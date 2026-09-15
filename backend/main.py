"""FastAPI-приложение: REST API + отдача дашборда."""

import asyncio
import fcntl
import logging
import os
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import audio_pipeline, config, resource_guard, transcribe
from .accentizer import Accentizer
from .audio_pipeline import RenderSettings, SpeakerSettings
from .dialogue_parser import Replica, parse_dialogue, split_into_chunks
from .job_queue import JobPayload, JobStatus, get_queue
from .resource_guard import ResourceGuard
from .tts_engine import TTSEngine
from .voices_store import ALLOWED_AUDIO_SUFFIXES, MAX_AUDIO_BYTES, get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("tts.main")

MIME_BY_FORMAT = {"wav": "audio/wav", "mp3": "audio/mpeg"}

# Псевдо-спикер для задачи прослушивания голоса (в диалоге не встречается)
PREVIEW_SPEAKER = "preview"
# Псевдо-спикер для режима «Сплошной текст»: один голос на все куски
TEXT_SPEAKER = "text"
# Причина прерывания задачи watchdog'ом (см. resource_guard)
RESOURCE_ABORT_REASON = "превышен лимит памяти"
# Лок единственного инстанса (см. _acquire_instance_lock)
INSTANCE_LOCK_PATH = config.BASE_DIR / ".voice_syntez.lock"


# --- модели запросов ----------------------------------------------------------
class SpeakerConfig(BaseModel):
    voice_id: str
    speed: float = config.DEFAULT_SPEED
    cfg_strength: float = config.DEFAULT_CFG_STRENGTH
    nfe_step: int = config.DEFAULT_NFE_STEP

    @field_validator("speed")
    @classmethod
    def _check_speed(cls, value: float) -> float:
        return min(max(value, config.SPEED_RANGE[0]), config.SPEED_RANGE[1])

    @field_validator("cfg_strength")
    @classmethod
    def _check_cfg(cls, value: float) -> float:
        return min(max(value, config.CFG_RANGE[0]), config.CFG_RANGE[1])

    @field_validator("nfe_step")
    @classmethod
    def _check_nfe(cls, value: int) -> int:
        return value if value in config.NFE_ALLOWED else config.DEFAULT_NFE_STEP


class ParseRequest(BaseModel):
    dialogue_text: str


class PreviewRequest(SpeakerConfig):
    """Прослушивание одного голоса: та же логика синтеза, что и у реплики диалога."""

    text: str = config.DEFAULT_PREVIEW_TEXT


class TrackOptions(BaseModel):
    """Параметры сборки итоговой дорожки: пауза между репликами и кроссфейд."""

    pause_ms: int = config.DEFAULT_PAUSE_MS
    cross_fade_duration: float = config.DEFAULT_CROSS_FADE_DURATION

    @field_validator("pause_ms")
    @classmethod
    def _check_pause(cls, value: int) -> int:
        return min(max(value, config.PAUSE_MS_RANGE[0]), config.PAUSE_MS_RANGE[1])

    @field_validator("cross_fade_duration")
    @classmethod
    def _check_crossfade(cls, value: float) -> float:
        return min(max(value, config.CROSS_FADE_RANGE[0]), config.CROSS_FADE_RANGE[1])


class GenerateRequest(TrackOptions):
    dialogue_text: str
    speakers: dict[str, SpeakerConfig] = Field(default_factory=dict)
    auto_accent: bool = True
    output_format: Literal["wav", "mp3"] = config.DEFAULT_OUTPUT_FORMAT


class RenderTextRequest(SpeakerConfig, TrackOptions):
    """Сплошной текст одним голосом: те же параметры синтеза, что и у реплики диалога."""

    text: str
    auto_accent: bool = True
    output_format: Literal["wav", "mp3"] = config.DEFAULT_OUTPUT_FORMAT


# --- жизненный цикл -----------------------------------------------------------
def _target_port() -> int:
    """Порт, на который сядет uvicorn: из его же аргументов, иначе из env PORT."""
    argv = sys.argv
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            return int(argv[index + 1])
        if arg.startswith("--port="):
            return int(arg.split("=", 1)[1])
    return int(os.environ.get("PORT", "8000"))


def _assert_port_free() -> None:
    """Останавливает старт, если порт уже занят другим живым процессом.

    Запуск в обход run.sh (`uvicorn backend.main:app`) иначе даёт второй живой
    инстанс: лимиты потоков и памяти считаются внутри процесса, и два инстанса
    складываются. Lifespan выполняется до биндинга сокета uvicorn'ом, поэтому
    проверка успевает сработать заранее.
    """
    port = _target_port()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"Порт {port} уже занят ({exc}). Похоже, приложение уже запущено — "
            "остановите первый инстанс или задайте другой PORT."
        ) from exc
    finally:
        sock.close()


def _acquire_instance_lock():
    """Берёт эксклюзивный лок на время жизни процесса.

    PIDFILE в run.sh обходится прямым запуском (`uvicorn backend.main:app
    --port 8001`): второй инстанс получает свой Metal-контекст и свою долю
    общей памяти Apple Silicon, а лимиты потоков и RSS считаются внутри
    процесса и сумму по машине не ограничивают. Лок держится ядром до конца
    процесса, поэтому «протухший» файл после падения ничего не блокирует.
    """
    handle = INSTANCE_LOCK_PATH.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "Другой экземпляр приложения уже работает (держит .voice_syntez.lock). "
            "Остановите его — в том числе если он слушает другой порт."
        ) from exc
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _warmup() -> None:
    """Тяжёлая инициализация в фоне, чтобы сервер был доступен сразу."""
    try:
        get_store().inspect_existing()  # F0 и метки для голосов, загруженных раньше
    except Exception as exc:
        logger.error("Не удалось проверить загруженные голоса: %s", exc)
    try:
        TTSEngine.instance().load()
    except Exception as exc:
        logger.error("Не удалось загрузить TTS-модель: %s", exc)
    Accentizer.instance().load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.instance_lock = _acquire_instance_lock()
    _assert_port_free()
    torch = config.configure_torch()
    logger.info("MPS available: %s", torch.backends.mps.is_available())
    logger.info("torch threads: %s", torch.get_num_threads())
    logger.info("Устройство синтеза: %s", config.pick_device())

    audio_pipeline.cleanup_output()
    await get_queue().start()
    resource_guard.snapshot()  # прогрев счётчика CPU: первый вызов cpu_percent всегда 0.0
    guard_task = asyncio.create_task(
        ResourceGuard(
            lambda: get_queue().request_abort(RESOURCE_ABORT_REASON)
        ).run(),
        name="resource-guard",
    )
    warmup_task = asyncio.create_task(asyncio.to_thread(_warmup))
    try:
        yield
    finally:
        await get_queue().stop()
        guard_task.cancel()
        warmup_task.cancel()


app = FastAPI(title="TTS Dashboard", lifespan=lifespan)

config.FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


# --- роуты --------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict:
    engine = TTSEngine.instance()
    accentizer = Accentizer.instance()
    return {
        "model_loaded": engine.is_loaded,
        "device": engine.device,
        "accentizer_loaded": accentizer.is_loaded,
        "accentizer_state": accentizer.state,
        "accentizer_error": accentizer.last_error,
        "queue_size": get_queue().queue_size(),
        **resource_guard.snapshot(),
    }


@app.post("/api/parse")
async def parse(payload: ParseRequest) -> dict:
    try:
        parsed = parse_dialogue(payload.dialogue_text, config.MAX_REPLICA_CHARS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return parsed.to_dict()


@app.get("/api/voices")
async def list_voices() -> dict:
    return {"voices": [voice.to_dict() for voice in get_store().list()]}


@app.post("/api/voices", status_code=201)
async def create_voice(
    name: str = Form(...),
    gender: str = Form("other"),
    ref_text: str = Form(""),
    file: UploadFile = File(...),
) -> dict:
    # Размер проверяем до чтения в память: иначе 50-МБ файл целиком окажется в RSS.
    if file.size is not None and file.size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Файл больше {MAX_AUDIO_BYTES // (1024 * 1024)} МБ",
        )
    audio_bytes = await file.read()
    try:
        # Проверка F0 и демо-файлов — не в event loop: декодирование аудио с ffmpeg
        voice = await asyncio.to_thread(
            get_store().create,
            name=name,
            gender=gender,
            ref_text=ref_text,
            audio_filename=file.filename or "",
            audio_bytes=audio_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.to_dict()


@app.post("/api/voices/transcribe")
async def transcribe_voice(file: UploadFile = File(...)) -> dict:
    """Расшифровывает запись — кнопка «Распознать» в форме нового голоса."""
    if file.size is not None and file.size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Файл больше {MAX_AUDIO_BYTES // (1024 * 1024)} МБ",
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f"Неподдерживаемый формат аудио: {suffix or '(нет расширения)'}"
        )
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл аудио")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        # Для F5-TTS запись всё равно обрезается до ~12 с — расшифровываем ровно
        # тот фрагмент, который уйдёт в модель (см. transcribe_worker).
        result = await asyncio.to_thread(transcribe.transcribe_file, tmp_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "ref_text": result.ref_text,
        "effective_sec": result.effective_sec,
        "full_sec": result.full_sec,
    }


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    if not get_store().delete(voice_id):
        raise HTTPException(status_code=404, detail="Голос не найден")
    return {"deleted": voice_id}


@app.post("/api/preview", status_code=202)
async def preview(payload: PreviewRequest) -> dict:
    """Синтез одной фразы выбранным голосом — через ту же очередь, что и диалоги."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст для прослушивания")
    if len(text) > config.MAX_REPLICA_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Текст длиннее {config.MAX_REPLICA_CHARS} символов — сократите фразу",
        )

    job_payload = JobPayload(
        replicas=[Replica(voice=PREVIEW_SPEAKER, text=text, line_number=1)],
        speakers={PREVIEW_SPEAKER: SpeakerSettings.from_dict(payload.model_dump())},
        settings=RenderSettings(
            pause_ms=0,
            cross_fade_duration=config.DEFAULT_CROSS_FADE_DURATION,
            auto_accent=True,
            output_format="wav",
        ),
    )
    try:
        job = get_queue().submit(job_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.id, "status": job.status.value, "total_replicas": 1}


@app.post("/api/generate", status_code=202)
async def generate(payload: GenerateRequest) -> dict:
    try:
        parsed = parse_dialogue(payload.dialogue_text, config.MAX_REPLICA_CHARS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not parsed.replicas:
        raise HTTPException(status_code=400, detail="Диалог пуст — нечего озвучивать")

    labels = {r.voice: r.label for r in parsed.replicas}
    missing = sorted({r.voice for r in parsed.replicas if r.voice not in payload.speakers})
    if missing:
        names = ", ".join(f"«{labels[key]}»" for key in missing)
        raise HTTPException(status_code=400, detail=f"Не назначен голос для: {names}")

    job_payload = JobPayload(
        replicas=parsed.replicas,
        speakers={s: SpeakerSettings.from_dict(cfg.model_dump()) for s, cfg in payload.speakers.items()},
        settings=RenderSettings(
            pause_ms=payload.pause_ms,
            cross_fade_duration=payload.cross_fade_duration,
            auto_accent=payload.auto_accent,
            output_format=payload.output_format,
        ),
    )
    try:
        job = get_queue().submit(job_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.id, "status": job.status.value, "total_replicas": job.total_replicas}


@app.post("/api/render-text", status_code=202)
async def render_text(payload: RenderTextRequest) -> dict:
    """Озвучка сплошного текста одним голосом — без спикеров и маркеров."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст пуст — нечего озвучивать")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Текст длиннее {config.MAX_TEXT_CHARS} символов — разбейте его на части",
        )

    chunks = split_into_chunks(text, config.MAX_REPLICA_CHARS)
    if not chunks:
        raise HTTPException(status_code=400, detail="Текст пуст — нечего озвучивать")

    job_payload = JobPayload(
        replicas=[
            Replica(voice=TEXT_SPEAKER, text=chunk, line_number=index)
            for index, chunk in enumerate(chunks, start=1)
        ],
        speakers={TEXT_SPEAKER: SpeakerSettings.from_dict(payload.model_dump())},
        settings=RenderSettings(
            pause_ms=payload.pause_ms,
            cross_fade_duration=payload.cross_fade_duration,
            auto_accent=payload.auto_accent,
            output_format=payload.output_format,
        ),
    )
    try:
        job = get_queue().submit(job_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.id, "status": job.status.value, "total_replicas": job.total_replicas}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = get_queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    data = job.to_dict()
    data["audio_url"] = f"/api/jobs/{job_id}/audio" if job.status is JobStatus.DONE else None
    return data


@app.get("/api/jobs/{job_id}/audio")
async def job_audio(job_id: str, download: bool = False) -> FileResponse:
    job = get_queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if job.status is not JobStatus.DONE or job.output_path is None:
        raise HTTPException(status_code=409, detail="Файл ещё не готов")
    if not job.output_path.exists():
        raise HTTPException(status_code=410, detail="Файл удалён (истёк срок хранения)")
    return FileResponse(
        job.output_path,
        media_type=MIME_BY_FORMAT.get(job.output_format, "application/octet-stream"),
        filename=job.output_path.name if download else None,
    )
