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

from . import audio_analysis, audio_pipeline, config, denoise, resource_guard, transcribe
from .accentizer import Accentizer
from .audio_pipeline import RenderSettings, SpeakerSettings
from .dialogue_parser import Replica, parse_dialogue, split_into_chunks
from .engines.base import ENGINE_F5, ENGINE_INFOS
from .engines.registry import created_engines, get_engine
from .job_queue import Job, JobPayload, JobStatus, get_queue
from .resource_guard import ResourceGuard
from .voices_store import ALLOWED_AUDIO_SUFFIXES, MAX_AUDIO_BYTES, get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
# torch сообщает о фолбэке MPS→CPU обычным Python-warning'ом (TORCH_WARN →
# warnings.warn), а не через logging. Без перенаправления этот warning теряется,
# и «тихий» CPU-фолбэк внутри куска остаётся незамеченным — а он даёт микро-артефакты.
logging.captureWarnings(True)
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
# Стратегия нарезки текста на куски. Literal, а не свободная строка: опечатка с
# фронта должна быть ошибкой запроса, а не тихой подменой длины куска на дефолт.
ChunkStrategy = Literal["short", "paragraph"]


class SpeakerConfig(BaseModel):
    voice_id: str
    speed: float = config.DEFAULT_SPEED
    cfg_strength: float = config.DEFAULT_CFG_STRENGTH
    nfe_step: int = config.DEFAULT_NFE_STEP
    target_rms: float = config.DEFAULT_TARGET_RMS
    gain_db: float = config.DEFAULT_GAIN_DB
    pitch_semitones: float = config.DEFAULT_PITCH_SEMITONES
    # Пауза перед репликами спикера; None — использовать общую для всего диалога.
    pause_override_ms: int | None = None
    # Ручки движка, переопределённые для этого слота (у XTTS — температура и
    # штраф за повторы). Границы проверяет сам движок: они объявлены в его
    # паспорте (engines/base), и дублировать их здесь означало бы два источника правды.
    engine_params: dict = Field(default_factory=dict)

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

    @field_validator("target_rms")
    @classmethod
    def _check_target_rms(cls, value: float) -> float:
        return min(max(value, config.TARGET_RMS_RANGE[0]), config.TARGET_RMS_RANGE[1])

    @field_validator("gain_db")
    @classmethod
    def _check_gain(cls, value: float) -> float:
        return min(max(value, config.GAIN_DB_RANGE[0]), config.GAIN_DB_RANGE[1])

    @field_validator("pitch_semitones")
    @classmethod
    def _check_pitch(cls, value: float) -> float:
        low, high = config.PITCH_SEMITONES_RANGE
        return min(max(value, low), high)

    @field_validator("pause_override_ms")
    @classmethod
    def _check_pause_override(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return min(max(value, config.PAUSE_MS_RANGE[0]), config.PAUSE_MS_RANGE[1])


class ParseRequest(BaseModel):
    dialogue_text: str
    # Стратегия нарезки нужна и разбору: панель слотов должна показывать ровно
    # те реплики, на которые текст будет разбит при генерации.
    chunk_strategy: ChunkStrategy = config.CHUNK_STRATEGY_DEFAULT


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
    chunk_strategy: ChunkStrategy = config.CHUNK_STRATEGY_DEFAULT


class RenderTextRequest(SpeakerConfig, TrackOptions):
    """Сплошной текст одним голосом: те же параметры синтеза, что и у реплики диалога."""

    text: str
    auto_accent: bool = True
    output_format: Literal["wav", "mp3"] = config.DEFAULT_OUTPUT_FORMAT
    chunk_strategy: ChunkStrategy = config.CHUNK_STRATEGY_DEFAULT


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
        # Прогреваем только F5: он движок по умолчанию для новых голосов, а XTTS
        # поднимается по факту выбора (см. engines/registry — движок живёт тёплым
        # до конца процесса, лишние секунды и гигабайты на старте не нужны).
        get_engine(ENGINE_F5).load()
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
    accentizer = Accentizer.instance()
    engines = created_engines()
    f5 = engines.get(ENGINE_F5)
    return {
        # Совместимые поля: по ним дашборд показывает готовность основного движка.
        "model_loaded": bool(f5 and f5.is_loaded),
        "device": config.pick_device(),
        # Состояние каждого поднятого движка: XTTS грузится десятки секунд, и без
        # этого списка в интерфейсе не отличить «ещё грузится» от «упал».
        "engines": [engine.to_dict() for engine in engines.values()],
        "accentizer_loaded": accentizer.is_loaded,
        "accentizer_state": accentizer.state,
        "accentizer_error": accentizer.last_error,
        # Опциональная очистка референса: без DeepFilterNet тумблер в форме гаснет.
        "denoise_available": denoise.is_available(),
        "queue_size": get_queue().queue_size(),
        **resource_guard.snapshot(),
    }


@app.get("/api/engines")
async def list_engines() -> dict:
    """Паспорта движков: подписи, описания и границы ручек для интерфейса.

    Отдаются все объявленные движки, а не только загруженные: выбор движка в
    форме голоса нужен до того, как модель впервые понадобилась.
    """
    return {"engines": [info.to_dict() for info in ENGINE_INFOS.values()]}


@app.post("/api/parse")
async def parse(payload: ParseRequest) -> dict:
    try:
        parsed = parse_dialogue(payload.dialogue_text, config.chunk_chars(payload.chunk_strategy))
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
    verify_ref_text: bool = Form(True),
    engine: str = Form(""),
    denoise: bool = Form(False),
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
            verify_ref_text=verify_ref_text,
            engine=engine,
            denoise=denoise,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Очистка референса (DeepFilterNet) падает, например, если зависимость не поставлена.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.to_dict()


@app.post("/api/voices/analyze")
async def analyze_voice(file: UploadFile = File(...), gender: str = Form("other")) -> dict:
    """Проверяет запись до сохранения голоса: тон, пол, перегрузка, длительность.

    Запись из браузера нельзя оценить на клиенте: измерение тона живёт в
    `audio_analysis`, и его копия в JS разошлась бы с той проверкой, которая потом
    применяется при сохранении голоса.
    """
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

    report = await asyncio.to_thread(audio_analysis.analyze, audio_bytes, suffix)
    warnings: list[str] = []
    if report.duration_sec <= 0:
        warnings.append(
            "Не удалось прочитать запись: браузер сохранил формат, который не разбирает ffmpeg."
        )
    elif report.f0_hz is None:
        warnings.append(
            "Высоту тона измерить не удалось: запись пустая, слишком тихая или короче "
            "полсекунды. Проверьте, что микрофон выбран верный и вы говорили в него."
        )
    else:
        mismatch = audio_analysis.check_gender_mismatch(report.f0_hz, gender)
        if mismatch:
            warnings.append(mismatch)
    if report.clipping_warning:
        warnings.append(report.clipping_warning)
    if report.band_warning:
        warnings.append(report.band_warning)

    return {
        "duration_sec": round(report.duration_sec, 2),
        "f0_hz": report.f0_hz,
        "band_high_db": None if report.band_high_db is None else round(report.band_high_db, 1),
        "warnings": warnings,
    }


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


class VoiceEngineUpdate(BaseModel):
    """Смена движка и его ручек у готового голоса."""

    engine: str | None = None
    engine_params: dict | None = None


@app.patch("/api/voices/{voice_id}")
async def update_voice(voice_id: str, payload: VoiceEngineUpdate) -> dict:
    """Меняет движок голоса, не трогая запись референса.

    Движок — свойство голоса, а не проекта: один и тот же текст может звучать
    разными моделями, и переключаться между ними приходится на готовом голосе.
    """
    try:
        voice = get_store().update(
            voice_id, engine=payload.engine, engine_params=payload.engine_params
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Голос не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.to_dict()


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
        parsed = parse_dialogue(payload.dialogue_text, config.chunk_chars(payload.chunk_strategy))
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

    chunks = split_into_chunks(text, config.chunk_chars(payload.chunk_strategy))
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
    queue = get_queue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    data = job.to_dict()
    data["audio_url"] = f"/api/jobs/{job_id}/audio" if job.status is JobStatus.DONE else None
    if job.status is JobStatus.DONE and job.payload is not None:
        # Список реплик отдаём один раз — вместе с готовым файлом: по нему UI
        # показывает, какой кусок какому месту в аудио соответствует.
        data["replicas"] = [
            {
                "index": index,
                "label": replica.label,
                "text": replica.text,
                # Сид текущего звучания: без него понравившийся кусок не повторить.
                "seed": queue.active_seed(job, index),
                "variants": _variants_payload(job_id, index, job),
            }
            for index, replica in enumerate(job.payload.replicas)
        ]
    return data


def _variants_payload(job_id: str, index: int, job: Job) -> list[dict]:
    """Варианты одной реплики для интерфейса: подпись, сид, ссылка на прослушивание.

    Ссылка на файл варианта, а не на готовый трек: варианты сравнивают между
    собой, а в треке звучит только один из них.
    """
    active = job.active_variant.get(index)
    return [
        {
            "id": variant.id,
            "label": variant.label,
            "seed": variant.seed,
            "duration_sec": round(variant.duration_sec, 2),
            "active": variant.id == active,
            "audio_url": f"/api/jobs/{job_id}/replicas/{index}/variants/{variant.id}/audio",
        }
        for variant in job.variants.get(index, [])
    ]


@app.post("/api/jobs/{job_id}/replicas/{index}/regenerate", status_code=202)
async def regenerate_replica(job_id: str, index: int) -> dict:
    """Пересинтезирует одну реплику готового файла, не трогая остальные.

    Прежнее звучание реплики остаётся доступным вариантом: интерфейс даёт
    прослушать оба и вернуть любой обратно (`.../variants/{id}`).
    """
    if get_queue().get(job_id) is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    try:
        get_queue().submit_regenerate(job_id, index)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "index": index, "status": JobStatus.QUEUED.value}


@app.post("/api/jobs/{job_id}/replicas/{index}/variants/{variant_id}", status_code=202)
async def select_replica_variant(job_id: str, index: int, variant_id: str) -> dict:
    """Ставит в готовый файл ранее сохранённый вариант реплики.

    Без пересинтеза: вариант — это уже готовое аудио, поэтому возврат к прежнему
    звучанию мгновенный, а не «ещё один прогон модели».
    """
    queue = get_queue()
    if queue.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    try:
        queue.submit_variant(job_id, index, variant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job_id,
        "index": index,
        "variant_id": variant_id,
        "status": JobStatus.QUEUED.value,
    }


@app.get("/api/jobs/{job_id}/replicas/{index}/variants/{variant_id}/audio")
async def replica_variant_audio(job_id: str, index: int, variant_id: str) -> FileResponse:
    """Отдаёт файл варианта — для прослушивания и сравнения на слух."""
    queue = get_queue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    try:
        variant = queue.find_variant(job, index, variant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not variant.path.exists():
        raise HTTPException(status_code=410, detail="Файл варианта удалён")
    # Варианты всегда wav: они служат для сравнения, а не для выдачи пользователю.
    return FileResponse(variant.path, media_type="audio/wav")


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
