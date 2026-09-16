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
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, BeforeValidator, Field, field_validator

from . import (
    audio_analysis,
    audio_pipeline,
    benchmark,
    config,
    denoise,
    engine_lifecycle,
    model_manager,
    resource_guard,
    timeline,
    transcribe,
)
from .accentizer import Accentizer
from .audio_pipeline import QaOutcome, QaSettings, RenderSettings, SpeakerSettings
from .db.connection import init_db
from .db.store import UNSET, get_projects_store
from .dialogue_parser import Replica, parse_dialogue, split_into_chunks
from .engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    STATE_IDLE,
    EngineBusyError,
    EngineInfo,
)
from .engines.registry import created_engine, created_engines, get_engine
from .job_queue import (
    PRIORITY_BACKGROUND,
    PRIORITY_PREVIEW,
    PRIORITY_RENDER,
    Job,
    JobPayload,
    JobStatus,
    get_queue,
)
from .model_manager import (
    ModelBusyError,
    ModelNotDownloadableError,
    ModelPathError,
    get_manager,
)
from .pronunciation import PronunciationConflict
from .pronunciation import get_store as get_pronunciation_store
from .resource_guard import ResourceGuard
from .text_normalization import normalize, normalize_report
from .voices_store import ALLOWED_AUDIO_SUFFIXES, MAX_AUDIO_BYTES, Voice, get_store

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


def _qa_mode(value: Any, fallback: str | None = None) -> str:
    """Приводит режим проверки к одному из `config.QA_MODES`.

    Булево — это прежний интерфейс, когда режимов было два: `true` означал
    строгую проверку каждого куска, `false` — выключенную. Значение без
    соответствия — ошибка запроса, кроме случая с `fallback`: в сохранённых
    настройках проекта испорченное значение должно откатываться к дефолту, а не
    валить рендер.
    """
    if isinstance(value, bool):
        return config.QA_MODE_STRICT if value else config.QA_MODE_OFF
    if value is None:
        return config.QA_MODE_OFF
    text = str(value).strip().lower()
    if text in config.QA_MODES:
        return text
    if fallback is not None:
        return fallback
    raise ValueError(f"неизвестный режим проверки: {value!r}")


def _qa_mode_or_none(value: Any) -> str | None:
    """Режим из необязательного поля: None здесь значит «не трогать сохранённое».

    Для `ProjectRenderRequest` непришедшее поле — это «как в прошлый раз», а не
    «выключено»: рендер без тела запроса обязан звучать так же, как предыдущий.
    """
    return None if value is None else _qa_mode(value)


QaMode = Annotated[Literal["off", "smart", "strict"], BeforeValidator(_qa_mode)]
QaModeOrNone = Annotated[
    Literal["off", "smart", "strict"] | None, BeforeValidator(_qa_mode_or_none)
]


class GenerateRequest(TrackOptions):
    dialogue_text: str
    speakers: dict[str, SpeakerConfig] = Field(default_factory=dict)
    auto_accent: bool = True
    # Проверка качества (Фаза 7): `off` — как раньше без проверки, `smart` —
    # сначала дешёвый отбор по waveform и только подозрительные куски в Whisper,
    # `strict` — каждый кусок через Whisper. По умолчанию выключено: попытка —
    # это синтез плюс отдельный процесс Whisper, и на длинном диалоге цена растёт
    # кратно. Числа (порог, попытки, бюджет) — из конфига: это не то, что имеет
    # смысл крутить на каждой задаче.
    qa: QaMode = config.QA_MODE_OFF
    output_format: Literal["wav", "mp3"] = config.DEFAULT_OUTPUT_FORMAT
    chunk_strategy: ChunkStrategy = config.CHUNK_STRATEGY_DEFAULT
    # Фоновая задача (Фаза 11): рендер уходит в приоритет 3 и пропускает вперёд
    # preview и перегенерацию. Поле необязательное — без него поведение прежнее.
    background: bool = False


class RenderTextRequest(SpeakerConfig, TrackOptions):
    """Сплошной текст одним голосом: те же параметры синтеза, что и у реплики диалога."""

    text: str
    auto_accent: bool = True
    qa: QaMode = config.QA_MODE_OFF
    output_format: Literal["wav", "mp3"] = config.DEFAULT_OUTPUT_FORMAT
    chunk_strategy: ChunkStrategy = config.CHUNK_STRATEGY_DEFAULT
    # См. `GenerateRequest.background`.
    background: bool = False


# --- модели запросов проектов -------------------------------------------------
class ProjectSpeakerUpdate(BaseModel):
    """Назначение голоса спикеру проекта и его персональные переопределения."""

    voice_id: str = ""
    overrides: dict = Field(default_factory=dict)


class ProjectCreateRequest(BaseModel):
    name: str
    source_text: str = ""
    mode: Literal["dialogue", "text"] = config.PROJECT_MODE_DIALOGUE
    render_settings: dict = Field(default_factory=dict)


class ProjectUpdateRequest(BaseModel):
    """Правка проекта: любое поле по отдельности, включая назначения голосов."""

    name: str | None = None
    source_text: str | None = None
    mode: Literal["dialogue", "text"] | None = None
    render_settings: dict | None = None
    speakers: dict[str, ProjectSpeakerUpdate] | None = None


class ProjectParseRequest(BaseModel):
    """Разбор исходного текста проекта в реплики.

    Стратегия нарезки необязательна: у проекта она уже сохранена в настройках, и
    повторный разбор без параметров должен давать то же разбиение, что и первое.
    """

    chunk_strategy: ChunkStrategy | None = None


class ReplicaUpdateRequest(BaseModel):
    """Правка одной реплики: голос и/или отдельные параметры.

    Различать «поле не пришло» и «пришло null» обязательно: непришедшее значит
    «не трогать», а null — «вернуть наследуемое у спикера» (голос, паузу, ручку).
    Поэтому обработчик смотрит на `exclude_unset`, а не на значения полей.
    """

    # Голос этой реплики; null — снова голос спикера.
    voice_id: str | None = None
    # Точечные правки параметров; null у ключа — сброс к значению спикера.
    overrides: dict[str, Any] | None = None
    # Сбросить все правки реплики разом (кнопка «вернуть как у спикера»).
    reset_overrides: bool = False


class ProjectRenderRequest(BaseModel):
    """Запуск рендера проекта сохранёнными настройками.
    запуск без тела запроса должен звучать так же, как предыдущий. Поэтому здесь
    нет ни наследования `TrackOptions` с его дефолтами, ни его валидаторов:
    непришедшее поле означает «как сохранено», а не «значение по умолчанию».
    """

    pause_ms: int | None = None
    cross_fade_duration: float | None = None
    auto_accent: bool | None = None
    qa: QaModeOrNone = None
    output_format: Literal["wav", "mp3"] | None = None
    chunk_strategy: ChunkStrategy | None = None
    # Фоновая задача (Фаза 11): приоритет 3. Не запоминается в настройках проекта:
    # это свойство запуска, а не сборки.
    background: bool = False


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
    init_db()  # база проектов создаётся сама: ручного SQL от пользователя не требуется
    await get_queue().start()
    resource_guard.snapshot()  # прогрев счётчика CPU: первый вызов cpu_percent всегда 0.0
    guard_task = asyncio.create_task(
        ResourceGuard(
            lambda: get_queue().request_abort(RESOURCE_ABORT_REASON)
        ).run(),
        name="resource-guard",
    )
    warmup_task = asyncio.create_task(asyncio.to_thread(_warmup))
    # Выгрузка простаивающих движков: возвращает память без рестарта, но только
    # когда очередь пуста и порог простоя пройден (см. engine_lifecycle).
    idle_unload_task = asyncio.create_task(
        engine_lifecycle.IdleUnloadWatcher().run(), name="engine-idle-unload"
    )
    try:
        yield
    finally:
        await get_queue().stop()
        guard_task.cancel()
        warmup_task.cancel()
        idle_unload_task.cancel()


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
        # Память MPS: на Apple Silicon вес моделей не виден в RSS (см. MAX_RSS_MB),
        # поэтому здесь best-effort счётчики torch. `None` — torch/MPS недоступны;
        # ошибка чтения метрики не должна валить статус.
        "mps_memory": resource_guard.mps_memory_snapshot(),
        **resource_guard.snapshot(),
    }


@app.get("/api/engines")
async def list_engines() -> dict:
    """Паспорта движков: подписи, описания и границы ручек для интерфейса.

    Отдаются все объявленные движки, а не только загруженные: выбор движка в
    форме голоса нужен до того, как модель впервые понадобилась.
    """
    return {"engines": [info.to_dict() for info in ENGINE_INFOS.values()]}


# --- выгрузка движков ---------------------------------------------------------
# Возврат памяти без перезапуска приложения (фаза 10). Ручные роуты и фоновую
# задачу объединяет одно правило: движок нельзя выгружать, пока он синтезирует
# или пока очередь занята задачей — иначе следующей реплике придётся поднимать
# его заново, а перегенерация получит выгруженную модель посреди работы.
ENGINE_UNLOAD_QUEUE_BUSY = (
    "В очереди есть задача: движок может понадобиться следующей реплике. "
    "Дождитесь окончания задачи и повторите выгрузку."
)


def _unknown_engine(engine_id: str) -> HTTPException:
    """404 с перечнем доступных движков — общий текст для обоих роутов."""
    return HTTPException(
        status_code=404,
        detail=f"Движок «{engine_id}» не найден. Доступны: " + ", ".join(ENGINE_INFOS),
    )


@app.post("/api/engines/{engine_id}/unload")
async def unload_engine(engine_id: str) -> dict:
    """Выгружает движок и возвращает занятую им память.

    409 в двух случаях: движок прямо сейчас синтезирует (`EngineBusyError`) или
    очередь занята задачей. 404 — неизвестный id: молчаливый no-op скрыл бы
    опечатку. Повторная выгрузка незагруженного движка — успешный no-op.
    """
    info = ENGINE_INFOS.get(engine_id)
    if info is None:
        raise _unknown_engine(engine_id)
    engine = created_engine(engine_id)
    if engine is None:
        return {
            "engine": engine_id,
            "state": STATE_IDLE,
            "unloaded": False,
            "message": f"Движок «{info.label}» и так не поднят — выгружать нечего.",
        }
    if engine.is_loaded and engine_lifecycle.queue_busy_now():
        raise HTTPException(status_code=409, detail=ENGINE_UNLOAD_QUEUE_BUSY)
    try:
        await asyncio.to_thread(engine.unload)
    except EngineBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        # Освободить не удалось, но сервис жив: отдаём причину текстом и 500.
        logger.exception("Не удалось выгрузить движок %s", engine_id)
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось выгрузить движок «{info.label}»: {exc}",
        ) from exc
    return {
        "engine": engine_id,
        "state": engine.state,
        "unloaded": True,
        "message": f"Движок «{info.label}» выгружен: память освобождена.",
    }


@app.post("/api/engines/{engine_id}/load")
async def load_engine(engine_id: str) -> dict:
    """Поднимает движок заранее, без синтеза — чтобы вернуть выгруженный.

    Модель поднимается десятки секунд, поэтому роут синхронный для вызывающего,
    но не блокирует цикл событий: загрузка идёт в отдельном потоке. Повторный
    вызов на поднятом движке — идемпотентный no-op.
    """
    info = ENGINE_INFOS.get(engine_id)
    if info is None:
        raise _unknown_engine(engine_id)
    try:
        engine = get_engine(engine_id)
        await asyncio.to_thread(engine.load)
    except Exception as exc:
        # Не поднялся: причина уходит в текст ответа, состояние движка уже `failed`.
        logger.exception("Не удалось поднять движок %s", engine_id)
        raise HTTPException(
            status_code=500,
            detail=f"Движок «{info.label}» не поднялся: {exc}",
        ) from exc
    return {
        "engine": engine_id,
        "state": engine.state,
        "loaded": engine.is_loaded,
        "message": f"Движок «{info.label}» поднят.",
    }


# --- менеджер моделей ---------------------------------------------------------
# Веса — это диск и сеть, а не инференс: скачивание и удаление идут мимо очереди
# синтеза. Но удаление обязано уважать занятость движка: файлы работающей модели
# трогать нельзя (см. `model_manager.guard_delete`).
@app.get("/api/models")
async def list_models() -> dict:
    """Все модели проекта: установка, размер, состояние движка и скачивания.

    Пути и размеры считаются на диске; в сеть этот роут не ходит — список
    моделей открывается и в офлайне.
    """
    manager = get_manager()
    states = await asyncio.to_thread(manager.states)
    return {
        "models": [state.to_dict() for state in states],
        "disk": manager.disk_report(states),
    }


@app.get("/api/models/{model_id}")
async def get_model(model_id: str) -> dict:
    """Одна модель — для опроса прогресса скачивания."""
    try:
        spec = model_manager.model_spec(model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Модель «{model_id}» не найдена. Доступны: "
            + ", ".join(spec.id for spec in model_manager.model_specs()),
        ) from exc
    return await asyncio.to_thread(lambda: get_manager().state(spec).to_dict())


@app.post("/api/models/{model_id}/download", status_code=202)
async def download_model(model_id: str) -> dict:
    """Запускает скачивание модели в фоне и сразу отдаёт её состояние.

    Уже установленная модель не скачивается повторно: её файлы остаются
    нетронутыми, а состояние помечается `done`.
    """
    try:
        spec = model_manager.model_spec(model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Модель «{model_id}» не найдена. Доступны: "
            + ", ".join(spec.id for spec in model_manager.model_specs()),
        ) from exc
    try:
        return await asyncio.to_thread(get_manager().download, spec.id)
    except ModelPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotDownloadableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    """Удаляет файлы модели.

    409, пока движок модели занят: поднятая модель держит файлы открытыми, а
    очередь может синтезировать прямо сейчас. Выгрузить движок можно роутом
    `POST /api/engines/{id}/unload` — текст отказа говорит, что именно сделать.
    """
    try:
        spec = model_manager.model_spec(model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Модель «{model_id}» не найдена. Доступны: "
            + ", ".join(spec.id for spec in model_manager.model_specs()),
        ) from exc
    try:
        return await asyncio.to_thread(get_manager().delete, spec.id)
    except ModelPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelNotDownloadableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    """Смена движка, его ручек и пресета у готового голоса."""

    engine: str | None = None
    engine_params: dict | None = None
    # Пресет: значения, подобранные в «Прослушать». Отсутствие поля означает
    # «не трогать», пустой словарь — «сбросить к паспорту движка».
    preset: dict | None = None


@app.patch("/api/voices/{voice_id}")
async def update_voice(voice_id: str, payload: VoiceEngineUpdate) -> dict:
    """Меняет движок, его ручки и пресет голоса, не трогая запись референса.

    Движок — свойство голоса, а не проекта: один и тот же текст может звучать
    разными моделями, и переключаться между ними приходится на готовом голосе.
    Пресет — тоже свойство голоса: настроив его один раз, пользователь получает
    эти значения во всех новых диалогах (см. settings_resolution).
    """
    try:
        voice = get_store().update(
            voice_id,
            engine=payload.engine,
            engine_params=payload.engine_params,
            preset=payload.preset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Голос не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.to_dict()


class PronunciationEntryRequest(BaseModel):
    """Новое правило словаря произношения.

    Повторное добавление того же источника (с тем же режимом регистра) обновляет
    правило: пользователь, дважды отправивший форму, ждёт исправленную замену, а не
    409. Так же ведёт себя и импорт словаря из внешнего скрипта.
    """

    source: str
    target: str
    case_sensitive: bool = False
    whole_word: bool = True
    enabled: bool = True
    note: str = ""


class PronunciationUpdateRequest(BaseModel):
    """Правка правила: непришедшее поле значит «не трогать»."""

    source: str | None = None
    target: str | None = None
    case_sensitive: bool | None = None
    whole_word: bool | None = None
    enabled: bool | None = None
    note: str | None = None


class PronunciationPreviewRequest(BaseModel):
    """Проверка текста словарём. `engine` необязателен и не поднимает модель."""

    text: str
    engine: str | None = None


class TextPreviewRequest(BaseModel):
    """Что услышит модель: текст и движок, по которым показать все стадии.

    Источник текста ровно один из двух, и это осознанно разные сценарии: панель
    «Что услышит модель» присылает произвольный `text`, а карточка реплики —
    `project_id` + `replica_index`, чтобы preview показывал текст и голос именно
    этой реплики, а не набранный заново. Движок разрешается по цепочке
    реплика → её голос → `voice_id` → `engine`; чем-то одним он должен
    определиться, иначе непонятно, ставить ли ударения.
    """

    text: str | None = None
    project_id: str | None = None
    replica_index: int | None = None
    voice_id: str | None = None
    engine: str | None = None
    auto_accent: bool = True


@app.get("/api/pronunciation")
async def list_pronunciation() -> dict:
    """Правила словаря целиком, включая выключенные: интерфейс показывает и их."""
    entries = get_pronunciation_store().list_entries()
    return {"entries": entries, "count": len(entries)}


@app.post("/api/pronunciation", status_code=201)
async def create_pronunciation(payload: PronunciationEntryRequest) -> dict:
    """Добавляет правило; если источник уже есть — обновляет его, а не плодит дубли."""
    try:
        return get_pronunciation_store().create(
            source=payload.source,
            target=payload.target,
            case_sensitive=payload.case_sensitive,
            whole_word=payload.whole_word,
            enabled=payload.enabled,
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/pronunciation/{entry_id}")
async def update_pronunciation(entry_id: int, payload: PronunciationUpdateRequest) -> dict:
    """Меняет поля правила, в том числе `enabled`, не удаляя его из словаря."""
    try:
        return get_pronunciation_store().update(
            entry_id, **payload.model_dump(exclude_unset=True)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Правило словаря не найдено") from exc
    except PronunciationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/pronunciation/{entry_id}")
async def delete_pronunciation(entry_id: int) -> dict:
    if not get_pronunciation_store().delete(entry_id):
        raise HTTPException(status_code=404, detail="Правило словаря не найдено")
    return {"deleted": entry_id}


@app.post("/api/pronunciation/preview")
async def preview_pronunciation(payload: PronunciationPreviewRequest) -> dict:
    """Показывает стадии обработки текста: нормализация → словарь → движок.

    Ничего не пишет в базу и не поднимает TTS-движок: `engine` нужен только для
    флага `supports_accents` из паспорта, а не для синтеза. `matches` — какие
    правила сработали и сколько раз.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст для проверки")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Текст длиннее {config.MAX_TEXT_CHARS} символов — сократите его",
        )
    supports_accents = True
    if payload.engine:
        info = ENGINE_INFOS.get(payload.engine)
        if info is None:
            known = ", ".join(sorted(ENGINE_INFOS))
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестный движок: {payload.engine}. Доступны: {known}",
            )
        supports_accents = info.supports_accents

    entries = get_pronunciation_store().active_rules()
    normalized = normalize(text)
    result, matches = normalize_report(text, entries, supports_accents=supports_accents)
    return {"original": text, "normalized": normalized, "result": result, "matches": matches}


def _preview_source(payload: TextPreviewRequest) -> tuple[str, str, EngineInfo]:
    """Разрешает текст и движок preview по цепочке из запроса.

    Возвращает текст, идентификатор движка и его паспорт. Цепочка ровно та же,
    что у синтеза реплики: реплика проекта → её голос (уже с учётом override) →
    `voice_id` из запроса → `engine`. Порядок здесь не формальность: если у
    реплики назначен F5, а в поле `engine` случайно остался XTTS, preview обязан
    показать F5 — иначе пользователь увидит текст без ударений, а услышит с ними.

    Ошибки — 400/404 с понятным русским текстом: preview дешёвый, и лучше
    отказать, чем молча показать «как для F5» чужой сценарий.

    Если пришли и `text`, и реплика, текст берётся из запроса, а голос — из
    реплики: панель может проверить произвольную фразу голосом конкретной строки,
    а карточка реплики просто не присылает `text` и получает и то, и другое.
    """
    if payload.replica_index is not None and not payload.project_id:
        raise HTTPException(status_code=400, detail="Для реплики проекта нужен project_id")
    if payload.project_id and payload.replica_index is None:
        raise HTTPException(
            status_code=400,
            detail="Укажите replica_index — preview показывает текст одной реплики",
        )

    replica_text = ""
    replica_voice_id = ""
    if payload.project_id:
        project = _project_or_404(payload.project_id)
        replica = _replica_or_404(project, payload.replica_index)
        replica_text = str(replica.get("text") or "")
        # `voice_id` реплики уже учитывает её собственный override (см. sync_voices).
        replica_voice_id = str(replica.get("voice_id") or "")

    text = payload.text if payload.text and payload.text.strip() else replica_text
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Пустой текст: введите текст или выберите непустую реплику проекта",
        )
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Текст длиннее {config.MAX_TEXT_CHARS} символов — сократите его",
        )

    voices: dict = {}
    engine_id = ""
    replica_voice = _resolved_voice(replica_voice_id, voices) if replica_voice_id else None
    if replica_voice is not None:
        engine_id = replica_voice.engine
    elif payload.voice_id:
        voice = _resolved_voice(payload.voice_id, voices)
        if voice is None:
            raise HTTPException(
                status_code=404, detail=f"Голос не найден: {payload.voice_id}"
            )
        engine_id = voice.engine

    if not engine_id:
        if not payload.engine:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Не удалось определить движок: укажите голос, engine "
                    "или реплику с назначенным голосом"
                ),
            )
        engine_id = payload.engine

    info = ENGINE_INFOS.get(engine_id)
    if info is None:
        known = ", ".join(sorted(ENGINE_INFOS))
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный движок: {engine_id}. Доступны: {known}",
        )
    return text, engine_id, info


@app.post("/api/text/preview")
async def preview_text_stages(payload: TextPreviewRequest) -> dict:
    """Что услышит модель: все стадии preprocessing без единого вызова синтеза.

    Модель TTS здесь не поднимается: `supports_accents` берётся из паспорта
    (`ENGINE_INFOS`), а не из живого движка, — поэтому preview стоит миллисекунды и
    не занимает память моделью. RUAccent — исключение: он и есть предмет проверки,
    и поднимается ровно так же, как при синтезе, чтобы preview не обещал ударений,
    которых не будет.

    Отказ RUAccent не роняет запрос: возвращается 200, текст без ударений и явные
    `accentizer.state`/`accentizer.error`. Иначе «почему нет ударений» пришлось бы
    выяснять по логам, а preview — ровно то место, где это должно быть видно.

    Стадии и итог считает `audio_pipeline.preview_text` — та же функция, которой
    пользуется синтез. Ничего в базу не пишется, исходный текст не меняется.
    """
    text, engine_id, info = _preview_source(payload)
    stages = await asyncio.to_thread(
        audio_pipeline.preview_text,
        text,
        supports_accents=info.supports_accents,
        auto_accent=payload.auto_accent,
    )
    accentizer = Accentizer.instance()
    return {
        "original": stages.original,
        "normalized": stages.normalized,
        "dictionary": stages.dictionary,
        "accentized": stages.accentized,
        "final": stages.final,
        "matches": stages.matches,
        "engine": engine_id,
        "engine_label": info.label,
        "supports_accents": stages.supports_accents,
        "accents_applied": stages.accents_applied,
        "accentizer": {"state": accentizer.state, "error": accentizer.last_error},
    }


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
        # Незаданные поля наследуются от голоса, а не подставляются дефолтами
        # движка: прослушивание и диалог должны считать слои одинаково, иначе
        # подобранные в «Прослушать» настройки в диалоге звучали бы иначе.
        speakers={
            PREVIEW_SPEAKER: SpeakerSettings.from_dict(
                payload.model_dump(exclude_unset=True, exclude={"text"})
            )
        },
        settings=RenderSettings(
            pause_ms=0,
            cross_fade_duration=config.DEFAULT_CROSS_FADE_DURATION,
            auto_accent=True,
            output_format="wav",
        ),
    )
    try:
        # Приоритет 0: прослушивание короткое, и человек ждёт его на месте.
        job = get_queue().submit(job_payload, priority=PRIORITY_PREVIEW)
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
            qa=QaSettings.for_mode(payload.qa),
        ),
    )
    try:
        priority = PRIORITY_BACKGROUND if payload.background else PRIORITY_RENDER
        job = get_queue().submit(job_payload, priority=priority)
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
            qa=QaSettings.for_mode(payload.qa),
        ),
    )
    try:
        priority = PRIORITY_BACKGROUND if payload.background else PRIORITY_RENDER
        job = get_queue().submit(job_payload, priority=priority)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.id, "status": job.status.value, "total_replicas": job.total_replicas}


# --- сравнение движков --------------------------------------------------------
class BenchmarkRequest(BaseModel):
    """Сравнение голоса на нескольких движках: одна фраза и один reference.

    Пустой `engines` означает «все объявленные движки»: сравнить голос на всём,
    что установлено, — самый частый сценарий, и заставлять отмечать их руками
    незачем. `qa` — тот же режим проверки, что и у рендера: `off` даёт чистое
    время синтеза, `smart`/`strict` добавляют к строке результата WER.
    """

    text: str = ""
    engines: list[str] = Field(default_factory=list)
    qa: QaMode = config.QA_MODE_OFF
    auto_accent: bool = True


class BenchmarkSelectRequest(BaseModel):
    """Выбор движка, который после сравнения станет движком голоса."""

    engine: str


def _benchmark_engines(requested: list[str]) -> list[str]:
    """Список движков для сравнения; пустой — все объявленные.

    Неизвестный id — ошибка запроса, а не молчаливый пропуск: опечатка должна
    быть видна сразу, иначе она выглядит как «движок почему-то не сравнился».
    """
    engines = list(dict.fromkeys(requested)) or list(ENGINE_INFOS)
    unknown = [engine for engine in engines if engine not in ENGINE_INFOS]
    if unknown:
        known = ", ".join(sorted(ENGINE_INFOS))
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный движок синтеза: {', '.join(unknown)}. Доступны: {known}",
        )
    return engines


@app.post("/api/voices/{voice_id}/benchmark", status_code=202)
async def start_benchmark(voice_id: str, payload: BenchmarkRequest) -> dict:
    """Ставит сравнение голоса на движках в очередь единственного воркера.

    Сам синтез здесь не запускается: движки поднимаются и читают фразу строго по
    одному в том же воркере, что и рендер (см. `job_queue`). Ошибки запроса —
    400/404, а недоступность модели видна уже в строке результата и не рушит
    остальные движки.
    """
    voice = get_store().get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Голос не найден")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст для сравнения")
    if len(text) > config.MAX_REPLICA_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Текст длиннее {config.MAX_REPLICA_CHARS} символов — сократите фразу",
        )
    engines = _benchmark_engines(payload.engines)
    try:
        benchmark.check_reference(voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        job, run = get_queue().submit_benchmark(
            voice, text, engines, payload.qa, payload.auto_accent
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"benchmark_id": run.id, "job_id": job.id, "engines": engines}


@app.get("/api/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str) -> dict:
    """Результаты сравнения: статус запуска и отдельная строка на каждый движок.

    Система ничего не ранжирует: WER и время лежат в строках как справка, выбор
    движка остаётся за пользователем (`.../select`).
    """
    run = benchmark.get_run(benchmark_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Запуск сравнения не найден")
    return run.to_dict()


@app.get("/api/benchmarks/{benchmark_id}/{engine}/audio")
async def benchmark_audio(benchmark_id: str, engine: str) -> FileResponse:
    """Файл результата одного движка — для прослушивания и сравнения на слух."""
    run = benchmark.get_run(benchmark_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Запуск сравнения не найден")
    result = next((item for item in run.results if item.engine == engine), None)
    if result is None or result.audio_path is None:
        raise HTTPException(status_code=404, detail="Для этого движка нет готового результата")
    if not result.audio_path.exists():
        raise HTTPException(status_code=410, detail="Файл сравнения удалён")
    return FileResponse(result.audio_path, media_type="audio/wav")


@app.post("/api/benchmarks/{benchmark_id}/select")
async def select_benchmark_engine(benchmark_id: str, payload: BenchmarkSelectRequest) -> dict:
    """Делает выбранный в сравнении движок движком голоса.

    Меняется только движок: reference, его расшифровка и пресет остаются как
    были. Ручки прошлого движка сбрасываются самим хранилищем — они к новому
    движку не относятся (см. `voices_store.update`).
    """
    run = benchmark.get_run(benchmark_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Запуск сравнения не найден")
    engine = payload.engine
    if engine not in ENGINE_INFOS:
        known = ", ".join(sorted(ENGINE_INFOS))
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный движок синтеза: {engine or '(пусто)'}. Доступны: {known}",
        )
    succeeded = [
        item
        for item in run.results
        if item.engine == engine and item.status == benchmark.STATUS_DONE
    ]
    if not succeeded:
        raise HTTPException(
            status_code=400,
            detail=f"Движок «{ENGINE_INFOS[engine].label}» не дал результата — выбирать нечего",
        )
    try:
        voice = get_store().update(run.voice_id, engine=engine)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Голос не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return voice.to_dict()


# --- проекты ------------------------------------------------------------------
def _project_or_404(project_id: str) -> dict:
    project = get_projects_store().get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return project


def _resolved_render_settings(project: dict, payload: ProjectRenderRequest) -> tuple[RenderSettings, dict]:
    """Настройки сборки: сохранённые в проекте плюс то, что пришло в запросе.

    Возвращается и то, что нужно запомнить: повторный запуск без тела запроса
    должен собирать файл так же, как предыдущий, а не «по дефолтам».
    """
    saved = project.get("render_settings") or {}

    def pick(name: str, default):
        value = getattr(payload, name, None)
        return saved.get(name, default) if value is None else value

    pause_ms = int(pick("pause_ms", config.DEFAULT_PAUSE_MS))
    pause_ms = min(max(pause_ms, config.PAUSE_MS_RANGE[0]), config.PAUSE_MS_RANGE[1])
    cross_fade = float(pick("cross_fade_duration", config.DEFAULT_CROSS_FADE_DURATION))
    cross_fade = min(max(cross_fade, config.CROSS_FADE_RANGE[0]), config.CROSS_FADE_RANGE[1])
    output_format = str(pick("output_format", config.DEFAULT_OUTPUT_FORMAT))
    if output_format not in MIME_BY_FORMAT:
        output_format = config.DEFAULT_OUTPUT_FORMAT
    strategy = str(pick("chunk_strategy", config.CHUNK_STRATEGY_DEFAULT))
    if strategy not in (config.CHUNK_STRATEGY_SHORT, config.CHUNK_STRATEGY_PARAGRAPH):
        strategy = config.CHUNK_STRATEGY_DEFAULT
    auto_accent = bool(pick("auto_accent", True))
    # В сохранённых настройках проекта режим может лежать булевым: проекты,
    # собранные до появления Smart, хранят `true`/`false`.
    qa = _qa_mode(pick("qa", config.QA_MODE_OFF), fallback=config.QA_MODE_OFF)

    settings = RenderSettings(
        pause_ms=pause_ms,
        cross_fade_duration=cross_fade,
        auto_accent=auto_accent,
        output_format=output_format,
        qa=QaSettings.for_mode(qa),
    )
    remembered = {
        "pause_ms": pause_ms,
        "cross_fade_duration": cross_fade,
        "auto_accent": auto_accent,
        "output_format": output_format,
        "chunk_strategy": strategy,
        "qa": qa,
    }
    return settings, remembered


def _project_speakers(project: dict) -> dict[str, SpeakerSettings]:
    """Настройки синтеза по спикерам проекта.

    Значения ручек живут в overrides спикера (в первой фазе их заполняет разбор
    текста из маркеров), а голос — в отдельном поле: без него синтез невозможен,
    и его отсутствие должно быть ошибкой запроса, а не тихим дефолтом.
    """
    speakers: dict[str, SpeakerSettings] = {}
    for item in project["speakers"]:
        data = dict(item.get("overrides") or {})
        data["voice_id"] = item.get("voice_id") or ""
        speakers[item["key"]] = SpeakerSettings.from_dict(data)
    return speakers


def _replica_or_404(project: dict, index: int) -> dict:
    """Реплика проекта по её индексу (позиции в диалоге)."""
    for replica in project["replicas"]:
        if int(replica["index"]) == index:
            return replica
    raise HTTPException(status_code=404, detail="Реплика не найдена")


def _resolved_voice(voice_id: str, voices: dict) -> Voice | None:
    """Голос по идентификатору с кэшем на один ответ.

    `voices.json` читается целиком на каждый `get`, поэтому в ответе про проект
    с десятками реплик голос берётся один раз: кэш живёт внутри сборки ответа.
    Хранилище берётся из пайплайна: это тот же голос, которым читается кусок, и
    второй путь к нему разошёлся бы с тем, что уходит в модель.
    """
    if not voice_id:
        return None
    if voice_id not in voices:
        voices[voice_id] = audio_pipeline.get_store().get(voice_id)
    return voices[voice_id]


def _speaker_layer(speaker: dict | None, voice_id: str) -> SpeakerSettings:
    """Слой спикера — только его собственные правки.

    Значения, досчитанные из дефолтов движка, в слой не попадают: иначе слот при
    каждом сохранении перекрывал бы собой пресет голоса, и «настроил голос один
    раз» перестало бы работать в новом диалоге.
    """
    return SpeakerSettings.from_dict({**((speaker or {}).get("overrides") or {}), "voice_id": voice_id})


def _replica_settings(replica: dict, speaker: dict | None, voices: dict) -> dict:
    """Параметры реплики с источником каждого значения.

    Резолв тот же, что и у синтеза (`audio_pipeline.settings_view`), поэтому
    «наследуется от голоса» в карточке и «что ушло в модель» не могут разойтись.
    """
    voice_id = str(replica.get("voice_id") or (speaker or {}).get("voice_id") or "")
    voice = _resolved_voice(voice_id, voices)
    if voice is None:
        return {}
    return audio_pipeline.settings_view(
        voice, _speaker_layer(speaker, voice_id), replica.get("overrides") or {}
    )


def _speakers_payload(project: dict, voices: dict) -> list[dict]:
    """Спикеры проекта с источником каждого значения — для панели слотов.

    Правки слота идут слоем спикера, а не реплики: в этой панели они и есть
    выбор пользователя, и помечать их «правкой реплики» значило бы обещать сброс
    не туда, куда он вернёт значение.
    """
    result: list[dict] = []
    for speaker in project["speakers"]:
        voice = _resolved_voice(str(speaker.get("voice_id") or ""), voices)
        settings = (
            {}
            if voice is None
            else audio_pipeline.settings_view(voice, _speaker_layer(speaker, voice.id))
        )
        result.append({**speaker, "settings": settings})
    return result


def _takes_payload(project_id: str, index: int, replica: dict) -> list[dict]:
    """Варианты реплики: то же, что в базе, плюс ссылка на прослушивание.

    Ссылка ведёт на файл варианта, а не на готовый трек: варианты сравнивают
    между собой, и в рендере звучит только один из них. Признак `active` нужен
    интерфейсу, чтобы отметить текущее звучание — по нему же видно, что выбор
    варианта ничего не синтезирует заново.
    """
    active = replica.get("selected_take_id")
    return [
        {
            **take,
            "active": take["id"] == active,
            "audio_url": (
                f"/api/projects/{project_id}/replicas/{index}/takes/{take['id']}/audio"
            ),
        }
        for take in replica.get("takes") or []
    ]


def _replica_payload(project: dict, index: int, voices: dict | None = None) -> dict:
    """Карточка реплики для редактора: правки, голос, движок и варианты.

    Отдельно от сырой строки базы: интерфейсу нужно то, что в базе не хранится, —
    движок эффективного голоса, голос спикера (к чему вернёт сброс) и источники
    значений параметров, по которым видно, что унаследовано, а что перекрыто.
    Эту же форму отдаёт таймлайн: инспектор сегмента — карточка реплики, и второй
    сборки её полей быть не должно.
    """
    replica = _replica_or_404(project, index)
    speaker = next(
        (item for item in project["speakers"] if item["key"] == replica["speaker"]), None
    )
    return {
        **replica,
        "inherited_voice_id": speaker["voice_id"] if speaker else "",
        "engine": audio_pipeline.voice_engine(replica["voice_id"]),
        "settings": _replica_settings(replica, speaker, {} if voices is None else voices),
        "takes": _takes_payload(project["id"], index, replica),
    }


def _project_payload(project: dict) -> dict:
    """Проект целиком, но реплики — в форме карточек редактора."""
    voices: dict = {}
    return {
        **project,
        "speakers": _speakers_payload(project, voices),
        "replicas": [
            _replica_payload(project, int(replica["index"]), voices)
            for replica in project["replicas"]
        ],
    }


def _replica_render_settings(project: dict) -> RenderSettings:
    """Настройки сборки, сохранённые в проекте, — без правок из запроса.

    Пересинтез одной реплики должен звучать так же, как её соседи в последнем
    рендере, поэтому берутся именно сохранённые настройки, а не дефолты.
    """
    settings, _ = _resolved_render_settings(project, ProjectRenderRequest())
    return settings


def _timeline_payload(project: dict) -> dict:
    """Таймлайн проекта: границы реплик по реальным длительностям активных take'ов.

    Ничего не синтезирует и не меняет: длительности уже лежат в базе вместе с
    take'ами, а границы считаются на каждый запрос (`backend/timeline.py`). Так
    замена take пересчитывает хвост сама собой — хранить start/end в базе значило
    бы держать вторую правду, которая разойдётся с собранным файлом.

    Настройки берутся сохранённые в проекте (`_replica_render_settings`): ровно
    те, которыми будет собран файл, а не дефолты. Сегменты — карточки реплик в
    той же форме, что у `GET /api/projects/{id}`: инспектор таймлайна переиспользует
    карточку реплики, а не собирает её заново.
    """
    voices: dict = {}
    render = _replica_render_settings(project)
    timings = timeline.replica_timings(
        project["replicas"],
        render,
        speakers=timeline.speaker_pause_overrides(project["speakers"]),
    )
    return {
        "project_id": project["id"],
        **timeline.timeline_view(
            [
                {
                    "key": speaker["key"],
                    "label": speaker.get("label") or speaker["key"],
                    "voice_id": speaker.get("voice_id") or "",
                    "voice_name": _voice_name(speaker.get("voice_id"), voices),
                }
                for speaker in project["speakers"]
            ],
            lambda index: _replica_payload(project, index, voices),
            timings,
            render,
            timeline.timeline_duration(timings),
        ),
    }


def _voice_name(voice_id: str | None, voices: dict) -> str:
    """Имя голоса для дорожки спикера; пустой голос — пустая подпись.

    Через `_resolved_voice` (то же хранилище, что у движка и настроек реплики):
    читать `voices.json` своим способом означало бы, что «не выбран» на таймлайне
    выглядит по-разному при разных путях к голосу.
    """
    voice = _resolved_voice(str(voice_id or ""), voices)
    return voice.name if voice is not None else ""


@app.post("/api/projects", status_code=201)
async def create_project(payload: ProjectCreateRequest) -> dict:
    try:
        project = get_projects_store().create_project(
            name=payload.name,
            source_text=payload.source_text,
            mode=payload.mode,
            render_settings=payload.render_settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _project_payload(project)


@app.get("/api/projects")
async def list_projects() -> dict:
    return {"projects": get_projects_store().list_projects()}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict:
    return _project_payload(_project_or_404(project_id))


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectUpdateRequest) -> dict:
    fields = payload.model_dump(exclude_unset=True, exclude={"speakers"})
    speakers = (
        None
        if payload.speakers is None
        else {key: item.model_dump() for key, item in payload.speakers.items()}
    )
    try:
        project = get_projects_store().update_project(project_id, speakers=speakers, **fields)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Проект не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _project_payload(project)


@app.get("/api/projects/{project_id}/timeline")
async def project_timeline(project_id: str) -> dict:
    """Таймлайн проекта: где какая реплика звучит, по реальным длительностям take'ов.

    Только чтение: ни синтеза, ни записи. Границы считаются при каждом запросе,
    поэтому замена take (или выбор другого take) сразу пересчитывает хвост — в
    ответе видно и новые границы, и новый активный take каждого сегмента.
    """
    return _timeline_payload(_project_or_404(project_id))


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    if not get_projects_store().delete_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")
    return {"deleted": project_id}


@app.post("/api/projects/{project_id}/parse")
async def parse_project(project_id: str, payload: ProjectParseRequest | None = None) -> dict:
    """Разбирает исходный текст проекта в реплики и сохраняет их."""
    strategy = payload.chunk_strategy if payload else None
    try:
        project = get_projects_store().parse_project(project_id, strategy)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Проект не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _project_payload(project)


@app.post("/api/projects/{project_id}/render", status_code=202)
async def render_project(project_id: str, payload: ProjectRenderRequest | None = None) -> dict:
    """Ставит в очередь рендер проекта сохранёнными репликами и голосами."""
    payload = payload or ProjectRenderRequest()
    project = _project_or_404(project_id)
    if not project["replicas"]:
        raise HTTPException(
            status_code=400, detail="В проекте нет реплик — сначала разберите текст"
        )

    speakers = _project_speakers(project)
    missing = sorted(
        key
        for key in {replica["speaker"] for replica in project["replicas"]}
        if not speakers.get(key) or not speakers[key].voice_id
    )
    if missing:
        names = ", ".join(f"«{key}»" for key in missing)
        raise HTTPException(status_code=400, detail=f"Не назначен голос для: {names}")

    settings, remembered = _resolved_render_settings(project, payload)
    get_projects_store().update_project(project_id, render_settings=remembered)

    job_payload = JobPayload(
        replicas=[
            Replica(
                voice=replica["speaker"],
                text=replica["text"],
                line_number=index + 1,
                overrides=replica["overrides"],
                # Свой голос реплики: NULL в базе — «как у спикера».
                voice_id=replica["voice_override"],
            )
            for index, replica in enumerate(project["replicas"])
        ],
        speakers=speakers,
        settings=settings,
        project_id=project_id,
    )
    try:
        priority = PRIORITY_BACKGROUND if payload.background else PRIORITY_RENDER
        job = get_queue().submit(job_payload, priority=priority)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    get_projects_store().set_project_status(
        project_id, config.PROJECT_STATUS_RENDERING, job.id
    )
    return {"job_id": job.id, "status": job.status.value, "total_replicas": job.total_replicas}


@app.patch("/api/projects/{project_id}/replicas/{index}")
async def update_replica(project_id: str, index: int, payload: ReplicaUpdateRequest) -> dict:
    """Правит одну реплику: голос, отдельные параметры или сброс правок.

    Отсутствие поля и `null` — разные вещи: непришедшее значит «не трогать», а
    `null` — «вернуть наследуемое у спикера». Поэтому смотрим на `exclude_unset`,
    а не на значения: иначе карточка не смогла бы отличить «сбросить скорость»
    от «скорость не меняли».
    """
    fields = payload.model_dump(exclude_unset=True)
    try:
        project = get_projects_store().patch_replica(
            project_id,
            index,
            voice_id=fields.get("voice_id", UNSET),
            overrides=fields.get("overrides"),
            reset_overrides=bool(fields.get("reset_overrides")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Реплика не найдена") from exc
    return {"project_id": project_id, "replica": _replica_payload(project, index)}


@app.post("/api/projects/{project_id}/replicas/{index}/regenerate", status_code=202)
async def regenerate_project_replica(project_id: str, index: int) -> dict:
    """Пересинтезирует одну реплику проекта — остальные не трогаются.

    Результат сохраняется вариантом реплики и становится активным: прежнее
    звучание остаётся доступным, и сравнить две версии можно на слух.
    """
    project = _project_or_404(project_id)
    replica = _replica_or_404(project, index)
    if not replica["voice_id"]:
        raise HTTPException(
            status_code=400, detail=f"Не назначен голос для «{replica['label']}»"
        )
    try:
        job = get_queue().submit_project_take(
            project_id=project_id,
            index=index,
            replica=Replica(
                voice=replica["speaker"],
                text=replica["text"],
                line_number=index + 1,
                overrides=replica["overrides"],
                voice_id=replica["voice_override"],
            ),
            speakers=_project_speakers(project),
            settings=_replica_render_settings(project),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_id": job.id, "index": index, "status": job.status.value}


@app.post("/api/projects/{project_id}/replicas/{index}/takes/{take_id}")
async def select_project_take(project_id: str, index: int, take_id: int) -> dict:
    """Ставит выбранный вариант активным — без повторного синтеза.

    Вариант — это готовое аудио: выбор между ними должен быть мгновенным, иначе
    сравнение двух версий на слух превращается в ожидание модели.
    """
    try:
        project = get_projects_store().select_take(project_id, index, take_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Вариант реплики не найден") from exc
    return {"project_id": project_id, "replica": _replica_payload(project, index)}


@app.get("/api/projects/{project_id}/replicas/{index}/takes/{take_id}/audio")
async def project_take_audio(project_id: str, index: int, take_id: int) -> FileResponse:
    """Отдаёт файл варианта реплики — для прослушивания и сравнения на слух."""
    take = get_projects_store().get_take(project_id, index, take_id)
    if take is None:
        raise HTTPException(status_code=404, detail="Вариант реплики не найден")
    path = Path(take["audio_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="Файл варианта удалён")
    # Варианты всегда wav: они служат для сравнения, а не для выдачи пользователю.
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Отменяет задачу: ожидающую — сразу, идущую — на безопасной точке.

    Повторная отмена — не ошибка: ответ тот же, а состояние задачи не меняется
    (`done`/`error`/`cancelled` остаются собой). Неизвестный id — 404.
    """
    job = get_queue().cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {"job": job.to_dict()}


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
                # Итог строгой проверки этого звучания; None — проверка не гонялась.
                "qa": _qa_payload(queue.active_qa(job, index)),
                "variants": _variants_payload(job_id, index, job),
            }
            for index, replica in enumerate(job.payload.replicas)
        ]
    return data


def _qa_payload(outcome: QaOutcome | None) -> dict | None:
    """Итог строгой проверки для интерфейса; None — кусок не проверялся."""
    return None if outcome is None else outcome.to_dict()


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
