"""Реестр моделей проекта: что уже лежит на диске, что поднято и сколько занимает.

Задача модуля — убрать ручное управление файлами весов. Он ничего не решает про
инференс: движки по-прежнему поднимаются лениво и качают свои веса сами
(`tts_engine._resolve_weights`), а здесь только показывается состояние, по
запросу скачивается репозиторий и удаляются файлы модели.

Ключевые решения:

* пути к моделям читаются из `config` **в момент вызова** (`model_specs()`), а не
  на импорте: тесты подменяют `config.MODELS_DIR`, и закешированный путь сделал
  бы подмену бессмысленной;
* сеть — единственная точка входа `_hf_download`: её мокают тесты, поэтому
  реальные скачивания в прогоне невозможны;
* ничто здесь не запускается само при старте сервера: скачивание инициирует
  пользователь, а `MODELS_DIR` только читается;
* любой путь модели обязан лежать внутри `config.MODELS_DIR` (для `kind="local"`):
  настройка, уводящая путь наружу, отклоняется до чтения и записи файлов.

Whisper живёт не в `models/`, а в кеше `huggingface_hub`, и его наличие
определяется чтением кеша без сети (`try_to_load_from_cache`).
"""

import logging
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import config
from .engines.base import ENGINE_F5, ENGINE_XTTS, ENGINE_XTTS_BANANA, engine_info
from .engines.registry import created_engines

logger = logging.getLogger(__name__)

# Модели, чьи файлы лежат в `MODELS_DIR` и раскладываются как в HF-репозитории.
KIND_LOCAL = "local"
# Модель, живущая в кеше `huggingface_hub` (её качает сама библиотека модели).
KIND_CACHE = "cache"

# Состояния скачивания. `idle` — «ничего не происходило», `interrupted` —
# процесс упал или был прерван, не доведя дело до конца.
DOWNLOAD_IDLE = "idle"
DOWNLOAD_DOWNLOADING = "downloading"
DOWNLOAD_DONE = "done"
DOWNLOAD_ERROR = "error"
DOWNLOAD_INTERRUPTED = "interrupted"

# Идентификатор модели Whisper — тот же, что в transcribe_worker.ASR_MODEL_ID.
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"

# Как часто пересчитывать занятое место во время скачивания. Чаще незачем:
# обход каталога с гигабайтными файлами — это десятки `stat`, а прогресс нужен
# глазу, а не измерительному прибору.
PROGRESS_INTERVAL_SEC = 2.0

# Скачивания идут строго по одному: два потока одновременно писали бы в один
# каталог моделей и мешали друг другу.
_download_lock = threading.Lock()
# Уже запущенные фоновые потоки скачивания — чтобы повторный POST не портил
# состояние работающего скачивания.
_active_downloads: set[str] = set()
_active_lock = threading.Lock()


class ModelPathError(ValueError):
    """Путь модели выведен за пределы `MODELS_DIR` — операция отклонена."""


class ModelBusyError(RuntimeError):
    """Модель занята: движок поднят или очередь синтеза занята задачей."""


class ModelNotDownloadableError(RuntimeError):
    """Модель не скачивается этим менеджером (кеш `huggingface_hub`)."""


@dataclass(frozen=True)
class ModelSpec:
    """Паспорт модели: где лежит, что обязательно и каким движком используется.

    `required_files` — относительные пути **или** шаблоны (`model.pth`), по
    которым файл ищется в каталоге модели на любой глубине: banana-файнтюн
    раскладывается по вложенным папкам, и жёсткий путь сломался бы при
    обновлении репозитория.
    """

    id: str
    label: str
    description: str
    repo_id: str
    local_dir: Path
    required_files: tuple[str, ...]
    approx_size_bytes: int
    engine_id: str | None
    kind: str = KIND_LOCAL

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "repo_id": self.repo_id,
            "required_files": list(self.required_files),
            "approx_size_bytes": self.approx_size_bytes,
            "engine_id": self.engine_id,
            "kind": self.kind,
        }


@dataclass
class DownloadState:
    """Состояние скачивания одной модели — то, что опрашивает интерфейс."""

    state: str = DOWNLOAD_IDLE
    progress: float = 0.0
    bytes_downloaded: int = 0
    bytes_total: int = 0
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "progress": round(self.progress, 4),
            "bytes_downloaded": self.bytes_downloaded,
            "bytes_total": self.bytes_total,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class ModelState:
    """Полное состояние модели в один момент времени."""

    spec: ModelSpec
    path: Path
    installed: bool
    missing_files: list[str] = field(default_factory=list)
    size_bytes: int = 0
    loaded: bool = False
    engine_state: str | None = None
    engine_in_use: bool = False
    download: DownloadState = field(default_factory=DownloadState)

    def to_dict(self) -> dict:
        return {
            **self.spec.to_dict(),
            "path": str(self.path),
            "installed": self.installed,
            "missing_files": list(self.missing_files),
            "size_bytes": self.size_bytes,
            "loaded": self.loaded,
            "engine_state": self.engine_state,
            "engine_in_use": self.engine_in_use,
            "download": self.download.to_dict(),
        }


# --- паспорта моделей ---------------------------------------------------------
def model_specs() -> tuple[ModelSpec, ...]:
    """Все модели проекта. Пути читаются из `config` при каждом вызове.

    Поэтому подмена `config.MODELS_DIR` в тестах (или `TTS_MODELS_DIR` в среде)
    действительно меняет реестр, а не остаётся незамеченной.
    """
    return (
        ModelSpec(
            id="f5",
            label=engine_info(ENGINE_F5).label,
            description=(
                "Основной движок проекта: русский файнтюн F5-TTS с разметкой ударений. "
                "Два файла — чекпоинт и словарь."
            ),
            repo_id=config.HF_REPO_ID,
            local_dir=config.MODELS_DIR,
            # Пути взяты из тех же констант, по которым ходит авто-догрузка
            # `tts_engine._resolve_weights`: реестр и движок обязаны смотреть на
            # одни и те же файлы, иначе интерфейс показывал бы не то, что грузится.
            required_files=(config.HF_CKPT_PATH, config.HF_VOCAB_PATH),
            approx_size_bytes=1_400_000_000,
            engine_id=ENGINE_F5,
        ),
        ModelSpec(
            id=ENGINE_XTTS,
            label=engine_info(ENGINE_XTTS).label,
            description="Многоязычная модель Coqui: клонирование по 6–15 с референса.",
            repo_id="coqui/XTTS-v2",
            local_dir=config.XTTS_BASE_DIR,
            required_files=("model.pth", "config.json"),
            approx_size_bytes=1_900_000_000,
            engine_id=ENGINE_XTTS,
        ),
        ModelSpec(
            id=ENGINE_XTTS_BANANA,
            label=engine_info(ENGINE_XTTS_BANANA).label,
            description="Русский комьюнити-файнтюн XTTS: разговорные ударения, ~5.2 ГБ.",
            repo_id="Ftfyhh/xttsv2_banana",
            local_dir=config.XTTS_BANANA_DIR,
            required_files=("model.pth", "config.json"),
            approx_size_bytes=5_200_000_000,
            engine_id=ENGINE_XTTS_BANANA,
        ),
        ModelSpec(
            id="whisper",
            label="Whisper large-v3-turbo",
            description=(
                "Распознавание речи: расшифровка референса и строгая проверка синтеза. "
                "Лежит в кеше huggingface_hub, а не в models/."
            ),
            repo_id=WHISPER_MODEL_ID,
            # Значение по умолчанию — только запасной путь: настоящий каталог
            # снапшота вычисляется в состоянии модели (см. `_model_dir`).
            local_dir=_hf_cache_dir() / "models--openai--whisper-large-v3-turbo",
            required_files=("config.json", "model.safetensors"),
            approx_size_bytes=1_600_000_000,
            # Движка синтеза у Whisper нет: он работает отдельным процессом и в
            # реестре движков не появляется.
            engine_id=None,
            kind=KIND_CACHE,
        ),
    )


def model_spec(model_id: str) -> ModelSpec:
    """Паспорт по id. Неизвестный id — `KeyError`: вызывающий сам решает про 404."""
    for spec in model_specs():
        if spec.id == model_id:
            return spec
    raise KeyError(model_id)


def _hf_cache_dir() -> Path:
    """Каталог кеша huggingface_hub. Читается без сети и без импорта тяжёлых вещей."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — без библиотеки функция всё равно должна работать
        return Path.home() / ".cache" / "huggingface" / "hub"


# --- пути и файлы -------------------------------------------------------------
def _inside(path: Path, root: Path) -> bool:
    """Лежит ли `path` внутри `root` (с учётом `..`, `.` и симлинков).

    `resolve()` разворачивает и относительные сегменты, и симлинки, поэтому
    проверка не обманывается ни `../..`, ни ссылкой наружу. Если каталога ещё
    нет, `resolve()` работает как `absolute()` — путь всё равно нормализуется.
    """
    try:
        resolved = Path(path).resolve()
        resolved_root = Path(root).resolve()
    except OSError:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def validate_model_path(spec: ModelSpec) -> Path:
    """Путь модели, если он внутри каталога моделей; иначе — понятная ошибка.

    Единственное место, где проверяется граница `MODELS_DIR`. Все операции
    (чтение состояния, скачивание, удаление) обязаны идти через него, иначе
    неверная настройка окружения позволила бы писать или удалять файлы за
    пределами каталога моделей.
    """
    if spec.kind != KIND_LOCAL:
        return spec.local_dir
    root = Path(config.MODELS_DIR)
    if not _inside(spec.local_dir, root):
        raise ModelPathError(
            f"Путь модели «{spec.label}» ({spec.local_dir}) выходит за пределы каталога "
            f"моделей ({root}). Подпути вроде «..» запрещены: проверьте настройки "
            f"TTS_MODELS_DIR и каталоги моделей."
        )
    return spec.local_dir


def _match_required(path: Path, pattern: str) -> Path | None:
    """Ищет обязательный файл: сначала точный относительный путь, затем по имени.

    Рекурсивный поиск — не «на всякий случай»: huggingface_hub раскладывает
    banana-файнтюн по вложенным папкам (`model_banana/v2.0.2/model.pth`), и
    перекладывать файлы руками пользователь не должен.
    """
    exact = path / pattern
    if exact.is_file():
        return exact
    name = Path(pattern).name
    if not path.is_dir():
        return None
    return next((found for found in sorted(path.rglob(name)) if found.is_file()), None)


def missing_files(spec: ModelSpec, path: Path | None = None) -> list[str]:
    """Обязательные файлы, которых нет в каталоге модели.

    `path` передаётся снаружи, чтобы не проверять один и тот же каталог дважды
    в пределах одного снимка состояния.
    """
    if spec.kind == KIND_CACHE:
        return _missing_cache_files(spec)
    root = Path(path) if path is not None else spec.local_dir
    return [pattern for pattern in spec.required_files if _match_required(root, pattern) is None]


def required_paths(spec: ModelSpec, path: Path | None = None) -> list[Path]:
    """Найденные на диске обязательные файлы модели.

    По ним считается «свой» размер модели: у F5 файлы лежат прямо в `models/`,
    рядом с весами других моделей, и размер всего каталога был бы чужой.
    """
    root = Path(path) if path is not None else _model_dir(spec)
    if spec.kind == KIND_CACHE:
        return [
            Path(found)
            for name in spec.required_files
            if (found := _cache_lookup(_cache_loader(), spec.repo_id, name)) is not None
        ]
    found_files = []
    for pattern in spec.required_files:
        found = _match_required(root, pattern)
        if found is not None:
            found_files.append(found)
    return found_files


def _cache_loader():
    """`try_to_load_from_cache`, если библиотека доступна; иначе заглушка."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return lambda *_args: None
    return try_to_load_from_cache


def _missing_cache_files(spec: ModelSpec) -> list[str]:
    """Отсутствующие файлы снапшота в кеше HF — рекурсивных путей там нет.

    `try_to_load_from_cache` только читает кеш: сети он не касается и в
    офлайне работает так же.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 — без библиотеки считаем, что модели нет
        return list(spec.required_files)

    missing: list[str] = []
    for name in spec.required_files:
        cached = _cache_lookup(try_to_load_from_cache, spec.repo_id, name)
        if not isinstance(cached, str) or not Path(cached).is_file():
            missing.append(name)
    return missing


def _cache_lookup(loader: Callable, repo_id: str, filename: str):
    """Значение из кеша HF. `_CACHED_NO_EXIST` и пропуски приводим к None."""
    try:
        value = loader(repo_id, filename)
    except Exception:  # noqa: BLE001 — кеш может быть битым; это не повод падать
        return None
    return value if isinstance(value, str) else None


def _cache_snapshot_dir(repo_id: str, names: Iterable[str]) -> Path | None:
    """Каталог снапшота в кеше HF: его путь содержит хеш ревизии и угадать его нельзя.

    Берётся каталог первого найденного файла: у одного снапшота все файлы лежат
    рядом, а `try_to_load_from_cache` читает только кеш и в сеть не ходит.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 — без библиотеки считаем, что кеша нет
        return None
    for name in names:
        cached = _cache_lookup(try_to_load_from_cache, repo_id, name)
        if cached is not None:
            return Path(cached).parent
    return None


def dir_size(path: Path, follow_symlinks: bool = False) -> int:
    """Реально занятое место в каталоге.

    По умолчанию симлинки не разворачиваются: в кеше `huggingface_hub` файлы
    снапшота — это ссылки на блобы, и разыменование посчитало бы одни и те же
    байты дважды. Для отдельного файла (`follow_symlinks=True`) размер берётся у
    того, на что ссылка указывает, — иначе кешированная модель «весила» бы ноль.
    Недоступные файлы (гонка с очисткой кеша) пропускаются, а не роняют запрос.
    """
    try:
        if path.is_symlink() and not follow_symlinks:
            return 0
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() or (follow_symlinks and entry.is_symlink()):
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _model_dir(spec: ModelSpec) -> Path:
    """Каталог, в котором лежат файлы модели.

    Для локальной модели это настройка из `config`, для кеша — каталог снапшота,
    который находит `huggingface_hub` (путь содержит хеш ревизии).
    """
    if spec.kind != KIND_CACHE:
        return spec.local_dir
    return _cache_snapshot_dir(spec.repo_id, spec.required_files) or spec.local_dir


# Размеры каталогов кешируются: обход `models/` с файлами на 5 ГБ — это десятки
# `stat`, а интерфейс опрашивает список моделей регулярно. Значение меняется
# только при скачивании и удалении, поэтому минуты между замерами достаточно.
SIZE_CACHE_TTL_SEC = 60.0
_size_cache: dict[str, tuple[float, int]] = {}
_size_cache_lock = threading.Lock()


def _model_size_bytes(spec: ModelSpec, root: Path) -> int:
    """Сколько занимает сама модель — по её обязательным файлам.

    Не размер каталога: F5 живёт прямо в общем `models/`, и каталог включал бы
    веса других моделей. У кеша HF файлы снапшота — симлинки на блобы, поэтому
    размер берётся у самих блобов (`required_paths` возвращает уже развёрнутые
    пути), и одни и те же байты не считаются дважды.
    """
    return sum(dir_size(found, follow_symlinks=True) for found in required_paths(spec, root))


def cached_dir_size(path: Path) -> int:
    """`dir_size` с коротким кешем — для регулярного опроса из интерфейса."""
    try:
        key = str(Path(path))
    except TypeError:  # pragma: no cover - защита от нестрокового пути
        return 0
    now = time.monotonic()
    with _size_cache_lock:
        cached = _size_cache.get(key)
        if cached is not None and now - cached[0] < SIZE_CACHE_TTL_SEC:
            return cached[1]
    size = dir_size(path)
    with _size_cache_lock:
        _size_cache[key] = (now, size)
    return size


def forget_dir_size(path: Path) -> None:
    """Сбрасывает замер каталога: после скачивания и удаления он уже неверен."""
    with _size_cache_lock:
        _size_cache.pop(str(Path(path)), None)


# --- состояние движков --------------------------------------------------------
def _engine_status(engine_id: str | None) -> tuple[bool, str | None, bool]:
    """`(loaded, state, in_use)` для движка модели — без создания нового экземпляра.

    `created_engines()` отдаёт только уже созданные движки: проверка состояния
    модели не должна поднимать модель в память (иначе открытие вкладки
    «Модели» грузило бы XTTS на 5 ГБ).
    """
    if engine_id is None:
        return False, None, False
    engine = created_engines().get(engine_id)
    if engine is None:
        return False, None, False
    return engine.is_loaded, engine.state, False


def engine_in_use(engine_id: str | None) -> bool:
    """Занят ли движок прямо сейчас: поднят или в очереди идёт задача с ним.

    Операция с файлами модели во время синтеза — это чтение `model.pth` из-под
    работающего движка; поэтому проверка нужна и для загруженной модели, и для
    очереди. Это и есть тот запрет, которым `DELETE /api/models/{id}` отказывает
    в удалении занятой модели.
    """
    if engine_id is None:
        return False
    loaded, _state, _ = _engine_status(engine_id)
    if loaded:
        return True
    return _queue_uses_engine(engine_id)


def _queue_uses_engine(engine_id: str) -> bool:
    """Есть ли в очереди задача, которая синтезирует этим движком."""
    try:
        from .job_queue import get_queue
        from .voices_store import get_store
    except Exception:  # noqa: BLE001 — без очереди считаем, что занятости нет
        return False
    queue = get_queue()
    job_id = getattr(queue, "current_job_id", None)
    if job_id is None:
        return False
    job = queue.get(job_id)
    payload = getattr(job, "payload", None)
    if payload is None:
        return False
    try:
        store = get_store()
        for replica in payload.replicas:
            voice = store.get(replica.voice)
            if voice is not None and voice.engine == engine_id:
                return True
    except Exception:  # noqa: BLE001 — хранилище может быть недоступно; не выдумываем занятость
        return False
    return False


# --- менеджер -----------------------------------------------------------------
class ModelManager:
    """Снимок состояния всех моделей плюс скачивание и удаление.

    Скачивание идёт в фоновом потоке (это сеть и диск, а не инференс), но
    очередь синтеза здесь ни при чём: менеджер не ставит задач и не поднимает
    движки.
    """

    def __init__(self) -> None:
        self._states: dict[str, DownloadState] = {}
        self._lock = threading.Lock()

    # -- состояние -------------------------------------------------------------
    def state(self, spec: ModelSpec) -> ModelState:
        """Состояние одной модели. Исключения путей наружу не пробрасываются:
        неверная настройка должна показываться в интерфейсе, а не ломать список."""
        try:
            validate_model_path(spec)
        except ModelPathError as exc:
            return ModelState(
                spec=spec,
                path=spec.local_dir,
                installed=False,
                missing_files=list(spec.required_files),
                engine_state=_engine_status(spec.engine_id)[1],
                download=_failed_download(exc),
            )
        root = _model_dir(spec)
        missing = missing_files(spec, root)
        loaded, engine_state, _ = _engine_status(spec.engine_id)
        return ModelState(
            spec=spec,
            path=root,
            installed=not missing,
            missing_files=missing,
            size_bytes=_model_size_bytes(spec, root),
            loaded=loaded,
            engine_state=engine_state,
            engine_in_use=engine_in_use(spec.engine_id),
            download=self.download_state(spec.id),
        )

    def states(self) -> list[ModelState]:
        return [self.state(spec) for spec in model_specs()]

    def models(self) -> list[dict]:
        return [state.to_dict() for state in self.states()]

    def get(self, model_id: str) -> dict:
        """Одна модель по id. Неизвестный id — `KeyError` (роут отдаёт 404)."""
        return self.state(model_spec(model_id)).to_dict()

    def download_state(self, model_id: str) -> DownloadState:
        with self._lock:
            current = self._states.get(model_id)
            if current is None:
                return DownloadState()
            # Копия: поток скачивания продолжит менять оригинал, а читатель
            # получит согласованный снимок.
            return replace(current)

    def _set_download(self, model_id: str, state: DownloadState) -> None:
        with self._lock:
            self._states[model_id] = state

    # -- занятость -------------------------------------------------------------
    @staticmethod
    def guard_delete(spec: ModelSpec) -> None:
        """Отказ на удаление занятой модели — до любого касания файлов.

        Выгрузить движок теперь можно самому (`POST /api/engines/{id}/unload`
        или кнопка «Выгрузить» во вкладке «Модели»): после выгрузки вес никто не
        читает, и файлы удаляются. Пока движок поднят или в очереди идёт задача,
        удаление отклоняется — текст говорит, что именно сделать.
        """
        if engine_in_use(spec.engine_id):
            raise ModelBusyError(
                f"Модель «{spec.label}» сейчас используется движком. Дождитесь "
                f"окончания задачи в очереди, затем выгрузите движок кнопкой "
                f"«Выгрузить» во вкладке «Модели» (или POST /api/engines/"
                f"{spec.engine_id}/unload) и повторите удаление: пока модель "
                f"поднята, её файлы читает синтез."
            )

    # -- скачивание ------------------------------------------------------------
    def download(self, model_id: str) -> dict:
        """Запускает скачивание модели в фоне и сразу отдаёт её состояние (202).

        Уже установленная модель не скачивается повторно: файлы остаются
        нетронутыми (идемпотентный no-op), а состояние помечается `done`.
        """
        spec = model_spec(model_id)
        validate_model_path(spec)
        if spec.kind != KIND_LOCAL:
            raise ModelNotDownloadableError(
                f"Модель «{spec.label}» живёт в кеше huggingface_hub: её скачивает сама "
                f"библиотека при первом распознавании. Управлять ею из дашборда нельзя."
            )
        if not missing_files(spec):
            # Идемпотентный no-op: файлы уже на месте, и трогать их незачем —
            # повторное скачивание не переписывает веса и не меняет mtime.
            forget_dir_size(spec.local_dir)
            self._set_download(
                model_id,
                DownloadState(
                    state=DOWNLOAD_DONE,
                    progress=1.0,
                    bytes_downloaded=sum(
                        dir_size(found, follow_symlinks=True) for found in required_paths(spec)
                    ),
                    bytes_total=spec.approx_size_bytes,
                    finished_at=time.time(),
                ),
            )
            return self.state(spec).to_dict()
        with _active_lock:
            if model_id in _active_downloads:
                return self.state(spec).to_dict()
            _active_downloads.add(model_id)
        forget_dir_size(spec.local_dir)
        # Сколько уже занято до скачивания: прогресс считается по приросту, а не
        # по полному размеру каталога. Иначе у F5 (его файлы лежат прямо в
        # `models/`, рядом с чужими весами) прогресс начинался бы с чужого места.
        baseline = dir_size(spec.local_dir)
        self._set_download(
            model_id,
            DownloadState(
                state=DOWNLOAD_DOWNLOADING,
                progress=0.0,
                bytes_downloaded=0,
                bytes_total=spec.approx_size_bytes,
                started_at=time.time(),
            ),
        )
        thread = threading.Thread(
            target=self._run_download,
            args=(spec, baseline),
            name=f"model-download-{model_id}",
            daemon=True,
        )
        thread.start()
        return self.state(spec).to_dict()

    def _run_download(self, spec: ModelSpec, baseline: int) -> None:
        started = self._states.get(spec.id)
        started_at = started.started_at if started else time.time()
        try:
            self._download_locked(spec, started_at, baseline)
        except Exception as exc:  # noqa: BLE001 — состояние ошибки важнее трейсбека
            logger.error("Не удалось скачать модель %s: %s", spec.id, exc)
            self._finish(
                spec,
                DOWNLOAD_ERROR,
                error=f"Не удалось скачать модель «{spec.label}»: {exc}",
                started_at=started_at,
            )
        finally:
            with _active_lock:
                _active_downloads.discard(spec.id)

    def _download_locked(self, spec: ModelSpec, started_at: float, baseline: int) -> None:
        """Скачивание под глобальным слотом: одновременно идёт не больше одного."""
        with _download_lock:
            spec.local_dir.mkdir(parents=True, exist_ok=True)
            reported_at = 0.0
            for filename in spec.required_files:
                if _match_required(spec.local_dir, filename) is not None:
                    continue  # файл уже на месте — докачиваем только недостающее
                _hf_download(spec.repo_id, filename, spec.local_dir)
                reported_at = self._update_progress(spec, started_at, reported_at, baseline)

            missing = missing_files(spec, spec.local_dir)
            if missing:
                self._finish(
                    spec,
                    DOWNLOAD_ERROR,
                    error=(
                        f"Скачивание завершилось, но файлы не найдены: {', '.join(missing)}. "
                        f"Проверьте содержимое {spec.local_dir}"
                    ),
                    started_at=started_at,
                )
                return
            self._finish(spec, DOWNLOAD_DONE, started_at=started_at)

    def _update_progress(
        self, spec: ModelSpec, started_at: float, reported_at: float, baseline: int
    ) -> float:
        """Дописывает прогресс по реально занятому месту, не выдумывая числа.

        `bytes_downloaded` — прирост относительно состояния до скачивания,
        `bytes_total` — ожидаемый размер репозитория. Числа не синтетические:
        и прирост, и потолок берутся с диска и из паспорта модели.
        """
        now = time.monotonic()
        if now - reported_at < PROGRESS_INTERVAL_SEC:
            return reported_at
        forget_dir_size(spec.local_dir)
        downloaded = max(0, dir_size(spec.local_dir) - baseline)
        total = spec.approx_size_bytes
        self._set_download(
            spec.id,
            DownloadState(
                state=DOWNLOAD_DOWNLOADING,
                progress=min(1.0, downloaded / total) if total else 0.0,
                bytes_downloaded=downloaded,
                bytes_total=total,
                started_at=started_at,
            ),
        )
        return now

    def _finish(
        self,
        spec: ModelSpec,
        state: str,
        error: str | None = None,
        started_at: float | None = None,
    ) -> None:
        forget_dir_size(spec.local_dir)
        downloaded = sum(dir_size(found, follow_symlinks=True) for found in required_paths(spec))
        self._set_download(
            spec.id,
            DownloadState(
                state=state,
                progress=1.0 if state == DOWNLOAD_DONE else 0.0,
                bytes_downloaded=downloaded,
                bytes_total=spec.approx_size_bytes,
                error=error,
                started_at=started_at,
                finished_at=time.time(),
            ),
        )

    # -- удаление --------------------------------------------------------------
    def delete(self, model_id: str) -> dict:
        """Удаляет файлы модели. Занятую модель не трогает (см. `guard_delete`).

        Порядок проверок важен: сначала «этой модели нет», затем «ею нельзя
        управлять из дашборда», затем «она занята», и только потом путь. Роут
        отдаёт по ним разные коды, и «занята» должен побеждать «путь запрещён»:
        это более полезная причина для пользователя.
        """
        spec = model_spec(model_id)
        if spec.kind != KIND_LOCAL:
            raise ModelNotDownloadableError(
                f"Модель «{spec.label}» лежит в кеше huggingface_hub и управляется "
                f"библиотекой распознавания, а не дашбордом. Удалите её через "
                f"`huggingface-cli delete-cache`, если она больше не нужна."
            )
        self.guard_delete(spec)
        path = validate_model_path(spec)
        forget_dir_size(path)
        freed = dir_size(path, follow_symlinks=True)
        if path.exists():
            # Удаляется ровно каталог модели: он уже проверен на принадлежность
            # MODELS_DIR, поэтому за его пределами ничего не пострадает.
            shutil.rmtree(path)
        forget_dir_size(path)
        with self._lock:
            self._states.pop(model_id, None)
        logger.info("Модель %s удалена (%s байт)", model_id, freed)
        return {"deleted": model_id, "freed_bytes": freed, "state": self.state(spec).to_dict()}

    # -- диск ------------------------------------------------------------------
    def disk_report(self, states: Iterable[ModelState] | None = None) -> dict:
        """Сколько занимают модели и сколько осталось на диске."""
        total = sum(state.size_bytes for state in (states if states is not None else self.states()))
        try:
            usage = shutil.disk_usage(config.MODELS_DIR)
            free_bytes, disk_total = usage.free, usage.total
        except OSError:
            free_bytes, disk_total = 0, 0
        return {"models_bytes": total, "free_bytes": free_bytes, "total_bytes": disk_total}


def _failed_download(exc: Exception) -> DownloadState:
    """Состояние «путь запрещён» в формате обычного состояния скачивания."""
    return DownloadState(state=DOWNLOAD_ERROR, error=str(exc), finished_at=time.time())


def _hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Единственная точка скачивания — её и мокают тесты.

    `hf_hub_download` с `local_dir` раскладывает файлы по путям репозитория, то
    есть каталог модели повторяет структуру HF (её и ждут движки).
    """
    from huggingface_hub import hf_hub_download

    logger.info("Скачиваю %s/%s в %s", repo_id, filename, local_dir)
    return Path(hf_hub_download(repo_id, filename, local_dir=str(local_dir)))


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    """Единственный менеджер на процесс: состояние скачиваний живёт в нём."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelManager()
    return _manager


def reset_manager() -> None:
    """Сбрасывает состояние скачиваний — нужно тестам, чтобы они не влияли друг на друга."""
    global _manager
    with _manager_lock:
        _manager = None
    with _active_lock:
        _active_downloads.clear()
    with _size_cache_lock:
        _size_cache.clear()
