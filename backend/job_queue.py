"""Очередь задач: один воркер, генерация строго последовательная."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from . import audio_pipeline, benchmark, config, eta, resource_guard, take_quality
from . import short_utterance as su
from .audio_pipeline import (
    NoteCallback,
    ProgressCallback,
    QaOutcome,
    RenderSettings,
    SpeakerSettings,
)
from .db.store import get_projects_store
from .dialogue_parser import Replica
from .engines import worker_protocol as proto
from .engines.base import STATE_LOADING, STATE_READY
from .engines.registry import created_engine, created_engines
from .llm import memory_policy as llm_memory
from .voices_store import Voice

logger = logging.getLogger(__name__)

MAX_KEPT_JOBS = 50
# Как часто перепроверять системную память, пока задача ждёт своей очереди.
MEMORY_WAIT_RETRY_SEC = 5.0

# Метка варианта, сохранённого из прерванного рендера: по ней видно, что это
# остаток незавершённой сборки, а не обычный результат (creash_report §13).
PARTIAL_TAKE_LABEL = "прерванный рендер"


class _RenderFailed(RuntimeError):
    """Рендер не удалось восстановить: попытки исчерпаны (внутреннее исключение).

    Нужно, чтобы «задача упала из-за падения воркера» отличалось от «задача упала
    из-за ошибки модели»: у первого есть тип отказа, число попыток и готовые
    куски, у второго — только текст ошибки.
    """

    def __init__(
        self,
        cause: BaseException,
        partial: "audio_pipeline.RenderPartial | None",
        error_type: str,
        attempts: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.partial = partial
        self.error_type = error_type
        self.attempts = attempts

# Приоритеты очереди (фаза 11). Воркер по-прежнему один: приоритет меняет только
# порядок, в котором задачи доходят до модели, а не число одновременных
# прогонов. Меньше число — раньше задача.
PRIORITY_PREVIEW = 0        # прослушивание голоса: короткое и интерактивное
PRIORITY_REGENERATE = 1     # пересинтез одной реплики: человек ждёт результат
PRIORITY_RENDER = 2         # обычная сборка файла
PRIORITY_BACKGROUND = 3     # сравнение движков и рендер, явно помеченный фоном

# Приоритет по виду задачи — для тех, кто ставит задачу без явного класса.
PRIORITY_BY_TASK = {
    "RenderTask": PRIORITY_RENDER,
    "RegenerateTask": PRIORITY_REGENERATE,
    "SelectVariantTask": PRIORITY_REGENERATE,
    "ProjectTakeTask": PRIORITY_REGENERATE,
    "BenchmarkTask": PRIORITY_BACKGROUND,
}

# Сообщения об отмене — их читает интерфейс, поэтому они на русском.
CANCEL_QUEUED_MESSAGE = "Отменено до запуска"
CANCEL_PROCESSING_MESSAGE = "Отменено: прервано на ближайшей безопасной точке"
# Кто остановил задачу. Сигнал в пайплайн один, а исход разный (см.
# `_finish_interrupted`): отмена — состояние `cancelled`, прерывание
# watchdog'ом — ошибка с причиной.
ABORT_USER = "user"
ABORT_WATCHDOG = "watchdog"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    total_replicas: int = 0
    current_replica: int = 0
    current_voice: str = ""
    output_format: str = "wav"
    message: str = "В очереди"
    error: str | None = None
    # Машинный тип ошибки (creash_report): `WORKER_CRASH`, `WORKER_TIMEOUT`,
    # `TTS_ERROR`, `CANCELLED`, `WATCHDOG`, `INTERRUPTED`. Отдельно от текста:
    # интерфейс по нему решает, что предложить («повторить» или «подождать»), а
    # логи — по чему группировать. Пустая строка — ошибки не было.
    error_type: str = ""
    output_path: Path | None = None
    duration_sec: float | None = None
    eta_sec: float | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    # Запрошена ли отмена этой задачи (фаза 11). Отдельно от статуса: у идущей
    # задачи отмена кооперативная — статус меняется тогда, когда воркер дойдёт до
    # безопасной точки, а флаг виден интерфейсу сразу после запроса.
    cancel_requested: bool = False
    # Кто попросил остановку: `"user"` или `"watchdog"`. Сигнал в пайплайн идёт
    # один (`cancel_requested`), но исход разный: отмена пользователя — это
    # состояние `cancelled`, а прерывание watchdog'ом — ошибка с причиной
    # («превышен лимит памяти»), и терять её нельзя: пользователь должен знать,
    # что рендер остановила не он.
    abort_kind: str = ""
    abort_reason: str | None = None
    # Данные готовой задачи: без них нельзя перегенерировать отдельную реплику.
    payload: "JobPayload | None" = field(default=None, repr=False)
    segments: list[tuple[int, int]] = field(default_factory=list, repr=False)
    # Сид каждого куска исходной сборки (`None` — движок сид не принимает).
    seeds: list[int | None] = field(default_factory=list, repr=False)
    # Итог строгой проверки по индексу реплики (`None` — проверка выключена или
    # кусок пришёл из варианта). Нужен, чтобы показать принятое без полного
    # прохождения аудио как непроверенное, а не как обычное.
    qa: list[QaOutcome | None] = field(default_factory=list, repr=False)
    # Диагностика исходного куска по индексу реплики (`None` — кусок пришёл из
    # варианта или задача не проект). Метрики, а не оценка: ранжирования по ним нет.
    quality: list[take_quality.TakeQuality | None] = field(default_factory=list, repr=False)
    # Варианты кусков: индекс реплики → список вариантов, от старых к новым.
    # Наличие варианта — это и есть история: без него перегенерация означала бы
    # «потерять то, что было», и сравнивать новое на слух было бы не с чем.
    variants: dict[int, list[audio_pipeline.ChunkVariant]] = field(
        default_factory=dict, repr=False
    )
    # Индекс реплики → id варианта, который сейчас стоит в файле.
    active_variant: dict[int, str] = field(default_factory=dict, repr=False)
    # Индекс реплики → сколько вариантов у неё уже было (нумерация в подписях).
    variant_seq: dict[int, int] = field(default_factory=dict, repr=False)
    # Индекс пересобираемой сейчас реплики; None — ничего не идёт. Одно поле и на
    # перегенерацию, и на выбор готового варианта: для интерфейса это одно
    # состояние («реплика занята, файл скоро обновится»).
    regenerating: int | None = None
    # Ошибка последней перегенерации: сама задача остаётся DONE, файл не испорчен.
    regen_error: str | None = None

    @property
    def progress(self) -> float:
        if self.status is JobStatus.DONE:
            return 1.0
        if not self.total_replicas:
            return 0.0
        return min(self.current_replica / self.total_replicas, 1.0)

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "current_replica": self.current_replica,
            "total_replicas": self.total_replicas,
            "current_voice": self.current_voice,
            "message": self.message,
            "error": self.error,
            "error_type": self.error_type or None,
            # Человеческий заголовок категории: интерфейсу не нужно знать
            # таксономию, а показывать пользователю `WORKER_CRASH` нельзя.
            "error_title": proto.ERROR_TITLES.get(self.error_type) if self.error_type else None,
            "cancel_requested": self.cancel_requested,
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec else None,
            "eta_sec": round(self.eta_sec, 1) if self.eta_sec else None,
            # Готовая строка для интерфейса: «2 мин 40 сек». `eta_sec` остаётся
            # на месте — по нему считают старые клиенты и тесты, — а формат
            # русского числительного живёт на бэкенде (см. backend/eta.py),
            # чтобы UI не повторял правила склонения у себя.
            "eta_text": eta.format_eta(self.eta_sec) if self.eta_sec else None,
            "output_format": self.output_format,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "regenerating_replica": self.regenerating,
            "regen_error": self.regen_error,
        }


@dataclass
class JobPayload:
    replicas: list[Replica]
    speakers: dict[str, SpeakerSettings]
    settings: RenderSettings
    # Проект, из которого пришла задача (`None` — разовая задача без проекта).
    # Нужен, чтобы после рендера сохранить куски вариантами реплик: сама очередь
    # про проекты ничего не знает и знать не должна.
    project_id: str | None = None


@dataclass
class RenderTask:
    job_id: str
    payload: JobPayload


@dataclass
class RegenerateTask:
    """Перегенерация одной реплики уже готового файла."""

    job_id: str
    index: int


@dataclass
class SelectVariantTask:
    """Постановка в файл ранее сохранённого варианта куска."""

    job_id: str
    index: int
    variant_id: str


@dataclass
class ProjectTakeTask:
    """Пересинтез одной реплики проекта: результат идёт новым вариантом (take).

    Отдельно от `RegenerateTask`: у проекта нет собираемого файла, куски которого
    можно переставлять, — есть только варианты реплик. Поэтому результат не
    вставляется в общий трек, а сохраняется вариантом выбранной реплики.
    """

    job_id: str
    project_id: str
    index: int
    replica: Replica
    speakers: dict[str, SpeakerSettings]
    settings: RenderSettings


@dataclass
class BenchmarkTask:
    """Сравнение голоса на нескольких движках: одна фраза, один reference.

    Голос передаётся снимком, готовым на момент постановки: очередь про
    хранилище голосов не знает (как и про проекты), а сравнение не должно
    менять движок посреди прогона, если пользователь переключил его в карточке.
    """

    job_id: str
    run_id: str
    voice: Voice
    text: str
    engines: list[str]
    qa: str
    auto_accent: bool


class JobQueue:
    """Единственный воркер: следующая задача не начнётся, пока не закончится текущая.

    Очередь приоритетная (`asyncio.PriorityQueue`), но воркер по-прежнему один:
    приоритет решает только, какая из ожидающих задач дойдёт до модели первой.
    Внутри одного приоритета порядок строгий FIFO — за это отвечает счётчик
    постановки `_seq`, второй элемент кортежа.
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue | None = None
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._worker_task: asyncio.Task | None = None
        # Монотонный счётчик постановки: приоритеты равны — решает порядок прихода.
        self._seq = 0
        # Текущая задача. Отмена адресуется конкретной задаче (`Job.cancel_requested`),
        # а не общему флагу процесса: иначе отмена одной задачи прервала бы следующую.
        self._current_job_id: str | None = None

    # -- жизненный цикл --------------------------------------------------------
    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._queue = asyncio.PriorityQueue()
        self._worker_task = asyncio.create_task(self._worker(), name="tts-worker")
        logger.info("Воркер очереди запущен")

    async def stop(self) -> None:
        """Останавливает очередь: отменяет текущую задачу и прерывает её инференс.

        Прерывание обязательно: результат куска выбрасывается (задача отменена),
        а без него остановка ждала бы таймаута куска, держа поток в нативном
        вызове. Супервизор к этому моменту уже помечен как выключающийся
        (см. `main.lifespan`), поэтому такое прерывание не считается падением.
        """
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        aborted = await asyncio.to_thread(
            audio_pipeline.abort_running_inference, "остановка очереди"
        )
        if aborted:
            logger.info("Прерван инференс при остановке: %s", ", ".join(aborted))
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None
        self._queue = None
        logger.info("Воркер очереди остановлен")

    # -- API -------------------------------------------------------------------
    def submit(self, payload: JobPayload, priority: int = PRIORITY_RENDER) -> Job:
        if self._queue is None:
            raise RuntimeError("Очередь не запущена")
        job = Job(
            id=uuid.uuid4().hex[:12],
            total_replicas=len(payload.replicas),
            output_format=payload.settings.output_format,
        )
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._prune()
        self._enqueue(RenderTask(job_id=job.id, payload=payload), priority)
        logger.info(
            "Задача %s принята (%s реплик, приоритет %s)",
            job.id, job.total_replicas, priority,
        )
        return job

    def submit_regenerate(
        self, job_id: str, index: int, priority: int = PRIORITY_REGENERATE
    ) -> Job:
        """Ставит в очередь перегенерацию одной реплики готовой задачи.

        Идёт через ту же очередь, что и генерация: модель тёплая одна на процесс,
        и два одновременных прогона — это и гонка за Metal-контекст, и двойной
        расход памяти. Флаг `regenerating` выставляется сразу, ещё до начала
        работы: иначе UI, опросив задачу до старта, принял бы «ещё в очереди»
        за «уже готово» и не дождался новой версии куска.
        """
        job, payload = self._ready_job(job_id)
        if not 0 <= index < len(payload.replicas):
            raise ValueError("Такой реплики в задаче нет")

        job.regenerating = index
        job.regen_error = None
        job.message = f"Перегенерирую реплику {index + 1}"
        self._enqueue(RegenerateTask(job_id=job_id, index=index), priority)
        logger.info("Задача %s: перегенерация реплики %s", job_id, index + 1)
        return job

    def submit_variant(
        self, job_id: str, index: int, variant_id: str, priority: int = PRIORITY_REGENERATE
    ) -> Job:
        """Ставит в очередь выбор уже сохранённого варианта куска.

        Через очередь, а не сразу в обработчике запроса: постановка варианта
        перезаписывает готовый файл, и одновременная перегенерация (она воркер
        как раз и занимает) дала бы два писателя в один файл.
        """
        job, _ = self._ready_job(job_id)
        variant = self.find_variant(job, index, variant_id)
        if job.active_variant.get(index) == variant.id:
            raise ValueError("Этот вариант уже стоит в файле")

        job.regenerating = index
        job.regen_error = None
        job.message = f"Ставлю «{variant.label}» в реплику {index + 1}"
        self._enqueue(
            SelectVariantTask(job_id=job_id, index=index, variant_id=variant.id), priority
        )
        logger.info("Задача %s: реплика %s → вариант %s", job_id, index + 1, variant.id)
        return job

    def submit_project_take(
        self,
        project_id: str,
        index: int,
        replica: Replica,
        speakers: dict[str, SpeakerSettings],
        settings: RenderSettings,
        priority: int = PRIORITY_REGENERATE,
    ) -> Job:
        """Ставит в очередь пересинтез одной реплики проекта.

        Через ту же очередь, что и сборка: модель в процессе одна, и параллельный
        прогон — это и гонка за неё, и двойной расход памяти. Параметры реплики
        передаются готовыми: очередь про проекты не знает, а собирать их здесь
        означало бы вторую копию разбора наследования (см. audio_pipeline).
        """
        if self._queue is None:
            raise RuntimeError("Очередь не запущена")
        job = Job(
            id=uuid.uuid4().hex[:12],
            total_replicas=1,
            output_format=settings.output_format,
            message=f"Реплика {index + 1}: синтез",
        )
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._prune()
        self._enqueue(
            ProjectTakeTask(
                job_id=job.id,
                project_id=project_id,
                index=index,
                replica=replica,
                speakers=speakers,
                settings=settings,
            ),
            priority,
        )
        logger.info("Проект %s: пересинтез реплики %s", project_id, index + 1)
        return job

    def submit_benchmark(
        self,
        voice: Voice,
        text: str,
        engines: list[str],
        qa: str,
        auto_accent: bool,
        priority: int = PRIORITY_BACKGROUND,
    ) -> tuple[Job, benchmark.BenchmarkRun]:
        """Ставит в очередь сравнение голоса на нескольких движках.

        Через ту же очередь, что и синтез: сравнение поднимает и держит модели
        ровно так же, и второй параллельный прогон — это гонка за Metal-контекст
        и двойной расход памяти. Запуск регистрируется в реестре `benchmark`
        сразу, ещё до старта: интерфейс, опросивший его до начала работы, должен
        увидеть «в очереди», а не «запуск не найден».
        """
        if self._queue is None:
            raise RuntimeError("Очередь не запущена")
        engines = list(engines)
        job = Job(
            id=uuid.uuid4().hex[:12],
            total_replicas=len(engines),
            output_format="wav",
            message="Сравнение движков: в очереди",
        )
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._prune()
        run = benchmark.register(
            benchmark.BenchmarkRun(
                id=benchmark.make_run_id(),
                voice_id=voice.id,
                text=text,
                engines=engines,
            )
        )
        self._enqueue(
            BenchmarkTask(
                job_id=job.id,
                run_id=run.id,
                voice=voice,
                text=text,
                engines=engines,
                qa=qa,
                auto_accent=auto_accent,
            ),
            priority,
        )
        logger.info("Сравнение %s: голос %s, движки %s", run.id, voice.id, ", ".join(engines))
        return job, run

    @staticmethod
    def find_variant(job: Job, index: int, variant_id: str) -> "audio_pipeline.ChunkVariant":
        """Вариант по id; отсутствие — ошибка запроса, а не пустая ветка.

        Вариант мог быть вытеснен более новыми (хранится не больше
        `config.MAX_REPLICA_VARIANTS`), поэтому «не найден» — ожидаемый ответ.
        """
        for variant in job.variants.get(index, []):
            if variant.id == variant_id:
                return variant
        raise KeyError("Такого варианта у реплики нет — возможно, вытеснен более новыми")

    def active_seed(self, job: Job, index: int) -> int | None:
        """Сид того куска, который сейчас стоит в файле: у варианта — из его карточки."""
        active = job.active_variant.get(index)
        if active is not None:
            for variant in job.variants.get(index, []):
                if variant.id == active:
                    return variant.seed
        return job.seeds[index] if index < len(job.seeds) else None

    def active_qa(self, job: Job, index: int) -> QaOutcome | None:
        """Итог строгой проверки куска, который сейчас стоит в файле.

        У варианта — его собственная отметка: варианты генерировались в разное
        время и с разным результатом проверки, и после выбора варианта у реплики
        должна показываться отметка именно выбранного аудио.
        """
        active = job.active_variant.get(index)
        if active is not None:
            for variant in job.variants.get(index, []):
                if variant.id == active:
                    return variant.qa
        return job.qa[index] if index < len(job.qa) else None

    def active_quality(self, job: Job, index: int) -> take_quality.TakeQuality | None:
        """Диагностика куска, который сейчас стоит в файле.

        Как и отметка проверки, метрики принадлежат звучанию, а не реплике: после
        выбора варианта показываться должны измерения выбранного аудио.
        """
        active = job.active_variant.get(index)
        if active is not None:
            for variant in job.variants.get(index, []):
                if variant.id == active:
                    return variant.quality
        return job.quality[index] if index < len(job.quality) else None

    def _ready_job(self, job_id: str) -> tuple[Job, "JobPayload"]:
        """Готовая к пересборке задача и её данные; всё остальное — ошибка запроса."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("Задача не найдена")
        if job.status is not JobStatus.DONE or job.payload is None or job.output_path is None:
            raise ValueError("Пересобрать можно только реплику готового файла")
        if job.regenerating is not None:
            raise ValueError(f"Реплика {job.regenerating + 1} уже пересобирается")
        return job, job.payload

    def _enqueue(
        self,
        task: RenderTask | RegenerateTask | SelectVariantTask | ProjectTakeTask | BenchmarkTask,
        priority: int | None = None,
    ) -> None:
        """Кладёт задачу в очередь: `(приоритет, номер постановки, задача)`.

        Меньший приоритет уходит первым; при равных решает `_seq`, то есть строгий
        FIFO. Кортеж с уникальным вторым элементом ещё и снимает сравнение самих
        датаклассов: `PriorityQueue` сравнивает элементы, а `RenderTask` с `JobPayload`
        несравнимы.
        """
        queue = self._queue
        if queue is None:
            raise RuntimeError("Очередь не запущена")
        if priority is None:
            priority = PRIORITY_BY_TASK.get(type(task).__name__, PRIORITY_RENDER)
        self._seq += 1
        queue.put_nowait((int(priority), self._seq, task))

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_size(self) -> int:
        return self._queue.qsize() if self._queue else 0

    def is_busy(self) -> bool:
        """Есть ли задачи в состоянии queued/processing — включая перегенерацию.

        По этому признаку фоновая и ручная выгрузка движка отказывается работать:
        выгружать модель, которая нужна следующей реплике (или прямо сейчас
        пересобирает одну), незачем — её тут же придётся поднимать заново.
        Проверяются три источника, потому что задача проходит через них
        последовательно: очередь (`queued`), `current_job_id` (`processing`) и
        флаг `regenerating`, который выставляется ещё до взятия задачи воркером.

        Отменённая задача занятостью не считается: она уже не ждёт синтеза и не
        синтезируется, и держать из-за неё движок в памяти незачем.
        """
        if self._current_job_id is not None:
            job = self._jobs.get(self._current_job_id)
            if job is None or job.status is not JobStatus.CANCELLED:
                return True
        if self._queue is not None and self._queue.qsize() > 0:
            return True
        return any(
            job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
            or (
                job.regenerating is not None
                and job.status is not JobStatus.CANCELLED
            )
            for job in self._jobs.values()
        )

    def cancel(self, job_id: str) -> Job | None:
        """Отменяет задачу: ожидающую — сразу, идущую — кооперативно.

        - `queued` → `cancelled` немедленно: синтез ещё не начинался, и воркер,
          достав такую задачу из очереди, обязан её пропустить.
        - `processing` → выставляется `cancel_requested`, а воркер прерывает её на
          ближайшей безопасной точке — между куском и куском или между попытками
          проверки. Поток внутри вызова модели не убивается: это и есть причина,
          по которой отмена кооперативная, а не `Task.cancel()`.
        - `done`/`error`/`cancelled` → ничего не меняется (идемпотентно):
          повторная отмена — не ошибка, а ответ с текущим состоянием.

        Неизвестный id → `None`: это ошибка запроса, и роут отдаёт 404.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status is JobStatus.QUEUED:
            self._mark_cancelled(job, CANCEL_QUEUED_MESSAGE)
            logger.info("Задача %s отменена до запуска", job.id)
            return job
        if job.status is JobStatus.PROCESSING:
            if not job.cancel_requested:
                job.cancel_requested = True
                job.abort_kind = ABORT_USER
                job.message = "Отмена запрошена: остановлюсь на ближайшей безопасной точке"
                logger.warning("Запрошена отмена задачи %s", job.id)
            return job
        # Готовый файл с идущей перегенерацией: отменяется именно она, а не сам
        # рендер — файл уже собран и должен остаться готовым.
        if job.regenerating is not None:
            if not job.cancel_requested:
                job.cancel_requested = True
                job.abort_kind = ABORT_USER
                job.regen_error = "Отменено"
                job.message = f"Реплика {job.regenerating + 1}: отмена перегенерации"
                logger.warning("Запрошена отмена перегенерации задачи %s", job.id)
            return job
        logger.info("Задача %s уже завершена (%s) — отменять нечего", job.id, job.status.value)
        return job

    @property
    def current_job_id(self) -> str | None:
        """Id задачи, которая сейчас в работе; None — воркер свободен.

        Публичное свойство, а не приватное поле: занятость воркера нужна снаружи
        очереди — по ней менеджер моделей отказывает в удалении весов движка,
        который прямо сейчас синтезирует.
        """
        return self._current_job_id

    # -- отмена ----------------------------------------------------------------
    def _cancelled(self, job_id: str) -> bool:
        """Признак отмены именно этой задачи — то, что видит пайплайн в `should_abort`."""
        job = self._jobs.get(job_id)
        return job is not None and job.cancel_requested

    def _mark_cancelled(self, job: Job, message: str = CANCEL_PROCESSING_MESSAGE) -> None:
        """Переводит задачу в `cancelled` и подчищает её недописанные файлы.

        Статус сохраняется в реестре: отменённая задача должна быть видна через
        `GET /api/jobs/{id}`, а не исчезать. Данные проектов не трогаются —
        удаляются только транзитные файлы самой задачи.
        """
        if job.status is not JobStatus.CANCELLED:
            job.status = JobStatus.CANCELLED
            job.finished_at = _now_iso()
        job.cancel_requested = True
        job.error = None
        job.message = message
        job.regenerating = None
        self._discard_partial(job)

    def _finish_interrupted(self, job: Job, project_id: str | None = None) -> None:
        """Завершает задачу, остановленную на безопасной точке.

        Один и тот же сигнал приводит к двум разным исходам, и различать их
        обязан именно этот метод: отмена пользователем — состояние `cancelled`
        (проект возвращается в черновик, работа не потеряна), прерывание
        watchdog'ом — ошибка с причиной, потому что рендер упал не по воле
        пользователя, и проект должен показать ошибку, а не «отменено».
        """
        if job.abort_kind == ABORT_WATCHDOG:
            reason = job.abort_reason or "нехватка памяти"
            job.status = JobStatus.ERROR
            job.error = reason
            job.error_type = proto.ERROR_WATCHDOG
            job.message = "Прервано"
            job.regenerating = None
            self._discard_partial(job)
            self._mark_project(project_id, config.PROJECT_STATUS_ERROR, job.id, reason)
            logger.warning("Задача %s прервана watchdog'ом: %s", job.id, reason)
            return
        job.error_type = proto.ERROR_CANCELLED
        self._mark_cancelled(job)
        self._mark_project(project_id, config.PROJECT_STATUS_DRAFT, job.id)
        logger.info("Задача %s отменена пользователем", job.id)

    def _discard_partial(self, job: Job) -> None:
        """Удаляет недописанный вывод задачи: готовый файл и её варианты кусков.

        Файлы проектов (`output/projects/{id}/`) не трогаются: уже сохранённые
        takes реплик — это данные проекта, и отмена задачи не должна их портить.
        """
        if job.output_path is not None and job.output_path.exists():
            try:
                job.output_path.unlink()
                logger.info("Задача %s: удалён недописанный файл %s", job.id, job.output_path)
            except OSError as exc:
                logger.warning("Задача %s: не удалось удалить %s (%s)", job.id, job.output_path, exc)
        if job.payload is not None:
            for variant in (v for items in job.variants.values() for v in items):
                audio_pipeline.drop_variant(variant)
    def _prune(self) -> None:
        while len(self._order) > MAX_KEPT_JOBS:
            oldest = self._order.pop(0)
            job = self._jobs.get(oldest)
            if job and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                self._order.append(oldest)
                break
            self._jobs.pop(oldest, None)

    async def _worker(self) -> None:
        # Своя ссылка на очередь: stop() обнуляет self._queue, и воркер, читающий
        # поле на каждой итерации, упал бы на уже остановленной очереди.
        queue = self._queue
        if queue is None:
            return
        while True:
            _priority, _seq, task = await queue.get()
            try:
                # Флаг отмены обнуляется до проверки очереди: запрос, пришедший
                # между проверкой и стартом задачи, не должен потеряться.
                job = self._jobs.get(task.job_id)
                if job is not None:
                    job.cancel_requested = False
                    job.abort_kind = ""
                    job.abort_reason = None
                if self._skip_cancelled(task):
                    continue
                await self._wait_for_memory(task.job_id)
                if self._skip_cancelled(task):
                    continue
                # Слот нужен только там, где реально работает модель. Выбор
                # варианта — запись в базу: держать его за LLM-анализом значило бы
                # задерживать отзывчивое действие интерфейса без причины.
                if isinstance(task, SelectVariantTask):
                    await self._select_variant(task)
                    continue
                # Один тяжёлый inference за раз: пока идёт LLM-анализ или чужое
                # распознавание, синтез ждёт своей очереди на слот. Лёгкие API при
                # этом продолжают отвечать — они слот не занимают.
                if not await self._wait_for_heavy_slot(task.job_id):
                    continue
                with llm_memory.get_gate().hold(
                    llm_memory.HEAVY_TTS, owner=f"job:{task.job_id}"
                ):
                    if isinstance(task, RegenerateTask):
                        await self._regenerate(task)
                    elif isinstance(task, ProjectTakeTask):
                        await self._project_take(task)
                    elif isinstance(task, BenchmarkTask):
                        await self._benchmark(task)
                    else:
                        await self._render(task)
            finally:
                queue.task_done()

    async def _wait_for_heavy_slot(self, job_id: str) -> bool:
        """Ждёт свободный тяжёлый слот (LLM/Whisper/другой синтез).

        Возвращает `False`, если задача отменена во время ожидания. Ждать, а не
        отказывать: чужая тяжёлая задача конечна (анализ реплики — секунды), а
        отказ заставил бы пользователя перезапускать рендер вручную.
        """
        job = self._jobs.get(job_id)
        warned = False
        while True:
            holder = llm_memory.get_gate().busy()
            if not holder:
                if warned:
                    logger.info("Задача %s: тяжёлый слот свободен, начинаю", job_id)
                return True
            if self._cancelled(job_id):
                return False
            if job is not None:
                job.message = f"Ожидание: {holder}"
            if not warned:
                warned = True
                logger.info("Задача %s ждёт тяжёлый слот: %s", job_id, holder)
            await asyncio.sleep(MEMORY_WAIT_RETRY_SEC)

    async def _wait_for_memory(self, job_id: str) -> None:
        """Не начинает задачу, пока память не в порядке (creash_report §20–§22).

        Проверка стоит перед взятием job, а не только в фоновом watchdog: задача
        должна остаться в очереди с понятным сообщением, а не стартовать и лечь
        дополнительной нагрузкой поверх уже существующей. Состояние берётся у
        `memory_monitor` — он видит и системную память, и RSS процессов синтеза,
        где и живут модели; свой потолок бэкенда при этом действует отдельно.

        Перед ожиданием очередь пробует вернуть память самым дешёвым способом:
        выгружает простаивающие движки. Выгружать занятый движок нельзя — это
        оборвало бы чужой инференс, — а простаивающий держит гигабайты.
        """
        job = self._jobs.get(job_id)
        warned = False
        while resource_guard.is_memory_critical():
            if self._cancelled(job_id):
                return
            if not warned:
                unloaded = await asyncio.to_thread(self._free_memory_for_job)
                if unloaded:
                    logger.warning(
                        "Задача %s ждёт памяти: выгружены простаивающие движки %s",
                        job_id, ", ".join(unloaded),
                    )
                    # После выгрузки состояние могло измениться — перепроверяем
                    # до сообщения «перегружено», чтобы не пугать зря.
                    if not resource_guard.is_memory_critical():
                        logger.info("Задача %s: память освободилась после выгрузки", job_id)
                        return
            state = resource_guard.memory_state()
            if job is not None:
                job.message = f"Система перегружена: {state['reason']} — ожидание"
            if not warned:
                warned = True
                logger.warning("Задача %s ждёт: %s", job_id, state["reason"])
            await asyncio.sleep(MEMORY_WAIT_RETRY_SEC)
        if warned:
            logger.info("Задача %s: память освободилась, начинаю", job_id)

    def _free_memory_for_job(self) -> list[str]:
        """Выгружает простаивающие движки, если прямо сейчас никто не синтезирует.

        Проверка именно по текущей задаче, а не по `is_busy()`: ожидающая задача
        тоже делает очередь «занятой», но синтеза в ней нет — выгружать движок
        как раз и нужно ради неё. Во время рендера (`_current_job_id` выставлен)
        выгрузка не делается: модель понадобится следующей реплике.
        """
        if self._current_job_id is not None:
            return []
        unloaded: list[str] = []
        for engine_id, engine in created_engines().items():
            if not engine.is_loaded or engine.active_synthesizes:
                continue
            try:
                engine.unload()
            except Exception as exc:  # noqa: BLE001 — уборка не повод падать
                logger.warning("Движок %s: выгрузка перед ожиданием памяти не удалась (%s)", engine_id, exc)
                continue
            unloaded.append(engine_id)
        return unloaded

    def _drop_variants(self, job: Job) -> None:
        """Убирает снятые варианты и их файлы.

        Готовый файл при отмене не переписывается, поэтому и «исходный» вариант,
        вырезанный из него перед пересинтезом, теряет смысл: он лишь обозначил бы
        историю, которой не было.
        """
        for variants in job.variants.values():
            for variant in variants:
                audio_pipeline.drop_variant(variant)
        job.variants.clear()
        job.active_variant.clear()

    def _cancel_pending_take(self, job: Job) -> None:
        """Снимает перегенерацию, которую воркер так и не начал.

        Статус задачи не трогается: у готового рендера он и остаётся `done`, а
        отменяется именно пересборка реплики — файл не переписан, прежнее звучание
        на месте. Интерфейс узнаёт об отмене из `regen_error` и снимает занятость
        реплики (`regenerating`).
        """
        index = job.regenerating
        self._drop_variants(job)
        job.regenerating = None
        job.cancel_requested = False
        job.abort_kind = ""
        job.abort_reason = None
        job.regen_error = "Отменено"
        if index is not None:
            job.message = f"Реплика {index + 1}: перегенерация отменена"
        logger.info("Задача %s: перегенерация отменена до запуска", job.id)

    def _skip_cancelled(self, task) -> bool:
        """Пропускает задачу, отменённую до запуска: синтез не начинается вовсе."""
        job = self._jobs.get(task.job_id)
        if job is None or job.status is JobStatus.CANCELLED:
            return True
        if not job.cancel_requested:
            return False
        if isinstance(task, RenderTask) or job.status is JobStatus.QUEUED:
            self._mark_cancelled(job, CANCEL_QUEUED_MESSAGE)
            return True
        # Отменённая перегенерация или выбор варианта у готового рендера: статус
        # остаётся `done` — файл на месте, отменена только пересборка реплики.
        self._cancel_pending_take(job)
        return True

    async def _render(self, task: RenderTask) -> None:
        job = self._jobs[task.job_id]
        payload = task.payload
        job.status = JobStatus.PROCESSING
        job.started_at = _now_iso()
        job.message = "Готовлю модель"
        self._current_job_id = job.id
        self._mark_project(payload.project_id, config.PROJECT_STATUS_RENDERING, job.id)
        started = time.monotonic()
        # План рендера и статистика времени: до старта видно оценку по кускам
        # (у каждого свой движок и своя длина текста), после каждого куска
        # оценка пересчитывается по остатку (см. backend/eta.py).
        plan = self._render_plan(payload)
        tracker = eta.get_tracker()
        job.eta_sec = await asyncio.to_thread(tracker.estimate, plan)
        try:
            result = await self._render_with_recovery(job, payload, plan, tracker)
            job.output_path = result.output_path
            job.duration_sec = result.duration_sec
            # Метаданные кусков нужны, чтобы задним числом заменить одну реплику.
            job.payload = payload
            job.segments = result.segments
            job.seeds = result.seeds
            job.qa = result.qa
            job.quality = list(result.qualities)
            job.current_replica = job.total_replicas
            job.eta_sec = 0.0
            # Статус DONE ставится после записи кусков в проект: «готово» должно
            # означать, что открытый следом проект уже показывает этот рендер,
            # а не что файл есть, а история реплик появится когда-нибудь потом.
            await self._persist_project_takes(job, payload, result)
            job.message = f"Готово: {result.duration_sec:.1f} c аудио"
            job.status = JobStatus.DONE
            logger.info("Задача %s завершена за %.1f c", job.id, time.monotonic() - started)
        except asyncio.CancelledError as exc:
            # Остановка сервера: задача не «упала», а не пережила выключение.
            # Готовое сохраняется так же, как при падении воркера, а реплика
            # помечается прерванной — иначе после рестарта она выглядела бы как
            # «ещё синтезируется» (см. recovery при старте).
            job.status = JobStatus.ERROR
            job.error = "Отменено (остановка сервера)"
            job.error_type = proto.ERROR_INTERRUPTED
            job.cancel_requested = True
            partial = getattr(exc, "partial_render", None)
            if payload.project_id and partial is not None and partial.failed_index is not None:
                self._set_replica_status(
                    payload.project_id, partial.failed_index, config.REPLICA_STATUS_INTERRUPTED
                )
            logger.warning(
                "Задача %s прервана остановкой сервера (готово реплик: %s)",
                job.id, 0 if partial is None else partial.done,
            )
            # Готовое сохраняется синхронно: ждать запись в момент отмены поздно.
            self._save_partial_takes_now(job, payload, partial)
            self._discard_partial(job)
            raise
        except _RenderFailed as failed:
            # Воркер падал, попытки кончились. Диагностика и статус прерванной
            # реплики записаны в `_render_with_recovery`, готовые куски — тоже.
            job.status = JobStatus.ERROR
            job.error_type = failed.error_type
            job.error = str(failed.cause)
            job.message = "Прервано: процесс синтеза не удержался"
            self._mark_project(
                payload.project_id, config.PROJECT_STATUS_ERROR, job.id, str(failed.cause)
            )
            logger.error(
                "Задача %s: синтез прерван (%s), попыток %d: %s",
                job.id, failed.error_type, failed.attempts, failed.cause,
            )
        except audio_pipeline.JobCancelledError:
            # Отмена пользователя — `cancelled`, прерывание watchdog'ом — ошибка
            # с причиной; различает их `_finish_interrupted`.
            self._finish_interrupted(job, payload.project_id)
        except audio_pipeline.JobAbortedError as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.error_type = proto.ERROR_WATCHDOG
            job.message = "Прервано"
            self._mark_project(payload.project_id, config.PROJECT_STATUS_ERROR, job.id, str(exc))
            logger.warning("Задача %s прервана watchdog'ом: %s", job.id, exc)
        except Exception as exc:
            # Ошибку мог вернуть сам движок (модель ответила отказом) — тогда тип
            # известен; всё остальное считаем ошибкой синтеза. Готовые куски при
            # этом не выбрасываем: рендер мог упасть на последней реплике.
            partial = getattr(exc, "partial_render", None)
            if payload.project_id and partial is not None and partial.failed_index is not None:
                # Реплика прервана и здесь: статус `rendering`, оставшийся навсегда,
                # скрыл бы от пользователя, что попытка была и не удалась.
                self._set_replica_status(
                    payload.project_id, partial.failed_index, config.REPLICA_STATUS_INTERRUPTED
                )
            await self._save_partial_takes(job, payload, partial, str(exc))
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.error_type = getattr(exc, "error_type", proto.ERROR_TTS)
            job.message = "Ошибка"
            self._mark_project(payload.project_id, config.PROJECT_STATUS_ERROR, job.id, str(exc))
            logger.exception("Задача %s упала", job.id)
        finally:
            self._current_job_id = None
            job.finished_at = _now_iso()

    async def _render_with_recovery(
        self,
        job: Job,
        payload: JobPayload,
        plan: list[eta.EtaStep],
        tracker: eta.EtaTracker,
    ) -> audio_pipeline.RenderResult:
        """Рендер с восстановлением после отказа воркера (creash_report §12–§14).

        Отказ процесса синтеза не отменяет задачу сразу: супервизор уже поднял
        замену, а готовые куски лежат в памяти (`RenderPartial`). Пока позволяет
        бюджет попыток (`TTS_WORKER_CRASH_RETRIES`), рендер продолжается с
        прерванной реплики — уже синтезированное не пересоздаётся и не теряется.
        Когда попытки кончились, готовое сохраняется вариантами реплик, а задача
        завершается ошибкой с типом отказа: «доделать как-нибудь» здесь означало
        бы скрыть, что сборка неполная.

        Повтор не делается, если движок переведён супервизором в `DEGRADED`:
        немедленная попытка упёрлась бы в тот же отказ, а пользователю нужен не
        цикл, а понятное «подождите» (см. `WorkerUnavailableError`).
        """
        kwargs = {
            "on_progress": self._progress_callback(job, plan, tracker, payload.project_id),
            "should_abort": lambda: self._cancelled(job.id),
            "wait_for_memory": lambda: self._wait_for_memory(job.id),
            "on_note": self._note_callback(job),
            "on_engine_loaded": self._engine_loaded_callback(tracker),
            "on_chunk_timing": self._timing_callback(job, plan, tracker),
        }
        resume: audio_pipeline.RenderPartial | None = None
        attempt = 0
        while True:
            try:
                return await audio_pipeline.render_dialogue(
                    job_id=job.id,
                    replicas=payload.replicas,
                    speakers=payload.speakers,
                    settings=payload.settings,
                    resume=resume,
                    llm_hints=JobQueue._llm_hints(payload),
                    **kwargs,
                )
            except (proto.WorkerFailure, audio_pipeline.ChunkTimeoutError) as exc:
                attempt += 1
                partial = getattr(exc, "partial_render", None)
                error_type = getattr(exc, "error_type", proto.ERROR_WORKER_TIMEOUT)
                self._note_worker_failure(
                    job,
                    payload.project_id,
                    exc,
                    index=None if partial is None else partial.failed_index,
                    attempt=attempt,
                    error_type=error_type,
                )
                if not self._can_retry(job, exc, attempt):
                    await self._save_partial_takes(job, payload, partial, str(exc))
                    raise _RenderFailed(exc, partial, error_type, attempt) from exc
                resume = partial if partial is not None and partial.done else None
                done = 0 if resume is None else resume.done
                job.message = (
                    f"Процесс синтеза упал: перезапущен, продолжаю с реплики {done + 1}"
                    if done
                    else "Процесс синтеза упал: перезапущен, повторяю сначала"
                )
                logger.warning(
                    "Задача %s: %s — повтор %d из %d (готово реплик: %d)",
                    job.id, error_type, attempt, config.WORKER_CRASH_RETRIES + 1, done,
                )

    def _can_retry(self, job: Job, exc: BaseException, attempt: int) -> bool:
        """Есть ли смысл в повторе: бюджет, состояние движка и отмена."""
        if attempt > config.WORKER_CRASH_RETRIES:
            return False
        if isinstance(exc, proto.WorkerUnavailableError):
            return False
        return not self._cancelled(job.id)

    def _note_worker_failure(
        self,
        job: Job,
        project_id: str | None,
        exc: BaseException,
        *,
        index: int | None,
        attempt: int,
        error_type: str,
    ) -> None:
        """Помечает прерванную реплику и записывает диагностику падения.

        Запись идёт **до** повтора: если повтор удастся, пользователь всё равно
        должен увидеть, что процесс падал и на какой реплике, — иначе «тихое»
        восстановление скрыло бы нестабильность движка (creash_report §8).
        """
        if project_id is not None and index is not None:
            self._set_replica_status(project_id, index, config.REPLICA_STATUS_INTERRUPTED)
        details = dict(getattr(exc, "details", {}) or {})
        engine = str(getattr(exc, "engine", "") or "")
        try:
            memory_percent = float(resource_guard.snapshot()["system_mem_percent"])
        except Exception:  # noqa: BLE001 — метрика не повод терять запись о падении
            memory_percent = None
        try:
            get_projects_store().record_worker_crash(
                job_id=job.id,
                project_id=project_id,
                replica_index=index,
                engine=engine,
                error_type=error_type,
                message=str(exc),
                pid=details.get("pid"),
                exit_code=details.get("exit_code"),
                signal_number=details.get("signal"),
                signal_name=details.get("signal_name"),
                reason=str(details.get("reason") or ""),
                retry_count=config.WORKER_CRASH_RETRIES,
                attempt=attempt,
                started_at=job.started_at,
                interrupted_at=_now_iso(),
                memory_percent=memory_percent,
            )
        except Exception as store_exc:  # noqa: BLE001 — база не важнее задачи
            logger.warning("Задача %s: не удалось записать диагностику падения (%s)", job.id, store_exc)
        title = proto.ERROR_TITLES.get(error_type, "Синтез прерван")
        where = "" if index is None else f" на реплике {index + 1}"
        job.error_type = error_type
        job.message = f"{title}{where}"

    @staticmethod
    def _short_plan_dict(short_run) -> dict | None:
        """Метаданные короткой реплики для карточки куска (§24) или None."""
        if short_run is None:
            return None
        return short_run.to_dict()

    @staticmethod
    def _llm_hints(payload: JobPayload | None) -> dict:
        """Подсказки LLM для коротких реплик проекта (пусто — слой работает сам).

        Ошибка чтения не должна мешать синтезу: подсказка вторична, а рендер важнее.
        """
        if payload is None or not payload.project_id:
            return {}
        try:
            return get_projects_store().llm_utterance_hints(payload.project_id)
        except Exception as exc:  # noqa: BLE001 — подсказка не повод падать
            logger.warning(
                "Проект %s: подсказки LLM не прочитаны (%s)", payload.project_id, exc
            )
            return {}

    @staticmethod
    def _short_context(job: Job, index: int):
        """Контекст соседей для перегенерации одной реплики.

        Перегенерация идёт по сохранённому payload, поэтому соседи известны — и
        короткая реплика после пересборки звучит так же, как при общей сборке.
        Если контекст собрать не удалось, слой честно работает без него.
        """
        payload = job.payload
        settings = payload.settings if payload else None
        if payload is None or settings is None or settings.short_utterance is None:
            return None
        if not settings.short_utterance.enabled:
            return None
        try:
            contexts = su.build_contexts(
                payload.replicas,
                thresholds=settings.short_utterance.thresholds,
                llm_hints=JobQueue._llm_hints(payload),
            )
        except Exception as exc:  # noqa: BLE001 — контекст вторичен, синтез важнее
            logger.warning("Задача %s: контекст короткой реплики не собран (%s)", job.id, exc)
            return None
        return contexts[index] if 0 <= index < len(contexts) else None

    @staticmethod
    def _set_replica_status(project_id: str, index: int, status: str) -> None:
        """Статус реплики проекта. Ошибка записи задачу не губит (см. `_mark_project`)."""
        try:
            get_projects_store().set_replica_status(project_id, index, status)
        except Exception as exc:  # noqa: BLE001 — фоновая запись, задача важнее
            logger.warning(
                "Проект %s: не удалось записать статус реплики %s (%s)", project_id, index + 1, exc
            )

    async def _save_partial_takes(
        self,
        job: Job,
        payload: JobPayload,
        partial: audio_pipeline.RenderPartial | None,
        error: str,
    ) -> None:
        """Сохраняет куски прерванного рендера вариантами реплик проекта.

        До этого момента готовые куски жили только в памяти: падение процесса их
        не теряло, а вот перезапуск приложения потерял бы. Поэтому перед тем как
        признать рендер неудачным, сделанное кладётся на диск — и становится
        текущим звучанием готовых реплик, а проект при этом **не** помечается
        `rendered`: сборка не завершена.
        """
        project_id = payload.project_id
        if not project_id or partial is None or not partial.pieces:
            return
        try:
            takes = await asyncio.to_thread(self._build_partial_takes, job, payload, partial)
            await asyncio.to_thread(
                get_projects_store().save_partial_takes, project_id, job.id, takes
            )
            logger.warning(
                "Задача %s: сохранено %d кусков прерванного рендера (%s)",
                job.id, len(takes), error,
            )
        except Exception as exc:  # noqa: BLE001 — диагностика важнее, файл уже не собран
            logger.warning(
                "Проект %s: не удалось сохранить куски прерванного рендера (%s: %s)",
                project_id, type(exc).__name__, exc,
            )

    def _save_partial_takes_now(
        self, job: Job, payload: JobPayload, partial: audio_pipeline.RenderPartial | None
    ) -> None:
        """То же, но без ожидания: вызывается при остановке сервера.

        Ждать асинхронную запись в момент отмены уже поздно — задача отменена,
        цикл событий останавливается. Путь синхронный (файлы и база пишутся
        синхронными функциями) и короткий: цена секунды задержки при выключении
        ниже цены потерянных реплик, которые уже синтезированы.
        """
        project_id = payload.project_id
        if not project_id or partial is None or not partial.pieces:
            return
        try:
            takes = self._build_partial_takes(job, payload, partial)
            get_projects_store().save_partial_takes(project_id, job.id, takes)
            logger.warning(
                "Задача %s: сохранено %d кусков при остановке сервера", job.id, len(takes)
            )
        except Exception as exc:  # noqa: BLE001 — выключение важнее аккуратности
            logger.warning(
                "Проект %s: не удалось сохранить куски при остановке (%s: %s)",
                project_id, type(exc).__name__, exc,
            )

    @staticmethod
    def _build_partial_takes(
        job: Job, payload: JobPayload, partial: audio_pipeline.RenderPartial
    ) -> list[dict]:
        """Файлы и карточки кусков прерванного рендера — общая часть обоих путей."""
        paths = audio_pipeline.export_partial_takes(payload.project_id, partial.pieces)
        takes: list[dict] = []
        count = min(len(paths), len(partial.pieces), len(payload.replicas), len(partial.segments))
        for index in range(count):
            replica = payload.replicas[index]
            speaker = payload.speakers.get(replica.voice)
            start, end = partial.segments[index]
            takes.append(
                {
                    "index": index,
                    "audio_path": str(paths[index]),
                    "label": f"рендер {job.id} · {PARTIAL_TAKE_LABEL}",
                    "seed": partial.seeds[index] if index < len(partial.seeds) else None,
                    "engine": audio_pipeline.voice_engine(speaker.voice_id if speaker else ""),
                    "parameters": {
                        **(audio_pipeline.chunk_parameters(replica, speaker)
                           if speaker else {}),
                        **(JobQueue._short_plan_dict(partial.short_runs.get(index)) or {}),
                    },
                    "duration_sec": (end - start) / audio_pipeline.SAMPLE_RATE,
                    "qa": partial.qa[index].to_dict()
                    if index < len(partial.qa) and partial.qa[index] is not None
                    else None,
                    "quality": partial.qualities[index].to_dict()
                    if index < len(partial.qualities) and partial.qualities[index] is not None
                    else None,
                }
            )
        return takes

    @staticmethod
    def _mark_project(
        project_id: str | None, status: str, job_id: str | None = None, error: str | None = None
    ) -> None:
        """Обновляет статус проекта. Ошибка записи не должна губить задачу.

        База — не модель: если она недоступна, уже сгенерированное аудио терять
        незачем, поэтому сбой записи только логируется.
        """
        if not project_id:
            return
        try:
            get_projects_store().set_project_status(project_id, status, job_id, error)
        except Exception as exc:  # noqa: BLE001 — фоновая запись, задача важнее
            logger.warning("Проект %s: не удалось записать статус %s (%s)", project_id, status, exc)

    async def _persist_project_takes(
        self, job: Job, payload: JobPayload, result: audio_pipeline.RenderResult
    ) -> None:
        """Сохраняет куски готового рендера вариантами реплик проекта.

        Проект переживает задачу и очистку output/, поэтому куски копируются в его
        каталог отдельными файлами: у реплики должна остаться возможность
        прослушать и выбрать звучание, даже когда задача из очереди уже вытеснена.
        """
        project_id = payload.project_id
        if not project_id:
            return
        try:
            paths = await asyncio.to_thread(
                audio_pipeline.export_project_chunks,
                project_id,
                result.output_path,
                payload.settings.output_format,
                result.segments,
            )
            takes: list[dict] = []
            for index, replica in enumerate(payload.replicas):
                if index >= len(paths):
                    break
                speaker = payload.speakers.get(replica.voice)
                start, end = result.segments[index]
                takes.append(
                    {
                        "index": index,
                        "audio_path": str(paths[index]),
                        "label": f"рендер {job.id}",
                        "seed": result.seeds[index] if index < len(result.seeds) else None,
                        "engine": audio_pipeline.voice_engine(
                            speaker.voice_id if speaker else ""
                        ),
                        "parameters": {
                            **(audio_pipeline.chunk_parameters(replica, speaker)
                               if speaker else {}),
                            **(self._short_plan_dict(result.short_runs.get(index)) or {}),
                        },
                        "duration_sec": (end - start) / audio_pipeline.SAMPLE_RATE,
                        "qa": result.qa[index].to_dict()
                        if index < len(result.qa) and result.qa[index] is not None
                        else None,
                        "quality": result.qualities[index].to_dict()
                        if index < len(result.qualities) and result.qualities[index] is not None
                        else None,
                    }
                )
            await asyncio.to_thread(
                get_projects_store().save_render_takes, project_id, job.id, takes
            )
        except Exception as exc:  # noqa: BLE001 — файл готов, история кусков вторична
            logger.warning(
                "Проект %s: не удалось сохранить куски рендера (%s: %s)",
                project_id, type(exc).__name__, exc,
            )
            self._mark_project(project_id, config.PROJECT_STATUS_RENDERED, job.id)

    async def _regenerate(self, task: RegenerateTask) -> None:
        """Пересинтезирует один кусок и вставляет его в готовый файл.

        Прежнее содержимое куска не теряется: до замены оно сохраняется
        отдельным вариантом, новый результат добавляется вторым. Раньше
        «перегенерировать» означало «надеяться, что новая версия лучше, и
        потерять старую» — сравнить их на слух было нельзя.
        """
        job = self._jobs.get(task.job_id)
        if job is None or job.payload is None or job.output_path is None:
            return
        replica = job.payload.replicas[task.index]
        speaker = job.payload.speakers.get(replica.voice)
        self._current_job_id = job.id
        try:
            if speaker is None:
                raise ValueError(f"Для «{replica.label}» не найден голос")
            await self._keep_current(job, task.index, job.payload.settings, job.output_path)
            if self._cancelled(job.id):
                raise audio_pipeline.JobCancelledError("Отменено до синтеза")
            prepared, seed, qa_outcome, quality, short_run = (
                await audio_pipeline.synthesize_replica(
                    replica=replica,
                    speaker=speaker,
                    settings=job.payload.settings,
                    index=task.index,
                    wait_for_memory=lambda: self._wait_for_memory(job.id),
                    on_note=self._note_callback(job),
                    should_abort=lambda: self._cancelled(job.id),
                    # Перегенерация идёт по сохранённому payload, поэтому соседи
                    # известны: короткая реплика получает тот же контекст, что и
                    # при обычной сборке, а не синтезируется «в вакууме».
                    context=self._short_context(job, task.index),
                )
            )
            # Синтез мог вернуться уже после запроса отмены: тогда вариант не
            # сохраняем — в файл попадёт недоделанная пересборка.
            if self._cancelled(job.id):
                raise audio_pipeline.JobCancelledError("Отменено до сохранения варианта")
            variant = await self._add_variant(
                job, task.index, prepared, seed, qa_outcome, quality,
                plan=self._short_plan_dict(short_run),
            )
            duration, bounds = await audio_pipeline.apply_variant(
                source_path=job.output_path,
                settings=job.payload.settings,
                segments=job.segments,
                index=task.index,
                variant=variant,
            )
            self._shift_segments(job.segments, task.index, bounds)
            job.duration_sec = duration
            job.message = f"Реплика {task.index + 1}: {variant.label}, сид {variant.seed}"
            logger.info(
                "Задача %s: реплика %s — %s (сид %s)",
                job.id, task.index + 1, variant.label, variant.seed,
            )
        except asyncio.CancelledError:
            raise
        except audio_pipeline.JobCancelledError:
            # Отмена не портит исходный файл: новая версия куска ещё не встала.
            # Убираем транзитные варианты — «исходный» и, если успел появиться,
            # новый: рендер остаётся готовым и звучит как прежде.
            self._drop_variants(job)
            job.regen_error = "Отменено"
            job.message = f"Реплика {task.index + 1}: перегенерация отменена"
            logger.info("Задача %s: перегенерация реплики %s отменена", job.id, task.index + 1)
        except (proto.WorkerFailure, audio_pipeline.ChunkTimeoutError) as exc:
            # Падение процесса при пересборке одной реплики: прежнее звучание
            # цело (файл не переписан), новая версия не появилась. Автоповтора
            # здесь нет намеренно: пересборка одной реплики — действие по
            # требованию, и решение «пробовать ещё» остаётся за пользователем.
            error_type = getattr(exc, "error_type", proto.ERROR_WORKER_TIMEOUT)
            self._note_worker_failure(
                job,
                job.payload.project_id if job.payload else None,
                exc,
                index=task.index,
                attempt=1,
                error_type=error_type,
            )
            job.error_type = error_type
            job.regen_error = proto.ERROR_TITLES.get(error_type, str(exc))
            job.message = f"Реплика {task.index + 1}: процесс синтеза упал"
            logger.error(
                "Задача %s: перегенерация реплики %s прервана (%s)",
                job.id, task.index + 1, exc,
            )
        except Exception as exc:
            job.error_type = getattr(exc, "error_type", proto.ERROR_TTS)
            job.regen_error = str(exc)
            job.message = "Перегенерация не удалась"
            logger.exception("Задача %s: перегенерация реплики %s упала", job.id, task.index + 1)
        finally:
            self._current_job_id = None
            job.regenerating = None

    async def _select_variant(self, task: SelectVariantTask) -> None:
        """Ставит в файл уже сохранённый вариант куска.

        Пересинтеза нет: вариант — это готовое аудио, и выбор между вариантами
        должен быть быстрым, иначе A/B на слух превращается в ожидание модели.
        """
        job = self._jobs.get(task.job_id)
        if job is None or job.payload is None or job.output_path is None:
            return
        self._current_job_id = job.id
        try:
            variant = self.find_variant(job, task.index, task.variant_id)
            duration, bounds = await audio_pipeline.apply_variant(
                source_path=job.output_path,
                settings=job.payload.settings,
                segments=job.segments,
                index=task.index,
                variant=variant,
            )
            self._shift_segments(job.segments, task.index, bounds)
            job.active_variant[task.index] = variant.id
            job.duration_sec = duration
            job.message = f"Реплика {task.index + 1}: {variant.label}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            job.regen_error = str(exc)
            job.message = "Не удалось поставить вариант"
            logger.exception("Задача %s: выбор варианта реплики %s упал", job.id, task.index + 1)
        finally:
            self._current_job_id = None
            job.regenerating = None

    async def _project_take(self, task: ProjectTakeTask) -> None:
        """Синтезирует реплику проекта заново и сохраняет её новым вариантом.

        Остальные реплики не трогаются: у проекта нет пересобираемого файла, и
        «перегенерировать» здесь означает «добавить реплике ещё одно звучание,
        сделав его текущим» — прежнее остаётся вариантом и его можно вернуть.
        """
        job = self._jobs.get(task.job_id)
        if job is None:
            return
        speaker = task.speakers.get(task.replica.voice)
        self._current_job_id = job.id
        job.status = JobStatus.PROCESSING
        job.started_at = _now_iso()
        try:
            if speaker is None:
                raise ValueError(f"Для «{task.replica.label}» не найден голос")
            # Реплика помечается занятой до передачи воркеру: после падения видно,
            # какую именно реплику собирали (creash_report §7).
            self._set_replica_status(task.project_id, task.index, config.REPLICA_STATUS_RENDERING)
            prepared, seed, qa_outcome, quality, short_run = (
                await audio_pipeline.synthesize_replica(
                    replica=task.replica,
                    speaker=speaker,
                    settings=task.settings,
                    index=task.index,
                    wait_for_memory=lambda: self._wait_for_memory(job.id),
                    on_note=self._note_callback(job),
                    should_abort=lambda: self._cancelled(job.id),
                )
            )
            variant = await asyncio.to_thread(
                audio_pipeline.save_chunk_variant,
                job.id,
                task.index,
                "take",
                prepared,
                seed,
                f"пересинтез {job.id}",
                qa_outcome,
                quality,
            )
            # Файл сначала живёт в output/, а в проект копируется: вариант задачи
            # чистится вместе с output/, вариант проекта должен его пережить.
            await asyncio.to_thread(
                get_projects_store().save_replica_take,
                task.project_id,
                task.index,
                {
                    "audio_path": str(variant.path),
                    "label": variant.label,
                    "seed": seed,
                    "engine": audio_pipeline.voice_engine(
                        str(task.replica.voice_id or speaker.voice_id)
                    ),
                    "parameters": {
                        **audio_pipeline.chunk_parameters(task.replica, speaker),
                        **(self._short_plan_dict(short_run) or {}),
                    },
                    "duration_sec": variant.duration_sec,
                    "qa": qa_outcome.to_dict() if qa_outcome is not None else None,
                    "quality": variant.quality.to_dict()
                    if variant.quality is not None
                    else None,
                },
            )
            await asyncio.to_thread(audio_pipeline.drop_variant, variant)
            job.current_replica = 1
            job.duration_sec = variant.duration_sec
            job.message = f"Реплика {task.index + 1}: готово, {variant.duration_sec:.1f} c"
            job.status = JobStatus.DONE
            logger.info(
                "Проект %s: реплика %s пересинтезирована (сид %s)",
                task.project_id, task.index + 1, seed,
            )
        except asyncio.CancelledError:
            job.status = JobStatus.ERROR
            job.error = "Отменено (остановка сервера)"
            job.cancel_requested = True
            raise
        except audio_pipeline.JobCancelledError:
            # Вариант в проект ещё не сохранён — терять нечего, и данные проекта
            # остаются согласованными: реплика просто не получила нового take.
            # Прерывание watchdog'ом — исключение: реплика не пересинтезирована
            # из-за нехватки памяти, и об этом надо сказать внятно.
            if job.abort_kind == ABORT_WATCHDOG:
                job.regen_error = job.abort_reason or "прервано из-за нехватки памяти"
                job.message = f"Реплика {task.index + 1}: прервано"
                self._drop_variants(job)
                logger.warning(
                    "Проект %s: пересинтез реплики %s прерван watchdog'ом: %s",
                    task.project_id, task.index + 1, job.regen_error,
                )
            else:
                self._mark_cancelled(job)
                logger.info("Проект %s: пересинтез реплики %s отменён", task.project_id, task.index + 1)
        except (proto.WorkerFailure, audio_pipeline.ChunkTimeoutError) as exc:
            # Падение процесса при пересинтезе реплики проекта: в проект ничего
            # не записано (вариант сохраняется последним шагом), поэтому данные
            # проекта согласованы, а реплика помечается прерванной и объясняется
            # в `regen_error` — повтор остаётся за пользователем.
            error_type = getattr(exc, "error_type", proto.ERROR_WORKER_TIMEOUT)
            self._note_worker_failure(
                job,
                task.project_id,
                exc,
                index=task.index,
                attempt=1,
                error_type=error_type,
            )
            job.status = JobStatus.ERROR
            job.error_type = error_type
            job.error = str(exc)
            job.message = f"Реплика {task.index + 1}: процесс синтеза упал"
            logger.error(
                "Проект %s: пересинтез реплики %s прерван (%s)",
                task.project_id, task.index + 1, exc,
            )
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.error_type = getattr(exc, "error_type", proto.ERROR_TTS)
            job.message = f"Реплика {task.index + 1}: не удалось перегенерировать"
            logger.exception(
                "Проект %s: пересинтез реплики %s упал", task.project_id, task.index + 1
            )
        finally:
            self._current_job_id = None
            job.finished_at = _now_iso()

    async def _benchmark(self, task: BenchmarkTask) -> None:
        """Сравнивает голос на выбранных движках тем же единственным воркером.

        Ошибка одного движка живёт в его строке результата, а не в задаче: run
        завершается `done`, если прогон состоялся, — иначе пропала бы и та
        часть сравнения, которая получилась.
        """
        job = self._jobs.get(task.job_id)
        run = benchmark.get_run(task.run_id)
        if job is None or run is None:
            return
        self._current_job_id = job.id
        job.status = JobStatus.PROCESSING
        job.started_at = _now_iso()
        run.status = benchmark.RUN_RUNNING
        # Свой счётчик, а не общий прогресс рендера: у сравнения шаг — движок,
        # а не реплика, и «Реплика 1 из 3» в статусе читалось бы неверно.
        def on_progress(index: int, total: int, label: str) -> None:
            job.current_replica = index - 1
            job.current_voice = label
            job.message = f"Движок {index} из {total} — {label}"

        try:
            await benchmark.run_benchmark(
                run,
                task.voice,
                qa_mode=task.qa,
                auto_accent=task.auto_accent,
                wait_for_memory=lambda: self._wait_for_memory(job.id),
                on_note=self._note_callback(job),
                on_progress=on_progress,
                should_abort=lambda: self._cancelled(job.id),
            )
            job.current_replica = job.total_replicas
            job.message = "Сравнение готово"
            job.status = JobStatus.DONE
            run.status = benchmark.RUN_DONE
        except asyncio.CancelledError:
            job.status = JobStatus.ERROR
            job.error = "Отменено (остановка сервера)"
            job.cancel_requested = True
            run.status = benchmark.RUN_ERROR
            run.error = job.error
            raise
        except audio_pipeline.JobCancelledError:
            # Уже снятые take-файлы движков остаются: это результат сравнения,
            # а не транзитный вывод задачи, и доигранные строки в нём полезны.
            if job.abort_kind == ABORT_WATCHDOG:
                job.status = JobStatus.ERROR
                job.error = job.abort_reason or "прервано из-за нехватки памяти"
                job.message = "Прервано"
                logger.warning("Сравнение %s прервано watchdog'ом: %s", task.run_id, job.error)
            else:
                self._mark_cancelled(job, "Отменено: прервано между движками")
            run.status = benchmark.RUN_ERROR
            run.error = job.error or job.message
            logger.info("Сравнение %s остановлено (%s)", task.run_id, job.status.value)
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.message = "Ошибка"
            run.status = benchmark.RUN_ERROR
            run.error = str(exc)
            logger.exception("Сравнение %s упало", task.run_id)
        finally:
            self._current_job_id = None
            job.finished_at = _now_iso()
            run.finished_at = job.finished_at

    async def _keep_current(
        self, job: Job, index: int, settings: RenderSettings, source_path: Path
    ) -> None:
        """Сохраняет текущее содержимое куска вариантом «исходный».

        Только при первой замене: дальше все предыдущие версии уже лежат
        вариантами, а вырезать кусок из готового файла — операция не бесплатная
        (файл читается целиком).
        """
        if job.variants.get(index):
            return
        chunk = await asyncio.to_thread(
            audio_pipeline.read_segment,
            source_path, settings.output_format, job.segments[index],
        )
        # Исходный идёт вне нумерации (`v0`), чтобы первый пересинтез назывался
        # «вариант 1», а не «вариант 2»: в списке это порядок версий для человека.
        variant = await asyncio.to_thread(
            audio_pipeline.save_chunk_variant,
            job.id, index, "v0", chunk, self.active_seed(job, index), "исходный",
            self.active_qa(job, index),
            # Исходный вариант — вырезанный из файла кусок: его диагностика
            # считается по тому же срезу, что слышит пользователь.
            await asyncio.to_thread(
                take_quality.measure,
                chunk,
                job.payload.replicas[index].text,
                qa=self.active_qa(job, index),
            ),
        )
        job.variants[index] = [variant]
        job.active_variant[index] = variant.id

    async def _add_variant(
        self,
        job: Job,
        index: int,
        chunk: np.ndarray,
        seed: int | None,
        qa: QaOutcome | None = None,
        quality: take_quality.TakeQuality | None = None,
        plan: dict | None = None,
    ) -> audio_pipeline.ChunkVariant:
        """Добавляет новый вариант куска и делает его текущим."""
        seq = self._take_seq(job, index)
        variant = await asyncio.to_thread(
            audio_pipeline.save_chunk_variant,
            job.id, index, f"v{seq}", chunk, seed, f"вариант {seq}", qa, quality, plan,
        )
        job.variants.setdefault(index, []).append(variant)
        job.active_variant[index] = variant.id
        self._drop_extra_variants(job, index)
        return variant

    def _drop_extra_variants(self, job: Job, index: int) -> None:
        """Держит не больше `config.MAX_REPLICA_VARIANTS` вариантов на реплику.

        Вытесняется самый старый из тех, что не стоят в файле: активный вариант
        удалять нельзя — это и есть текущее звучание, а не история.
        """
        variants = job.variants[index]
        while len(variants) > config.MAX_REPLICA_VARIANTS:
            active = job.active_variant.get(index)
            victim = next((item for item in variants if item.id != active), None)
            if victim is None:  # остались только активные — вытеснять нечего
                return
            variants.remove(victim)
            audio_pipeline.drop_variant(victim)
            logger.info(
                "Задача %s: вариант «%s» реплики %s вытеснен",
                job.id, victim.label, index + 1,
            )

    @staticmethod
    def _take_seq(job: Job, index: int) -> int:
        """Порядковый номер следующего варианта этой реплики.

        Номер, а не случайный id: он попадает и в имя файла, и в подпись в
        интерфейсе («вариант 2»), и по нему видно порядок версий.
        """
        seq = job.variant_seq.get(index, 0) + 1
        job.variant_seq[index] = seq
        return seq

    @staticmethod
    def _shift_segments(
        segments: list[tuple[int, int]], index: int, bounds: tuple[int, int]
    ) -> None:
        """Записывает новые границы куска и сдвигает все следующие.

        Куски хранятся как смещения в сэмплах, поэтому замена реплики другой длины
        сдвигает хвост файла: без сдвига следующая перегенерация разрезала бы файл
        не там, где стоит реплика.
        """
        delta = bounds[1] - segments[index][1]
        segments[index] = bounds
        if not delta:
            return
        for position in range(index + 1, len(segments)):
            begin, end = segments[position]
            segments[position] = (begin + delta, end + delta)

    @staticmethod
    def _progress_callback(
        job: Job,
        plan: list[eta.EtaStep] | None = None,
        tracker: eta.EtaTracker | None = None,
        project_id: str | None = None,
    ) -> ProgressCallback:
        """Ход рендера: текущая реплика и оценка остатка по плану.

        Оценка здесь только уточняет уже показанную: настоящий пересчёт делает
        `_timing_callback` по факту законченного куска. Прежнее «среднее по
        длительности аудио × число реплик» убрано: одно число на все реплики
        не различало ни движки, ни длину текста, ни холодный старт, ни проверку.

        Заодно здесь реплика помечается `rendering` в базе: колбэк вызывается
        ровно перед передачей куска движку, и после падения воркера видно, на
        какой реплике остановился рендер (creash_report §7). Продолжение рендера
        (`resume`) пропускает готовые реплики и, значит, не переписывает их
        статус задним числом.
        """

        def on_progress(index: int, total: int, label: str) -> None:
            job.current_replica = index - 1
            job.current_voice = label
            job.message = f"Реплика {index} из {total} — {label}"
            if project_id:
                JobQueue._set_replica_status(
                    project_id, index - 1, config.REPLICA_STATUS_RENDERING
                )
            if plan and tracker is not None:
                # Index у пайплайна с единицы: закончены куски до index - 1.
                job.eta_sec = tracker.estimate(plan[index - 1 :])

        return on_progress

    @staticmethod
    def _engine_loaded(engine_id: str) -> bool:
        """Движок уже поднят (или поднимается) — холодный старт ему не нужен.

        Проверка ничего не создаёт: `created_engine` смотрит только на живые
        экземпляры реестра, поэтому ETA не поднимает модель ради оценки.
        `loading` считается тёплым: загрузка уже началась (например, прогрев
        при старте сервера), и второй раз её оплачивать не нужно.
        """
        engine = created_engine(engine_id)
        return engine is not None and engine.state in (STATE_READY, STATE_LOADING)

    def _render_plan(self, payload: JobPayload) -> list[eta.EtaStep]:
        """План рендера: по куску на реплику — движок, длина текста, проверка.

        Строится до старта и по тем же данным, что уйдут в синтез: движок
        берётся у эффективного голоса реплики (правка карточки важнее слота),
        а «холодный» кусок — только первый для ещё не поднятого движка. Именно
        поэтому ETA не может быть одним средним на число реплик: у кусков
        разные движки, длины и цена проверки.
        """
        qa_mode = eta.qa_mode(payload.settings.qa)
        seen_engines: set[str] = set()
        steps: list[eta.EtaStep] = []
        for replica in payload.replicas:
            speaker = payload.speakers.get(replica.voice)
            voice_id = str(replica.voice_id or (speaker.voice_id if speaker else ""))
            engine = audio_pipeline.voice_engine(voice_id)
            # Холодный — ровно первый кусок каждого движка, который ещё не поднят;
            # у остальных его кусков загрузка уже оплачена первым.
            cold = engine not in seen_engines and not self._engine_loaded(engine)
            seen_engines.add(engine)
            steps.append(
                eta.EtaStep(
                    engine=engine,
                    chars=audio_pipeline.replica_chars(replica.text),
                    qa_mode=qa_mode,
                    cold=cold,
                )
            )
        return steps

    @staticmethod
    def _engine_loaded_callback(tracker: eta.EtaTracker):
        """Загрузка движка — это и есть его холодный старт; запоминаем цену."""

        def on_loaded(engine_id: str, seconds: float) -> None:
            tracker.note_load(engine_id, seconds)
            logger.info("Движок %s поднят за %.1f c (учтено в ETA)", engine_id, seconds)

        return on_loaded

    @staticmethod
    def _timing_callback(job: Job, plan: list[eta.EtaStep], tracker: eta.EtaTracker):
        """Факт куска — в статистику, оценка остатка — в статус задачи.

        Кусков «до индекса» уже нет: план знает о них только по оценке, а их
        настоящее время пришло этим колбэком. Поэтому остаток считается от
        `index + 1`, а не от начала: выполненная работа обязана уменьшать
        остаток, а не подтверждать прежнюю оценку.
        """

        def on_chunk_timing(
            index: int, seconds: float, audio_sec: float, qa_outcome: QaOutcome | None
        ) -> None:
            if not 0 <= index < len(plan):
                return
            step = plan[index]
            tracker.observe(
                engine=step.engine,
                chars=step.chars,
                render_sec=seconds,
                qa_mode=step.qa_mode,
                attempts=qa_outcome.attempts if qa_outcome is not None else 1,
                cold=step.cold,
                audio_sec=audio_sec,
            )
            remaining = tracker.estimate(plan[index + 1 :])
            job.eta_sec = remaining

        return on_chunk_timing

    @staticmethod
    def _note_callback(job: Job) -> NoteCallback:
        """Ход строгой проверки — в то же сообщение задачи, что и прогресс.

        Отдельного поля нет намеренно: это не состояние задачи, а пояснение к
        текущему шагу («попытка 2 из 4»), и рядом с «Реплика 3 из 10» оно читается
        ровно так же, как остальные сообщения воркера.
        """

        def on_note(text: str) -> None:
            job.message = text

        return on_note

    def request_abort(self, reason: str) -> str | None:
        """Прервать текущую задачу между кусками (использует ResourceGuard).

        Возвращает id задачи, которая будет прервана, или None, если очередь
        свободна. Процесс приложения при этом не останавливается: прерывается
        только конкретная задача, чтобы память успела освободиться.

        Это тот же кооперативный механизм, что и у `cancel`, поэтому флаг ставится
        конкретной задаче: отмена одной задачи не должна задевать следующую.
        """
        job_id = self._current_job_id
        if job_id is None:
            return None
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if self.cancel(job_id) is None:
            return None
        # Порядок важен: `cancel` помечает остановку как пользовательскую, а здесь
        # причина известна точнее. Причина важнее самого факта — по ней
        # `_finish_interrupted` покажет ошибку вместо тихого «отменено».
        job.abort_kind = ABORT_WATCHDOG
        job.abort_reason = reason
        job.message = f"Прерываю: {reason}"
        logger.warning("Запрошено прерывание задачи %s: %s", job_id, reason)
        return job_id


_queue_instance: JobQueue | None = None


def get_queue() -> JobQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = JobQueue()
    return _queue_instance
