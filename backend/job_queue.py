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

from . import audio_pipeline, config, resource_guard
from .audio_pipeline import (
    NoteCallback,
    ProgressCallback,
    QaOutcome,
    RenderSettings,
    SpeakerSettings,
)
from .dialogue_parser import Replica

logger = logging.getLogger(__name__)

MAX_KEPT_JOBS = 50
# Как часто перепроверять системную память, пока задача ждёт своей очереди.
MEMORY_WAIT_RETRY_SEC = 5.0


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


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
    output_path: Path | None = None
    duration_sec: float | None = None
    eta_sec: float | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    # Данные готовой задачи: без них нельзя перегенерировать отдельную реплику.
    payload: "JobPayload | None" = field(default=None, repr=False)
    segments: list[tuple[int, int]] = field(default_factory=list, repr=False)
    # Сид каждого куска исходной сборки (`None` — движок сид не принимает).
    seeds: list[int | None] = field(default_factory=list, repr=False)
    # Итог строгой проверки по индексу реплики (`None` — проверка выключена или
    # кусок пришёл из варианта). Нужен, чтобы показать принятое без полного
    # прохождения аудио как непроверенное, а не как обычное.
    qa: list[QaOutcome | None] = field(default_factory=list, repr=False)
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
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec else None,
            "eta_sec": round(self.eta_sec, 1) if self.eta_sec else None,
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


class JobQueue:
    """Единственный воркер: следующая задача не начнётся, пока не закончится текущая."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue | None = None
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._worker_task: asyncio.Task | None = None
        # Текущая задача и запрос на её прерывание (ставит ResourceGuard)
        self._current_job_id: str | None = None
        self._abort_requested = False

    # -- жизненный цикл --------------------------------------------------------
    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker(), name="tts-worker")
        logger.info("Воркер очереди запущен")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None
        self._queue = None
        logger.info("Воркер очереди остановлен")

    # -- API -------------------------------------------------------------------
    def submit(self, payload: JobPayload) -> Job:
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
        self._queue.put_nowait(RenderTask(job_id=job.id, payload=payload))
        logger.info("Задача %s принята (%s реплик)", job.id, job.total_replicas)
        return job

    def submit_regenerate(self, job_id: str, index: int) -> Job:
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
        self._enqueue(RegenerateTask(job_id=job_id, index=index))
        logger.info("Задача %s: перегенерация реплики %s", job_id, index + 1)
        return job

    def submit_variant(self, job_id: str, index: int, variant_id: str) -> Job:
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
        self._enqueue(SelectVariantTask(job_id=job_id, index=index, variant_id=variant.id))
        logger.info("Задача %s: реплика %s → вариант %s", job_id, index + 1, variant.id)
        return job

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

    def _enqueue(self, task: RenderTask | RegenerateTask | SelectVariantTask) -> None:
        queue = self._queue
        if queue is None:
            raise RuntimeError("Очередь не запущена")
        queue.put_nowait(task)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_size(self) -> int:
        return self._queue.qsize() if self._queue else 0

    # -- внутреннее ------------------------------------------------------------
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
            task = await queue.get()
            try:
                await self._wait_for_memory(task.job_id)
                if isinstance(task, RegenerateTask):
                    await self._regenerate(task)
                elif isinstance(task, SelectVariantTask):
                    await self._select_variant(task)
                else:
                    await self._render(task)
            finally:
                queue.task_done()

    async def _wait_for_memory(self, job_id: str) -> None:
        """Не начинает задачу, пока память занята не только этим процессом.

        Проверка стоит именно перед взятием job, а не в фоновом watchdog: задача
        должна остаться в очереди с понятным сообщением, а не стартовать и лечь
        дополнительной нагрузкой поверх уже существующей. Свой потолок
        (`MAX_RSS_MB`) при этом продолжает действовать отдельно.
        """
        job = self._jobs.get(job_id)
        warned = False
        while resource_guard.check_system_memory_pressure():
            percent = resource_guard.snapshot()["system_mem_percent"]
            if job is not None:
                job.message = f"Система перегружена: память занята на {percent:.0f}% — ожидание"
            if not warned:
                warned = True
                logger.warning(
                    "Задача %s ждёт: системная память %.0f%% выше порога %d%%",
                    job_id, percent, config.SYSTEM_MEM_THRESHOLD_PERCENT,
                )
            await asyncio.sleep(MEMORY_WAIT_RETRY_SEC)
        if warned:
            logger.info("Задача %s: память освободилась, начинаю", job_id)

    async def _render(self, task: RenderTask) -> None:
        job = self._jobs[task.job_id]
        payload = task.payload
        job.status = JobStatus.PROCESSING
        job.started_at = _now_iso()
        job.message = "Готовлю модель"
        self._current_job_id = job.id
        self._abort_requested = False
        started = time.monotonic()
        try:
            result = await audio_pipeline.render_dialogue(
                job_id=job.id,
                replicas=payload.replicas,
                speakers=payload.speakers,
                settings=payload.settings,
                on_progress=self._progress_callback(job),
                should_abort=lambda: self._abort_requested,
                wait_for_memory=lambda: self._wait_for_memory(job.id),
                on_note=self._note_callback(job),
            )
            job.status = JobStatus.DONE
            job.output_path = result.output_path
            job.duration_sec = result.duration_sec
            # Метаданные кусков нужны, чтобы задним числом заменить одну реплику.
            job.payload = payload
            job.segments = result.segments
            job.seeds = result.seeds
            job.qa = result.qa
            job.current_replica = job.total_replicas
            job.eta_sec = 0.0
            job.message = f"Готово: {result.duration_sec:.1f} c аудио"
            logger.info("Задача %s завершена за %.1f c", job.id, time.monotonic() - started)
        except asyncio.CancelledError:
            job.status = JobStatus.ERROR
            job.error = "Отменено (остановка сервера)"
            raise
        except audio_pipeline.JobAbortedError as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.message = "Прервано"
            logger.warning("Задача %s прервана watchdog'ом: %s", job.id, exc)
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.error = str(exc)
            job.message = "Ошибка"
            logger.exception("Задача %s упала", job.id)
        finally:
            self._current_job_id = None
            self._abort_requested = False
            job.finished_at = _now_iso()

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
            prepared, seed, qa_outcome = await audio_pipeline.synthesize_replica(
                replica=replica,
                speaker=speaker,
                settings=job.payload.settings,
                index=task.index,
                wait_for_memory=lambda: self._wait_for_memory(job.id),
                on_note=self._note_callback(job),
            )
            variant = await self._add_variant(job, task.index, prepared, seed, qa_outcome)
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
        except Exception as exc:
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
    ) -> audio_pipeline.ChunkVariant:
        """Добавляет новый вариант куска и делает его текущим."""
        seq = self._take_seq(job, index)
        variant = await asyncio.to_thread(
            audio_pipeline.save_chunk_variant,
            job.id, index, f"v{seq}", chunk, seed, f"вариант {seq}", qa,
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
    def _progress_callback(job: Job) -> ProgressCallback:
        durations: list[float] = []
        last = time.monotonic()

        def on_progress(index: int, total: int, label: str) -> None:
            nonlocal last
            now = time.monotonic()
            if index > 1:
                durations.append(now - last)
            last = now
            if durations:  # ETA по средней длительности уже готовых реплик
                job.eta_sec = (sum(durations) / len(durations)) * (total - index + 1)
            job.current_replica = index - 1
            job.current_voice = label
            job.message = f"Реплика {index} из {total} — {label}"

        return on_progress

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
        """
        if self._current_job_id is None:
            return None
        self._abort_requested = True
        logger.warning("Запрошено прерывание задачи %s: %s", self._current_job_id, reason)
        return self._current_job_id


_queue_instance: JobQueue | None = None


def get_queue() -> JobQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = JobQueue()
    return _queue_instance
