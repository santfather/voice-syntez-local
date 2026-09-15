"""Очередь задач: один воркер, генерация строго последовательная."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from . import audio_pipeline
from .audio_pipeline import ProgressCallback, RenderSettings, SpeakerSettings
from .dialogue_parser import Replica

logger = logging.getLogger(__name__)

MAX_KEPT_JOBS = 50


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
        }


@dataclass
class JobPayload:
    replicas: list[Replica]
    speakers: dict[str, SpeakerSettings]
    settings: RenderSettings


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
        self._queue.put_nowait((job.id, payload))
        logger.info("Задача %s принята (%s реплик)", job.id, job.total_replicas)
        return job

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
        while True:
            job_id, payload = await self._queue.get()
            job = self._jobs[job_id]
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
                )
                job.status = JobStatus.DONE
                job.output_path = result.output_path
                job.duration_sec = result.duration_sec
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
                self._queue.task_done()

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
