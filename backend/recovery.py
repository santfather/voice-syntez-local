"""Восстановление после перезапуска приложения (creash_report §15).

Очередь задач и воркеры синтеза живут в памяти процесса: после падения или
перезапуска бэкенда незавершённая работа не продолжается. В базе при этом
остаются состояния «идёт» — проект `rendering`, реплика `rendering`, анализ
`analyzing`, — и без восстановления они превращаются в вечное «синтез идёт»:
проект выглядит занятым, повторный рендер кажется ненужным, а недописанные файлы
аудио копятся в каталогах.

Модуль делает четыре вещи и только их:

* переводит состояния «идёт» в прерванные (`interrupt_stale_state`);
* записывает причину в диагностику падений (`worker_crashes`) — чтобы после
  рестарта было видно, что именно прервалось;
* помечает осиротевшие задачи очереди (`jobs`) прерванными — очередь живёт в
  памяти, и её задача без этого исчезала бы бесследно;
* убирает недописанные `*.part`, оставшиеся от прерванной записи.

Готовые файлы, варианты реплик и всё остальное не трогается: восстановление не
имеет права удалять то, что уже сделано.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import audio_pipeline, config
from .db.connection import transaction
from .db.repositories.jobs import JobsRepository
from .db.store import get_projects_store
from .engines import worker_protocol as proto

logger = logging.getLogger("tts.recovery")


@dataclass
class RecoveryReport:
    """Что именно восстановлено — для лога и тестов."""

    projects: list[dict] = field(default_factory=list)
    replicas: int = 0
    analyses: int = 0
    partial_files: int = 0
    crashes: int = 0
    # Задачи очереди, оставшиеся «в работе» с прошлого запуска.
    jobs: int = 0

    @property
    def empty(self) -> bool:
        return not (
            self.projects or self.replicas or self.analyses or self.partial_files or self.jobs
        )

    def to_dict(self) -> dict:
        return {
            "projects": len(self.projects),
            "replicas": self.replicas,
            "analyses": self.analyses,
            "partial_files": self.partial_files,
            "crashes": self.crashes,
            "jobs": self.jobs,
        }


def recover_after_restart() -> RecoveryReport:
    """Приводит базу и каталоги вывода в согласованное состояние при старте.

    Ошибка восстановления не должна мешать приложению подняться: без него
    пользователь увидит неверные статусы, но работать сможет, а падение на старте
    не оставило бы вообще ничего. Поэтому каждая часть под своим `try`.
    """
    report = RecoveryReport()
    store = get_projects_store()

    try:
        stale = store.interrupt_stale_state()
    except Exception as exc:  # noqa: BLE001 — подниматься важнее, чем восстановиться
        logger.error("Не удалось восстановить состояния после перезапуска: %s", exc)
        stale = {"projects": [], "replicas": 0, "replica_indexes": [], "analyses": 0}

    report.projects = list(stale.get("projects", []))
    report.replicas = int(stale.get("replicas", 0))
    report.analyses = int(stale.get("analyses", 0))

    # Диагностика по каждому прерванному проекту: после рестарта задача из памяти
    # исчезла, и единственное место, где останется причина, — эта запись.
    for project in report.projects:
        try:
            store.record_worker_crash(
                job_id=str(project.get("job_id") or ""),
                project_id=str(project["id"]),
                error_type=proto.ERROR_INTERRUPTED,
                message=config.RECOVERY_RENDER_MESSAGE,
                reason="перезапуск приложения",
                retry_count=0,
                attempt=1,
            )
            report.crashes += 1
        except Exception as exc:  # noqa: BLE001 — запись вторична
            logger.warning("Проект %s: не удалось записать прерывание (%s)", project.get("id"), exc)

    # Задачи очереди: в памяти их после рестарта нет, а строки в `jobs` остались
    # в `queued`/`processing`. Без этой пометки задача выглядела бы идущей вечно —
    # и в интерфейсе, и в диагностике.
    try:
        with transaction() as connection:
            orphaned = JobsRepository(connection).mark_unfinished_interrupted(
                config.RECOVERY_QUEUE_MESSAGE, proto.ERROR_INTERRUPTED
            )
        report.jobs = len(orphaned)
    except Exception as exc:  # noqa: BLE001 — старт важнее полноты восстановления
        logger.warning("Не удалось пометить прерванные задачи очереди: %s", exc)

    try:
        report.partial_files = audio_pipeline.cleanup_partials()
    except Exception as exc:  # noqa: BLE001 — остатки на диске не повод не стартовать
        logger.warning("Не удалось убрать недописанные файлы: %s", exc)

    if report.empty:
        logger.info("Восстановление после перезапуска: нечего восстанавливать")
    else:
        logger.warning(
            "Восстановление после перезапуска: проектов %d, реплик %d, анализов %d, "
            "задач очереди %d, недописанных файлов %d",
            len(report.projects), report.replicas, report.analyses, report.jobs,
            report.partial_files,
        )
    return report
