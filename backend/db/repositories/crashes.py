"""Диагностика падений процесса синтеза (creash_report).

Отдельная таблица, а не поля реплики или задачи: падение относится к процессу и
движку, случается и вне проекта (разовые задачи), и его ценность — в истории.
Запись переживает перезапуск приложения: задача в очереди живёт в памяти, а
вопрос «почему вчера упало» задаётся уже после рестарта.

Тексты реплик здесь не хранятся: только индекс реплики, движок, коды и причина.
Диагностика не должна становиться вторым архивом пользовательского текста —
для логов есть отпечаток (`worker_protocol.text_fingerprint`).
"""

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def row_to_crash(row: sqlite3.Row) -> dict:
    """Запись для API: имена полей совпадают с колонками, чтобы не плодить словари."""
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "job_id": row["job_id"],
        "project_id": row["project_id"],
        "replica_id": row["replica_id"],
        "replica_index": row["replica_index"],
        "engine": row["engine"],
        "error_type": row["error_type"],
        "message": row["message"],
        "pid": row["pid"],
        "exit_code": row["exit_code"],
        "signal": row["signal"],
        "signal_name": row["signal_name"],
        "reason": row["reason"],
        "retry_count": row["retry_count"],
        "attempt": row["attempt"],
        "started_at": row["started_at"],
        "interrupted_at": row["interrupted_at"],
        "memory_percent": row["memory_percent"],
    }


class WorkerCrashesRepository:
    """Записи о падениях воркеров: добавить и прочитать последние."""

    COLUMNS = (
        "created_at",
        "job_id",
        "project_id",
        "replica_id",
        "replica_index",
        "engine",
        "error_type",
        "message",
        "pid",
        "exit_code",
        "signal",
        "signal_name",
        "reason",
        "retry_count",
        "attempt",
        "started_at",
        "interrupted_at",
        "memory_percent",
    )

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def add(
        self,
        *,
        job_id: str = "",
        project_id: str | None = None,
        replica_id: int | None = None,
        replica_index: int | None = None,
        engine: str = "",
        error_type: str = "",
        message: str = "",
        pid: int | None = None,
        exit_code: int | None = None,
        signal_number: int | None = None,
        signal_name: str | None = None,
        reason: str = "",
        retry_count: int = 0,
        attempt: int = 1,
        started_at: str | None = None,
        interrupted_at: str | None = None,
        memory_percent: float | None = None,
        created_at: str | None = None,
    ) -> dict:
        """Записывает падение воркера и отдаёт сохранённую строку целиком."""
        cursor = self._conn.execute(
            "INSERT INTO worker_crashes (created_at, job_id, project_id, replica_id,"
            " replica_index, engine, error_type, message, pid, exit_code, signal,"
            " signal_name, reason, retry_count, attempt, started_at, interrupted_at,"
            " memory_percent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                created_at or _now(),
                job_id,
                project_id,
                replica_id,
                replica_index,
                engine,
                error_type,
                message,
                pid,
                exit_code,
                signal_number,
                signal_name,
                reason,
                int(retry_count),
                int(attempt),
                started_at,
                interrupted_at,
                None if memory_percent is None else float(memory_percent),
            ),
        )
        row = self._conn.execute(
            "SELECT * FROM worker_crashes WHERE id = ?", (int(cursor.lastrowid),)
        ).fetchone()
        return row_to_crash(row)

    def list_recent(self, limit: int = 20, project_id: str | None = None) -> list[dict]:
        """Свежие падения, не больше двухсот; проект сужает выборку."""
        limit = max(1, min(int(limit), 200))
        if project_id:
            rows = self._conn.execute(
                "SELECT * FROM worker_crashes WHERE project_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM worker_crashes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [row_to_crash(row) for row in rows]

    def count_for_replica(self, project_id: str, index: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total FROM worker_crashes WHERE project_id = ? AND replica_index = ?",
            (project_id, int(index)),
        ).fetchone()
        return int(row["total"]) if row else 0
