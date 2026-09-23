"""Задачи очереди в базе (F-Q1).

Очередь живёт в памяти процесса, и это правильно: она держит `asyncio`-очередь и
исполняет синтез. Но запись о задаче должна переживать рестарт: без неё после
краха не остаётся ни статуса, ни причины, и пользователь не отличает «задача
пропала» от «задача никогда не ставилась». Таблица хранит снимок задачи, а
репозиторий умеет ровно две вещи: сохранить снимок и найти осиротевшие задачи.

Снимок, а не журнал переходов: очередь пишет текущее состояние (`upsert`), потому
что история статусов одной задачи ничего не добавляет к её итогу, а место в базе
занимает.
"""

import sqlite3
from datetime import datetime, timezone

# Статусы, в которых задача считается незавершённой: если после старта
# приложения строка осталась в одном из них, задачи в памяти уже нет.
UNFINISHED_STATUSES = ("queued", "processing")

# Колонки снимка: имена совпадают с полями `Job`, чтобы вызывающему коду не
# приходилось переводить одно в другое.
COLUMNS = (
    "id",
    "status",
    "total_replicas",
    "current_replica",
    "current_voice",
    "output_format",
    "message",
    "error",
    "error_type",
    "output_path",
    "duration_sec",
    "created_at",
    "started_at",
    "finished_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def row_to_job(row: sqlite3.Row) -> dict:
    return {name: row[name] for name in COLUMNS}


class JobsRepository:
    """Снимки задач: сохранить и прочитать осиротевшие."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def save(self, row: dict) -> None:
        """Записывает снимок задачи целиком (по `id` — вставка или обновление)."""
        values = tuple(row.get(name) for name in COLUMNS)
        placeholders = ", ".join("?" for _ in COLUMNS)
        updates = ", ".join(f"{name} = excluded.{name}" for name in COLUMNS if name != "id")
        self._conn.execute(
            f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES ({placeholders})"
            f" ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )

    def list_unfinished(self) -> list[dict]:
        """Задачи, оставшиеся «в работе» — то есть осиротевшие после рестарта."""
        placeholders = ", ".join("?" for _ in UNFINISHED_STATUSES)
        rows = self._conn.execute(
            f"SELECT * FROM jobs WHERE status IN ({placeholders})"
            " ORDER BY created_at, rowid",
            UNFINISHED_STATUSES,
        ).fetchall()
        return [row_to_job(row) for row in rows]

    def mark_unfinished_interrupted(self, message: str, error_type: str) -> list[dict]:
        """Переводит незавершённые задачи в ошибку с причиной. Возвращает их.

        Возвращаются строки **до** обновления: в отчёте восстановления нужен
        прежний статус («в очереди» против «синтезировалась») — по нему видно,
        успела задача начаться или нет.
        """
        rows = self.list_unfinished()
        if not rows:
            return []
        placeholders = ", ".join("?" for _ in UNFINISHED_STATUSES)
        self._conn.execute(
            "UPDATE jobs SET status = 'error', error = ?, error_type = ?,"
            " finished_at = ? WHERE status IN (" + placeholders + ")",
            (message, error_type, _now(), *UNFINISHED_STATUSES),
        )
        return rows

    def prune(self, keep: int) -> int:
        """Оставляет последние `keep` завершённых задач. Возвращает число удалённых.

        Незавершённые не удаляются никогда: их разбирает восстановление при
        старте, и «убрать вместе с историей» означало бы снова спрятать следы.
        """
        cursor = self._conn.execute(
            "DELETE FROM jobs WHERE status NOT IN (?, ?) AND id NOT IN ("
            " SELECT id FROM jobs ORDER BY rowid DESC LIMIT ?)",
            (*UNFINISHED_STATUSES, int(keep)),
        )
        return int(cursor.rowcount or 0)
