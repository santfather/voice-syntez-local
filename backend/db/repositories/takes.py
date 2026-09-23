"""Варианты (takes) реплик: готовое аудио и то, чем оно получено."""

import sqlite3
from datetime import datetime, timezone

from . import dumps, loads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def row_to_take(row: sqlite3.Row) -> dict:
    """Собирает вариант из строки БД; пустые `qa`/`quality` — это `None`.

    «Метрик нет» и «метрики пусты» должны различаться, поэтому пустая колонка
    не превращается в пустой объект.
    """
    return {
        "id": row["id"],
        "replica_id": row["replica_id"],
        "label": row["label"],
        "audio_path": row["audio_path"],
        "seed": row["seed"],
        "engine": row["engine"],
        "parameters": loads(row["parameters"], {}),
        "duration_sec": row["duration_sec"],
        "qa": loads(row["qa"], {}) if row["qa"] else None,
        # Диагностика take'а. Пустая колонка (записи до фазы 14) — это `None`, а
        # не пустой объект: «метрик нет» и «метрики пусты» должны различаться.
        "quality": loads(row["quality"], {}) if row["quality"] else None,
        "created_at": row["created_at"],
    }


class TakesRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def add(
        self,
        replica_id: int,
        audio_path: str,
        label: str = "",
        seed: int | None = None,
        engine: str = "",
        parameters: dict | None = None,
        duration_sec: float = 0.0,
        qa: dict | None = None,
        quality: dict | None = None,
    ) -> dict:
        """Записывает готовое аудио варианта и отдаёт сохранённую строку."""
        cursor = self._conn.execute(
            "INSERT INTO takes (replica_id, label, audio_path, seed, engine, parameters,"
            " duration_sec, qa, quality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                replica_id,
                label,
                audio_path,
                None if seed is None else int(seed),
                engine,
                dumps(parameters or {}),
                float(duration_sec),
                None if qa is None else dumps(qa),
                None if quality is None else dumps(quality),
                _now(),
            ),
        )
        return self.get(int(cursor.lastrowid))  # type: ignore[return-value]

    def get(self, take_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
        return None if row is None else row_to_take(row)

    def list_for_replica(self, replica_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM takes WHERE replica_id = ? ORDER BY id", (replica_id,)
        ).fetchall()
        return [row_to_take(row) for row in rows]

    def list_for_project(self, project_id: str) -> list[dict]:
        """Варианты всех реплик проекта с индексом реплики — для карточки проекта."""
        rows = self._conn.execute(
            "SELECT t.*, r.idx AS replica_idx FROM takes t"
            " JOIN replicas r ON r.id = t.replica_id WHERE r.project_id = ? ORDER BY t.id",
            (project_id,),
        ).fetchall()
        return [{**row_to_take(row), "index": row["replica_idx"]} for row in rows]

    def delete(self, take_id: int) -> str | None:
        """Удаляет вариант и возвращает путь файла — его удаляет вызывающий."""
        row = self._conn.execute("SELECT audio_path FROM takes WHERE id = ?", (take_id,)).fetchone()
        if row is None:
            return None
        self._conn.execute("DELETE FROM takes WHERE id = ?", (take_id,))
        return str(row["audio_path"])

    def prune(self, replica_id: int, limit: int) -> list[str]:
        """Оставляет у реплики не больше `limit` вариантов, вытесняя самые старые.

        Возвращает пути вытесненных файлов: строки удаляются здесь, а файлы —
        вызывающим, потому что репозиторий о диске ничего не знает (см. store).

        Активный (`selected_take_id`) вариант не вытесняется никогда: это текущее
        звучание реплики, а не история, и его удаление сломало бы воспроизведение
        при следующем же открытии проекта.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT id, audio_path FROM takes WHERE replica_id = ? ORDER BY id",
            (replica_id,),
        ).fetchall()
        excess = len(rows) - limit
        if excess <= 0:
            return []
        row = self._conn.execute(
            "SELECT selected_take_id FROM replicas WHERE id = ?", (replica_id,)
        ).fetchone()
        selected = None if row is None else row["selected_take_id"]
        victims = [
            item
            for item in rows
            if selected is None or int(item["id"]) != int(selected)
        ][:excess]
        for item in victims:
            self._conn.execute("DELETE FROM takes WHERE id = ?", (item["id"],))
        return [str(item["audio_path"]) for item in victims]

    def paths(self, project_id: str) -> list[str]:
        """Пути файлов всех вариантов проекта: нужны при удалении проекта."""
        rows = self._conn.execute(
            "SELECT t.audio_path FROM takes t JOIN replicas r ON r.id = t.replica_id"
            " WHERE r.project_id = ?",
            (project_id,),
        ).fetchall()
        return [str(row["audio_path"]) for row in rows]
