"""Реплики проекта: порядок, текст, спикер и назначенный голос."""

import sqlite3

from ... import config
from . import dumps, loads


def row_to_replica(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "index": row["idx"],
        "text": row["text"],
        "speaker": row["speaker"],
        "voice_id": row["voice_id"],
        # Собственный голос реплики; None — наследует голос спикера. Отдельно от
        # voice_id, потому что «как звучит» и «кто это выбрал» — разные вопросы:
        # без этого поля сброс правки было бы некуда вернуть.
        "voice_override": row["voice_override"],
        "overrides": loads(row["overrides"], {}),
        "status": row["status"],
        "selected_take_id": row["selected_take_id"],
    }


class ReplicasRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def list_for_project(self, project_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM replicas WHERE project_id = ? ORDER BY idx", (project_id,)
        ).fetchall()
        return [row_to_replica(row) for row in rows]

    def get(self, replica_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM replicas WHERE id = ?", (replica_id,)).fetchone()
        return None if row is None else row_to_replica(row)

    def by_index(self, project_id: str, index: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM replicas WHERE project_id = ? AND idx = ?", (project_id, index)
        ).fetchone()
        return None if row is None else row_to_replica(row)

    def sync_voices(self, project_id: str, speakers: dict[str, dict]) -> None:
        """Разносит назначенный голос спикера по его репликам.

        Реплики с собственным голосом не трогаются: пользователь выбрал голос
        именно этой реплике, и смена голоса спикера не должна её переписывать.
        """
        for key, data in speakers.items():
            if "voice_id" not in data:
                continue
            self._conn.execute(
                "UPDATE replicas SET voice_id = ? WHERE project_id = ? AND speaker = ?"
                " AND voice_override IS NULL",
                (str(data.get("voice_id") or ""), project_id, key),
            )

    def replace(self, project_id: str, replicas: list[dict]) -> None:
        """Приводит список реплик проекта к новому результату разбора.

        Реплика, у которой текст и спикер не изменились, остаётся той же строкой
        в базе: у неё сохраняются варианты и выбранный take. Иначе правка одной
        фразы в исходном тексте обнуляла бы уже проделанную работу по всему
        диалогу, хотя изменилась одна строчка.
        """
        existing = {
            int(row["idx"]): row
            for row in self._conn.execute(
                "SELECT * FROM replicas WHERE project_id = ?", (project_id,)
            )
        }
        keep: list[int] = []
        for item in replicas:
            index = int(item["index"])
            keep.append(index)
            old = existing.get(index)
            same = (
                old is not None
                and old["text"] == item["text"]
                and old["speaker"] == item["speaker"]
            )
            if same:
                # Эффективный голос пересчитывается от наследования: у реплики с
                # собственным голосом повторный разбор не должен его отбирать.
                self._conn.execute(
                    "UPDATE replicas SET voice_id = COALESCE(voice_override, ?),"
                    " overrides = ? WHERE id = ?",
                    (
                        str(item.get("voice_id", old["voice_id"])),
                        dumps(item.get("overrides", loads(old["overrides"], {}))),
                        old["id"],
                    ),
                )
                continue
            if old is not None:
                self._conn.execute("DELETE FROM replicas WHERE id = ?", (old["id"],))
            self._conn.execute(
                "INSERT INTO replicas (project_id, idx, text, speaker, voice_id, overrides,"
                " status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    index,
                    item["text"],
                    item["speaker"],
                    str(item.get("voice_id", "")),
                    dumps(item.get("overrides", {})),
                    config.REPLICA_STATUS_PENDING,
                ),
            )
        if keep:
            placeholders = ", ".join("?" for _ in keep)
            self._conn.execute(
                f"DELETE FROM replicas WHERE project_id = ? AND idx NOT IN ({placeholders})",
                (project_id, *keep),
            )
        else:
            self._conn.execute("DELETE FROM replicas WHERE project_id = ?", (project_id,))

    def set_voice(self, replica_id: int, voice_id: str | None) -> bool:
        """Голос одной реплики: свой (`voice_id`) или наследуемый (`None`).

        Сброс к наследуемому — не «пустая строка», а возврат к текущему голосу
        спикера: иначе реплика осталась бы без голоса и рендер упал бы.
        """
        if voice_id:
            cursor = self._conn.execute(
                "UPDATE replicas SET voice_override = ?, voice_id = ? WHERE id = ?",
                (voice_id, voice_id, replica_id),
            )
            return cursor.rowcount > 0
        cursor = self._conn.execute(
            "UPDATE replicas SET voice_override = NULL, voice_id = COALESCE(("
            " SELECT s.voice_id FROM speakers s JOIN replicas r"
            " ON r.project_id = s.project_id AND r.speaker = s.key WHERE r.id = ?"
            "), '') WHERE id = ?",
            (replica_id, replica_id),
        )
        return cursor.rowcount > 0

    def set_overrides(self, replica_id: int, overrides: dict) -> bool:
        cursor = self._conn.execute(
            "UPDATE replicas SET overrides = ? WHERE id = ?", (dumps(overrides), replica_id)
        )
        return cursor.rowcount > 0

    def set_status(self, replica_id: int, status: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE replicas SET status = ? WHERE id = ?", (status, replica_id)
        )
        return cursor.rowcount > 0

    def select_take(self, replica_id: int, take_id: int | None) -> bool:
        cursor = self._conn.execute(
            "UPDATE replicas SET selected_take_id = ?, status = ? WHERE id = ?",
            (
                take_id,
                config.REPLICA_STATUS_RENDERED
                if take_id is not None
                else config.REPLICA_STATUS_PENDING,
                replica_id,
            ),
        )
        return cursor.rowcount > 0
