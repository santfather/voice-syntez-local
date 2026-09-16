"""Таблица словаря произношения: пользовательские правила чтения терминов.

Репозиторий, как и остальные, ничего не знает про транзакции и кеш: он принимает
открытое соединение и работает в его рамках. Снимок активных правил и сброс кеша —
дело сервиса (`backend/pronunciation.py`).

Флаги хранятся целыми 0/1 (так их понимает SQLite), наружу отдаются булевыми:
и SQL, и Python-код должны видеть одно и то же правило одинаково.
"""

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_entry(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "source": row["source"],
        "target": row["target"],
        "case_sensitive": bool(row["case_sensitive"]),
        "whole_word": bool(row["whole_word"]),
        "enabled": bool(row["enabled"]),
        "note": row["note"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class PronunciationRepository:
    """CRUD правил словаря. Уникальность — по (`source`, `case_sensitive`)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def list_all(self) -> list[dict]:
        """Все правила, включая выключенные: список в интерфейсе показывает их тоже."""
        rows = self._conn.execute(
            "SELECT * FROM pronunciation_entries ORDER BY source COLLATE NOCASE, id"
        ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def list_enabled(self) -> list[dict]:
        """Только включённые, длинные источники первыми.

        Порядок задан и здесь, и в чистой логике: правило с более длинным
        источником обязано применяться раньше короткого, иначе «OpenAI API»
        развалилось бы на «OpenAI» + «API».
        """
        rows = self._conn.execute(
            "SELECT * FROM pronunciation_entries WHERE enabled = 1"
            " ORDER BY LENGTH(source) DESC, id"
        ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def get(self, entry_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM pronunciation_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return None if row is None else _row_to_entry(row)

    def find(self, source: str, case_sensitive: bool) -> dict | None:
        """Правило с тем же источником и режимом регистра — цель повторного добавления."""
        row = self._conn.execute(
            "SELECT * FROM pronunciation_entries WHERE source = ? AND case_sensitive = ?",
            (source, int(bool(case_sensitive))),
        ).fetchone()
        return None if row is None else _row_to_entry(row)

    def create(
        self,
        source: str,
        target: str,
        case_sensitive: bool = False,
        whole_word: bool = True,
        enabled: bool = True,
        note: str = "",
    ) -> dict:
        now = _now()
        cursor = self._conn.execute(
            "INSERT INTO pronunciation_entries"
            " (source, target, case_sensitive, whole_word, enabled, note, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                target,
                int(bool(case_sensitive)),
                int(bool(whole_word)),
                int(bool(enabled)),
                note,
                now,
                now,
            ),
        )
        return self.get(int(cursor.lastrowid))  # type: ignore[return-value]

    def update(self, entry_id: int, **fields) -> dict | None:
        """Меняет переданные поля правила; незнакомые ключи отбрасываются."""
        allowed = {"source", "target", "case_sensitive", "whole_word", "enabled", "note"}
        values: dict[str, object] = {}
        for key, value in fields.items():
            if key in ("case_sensitive", "whole_word", "enabled"):
                values[key] = int(bool(value))
            elif key in allowed:
                values[key] = value
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        cursor = self._conn.execute(
            f"UPDATE pronunciation_entries SET {assignments} WHERE id = ?",
            (*values.values(), entry_id),
        )
        if cursor.rowcount == 0:
            return None
        return self.get(entry_id)

    def delete(self, entry_id: int) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM pronunciation_entries WHERE id = ?", (entry_id,)
        )
        return cursor.rowcount > 0
