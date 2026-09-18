"""Проект и его спикеры."""

import sqlite3
import uuid
from datetime import datetime, timezone

from ... import config
from . import dumps, loads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_project(row: sqlite3.Row, replicas: int = 0) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "source_text": row["source_text"],
        "mode": row["mode"],
        "render_settings": loads(row["render_settings"], {}),
        "status": row["status"],
        "job_id": row["job_id"],
        "last_error": row["last_error"],
        # Готовность текста к синтезу — отдельная ось от `status` (см. миграцию 5).
        "analysis_status": row["analysis_status"],
        "analysis_version": row["analysis_version"],
        "analysis_error": row["analysis_error"],
        "analysis_started_at": row["analysis_started_at"],
        "analysis_finished_at": row["analysis_finished_at"],
        # Подстатус LLM-анализа (Task 2): отдельная ось, см. миграцию 8.
        "llm_analysis_status": row["llm_analysis_status"],
        "llm_analysis_model": row["llm_analysis_model"],
        "llm_analysis_error": row["llm_analysis_error"],
        "llm_analysis_updated_at": row["llm_analysis_updated_at"],
        "replicas_count": replicas,
    }


def _row_to_speaker(row: sqlite3.Row) -> dict:
    return {
        "key": row["key"],
        "label": row["label"],
        "voice_id": row["voice_id"],
        "overrides": loads(row["overrides"], {}),
    }


class ProjectsRepository:
    """CRUD проекта и назначений голосов по спикерам."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    # -- проект ----------------------------------------------------------------
    def create(
        self,
        name: str,
        source_text: str = "",
        mode: str = config.PROJECT_MODE_DIALOGUE,
        render_settings: dict | None = None,
    ) -> dict:
        project_id = uuid.uuid4().hex[:12]
        now = _now()
        self._conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at, source_text, mode,"
            " render_settings, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                name,
                now,
                now,
                source_text,
                mode,
                dumps(render_settings or {}),
                config.PROJECT_STATUS_DRAFT,
            ),
        )
        return self.get(project_id)  # type: ignore[return-value]

    def list_projects(self) -> list[dict]:
        """Проекты без реплик: в списке нужны только имя, статус и объём."""
        rows = self._conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM replicas r WHERE r.project_id = p.id)"
            " AS replicas_count FROM projects p ORDER BY p.updated_at DESC, p.id"
        ).fetchall()
        return [_row_to_project(row, int(row["replicas_count"])) for row in rows]

    def get(self, project_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM replicas r WHERE r.project_id = p.id)"
            " AS replicas_count FROM projects p WHERE p.id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_project(row, int(row["replicas_count"]))

    def list_by_status(self, status: str, *, column: str = "status") -> list[dict]:
        """Проекты в заданном состоянии — по колонке статуса рендера или анализа.

        `column` ограничен двумя известными именами: подставлять сюда произвольную
        строку из запроса нельзя, а два состояния, которые нужно восстанавливать
        после перезапуска (`rendering` и `analyzing`), различаются именно колонкой.
        """
        if column not in ("status", "analysis_status"):
            raise ValueError(f"Неизвестная колонка состояния: {column}")
        rows = self._conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM replicas r WHERE r.project_id = p.id)"
            f" AS replicas_count FROM projects p WHERE p.{column} = ? ORDER BY p.updated_at DESC",
            (status,),
        ).fetchall()
        return [_row_to_project(row, int(row["replicas_count"])) for row in rows]

    def update(self, project_id: str, **fields) -> dict | None:
        """Меняет переданные поля проекта; незнакомые ключи отбрасываются.

        Вызов без полей — это «проект изменился»: так отмечаются правки, которые
        лежат в других таблицах (назначение голосов, новый вариант реплики), иначе
        список проектов остался бы отсортированным по времени открытия, а не правки.
        """
        allowed = {
            "name",
            "source_text",
            "mode",
            "status",
            "job_id",
            "last_error",
            # Состояние подготовки текста пишется только через `store`: рендер
            # обязан видеть его согласованным со стадиями реплик, а прямая запись
            # из обработчика запроса легко оставила бы «ready» без стадий.
            "analysis_status",
            "analysis_version",
            "analysis_error",
            "analysis_started_at",
            "analysis_finished_at",
            # Подстатус LLM пишется только через `store`: он обязан меняться
            # согласованно с сохранёнными разборами и статусом подготовки.
            "llm_analysis_status",
            "llm_analysis_model",
            "llm_analysis_error",
            "llm_analysis_updated_at",
        }
        values: dict[str, object] = {}
        for key, value in fields.items():
            if key == "render_settings":
                values[key] = dumps(value or {})
            elif key in allowed:
                values[key] = value
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        cursor = self._conn.execute(
            f"UPDATE projects SET {assignments} WHERE id = ?",
            (*values.values(), project_id),
        )
        if cursor.rowcount == 0:
            return None
        return self.get(project_id)

    def delete(self, project_id: str) -> bool:
        """Удаляет проект; спикеры, реплики и варианты уходят каскадом."""
        cursor = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0

    # -- спикеры ---------------------------------------------------------------
    def speakers(self, project_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM speakers WHERE project_id = ? ORDER BY id", (project_id,)
        ).fetchall()
        return [_row_to_speaker(row) for row in rows]

    def replace_speakers(self, project_id: str, speakers: dict[str, dict]) -> None:
        """Приводит список спикеров проекта к переданному словарю.

        Голоса уже известных ключей сохраняются: повторный разбор текста после
        правки реплики не должен сбрасывать выбранные голоса.
        """
        existing = {item["key"]: item for item in self.speakers(project_id)}
        keep: list[str] = []
        for key, data in speakers.items():
            keep.append(key)
            voice_id = str(data.get("voice_id") or existing.get(key, {}).get("voice_id", ""))
            overrides = data.get("overrides")
            if overrides is None:
                overrides = existing.get(key, {}).get("overrides", {})
            label = str(data.get("label") or key)
            self._conn.execute(
                "INSERT INTO speakers (project_id, key, label, voice_id, overrides)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (project_id, key) DO UPDATE SET"
                " label = excluded.label, voice_id = excluded.voice_id, overrides = excluded.overrides",
                (project_id, key, label, voice_id, dumps(overrides)),
            )
        if keep:
            placeholders = ", ".join("?" for _ in keep)
            self._conn.execute(
                f"DELETE FROM speakers WHERE project_id = ? AND key NOT IN ({placeholders})",
                (project_id, *keep),
            )
        else:
            self._conn.execute("DELETE FROM speakers WHERE project_id = ?", (project_id,))

    def assign_voice(self, project_id: str, key: str, voice_id: str) -> bool:
        """Назначает голос спикеру; список реплик обновляется следом (см. store)."""
        cursor = self._conn.execute(
            "UPDATE speakers SET voice_id = ? WHERE project_id = ? AND key = ?",
            (voice_id, project_id, key),
        )
        return cursor.rowcount > 0

    def set_speaker_overrides(self, project_id: str, key: str, overrides: dict) -> bool:
        cursor = self._conn.execute(
            "UPDATE speakers SET overrides = ? WHERE project_id = ? AND key = ?",
            (dumps(overrides), project_id, key),
        )
        return cursor.rowcount > 0
