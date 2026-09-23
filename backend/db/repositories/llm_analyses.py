"""Таблица разборов локальной LLM: аннотации, их вход и версии.

Разбор хранится **отдельно от текста**, потому что это производная данных разного
времени жизни: текст реплики пользовательский и неизменяемый, а разбор зависит от
модели, prompt'а, схемы, контекста и словаря. Хранить их вместе значило бы либо
переписывать текст под ответ модели (запрещено), либо терять разбор при каждой
правке (тоже неверно — история нужна для аудита).

Действительность разбора решается сравнением хешей входа, а не временем: в строке
лежат `source_text_hash`, `context_hash`, `dictionary_hash`, `model_digest`,
`prompt_version`, `schema_version`, `inference_hash`. Совпали все — разбор ещё про
этот вход; любое расхождение делает его устаревшим (§11, §18 Task 2).

Репозиторий не знает про транзакции и не решает, что делать с устаревшим разбором:
он принимает соединение и работает в его рамках. Политика кеша — в `llm/analysis_cache.py`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# Поля, которые сравниваются при проверке действительности разбора.
CACHE_KEY_FIELDS: tuple[str, ...] = (
    "source_text_hash",
    "context_hash",
    "dictionary_hash",
    "model_digest",
    "prompt_version",
    "schema_version",
    "inference_hash",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_analysis(row: sqlite3.Row) -> dict:
    return {
        "analysis_id": row["analysis_id"],
        "project_id": row["project_id"],
        "replica_id": int(row["replica_id"]),
        "replica_index": int(row["replica_index"]),
        "source_text_hash": row["source_text_hash"],
        "context_hash": row["context_hash"],
        "dictionary_hash": row["dictionary_hash"],
        "model_tag": row["model_tag"],
        "model_digest": row["model_digest"],
        "prompt_version": row["prompt_version"],
        "schema_version": row["schema_version"],
        "inference_hash": row["inference_hash"],
        "analysis_json": row["analysis_json"],
        "status": row["status"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class LlmAnalysesRepository:
    """Хранение разборов: одна актуальная запись на реплику проекта."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def upsert(
        self,
        *,
        analysis_id: str,
        project_id: str,
        replica_id: int,
        replica_index: int,
        analysis_json: str,
        status: str,
        source_text_hash: str = "",
        context_hash: str = "",
        dictionary_hash: str = "",
        model_tag: str = "",
        model_digest: str = "",
        prompt_version: str = "",
        schema_version: str = "",
        inference_hash: str = "",
        error: str = "",
    ) -> dict:
        """Сохраняет разбор реплики, заменяя прежний.

        Одна запись на реплику, а не журнал: пользователю нужен актуальный разбор, а
        история изменений текста и так живёт в проекте. `created_at` при замене
        сохраняется — он отвечает на вопрос «когда реплику впервые разобрали».
        """
        now = _now()
        self._conn.execute(
            """
            INSERT INTO llm_analyses (
                analysis_id, project_id, replica_id, replica_index,
                source_text_hash, context_hash, dictionary_hash,
                model_tag, model_digest, prompt_version, schema_version,
                inference_hash, analysis_json, status, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, replica_index) DO UPDATE SET
                analysis_id = excluded.analysis_id,
                replica_id = excluded.replica_id,
                source_text_hash = excluded.source_text_hash,
                context_hash = excluded.context_hash,
                dictionary_hash = excluded.dictionary_hash,
                model_tag = excluded.model_tag,
                model_digest = excluded.model_digest,
                prompt_version = excluded.prompt_version,
                schema_version = excluded.schema_version,
                inference_hash = excluded.inference_hash,
                analysis_json = excluded.analysis_json,
                status = excluded.status,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                analysis_id,
                project_id,
                int(replica_id),
                int(replica_index),
                source_text_hash,
                context_hash,
                dictionary_hash,
                model_tag,
                model_digest,
                prompt_version,
                schema_version,
                inference_hash,
                analysis_json,
                status,
                error,
                now,
                now,
            ),
        )
        row = self.get(project_id, replica_index)
        assert row is not None  # только что записали
        return row

    def get(self, project_id: str, replica_index: int) -> dict | None:
        """Разбор по месту реплики в проекте (индекс), а не по её id в базе."""
        row = self._conn.execute(
            "SELECT * FROM llm_analyses WHERE project_id = ? AND replica_index = ?",
            (project_id, int(replica_index)),
        ).fetchone()
        return _row_to_analysis(row) if row else None

    def list_for_project(self, project_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM llm_analyses WHERE project_id = ? ORDER BY replica_index",
            (project_id,),
        ).fetchall()
        return [_row_to_analysis(row) for row in rows]

    def mark_stale(self, project_id: str, replica_indexes: list[int] | None = None) -> int:
        """Помечает разборы устаревшими, не удаляя их.

        Удалять нельзя: устаревший разбор — это история («что предлагала модель до
        правки текста»), и он дешевле пересчитывается повторно, чем теряется.
        """
        params: list[object] = [project_id]
        clause = ""
        if replica_indexes is not None:
            if not replica_indexes:
                return 0
            placeholders = ",".join("?" for _ in replica_indexes)
            clause = f" AND replica_index IN ({placeholders})"
            params.extend(int(index) for index in replica_indexes)
        cursor = self._conn.execute(
            "UPDATE llm_analyses SET status = 'STALE', updated_at = ? WHERE project_id = ?"
            + clause,
            [_now(), *params],
        )
        return int(cursor.rowcount or 0)

    def delete_for_replicas(self, project_id: str, replica_indexes: list[int]) -> int:
        """Удаляет разборы реплик, которых больше нет (пересборка диалога)."""
        if not replica_indexes:
            return 0
        placeholders = ",".join("?" for _ in replica_indexes)
        cursor = self._conn.execute(
            f"DELETE FROM llm_analyses WHERE project_id = ? AND replica_index IN ({placeholders})",
            (project_id, *[int(index) for index in replica_indexes]),
        )
        return int(cursor.rowcount or 0)

    def delete_for_project(self, project_id: str) -> int:
        """Удаляет все разборы проекта (он сам удалён).

        Явное удаление, а не `ON DELETE CASCADE`: ключом `project_id` бывает и
        синтетическое значение `text-<хеш>` (разбор сплошного текста без проекта),
        поэтому внешний ключ на `projects` здесь поставить нельзя. Разбор —
        производная данных, и вместе с проектом он не нужен.
        """
        cursor = self._conn.execute(
            "DELETE FROM llm_analyses WHERE project_id = ?", (project_id,)
        )
        return int(cursor.rowcount or 0)
