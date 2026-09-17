"""Реплики проекта: порядок, текст, спикер и назначенный голос."""

import sqlite3

from ... import config
from . import dumps, loads


def row_to_replica(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "index": row["idx"],
        "text": row["text"],
        # `source_text` — то же самое, что `text`: отдельной колонки нет, чтобы не
        # держать два источника правды об исходнике. Второе имя нужно интерфейсу и
        # API: рядом лежат производные стадии, и «исходный» рядом с «подготовленным»
        # читается однозначнее, чем `text` рядом с `final_text`.
        "source_text": row["text"],
        "speaker": row["speaker"],
        "voice_id": row["voice_id"],
        # Собственный голос реплики; None — наследует голос спикера. Отдельно от
        # voice_id, потому что «как звучит» и «кто это выбрал» — разные вопросы:
        # без этого поля сброс правки было бы некуда вернуть.
        "voice_override": row["voice_override"],
        "overrides": loads(row["overrides"], {}),
        "status": row["status"],
        "selected_take_id": row["selected_take_id"],
        # Стадии подготовки (см. миграцию 5). Пустые значения — реплика ещё не
        # анализировалась; `analysis_status` говорит, можно ли брать `final_text`.
        "normalized_text": row["normalized_text"],
        "yo_text": row["yo_text"],
        "dictionary_text": row["dictionary_text"],
        "accentized_text": row["accentized_text"],
        "final_text": row["final_text"],
        "analysis_status": row["analysis_status"],
        "analysis_version": row["analysis_version"],
        "analysis_error": row["analysis_error"],
        "dictionary_matches": loads(row["dictionary_matches"], []),
        "pronunciation_candidates": loads(row["pronunciation_candidates"], []),
        "supports_accents": bool(row["supports_accents"]),
        "auto_accent": bool(row["auto_accent"]),
        "effective_engine_params": loads(row["effective_engine_params"], {}),
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

    def list_by_status(self, *statuses: str) -> list[dict]:
        """Реплики в указанных статусах — по всем проектам.

        Нужно восстановлению после перезапуска: статусы «идёт» (`rendering`) не
        могут пережить смерть процесса, и их надо найти одним запросом, а не
        обходить проекты по одному.
        """
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        rows = self._conn.execute(
            f"SELECT * FROM replicas WHERE status IN ({placeholders}) ORDER BY project_id, idx",
            statuses,
        ).fetchall()
        return [row_to_replica(row) for row in rows]

    def sync_voices(self, project_id: str, speakers: dict[str, dict]) -> list[int]:
        """Разносит назначенный голос спикера по его репликам.

        Реплики с собственным голосом не трогаются: пользователь выбрал голос
        именно этой реплике, и смена голоса спикера не должна её переписывать.

        Возвращает индексы реплик, у которых голос **действительно** изменился: по
        ним инвалидируется подготовка (ударения зависят от движка голоса), а
        сохранение карточки спикера без правок голоса анализ не сбрасывает —
        иначе каждое нажатие «сохранить» требовало бы прогонять анализ заново.
        """
        changed: list[int] = []
        for key, data in speakers.items():
            if "voice_id" not in data:
                continue
            new_voice = str(data.get("voice_id") or "")
            rows = self._conn.execute(
                "SELECT idx, voice_id FROM replicas WHERE project_id = ? AND speaker = ?"
                " AND voice_override IS NULL",
                (project_id, key),
            ).fetchall()
            changed.extend(int(row["idx"]) for row in rows if row["voice_id"] != new_voice)
            if rows:
                self._conn.execute(
                    "UPDATE replicas SET voice_id = ? WHERE project_id = ? AND speaker = ?"
                    " AND voice_override IS NULL",
                    (new_voice, project_id, key),
                )
        return sorted(changed)

    def set_text(self, replica_id: int, text: str) -> bool:
        """Меняет исходный текст одной реплики (правка в карточке)."""
        cursor = self._conn.execute(
            "UPDATE replicas SET text = ? WHERE id = ?", (text, replica_id)
        )
        return cursor.rowcount > 0

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

    # -- подготовка текста (см. миграцию 5) ------------------------------------
    def save_analysis(self, replica_id: int, analysis: dict) -> bool:
        """Записывает стадии подготовки реплики одним обновлением.

        Одним, а не по шагам: половина записанных стадий — это реплика, у которой
        `final_text` от одной версии анализа, а кандидаты от другой. Рендер потом
        берёт `final_text`, и такая половинчатость означала бы, что в модель ушёл
        текст, которого пользователь не видел.
        """
        cursor = self._conn.execute(
            "UPDATE replicas SET normalized_text = ?, yo_text = ?, dictionary_text = ?,"
            " accentized_text = ?, final_text = ?, analysis_status = ?, analysis_version = ?,"
            " analysis_error = ?, dictionary_matches = ?, pronunciation_candidates = ?,"
            " supports_accents = ?, auto_accent = ?, effective_engine_params = ?"
            " WHERE id = ?",
            (
                analysis.get("normalized_text"),
                analysis.get("yo_text"),
                analysis.get("dictionary_text"),
                analysis.get("accentized_text"),
                analysis.get("final_text"),
                analysis.get("analysis_status", config.REPLICA_ANALYSIS_DONE),
                int(analysis.get("analysis_version") or 0),
                analysis.get("analysis_error"),
                dumps(analysis.get("dictionary_matches") or []),
                dumps(analysis.get("pronunciation_candidates") or []),
                1 if analysis.get("supports_accents", True) else 0,
                1 if analysis.get("auto_accent", True) else 0,
                dumps(analysis.get("effective_engine_params") or {}),
                replica_id,
            ),
        )
        return cursor.rowcount > 0

    def mark_analysis_pending(self, project_id: str, indexes: list[int] | None = None) -> int:
        """Помечает реплики как требующие повторной подготовки.

        Стадии при этом не стираются: пользователю полезно видеть, что было
        подготовлено в прошлый раз, а `analysis_status = pending` уже говорит, что
        брать этот `final_text` для рендера нельзя. Без списка индексов помечаются
        все реплики проекта — например, когда поменялся движок сразу у всех.
        """
        if indexes is None:
            cursor = self._conn.execute(
                "UPDATE replicas SET analysis_status = ? WHERE project_id = ?",
                (config.REPLICA_ANALYSIS_PENDING, project_id),
            )
            return cursor.rowcount
        if not indexes:
            return 0
        placeholders = ", ".join("?" for _ in indexes)
        cursor = self._conn.execute(
            f"UPDATE replicas SET analysis_status = ? WHERE project_id = ?"
            f" AND idx IN ({placeholders})",
            (config.REPLICA_ANALYSIS_PENDING, project_id, *indexes),
        )
        return cursor.rowcount

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
