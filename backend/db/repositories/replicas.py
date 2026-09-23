"""Реплики проекта: порядок, текст, спикер и назначенный голос."""

import sqlite3

from ... import config, emotions
from . import dumps, loads


def _optional_float(value: object) -> float | None:
    """Число из колонки, где NULL — «модель не сказала», а 0.0 — значение.

    Разница существенна: интенсивность `0.0` и отсутствие интенсивности — разные
    факты, и схлопывать их в одно значило бы выдумывать ответ модели.
    """
    return None if value is None else float(value)


def _number_or_none(value: object) -> float | None:
    """Число просодии для записи: вне 0..1 или не число — NULL («не сказала»)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def _pace_or_empty(value: object) -> str:
    """Темп из словаря §12; неизвестное значение — пустая строка, а не догадка."""
    text = str(value or "").strip().upper()
    return text if text in config.PROSODY_PACES else ""


def row_to_replica(row: sqlite3.Row) -> dict:
    """Собирает реплику из строки БД для API и хранилища.

    Действующие эмоция и просодия вычисляются здесь, а не читаются: правило
    `override → detected` одно на всё приложение, и колонка-дубликат однажды
    разошлась бы с ним.
    """
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
        # Эмоция и референс (миграция 9). `emotion_effective` вычисляется здесь, а
        # не хранится: правило `override → detected → NEUTRAL` одно на всё
        # приложение, и колонка-дубликат однажды разошлась бы с ним.
        "emotion_detected": row["emotion_detected"],
        "emotion_confidence": float(row["emotion_confidence"] or 0.0),
        "emotion_override": row["emotion_override"],
        "dialogue_act": row["dialogue_act"],
        "context_dependency": row["context_dependency"],
        "reference_profile_id": row["reference_profile_id"],
        "reference_emotion": row["reference_emotion"],
        "reference_fallback_used": bool(row["reference_fallback_used"]),
        "emotion_effective": emotions.emotion_effective(
            row["emotion_detected"], row["emotion_override"]
        ),
        # Просодия (миграция 10). `prosody_profile` — что **рекомендовала** модель,
        # `prosody_effective` — куда идти резолверу сейчас: ручной выбор сильнее
        # рекомендации, а без неё работает прежнее правило по эмоции. Вычисляется
        # здесь по той же причине, что и `emotion_effective`: правило одно на всё
        # приложение, и колонка-дубликат однажды разошлась бы с ним.
        "prosody_profile": row["prosody_profile"],
        "prosody_intensity": _optional_float(row["prosody_intensity"]),
        "prosody_pace": row["prosody_pace"],
        "prosody_confidence": _optional_float(row["prosody_confidence"]),
        "prosody_effective": emotions.prosody_effective(
            row["emotion_detected"], row["emotion_override"], row["prosody_profile"]
        ),
        "reference_profile_key": row["reference_profile_key"],
        "reference_fallback_reason": row["reference_fallback_reason"],
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

    def replace(self, project_id: str, replicas: list[dict]) -> list[str]:
        """Приводит список реплик проекта к новому результату разбора.

        Реплика, у которой текст и спикер не изменились, остаётся той же строкой
        в базе: у неё сохраняются варианты и выбранный take. Иначе правка одной
        фразы в исходном тексте обнуляла бы уже проделанную работу по всему
        диалогу, хотя изменилась одна строчка.

        Возвращает пути файлов вариантов удалённых реплик: `ON DELETE CASCADE`
        чистит только строки в базе, а куски на диске остались бы сиротами
        (F-S2). Файлы удаляет вызывающий — репозиторий о диске не знает.
        """
        existing = {
            int(row["idx"]): row
            for row in self._conn.execute(
                "SELECT * FROM replicas WHERE project_id = ?", (project_id,)
            )
        }
        orphan_paths: list[str] = []
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
                orphan_paths.extend(self._take_paths([int(old["id"])]))
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
        # Реплики, которых нет в новом разборе, уходят вместе с вариантами.
        removed = self._extra_replica_ids(project_id, keep)
        orphan_paths.extend(self._take_paths(removed))
        if keep:
            placeholders = ", ".join("?" for _ in keep)
            self._conn.execute(
                f"DELETE FROM replicas WHERE project_id = ? AND idx NOT IN ({placeholders})",
                (project_id, *keep),
            )
        else:
            self._conn.execute("DELETE FROM replicas WHERE project_id = ?", (project_id,))
        return orphan_paths

    def _extra_replica_ids(self, project_id: str, keep: list[int]) -> list[int]:
        """Идентификаторы реплик, которых нет в новом разборе (уходят целиком)."""
        if keep:
            placeholders = ", ".join("?" for _ in keep)
            rows = self._conn.execute(
                f"SELECT id FROM replicas WHERE project_id = ? AND idx NOT IN ({placeholders})",
                (project_id, *keep),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id FROM replicas WHERE project_id = ?", (project_id,)
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def _take_paths(self, replica_ids: list[int]) -> list[str]:
        """Пути файлов вариантов указанных реплик — до их каскадного удаления."""
        if not replica_ids:
            return []
        placeholders = ", ".join("?" for _ in replica_ids)
        rows = self._conn.execute(
            f"SELECT audio_path FROM takes WHERE replica_id IN ({placeholders})",
            tuple(replica_ids),
        ).fetchall()
        return [str(row["audio_path"]) for row in rows]

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

    def save_emotion(self, replica_id: int, emotion: dict) -> bool:
        """Пишет эмоцию реплики, просодию и фактический референс последнего синтеза.

        Отдельным обновлением от текста, а не вместе с `save_analysis`: анализ
        меняет произносимый текст, а эмоция — только выбор референса, и ручная
        смена эмоции не должна требовать пересчёта подготовки (§11).

        Просодия пишется здесь же, потому что приходит из того же ответа модели,
        что и эмоция: разнести их по двум обновлениям значило бы допустить
        состояние, где рекомендация профиля уже новая, а эмоция ещё старая.
        """
        cursor = self._conn.execute(
            "UPDATE replicas SET emotion_detected = ?, emotion_confidence = ?,"
            " emotion_override = ?, dialogue_act = ?, context_dependency = ?,"
            " prosody_profile = ?, prosody_intensity = ?, prosody_pace = ?,"
            " prosody_confidence = ?, reference_profile_id = ?, reference_emotion = ?,"
            " reference_fallback_used = ?, reference_profile_key = ?,"
            " reference_fallback_reason = ?"
            " WHERE id = ?",
            (
                emotions.normalize_emotion(emotion.get("emotion_detected"), default=""),
                float(emotion.get("emotion_confidence") or 0.0),
                emotions.normalize_emotion(emotion.get("emotion_override"), default=""),
                str(emotion.get("dialogue_act") or ""),
                str(emotion.get("context_dependency") or ""),
                emotions.normalize_profile_key(emotion.get("prosody_profile"), default=""),
                _number_or_none(emotion.get("prosody_intensity")),
                _pace_or_empty(emotion.get("prosody_pace")),
                _number_or_none(emotion.get("prosody_confidence")),
                str(emotion.get("reference_profile_id") or ""),
                emotions.normalize_emotion(emotion.get("reference_emotion"), default=""),
                1 if emotion.get("reference_fallback_used") else 0,
                emotions.normalize_profile_key(emotion.get("reference_profile_key"), default=""),
                str(emotion.get("reference_fallback_reason") or ""),
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

        Просодия — исключение: **рекомендация профиля стирается**. Она не просто
        «устарела для показа» — она выбирает файл референса, и оставленная молча
        привела бы к синтезу с интонацией, выведенной по тексту, которого уже нет
        (§36, §37). Ручной выбор (`emotion_override`) не трогается никогда: он
        принадлежит пользователю, а не анализу.
        """
        stale_prosody = (
            "prosody_profile = '', prosody_intensity = NULL, prosody_pace = '',"
            " prosody_confidence = NULL"
        )
        if indexes is None:
            cursor = self._conn.execute(
                f"UPDATE replicas SET analysis_status = ?, {stale_prosody} WHERE project_id = ?",
                (config.REPLICA_ANALYSIS_PENDING, project_id),
            )
            return cursor.rowcount
        if not indexes:
            return 0
        placeholders = ", ".join("?" for _ in indexes)
        cursor = self._conn.execute(
            f"UPDATE replicas SET analysis_status = ?, {stale_prosody} WHERE project_id = ?"
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
