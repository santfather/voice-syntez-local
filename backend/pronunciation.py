"""Хранилище словаря произношения: CRUD в SQLite и снимок активных правил.

Словарь **глобальный**, без привязки к проекту: смысл фазы в том, что правило
«SQL читается как эскьюэль» пользователь задаёт один раз и получает во всех
диалогах сразу. Правила лежат в `pronunciation_entries` (миграция 3) и переживают
перезапуск.

Чистая логика применения — в `text_normalization/pronunciation.py`; здесь только
хранение и кеш. Кеш нужен потому, что пайплайн спрашивает правила на каждом куске
длинного рендера, а читать ради этого базу каждый раз незачем. Снимок сбрасывается
при любой записи, поэтому правка видна следующей же реплике.

Путь к базе читается в момент вызова (`config.DB_PATH`), а не на импорте: тесты
уводят базу в `tmp_path`, и закешированные правила от прошлого файла не должны
протечь в следующий тест. Тот же приём, что в `db/connection.py`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from . import config
from .db.connection import transaction
from .db.repositories.pronunciation import PronunciationRepository
from .text_normalization.pronunciation import PronunciationRule

logger = logging.getLogger(__name__)


class PronunciationConflict(ValueError):
    """Правило с таким источником и режимом регистра уже есть — для API это 409."""


def _clean_source(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Источник правила не может быть пустым")
    return source.strip()


def _clean_target(target: object) -> str:
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Замена не может быть пустой — удаляйте правило, а не слово")
    return target.strip()


class PronunciationStore:
    """Правила словаря: чтение, запись и кеш активного снимка."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: tuple[Path, list[PronunciationRule]] | None = None

    # -- чтение ----------------------------------------------------------------
    def list_entries(self) -> list[dict]:
        """Все правила, включая выключенные: список в интерфейсе показывает их тоже."""
        with transaction() as connection:
            return PronunciationRepository(connection).list_all()

    def get(self, entry_id: int) -> dict | None:
        with transaction() as connection:
            return PronunciationRepository(connection).get(entry_id)

    def active_rules(self) -> list[PronunciationRule]:
        """Снимок включённых правил для пайплайна — с кешем по пути к базе."""
        path = config.DB_PATH
        with self._lock:
            if self._cache is not None and self._cache[0] == path:
                return self._cache[1]
        rules = self._read_active()
        with self._lock:
            self._cache = (path, rules)
        return rules

    def _read_active(self) -> list[PronunciationRule]:
        with transaction() as connection:
            rows = PronunciationRepository(connection).list_enabled()
        return [
            PronunciationRule(
                source=row["source"],
                target=row["target"],
                case_sensitive=row["case_sensitive"],
                whole_word=row["whole_word"],
            )
            for row in rows
        ]

    def invalidate(self) -> None:
        """Сбрасывает снимок: следующее чтение снова заглянет в базу."""
        with self._lock:
            self._cache = None

    # -- запись ----------------------------------------------------------------
    def create(
        self,
        source: str,
        target: str,
        case_sensitive: bool = False,
        whole_word: bool = True,
        enabled: bool = True,
        note: str = "",
    ) -> dict:
        """Создаёт правило; повторное добавление того же источника его обновляет.

        Выбран update, а не ошибка: пользователь, дважды отправивший форму, ждёт
        исправленную замену, а не 409. Уникальность — по (`source`, `case_sensitive`)
        (см. миграцию 3).
        """
        source = _clean_source(source)
        target = _clean_target(target)
        case_sensitive = bool(case_sensitive)
        with transaction() as connection:
            repository = PronunciationRepository(connection)
            existing = repository.find(source, case_sensitive)
            if existing is None:
                entry = repository.create(
                    source=source,
                    target=target,
                    case_sensitive=case_sensitive,
                    whole_word=bool(whole_word),
                    enabled=bool(enabled),
                    note=str(note or ""),
                )
                logger.info("Словарь: добавлено правило «%s» → «%s»", source, target)
            else:
                entry = repository.update(
                    existing["id"],
                    target=target,
                    whole_word=bool(whole_word),
                    enabled=bool(enabled),
                    note=str(note or ""),
                )
                logger.info("Словарь: правило «%s» обновлено → «%s»", source, target)
        self.invalidate()
        return entry  # type: ignore[return-value]

    def update(self, entry_id: int, **fields) -> dict:
        """Меняет поля правила. Неизвестный id — `KeyError`, конфликт — `PronunciationConflict`."""
        allowed = {"source", "target", "case_sensitive", "whole_word", "enabled", "note"}
        patch: dict[str, object] = {
            key: value for key, value in fields.items() if key in allowed and value is not None
        }
        if "source" in patch:
            patch["source"] = _clean_source(patch["source"])
        if "target" in patch:
            patch["target"] = _clean_target(patch["target"])
        if "note" in patch:
            patch["note"] = str(patch["note"] or "")
        for flag in ("case_sensitive", "whole_word", "enabled"):
            if flag in patch:
                patch[flag] = bool(patch[flag])

        try:
            with transaction() as connection:
                repository = PronunciationRepository(connection)
                if repository.get(entry_id) is None:
                    raise KeyError(entry_id)
                entry = repository.update(entry_id, **patch)
        except sqlite3.IntegrityError as exc:
            raise PronunciationConflict(
                "Правило с таким источником и режимом регистра уже есть — измените источник"
            ) from exc
        if entry is None:
            raise KeyError(entry_id)
        self.invalidate()
        return entry

    def delete(self, entry_id: int) -> bool:
        with transaction() as connection:
            deleted = PronunciationRepository(connection).delete(entry_id)
        if deleted:
            self.invalidate()
            logger.info("Словарь: правило %s удалено", entry_id)
        return deleted


_store = PronunciationStore()


def get_store() -> PronunciationStore:
    return _store


def active_rules() -> list[PronunciationRule]:
    """Снимок включённых правил — то, что пайплайн передаёт в `normalize`."""
    return _store.active_rules()
