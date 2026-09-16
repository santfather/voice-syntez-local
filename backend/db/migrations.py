"""Схема базы и её версии.

Схема создаётся сама при первом обращении: требовать от пользователя ручного
SQL нельзя — приложение локальное и ставится распаковкой архива. Миграции
пронумерованы, потому что база уже может существовать у того, кто обновился:
версия в `schema_version` говорит, что именно нужно досоздать.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Проект: то, что пользователь открывает и закрывает. Текст хранится целиком,
# чтобы диалог не приходилось склеивать обратно из реплик — иначе при повторном
# разборе потерялись бы пустые строки и разметка, которую пользователь написал.
_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    source_text     TEXT NOT NULL DEFAULT '',
    mode            TEXT NOT NULL DEFAULT 'dialogue',
    render_settings TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'draft',
    job_id          TEXT,
    last_error      TEXT
);

-- Спикер проекта: слот или имя из текста плюс назначенный ему голос.
-- Назначение живёт отдельно от реплик: повторный разбор текста не должен
-- сбрасывать голоса, которые пользователь уже выбрал.
CREATE TABLE IF NOT EXISTS speakers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    label      TEXT NOT NULL,
    voice_id   TEXT NOT NULL DEFAULT '',
    overrides  TEXT NOT NULL DEFAULT '{}',
    UNIQUE (project_id, key)
);

CREATE TABLE IF NOT EXISTS replicas (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    text             TEXT NOT NULL,
    speaker          TEXT NOT NULL,
    voice_id         TEXT NOT NULL DEFAULT '',
    overrides        TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    selected_take_id INTEGER,
    UNIQUE (project_id, idx)
);

-- Вариант (take) реплики: готовое аудио плюс то, чем оно получено. Отдельный
-- файл на реплику, а не вырезка из итогового трека: варианты сравнивают на слух,
-- и после замены одного из них остальные не должны меняться.
CREATE TABLE IF NOT EXISTS takes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    replica_id  INTEGER NOT NULL REFERENCES replicas(id) ON DELETE CASCADE,
    label       TEXT NOT NULL DEFAULT '',
    audio_path  TEXT NOT NULL,
    seed        INTEGER,
    engine      TEXT NOT NULL DEFAULT '',
    parameters  TEXT NOT NULL DEFAULT '{}',
    duration_sec REAL NOT NULL DEFAULT 0,
    qa          TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_replicas_project ON replicas(project_id, idx);
CREATE INDEX IF NOT EXISTS idx_speakers_project ON speakers(project_id);
CREATE INDEX IF NOT EXISTS idx_takes_replica ON takes(replica_id, id);
"""

# Собственный голос реплики поверх голоса спикера. NULL — реплика наследует
# голос спикера, и тогда смена голоса спикера её тоже меняет; непустое значение —
# пользователь выбрал голос именно этой реплике. Без отдельной колонки отличить
# «унаследовано» от «выбрано вручную» нельзя: у реплики лежит уже вычисленное
# значение, и сброс правки было бы некуда вернуть.
_MIGRATION_2 = """
ALTER TABLE replicas ADD COLUMN voice_override TEXT;
"""

# Словарь произношения. Хранится глобально, без привязки к проекту: правило
# «SQL читается как эскьюэль» пользователь задаёт один раз и ожидает его во всех
# диалогах сразу. Уникальность по источнику и режиму регистра — это то, по чему
# повторное добавление обновляет правило, а не плодит дубли (см. pronunciation.py).
_MIGRATION_3 = """
CREATE TABLE IF NOT EXISTS pronunciation_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT NOT NULL,
    target         TEXT NOT NULL,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    whole_word     INTEGER NOT NULL DEFAULT 1,
    enabled        INTEGER NOT NULL DEFAULT 1,
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (source, case_sensitive)
);

CREATE INDEX IF NOT EXISTS idx_pronunciation_enabled ON pronunciation_entries(enabled);
"""

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _MIGRATION_1),
    (2, _MIGRATION_2),
    (3, _MIGRATION_3),
)


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Доводит схему до последней версии. Возвращает итоговую версию."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT version FROM schema_version").fetchone()
    current = int(row[0]) if row else 0
    for version, script in MIGRATIONS:
        if version <= current:
            continue
        connection.executescript(script)
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        logger.info("Схема базы обновлена до версии %s", version)
        current = version
    return current
