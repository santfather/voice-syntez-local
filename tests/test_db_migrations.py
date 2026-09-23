"""Миграции базы: копия перед обновлением схемы и проверка целостности (F-D1).

База — единственное место, где лежат проекты пользователя, и портится она тихо:
повреждённый файл до первой выборки выглядит обычным. Поэтому проверка
целостности стоит до миграций, а перед обновлением схемы кладётся копия, к
которой можно вернуться, если миграция испортит базу.

Тесты работают на tmp_path и на своём файле: настоящая `data/voice_syntez.db`
не читается и не пишется.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend import config
from backend.db import migrations
from backend.db.connection import init_db
from backend.db.migrations import (
    BACKUP_SUFFIX,
    MIGRATIONS,
    DatabaseCorruptedError,
    apply_migrations,
    backup_before_migration,
    check_integrity,
)


def _old_database(path, version: int = 6) -> None:
    """База «прошлой версии» с одной строкой проекта — как у обновившегося пользователя."""
    connection = sqlite3.connect(path)
    try:
        for number, script in MIGRATIONS:
            if number > version:
                break
            connection.executescript(script)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        connection.execute(
            "INSERT INTO projects"
            " (id, name, source_text, mode, render_settings, status, created_at, updated_at)"
            " VALUES ('p1', 'Старый проект', 'текст', 'dialogue', '{}', 'draft',"
            " '2026-01-01', '2026-01-01')"
        )
        connection.commit()
    finally:
        connection.close()


def test_migration_backs_up_database_before_upgrade(tmp_path):
    """Копия — состояние до миграции: в ней старая версия схемы и данные на месте."""
    path = tmp_path / "voice_syntez.db"
    _old_database(path)

    connection = sqlite3.connect(path)
    try:
        assert apply_migrations(connection, path) == MIGRATIONS[-1][0]
    finally:
        connection.close()

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    assert backup.exists()
    copy = sqlite3.connect(backup)
    try:
        assert copy.execute("SELECT version FROM schema_version").fetchone()[0] == 6
        # Таблица поздней миграции в копии ещё не появилась — это снимок «до».
        later = copy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'llm_analyses'"
        ).fetchone()
        assert later is None
        assert copy.execute("SELECT name FROM projects WHERE id = 'p1'").fetchone()[0] == (
            "Старый проект"
        )
    finally:
        copy.close()


def test_first_run_and_current_schema_leave_no_backup(tmp_path):
    """Копировать нечего: на первом запуске файла нет, при текущей схеме — нечего терять."""
    path = tmp_path / "voice_syntez.db"
    backup = path.with_name(path.name + BACKUP_SUFFIX)

    connection = sqlite3.connect(path)
    try:
        apply_migrations(connection, path)  # создаёт схему с нуля
        assert not backup.exists()
        apply_migrations(connection, path)  # схема уже последней версии
        assert not backup.exists()
    finally:
        connection.close()


def test_backup_is_skipped_without_pending_migrations(tmp_path):
    """Прямой вызов без незакрытых миграций копию не делает — иначе она плодилась бы зря."""
    path = tmp_path / "voice_syntez.db"
    connection = sqlite3.connect(path)
    try:
        apply_migrations(connection, path)
        assert backup_before_migration(connection, path) is None
    finally:
        connection.close()

    # Файла ещё нет — копировать нечего.
    missing = tmp_path / "нет-такого.db"
    memory = sqlite3.connect(":memory:")
    try:
        assert backup_before_migration(memory, missing) is None
    finally:
        memory.close()


def test_corrupted_database_is_rejected_before_migration(tmp_path):
    """Порченый файл не проходит проверку молча: миграция не применяется."""
    path = tmp_path / "voice_syntez.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload TEXT)")
        connection.executemany(
            "INSERT INTO t (payload) VALUES (?)", [("строка " * 40,)] * 100
        )
        connection.commit()
    finally:
        connection.close()
    assert path.stat().st_size > 8192

    # Обрезаем файл: в заголовке записано больше страниц, чем осталось, — SQLite
    # сообщает о повреждении, а не читает «что получилось».
    with open(path, "r+b") as handle:
        handle.truncate(4096 + 16)

    connection = sqlite3.connect(path)
    try:
        with pytest.raises(DatabaseCorruptedError, match="повреждена"):
            check_integrity(connection)
        with pytest.raises(DatabaseCorruptedError, match="повреждена"):
            apply_migrations(connection, path)
    finally:
        connection.close()


def test_init_db_backs_up_before_migrating_existing_database(workspace):
    """Старт приложения передаёт путь: обновляемая база получает копию автоматически."""
    path = config.DB_PATH
    _old_database(path)

    init_db()

    assert path.with_name(path.name + BACKUP_SUFFIX).exists()


def test_integrity_check_accepts_healthy_database(tmp_path):
    """Исправная база проходит проверку — иначе миграции встали бы всегда."""
    path = tmp_path / "voice_syntez.db"
    connection = sqlite3.connect(path)
    try:
        apply_migrations(connection, path)
        assert migrations.check_integrity(connection) is None
    finally:
        connection.close()
