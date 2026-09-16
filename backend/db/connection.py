"""Подключение к SQLite: соединение на операцию, схема создаётся сама.

Своё соединение на каждую операцию, а не одно общее на процесс: `sqlite3.Connection`
не предназначен для работы из разных потоков, а здесь в базу пишут и обработчики
FastAPI, и воркер очереди (он уводит синтез в отдельный поток). Открыть соединение
в SQLite дешевле, чем согласовывать блокировки между потоками.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from .. import config
from .migrations import apply_migrations

logger = logging.getLogger(__name__)

# Путь, для которого схема уже проверена. Кеш нужен, чтобы не гонять миграции на
# каждом соединении; путь в ключе — потому что тесты уводят базу в tmp_path, и
# «уже инициализировано» не должно означать «инициализировано для другого файла».
_init_lock = threading.Lock()
_initialized_path: Path | None = None


def db_path() -> Path:
    return config.DB_PATH


def connect() -> sqlite3.Connection:
    """Новое соединение с включёнными внешними ключами и WAL."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    # Без этого ON DELETE CASCADE в схеме не работает: sqlite по умолчанию
    # внешние ключи игнорирует, и удаление проекта оставляло бы его реплики.
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_db() -> None:
    """Создаёт базу и схему, если их ещё нет. Идемпотентно."""
    global _initialized_path
    path = db_path()
    with _init_lock:
        if _initialized_path == path:
            return
        connection = connect()
        try:
            apply_migrations(connection)
            connection.commit()
        finally:
            connection.close()
        _initialized_path = path
        logger.info("База проектов готова: %s", path)


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Транзакция: коммит на выходе, откат при исключении."""
    init_db()
    connection = connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()
