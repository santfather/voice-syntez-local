"""Журнал приложения: файл с ротацией и логи uvicorn в нём (F-F1).

До этой фазы в файл всё попадало перенаправлением вывода лаунчера: файл рос без
предела, а приложение про свой журнал ничего не знало. Здесь проверяется, что
журнал ведёт приложение — обработчиком с ротацией, что настройка повторного старта
ничего не копит и что сообщения uvicorn доходят до того же файла.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from backend import config, main


def _file_handler() -> RotatingFileHandler:
    """Наш файловый обработчик в корневом логгере — его и проверяем."""
    handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert handlers, "файловый обработчик журнала не подключён"
    return handlers[-1]


@pytest.fixture(autouse=True)
def _no_handler_leaks():
    """Обработчик и уровень живут до конца теста: иначе они остались бы в процессе.

    Корневой логгер — общий на процесс, поэтому утечка сказалась бы на других
    тестах: строка из следующего теста ушла бы в файл, которого уже нет.
    """
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, main.FILE_HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()


def test_journal_is_written_to_file_and_rotates(workspace, monkeypatch):
    """Файл ведёт приложение, и при отрастании старый лог становится копией."""
    monkeypatch.setattr(config, "LOG_MAX_BYTES", 512)
    monkeypatch.setattr(config, "LOG_BACKUP_COUNT", 2)

    main._configure_file_logging()
    handler = _file_handler()
    assert Path(handler.baseFilename) == config.LOG_PATH
    assert (handler.maxBytes, handler.backupCount) == (512, 2)

    line = "строка журнала " + "x" * 80
    for _ in range(40):
        logging.getLogger("tts.test.rotation").info(line)

    assert "строка журнала" in config.LOG_PATH.read_text(encoding="utf-8")
    rotated = config.LOG_PATH.with_name(config.LOG_PATH.name + ".1")
    assert rotated.exists(), "старый журнал должен сохраниться копией, а не пропасть"


def test_repeated_setup_keeps_one_handler(workspace):
    """Повторный старт в одном процессе (тесты, перезапуск) не копит обработчики."""
    main._configure_file_logging()
    main._configure_file_logging()

    logging.getLogger("tts.test.repeat").info("одна строка журнала")
    text = config.LOG_PATH.read_text(encoding="utf-8")
    assert text.count("одна строка журнала") == 1


def test_uvicorn_messages_reach_the_journal(workspace):
    """Сообщения uvicorn (старт, отказы, трейсбеки) — в том же файле."""
    main._configure_file_logging()
    main._adopt_uvicorn_loggers()

    logging.getLogger("uvicorn.error").error("проверка журнала uvicorn")
    assert "проверка журнала uvicorn" in config.LOG_PATH.read_text(encoding="utf-8")
    # Иначе обработчики uvicorn остались бы отдельными и файла не видели.
    assert logging.getLogger("uvicorn.error").propagate is True
