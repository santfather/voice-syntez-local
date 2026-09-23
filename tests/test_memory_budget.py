"""Проверки проактивного бюджета памяти перед загрузкой движка.

Проверяется сама функция (`check_engine_load_budget`), а не вход, которым
пользуются роут и пайплайн: `guard_engine_load` выключен фикстурой
`no_memory_pressure` нарочно — он читает свободную память машины, и тест на нём
зависел бы от того, что ещё запущено рядом.
"""

import psutil
import pytest

from backend import config
from backend.memory_guard import MemoryBudgetExceeded, check_engine_load_budget


def test_rejects_third_engine(monkeypatch):
    """Третий TTS-движок при лимите 2 — отказ, и до любой проверки памяти."""
    monkeypatch.setattr(config, "MAX_CONCURRENT_TTS_ENGINES", 2)
    with pytest.raises(MemoryBudgetExceeded, match="лимит — 2"):
        check_engine_load_budget(
            "xtts-banana",
            current_loaded=["f5", "xtts"],
            engine_size_mb=5220,
        )


def test_rejects_insufficient_memory(monkeypatch):
    """Недостаток свободной памяти — отказ с числами, а не общим текстом."""

    class FakeVM:
        available = 300 * 1024 * 1024  # 300 МБ
        percent = 92.0

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())
    with pytest.raises(MemoryBudgetExceeded, match="Недостаточно свободной"):
        check_engine_load_budget(
            "xtts-banana",
            current_loaded=["f5"],
            engine_size_mb=5220,
        )


def test_allows_when_budget_ok(monkeypatch):
    """Достаточно памяти и движков меньше лимита — разрешено."""

    class FakeVM:
        available = 8 * 1024 * 1024 * 1024  # 8 ГБ
        percent = 50.0

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())
    # Не должно поднять исключение.
    check_engine_load_budget(
        "xtts",
        current_loaded=["f5"],
        engine_size_mb=1900,
    )
