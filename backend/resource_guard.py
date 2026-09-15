"""Независимый наблюдатель за потреблением ресурсов процессом.

Точечные лимиты (потоки torch/OMP, один воркер очереди, таймаут инференса)
закрывают известные причины перерасхода, но не защищают от неизвестных:
утечка в новой версии библиотеки, неучтённый параллелизм, разросшийся список
кусков в памяти. Watchdog следит за RSS и прерывает текущую задачу, не убивая
процесс приложения.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable

import psutil

from . import config

logger = logging.getLogger("tts.resource_guard")

CHECK_INTERVAL_SEC = float(os.environ.get("TTS_RSS_CHECK_INTERVAL_SEC", "5"))
# Не дёргаться на разовый всплеск: RSS должен превышать лимит несколько проверок подряд.
SUSTAINED_BREACH_CHECKS = 3

_process: psutil.Process | None = None


def _proc() -> psutil.Process:
    global _process
    if _process is None:
        _process = psutil.Process(os.getpid())
    return _process


def snapshot() -> dict:
    """Текущие RSS и загрузка CPU процесса (для /api/status).

    `cpu_percent(interval=None)` считает загрузку с момента предыдущего вызова,
    поэтому первый вызов после старта всегда возвращает 0.0 — это нормально.
    """
    process = _proc()
    return {
        "rss_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        "cpu_percent": round(process.cpu_percent(interval=None), 1),
        "system_mem_percent": round(psutil.virtual_memory().percent, 1),
    }


def check_system_memory_pressure() -> bool:
    """True, если память занята не только этим процессом, но и всей системой.

    Второй, независимый от `MAX_RSS_MB` слой защиты: свой лимит не видит, когда
    машину уже загрузили редактор с индексацией и браузер, а питон-процесс ещё
    в пределах своего потолка. Обратная слепота тоже есть — системный порог не
    различает, кто именно занял память, поэтому обе проверки нужны вместе.

    Сама по себе не логирует: вызывается в цикле ожидания, и предупреждение
    на каждой итерации превратилось бы в спам. Решение о сообщении принимает
    вызывающий (см. `job_queue._wait_for_memory`).
    """
    return psutil.virtual_memory().percent > config.SYSTEM_MEM_THRESHOLD_PERCENT


class ResourceGuard:
    """Прерывает текущую задачу при устойчивом превышении лимита памяти."""

    def __init__(self, on_breach: Callable[[], str | None]) -> None:
        self._on_breach = on_breach
        self._breach_streak = 0
        self._last_breach_at = 0.0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SEC)
            rss_mb = _proc().memory_info().rss / (1024 * 1024)
            if rss_mb > config.MAX_RSS_MB:
                self._breach_streak += 1
                logger.warning(
                    "RSS %.0f МБ > лимита %d МБ (%d/%d)",
                    rss_mb, config.MAX_RSS_MB, self._breach_streak, SUSTAINED_BREACH_CHECKS,
                )
            else:
                self._breach_streak = 0

            if self._breach_streak >= SUSTAINED_BREACH_CHECKS:
                # Не спамим при длительной утечке: не чаще одной попытки в минуту.
                now = time.monotonic()
                if now - self._last_breach_at > 60:
                    self._last_breach_at = now
                    logger.error(
                        "Устойчивое превышение памяти (%.0f МБ) — прерываю текущую задачу", rss_mb
                    )
                    self._on_breach()
                self._breach_streak = 0
