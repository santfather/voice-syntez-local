"""Независимый наблюдатель за потреблением ресурсов процессом.

Точечные лимиты (потоки torch/OMP, один воркер очереди, таймаут инференса)
закрывают известные причины перерасхода, но не защищают от неизвестных:
утечка в новой версии библиотеки, неучтённый параллелизм, разросшийся список
кусков в памяти. Watchdog следит за памятью и прерывает текущую задачу, не убивая
процесс приложения.

Картину памяти собирает `memory_monitor` (свой RSS + RSS воркеров синтеза +
заполненность системы), а здесь остаются решения: ждать перед задачей
(`is_memory_critical`, см. `job_queue._wait_for_memory`) или прервать идущую
(`ResourceGuard`). Разделение важно: очередь не начинает новую работу в
критическом состоянии, а watchdog останавливает ту, из-за которой оно наступило.
"""

import asyncio
import logging
import os
import sys
import time
from collections.abc import Callable

import psutil

from . import config, memory_monitor

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


def mps_memory_snapshot() -> dict | None:
    """Best-effort память MPS: `None`, если torch/MPS недоступны.

    На Apple Silicon память моделей не попадает в RSS (см. MAX_RSS_MB), поэтому
    единственная честная метрика — счётчики самого torch. Значение читается
    только если torch уже импортирован: `/api/status` опрашивается часто, а
    первый импорт torch стоит секунды, и тянуть его ради метрики нельзя. В
    работающем приложении torch поднимается на старте (`configure_torch`), так
    что метрика есть; в тестах и на машинах без MPS её просто нет.

    Ошибка чтения не должна валить `/api/status`: счётчик может быть недоступен
    в принципе (старый torch, отсутствие Metal), и это не ошибка сервиса.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.backends.mps.is_available():
            return None
        return {
            "current_allocated_mb": round(
                torch.mps.current_allocated_memory() / (1024 * 1024), 1
            ),
            "driver_allocated_mb": round(
                torch.mps.driver_allocated_memory() / (1024 * 1024), 1
            ),
        }
    except Exception as exc:  # noqa: BLE001 — метрика не повод валить статус
        logger.warning("Не удалось прочитать память MPS: %s", exc)
        return None


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
    return memory_monitor.sample()["system_percent"] > config.SYSTEM_MEM_THRESHOLD_PERCENT


def memory_state() -> dict:
    """Состояние памяти целиком: для `/api/status` и объяснения пауз в очереди.

    Отдаёт и числа, и причину: пользователю нужно не «CRITICAL», а «память
    системы занята на 91 %», иначе непонятно, что закрывать.
    """
    return memory_monitor.get_state()


def is_memory_critical() -> bool:
    """Критическое состояние: новую тяжёлую работу не начинать.

    Учитывает не только системный порог, но и RSS процессов синтеза — модель
    живёт в воркере, и её вес в RSS бэкенда не виден (см. `memory_monitor`).
    """
    return memory_monitor.is_critical()


class ResourceGuard:
    """Прерывает текущую задачу при устойчивом превышении лимита памяти."""

    def __init__(self, on_breach: Callable[[], str | None]) -> None:
        self._on_breach = on_breach
        self._breach_streak = 0
        self._last_breach_at = 0.0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SEC)
            state = memory_monitor.get_state()
            if state["state"] == memory_monitor.STATE_CRITICAL:
                self._breach_streak += 1
                logger.warning(
                    "Память: %s (%d/%d)",
                    state["reason"], self._breach_streak, SUSTAINED_BREACH_CHECKS,
                )
            else:
                self._breach_streak = 0

            if self._breach_streak >= SUSTAINED_BREACH_CHECKS:
                # Не спамим при длительной утечке: не чаще одной попытки в минуту.
                now = time.monotonic()
                if now - self._last_breach_at > 60:
                    self._last_breach_at = now
                    logger.error(
                        "Устойчивое превышение памяти (%s) — прерываю текущую задачу",
                        state["reason"],
                    )
                    self._on_breach()
                self._breach_streak = 0
