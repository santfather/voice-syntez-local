"""Состояние памяти: NORMAL / WARNING / CRITICAL — для очереди, watchdog'а и health.

Три источника, потому что ни один не видит картину целиком:

* **RSS процесса бэкенда** — буферы пайплайна, numpy-массивы, распознавание. На
  Apple Silicon вес моделей в него не попадает (Metal считает отдельно), поэтому
  этот лимит ловит утечки, но не модели;
* **RSS процессов-воркеров** — именно там живут модели (creash_report), и без
  этого слагаемого watchdog не увидел бы главного потребителя памяти; в
  классификацию идёт самый тяжёлый процесс, а не сумма по всем (см. `classify`);
* **заполненность памяти системы** — видит чужие процессы (браузер, редактор с
  индексацией) и не зависит от того, как учтена память в Python.

Сводка нужна разным потребителям с разными решениями: очередь не начинает новую
задачу в CRITICAL, watchdog прерывает уже идущую, `/api/status` показывает
состояние пользователю. Поэтому здесь чистые числа и одна классификация, а
решения принимают вызывающие.

Сбор чисел инъектируется (`sampler`): иначе проверять классификацию пришлось бы
на живой памяти машины, то есть тест зависел бы от того, что ещё запущено рядом.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from . import config

logger = logging.getLogger("tts.memory")

STATE_NORMAL = "normal"
STATE_WARNING = "warning"
STATE_CRITICAL = "critical"

# WARNING — это «ещё можно, но близко к пределу». Порог системной памяти минус
# пять процентов и 80 % от своего потолка: значения подобраны так, чтобы
# предупреждение появлялось до отказа, а не вместе с ним.
WARNING_SYSTEM_MARGIN = 5.0
WARNING_RSS_RATIO = 0.8

Sampler = Callable[[], dict]


def _worker_pids() -> list[int]:
    """PID живых воркеров синтеза. Супервизор может быть не создан — это норма."""
    try:
        from .engines.supervisor import get_supervisor

        return get_supervisor().alive_pids()
    except Exception as exc:  # noqa: BLE001 — метрика не повод падать
        logger.debug("Не удалось получить список воркеров: %s", exc)
        return []


def _default_sampler() -> dict:
    """Сырые числа: свой RSS, RSS воркеров и заполненность памяти системы."""
    import psutil

    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    system_percent = float(psutil.virtual_memory().percent)
    workers: list[dict] = []
    for pid in _worker_pids():
        try:
            workers.append(
                {
                    "pid": pid,
                    "rss_mb": round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 1),
                }
            )
        except Exception as exc:  # noqa: BLE001 — процесс мог умереть между вызовами
            logger.debug("Не удалось прочитать память воркера %s: %s", pid, exc)
    return {
        "rss_mb": round(rss_mb, 1),
        "system_percent": round(system_percent, 1),
        "workers": workers,
        "workers_rss_mb": round(sum(item["rss_mb"] for item in workers), 1),
        # Самый тяжёлый воркер — по нему идёт классификация: сумма по всем
        # процессам делила бы единый потолок между движками, и два движка
        # упирались бы в лимит, ни один из которых его не превысил (см. `classify`).
        "worker_peak_rss_mb": round(max((item["rss_mb"] for item in workers), default=0.0), 1),
    }


def sample(sampler: Sampler | None = None) -> dict:
    """Собирает числа. Ошибка сбора не должна ломать задачу: возвращаем нули.

    Нули в классификации означают NORMAL — то есть «нет данных, не мешаем
    работать». Обратный выбор («нет данных — значит критично») останавливал бы
    синтез из-за сломанной метрики.
    """
    collect = sampler or _default_sampler
    try:
        data = collect()
    except Exception as exc:  # noqa: BLE001 — метрика вторична
        logger.warning("Не удалось собрать состояние памяти: %s", exc)
        data = {}
    return {
        "rss_mb": float(data.get("rss_mb") or 0.0),
        "workers_rss_mb": float(data.get("workers_rss_mb") or 0.0),
        "worker_peak_rss_mb": float(data.get("worker_peak_rss_mb") or 0.0),
        "system_percent": float(data.get("system_percent") or 0.0),
        "workers": list(data.get("workers") or []),
    }


def classify(
    *,
    rss_mb: float,
    worker_peak_rss_mb: float,
    system_percent: float,
    max_rss_mb: float | None = None,
    worker_max_rss_mb: float | None = None,
    system_threshold_percent: float | None = None,
) -> tuple[str, str]:
    """Чистая классификация: состояние и причина. Без psutil, времени и глобального состояния.

    Порядок проверок — от самого общего к частному: системная память важнее
    своего RSS, потому что при перегруженной машине новая задача навредит всем,
    а не только нам.

    Потолок воркеров сравнивается с **самым тяжёлым** процессом синтеза, а не с
    суммой по всем: `WORKER_MAX_RSS_MB` — это потолок одного процесса, и на
    сумме два движка делили бы один бюджет. Хуже, что движки не равны по
    аппетиту: CPU-движок (Kokoro-ru) держит модель и произносительный словарь в
    RSS целиком, а MPS-движок почти не виден в RSS, поэтому общий бюджет
    срабатывал бы на первом же честном прогоне CPU-движка.
    """
    max_rss = config.MAX_RSS_MB if max_rss_mb is None else max_rss_mb
    worker_max = config.WORKER_MAX_RSS_MB if worker_max_rss_mb is None else worker_max_rss_mb
    threshold = (
        config.SYSTEM_MEM_THRESHOLD_PERCENT
        if system_threshold_percent is None
        else system_threshold_percent
    )

    if system_percent > threshold:
        return (
            STATE_CRITICAL,
            f"память системы занята на {system_percent:.0f}% (порог {threshold:.0f}%)",
        )
    if rss_mb > max_rss:
        return STATE_CRITICAL, f"бэкенд занимает {rss_mb:.0f} МБ (лимит {max_rss:.0f} МБ)"
    if worker_peak_rss_mb > worker_max:
        return (
            STATE_CRITICAL,
            f"процесс синтеза занимает {worker_peak_rss_mb:.0f} МБ "
            f"(лимит {worker_max:.0f} МБ)",
        )

    if system_percent > threshold - WARNING_SYSTEM_MARGIN:
        return (
            STATE_WARNING,
            f"память системы занята на {system_percent:.0f}% — близко к порогу {threshold:.0f}%",
        )
    if rss_mb > max_rss * WARNING_RSS_RATIO:
        return STATE_WARNING, f"бэкенд занимает {rss_mb:.0f} МБ — близко к лимиту {max_rss:.0f} МБ"
    if worker_peak_rss_mb > worker_max * WARNING_RSS_RATIO:
        return (
            STATE_WARNING,
            f"процесс синтеза занимает {worker_peak_rss_mb:.0f} МБ — близко к лимиту "
            + f"{worker_max:.0f} МБ",
        )
    return STATE_NORMAL, "памяти достаточно"


def get_state(sampler: Sampler | None = None) -> dict:
    """Состояние памяти целиком: для очереди (решение), health (показ) и логов."""
    data = sample(sampler)
    state, reason = classify(
        rss_mb=data["rss_mb"],
        worker_peak_rss_mb=data["worker_peak_rss_mb"],
        system_percent=data["system_percent"],
    )
    return {
        "state": state,
        "reason": reason,
        **data,
        "thresholds": {
            "max_rss_mb": config.MAX_RSS_MB,
            "worker_max_rss_mb": config.WORKER_MAX_RSS_MB,
            "system_percent": config.SYSTEM_MEM_THRESHOLD_PERCENT,
        },
    }


def is_critical(sampler: Sampler | None = None) -> bool:
    """Критическое состояние — «новую тяжёлую работу не начинать»."""
    return get_state(sampler)["state"] == STATE_CRITICAL
