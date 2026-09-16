#!/usr/bin/env python3
"""Живой замер памяти при загрузке и выгрузке движков (фаза 10).

Инструмент поднимает F5-TTS, затем XTTS v2, выгружает их по одному и после
каждого шага печатает метрики: RSS процесса, заполненность памяти системы и
счётчики MPS у torch. Ничего не мокает и не ходит по сети — это замер на
реальных весах, поэтому в pytest он не входит (см. `pytest.ini`).

Запускать вручную, когда модели не держит другой процесс:

    ./venv/bin/python tools/memory_check.py
    ./venv/bin/python tools/memory_check.py --xtts-engine xtts-banana

Дашборд на порту 8000 при этом лучше остановить: два процесса с поднятыми
моделями не влезут в память Apple Silicon и исказят замер. Если сервер оставить
запущенным, у него должна быть выгружена модель (вкладка «Модели» → «Выгрузить»)
и пуста очередь.

Про цифры. `torch.mps.current_allocated_memory()` — память, которую учитывает
сам torch, `driver_allocated_memory()` — то, что драйвер Metal выделил процессу,
а RSS из psutil память MPS на Apple Silicon не видит вовсе. Точное освобождение
MPS без такого замера утверждать нельзя; драйверный счётчик может снижаться не
сразу и не до нуля — это ожидаемо и печатается как есть.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Инструмент запускается из корня репозитория и как `tools/memory_check.py`:
# корень нужен в sys.path, чтобы импортировался пакет `backend`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, resource_guard
from backend.engines.base import ENGINE_F5, ENGINE_XTTS
from backend.engines.registry import get_engine

ROW_TEMPLATE = "{title:<30}{rss:>11}{system:>13}{torch_mps:>16}{driver:>18}"


def metrics() -> dict:
    """Текущие метрики: RSS, система и — best-effort — память MPS."""
    snapshot = resource_guard.snapshot()
    mps = resource_guard.mps_memory_snapshot() or {}
    return {
        "rss_mb": snapshot["rss_mb"],
        "system_mem_percent": snapshot["system_mem_percent"],
        "mps_current_mb": mps.get("current_allocated_mb"),
        "mps_driver_mb": mps.get("driver_allocated_mb"),
    }


def fmt(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.1f}"


def print_row(title: str, values: dict) -> None:
    print(ROW_TEMPLATE.format(
        title=title,
        rss=fmt(values["rss_mb"]),
        system=fmt(values["system_mem_percent"]),
        torch_mps=fmt(values["mps_current_mb"]),
        driver=fmt(values["mps_driver_mb"]),
    ))


def print_header() -> None:
    print(ROW_TEMPLATE.format(
        title="шаг", rss="RSS, МБ", system="система, %", torch_mps="MPS torch, МБ",
        driver="MPS драйвер, МБ",
    ))
    print("-" * 88)


def delta(after: dict, before: dict) -> str:
    """Изменение RSS и памяти драйвера MPS между двумя шагами (МБ)."""
    rss = f"RSS {after['rss_mb'] - before['rss_mb']:+.1f} МБ"
    driver_after, driver_before = after["mps_driver_mb"], before["mps_driver_mb"]
    if driver_after is None or driver_before is None:
        return rss + " · MPS: н/д"
    return rss + f" · MPS-драйвер {driver_after - driver_before:+.1f} МБ"


def load_engine(engine_id: str) -> tuple[object | None, float]:
    engine = get_engine(engine_id)
    started = time.monotonic()
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001 — инструмент печатает причину и продолжает
        print(f"  ! {engine.info.label} не поднялся: {exc}")
        return None, 0.0
    return engine, time.monotonic() - started


def unload_engine(engine, title: str) -> None:
    if engine is None:
        return
    try:
        engine.unload()
    except Exception as exc:  # noqa: BLE001 — выгрузка может отказать, это тоже результат
        print(f"  ! {title}: выгрузка не удалась ({exc})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер памяти при load/unload движков")
    parser.add_argument(
        "--xtts-engine",
        choices=(ENGINE_XTTS, "xtts-banana"),
        default=ENGINE_XTTS,
        help="какой XTTS поднимать вторым (по умолчанию базовая модель)",
    )
    args = parser.parse_args()

    print("Замер памяти: F5 → XTTS → выгрузка XTTS → выгрузка F5")
    print(f"Устройство: {config.pick_device()} · модель XTTS: {args.xtts_engine}")
    print()

    rows: list[tuple[str, dict]] = []
    print_header()
    start = metrics()
    print_row("старт (модели не подняты)", start)
    rows.append(("старт", start))

    f5, f5_sec = load_engine(ENGINE_F5)
    after_f5 = metrics()
    print_row(f"после загрузки F5 ({f5_sec:.0f} c)", after_f5)
    rows.append(("после загрузки F5", after_f5))

    xtts, xtts_sec = load_engine(args.xtts_engine)
    after_xtts = metrics()
    if xtts is not None:
        print_row(f"после загрузки {args.xtts_engine} ({xtts_sec:.0f} c)", after_xtts)
        rows.append((f"после загрузки {args.xtts_engine}", after_xtts))

    unload_engine(xtts, args.xtts_engine)
    after_xtts_unload = metrics()
    print_row(f"после выгрузки {args.xtts_engine}", after_xtts_unload)
    rows.append((f"после выгрузки {args.xtts_engine}", after_xtts_unload))

    unload_engine(f5, "F5")
    after_f5_unload = metrics()
    print_row("после выгрузки F5", after_f5_unload)
    rows.append(("после выгрузки F5", after_f5_unload))

    print()
    print("До/после (относительно предыдущего шага):")
    for (title, values), (prev_title, prev_values) in zip(rows[1:], rows[:-1]):
        print(f"  {title:<30} {delta(values, prev_values)}")
    print()
    print("Итог относительно старта:")
    print_row("  всего за прогон", {
        "rss_mb": after_f5_unload["rss_mb"],
        "system_mem_percent": after_f5_unload["system_mem_percent"],
        "mps_current_mb": after_f5_unload["mps_current_mb"],
        "mps_driver_mb": after_f5_unload["mps_driver_mb"],
    })
    print(f"  {delta(after_f5_unload, start)}")
    print()
    print(
        "Память MPS, возвращённая драйвером, может снижаться не сразу и не до нуля:\n"
        "`empty_cache()` освобождает кеш аллокатора, а ОС забирает страницы по своим\n"
        "правилам. Цифры выше — измеренный факт, а не обещание конкретного объёма."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
