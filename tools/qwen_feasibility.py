#!/usr/bin/env python3
"""Проба выполнимости Qwen3-TTS на этой машине: устройство, dtype, память, скорость.

Модель Qwen3-TTS документирует CUDA, а проект живёт на Apple Silicon. Вопросы,
на которые нельзя ответить чтением кода — поднимается ли 1.7B на MPS, хватает ли
памяти, во что обходится откат на CPU и есть ли вообще путь через MLX, — решает
эта проба: она поднимает настоящие веса теми же двумя способами, которыми их
поднимет дашборд (`auto` и `cpu`), и печатает измеренные числа.

Ничего не мокает: веса, сеть (только если модель ещё не скачана) и минуты времени.
Поэтому в pytest не входит (см. `pytest.ini`) и запускается вручную:

    ./venv/bin/python tools/qwen_feasibility.py --ref /путь/к/речи.wav
    ./venv/bin/python tools/qwen_feasibility.py --plan          # план без моделей
    ./venv/bin/python tools/qwen_feasibility.py --ref ... --json

Референс — обычная запись голоса 3–15 с: столько просит карточка модели, и это
ровно то, что проект умеет отдавать как `ReferenceProfile`. Расшифровку можно
передать через `--ref-text`; без неё проба синтезирует в режиме «эксперимент»
(только эмбеддинг спикера), а с ней — полным клонированием.

MLX проба не «поддерживает», а проверяет: отдельной реализации Qwen3-TTS на MLX
в проекте нет, и инструмент честно печатает, установлен ли рантайм, вместо того
чтобы обещать ускорение, которого никто не измерял.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Инструмент запускается из корня репозитория и как `tools/qwen_feasibility.py`:
# корень нужен в sys.path, чтобы импортировался пакет `backend`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, model_manager, resource_guard
from backend.engines.base import (
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_MODE_KEY,
    ENGINE_MODE_QUALITY,
    ENGINE_QWEN,
    engine_info,
)
from backend.engines.qwen_engine import QwenTTSEngine

PROBE_TEXT = "Проверка связи: раз, два, три. Модель должна прочитать эту фразу целиком."

# Два способа поднять модель, которыми пользуется дашборд: `auto` — то, что
# выбирается само (на Apple Silicon это MPS), `cpu` — аварийный тормоз
# `TTS_QWEN_FORCE_CPU=1`. Третьей комбинации в проекте нет, поэтому и не пробуем:
# измерять то, что нельзя включить, — значит выдавать числа без адреса.
PROBES: tuple[tuple[str, str], ...] = (("auto", "auto"), ("cpu", "cpu"))

ROW_TEMPLATE = "{title:<28}{device:>6}{load:>10}{synth:>10}{audio:>9}{rtf:>8}{rss:>12}{mps:>13}"


def _fmt(value: float | None, digits: int = 1) -> str:
    return "н/д" if value is None else f"{value:.{digits}f}"


def prerequisites() -> list[str]:
    """Что мешает пробе. Пустой список — можно запускать."""
    problems: list[str] = []
    if importlib.util.find_spec("qwen_tts") is None:
        problems.append(
            "пакет `qwen-tts` не установлен: "
            "./venv/bin/pip install --no-deps -r requirements-qwen.txt"
        )
    missing = [
        spec.id
        for spec in model_manager.model_specs()
        if spec.engine_id == ENGINE_QWEN and model_manager.missing_files(spec)
    ]
    if missing:
        problems.append(
            "не скачаны модели: "
            + ", ".join(missing)
            + " (вкладка «Модели» в дашборде или `huggingface-cli download`)"
        )
    return problems


def environment() -> dict:
    """Окружение и то, что видно без поднятия модели: версии, MPS, MLX, MPS-fallback."""
    import torch
    import transformers

    pytorch_utils = transformers.pytorch_utils
    info = engine_info(ENGINE_QWEN)
    return {
        "engine": info.id,
        "label": info.label,
        "languages": list(info.languages),
        "fallback_engine": info.fallback_engine or "(нет)",
        "modes": [mode.id for mode in info.modes],
        "torch": torch.__version__,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "transformers": transformers.__version__,
        # Шим нужен только потому, что пятая версия transformers вычистила
        # функцию, которую ждёт qwen-tts. Отсутствие функции — не ошибка, но её
        # наличие означает, что без шима импорт упал бы.
        "shim_isin_mps_friendly_needed": not hasattr(pytorch_utils, "isin_mps_friendly"),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "qwen_device_setting": config.QWEN_DEVICE,
        "force_cpu_env": config.qwen_force_cpu(),
        "attn_implementation": config.QWEN_ATTN_IMPLEMENTATION,
        "dtype_on_mps": "bfloat16",
        "dtype_on_cpu": "float32",
        "mlx_installed": importlib.util.find_spec("mlx") is not None,
        "mlx_implementation": "нет: отдельного MLX-пути для Qwen3-TTS в проекте не написано",
    }


def probe(device_setting: str, reference: str, ref_text: str, mode: str) -> dict:
    """Один прогон: поднять модель выбранным способом, синтезировать, выгрузить.

    Любая ошибка возвращается как результат прогона, а не поднимается наружу:
    «на MPS не поднялось» — это и есть ответ пробы, ради которого она запущена.
    """
    config.QWEN_DEVICE = device_setting
    engine = QwenTTSEngine()
    result: dict = {
        "requested": device_setting,
        "device": None,
        "dtype": None,
        "load_sec": None,
        "synth_sec": None,
        "audio_sec": None,
        "rtf": None,
        "peak_rss_mb": None,
        "mps_driver_mb": None,
        "error": "",
    }
    started = time.monotonic()
    try:
        engine.load()
        result["load_sec"] = time.monotonic() - started
        result["device"] = engine.device
        result["dtype"] = engine.dtype
        synth_started = time.monotonic()
        waveform, sample_rate = engine.synthesize(
            PROBE_TEXT, reference, ref_text, 1.0, **{ENGINE_MODE_KEY: mode}
        )
        result["synth_sec"] = time.monotonic() - synth_started
        result["audio_sec"] = waveform.size / float(sample_rate or 1)
        if result["audio_sec"]:
            # RTF (real-time factor): сколько секунд счёта стоит секунда звука.
            # Меньше единицы — быстрее реального времени; это и есть мера
            # пригодности для диалога.
            result["rtf"] = result["synth_sec"] / result["audio_sec"]
        snapshot = resource_guard.snapshot()
        mps = resource_guard.mps_memory_snapshot() or {}
        result["peak_rss_mb"] = snapshot["rss_mb"]
        result["mps_driver_mb"] = mps.get("driver_allocated_mb")
    except Exception as exc:  # noqa: BLE001 — провал пробы это её результат
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            engine.unload()
        except Exception as exc:  # noqa: BLE001 — выгрузка тоже часть пробы
            result["error"] = result["error"] or f"выгрузка: {exc}"
        del engine
    return result


def print_plan() -> None:
    print("Что будет проверено (модели не поднимаются):")
    print()
    print("  1. окружение: пакет qwen-tts, веса, torch/transformers, MPS, MLX")
    for device_setting, title in PROBES:
        print(f"  2. прогон «{title}»: load → синтез фразы {len(PROBE_TEXT)} знаков → unload")
        print(f"     устройство: config.QWEN_DEVICE={device_setting!r}, dtype по устройству")
    print()
    print("Что получится на выходе: время загрузки, время синтеза, длительность звука,")
    print("RTF, RSS и память драйвера MPS на каждом шаге — плюс вердикт по пригодности.")
    print()
    print("Нужны: скачанные веса (~5 ГБ), установленный `qwen-tts`, запись голоса 3–15 с.")
    print("Дашборд перед прогоном лучше остановить: двум процессам с моделью не хватит памяти.")


def print_environment(env: dict) -> None:
    print("Окружение:")
    print(f"  движок: {env['engine']} — {env['label']}")
    print(f"  языки: {', '.join(env['languages'])} · откат: {env['fallback_engine']} · режимы: {', '.join(env['modes'])}")
    print(f"  torch {env['torch']} · transformers {env['transformers']}")
    print(f"  MPS доступен: {'да' if env['torch_mps_available'] else 'нет'}")
    print(
        "  шим isin_mps_friendly: "
        + ("нужен (функции нет в transformers)" if env["shim_isin_mps_friendly_needed"] else "не нужен")
    )
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK={env['mps_fallback_env'] or '(не задан)'}")
    print(f"  устройство: {env['qwen_device_setting']} · force-cpu: {'да' if env['force_cpu_env'] else 'нет'}")
    print(f"  dtype: MPS → {env['dtype_on_mps']}, CPU → {env['dtype_on_cpu']} · attention: {env['attn_implementation']}")
    print(
        "  MLX: "
        + ("рантайм есть" if env["mlx_installed"] else "рантайм не найден")
        + f", но {env['mlx_implementation']}"
    )
    print()


def print_results(results: list[dict]) -> None:
    print(ROW_TEMPLATE.format(
        title="прогон", device="устр.", load="загрузка,с", synth="синтез,с", audio="звук,с",
        rtf="RTF", rss="RSS,МБ", mps="MPS драйвер,МБ",
    ))
    print("-" * 108)
    for item in results:
        print(ROW_TEMPLATE.format(
            title=f"«{item['requested']}»" + (" — ошибка" if item["error"] else ""),
            device=item["device"] or "—",
            load=_fmt(item["load_sec"]),
            synth=_fmt(item["synth_sec"]),
            audio=_fmt(item["audio_sec"]),
            rtf=_fmt(item["rtf"], 2),
            rss=_fmt(item["peak_rss_mb"], 0),
            mps=_fmt(item["mps_driver_mb"], 0),
        ))
    print()
    for item in results:
        if item["error"]:
            print(f"  ! «{item['requested']}»: {item['error']}")
    if any(item["error"] for item in results):
        print()


def verdict(env: dict, results: list[dict]) -> list[str]:
    """Выводы по измеренному: чем можно пользоваться, а что придётся обойти."""
    lines: list[str] = []
    by_setting = {item["requested"]: item for item in results}
    auto = by_setting.get("auto")
    cpu = by_setting.get("cpu")

    if env["torch_mps_available"]:
        default = "mps" if not env["force_cpu_env"] else "cpu (MPS запрещён переменной)"
    else:
        default = "cpu (MPS недоступен)"
    lines.append(f"Значение по умолчанию берёт: {default}.")

    if auto and auto["error"]:
        lines.append(
            "Прогон на выбранном устройстве не прошёл: движок откатится на CPU "
            "и повторит кусок (см. engines/qwen_engine.py), но каждый кусок будет "
            "платить за это дважды — сначала падением, потом повтором."
        )
        if default == "mps":
            lines.append(
                "Практический вывод: держать TTS_QWEN_FORCE_CPU=1, пока MPS-путь не починится."
            )
    elif auto and auto["rtf"] is not None:
        lines.append(
            f"Рабочий прогон идёт на {auto['device']} с RTF {auto['rtf']:.2f}: "
            + ("быстрее реального времени — диалог можно слушать почти сразу"
               if auto["rtf"] < 1
               else "медленнее реального времени — для длинных проектов это ощутимо")
        )

    if auto and cpu and auto["rtf"] and cpu["rtf"]:
        lines.append(f"Откат на CPU стоит ×{cpu['rtf'] / auto['rtf']:.1f} по времени синтеза.")

    if auto and cpu and auto["peak_rss_mb"] and cpu["peak_rss_mb"]:
        lines.append(
            f"RSS: {auto['peak_rss_mb']:.0f} МБ против {cpu['peak_rss_mb']:.0f} МБ на CPU. "
            "На Apple Silicon память MPS в RSS не попадает — судить по ней о запасе нельзя."
        )
    lines.append(
        "Числа выше — замер на этой машине и этих весах; переносить их на другую "
        "конфигурацию нельзя."
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Проба выполнимости Qwen3-TTS на этой машине")
    parser.add_argument("--ref", help="путь к записи голоса 3–15 с (WAV/MP3)")
    parser.add_argument("--ref-text", default="", help="расшифровка записи; без неё — режим «эксперимент»")
    parser.add_argument("--plan", action="store_true", help="напечатать план и выйти, моделей не трогая")
    parser.add_argument("--json", action="store_true", help="напечатать результат машинночитаемым JSON")
    args = parser.parse_args()

    if args.plan:
        print_plan()
        return 0

    if not args.ref:
        print("Нужен референс: --ref /путь/к/речи.wav (3–15 с). План без моделей: --plan")
        return 2
    reference = Path(args.ref)
    if not reference.is_file():
        print(f"Файл референса не найден: {reference}")
        return 2

    problems = prerequisites()
    if problems:
        for problem in problems:
            print(f"! {problem}")
        return 1

    env = environment()
    mode = "quality" if args.ref_text else ENGINE_MODE_EXPERIMENTAL
    if not args.json:
        print_environment(env)
        print(f"Референс: {reference} · режим: {mode}")
        print()

    results = []
    for device_setting, title in PROBES:
        if not args.json:
            print(f"Прогон «{title}» ({device_setting})…")
        results.append(probe(device_setting, str(reference), args.ref_text, mode))

    if args.json:
        print(json.dumps({"environment": env, "probes": results}, ensure_ascii=False, indent=2))
        return 0

    print_results(results)
    print("Вердикт:")
    for line in verdict(env, results):
        print(f"  · {line}")
    return 0 if not any(item["error"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
