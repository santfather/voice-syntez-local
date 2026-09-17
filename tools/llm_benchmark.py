"""CLI benchmark'а русской лингвистики (§12 постановки Task 1).

Запуск:

```bash
# верхняя граница без моделей и сети: проверяет gold, prompt, метрики и отчёты
./venv/bin/python tools/llm_benchmark.py --dry-run --limit 20

# smoke на 15 кейсах: убедиться, что модели отвечают и память в норме
./venv/bin/python tools/llm_benchmark.py --smoke

# полный прогон четырёх обязательных моделей
./venv/bin/python tools/llm_benchmark.py --full --resume

# сравнить с сохранённым baseline и обновить его
./venv/bin/python tools/llm_benchmark.py --full --compare-baseline
./venv/bin/python tools/llm_benchmark.py --full --save-baseline --baseline-model qwen3:8b
```

Почему CLI, а не «скрипт по месту»: прогон должен быть воспроизводимым одной
командой с зафиксированными версиями, а результат — лежать в предсказуемых файлах
(`raw.jsonl`, `metrics.json`, `run.json`, `benchmark.json`). Тогда чужой человек (и
следующая сессия агента) может перепроверить цифры, не читая код.

Разделение каталогов: полный прогон, smoke и dry-run пишут в разные подкаталоги,
поэтому smoke не портит baseline и не смешивается с полными результатами.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm import metrics as metrics_mod
from backend.llm import runner as runner_mod
from backend.llm import schemas as s
from backend.llm import versioning
from backend.llm.ollama_client import OllamaClient, OllamaUnavailableError
from backend.llm.runner import (
    BenchmarkError,
    BenchmarkRunner,
    dry_run_client,
)

DEFAULT_DATASET = ROOT / "benchmarks" / "russian_linguistics" / "dataset.v1.jsonl"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "russian_linguistics" / "results"
# Модели из постановки (§3): 14B и больше в этой задаче запрещены.
DEFAULT_MODELS: tuple[str, ...] = (
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:4b-instruct-2507-q8_0",
    "qwen3:8b",
    "gemma3:4b",
)
SMOKE_LIMIT = 15
EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark локальных LLM для подготовки русского текста к синтезу",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL-корпус")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="модель Ollama (можно несколько раз); по умолчанию — четыре обязательные",
    )
    parser.add_argument("--models", default="", help="то же, через запятую")
    parser.add_argument(
        "--category", action="append", default=[], help="категория корпуса (можно несколько)"
    )
    parser.add_argument("--limit", type=int, default=None, help="сколько кейсов взять")
    parser.add_argument("--repeat", type=int, default=1, help="повторов на кейс")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="каталог результатов"
    )
    parser.add_argument("--resume", action="store_true", help="продолжить прерванный прогон")
    parser.add_argument(
        "--smoke", action="store_true", help=f"короткий прогон ({SMOKE_LIMIT} кейсов)"
    )
    parser.add_argument(
        "--full", action="store_true", help="полный корпус (значение по умолчанию)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="без моделей и сети: подставной клиент отвечает gold-аннотациями",
    )
    parser.add_argument(
        "--response-format",
        choices=list(runner_mod.RESPONSE_FORMATS),
        default=runner_mod.RESPONSE_FORMAT_JSON,
        help="как передавать схему ответа: текстом в prompt (json) или грамматикой Ollama (schema)",
    )
    parser.add_argument("--num-ctx", type=int, default=8192, help="размер контекста")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--save-baseline", action="store_true", help="сохранить метрики как baseline"
    )
    parser.add_argument(
        "--baseline-model",
        default="",
        help="какая модель идёт в baseline (по умолчанию — единственная в прогоне)",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true", help="сравнить прогон с baseline"
    )
    parser.add_argument("--list-categories", action="store_true", help="показать категории и выйти")
    parser.add_argument("--quiet", action="store_true", help="не печатать прогресс")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    return parser


def resolve_models(args: argparse.Namespace) -> list[str]:
    """Список моделей: явные `--model`/`--models` или четыре обязательные."""
    models: list[str] = []
    for item in args.model:
        models.extend(part.strip() for part in str(item).split(",") if part.strip())
    for item in str(args.models or "").split(","):
        if item.strip():
            models.append(item.strip())
    return models or list(DEFAULT_MODELS)


def resolve_categories(args: argparse.Namespace) -> list[str]:
    categories: list[str] = []
    for item in args.category:
        categories.extend(part.strip() for part in str(item).split(",") if part.strip())
    return categories


def resolve_output(args: argparse.Namespace) -> Path:
    """Каталог результатов: dry-run, smoke и полный прогон не смешиваются."""
    if args.output != DEFAULT_OUTPUT:
        return Path(args.output)
    if args.dry_run:
        return DEFAULT_OUTPUT / "dry-run"
    if args.smoke:
        return DEFAULT_OUTPUT / "smoke"
    return DEFAULT_OUTPUT / "full"


def resolve_limit(args: argparse.Namespace) -> int | None:
    if args.limit is not None:
        return args.limit
    return SMOKE_LIMIT if args.smoke else None


def run_benchmark(args: argparse.Namespace, *, client=None) -> int:
    """Тело CLI, отделённое от разбора аргументов: так его проверяют тесты."""
    cases = s.load_dataset(args.dataset)
    models = resolve_models(args)
    output_dir = resolve_output(args)
    if client is None:
        client = (
            dry_run_client(cases, model_tags=models) if args.dry_run else OllamaClient()
        )

    options = {
        "num_ctx": int(args.num_ctx),
        "temperature": float(args.temperature),
        "seed": args.seed,
    }
    progress = (lambda message: None) if args.quiet else _printer
    runner = BenchmarkRunner(
        client=client,
        cases=cases,
        output_dir=output_dir,
        options=options,
        response_format=args.response_format,
        check_memory=not args.dry_run,
        dataset_version=versioning.BENCHMARK_VERSION,
        on_progress=progress,
    )
    try:
        report = runner.run(
            models,
            categories=resolve_categories(args) or None,
            limit=resolve_limit(args),
            repeat=args.repeat,
            resume=args.resume,
        )
    except BenchmarkError as exc:
        print(f"Прогон не выполнен: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OllamaUnavailableError as exc:
        print(f"Ollama недоступна: {exc}", file=sys.stderr)
        return EXIT_ERROR

    _print_report(report, output_dir)
    if args.compare_baseline or args.save_baseline:
        _handle_baseline(args, report, output_dir)
    return _exit_code(report)


def _handle_baseline(args: argparse.Namespace, report: dict, output_dir: Path) -> None:
    path = versioning.baseline_path(output_dir, report["dataset_version"])
    if args.compare_baseline:
        if not path.exists():
            print(f"baseline не найден: {path} (сначала --save-baseline)", file=sys.stderr)
        else:
            baseline = json.loads(path.read_text(encoding="utf-8"))
            stored_models = baseline.get("models") or {}
            for tag, data in report["models"].items():
                stored = stored_models.get(tag)
                if stored is None and baseline.get("model") == tag:
                    # Baseline хранит одну модель: «модель + метрики» верхним уровнем.
                    stored = {"metrics": baseline.get("metrics") or {}}
                if not stored:
                    print(f"{tag}: в baseline нет записи — сравнить не с чем")
                    continue
                diff = versioning.compare_with_baseline(data.get("metrics") or {}, stored)
                print(f"\n### Сравнение с baseline: `{tag}`")
                print(f"регрессий: {len(diff['regressions'])}, улучшений: {len(diff['improvements'])}")
                for name, values in diff["regressions"].items():
                    print(f"  - регрессия {name}: {values['baseline']} → {values['current']}")
                for name, values in diff["improvements"].items():
                    print(f"  + улучшение {name}: {values['baseline']} → {values['current']}")
    if args.save_baseline:
        tag = args.baseline_model or (
            next(iter(report["models"])) if report["models"] else ""
        )
        data = report["models"].get(tag)
        if data is None:
            print(f"нечего сохранять: модели «{tag}» нет в прогоне", file=sys.stderr)
            return
        metrics = data.get("metrics") or {}
        selected = {
            name: metrics["metrics"][name]
            for name in sorted(metrics.get("metrics") or {})
        }
        payload = {
            "dataset_version": report["dataset_version"],
            "prompt_version": report["prompt_version"],
            "schema_version": report["schema_version"],
            "benchmark_version": versioning.BENCHMARK_VERSION,
            "model": tag,
            "metrics": selected,
            "categories": {
                category: values.get("issue_f1") for category, values in metrics.get(
                    "categories", {}
                ).items()
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nbaseline сохранён: {path} (модель `{tag}`)")


def _exit_code(report: dict) -> int:
    """Код возврата: неполный прогон не должен выглядеть как успешный."""
    problems = [
        f"{tag}: {data.get('status')} — {data.get('reason')}"
        for tag, data in report["models"].items()
        if data.get("status") not in ("ok", "ok_unload_failed")
    ]
    if report.get("stopped_reason"):
        problems.append(report["stopped_reason"])
    if problems:
        print("\nПрогон неполный:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_INCOMPLETE
    return EXIT_OK


def _print_report(report: dict, output_dir: Path) -> None:
    reports = {
        tag: data["metrics"]
        for tag, data in report["models"].items()
        if data.get("metrics", {}).get("metrics")
    }
    print(f"\nКейсов: {report['cases']}, повторов: {report['repeat']}")
    if reports:
        print("\n" + metrics_mod.tradeoff_table(reports))
    for tag, data in report["models"].items():
        if data.get("status") in ("ok", "ok_unload_failed"):
            critical = (data["metrics"] or {}).get("critical_errors") or {}
            print(
                f"{tag}: {data['status']}, оценок {data['cases']} "
                + f"из {data.get('planned_cases', data['cases'])} кейсов, "
                + f"критических ошибок {critical.get('count', 0)} "
                + f"{critical.get('by_kind') or ''}"
            )
        else:
            print(f"{tag}: {data['status']} — {data.get('reason', '')}")
    print(f"\nРезультаты: {output_dir}")


def _printer(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.list_categories:
        cases = s.load_dataset(args.dataset)
        counts = s.category_counts(cases)
        for category, count in counts.items():
            print(f"{category}\t{count}")
        return EXIT_OK
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
