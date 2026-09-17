"""Сборка отчёта benchmark'а из результатов прогона (§17 постановки Task 1).

Читает `benchmarks/russian_linguistics/results/full/`: `benchmark.json` (сводка),
`<модель>/metrics.json` (метрики и паспорт) и `<модель>/run.json` (условия), и
печатает готовый markdown: таблицу trade-off, метрики по категориям, критические
ошибки, замеры памяти и рекомендацию по primary/fallback.

Почему отдельный инструмент, а не ручная таблица в отчёте: числа должны
пересчитываться из сохранённых результатов одной командой, иначе отчёт устареет
на первом же повторном прогоне, а расхождение никто не заметит.

Порядок выбора модели (Phase 6) — не «кто быстрее», а:
1) качество на русском (`issue_f1`, точность омографов, `yo_f1`);
2) `critical_error_rate`;
3) `false_positive_rate`;
4) надёжность JSON/схемы;
5) память;
6) задержка.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm import metrics as metrics_mod
from backend.llm import schemas as schemas_mod
from backend.llm import versioning

DEFAULT_RESULTS = ROOT / "benchmarks" / "russian_linguistics" / "results" / "full"
REPORT_PATH = ROOT / "llm_benchmark_report.md"
# Порядок критериев выбора (Phase 6). Первое отличие решает; задержка — последняя,
# потому что «быстрее» не значит «правильнее».
RANKING: tuple[tuple[str, bool], ...] = (
    ("issue_f1", True),
    ("homograph_accuracy", True),
    ("yo_f1", True),
    ("critical_error_rate", False),
    ("false_positive_rate", False),
    ("schema_valid_rate", True),
    ("peak_process_memory", False),
    ("latency_p95", False),
)
METRIC_TABLE = (
    "issue_precision",
    "issue_recall",
    "issue_f1",
    "false_positive_rate",
    "critical_error_rate",
    "schema_valid_rate",
    "valid_json_rate",
    "source_preservation_rate",
    "homograph_accuracy",
    "yo_f1",
    "span_accuracy",
    "utterance_accuracy",
    "needs_review_precision",
)
RESOURCE_TABLE = (
    "latency_p50",
    "latency_p95",
    "latency_mean",
    "latency_max",
    "tokens_per_second",
    "peak_process_memory",
    "peak_system_memory_percent",
    "memory_pressure_events",
)


def span_diagnostics(results_dir: Path, model_dir_name: str, cases: dict) -> dict:
    """Разбирает, как именно модели ошибаются в границах аннотаций.

    Это отдельный, самый полезный для Task 2 разрез: если модель находит нужное
    слово, но считает символы неверно, то проблема не в русском языке, а в
    протоколе — и backend обязан искать слово сам, а не доверять offset'ам.
    """
    path = results_dir / model_dir_name / "raw.jsonl"
    if not path.exists():
        return {}
    stats = {"responses": 0, "empty_items": 0, "exact_spans": 0, "word_found_wrong_bounds": 0,
             "source_not_in_text": 0, "no_json": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["responses"] += 1
        row = json.loads(line)
        case = cases.get(row.get("case_id"))
        payload, _ = schemas_mod.extract_json_object(row.get("raw_text") or "")
        if case is None or not payload:
            stats["no_json"] += 1
            continue
        items = payload.get("items") or []
        if not items:
            stats["empty_items"] += 1
            continue
        first = items[0]
        source = str(first.get("source") or "")
        start, end = first.get("span_start"), first.get("span_end")
        claimed = None
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(case.target_text):
            claimed = case.target_text[start:end]
        if source and claimed == source:
            stats["exact_spans"] += 1
        elif source and source in case.target_text:
            stats["word_found_wrong_bounds"] += 1
        elif source:
            stats["source_not_in_text"] += 1
    return stats


def load_run(results_dir: Path) -> dict:
    benchmark_path = results_dir / "benchmark.json"
    if not benchmark_path.exists():
        raise SystemExit(f"нет сводки прогона: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    models: dict[str, dict] = {}
    for tag in benchmark.get("models") or {}:
        model_dir = results_dir / metrics_mod_slug(tag)
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        models[tag] = {
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "metadata": json.loads((model_dir / "run.json").read_text(encoding="utf-8")),
            "slowest": json.loads((model_dir / "run.json").read_text(encoding="utf-8")),
        }
    return {"benchmark": benchmark, "models": models}


def metrics_mod_slug(tag: str) -> str:
    """Тот же slug, что у runner'а: иначе отчёт не найдёт каталог модели."""
    from backend.llm.runner import model_slug

    return model_slug(tag)


def rank_models(models: dict) -> list[tuple[str, list[str]]]:
    """Сортирует модели по критериям Phase 6 и объясняет порядок словами."""
    def sort_key(item: tuple[str, dict]):
        values = []
        for name, higher_is_better in RANKING:
            value = (item[1]["metrics"].get("metrics") or {}).get(name)
            if value is None:
                # «Нет данных» — худший случай: метрику не на чем было считать.
                value = -1.0 if higher_is_better else 1e9
            values.append(-value if higher_is_better else value)
        return tuple(values)

    ranked = sorted(models.items(), key=sort_key)
    reasons: list[tuple[str, list[str]]] = []
    for tag, data in ranked:
        metrics = data["metrics"].get("metrics") or {}
        reasons.append(
            (
                tag,
                [
                    f"issue_f1={_fmt(metrics.get('issue_f1'))}",
                    f"critical_error_rate={_fmt(metrics.get('critical_error_rate'))}",
                    f"false_positive_rate={_fmt(metrics.get('false_positive_rate'))}",
                    f"schema_valid_rate={_fmt(metrics.get('schema_valid_rate'))}",
                    f"latency_p95={_fmt(metrics.get('latency_p95'))} c",
                    f"peak_system_memory_percent={_fmt(metrics.get('peak_system_memory_percent'))}",
                ],
            )
        )
    return reasons


def build_report(run: dict, *, dataset: dict | None = None) -> str:
    benchmark = run["benchmark"]
    models = run["models"]
    ranked = rank_models(models)
    lines: list[str] = [
        "# Benchmark локальных LLM для подготовки русского текста к синтезу",
        "",
        (
            "Отчёт собран `tools/llm_benchmark_report.py` из сохранённых результатов "
            f"(`{DEFAULT_RESULTS.relative_to(ROOT)}`). Числа не переносились руками: "
            "любой повторный прогон пересобирает отчёт той же командой."
        ),
        "",
        "## Условия прогона",
        "",
        f"- benchmark version: `{versioning.BENCHMARK_VERSION}`",
        f"- dataset version: `{benchmark.get('dataset_version')}`"
        + (f", кейсов: {dataset.get('cases')}" if dataset else ""),
        f"- prompt version: `{benchmark.get('prompt_version')}`",
        f"- schema version: `{benchmark.get('schema_version')}`",
        f"- режим схемы: `{(benchmark.get('options') or {}).get('response_format', 'json')}`",
        (
            f"- num_ctx: `{(benchmark.get('options') or {}).get('num_ctx')}`, "
            f"temperature: `{(benchmark.get('options') or {}).get('temperature')}`, "
            f"seed: `{(benchmark.get('options') or {}).get('seed')}`"
        ),
        f"- Ollama: `{benchmark.get('ollama_version')}`",
        f"- длительность прогона: `{benchmark.get('duration_sec')}` c",
    ]
    if benchmark.get("stopped_reason"):
        lines.append(f"- прогон остановлен: {benchmark['stopped_reason']}")
    lines += ["", "## Модели", "", "| Модель | digest | размер, GiB | кейсов | статус |",
              "|---|---|---|---|---|"]
    for tag, data in models.items():
        metadata = data["metadata"]
        status = (benchmark.get("models") or {}).get(tag) or {}
        lines.append(
            f"| `{tag}` | `{metadata.get('model_digest', '')[:12]}` | "
            f"{metadata.get('model_size_gb')} | {data['metrics'].get('cases')} | "
            f"{status.get('status', '')} |"
        )

    lines += ["", "## Метрики качества", "", "| Модель | " +
              " | ".join(f"`{name}`" for name in METRIC_TABLE) + " |",
              "|---" * (len(METRIC_TABLE) + 1) + "|"]
    for tag, data in models.items():
        metrics = data["metrics"].get("metrics") or {}
        lines.append(
            f"| `{tag}` | " + " | ".join(_fmt(metrics.get(name)) for name in METRIC_TABLE) + " |"
        )

    lines += ["", "## Ресурсы и память", "", "| Модель | " +
              " | ".join(f"`{name}`" for name in RESOURCE_TABLE) + " |",
              "|---" * (len(RESOURCE_TABLE) + 1) + "|"]
    for tag, data in models.items():
        metrics = data["metrics"].get("metrics") or {}
        lines.append(
            f"| `{tag}` | " + " | ".join(_fmt(metrics.get(name)) for name in RESOURCE_TABLE) + " |"
        )

    lines += ["", "## Критические ошибки", "", "| Модель | всего кейсов | по видам |",
              "|---|---|---|"]
    for tag, data in models.items():
        critical = data["metrics"].get("critical_errors") or {}
        kinds = ", ".join(f"{name}: {count}" for name, count in (critical.get("by_kind") or {}).items())
        lines.append(f"| `{tag}` | {critical.get('count', 0)} | {kinds or '—'} |")

    lines += ["", "## Итог по категориям (issue_f1)", ""]
    categories = sorted(
        {
            category
            for data in models.values()
            for category in (data["metrics"].get("categories") or {})
        }
    )
    lines += ["| Категория | " + " | ".join(f"`{tag}`" for tag in models) + " |",
              "|---" * (len(models) + 1) + "|"]
    for category in categories:
        cells = []
        for data in models.values():
            values = (data["metrics"].get("categories") or {}).get(category) or {}
            cells.append(_fmt(values.get("issue_f1")))
        lines.append(f"| `{category}` | " + " | ".join(cells) + " |")

    diagnostics = run.get("diagnostics") or {}
    if diagnostics:
        lines += [
            "",
            "## Диагностика границ аннотаций",
            "",
            (
                "Ответ с неверными границами backend не применяет (правка текста без "
                "основания недопустима), поэтому важно понимать, в чём именно ошибка: "
                "модель не нашла место или нашла, но неверно посчитала символы."
            ),
            "",
            "| Модель | ответов | точные границы | слово найдено, границы неверны | слова нет в тексте | пустой items | нет JSON |",
            "|---|---|---|---|---|---|---|",
        ]
        for tag, stats in diagnostics.items():
            lines.append(
                f"| `{tag}` | {stats.get('responses', 0)} | {stats.get('exact_spans', 0)} | "
                f"{stats.get('word_found_wrong_bounds', 0)} | "
                f"{stats.get('source_not_in_text', 0)} | {stats.get('empty_items', 0)} | "
                f"{stats.get('no_json', 0)} |"
            )

    lines += ["", "## Рекомендация (Phase 6)", "",
              (
                  "Порядок выбора: качество на русском → критические ошибки → ложные "
                  "срабатывания → надёжность JSON/схемы → память → задержка. "
                  "Скорость сама по себе модель не выбирает."
              ), ""]
    for index, (tag, reasons) in enumerate(ranked):
        role = "**primary**" if index == 0 else ("**fallback**" if index == 1 else "резерв")
        lines.append(f"{index + 1}. `{tag}` — {role}: " + "; ".join(reasons))
    lines += [
        "",
        (
            "Primary — модель с лучшим качеством при приемлемом риске; fallback — "
            "следующая по тем же критериям, чтобы был запас при недоступности или "
            "деградации primary."
        ),
        "",
        "## Ограничения",
        "",
        (
            "- Метрики качества считаются по gold-корпусу; спорные кейсы (см. "
            "`expected/review.md`) подтверждены человеком до прогона."
        ),
        (
            "- Качество чтения омографов проверяется по ключевым словам значения, а не "
            "по знаку ударения: ударение размечает отдельный слой (RUAccent)."
        ),
        (
            "- Ответ с одной битой аннотацией отвергается целиком; в метриках видны и "
            "`critical_error_rate`, и `recovered_annotations` — сколько аннотаций из "
            "отвергнутых ответов было корректно по отдельности."
        ),
        (
            "- Замеры памяти зависят от того, что ещё запущено на машине; пороги "
            "политики указаны рядом с замерами."
        ),
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    return metrics_mod._format_value(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Отчёт по результатам benchmark'а")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--stdout", action="store_true", help="печатать отчёт вместо записи")
    args = parser.parse_args(argv)

    dataset_path = ROOT / "benchmarks" / "russian_linguistics" / "expected" / "summary.json"
    dataset = (
        json.loads(dataset_path.read_text(encoding="utf-8")) if dataset_path.exists() else None
    )
    run = load_run(args.results)
    dataset_path_full = (
        ROOT / "benchmarks" / "russian_linguistics" / "dataset.v1.jsonl"
    )
    cases = (
        {case.id: case for case in schemas_mod.load_dataset(dataset_path_full)}
        if dataset_path_full.exists()
        else {}
    )
    run["diagnostics"] = {
        tag: span_diagnostics(args.results, metrics_mod_slug(tag), cases)
        for tag in run["models"]
    }
    report = build_report(run, dataset=dataset)
    if args.stdout:
        print(report)
        return 0
    args.output.write_text(report, encoding="utf-8")
    print(f"отчёт записан: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
