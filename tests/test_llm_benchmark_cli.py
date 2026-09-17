"""CLI benchmark'а: аргументы, каталоги результатов, baseline и коды возврата.

Настоящие модели здесь не поднимаются: `--dry-run` идёт через подставной клиент, а
там, где нужен другой ответ, клиент передаётся в `run_benchmark` напрямую. Проверяем
именно то, что легко сломать незаметно: куда пишутся результаты, что попадает в
baseline и какой код возврата получает автоматизация.
"""

from __future__ import annotations

import json
from pathlib import Path

import tools.llm_benchmark as cli
from backend.llm.fake_client import FakeOllamaClient
from backend.llm.ollama_client import OllamaModel

MODELS = [
    OllamaModel(tag="qwen3:4b-instruct-2507-q4_K_M", digest="q4", size_bytes=2 * 1024**3),
    OllamaModel(tag="qwen3:8b", digest="q8", size_bytes=5 * 1024**3),
]


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return ["--dry-run", "--output", str(tmp_path), "--quiet", *extra]


def test_cli_dry_run_writes_full_result_set(tmp_path):
    """`--dry-run` проходит весь путь и пишет результаты в указанный каталог."""
    code = cli.main(_args(tmp_path, "--limit", "4", "--model", "qwen3:8b"))
    assert code == cli.EXIT_OK
    model_dir = tmp_path / "qwen3-8b"
    assert (tmp_path / "benchmark.json").exists()
    assert (model_dir / "raw.jsonl").exists()
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "run.json").exists()
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["issue_f1"] == 1.0
    assert metrics["metadata"]["model_digest"] == "dry-run-0"
    report = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert report["models"]["qwen3:8b"]["cases"] == 4


def test_cli_dry_run_uses_requested_models(tmp_path):
    """Dry-run объявляет именно запрошенные модели, а не одну подставную."""
    code = cli.main(
        _args(
            tmp_path,
            "--limit",
            "2",
            "--model",
            "qwen3:4b-instruct-2507-q4_K_M",
            "--model",
            "qwen3:8b",
        )
    )
    assert code == cli.EXIT_OK
    report = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert set(report["models"]) == {"qwen3:4b-instruct-2507-q4_K_M", "qwen3:8b"}
    # Модели объявлены с правдоподобным размером: иначе dry-run проверял бы не тот
    # путь решения о памяти.
    metadata = json.loads(
        (tmp_path / "qwen3-8b" / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["model_size_gb"] == 4.0


def test_cli_smoke_limit_and_separate_directory():
    """`--smoke` берёт 15 кейсов и пишет в свой каталог: полный прогон не портится."""
    args = cli.build_parser().parse_args(["--smoke"])
    assert cli.resolve_limit(args) == cli.SMOKE_LIMIT
    assert cli.resolve_output(args) == cli.DEFAULT_OUTPUT / "smoke"
    dry = cli.build_parser().parse_args(["--dry-run"])
    assert cli.resolve_output(dry) == cli.DEFAULT_OUTPUT / "dry-run"
    full = cli.build_parser().parse_args(["--full"])
    assert cli.resolve_output(full) == cli.DEFAULT_OUTPUT / "full"
    explicit = cli.build_parser().parse_args(["--output", "/tmp/x"])
    assert cli.resolve_output(explicit) == Path("/tmp/x")


def test_cli_models_and_categories_parsing():
    args = cli.build_parser().parse_args(
        ["--model", "a:1,b:2", "--models", "c:3", "--category", "yo,homograph"]
    )
    assert cli.resolve_models(args) == ["a:1", "b:2", "c:3"]
    assert cli.resolve_categories(args) == ["yo", "homograph"]
    default = cli.build_parser().parse_args([])
    assert cli.resolve_models(default) == list(cli.DEFAULT_MODELS)
    assert cli.resolve_categories(default) == []


def test_cli_lists_categories(capsys):
    """`--list-categories` — быстрый способ увидеть состав корпуса без прогона."""
    assert cli.main(["--list-categories"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "homograph\t40" in out
    assert "no_issue\t30" in out


def test_cli_unknown_category_is_an_error(tmp_path):
    """Опечатка в категории не должна молча давать пустой прогон."""
    code = cli.main(_args(tmp_path, "--category", "магия", "--limit", "1"))
    assert code == cli.EXIT_ERROR


def test_cli_incomplete_run_returns_one(tmp_path):
    """Пропущенная модель — код возврата 1, автоматизация увидит неполный прогон."""
    args = cli.build_parser().parse_args(
        ["--output", str(tmp_path), "--quiet", "--model", "qwen3:8b", "--limit", "2"]
    )
    client = FakeOllamaClient(models=MODELS[:1])
    assert cli.run_benchmark(args, client=client) == cli.EXIT_INCOMPLETE


def test_cli_benchmark_error_returns_two(tmp_path):
    """Ошибка прогона (нет Ollama) — код 2 и сообщение в stderr."""

    class BrokenClient(FakeOllamaClient):
        def health(self):
            from backend.llm.ollama_client import OllamaStatus

            return OllamaStatus(available=False, error="нет соединения")

    args = cli.build_parser().parse_args(
        ["--output", str(tmp_path), "--quiet", "--limit", "1"]
    )
    assert cli.run_benchmark(args, client=BrokenClient()) == cli.EXIT_ERROR


def test_cli_num_ctx_and_seed_reach_metadata(tmp_path):
    """Контекст и seed сохраняются: без них сравнение прогонов невоспроизводимо."""
    code = cli.main(_args(tmp_path, "--limit", "2", "--num-ctx", "4096", "--seed", "7"))
    assert code == cli.EXIT_OK
    metadata = json.loads(
        (tmp_path / "qwen3-4b-instruct-2507-q4_K_M" / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["context"] == 4096
    assert metadata["seed"] == 7
    assert metadata["options"]["num_ctx"] == 4096


def test_cli_saves_and_compares_baseline(tmp_path, capsys):
    """Baseline сохраняется и сравнивается: регрессии видны по именам метрик."""
    assert cli.main(
        _args(tmp_path, "--limit", "3", "--model", "qwen3:8b", "--save-baseline")
    ) == cli.EXIT_OK
    baseline_path = tmp_path / "baseline.v1.json"
    assert baseline_path.exists()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["model"] == "qwen3:8b"
    assert baseline["prompt_version"] == "1"
    assert baseline["metrics"]["issue_f1"] == 1.0
    assert baseline["categories"]["homograph"] == 1.0

    code = cli.main(
        _args(tmp_path, "--limit", "3", "--model", "qwen3:8b", "--compare-baseline")
    )
    assert code == cli.EXIT_OK

    # Сравнение обязано различать направление метрик: для issue_f1 рост — улучшение,
    # для tokens_per_second падение — регрессия. Иначе таблица baseline бесполезна.
    baseline["metrics"]["issue_f1"] = 0.25
    baseline["metrics"]["tokens_per_second"] = 10**9
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
    cli.main(_args(tmp_path, "--limit", "3", "--model", "qwen3:8b", "--compare-baseline"))
    printed = capsys.readouterr().out
    assert "+ улучшение issue_f1: 0.25 → 1.0" in printed
    assert "- регрессия tokens_per_second" in printed


def test_cli_baseline_missing_is_reported_not_crash(tmp_path, capsys):
    """Нет baseline — понятное сообщение, а не падение."""
    code = cli.main(
        _args(tmp_path, "--limit", "2", "--model", "qwen3:8b", "--compare-baseline")
    )
    assert code == cli.EXIT_OK
    assert "baseline не найден" in capsys.readouterr().err


def test_cli_repeat_makes_evaluations_and_keeps_cases(tmp_path):
    """`--repeat` добавляет оценки, но не кейсы: повторы не раздувают корпус."""
    assert cli.main(
        _args(tmp_path, "--limit", "3", "--model", "qwen3:8b", "--repeat", "2")
    ) == cli.EXIT_OK
    report = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert report["repeat"] == 2
    data = report["models"]["qwen3:8b"]
    assert data["cases"] == 6  # 3 кейса × 2 повтора
    assert data["metrics"]["evaluations"] == 6
    assert data["metrics"]["cases"] == 3
    assert data["metrics"]["planned_cases"] == 3
