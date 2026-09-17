"""Последовательный runner: очередь моделей, память, resume и метаданные.

Настоящие модели здесь не поднимаются (§13): клиент подставной, датчики памяти —
функции-заглушки. Проверяется то, что нельзя проверить глазами в отчёте:
порядок запуска, отсутствие конкурентного инференса, честный resume и паспорт
прогона.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.llm import memory_policy as memory
from backend.llm import metrics as metrics_mod
from backend.llm import runner as runner_mod
from backend.llm import schemas as s
from backend.llm.fake_client import FakeOllamaClient
from backend.llm.ollama_client import OllamaModel

MODELS = [
    OllamaModel(tag="qwen3:4b", digest="aaa111", size_bytes=5 * 1024**3),
    OllamaModel(tag="gemma3:4b", digest="bbb222", size_bytes=3 * 1024**3),
]


def _cases(count: int = 3) -> list[s.DatasetCase]:
    texts = [
        ("homograph-0001", "Мы гуляли вокруг старого замка на холме.", s.CATEGORY_HOMOGRAPH),
        ("yo-0002", "Он все понял без лишних слов.", s.CATEGORY_YO),
        ("no_issue-0003", "На кухне пахло свежим хлебом и корицей.", s.CATEGORY_NEGATIVE),
    ]
    cases = []
    for case_id, text, category in texts[:count]:
        expected = ()
        if category == s.CATEGORY_HOMOGRAPH:
            start = text.index("замка")
            expected = (
                s.Annotation(
                    span_start=start,
                    span_end=start + 5,
                    source="замка",
                    type=s.TYPE_HOMOGRAPH,
                    meaning="строение, крепость",
                ),
            )
        elif category == s.CATEGORY_YO:
            start = text.index("все")
            expected = (
                s.Annotation(
                    span_start=start,
                    span_end=start + 3,
                    source="все",
                    type=s.TYPE_YO,
                    suggested_form="всё",
                ),
            )
        cases.append(
            s.DatasetCase(
                id=case_id, category=category, target_text=text, expected=expected
            )
        )
    return cases


class RecordingClient(FakeOllamaClient):
    """Подставной клиент, который следит за конкурентной загрузкой моделей."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_loaded = 0
        self.loaded_sequence: list[str] = []

    def chat(self, model, messages, **kwargs):
        self.max_loaded = max(self.max_loaded, len(set(self._loaded) | {model}))
        self.loaded_sequence.append(model)
        return super().chat(model, messages, **kwargs)


def _gold(cases) -> dict:
    """Gold по id реплики — его получает подставной клиент, а не prompt модели."""
    return {case.replica_id: [item.to_dict() for item in case.expected] for case in cases}


def _perfect_client(cases, **kwargs) -> RecordingClient:
    """Клиент, отвечающий gold-аннотациями: верхняя граница метрик без моделей."""
    return RecordingClient(models=MODELS, gold_by_replica=_gold(cases), **kwargs)


def _runner(
    tmp_path: Path,
    *,
    client=None,
    sensor=None,
    cases=None,
    gate=None,
    on_progress=None,
    **kwargs,
) -> runner_mod.BenchmarkRunner:
    selected = cases if cases is not None else _cases()
    return runner_mod.BenchmarkRunner(
        client=client if client is not None else _perfect_client(selected),
        cases=selected,
        output_dir=tmp_path,
        sensor=sensor or (lambda: {"system_percent": 40.0, "available_gb": 12.0, "pressure": "green"}),
        gate=gate,
        on_progress=on_progress or (lambda message: None),
        dataset_version="1",
        **kwargs,
    )


def test_benchmark_models_run_sequentially(tmp_path):
    """Модели идут по очереди: вторая не загружается, пока первая в памяти."""
    client = _perfect_client(_cases(), latency_sec=0.0)
    runner = _runner(tmp_path, client=client)
    report = runner.run(["qwen3:4b", "gemma3:4b"])

    assert client.max_loaded == 1
    assert client.loaded_sequence == ["qwen3:4b"] * 3 + ["gemma3:4b"] * 3
    assert client.unloaded == ["qwen3:4b", "gemma3:4b"]
    assert report["models"]["qwen3:4b"]["status"] == "ok"
    assert report["models"]["gemma3:4b"]["unload_confirmed"] is True
    # Идеальный ответ fake-клиента: gold-аннотации, значит метрики максимальные.
    assert report["models"]["qwen3:4b"]["metrics"]["metrics"]["issue_f1"] == 1.0
    assert (tmp_path / "qwen3-4b" / "raw.jsonl").exists()
    assert (tmp_path / "gemma3-4b" / "raw.jsonl").exists()


def test_benchmark_stops_before_second_model_on_hard_stop(tmp_path):
    """HARD STOP не запускает модель вообще и останавливает весь прогон."""
    client = RecordingClient(models=MODELS)
    runner = _runner(
        tmp_path,
        client=client,
        sensor=lambda: {"system_percent": 91.0, "available_gb": 2.0, "pressure": "green"},
    )
    report = runner.run(["qwen3:4b", "gemma3:4b"])
    assert client.calls == []
    assert report["models"]["qwen3:4b"]["status"] == "hard_stop"
    assert "gemma3:4b" not in report["models"]
    assert "HARD STOP" in report["stopped_reason"]
    assert report["models"]["qwen3:4b"]["metrics"]["memory"]["level"] == memory.LEVEL_HARD_STOP


def test_memory_warning_blocks_next_heavy_job(tmp_path):
    """WARNING: новая тяжёлая задача не начинается (и gate не пускает вторую)."""
    client = RecordingClient(models=MODELS)
    runner = _runner(
        tmp_path,
        client=client,
        sensor=lambda: {"system_percent": 78.0, "available_gb": 9.0, "pressure": "green"},
    )
    report = runner.run(["qwen3:4b"])
    assert client.calls == []
    assert report["models"]["qwen3:4b"]["status"] == "blocked"
    assert "WARNING" in report["models"]["qwen3:4b"]["reason"]
    assert report["models"]["qwen3:4b"]["metrics"]["memory"]["allow_llm_start"] is False

    # Занятый тяжёлый ресурс (например, идущий синтез) тоже блокирует прогон.
    gate = memory.HeavyGate()
    ticket = gate.try_acquire(memory.HEAVY_TTS, owner="tts")
    try:
        busy_client = RecordingClient(models=MODELS)
        busy_runner = _runner(tmp_path / "busy", client=busy_client, gate=gate)
        busy_report = busy_runner.run(["qwen3:4b"])
        assert busy_client.calls == []
        assert busy_report["models"]["qwen3:4b"]["status"] == "blocked"
        assert "тяжёлый ресурс занят" in busy_report["models"]["qwen3:4b"]["reason"]
    finally:
        gate.release(ticket)


def test_memory_critical_blocks_llm_start_and_unloads(tmp_path):
    """CRITICAL: модель не стартует, а загруженные модели выгружаются."""
    client = RecordingClient(models=MODELS)
    client._loaded.append("qwen3:14b")  # чужая модель уже в памяти
    runner = _runner(
        tmp_path,
        client=client,
        sensor=lambda: {"system_percent": 84.0, "available_gb": 9.0, "pressure": "green"},
    )
    report = runner.run(["qwen3:4b"])
    assert client.calls == []
    assert report["models"]["qwen3:4b"]["status"] == "blocked"
    assert client.unloaded == ["qwen3:14b"]
    assert report["models"]["qwen3:4b"]["unloaded_before"] == ["qwen3:14b"]
    assert report["models"]["qwen3:4b"]["metrics"]["memory"]["unload_recommended"] is True


def test_hard_stop_never_starts_model(tmp_path):
    """HARD STOP по давлению macOS не даёт запустить модель даже при низком проценте."""
    client = RecordingClient(models=MODELS)
    runner = _runner(
        tmp_path,
        client=client,
        sensor=lambda: {"system_percent": 45.0, "available_gb": 12.0, "pressure": "red"},
    )
    report = runner.run(["qwen3:4b"])
    assert client.calls == []
    assert report["models"]["qwen3:4b"]["status"] == "hard_stop"
    assert "pressure" in report["models"]["qwen3:4b"]["reason"]


def test_benchmark_resume_does_not_duplicate_cases(tmp_path):
    """Повторный запуск с --resume досчитывает только недостающие кейсы."""
    client = RecordingClient(models=MODELS)
    runner = _runner(tmp_path, client=client, cases=_cases(3))
    first = runner.run(["qwen3:4b"], limit=2, resume=True)
    assert first["models"]["qwen3:4b"]["cases"] == 2
    assert len(client.calls) == 2

    second_runner = _runner(tmp_path, client=client, cases=_cases(3))
    second = second_runner.run(["qwen3:4b"], limit=3, resume=True)
    # Досчитан ровно один кейс, а в отчёте — все три.
    assert len(client.calls) == 3
    assert second["models"]["qwen3:4b"]["cases"] == 3
    lines = (tmp_path / "qwen3-4b" / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["case_id"] for line in lines]
    assert ids == ["homograph-0001", "yo-0002", "no_issue-0003"]
    assert len(ids) == len(set(ids))
    assert second["models"]["qwen3:4b"]["metrics"]["metrics"]["cases"] == 3


def test_benchmark_resume_reruns_without_flag(tmp_path):
    """Без --resume прогон повторяется целиком (сравнение прогонов важнее экономии)."""
    client = RecordingClient(models=MODELS)
    runner = _runner(tmp_path, client=client, cases=_cases(2))
    runner.run(["qwen3:4b"], resume=True)
    runner.run(["qwen3:4b"], resume=False)
    assert len(client.calls) == 4
    # Одинаковые ответы не должны задваиваться в файле сырых ответов.
    lines = (tmp_path / "qwen3-4b" / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_model_metadata_is_saved(tmp_path):
    """Паспорт прогона: тег, digest, размер, контекст и версии рядом с метриками."""
    runner = _runner(tmp_path, client=RecordingClient(models=MODELS))
    runner.run(["qwen3:4b"])
    model_dir = tmp_path / "qwen3-4b"
    metadata = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["model_tag"] == "qwen3:4b"
    assert metadata["model_digest"] == "aaa111"
    assert metadata["model_size_gb"] == 5.0
    assert metadata["context"] == runner_mod.DEFAULT_NUM_CTX
    assert metadata["benchmark_version"]
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metadata"]["model_digest"] == "aaa111"


def test_prompt_version_is_saved(tmp_path):
    """Версия prompt'а сохраняется: без неё старые числа невоспроизводимы."""
    runner = _runner(tmp_path, client=RecordingClient(models=MODELS))
    report = runner.run(["qwen3:4b"])
    assert report["prompt_version"] == runner.prompt.version == "1"
    assert report["schema_version"] == s.SCHEMA_VERSION
    assert report["dataset_version"] == "1"
    metadata = json.loads(
        (tmp_path / "qwen3-4b" / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["prompt_version"] == runner.prompt.version
    assert metadata["schema_version"] == s.SCHEMA_VERSION
    assert metadata["dataset_version"] == "1"


def test_benchmark_missing_model_is_reported_not_skipped_silently(tmp_path):
    """Отсутствующая модель — это запись в отчёте, а не молчаливый пропуск."""
    runner = _runner(tmp_path, client=RecordingClient(models=MODELS[:1]))
    report = runner.run(["qwen3:4b", "qwen3:14b"])
    assert report["models"]["qwen3:4b"]["status"] == "ok"
    assert report["models"]["qwen3:14b"]["status"] == "missing"
    assert "не найдена" in report["models"]["qwen3:14b"]["reason"]


def test_benchmark_repairs_invalid_answer_once(tmp_path):
    """Битый JSON после одного повтора остаётся критической ошибкой, а не тишиной."""

    def responder(case):
        return "конечно! вот результат:"

    client = RecordingClient(models=MODELS, responder=responder)
    runner = _runner(tmp_path, client=client)
    report = runner.run(["qwen3:4b"])
    assert len(client.calls) == 6  # 3 кейса × (запрос + repair)
    metrics = report["models"]["qwen3:4b"]["metrics"]["metrics"]
    assert metrics["valid_json_rate"] == 0.0
    assert metrics["critical_error_rate"] == 1.0
    kinds = report["models"]["qwen3:4b"]["metrics"]["critical_errors"]["by_kind"]
    assert kinds[metrics_mod.CRITICAL_INVALID_JSON] == 3


def test_benchmark_cancel_keeps_partial_results(tmp_path):
    """Отмена не теряет уже полученные ответы и продолжается позже."""
    client = RecordingClient(models=MODELS)
    runner = _runner(tmp_path, client=client, cases=_cases(3))
    cancelled = {"flag": False}

    def progress(message):
        if "получен" in message:  # pragma: no cover — заглушка прогресса
            cancelled["flag"] = True

    report = runner.run(["qwen3:4b"], cancel=lambda: len(client.calls) >= 1)
    assert report["models"]["qwen3:4b"]["status"] == "cancelled"
    assert report["models"]["qwen3:4b"]["cases"] == 1
    resumed = _runner(tmp_path, client=client, cases=_cases(3)).run(
        ["qwen3:4b"], resume=True
    )
    assert resumed["models"]["qwen3:4b"]["cases"] == 3
    assert cancelled["flag"] is False


def test_benchmark_select_cases_filters_and_limits(tmp_path):
    """Фильтр по категориям и --limit валидируются, а не игнорируются."""
    runner = _runner(tmp_path, cases=_cases(3))
    assert [case.id for case in runner.select_cases(limit=2)] == ["homograph-0001", "yo-0002"]
    assert [case.id for case in runner.select_cases(categories=[s.CATEGORY_NEGATIVE])] == [
        "no_issue-0003"
    ]
    with pytest.raises(runner_mod.BenchmarkError, match="неизвестные категории"):
        runner.select_cases(categories=["магия"])


def test_benchmark_fails_loudly_when_ollama_unavailable(tmp_path):
    """Ollama недоступна — прогон не начинается и не подменяется заглушкой."""
    runner = _runner(tmp_path, client=RecordingClient(models=MODELS, available=False))
    with pytest.raises(runner_mod.BenchmarkError, match="Ollama недоступна"):
        runner.run(["qwen3:4b"])


def test_benchmark_metrics_and_raw_are_separate_files(tmp_path):
    """Сырые ответы лежат отдельно от метрик: метрики пересчитываются, ответы — нет."""
    runner = _runner(tmp_path, client=RecordingClient(models=MODELS))
    runner.run(["qwen3:4b"], limit=2)
    model_dir = tmp_path / "qwen3-4b"
    raw = [json.loads(line) for line in (model_dir / "raw.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert {item["case_id"] for item in raw} == {"homograph-0001", "yo-0002"}
    assert all("raw_text" in item and "latency_sec" in item for item in raw)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "raw_text" not in json.dumps(metrics)[:200]
    assert metrics["planned_cases"] == 2
