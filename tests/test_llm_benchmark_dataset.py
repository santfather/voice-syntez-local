"""Фаза 2 Task 1: gold-корпус русской лингвистики и его валидация.

Проверяется сам датасет, а не модели: покрытие категорий, целостность spans,
дубликаты, наличие негативных проб и границы контекста. Ошибка в gold измеряется
метриками как ошибка модели, поэтому проверять его надо тестами.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.llm import schemas as s

DATASET_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "russian_linguistics"
DATASET_PATH = DATASET_DIR / "dataset.v1.jsonl"
PROMPT_PATH = DATASET_DIR / "prompts" / "analyzer.v1.txt"
MIN_CASES = 300
# Политика контекста (Task 2 §6): цель + не больше двух реплик с каждой стороны.
MAX_CONTEXT_SIDE = 2


@pytest.fixture(scope="module")
def dataset() -> list[s.DatasetCase]:
    return s.load_dataset(DATASET_PATH)


def test_benchmark_dataset_schema(dataset):
    """Датасет читается, валидируется целиком и покрывает все категории."""
    assert len(dataset) >= MIN_CASES, f"кейсов {len(dataset)}, нужно минимум {MIN_CASES}"
    problems = s.dataset_problems(dataset)
    assert problems == [], problems[:10]

    counts = s.category_counts(dataset)
    assert set(counts) == set(s.CATEGORIES)
    for category, count in counts.items():
        assert count > 0, f"категория {category} не покрыта"
    # По каждой категории должны быть не единичные пробы: иначе метрика по ней —
    # это шум из одного случая.
    thin = {name: count for name, count in counts.items() if count < 10}
    assert thin == {}, f"слишком мало кейсов в категориях: {thin}"

    ids = [case.id for case in dataset]
    assert len(ids) == len(set(ids)), "повторяющиеся id"
    assert all(case.language == "ru" for case in dataset)


def test_benchmark_gold_spans_match_source(dataset):
    """`source` обязан посимвольно совпадать с текстом в указанных границах."""
    for case in dataset:
        for item in case.expected:
            assert 0 <= item.span_start < item.span_end <= len(case.target_text), case.id
            actual = case.target_text[item.span_start : item.span_end]
            assert actual == item.source, f"{case.id}: {actual!r} != {item.source!r}"


def test_benchmark_rejects_invalid_gold_span():
    """Валидатор ловит gold с неверной границей, типом и дубликатом."""
    bad = s.DatasetCase(
        id="bad-0001",
        category=s.CATEGORY_HOMOGRAPH,
        target_text="Он поменял замок на двери.",
        expected=(
            s.Annotation(span_start=11, span_end=15, source="замок", type=s.TYPE_HOMOGRAPH),
        ),
    )
    problems = bad.validate()
    assert any("источник не совпадает" in problem for problem in problems)

    unknown_type = s.DatasetCase(
        id="bad-0002",
        category=s.CATEGORY_HOMOGRAPH,
        target_text="Да.",
        expected=(s.Annotation(span_start=0, span_end=2, source="Да", type="магия"),),
    )
    assert any("неизвестный тип" in problem for problem in unknown_type.validate())

    out_of_bounds = s.DatasetCase(
        id="bad-0003",
        category=s.CATEGORY_HOMOGRAPH,
        target_text="Да.",
        expected=(s.Annotation(span_start=0, span_end=99, source="Да", type=s.TYPE_HOMOGRAPH),),
    )
    assert any("вне текста" in problem for problem in out_of_bounds.validate())

    no_category = s.DatasetCase(
        id="bad-0004", category="выдуманная", target_text="Да.", expected=()
    )
    assert any("неизвестная категория" in problem for problem in no_category.validate())

    empty_gold = s.DatasetCase(
        id="bad-0005", category=s.CATEGORY_HOMOGRAPH, target_text="Да.", expected=()
    )
    assert any("ambiguous" in problem for problem in empty_gold.validate())


def test_benchmark_dataset_duplicate_ids_are_detected(dataset):
    doubled = list(dataset[:2]) + [dataset[0]]
    problems = s.dataset_problems(doubled)
    assert any("повтор id" in problem for problem in problems)


def test_benchmark_dataset_has_negative_and_ambiguous_cases(dataset):
    """Негативные пробы нужны для false positive rate, ambiguous — для ревью."""
    negative = [case for case in dataset if not case.expected]
    assert len(negative) >= 30, f"негативных проб всего {len(negative)}"
    assert any(case.category == s.CATEGORY_NEGATIVE for case in negative)
    assert any(case.expect_no_issue and case.category != s.CATEGORY_NEGATIVE for case in negative)

    ambiguous = [case for case in dataset if case.ambiguous]
    assert ambiguous, "ни один спорный кейс не помечен ambiguous"
    for case in ambiguous:
        # Спорный кейс обязан быть помечен к ревью: либо правками с needs_review,
        # либо (как короткие реплики) пояснением ожидаемого класса.
        if case.expected:
            assert all(item.needs_review for item in case.expected), case.id
        else:
            assert case.notes.strip(), f"{case.id}: спорный кейс без пояснения"


def test_benchmark_replica_ids_are_unique(dataset):
    """`replica_id` из id кейса уникален — иначе проверка id реплики бессмысленна."""
    replica_ids = [case.replica_id for case in dataset]
    assert len(replica_ids) == len(set(replica_ids)), "коллизия replica_id"


def test_benchmark_yo_cases_contain_restorable_form(dataset):
    """В ё-кейсах текст содержит «е»-форму, а gold предлагает «ё»."""
    yo_cases = [case for case in dataset if case.category == s.CATEGORY_YO]
    positive = [case for case in yo_cases if case.expected]
    assert len(positive) >= 15
    for case in positive:
        item = case.expected[0]
        assert item.type == s.TYPE_YO
        assert "ё" not in item.source.lower(), f"{case.id}: в исходнике уже есть «ё»"
        assert item.suggested_form, case.id
        # Форма меняется только заменой «е» → «ё», и хотя бы в одной позиции:
        # никакого переписывания слова gold не допускает.
        assert len(item.suggested_form) == len(item.source), case.id
        changes = 0
        for original, suggested in zip(item.source, item.suggested_form):
            if original == suggested:
                continue
            assert original in "еЕ" and suggested in "ёЁ", f"{case.id}: {item.source!r}→{item.suggested_form!r}"
            changes += 1
        assert changes >= 1, f"{case.id}: форма не изменилась"


def test_benchmark_context_is_bounded(dataset):
    """Контекст корпуса не превышает production-политику (2 + 2 реплики)."""
    for case in dataset:
        assert len(case.context_before) <= MAX_CONTEXT_SIDE, case.id
        assert len(case.context_after) <= MAX_CONTEXT_SIDE, case.id


def test_benchmark_expected_types_and_reasons_are_known(dataset):
    for case in dataset:
        for item in case.expected:
            assert item.type in s.ANNOTATION_TYPES, case.id
            if item.reason_code:
                assert item.reason_code in s.REASON_CODES, case.id
            assert 0.0 <= item.confidence <= 1.0, case.id
            assert "+" not in item.suggested_form, f"{case.id}: в gold попала разметка ударения"


def test_benchmark_short_replica_cases_expect_hint_not_text(dataset):
    """Короткие реплики: правок нет, но есть контекст и пояснение класса."""
    short_cases = [case for case in dataset if case.category == s.CATEGORY_SHORT]
    assert len(short_cases) >= 15
    for case in short_cases:
        assert not case.expected, f"{case.id}: у короткой реплики не должно быть правок"
        assert case.expect_no_issue, f"{case.id}: короткая реплика не помечена как «нет проблем»"
        assert case.notes.strip(), f"{case.id}: не указано, что это за реплика"
        # Смысл короткой реплики берётся из контекста, значит он обязателен.
        assert case.context_before or case.context_after, f"{case.id}: короткая реплика без контекста"


def test_benchmark_files_are_present_and_parse():
    """В каталоге корпуса лежат все обещанные файлы, и они читаются."""
    for name in ("dataset.v1.jsonl", "schema.json", "README.md"):
        assert (DATASET_DIR / name).exists(), name
    assert PROMPT_PATH.exists()
    assert (DATASET_DIR / "expected" / "analyzer.response.schema.json").exists()
    assert (DATASET_DIR / "expected" / "summary.json").exists()

    schema = json.loads((DATASET_DIR / "schema.json").read_text(encoding="utf-8"))
    assert schema["required"] == ["id", "category", "target_text", "expected"]

    summary = json.loads(
        (DATASET_DIR / "expected" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["cases"] >= MIN_CASES
    assert summary["ambiguous_cases"]


def test_dataset_schema_matches_response_schema_types():
    """Схема ответа и валидатор не должны разойтись по списку типов."""
    response_schema = s.analysis_json_schema()
    item_types = response_schema["properties"]["items"]["items"]["properties"]["type"]["enum"]
    assert tuple(item_types) == s.ANNOTATION_TYPES
    utterance_classes = response_schema["properties"]["utterance"]["properties"]["class"]["enum"]
    assert tuple(utterance_classes) == s.UTTERANCE_CLASSES


def test_dataset_case_prompt_payload_is_bounded_and_keeps_target(dataset):
    """Вход для модели: цель, ограниченный контекст, язык и числовой replica_id."""
    case = next(case for case in dataset if case.context_before)
    payload = case.prompt_payload()
    assert payload["target_text"] == case.target_text
    assert payload["language"] == "ru"
    assert isinstance(payload["replica_id"], int)
    assert set(payload) == {"replica_id", "target_text", "context_before", "context_after", "language"}
