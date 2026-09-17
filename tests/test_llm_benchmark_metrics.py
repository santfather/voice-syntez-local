"""Метрики benchmark'а: проверка на маленьких ручных примерах.

Метрики — единственное место, где качество модели превращается в числа, и ошибка
здесь не видна глазом: неверная формула просто даёт «красивый» отчёт. Поэтому
каждая метрика проверяется на примере, где правильный ответ известен заранее.
"""

from __future__ import annotations

import json

from backend.llm import metrics as m
from backend.llm import schemas as s


def _item(text: str, source: str, type_: str, **kwargs) -> s.Annotation:
    start = text.index(source)
    return s.Annotation(
        span_start=start,
        span_end=start + len(source),
        source=source,
        type=type_,
        **kwargs,
    )


def _gold(
    case_id: str,
    text: str,
    category: str,
    items: tuple[s.Annotation, ...],
    **kwargs,
) -> s.DatasetCase:
    return s.DatasetCase(
        id=case_id, category=category, target_text=text, expected=items, **kwargs
    )


def _respond(
    case: s.DatasetCase,
    items: tuple[s.Annotation, ...] = (),
    *,
    extra: dict | None = None,
    utterance: str = "NORMAL",
    latency: float = 1.0,
    tokens_per_second: float = 20.0,
    repair_attempted: bool = False,
    raw: str | None = None,
) -> m.Prediction:
    """Строит ответ так, как его строит runner: через настоящую валидацию."""
    if raw is None:
        payload = {
            "schema_version": s.SCHEMA_VERSION,
            "replica_id": case.replica_id,
            "items": [item.to_dict() for item in items],
            "utterance": {"class": utterance, "context_dependency": "LOW"},
        }
        payload.update(extra or {})
        raw = json.dumps(payload, ensure_ascii=False)
    analysis, errors = s.parse_analysis(
        raw, expected_replica_id=case.replica_id, target_text=case.target_text
    )
    return m.Prediction(
        case_id=case.id,
        category=case.category,
        raw_text=raw,
        analysis=analysis,
        errors=tuple(errors),
        latency_sec=latency,
        tokens_per_second=tokens_per_second,
        repair_attempted=repair_attempted,
    )


CASTLE = "Мы гуляли вокруг старого замка на холме."
LOCK = "Он поменял замок на входной двери."


def _castle_case(case_id: str = "homograph-0001") -> s.DatasetCase:
    return _gold(
        case_id,
        CASTLE,
        s.CATEGORY_HOMOGRAPH,
        (_item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="строение, крепость, здание"),),
    )


def _lock_case(case_id: str = "homograph-0002") -> s.DatasetCase:
    return _gold(
        case_id,
        LOCK,
        s.CATEGORY_HOMOGRAPH,
        (_item(LOCK, "замок", s.TYPE_HOMOGRAPH, meaning="запор, механизм, ключ"),),
    )


def test_metrics_homograph_accuracy():
    """Омограф засчитан только при верном чтении, а не при верном месте."""
    right = _respond(
        _castle_case(),
        (_item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="речь о строении", confidence=0.9),),
    )
    wrong_reading = _respond(
        _lock_case(),
        (_item(LOCK, "замок", s.TYPE_HOMOGRAPH, meaning="имеется в виду строение", confidence=0.95),),
    )
    cautious_wrong = _respond(
        _castle_case("homograph-0003"),
        (_item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="запор", confidence=0.4),),
    )
    missing = _respond(_lock_case("homograph-0004"))

    cases = [
        _castle_case(),
        _lock_case(),
        _castle_case("homograph-0003"),
        _lock_case("homograph-0004"),
    ]
    report = m.score_run(cases, [right, wrong_reading, cautious_wrong, missing])
    assert report["metrics"]["homograph_accuracy"] == 0.25
    assert right is not None
    assert m.score_case(cases[0], right).homograph_ok is True
    assert m.score_case(cases[1], wrong_reading).homograph_ok is False
    assert m.score_case(cases[2], cautious_wrong).homograph_ok is False
    # Уверенное неверное чтение — критическая ошибка, осторожное (0.4) — нет:
    # задача модели в спорном месте — уйти в needs_review, а не угадать.
    assert m.CRITICAL_WRONG_HOMOGRAPH in m.score_case(cases[1], wrong_reading).critical
    assert m.CRITICAL_WRONG_HOMOGRAPH not in m.score_case(cases[2], cautious_wrong).critical


def test_metrics_homograph_extra_guess_breaks_case():
    """Лишний уверенный омограф «от себя» обнуляет кейс, даже если gold найден."""
    good = _item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="строение", confidence=0.9)
    invented = _item(CASTLE, "старого", s.TYPE_HOMOGRAPH, meaning="возраст", confidence=0.9)
    case = _castle_case()
    score = m.score_case(case, _respond(case, (good, invented)))
    assert score.homograph_ok is False
    assert m.CRITICAL_WRONG_HOMOGRAPH in score.critical


def test_metrics_yo_precision_recall():
    """Для «ё» проверяется и место, и предложенная форма."""
    text = "Он все понял без лишних слов."
    case = _gold("yo-0001", text, s.CATEGORY_YO, (_item(text, "все", s.TYPE_YO, suggested_form="всё"),))
    negative = _gold("no_issue-0002", "Мы ели яблоки и слушали радио.", s.CATEGORY_NEGATIVE, ())

    correct = _respond(case, (_item(text, "все", s.TYPE_YO, suggested_form="всё"),))
    wrong_form = _respond(case, (_item(text, "все", s.TYPE_YO, suggested_form="все"),))
    missing = _respond(case)
    false_alarm = _respond(
        negative,
        (_item("Мы ели яблоки и слушали радио.", "ели", s.TYPE_YO, suggested_form="ёл"),),
    )

    report = m.score_run([case, negative], [correct, _respond(negative)])
    assert report["metrics"]["yo_precision"] == 1.0
    assert report["metrics"]["yo_recall"] == 1.0
    assert report["metrics"]["yo_f1"] == 1.0

    # Лишняя «ё» на негативном кейсе — ложная правка: recall не падает, точность
    # падает вдвое, и именно это должно быть видно в отчёте.
    report = m.score_run([case, negative], [correct, false_alarm])
    assert report["metrics"]["yo_precision"] == 0.5
    assert report["metrics"]["yo_recall"] == 1.0
    assert report["metrics"]["yo_f1"] == 0.6667

    # Место найдено, а форма — нет: это одновременно пропуск и ложная правка.
    report = m.score_run([case, negative], [wrong_form, _respond(negative)])
    assert report["metrics"]["yo_precision"] == 0.0
    assert report["metrics"]["yo_recall"] == 0.0

    # Модель не предложила ни одной «ё»: точности нет (нечего оценивать), а
    # полнота честно нулевая.
    report = m.score_run([case, negative], [missing, _respond(negative)])
    assert report["metrics"]["yo_precision"] is None
    assert report["metrics"]["yo_recall"] == 0.0
    assert report["metrics"]["yo_f1"] is None


def test_metrics_false_positive_rate():
    """False positive rate считается по кейсам, где в gold нет ни одной правки."""
    clean = "Сегодня хорошая погода, и я рад тебя видеть."
    cases = [
        _gold("no_issue-0001", clean, s.CATEGORY_NEGATIVE, ()),
        _gold("no_issue-0002", "Кот спал на подоконнике, свернувшись клубком.", s.CATEGORY_NEGATIVE, ()),
        _gold("short-0003", "Да.", s.CATEGORY_SHORT, (), expect_no_issue=True),
    ]
    invented = _respond(
        cases[0],
        (_item(clean, "хорошая", s.TYPE_TERM, suggested_form="хорошая", confidence=0.8),),
    )
    report = m.score_run(cases, [invented, _respond(cases[1]), _respond(cases[2])])
    assert report["metrics"]["false_positive_rate"] == 0.3333
    assert report["metrics"]["issue_precision"] == 0.0
    # В gold нет ни одной правки — recall мерить не на чем, и это «—», а не 0.0:
    # иначе чистый корпус выглядел бы как провал по полноте.
    assert report["metrics"]["issue_recall"] is None
    assert report["metrics"]["issue_f1"] is None


def test_metrics_valid_json_rate():
    """`valid_json_rate` считает JSON, а `schema_valid_rate` — прошедший контракт."""
    text = "Он поменял замок на входной двери."
    case = _gold("homograph-0001", text, s.CATEGORY_HOMOGRAPH, (_item(text, "замок", s.TYPE_HOMOGRAPH, meaning="запор"),))
    valid = _respond(case, (_item(text, "замок", s.TYPE_HOMOGRAPH, meaning="запор"),))
    broken_json = _respond(case, raw="конечно, вот ответ:")
    not_object = _respond(case, raw="[1, 2, 3]")
    extra_field = _respond(
        case,
        (_item(text, "замок", s.TYPE_HOMOGRAPH, meaning="запор"),),
        extra={"final_text": "Он сменил замок."},
    )
    report = m.score_run([case] * 4, [valid, broken_json, not_object, extra_field])
    assert report["metrics"]["valid_json_rate"] == 0.5
    assert report["metrics"]["schema_valid_rate"] == 0.25


def test_metrics_critical_error_rate():
    """Каждый вид критической ошибки из §10 попадает в отчёт своим именем."""
    text = "Роман Достоевского перевели на сорок языков."
    review_case = _gold(
        "name-0001",
        text,
        s.CATEGORY_NAME,
        (
            _item(
                text,
                "Достоевского",
                s.TYPE_PROPER_NAME,
                needs_review=True,
                confidence=0.4,
                reason_code=s.REASON_UNKNOWN_WORD,
            ),
        ),
    )
    clean = _gold("no_issue-0001", "На кухне пахло свежим хлебом и корицей.", s.CATEGORY_NEGATIVE, ())
    clean_text = clean.target_text

    # 1) уверенный ответ там, где gold требует проверку;
    ignored_review = _respond(
        review_case,
        (
            _item(
                text,
                "Достоевского",
                s.TYPE_PROPER_NAME,
                needs_review=False,
                confidence=0.95,
            ),
        ),
    )
    # 2) попытка вернуть переписанный текст;
    rewrite = _respond(clean, extra={"text": "На кухне пахло хлебом и корицей."})
    # 3) `source` не совпал с текстом в границах;
    mismatch = _respond(
        clean,
        raw=json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": clean.replica_id,
                "items": [
                    {
                        "span_start": 3,
                        "span_end": 8,
                        "source": "кухне пахло",
                        "type": "term",
                        "confidence": 0.9,
                        "needs_review": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )
    # 4) выдуманное имя с высокой уверенностью;
    invented = _respond(
        clean,
        (
            _item(
                clean_text,
                "корицей",
                s.TYPE_PROPER_NAME,
                confidence=0.9,
                needs_review=False,
            ),
        ),
    )
    # 5) невалидный JSON, который не спас даже повторный запрос;
    broken = _respond(clean, raw="{не json}", repair_attempted=True)
    # 6) ответ про другую реплику.
    wrong_replica = _respond(
        clean,
        raw=json.dumps(
            {"schema_version": s.SCHEMA_VERSION, "replica_id": 999_999, "items": []},
            ensure_ascii=False,
        ),
    )
    cases = [review_case, clean]
    predictions = [ignored_review, rewrite, mismatch, invented, broken, wrong_replica]
    report = m.score_run(cases, predictions)
    kinds = report["critical_errors"]["by_kind"]
    # Ответов шесть, и каждый — со своей критической ошибкой; кейсов два.
    assert report["cases"] == 2
    assert report["evaluations"] == 6
    assert report["critical_errors"]["count"] == 6
    assert report["metrics"]["critical_error_rate"] == 1.0
    assert kinds[m.CRITICAL_REVIEW_IGNORED] == 1
    for kind in (m.CRITICAL_TEXT_CHANGE, m.CRITICAL_SOURCE_MISMATCH, m.CRITICAL_INVENTED,
                 m.CRITICAL_INVALID_JSON, m.CRITICAL_UNKNOWN_REPLICA):
        assert kinds[kind] == 1, kind
    assert set(kinds) <= set(m.CRITICAL_ERROR_KINDS)


def test_metrics_span_accuracy_and_type_aliases():
    """Пересечение интервалов — это находка, совпадение границ — точность."""
    text = "Рейкьявик — самая северная столица."
    case = _gold("toponym-0001", text, s.CATEGORY_TOPONYM, (_item(text, "Рейкьявик", s.TYPE_TOPONYM, needs_review=True, confidence=0.4),))
    wide = _item(text, "Рейкьявик", s.TYPE_PROPER_NAME, needs_review=True, confidence=0.4)
    wide = s.Annotation(
        span_start=wide.span_start, span_end=wide.span_end - 1, source="Рейкьяви", type=wide.type,
        needs_review=True, confidence=0.4,
    )
    # Тип `proper_name` совместим с gold `toponym`, границы шире — находка есть.
    prediction = _respond(case, (wide,))
    score = m.score_case(case, prediction)
    assert score.matched == 1
    assert score.exact_spans == 0
    assert score.review_flagged == 1
    report = m.score_run([case], [prediction])
    assert report["metrics"]["span_accuracy"] == 0.0
    assert report["metrics"]["needs_review_precision"] == 1.0


def test_metrics_issue_and_utterance_and_categories():
    """Полнота по проблемам, класс реплики и разбивка по категориям."""
    text = "Он все понял без лишних слов."
    yo_case = _gold("yo-0001", text, s.CATEGORY_YO, (_item(text, "все", s.TYPE_YO, suggested_form="всё"),))
    short = _gold(
        "short-0002",
        "Да.",
        s.CATEGORY_SHORT,
        (),
        expect_no_issue=True,
        expected_utterance="CONFIRMATION",
        context_before=("Ты придёшь завтра?",),
    )
    correct_yo = _respond(yo_case, (_item(text, "все", s.TYPE_YO, suggested_form="всё"),))
    correct_short = _respond(short, utterance="NEGATION")
    report = m.score_run([yo_case, short], [correct_yo, correct_short])
    assert report["metrics"]["issue_recall"] == 1.0
    assert report["metrics"]["issue_precision"] == 1.0
    assert report["metrics"]["utterance_accuracy"] == 0.0
    assert set(report["categories"]) == {s.CATEGORY_YO, s.CATEGORY_SHORT}
    assert report["categories"][s.CATEGORY_SHORT]["utterance_accuracy"] == 0.0
    report = m.score_run([yo_case, short], [correct_yo, _respond(short, utterance="CONFIRMATION")])
    assert report["metrics"]["utterance_accuracy"] == 1.0


def test_metrics_undefined_per_category_is_dash_not_zero():
    """В категории без gold-правок precision/recall не «ноль», а «—»."""
    text = "На кухне пахло свежим хлебом и корицей."
    clean = _gold("no_issue-0001", text, s.CATEGORY_NEGATIVE, ())
    report = m.score_run([clean], [_respond(clean)])
    category = report["categories"][s.CATEGORY_NEGATIVE]
    assert category["issue_precision"] is None
    assert category["issue_recall"] is None
    assert category["false_positive_rate"] == 0.0
    assert report["metrics"]["false_positive_rate"] == 0.0
    assert m._format_value(category["issue_precision"]) == "—"
    assert m._format_value(category["false_positive_rate"]) == "0"


def test_metrics_missing_response_is_a_failure_not_a_gap():
    """Кейс без ответа — это результат модели (ноль), а не пропуск в отчёте."""
    text = "Он все понял без лишних слов."
    case = _gold("yo-0001", text, s.CATEGORY_YO, (_item(text, "все", s.TYPE_YO, suggested_form="всё"),))
    report = m.score_run([case], [])
    assert report["metrics"]["cases"] == 1
    assert report["metrics"]["issue_recall"] == 0.0
    assert report["metrics"]["valid_json_rate"] == 0.0
    assert report["metrics"]["schema_valid_rate"] == 0.0
    assert report["metrics"]["source_preservation_rate"] == 0.0


def test_metrics_latency_and_memory_and_ranking():
    """Задержки считаются по ближайшему рангу, память переносится из замеров."""
    text = "Он поменял замок на входной двери."
    case = _gold("homograph-0001", text, s.CATEGORY_HOMOGRAPH, (_item(text, "замок", s.TYPE_HOMOGRAPH, meaning="запор"),))
    predictions = [
        _respond(case, latency=value, tokens_per_second=10.0 + value) for value in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    report = m.score_run(
        [case] * 5,
        predictions,
        memory={
            "peak_process_memory": 5.5,
            "peak_system_memory_percent": 71.25,
            "memory_pressure_events": 2,
        },
    )
    assert report["metrics"]["latency_p50"] == 3.0
    assert report["metrics"]["latency_p95"] == 5.0
    assert report["metrics"]["latency_mean"] == 3.0
    assert report["metrics"]["tokens_per_second"] == 13.0
    assert report["metrics"]["peak_process_memory"] == 5.5
    assert report["metrics"]["peak_system_memory_percent"] == 71.25
    assert report["metrics"]["memory_pressure_events"] == 2
    # Направление метрик: без него рост задержки считался бы улучшением.
    assert m.HIGHER_IS_BETTER["latency_p95"] is False
    assert m.HIGHER_IS_BETTER["critical_error_rate"] is False
    assert m.HIGHER_IS_BETTER["issue_f1"] is True


def test_metrics_report_and_tradeoff_render():
    """Отчёт печатает метрики, категории, критические ошибки и trade-off."""
    case = _castle_case()
    prediction = _respond(
        case, (_item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="строение", confidence=0.9),),
        latency=2.5, tokens_per_second=18.0,
    )
    report = m.score_run([case], [prediction])
    text = m.format_run_report(report, title="qwen3:4b")
    assert "# qwen3:4b" in text
    assert "homograph_accuracy" in text
    assert "По категориям" in text
    assert "Критические ошибки" in text
    table = m.tradeoff_table({"qwen3:4b": report, "gemma3:4b": report})
    assert "`qwen3:4b`" in table and "`gemma3:4b`" in table
    assert "latency_p95" in table
    assert m._format_value(None) == "—"


def test_metrics_known_metric_names_are_declared():
    """Все метрики отчёта имеют направление сравнения с baseline."""
    case = _castle_case()
    report = m.score_run(
        [case],
        [_respond(case, (_item(CASTLE, "замка", s.TYPE_HOMOGRAPH, meaning="строение"),))],
    )
    for name in report["metrics"]:
        if name in {
            "cases",
            "positive_cases",
            "negative_cases",
            "gold_annotations",
            "predicted_annotations",
            "matched_annotations",
            "latency_max",
        }:
            continue
        assert name in m.HIGHER_IS_BETTER, name
