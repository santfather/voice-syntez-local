"""Метрики benchmark'а (§9–§10 постановки): качество, риск, скорость, память.

Считаются по gold-корпусу и ответам модели. Сырые ответы сохраняются рядом с
метриками (это делает runner), поэтому любую цифру можно перепроверить руками, а
не верить агрегату.

Почему метрик много и нет одного score:

- у моделей разный trade-off (качество / память / скорость), и одна свёртка
  спрятала бы ровно то, ради чего benchmark делается;
- часть метрик измеряет риск (`critical_error_rate`, `false_positive_rate`),
  часть — полноту (`issue_recall`), и они тянут в разные стороны: модель, которая
  «находит проблемы везде», выигрывала бы по recall и проигрывала по риску.

Как сопоставляются аннотации gold и ответа:

- типы считаются совпадающими, если они равны или лежат в одной группе
  (`TYPE_ALIASES`): `toponym` — частный случай `proper_name`, `pronunciation` —
  частный случай `stress`;
- интервалы — если пересекаются; строгое совпадение границ считается отдельно
  (`span_accuracy`), чтобы различать «нашёл место» и «нашёл место точно»;
- сопоставление жадное: каждая аннотация gold получает не более одного партнёра,
  поэтому «одна аннотация на два gold-места» не даёт ложного TP.

Как читается омограф: модель не размечает ударение (это работа RUAccent), поэтому
чтение проверяется по `meaning` — набору ключевых слов правильного значения,
который лежит в gold. Это приблизительная проверка, и она названа приблизительной:
человек видит сырой ответ рядом с метрикой. Замена ключевых слов на «умную»
проверку смысла здесь не нужна: все модели сравниваются по одному правилу.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from . import schemas as s

# --- направление метрик -------------------------------------------------------
# Нужно для сравнения с baseline: без этой таблицы рост задержки выглядел бы как
# улучшение. Значение True — «чем больше, тем лучше».
HIGHER_IS_BETTER: dict[str, bool] = {
    "homograph_accuracy": True,
    "yo_precision": True,
    "yo_recall": True,
    "yo_f1": True,
    "issue_precision": True,
    "issue_recall": True,
    "issue_f1": True,
    "false_positive_rate": False,
    "needs_review_precision": True,
    "span_accuracy": True,
    "valid_json_rate": True,
    "schema_valid_rate": True,
    "source_preservation_rate": True,
    "utterance_accuracy": True,
    "critical_error_rate": False,
    "latency_p50": False,
    "latency_p95": False,
    "latency_mean": False,
    "tokens_per_second": True,
    "peak_process_memory": False,
    "peak_system_memory_percent": False,
    "memory_pressure_events": False,
}

# Виды критических ошибок (§10). Список закрытый: отчёт печатает раскладку по
# видам, и «прочие» в нём быть не должно — иначе вид ошибки непонятен.
CRITICAL_TEXT_CHANGE = "text_change"
CRITICAL_SOURCE_MISMATCH = "source_mismatch"
CRITICAL_INVALID_JSON = "invalid_json_after_repair"
CRITICAL_WRONG_HOMOGRAPH = "wrong_homograph_reading"
CRITICAL_INVENTED = "invented_name_or_number"
CRITICAL_REVIEW_IGNORED = "review_ignored"
CRITICAL_UNKNOWN_REPLICA = "unknown_replica"
CRITICAL_ERROR_KINDS: tuple[str, ...] = (
    CRITICAL_TEXT_CHANGE,
    CRITICAL_SOURCE_MISMATCH,
    CRITICAL_INVALID_JSON,
    CRITICAL_WRONG_HOMOGRAPH,
    CRITICAL_INVENTED,
    CRITICAL_REVIEW_IGNORED,
    CRITICAL_UNKNOWN_REPLICA,
)

# Уверенность, начиная с которой ответ считается «уверенным»: именно уверенные
# неверные утверждения опасны, осторожная модель должна уходить в needs_review.
CONFIDENT_THRESHOLD = 0.7

# Типы, где уверенное предсказание без опоры в gold — это выдуманное имя/число/
# произношение, а не безобидная лишняя аннотация.
INVENTED_TYPES: frozenset[str] = frozenset(
    {
        s.TYPE_PROPER_NAME,
        s.TYPE_TOPONYM,
        s.TYPE_NUMBER,
        s.TYPE_ABBREVIATION,
        s.TYPE_PRONUNCIATION,
    }
)

TYPE_ALIASES: tuple[frozenset[str], ...] = (
    frozenset({s.TYPE_PROPER_NAME, s.TYPE_TOPONYM}),
    frozenset({s.TYPE_STRESS, s.TYPE_PRONUNCIATION}),
)


@dataclass(frozen=True)
class Prediction:
    """Ответ модели на один кейс: сырой текст, разбор, тайминги и признаки ремонта."""

    case_id: str
    category: str = ""
    raw_text: str = ""
    analysis: s.LinguisticAnalysis | None = None
    errors: tuple[str, ...] = ()
    latency_sec: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Был ли повторный запрос после невалидного JSON и помог ли он. Критической
    # ошибкой невалидный JSON считается только тогда, когда repair не помог.
    repair_attempted: bool = False

    @property
    def json_object(self) -> dict | None:
        """Разобранный JSON-объект ответа (даже если валидация его отвергла)."""
        payload, _ = s.extract_json_object(self.raw_text)
        return payload

    @property
    def json_valid(self) -> bool:
        return self.json_object is not None

    @property
    def items(self) -> tuple[s.Annotation, ...]:
        return self.analysis.items if self.analysis is not None else ()

    def extra_fields(self) -> tuple[str, ...]:
        """Лишние поля ответа — в том числе поля с переписанным текстом."""
        payload = self.json_object
        if not payload:
            return ()
        return tuple(sorted(set(payload) - set(s.RESPONSE_FIELDS)))

    def rewrite_fields(self) -> tuple[str, ...]:
        return tuple(name for name in self.extra_fields() if name in s.REWRITE_FIELDS)


@dataclass
class CaseScore:
    """Разбор одного кейса: что совпало, что нет и какие сработали риски."""

    case_id: str
    category: str
    json_valid: bool = False
    schema_valid: bool = False
    source_preserved: bool = True
    gold_items: int = 0
    pred_items: int = 0
    matched: int = 0
    exact_spans: int = 0
    gold_has_issue: bool = False
    pred_has_issue: bool = False
    yo_tp: int = 0
    yo_fp: int = 0
    yo_fn: int = 0
    review_required: int = 0
    review_flagged: int = 0
    homograph_checked: bool = False
    homograph_ok: bool = False
    utterance_expected: str = ""
    utterance_got: str = ""
    critical: tuple[str, ...] = ()

    @property
    def utterance_ok(self) -> bool:
        return bool(self.utterance_expected) and self.utterance_expected == self.utterance_got


def same_type(left: str, right: str) -> bool:
    """Совпадают ли типы аннотаций (с учётом близких пар)."""
    if left == right:
        return True
    return any(left in group and right in group for group in TYPE_ALIASES)


def spans_overlap(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def reading_keywords(gold_meaning: str) -> tuple[str, ...]:
    """Ключевые слова чтения из gold: «строение, крепость» → («строение», «крепость»)."""
    return tuple(part.strip().lower() for part in gold_meaning.split(",") if part.strip())


def reading_stem(keyword: str) -> str:
    """Грубая основа слова: русские окончания меняются, корень — нет.

    Без этого «строение» в gold не совпало бы с «строении» в ответе модели, и
    метрика наказывала бы за верное чтение. Основа — не морфология, а грубое
    отсечение хвоста: для сравнения ключевых слов этого достаточно, а полноценный
    лемматизатор здесь был бы ещё одной моделью в benchmark'е.
    """
    return keyword[: max(4, len(keyword) - 2)]


def reading_matches(meaning: str, suggested_form: str, gold_meaning: str) -> bool:
    """Содержит ли пояснение модели ключевые слова правильного чтения.

    Проверка нарочно простая и одинаковая для всех моделей: ответ разбирается по
    ключевым словам gold, а не «по смыслу». Сырой ответ сохраняется рядом, поэтому
    спорный случай человек может перечитать сам.
    """
    keywords = reading_keywords(gold_meaning)
    if not keywords:
        return False
    tokens = re.findall(r"[а-яёa-z0-9-]+", f"{meaning} {suggested_form}".lower())
    stems = {reading_stem(token) for token in tokens}
    return any(reading_stem(keyword) in stems for keyword in keywords)


def match_items(
    gold: Sequence[s.Annotation], predicted: Sequence[s.Annotation]
) -> tuple[list[tuple[s.Annotation, s.Annotation]], list[s.Annotation], list[s.Annotation]]:
    """Жадно сопоставляет gold и ответ: пары, непокрытый gold, лишние ответы.

    Сопоставление идёт по индексам, а не по значениям: две одинаковые аннотации в
    gold (такое бывает в диалоговых кейсах) не должны «засчитываться» обе по
    одному предсказанию.
    """
    pairs: list[tuple[s.Annotation, s.Annotation]] = []
    matched_gold: set[int] = set()
    used: set[int] = set()
    for gold_index, gold_item in enumerate(gold):
        for index, pred_item in enumerate(predicted):
            if index in used:
                continue
            if not same_type(gold_item.type, pred_item.type):
                continue
            if not spans_overlap(
                gold_item.span_start, gold_item.span_end, pred_item.span_start, pred_item.span_end
            ):
                continue
            pairs.append((gold_item, pred_item))
            matched_gold.add(gold_index)
            used.add(index)
            break
    unmatched_gold = [item for index, item in enumerate(gold) if index not in matched_gold]
    unmatched_pred = [item for index, item in enumerate(predicted) if index not in used]
    return pairs, unmatched_gold, unmatched_pred


def score_case(case: s.DatasetCase, prediction: Prediction) -> CaseScore:
    """Считает всё, что можно посчитать по одному кейсу."""
    score = CaseScore(
        case_id=case.id,
        category=case.category,
        json_valid=prediction.json_valid,
        schema_valid=prediction.analysis is not None,
        gold_items=len(case.expected),
        pred_items=len(prediction.items),
        gold_has_issue=bool(case.expected),
        pred_has_issue=bool(prediction.items),
        utterance_expected=case.expected_utterance,
    )
    critical: list[str] = []
    # Изменение текста вне разрешённого span: либо лишнее поле с текстом, либо
    # `source`, не совпавший с текстом в указанных границах.
    if prediction.rewrite_fields():
        critical.append(CRITICAL_TEXT_CHANGE)
    if s.ERROR_SOURCE_MISMATCH in prediction.errors or s.ERROR_SPAN_BOUNDS in prediction.errors:
        critical.append(CRITICAL_SOURCE_MISMATCH)
    if prediction.errors and not prediction.json_valid and prediction.repair_attempted:
        critical.append(CRITICAL_INVALID_JSON)
    if s.ERROR_UNKNOWN_REPLICA in prediction.errors:
        critical.append(CRITICAL_UNKNOWN_REPLICA)
    # Источник сохранён, если модель не пыталась менять границы/текст и ответ
    # вообще разобрался: у невалидного ответа проверить нечего, и это не «ок».
    score.source_preserved = (
        prediction.json_valid
        and s.ERROR_SOURCE_MISMATCH not in prediction.errors
        and s.ERROR_SPAN_BOUNDS not in prediction.errors
        and s.ERROR_EMPTY_SOURCE not in prediction.errors
    )

    pairs, unmatched_gold, unmatched_pred = match_items(case.expected, prediction.items)
    score.matched = len(pairs)
    score.exact_spans = sum(
        1
        for gold_item, pred_item in pairs
        if gold_item.span_start == pred_item.span_start and gold_item.span_end == pred_item.span_end
    )

    for gold_item, pred_item in pairs:
        if gold_item.type == s.TYPE_YO:
            if gold_item.suggested_form.strip().lower() == pred_item.suggested_form.strip().lower():
                score.yo_tp += 1
            else:
                # Место найдено, а форма — нет: это и пропуск, и ложная правка.
                score.yo_fn += 1
                score.yo_fp += 1
        if gold_item.needs_review:
            score.review_required += 1
            if pred_item.needs_review:
                score.review_flagged += 1
            elif pred_item.confidence >= CONFIDENT_THRESHOLD:
                critical.append(CRITICAL_REVIEW_IGNORED)
        if gold_item.type == s.TYPE_HOMOGRAPH:
            score.homograph_checked = True
            wrong_reading = not reading_matches(
                pred_item.meaning, pred_item.suggested_form, gold_item.meaning
            )
            if wrong_reading and pred_item.confidence >= CONFIDENT_THRESHOLD:
                critical.append(CRITICAL_WRONG_HOMOGRAPH)
    for gold_item in unmatched_gold:
        if gold_item.type == s.TYPE_YO:
            score.yo_fn += 1
        if gold_item.type == s.TYPE_HOMOGRAPH:
            score.homograph_checked = True
    for pred_item in unmatched_pred:
        if pred_item.type == s.TYPE_YO:
            score.yo_fp += 1
        if pred_item.type in INVENTED_TYPES and pred_item.confidence >= CONFIDENT_THRESHOLD:
            critical.append(CRITICAL_INVENTED)
        # Уверенный омограф там, где его нет в gold, — та же ошибка чтения:
        # модель навязывает произношение, которого контекст не подтверждает.
        if pred_item.type == s.TYPE_HOMOGRAPH and pred_item.confidence >= CONFIDENT_THRESHOLD:
            critical.append(CRITICAL_WRONG_HOMOGRAPH)

    # Омографный кейс засчитан, только если все gold-омографы распознаны верно и
    # модель не добавила уверенных омографов «от себя».
    homographs = [item for item in case.expected if item.type == s.TYPE_HOMOGRAPH]
    if homographs:
        recognised = 0
        for gold_item in homographs:
            partner = next(
                (
                    pred_item
                    for pair_gold, pred_item in pairs
                    if pair_gold is gold_item
                    and reading_matches(
                        pred_item.meaning, pred_item.suggested_form, gold_item.meaning
                    )
                ),
                None,
            )
            if partner is not None:
                recognised += 1
        extra_homographs = [item for item in unmatched_pred if item.type == s.TYPE_HOMOGRAPH]
        score.homograph_ok = (
            recognised == len(homographs)
            and not extra_homographs
            and CRITICAL_WRONG_HOMOGRAPH not in critical
        )

    if prediction.analysis is not None:
        got = prediction.analysis.utterance.cls
        score.utterance_got = got if case.expected_utterance else ""
    score.critical = tuple(dict.fromkeys(critical))
    return score


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Процентиль по ближайшему рангу: у benchmark'а десятки замеров, не тысячи.

    Интерполяция на таком объёме создаёт ложное ощущение точности, поэтому берём
    фактически измеренное значение.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 4)


def _ratio(numerator: int, denominator: int) -> float:
    """Отношение; при пустом знаменателе — 0.0, а не «нет данных».

    Иначе модель, не ответившая ни разу, выглядела бы нейтрально по precision.
    """
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def aggregate(scores: Sequence[CaseScore]) -> dict:
    """Метрики качества по набору разобранных кейсов (без таймингов)."""
    total = len(scores)
    if not total:
        return {"cases": 0}
    positive = [score for score in scores if score.gold_has_issue]
    negative = [score for score in scores if not score.gold_has_issue]
    issue_tp = sum(1 for score in positive if score.pred_has_issue)
    issue_fn = len(positive) - issue_tp
    issue_fp = sum(1 for score in negative if score.pred_has_issue)

    yo_tp = sum(score.yo_tp for score in scores)
    yo_fp = sum(score.yo_fp for score in scores)
    yo_fn = sum(score.yo_fn for score in scores)
    yo_precision = _ratio(yo_tp, yo_tp + yo_fp)
    yo_recall = _ratio(yo_tp, yo_tp + yo_fn)

    review_required = sum(score.review_required for score in scores)
    review_flagged = sum(score.review_flagged for score in scores)

    issue_precision = _ratio(issue_tp, issue_tp + issue_fp)
    issue_recall = _ratio(issue_tp, issue_tp + issue_fn)

    homograph_scores = [score for score in scores if score.homograph_checked]
    critical_cases = [score for score in scores if score.critical]
    gold_annotations = sum(score.gold_items for score in scores)
    predicted_annotations = sum(score.pred_items for score in scores)
    exact_spans = sum(score.exact_spans for score in scores)
    utterance_scores = [score for score in scores if score.utterance_expected]

    return {
        "cases": total,
        "gold_annotations": gold_annotations,
        "predicted_annotations": predicted_annotations,
        "matched_annotations": sum(score.matched for score in scores),
        "homograph_accuracy": _ratio(
            sum(1 for score in homograph_scores if score.homograph_ok), len(homograph_scores)
        ),
        "yo_precision": yo_precision,
        "yo_recall": yo_recall,
        "yo_f1": _f1(yo_precision, yo_recall),
        "issue_precision": issue_precision,
        "issue_recall": issue_recall,
        "issue_f1": _f1(issue_precision, issue_recall),
        "false_positive_rate": _ratio(issue_fp, len(negative)),
        "needs_review_precision": _ratio(review_flagged, review_required),
        "span_accuracy": _ratio(exact_spans, gold_annotations),
        "valid_json_rate": _ratio(sum(1 for score in scores if score.json_valid), total),
        "schema_valid_rate": _ratio(sum(1 for score in scores if score.schema_valid), total),
        "source_preservation_rate": _ratio(
            sum(1 for score in scores if score.source_preserved), total
        ),
        "utterance_accuracy": _ratio(
            sum(1 for score in utterance_scores if score.utterance_ok), len(utterance_scores)
        ),
        "critical_error_rate": _ratio(len(critical_cases), total),
    }


def timing_metrics(predictions: Iterable[Prediction]) -> dict:
    """Скорость: задержки, разброс и скорость генерации.

    `latency_p95` важнее среднего: пользователь замечает именно медленные ответы,
    а среднее по M3 скрывает длинный хвост.
    """
    latencies = [
        prediction.latency_sec
        for prediction in predictions
        if prediction.latency_sec is not None
    ]
    speeds = [
        prediction.tokens_per_second
        for prediction in predictions
        if prediction.tokens_per_second is not None
    ]
    return {
        "latency_p50": percentile(latencies, 0.5),
        "latency_p95": percentile(latencies, 0.95),
        "latency_mean": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "latency_max": round(max(latencies), 4) if latencies else None,
        "tokens_per_second": round(sum(speeds) / len(speeds), 4) if speeds else None,
    }


def memory_metrics(memory: Mapping[str, object] | None) -> dict:
    """Память прогона: пики и события pressure — из замеров runner'а."""
    values = {
        "peak_process_memory": None,
        "peak_system_memory_percent": None,
        "memory_pressure_events": None,
    }
    for name in values:
        if memory and isinstance(memory.get(name), (int, float)):
            values[name] = round(float(memory[name]), 4)
    return values


def score_run(
    cases: Sequence[s.DatasetCase],
    predictions: Sequence[Prediction],
    *,
    memory: Mapping[str, object] | None = None,
) -> dict:
    """Полный отчёт по одному прогону: метрики, категории, критические ошибки.

    Кейс без ответа считается проваленным по всем метрикам, а не пропущенным: если
    модель упала на кейсе, это её результат, а не отсутствие данных.
    """
    by_case = {case.id: case for case in cases}
    scores: list[CaseScore] = []
    answered: set[str] = set()
    for prediction in predictions:
        case = by_case.get(prediction.case_id)
        if case is None:
            # Ответ на кейс, которого нет в корпусе: это ошибка runner'а, а не
            # модели, и молча портить метрики ей не даём.
            continue
        scores.append(score_case(case, prediction))
        answered.add(case.id)
    for case in cases:
        if case.id in answered:
            continue
        # Кейс без ответа — результат модели (ноль по всем метрикам), а не пропуск.
        scores.append(score_case(case, Prediction(case_id=case.id, category=case.category)))

    metrics = aggregate(scores)
    metrics.update(timing_metrics(predictions))
    metrics.update(memory_metrics(memory))

    categories: dict[str, dict] = {}
    for category in s.CATEGORIES:
        subset = [score for score in scores if score.category == category]
        if not subset:
            continue
        category_metrics = aggregate(subset)
        category_metrics.update(
            timing_metrics(
                prediction
                for prediction in predictions
                if prediction.category == category
            )
        )
        categories[category] = category_metrics

    critical_by_kind: Counter[str] = Counter()
    critical_cases: list[dict] = []
    for score in scores:
        if not score.critical:
            continue
        critical_by_kind.update(score.critical)
        critical_cases.append({"case_id": score.case_id, "kinds": list(score.critical)})

    return {
        "cases": len(cases),
        "evaluations": len(scores),
        "metrics": metrics,
        "categories": categories,
        "critical_errors": {
            "count": len(critical_cases),
            "by_kind": dict(sorted(critical_by_kind.items())),
            "cases": critical_cases,
        },
        "memory": memory_metrics(memory),
    }


def format_run_report(report: Mapping[str, object], *, title: str = "Прогон") -> str:
    """Markdown-отчёт по одному прогону: метрики, категории, критические ошибки."""
    metrics = report.get("metrics") or {}
    lines = [f"# {title}", "", f"Кейсов: {report.get('cases', 0)}", "", "## Метрики", "",
             "| Метрика | Значение |", "|---|---|"]
    for name in sorted(metrics):
        lines.append(f"| `{name}` | {_format_value(metrics[name])} |")

    categories = report.get("categories") or {}
    if categories:
        names = [
            "cases", "issue_f1", "false_positive_rate", "homograph_accuracy",
            "yo_f1", "span_accuracy", "critical_error_rate", "latency_p50", "latency_p95",
        ]
        lines += ["", "## По категориям", "",
                  "| Категория | " + " | ".join(names) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for category, values in categories.items():
            cells = " | ".join(_format_value(values.get(name)) for name in names)
            lines.append(f"| `{category}` | {cells} |")

    critical = report.get("critical_errors") or {}
    lines += ["", "## Критические ошибки", "",
              f"Всего кейсов с критической ошибкой: {critical.get('count', 0)}"]
    by_kind = critical.get("by_kind") or {}
    if by_kind:
        lines += ["", "| Вид | Кейсов |", "|---|---|"]
        lines += [f"| `{kind}` | {count} |" for kind, count in by_kind.items()]
    return "\n".join(lines) + "\n"


def tradeoff_table(reports: Mapping[str, Mapping[str, object]]) -> str:
    """Таблица trade-off по моделям: качество, риск, скорость, память.

    Один score здесь не считается намеренно: выбор модели — это выбор компромисса,
    и таблица показывает его целиком, а не подменяет решением.
    """
    names = [
        "issue_f1", "false_positive_rate", "critical_error_rate", "homograph_accuracy",
        "yo_f1", "span_accuracy", "utterance_accuracy", "latency_p50", "latency_p95",
        "tokens_per_second", "peak_process_memory", "peak_system_memory_percent",
        "memory_pressure_events",
    ]
    lines = ["| Модель | " + " | ".join(f"`{name}`" for name in names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for model, report in reports.items():
        metrics = report.get("metrics") or {}
        cells = " | ".join(_format_value(metrics.get(name)) for name in names)
        lines.append(f"| `{model}` | {cells} |")
    return "\n".join(lines) + "\n"


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return str(value)
