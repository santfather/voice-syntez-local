"""Схемы лингвистического анализа: что подаём в LLM и что принимаем обратно.

Контракт один на два применения: benchmark (сравнить модели) и Analyzer (готовить
русский текст к синтезу). Поэтому схемы живут отдельным модулем, а не внутри
инструмента: разойдись они — benchmark мерил бы не то, что работает в бою.

Главное правило контракта: **модель возвращает только аннотации**, то есть
участки исходной строки и предложения по ним. Она не возвращает переписанный
текст, и backend никогда не берёт текст из ответа. Отсюда и проверки: границы
обязаны попадать в строку, `source` — совпадать с тем, что в этих границах лежит,
типы — быть из известного списка, интервалы — не перекрываться. Если ответ не
проходит, он не применяется **целиком** (частичное применение аннотаций к
пользовательскому тексту — это ровно тот «свободный rewrite», который запрещён).

Схема версионируется: `SCHEMA_VERSION` уходит и в prompt, и в сохранённый анализ,
и в метаданные прогона — без этого нельзя понять, почему старый результат
перестал воспроизводиться.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"

# --- категории корпуса (§6) ---------------------------------------------------
CATEGORY_HOMOGRAPH = "homograph"
CATEGORY_YO = "yo"
CATEGORY_MORPHOLOGY = "morphology"
CATEGORY_NAME = "name"
CATEGORY_TOPONYM = "toponym"
CATEGORY_ABBREVIATION = "abbreviation"
CATEGORY_NUMBER = "number"
CATEGORY_LATIN = "latin"
CATEGORY_TERM = "term"
CATEGORY_DIALOGUE = "dialogue_context"
CATEGORY_SHORT = "short_replica"
CATEGORY_NEGATIVE = "no_issue"
CATEGORIES: tuple[str, ...] = (
    CATEGORY_HOMOGRAPH,
    CATEGORY_YO,
    CATEGORY_MORPHOLOGY,
    CATEGORY_NAME,
    CATEGORY_TOPONYM,
    CATEGORY_ABBREVIATION,
    CATEGORY_NUMBER,
    CATEGORY_LATIN,
    CATEGORY_TERM,
    CATEGORY_DIALOGUE,
    CATEGORY_SHORT,
    CATEGORY_NEGATIVE,
)

# --- типы аннотаций -----------------------------------------------------------
# Тип отвечает на вопрос «что это за место в тексте», а не «что с ним делать»:
# решение принимает backend по правилам (`needs_review`, словарь, review).
TYPE_HOMOGRAPH = "homograph"
TYPE_YO = "yo"
TYPE_STRESS = "stress"
TYPE_PRONUNCIATION = "pronunciation"
TYPE_PROPER_NAME = "proper_name"
TYPE_TOPONYM = "toponym"
TYPE_ABBREVIATION = "abbreviation"
TYPE_NUMBER = "number"
TYPE_FOREIGN = "foreign"
TYPE_TERM = "term"
TYPE_PUNCTUATION = "punctuation"
TYPE_AMBIGUOUS = "ambiguous"
ANNOTATION_TYPES: tuple[str, ...] = (
    TYPE_HOMOGRAPH,
    TYPE_YO,
    TYPE_STRESS,
    TYPE_PRONUNCIATION,
    TYPE_PROPER_NAME,
    TYPE_TOPONYM,
    TYPE_ABBREVIATION,
    TYPE_NUMBER,
    TYPE_FOREIGN,
    TYPE_TERM,
    TYPE_PUNCTUATION,
    TYPE_AMBIGUOUS,
)

# Классы реплики в ответе (§3.5 Таска 2). Верхний регистр — как в документе.
UTTERANCE_CLASSES: tuple[str, ...] = (
    "NORMAL",
    "SHORT_REPLY",
    "QUESTION",
    "CONFIRMATION",
    "NEGATION",
    "SURPRISE",
    "CONTEXT_DEPENDENT",
)
CONTEXT_DEPENDENCIES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")

# Коды причин: короткие, машинные, без chain-of-thought (§15 Таска 2).
REASON_CONTEXT_DISAMBIGUATION = "CONTEXT_DISAMBIGUATION"
REASON_YO_REQUIRED = "YO_REQUIRED"
REASON_YO_UNSURE = "YO_UNSURE"
REASON_UNKNOWN_WORD = "UNKNOWN_WORD"
REASON_PROPER_NAME = "PROPER_NAME"
REASON_ABBREVIATION = "ABBREVIATION"
REASON_NUMBER_FORM = "NUMBER_FORM"
REASON_FOREIGN_WORD = "FOREIGN_WORD"
REASON_TERM = "TERM"
REASON_NO_ISSUE = "NO_ISSUE"
REASON_CODES: tuple[str, ...] = (
    REASON_CONTEXT_DISAMBIGUATION,
    REASON_YO_REQUIRED,
    REASON_YO_UNSURE,
    REASON_UNKNOWN_WORD,
    REASON_PROPER_NAME,
    REASON_ABBREVIATION,
    REASON_NUMBER_FORM,
    REASON_FOREIGN_WORD,
    REASON_TERM,
    REASON_NO_ISSUE,
)

# Ошибки разбора и валидации — кодами: по ним считаются метрики и объясняется
# пользователю, почему ответ модели не применён.
ERROR_NOT_JSON = "not_json"
ERROR_NOT_OBJECT = "not_object"
ERROR_SCHEMA_VERSION = "schema_version"
ERROR_UNKNOWN_REPLICA = "unknown_replica"
ERROR_UNKNOWN_TYPE = "unknown_type"
ERROR_UNKNOWN_REASON = "unknown_reason"
ERROR_SPAN_BOUNDS = "span_bounds"
ERROR_SOURCE_MISMATCH = "source_mismatch"
ERROR_EMPTY_SOURCE = "empty_source"
ERROR_CONFIDENCE_RANGE = "confidence_range"
ERROR_OVERLAP = "overlap"
ERROR_UNKNOWN_UTTERANCE_CLASS = "unknown_utterance_class"
ERROR_UNKNOWN_FIELD = "unknown_field"

# Поля, которые модели разрешено возвращать. Список закрытый: схема ответа и так
# не содержит поля для текста, но модель может добавить его «от себя» — и тогда
# невнимательный код однажды возьмёт текст из ответа. Поэтому лишнее поле — это
# ошибка ответа, а не безобидный мусор.
RESPONSE_FIELDS: tuple[str, ...] = ("schema_version", "replica_id", "items", "utterance")
ANNOTATION_FIELDS: tuple[str, ...] = (
    "span_start",
    "span_end",
    "source",
    "type",
    "meaning",
    "suggested_form",
    "confidence",
    "needs_review",
    "reason_code",
)
UTTERANCE_FIELDS: tuple[str, ...] = ("class", "context_dependency", "relevant_replica_ids")
# Ключи, которыми модель чаще всего пытается вернуть переписанный текст. Выделены
# отдельно: метрики считают такую попытку критической ошибкой (`text_change`).
REWRITE_FIELDS: tuple[str, ...] = (
    "text",
    "final_text",
    "source_text",
    "normalized",
    "normalized_text",
    "rewritten",
    "rewritten_text",
    "corrected",
    "corrected_text",
    "output",
    "result",
    "translation",
)

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


@dataclass(frozen=True)
class Annotation:
    """Одно предложение модели по участку текста.

    `source` — то, что модель прочитала в этих границах; backend сверяет его с
    исходной строкой, поэтому «модель передвинула границы и пересказала» ловится
    сразу. `suggested_form` — предлагаемая замена/разметка (для `yo` — форма с
    «ё», для `stress` — с ударением, для `pronunciation` — словарная замена).
    """

    span_start: int
    span_end: int
    source: str
    type: str
    meaning: str = ""
    suggested_form: str = ""
    confidence: float = 0.0
    needs_review: bool = True
    reason_code: str = ""

    @property
    def length(self) -> int:
        return max(self.span_end - self.span_start, 0)

    def to_dict(self) -> dict:
        return {
            "span_start": self.span_start,
            "span_end": self.span_end,
            "source": self.source,
            "type": self.type,
            "meaning": self.meaning,
            "suggested_form": self.suggested_form,
            "confidence": round(float(self.confidence), 4),
            "needs_review": bool(self.needs_review),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Annotation:
        return cls(
            span_start=int(raw.get("span_start", -1)),
            span_end=int(raw.get("span_end", -1)),
            source=str(raw.get("source") or ""),
            type=str(raw.get("type") or ""),
            meaning=str(raw.get("meaning") or ""),
            suggested_form=str(raw.get("suggested_form") or ""),
            confidence=float(raw.get("confidence") or 0.0),
            needs_review=bool(raw.get("needs_review", True)),
            reason_code=str(raw.get("reason_code") or ""),
        )


@dataclass(frozen=True)
class UtteranceHint:
    """Подсказка о реплике целиком (§3.5 Таска 2).

    Нужна коротким репликам: модель говорит, что это ответ/вопрос/подтверждение и
    какие соседи важны, но **не управляет аудио** и не добавляет ничего в
    `final_text` — это делает Short Utterance Strategy отдельно.
    """

    cls: str = "NORMAL"
    context_dependency: str = "LOW"
    relevant_replica_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "class": self.cls,
            "context_dependency": self.context_dependency,
            "relevant_replica_ids": list(self.relevant_replica_ids),
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> UtteranceHint:
        raw = raw or {}
        ids = raw.get("relevant_replica_ids") or []
        return cls(
            cls=str(raw.get("class") or "NORMAL").upper(),
            context_dependency=str(raw.get("context_dependency") or "LOW").upper(),
            relevant_replica_ids=tuple(int(item) for item in ids if _is_int(item)),
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class LinguisticAnalysis:
    """Разобранный ответ модели: аннотации и подсказка о реплике."""

    replica_id: int
    items: tuple[Annotation, ...] = ()
    utterance: UtteranceHint = field(default_factory=UtteranceHint)
    schema_version: str = SCHEMA_VERSION
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "replica_id": self.replica_id,
            "items": [item.to_dict() for item in self.items],
            "utterance": self.utterance.to_dict(),
        }

    def by_type(self) -> dict[str, list[Annotation]]:
        result: dict[str, list[Annotation]] = {}
        for item in self.items:
            result.setdefault(item.type, []).append(item)
        return result


def _extract_json(raw_text: str) -> tuple[dict | None, list[str]]:
    """Достаёт JSON-объект из ответа модели.

    Модель обязана отвечать JSON, но страховка нужна: иногда вокруг объекта
    оказывается текст или markdown-забор. Мы **не** «ремонтируем» JSON
    эвристиками — только снимаем забор и пробуем найти объект; всё остальное
    считается невалидным ответом (метрика `valid_json_rate`).
    """
    text = (raw_text or "").strip()
    if not text:
        return None, [ERROR_NOT_JSON]
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None, [ERROR_NOT_JSON]
        try:
            payload = json.loads(text[start : end + 1])
        except ValueError:
            return None, [ERROR_NOT_JSON]
    if not isinstance(payload, dict):
        return None, [ERROR_NOT_OBJECT]
    return payload, []


def normalize_schema_version(value: object) -> str:
    """Приводит версию схемы к сравнимому виду.

    Модели почти всегда отдают `"1.0"` там, где в контракте стоит `"1"` — это та же
    самая версия, а не другая. Считать такое расхождение ошибкой значило бы мерить
    педантичность формата вместо русского языка; при этом настоящая смена версии
    (`"2"`, `"1.1"`) обязана ломать разбор. Поэтому снимаем косметику: ведущая `v`,
    пробелы и хвостовой `.0`.
    """
    text = str(value or "").strip().lstrip("vV")
    text = text.removesuffix(".0")
    return text


def extract_json_object(raw_text: str) -> tuple[dict | None, list[str]]:
    """Публичный разбор JSON-объекта из ответа — нужен метрикам.

    Метрики считают `valid_json_rate` и проверяют попытку вернуть переписанный
    текст. Для этого нужен **сырой** объект ответа, в том числе когда валидация
    его целиком отвергла.
    """
    return _extract_json(raw_text)


def parse_analysis(
    raw_text: str,
    *,
    expected_replica_id: int | None = None,
    target_text: str | None = None,
    allowed_types: Iterable[str] = ANNOTATION_TYPES,
    check_schema_version: bool = True,
) -> tuple[LinguisticAnalysis | None, list[str]]:
    """Разбирает и валидирует ответ модели.

    Возвращает `(анализ, ошибки)`. При любой ошибке анализ — `None`: ответ
    применяется целиком или не применяется вовсе. Это осознанная строгость: одна
    принятая аннотация из сломанного ответа — это изменение пользовательского
    текста без основания.
    """
    payload, errors = _extract_json(raw_text)
    if payload is None:
        return None, errors

    errors = []
    allowed = set(allowed_types)
    if set(payload) - set(RESPONSE_FIELDS):
        errors.append(ERROR_UNKNOWN_FIELD)
    raw_version = payload.get("schema_version")
    version = normalize_schema_version(raw_version) or SCHEMA_VERSION
    if check_schema_version and version != SCHEMA_VERSION:
        errors.append(ERROR_SCHEMA_VERSION)

    replica_id = payload.get("replica_id")
    if not _is_int(replica_id):
        errors.append(ERROR_UNKNOWN_REPLICA)
        replica_id = -1
    replica_id = int(replica_id)
    if expected_replica_id is not None and replica_id != expected_replica_id:
        errors.append(ERROR_UNKNOWN_REPLICA)

    items: list[Annotation] = []
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict):
            errors.append(ERROR_NOT_OBJECT)
            continue
        if set(raw_item) - set(ANNOTATION_FIELDS):
            errors.append(ERROR_UNKNOWN_FIELD)
        item = Annotation.from_dict(raw_item)
        if item.type not in allowed:
            errors.append(ERROR_UNKNOWN_TYPE)
        if item.reason_code and item.reason_code not in REASON_CODES:
            errors.append(ERROR_UNKNOWN_REASON)
        if not (CONFIDENCE_MIN <= item.confidence <= CONFIDENCE_MAX):
            errors.append(ERROR_CONFIDENCE_RANGE)
        if not item.source:
            errors.append(ERROR_EMPTY_SOURCE)
        if target_text is not None:
            if item.span_start < 0 or item.span_end > len(target_text) or item.span_end <= item.span_start:
                errors.append(ERROR_SPAN_BOUNDS)
            elif target_text[item.span_start : item.span_end] != item.source:
                errors.append(ERROR_SOURCE_MISMATCH)
        items.append(item)

    # Пересекающиеся аннотации — конфликт: неясно, какая из них описывает текст.
    ordered = sorted(items, key=lambda item: (item.span_start, item.span_end))
    for previous, current in itertools.pairwise(ordered):
        if current.span_start < previous.span_end:
            errors.append(ERROR_OVERLAP)

    utterance_raw = payload.get("utterance")
    if isinstance(utterance_raw, dict) and set(utterance_raw) - set(UTTERANCE_FIELDS):
        errors.append(ERROR_UNKNOWN_FIELD)
    utterance = UtteranceHint.from_dict(utterance_raw)
    if utterance.cls not in UTTERANCE_CLASSES:
        errors.append(ERROR_UNKNOWN_UTTERANCE_CLASS)
    if utterance.context_dependency not in CONTEXT_DEPENDENCIES:
        errors.append(ERROR_UNKNOWN_UTTERANCE_CLASS)

    unique = list(dict.fromkeys(errors))
    if unique:
        return None, unique
    return (
        LinguisticAnalysis(
            replica_id=replica_id,
            items=tuple(items),
            utterance=utterance,
            schema_version=version,
            raw_text=raw_text,
        ),
        [],
    )


# --- корпус -------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetCase:
    """Один кейс gold-корпуса: текст, контекст и проверенные аннотации.

    `expected` — то, что считается правильным ответом. Пустой список ожидаем для
    негативных кейсов (в тексте нечего исправлять): именно на них измеряется
    false positive rate.
    """

    id: str
    category: str
    target_text: str
    expected: tuple[Annotation, ...] = ()
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    ambiguous: bool = False
    # Внутри категории тоже бывают негативные кейсы: например, «ё» здесь не нужно.
    # Явный флаг вместо догадки по пустому `expected` — иначе валидатор не отличит
    # намеренную негативную пробу от забытой разметки.
    expect_no_issue: bool = False
    # Ожидаемый класс реплики для коротких реплик и диалоговых кейсов: смысл там
    # берётся из контекста, а не из самого текста, и проверяется отдельной метрикой.
    expected_utterance: str = ""
    notes: str = ""
    language: str = "ru"

    def prompt_payload(self, replica_id: int | None = None) -> dict:
        """Вход для модели (§7): target, ограниченный контекст и язык."""
        return {
            "replica_id": _case_replica_id(self.id) if replica_id is None else replica_id,
            "target_text": self.target_text,
            "context_before": list(self.context_before),
            "context_after": list(self.context_after),
            "language": self.language,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "target_text": self.target_text,
            "context_before": list(self.context_before),
            "context_after": list(self.context_after),
            "expected": [item.to_dict() for item in self.expected],
            "ambiguous": bool(self.ambiguous),
            "expect_no_issue": bool(self.expect_no_issue),
            "expected_utterance": self.expected_utterance,
            "notes": self.notes,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DatasetCase:
        return cls(
            id=str(raw.get("id") or ""),
            category=str(raw.get("category") or ""),
            target_text=str(raw.get("target_text") or ""),
            expected=tuple(
                Annotation.from_dict(item) for item in (raw.get("expected") or []) if isinstance(item, dict)
            ),
            context_before=tuple(str(item) for item in (raw.get("context_before") or [])),
            context_after=tuple(str(item) for item in (raw.get("context_after") or [])),
            ambiguous=bool(raw.get("ambiguous", False)),
            expect_no_issue=bool(raw.get("expect_no_issue", False)),
            expected_utterance=str(raw.get("expected_utterance") or ""),
            notes=str(raw.get("notes") or ""),
            language=str(raw.get("language") or "ru"),
        )

    def validate(self) -> list[str]:
        """Проверяет кейс: границы, источник, категорию, типы.

        Это проверка **gold**, а не ответа модели: если gold сам не сходится с
        исходной строкой, метрики будут считать ошибки не там.
        """
        problems: list[str] = []
        if not self.id:
            problems.append("нет id")
        if self.category not in CATEGORIES:
            problems.append(f"неизвестная категория: {self.category}")
        if not self.target_text.strip():
            problems.append("пустой target_text")
        seen: set[tuple[int, int]] = set()
        for item in self.expected:
            if item.type not in ANNOTATION_TYPES:
                problems.append(f"неизвестный тип: {item.type}")
            if item.span_start < 0 or item.span_end > len(self.target_text):
                problems.append(f"границы вне текста: {item.span_start}:{item.span_end}")
                continue
            if item.span_end <= item.span_start:
                problems.append(f"пустой интервал: {item.span_start}:{item.span_end}")
                continue
            actual = self.target_text[item.span_start : item.span_end]
            if actual != item.source:
                problems.append(
                    f"источник не совпадает: ожидалось {actual!r}, указано {item.source!r}"
                )
            if (item.span_start, item.span_end) in seen:
                problems.append(f"повтор интервала: {item.span_start}:{item.span_end}")
            seen.add((item.span_start, item.span_end))
        if self.category == CATEGORY_NEGATIVE and self.expected:
            problems.append("негативный кейс с ожидаемыми аннотациями")
        if self.expected_utterance and self.expected_utterance not in UTTERANCE_CLASSES:
            problems.append(f"неизвестный класс реплики: {self.expected_utterance}")
        if not self.expected and not self.ambiguous and not self.expect_no_issue \
                and self.category != CATEGORY_NEGATIVE:
            problems.append("нет ожидаемых аннотаций и нет пометки ambiguous/expect_no_issue")
        return problems

    @property
    def replica_id(self) -> int:
        return _case_replica_id(self.id)

    def with_expected(self, expected: Iterable[Annotation]) -> DatasetCase:
        return replace(self, expected=tuple(expected))


def _case_replica_id(case_id: str) -> int:
    """Стабильный числовой id реплики из строкового id кейса.

    Модель получает числовой `replica_id` (как в проде), а проверка потом сверяет
    его с тем же кейсом — поэтому преобразование обязано быть детерминированным и
    не зависеть от порядка в файле.
    """
    digits = re.findall(r"(\d+)$", case_id or "")
    if digits:
        return int(digits[-1])
    return abs(hash(case_id)) % 10_000


def load_dataset(path: Path | str) -> list[DatasetCase]:
    """Читает JSONL-датасет: одна строка — один кейс."""
    cases: list[DatasetCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: не JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise TypeError(f"{path}:{number}: ожидался объект")
        cases.append(DatasetCase.from_dict(raw))
    return cases


def dataset_problems(cases: list[DatasetCase]) -> list[str]:
    """Проблемы всего датасета: повторы id, проблемы кейсов, покрытие категорий."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            problems.append(f"{case.id}: повтор id")
        seen_ids.add(case.id)
        problems.extend(f"{case.id}: {problem}" for problem in case.validate())
    if not cases:
        problems.append("датасет пуст")
    return problems


def category_counts(cases: Iterable[DatasetCase]) -> dict[str, int]:
    counts: dict[str, int] = {category: 0 for category in CATEGORIES}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts


def analysis_json_schema() -> dict:
    """JSON-схема ответа — её получает Ollama в `format` (§8).

    Схема описывает **только аннотации**: у модели нет поля, куда можно положить
    переписанный текст, и это главная защита от «свободного rewrite» на уровне
    самого протокола, а не только валидации.
    """
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "replica_id": {"type": "integer"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "span_start": {"type": "integer"},
                        "span_end": {"type": "integer"},
                        "source": {"type": "string"},
                        "type": {"type": "string", "enum": list(ANNOTATION_TYPES)},
                        "meaning": {"type": "string"},
                        "suggested_form": {"type": "string"},
                        "confidence": {"type": "number"},
                        "needs_review": {"type": "boolean"},
                        "reason_code": {"type": "string", "enum": list(REASON_CODES)},
                    },
                    "required": ["span_start", "span_end", "source", "type", "needs_review"],
                },
            },
            "utterance": {
                "type": "object",
                "properties": {
                    "class": {"type": "string", "enum": list(UTTERANCE_CLASSES)},
                    "context_dependency": {"type": "string", "enum": list(CONTEXT_DEPENDENCIES)},
                    "relevant_replica_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["class"],
            },
        },
        "required": ["schema_version", "replica_id", "items"],
    }
