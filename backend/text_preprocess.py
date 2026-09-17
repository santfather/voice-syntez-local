"""Шим обратной совместимости: нормализация переехала в `text_normalization`.

Файл остаётся рабочим намеренно. Во-первых, `audio_pipeline` и тесты
импортируют `normalize` именно отсюда, и менять их импорты ради красоты
структуры — это править работающий пайплайн без нужды. Во-вторых, старый путь
`backend.text_preprocess` — публичный для внешних скриптов и заметок; он должен
продолжать работать, даже когда реализация живёт в пакете.

Нового кода здесь быть не должно: любая логика — это шаг в
`backend/text_normalization/`, а этот модуль только ре-экспортирует имена.
"""

from .text_normalization import (
    YEAR_CASE_BY_TAIL,
    YEAR_RANGE,
    NormalizeStages,
    PronunciationRule,
    agree,
    agree_decimal,
    apply_pronunciation,
    cardinal,
    coverage_predicate,
    expand_numbers,
    expand_phones,
    expand_roman,
    normalize,
    normalize_report,
    normalize_stages,
    numeral_case,
    ordinal,
    plural_form,
    restore_yo,
)

__all__ = [
    "YEAR_CASE_BY_TAIL",
    "YEAR_RANGE",
    "NormalizeStages",
    "PronunciationRule",
    "agree",
    "agree_decimal",
    "apply_pronunciation",
    "cardinal",
    "coverage_predicate",
    "expand_numbers",
    "expand_phones",
    "expand_roman",
    "normalize",
    "normalize_report",
    "normalize_stages",
    "numeral_case",
    "ordinal",
    "plural_form",
    "restore_yo",
]
