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
    agree,
    agree_decimal,
    apply_pronunciation,
    cardinal,
    expand_numbers,
    expand_phones,
    expand_roman,
    normalize,
    numeral_case,
    ordinal,
    plural_form,
)

__all__ = [
    "YEAR_CASE_BY_TAIL",
    "YEAR_RANGE",
    "agree",
    "agree_decimal",
    "apply_pronunciation",
    "cardinal",
    "expand_numbers",
    "expand_phones",
    "expand_roman",
    "normalize",
    "numeral_case",
    "ordinal",
    "plural_form",
]
