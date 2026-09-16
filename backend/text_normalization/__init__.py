"""Нормализация русского текста до модели: числа, даты, единицы, латиница.

Пакет разбит по ответственностям, а порядок шагов живёт в `pipeline.py`.
Снаружи нужен почти всегда только `normalize`; остальные имена экспортируются
для тестов и для будущих фаз (словарь произношения, редактор правил).
"""

from .morphology import (
    agree,
    agree_decimal,
    cardinal,
    numeral_case,
    ordinal,
    plural_form,
)
from .numbers import (
    YEAR_CASE_BY_TAIL,
    YEAR_RANGE,
    expand_numbers,
    expand_phones,
    expand_roman,
)
from .pipeline import normalize
from .pronunciation import apply_pronunciation

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
