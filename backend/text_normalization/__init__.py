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
from .pipeline import NormalizeStages, normalize, normalize_report, normalize_stages
from .pronunciation import PronunciationRule, apply_pronunciation, coverage_predicate
from .yo_restoration import restore_yo

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
