"""Деньги: символы и словесные обозначения валют.

Валюта — такое же существительное после числа, как единица измерения, и
согласуется по тем же правилам: «двадцать пять рублей», «двадцать два рубля»,
«один рубль». Разница только в том, что знак валюты может стоять и до числа
(«$5»), поэтому у модуля два шаблона, а не один.

Евро — исключение: во всех формах «евро» (несклоняемое), поэтому у него три
одинаковые формы.

Дробное количество тоже осмысленно: «2,5 рубля» → «две целых пять десятых
рубля» — родительный единственного, как и у любой дроби.
"""

import re

from .morphology import agree, agree_decimal
from .numbers import NUMBER_PATTERN, number_value

Forms = tuple[str, str, str]
RUBLE: Forms = ("рубль", "рубля", "рублей")
KOPECK: Forms = ("копейка", "копейки", "копеек")
DOLLAR: Forms = ("доллар", "доллара", "долларов")
EURO: Forms = ("евро", "евро", "евро")

# Вариант записи → (формы, род). Точка в «руб.» и «коп.» — часть сокращения,
# поэтому она входит в вариант и не остаётся в готовом тексте.
_MONEY_BY_ALIAS: dict[str, tuple[Forms, str]] = {}
for _aliases, _forms, _gender in (
    (("₽", "руб.", "рублей", "рубля", "рубль", "руб", "р."), RUBLE, "m"),
    (("коп.", "копеек", "копейки", "копейка", "коп"), KOPECK, "f"),
    (("$", "долларов", "доллара", "доллар"), DOLLAR, "m"),
    (("€", "евро"), EURO, "n"),
):
    for _alias in _aliases:
        _MONEY_BY_ALIAS[_alias] = (_forms, _gender)

_MONEY_SUFFIX_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё])({NUMBER_PATTERN})\s*"
    rf"(₽|\$|€|рублей|рубля|рубль|руб\.?|руб|р\.|копеек|копейки|копейка|коп\.?|коп"
    rf"|долларов|доллара|доллар|евро)(?![A-Za-zА-Яа-яЁё])"
)
_MONEY_PREFIX_RE = re.compile(rf"([$€₽])\s*({NUMBER_PATTERN})")


def _phrase(raw: str, forms: Forms, gender: str) -> str:
    value = number_value(raw)
    if isinstance(value, float):
        return agree_decimal(value, forms, gender=gender)
    return agree(value, forms, gender=gender)


def expand_money(text: str) -> str:
    """Разворачивает «25 руб.», «5$» и «€3» в согласованную сумму словами."""

    def replace_prefix(match: re.Match) -> str:
        forms, gender = _MONEY_BY_ALIAS[match.group(1)]
        return _phrase(match.group(2), forms, gender)

    def replace_suffix(match: re.Match) -> str:
        forms, gender = _MONEY_BY_ALIAS[match.group(2)]
        return _phrase(match.group(1), forms, gender)

    text = _MONEY_PREFIX_RE.sub(replace_prefix, text)
    return _MONEY_SUFFIX_RE.sub(replace_suffix, text)
