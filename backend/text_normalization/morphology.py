"""Согласование числительного с существительным — правилами, без морфоразбора.

pymorphy здесь нет и не будет: это ещё одна тяжёлая зависимость в offline-first
runtime ради одной задачи. Русская форма после числа определяется последними
двумя цифрами, и этого правила достаточно для единиц, денег и времени:

* 1, 21, 31 … — единственное число, именительный («один километр»);
* 2–4, 22–24 … — единственное число, родительный («два километра»);
* 0, 5–20, 11–14, 25–30 … — множественное число, родительный («пять километров»).

Отдельно стоит случай 11–14: он ломает обычное правило по последней цифре
(«одиннадцать километров», а не «одиннадцать километра»), поэтому проверяется
раньше. Род нужен только для «один/одна» и «два/две» — остальные числительные
от рода не зависят.

Нерегулярные формы («год/года/лет», «человек/человека/человек») правилом не
выводятся: три формы хранятся рядом с единицей в `units.py` и `money.py`.
"""

from num2words import num2words

# Порядок форм один для всех таблиц проекта: «один / два / пять».
FORM_ONE = 0
FORM_FEW = 1
FORM_MANY = 2

Forms = tuple[str, str, str]

_CARDINAL_CASES = {
    "nom": "n",
    "gen": "g",
    "dat": "d",
    "acc": "a",
    "ins": "i",
    "prep": "p",
}

# Окончания русских порядковых прилагательных. Меняем только последнее слово
# числительного: «две тысячи двадцать шестой» → «…шестого». Ветка «neut» — это
# даты: «двенадцатый» → «двенадцатое мая».
_ORDINAL_ENDINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "nom": (("ый", "ый"), ("ой", "ой"), ("ий", "ий")),
    "neut": (("ый", "ое"), ("ой", "ое"), ("ий", "ее")),
    "gen": (("ый", "ого"), ("ой", "ого"), ("ий", "его")),
    "dat": (("ый", "ому"), ("ой", "ому"), ("ий", "ему")),
    "prep": (("ый", "ом"), ("ой", "ом"), ("ий", "ем")),
}


def plural_form(value: int, forms: Forms) -> str:
    """Форма существительного по последним двум цифрам числа."""
    last_two = abs(int(value)) % 100
    if 11 <= last_two <= 14:
        return forms[FORM_MANY]
    last = last_two % 10
    if last == 1:
        return forms[FORM_ONE]
    if 2 <= last <= 4:
        return forms[FORM_FEW]
    return forms[FORM_MANY]


def cardinal(value: float, case: str = "nom", gender: str = "m") -> str:
    """Количественное числительное словами в нужном падеже и роде.

    Целые значения приходят сюда так же часто, как дробные: падеж управляет
    только ими, а у дробей своя конструкция («две целых пять десятых»), и
    родительный к ней не применим.
    """
    if isinstance(value, float):
        return num2words(value, lang="ru", gender=gender)
    return num2words(value, lang="ru", case=_CARDINAL_CASES.get(case, "n"), gender=gender)


def ordinal(value: int, case: str = "nom") -> str:
    """Порядковое числительное; падеж задаёт слово после числа («в … году»)."""
    words = num2words(int(value), lang="ru", to="ordinal").split()
    endings = _ORDINAL_ENDINGS.get(case, ())
    for suffix, replacement in endings:
        if words[-1].endswith(suffix):
            words[-1] = words[-1][: -len(suffix)] + replacement
            break
    return " ".join(words)


def agree(value: int, forms: Forms, gender: str = "m", case: str = "nom") -> str:
    """Числительное вместе с согласованным существительным: «два километра»."""
    return f"{cardinal(value, case=case, gender=gender)} {plural_form(value, forms)}"


def agree_decimal(value: float, forms: Forms, gender: str = "m") -> str:
    """Дробь вместе с существительным в родительном единственного.

    «Две целых пять десятых процента»: после дробной части существительное
    всегда стоит в родительном единственного, независимо от величины.
    """
    return f"{cardinal(value, gender=gender)} {forms[FORM_FEW]}"


def numeral_case(value: float, case: str = "gen") -> str:
    """Числительное в падеже без существительного — для диапазонов.

    «От пяти до десяти»: обе границы диапазона читаются в родительном, а форма
    существительного при них берётся множественная.
    """
    return cardinal(value, case=case)
