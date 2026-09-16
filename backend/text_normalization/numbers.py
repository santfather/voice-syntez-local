"""Числа словами, телефоны и римские числительные.

Это последний шаг, который читает цифры: к моменту его вызова даты, время,
деньги, единицы и диапазоны уже развёрнуты, а технические конструкции
(URL, email, версии, IP) спрятаны в плейсхолдеры. Поэтому здесь остаётся
честное правило «любая оставшаяся цифра читается словами» — и оно же
гарантирует, что в выходном тексте не остаётся случайных цифр.

Три исключения осознанные:

* ведущие нули («007») — это код, а не число семь;
* год распознаётся по следующему слову и читается порядковым в его падеже
  («в 2026 году» → «в две тысячи двадцать шестом году»);
* телефон — не число, а последовательность цифр, и читается по одной.
"""

import re

from .morphology import cardinal, ordinal

# Число целиком: разряды через пробел («3 500») или дробь с одним разделителем.
NUMBER_PATTERN = r"\d{1,3}(?:[ \u00a0]\d{3})+|\d+(?:[.,]\d+)?"
NUMBER_RE = re.compile(NUMBER_PATTERN)
# «007» — это код, а не число семь.
LEADING_ZERO_RE = re.compile(r"0\d+")

# Год распознаём по следующему за числом слову: падеж порядкового числительного
# задаёт именно оно. «2026 год» → «две тысячи двадцать шестой год»,
# «в 2026 году» → «в две тысячи двадцать шестом году».
YEAR_TAIL_RE = re.compile(r"\s*(год|году|года|годы|годов)\b", re.IGNORECASE)
YEAR_CASE_BY_TAIL = {
    "год": "nom",
    "году": "prep",
    "года": "gen",
    "годы": "nom",
    "годов": "nom",
}
YEAR_RANGE = (1500, 2199)

# Телефон — не число: «+7 999 123-45-67» как количество было бы суммой в десятки
# миллиардов. Политика — читать цифра за цифрой через дефис.
_PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s\u00a0()\-]{7,}\d")
_DIGIT_NAMES = (
    "ноль", "один", "два", "три", "четыре",
    "пять", "шесть", "семь", "восемь", "девять",
)

# Римские числительные. Только заглавные и только от двух знаков: одиночная «I»
# в английской фразе — это местоимение, а не единица. Дефис рядом исключён,
# чтобы «CD-ROM» не превратился в «четырёхсотый-ROM».
_ROMAN_RE = re.compile(r"(?<![A-Za-z\-])([IVXLCDM]{2,})(?![A-Za-z\-])")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_TAIL_RE = re.compile(r"\s*(век|века|веку|веке|веков)\b", re.IGNORECASE)
_ROMAN_CASE_BY_TAIL = {
    "век": "nom",
    "века": "gen",
    "веку": "dat",
    "веке": "prep",
    "веков": "gen",
}


def is_year(value: int) -> bool:
    """Похоже ли число на год: диапазон, в котором годы вообще встречаются."""
    return YEAR_RANGE[0] <= value <= YEAR_RANGE[1]


def number_value(raw: str) -> int | float:
    """Значение записанного числа: пробелы — разряды, запятая и точка — дробь."""
    if "," in raw or "." in raw:
        return float(raw.replace(",", "."))
    return int(raw.replace(" ", "").replace("\u00a0", ""))


def expand_numbers(text: str) -> str:
    """Читает словами все оставшиеся числа: год, дробь или код — по контексту."""

    def replace(match: re.Match) -> str:
        raw = match.group()
        if LEADING_ZERO_RE.fullmatch(raw):
            return raw
        if "," in raw or "." in raw:
            return cardinal(float(raw.replace(",", ".")))
        value = int(raw.replace(" ", "").replace("\u00a0", ""))
        tail = YEAR_TAIL_RE.match(text, match.end())
        if tail and is_year(value):
            return ordinal(value, YEAR_CASE_BY_TAIL.get(tail.group(1).lower(), "nom"))
        return cardinal(value)

    return NUMBER_RE.sub(replace, text)


def _looks_like_phone(raw: str) -> bool:
    """Отличает телефон от числа: длина, код страны и разделители.

    Без кода (7/8/+…) десятизначный набор цифр остаётся числом: «5 000 000 000»
    — это сумма, а не номер. Разделители обязательны, иначе телефоном считаем
    только одиннадцатизначный набор с кодом — так обычно и пишут в мессенджерах.
    """
    digits = re.sub(r"\D", "", raw)
    if not 10 <= len(digits) <= 15:
        return False
    if not (raw.startswith("+") or digits[0] in "78"):
        return False
    return bool(re.search(r"[\s()\-]", raw.strip())) or len(digits) == 11


def expand_phones(text: str) -> str:
    """Телефоны читаются цифра за цифрой: «восемь-девять-девять-…»."""

    def replace(match: re.Match) -> str:
        raw = match.group()
        if not _looks_like_phone(raw):
            return raw
        digits = re.sub(r"\D", "", raw)
        return "-".join(_DIGIT_NAMES[int(digit)] for digit in digits)

    return _PHONE_CANDIDATE_RE.sub(replace, text)


def _to_roman(value: int) -> str:
    """Обратная запись: нужна, чтобы отсеять нечисла вроде «IIX» и «IIII»."""
    parts: list[str] = []
    for number, sign in (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ):
        while value >= number:
            parts.append(sign)
            value -= number
    return "".join(parts)


def _roman_value(token: str) -> int | None:
    """Строгое значение римского числа; None — если токен им не является."""
    total = 0
    previous = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    if not 0 < total < 4000 or _to_roman(total) != token:
        return None
    return total


def expand_roman(text: str) -> str:
    """Римские числительные — порядковым словом; «век» задаёт падеж.

    Вызывается до латиницы: иначе «XXI» ушло бы в чтение по буквам. Без слова
    «век» рядом падеж взять неоткуда, поэтому читаем именительным мужского рода
    («том II» → «том второй»).
    """

    def replace(match: re.Match) -> str:
        value = _roman_value(match.group())
        if value is None:
            return match.group()
        tail = _ROMAN_TAIL_RE.match(text, match.end())
        case = _ROMAN_CASE_BY_TAIL.get(tail.group(1).lower(), "nom") if tail else "nom"
        return ordinal(value, case)

    return _ROMAN_RE.sub(replace, text)
