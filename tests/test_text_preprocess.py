"""Предобработка текста до модели: числа, сокращения и латиница.

Таблицы «вход → ожидание» держат нормализацию как контракт: любая правка
правила сразу видна в диффе теста, а не только на слух в готовом файле.
Отдельным блоком идёт идемпотентность — повторный проход по результату не
должен ничего менять, иначе текст, попавший в нормализацию дважды, разъедется
с текстом, который нормализовали один раз.
"""

import re

import pytest

from backend.text_preprocess import normalize

# --- согласование числительного с существительным ------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 км", "один километр"),
        ("2 км", "два километра"),
        ("5 км", "пять километров"),
        ("11 км", "одиннадцать километров"),
        ("21 км", "двадцать один километр"),
        ("22 км", "двадцать два километра"),
        ("25 км", "двадцать пять километров"),
        ("101 км", "сто один километр"),
        ("111 км", "сто одиннадцать километров"),
        ("0 км", "ноль километров"),
    ],
)
def test_kilometres_agree(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 рубль", "один рубль"),
        ("2 рубля", "два рубля"),
        ("5 рублей", "пять рублей"),
        ("11 рублей", "одиннадцать рублей"),
        ("21 рубль", "двадцать один рубль"),
        ("22 рубля", "двадцать два рубля"),
        ("25 рублей", "двадцать пять рублей"),
    ],
)
def test_rubles_agree(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 м", "два метра"),
        ("5 м", "пять метров"),
        ("2 см", "два сантиметра"),
        ("5 см", "пять сантиметров"),
        ("2 мм", "два миллиметра"),
        ("5 мм", "пять миллиметров"),
        ("2 кг", "два килограмма"),
        ("5 кг", "пять килограммов"),
        ("2 г", "два грамма"),
        ("5 г", "пять граммов"),
        ("1 т", "одна тонна"),
        ("2 т", "две тонны"),
        ("5 т", "пять тонн"),
        ("1 тыс", "одна тысяча"),
        ("2 тыс", "две тысячи"),
        ("5 тыс", "пять тысяч"),
        ("1 млн", "один миллион"),
        ("2 млн", "два миллиона"),
        ("5 млн", "пять миллионов"),
        ("1 млрд", "один миллиард"),
        ("5 млрд", "пять миллиардов"),
        ("1 %", "один процент"),
        ("2 %", "два процента"),
        ("5 %", "пять процентов"),
        ("11 %", "одиннадцать процентов"),
        ("1 час", "один час"),
        ("2 ч", "два часа"),
        ("5 ч", "пять часов"),
        ("1 мин", "одна минута"),
        ("2 мин", "две минуты"),
        ("5 мин", "пять минут"),
        ("1 сек", "одна секунда"),
        ("2 сек", "две секунды"),
        ("5 сек", "пять секунд"),
        ("1 год", "один год"),
        ("2 года", "два года"),
        ("5 лет", "пять лет"),
        ("1 человек", "один человек"),
        ("2 человека", "два человека"),
        ("5 человек", "пять человек"),
        ("1 раз", "один раз"),
        ("2 раза", "два раза"),
        ("5 раз", "пять раз"),
        ("1 штука", "одна штука"),
        ("2 штуки", "две штуки"),
        ("5 штук", "пять штук"),
        ("1 л", "один литр"),
        ("5 л", "пять литров"),
        ("1 мл", "один миллилитр"),
        ("5 мл", "пять миллилитров"),
        ("1 байт", "один байт"),
        ("2 байта", "два байта"),
        ("5 байт", "пять байтов"),
        ("1 КБ", "один килобайт"),
        ("5 КБ", "пять килобайтов"),
        ("5 МБ", "пять мегабайтов"),
        ("5 ГБ", "пять гигабайтов"),
        ("5 ТБ", "пять терабайтов"),
        ("5 GB", "пять гигабайтов"),
    ],
)
def test_units_table_agrees(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_unknown_noun_is_not_inflected():
    # Единица не из таблицы: число читается, а существительное остаётся как было.
    assert normalize("2 яблока") == "два яблока"
    assert normalize("5 яблок") == "пять яблок"


# --- даты ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "12.05.2026",
            "двенадцатое мая две тысячи двадцать шестого года",
        ),
        (
            "12/05/2026",
            "двенадцатое мая две тысячи двадцать шестого года",
        ),
        ("01.01.2000", "первое января двухтысячного года"),
        (
            "31.12.1999",
            "тридцать первое декабря тысяча девятьсот девяносто девятого года",
        ),
        (
            "12 мая 2026 года",
            "двенадцатое мая две тысячи двадцать шестого года",
        ),
        ("12 мая", "двенадцатое мая"),
        ("9 мая 2025 года", "девятое мая две тысячи двадцать пятого года"),
    ],
)
def test_dates_are_spoken(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize("text", ["32.13.2026", "31.02.2026"])
def test_invalid_dates_are_left_alone(text: str) -> None:
    assert normalize(text) == text


def test_year_with_g_marker_is_ordinal():
    assert normalize("1917 г.") == "тысяча девятьсот семнадцатый год"


# --- время ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12:30", "двенадцать часов тридцать минут"),
        ("12:00", "двенадцать часов"),
        ("00:05", "ноль часов пять минут"),
        ("01:01", "один час одна минута"),
        ("23:59", "двадцать три часа пятьдесят девять минут"),
        ("в 9:00", "в девять часов"),
    ],
)
def test_time_is_read_as_clock(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_time_is_not_split_into_two_numbers():
    result = normalize("Встреча в 12:30.")
    assert result == "Встреча в двенадцать часов тридцать минут."
    assert re.search(r"\d", result) is None


def test_invalid_time_is_left_alone():
    assert normalize("25:99") == "25:99"


# --- деньги --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("25 руб.", "двадцать пять рублей"),
        ("25 ₽", "двадцать пять рублей"),
        ("1 копейка", "одна копейка"),
        ("2 копейки", "две копейки"),
        ("5 коп.", "пять копеек"),
        ("$5", "пять долларов"),
        ("5$", "пять долларов"),
        ("€5", "пять евро"),
        ("5 €", "пять евро"),
        ("2,5 рубля", "две целых пять десятых рубля"),
    ],
)
def test_money_is_spoken_with_agreement(text: str, expected: str) -> None:
    assert normalize(text) == expected


# --- проценты и дроби ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5%", "пять процентов"),
        ("1%", "один процент"),
        ("2%", "два процента"),
        ("11%", "одиннадцать процентов"),
        ("100%", "сто процентов"),
        ("2,5%", "две целых пять десятых процента"),
    ],
)
def test_percent_agrees(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,5", "одна целая пять десятых"),
        ("1.5", "одна целая пять десятых"),
        ("0,5", "ноль целых пять десятых"),
        ("3,14", "три целых четырнадцать сотых"),
    ],
)
def test_fractions_keep_num2words_behaviour(text: str, expected: str) -> None:
    assert normalize(text) == expected


# --- диапазоны -----------------------------------------------------------------


@pytest.mark.parametrize("text", ["5–10 км", "5-10 км", "5 — 10 км"])
def test_range_dash_variants_read_the_same(text: str) -> None:
    assert normalize(text) == "от пяти до десяти километров"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1–2 км", "от одного до двух километров"),
        ("5–10%", "от пяти до десяти процентов"),
        ("5–10", "от пяти до десяти"),
        (
            "2020–2025 годы",
            "от две тысячи двадцатого до две тысячи двадцать пятого года",
        ),
    ],
)
def test_ranges_use_from_to_policy(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_hyphenated_digits_are_not_a_range():
    # «123-45-67» — фрагмент номера, а не диапазон: разрезать его нельзя.
    assert "от " not in normalize("123-45-67")


# --- римские числительные и знак номера ----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("XXI век", "двадцать первый век"),
        ("XIX века", "девятнадцатого века"),
        ("в XXI веке", "в двадцать первом веке"),
        ("том II", "том второй"),
    ],
)
def test_roman_numerals_become_ordinals(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("№ 15", "номер пятнадцать"),
        ("№15", "номер пятнадцать"),
    ],
)
def test_number_sign_means_nomer(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_hash_is_not_number_sign():
    assert normalize("# 15") == "# пятнадцать"


# --- технические конструкции ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/a/1",
        "www.example.ru",
        "user@example.ru",
        "1.2.3",
        "192.168.1.1",
        "версия 2.0.1",
    ],
)
def test_urls_emails_versions_and_ips_are_protected(text: str) -> None:
    assert normalize(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "+7 999 123-45-67",
            "семь-девять-девять-девять-один-два-три-четыре-пять-шесть-семь",
        ),
        (
            "8 (999) 123-45-67",
            "восемь-девять-девять-девять-один-два-три-четыре-пять-шесть-семь",
        ),
    ],
)
def test_phones_are_read_digit_by_digit(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_ten_digit_number_is_not_a_phone():
    # Без кода страны десятизначный набор — это сумма, а не номер телефона.
    assert normalize("5 000 000 000") == "пять миллиардов"


# --- сокращения, латиница, коды ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("т.е. вдвое больше", "то есть вдвое больше"),
        ("т.д. и т.п.", "так далее и тому подобное"),
        ("ул. Ленина", "улица Ленина"),
        ("5 тыс. рублей", "пять тысяч рублей"),
        ("см", "сантиметров"),
        ("смех в зале", "смех в зале"),
        ("300 руб.", "триста рублей"),
    ],
)
def test_abbreviations_expand_as_whole_words(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Открой PDF и API.", "Открой пи-ди-эф и апи."),
        ("IT-отдел", "ай-ти-отдел"),
        ("Python и speech recognition", "питон и speech recognition"),
    ],
)
def test_latin_acronyms_and_words(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("007", "007"),
        ("Пин-код 007", "Пин-код 007"),
        ("0", "ноль"),
    ],
)
def test_leading_zero_codes_are_kept(text: str, expected: str) -> None:
    assert normalize(text) == expected


# --- общие требования ----------------------------------------------------------


def test_year_is_read_as_ordinal_in_right_case():
    assert normalize("В 2026 году мы сделали 12 попыток.") == (
        "В две тысячи двадцать шестом году мы сделали двенадцать попыток."
    )
    assert normalize("2026 год начался хорошо.") == "две тысячи двадцать шестой год начался хорошо."


def test_numbers_spelled_out_but_codes_kept():
    assert normalize("Пришло 3500 человек.") == "Пришло три тысячи пятьсот человек."
    assert normalize("Рост 1,5 раза.") == "Рост одна целая пять десятых раза."
    # Ведущие нули — код, а не число.
    assert normalize("007") == "007"


def test_normalized_text_has_no_leftover_digits():
    text = "В 2026 году мы сделали 12 попыток и потратили 3500 рублей."
    assert re.search(r"\d", normalize(text)) is None


def test_expected_phrase_agrees():
    assert normalize("Я прошёл 2 км и заплатил 25 руб.") == (
        "Я прошёл два километра и заплатил двадцать пять рублей"
    )


def test_empty_text_is_returned_as_is():
    assert normalize("") == ""


def test_pronunciation_point_is_transparent():
    from backend.text_preprocess import apply_pronunciation

    assert apply_pronunciation("привет") == "привет"
    assert apply_pronunciation("привет", []) == "привет"
    assert apply_pronunciation("привет", {"привет": "привет!"}) == "привет!"


# --- идемпотентность -----------------------------------------------------------

IDEMPOTENT_SAMPLES = [
    "Я прошёл 2 км и заплатил 25 руб.",
    "5 км",
    "21 рубль",
    "25 рублей",
    "2,5%",
    "12.05.2026",
    "12/05/2026",
    "12 мая 2026 года",
    "32.13.2026",
    "1917 г.",
    "12:30",
    "00:05",
    "25:99",
    "5–10 км",
    "5 — 10 км",
    "2020–2025 годы",
    "XXI век",
    "XIX века",
    "№ 15",
    "007",
    "1,5",
    "3500 человек.",
    "https://example.com/a/1",
    "user@example.ru",
    "1.2.3",
    "192.168.1.1",
    "+7 999 123-45-67",
    "8 (999) 123-45-67",
    "$5",
    "€5",
    "5 тыс. рублей",
    "2 яблока",
    "т.е. вдвое больше",
    "IT-отдел",
    "Python и speech recognition",
    "В 2026 году мы сделали 12 попыток.",
    "5 000 000 000",
]


@pytest.mark.parametrize("text", IDEMPOTENT_SAMPLES)
def test_normalize_is_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once
