"""Предобработка текста до модели: числа, сокращения и латиница."""

import re

from backend.text_preprocess import normalize


def test_year_is_read_as_ordinal_in_right_case():
    assert normalize("В 2026 году мы сделали 12 попыток.") == (
        "В две тысячи двадцать шестом году мы сделали двенадцать попыток."
    )
    assert normalize("2026 год начался хорошо.") == "две тысячи двадцать шестой год начался хорошо."


def test_numbers_spelled_out_but_codes_and_dates_kept():
    assert normalize("Пришло 3500 человек.") == "Пришло три тысячи пятьсот человек."
    assert normalize("Рост 1,5 раза.") == "Рост одна целая пять десятых раза."
    # Ведущие нули — код, а время и даты читаются не как отдельные числа.
    assert normalize("007") == "007"
    assert normalize("12:30") == "12:30"
    assert normalize("12.05.2026") == "12.05.2026"


def test_abbreviations_and_units_expand():
    assert normalize("т.е. вдвое больше") == "то есть вдвое больше"
    assert normalize("На улице 5 км и 300 руб.") == "На улице пять километров и триста рублей"
    # Сокращение ищется как слово: «см» внутри «смех» — не сантиметры.
    assert normalize("смех в зале") == "смех в зале"


def test_latin_acronyms_read_letter_by_letter_and_words_from_dictionary():
    assert normalize("Открой PDF и API.") == "Открой пи-ди-эф и апи."
    assert normalize("IT-отдел") == "ай-ти-отдел"
    # Английские слова не транслитерируются: их прочитает многоязычная XTTS.
    assert normalize("speech recognition") == "speech recognition"


def test_normalized_text_has_no_leftover_digits():
    text = "В 2026 году мы сделали 12 попыток и потратили 3500 рублей."
    assert re.search(r"\d", normalize(text)) is None


def test_empty_text_is_returned_as_is():
    assert normalize("") == ""
