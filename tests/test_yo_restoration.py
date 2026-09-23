"""Восстановление «ё»: гарантированные замены, регистр, границы и fallback.

Главное, что здесь проверяется, — узость шага. Автоматика обязана менять только
бесспорные слова и не имеет права трогать ни неоднозначные омографы («все/всё»),
ни ручную разметку ударения, ни технические строки, ни то, что уже покрыто
правилом словаря. Данные лежат в репозитории (`data/yo_safe.tsv.gz`), поэтому
тесты не ходят в сеть и не поднимают моделей.
"""

import gzip
import unicodedata

import pytest

from backend.text_normalization import (
    normalize,
    normalize_stages,
    restore_yo,
    yo_restoration,
)
from backend.text_normalization.pronunciation import PronunciationRule


@pytest.fixture(autouse=True)
def fresh_dictionaries():
    """Ленивые словари сбрасываются до и после теста: подмена пути не протекает."""
    yo_restoration.reset_cache()
    yield
    yo_restoration.reset_cache()


# --- гарантированные замены ---------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ежик", "ёжик"),
        ("шел", "шёл"),
        ("зеленую", "зелёную"),
        ("еще", "ещё"),
        ("актер", "актёр"),
        ("Мы пошли в парк, где живет ежик.", "Мы пошли в парк, где живёт ёжик."),
    ],
)
def test_guaranteed_words_are_restored(text: str, expected: str) -> None:
    assert restore_yo(text) == expected


def test_step_is_part_of_normalize():
    # Шаг «ё» — часть общего пайплайна, а не отдельная функция «для галочки».
    assert normalize("ежик и SQL") == "ёжик и эс-кью-эл"


# --- регистр ------------------------------------------------------------------
def test_case_is_preserved():
    # Регистр переносится на замену: строчное, заглавное и капсом — по-разному.
    assert restore_yo("Ежик ЕЖИК ежик") == "Ёжик ЁЖИК ёжик"
    assert restore_yo("Еще ЕЩЕ еще") == "Ещё ЕЩЁ ещё"


def test_ambiguous_word_is_not_restored_in_any_case():
    # «все» — е/ё-омограф (всё), а не гарантированная замена: автоматика молчит
    # и в нижнем регистре, и капсом. Разрешает такой случай только словарь.
    assert restore_yo("Все ВСЕ все") == "Все ВСЕ все"


# --- идемпотентность ----------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "ежик шел зеленую еще актер",
        "Ежик и ЕЖИК",
        "все небо пчелы",
        "шел+ит ежик",
        "https://example.com/ежик?шел=1",
        "Ни одного слова с буквой е",
    ],
)
def test_restore_yo_is_idempotent(text: str) -> None:
    once = restore_yo(text)
    assert restore_yo(once) == once


def test_normalize_stays_idempotent_with_yo():
    once = normalize("В 2026 году ежик прошел 2 км.")
    assert normalize(once) == once
    assert "ёжик" in once and "прошёл" in once


# --- неоднозначное не меняется автоматически ----------------------------------
@pytest.mark.parametrize("text", ["все", "небо", "пчелы", "Мы все видим небо."])
def test_ambiguous_words_are_left_alone(text: str) -> None:
    assert restore_yo(text) == text


# --- ручная разметка ударения -------------------------------------------------
def test_word_with_manual_stress_is_not_touched():
    assert restore_yo("шел+ит") == "шел+ит"
    assert restore_yo("пош+ел") == "пош+ел"
    # Соседнее слово без «+» при этом восстанавливается как обычно.
    assert restore_yo("шел+ит ежик") == "шел+ит ёжик"


# --- технические строки, латиница, цифры --------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/a/1",
        "www.example.ru",
        "user@example.ru",
        "1.2.3",
        "192.168.1.1",
        "Python и speech recognition",
        "ежик2",
        "2ежик",
        "version 1.0",
    ],
)
def test_technical_and_latin_text_is_not_touched(text: str) -> None:
    assert restore_yo(text) == text


def test_text_without_yo_candidate_is_returned_as_is():
    text = "Привет, мир! Тут нет ни одного длинного слова."
    assert restore_yo(text) == text


def test_text_without_letter_e_is_returned_as_is():
    # Ни «е», ни «Е» — восстанавливать нечего, текст не трогается вовсе.
    text = "Мой друг шёл домой. ЁЖИК"
    assert restore_yo(text) == text


# --- приоритет правила словаря ------------------------------------------------
def test_word_covered_by_enabled_rule_is_not_restored():
    rules = [PronunciationRule(source="ежик", target="еж-ик")]
    assert normalize("ежик", rules) == "еж-ик"
    # Без правила автоматика своё берёт.
    assert normalize("ежик") == "ёжик"


def test_skip_predicate_blocks_restoration():
    assert restore_yo("ежик и шел", skip=lambda word: word == "ежик") == "ежик и шёл"


# --- стадии preview -----------------------------------------------------------
def test_normalize_stages_exposes_yo_between_normalization_and_dictionary():
    stages = normalize_stages("ежик и SQL", [PronunciationRule("SQL", "эскьюэль")])
    assert stages.normalized == "ежик и эс-кью-эл"
    assert stages.yo == "ёжик и эс-кью-эл"
    assert stages.result == "ёжик и эскьюэль"
    assert stages.matches == [{"source": "SQL", "target": "эскьюэль", "count": 1}]


# --- юникод-форма входа -------------------------------------------------------
def test_decomposed_yo_is_composed_on_pipeline_input():
    """«е» + U+0308 из чужого файла приводится к «ё» ещё до шага «ё»."""
    decomposed = unicodedata.normalize("NFD", "ёжик и ещё")
    assert decomposed != "ёжик и ещё"  # вход действительно декомпозирован
    assert normalize(decomposed) == "ёжик и ещё"


def test_restoration_of_composed_e_works_on_decomposed_input():
    """Реплика с декомпозированной «ё» проходит пайплайн так же, как композированная.

    Здесь в одной строке и «Ёлка» с диакритикой из чужого источника, и «ежик» из
    однозначного словаря: NFC обязан сложить первую, не помешав второй.
    """
    text = "Ёлка, ежик и чёрный кот"
    decomposed = unicodedata.normalize("NFD", text)
    assert len(decomposed) > len(text)
    assert normalize(decomposed) == normalize(text) == "Ёлка, ёжик и чёрный кот"


# --- fallback при отсутствии/поломке данных -----------------------------------
def test_missing_data_file_falls_back_without_breaking(monkeypatch, tmp_path):
    monkeypatch.setattr(yo_restoration, "SAFE_PATH", tmp_path / "нет-такого.tsv.gz")
    yo_restoration.reset_cache()
    # Запасной список покрывает частые слова — синтез не ломается.
    assert restore_yo("ежик шел еще") == "ёжик шёл ещё"
    assert normalize("ежик") == "ёжик"


def test_broken_data_file_falls_back(monkeypatch, tmp_path):
    broken = tmp_path / "yo_safe.tsv.gz"
    broken.write_bytes("это не gzip".encode())
    monkeypatch.setattr(yo_restoration, "SAFE_PATH", broken)
    yo_restoration.reset_cache()
    assert restore_yo("ежик") == "ёжик"


def test_fallback_data_is_sane():
    # Пары запасного списка не должны противоречить основному словарю: иначе при
    # отсутствии файла поведение отличалось бы от обычного.
    for target in yo_restoration.FALLBACK.values():
        assert "ё" in target
    assert yo_restoration.safe_dictionary().get("ежик") == "ёжик"


def test_shipped_dictionary_matches_documented_size():
    """Словарь из репозитория читается целиком и не повреждён."""
    with gzip.open(yo_restoration.SAFE_PATH, "rt", encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    assert count == 106_298
    # Неоднозначные пары — отдельный файл, и «все» лежит именно там.
    assert yo_restoration.ambiguous_dictionary().get("все") == "всё"
    assert yo_restoration.safe_dictionary().get("все") is None
