"""Латиница: аббревиатуры по буквам или словарём, английские слова — как есть.

Транслитерация английских *слов* здесь намеренно не делается: XTTS v2
многоязычная и читает их сама, а практическая транскрипция только испортила бы
её произношение («speech» → «спич» вместо английского). Разворачиваются только
аббревиатуры и слова из небольшого словаря, где русское чтение однозначно.

Шаг идёт последним из текстовых: к этому моменту римские числительные уже
прочитаны словами, и «XXI» не попадёт в чтение по буквам.
"""

import re

# Чтение по буквам: незнакомая аббревиатура из заглавных латинских букв
# произносится именно так, и это предсказуемо — в отличие от попытки угадать.
LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед",
}
LATIN_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,5}(?![A-Za-z0-9])")
LATIN_WORD_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*")

# Слова, которые в русской речи звучат не по буквам, а целиком. Ключи — в нижнем
# регистре; значение — то, как это слово реально произносят.
LATIN_READINGS = {
    "api": "апи", "ai": "эй-ай", "gpt": "джи-пи-ти", "it": "ай-ти",
    "pdf": "пи-ди-эф", "os": "о-эс", "ui": "ю-ай", "ux": "ю-экс",
    "iphone": "айфон", "android": "андроид", "google": "гугл", "youtube": "ютуб",
    "python": "питон", "javascript": "джаваскрипт", "telegram": "телеграм",
    "whatsapp": "вотсап", "windows": "виндовс", "linux": "линукс", "docker": "докер",
}
# Короткое латинское слово — почти всегда аббревиатура («IT», «AI»); длинное
# без словарного чтения оставляем как есть, его прочитает многоязычная XTTS.
ACRONYM_MAX_LEN = 5


def _read_acronym(token: str) -> str | None:
    """Читает латинскую аббревиатуру: словарём, иначе по буквам."""
    known = LATIN_READINGS.get(token.lower())
    if known:
        return known
    if not token.isalpha() or len(token) > ACRONYM_MAX_LEN:
        return None
    letters: list[str] = []
    for char in token:
        name = LETTER_NAMES.get(char.lower())
        if name is None:
            return None
        letters.append(name)
    return "-".join(letters)


def expand_latin(text: str) -> str:
    """Разворачивает латинские аббревиатуры, слова без словаря оставляет."""

    def replace_acronym(match: re.Match) -> str:
        return _read_acronym(match.group()) or match.group()

    def replace_word(match: re.Match) -> str:
        word = match.group()
        known = LATIN_READINGS.get(word.lower())
        return known if known is not None else word

    return LATIN_WORD_RE.sub(replace_word, LATIN_ACRONYM_RE.sub(replace_acronym, text))
