"""Предобработка текста до модели: числа, сокращения и латиница.

Текст уходит в модель почти как есть — это осознанно: пунктуация и интонация
остаются авторскими. Но три вещи движки читают заведомо плохо, и обе модели
ошибаются на них одинаково, поэтому шаг общий и не зависит от выбора F5/XTTS:

* числа — «2026» модель читает как угодно, а «в 2026 году» требует ещё и
  порядкового числительного в нужном падеже;
* сокращения — «т.е.» и «руб.» с точками движок произносит по буквам;
* латиница — незнакомая аббревиатура читается как попало.

Транслитерация английских *слов* здесь намеренно не делается: XTTS v2
многоязычная и читает их сама, а практическая транскрипция только испортила бы
её произношение («speech» → «спич» вместо английского). Разворачиваются только
аббревиатуры и слова из небольшого словаря, где русское чтение однозначно.

Акцентуация (RUAccent) идёт после этого шага и по-прежнему только у F5 — знак
«+» XTTS не понимает.
"""

import logging
import re

from num2words import num2words

logger = logging.getLogger(__name__)

# --- числа --------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\d{1,3}(?:[ \u00a0]\d{3})+|\d+(?:[.,]\d+)?")
# «12:30» и «12.05.2026» числами не являются: развернуть их в слова — не то же
# самое, что прочитать, а по отдельности вышло бы «двенадцать:тридцать».
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")
_DOTTED_RE = re.compile(r"\d+(?:[.,]\d+){2,}")
# «007» — это код, а не число семь.
_LEADING_ZERO_RE = re.compile(r"0\d+")

# Год распознаём по следующему за числом слову: падеж порядкового числительного
# задаёт именно оно. «2026 год» → «две тысячи двадцать шестой год»,
# «в 2026 году» → «в две тысячи двадцать шестом году».
_YEAR_TAIL_RE = re.compile(r"\s*(год|году|года|годы|годов)\b", re.IGNORECASE)
_YEAR_CASE_BY_TAIL = {
    "год": "nom",
    "году": "prep",
    "года": "gen",
    "годы": "nom",
    "годов": "nom",
}
YEAR_RANGE = (1500, 2199)

# Окончания русских порядковых прилагательных по падежам. Меняем только
# последнее слово числительного: «две тысячи двадцать шестой» → «…шестом».
_ORDINAL_ENDINGS = {
    "nom": (("ый", "ый"), ("ой", "ой"), ("ий", "ий")),
    "gen": (("ый", "ого"), ("ой", "ого"), ("ий", "его")),
    "dat": (("ый", "ому"), ("ой", "ому"), ("ий", "ему")),
    "prep": (("ый", "ом"), ("ой", "ом"), ("ий", "ем")),
}

# --- сокращения ---------------------------------------------------------------
# Пары «шаблон → разворот». Идём сверху вниз, поэтому многоточечные
# сокращения стоят раньше одинарных: иначе «т.д.» распалось бы на «т» и «д».
_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    (r"т\.\s*е\.", "то есть"),
    (r"т\.\s*д\.", "так далее"),
    (r"т\.\s*п\.", "тому подобное"),
    (r"т\.\s*к\.", "так как"),
    (r"т\.\s*н\.", "так называемый"),
    (r"т\.\s*о\.", "таким образом"),
    (r"н\.\s*э\.", "нашей эры"),
    (r"г\.\s*г\.", "годы"),
    (r"им\.", "имени"),
    (r"напр\.", "например"),
    (r"рис\.", "рисунок"),
    (r"стр\.", "страница"),
    (r"др\.", "другие"),
    (r"руб\.", "рублей"),
    (r"коп\.", "копеек"),
    (r"тыс\.", "тысяч"),
    (r"чел\.", "человек"),
    (r"мин\.", "минут"),
    (r"сек\.", "секунд"),
    (r"кв\.", "квартира"),
    (r"ул\.", "улица"),
    (r"млн\b", "миллионов"),
    (r"млрд\b", "миллиардов"),
    (r"км\b", "километров"),
    (r"кг\b", "килограммов"),
    (r"мм\b", "миллиметров"),
    (r"см\b", "сантиметров"),
)
# Аббревиатуру ищем как отдельное слово, а не как подстроку: «др.» внутри
# «мудр.» — не сокращение, а «см» внутри «смех» — не сантиметры.
_ABBREVIATION_ANCHOR = r"(?<![0-9A-Za-zА-Яа-яЁё])"

# --- латиница -----------------------------------------------------------------
# Чтение по буквам: незнакомая аббревиатура из заглавных латинских букв
# произносится именно так, и это предсказуемо — в отличие от попытки угадать.
_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед",
}
_LATIN_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,5}(?![A-Za-z0-9])")
_LATIN_WORD_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*")

# Слова, которые в русской речи звучат не по буквам, а целиком. Ключи — в нижнем
# регистре; значение — то, как это слово реально произносят.
_LATIN_READINGS = {
    "api": "апи", "ai": "эй-ай", "gpt": "джи-пи-ти", "it": "ай-ти",
    "pdf": "пи-ди-эф", "os": "о-эс", "ui": "ю-ай", "ux": "ю-экс",
    "iphone": "айфон", "android": "андроид", "google": "гугл", "youtube": "ютуб",
    "python": "питон", "javascript": "джаваскрипт", "telegram": "телеграм",
    "whatsapp": "вотсап", "windows": "виндовс", "linux": "линукс", "docker": "докер",
}
# Короткое латинское слово — почти всегда аббревиатура («IT», «AI»); длинное
# без словарного чтения оставляем как есть, его прочитает многоязычная XTTS.
_ACRONYM_MAX_LEN = 5


def _read_acronym(token: str) -> str | None:
    """Читает латинскую аббревиатуру: словарём, иначе по буквам."""
    known = _LATIN_READINGS.get(token.lower())
    if known:
        return known
    if not token.isalpha() or len(token) > _ACRONYM_MAX_LEN:
        return None
    letters: list[str] = []
    for char in token:
        name = _LETTER_NAMES.get(char.lower())
        if name is None:
            return None
        letters.append(name)
    return "-".join(letters)


def _expand_latin(text: str) -> str:
    def replace_acronym(match: re.Match) -> str:
        return _read_acronym(match.group()) or match.group()

    def replace_word(match: re.Match) -> str:
        word = match.group()
        known = _LATIN_READINGS.get(word.lower())
        return known if known is not None else word

    return _LATIN_WORD_RE.sub(replace_word, _LATIN_ACRONYM_RE.sub(replace_acronym, text))


def _expand_abbreviations(text: str) -> str:
    """Разворачивает сокращения по списку сверху вниз.

    По одному шаблону за раз, а не общим регэкспом: замены в списке — это готовый
    текст («т.д.» → «так далее»), и общий алфавитный разбор пришлось бы держать
    отдельной таблицей соответствия, которую легко рассинхронизировать.
    """
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(_ABBREVIATION_ANCHOR + pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _decline_ordinal(value: int, case: str) -> str:
    """Порядковое числительное в нужном падеже (меняем последнее слово)."""
    words = num2words(value, lang="ru", to="ordinal").split()
    endings = _ORDINAL_ENDINGS.get(case, ())
    for suffix, replacement in endings:
        if words[-1].endswith(suffix):
            words[-1] = words[-1][: -len(suffix)] + replacement
            break
    return " ".join(words)


def _read_number(raw: str, match: re.Match, text: str) -> str:
    """Одно число словами: с учётом года, дроби и ведущих нулей."""
    if _LEADING_ZERO_RE.fullmatch(raw):
        return raw
    if "," in raw or "." in raw:
        return num2words(float(raw.replace(",", ".")), lang="ru")
    value = int(raw.replace(" ", "").replace("\u00a0", ""))
    tail = _YEAR_TAIL_RE.match(text, match.end())
    if tail and YEAR_RANGE[0] <= value <= YEAR_RANGE[1]:
        return _decline_ordinal(value, _YEAR_CASE_BY_TAIL.get(tail.group(1).lower(), "nom"))
    return num2words(value, lang="ru")


def _expand_numbers(text: str) -> str:
    protected = [match.span() for match in _TIME_RE.finditer(text)]
    protected += [match.span() for match in _DOTTED_RE.finditer(text)]

    def replace(match: re.Match) -> str:
        if any(start <= match.start() < end for start, end in protected):
            return match.group()
        return _read_number(match.group(), match, text)

    return _NUMBER_RE.sub(replace, text)


def normalize(text: str) -> str:
    """Готовит текст реплики к отправке в модель. Идемпотентна по смыслу:
    результат уже не содержит ни цифр, ни тех сокращений, что размечены выше.
    """
    if not text:
        return text
    result = _expand_abbreviations(text)
    result = _expand_numbers(result)
    result = _expand_latin(result)
    if result != text:
        logger.info("Текст до модели: «%s» → «%s»", text[:120], result[:120])
    return result
