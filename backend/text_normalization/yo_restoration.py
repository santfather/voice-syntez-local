"""Восстановление буквы «ё» по словарю гарантированных случаев.

Русская типографика почти всегда пишет «е» там, где читается «ё» («ежик», «шел»,
«черный»). Для модели это не орфографическая мелочь: «е» и «ё» — разные символы,
и метка ударения саму букву не подменяет. Поэтому в тексте с «е» модель произносит
«е» даже под ударением.

Шаг здесь детерминированный и узкий: меняются **только бесспорные** случаи, где
«е» на месте «ё» — исключительно издательская привычка, а не другое слово. Данные —
`data/yo_safe.tsv.gz` (словарь проекта `eyo-kernel`, MIT; уведомление лицензии
лежит рядом в `data/LICENSE.eyo-kernel.txt`). Неоднозначные пары («все/всё»,
«небо/нёбо», «пчелы/пчёлы») лежат отдельно (`yo_ambiguous.tsv.gz`) и здесь не
применяются вообще: их разрешает пользователь через словарь произношения
(см. `backend/pronunciation_suggest.py`).

Место шага в пайплайне — после чисел и сокращений (там уже готовые слова) и до
словаря произношения: явное правило пользователя обязано оставаться приоритетнее
автоматики. Поэтому у функции есть `skip` — предикат «слово покрыто правилом
словаря», и такие слова шаг не трогает.

Словарь грузится лениво, один раз на процесс, под блокировкой. Файла данных может
не быть (урезанная сборка) — тогда работает небольшой встроенный список частых
слов, а синтез продолжается: шаг «ё» не имеет права уронить озвучку.
"""

from __future__ import annotations

import gzip
import logging
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).with_name("data")
# Пути — переменные модуля, а не константы внутри загрузчика: тест подменяет их на
# несуществующий файл и проверяет fallback, не трогая репозиторий.
SAFE_PATH = _DATA_DIR / "yo_safe.tsv.gz"
AMBIGUOUS_PATH = _DATA_DIR / "yo_ambiguous.tsv.gz"

# Границы слова — тот же явный класс, что в `pronunciation.py` (кириллица, латиница,
# цифры), плюс «+»: слово с ручной разметкой ударения шаг не трогает целиком, а не
# по частям вокруг знака.
_BOUNDARY_CHARS = "0-9A-Za-z\u0400-\u04FF+"
_WORD_RE = re.compile(rf"(?<![{_BOUNDARY_CHARS}])([А-Яа-яЁё]+)(?![{_BOUNDARY_CHARS}])")

# Запасной список самых частых слов на случай отсутствия/повреждения данных.
# Каждая пара проверена по `yo_safe.tsv.gz`: неоднозначных слов здесь быть не может.
FALLBACK: dict[str, str] = {
    "ее": "её",
    "ежик": "ёжик",
    "ежика": "ёжика",
    "ежики": "ёжики",
    "елка": "ёлка",
    "елки": "ёлки",
    "елку": "ёлку",
    "еще": "ещё",
    "шел": "шёл",
    "пошел": "пошёл",
    "пришел": "пришёл",
    "ушел": "ушёл",
    "нашел": "нашёл",
    "прошел": "прошёл",
    "вошел": "вошёл",
    "зашел": "зашёл",
    "подошел": "подошёл",
    "отошел": "отошёл",
    "черный": "чёрный",
    "черная": "чёрная",
    "черное": "чёрное",
    "черные": "чёрные",
    "желтый": "жёлтый",
    "желтая": "жёлтая",
    "зеленый": "зелёный",
    "зеленая": "зелёная",
    "зеленую": "зелёную",
    "зеленые": "зелёные",
    "актер": "актёр",
    "актера": "актёра",
    "ребенок": "ребёнок",
    "ребенка": "ребёнка",
    "мед": "мёд",
    "меда": "мёда",
    "легкий": "лёгкий",
    "теплый": "тёплый",
    "темный": "тёмный",
    "серьезный": "серьёзный",
    "живет": "живёт",
    "везет": "везёт",
    "несет": "несёт",
    "идет": "идёт",
    "поет": "поёт",
    "дает": "даёт",
    "зовет": "зовёт",
    "пьет": "пьёт",
    "льет": "льёт",
    "вьет": "вьёт",
}

_lock = threading.Lock()
_safe: dict[str, str] | None = None
_ambiguous: dict[str, str] | None = None
_safe_warned = False
_ambiguous_warned = False


def _read_tsv(path: Path) -> dict[str, str]:
    """Читает gzip-TSV «е-форма<TAB>ё-форма»; битые строки пропускает."""
    rows: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.rstrip("\n").partition("\t")
            if key and value:
                rows[key] = value
    return rows


def _load_safe() -> dict[str, str]:
    """Гарантированные пары; при недоступном файле — запасной список и warning."""
    global _safe, _safe_warned
    with _lock:
        if _safe is not None:
            return _safe
        try:
            loaded = _read_tsv(SAFE_PATH)
        except (OSError, EOFError, UnicodeDecodeError) as exc:
            loaded = {}
            if not _safe_warned:
                _safe_warned = True
                logger.warning(
                    "Словарь «ё» недоступен (%s: %s) — работаю на запасном списке из %s слов",
                    type(exc).__name__,
                    exc,
                    len(FALLBACK),
                )
        if not loaded:
            loaded = dict(FALLBACK)
        else:
            logger.info("Словарь «ё»: загружено %s гарантированных пар", len(loaded))
        _safe = loaded
        return _safe


def _load_ambiguous() -> dict[str, str]:
    """Неоднозначные пары — только для подсказок, автоматически не применяются."""
    global _ambiguous, _ambiguous_warned
    with _lock:
        if _ambiguous is not None:
            return _ambiguous
        try:
            loaded = _read_tsv(AMBIGUOUS_PATH)
        except (OSError, EOFError, UnicodeDecodeError) as exc:
            loaded = {}
            if not _ambiguous_warned:
                _ambiguous_warned = True
                logger.warning(
                    "Список неоднозначных «е/ё» недоступен (%s) — подсказки по ё-омографам пусты",
                    exc,
                )
        _ambiguous = loaded
        return _ambiguous


def safe_dictionary() -> Mapping[str, str]:
    """Словарь гарантированных замен (только чтение). Ключи без «ё», значения с «ё»."""
    return _load_safe()


def ambiguous_dictionary() -> Mapping[str, str]:
    """Словарь неоднозначных пар для предложений в словарь (только чтение)."""
    return _load_ambiguous()


def reset_cache() -> None:
    """Сбрасывает ленивые словари — для тестов и для смены файла данных."""
    global _safe, _ambiguous, _safe_warned, _ambiguous_warned
    with _lock:
        _safe = None
        _ambiguous = None
        _safe_warned = False
        _ambiguous_warned = False


def match_case(word: str, target: str) -> str:
    """Переносит регистр исходного слова на замену: `Все → Всё`, `ВСЕ → ВСЁ`."""
    if word.isupper():
        return target.upper()
    if word[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def restore_yo(text: str, *, skip: Callable[[str], bool] | None = None) -> str:
    """Восстанавливает «ё» в бесспорных словах, сохраняя регистр.

    `skip` — предикат «слово покрыто правилом словаря пользователя». Если правило
    (включённое) матчит слово, шаг его не меняет: ручное правило приоритетнее.

    Идемпотентна: ключи словаря записаны без «ё», поэтому повторный проход по
    результату ничего не находит. Слова с «+» не трогаются вовсе — иначе ручная
    разметка ударения разъехалась бы с текстом.
    """
    if not text or ("е" not in text and "Е" not in text):
        # «ё» уже на месте или восстанавливать нечего.
        return text

    safe = _load_safe()
    if not safe:
        return text

    def replace(match: re.Match) -> str:
        word = match.group(1)
        target = safe.get(word.lower())
        if target is None:
            return word
        if skip is not None and skip(word):
            return word
        return match_case(word, target)

    # Разбиение по пробелам: слово с «+» отсекается целым токеном, а не по частям.
    parts = re.split(r"(\s+)", text)
    for index in range(0, len(parts), 2):
        token = parts[index]
        if not token or "+" in token:
            continue
        parts[index] = _WORD_RE.sub(replace, token)
    return "".join(parts)
