"""Порядок шагов нормализации и защита технических конструкций.

Порядок здесь — не деталь реализации, а часть контракта, потому что шаги
пересекаются по тексту:

1. URL и email прячутся первыми: внутри них есть и цифры, и латиница, и
   трогать их нельзя ни одним следующим шагом.
2. Телефоны читаются цифра за цифрой, пока числа ещё не развёрнуты.
3. Римские числительные — до латиницы, иначе «XXI» уйдёт в чтение по буквам.
4. Даты — до «версий»: «12.05.2026» и «1.2.3» выглядят одинаково, но первая
   обязана стать датой, а вторая — остаться технической строкой.
5. Время — до чисел: «12:30» не должно распасться на «двенадцать» и «тридцать»
   через двоеточие.
6. IP и версии прячутся после дат: всё, что похоже на цепочку чисел с точками
   и не оказалось датой, остаётся как есть.
7. Знак номера, диапазоны, деньги, единицы — до общих чисел, иначе число
   прочитается само, а существительное после него останется без согласования.
8. Общие числа и сокращения — по остаточному принципу.
9. Восстановление «ё» — после чисел и сокращений, то есть уже по готовым словам:
   автоматика меняет только бесспорные случаи («ежик» → «ёжик»), но **пропускает**
   слово, покрытое включённым правилом словаря, — ручное правило приоритетнее.
   Неоднозначные омографы е/ё здесь не трогаются вовсе (см. `yo_restoration`).
10. Словарь произношения — после «ё» и до латиницы и до возврата технических строк.
    Числа и сокращения к этому моменту уже развёрнуты, поэтому правило видит готовое
    слово; а латиница ещё не прочитана по буквам, поэтому пользователь может
    переопределить чтение аббревиатуры («SQL»), которое шаг латиницы иначе сделал
    бы сам. Технические строки спрятаны в плейсхолдеры — правило не перепишет URL
    или версию случайно.
11. Латиница — последней: она читает то, что осталось латинским, включая
    латиницу внутри замены словаря.

Дополнительно словарь умеет вычитать ударения для движков, которые знак «+» не
понимают (`supports_accents=False`). Нормализация об этом не знает: она лишь
передаёт флаг в шаг словаря.

Плейсхолдеры — символы из области для приватного использования: они не цифры,
не латиница и не кириллица, поэтому ни один шаблон их не видит. Так защита не
требует ни отдельного реестра диапазонов, ни риска задеть соседний текст.
"""

import logging
import re
from dataclasses import dataclass

from . import abbreviations, dates, latin, money, numbers, time, units, yo_restoration
from .pronunciation import (
    PronunciationEntries,
    apply_pronunciation_report,
    coverage_predicate,
)

logger = logging.getLogger(__name__)

# Почта раньше URL: адрес внутри ссылки должен целиком уйти в плейсхолдер.
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
# Домен без схемы: список зон закрытый — так «1.2.3» и «Python 3.11» не попадут
# сюда по ошибке.
DOMAIN_RE = re.compile(
    r"(?<![@\w.-])(?:[A-Za-z0-9-]+\.)+"
    r"(?:ru|com|org|net|io|ai|dev|app|py|su|by|kz|ua|me|co|uk|de|fr|tech)"
    r"(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)
# Цепочка чисел с точками и запятыми от трёх групп: версии («1.2.3») и IP
# («192.168.1.1»). Невалидная дата («32.13.2026») тоже остаётся здесь как есть.
TECHNICAL_RE = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+){2,}(?![\d.,])")


class Protector:
    """Прячет фрагменты в плейсхолдеры и возвращает их после всех шагов.

    Лимит в 512 фрагментов выбран с запасом: это не счётчик ссылок в реплике, а
    предохранитель от бесконечного роста, если шаблон когда-нибудь станет
    слишком жадным.
    """

    _FIRST = 0xE100
    _LIMIT = _FIRST + 512

    def __init__(self) -> None:
        self._originals: list[tuple[str, str]] = []
        self._next = self._FIRST

    def protect(self, text: str, pattern: re.Pattern) -> str:
        """Заменяет совпадения плейсхолдерами; при переполнении оставляет как есть."""

        def replace(match: re.Match) -> str:
            if self._next >= self._LIMIT:
                return match.group()
            placeholder = chr(self._next)
            self._next += 1
            self._originals.append((placeholder, match.group()))
            return placeholder

        return pattern.sub(replace, text)

    def hide(self, fragment: str) -> str:
        """Прячет один фрагмент — для шагов, которые сами решают, что не трогать."""
        if self._next >= self._LIMIT:
            return fragment
        placeholder = chr(self._next)
        self._next += 1
        self._originals.append((placeholder, fragment))
        return placeholder

    def restore(self, text: str) -> str:
        for placeholder, original in self._originals:
            text = text.replace(placeholder, original)
        return text


def _protect_first(text: str, protector: Protector) -> str:
    for pattern in (EMAIL_RE, URL_RE, DOMAIN_RE):
        text = protector.protect(text, pattern)
    return text


@dataclass(frozen=True)
class NormalizeStages:
    """Промежуточные итоги прохода — для preview «что услышит модель».

    `normalized` — та же стадия, что показывал preview и раньше: числа, сокращения и
    латиница, но **без** «ё» и без словаря. `yo` — она же после восстановления «ё».
    `result` — после словаря произношения (то, что уйдёт в модель до ударений).
    Отдельные поля, а не повторные прогоны: стадии обязаны быть срезами одного
    прохода, иначе preview однажды разойдётся с синтезом.
    """

    normalized: str
    yo: str
    result: str
    matches: list[dict]


def _normalize(
    text: str,
    pronunciation: PronunciationEntries,
    supports_accents: bool,
    *,
    with_yo: bool = True,
    collect_stages: bool = False,
) -> tuple[str, list[dict], NormalizeStages | None]:
    """Общий проход нормализации: текст, отчёт словаря и (по запросу) срезы стадий."""
    if not text:
        stages = None
        if collect_stages:
            stages = NormalizeStages(normalized="", yo="", result="", matches=[])
        return text, [], stages

    protector = Protector()
    result = _protect_first(text, protector)
    result = numbers.expand_phones(result)
    result = numbers.expand_roman(result)
    result = dates.expand_dates(result)
    result = time.expand_time(result, protector.hide)
    result = protector.protect(result, TECHNICAL_RE)
    result = abbreviations.expand_number_sign(result)
    result = units.expand_ranges(result)
    result = money.expand_money(result)
    result = units.expand_units(result)
    result = numbers.expand_numbers(result)
    result = abbreviations.expand_abbreviations(result)

    # «Нормализация» как стадия preview исторически показывает числа, сокращения и
    # латиницу, но не «ё»: считаем её до шага «ё» и отдельно прогоняем латиницу.
    normalized_stage = latin.expand_latin(result) if collect_stages else ""

    if with_yo:
        # Слова, покрытые включённым правилом словаря, автоматика не переписывает.
        result = yo_restoration.restore_yo(
            result, skip=coverage_predicate(pronunciation)
        )

    yo_stage = latin.expand_latin(result) if collect_stages else ""

    result, matches = apply_pronunciation_report(
        result, pronunciation, supports_accents=supports_accents
    )
    result = latin.expand_latin(result)
    result = protector.restore(result)

    if result != text:
        logger.info("Текст до модели: «%s» → «%s»", text[:120], result[:120])

    stages = None
    if collect_stages:
        stages = NormalizeStages(
            normalized=protector.restore(normalized_stage),
            yo=protector.restore(yo_stage),
            result=result,
            matches=matches,
        )
    return result, matches, stages


def normalize(
    text: str,
    pronunciation: PronunciationEntries = None,
    supports_accents: bool = True,
) -> str:
    """Готовит текст реплики к отправке в модель.

    Идемпотентна: после шага в тексте не остаётся ни цифр, требующих чтения, ни
    разобранных сокращений, ни слов с гарантированной «ё», а защищённые URL, email,
    версии и IP возвращаются в исходном виде — повторный проход ничего не меняет.

    `pronunciation` — правила словаря произношения (или устаревший словарь замен),
    `supports_accents` — понимает ли движок знак «+». Оба параметра необязательны:
    вызов `normalize(text)` работает как раньше.
    """
    return _normalize(text, pronunciation, supports_accents)[0]


def normalize_report(
    text: str,
    pronunciation: PronunciationEntries = None,
    supports_accents: bool = True,
) -> tuple[str, list[dict]]:
    """Как `normalize`, но вместе с отчётом словаря — для preview «что услышит модель».

    Отчёт считается в том же проходе, что и результат: preview не может показать
    одно, а синтез получить другое.
    """
    result, matches, _ = _normalize(text, pronunciation, supports_accents)
    return result, matches


def normalize_stages(
    text: str,
    pronunciation: PronunciationEntries = None,
    supports_accents: bool = True,
) -> NormalizeStages:
    """Срезы одного прохода: нормализация → «ё» → словарь. Для стадий preview."""
    _, _, stages = _normalize(
        text, pronunciation, supports_accents, collect_stages=True
    )
    assert stages is not None  # collect_stages=True всегда заполняет срезы
    return stages
