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
8. Общие числа, сокращения, латиница — по остаточному принципу.
9. Словарь произношения (пока no-op) — последним, уже по готовому тексту.

Плейсхолдеры — символы из области для приватного использования: они не цифры,
не латиница и не кириллица, поэтому ни один шаблон их не видит. Так защита не
требует ни отдельного реестра диапазонов, ни риска задеть соседний текст.
"""

import logging
import re

from . import abbreviations, dates, latin, money, numbers, pronunciation, time, units

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


def normalize(text: str) -> str:
    """Готовит текст реплики к отправке в модель.

    Идемпотентна: после шага в тексте не остаётся ни цифр, требующих чтения, ни
    разобранных сокращений, а защищённые URL, email, версии и IP возвращаются в
    исходном виде — повторный проход ничего не меняет.
    """
    if not text:
        return text

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
    result = latin.expand_latin(result)
    result = pronunciation.apply_pronunciation(result)
    result = protector.restore(result)

    if result != text:
        logger.info("Текст до модели: «%s» → «%s»", text[:120], result[:120])
    return result
