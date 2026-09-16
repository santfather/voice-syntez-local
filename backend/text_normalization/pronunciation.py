"""Словарь произношения: пользовательские замены текста перед моделью.

Здесь только чистая логика — ни базы, ни файлов, ни конфига. Хранилище живёт в
`backend/pronunciation.py`, сюда приходит уже готовый список правил: так правило
проверяется тестом без SQLite, а сервис ничего не знает про регулярки.

Место шага — последним в нормализации, уже по готовому тексту: пользователь должен
видеть для замены «25 руб.», а не «25 руб.». Каждое следующее правило применяется
к результату предыдущего, а более длинный источник идёт раньше короткого — иначе
«OpenAI API» развалилось бы на «OpenAI» + «API».

Границы слова заданы явным классом символов (кириллица, латиница, цифры), а не
`\b`: `\b` считает словом ещё и подчёркивание, а его поведение на стыке алфавитов
зависит от Unicode-категорий. Явный класс — ровно то правило, которое обещает UI.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# «Словесные» символы для границ. \u0400–\u04FF — вся кириллица, а не только
# русские буквы: правила должны одинаково вести себя на «ё» и на украинских «і/ї».
_WORD_CHARS = "0-9A-Za-z\u0400-\u04FF"
_WORD_RE = re.compile(f"[{_WORD_CHARS}]")

# Ручное ударение F5: «+» перед гласной. Только оно считается разметкой, поэтому
# «C++» или «плюс+» не теряют свой знак при отправке в движок без ударений.
_STRESS_RE = re.compile(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY])")


@dataclass(frozen=True)
class PronunciationRule:
    """Одно правило словаря.

    `whole_word` и `case_sensitive` — то, что пользователь переключает в UI;
    `enabled` позволяет выключить правило, не удаляя его из словаря.
    """

    source: str
    target: str
    case_sensitive: bool = False
    whole_word: bool = True
    enabled: bool = True


# Обратная совместимость: старый формат «источник → цель» одним словарём и новый
# список правил (dataclass, dict или пара) — всё это принимает `apply_pronunciation`.
PronunciationEntries = Mapping[str, str] | Iterable[Any] | None


def strip_stress(text: str) -> str:
    """Убирает ручные ударения F5: движок без их поддержки прочитал бы «+» вслух."""
    return _STRESS_RE.sub(r"\1", text)


@dataclass(frozen=True)
class _CompiledRule:
    source: str
    target: str
    pattern: re.Pattern


def _coerce_rule(value: Any) -> PronunciationRule | None:
    """Приводит запись словаря к правилу. Непонятный формат — `None` (и warning)."""
    if isinstance(value, PronunciationRule):
        return value
    if isinstance(value, Mapping):
        if "source" in value or "target" in value:
            return PronunciationRule(
                source=value.get("source"),
                target=value.get("target"),
                case_sensitive=bool(value.get("case_sensitive", False)),
                whole_word=bool(value.get("whole_word", True)),
                enabled=bool(value.get("enabled", True)),
            )
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return PronunciationRule(source=value[0], target=value[1])
    return None


def _iter_entries(entries: Any) -> Iterable[Any]:
    """Разворачивает коллекцию записей, включая устаревший формат «источник → цель»."""
    if isinstance(entries, Mapping):
        if "source" in entries or "target" in entries:
            yield entries  # одна запись, а не словарь замен
            return
        yield from entries.items()
        return
    yield from entries


def _validate(rule: PronunciationRule) -> str | None:
    if not isinstance(rule.source, str) or not rule.source.strip():
        return "пустой источник"
    if not isinstance(rule.target, str) or not rule.target.strip():
        return "пустая замена"
    return None


def compile_rule(rule: PronunciationRule, supports_accents: bool = True) -> _CompiledRule | None:
    """Собирает регулярку правила. Некорректное правило возвращает `None` и не роняет синтез."""
    problem = _validate(rule)
    if problem is not None:
        logger.warning(
            "Правило словаря пропущено (%s): %r → %r", problem, rule.source, rule.target
        )
        return None

    source = rule.source.strip()
    target = rule.target.strip()
    if not supports_accents:
        # Движок без поддержки «+» (XTTS) прочитал бы знак как отдельный символ:
        # для него правило превращается в «совместимую» замену без разметки.
        target = strip_stress(target)

    escaped = re.escape(source)
    prefix = f"(?<![{_WORD_CHARS}])" if _WORD_RE.match(source[0]) else ""
    suffix = f"(?![{_WORD_CHARS}])" if _WORD_RE.match(source[-1]) else ""
    body = f"{prefix}{escaped}{suffix}" if rule.whole_word else escaped
    flags = 0 if rule.case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(body, flags)
    except re.error as exc:  # защита от будущих изменений в сборке шаблона
        logger.warning("Правило словаря пропущено (некорректный шаблон %r): %s", source, exc)
        return None
    return _CompiledRule(source=source, target=target, pattern=pattern)


def apply_pronunciation_report(
    text: str, entries: PronunciationEntries = None, supports_accents: bool = True
) -> tuple[str, list[dict]]:
    """Применяет правила и возвращает текст вместе с отчётом по сработавшим правилам.

    Отчёт (`[{source, target, count}]`) нужен preview-эндпоинту: пользователь должен
    видеть не только результат, но и то, какие правила его сделали. Счётчик считает
    фактические замены, а не совпадения шаблона, поэтому совпадает с результатом.
    """
    if not text or not entries:
        return text, []

    compiled: list[_CompiledRule] = []
    for value in _iter_entries(entries):
        rule = _coerce_rule(value)
        if rule is None:
            logger.warning("Правило словаря пропущено (непонятный формат): %r", value)
            continue
        if not rule.enabled:
            continue
        item = compile_rule(rule, supports_accents=supports_accents)
        if item is not None:
            compiled.append(item)

    # Длинный источник раньше короткого; при равной длине порядок исходный —
    # сортировка стабильна, и словарь не «переставляет» равные правила сам.
    compiled.sort(key=lambda item: len(item.source), reverse=True)

    matches: list[dict] = []
    for item in compiled:
        count = 0

        def replace(_match: re.Match, target: str = item.target) -> str:
            nonlocal count
            count += 1
            return target

        text = item.pattern.sub(replace, text)
        if count:
            matches.append({"source": item.source, "target": item.target, "count": count})
    return text, matches


def apply_pronunciation(
    text: str, entries: PronunciationEntries = None, supports_accents: bool = True
) -> str:
    """Применяет пользовательские замены произношения; без словаря — no-op."""
    result, _ = apply_pronunciation_report(text, entries, supports_accents=supports_accents)
    return result
