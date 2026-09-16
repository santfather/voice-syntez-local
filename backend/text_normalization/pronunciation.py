"""Словарь произношения — зарезервированная точка расширения фазы 6.

Смысл точки: пользователь должен уметь сказать «это слово читай вот так»,
и правка должна доехать до модели независимо от того, какой движок читает
реплику. Место для неё — здесь, последним шагом нормализации, уже после того,
как числа и сокращения стали словами: пользовательский словарь должен видеть
готовый текст, а не «25 руб.».

До фазы 6 словаря нет, и функция обязана быть прозрачной: `entries=None` и
пустой список значат «не менять ничего». Pipeline вызывает её всегда — так
точка расширения остаётся настоящим шагом, а не мёртвым кодом, который
однажды забудут подключить.
"""

from collections.abc import Iterable, Mapping

PronunciationEntries = Mapping[str, str] | Iterable[tuple[str, str]] | None


def apply_pronunciation(text: str, entries: PronunciationEntries = None) -> str:
    """Применяет пользовательские замены произношения; без словаря — no-op."""
    if not entries:
        return text
    items = entries.items() if isinstance(entries, Mapping) else entries
    for source, target in items:
        if source:
            text = text.replace(source, target)
    return text
