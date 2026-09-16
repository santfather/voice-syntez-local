"""Репозитории таблиц: проекты, реплики и варианты реплик.

Репозиторий ничего не знает про транзакции и файлы на диске: он принимает
открытое соединение и работает в его рамках. Границы транзакций и удаление
файлов — дело фасада (`store.py`), потому что «удалить проект» означает и
записи в четырёх таблицах, и файлы вариантов в output/.
"""

import json
from typing import Any


def dumps(value: Any) -> str:
    """JSON для TEXT-колонки: словари и списки хранятся как есть, без схемы."""
    return json.dumps(value, ensure_ascii=False)


def loads(raw: Any, default: Any) -> Any:
    """Разбирает JSON-колонку. Испорченное значение — не повод падать.

    Базу могли править руками или откатить; потерять из-за одной кривой ячейки
    весь проект хуже, чем показать настройки по умолчанию.
    """
    if raw is None or raw == "":
        return default
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default
