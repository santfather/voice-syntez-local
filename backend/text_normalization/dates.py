"""Даты: числовые, словесные и «1917 г.» — с явной политикой разбора.

Политика одна и та же для всех записей: **день → месяц → год**, как принято в
русском тексте. Поэтому «12/05/2026» и «12.05.2026» — двенадцатое мая, а не
5 декабря: американский порядок месяц-день здесь сознательно не поддерживается,
иначе одна и та же запись читалась бы по-разному в зависимости от разделителя.

Невалидная или неоднозначная дата не трогается вообще («32.13.2026»): лучше
оставить техническую строку как есть, чем прочитать её наугад. Так же ведёт себя
дата с годом вне разумного диапазона.

Год в дате читается порядковым в родительном: «2026 года» → «две тысячи двадцать
шестого года». Словесная форма («12 мая 2026 года») уже содержит месяц словом —
месяц остаётся как есть, а год обязан получить тот же родительный.
"""

import datetime as _datetime
import re

from .morphology import ordinal
from .numbers import is_year

MONTHS_NOM = (
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
)
MONTHS_GEN = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
# Падеж месяца в дате всегда родительный, поэтому именительная форма входа —
# просто вариант записи, который приводится к тому же виду.
_MONTH_INDEX = {
    name.lower(): index
    for index, names in enumerate(zip(MONTHS_NOM, MONTHS_GEN))
    for name in names
}

# Разделитель — точка или слэш, но не дефис: «12-05-2026» слишком похоже на
# диапазон, и портить его нельзя.
DATE_NUMERIC_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
DATE_VERBAL_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+([А-Яа-яЁё]+)(?:\s+(\d{4}))?\s*(?:года|году|год|г\.)?"
)
YEAR_DOT_RE = re.compile(r"(?<!\d)(\d{4})\s*г\.(?!\w)")


def _numeric_date(match: re.Match) -> str:
    day, month, year = (int(match.group(index)) for index in (1, 2, 3))
    if not is_year(year):
        return match.group()
    try:
        _datetime.date(year, month, day)
    except ValueError:
        return match.group()  # 32.13.2026 — не дата, оставляем как есть
    return f"{ordinal(day, 'neut')} {MONTHS_GEN[month - 1]} {ordinal(year, 'gen')} года"


def _verbal_date(match: re.Match) -> str:
    day = int(match.group(1))
    month = _MONTH_INDEX.get(match.group(2).lower())
    if month is None or not 1 <= day <= 31:
        return match.group()
    year_raw = match.group(3)
    if year_raw is not None and not is_year(int(year_raw)):
        return match.group()
    parts = [ordinal(day, "neut"), MONTHS_GEN[month]]
    if year_raw is not None:
        parts.extend([ordinal(int(year_raw), "gen"), "года"])
    return " ".join(parts)


def _year_dot(match: re.Match) -> str:
    """«1917 г.» — это год, а не «грамм», поэтому разбирается здесь, а не в units."""
    year = int(match.group(1))
    if not is_year(year):
        return match.group()
    return f"{ordinal(year, 'nom')} год"


def expand_dates(text: str) -> str:
    """Разворачивает даты всех трёх записей; невалидные оставляет без изменений."""
    text = DATE_NUMERIC_RE.sub(_numeric_date, text)
    text = DATE_VERBAL_RE.sub(_verbal_date, text)
    return YEAR_DOT_RE.sub(_year_dot, text)
