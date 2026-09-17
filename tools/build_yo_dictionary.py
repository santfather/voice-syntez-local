#!/usr/bin/env python3
"""Собирает словарь восстановления «ё» из открытого источника.

Зачем отдельный скрипт, а не список, набранный руками. Гарантированных слов с «ё»
десятки тысяч, и любой рукописный список был бы и неполным, и непроверяемым. Берём
готовый открытый словарь проекта `eyo-kernel` (MIT): у него уже разделены
**бесспорные** случаи (`safe.txt` — там «е» на месте «ё» всегда опечатка традиции) и
**неоднозначные** (`not_safe.txt` — фамилии и омографы вида «Ежиков/Ёжиков»). Ровно
это разделение и нужно: первое можно применять автоматически, второе — только
предлагать пользователю.

Скрипт запускается вручную и **не** входит в pytest: он ходит в сеть. Результат
(два сжатых TSV) лежит в репозитории, поэтому рантайму сеть не нужна вообще.

    ./venv/bin/python tools/build_yo_dictionary.py            # из GitHub
    ./venv/bin/python tools/build_yo_dictionary.py --safe f.txt --ambiguous g.txt

Формат источника — «Ёжиков(а|ой|у|ы)» или «Слово# комментарий». После разбора
получаются пары «е-форма → ё-форма»:

    ежиков\tёжиков

Правила отбора намеренно консервативные: всё, что непонятно (маркеры `_`, знаки
ударения, несколько слов, не-кириллица) — выбрасывается. Потерять слово значит
всего лишь не восстановить «ё»; применить замену не там — испортить текст.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_REPO = "https://raw.githubusercontent.com/e2yo/eyo-kernel/master"
SOURCE_SAFE = f"{SOURCE_REPO}/dictionary/safe.txt"
SOURCE_AMBIGUOUS = f"{SOURCE_REPO}/dictionary/not_safe.txt"
SOURCE_LICENSE = f"{SOURCE_REPO}/LICENSE"

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "text_normalization" / "data"

# Только кириллица, скобки и разделитель вариантов: всё остальное (пробелы, знаки
# ударения, латиница, дефисы) означает запись, которую мы не берёмся толковать.
_PLAIN_RE = re.compile(r"^[А-Яа-яЁё()|]+$")
_ALTERNATIVES_RE = re.compile(r"^([^()]+)\(([^()]*)\)$")
# Маркеры php-yoficator, смысл которых нам неизвестен: строку пропускаем целиком.
_UNKNOWN_MARKERS = ("_", "~", "+", "*", "-", "!")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _expand(line: str) -> list[str]:
    """Разворачивает «Ёжиков(а|ой|у)» в отдельные словоформы.

    Пустой вариант в скобках (`Ёлкин(|а|е)`) — это «без окончания», а не мусор.
    """
    if not line or line[0] in _UNKNOWN_MARKERS:
        return []
    if not _PLAIN_RE.match(line):
        return []
    match = _ALTERNATIVES_RE.match(line)
    if match:
        stem, variants = match.group(1), match.group(2).split("|")
        return [stem + variant for variant in variants]
    return [line]


def _to_yo_forms(text: str) -> list[str]:
    """Словоформы в нижнем регистре — такими они и хранятся в словаре."""
    return [form.lower() for form in _expand(_strip_comment(text))]


def _pairs(forms: list[str]) -> tuple[dict[str, str], int]:
    """Пары «е-форма → ё-форма» без внутренних противоречий.

    Если два написания с «ё» дают одну и ту же е-форму (например «всё» и «все»),
    случай неоднозначный по определению, и в гарантированный словарь он не
    попадает: там должны остаться только замены, которые нельзя сделать неправильно.
    """
    pairs: dict[str, str] = {}
    conflicts = 0
    for form in forms:
        if "ё" not in form:
            continue  # слово без «ё» в источнике — это не наш случай
        key = form.replace("ё", "е")
        if key == form:
            continue
        existing = pairs.get(key)
        if existing is not None and existing != form:
            conflicts += 1
            pairs.pop(key, None)
            continue
        if existing is None and key not in pairs:
            pairs[key] = form
    return pairs, conflicts


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _write_gz(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: иначе каждый прогон даёт новый бинарник, и коммит выглядит изменённым.
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))


def _load_lines(safe: str, ambiguous: str) -> tuple[list[str], list[str]]:
    safe_forms: list[str] = []
    for line in safe.splitlines():
        safe_forms += _to_yo_forms(line)
    ambiguous_forms: list[str] = []
    for line in ambiguous.splitlines():
        ambiguous_forms += _to_yo_forms(line)
    return safe_forms, ambiguous_forms


def build(safe_text: str, ambiguous_text: str) -> tuple[int, int]:
    """Собирает оба словаря и печатает сводку. Возвращает число пар."""
    safe_forms, ambiguous_forms = _load_lines(safe_text, ambiguous_text)

    safe_pairs, safe_conflicts = _pairs(safe_forms)
    ambiguous_pairs, ambiguous_conflicts = _pairs(ambiguous_forms)

    # Неоднозначное всегда важнее: если слово есть в обоих списках, автоматически
    # его менять нельзя — источник сам называет такой случай небесспорным.
    overlapped = sorted(set(safe_pairs) & set(ambiguous_pairs))
    for key in overlapped:
        safe_pairs.pop(key)

    safe_lines = [f"{key}\t{value}" for key, value in sorted(safe_pairs.items())]
    ambiguous_lines = [f"{key}\t{value}" for key, value in sorted(ambiguous_pairs.items())]

    _write_gz(DATA_DIR / "yo_safe.tsv.gz", safe_lines)
    _write_gz(DATA_DIR / "yo_ambiguous.tsv.gz", ambiguous_lines)

    print(f"бесспорных замен:      {len(safe_lines)}")
    print(f"неоднозначных (предложений): {len(ambiguous_lines)}")
    print(f"противоречий внутри списков: {safe_conflicts} + {ambiguous_conflicts}")
    print(f"убрано из безопасного как неоднозначное: {len(overlapped)}")
    print(f"записано: {DATA_DIR}/yo_safe.tsv.gz, {DATA_DIR}/yo_ambiguous.tsv.gz")
    return len(safe_lines), len(ambiguous_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка словаря восстановления «ё»")
    parser.add_argument("--safe", help="локальный safe.txt вместо загрузки из сети")
    parser.add_argument("--ambiguous", help="локальный not_safe.txt вместо загрузки из сети")
    parser.add_argument(
        "--license", action="store_true", help="только обновить текст лицензии источника"
    )
    args = parser.parse_args()

    if args.license:
        text = (
            Path(args.safe).read_text(encoding="utf-8")
            if args.safe and Path(args.safe).suffix == ".txt" and "MIT" in Path(args.safe).read_text(encoding="utf-8")[:200]
            else _fetch(SOURCE_LICENSE)
        )
        (DATA_DIR / "LICENSE.eyo-kernel.txt").write_text(text, encoding="utf-8")
        print(f"лицензия источника записана в {DATA_DIR}/LICENSE.eyo-kernel.txt")
        return 0

    safe_text = Path(args.safe).read_text(encoding="utf-8") if args.safe else _fetch(SOURCE_SAFE)
    ambiguous_text = (
        Path(args.ambiguous).read_text(encoding="utf-8") if args.ambiguous else _fetch(SOURCE_AMBIGUOUS)
    )
    build(safe_text, ambiguous_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
