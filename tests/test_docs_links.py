"""Ссылки в документации ведут в существующие файлы (§«Соглашения» docs/README.md).

Документация без такой проверки гниёт молча: страницу переименовали, ссылку не
поправили — и переход ведёт в никуда, притом что сам текст выглядит целым. Тест
разбирает только настоящие markdown-ссылки вне кодовых блоков: примеры команд и
URL в тексте он не трогает.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# `[текст](цель)` — цель берём без заголовка-анкора и без обрамляющих пробелов.
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")


def _documents() -> list[Path]:
    """README в корне и все страницы docs/ — единственные документы, которые читает пользователь."""
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _links(path: Path) -> list[tuple[int, str]]:
    """Ссылки из файла: (строка, цель). Кодовые блоки пропускаются целиком."""
    found: list[tuple[int, str]] = []
    fence = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        for target in LINK_RE.findall(raw):
            found.append((number, target))
    return found


@pytest.mark.parametrize("document", _documents(), ids=lambda path: str(path.relative_to(ROOT)))
def test_relative_links_point_to_existing_files(document: Path):
    """Каждая относительная ссылка ведёт на существующий файл, а не в пустоту."""
    missing = [
        (number, target)
        for number, target in _links(document)
        if not target.startswith(("http://", "https://", "mailto:", "#", "<"))
        and not (document.parent / target.split("#", 1)[0]).exists()
    ]
    assert not missing, f"{document.relative_to(ROOT)}: битые ссылки {missing}"


@pytest.mark.parametrize("document", _documents(), ids=lambda path: str(path.relative_to(ROOT)))
def test_documents_do_not_link_to_removed_root_files(document: Path):
    """Ссылок на удалённые планы и отчёты не осталось: их место — история git.

    Упоминание имени в тексте (например, в журнале фаз) допустимо — недопустима
    именно ссылка: она вела бы на файл, которого в репозитории больше нет.
    """
    removed = {
        "AUDIT.md",
        "PHASE_2_SAM_SEBE_ZVUKOREZHISSER.md",
        "F5_LLM_WARMUP_IMPLEMENTATION_SPEC.md",
        "update_2.md",
        "update_3.md",
        "updates.md",
        "warmup_prefix_report.md",
        "short_phrasases.md",
        "sam_sebe_zvukorezhisser_report.md",
    }
    hits = [
        (number, target)
        for number, target in _links(document)
        if target.split("#", 1)[0].rsplit("/", 1)[-1] in removed
    ]
    assert not hits, f"{document.relative_to(ROOT)}: ссылки на удалённые файлы {hits}"
