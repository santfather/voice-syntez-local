#!/usr/bin/env python3
"""Диагностический архив проекта без запуска моделей: собрать данные для разбора.

Отвечает на жалобу «короткие реплики звучат плохо, и непонятно почему»: собирает
тот же архив, что и кнопка «Собрать архив диагностики» в интерфейсе, но из
командной строки — удобно, когда данные нужно передать человеку или приложить к
отчёту.

Инструмент работает **в процессе**, без ASGI-lifespan: сервер при старте греет
F5-TTS (несколько гигабайт), а диагностике модели не нужны вовсе — она только
читает базу, файлы take'ов, журнал и состояние окружения. Поэтому запускать можно
рядом с работающим дашбордом и на загруженной машине.

    ./venv/bin/python tools/diagnostics_smoke.py --list
    ./venv/bin/python tools/diagnostics_smoke.py --project <id или имя>
    ./venv/bin/python tools/diagnostics_smoke.py --project <id> --no-references

Готовый архив остаётся в `output/diagnostics/` (`TTS_DIAGNOSTICS_DIR`): его можно
открыть в Finder и передать целиком, вместе с README внутри.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config, diagnostics
from backend.db.connection import init_db
from backend.db.store import get_projects_store


def _pick_project(store, wanted: str) -> dict:
    """Проект по идентификатору или по имени: имя удобнее, id надёжнее.

    Несколько проектов с одним именем — не ошибка пользователя (имена не
    уникальны), поэтому выбирается самый свежий, а найденные варианты печатаются:
    молча взять «какой-то из них» значило бы собрать архив не о том диалоге.
    """
    by_id = store.get_project(wanted)
    if by_id is not None:
        return by_id
    matches = [item for item in store.list_projects() if item.get("name") == wanted]
    if not matches:
        raise SystemExit(f"Проект «{wanted}» не найден: ни по id, ни по имени")
    if len(matches) > 1:
        print(f"Проектов с именем «{wanted}» несколько, беру самый свежий:")
        for item in matches:
            print(f"  {item['id']}  {item.get('updated_at')}")
    return store.get_project(matches[0]["id"])


def _print_archives(project: dict) -> None:
    archives = diagnostics.list_archives(project)
    if not archives:
        print("Архивов диагностики у проекта пока нет.")
        return
    print(f"Архивы диагностики проекта «{project.get('name')}»:")
    for item in archives:
        print(f"  {item['name']}  {item['size_mb']} МБ  {item['created_at']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="id или имя проекта")
    parser.add_argument("--list", action="store_true", help="показать проекты и их архивы")
    parser.add_argument(
        "--job-output",
        help="файл готового рендера, который нужно положить в архив (путь на диске)",
    )
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="не класть референсы голосов (архив заметно меньше)",
    )
    parser.add_argument(
        "--max-audio-mb",
        type=float,
        default=None,
        help=f"предел аудио в архиве (по умолчанию {config.DIAGNOSTICS_MAX_AUDIO_MB})",
    )
    args = parser.parse_args()

    init_db()
    store = get_projects_store()

    if args.list or not args.project:
        projects = store.list_projects()
        if not projects:
            print("Проектов нет — сначала разберите текст в дашборде.")
            return 0
        print("Проекты:")
        for item in projects:
            archives = len(diagnostics.list_archives(store.get_project(item["id"]) or {}))
            print(
                f"  {item['id']}  {item.get('name')}  "
                f"реплик: {item.get('replicas_count')}  архивов: {archives}  "
                f"{item.get('updated_at')}"
            )
        if not args.project:
            print("\nУкажите --project <id или имя>, чтобы собрать архив.")
            return 0

    project = _pick_project(store, args.project)
    if args.list:
        _print_archives(project)
        return 0

    job_output = None
    if args.job_output:
        job_output = Path(args.job_output).expanduser()
        if not job_output.is_file():
            raise SystemExit(f"Файл рендера не найден: {job_output}")

    # Настройки берутся сохранённые в проекте: конкретного запуска здесь нет.
    # Если нужен архив именно того рендера, который слушали, его собирает кнопка в
    # интерфейсе — там задача ещё жива в памяти очереди.
    bundle = diagnostics.create_diagnostics_archive(
        project,
        render_settings=project.get("render_settings") or {},
        job_output=job_output,
        include_references=not args.no_references,
        max_audio_mb=args.max_audio_mb,
    )
    print(f"Архив: {bundle.path}")
    print(f"Размер: {bundle.size_mb} МБ, файлов: {len(bundle.entries)}")
    for entry in bundle.entries:
        print(f"  {entry}")
    if bundle.warnings:
        print("Предупреждения сборки:")
        for warning in bundle.warnings:
            print(f"  * {warning}")
    print(
        "\nНачните с README.md внутри архива: там порядок чтения — план короткой "
        "реплики, откат слоя, QA и журнал."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
