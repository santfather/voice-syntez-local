#!/usr/bin/env python3
"""Разбор отчётов macOS о падениях после прогона тестов: отметить тестовые и убрать их.

Каждый SIGABRT/SIGSEGV в процессе Python оставляет файл `Python-*.ips` в
`~/Library/Logs/DiagnosticReports/`. Тесты изоляции роняют воркер намеренно
(`tests/fake_worker_engine.py`), поэтому один прогон добавляет туда десятки
файлов, и настоящий сигнал тонет в шуме. Инструмент берёт отчёты, появившиеся в
окне прогона (`--since-min`), проверяет по журналу приложения, подтверждает ли
падение метка `FAKE-WORKER-CRASH`, и по флагу убирает их из окна наблюдения.

    # посмотреть, что оставил последний прогон (ничего не меняет)
    ./venv/bin/python tools/retire_test_crashes.py --since-min 60

    # убрать их из окна наблюдения: перенос в Retired/test-crashes/YYYY-MM-DD
    ./venv/bin/python tools/retire_test_crashes.py --since-min 60 --retire

    # то же, без архива — вариант для CI
    ./venv/bin/python tools/retire_test_crashes.py --since-min 60 --delete

Границы окна задаёт вызывающий: сам прогон инструмент не видит. Перенос обратим,
поэтому `--retire` убирает всё, что попало в окно, а удаление — только отчёты,
подтверждённые меткой: неподтверждённый отчёт остаётся на месте, чтобы настоящее
падение не пропало вместе с тестовым шумом.

Отдельно про падения, поднятые самим pytest: фикстура `workspace` уводит журнал
приложения в `tmp_path`, поэтому метка тестового падения в `logs/voice_syntez.log`
не появляется — и `--delete` такие отчёты не тронет, хотя они заведомо тестовые.
Убирает их `--retire`: он переносит в архив всё окно, ничего не удаляя.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = Path.home() / "Library" / "Logs" / "DiagnosticReports"
DEFAULT_LOG = ROOT / "logs" / "voice_syntez.log"
# Метка из tests/fake_worker_engine.py: заглушка печатает её перед os.abort().
TEST_MARKER = "FAKE-WORKER-CRASH"
# Окно поиска метки в журнале вокруг момента запуска упавшего процесса.
LOG_WINDOW_SEC = 120
ARCHIVE_SUBDIR = Path("Retired") / "test-crashes"


def _report(path: Path) -> dict:
    """Заголовок и тело отчёта одним словарём.

    `.ips` — это два JSON подряд: первая строка — заголовок ядра (имя процесса и
    время), а со второй строки начинается тело с подробностями падения. Именно там
    лежат `pid`, `parentProc` и `procLaunch`, поэтому по одной первой строке о
    падении известно только то, что оно было.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    header = json.loads(lines[0])
    body = json.loads("".join(lines[1:])) if len(lines) > 1 else {}
    return {**header, **body}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: str) -> datetime | None:
    """Момент события местным временем.

    Источники пишут время по-разному: отчёт — `2026-09-23 17:42:40.8121 +0200`,
    журнал приложения — `2026-09-23 17:42:41,123` без зоны. Для окна в пару минут
    достаточно секунд, а местное время у обоих одно и то же (машина одна), поэтому
    отметка берётся по секундам и приводится к местной зоне: без общей зоны отметки
    из отчёта и из журнала несравнимы.
    """
    try:
        moment = datetime.fromisoformat(value[:19])
    except ValueError:
        return None
    return moment.astimezone()


def _marker_confirms(log_path: Path, moment: datetime | None) -> bool:
    """Подтверждено ли падение меткой тестового падения в журнале приложения.

    Метку заглушка пишет в stderr воркера, а родитель переносит stderr в общий
    журнал, поэтому искать её нужно там же, где всё остальное. Журнал читается
    построчно: он ротируется по 5 МБ и в память целиком не берётся.
    """
    if moment is None or not log_path.exists():
        return False
    window = timedelta(seconds=LOG_WINDOW_SEC)
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if TEST_MARKER not in line:
                continue
            stamp = _timestamp(line)
            if stamp is not None and abs(stamp - moment) <= window:
                return True
    return False


def _recent_reports(reports_dir: Path, since: datetime) -> list[Path]:
    """Отчёты, изменённые не раньше `since`: окно прогона задаётся снаружи."""
    found = [
        path
        for path in reports_dir.glob("Python-*.ips")
        if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) >= since
    ]
    return sorted(found, key=lambda item: item.stat().st_mtime)


def _archive(path: Path, reports_dir: Path) -> str:
    target_dir = reports_dir / ARCHIVE_SUBDIR / _now().astimezone().strftime("%Y-%m-%d")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    shutil.move(str(path), str(target))
    return f"перенесён в {target.relative_to(reports_dir)}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--since-min",
        type=float,
        default=60.0,
        help="окно прогона тестов в минутах, по умолчанию 60",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="каталог DiagnosticReports",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="журнал приложения, в котором ищется метка",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--retire", action="store_true", help="перенести отчёты окна в Retired/test-crashes/"
    )
    action.add_argument(
        "--delete", action="store_true", help="удалить подтверждённые тестовые отчёты"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.reports_dir.is_dir():
        print(f"Каталог отчётов не найден: {args.reports_dir}")
        return 1

    since = _now() - timedelta(minutes=args.since_min)
    reports = _recent_reports(args.reports_dir, since)
    if not reports:
        print(f"Отчётов за последние {args.since_min:g} мин нет — окно наблюдения чистое.")
        return 0

    print(f"Отчётов за последние {args.since_min:g} мин: {len(reports)}")
    unconfirmed = 0
    for path in reports:
        try:
            report = _report(path)
        except (OSError, ValueError) as exc:
            print(f"  {path.name}: отчёт не прочитан ({exc})")
            unconfirmed += 1
            continue
        moment = _timestamp(str(report.get("procLaunch", ""))) or _timestamp(
            str(report.get("timestamp", ""))
        )
        confirmed = _marker_confirms(args.log, moment)
        verdict = "тестовое, метка найдена" if confirmed else "метки нет — проверить вручную"
        print(
            f"  {path.name}: pid={report.get('pid')} parent={report.get('parentProc')} "
            f"procLaunch={report.get('procLaunch')} coalition={report.get('coalitionName')} "
            f"— {verdict}"
        )
        if args.retire:
            print(f"    {_archive(path, args.reports_dir)}")
        elif args.delete and confirmed:
            path.unlink()
            print("    удалён")
        else:
            unconfirmed += 1

    if not (args.retire or args.delete):
        print("\nНичего не изменено: добавьте --retire (в архив) или --delete (удалить).")
    elif unconfirmed:
        print(
            f"\nОсталось на месте {unconfirmed} отчётов без метки: убедитесь, что это те же "
            "тесты, а не продуктовое падение."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
