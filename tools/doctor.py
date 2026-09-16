#!/usr/bin/env python3
"""Самопроверка окружения одной командой (Фаза 16).

Инструмент отвечает на вопрос «почему не запускается / что настроено не так» до
первого синтеза: версия Python и venv, ffmpeg, наличие и размер моделей, место на
диске, права на `data/` и `output/`, состояние порта и замка единственного
инстанса, пробелы/не-ASCII в пути проекта и доступность MPS (best-effort).

Модели здесь **не поднимаются**: состояние берётся у `model_manager` (он смотрит
только файлы и уже созданные движки), `get_engine`/`load` не вызываются вовсе.
Поэтому инструмент безопасен на машине с запущенным дашбордом.

    ./venv/bin/python tools/doctor.py            # человекочитаемый отчёт
    ./venv/bin/python tools/doctor.py --json     # то же машинночитаемым JSON

Код возврата: 0 — всё в порядке, 1 — есть проблемы (у каждой есть подсказка).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Инструмент запускается из корня репозитория и как `tools/doctor.py`: корень
# нужен в sys.path, чтобы импортировался пакет `backend`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, model_manager
from tools import launcher

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Модель без файлов — это проблема: без основного движка дашборд не синтезирует.
REQUIRED_MODEL_IDS = ("f5",)
# Файлы, без которых приложение заведомо не поднимется.
REQUIRED_PROJECT_FILES = (
    "run.sh",
    "VOICE_SYNTEZ.command",
    "requirements.txt",
    "backend/main.py",
    "frontend/index.html",
    "tools/launcher.py",
)


def check(name: str, status: str, detail: str, hint: str = "") -> dict:
    """Один пункт отчёта: что проверяли, что нашли и что с этим делать."""
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def _python_check() -> tuple[dict, dict]:
    """Версия Python и venv. Возвращает (пункт отчёта, машиночитаемый блок)."""
    version = tuple(sys.version_info[:2])
    in_venv = launcher.python_is_venv(sys.prefix, sys.base_prefix)
    data = {
        "version": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": sys.executable,
        "venv": in_venv,
    }
    if not launcher.python_version_ok(version):
        return (
            check(
                "Python",
                FAIL,
                f"версия {data['version']} — нужна {launcher.MIN_PYTHON[0]}.{launcher.MIN_PYTHON[1]}+",
                "Установите Python 3.11+: brew install python@3.11, затем запустите через VOICE_SYNTEZ.command.",
            ),
            data,
        )
    if not in_venv:
        return (
            check(
                "Python",
                WARN,
                f"{data['version']} — запущено вне venv",
                "Для приложения нужны зависимости из venv: запускайте ./venv/bin/python tools/doctor.py.",
            ),
            data,
        )
    return check("Python", OK, f"{data['version']} в venv ({sys.executable})"), data


def _venv_check() -> dict:
    """Есть ли venv и актуальны ли зависимости по хешу requirements.txt."""
    venv_dir = Path(config.BASE_DIR) / "venv"
    venv_python = launcher.venv_python(venv_dir)
    if not venv_python.exists():
        return check(
            "venv",
            WARN,
            f"{venv_dir} не найден",
            "Лаунчер создаст его сам, либо: python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt",
        )
    requirements = Path(config.BASE_DIR) / "requirements.txt"
    marker = venv_dir / ".requirements.sha256"
    try:
        stale = launcher.install_needed(requirements, marker)
    except FileNotFoundError:
        return check("venv", WARN, f"{venv_dir} есть, но requirements.txt не найден", "Проверьте каталог проекта.")
    if stale:
        return check(
            "venv",
            WARN,
            "venv есть, но зависимости могли устареть (маркер хеша не совпал)",
            "Следующий запуск VOICE_SYNTEZ.command переустановит requirements.txt.",
        )
    return check("venv", OK, f"{venv_dir} есть, зависимости соответствуют requirements.txt")


def _ffmpeg_check() -> tuple[dict, dict]:
    path = shutil.which("ffmpeg")
    if not path:
        return (
            check(
                "ffmpeg",
                WARN,
                "не найден",
                "Без ffmpeg не работает MP3-экспорт и часть mp3/m4a-референсов: brew install ffmpeg.",
            ),
            {"path": None},
        )
    return check("ffmpeg", OK, path), {"path": path}


def _models_check() -> tuple[list[dict], dict]:
    """Состояние моделей через model_manager — без создания движков."""
    items: list[dict] = []
    manager = model_manager.get_manager()
    installed = []
    missing = []
    total_bytes = 0
    for state in manager.states():
        spec = state.spec
        if state.installed:
            installed.append(spec.id)
            total_bytes += state.size_bytes
            items.append(
                check(
                    f"модель {spec.id}",
                    OK,
                    f"{spec.label}: установлена, {state.size_bytes / (1024 ** 3):.2f} ГБ",
                )
            )
        else:
            missing.append(spec.id)
            # Без F5 синтез невозможен вовсе; остальные можно доставить позже.
            status = FAIL if spec.id in REQUIRED_MODEL_IDS else WARN
            items.append(
                check(
                    f"модель {spec.id}",
                    status,
                    f"{spec.label}: нет файлов ({', '.join(state.missing_files)})",
                    "Скачайте модель во вкладке «Модели» (05) дашборда; файлы ищутся в "
                    f"{state.path}",
                )
            )
    disk = manager.disk_report()
    data = {
        "installed": installed,
        "missing": missing,
        "total_bytes": total_bytes,
        **disk,
    }
    return items, data


def _disk_check() -> tuple[dict, dict]:
    try:
        usage = shutil.disk_usage(config.MODELS_DIR)
    except OSError as exc:
        return (
            check("диск", WARN, f"не удалось прочитать {config.MODELS_DIR}: {exc}", "Проверьте путь TTS_MODELS_DIR."),
            {"free_bytes": 0, "total_bytes": 0},
        )
    free_gb = usage.free / (1024 ** 3)
    status = OK if free_gb >= 2 else WARN
    hint = "" if status == OK else "Модели занимают гигабайты: освободите место или смените TTS_MODELS_DIR."
    return (
        check(
            "диск",
            status,
            f"свободно {free_gb:.1f} ГБ из {usage.total / (1024 ** 3):.1f} ГБ",
            hint,
        ),
        {"free_bytes": usage.free, "total_bytes": usage.total},
    )


def _writable_check(name: str, path: Path) -> dict:
    """Права на запись в каталог — проверка делом, а не только битами режима."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return check(name, FAIL, f"{path} недоступен: {exc}", f"Выдайте права на запись: chmod u+w '{path}'")
    if not os.access(path, os.W_OK):
        return check(name, FAIL, f"{path} без права на запись", f"chmod u+w '{path}'")
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-", delete=True):
            pass
    except OSError as exc:
        return check(name, FAIL, f"не удалось создать файл в {path}: {exc}", f"Проверьте права на {path}")
    return check(name, OK, f"{path} доступен на запись")


def _path_check() -> tuple[dict, dict]:
    root = str(config.BASE_DIR)
    has_space = " " in root
    non_ascii = not root.isascii()
    notes = []
    if has_space:
        notes.append("есть пробелы")
    if non_ascii:
        notes.append("есть не-ASCII")
    data = {"root": root, "has_space": has_space, "non_ascii": non_ascii}
    if not notes:
        return check("путь проекта", OK, root), data
    # Это не проблема: лаунчер и run.sh подставляют пути только в кавычках.
    return (
        check("путь проекта", OK, f"{root} ({', '.join(notes)} — лаунчер работает с такими путями)"),
        data,
    )


def _dependencies_check() -> dict:
    """Вспомогательные модули проекта на месте и самодостаточны."""
    missing = [name for name in REQUIRED_PROJECT_FILES if not (Path(config.BASE_DIR) / name).exists()]
    if missing:
        return check("файлы проекта", FAIL, f"нет: {', '.join(missing)}", "Восстановите каталог проекта из репозитория.")
    return check("файлы проекта", OK, f"{len(REQUIRED_PROJECT_FILES)} обязательных файлов на месте")


def _port_and_lock_check(start: int = 8000, tries: int = 20) -> tuple[list[dict], dict]:
    lock_path = Path(config.BASE_DIR) / ".voice_syntez.lock"
    running = launcher.running_port(start, tries)
    held = launcher.lock_is_held(lock_path)
    data = {"running_port": running, "lock_held": held, "lock_path": str(lock_path)}
    items: list[dict] = []
    if running is not None:
        items.append(check("порт", OK, f"приложение уже отвечает на http://127.0.0.1:{running}"))
    else:
        free = launcher.first_free_port(start, tries)
        if free is None:
            items.append(
                check("порт", WARN, f"нет свободного порта в {start}..{start + tries}", "Задайте PORT другим значением.")
            )
        else:
            items.append(check("порт", OK, f"{free} свободен — приложение поднимется на нём"))
    if held:
        items.append(check("замок", OK, "замок .voice_syntez.lock держит живой процесс (приложение запущено)"))
    elif lock_path.exists():
        items.append(
            check(
                "замок",
                WARN,
                "остался файл .voice_syntez.lock от прошлого запуска, но он никем не удерживается",
                "Это не блокирует запуск; файл можно удалить.",
            )
        )
    else:
        items.append(check("замок", OK, "свободен — другой экземпляр не мешает"))
    return items, data


def _mps_check() -> tuple[dict, dict]:
    """Best-effort доступность MPS: тяжёлый torch импортируется только здесь."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — отсутствие torch это диагностика, а не падение
        return (
            check("MPS", WARN, f"torch не импортируется: {exc}", "Проверьте установку зависимостей: pip install -r requirements.txt"),
            {"available": None, "torch": False},
        )
    try:
        available = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 — счётчик MPS может быть недоступен в принципе
        return check("MPS", WARN, f"не удалось определить: {exc}"), {"available": None, "torch": True}
    if available:
        return check("MPS", OK, f"MPS доступен (torch {torch.__version__})"), {"available": True, "torch": True}
    return (
        check("MPS", WARN, "MPS недоступен — синтез пойдёт на CPU (медленнее)"),
        {"available": False, "torch": True},
    )


def collect(start_port: int = 8000) -> dict:
    """Собирает полный отчёт. Модели не поднимает, сеть не трогает."""
    checks: list[dict] = []
    python_item, python_data = _python_check()
    checks.append(python_item)
    checks.append(_venv_check())
    checks.append(_dependencies_check())
    ffmpeg_item, ffmpeg_data = _ffmpeg_check()
    checks.append(ffmpeg_item)
    model_items, models_data = _models_check()
    checks.extend(model_items)
    checks.append(_writable_check("data/", Path(config.DATA_DIR)))
    checks.append(_writable_check("output/", Path(config.OUTPUT_DIR)))
    disk_item, disk_data = _disk_check()
    checks.append(disk_item)
    path_item, path_data = _path_check()
    checks.append(path_item)
    port_items, port_data = _port_and_lock_check(start_port)
    checks.extend(port_items)
    mps_item, mps_data = _mps_check()
    checks.append(mps_item)

    failures = [item for item in checks if item["status"] == FAIL]
    warnings = [item for item in checks if item["status"] == WARN]
    return {
        "ok": not failures,
        "root": str(config.BASE_DIR),
        "python": python_data,
        "ffmpeg": ffmpeg_data,
        "models": models_data,
        "disk": disk_data,
        "path": path_data,
        "port": port_data,
        "mps": mps_data,
        "checks": checks,
        "summary": {"ok": len(checks) - len(failures) - len(warnings), "warn": len(warnings), "fail": len(failures)},
    }


_MARK = {OK: "[ ok ]", WARN: "[ !  ]", FAIL: "[ОШИБКА]"}


def render(report: dict) -> str:
    """Человекочитаемый отчёт: одна строка на проверку плюс итог."""
    lines = [f"Диагностика окружения: {report['root']}", ""]
    for item in report["checks"]:
        lines.append(f"{_MARK[item['status']]} {item['name']}: {item['detail']}")
        if item["hint"]:
            lines.append(f"        → {item['hint']}")
    summary = report["summary"]
    lines.append("")
    if report["ok"]:
        lines.append(
            f"Итог: всё в порядке ({summary['ok']} проверок, предупреждений {summary['warn']}). "
            "Запускайте: bash VOICE_SYNTEZ.command"
        )
    else:
        lines.append(
            f"Итог: проблем {summary['fail']} (предупреждений {summary['warn']}). "
            "Исправьте пункты с [ОШИБКА] и запустите doctor снова."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doctor.py",
        description="Самопроверка окружения TTS-дашборда (модели не поднимаются).",
    )
    parser.add_argument("--json", action="store_true", help="машинночитаемый отчёт вместо текста")
    parser.add_argument("--port", type=int, default=8000, help="с какого порта проверять занятость")
    args = parser.parse_args(argv)

    report = collect(start_port=args.port)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
