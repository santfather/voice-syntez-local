#!/usr/bin/env python3
"""Чистые помощники лёгкого лаунчера `VOICE_SYNTEZ.command`.

Скрипт запускается **системным** Python (venv может ещё не существовать), поэтому
здесь нет ни импортов проекта, ни сторонних библиотек — только стандартная
библиотека. Вся логика, которую имеет смысл проверять без запуска сервера и без
моделей, вынесена в функции: поиск интерпретатора 3.11+, свежесть зависимостей по
хешу `requirements.txt`, свободный порт и замок единственного инстанса
(см. `tests/test_launcher.py`).

Команды для оболочки (коды возврата — часть контракта):

    find-python                                   # путь к Python 3.11+ в stdout
    needs-install --requirements R --marker M     # 0 — ставить, 1 — уже стоит
    write-marker  --requirements R --marker M     # запоминает хеш requirements.txt
    free-port     --start 8000                    # первый свободный порт
    running-port  --start 8000                    # порт уже запущенного приложения
    lock-held     --path .voice_syntez.lock       # 0 — замок держит живой процесс
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Минимальная версия интерпретатора: на 3.10 нет синтаксиса, которым написан
# backend (`X | None` в рантайме), и torch/coqui-tts из requirements.txt.
MIN_PYTHON = (3, 11)
# Порядок поиска: сначала точная версия, потом общий `python3`. `python` без
# цифры — последний шанс на машинах, где по-другому не называется.
PYTHON_CANDIDATES = ("python3.11", "python3", "python")
# Насколько далеко от базового порта искать свободный/занятый.
DEFAULT_PORT_TRIES = 20
# Признаки «это наш /api/status», а не чужой сервис на том же порту.
STATUS_MARKERS = ("device", "engines", "accentizer_state")


# --- интерпретатор -------------------------------------------------------------
def python_version_ok(version: tuple[int, ...]) -> bool:
    """Достаточно ли версии интерпретатора для запуска backend."""
    return version[:2] >= MIN_PYTHON


def find_python(which, probe) -> str | None:
    """Путь к Python 3.11+ или `None`.

    `which` и `probe` передаются снаружи, чтобы тест мог проверить выбор версии
    без реальных интерпретаторов: `which(name) -> str | None` ищет команду в PATH,
    `probe(path) -> tuple[int, ...] | None` возвращает версию.
    """
    for name in PYTHON_CANDIDATES:
        path = which(name)
        if not path:
            continue
        version = probe(path)
        if version and python_version_ok(version):
            return str(path)
    return None


def _probe_system_python(path: str) -> tuple[int, ...] | None:
    """Версия интерпретатора подпроцессом: импортировать его в текущий нельзя."""
    import subprocess

    try:
        result = subprocess.run(
            [path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return tuple(int(part) for part in result.stdout.strip().split(".")[:2])
    except ValueError:
        return None


def python_is_venv(prefix: str, base_prefix: str) -> bool:
    """Запущен ли интерпретатор из virtualenv (а не системный)."""
    return prefix != base_prefix


def venv_python(venv_dir: Path) -> Path:
    """Путь к интерпретатору внутри venv."""
    return Path(venv_dir) / "bin" / "python"


# --- зависимости ---------------------------------------------------------------
def requirements_fingerprint(requirements: Path) -> str:
    """SHA-256 файла зависимостей; пустая строка — файла нет.

    Хеш, а не время изменения: `pip install` сам трогает файлы в venv, и по mtime
    повторный запуск переустанавливал бы зависимости каждый раз.
    """
    path = Path(requirements)
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_marker(marker: Path) -> str:
    """Сохранённый хеш зависимостей; пустая строка — маркера нет или он пуст."""
    path = Path(marker)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def install_needed(requirements: Path, marker: Path) -> bool:
    """Нужно ли ставить зависимости: маркер не совпал с хешем `requirements.txt`.

    Идемпотентность — требование к лаунчеру: повторный запуск без правок
    requirements.txt не должен снова тянуть гигабайты из сети. Хеш берётся у
    файла целиком, поэтому любая правка (даже комментарий) честно запускает
    переустановку, а совпадение — гарантирует пропуск.
    """
    fingerprint = requirements_fingerprint(requirements)
    if not fingerprint:
        raise FileNotFoundError(f"Не найден файл зависимостей: {requirements}")
    return read_marker(marker) != fingerprint


def write_marker(requirements: Path, marker: Path) -> str:
    """Запоминает хеш `requirements.txt` после успешной установки."""
    fingerprint = requirements_fingerprint(requirements)
    if not fingerprint:
        raise FileNotFoundError(f"Не найден файл зависимостей: {requirements}")
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fingerprint + "\n", encoding="utf-8")
    return fingerprint


# --- порт и замок --------------------------------------------------------------
def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Свободен ли порт. `SO_REUSEADDR` — как у uvicorn: TIME_WAIT не считается занятостью."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, int(port)))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def first_free_port(start: int, tries: int = DEFAULT_PORT_TRIES, host: str = "127.0.0.1") -> int | None:
    """Первый свободный порт в диапазоне `[start, start + tries)`."""
    for port in range(int(start), int(start) + int(tries)):
        if port_is_free(port, host):
            return port
    return None


def lock_is_held(lock_path: Path) -> bool:
    """Держит ли замок живой процесс.

    Проверка честная: `flock` неблокирующий, и ядро снимает замок вместе с
    процессом. Поэтому оставшийся после падения файл ничего не блокирует, а
    «файл существует» само по себе не означает «приложение запущено».
    """
    path = Path(lock_path)
    if not path.exists():
        return False
    try:
        handle = path.open("r")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Замок уже кем-то удерживается — это и есть «приложение запущено».
        return True
    finally:
        # Замок снимается закрытием: чужой инстанс не должен пострадать.
        handle.close()
    return False


def status_looks_like_app(url: str, timeout: float = 1.0) -> bool:
    """Отвечает ли по URL наш backend (`GET /api/status`) — без импорта проекта."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False
    return isinstance(data, dict) and all(marker in data for marker in STATUS_MARKERS)


def running_port(
    start: int, tries: int = DEFAULT_PORT_TRIES, host: str = "127.0.0.1", timeout: float = 1.0
) -> int | None:
    """Порт уже запущенного приложения в диапазоне `[start, start + tries)` или `None`.

    Именно опрос `/api/status`, а не «порт занят»: на 8000 может висеть чужой
    проект, и подменять его своим — ошибка. Ответ должен содержать поля нашего
    статуса, иначе порт считается чужим.
    """
    for port in range(int(start), int(start) + int(tries)):
        if status_looks_like_app(f"http://{host}:{port}/api/status", timeout=timeout):
            return port
    return None


# --- CLI -----------------------------------------------------------------------
def _cmd_find_python(_args: argparse.Namespace) -> int:
    import shutil

    found = find_python(shutil.which, _probe_system_python)
    if not found:
        print(
            f"Не найден Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+. Установите его: "
            f"`brew install python@3.11` (или с python.org), затем запустите снова.",
            file=sys.stderr,
        )
        return 1
    print(found)
    return 0


def _cmd_needs_install(args: argparse.Namespace) -> int:
    needed = install_needed(Path(args.requirements), Path(args.marker))
    print("install" if needed else "skip")
    return 0 if needed else 1


def _cmd_write_marker(args: argparse.Namespace) -> int:
    print(write_marker(Path(args.requirements), Path(args.marker)))
    return 0


def _cmd_free_port(args: argparse.Namespace) -> int:
    port = first_free_port(args.start, args.tries, args.host)
    if port is None:
        print(f"Не нашёл свободный порт в диапазоне {args.start}..{args.start + args.tries}", file=sys.stderr)
        return 1
    print(port)
    return 0


def _cmd_running_port(args: argparse.Namespace) -> int:
    port = running_port(args.start, args.tries, args.host, args.timeout)
    if port is None:
        return 1
    print(port)
    return 0


def _cmd_lock_held(args: argparse.Namespace) -> int:
    return 0 if lock_is_held(Path(args.path)) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launcher.py",
        description="Помощники лёгкого лаунчера: Python, зависимости, порт, замок.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    find = subparsers.add_parser("find-python", help="путь к Python 3.11+")
    find.set_defaults(func=_cmd_find_python)

    needs = subparsers.add_parser("needs-install", help="0 — зависимости устарели")
    needs.add_argument("--requirements", required=True)
    needs.add_argument("--marker", required=True)
    needs.set_defaults(func=_cmd_needs_install)

    write = subparsers.add_parser("write-marker", help="запомнить хеш requirements.txt")
    write.add_argument("--requirements", required=True)
    write.add_argument("--marker", required=True)
    write.set_defaults(func=_cmd_write_marker)

    free = subparsers.add_parser("free-port", help="первый свободный порт")
    free.add_argument("--start", type=int, default=8000)
    free.add_argument("--tries", type=int, default=DEFAULT_PORT_TRIES)
    free.add_argument("--host", default="127.0.0.1")
    free.set_defaults(func=_cmd_free_port)

    running = subparsers.add_parser("running-port", help="порт запущенного приложения")
    running.add_argument("--start", type=int, default=8000)
    running.add_argument("--tries", type=int, default=DEFAULT_PORT_TRIES)
    running.add_argument("--host", default="127.0.0.1")
    running.add_argument("--timeout", type=float, default=1.0)
    running.set_defaults(func=_cmd_running_port)

    lock = subparsers.add_parser("lock-held", help="держит ли замок живой процесс")
    lock.add_argument("--path", required=True)
    lock.set_defaults(func=_cmd_lock_held)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
