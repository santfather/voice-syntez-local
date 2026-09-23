"""Фаза 16: лёгкий лаунчер, doctor и smoke-инструмент — то, что проверяемо без моделей.

Ни один тест здесь не поднимает TTS-движки, не ходит в интернет и не запускает
реальный сервер: проверяются синтаксис и цитирование путей в `VOICE_SYNTEZ.command`,
чистые помощники `tools/launcher.py`, отчёт `tools/doctor.py`, режимы
`tools/smoke.py` и отсутствие поломок в `run.sh`. Живой end-to-end сценарий §7
(реальная синтезация) остаётся ручным прогоном `tools/smoke.py` — подменить его
фиктивными проверками нельзя.
"""

from __future__ import annotations

import fcntl
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from backend import config, model_manager
from tools import doctor, launcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_SCRIPT = PROJECT_ROOT / "VOICE_SYNTEZ.command"
LAUNCHER_MODULE = PROJECT_ROOT / "tools" / "launcher.py"
DOCTOR_MODULE = PROJECT_ROOT / "tools" / "doctor.py"
SMOKE_MODULE = PROJECT_ROOT / "tools" / "smoke.py"

# Переменные с путями: каждая подстановка обязана быть в двойных кавычках,
# иначе пробел или кириллица в каталоге проекта развалят команду на слова.
PATH_VARS = (
    "PROJECT_DIR",
    "VENV_DIR",
    "VENV_PY",
    "LAUNCHER_PY",
    "LOG_DIR",
    "LOG_FILE",
    "REQ_FILE",
    "REQ_MARKER",
    "LOCK_FILE",
    "SCRIPT_PATH",
    "BOOT_PYTHON",
    "SYSTEM_PYTHON",
)
_VAR_AT = re.compile(r"\$(?:\{)?(" + "|".join(PATH_VARS) + r")(?:\})?")


# --- 1. Синтаксис и исполняемость ---------------------------------------------
def test_launcher_script_exists_executable_and_syntax_valid():
    assert LAUNCHER_SCRIPT.is_file(), "VOICE_SYNTEZ.command должен лежать в корне проекта"
    assert os.access(LAUNCHER_SCRIPT, os.X_OK), "двойной клик из Finder требует исполняемый бит"
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


# --- 2. Подстановки путей ------------------------------------------------------
def _path_vars_outside_double_quotes(code: str) -> list[str]:
    """Переменные-пути, раскрытые вне двойных кавычек.

    Разбор учитывает вложенные подстановки `"$(... "..." ...)"`: внутри `$(...)`
    начинается отдельный контекст цитирования, и внутренние кавычки не закрывают
    внешние. Наивный счётчик кавычек на такой строке ошибается, поэтому здесь
    стек контекстов: `None` — код, `"` — двойные кавычки, `'` — одиночные.
    """
    found: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.split("#", 1)[0]
        contexts: list[str | None] = [None]
        index = 0
        while index < len(line):
            char = line[index]
            current = contexts[-1]
            if current == "'":
                if char == "'":
                    contexts.pop()
                index += 1
                continue
            if char == "\\":
                index += 2
                continue
            if char == '"':
                if current == '"':
                    contexts.pop()
                else:
                    contexts.append('"')
                index += 1
                continue
            if char == "'" and current != '"':
                contexts.append("'")
                index += 1
                continue
            if char == "$" and line[index + 1 : index + 2] == "(":
                contexts.append(None)  # команда внутри $(...) цитируется заново
                index += 2
                continue
            if char == ")" and current is None and len(contexts) > 1:
                contexts.pop()
                index += 1
                continue
            if char == "$":
                match = _VAR_AT.match(line, index)
                if match and current != '"':
                    found.append(match.group(1))
            index += 1
    return found


def test_launcher_quotes_path_substitutions_and_has_no_dynamic_parsing():
    code = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    assert _path_vars_outside_double_quotes(code) == [], "пути обязаны быть в кавычках"
    for token in ("eval", "awk", "sed"):
        assert not re.search(rf"(?<![A-Za-z_]){token}(?![A-Za-z_])", code), (
            f"лаунчер не должен использовать {token}: путь может содержать пробелы"
        )
    # Каталог определяется через "$0", а не через текущий каталог.
    assert '"$0"' in code
    assert 'PROJECT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"' in code


# --- 3. doctor: JSON, коды возврата, отсутствие загрузки моделей ---------------
def _run_doctor(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DOCTOR_MODULE), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_doctor_json_is_valid_with_expected_keys_and_zero_on_healthy_env():
    result = _run_doctor("--json")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    for key in ("ok", "root", "python", "ffmpeg", "models", "disk", "path", "port", "mps", "checks", "summary"):
        assert key in report, f"в JSON доктора нет ключа {key}"
    assert report["ok"] is True
    assert isinstance(report["checks"], list) and report["checks"]
    for item in report["checks"]:
        assert set(item) == {"name", "status", "detail", "hint"}
        assert item["status"] in ("ok", "warn", "fail")
    assert {"ok", "warn", "fail"} == set(report["summary"])
    # Человекочитаемый прогон заканчивается тем же итогом и тем же кодом.
    human = _run_doctor()
    assert human.returncode == 0
    assert "Итог: всё в порядке" in human.stdout


def test_doctor_reports_one_on_broken_environment(monkeypatch, tmp_path, capsys):
    # «Сломанное» окружение: каталог моделей пуст — основной движок не установлен.
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", empty_models)
    monkeypatch.setattr(config, "XTTS_BASE_DIR", empty_models / "xtts_v2")
    monkeypatch.setattr(config, "XTTS_BANANA_DIR", empty_models / "xtts_banana")
    model_manager.reset_manager()

    report = doctor.collect()
    assert report["ok"] is False
    assert report["summary"]["fail"] >= 1
    f5 = next(item for item in report["checks"] if item["name"] == "модель f5")
    assert f5["status"] == doctor.FAIL
    assert f5["hint"], "у проблемы должна быть подсказка, что делать"
    assert "[ОШИБКА]" in doctor.render(report)

    assert doctor.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_doctor_never_loads_models(monkeypatch):
    code = DOCTOR_MODULE.read_text(encoding="utf-8")
    assert "get_engine(" not in code, "doctor не должен создавать движки"
    assert ".load(" not in code, "doctor не должен поднимать модели"

    # Даже если реестр подменён на «падающий», отчёт собирается: движки не зовутся.
    from backend.engines import registry

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("doctor не должен вызывать get_engine")

    monkeypatch.setattr(registry, "get_engine", _forbidden)
    report = doctor.collect()
    assert report["checks"]


# --- 3b. давление памяти в doctor (Фаза 3, §3.4) --------------------------------
def _sysctl(monkeypatch, stdout: str, *, returncode: int = 0, error: Exception | None = None):
    """Подменяет вызов sysctl: команда настоящая не запускается."""

    def fake_run(args, **_kwargs):
        assert args[:2] == ["sysctl", "-n"], args
        if error is not None:
            raise error
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)


def test_doctor_reports_memory_pressure_by_level(monkeypatch):
    """Давление памяти ядра читается без `sudo` — в отличие от `powermetrics`."""
    _sysctl(monkeypatch, "1\n")
    item, data = doctor._vm_pressure_check()
    assert item["status"] == doctor.OK
    assert data == {"level": 1, "level_name": "норма"}

    _sysctl(monkeypatch, "2\n")
    item, data = doctor._vm_pressure_check()
    assert item["status"] == doctor.WARN
    assert "почти занята" in item["detail"]
    assert item["hint"], "у предупреждения должна быть подсказка, что делать"
    assert data == {"level": 2, "level_name": "предупреждение"}

    _sysctl(monkeypatch, "4\n")
    item, _ = doctor._vm_pressure_check()
    assert item["status"] == doctor.WARN
    assert "критично" in item["detail"]


def test_doctor_survives_unreadable_memory_pressure(monkeypatch):
    """Остальные проверки сохраняют смысл, если счётчик недоступен или ответ чужой."""
    _sysctl(monkeypatch, "", error=OSError("sysctl не найден"))
    item, data = doctor._vm_pressure_check()
    assert item["status"] == doctor.WARN
    assert data["level"] is None

    _sysctl(monkeypatch, "не число\n")
    item, data = doctor._vm_pressure_check()
    assert item["status"] == doctor.WARN
    assert data["level"] is None


def test_doctor_skips_memory_pressure_outside_macos(monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    item, data = doctor._vm_pressure_check()
    assert item["status"] == doctor.OK
    assert "неприменима" in item["detail"]
    assert data == {"level": None, "level_name": None}


def test_collect_includes_memory_pressure(monkeypatch):
    """Проверка попала в общий отчёт: имя в списке и число в JSON."""
    _sysctl(monkeypatch, "1\n")
    report = doctor.collect()
    assert any(item["name"] == "давление памяти" for item in report["checks"])
    assert report["vm_pressure"]["level"] == 1


# --- 4. smoke: --help и --dry-run без сервера ----------------------------------
def _closed_port() -> int:
    """Свободный порт, на котором заведомо никто не слушает."""
    port = launcher.first_free_port(9100, tries=200)
    assert port is not None
    return port


def test_smoke_help_works_without_server():
    result = subprocess.run(
        [sys.executable, str(SMOKE_MODULE), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_smoke_dry_run_works_without_server_and_does_not_synthesize():
    port = _closed_port()
    result = subprocess.run(
        [sys.executable, str(SMOKE_MODULE), "--dry-run", "--base-url", f"http://127.0.0.1:{port}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "рендер F5 + XTTS" in result.stdout, "план шагов должен печататься"
    assert "Сервер недоступен" in result.stdout


def test_smoke_full_run_refuses_to_start_without_server():
    port = _closed_port()
    result = subprocess.run(
        [sys.executable, str(SMOKE_MODULE), "--base-url", f"http://127.0.0.1:{port}", "--no-restart"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, "без сервера полный прогон обязан отказаться до синтеза"
    assert "недоступен" in result.stderr
    assert "не поднимает" in result.stderr


# --- 5. Путь проекта с пробелами и не-ASCII ------------------------------------
def test_launcher_and_doctor_survive_spaces_and_non_ascii_in_path():
    base = Path(tempfile.mkdtemp(prefix="voice-syntez-", dir="/tmp"))
    project = base / "Проект с пробелами"
    project.mkdir()
    try:
        shutil.copy2(LAUNCHER_SCRIPT, project / "VOICE_SYNTEZ.command")
        (project / "tools").mkdir()
        shutil.copy2(LAUNCHER_MODULE, project / "tools" / "launcher.py")
        (project / "requirements.txt").write_text("fastapi==0.141.1\n", encoding="utf-8")

        # Скрипт запускается из чужого каталога: путь берётся из "$0", не из cwd.
        result = subprocess.run(
            ["bash", str(project / "VOICE_SYNTEZ.command"), "--paths"],
            cwd="/",
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        paths = dict(
            line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line
        )
        assert paths["PROJECT_DIR"] == str(project.resolve())
        assert paths["LAUNCHER_PY"] == str(project.resolve() / "tools" / "launcher.py")
        assert paths["LOG_FILE"] == str(project.resolve() / "logs" / "voice_syntez.log")
        assert " " in paths["PROJECT_DIR"] and not paths["PROJECT_DIR"].isascii()

        # Идемпотентность зависимостей на таком пути — тем же помощником.
        helper = project / "tools" / "launcher.py"
        marker = project / "venv" / ".requirements.sha256"
        first = subprocess.run(
            [sys.executable, str(helper), "needs-install", "--requirements", str(project / "requirements.txt"), "--marker", str(marker)],
            capture_output=True, text=True, check=False,
        )
        assert first.returncode == 0 and first.stdout.strip() == "install"
        subprocess.run(
            [sys.executable, str(helper), "write-marker", "--requirements", str(project / "requirements.txt"), "--marker", str(marker)],
            capture_output=True, text=True, check=True,
        )
        second = subprocess.run(
            [sys.executable, str(helper), "needs-install", "--requirements", str(project / "requirements.txt"), "--marker", str(marker)],
            capture_output=True, text=True, check=False,
        )
        assert second.returncode == 1 and second.stdout.strip() == "skip"

        # doctor с data/ и output/ внутри не-ASCII каталога не ломает пути.
        env = dict(os.environ)
        env["TTS_DATA_DIR"] = str(project / "данные")
        env["TTS_OUTPUT_DIR"] = str(project / "вывод")
        report = _run_doctor("--json", env=env)
        assert report.returncode == 0, report.stdout + report.stderr
        payload = json.loads(report.stdout)
        writable = {
            item["name"]: item
            for item in payload["checks"]
            if item["name"] in ("data/", "output/")
        }
        assert set(writable) == {"data/", "output/"}
        for item in writable.values():
            assert item["status"] == "ok", item
            assert " " in item["detail"] and not item["detail"].isascii()
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --- 6. Идемпотентность первого/повторного запуска -----------------------------
def test_install_needed_is_idempotent_until_requirements_change(tmp_path):
    requirements = tmp_path / "requirements.txt"
    marker = tmp_path / "venv" / ".requirements.sha256"
    requirements.write_text("fastapi==0.141.1\n", encoding="utf-8")

    assert launcher.install_needed(requirements, marker) is True, "первый запуск ставит зависимости"
    fingerprint = launcher.write_marker(requirements, marker)
    assert launcher.read_marker(marker) == fingerprint
    assert launcher.install_needed(requirements, marker) is False, "повторный запуск ничего не меняет"

    requirements.write_text("fastapi==0.141.1\n# правка\n", encoding="utf-8")
    assert launcher.install_needed(requirements, marker) is True, "правка requirements запускает установку"

    with pytest.raises(FileNotFoundError):
        launcher.install_needed(tmp_path / "нет-такого.txt", marker)


# --- 7. .gitignore -------------------------------------------------------------
def test_gitignore_ignores_root_runtime_directories():
    """Рабочие каталоги в корне игнорируются, а одноимённые вложенные — нет.

    Шаблон без ведущего «/» (`data/`) действует на любой глубине: однажды он
    совпал с `backend/text_normalization/data/` и выкинул из репозитория словарь
    восстановления «ё». Поэтому здесь проверяется не текст строки, а её смысл:
    корневые каталоги игнорируются, вложенный `data/` — нет.
    """
    lines = [line.strip() for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "/logs/" in lines
    for directory in ("/venv/", "/models/", "/voices/", "/output/", "/data/"):
        assert directory in lines, f"корневой {directory} должен игнорироваться"

    def ignored(relative: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=PROJECT_ROOT, check=False
        ).returncode == 0

    assert ignored("data/voice_syntez.db"), "база проектов не должна попадать в git"
    assert ignored("logs/voice_syntez.log")
    assert not ignored("backend/text_normalization/data/yo_safe.tsv.gz"), (
        "словарь «ё» обязан быть в репозитории: без него шаг не работает на чистой установке"
    )


# --- 8. run.sh не сломан -------------------------------------------------------
def test_run_sh_is_still_valid_and_unchanged_in_behavior():
    script = PROJECT_ROOT / "run.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    text = script.read_text(encoding="utf-8")
    assert "source venv/bin/activate" in text
    assert "uvicorn backend.main:app" in text


# --- 9. run.sh пишет свой вывод в журнал (F-L2) --------------------------------
def test_run_sh_logs_its_failure_to_the_journal(tmp_path):
    """Падение до старта Python обязано остаться в журнале, а не только в окне.

    Проверяется на самом раннем выходе run.sh — чужом pidfile: он не требует ни
    venv, ни зависимостей, но проходит через ту же настройку журнала.
    """
    shutil.copy2(PROJECT_ROOT / "run.sh", tmp_path / "run.sh")
    pidfile = tmp_path / "busy.pid"
    pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")  # живой PID: место занято

    result = subprocess.run(
        ["bash", str(tmp_path / "run.sh")],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TTS_PIDFILE": str(pidfile)},
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "уже запущено" in result.stderr, "сообщение остаётся ошибкой, а не уезжает в stdout"

    journal = tmp_path / "logs" / "voice_syntez.log"
    assert journal.exists(), "журнал заводится даже без запуска сервера"
    assert "уже запущено" in journal.read_text(encoding="utf-8")


# --- 10. Прямые зависимости объявлены в requirements.txt ------------------------
def test_directly_imported_packages_are_declared_in_requirements():
    """То, что код импортирует напрямую, объявлено строкой, а не приходит транзитивно.

    `pydantic` и `starlette` backend импортирует сам, `httpx` — тесты; до сих пор
    все три приходили через `fastapi`. Смена мажорной версии FastAPI могла убрать
    любую из них из окружения молча, и импорт упал бы только при запуске.
    """
    declared = set()
    for raw in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            declared.add(re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0].lower())

    for package in ("pydantic", "starlette", "httpx"):
        assert package in declared, f"{package} не объявлен в requirements.txt"


# --- помощники лаунчера: версия Python, порт, замок ----------------------------
def test_find_python_prefers_supported_version():
    present = {"python3.11": "/opt/python3.11", "python3": "/usr/bin/python3", "python": "/usr/bin/python"}
    versions = {"/opt/python3.11": (3, 11), "/usr/bin/python3": (3, 9), "/usr/bin/python": (3, 9)}

    assert launcher.find_python(present.get, lambda path: versions[path]) == "/opt/python3.11"
    # python3.11 нет, но python3 уже 3.11 — он и выбирается.

    def only_python3(name: str) -> str | None:
        return "/usr/bin/python3" if name == "python3" else None

    assert launcher.find_python(only_python3, lambda path: (3, 11)) == "/usr/bin/python3"
    # Ни один кандидат не подходит — честный None, а не «первый попавшийся».
    assert launcher.find_python(present.get, lambda path: (3, 9)) is None
    assert launcher.python_version_ok((3, 11)) and launcher.python_version_ok((3, 12))
    assert not launcher.python_version_ok((3, 10))


def test_lock_is_held_only_while_live_process_holds_it(tmp_path):
    lock_path = tmp_path / ".voice_syntez.lock"
    assert launcher.lock_is_held(lock_path) is False, "отсутствующий файл — не замок"

    lock_path.write_text("", encoding="utf-8")
    assert launcher.lock_is_held(lock_path) is False, "оставшийся файл без владельца ничего не держит"

    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert launcher.lock_is_held(lock_path) is True
    finally:
        handle.close()
    assert launcher.lock_is_held(lock_path) is False


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    """Локальная заглушка /api/status: проверяем распознавание «своего» сервера."""

    payload = json.dumps({"device": "cpu", "engines": [], "accentizer_state": "idle"}).encode()

    def do_GET(self):
        if self.path != "/api/status":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args):
        """Тишина в выводе теста."""


def test_running_port_recognizes_only_our_status():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StatusHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert launcher.port_is_free(port) is False, "занятый порт не считается свободным"
        assert launcher.status_looks_like_app(f"http://127.0.0.1:{port}/api/status") is True
        assert launcher.running_port(port, tries=1) == port
        assert launcher.status_looks_like_app(f"http://127.0.0.1:{port}/api/voices") is False
        assert launcher.running_port(port + 1, tries=1) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
