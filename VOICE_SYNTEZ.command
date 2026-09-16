#!/bin/bash
# Лёгкий лаунчер TTS-дашборда для macOS: двойной клик по этому файлу из Finder.
#
# Что делает:
#   1. находит каталог проекта независимо от текущего каталога;
#   2. находит Python 3.11+;
#   3. создаёт venv, если его нет, и ставит requirements.txt только когда он
#      создан или изменился (хеш сохранён в venv/.requirements.sha256);
#   4. предупреждает, если не найден ffmpeg (без него нет MP3-экспорта);
#   5. запускает backend на свободном порту и открывает браузер;
#   6. пишет лог в logs/voice_syntez.log и показывает его в терминале;
#   7. по Ctrl+C останавливает сервер и не оставляет процессов.
#
# Сеть нужна только для pip при первом запуске (или после правки
# requirements.txt). Веса моделей лаунчер не трогает.
#
# Диагностика без запуска сервера:  bash VOICE_SYNTEZ.command --paths

set -u

# --- каталог проекта -----------------------------------------------------------
# "$0" — путь к этому файлу, а не текущий каталог: Finder запускает .command из
# домашнего каталога. Подстановки только в кавычках, без динамического разбора
# пути: пробелы и не-ASCII в именах каталогов должны доезжать как есть.
SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$PWD/$SCRIPT_PATH" ;;
esac
PROJECT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)" || exit 1
cd "$PROJECT_DIR" || exit 1

VENV_DIR="$PROJECT_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
LAUNCHER_PY="$PROJECT_DIR/tools/launcher.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/voice_syntez.log"
REQ_FILE="$PROJECT_DIR/requirements.txt"
REQ_MARKER="$VENV_DIR/.requirements.sha256"
PORT_BASE="${PORT:-8000}"

if [ "${1:-}" = "--paths" ]; then
  printf 'PROJECT_DIR=%s\n' "$PROJECT_DIR"
  printf 'VENV_DIR=%s\n' "$VENV_DIR"
  printf 'VENV_PY=%s\n' "$VENV_PY"
  printf 'LAUNCHER_PY=%s\n' "$LAUNCHER_PY"
  printf 'LOG_FILE=%s\n' "$LOG_FILE"
  printf 'REQUIREMENTS=%s\n' "$REQ_FILE"
  printf 'REQUIREMENTS_MARKER=%s\n' "$REQ_MARKER"
  exit 0
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  printf '%s\n' \
    "Запуск:   bash VOICE_SYNTEZ.command        (или двойной клик из Finder)" \
    "Пути:     bash VOICE_SYNTEZ.command --paths" \
    "Справка:  bash VOICE_SYNTEZ.command --help"
  exit 0
fi

echo "Каталог проекта: $PROJECT_DIR"

if [ ! -f "$LAUNCHER_PY" ]; then
  echo "Не найден вспомогательный модуль: $LAUNCHER_PY" >&2
  echo "Похоже, это копия без tools/ — возьмите полный каталог проекта." >&2
  exit 1
fi

# --- Python, которым можно выполнить помощник ----------------------------------
# Сам помощник — только стандартная библиотека, поэтому подойдёт любой Python 3;
# backend потом получит строго 3.11+ (см. find-python).
BOOT_PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    BOOT_PYTHON="$(command -v "$candidate")"
    break
  fi
done

if [ -z "$BOOT_PYTHON" ]; then
  echo "Python 3 не найден — запускать backend нечем." >&2
  echo "Установите Python: brew install python@3.11 (или с python.org), затем запустите снова." >&2
  exit 1
fi

# --- уже запущенный экземпляр --------------------------------------------------
# Второй инстанс нельзя: лимиты потоков и памяти считаются внутри процесса, и
# два процесса складываются. Замок держит живой backend (backend/main.py),
# поэтому проверяем именно его, а не занятость порта: на 8000 может висеть
# чужой проект.
LOCK_FILE="$PROJECT_DIR/.voice_syntez.lock"
RUNNING_PORT="$("$BOOT_PYTHON" "$LAUNCHER_PY" running-port --start "$PORT_BASE" 2>/dev/null || true)"
if [ -z "$RUNNING_PORT" ]; then
  if "$BOOT_PYTHON" "$LAUNCHER_PY" lock-held --path "$LOCK_FILE" >/dev/null 2>&1; then
    echo "Приложение уже запущено (держит .voice_syntez.lock), но порт ещё не отвечает."
    echo "Подождите несколько секунд и откройте http://127.0.0.1:${PORT_BASE} — второй запускать нельзя."
    exit 0
  fi
else
  echo "Приложение уже запущено: http://127.0.0.1:${RUNNING_PORT}"
  open "http://127.0.0.1:${RUNNING_PORT}" >/dev/null 2>&1 || true
  exit 0
fi

# --- Python 3.11+ --------------------------------------------------------------
if ! SYSTEM_PYTHON="$("$BOOT_PYTHON" "$LAUNCHER_PY" find-python)"; then
  exit 1
fi
echo "Python: $SYSTEM_PYTHON"

# --- ffmpeg --------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ВНИМАНИЕ: ffmpeg не найден — MP3-экспорт работать не будет."
  echo "          Установите его: brew install ffmpeg"
fi

# --- venv и зависимости --------------------------------------------------------
if [ ! -f "$REQ_FILE" ]; then
  echo "Не найден requirements.txt: $REQ_FILE" >&2
  exit 1
fi

INSTALL_DEPS=0
if [ ! -x "$VENV_PY" ]; then
  echo "Виртуальное окружение не найдено — создаю venv (первый запуск)…"
  if ! "$SYSTEM_PYTHON" -m venv "$VENV_DIR"; then
    echo "Не удалось создать venv в $VENV_DIR" >&2
    exit 1
  fi
  INSTALL_DEPS=1
elif "$BOOT_PYTHON" "$LAUNCHER_PY" needs-install --requirements "$REQ_FILE" --marker "$REQ_MARKER" >/dev/null 2>&1; then
  echo "requirements.txt изменился — обновляю зависимости…"
  INSTALL_DEPS=1
else
  echo "Зависимости актуальны — установка не требуется."
fi

if [ "$INSTALL_DEPS" = "1" ]; then
  echo "Ставлю зависимости (это может занять несколько минут; сеть нужна только к PyPI)…"
  "$VENV_PY" -m pip install --upgrade pip || echo "pip не обновился — продолжаю с текущим."
  if ! "$VENV_PY" -m pip install -r "$REQ_FILE"; then
    echo "Установка зависимостей не удалась. Проверьте сеть и запустите снова." >&2
    exit 1
  fi
  "$BOOT_PYTHON" "$LAUNCHER_PY" write-marker --requirements "$REQ_FILE" --marker "$REQ_MARKER" >/dev/null
fi

# --- свободный порт ------------------------------------------------------------
PORT="$("$BOOT_PYTHON" "$LAUNCHER_PY" free-port --start "$PORT_BASE" 2>/dev/null || true)"
if [ -z "$PORT" ]; then
  echo "Не нашёл свободный порт начиная с $PORT_BASE. Задайте другой: PORT=8100 bash VOICE_SYNTEZ.command" >&2
  exit 1
fi

# --- запуск --------------------------------------------------------------------
# Те же переменные, что в run.sh: предсказуемый хеш-сид и отключение телеметрии
# onnxruntime (без него RUAccent убивает процесс во время синтеза).
export PYTHONHASHSEED=0
export ORT_DISABLE_TELEMETRY=1

mkdir -p "$LOG_DIR"
echo "Логи:      $LOG_FILE"
echo "Дашборд:   http://127.0.0.1:${PORT}"
echo "Остановка: Ctrl+C"

SERVER_CMD=("$VENV_PY" -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT")
if command -v taskpolicy >/dev/null 2>&1; then
  SERVER_CMD=(taskpolicy -c utility "${SERVER_CMD[@]}")
fi

"${SERVER_CMD[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Хвост лога в терминале: иначе окно выглядит зависшим, пока грузится модель.
tail -n 0 -f "$LOG_FILE" &
TAIL_PID=$!

cleanup() {
  trap - INT TERM EXIT
  if kill -0 "$TAIL_PID" 2>/dev/null; then
    kill "$TAIL_PID" 2>/dev/null
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    # SIGINT — то же, что Ctrl+C для uvicorn. Но фоновый процесс в
    # неинтерактивном скрипте наследует SIGINT как SIG_IGN (bash так делает для
    # асинхронных команд без job control), поэтому сразу дублируем SIGTERM:
    # его uvicorn обрабатывает всегда и завершается так же аккуратно.
    kill -INT "$SERVER_PID" 2>/dev/null
    sleep 0.3
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID" 2>/dev/null
    fi
    waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null && [ "$waited" -lt 30 ]; do
      sleep 0.5
      waited=$((waited + 1))
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      # Последняя мера: процесс не отвечает на сигналы — не оставляем его жить.
      kill -KILL "$SERVER_PID" 2>/dev/null
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null
}
trap cleanup INT TERM EXIT

# Открываем браузер, когда сервер уже отвечает (или сразу, если он не поднялся —
# тогда диагноз будет в терминале и логе).
waited=0
while [ "$waited" -lt 20 ]; do
  if "$BOOT_PYTHON" "$LAUNCHER_PY" running-port --start "$PORT" --tries 1 --timeout 0.3 >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
  waited=$((waited + 1))
done
open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true

wait "$SERVER_PID"
