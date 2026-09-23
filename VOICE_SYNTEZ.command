#!/bin/bash
# Лёгкий лаунчер TTS-дашборда для macOS: двойной клик по этому файлу из Finder.
#
# Сам запуск живёт в run.sh — есть один источник правды о том, как поднимается
# backend (venv, зависимости, порт, замок, uvicorn). Здесь только то, ради чего
# нужен Finder'у отдельный файл:
#   1. находит каталог проекта независимо от текущего каталога;
#   2. предупреждает, если не найден ffmpeg (без него нет MP3-экспорта);
#   3. запускает run.sh и показывает его вывод в терминале (журнал ведёт сам
#      run.sh и приложение: `logs/voice_syntez.log`);
#   4. открывает браузер на том порту, который выбрал run.sh;
#   5. по Ctrl+C останавливает сервер и не оставляет процессов.
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
RUN_SCRIPT="$PROJECT_DIR/run.sh"
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
  printf 'RUN_SCRIPT=%s\n' "$RUN_SCRIPT"
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

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Не найден скрипт запуска: $RUN_SCRIPT" >&2
  echo "Похоже, это копия без run.sh — возьмите полный каталог проекта." >&2
  exit 1
fi

if [ ! -f "$LAUNCHER_PY" ]; then
  echo "Не найден вспомогательный модуль: $LAUNCHER_PY" >&2
  echo "Похоже, это копия без tools/ — возьмите полный каталог проекта." >&2
  exit 1
fi

# --- Python, которым можно выполнить помощник ----------------------------------
# Он нужен, чтобы узнать порт уже поднятого приложения (см. ниже): run.sh выбирает
# порт сам, и из лога его не всегда видно раньше, чем сервер ответит.
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

# --- ffmpeg --------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ВНИМАНИЕ: ffmpeg не найден — MP3-экспорт работать не будет."
  echo "          Установите его: brew install ffmpeg"
fi

# --- запуск --------------------------------------------------------------------
mkdir -p "$LOG_DIR"
echo "Логи:      $LOG_FILE"
echo "Остановка: Ctrl+C"

# run.sh сам поднимает venv, ставит зависимости (по хешу requirements.txt),
# выбирает свободный порт и делает exec uvicorn. Журнал он ведёт сам (вместе с
# приложением), поэтому перенаправлять его вывод в тот же файл не нужно: строки
# дублировались бы, а журнал получал бы двух писателей. PID остаётся тем же (exec),
# поэтому kill в cleanup бьёт по самому серверу.
SERVER_CMD=(bash "$RUN_SCRIPT")
"${SERVER_CMD[@]}" &
SERVER_PID=$!

cleanup() {
  trap - INT TERM EXIT
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

# Порт выбирает run.sh, поэтому его нужно обнаружить опросом /api/status — так же
# узнаётся уже запущенный экземпляр (в этом случае run.sh сразу выйдет с pidfile-
# ошибкой, а браузер всё равно откроется на живом порту).
RUNNING_PORT=""
waited=0
while [ "$waited" -lt 40 ]; do
  RUNNING_PORT="$("$BOOT_PYTHON" "$LAUNCHER_PY" running-port --start "$PORT_BASE" --timeout 0.3 2>/dev/null || true)"
  if [ -n "$RUNNING_PORT" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
  waited=$((waited + 1))
done

# Если сервер не поднялся — открываем базовый порт: диагноз уже в терминале и логе.
open "http://127.0.0.1:${RUNNING_PORT:-$PORT_BASE}" >/dev/null 2>&1 || true

wait "$SERVER_PID"
