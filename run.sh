#!/usr/bin/env bash
# Запуск TTS-дашборда: http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

# --- Лок против параллельного запуска -----------------------------------------
# Лимиты потоков и памяти действуют только внутри одного процесса, поэтому два
# живых инстанса суммируются и выводят нагрузку за все границы. PIDFILE указывает
# на сам uvicorn (ниже используется exec), так что проверка `kill -0` честная.
PIDFILE="${TTS_PIDFILE:-.voice_syntez.pid}"

if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Приложение уже запущено (PID $OLD_PID). Останови его или удали $PIDFILE, если это устаревший файл." >&2
    exit 1
  else
    echo "Найден устаревший pidfile (процесс $OLD_PID не существует), удаляю." >&2
    rm -f "$PIDFILE"
  fi
fi

echo $$ > "$PIDFILE"

# Фиксируем хеш-сид до старта интерпретатора. Сам по себе этого мало: f5_tts
# перезаписывает PYTHONHASHSEED внутри процесса (backend/tts_engine.py передаёт
# ему корректный сид), но стартовое значение делает поведение предсказуемым.
export PYTHONHASHSEED=0

# onnxruntime (нужен RUAccent) при старте пишет телеметрию в
# ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime. Если эта
# запись запрещена (например, песочницей IDE), процесс убивается прямо во время
# синтеза — в браузере это выглядит как «Failed to fetch». Телеметрия нам не
# нужна, выключаем.
export ORT_DISABLE_TELEMETRY=1

if [ ! -d venv ]; then
  echo "venv не найден — создаю и ставлю зависимости (это займёт время)…"
  python3.11 -m venv venv
  ./venv/bin/python -m pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi

# shellcheck disable=SC1091
source venv/bin/activate

# Порт по умолчанию 8000, но на нём часто висит другой проект — ищем свободный.
port_free() {
  python - "$1" <<'PY'
import socket
import sys

sock = socket.socket()
# Как uvicorn: без SO_REUSEADDR порт в TIME_WAIT после рестарта считается занятым
# и приложение молча уезжает на следующий порт.
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

PORT="${PORT:-8000}"
while ! port_free "$PORT"; do
  echo "Порт $PORT занят — пробую $((PORT + 1))"
  PORT=$((PORT + 1))
done

echo "Дашборд: http://localhost:${PORT}"

# Понижаем приоритет процесса: при перегрузке система остаётся отзывчивой для
# остальных приложений (в macOS нет cgroups, taskpolicy — ближайший аналог).
if command -v taskpolicy >/dev/null 2>&1; then
  exec taskpolicy -c utility uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
fi
exec uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
