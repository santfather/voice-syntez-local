#!/usr/bin/env bash
# Единственный лаунчер TTS-дашборда: http://localhost:8000
#
# Здесь живёт всё, что нужно для запуска: venv и зависимости, свободный порт,
# замок единственного инстанса, переменные окружения и `exec uvicorn`.
# `VOICE_SYNTEZ.command` — только обёртка для двойного клика из Finder (показать
# вывод, открыть браузер, остановить по Ctrl+C) и зовёт этот скрипт: о запуске
# должен быть один источник правды, иначе две копии логики расходятся.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_DIR="$PWD"

VENV_DIR="$PROJECT_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
LAUNCHER_PY="$PROJECT_DIR/tools/launcher.py"
REQ_FILE="$PROJECT_DIR/requirements.txt"
REQ_MARKER="$VENV_DIR/.requirements.sha256"
PORT_BASE="${PORT:-8000}"

# --- журнал лаунчера ----------------------------------------------------------
# Вывод лаунчера дублируется в общий журнал: падения до старта Python (нет
# интерпретатора, не встали зависимости, занят порт, чужой pidfile) иначе остались
# бы только в терминале (F-L2). `tee` в каждом потоке: сообщения и остаются на
# экране, и не съезжают в один поток — stderr остаётся stderr.
# Перед `exec uvicorn` вывод возвращается на терминал: журнал с этого момента ведёт
# само приложение, обработчиком с ротацией, и второй писатель дал бы в файле
# двойные строки.
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/voice_syntez.log"
mkdir -p "$LOG_DIR"
exec 3>&1 4>&2
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

# --- Лок против параллельного запуска -----------------------------------------
# Лимиты потоков и памяти действуют только внутри одного процесса, поэтому два
# живых инстанса суммируются и выводят нагрузку за все границы. PIDFILE указывает
# на сам uvicorn (ниже используется exec), так что проверка `kill -0` честная.
# Второй замок — `.voice_syntez.lock` внутри backend: этот держится ядром и
# закрывает запуск в обход скрипта (см. _acquire_instance_lock).
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

# --- Python, которым можно выполнить помощник ----------------------------------
# Помощник — только стандартная библиотека, поэтому подойдёт любой Python 3;
# backend потом получит строго 3.11+ (см. find-python). Нужен до venv: без него
# нечем создать сам venv.
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

if [ ! -f "$LAUNCHER_PY" ]; then
  echo "Не найден вспомогательный модуль: $LAUNCHER_PY" >&2
  exit 1
fi

# --- venv и зависимости -------------------------------------------------------
# Хеш requirements.txt, а не mtime: pip сам трогает файлы в venv, и по времени
# повторный запуск переустанавливал бы гигабайты каждый раз.
INSTALL_DEPS=0
if [ ! -x "$VENV_PY" ]; then
  echo "venv не найден — создаю и ставлю зависимости (это займёт время)…"
  SYSTEM_PYTHON="$("$BOOT_PYTHON" "$LAUNCHER_PY" find-python)" || exit 1
  "$SYSTEM_PYTHON" -m venv "$VENV_DIR" || exit 1
  INSTALL_DEPS=1
elif "$BOOT_PYTHON" "$LAUNCHER_PY" needs-install --requirements "$REQ_FILE" --marker "$REQ_MARKER" >/dev/null 2>&1; then
  echo "requirements.txt изменился — обновляю зависимости…"
  INSTALL_DEPS=1
fi

if [ "$INSTALL_DEPS" = "1" ]; then
  "$VENV_PY" -m pip install --upgrade pip || echo "pip не обновился — продолжаю с текущим."
  if ! "$VENV_PY" -m pip install -r "$REQ_FILE"; then
    echo "Установка зависимостей не удалась. Проверьте сеть и запустите снова." >&2
    exit 1
  fi
  "$BOOT_PYTHON" "$LAUNCHER_PY" write-marker --requirements "$REQ_FILE" --marker "$REQ_MARKER" >/dev/null
fi

# shellcheck disable=SC1091
source venv/bin/activate

# --- свободный порт ------------------------------------------------------------
# Порт по умолчанию 8000, но на нём часто висит другой проект — ищем свободный.
PORT="$("$BOOT_PYTHON" "$LAUNCHER_PY" free-port --start "$PORT_BASE" 2>/dev/null)" || {
  echo "Не нашёл свободный порт начиная с $PORT_BASE. Задайте другой: PORT=8100 ./run.sh" >&2
  exit 1
}
if [ "$PORT" != "$PORT_BASE" ]; then
  echo "Порт $PORT_BASE занят — беру $PORT"
fi

echo "Дашборд: http://localhost:${PORT}"

# Дальше журнал ведёт приложение (F-F1), а вывод uvicorn идёт на терминал как есть.
exec 1>&3 2>&4
exec 3>&- 4>&-

# Понижаем приоритет процесса: при перегрузке система остаётся отзывчивой для
# остальных приложений (в macOS нет cgroups, taskpolicy — ближайший аналог).
if command -v taskpolicy >/dev/null 2>&1; then
  exec taskpolicy -c utility uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
fi
exec uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
