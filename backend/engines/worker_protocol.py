"""Контракт между бэкендом и процессом-воркером TTS.

Модуль общий для обеих сторон и намеренно ничего не импортирует, кроме
стандартной библиотеки: его читает и процесс бэкенда (там импорт torch стоит
секунды), и дочерний процесс (там torch появится позже и только по делу).

Здесь лежат три вещи, которые обязаны совпадать у родителя и ребёнка:

* имена запросов (`REQUEST_*`) и состояния воркера (`WORKER_STATE_*`);
* таксономия ошибок (`ERROR_*`): по ней задача получает `error_type`, а
  интерфейс — понятный текст вместо «что-то пошло не так»;
* разбор кода возврата процесса: падение нативной библиотеки приходит сюда
  отрицательным кодом (`-6` — SIGABRT), и отличить его от обычной ошибки
  модели можно только по знаку и номеру сигнала.

Плюс классы исключений: они поднимаются в процессе бэкенда, поэтому не должны
тянуть за собой ничего тяжёлого.
"""

from __future__ import annotations

import hashlib
import signal
import textwrap

# --- запросы ------------------------------------------------------------------
# Ровно один запрос в работе: очередь задач последовательная, а модель на MPS
# не любит параллельный инференс из двух потоков. Супервизор сериализует вызовы
# и не даёт отправить второй запрос, пока не пришёл ответ на первый.
REQUEST_LOAD = "load"
REQUEST_SYNTHESIZE = "synthesize"
REQUEST_UNLOAD = "unload"
REQUEST_PING = "ping"
REQUEST_SHUTDOWN = "shutdown"

REQUEST_KINDS = frozenset(
    {REQUEST_LOAD, REQUEST_SYNTHESIZE, REQUEST_UNLOAD, REQUEST_PING, REQUEST_SHUTDOWN}
)

# --- состояния воркера --------------------------------------------------------
# STARTING    — процесс запущен, но ещё не подтвердил готовность;
# IDLE        — жив и свободен (модель может быть не загружена: грузится лениво);
# BUSY        — выполняет запрос (в том числе загрузку модели);
# CRASHED     — процесс умер: код возврата и сигнал записаны в диагностику;
# RESTARTING  — поднимается замена после падения;
# STOPPED     — остановлен штатно (выгрузка, выключение приложения, простой);
# DEGRADED    — движок временно отключён: падений слишком много подряд.
WORKER_STATE_STARTING = "starting"
WORKER_STATE_IDLE = "idle"
WORKER_STATE_BUSY = "busy"
WORKER_STATE_CRASHED = "crashed"
WORKER_STATE_RESTARTING = "restarting"
WORKER_STATE_STOPPED = "stopped"
WORKER_STATE_DEGRADED = "degraded"

# Состояния, при которых процесс обязан быть жив. Нужны и супервизору (решать,
# поднимать ли замену), и `/api/status` (объяснять, чего ждать пользователю).
WORKER_STATE_ALIVE = frozenset({WORKER_STATE_STARTING, WORKER_STATE_IDLE, WORKER_STATE_BUSY})

# --- таксономия ошибок --------------------------------------------------------
# TTS_ERROR        — модель или движок вернул ошибку (процесс жив);
# WORKER_CRASH     — процесс воркера умер (сигнал, ненулевой код, потеря канала);
# WORKER_TIMEOUT   — ответа не дождались: воркер убит, чтобы не держать зависший MPS;
# CANCELLED        — отмена пользователем: это не ошибка движка;
# WATCHDOG         — задачу прервал watchdog по памяти (см. resource_guard);
# INTERRUPTED      — задача не пережила перезапуск приложения (см. recovery).
ERROR_TTS = "TTS_ERROR"
ERROR_WORKER_CRASH = "WORKER_CRASH"
ERROR_WORKER_TIMEOUT = "WORKER_TIMEOUT"
ERROR_CANCELLED = "CANCELLED"
ERROR_WATCHDOG = "WATCHDOG"
ERROR_INTERRUPTED = "INTERRUPTED"
# Записанное аудио не прошло проверку: пустой, обрезанный или нечитаемый файл
# (см. `audio_pipeline.validate_audio_file`). Не ошибка движка: причина обычно в
# диске или в прерванной записи, и совет пользователю другой.
ERROR_AUDIO = "AUDIO_ERROR"

# Заголовок «человеческого» текста для каждой категории. Формулировки на русском:
# они уходят в интерфейс как есть (см. job_queue).
ERROR_TITLES = {
    ERROR_TTS: "Ошибка движка синтеза",
    ERROR_WORKER_CRASH: "Процесс синтеза аварийно завершился",
    ERROR_WORKER_TIMEOUT: "Процесс синтеза не ответил вовремя",
    ERROR_CANCELLED: "Отменено",
    ERROR_WATCHDOG: "Прервано watchdog'ом",
    ERROR_INTERRUPTED: "Прервано перезапуском приложения",
    ERROR_AUDIO: "Аудио записано некорректно",
}

# Причина падения по сигналу — для диагностики и сообщения пользователю. Список
# короткий не по недосмотру: остальные сигналы для TTS-воркера не встречаются, а
# выдумывать текст для них — значит угадывать.
_SIGNAL_REASONS = {
    signal.SIGABRT: "аварийное завершение нативной библиотеки (SIGABRT)",
    signal.SIGSEGV: "нарушение доступа к памяти (SIGSEGV)",
    signal.SIGBUS: "ошибка шины памяти (SIGBUS)",
    signal.SIGKILL: "процесс убит ядром или пользователем (SIGKILL) — обычно нехватка памяти",
    signal.SIGTERM: "процесс завершён по сигналу (SIGTERM)",
    signal.SIGINT: "процесс прерван (SIGINT)",
    signal.SIGILL: "недопустимая инструкция (SIGILL)",
    signal.SIGFPE: "ошибка вычислений (SIGFPE)",
}

PREVIEW_LIMIT = 80


def describe_exit(returncode: int | None) -> dict:
    """Разбирает код возврата процесса в диагностику.

    `None` — процесс ещё жив: это не ошибка, а состояние. Отрицательный код —
    сигнал (`subprocess` отдаёт его со знаком минус), положительный — код выхода.
    """
    if returncode is None:
        return {"exit_code": None, "signal": None, "signal_name": None, "reason": "процесс жив"}
    if returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:  # экзотический сигнал: имя не выдумываем
            name = f"сигнал {number}"
        return {
            "exit_code": returncode,
            "signal": number,
            "signal_name": name,
            "reason": _SIGNAL_REASONS.get(number, f"процесс убит сигналом {name}"),
        }
    if returncode == 0:
        return {
            "exit_code": 0,
            "signal": None,
            "signal_name": None,
            "reason": "процесс завершился нормально",
        }
    return {
        "exit_code": returncode,
        "signal": None,
        "signal_name": None,
        "reason": f"процесс завершился с кодом {returncode}",
    }


def text_fingerprint(text: str, limit: int = PREVIEW_LIMIT) -> dict:
    """Отпечаток текста для логов и диагностики: длина, sha1 и короткая выжимка.

    Текст реплики в логи целиком не пишется: это пользовательский контент, и в
    логе он не нужен. Но при разборе падения важно понять, *на какой* реплике
    упало, и сверить это с тем, что человек видит в интерфейсе, — поэтому
    остаётся длина, хеш и первые слова одной строкой.
    """
    normalized = " ".join((text or "").split())
    return {
        "length": len(text or ""),
        "sha1": hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8],
        "preview": textwrap.shorten(normalized, width=limit, placeholder="…") if normalized else "",
    }


class WorkerFailure(RuntimeError):
    """Общая база отказов воркера: у каждой ошибки есть машинный `error_type`.

    Пайплайн и очередь не разбирают текст сообщения — они смотрят на `error_type`,
    поэтому таксономия живёт в одном месте, а не в регулярках по строкам.
    """

    error_type = ERROR_WORKER_CRASH

    def __init__(self, message: str, *, engine: str = "", details: dict | None = None) -> None:
        super().__init__(message)
        self.engine = engine
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        """Диагностика для API и БД: без текста реплик и без огромных traceback."""
        return {
            "error_type": self.error_type,
            "title": ERROR_TITLES.get(self.error_type, "Ошибка синтеза"),
            "message": str(self),
            "engine": self.engine,
            **self.details,
        }


class WorkerCrashError(WorkerFailure):
    """Процесс воркера умер: сигнал, код возврата или потеря канала."""

    error_type = ERROR_WORKER_CRASH


class WorkerTimeoutError(WorkerFailure):
    """Воркер не ответил за отведённое время; процесс убит, состояние неясно."""

    error_type = ERROR_WORKER_TIMEOUT


class WorkerUnavailableError(WorkerFailure):
    """Движок временно отключён после серии падений (состояние DEGRADED).

    Отдельный класс, а не общий `WorkerCrashError`: задача падает не из-за
    падения процесса, а потому что супервизор не даёт запускать новый, пока не
    истечёт пауза. Пользователю это разные советы — «повторите» против
    «подождите».
    """

    error_type = ERROR_WORKER_CRASH


class TtsEngineError(RuntimeError):
    """Ошибка внутри движка: процесс жив, модель ответила отказом.

    Приходит из дочернего процесса коротким текстом и усечённым traceback: в
    базе и в интерфейсе нужен первый, второй — только в логе.
    """

    error_type = ERROR_TTS

    def __init__(self, message: str, *, engine: str = "", traceback_text: str = "") -> None:
        super().__init__(message)
        self.engine = engine
        self.traceback_text = traceback_text

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "title": ERROR_TITLES[self.error_type],
            "message": str(self),
            "engine": self.engine,
        }
