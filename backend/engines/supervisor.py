"""Супервизор процессов-воркеров: запуск, надзор, перезапуск после падения.

Один воркер на движок (`f5`, `xtts`, `xtts-banana`) — ровно как раньше был один
тёплый экземпляр движка на процесс. Отличие в том, что теперь падение движка
видно снаружи: `subprocess` отдаёт код возврата и сигнал, а канал обмена при
смерти процесса закрывается, и это отличается от «модель вернула ошибку».

Что делает супервизор:

* поднимает процесс по требованию (`ensure`) и держит трубу запросов;
* сериализует запросы: очередь задач последовательная, но пайплайн синтезирует
  в отдельном потоке, а `unload()` приходит из событийного цикла — без лока
  «выгрузка против идущего инференса» уже была бы гонкой;
* замечает падение (EOF в канале, код возврата, потеря процесса между
  запросами) и записывает диагностику: pid, код, сигнал, причина;
* поднимает замену сразу после падения — со следующим запросом задача должна
  работать, а не обнаруживать мёртвый процесс;
* считает отказы в окне и переводит движок в `DEGRADED`, если они повторяются:
  иначе «падает на каждой реплике» превратилось бы в бесконечный цикл
  перезапусков и бессмысленную нагрузку;
* отдаёт состояние для `/api/status`, не заглядывая в модель и не ожидая
  текущий запрос: лёгкие роуты обязаны отвечать, даже когда воркер мёртв.

Воркер не «присматривает» за родителем отдельно: канал запросов закрывается
вместе с процессом бэкенда, ребёнок получает EOF и выходит — осиротевший
процесс с моделью в памяти не остаётся даже при `kill -9` бэкенда.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any

from .. import config
from . import worker_protocol as proto

logger = logging.getLogger("tts.worker.supervisor")

# Сколько ждать добровольного выхода после `shutdown`/`SIGTERM`. Короткий шаг:
# процесс с моделью, который «задумался» на выходе, всё равно не сохраняет
# ничего полезного, а очередь уже остановлена.
TERMINATE_GRACE_SEC = 5.0
# Ожидание кода возврата уже упавшего процесса: `poll()` может вернуть None
# доли миллисекунды после смерти. Дольше ждать незачем — диагностика важна, но
# не за счёт задержки ответа пользователю.
EXIT_WAIT_SEC = 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class WorkerHandle:
    """Один воркер: процесс, каналы и накопленная диагностика.

    Два лока решают разные задачи, и разделены они не для красоты.
    `lock` сериализует длинные операции (запрос, выгрузку) — его нельзя брать
    там, где ждут немедленного ответа: `/api/status` и `abort_current` приходят
    снаружи и обязаны работать, пока запрос висит. Состояние же защищает
    короткий `state_lock`, и только его берут все остальные.
    """

    engine_id: str
    state: str = proto.WORKER_STATE_STOPPED
    process: subprocess.Popen | None = None
    request: Connection | None = None
    response: Connection | None = None
    started_monotonic: float | None = None
    started_at: str | None = None
    busy_since: float | None = None
    restarts: int = 0
    failures: int = 0
    failure_times: list[float] = field(default_factory=list)
    last_failure: dict | None = None
    degraded_until: float | None = None
    # Отказ, уже описанный вызывающим (таймаут пайплайна убил воркер): висящий
    # запрос обязан вернуть именно его, а не записать второе падение.
    pending_failure: proto.WorkerFailure | None = None
    stop_reason: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    drain_threads: list[threading.Thread] = field(default_factory=list)

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def busy_sec(self) -> float | None:
        if self.busy_since is None:
            return None
        return round(time.monotonic() - self.busy_since, 1)

    def to_dict(self) -> dict:
        """Состояние для `/api/status`: только то, что не требует ожидания."""
        with self.state_lock:
            return {
                "engine": self.engine_id,
                "state": self.state,
                "pid": self.pid,
                "alive": self.alive(),
                "started_at": self.started_at,
                "busy_sec": self.busy_sec(),
                "restarts": self.restarts,
                "failures": self.failures,
                "degraded_until": (
                    _now_iso_from(self.degraded_until) if self.degraded_until else None
                ),
                "last_failure": self.last_failure,
                "stop_reason": self.stop_reason or None,
            }


def _now_iso_from(monotonic_at: float | None) -> str | None:
    """Переводит момент `time.monotonic()` в настенное время для интерфейса."""
    if monotonic_at is None:
        return None
    delta = monotonic_at - time.monotonic()
    return datetime.fromtimestamp(time.time() + delta, timezone.utc).isoformat(
        timespec="seconds"
    )


class WorkerSupervisor:
    """Все воркеры процесса бэкенда плюс их диагностика."""

    def __init__(self) -> None:
        self._handles: dict[str, WorkerHandle] = {}
        self._lock = threading.RLock()
        self._listeners: dict[str, list[Any]] = {}
        self._stopping = False

    # -- реестр хэндлов --------------------------------------------------------
    def handle(self, engine_id: str) -> WorkerHandle:
        """Запись о воркере. Процесс здесь не запускается — только при запросе."""
        with self._lock:
            handle = self._handles.get(engine_id)
            if handle is None:
                handle = WorkerHandle(engine_id=engine_id)
                self._handles[engine_id] = handle
            return handle

    def peek(self, engine_id: str) -> WorkerHandle | None:
        with self._lock:
            return self._handles.get(engine_id)

    def add_failure_listener(self, engine_id: str, listener: Any) -> None:
        """Подписка на отказы: движок-прокси по ней помечает себя `failed`.

        Слушатели вызываются из потока, который заметил отказ, и не должны
        бросать исключения — их ошибки только логируются.
        """
        with self._lock:
            self._listeners.setdefault(engine_id, []).append(listener)

    def status(self) -> list[dict]:
        """Состояние всех известных воркеров (без создания новых)."""
        with self._lock:
            handles = list(self._handles.values())
        return [handle.to_dict() for handle in sorted(handles, key=lambda h: h.engine_id)]

    # -- запросы ---------------------------------------------------------------
    def request(self, engine_id: str, payload: dict, *, timeout: float) -> dict:
        """Отправляет запрос и возвращает ответ воркера.

        Отказы приходят исключениями: падение процесса (`WorkerCrashError`),
        отсутствие ответа (`WorkerTimeoutError`), ошибка модели
        (`TtsEngineError`). Первые два записываются в диагностику и поднимают
        замену, третье — нет: процесс жив, следующий запрос имеет смысл.
        """
        handle = self.handle(engine_id)
        with handle.lock:
            self._ensure_ready(handle)
            request_conn = handle.request
            response_conn = handle.response
            if request_conn is None or response_conn is None:
                raise self._register_failure(
                    handle, message="канал обмена с воркером не открыт"
                )
            with handle.state_lock:
                handle.state = proto.WORKER_STATE_BUSY
                handle.busy_since = time.monotonic()
                handle.pending_failure = None
            try:
                request_conn.send(payload)
            except (OSError, ValueError) as exc:
                raise self._register_failure(
                    handle, message=f"канал запросов закрыт ({exc})"
                ) from exc
            if not response_conn.poll(timeout):
                raise self._register_failure(
                    handle,
                    error_type=proto.ERROR_WORKER_TIMEOUT,
                    message=f"воркер не ответил за {timeout:.0f} с",
                    kill=True,
                )
            try:
                answer = response_conn.recv()
            except EOFError:
                raise self._register_failure(
                    handle, message="процесс воркера завершился во время синтеза"
                ) from None
            except (OSError, ValueError) as exc:
                raise self._register_failure(
                    handle, message=f"канал ответов закрыт ({exc})"
                ) from exc
            finally:
                with handle.state_lock:
                    handle.busy_since = None
                    if handle.state == proto.WORKER_STATE_BUSY:
                        handle.state = proto.WORKER_STATE_IDLE
        if not isinstance(answer, dict) or not answer.get("ok"):
            raise self._engine_error(engine_id, answer)
        return answer

    def request_ping(self, engine_id: str, timeout: float = 5.0) -> dict:
        """Короткий пинг: используется тестами и диагностикой."""
        return self.request(engine_id, {"kind": proto.REQUEST_PING}, timeout=timeout)

    # -- подъём, остановка, отказ ----------------------------------------------
    def _ensure_ready(self, handle: WorkerHandle) -> None:
        """Гарантирует живой процесс перед запросом."""
        with handle.state_lock:
            state = handle.state
            degraded_until = handle.degraded_until
        if state == proto.WORKER_STATE_DEGRADED:
            if degraded_until is not None and time.monotonic() < degraded_until:
                left = max(degraded_until - time.monotonic(), 0.0)
                raise proto.WorkerUnavailableError(
                    f"Движок «{handle.engine_id}» временно отключён после "
                    f"{handle.failures} аварийных завершений подряд. "
                    f"Повторите через {left:.0f} с или перезапустите приложение.",
                    engine=handle.engine_id,
                    details={"degraded": True, "retry_in_sec": round(left, 1)},
                )
            # Пауза истекла: одна попытка. Счётчик окна чистим, иначе первый же
            # новый отказ снова вернул бы DEGRADED и движок не поднялся бы никогда.
            logger.info("Воркер %s: пауза после серии падений истекла, пробую снова", handle.engine_id)
            with handle.state_lock:
                handle.state = proto.WORKER_STATE_STOPPED
                handle.degraded_until = None
                handle.failure_times.clear()
        if handle.alive():
            return
        if handle.process is not None:
            # Процесс умер между запросами: об этом никто не узнал, потому что
            # падать в простое он может от чужого OOM-killer'а или от сигнала
            # снаружи. Записываем отказ — иначе диагностика потеряла бы причину,
            # а счётчик падений не заметил бы циклическую смерть.
            failure = self._register_failure(
                handle, message="процесс воркера завершился в простое", restart=False
            )
            if handle.state == proto.WORKER_STATE_DEGRADED:
                raise proto.WorkerUnavailableError(
                    f"Движок «{handle.engine_id}» временно отключён после серии падений.",
                    engine=handle.engine_id,
                    details={"degraded": True, **failure.details},
                )
        self._spawn(handle)

    def _spawn(self, handle: WorkerHandle) -> None:
        """Запускает процесс воркера и подключается к его каналам."""
        if self._stopping:
            raise proto.WorkerUnavailableError(
                "Приложение выключается: новые задачи синтеза не запускаются.",
                engine=handle.engine_id,
            )
        with handle.state_lock:
            handle.state = proto.WORKER_STATE_STARTING
        # Каналы создаёт родитель: дескрипторы передаются ребёнку через
        # `pass_fds`, поэтому воркер не наследует ничего лишнего (в том числе
        # не наследует инициализированный MPS — он поднимает его сам).
        request_recv, request_send = Pipe(duplex=False)
        response_recv, response_send = Pipe(duplex=False)
        command = [
            sys.executable,
            "-m",
            config.WORKER_MODULE,
            handle.engine_id,
            str(request_recv.fileno()),
            str(response_send.fileno()),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(config.BASE_DIR),
                env=self._child_env(),
                pass_fds=(request_recv.fileno(), response_send.fileno()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            for connection in (request_recv, request_send, response_recv, response_send):
                connection.close()
            with handle.state_lock:
                handle.state = proto.WORKER_STATE_STOPPED
            raise proto.WorkerUnavailableError(
                f"Не удалось запустить процесс синтеза: {exc}", engine=handle.engine_id
            ) from exc
        # Свои копии «чужих» концов закрываем: иначе EOF при смерти ребёнка не
        # наступит никогда — дескриптор останется открытым у родителя.
        request_recv.close()
        response_send.close()
        with handle.state_lock:
            handle.process = process
            handle.request = request_send
            handle.response = response_recv
            handle.started_monotonic = time.monotonic()
            handle.started_at = _now_iso()
            handle.state = proto.WORKER_STATE_IDLE
            handle.pending_failure = None
            handle.stop_reason = ""
            handle.degraded_until = None
        self._start_drains(handle, process)
        logger.info("Воркер %s запущен: pid %s", handle.engine_id, process.pid)

    def _start_drains(self, handle: WorkerHandle, process: subprocess.Popen) -> None:
        """Перенаправляет stdout/stderr воркера в общий лог с префиксом.

        Читать обязательно: если оставить потоки неразобранными, буфер трубы
        заполнится и воркер встанет на `print` посреди инференса. Префикс нужен,
        потому что после падения в логе должно быть видно, что именно печатал
        упавший процесс.
        """
        with handle.state_lock:
            handle.drain_threads = []
        for stream, level in ((process.stdout, logging.INFO), (process.stderr, logging.INFO)):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._drain_stream,
                args=(handle.engine_id, process.pid, stream, level),
                name=f"worker-drain-{handle.engine_id}",
                daemon=True,
            )
            with handle.state_lock:
                handle.drain_threads.append(thread)
            thread.start()

    @staticmethod
    def _drain_stream(engine_id: str, pid: int, stream: Any, level: int) -> None:
        for line in stream:
            text = line.rstrip("\n")
            if not text:
                continue
            logger.log(level, "[%s pid=%s] %s", engine_id, pid, text)
        try:
            stream.close()
        except OSError:  # поток уже закрыт вместе с процессом
            pass

    def unload(self, engine_id: str, *, reason: str = "выгрузка") -> bool:
        """Останавливает воркер: модель живёт в нём, значит уходит вместе с ним.

        Возвращает `True`, если процесс был жив и остановлен. Ожидание лока
        запроса здесь намеренное: убить процесс посреди синтеза — значит уронить
        чужую задачу, а `SynthesisEngine.unload()` и так отказывается выгружать
        занятый движок.
        """
        handle = self.peek(engine_id)
        if handle is None:
            return False
        with handle.lock:
            process = handle.process
            if process is None or process.poll() is not None:
                self._close_channels(handle)
                with handle.state_lock:
                    handle.process = None
                    handle.state = proto.WORKER_STATE_STOPPED
                    handle.stop_reason = reason
                return False
            with handle.state_lock:
                handle.state = proto.WORKER_STATE_STOPPED
            try:
                if handle.request is not None:
                    handle.request.send({"kind": proto.REQUEST_SHUTDOWN})
            except (OSError, ValueError):
                pass
            if not self._wait_exit(process, config.WORKER_SHUTDOWN_GRACE_SEC):
                logger.warning(
                    "Воркер %s (pid %s) не вышел за %.0f с — завершаю принудительно",
                    engine_id, process.pid, config.WORKER_SHUTDOWN_GRACE_SEC,
                )
                self._kill(process)
            self._close_channels(handle)
            with handle.state_lock:
                handle.process = None
                handle.stop_reason = reason
                handle.busy_since = None
            logger.info("Воркер %s остановлен (%s): pid %s", engine_id, reason, process.pid)
            return True

    def abort_current(self, engine_id: str, reason: str = "таймаут") -> bool:
        """Убивает воркер, который не ответил вовремя.

        Вызывается из событийного цикла, пока висящий запрос держит `lock`,
        поэтому здесь берётся только `state_lock`: иначе ожидание лока превратило
        бы таймаут в зависание. Отказ описывается один раз — висящий запрос
        подберёт его через `pending_failure` и не запишет второе падение.
        """
        handle = self.peek(engine_id)
        if handle is None:
            return False
        with handle.state_lock:
            if handle.state != proto.WORKER_STATE_BUSY:
                return False
            failure = proto.WorkerTimeoutError(
                f"синтез прерван: {reason}",
                engine=engine_id,
                details=self._exit_info(handle) | {"abort_reason": reason},
            )
            handle.pending_failure = failure
        self._register_failure(handle, failure=failure, message=str(failure), kill=True)
        return True

    def _register_failure(
        self,
        handle: WorkerHandle,
        *,
        message: str,
        error_type: str = proto.ERROR_WORKER_CRASH,
        failure: proto.WorkerFailure | None = None,
        kill: bool = False,
        restart: bool = True,
    ) -> proto.WorkerFailure:
        """Записывает отказ и возвращает исключение для `raise`.

        Единая точка: и падение в запросе, и смерть в простое, и таймаут проходят
        здесь, поэтому счётчик окна, состояние, лог и перезапуск не могут
        разойтись между ветками.
        """
        with handle.state_lock:
            pending = handle.pending_failure
        if pending is not None and failure is None:
            return pending
        if kill:
            self._kill(handle.process)
        info = self._exit_info(handle)
        if failure is None:
            text = f"{message} ({info['reason']})" if info.get("reason") else message
            failure = (
                proto.WorkerTimeoutError(text, engine=handle.engine_id, details=info)
                if error_type == proto.ERROR_WORKER_TIMEOUT
                else proto.WorkerCrashError(text, engine=handle.engine_id, details=info)
            )
        else:
            failure.details.update(info)
        self._close_channels(handle)
        # Умерший процесс забываем сразу: его pid уже записан в диагностику, а
        # оставленный объект позже выглядел бы как «процесс умер в простое» и
        # записывал бы второй отказ по тому же падению.
        with handle.state_lock:
            handle.process = None
        now = time.monotonic()
        with handle.state_lock:
            handle.failures += 1
            handle.failure_times = [
                moment
                for moment in handle.failure_times
                if now - moment <= config.WORKER_CRASH_WINDOW_SEC
            ]
            handle.failure_times.append(now)
            handle.busy_since = None
            handle.last_failure = {
                **failure.to_dict(),
                "at": _now_iso(),
                "pid": info.get("pid"),
                "exit_code": info.get("exit_code"),
                "signal": info.get("signal"),
                "signal_name": info.get("signal_name"),
                "reason": info.get("reason"),
                "state": proto.WORKER_STATE_DEGRADED
                if len(handle.failure_times) >= config.WORKER_CRASH_LIMIT
                else proto.WORKER_STATE_CRASHED,
            }
            degraded = len(handle.failure_times) >= config.WORKER_CRASH_LIMIT
            if degraded:
                handle.state = proto.WORKER_STATE_DEGRADED
                handle.degraded_until = now + config.WORKER_DEGRADED_COOLDOWN_SEC
            else:
                handle.state = proto.WORKER_STATE_CRASHED
        if degraded:
            logger.error(
                "Воркер %s: %d отказа за %.0f с — движок отключён на %.0f с (%s)",
                handle.engine_id, len(handle.failure_times), config.WORKER_CRASH_WINDOW_SEC,
                config.WORKER_DEGRADED_COOLDOWN_SEC, failure,
            )
        else:
            logger.error(
                "Воркер %s: %s [pid=%s, код=%s, сигнал=%s]",
                handle.engine_id, failure, info.get("pid"), info.get("exit_code"),
                info.get("signal_name"),
            )
        self._notify(handle.engine_id, failure)
        if restart and not degraded:
            self._restart(handle)
        return failure

    def _restart(self, handle: WorkerHandle) -> None:
        """Поднимает замену сразу после падения.

        Сразу, а не «при следующем запросе»: задача-повтор приходит через
        миллисекунды, и ей нужен живой процесс, а `/api/status` должен успеть
        показать `restarting`, а не `crashed` до конца времён.
        """
        if self._stopping:
            return
        with handle.state_lock:
            handle.state = proto.WORKER_STATE_RESTARTING
            handle.restarts += 1
            restarts = handle.restarts
        try:
            self._spawn(handle)
        except proto.WorkerFailure as exc:
            logger.error("Воркер %s: замена не поднялась (%s)", handle.engine_id, exc)
            return
        logger.warning("Воркер %s перезапущен после падения (перезапуск №%d)", handle.engine_id, restarts)

    def _notify(self, engine_id: str, failure: proto.WorkerFailure) -> None:
        with self._lock:
            listeners = list(self._listeners.get(engine_id, ()))
        for listener in listeners:
            try:
                listener(failure)
            except Exception as exc:  # noqa: BLE001 — слушатель не важнее отказа
                logger.warning("Слушатель отказов %s упал: %s", engine_id, exc)

    def _engine_error(self, engine_id: str, answer: Any) -> Exception:
        """Ошибка модели: процесс жив, это не падение воркера."""
        payload = answer if isinstance(answer, dict) else {}
        message = str(payload.get("error") or "движок синтеза вернул пустой ответ")
        traceback_text = str(payload.get("traceback") or "")
        if traceback_text:
            logger.error("Движок %s: %s\n%s", engine_id, message, traceback_text)
        return proto.TtsEngineError(message, engine=engine_id, traceback_text=traceback_text)

    # -- служебное -------------------------------------------------------------
    def _exit_info(self, handle: WorkerHandle) -> dict:
        process = handle.process
        if process is None:
            return proto.describe_exit(None) | {"pid": None}
        code = process.poll()
        if code is None:
            try:
                code = process.wait(timeout=EXIT_WAIT_SEC)
            except subprocess.TimeoutExpired:
                code = None
        return proto.describe_exit(code) | {"pid": process.pid}

    def _close_channels(self, handle: WorkerHandle) -> None:
        with handle.state_lock:
            request, response = handle.request, handle.response
            handle.request = None
            handle.response = None
        for connection in (request, response):
            if connection is None:
                continue
            try:
                connection.close()
            except OSError:
                pass

    def _wait_exit(self, process: subprocess.Popen, grace: float) -> bool:
        try:
            process.wait(timeout=grace)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _kill(self, process: subprocess.Popen | None) -> None:
        """Гарантированно завершает процесс: `SIGTERM`, затем `SIGKILL`."""
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            return
        if not self._wait_exit(process, TERMINATE_GRACE_SEC):
            try:
                process.kill()
            except OSError:
                return
            self._wait_exit(process, TERMINATE_GRACE_SEC)

    def _child_env(self) -> dict:
        """Окружение воркера: маркер потомка и пути импорта родителя.

        `sys.path` родителя передаётся целиком осознанно: тесты подменяют движок
        фабрикой из своих модулей (`TTS_WORKER_ENGINE_FACTORY`), а pytest кладёт
        каталог тестов именно в `sys.path`. Без этого шва проверить изоляцию без
        настоящей модели было бы нечем.
        """
        env = os.environ.copy()
        env.pop(config.WORKER_CHILD_ENV, None)
        env["PYTHONUNBUFFERED"] = "1"
        paths = [str(config.BASE_DIR)]
        paths.extend(entry for entry in sys.path if entry)
        existing = env.get("PYTHONPATH")
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        return env

    def stop_all(self, grace: float | None = None) -> int:
        """Останавливает все воркеры. Возвращает число живых, которые были остановлены.

        `_stopping` выставляется до остановки: падение воркера в этот момент не
        должно поднимать замену — иначе выключение приложения оставляло бы за
        собой свежезапущенный процесс с моделью.
        """
        self._stopping = True
        with self._lock:
            handles = list(self._handles.values())
        stopped = 0
        for handle in handles:
            try:
                if self.unload(handle.engine_id, reason="выключение приложения"):
                    stopped += 1
            except Exception as exc:  # noqa: BLE001 — выключение важнее аккуратности
                logger.warning("Воркер %s не остановился штатно: %s", handle.engine_id, exc)
                self._kill(handle.process)
        if stopped:
            logger.info("Остановлено воркеров: %d", stopped)
        return stopped

    def reset_after_stop(self) -> None:
        """Возвращает супервизор в рабочее состояние (нужно тестам и рестарту)."""
        self._stopping = False


_supervisor: WorkerSupervisor | None = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> WorkerSupervisor:
    """Единственный супервизор на процесс бэкенда."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is None:
            _supervisor = WorkerSupervisor()
            # Страховка на случай, если приложение завершится без lifespan
            # (падение теста, аварийный выход): процессы с моделью не должны
            # переживать родителя. Короткая пауза — выключение важнее вежливости.
            atexit.register(_stop_at_exit, _supervisor)
        return _supervisor


def _stop_at_exit(supervisor: WorkerSupervisor) -> None:
    # На выходе логировать уже некуда, но и падать нельзя: исключение в atexit
    # печатает traceback поверх вывода приложения и пугает пользователя.
    try:
        supervisor.stop_all(grace=1.0)
    except Exception:  # noqa: BLE001, S110 — см. комментарий выше
        pass
