"""Процесс-воркер TTS: модель живёт здесь, а не в процессе бэкенда.

    python -m backend.engines.worker_process <engine_id> <fd запросов> <fd ответов>

Зачем отдельный процесс. Инференс F5-TTS и XTTS идёт через нативные библиотеки
(Metal/MPS, torch, ffmpeg-подобные обёртки), и их падение — SIGABRT, SIGSEGV,
SIGBUS или тихое убийство по памяти — уносит с собой весь процесс Python. Пока
модель жила в бэкенде, это означало падение всего приложения вместе с базой, уже
готовыми репликами и интерфейсом. Теперь падать может только воркер: бэкенд
видит код возврата, объясняет причину и поднимает замену.

Почему `Popen`, а не `fork`/`multiprocessing.Process`. На macOS `fork` после
инициализации MPS — известный источник падений: дочерний процесс наследует
состояние Metal, которое принадлежит родителю. Здесь процесс стартует заново
(`python -m ...`), ничего не наследует и поднимает MPS сам, при первом
обращении к модели.

Как устроен обмен. Родитель передаёт два дескриптора каналов (запросы и ответы)
через `pass_fds`; по ним идёт pickle с автоматической разметкой длины
(`multiprocessing.connection`). `stdin` закрыт, `stdout`/`stderr` — обычные
потоки: их читает родитель и пишет в общий лог с префиксом движка, поэтому
болтливые `print` из f5_tts не теряются и не превращаются в дедлок.

Смерть родителя воркер переживает ровно настолько, чтобы успеть выйти: канал
запросов закрывается вместе с процессом бэкенда, `recv()` получает EOF, цикл
завершается. Отдельного «присмотра» за родителем не нужно — ядро закрывает
дескрипторы само.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

from . import worker_protocol as proto

logger = logging.getLogger("tts.worker")

# Traceback в ответе нужен для лога родителя, но не для базы и не для интерфейса:
# ограничиваем, чтобы multimegabyte-простыня не уехала по каналу.
TRACEBACK_LIMIT = 4000
# Текст ошибки тоже усекаем: некоторые библиотеки вставляют в сообщение весь
# тензор целиком.
ERROR_LIMIT = 600


def _configure_logging(engine_id: str) -> None:
    """Логи воркера — в stderr; их читает родитель и подписывает своим префиксом.

    Уровень берётся из `TTS_WORKER_LOG_LEVEL` (по умолчанию INFO): при разборе
    падений хочется видеть, на чём именно упало, а лишние строки теряются в логе
    бэкенда без вреда.
    """
    level = os.environ.get("TTS_WORKER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=f"%(asctime)s %(levelname)-7s [{engine_id}] %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def create_engine(engine_id: str):
    """Создаёт настоящий движок по id.

    Фабрика из `TTS_WORKER_ENGINE_FACTORY` (`модуль:функция`) — не украшение, а
    единственный способ проверить изоляцию без модели: тесты подставляют сюда
    лёгкую заглушку, и в дочернем процессе не оказывается ни torch, ни весов.
    Формат намеренно повторяет `uvicorn --factory`: путь к объекту строкой.
    """
    spec = os.environ.get("TTS_WORKER_ENGINE_FACTORY", "").strip()
    if spec:
        module_name, _, attribute = spec.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute or "create")
        logger.info("Воркер %s: движок создаётся фабрикой %s", engine_id, spec)
        return factory(engine_id)
    # Настоящий движок собирает тот же реестр, что и раньше: второй реализации
    # F5/XTTS не появляется, воркер лишь место, где она живёт.
    from .registry import create_local_engine

    return create_local_engine(engine_id)


class _EngineHolder:
    """Ленивая обёртка над движком: до первого запроса модель не поднимается.

    Пинг и выгрузка не должны стоить импорта torch: `/api/status` дёргает воркер
    часто, а подъём модели — десятки секунд.
    """

    def __init__(self, engine_id: str) -> None:
        self._engine_id = engine_id
        self._engine: Any = None
        self.create_count = 0

    @property
    def engine(self):
        if self._engine is None:
            started = time.monotonic()
            self._engine = create_engine(self._engine_id)
            self.create_count += 1
            logger.info(
                "Воркер %s: движок создан за %.2f c", self._engine_id, time.monotonic() - started
            )
        return self._engine

    @property
    def created(self) -> bool:
        return self._engine is not None

    def state(self) -> str:
        return self._engine.state if self._engine is not None else proto.WORKER_STATE_IDLE

    def release(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.unload()
        except Exception as exc:  # noqa: BLE001 — выгрузка не должна мешать выходу
            logger.warning("Воркер %s: ошибка выгрузки при завершении: %s", self._engine_id, exc)


def _error_response(exc: BaseException, engine_id: str) -> dict:
    """Ошибка модели как значение: процесс остаётся живым и готов к следующему запросу."""
    message = f"{type(exc).__name__}: {exc}"[:ERROR_LIMIT]
    logger.error("Воркер %s: ошибка синтеза: %s", engine_id, message)
    return {
        "ok": False,
        "error_type": proto.ERROR_TTS,
        "error": message,
        "traceback": traceback.format_exc()[-TRACEBACK_LIMIT:],
        "engine": engine_id,
        "pid": os.getpid(),
    }


def handle_request(message: dict, holder: _EngineHolder, engine_id: str) -> dict:
    """Выполняет один запрос. Исключения движка не выходят наружу: они — ответ."""
    kind = message.get("kind")
    if kind == proto.REQUEST_PING:
        return {
            "ok": True,
            "pid": os.getpid(),
            "engine": engine_id,
            "state": holder.state(),
            "created": holder.created,
        }
    if kind == proto.REQUEST_LOAD:
        try:
            holder.engine.load()
        except Exception as exc:  # noqa: BLE001 — ошибку загрузки возвращаем ответом
            return _error_response(exc, engine_id)
        return {"ok": True, "pid": os.getpid(), "engine": engine_id, "state": holder.state()}
    if kind == proto.REQUEST_UNLOAD:
        holder.release()
        return {"ok": True, "pid": os.getpid(), "engine": engine_id}
    if kind == proto.REQUEST_SYNTHESIZE:
        started = time.monotonic()
        try:
            waveform, sample_rate = holder.engine.synthesize(
                text=message["text"],
                ref_audio_path=message["ref_audio_path"],
                ref_text=message["ref_text"],
                speed=float(message.get("speed", 1.0)),
                **dict(message.get("params") or {}),
            )
        except Exception as exc:  # noqa: BLE001 — ошибку движка возвращаем ответом
            return _error_response(exc, engine_id)
        return {
            "ok": True,
            "waveform": waveform,
            "sample_rate": int(sample_rate),
            "pid": os.getpid(),
            "engine": engine_id,
            "elapsed_sec": time.monotonic() - started,
        }
    return {
        "ok": False,
        "error_type": proto.ERROR_TTS,
        "error": f"неизвестный запрос: {kind!r}",
        "engine": engine_id,
        "pid": os.getpid(),
    }


def _parse_args(argv: list[str]) -> tuple[str, int, int]:
    if len(argv) != 3:
        raise SystemExit("usage: python -m backend.engines.worker_process <engine> <req_fd> <resp_fd>")
    engine_id, request_fd, response_fd = argv
    return engine_id, int(request_fd), int(response_fd)


def main(argv: list[str] | None = None) -> int:
    engine_id, request_fd, response_fd = _parse_args(list(sys.argv[1:] if argv is None else argv))
    _configure_logging(engine_id)
    # Порядок важен: `config` выставляет лимиты потоков в переменных окружения, и
    # сделать это нужно до первого импорта torch (внутри движка).
    from .. import config  # noqa: F401

    # Явный маркер: `worker_isolation_enabled()` в дочернем процессе обязан
    # вернуть False, иначе реестр собрал бы здесь ещё одного воркера — рекурсия.
    os.environ["TTS_WORKER_CHILD"] = "1"
    request = Connection(request_fd, readable=True, writable=False)
    response = Connection(response_fd, readable=False, writable=True)
    holder = _EngineHolder(engine_id)
    logger.info("Воркер %s запущен: pid %s", engine_id, os.getpid())

    try:
        while True:
            try:
                message = request.recv()
            except EOFError:
                # Родитель закрыл канал: либо штатное выключение, либо бэкенд
                # убит. В обоих случаях воркер обязан уйти — иначе на машине
                # останется процесс с моделью в памяти и без хозяина.
                logger.info("Воркер %s: канал закрыт родителем, выхожу", engine_id)
                return 0
            kind = message.get("kind") if isinstance(message, dict) else None
            try:
                answer = handle_request(message, holder, engine_id)
            except BaseException as exc:
                logger.exception("Воркер %s: сбой обработки запроса %s", engine_id, kind)
                answer = _error_response(exc, engine_id)
            try:
                response.send(answer)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                logger.warning("Воркер %s: не смог ответить (%s), выхожу", engine_id, exc)
                return 0
            if kind == proto.REQUEST_SHUTDOWN:
                logger.info("Воркер %s: получен shutdown, выхожу", engine_id)
                return 0
    finally:
        holder.release()
        request.close()
        response.close()


if __name__ == "__main__":
    raise SystemExit(main())
