"""Выгрузка простаивающих движков: ручная и по таймауту (фаза 10).

Модуль отвечает на один вопрос — «можно ли сейчас вернуть память, которую
держит движок», — и держит эту логику отдельно от API и очереди:

* само решение вынесено в чистую функцию `decide_idle_unload`: её проверяют
  тесты с подставным временем, без потоков, таймеров и моделей;
* живое состояние (поднят ли движок, занят ли синтезом, пуста ли очередь)
  собирает `should_unload_now`: ручной роут и фоновая задача зовут его
  одинаково, поэтому «почему не выгрузилось» объясняется одним и тем же кодом;
* фоновая задача `IdleUnloadWatcher` раз в минуту проходит по уже созданным
  движкам и выгружает те, что простояли дольше `TTS_ENGINE_IDLE_UNLOAD_MIN`.

Пайплайн синтезирует в отдельном потоке, поэтому гонка «выгрузка против
стартующей задачи» закрыта в `SynthesisEngine`: `unload()` отказывает, пока
счётчик активных синтезов не ноль, а `synthesize()` поднимает модель обратно.
Здесь эта защита только используется, а не дублируется.
"""

import asyncio
import logging
import time
from collections.abc import Callable

from . import config
from .engines.base import SynthesisEngine
from .engines.registry import created_engines

logger = logging.getLogger(__name__)

# Как часто фоновая задача проверяет простой. Минута — компромисс: движок,
# простоявший 15 минут, не обязан выгружаться ровно на пятнадцатой, а частые
# проверки только читают состояние.
IDLE_CHECK_INTERVAL_SEC = 60.0

# Причины решения — текстом, а не флагом: их пишет лог, ими объясняется отказ,
# и по ним же проверяются тесты. Формулировки на русском, потому что попадают
# в сообщения интерфейса.
REASON_DISABLED = "политика выключена"
REASON_NOT_LOADED = "движок не поднят"
REASON_BUSY = "движок занят синтезом"
REASON_QUEUE_BUSY = "очередь занята задачей"
REASON_TOO_EARLY = "порог простоя не достигнут"
REASON_IDLE = "движок простаивает дольше порога"


def decide_idle_unload(
    *,
    policy_minutes: float,
    last_used_at: float | None,
    now: float,
    loaded: bool,
    busy: bool,
    queue_busy: bool,
) -> tuple[bool, str]:
    """Чистое решение о выгрузке: без обращений к движку, очереди и времени.

    Порядок проверок — от «выгрузка вообще выключена» к «пора»: так причина
    отказа всегда самая общая из подходящих, и сообщение не уводит в сторону.
    `policy_minutes <= 0` — политика выключена (`TTS_ENGINE_IDLE_UNLOAD_MIN=0`).
    """
    if policy_minutes <= 0:
        return False, REASON_DISABLED
    if not loaded:
        return False, REASON_NOT_LOADED
    if busy:
        return False, REASON_BUSY
    if queue_busy:
        return False, REASON_QUEUE_BUSY
    if last_used_at is None:
        return False, REASON_TOO_EARLY
    if now - last_used_at < policy_minutes * 60:
        return False, REASON_TOO_EARLY
    return True, REASON_IDLE


def queue_busy_now() -> bool:
    """Занята ли очередь задачей. Если очередь недоступна — считаем, что занята.

    Ошибка чтения не должна приводить к выгрузке: осторожный ответ здесь дешевле
    лишней загрузки модели.
    """
    try:
        from .job_queue import get_queue

        return get_queue().is_busy()
    except Exception as exc:  # noqa: BLE001 — без очереди выгружать нечего
        logger.warning("Не удалось проверить занятость очереди, пропускаю выгрузку: %s", exc)
        return True


def should_unload_now(
    engine: SynthesisEngine,
    *,
    now: float | None = None,
    queue_busy: bool | None = None,
) -> tuple[bool, str]:
    """Решение по живому движку: порог берётся из `config` в момент вызова.

    `now` и `queue_busy` подставляются в тестах: функцию тогда не нужно ждать и
    не нужно поднимать очередь.
    """
    return decide_idle_unload(
        policy_minutes=config.engine_idle_unload_minutes(),
        last_used_at=engine.last_used_at,
        now=time.monotonic() if now is None else now,
        loaded=engine.is_loaded,
        busy=engine.active_synthesizes > 0,
        queue_busy=queue_busy_now() if queue_busy is None else queue_busy,
    )


class IdleUnloadWatcher:
    """Фоновая задача: выгружает движки, простоявшие дольше порога.

    Живёт в `lifespan` рядом с watchdog'ом памяти. Ничего не поднимает: работает
    только с уже созданными движками (`created_engines`), поэтому простаивающая
    вкладка «Модели» не приводит к загрузке модели.
    """

    def __init__(
        self,
        *,
        engines: Callable[[], dict[str, SynthesisEngine]] = created_engines,
        interval_sec: float = IDLE_CHECK_INTERVAL_SEC,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engines = engines
        self._interval_sec = interval_sec
        self._now = now

    async def run(self) -> None:
        """Проверяет простой до тех пор, пока задача не отменена из `lifespan`."""
        while True:
            await asyncio.sleep(self._interval_sec)
            await self.sweep()

    async def sweep(self) -> list[str]:
        """Одна проверка: возвращает id выгруженных движков (для логов и тестов).

        Ошибка выгрузки не роняет backend: она логируется, а движок остаётся в
        согласованном состоянии (`failed`, если не удалось освободить ресурсы), и
        его можно выгрузить или поднять снова.
        """
        unloaded: list[str] = []
        queue_busy = queue_busy_now()
        for engine_id, engine in self._engines().items():
            should, reason = should_unload_now(
                engine, now=self._now(), queue_busy=queue_busy
            )
            if not should:
                logger.debug("Движок %s не выгружаю: %s", engine_id, reason)
                continue
            try:
                await asyncio.to_thread(engine.unload)
            except Exception as exc:  # noqa: BLE001 — фоновая уборка не повод падать
                logger.error("Движок %s: выгрузка по простою не удалась (%s)", engine_id, exc)
                continue
            unloaded.append(engine_id)
            logger.info("Движок %s выгружен по простою: %s", engine_id, reason)
        return unloaded
