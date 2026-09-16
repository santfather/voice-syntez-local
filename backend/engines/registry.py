"""Реестр движков: по одному тёплому экземпляру на движок на весь процесс.

Конкретные классы импортируются лениво, внутри `get_engine`: импорт XTTS тянет
torch и coqui-tts (десятки секунд и сотни мегабайт RSS), и делать это при старте
сервера незачем — движок поднимается, только когда его выбрал хотя бы один голос.
"""

import logging
import threading

from .base import (
    ENGINE_F5,
    ENGINE_INFOS,
    ENGINE_XTTS,
    ENGINE_XTTS_BANANA,
    SynthesisEngine,
)

logger = logging.getLogger(__name__)

_instances: dict[str, SynthesisEngine] = {}
_lock = threading.Lock()


def _create(engine_id: str) -> SynthesisEngine:
    if engine_id == ENGINE_F5:
        from .f5_engine import F5Engine

        return F5Engine()
    if engine_id in (ENGINE_XTTS, ENGINE_XTTS_BANANA):
        from .xtts_engine import create_xtts_engine

        return create_xtts_engine(engine_id)
    raise ValueError(f"Неизвестный движок синтеза: {engine_id}")


def get_engine(engine_id: str) -> SynthesisEngine:
    """Экземпляр движка (создаётся один раз). Неизвестный id — ошибка, а не молчаливый F5.

    Подмена движка по умолчанию скрыла бы опечатку в `voices.json` до момента,
    когда пользователь услышит не тот голос, который выбрал.
    """
    if engine_id not in ENGINE_INFOS:
        raise ValueError(f"Неизвестный движок синтеза: {engine_id}")
    with _lock:
        engine = _instances.get(engine_id)
        if engine is None:
            engine = _create(engine_id)
            _instances[engine_id] = engine
            logger.info("Движок %s создан", engine_id)
        return engine


def created_engines() -> dict[str, SynthesisEngine]:
    """Уже созданные движки — для /api/status и /api/models, без запуска новых.

    Состояние берётся у живых объектов (`engine.state`/`engine.is_loaded`), а не
    кешируется здесь: после `unload()` и `/api/status`, и вкладка «Модели»
    обязаны видеть `idle`, а не запомненный `ready`.
    """
    with _lock:
        return dict(_instances)


def created_engine(engine_id: str) -> SynthesisEngine | None:
    """Уже созданный движок по id или `None` — без создания нового.

    Нужен ручной выгрузке: поднимать модель ради того, чтобы её выгрузить,
    бессмысленно, а неизвестный id проверяется вызывающим отдельно.
    """
    with _lock:
        return _instances.get(engine_id)
