"""Движок синтеза, который на самом деле живёт в отдельном процессе.

Класс — тонкая замена настоящему движку: он наследует тот же ABC, объявляет тот
же паспорт и проходит тот же путь `synthesize()` → `load()` → `_synthesize()`,
поэтому пайплайн, QA-цикл, варианты, сиды, ETA, перегенерация и выгрузка по
простою работают без единой правки. Разница только в том, что `_synthesize()`
не вызывает модель, а отправляет запрос воркеру (см. `supervisor`).

Почему прокси, а не «второй движок». Вся логика конкретной модели — в
`f5_engine`/`xtts_engine`, и она не дублируется: дочерний процесс собирает
ровно тот же объект через реестр. Прокси отвечает только за транспорт,
состояние и превращение отказа процесса в понятное исключение.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import config
from . import worker_protocol as proto
from .base import (
    ENGINE_INFOS,
    STATE_FAILED,
    STATE_LOADING,
    STATE_READY,
    SynthesisEngine,
)
from .supervisor import WorkerHandle, get_supervisor

logger = logging.getLogger(__name__)


class WorkerEngine(SynthesisEngine):
    """Прокси движка в изолированном процессе."""

    def __init__(self, engine_id: str) -> None:
        super().__init__()
        self.info = ENGINE_INFOS[engine_id]
        # Последний отказ воркера: нужен, чтобы `last_error` движка объяснял
        # «failed» не общим текстом, а причиной (сигнал, код возврата).
        self._last_failure: proto.WorkerFailure | None = None
        # Супервизор сообщает о падении сам: состояние движка обязано стать
        # `failed` сразу, а не при следующем запросе — иначе интерфейс показывал
        # бы «ready» у движка, чей процесс уже мёртв.
        get_supervisor().add_failure_listener(engine_id, self._on_failure)

    # -- состояние -------------------------------------------------------------
    def _on_failure(self, failure: proto.WorkerFailure) -> None:
        self._last_failure = failure
        self._mark(STATE_FAILED, failure)

    @property
    def worker(self) -> WorkerHandle:
        """Хэндл воркера: pid, состояние процесса, счётчик перезапусков."""
        return get_supervisor().handle(self.id)

    @property
    def worker_pid(self) -> int | None:
        return self.worker.pid

    def to_dict(self) -> dict:
        """Паспорт движка плюс состояние процесса — для `/api/status` и вкладки «Модели»."""
        return {**super().to_dict(), "worker": self.worker.to_dict()}

    # -- загрузка и синтез -----------------------------------------------------
    def load(self) -> None:
        """Поднимает модель в воркере (процесс запускается здесь же, если нужно)."""
        self._mark(STATE_LOADING)
        try:
            answer = get_supervisor().request(
                self.id, {"kind": proto.REQUEST_LOAD}, timeout=config.WORKER_LOAD_TIMEOUT_SEC
            )
        except proto.WorkerFailure:
            raise
        except Exception as exc:
            self._mark(STATE_FAILED, exc)
            raise
        logger.info(
            "Движок %s загружен в воркере (pid %s)", self.id, answer.get("pid")
        )
        self._mark(STATE_READY)

    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[Any, int]:
        """Отправляет кусок воркеру и получает waveform.

        Путь к референсу передаётся абсолютным: воркер работает с `cwd` проекта,
        но полагаться на это в обмене между процессами не стоит — цена ошибки
        «не найден файл голоса» несопоставима с одной строкой приведения.
        """
        answer = get_supervisor().request(
            self.id,
            {
                "kind": proto.REQUEST_SYNTHESIZE,
                "text": text,
                "ref_audio_path": str(ref_audio_path),
                "ref_text": ref_text,
                "speed": float(speed),
                "params": dict(params),
            },
            timeout=config.WORKER_REQUEST_TIMEOUT_SEC,
        )
        return answer["waveform"], int(answer["sample_rate"])

    def _release(self) -> None:
        """Выгрузка = остановка процесса: модель живёт в нём и уходит вместе с ним.

        Никакого «оставить процесс, но освободить модель» здесь нет намеренно:
        именно процесс — единица изоляции, и после падения замена поднимается с
        чистым состоянием, а не с огрызками предыдущего инференса.
        """
        get_supervisor().unload(self.id, reason="выгрузка движка")

    def abort_current(self, reason: str = "таймаут") -> bool:
        """Убивает зависший инференс. Возвращает `True`, если воркер был занят.

        Не часть общего интерфейса движков: у модели в своём процессе прервать
        нативный вызов извне нельзя, и единственный честный способ остановить
        зависание — убить процесс. Пайплайн зовёт этот хук, только если он есть
        (см. `audio_pipeline._synthesize_chunk`), поэтому движки в процессе
        бэкенда о нём не знают.
        """
        killed = get_supervisor().abort_current(self.id, reason)
        if killed:
            logger.warning("Зависший синтез %s прерван: %s", self.id, reason)
        return killed
