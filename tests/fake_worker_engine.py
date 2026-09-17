"""Движок-заглушка для дочернего процесса воркера (тесты изоляции).

Модуль импортируется **только** в процессе-воркере: супервизор передаёт его
через `TTS_WORKER_ENGINE_FACTORY`, и настоящая модель не поднимается — иначе
проверить падение и перезапуск было бы нельзя (падение F5 на тестовой машине
занимает минуты, а SIGABRT в тесте надо получать детерминированно).

Поведение задаётся текстом реплики, потому что весь путь запроса — это текст:

* `CRASH` — процесс падает нативно (`os.abort()`, то есть SIGABRT);
* `SEGV` — падение с SIGSEGV (проверяет разбор другого сигнала);
* `BOOM`  — обычная ошибка Python внутри модели (воркер обязан остаться жив);
* `HANG`  — зависание: процесс жив, ответа нет (проверяет таймаут и убийство);
* `SLOW`  — долгий, но честный инференс (проверяет, что занятость воркера видна
  снаружи и что выгрузка не влезает в идущий синтез);
* всё остальное — предсказуемая синусоида длиной от текста.
"""

from __future__ import annotations

import os
import time

import numpy as np

from backend.engines.base import SAMPLE_RATE, STATE_READY, EngineInfo, SynthesisEngine

# Сколько «синтезировать» на знак: мало, чтобы тесты шли секунды.
SECONDS_PER_CHAR = 0.01
SLOW_SECONDS = 3.0
HANG_SECONDS = 120.0


class FakeWorkerEngine(SynthesisEngine):
    """Заглушка с управляемым поведением (см. модульную строку документации)."""

    info = EngineInfo(
        id="f5",
        label="Заглушка воркера",
        description="Тестовый движок внутри процесса-воркера.",
        supports_accents=True,
        supports_seed=True,
    )

    def __init__(self, engine_id: str) -> None:
        super().__init__()
        self.engine_id = engine_id
        self.loads = 0
        self.synthesizes = 0

    def load(self) -> None:
        self.loads += 1
        self._mark(STATE_READY)

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        self.synthesizes += 1
        if "CRASH" in text:
            # Нативное падение: исключение поймать нельзя, и это ровно тот
            # сценарий, ради которого воркер вынесен в отдельный процесс.
            os.abort()
        if "SEGV" in text:
            import ctypes

            ctypes.string_at(0)  # SIGSEGV: проверяем разбор другого сигнала
        if "BOOM" in text:
            raise RuntimeError("модель сломалась внутри воркера")
        if "HANG" in text:
            time.sleep(HANG_SECONDS)
        if "SLOW" in text:
            time.sleep(SLOW_SECONDS)
        seconds = 0.2 + SECONDS_PER_CHAR * max(len(text), 1)
        samples = int(SAMPLE_RATE * seconds)
        timeline = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
        wave = (0.25 * np.sin(2 * np.pi * 220.0 * timeline)).astype(np.float32)
        return wave, SAMPLE_RATE


def create(engine_id: str) -> FakeWorkerEngine:
    """Фабрика для `TTS_WORKER_ENGINE_FACTORY=tests...:create`."""
    return FakeWorkerEngine(engine_id)
