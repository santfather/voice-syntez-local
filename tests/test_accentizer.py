"""Accentizer: сериализация инференса одной модели на процесс (F-T4).

Модель RUAccent живёт синглтоном, а её инференс не потокобезопасен: два
одновременных запроса (реплика проекта, preview, предложения для словаря) делят
внутреннее состояние модели и портят результат. Поэтому вызовы модели
выстраиваются в очередь локи `_infer_lock` — ровно так же, как это сделано у
движков синтеза (`_infer_lock` в `backend/engines/*`).

Тест подставляет модель, которая сама считает, сколько её вызовов выполнялось
одновременно: гонка видна детерминированно (внутри вызова есть пауза — окно, в
которое второй поток успел бы войти), без опоры на тайминги и повторы.

Настоящий RUAccent не поднимается: `_accent` подменён, состояние синглтона
объявлено готовым, библиотека не нужна.
"""

import threading
import time
from contextlib import contextmanager

from backend.accentizer import STATE_READY, Accentizer


class _Probe:
    """Считает одновременные вызовы модели: `peak > 1` означает гонку."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inside = 0
        self.peak = 0

    @contextmanager
    def call(self):
        with self._lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        try:
            time.sleep(0.02)
            yield
        finally:
            with self._lock:
                self.inside -= 1


class _FakeYoScores:
    """Модель ё-омографов: та же проверка гонки, что и у process_all."""

    def __init__(self, probe: _Probe) -> None:
        self._probe = probe

    def predict_yo_homographs(self, sentence: str) -> list[dict]:
        with self._probe.call():
            return []


class _FakeAccent:
    """Заглушка RUAccent: обе точки инференса считает один общий счётчик."""

    def __init__(self, probe: _Probe) -> None:
        self._probe = probe
        self.yo_homographs: dict[str, str] = {}
        self.yo_homograph_model = _FakeYoScores(probe)

    def process_all(self, text: str) -> str:
        with self._probe.call():
            return f"[{text}]"


def _ready_accentizer(probe: _Probe) -> Accentizer:
    """Синглтон с «уже загруженной» моделью: `load()` вернёт True без библиотеки."""
    accentizer = Accentizer()
    accentizer._accent = _FakeAccent(probe)
    accentizer._state = STATE_READY
    return accentizer


def _run_parallel(worker, count: int = 8) -> list:
    """Запускает worker(index) в count потоках и возвращает результаты по индексу."""
    results: list = [None] * count
    threads = [
        threading.Thread(target=lambda index=index: results.__setitem__(index, worker(index)))
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_accentuate_serializes_model_calls():
    probe = _Probe()
    accentizer = _ready_accentizer(probe)

    results = _run_parallel(lambda index: accentizer.accentuate(f"фраза {index}"))

    assert probe.peak == 1, "инференс выполнялся параллельно на одной модели"
    # Результаты не перепутались: каждому потоку достался его собственный вход.
    assert sorted(results) == sorted(f"[фраза {index}]" for index in range(8))


def test_yo_scores_serialize_model_calls():
    probe = _Probe()
    accentizer = _ready_accentizer(probe)

    results = _run_parallel(lambda index: accentizer.yo_homograph_scores(f"Фраза {index}."))

    assert probe.peak == 1, "модель ё-омографов вызывалась параллельно"
    assert all(result == [] for result in results)
