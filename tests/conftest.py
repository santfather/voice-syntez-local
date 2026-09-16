"""Общие фикстуры тестов: движок-заглушка, временные каталоги и хранилище голосов.

Тесты не поднимают F5-TTS и XTTS: движок подменяется заглушкой `StubEngine`,
которая пишет вызовы и возвращает предсказуемую синусоиду. Каталоги голосов и
output уводятся в `tmp_path`, поэтому прогон идёт секунды и не трогает рабочие
файлы проекта.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend import audio_pipeline, config, job_queue
from backend.engines.base import EngineInfo, SAMPLE_RATE, STATE_READY, SynthesisEngine
from backend.voices_store import Voice

STUB_ENGINE_ID = "stub"


class StubEngine(SynthesisEngine):
    """Движок вместо модели: предсказуемая волна и журнал вызовов."""

    info = EngineInfo(
        id=STUB_ENGINE_ID,
        label="Заглушка",
        description="Тестовый движок вместо F5-TTS и XTTS — модели не поднимаются.",
        # RUAccent в тестах не нужен: разметку ударений проверяет отдельный тест.
        supports_accents=False,
    )
    supports_seed = True

    def __init__(self, seconds_per_char: float = 0.02) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.loads = 0
        self.seconds_per_char = seconds_per_char
        self.sample_rate = SAMPLE_RATE

    def load(self) -> None:
        self.loads += 1
        self._mark(STATE_READY)

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        self.calls.append(
            {
                "text": text,
                "ref_audio_path": ref_audio_path,
                "ref_text": ref_text,
                "speed": speed,
                "params": dict(params),
            }
        )
        seed = params.get("seed")
        # Длина зависит от текста и номера вызова: перегенерация обязана давать кусок
        # другой длины, иначе сдвиг хвоста файла (_shift_segments) нечем проверить.
        # Считать jitter от сида нельзя — два разных сида легко совпадают по модулю,
        # и тест на перегенерацию становится лотереей.
        jitter = 0.0 if seed is None else 0.04 * len(self.calls)
        samples = int(SAMPLE_RATE * (0.4 + self.seconds_per_char * len(text) + jitter))
        t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
        return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), self.sample_rate


def sine(seconds: float, freq: float = 220.0, amplitude: float = 0.3, sr: int = SAMPLE_RATE):
    """Синусоида заданной частоты — заготовка для кусков и референсов."""
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_hz(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Частота самой сильной спектральной составляющей: по ней видно, какой кусок в файле."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    return float(freqs[int(np.argmax(spectrum))])


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Уводит голоса и output в tmp_path: тесты не трогают рабочие файлы проекта."""
    voices_dir = tmp_path / "voices"
    output_dir = tmp_path / "output"
    voices_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(config, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "VOICES_JSON", voices_dir / "voices.json")
    # База проектов тоже в tmp_path: иначе тесты писали бы в рабочую базу, а
    # прогон зависел бы от того, какие проекты уже созданы на машине.
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "projects.db")
    monkeypatch.setattr(config, "PROJECTS_OUTPUT_DIR", output_dir / "projects")
    return tmp_path


@pytest.fixture(autouse=True)
def no_memory_pressure(monkeypatch):
    """Очередь не должна засыпать в ожидании свободной системной памяти."""
    monkeypatch.setattr(job_queue.resource_guard, "check_system_memory_pressure", lambda: False)


@pytest.fixture
def stub(monkeypatch):
    """Подменяет реестр движков: на любой engine_id отдаётся одна заглушка."""
    engine = StubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)
    return engine


@pytest.fixture
def voice(workspace):
    """Голос с настоящим файлом референса — его требуют `_resolve_voice` и движок."""
    sf.write(workspace / "voices" / "ref.wav", sine(2.0, 180.0), SAMPLE_RATE)
    return Voice(
        id="voice1",
        name="Тестовый",
        gender="male",
        ref_text="Привет, это тест",
        audio_file="ref.wav",
        engine=STUB_ENGINE_ID,
    )


@pytest.fixture
def fake_store(monkeypatch, voice):
    """Хранилище из одного голоса: тестам не нужен voices.json."""

    class Store:
        def get(self, voice_id: str) -> Voice | None:
            return voice if voice_id == voice.id else None

    monkeypatch.setattr(audio_pipeline, "get_store", lambda: Store())
    return voice
