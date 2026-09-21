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

from backend import audio_pipeline, config, job_queue, model_manager
from backend.engines.base import SAMPLE_RATE, STATE_READY, EngineInfo, SynthesisEngine
from backend.engines.supervisor import get_supervisor
from backend.voices_store import Voice

STUB_ENGINE_ID = "stub"


@pytest.fixture(autouse=True)
def _no_worker_isolation(monkeypatch):
    """По умолчанию изоляция синтеза выключена — как и раньше, модели не поднимаются.

    Иначе любой тест, дёрнувший настоящий реестр (`registry.get_engine("f5")`),
    запускал бы дочерний процесс с torch и весами. Тесты самой изоляции включают
    её сами и подставляют лёгкую фабрику движка (см. test_worker_isolation.py).
    """
    monkeypatch.setenv(config.WORKER_ISOLATION_ENV, "0")
    yield
    # Процессы, поднятые тестом изоляции, не должны переживать тест: осиротевший
    # воркер держал бы память и портил бы следующий прогон.
    supervisor = get_supervisor()
    supervisor.stop_all(grace=1.0)
    supervisor.reset_after_stop()


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
    monkeypatch.setattr(config, "BENCHMARKS_DIR", output_dir / "benchmarks")
    # Каталог диагностики и лог — туда же: архив качества не должен собираться в
    # рабочий `output/diagnostics`, а чужой лог делал бы тест недетерминированным.
    monkeypatch.setattr(config, "DIAGNOSTICS_DIR", output_dir / "diagnostics")
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "voice_syntez.log")
    # Записи пользователя («Сам себе звукорежиссер») — туда же: иначе тесты писали
    # бы в рабочий каталог `recordings/`, а прогон зависел бы от прошлых записей.
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "recordings")
    from backend import recording_pipeline, recording_store

    recording_store.reset_store()
    recording_pipeline.reset_states()
    # Менеджер моделей держит состояние скачиваний в памяти, а замеры размеров —
    # в кеше модуля: без сброса тесты влияли бы друг на друга.
    model_manager.reset_manager()
    yield tmp_path
    model_manager.reset_manager()


@pytest.fixture(autouse=True)
def isolated_llm_settings(tmp_path, monkeypatch):
    """Настройки анализатора из `data/llm_settings.json` не влияют на тесты.

    Файл настроек переживает перезапуск — это его смысл, — но и тесты он тогда
    переживает тоже: включённый в интерфейсе анализатор делал
    `test_analyzer_disabled_by_default` красным на машине пользователя, хотя код
    исправен. Поэтому каждый тест получает свой пустой файл настроек.
    """
    from backend.llm import settings_store

    monkeypatch.setenv(settings_store.SETTINGS_ENV, str(tmp_path / "llm_settings.json"))


@pytest.fixture(autouse=True)
def no_memory_pressure(monkeypatch):
    """Очередь не должна засыпать в ожидании свободной памяти.

    Патчатся оба входа: `is_memory_critical` — то, чем пользуется очередь, и
    `check_system_memory_pressure` — прежняя проверка, на которую опираются
    отдельные тесты.
    """
    monkeypatch.setattr(job_queue.resource_guard, "is_memory_critical", lambda: False)
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


# --- общий шаг сценариев --------------------------------------------------------
async def analyze_project(client, project_id: str, **body) -> dict:
    """Доводит проект до готовности к рендеру — тем же путём, что интерфейс.

    Рендер неподготовленного проекта запрещён (409), поэтому каждый сценарий,
    доходящий до звука, проходит обязательную подготовку: анализ, а затем review
    найденных слов. Кандидаты подтверждаются предложенным вариантом — ровно то,
    что делает пользователь кнопкой «Добавить». Хелпер общий нарочно: если бы
    каждый тест готовил проект по-своему, они проверяли бы разные пути подготовки,
    а не один канонический.
    """
    state = await _analyze(client, project_id, **body)
    # Review: неоднозначные слова нельзя разрешать молча, но и оставлять проект
    # неготовым в сценарии, который проверяет не review, нельзя. Подтверждаем
    # предложенное и пересчитываем только затронутые реплики.
    for _ in range(5):
        if state["status"] != "needs_review" or not state["candidates"]:
            break
        touched: list[int] = []
        for candidate in state["candidates"]:
            # Сценарии, которые проверяют не review, проходят его «пропуском»:
            # пропуск ничего не меняет в произношении (правило выключено) и лишь
            # снимает предложение с показа. Принятие предложения, наоборот,
            # поменяло бы текст — и тест, сверяющий текст, проверял бы уже другое.
            response = await client.post(
                "/api/pronunciation",
                json={
                    "source": candidate["word"],
                    "target": candidate["target"] or candidate["word"],
                    "enabled": False,
                    "note": "пропущено в тестовом сценарии",
                },
            )
            assert response.status_code in (200, 201), response.text
            if candidate["replica_index"] not in touched:
                touched.append(candidate["replica_index"])
        state = await _analyze(client, project_id, indexes=touched)
    assert state["status"] == "ready", state
    return state


async def _analyze(client, project_id: str, **body) -> dict:
    """Один прогон анализа: состояние проекта и найденные кандидаты."""
    response = await client.post(f"/api/projects/{project_id}/analyze", json=body or {})
    assert response.status_code == 200, response.text
    return response.json()


# --- общие фикстуры двух движков -------------------------------------------------
# Живут здесь, а не в одном из тестовых модулей: их используют и preview, и
# сценарии обязательной подготовки, а импортировать фикстуры из чужого тестового
# модуля — значит связывать тесты между собой.
class AccentStubEngine(StubEngine):
    """Заглушка F5: говорит, что понимает «+»-ударения, как настоящий движок."""

    info = EngineInfo(
        id="stub-accents",
        label="Заглушка с ударениями",
        description="Тестовый движок вместо F5-TTS — модель не поднимается.",
        supports_accents=True,
    )


class _VoiceStore:
    """Хранилище из нескольких голосов: есть из чего выбирать движок."""

    def __init__(self, *voices: Voice) -> None:
        self._voices = {voice.id: voice for voice in voices}

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)


F5_VOICE = "voice-f5"
XTTS_VOICE = "voice-xtts"


@pytest.fixture
def voices(workspace, monkeypatch):
    """Два голоса разных движков — по одному на каждую ветку ударений."""
    import soundfile as sf

    from backend.engines.base import ENGINE_F5, ENGINE_XTTS

    sf.write(workspace / "voices" / "f5.wav", sine(2.0, 180.0), SAMPLE_RATE)
    sf.write(workspace / "voices" / "xtts.wav", sine(2.0, 240.0), SAMPLE_RATE)
    f5 = Voice(
        id=F5_VOICE, name="Ф5", gender="male", ref_text="Привет, это тест",
        audio_file="f5.wav", engine=ENGINE_F5,
    )
    xtts = Voice(
        id=XTTS_VOICE, name="Икс", gender="female", ref_text="Привет, это тест",
        audio_file="xtts.wav", engine=ENGINE_XTTS,
    )
    store = _VoiceStore(f5, xtts)
    # Эндпоинт разрешает голос через `main`, синтез — через `audio_pipeline`:
    # в бою это один и тот же singleton, в тесте подменяем оба, чтобы они не
    # разошлись и подготовка не оказалась «про другой голос».
    import backend.main as main_module

    monkeypatch.setattr(main_module, "get_store", lambda: store)
    monkeypatch.setattr(audio_pipeline, "get_store", lambda: store)
    return f5, xtts


@pytest.fixture
def fake_accent(monkeypatch):
    """Предсказуемая «RUAccent»: маркер вместо настоящей модели ударений."""

    def accent(text: str) -> str:
        return f"[{text}]"

    monkeypatch.setattr(audio_pipeline, "accentuate", accent)


@pytest.fixture
def accent_stub(monkeypatch):
    """Движок рендера, понимающий ударения (как F5): без него их не проверить."""
    engine = AccentStubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)
    return engine
