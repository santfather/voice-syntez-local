"""Движок Qwen3-TTS (ФАЗА 9): загрузка, кеш clone-prompt, инференс, откат на CPU.

Настоящие веса здесь не поднимаются: `qwen_tts` подменяется подставным пакетом,
который пишет вызовы и отдаёт предсказуемый звук. Поэтому проверяются ровно те
вещи, которые не видны на реальной модели: какой каталог выбран, с каким
устройством и dtype вызван `from_pretrained`, когда промпт референса берётся из
кеша, что уходит в `generate_voice_clone` и что происходит, когда MPS падает.

Реальные веса — отдельный, gated прогон (`TTS_RUN_REAL_QWEN=1`, см. docs/testing.md).
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from backend import config, model_manager
from backend.engines.base import (
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_QWEN,
    SAMPLE_RATE,
    STATE_FAILED,
)
from backend.engines.qwen_engine import QwenTTSEngine, resolve_model_dir
from backend.engines.registry import create_local_engine

REF_TEXT = "Привет, это референс"

# Флаги реального прогона (см. `_real_skip_reason` и docs/testing.md).
REAL_FLAG = "TTS_RUN_REAL_QWEN"
REAL_REF_ENV = "TTS_REAL_REF_AUDIO"
REAL_REF_TEXT_ENV = "TTS_REAL_REF_TEXT"


# --- подставной пакет ---------------------------------------------------------
class _Recorder(dict):
    """Журнал подставного пакета: загрузки, промпты и вызовы генерации."""

    def __init__(self, **kwargs):
        super().__init__(
            loads=[],
            prompts=[],
            generations=[],
            sample_rate=SAMPLE_RATE,
            samples=2400,
            fail_generation_at=None,
            **kwargs,
        )


def fake_package(recorder: _Recorder) -> types.ModuleType:
    """Модуль `qwen_tts` вместо настоящего: тот же интерфейс, никаких весов."""

    class FakeModel:
        def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
            recorder["prompts"].append(
                {
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                    "x_vector_only_mode": x_vector_only_mode,
                }
            )
            return {"ref_audio": ref_audio, "x_vector_only_mode": x_vector_only_mode}

        def generate_voice_clone(self, **kwargs):
            recorder["generations"].append(kwargs)
            if recorder["fail_generation_at"] == len(recorder["generations"]):
                # Так падает операция, которой нет на MPS: движок обязан
                # пережить это и продолжить на CPU, а не потерять кусок.
                raise RuntimeError("MPS backend does not support this op")
            sound = np.zeros(int(recorder["samples"]), dtype=np.float32)
            return [sound], int(recorder["sample_rate"])

    class Qwen3TTSModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            recorder["loads"].append({"path": path, **kwargs})
            return FakeModel()

    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = Qwen3TTSModel
    return module


@pytest.fixture
def qwen(monkeypatch, tmp_path):
    """Движок на подставном пакете плюс журнал его вызовов."""
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_package(recorder))

    model_dir = tmp_path / "models" / "qwen3_tts_base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard")

    monkeypatch.setattr(config, "QWEN_BASE_DIR", model_dir)
    monkeypatch.setattr(config, "QWEN_DEVICE", "auto")
    # Устройство задаётся тестом: проба MPS требует настоящего Apple Silicon.
    monkeypatch.setattr(config, "pick_device", lambda: "cpu")
    monkeypatch.delenv(config.QWEN_FORCE_CPU_ENV, raising=False)

    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"not-a-real-wav")

    engine = QwenTTSEngine(model_dir)
    return engine, recorder, reference


# --- 1. каталог весов ---------------------------------------------------------
def test_resolve_model_dir_finds_sharded_weights(tmp_path):
    root = tmp_path / "hub"
    nested = root / "models--Qwen--Qwen3-TTS" / "snapshots" / "abc"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")
    (nested / "model-00001-of-00002.safetensors").write_bytes(b"x")

    assert resolve_model_dir(root, required=("config.json", "*.safetensors")) == nested


def test_resolve_model_dir_without_weights_is_none(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert resolve_model_dir(root, required=("config.json", "*.safetensors")) is None
    assert resolve_model_dir(tmp_path / "нет-такого-каталога", required=("config.json",)) is None


# --- 2. загрузка --------------------------------------------------------------
def test_load_reports_missing_weights_before_package(tmp_path, monkeypatch):
    """Пустой каталог — понятная ошибка про веса, а не про отсутствие пакета.

    Порядок проверок важен и проверяется здесь: сначала файлы, потом импорт.
    Иначе пользователь без скачанных весов получал бы «нет модуля qwen_tts» и
    чинил бы не то.
    """
    empty = tmp_path / "models" / "qwen3_tts_base"
    empty.mkdir(parents=True)
    monkeypatch.setattr(config, "pick_device", lambda: "cpu")
    monkeypatch.setitem(sys.modules, "qwen_tts", None)  # пакета как будто нет

    engine = QwenTTSEngine(empty)
    with pytest.raises(RuntimeError) as exc:
        engine.load()

    assert "не найдены" in str(exc.value)
    assert engine.state == STATE_FAILED


def test_load_reports_missing_package(tmp_path, monkeypatch):
    """Веса на месте, пакета нет — ошибка называет установку без зависимостей."""
    model_dir = tmp_path / "models" / "qwen3_tts_base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(config, "pick_device", lambda: "cpu")
    monkeypatch.setitem(sys.modules, "qwen_tts", None)

    engine = QwenTTSEngine(model_dir)
    with pytest.raises(RuntimeError) as exc:
        engine.load()

    assert "qwen-tts" in str(exc.value)
    assert "no-deps" in str(exc.value)


def test_load_passes_device_dtype_and_attention(qwen):
    engine, recorder, _reference = qwen
    engine.load()

    assert len(recorder["loads"]) == 1
    load = recorder["loads"][0]
    assert load["device_map"] == "cpu"
    # На CPU bf16 у большинства операций эмулируется — берётся float32.
    assert load["dtype"] == torch.float32
    assert load["attn_implementation"] == config.QWEN_ATTN_IMPLEMENTATION
    assert engine.is_loaded


def test_second_load_is_noop(qwen):
    engine, recorder, _reference = qwen
    engine.load()
    engine.load()
    assert len(recorder["loads"]) == 1


def test_mps_uses_bfloat16(qwen, monkeypatch):
    engine, recorder, _reference = qwen
    monkeypatch.setattr(config, "pick_device", lambda: "mps")

    engine.load()

    load = recorder["loads"][0]
    assert load["device_map"] == "mps"
    # 1.7B в float32 на MPS — это лишние гигабайты и заметно медленнее.
    assert load["dtype"] == torch.bfloat16
    assert engine.device == "mps"
    # Свойство читает проба выполнимости (`tools/qwen_feasibility.py`): dtype
    # должен быть виден снаружи, а не только в журнале подставного пакета.
    assert engine.dtype == "bfloat16"


def test_force_cpu_wins_over_mps(qwen, monkeypatch):
    """Аварийная переменная важнее выбранного устройства: иначе от неё нет толку."""
    engine, recorder, _reference = qwen
    monkeypatch.setattr(config, "pick_device", lambda: "mps")
    monkeypatch.setenv(config.QWEN_FORCE_CPU_ENV, "1")

    engine.load()

    assert recorder["loads"][0]["device_map"] == "cpu"


def test_explicit_device_setting_is_honoured(qwen, monkeypatch):
    engine, recorder, _reference = qwen
    monkeypatch.setattr(config, "QWEN_DEVICE", "cpu")

    engine.load()

    assert recorder["loads"][0]["device_map"] == "cpu"


# --- 3. инференс --------------------------------------------------------------
def test_synthesize_matches_contract(qwen):
    engine, recorder, reference = qwen

    waveform, sample_rate = engine.synthesize("Привет", str(reference), REF_TEXT, 1.0)

    assert sample_rate == SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1

    call = recorder["generations"][0]
    assert call["text"] == "Привет"
    assert call["language"] == config.QWEN_LANGUAGE
    assert call["temperature"] == pytest.approx(config.DEFAULT_QWEN_TEMPERATURE)
    assert call["top_p"] == pytest.approx(config.DEFAULT_QWEN_TOP_P)
    assert call["repetition_penalty"] == pytest.approx(config.DEFAULT_QWEN_REPETITION_PENALTY)
    assert call["max_new_tokens"] == config.DEFAULT_QWEN_MAX_NEW_TOKENS


def test_foreign_sample_rate_is_resampled(qwen):
    """Чужой sample rate — брак сборки, поэтому он приводится к 24 кГц, а не теряется."""
    engine, recorder, reference = qwen
    recorder["sample_rate"] = 16_000
    recorder["samples"] = 16_000  # секунда при 16 кГц

    waveform, sample_rate = engine.synthesize("Привет", str(reference), REF_TEXT, 1.0)

    assert sample_rate == SAMPLE_RATE
    assert len(waveform) == pytest.approx(SAMPLE_RATE, rel=0.02)


def test_speed_changes_length_without_changing_pitch(qwen):
    engine, recorder, reference = qwen
    recorder["samples"] = SAMPLE_RATE  # секунда при 24 кГц

    normal, _ = engine.synthesize("Привет", str(reference), REF_TEXT, 1.0)
    slow, _ = engine.synthesize("Привет", str(reference), REF_TEXT, 0.5)

    assert len(slow) == pytest.approx(2 * len(normal), rel=0.1)


def test_seed_is_applied(qwen, monkeypatch):
    engine, _recorder, reference = qwen
    seen: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda value: seen.append(int(value)))

    engine.synthesize("Привет", str(reference), REF_TEXT, 1.0, seed=42)

    assert seen == [42]


# --- 4. clone-prompt ----------------------------------------------------------
def test_clone_prompt_is_cached_per_reference(qwen, tmp_path):
    engine, recorder, reference = qwen

    engine.synthesize("Раз", str(reference), REF_TEXT, 1.0)
    engine.synthesize("Два", str(reference), REF_TEXT, 1.0)
    # Второй кусок той же реплики не должен пересчитывать промпт: промпт — это
    # фичи на устройстве модели, и их пересчёт стоит как половина синтеза.
    assert len(recorder["prompts"]) == 1
    assert len(recorder["generations"]) == 2

    other = tmp_path / "ref2.wav"
    other.write_bytes(b"another")
    engine.synthesize("Три", str(other), REF_TEXT, 1.0)
    assert len(recorder["prompts"]) == 2


def test_prompt_cache_evicts_oldest(qwen, tmp_path, monkeypatch):
    engine, recorder, reference = qwen
    monkeypatch.setattr(config, "QWEN_PROMPT_CACHE_SIZE", 1)

    other = tmp_path / "ref2.wav"
    other.write_bytes(b"another")

    engine.synthesize("Раз", str(reference), REF_TEXT, 1.0)
    engine.synthesize("Два", str(other), REF_TEXT, 1.0)
    engine.synthesize("Три", str(reference), REF_TEXT, 1.0)

    # Вытеснённый промпт считается заново — иначе кеш рос бы без предела.
    assert len(recorder["prompts"]) == 3


def test_experimental_mode_clones_by_embedding_only(qwen):
    """Экспериментальный режим: клонирование без расшифровки референса."""
    engine, recorder, reference = qwen

    engine.synthesize(
        "Привет", str(reference), REF_TEXT, 1.0, mode=ENGINE_MODE_EXPERIMENTAL
    )

    prompt = recorder["prompts"][0]
    assert prompt["x_vector_only_mode"] is True
    assert prompt["ref_text"] == ""


def test_quality_mode_uses_reference_text(qwen):
    engine, recorder, reference = qwen

    engine.synthesize("Привет", str(reference), REF_TEXT, 1.0, mode="quality")

    prompt = recorder["prompts"][0]
    assert prompt["x_vector_only_mode"] is False
    assert prompt["ref_text"] == REF_TEXT


# --- 5. память и откат на CPU -------------------------------------------------
def test_mps_failure_reloads_on_cpu_and_keeps_the_chunk(qwen, monkeypatch):
    engine, recorder, reference = qwen
    monkeypatch.setattr(config, "pick_device", lambda: "mps")
    recorder["fail_generation_at"] = 1  # первая генерация падает

    waveform, sample_rate = engine.synthesize("Привет", str(reference), REF_TEXT, 1.0)

    assert sample_rate == SAMPLE_RATE
    assert len(waveform) > 0
    assert [load["device_map"] for load in recorder["loads"]] == ["mps", "cpu"]
    assert engine.device == "cpu"


def test_unload_releases_model_and_prompts(qwen):
    engine, recorder, reference = qwen
    engine.synthesize("Привет", str(reference), REF_TEXT, 1.0)
    assert engine.is_loaded

    engine.unload()

    assert not engine.is_loaded
    assert engine.device is None
    assert engine.dtype == ""  # выгруженный движок не притворяется загруженным
    # Промпты — тензоры на устройстве модели: выгрузка без их очистки оставила бы
    # память занятой при внешне «пустом» движке.
    assert engine._prompts == {}

    engine.synthesize("Снова", str(reference), REF_TEXT, 1.0)
    assert len(recorder["loads"]) == 2


# --- 6. реестр ----------------------------------------------------------------
def test_registry_builds_qwen_engine(qwen):
    engine = create_local_engine("qwen3-tts")
    assert isinstance(engine, QwenTTSEngine)
    assert engine.id == "qwen3-tts"


# --- 7. настоящие веса: отдельный, gated прогон -------------------------------
# Всё выше проверено на подставном пакете. Здесь — единственная проверка, которую
# подстановка дать не может: что настоящий `qwen_tts` действительно принимает те
# аргументы, которые ему передаёт адаптер, и что веса на MPS/CPU поднимаются.
# Прогон выключен по умолчанию: он требует ~5 ГБ весов, минуты на загрузку и
# скачивание моделей из сети — то есть не может быть частью `pytest`.
def _real_skip_reason() -> str:
    """Почему реальный прогон сейчас невозможен; пустая строка — возможен.

    Причина формулируется словами, а не через `skipif` с булевым условием: без
    неё непонятно, чего не хватает — флага, весов или записи голоса.
    """
    if os.environ.get(REAL_FLAG, "").strip().lower() not in ("1", "true", "yes"):
        return f"реальный прогон не запрошен: {REAL_FLAG}=1"
    if not model_manager.engine_available(ENGINE_QWEN):
        return "веса Qwen3-TTS не установлены (см. requirements-qwen.txt и docs/install.md)"
    reference = os.environ.get(REAL_REF_ENV, "").strip()
    if not reference:
        return f"не задан путь к записи голоса: {REAL_REF_ENV}=/путь/к/речи.wav (3–15 с)"
    if not Path(reference).is_file():
        return f"файл референса не найден: {reference}"
    return ""


def test_real_model_synthesizes_a_chunk():
    """Настоящая модель: загрузка, клонирование по референсу и контракт звука.

    Проверяется только контракт (частота, тип, непустой звук, длительность), а не
    качество клонирования: его оценивают слухом, и автотест, «проверяющий» его
    числом, создавал бы ложную уверенность. Задача этого теста — поймать
    несовпадение API пакета с адаптером, которое на подставном пакете невидимо.
    """
    reason = _real_skip_reason()
    if reason:
        pytest.skip(reason)

    reference = os.environ[REAL_REF_ENV].strip()
    engine = QwenTTSEngine()
    try:
        engine.load()
        assert engine.is_loaded, engine.last_error
        assert engine.device in ("cpu", "mps", "cuda")
        waveform, sample_rate = engine.synthesize(
            "Проверка связи: один, два, три.",
            reference,
            os.environ.get(REAL_REF_TEXT_ENV, ""),
            1.0,
        )
    finally:
        engine.unload()

    assert sample_rate == SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    # «Слово», а не щелчок: слишком короткий выход означает, что промпт референса
    # разобран неверно и модель не начала говорить.
    assert waveform.size > SAMPLE_RATE * 0.2
    assert float(np.max(np.abs(waveform))) > 0.0
