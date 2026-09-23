"""Паспорта движков: границы ручек, подсказка по полу и контракт `synthesize`."""

from pathlib import Path

import numpy as np
import pytest

from backend import config
from backend.engines import registry, xtts_engine
from backend.engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    ENGINE_KOKORO,
    ENGINE_MODE_DRAFT,
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_MODE_QUALITY,
    ENGINE_QWEN,
    ENGINE_XTTS,
    SAMPLE_RATE,
    EngineInfo,
    EngineMode,
    EngineParam,
    SynthesisEngine,
    builtin_voices,
    default_engine_for_gender,
    engine_info,
    fallback_engine,
    normalize_engine_mode,
    normalize_engine_params,
    requires_reference,
)


def test_engine_param_clamps_and_rounds():
    param = EngineParam(name="p", label="P", default=0.5, minimum=0.0, maximum=1.0, step=0.1)
    assert param.clamp(2.0) == 1.0
    assert param.clamp(-5) == 0.0
    assert param.clamp("мусор") == 0.5  # нечисло — дефолт, а не падение

    integer = EngineParam(
        name="n", label="N", default=16, minimum=8, maximum=32, step=1, integer=True
    )
    assert integer.clamp(17.6) == 18
    assert isinstance(integer.clamp("20"), int)


def test_normalize_engine_params_keeps_unknown_keys_but_clamps_known():
    raw = {
        "temperature": 9.0,
        "repetition_penalty": 0.1,
        "target_rms": 0.05,  # общая ручка пайплайна — проходит как есть
        "seed": 123,
    }
    result = normalize_engine_params(ENGINE_XTTS, raw)
    assert result["temperature"] == config.XTTS_TEMPERATURE_RANGE[1]
    assert result["repetition_penalty"] == config.XTTS_REPETITION_PENALTY_RANGE[0]
    assert result["target_rms"] == 0.05
    assert result["seed"] == 123


def test_engine_info_falls_back_to_f5_and_exposes_params():
    assert engine_info(ENGINE_XTTS).params  # у XTTS ручки объявлены
    assert engine_info("нет-такого").id == ENGINE_F5
    payload = ENGINE_INFOS[ENGINE_XTTS].to_dict()
    assert {"temperature", "repetition_penalty"} <= {param["name"] for param in payload["params"]}


def test_default_engine_hint_by_gender():
    assert default_engine_for_gender("female") == ENGINE_XTTS
    assert default_engine_for_gender("male") == ENGINE_F5
    assert default_engine_for_gender("other") == ENGINE_F5


def test_registry_rejects_unknown_engine_without_silent_substitution():
    with pytest.raises(ValueError):
        registry.get_engine("нет-такого")


class _WrongRateEngine(SynthesisEngine):
    """Движок, отдающий чужую частоту дискретизации."""

    info = ENGINE_INFOS[ENGINE_F5]

    def load(self) -> None:
        self._mark("ready")

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        return np.zeros(SAMPLE_RATE, dtype=np.float32), 16_000


def test_synthesize_rejects_wrong_sample_rate():
    engine = _WrongRateEngine()
    with pytest.raises(RuntimeError, match="вместо"):
        engine.synthesize("Привет", "/tmp/ref.wav", "Привет")


# --- паспорт: возможности, а не догадки вызывающего кода ----------------------
def test_engine_passport_declares_capabilities():
    qwen = ENGINE_INFOS[ENGINE_QWEN]
    assert qwen.supports_seed is True
    assert qwen.supports_cloning is True
    assert qwen.supports_prosody_profiles is True
    assert "ru" in qwen.languages
    assert qwen.fallback_engine == ENGINE_F5
    assert qwen.default_mode == ENGINE_MODE_QUALITY

    payload = qwen.to_dict()
    assert {"supports_cloning", "supports_prosody_profiles", "languages", "modes"} <= set(payload)
    assert [mode["id"] for mode in payload["modes"]] == [
        ENGINE_MODE_DRAFT,
        ENGINE_MODE_QUALITY,
        ENGINE_MODE_EXPERIMENTAL,
    ]


def test_all_declared_engines_have_their_rules_in_the_passport():
    """F5 и XTTS остались клонирующими: новые поля не поменяли их поведение."""
    for engine_id in (ENGINE_F5, ENGINE_XTTS):
        info = ENGINE_INFOS[engine_id]
        assert info.supports_cloning is True
        assert info.supports_prosody_profiles is True
        assert info.languages == ("ru",)
        # Откат по недоступности никому, кроме Qwen, не разрешён: молчаливая
        # подмена звучания F5 на XTTS была бы хуже падения.
        assert info.fallback_engine == ""
        assert info.modes == ()


def test_normalize_engine_mode_keeps_declared_and_rejects_unknown():
    assert normalize_engine_mode(ENGINE_QWEN, ENGINE_MODE_DRAFT) == ENGINE_MODE_DRAFT
    # Пустой выбор означает объявленный по умолчанию режим, а не «никакого».
    assert normalize_engine_mode(ENGINE_QWEN, "") == ENGINE_MODE_QUALITY
    # Выдуманный режим не подменяется дефолтным молча — иначе «эксперимент»
    # превратился бы в «качество», и пользователь слушал бы не то, что выбрал.
    assert normalize_engine_mode(ENGINE_QWEN, "выдумка") == ""
    assert normalize_engine_mode(ENGINE_F5, ENGINE_MODE_DRAFT) == ""  # режимов нет


def test_fallback_and_reference_helpers():
    assert requires_reference(ENGINE_QWEN) is True
    assert fallback_engine(ENGINE_QWEN) == ENGINE_F5
    assert fallback_engine(ENGINE_F5) == ""  # отката нет — и не выдумывается
    assert fallback_engine("нет-такого") == ""  # неизвестный id тоже не цель отката


def test_builtin_voices_are_declared_only_by_the_engine_that_speaks_them():
    """Встроенные голоса объявляет паспорт, и только у движка без клонирования.

    Этот список — единственный источник имён: и `voices_store` заводит по нему
    карточки, и `/api/engines` отдаёт его интерфейсу. Выдумывать имена в
    хранилище или на фронте значило бы завести второй источник правды и
    разойтись с движком при первой же смене чекпоинта.
    """
    assert requires_reference(ENGINE_KOKORO) is False
    voices = builtin_voices(ENGINE_KOKORO)
    # По голосу на каждый пол: движок выбирает встроенный голос по полу карточки,
    # и без пары «мужской/женский» часть выбора просто не звучала бы.
    assert [voice.gender for voice in voices] == ["female", "male", "other"]
    assert all(voice.id and voice.label for voice in voices)
    # Клонирующему движку заводить нечего: пустой набор, а не выдуманные имена.
    assert builtin_voices(ENGINE_F5) == ()
    assert builtin_voices(ENGINE_QWEN) == ()
    # Опечатка в id не рождает голосов: паспорт откатывается к F5.
    assert builtin_voices("нет-такого") == ()

    payload = ENGINE_INFOS[ENGINE_KOKORO].to_dict()
    assert payload["builtin_voices"] == [voice.to_dict() for voice in voices]
    assert set(payload["builtin_voices"][0]) == {"id", "label", "gender"}


class _ModeEngine(SynthesisEngine):
    """Движок с режимами: проверяет слияние пресета, дефолтов и явных ручек."""

    info = EngineInfo(
        id="тест-режимы",
        label="Режимы",
        description="",
        supports_accents=False,
        params=(
            EngineParam(
                name="steps",
                label="Шаги",
                default=32,
                minimum=8,
                maximum=64,
                step=1,
                integer=True,
            ),
        ),
        modes=(
            EngineMode(id=ENGINE_MODE_DRAFT, label="Черновик", overrides=(("steps", 8),)),
            EngineMode(id=ENGINE_MODE_QUALITY, label="Качество"),
        ),
        default_mode=ENGINE_MODE_QUALITY,
    )

    def __init__(self, info: EngineInfo | None = None) -> None:
        super().__init__()
        if info is not None:
            self.info = info
        self.seen: dict = {}

    def load(self) -> None:
        self._mark("ready")

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        self.seen = dict(params)
        return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE


def test_mode_overrides_defaults_but_explicit_params_win():
    engine = _ModeEngine()

    engine.synthesize("Привет", "/tmp/ref.wav", "Привет")
    # Режим по умолчанию — «качество»: у него нет переопределений.
    assert engine.seen["steps"] == 32

    engine.synthesize("Привет", "/tmp/ref.wav", "Привет", mode=ENGINE_MODE_DRAFT)
    assert engine.seen["steps"] == 8

    # Явная ручка важнее пресета: иначе выбранное значение молча вернулось бы к режимному.
    engine.synthesize("Привет", "/tmp/ref.wav", "Привет", mode=ENGINE_MODE_DRAFT, steps=64)
    assert engine.seen["steps"] == 64


def test_mode_overrides_are_clamped_to_declared_bounds():
    """Пресет режима проходит ту же валидацию, что и ручки с фронта."""
    info = EngineInfo(
        id="тест-границ",
        label="Границы",
        description="",
        supports_accents=False,
        params=(
            EngineParam(name="steps", label="Шаги", default=32, minimum=8, maximum=64, step=1),
        ),
        modes=(EngineMode(id=ENGINE_MODE_DRAFT, label="Черновик", overrides=(("steps", 1),)),),
        default_mode=ENGINE_MODE_DRAFT,
    )
    engine = _ModeEngine(info)

    engine.synthesize("Привет", "/tmp/ref.wav", "Привет")

    # Значение пресета ниже минимума: приведено к границе, а не ушло в модель.
    assert engine.seen["steps"] == 8


# --- откат XTTS с MPS на CPU --------------------------------------------------
def _xtts_on_mps(monkeypatch, error: Exception):
    """XTTS с MPS и подставным инференсом, падающим первым вызовом.

    Веса не грузятся: проверяется только решение «перезагружать на CPU или нет».
    """
    engine = xtts_engine.XTTSEngine(ENGINE_XTTS, Path("/нет-такого-чекпоинта"))
    engine._device = "mps"
    engine._model = object()
    calls = {"infer": 0, "loaded": [], "released": 0}

    def fake_infer(text, ref_audio_path, speed, params):
        calls["infer"] += 1
        if calls["infer"] == 1:
            raise error
        return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE

    def fake_load(device):
        calls["loaded"].append(device)
        engine._device = device
        engine._model = object()

    def fake_release():
        calls["released"] += 1

    monkeypatch.setattr(engine, "_infer", fake_infer)
    monkeypatch.setattr(engine, "_load_model", fake_load)
    monkeypatch.setattr(xtts_engine, "release_torch_memory", fake_release)
    return engine, calls


def test_xtts_fallback_ignores_errors_that_are_not_about_mps(monkeypatch):
    """Сбой синтеза на MPS — это не сбой MPS: тихий перевод на CPU прятал ошибку."""
    engine, calls = _xtts_on_mps(monkeypatch, ValueError("Пустой текст реплики"))

    with pytest.raises(ValueError, match="Пустой текст"):
        engine._synthesize("", "/tmp/ref.wav", "", 1.0, {})

    assert calls["loaded"] == []  # модель не перезагружалась
    assert calls["released"] == 0


def test_xtts_fallback_moves_to_cpu_and_frees_metal_pool(monkeypatch):
    engine, calls = _xtts_on_mps(monkeypatch, RuntimeError("MPS backend out of memory"))

    wave, rate = engine._synthesize("Привет", "/tmp/ref.wav", "Привет", 1.0, {})

    assert rate == SAMPLE_RATE and wave.size == SAMPLE_RATE
    assert calls["loaded"] == ["cpu"]
    # Старый Metal-пул освобождается до перезагрузки: иначе unified memory удваивается.
    assert calls["released"] == 1


def test_xtts_fallback_does_not_trigger_on_cpu_device(monkeypatch):
    """На CPU «mps» в тексте ошибки ничего не значит — перезагружать некуда."""
    engine, calls = _xtts_on_mps(monkeypatch, RuntimeError("unsupported MPS op"))
    engine._device = "cpu"

    with pytest.raises(RuntimeError, match="unsupported MPS"):
        engine._synthesize("Привет", "/tmp/ref.wav", "Привет", 1.0, {})

    assert calls["loaded"] == []
    assert calls["released"] == 0
