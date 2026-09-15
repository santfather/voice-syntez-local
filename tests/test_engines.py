"""Паспорта движков: границы ручек, подсказка по полу и контракт `synthesize`."""

import numpy as np
import pytest

from backend import config
from backend.engines import registry
from backend.engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    ENGINE_XTTS,
    SAMPLE_RATE,
    EngineParam,
    SynthesisEngine,
    default_engine_for_gender,
    engine_info,
    normalize_engine_params,
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
