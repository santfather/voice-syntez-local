"""Границы целевой реплики в контекстном синтезе (§11, §30, §31).

Распознавание подменяется провайдером (`asr=`): проверяется логика границ, а не
Whisper. Модели не поднимаются.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import short_utterance as su
from backend import short_utterance_boundary as b
from backend.engines.base import SAMPLE_RATE
from backend.transcribe import WordStamp


def _tone(seconds: float, amplitude: float = 0.2, frequency: float = 180.0) -> np.ndarray:
    timeline = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _phrases_audio() -> np.ndarray:
    """«Хорошо. Привет! Хорошо.» — контекст, цель, контекст (с паузами)."""
    return np.concatenate(
        [
            _silence(0.05),
            _tone(0.8),
            _silence(0.25),
            _tone(0.6, frequency=240.0),
            _silence(0.25),
            _tone(0.8),
        ]
    ).astype(np.float32)


def _stamps(*items: tuple[str, float, float]) -> list[WordStamp]:
    return [WordStamp(word=word, start=start, end=end) for word, start, end in items]


def test_asr_boundary_finds_target_in_middle_for_both_sides():
    """При carrier'е с двух сторон цель берётся из середины, а не первое вхождение."""
    audio = _phrases_audio()
    stamps = _stamps(
        ("хорошо", 0.05, 0.85),
        ("привет", 1.10, 1.70),
        ("хорошо", 1.95, 2.75),
    )
    cropped, boundary = b.crop_to_target(
        audio, "Привет!", method=b.METHOD_ASR, side=su.SIDE_BOTH, asr=lambda _: stamps
    )
    assert boundary is not None
    assert boundary.production_ready is True
    assert boundary.matched_words == 1
    # Длительность — около секунды с запасом по краям, а не вся запись.
    assert 0.5 <= cropped.size / SAMPLE_RATE <= 0.9
    assert cropped.size < audio.size


def test_asr_boundary_takes_prefix_target_at_the_end():
    audio = _phrases_audio()
    stamps = _stamps(("хорошо", 0.05, 0.85), ("да", 1.10, 1.45))
    cropped, boundary = b.crop_to_target(
        audio, "Да.", method=b.METHOD_ASR, side=su.SIDE_PREFIX, asr=lambda _: stamps
    )
    assert boundary is not None
    assert boundary.start > 0
    assert cropped.size < audio.size


def test_asr_boundary_returns_none_without_match():
    """Не нашли цель — работаем без контекста, а не режем наугад."""
    audio = _phrases_audio()
    stamps = _stamps(("совершенно", 0.05, 0.85), ("другое", 1.10, 1.45))
    cropped, boundary = b.crop_to_target(
        audio, "Привет!", method=b.METHOD_ASR, asr=lambda _: stamps
    )
    assert boundary is None
    assert cropped.size == audio.size


def test_asr_boundary_requires_confidence():
    """Пропуск одного слова из пяти ещё граница, двух — уже нет."""
    audio = _phrases_audio()
    target = "Раз два три четыре пять"
    four_of_five = _stamps(
        ("раз", 0.1, 0.3), ("два", 0.4, 0.6), ("три", 0.7, 0.9), ("четыре", 1.0, 1.2)
    )
    found = b.find_boundary_asr(audio, target, asr=lambda _: four_of_five)
    assert found is not None
    assert found.confidence == pytest.approx(0.8)

    three_of_five = _stamps(("раз", 0.1, 0.3), ("три", 0.7, 0.9), ("пять", 1.3, 1.5))
    assert b.find_boundary_asr(audio, target, asr=lambda _: three_of_five) is None


def test_asr_boundary_tolerates_broken_asr():
    """Распознавание недоступно — это не ошибка синтеза, а отсутствие границы."""
    audio = _phrases_audio()

    def broken(_audio):
        raise RuntimeError("Whisper не поднялся")

    cropped, boundary = b.crop_to_target(audio, "Привет!", asr=broken)
    assert boundary is None
    assert cropped.size == audio.size


def test_asr_boundary_returns_none_for_empty_target():
    assert b.find_boundary_asr(_phrases_audio(), "", asr=lambda _: []) is None


def test_silence_boundary_is_experiment_only():
    """Пауза не подтверждает текст, поэтому в производство такой границе нельзя."""
    audio = _phrases_audio()
    found = b.find_boundary_silence(audio, "Привет!", side=su.SIDE_PREFIX)
    assert found is not None
    assert found.method == b.METHOD_SILENCE
    assert found.production_ready is False

    cropped, boundary = b.crop_to_target(audio, "Привет!", method=b.METHOD_SILENCE)
    assert boundary is None, "в бою пауза не считается надёжной границей"
    assert cropped.size == audio.size

    # В режиме эксперимента (benchmark) та же граница возвращается.
    cropped, boundary = b.crop_to_target(
        audio, "Привет!", method=b.METHOD_SILENCE, require_production=False
    )
    assert boundary is not None
    assert cropped.size < audio.size


def test_boundary_pads_edges():
    """К найденному интервалу добавляется запас: атаку короткой фразы легко срезать."""
    audio = _phrases_audio()
    stamps = _stamps(("привет", 1.10, 1.70))
    found = b.find_boundary_asr(audio, "Привет!", asr=lambda _: stamps)
    assert found is not None
    assert found.start < int(1.10 * SAMPLE_RATE)
    assert found.end > int(1.70 * SAMPLE_RATE)


def test_unknown_boundary_method_is_rejected():
    with pytest.raises(ValueError, match="Неизвестный метод границ"):
        b.find_boundary(_phrases_audio(), "Привет!", method="магия")


def test_boundary_metadata_is_serializable():
    stamps = _stamps(("привет", 1.10, 1.70))
    found = b.find_boundary_asr(_phrases_audio(), "Привет!", asr=lambda _: stamps)
    payload = found.to_dict()
    assert payload["method"] == b.METHOD_ASR
    assert payload["matched_words"] == 1
    assert payload["production_ready"] is True
    assert payload["duration_sec"] > 0
