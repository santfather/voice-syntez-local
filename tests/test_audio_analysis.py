"""Проверка референса: полоса частот, перегрузка, высота тона и пол."""

import subprocess

import numpy as np
import pytest
import soundfile as sf
from conftest import sine

from backend import audio_analysis, config
from backend.engines.base import SAMPLE_RATE


def _narrow_band_signal(seconds: float = 3.0) -> np.ndarray:
    """Запись без верха: так звучит телефонная линия или сильно сжатый mp3."""
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    wave = np.zeros_like(t)
    for freq in (200.0, 400.0, 600.0, 800.0):
        wave += 0.2 * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)


def _full_band_signal(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.2 * rng.standard_normal(int(SAMPLE_RATE * seconds))).astype(np.float32)


def test_narrow_band_reference_is_reported(tmp_path):
    path = tmp_path / "narrow.wav"
    sf.write(path, _narrow_band_signal(), SAMPLE_RATE)
    report = audio_analysis.analyze(path)
    assert report.duration_sec == pytest.approx(3.0, abs=0.05)
    assert report.band_high_db is not None and report.band_high_db < audio_analysis.BAND_HIGH_MIN_DB
    assert report.band_warning is not None
    assert "узкополосная" in report.band_warning


def test_full_band_reference_passes_band_check(tmp_path):
    path = tmp_path / "wide.wav"
    sf.write(path, _full_band_signal(), SAMPLE_RATE)
    report = audio_analysis.analyze(path)
    assert report.band_high_db is not None and report.band_high_db > audio_analysis.BAND_HIGH_MIN_DB
    assert report.band_warning is None


def test_band_threshold_returns_warning_only_below_limit():
    assert audio_analysis.check_narrow_band(-40.0) is not None
    assert audio_analysis.check_narrow_band(-10.0) is None
    assert audio_analysis.check_narrow_band(None) is None


def test_clipping_detected_on_overloaded_recording():
    clipped = np.clip(_full_band_signal() * 20.0, -1.0, 1.0).astype(np.float32)
    assert audio_analysis.check_clipping(clipped) is not None
    assert audio_analysis.check_clipping(_full_band_signal()) is None
    assert audio_analysis.check_clipping(np.zeros(0, dtype=np.float32)) is None


def test_f0_estimate_finds_fundamental_of_harmonic_tone():
    t = np.arange(int(SAMPLE_RATE * 2.0), dtype=np.float32) / SAMPLE_RATE
    wave = np.zeros_like(t)
    for harmonic in range(1, 17):  # гармонический ряд 120 Гц до 2 кГц
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * 120.0 * harmonic * t)
    f0 = audio_analysis.estimate_f0_harmonic(wave.astype(np.float32), SAMPLE_RATE)
    assert f0 == pytest.approx(120.0, abs=4.0)


def test_check_gender_mismatch():
    assert audio_analysis.check_gender_mismatch(120.0, "female") is not None
    assert audio_analysis.check_gender_mismatch(200.0, "male") is not None
    assert audio_analysis.check_gender_mismatch(200.0, "female") is None
    assert audio_analysis.check_gender_mismatch(None, "male") is None


def test_analyze_does_not_raise_on_broken_audio(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"definitely not an audio stream")
    report = audio_analysis.analyze(path)
    # Проверка не должна блокировать загрузку голоса — только сообщить нулевую длительность.
    assert report.duration_sec == 0.0
    assert report.f0_hz is None


def test_analyze_reports_duration_of_plain_tone(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, sine(1.5, 180.0), SAMPLE_RATE)
    report = audio_analysis.analyze(path)
    assert report.duration_sec == pytest.approx(1.5, abs=0.05)
    assert report.f0_hz == pytest.approx(180.0, abs=4.0)


def test_load_mono_passes_timeout_to_ffmpeg(monkeypatch):
    """Без `timeout=` битый вход держал ffmpeg, а с ним и воркер, вечно."""
    seen: dict = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(audio_analysis.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        audio_analysis.load_mono(b"\x00\x01\x02")

    assert seen["timeout"] == config.FFMPEG_TIMEOUT_SEC
    # Ошибка объясняет причину: зависание молча заменено понятным отказом.
    assert "ffmpeg" in str(exc.value)
    assert "не уложился" in str(exc.value)
