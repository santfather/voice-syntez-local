"""DSP записанного диалога: декодирование, темп, высота, очистка и кодирование (§35).

Все сигналы синтетические (`conftest.sine`), реальные ML-модели не поднимаются:
DeepFilterNet подменяется моками, librosa используется как настоящая библиотека —
именно она отвечает за «темп без изменения высоты» и «высоту без изменения
длительности», и проверять её заглушкой значило бы не проверить ничего.

Ключевое свойство, которое ловят эти тесты: конвейер **неразрушающий** — он всегда
начинается с переданного сырого сигнала и не накапливает артефакты между
прогонами (§16).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from conftest import dominant_hz, sine

from backend import config, recording_audio
from backend.engines.base import SAMPLE_RATE
from backend.recording_audio import RecordingAudioError

# Допуски: librosa — фазовый вокодер, он не обязан сохранять длину до сэмпла.
LENGTH_TOLERANCE = 0.1
PITCH_TOLERANCE = 0.05


def _wav_bytes(seconds: float = 0.6, freq: float = 220.0, amplitude: float = 0.3) -> bytes:
    """WAV-байты, как их отдал бы браузер: моно, `SAMPLE_RATE`, float32."""
    buffer = io.BytesIO()
    sf.write(buffer, sine(seconds, freq, amplitude), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


# --- 9. Декодирование --------------------------------------------------------
def test_decode_returns_mono_float32_at_project_sample_rate(workspace):
    """Внутренний формат один для всего режима: моно float32 при `SAMPLE_RATE`."""
    audio = recording_audio.decode(_wav_bytes(0.5), ".wav")
    assert audio.dtype == np.float32
    assert audio.ndim == 1, "стерео или многоканальный сигнал сломал бы всю обработку"
    assert audio.size == int(SAMPLE_RATE * 0.5)
    assert recording_audio.duration_sec(audio) == pytest.approx(0.5, abs=1e-6)


def test_decode_rejects_broken_empty_and_foreign_format(workspace):
    """Битые байты, пустая запись и чужое расширение — ошибка запроса, а не 500."""
    with pytest.raises(RecordingAudioError):
        recording_audio.decode(b"not audio at all", ".wav")
    with pytest.raises(RecordingAudioError):
        recording_audio.decode(b"", ".wav")
    with pytest.raises(RecordingAudioError):
        recording_audio.decode(_wav_bytes(0.2), ".exe")
    # «Отсутствующий» суффикс: имя без расширения не поддерживается.
    with pytest.raises(RecordingAudioError):
        recording_audio.validate_suffix("take")
    with pytest.raises(RecordingAudioError):
        recording_audio.validate_suffix("take.exe")


def test_decode_rejects_recording_over_size_limit(workspace, monkeypatch):
    """Лимит размера отсекает запись до декодирования, а не после него."""
    monkeypatch.setattr(config, "MAX_RECORDING_BYTES", 512)
    with pytest.raises(RecordingAudioError):
        recording_audio.decode(_wav_bytes(0.6), ".wav")


def test_decode_normalizes_stereo_and_foreign_sample_rate(workspace):
    """Запись с чужой частотой и стерео приводится к внутреннему формату.

    Микрофон и браузер легко отдают 48 кГц и два канала, а весь режим считает
    сэмплы как «моно при `SAMPLE_RATE`». Без приведения формата секунда записи
    превращается в четыре секунды, а темп и паузы уезжают вчетверо.
    """
    source_rate = 48000
    timeline = np.arange(source_rate, dtype=np.float32) / source_rate
    stereo = np.stack(
        [
            0.3 * np.sin(2 * np.pi * 220.0 * timeline),
            0.3 * np.sin(2 * np.pi * 330.0 * timeline),
        ],
        axis=1,
    )
    buffer = io.BytesIO()
    sf.write(buffer, stereo, source_rate, format="WAV")

    audio = recording_audio.decode(buffer.getvalue(), ".wav")
    assert audio.ndim == 1, "каналы должны быть сведены в моно"
    assert recording_audio.duration_sec(audio) == pytest.approx(1.0, rel=0.02)


# --- 10. Уровень записи ------------------------------------------------------
def test_level_warning_flags_silence_quiet_and_clipping(workspace):
    """Тишина, тихая запись и перегруз распознаются, нормальный сигнал — молчит."""
    assert "тишин" in recording_audio.level_warning(np.zeros(SAMPLE_RATE // 4, np.float32))
    assert "тихая" in recording_audio.level_warning(sine(0.5, amplitude=0.005))
    assert "перегруз" in recording_audio.level_warning(sine(0.5, amplitude=1.0))
    # Рабочий уровень не должен показывать предупреждение: иначе пользователь
    # привыкнет игнорировать подсказку.
    assert recording_audio.level_warning(sine(0.5, amplitude=0.3)) == ""


# --- 11–12. Темп -------------------------------------------------------------
def test_time_stretch_shortens_by_rate_and_unity_is_a_noop(workspace, monkeypatch):
    """1.25 короче примерно в 1.25 раза, а 1.0 вообще не трогает librosa."""
    audio = sine(2.0, 220.0)
    faster = recording_audio.time_stretch(audio, 1.25)
    assert faster.size == pytest.approx(audio.size / 1.25, rel=LENGTH_TOLERANCE)

    calls: list[dict] = []
    import librosa

    monkeypatch.setattr(
        librosa.effects,
        "time_stretch",
        lambda *args, **kwargs: calls.append(kwargs) or audio,
    )
    same = recording_audio.time_stretch(audio, 1.0)
    assert calls == [], "на скорости 1.0 фазовый вокодер вызываться не должен"
    assert np.array_equal(same, audio)


def test_time_stretch_keeps_pitch_instead_of_resampling(workspace):
    """Темп меняется без изменения высоты — это и отличает stretch от resample.

    Ресемплинг сдвинул бы доминирующую частоту ровно в 1.25 раза (220 → 275 Гц),
    то есть дал бы «ускоренную кассету». Тест ловит именно такую подмену.
    """
    audio = sine(2.0, 220.0)
    stretched = recording_audio.time_stretch(audio, 1.25)
    assert dominant_hz(stretched) == pytest.approx(220.0, rel=PITCH_TOLERANCE)


# --- 13. Высота --------------------------------------------------------------
def test_pitch_shift_changes_pitch_keeping_duration(workspace):
    """+3 полутона поднимают частоту и не меняют длительность; 0 — не меняет сигнал."""
    audio = sine(2.0, 220.0)
    up = recording_audio.pitch_shift(audio, 3.0)
    assert up.size == pytest.approx(audio.size, rel=LENGTH_TOLERANCE)
    assert dominant_hz(up) == pytest.approx(220.0 * 2 ** (3 / 12), rel=PITCH_TOLERANCE)
    assert np.array_equal(recording_audio.pitch_shift(audio, 0.0), audio)


# --- 14. Валидация настроек --------------------------------------------------
def test_validate_speed_and_pitch_reject_out_of_range(workspace):
    """Настройки вне диапазона отклоняются, границы диапазона — допустимы."""
    speed_low, speed_high = config.RECORDING_SPEED_RANGE
    pitch_low, pitch_high = config.RECORDING_PITCH_RANGE
    for value in (speed_low - 0.1, speed_high + 0.1, 5.0):
        with pytest.raises(RecordingAudioError):
            recording_audio.validate_speed(value)
    for value in (pitch_low - 1, pitch_high + 1, 50.0):
        with pytest.raises(RecordingAudioError):
            recording_audio.validate_pitch(value)
    assert recording_audio.validate_speed(speed_low) == speed_low
    assert recording_audio.validate_pitch(pitch_high) == pitch_high


# --- 15. Очистка -------------------------------------------------------------
def test_denoise_falls_back_to_raw_when_unavailable_or_broken(workspace, monkeypatch):
    """Отсутствие DeepFilterNet — не ошибка записи: возвращается сырой сигнал и причина."""
    audio = sine(1.0, 220.0)

    monkeypatch.setattr(recording_audio.denoise, "is_available", lambda: False)
    silent, reason = recording_audio.denoise_audio(audio)
    assert np.array_equal(silent, audio), "без denoiser'а сигнал обязан остаться сырым"
    assert reason, "пользователь должен узнать, почему очистка не применилась"

    monkeypatch.setattr(recording_audio.denoise, "is_available", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("модель недоступна")

    monkeypatch.setattr(recording_audio.denoise, "clean_bytes", broken)
    failed, reason = recording_audio.denoise_audio(audio)
    assert np.array_equal(failed, audio)
    assert "модель недоступна" in reason


# --- 16–17. Конвейер ---------------------------------------------------------
def test_process_skips_denoiser_when_disabled_and_keeps_length(workspace, monkeypatch):
    """Выключенный тумблер не должен даже заходить в очистку; 1.0 и 0 — не меняют длину."""
    calls: list[np.ndarray] = []

    def spy(data: np.ndarray):
        calls.append(data)
        return np.asarray(data, dtype=np.float32), ""

    monkeypatch.setattr(recording_audio, "denoise_audio", spy)
    audio = sine(1.5, 220.0)
    processed, warnings = recording_audio.process(
        audio, speed=1.0, pitch_semitones=0.0, denoise_enabled=False
    )
    assert calls == [], "denoise_enabled=False обязан обходить очистку"
    assert warnings == []
    assert processed.size == pytest.approx(audio.size, rel=LENGTH_TOLERANCE)


def test_process_always_starts_from_raw_and_is_repeatable(workspace):
    """Второй прогон с другими настройками не зависит от первого (§16).

    Если бы конвейер обрабатывал «поверх результата», второй вызов с настройками
    по умолчанию вернул бы уже испорченный сигнал — тест это ловит.
    """
    raw = sine(1.5, 220.0)
    untouched = raw.copy()

    first, _ = recording_audio.process(raw, speed=1.5, pitch_semitones=4.0)
    second, _ = recording_audio.process(raw, speed=1.0, pitch_semitones=0.0)
    repeated, _ = recording_audio.process(raw, speed=1.0, pitch_semitones=0.0)

    assert np.array_equal(raw, untouched), "сырой сигнал изменён обработкой"
    assert first.size < second.size, "первый прогон должен был реально ускорить запись"
    assert np.allclose(second, repeated), "результат зависит от предыдущего прогона"
    assert second.size == pytest.approx(raw.size, rel=LENGTH_TOLERANCE)


# --- 18. Кодирование ---------------------------------------------------------
def test_encode_writes_wav_and_mp3_with_correct_magic(workspace):
    """Готовый мастер обязан открываться плеером: RIFF для WAV и MP3-контейнер для MP3."""
    audio = sine(1.0, 220.0)
    wav = recording_audio.encode(audio, "wav")
    assert wav[:4] == b"RIFF"
    mp3 = recording_audio.encode(audio, "mp3")
    assert mp3, "MP3 пустой — файл не откроется"
    assert mp3[:3] == b"ID3" or mp3[0] == 0xFF


# --- 19. Ключ настроек -------------------------------------------------------
def test_settings_key_changes_with_every_setting(workspace):
    """Ключ кэша различает скорость, высоту и очистку и совпадает при равных настройках."""
    base = recording_audio.settings_key(1.0, 0.0, False)
    assert base == recording_audio.settings_key(1.0, 0.0, False)
    assert base != recording_audio.settings_key(1.2, 0.0, False)
    assert base != recording_audio.settings_key(1.0, 3.0, False)
    assert base != recording_audio.settings_key(1.0, 0.0, True)
