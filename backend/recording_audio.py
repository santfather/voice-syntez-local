"""DSP записанного диалога: декодирование, обработка голоса, кодирование (§8, §11–§16).

Правило, из которого следует всё остальное: **сырая запись не изменяется никогда**.
Обработка всегда идёт «raw → текущие настройки → processed», и потому повторное
движение ползунка не накапливает артефакты (§16, §39).

Существующие помощники проекта переиспользуются, а не переписываются: обрезка
краёв, pitch shift, LUFS-нормализация и лимитер живут в `audio_pipeline` и
применяются к записанному звуку как к готовому waveform. Новыми являются только
две вещи, которых в проекте не было: декодирование браузерной записи в общий
формат и time-stretch (скорость без изменения высоты).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from . import audio_pipeline, config, denoise
from .engines.base import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Расширения, которые браузер реально отдаёт (WebM/Opus, MP4/AAC, WAV, OGG).
# Проверяется не имя файла, а успешность декодирования: контейнер может соврать,
# а декодер — нет (§23).
ALLOWED_SUFFIXES = (".webm", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".wav", ".mp3", ".flac")


class RecordingAudioError(ValueError):
    """Запись не декодируется или настройки обработки вне допустимого."""


def validate_speed(speed: float) -> float:
    low, high = config.RECORDING_SPEED_RANGE
    value = float(speed)
    if not (low <= value <= high):
        raise RecordingAudioError(
            f"Скорость {value:g} вне допустимого диапазона {low:g}–{high:g}"
        )
    return value


def validate_pitch(semitones: float) -> float:
    low, high = config.RECORDING_PITCH_RANGE
    value = float(semitones)
    if not (low <= value <= high):
        raise RecordingAudioError(
            f"Высота {value:g} пт вне допустимого диапазона {low:g}…{high:g}"
        )
    return value


def validate_suffix(filename: str) -> str:
    """Расширение из имени файла **или** из готового суффикса (`.wav`).

    Обе формы приходят с разных сторон: от браузера — имя файла, из хранилища —
    уже суффикс. Проверять их одним кодом дешевле, чем помнить о двух формах.
    """
    suffix = str(filename or "").strip().lower()
    if not suffix.startswith("."):
        suffix = Path(suffix).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise RecordingAudioError(
            f"Формат {suffix or 'без расширения'} не поддерживается записью"
        )
    return suffix


def decode(data: bytes, suffix: str = ".webm") -> np.ndarray:
    """Декодирует запись браузера в моно float32 при `SAMPLE_RATE`.

    Один внутренний формат на весь режим (§8): дальше всё — trim, stretch, pitch,
    LUFS — работает с PCM, а не с WebM/Opus. Ошибка декодирования — это ошибка
    запроса с понятным текстом, а не 500.
    """
    if not data:
        raise RecordingAudioError("Пустая запись")
    if len(data) > config.MAX_RECORDING_BYTES:
        raise RecordingAudioError(
            f"Запись больше {config.MAX_RECORDING_BYTES // (1024 * 1024)} МБ"
        )
    suffix = validate_suffix(suffix) if suffix else ".webm"
    with tempfile.TemporaryDirectory(prefix="tts-recording-") as tmpdir:
        source = Path(tmpdir) / f"source{suffix}"
        source.write_bytes(data)
        try:
            return audio_pipeline._read_audio(source, suffix.lstrip("."))
        except Exception as exc:
            raise RecordingAudioError(f"Запись не декодируется: {exc}") from exc


def duration_sec(audio: np.ndarray) -> float:
    return float(np.asarray(audio).size) / float(SAMPLE_RATE)


def level_warning(audio: np.ndarray) -> str:
    """Лёгкая проверка уровня записи: тихо/перегруз. Предупреждение, не блокер (§24).

    Никакой ML: пик и RMS по waveform. Задача — подсказать пользователю, что
    микрофон далеко или перегружен, до того как он запишет весь диалог.
    """
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return "пустая запись"
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    if peak <= 1e-6:
        return "в записи тишина"
    if peak >= 0.999:
        return "перегруз: пик упирается в потолок"
    if rms < 0.01:
        return "слишком тихая запись — микрофон дальше, чем нужно"
    return ""


def trim_edges(audio: np.ndarray, text: str = "") -> np.ndarray:
    """Срезает тишину по краям записи, сохраняя запас вокруг речи (§27).

    Переиспользуется существующая обрезка: пауза между репликами задаётся
    `pause_ms`, и тишина внутри записи не должна к ней добавляться. Запас шире
    для коротких реплик — у них край и есть слово (то же правило, что в синтезе).
    """
    data = np.asarray(audio, dtype=np.float32)
    guard = audio_pipeline._short_guard_ms(text) if text else config.EDGE_SILENCE_MARGIN_MS
    return audio_pipeline._trim_edge_silence(data, guard)


def time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    """Меняет темп **без изменения высоты** (§12).

    Обычный resample здесь запрещён: он меняет и длительность, и высоту, то есть
    даёт «ускоренную кассету» вместо коррекции темпа. Используется phase vocoder
    из librosa (`time_stretch`) — та же библиотека, что уже применяется для
    pitch shift, поэтому новых зависимостей нет.
    """
    rate = validate_speed(speed)
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if abs(rate - 1.0) < 1e-3 or data.size == 0:
        return data
    import librosa

    # `rate` у librosa — коэффициент скорости: 1.25 быстрее, 0.75 медленнее.
    stretched = librosa.effects.time_stretch(data, rate=float(rate))
    return np.asarray(stretched, dtype=np.float32)


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Меняет высоту, сохраняя длительность. Существующий помощник проекта (§13)."""
    value = validate_pitch(semitones)
    return audio_pipeline._pitch_shift(np.asarray(audio, dtype=np.float32), value)


def denoise_audio(audio: np.ndarray) -> tuple[np.ndarray, str]:
    """Очищает запись существующим DeepFilterNet'ом (§14).

    Возвращается и причина отказа: отсутствие denoiser'а — не ошибка записи, а
    выключенный тумблер с объяснением, и синтез/монтаж из-за него не страдает.
    Сырой дубль не затрагивается — очистка идёт над копией (§16, §39).
    """
    if not denoise.is_available():
        return np.asarray(audio, dtype=np.float32), "DeepFilterNet не установлен"
    try:
        cleaned = _as_wav_bytes(audio)
        result = denoise.clean_bytes(cleaned, ".wav")
        with tempfile.TemporaryDirectory(prefix="tts-recording-denoise-") as tmpdir:
            path = Path(tmpdir) / "cleaned.wav"
            path.write_bytes(result)
            return audio_pipeline._read_audio(path, "wav"), ""
    except Exception as exc:  # noqa: BLE001 — очистка необязательна
        logger.warning("Очистка записи не удалась: %s", exc)
        return np.asarray(audio, dtype=np.float32), f"очистка не выполнена: {exc}"


def _as_wav_bytes(audio: np.ndarray) -> bytes:
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="tts-recording-wav-") as tmpdir:
        path = Path(tmpdir) / "take.wav"
        sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV")
        return path.read_bytes()


def gain_safety(audio: np.ndarray, target_rms: float | None = None) -> np.ndarray:
    """Мягкое выравнивание уровня и защита от клиппинга.

    Не «нормализация до одинакового тембра»: уровень подтягивается к общей
    целевой громкости, а итоговая воспринимаемая громкость всё равно
    контролируется финальным LUFS-проходом (§29).
    """
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    target = config.DEFAULT_TARGET_RMS if target_rms is None else float(target_rms)
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms > 1e-6:
        data = data * (target / rms)
    return audio_pipeline._limit_peaks(data)


def settings_key(speed: float, pitch_semitones: float, denoise: bool) -> str:
    """Короткий ключ настроек для имени кэша обработки (§16)."""
    payload = json.dumps(
        {
            "speed": round(float(speed), 4),
            "pitch": round(float(pitch_semitones), 4),
            "denoise": bool(denoise),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def process(
    raw: np.ndarray,
    *,
    speed: float = 1.0,
    pitch_semitones: float = 0.0,
    denoise_enabled: bool = False,
    text: str = "",
) -> tuple[np.ndarray, list[str]]:
    """Единый неразрушающий конвейер обработки дубля (§15).

    Порядок зафиксирован пакетом: очистка → обрезка краёв → темп → высота →
    уровень. Обрезка идёт до растяжения, потому что после него границы речи
    размываются; уровень — последним, потому что он зависит от результата всех
    предыдущих шагов. Возвращаются также предупреждения (например, об отказе
    очистки), чтобы интерфейс мог их показать.
    """
    warnings: list[str] = []
    audio = np.asarray(raw, dtype=np.float32).reshape(-1)
    if denoise_enabled:
        audio, reason = denoise_audio(audio)
        if reason:
            warnings.append(reason)
    audio = trim_edges(audio, text)
    audio = time_stretch(audio, speed)
    audio = pitch_shift(audio, pitch_semitones)
    audio = gain_safety(audio)
    return audio, warnings


def encode(audio: np.ndarray, output_format: str = "wav") -> bytes:
    """Кодирует готовый мастер в WAV или MP3 (через существующую запись файлов)."""
    with tempfile.TemporaryDirectory(prefix="tts-recording-encode-") as tmpdir:
        path = Path(tmpdir) / f"master.{output_format}"
        audio_pipeline._write_audio(path, np.asarray(audio, dtype=np.float32), output_format)
        return path.read_bytes()


__all__ = [
    "ALLOWED_SUFFIXES",
    "RecordingAudioError",
    "decode",
    "denoise_audio",
    "duration_sec",
    "encode",
    "gain_safety",
    "level_warning",
    "pitch_shift",
    "process",
    "settings_key",
    "time_stretch",
    "trim_edges",
    "validate_pitch",
    "validate_speed",
    "validate_suffix",
]
