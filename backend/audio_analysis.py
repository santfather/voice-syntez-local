"""Измерение высоты тона (F0) референс-аудио: проверка заявленного пола голоса.

Метод — разнос гармоник: берём самый энергичный участок записи, считаем спектр,
выбираем основной тон по согласованности спектра с рядом гармоник.
Кросс-проверка — `librosa.yin` (если библиотека доступна), `pyin` для коротких
референсов ненадёжен и не используется.
"""

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_SR = 16_000
FFT_SIZE = 32_768
WINDOW_SEC = 1.5
MIN_AUDIO_SEC = 0.5

# Ориентиры для взрослых голосов
MALE_RANGE = (85.0, 155.0)
FEMALE_RANGE = (165.0, 255.0)

# Границы поиска основного тона
F0_MIN = 60.0
F0_MAX = 450.0
F0_SEARCH_MAX = 350.0
F0_STEP = 0.5
HARMONIC_LIMIT = 2_000.0

# Демо-файлы пакета f5_tts, которые нельзя выдавать за пользовательские записи
DEMO_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass
class F0Estimate:
    f0_hz: float | None  # итоговая оценка (разнос гармоник)
    yin_hz: float | None  # кросс-проверка, может отсутствовать


def load_mono(data: bytes | bytearray | str | Path, suffix: str = "", target_sr: int = ANALYSIS_SR):
    """Декодирует аудио (байты или путь) в моно float32 нужной частоты."""
    from pydub import AudioSegment

    source = BytesIO(bytes(data)) if isinstance(data, (bytes, bytearray)) else str(data)
    fmt = suffix.lstrip(".").lower() or None
    segment = AudioSegment.from_file(source, format=fmt)
    segment = segment.set_channels(1).set_frame_rate(target_sr)
    samples = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, target_sr


def _most_energetic(y: np.ndarray, sr: int, window_sec: float = WINDOW_SEC) -> np.ndarray:
    """Окно записи с максимальной энергией."""
    frame = int(window_sec * sr)
    if len(y) <= frame:
        return y
    # Префиксные суммы вместо np.convolve: O(N) вместо O(N · frame). На записи в
    # десятки миллионов сэмплов это разница между миллисекундами и десятками
    # секунд работы одного ядра (лимит 50 МБ на файл это допускает).
    cumsum = np.concatenate(([0.0], np.cumsum(np.square(y, dtype=np.float64))))
    energy = cumsum[frame:] - cumsum[:-frame]
    start = int(np.argmax(energy))
    return y[start:start + frame]


def _spectrum(segment: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1 / sr)
    return spectrum, freqs


def _smooth(spectrum: np.ndarray, freqs: np.ndarray, width_hz: float = 20.0) -> np.ndarray:
    """Сглаживает спектр: убирает мелкую структуру, оставляя только гармоники."""
    bin_hz = float(freqs[1] - freqs[0])
    half = max(round(width_hz / bin_hz / 2), 1)
    kernel = np.hanning(2 * half + 1)
    return np.convolve(spectrum, kernel / kernel.sum(), mode="same")


def _magnitude_at(spectrum: np.ndarray, freqs: np.ndarray, freq: float, tol: float = 0.02) -> float:
    """Максимум спектра в окне ±tol вокруг частоты (пик может не совпасть с бином)."""
    center = int(np.searchsorted(freqs, freq))
    center = min(max(center, 1), len(spectrum) - 2)
    half = max(round(tol * freq / (freqs[1] - freqs[0])), 1)
    lo, hi = max(center - half, 0), min(center + half + 1, len(spectrum))
    return float(spectrum[lo:hi].max())


def _harmonic_score(f0: float, spectrum: np.ndarray, freqs: np.ndarray, max_harmonics: int = 8) -> float:
    """Насколько спектр объясняется рядом гармоник f0 (0..1, больше — лучше).

    Вес 1/k у первой гармоники гасит октавные ошибки: ряд из удвоенной частоты
    объясняет только чётные гармоники и получает меньший балл.
    """
    harmonics = [f0 * k for k in range(1, max_harmonics + 1) if f0 * k <= HARMONIC_LIMIT]
    if len(harmonics) < 2:
        return -1.0
    peak = float(spectrum.max()) + 1e-9
    total, weights = 0.0, 0.0
    for k, harmonic in enumerate(harmonics, start=1):
        weight = 1.0 / k
        total += weight * _magnitude_at(spectrum, freqs, harmonic) / peak
        weights += weight
    return total / weights


def estimate_f0_harmonic(y: np.ndarray, sr: int) -> float | None:
    """Основной тон по самому энергетичному участку: ищем f0, лучше всего объясняющий гармоники."""
    if y.size < int(MIN_AUDIO_SEC * sr):
        return None
    segment = _most_energetic(y, sr)
    if np.max(np.abs(segment)) < 1e-4:  # тишина
        return None
    spectrum, freqs = _spectrum(segment, sr)
    spectrum = _smooth(spectrum, freqs)
    grid = np.arange(F0_MIN, F0_SEARCH_MAX, F0_STEP)
    scores = [_harmonic_score(f0, spectrum, freqs) for f0 in grid]
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return None
    return float(grid[best])


def estimate_f0_yin(y: np.ndarray, sr: int) -> float | None:
    """Кросс-проверка автокорреляционным трекером. Без librosa вернёт None."""
    try:
        import librosa
    except ImportError:
        logger.info("librosa недоступна — кросс-проверка F0 через yin пропущена")
        return None
    try:
        f0 = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX, sr=sr)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    except Exception as exc:  # noqa: BLE001 — кросс-проверка не должна ронять загрузку голоса
        logger.warning("Не удалось посчитать yin: %s", exc)
        return None
    length = min(len(f0), len(rms))
    f0, rms = np.asarray(f0[:length]), np.asarray(rms[:length])
    voiced = f0 > 0
    if not voiced.any():
        return None
    keep = voiced & (rms >= 0.25 * float(np.median(rms[voiced])))
    if int(keep.sum()) < 3:
        keep = voiced
    return float(np.median(f0[keep]))


def analyze(data: bytes | bytearray | str | Path, suffix: str = "") -> F0Estimate:
    """Оценка F0 референса. Ошибки декодирования не пробрасываются — вернётся пустой результат."""
    try:
        y, sr = load_mono(data, suffix)
    except Exception as exc:  # noqa: BLE001 — проверка не должна блокировать загрузку голоса
        logger.warning("Не удалось прочитать аудио для оценки F0: %s", exc)
        return F0Estimate(f0_hz=None, yin_hz=None)
    f0 = estimate_f0_harmonic(y, sr)
    yin = estimate_f0_yin(y, sr)
    logger.info("F0 референса: разнос гармоник %s Гц, yin %s Гц", f0, yin)
    return F0Estimate(f0_hz=f0, yin_hz=yin)


def check_gender_mismatch(f0: float | None, declared_gender: str) -> str | None:
    """Предупреждение, если измеренный F0 противоречит заявленному полу. Иначе None."""
    if f0 is None:
        return None
    if declared_gender == "female" and f0 < MALE_RANGE[1]:
        return f"Референс размечен как женский, но F0={f0:.0f} Гц попадает в мужской диапазон"
    if declared_gender == "male" and f0 > FEMALE_RANGE[0]:
        return f"Референс размечен как мужской, но F0={f0:.0f} Гц попадает в женский диапазон"
    return None


@lru_cache(maxsize=1)
def _demo_hashes() -> dict[str, str]:
    """md5 -> имя демо-файла из установленного пакета f5_tts."""
    # f5_tts — namespace-пакет, у него нет origin: берём spec подмодуля
    spec = importlib.util.find_spec("f5_tts.api")
    if spec is None or not spec.origin:
        return {}
    examples = Path(spec.origin).parent / "infer" / "examples"
    if not examples.is_dir():
        return {}
    known: dict[str, str] = {}
    for path in sorted(examples.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DEMO_AUDIO_SUFFIXES:
            continue
        known[hashlib.md5(path.read_bytes()).hexdigest()] = path.name
    return known


def find_demo_source(data: bytes | bytearray) -> str | None:
    """Имя демо-файла F5-TTS, если загруженные байты — точная его копия."""
    return _demo_hashes().get(hashlib.md5(bytes(data)).hexdigest())
