"""Проверка референс-аудио: высота тона (F0) против заявленного пола, перегрузка записи
и полоса частот.

Метод измерения тона — разнос гармоник: берём самый энергичный участок записи,
считаем спектр, выбираем основной тон по согласованности спектра с рядом гармоник.
Кросс-проверка — `librosa.yin` (если библиотека доступна), `pyin` для коротких
референсов ненадёжен и не используется.

Вторая проверка — клиппинг: у референса с перегруженным микрофоном пики срезаны,
и F5-TTS клонирует эти искажения вместе с тембром.

Третья — полоса частот: узкополосная запись (телефонная линия, сильно сжатый mp3)
даёт глухой тембр, который модель клонирует вместе с голосом и который настройками
синтеза потом не лечится.
"""

import hashlib
import importlib.util
import logging
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_SR = 16_000
FFT_SIZE = 32_768
WINDOW_SEC = 1.5
MIN_AUDIO_SEC = 0.5

# Полоса частот меряется на своей частоте дискретизации: на 16 кГц (ANALYSIS_SR)
# выше 8 кГц ничего нет, а проверке нужен диапазон до 11 кГц.
BAND_SR = 24_000
BAND_FRAME = 2_048
BAND_CORE_HZ = (300.0, 3_400.0)  # ядро речи — уровень, с которым сравниваем
BAND_HIGH_HZ = (7_000.0, 11_000.0)  # верх: его и срезают телефонные кодеки и сжатые mp3
# Замерено на референсах проекта и на их копиях через ФНЧ: записи в полной полосе
# дают −8…−19 дБ относительно ядра речи, обрезанные на 7 кГц — от −34 дБ и ниже,
# на 6 кГц и ниже — от −67 дБ. Порог −28 дБ посередине между этими группами.
BAND_HIGH_MIN_DB = -28.0

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

# Перегрузка: доля отсчётов, упёршихся в потолок шкалы. Порог по доле, а не по
# одному пику: одиночный отсчёт на пределе бывает и у нормальной записи, а вот
# срезанные вершины громких звуков дают их десятками на секунду.
CLIPPING_LEVEL = 0.99
CLIPPING_MIN_RATIO = 0.001


@dataclass
class ReferenceReport:
    """Итог проверки референса: высота тона и дефекты записи."""

    f0_hz: float | None  # итоговая оценка (разнос гармоник)
    yin_hz: float | None  # кросс-проверка, может отсутствовать
    duration_sec: float  # длительность записи; 0.0 — декодировать не удалось
    clipping_warning: str | None  # микрофон перегружен
    band_high_db: float | None = None  # уровень 6–8 кГц относительно ядра речи, дБ
    band_warning: str | None = None  # запись узкополосная (телефон, сжатый mp3)


def load_mono(data: bytes | bytearray | str | Path, suffix: str = "", target_sr: int = ANALYSIS_SR):
    """Декодирует аудио (байты или путь) в моно float32 нужной частоты.

    Через ffmpeg напрямую, а не через pydub: pydub отдаёт сырые сэмплы в том
    формате, который выбрал для вывода ffmpeg, и разрядность приходится угадывать
    по ширине сэмпла. Для opus (а это любая запись из браузера: webm/ogg) ffmpeg
    выдаёт 32-битный float, а `AudioSegment.sample_width` остаётся 4 — тот же
    размер и у int32. Разбор такого потока как int16 удваивал число сэмплов:
    одиннадцатисекундная запись превращалась в 22 секунды, а тон 150 Гц — в 76.
    Явный `s16le` на выходе убирает неоднозначность.

    `suffix` сохранён в подписи для совместимости с вызовами: ffmpeg определяет
    формат сам, в том числе по имени файла.
    """
    del suffix
    if isinstance(data, (bytes, bytearray)):
        # `cache:` делает трубу сэмплов ищущей: контейнеры mp4/m4a (их отдаёт
        # Safari) без этого не демультиплексируются — «Invalid data found».
        command = ["-read_ahead_limit", "-1", "-i", "cache:pipe:0"]
        stdin = bytes(data)
    else:
        command = ["-i", str(data)]
        stdin = None
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *command,
         "-ac", "1", "-ar", str(target_sr), "-f", "s16le", "-"],
        input=stdin,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        tail = process.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        raise RuntimeError(f"ffmpeg не смог прочитать аудио: {tail[0] if tail else process.returncode}")
    samples = np.frombuffer(process.stdout, dtype="<i2").astype(np.float32) / 32768.0
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


def clipping_ratio(y: np.ndarray) -> float:
    """Доля отсчётов, упёршихся в потолок шкалы.

    Единственная точка замера перегруза: `check_clipping` выносит по ней вердикт
    для референса, а диагностика take'а кладёт её же числом в метаданные. Второй
    счётчик той же величины разошёлся бы с первым при первой же правке порога.
    """
    if y.size == 0:
        return 0.0
    return int(np.count_nonzero(np.abs(y) >= CLIPPING_LEVEL)) / y.size


def check_clipping(y: np.ndarray) -> str | None:
    """Предупреждение, если микрофон был перегружен и вершины громких звуков срезаны."""
    if y.size == 0:
        return None
    ratio = clipping_ratio(y)
    if ratio < CLIPPING_MIN_RATIO:
        return None
    peak_db = 20 * math.log10(max(float(np.max(np.abs(y))), 1e-6))
    return (
        f"Запись перегружена: {ratio * 100:.1f}% отсчётов упираются в потолок "
        f"шкалы (пик {peak_db:.1f} дБ). Микрофон срезает вершины громких звуков, и модель "
        f"склонирует искажения вместе с тембром. Убавьте усиление микрофона или отойдите "
        f"дальше и запишите заново."
    )


def _band_mean(spectrum: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float | None:
    """Средняя амплитуда спектра в полосе [lo, hi)."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return None
    return float(np.mean(spectrum[mask]))


def estimate_high_band_db(y: np.ndarray, sr: int) -> float | None:
    """Уровень полосы 7–11 кГц относительно ядра речи (300–3400 Гц), дБ.

    Средний спектр берётся по всем кадрам, а не по самым громким: усредняются
    амплитуды, а вклад тихих кадров на порядки меньше речи, поэтому длинная пауза
    в записи оценку не сдвигает (проверено на записи, где речи — треть). Зато
    шипящие, которые тише гласных, остаются в расчёте — а именно они и пропадают
    у узкополосных записей.
    """
    if y.size < BAND_FRAME or float(np.max(np.abs(y))) < 1e-4:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(y, BAND_FRAME)[:: BAND_FRAME // 2]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(BAND_FRAME), axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(BAND_FRAME, d=1 / sr)
    core = _band_mean(spectrum, freqs, *BAND_CORE_HZ)
    high = _band_mean(spectrum, freqs, *BAND_HIGH_HZ)
    if not core or high is None:
        return None
    return 20 * math.log10(high / core)


def check_narrow_band(high_db: float | None) -> str | None:
    """Предупреждение, если в записи нет верха: тембр выйдет глухим. Иначе None."""
    if high_db is None or high_db >= BAND_HIGH_MIN_DB:
        return None
    return (
        f"Запись узкополосная: выше 7 кГц в ней практически ничего нет "
        f"(уровень {high_db:.0f} дБ от ядра речи). Так звучат телефонные записи и сильно "
        f"сжатые mp3 — синтез унаследует глухой тембр, и настройками это не лечится. "
        f"Запишите референс заново в полной полосе."
    )


def analyze(data: bytes | bytearray | str | Path, suffix: str = "") -> ReferenceReport:
    """Проверка референса: F0, дефекты записи и полоса частот. Ошибки декодирования не пробрасываются."""
    try:
        y, sr = load_mono(data, suffix)
    except Exception as exc:  # noqa: BLE001 — проверка не должна блокировать загрузку голоса
        logger.warning("Не удалось прочитать аудио для проверки референса: %s", exc)
        return ReferenceReport(f0_hz=None, yin_hz=None, duration_sec=0.0, clipping_warning=None)
    f0 = estimate_f0_harmonic(y, sr)
    yin = estimate_f0_yin(y, sr)
    clipping = check_clipping(y)
    duration = y.size / sr
    # Полоса меряется отдельной декодировкой на 24 кГц: на 16 кГц выше 8 кГц
    # ничего нет, а сравнивать нужно именно 6–8 кГц.
    try:
        wide, wide_sr = load_mono(data, suffix, target_sr=BAND_SR)
        band_high = estimate_high_band_db(wide, wide_sr)
    except Exception as exc:  # noqa: BLE001 — проверка полосы не должна ронять F0
        logger.warning("Не удалось измерить полосу референса: %s", exc)
        band_high = None
    band_warning = check_narrow_band(band_high)
    logger.info(
        "Референс: %.1f с, F0: разнос гармоник %s Гц, yin %s Гц, полоса 7–11 кГц %s дБ%s",
        duration, f0, yin,
        f"{band_high:.1f}" if band_high is not None else "—",
        " (узкая полоса)" if band_warning else "",
    )
    return ReferenceReport(
        f0_hz=f0,
        yin_hz=yin,
        duration_sec=duration,
        clipping_warning=clipping,
        band_high_db=band_high,
        band_warning=band_warning,
    )


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
