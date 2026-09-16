"""Дешёвый отбор подозрительных кусков для Smart QA.

Отбор смотрит только на сам waveform и на текст, который в него отправляли:
распознавание речи здесь не запускается. Смысл в асимметрии ошибок — лишний
запуск Whisper стоит десятков секунд, а пропущенный брак уезжает в готовый файл,
поэтому признаки подобраны так, чтобы нормальный кусок проходил наверняка, а
подозрительный уходил на расшифровку даже при небольшом сомнении.

Пороги живут в `config` (блок `QA_SCREEN_*`): они подобраны под 24 кГц выход
движков и русскую речь.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from . import config
from .audio_analysis import check_clipping
from .engines.base import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Причины — коды, а не готовые фразы: их видит и интерфейс (переводит в подписи),
# и тесты (сравнивают точно), и лог. Формулировки на русском тут не приживаются.
REASON_EMPTY = "empty"
REASON_TOO_SHORT = "too_short"
REASON_DURATION_SHORT = "duration_short"
REASON_DURATION_LONG = "duration_long"
REASON_SILENCE = "silence"
REASON_CLIPPING = "clipping"
REASON_LEVEL = "level"
REASON_REPEAT = "repeat"


@dataclass
class Screening:
    """Вердикт отбора: подозрительный ли кусок и почему."""

    suspicious: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"suspicious": self.suspicious, "reasons": list(self.reasons)}


def _frame_rms(y: np.ndarray, frame: int) -> np.ndarray:
    """RMS по кадрам без остатка в хвосте: кадр короче окна — это не измерение."""
    usable = y.size - y.size % frame
    if usable <= 0:
        return np.empty(0, dtype=np.float64)
    frames = y[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def _to_db(values: np.ndarray) -> np.ndarray:
    """Амплитуда в дБ с полом вместо -inf: цифровая тишина тоже должна сравниваться."""
    return 20.0 * np.log10(np.maximum(values, 1e-9))


def _expected_sec(text: str) -> float:
    """Сколько кусок должен звучать по своему тексту.

    Оценка грубая и намеренно широкая: темп речи, паузы и растянутые гласные
    гуляют сильно, а нам нужно поймать не «не тот темп», а обрыв или зацикливание.
    """
    return max(len(text) / config.QA_SCREEN_CHARS_PER_SEC, config.QA_SCREEN_MIN_EXPECTED_SEC)


def _has_repeat(y: np.ndarray, frame: int) -> bool:
    """Ищет участок, который повторяет более ранний участок того же куска.

    Нормированная автокорреляция при сдвиге 0.4–2 с: у зацикленной модели кусок
    повторяется почти дословно, и корреляция на этом сдвиге близка к единице.
    Замер идёт только там, где меняется громкость: ровный тон (затянутая гласная,
    гудок) коррелирует с собой при сдвиге, кратного периоду, и без этой оговорки
    отбор браковал бы каждую длинную гласную.
    """
    envelope = _frame_rms(y, frame)
    if envelope.size < 4:
        return False
    envelope_db = _to_db(envelope)
    if float(envelope_db.max() - envelope_db.min()) < config.QA_SCREEN_REPEAT_MIN_ENVELOPE_DB:
        return False

    min_lag = int(SAMPLE_RATE * config.QA_SCREEN_REPEAT_MIN_LAG_SEC)
    max_lag = min(int(SAMPLE_RATE * config.QA_SCREEN_REPEAT_MAX_LAG_SEC), y.size - 1)
    if max_lag <= min_lag:
        return False

    # Считаем всю автокорреляцию одним преобразованием, а нужный диапазон сдвигов
    # выбираем уже из результата: перебирать сдвиги по одному на минуте аудио
    # дороже, чем один rfft.
    signal = np.asarray(y, dtype=np.float64)
    size = signal.size
    window = 1
    while window < 2 * size:
        window <<= 1
    spectrum = np.fft.rfft(signal, window)
    acorr = np.fft.irfft(spectrum * np.conj(spectrum), window)[: max_lag + 1]

    # Перекрытие на каждом сдвиге разное, и без нормировки длинный кусок
    # «коррелировал» бы сам с собой просто потому, что он длинный.
    energy = np.cumsum(np.square(signal))
    lags = np.arange(min_lag, max_lag + 1)
    head = energy[size - lags - 1]
    tail = energy[-1] - np.where(lags > 0, energy[lags - 1], 0.0)
    denom = np.sqrt(np.maximum(head * tail, 0.0))
    corr = np.divide(
        acorr[lags], denom, out=np.zeros(lags.shape, dtype=np.float64), where=denom > 0
    )
    return bool(float(np.max(corr)) >= config.QA_SCREEN_REPEAT_CORR)


def screen_chunk(chunk: np.ndarray, text: str) -> Screening:
    """Вердикт по куску: отправлять ли его в Whisper.

    Читает сырой выход движка — до `_prepare_chunk`: тот выравнивает громкость и
    срезает края, и после него ни тихий, ни перегруженный кусок уже не отличить
    от нормального.
    """
    y = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if y.size == 0:
        # Пустой waveform — не «подозрительный», а заведомо сорвавшийся синтез;
        # измерять в нём нечего, поэтому дальше не идём.
        return Screening(suspicious=True, reasons=[REASON_EMPTY])

    duration = y.size / SAMPLE_RATE
    expected = _expected_sec(text)
    reasons: list[str] = []
    if duration < config.QA_SCREEN_MIN_DURATION_SEC:
        reasons.append(REASON_TOO_SHORT)
    if duration < expected * config.QA_SCREEN_SHORT_RATIO:
        reasons.append(REASON_DURATION_SHORT)
    elif duration > expected * config.QA_SCREEN_LONG_RATIO:
        reasons.append(REASON_DURATION_LONG)

    frame = max(1, int(SAMPLE_RATE * config.QA_SCREEN_FRAME_SEC))
    frames = _frame_rms(y, frame)
    if frames.size:
        frames_db = _to_db(frames)
        silent = float(np.count_nonzero(frames_db < config.QA_SCREEN_SILENCE_DB)) / frames.size
        if silent > config.QA_SCREEN_SILENCE_RATIO:
            reasons.append(REASON_SILENCE)
    level_db = float(_to_db(np.array([np.sqrt(np.mean(np.square(y, dtype=np.float64)))]))[0])
    if not (config.QA_SCREEN_RMS_FLOOR_DB <= level_db <= config.QA_SCREEN_RMS_CEIL_DB):
        reasons.append(REASON_LEVEL)
    # Детектор перегруза общий с загрузкой голосов: второй экземпляр той же
    # проверки разошёлся бы с первым при первой же правке порога.
    if check_clipping(y):
        reasons.append(REASON_CLIPPING)
    if frames.size and _has_repeat(y, frame):
        reasons.append(REASON_REPEAT)

    if reasons:
        logger.info(
            "Отбор куска (%.2f c, ожидалось %.2f c, уровень %.1f дБ): подозрительный — %s",
            duration, expected, level_db, ", ".join(reasons),
        )
    return Screening(suspicious=bool(reasons), reasons=reasons)


def describe_reasons(reasons: list[str]) -> str:
    """Причины одной строкой по-русски — для хода задачи в интерфейсе."""
    labels = {
        REASON_EMPTY: "пустое аудио",
        REASON_TOO_SHORT: "слишком короткий кусок",
        REASON_DURATION_SHORT: "короче текста",
        REASON_DURATION_LONG: "длиннее текста",
        REASON_SILENCE: "много тишины",
        REASON_CLIPPING: "перегруз",
        REASON_LEVEL: "аномальный уровень",
        REASON_REPEAT: "повтор участка",
    }
    return ", ".join(labels.get(reason, reason) for reason in reasons)
