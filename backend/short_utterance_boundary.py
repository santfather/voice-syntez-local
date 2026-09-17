"""Границы целевой реплики в контекстном синтезе (§11, §30, §31).

Когда модель получает больше текста, чем нужно озвучить («контекст + цель»),
готовый WAV должен содержать **только цель**. Значит, результат нужно разрезать —
и резать «примерно по времени» нельзя: расчётная длительность не совпадает с
фактической (модель дышит, тянет гласные, вставляет паузы), и в файл попадут
куски контекста.

Два способа, оба без новых зависимостей:

* `asr` — таймстемпы слов у того же Whisper, что уже используется для проверки
  качества (`transcribe.transcribe_words`). Целевые слова ищутся как
  подпоследовательность в распознанном тексте, границы берутся у первого и
  последнего совпавшего слова. Это единственный способ, пригодный для
  производства: он подтверждает, что найден именно целевой текст.
* `silence` — поиск паузы между контекстом и целью по кадрам тишины. Дешёвый, но
  ненадёжный: пауза внутри самой реплики неотличима от паузы между фрагментами.
  Поэтому он живёт только как вариант benchmark'а (`allow_production=False`).

Правило отката: нет границы с достаточной уверенностью — стратегия возвращается к
DIRECT. Оставить в файле часть carrier'а хуже, чем не улучшить короткую реплику.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import config
from . import short_utterance as su
from .engines.base import SAMPLE_RATE
from .transcribe import WordStamp

logger = logging.getLogger(__name__)

METHOD_ASR = "asr"
METHOD_SILENCE = "silence"
METHODS = (METHOD_ASR, METHOD_SILENCE)

# Минимальная доля найденных целевых слов, при которой границе можно верить.
MIN_ASR_CONFIDENCE = 0.8
# Уверенность «по паузе»: её заведомо мало для производства, и это не украшение.
SILENCE_CONFIDENCE = 0.4
# Запас по краям: атака короткой фразы и её хвост легко обрезаются «по границе
# слова», поэтому к найденному интервалу добавляется немного времени.
PAD_BEFORE_SEC = 0.06
PAD_AFTER_SEC = 0.10
# Пауза короче этого не считается границей между фрагментами: модель делает
# паузы и внутри фразы.
MIN_PAUSE_SEC = 0.12

AsrProvider = Callable[[np.ndarray], list[WordStamp]]


@dataclass(frozen=True)
class Boundary:
    """Найденный интервал целевой реплики в общем синтезе."""

    start: int  # сэмпл
    end: int
    method: str
    confidence: float
    matched_words: int = 0
    total_words: int = 0
    note: str = ""

    @property
    def duration_sec(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE

    @property
    def production_ready(self) -> bool:
        """Можно ли резать по этой границе в бою.

        Годится только ASR с достаточной уверенностью: «по паузе» остаётся
        экспериментом, потому что не подтверждает, что найден нужный текст.
        """
        return self.method == METHOD_ASR and self.confidence >= MIN_ASR_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "duration_sec": round(self.duration_sec, 3),
            "matched_words": self.matched_words,
            "total_words": self.total_words,
            "production_ready": self.production_ready,
            "note": self.note,
        }


def _normalized_words(text: str) -> list[str]:
    """Слова для сверки: без ударений, регистра и «ё» (Whisper пишет «е»)."""
    return [word.replace("ё", "е") for word in su._words(text)]


def _match_window(
    target: list[str], heard: list[str], start: int
) -> tuple[int, int, int]:
    """Жадно сопоставляет целевые слова с распознанными, начиная с позиции `start`.

    Возвращает `(matched, first_index, last_index)`. Пропуски в распознанном тексте
    допускаются (Whisper теряет служебные слова), но порядок обязан сохраняться —
    иначе это не наша фраза.
    """
    matched = 0
    first = -1
    last = -1
    position = start
    for word in target:
        while position < len(heard) and heard[position] != word:
            position += 1
        if position >= len(heard):
            break
        if first < 0:
            first = position
        last = position
        matched += 1
        position += 1
    return matched, first, last


def _expected_center_fraction(side: str) -> float:
    """Где примерно должна оказаться цель при данной стороне контекста.

    Нужно, чтобы выбрать правильное вхождение, когда целевой текст встречается
    дважды: при carrier'е с двух сторон (§7) цель в середине, при префиксе — в
    конце, при суффиксе — в начале.
    """
    if side == su.SIDE_SUFFIX:
        return 0.25
    if side == su.SIDE_BOTH:
        return 0.5
    return 0.75


def find_boundary_asr(
    audio: np.ndarray,
    target_text: str,
    *,
    side: str = su.SIDE_PREFIX,
    asr: AsrProvider | None = None,
) -> Boundary | None:
    """Границы цели по таймстемпам слов распознавания.

    `asr` инъектируется (в тестах — без Whisper): на вход идёт waveform, на выходе
    список слов с границами. Реальный провайдер пишет временный wav и зовёт
    `transcribe.transcribe_words`.
    """
    target = _normalized_words(target_text)
    if not target:
        return None
    try:
        stamps = (asr or _default_asr)(audio)
    except Exception as exc:  # noqa: BLE001 — нет распознавания → нет границы → DIRECT
        logger.warning("Границы по словам не найдены: распознавание недоступно (%s)", exc)
        return None
    if not stamps:
        return None

    heard = [stamp.word.lower().replace("ё", "е").strip(".,!?;:…—-\"'") for stamp in stamps]
    heard = [word for word in heard if word]
    if not heard:
        return None

    best: tuple[int, int, int, float] | None = None
    for start in range(len(heard)):
        matched, first, last = _match_window(target, heard, start)
        if matched == 0 or first < 0:
            continue
        center_fraction = ((first + last) / 2 + 0.5) / len(heard)
        distance = abs(center_fraction - _expected_center_fraction(side))
        if best is None or (matched, -distance) > (best[0], -best[3]):
            best = (matched, first, last, distance)
    if best is None:
        return None

    matched, first, last, _ = best
    confidence = matched / len(target)
    if confidence < MIN_ASR_CONFIDENCE:
        logger.info(
            "Граница цели ненадёжна: совпало %d слов из %d — работаю без контекста",
            matched, len(target),
        )
        return None

    start_sec = max(stamps[first].start - PAD_BEFORE_SEC, 0.0)
    end_sec = stamps[last].end + PAD_AFTER_SEC
    start = int(start_sec * SAMPLE_RATE)
    end = min(int(end_sec * SAMPLE_RATE), audio.size)
    if end <= start:
        return None
    return Boundary(
        start=start,
        end=end,
        method=METHOD_ASR,
        confidence=confidence,
        matched_words=matched,
        total_words=len(target),
        note="по таймстемпам слов",
    )


def find_boundary_silence(
    audio: np.ndarray,
    target_text: str,
    *,
    side: str = su.SIDE_PREFIX,
) -> Boundary | None:
    """Границы цели по самой длинной паузе (только эксперимент).

    Пауза ищется в средней половине записи: в начале и в конце паузы быть не может
    (там контекст и цель), а внутри фразы паузы короче `MIN_PAUSE_SEC`. Метод не
    подтверждает, что найден нужный текст, поэтому `production_ready` у него
    всегда `False`.
    """
    from . import qa_screening

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(SAMPLE_RATE * config.QA_SCREEN_FRAME_SEC))
    frames = qa_screening._frame_rms(y, frame)
    if frames.size < 3:
        return None
    quiet = qa_screening._to_db(frames) < config.QA_SCREEN_SILENCE_DB
    best_length = 0
    best_index = -1
    index = 0
    while index < quiet.size:
        if not quiet[index]:
            index += 1
            continue
        start = index
        while index < quiet.size and quiet[index]:
            index += 1
        length = index - start
        center = (start + index) / 2 / quiet.size
        # Пауза-кандидат: достаточно длинная и не в самом начале/конце записи.
        if length > best_length and 0.25 <= center <= 0.75:
            best_length = length
            best_index = start
    pause_sec = best_length * config.QA_SCREEN_FRAME_SEC
    if best_index < 0 or pause_sec < MIN_PAUSE_SEC:
        return None
    cut = best_index * frame
    if side == su.SIDE_SUFFIX:
        start, end = 0, cut
    else:
        start, end = cut, y.size
    if end - start <= 0:
        return None
    return Boundary(
        start=start,
        end=end,
        method=METHOD_SILENCE,
        confidence=SILENCE_CONFIDENCE,
        note=f"по паузе {pause_sec:.2f} c (эксперимент)",
    )


def _default_asr(audio: np.ndarray) -> list[WordStamp]:
    """Реальный провайдер: временный wav + Whisper с таймстемпами слов."""
    import tempfile
    from pathlib import Path

    from .audio_pipeline import _write_audio
    from .transcribe import transcribe_words

    with tempfile.NamedTemporaryFile(suffix=".wav", prefix="voice-syntez-", delete=False) as handle:
        path = Path(handle.name)
    try:
        _write_audio(path, np.asarray(audio, dtype=np.float32), "wav")
        return transcribe_words(path)
    finally:
        path.unlink(missing_ok=True)


def find_boundary(
    audio: np.ndarray,
    target_text: str,
    *,
    method: str = METHOD_ASR,
    side: str = su.SIDE_PREFIX,
    asr: AsrProvider | None = None,
) -> Boundary | None:
    """Единая точка: граница выбранным методом (или `None`, если не нашлась)."""
    if method == METHOD_ASR:
        return find_boundary_asr(audio, target_text, side=side, asr=asr)
    if method == METHOD_SILENCE:
        return find_boundary_silence(audio, target_text, side=side)
    raise ValueError(f"Неизвестный метод границ: {method}")


def crop_to_target(
    audio: np.ndarray,
    target_text: str,
    *,
    method: str = METHOD_ASR,
    side: str = su.SIDE_PREFIX,
    asr: AsrProvider | None = None,
    require_production: bool = True,
) -> tuple[np.ndarray, Boundary | None]:
    """Вырезает целевую реплику из контекстного синтеза.

    Возвращает `(аудио цели, граница)`. Если границы нет (или она не годится для
    производства при `require_production=True`), возвращается исходное аудио и
    `None` — вызывающий обязан в этом случае откатиться на DIRECT и синтезировать
    цель отдельно, а не отдавать пользователю контекст в файле.
    """
    boundary = find_boundary(audio, target_text, method=method, side=side, asr=asr)
    if boundary is None:
        return audio, None
    if require_production and not boundary.production_ready:
        logger.info(
            "Граница цели не годится для производства (%s, уверенность %.2f) — откат на DIRECT",
            boundary.method, boundary.confidence,
        )
        return audio, None
    return np.asarray(audio[boundary.start : boundary.end], dtype=np.float32), boundary
