"""Диагностические метрики take'а: объективные измерения готового куска.

Модуль считает то, что можно измерить в самом waveform и тексте: клиппинг,
долю тишины, длительность и секунды на знак, peak/RMS/LUFS, а также переносит
в метаданные вердикт QA (WER, попытки, режим, причины отбора). Из этих чисел
собираются понятные предупреждения — «Обнаружен клиппинг», «Необычно длинная
реплика» и т.п.

**Автоматической оценки естественности голоса здесь нет и не будет.** Метрики
не сворачиваются в один «балл качества», take'ы не ранжируются: LUFS, WER и
доля тишины описывают разные вещи, а «похожесть на живую речь» ими не
измеряется. Решение «этот вариант лучше» принимает человек на слух — ровно так
же, как сравнение движков (`backend/benchmark.py`) не выставляет победителя.
Приложение лишь показывает технические причины, по которым take может звучать
плохо.

Метрики измеряются по подготовленному куску (`_prepare_chunk`), а не по сырому
выходу модели: пользователь слышит именно подготовленное аудио. Пороги не
заводятся заново — они берутся у `qa_screening` и `audio_analysis`, иначе
«подозрительный» и «предупреждение» разошлись бы при первой же правке.

Любая метрика опциональна: отсутствие (`None`) — нормальная ситуация, а не
ошибка. `lufs` не считается на куске короче 400 мс (там `pyloudnorm` бросает
`ValueError`) и на цифровой тишине, где он равен `-inf`; вместо исключения и
бесконечности наружу идёт `None`. Сериализация устойчива к `NaN`/`inf` — они
превращаются в `None`, чтобы JSON оставался валидным.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import audio_analysis, config, qa_screening
from .engines.base import SAMPLE_RATE

if TYPE_CHECKING:  # только для аннотаций: пайплайн импортирует этот модуль сам
    from .audio_pipeline import QaOutcome

logger = logging.getLogger(__name__)

# LUFS считается по целым блокам 400 мс (ITU-R BS.1770): на более коротком куске
# `pyloudnorm.Meter.integrated_loudness` бросает ValueError. Порог взят из самой
# библиотеки, а не подобран: измерять громкость короче окна нечем.
MIN_LUFS_SEC = 0.4
# Ниже этого уровня сигнала LUFS смысла не имеет: цифровая тишина даёт -inf.
SILENCE_EPS = 1e-6
# С какого превышения дБ над полом шкалы число пика ещё информативно. Амплитуда
# ниже 1e-9 — это тишина, и вместо -inf в метаданных лежит пол.
DB_FLOOR = 1e-9

# Русские формулировки предупреждений. Коды — причины `qa_screening`, а не свои:
# у одного признака должен быть один код и в отборе, и в диагностике.
WARNING_TEXTS = {
    qa_screening.REASON_EMPTY: "Пустое аудио",
    qa_screening.REASON_TOO_SHORT: "Слишком короткая реплика",
    qa_screening.REASON_DURATION_SHORT: "Реплика короче своего текста",
    qa_screening.REASON_DURATION_LONG: "Необычно длинная реплика",
    qa_screening.REASON_SILENCE: "Высокая доля тишины",
    qa_screening.REASON_CLIPPING: "Обнаружен клиппинг",
    qa_screening.REASON_LEVEL: "Низкий уровень",
}
# У уровня две стороны, а код один: слишком громкий выход движка — та же причина
# `level`, но формулировка для человека другая.
LEVEL_HIGH_TEXT = "Слишком высокий уровень"


def _finite(value: float | None) -> float | None:
    """`NaN`/`inf`/`None` → `None`: в JSON бесконечность не сериализуется."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 4)


def _optional_float(value: Any) -> float | None:
    """Значение из JSON: числа и числовые строки — да, всё остальное — None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dbfs(amplitude: float) -> float:
    """Амплитуда в дБ с полом вместо -inf — как в `qa_screening._to_db`."""
    return 20.0 * math.log10(max(float(amplitude), DB_FLOOR))


@dataclass
class TakeQuality:
    """Измерения одного take'а и выведенные из них предупреждения.

    Поля необязательны по отдельности: `None` значит «не измерено», и это
    нормальный случай (например, `lufs` у слишком короткого куска). Свести их
    в одно число нельзя — это разные измерения, а не части одного балла.
    """

    duration_sec: float = 0.0
    chars: int = 0
    clipping: bool = False
    clipping_ratio: float = 0.0
    silence_ratio: float | None = None
    duration_per_char: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    lufs: float | None = None
    wer: float | None = None
    qa_attempts: int | None = None
    qa_mode: str | None = None
    screening_reasons: list[str] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Метаданные для БД и API. `NaN`/`inf` и пустые значения — `None`."""
        return {
            "duration_sec": _finite(self.duration_sec),
            "chars": int(self.chars),
            "clipping": bool(self.clipping),
            "clipping_ratio": _finite(self.clipping_ratio),
            "silence_ratio": _finite(self.silence_ratio),
            "duration_per_char": _finite(self.duration_per_char),
            "peak_dbfs": _finite(self.peak_dbfs),
            "rms_dbfs": _finite(self.rms_dbfs),
            "lufs": _finite(self.lufs),
            "wer": _finite(self.wer),
            "qa_attempts": None if self.qa_attempts is None else int(self.qa_attempts),
            "qa_mode": self.qa_mode,
            "screening_reasons": [str(reason) for reason in self.screening_reasons],
            "warnings": [
                {"code": str(item.get("code", "")), "text": str(item.get("text", ""))}
                for item in self.warnings
            ],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TakeQuality | None":
        """Читает метаданные обратно; `None` и неполный словарь — не ошибка.

        Записи, сохранённые до появления фазы, не содержат ни одного из этих
        полей: отсутствующий ключ читается как `None`/пустой список, а не как
        сбой. Незнакомые ключи игнорируются — набор полей будет расти.
        """
        if not data:
            return None
        raw_warnings = data.get("warnings") or []
        raw_reasons = data.get("screening_reasons") or []
        attempts = data.get("qa_attempts")
        return cls(
            duration_sec=_optional_float(data.get("duration_sec")) or 0.0,
            chars=int(_optional_float(data.get("chars")) or 0),
            clipping=bool(data.get("clipping")),
            clipping_ratio=_optional_float(data.get("clipping_ratio")) or 0.0,
            silence_ratio=_optional_float(data.get("silence_ratio")),
            duration_per_char=_optional_float(data.get("duration_per_char")),
            peak_dbfs=_optional_float(data.get("peak_dbfs")),
            rms_dbfs=_optional_float(data.get("rms_dbfs")),
            lufs=_optional_float(data.get("lufs")),
            wer=_optional_float(data.get("wer")),
            qa_attempts=None if attempts is None else int(_optional_float(attempts) or 0),
            qa_mode=None if data.get("qa_mode") is None else str(data["qa_mode"]),
            screening_reasons=[str(reason) for reason in raw_reasons],
            warnings=[
                {"code": str(item.get("code", "")), "text": str(item.get("text", ""))}
                for item in raw_warnings
                if isinstance(item, dict)
            ],
        )


def _lufs(chunk: np.ndarray) -> float | None:
    """Интегрированная громкость; None — кусок короче 400 мс или тишина.

    Исключение и `-inf` наружу не выходят: это ожидаемые случаи, а не сбой, и
    метаданные должны остаться сериализуемыми.
    """
    if chunk.size < int(SAMPLE_RATE * MIN_LUFS_SEC):
        return None
    if float(np.max(np.abs(chunk))) < SILENCE_EPS:
        return None
    try:
        import pyloudnorm as pyln

        loudness = float(pyln.Meter(SAMPLE_RATE).integrated_loudness(chunk))
    except ValueError:  # pyloudnorm: кусок короче своего окна
        return None
    except Exception as exc:  # noqa: BLE001 — метрика не должна ронять сохранение take'а
        logger.warning("Не удалось посчитать LUFS: %s", exc)
        return None
    return float(loudness) if math.isfinite(loudness) else None


def _warnings(reasons: list[str], rms_dbfs: float) -> list[dict]:
    """Причины измерения → понятные русские предупреждения.

    Коды остаются причинами `qa_screening`; формулировка отличается только у
    уровня, у которого две стороны при одном коде.
    """
    result: list[dict] = []
    for code in reasons:
        text = WARNING_TEXTS.get(code, code)
        if code == qa_screening.REASON_LEVEL and rms_dbfs > config.QA_SCREEN_RMS_CEIL_DB:
            text = LEVEL_HIGH_TEXT
        result.append({"code": code, "text": text})
    return result


def measure(
    chunk: np.ndarray,
    text: str,
    *,
    qa: "QaOutcome | None" = None,
    raw: np.ndarray | None = None,
) -> TakeQuality:
    """Измеряет подготовленный кусок и собирает метаданные take'а.

    `chunk` — то, что реально услышит пользователь (после `_prepare_chunk`), а
    не сырой выход модели: после выравнивания громкости часть дефектов уже
    неотличима от нормы, и мерить нужно именно услышанное. `qa` — итог цикла
    проверки, если он был; из него переносятся WER, попытки, режим и причины
    отбора, а сам кусок перемеряется независимо.

    `raw` — сырой выход модели до подготовки. Он нужен ровно для перегруза:
    `_prepare_chunk` подрезает пик, и клиппинг, который модель выдала сама,
    в подготовленном куске уже не виден. Считать его по «услышанному» значило бы
    показывать предупреждение только тогда, когда его кто-то добавил уже после
    нормализации, — то есть почти никогда.

    Синхронная и довольно дешёвая операция, кроме LUFS: вызывающий уводит её в
    поток, чтобы не занимать event loop (в очереди это `asyncio.to_thread`).
    """
    y = np.asarray(chunk, dtype=np.float32).reshape(-1)
    duration = y.size / SAMPLE_RATE
    chars = len(text)
    clipping_ratio = audio_analysis.clipping_ratio(y)
    if raw is not None and raw.size:
        # Худший из двух: перегруз модели важнее того, что осталось после
        # подготовки, а подготовка могла его только спрятать.
        clipping_ratio = max(clipping_ratio, audio_analysis.clipping_ratio(raw))
    silence = qa_screening.silence_ratio(y)
    rms_dbfs = qa_screening.level_dbfs(y)
    peak_dbfs = _dbfs(float(np.max(np.abs(y))) if y.size else 0.0)

    # Причины те же, что у отбора, и в том же порядке; повторов здесь нет —
    # автокорреляция ищет зацикливание для решения «идти ли в Whisper», а не
    # описывает качество take'а. Пустой waveform — отдельный случай: мерить в нём
    # нечего, и причина у него одна, как и в `screen_chunk`.
    if not y.size:
        reasons = [qa_screening.REASON_EMPTY]
    else:
        reasons = qa_screening.duration_reasons(duration, text)
        if silence is not None and silence > config.QA_SCREEN_SILENCE_RATIO:
            reasons.append(qa_screening.REASON_SILENCE)
        if not (config.QA_SCREEN_RMS_FLOOR_DB <= rms_dbfs <= config.QA_SCREEN_RMS_CEIL_DB):
            reasons.append(qa_screening.REASON_LEVEL)
        if clipping_ratio >= audio_analysis.CLIPPING_MIN_RATIO:
            reasons.append(qa_screening.REASON_CLIPPING)

    # Вердикт отбора важнее собственного замера: он вынесен по сырому выходу
    # движка, и именно он решал, идти ли куску в Whisper. Если отбора не было
    # (strict/off), причины считаются по подготовленному куску.
    screening = getattr(qa, "screening", None) if qa is not None else None
    screening_reasons = list(screening.reasons) if screening is not None else list(reasons)
    return TakeQuality(
        duration_sec=duration,
        chars=chars,
        clipping=clipping_ratio >= audio_analysis.CLIPPING_MIN_RATIO,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence,
        duration_per_char=(duration / chars) if chars else None,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        lufs=_lufs(y),
        wer=None if qa is None else getattr(qa, "wer", None),
        qa_attempts=None if qa is None else getattr(qa, "attempts", None),
        qa_mode=None if qa is None else getattr(qa, "mode", None),
        screening_reasons=screening_reasons,
        warnings=_warnings(reasons, rms_dbfs),
    )
