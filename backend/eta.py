"""Engine-aware ETA: статистика времени по движкам и оценка остатка рендера.

Здесь нет ни обращений к моделям, ни побочных эффектов: модуль только копит
наблюдения о прошедших кусках и по ним оценивает будущие. Число реплик для
оценки не годится: реплика из 12 знаков и реплика из 200 знаков стоят разного
времени, XTTS заметно медленнее F5, холодная загрузка модели стоит десятков
секунд, а строгая проверка добавляет к каждому куску ещё и Whisper. Поэтому
план рендера считается **по кускам**, каждый — со своим движком, длиной текста,
режимом проверки и признаком холодного старта.

Статистика живёт в процессе и уточняется по ходу работы: чем больше кусков
движка прошло, тем ближе оценка к тому, что человек реально видит на часах.
Пока наблюдений нет, работает грубый дефолт по движку — приблизительный, но
конечный, положительный и разный для разных движков.
"""

import logging
import math
import threading
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)

# Коэффициент EWMA: новое наблюдение весит 30 %. Не 1.0 — иначе одна случайная
# реплика переписала бы всю историю; не 0.0 — иначе история ничего не значила бы
# и оценка навсегда осталась бы дефолтом.
EWMA_ALPHA = 0.3

# Дефолты — приблизительные, до первых наблюдений: секунды инференса на знак
# текста (прогретый движок), стоимость холодной загрузки модели и множитель
# режима проверки. Уточняются наблюдениями (см. `observe`). F5 в этом проекте
# заметно быстрее XTTS, а её загрузка дешевле: по замерам нагрузочного теста
# (README, «Нагрузочный тест») XTTS на первой задаче тратит ~21 с только на
# подъём весов, F5 — секунды. Различие движков и есть то, ради чего статистика
# ведётся отдельно по каждому.
DEFAULT_SEC_PER_CHAR: dict[str, float] = {
    "f5": 0.012,
    "xtts": 0.055,
    "xtts-banana": 0.07,
    # Неизвестный id: середина между быстрым F5 и медленной XTTS.
    "": 0.03,
}
DEFAULT_COLD_START_SEC: dict[str, float] = {
    "f5": 8.0,
    "xtts": 25.0,
    "xtts-banana": 32.0,
    "": 20.0,
}
# Множитель проверки: 1.0 — цикл не запускается; smart — дешёвый отбор плюс
# Whisper на подозрительных кусках; strict — Whisper на каждом куске.
DEFAULT_QA_FACTOR: dict[str, float] = {
    config.QA_MODE_OFF: 1.0,
    config.QA_MODE_SMART: 1.15,
    config.QA_MODE_STRICT: 2.6,
}

# Границы, за которые статистика не выпускает оценку. Это защита от шума, а не
# оценка: цена проверки в разы, но не в сотни раз выше обычного синтеза.
MIN_RATE = 1e-4
MAX_RATE = 5.0
MIN_QA_FACTOR = 1.0
MAX_QA_FACTOR = 20.0


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма существительного при числительном (1 секунда / 2 секунды)."""
    tail = count % 100
    if 11 <= tail <= 14:
        return many
    tail %= 10
    if tail == 1:
        return one
    if 2 <= tail <= 4:
        return few
    return many


def format_eta(seconds: float) -> str:
    """Готовая к показу строка: `"2 мин 40 сек"`, `"1 ч 4 мин 12 сек"`.

    «≈» ставит интерфейс: бэкенд отдаёт само значение, а не оформление
    конкретного места в UI. Аббревиатуры («сек», «мин», «ч») выбраны короткими:
    строка живёт рядом с «18 / 47 реплик» в одну строку статуса.
    """
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        total = 0.0
    if total < 0 or math.isnan(total):  # шум статистики и NaN — это «осталось 0»
        total = 0.0
    total = round(total)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)

    parts: list[str] = []
    if hours:
        # Часы во всех формах — «ч»: строка короткая и живёт рядом с минутами.
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} {_plural(minutes, 'мин', 'мин', 'мин')}")
    # Секунды — самая мелкая ступень: если крупнее уже что-то есть, ноль не
    # пишется («1 ч», «1 мин»), а если нет — «0 сек» честнее пустой строки.
    if secs or not parts:
        parts.append(f"{secs} {_plural(secs, 'сек', 'сек', 'сек')}")
    return " ".join(parts)


def qa_mode(qa) -> str:
    """Режим проверки из `RenderSettings.qa` (None — проверка выключена)."""
    mode = getattr(qa, "mode", None)
    return str(mode) if mode else config.QA_MODE_OFF


@dataclass(frozen=True)
class EtaStep:
    """Один кусок плана рендера: чем он будет синтезирован и как долго ждать."""

    engine: str
    chars: int
    qa_mode: str = config.QA_MODE_OFF
    # Движок ещё не поднят: к этому куску добавляется стоимость загрузки модели.
    cold: bool = False


@dataclass
class _EngineStats:
    """История одного движка: скорость, холодный старт и цена проверки."""

    sec_per_char: float
    samples: int = 0
    cold_start_sec: float = 0.0
    qa_factor: dict[str, float] | None = None

    def factors(self) -> dict[str, float]:
        if self.qa_factor is None:
            self.qa_factor = dict(DEFAULT_QA_FACTOR)
        return self.qa_factor


def _blend(previous: float, observed: float) -> float:
    """EWMA: новое наблюдение подтягивает прежнее значение, а не заменяет его.

    Сглаживается и первое наблюдение — от дефолта: история должна значить, а
    не обнуляться первым же куском, и оценка обязана двигаться постепенно.
    """
    return (1.0 - EWMA_ALPHA) * previous + EWMA_ALPHA * observed


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class EtaTracker:
    """Статистика по движкам и оценка остатка. Потокобезопасен.

    Синтез идёт в отдельном потоке, а статус задачи читается из event loop,
    поэтому и `observe`, и `estimate` могут прийти одновременно.
    """

    def __init__(self, defaults: dict | None = None) -> None:
        self._lock = threading.Lock()
        # Дефолты можно подменить целиком: тестам нужна предсказуемая история,
        # а не боевые константы (см. tests/test_eta.py).
        self._defaults = defaults or {
            "sec_per_char": DEFAULT_SEC_PER_CHAR,
            "cold_start": DEFAULT_COLD_START_SEC,
            "qa_factor": DEFAULT_QA_FACTOR,
        }
        self._stats: dict[str, _EngineStats] = {}
        # Загрузка, которую уже оплатили наблюдением: её стоимость вычитается из
        # первого куска этого движка, иначе холодный старт посчитался бы дважды.
        self._pending_load: dict[str, float] = {}

    # -- статистика ------------------------------------------------------------
    def _stats_for(self, engine: str) -> _EngineStats:
        """История движка; незнакомый id получает разумный дефолт.

        У неизвестного движка своей строки в константах нет — он берёт общую
        (`""`), а не нули: оценка обязана остаться положительной и конечной,
        иначе ETA «сломается» на первом же новом движке.
        """
        stats = self._stats.get(engine)
        if stats is None:
            rates = self._defaults["sec_per_char"]
            colds = self._defaults["cold_start"]
            stats = _EngineStats(
                sec_per_char=float(rates.get(engine, rates.get("", DEFAULT_SEC_PER_CHAR[""]))),
                cold_start_sec=float(
                    colds.get(engine, colds.get("", DEFAULT_COLD_START_SEC[""]))
                ),
                qa_factor=dict(self._defaults["qa_factor"]),
            )
            self._stats[engine] = stats
        return stats

    def note_load(self, engine: str, seconds: float) -> None:
        """Движок поднят за `seconds` секунд: это и есть его холодный старт.

        Стоимость запоминается отдельно от скорости инференса: `observe` вычтет
        её из длительности куска, на котором она случилась, и скорость не будет
        выглядеть хуже, чем она есть.
        """
        seconds = max(float(seconds), 0.0)
        with self._lock:
            stats = self._stats_for(engine)
            stats.cold_start_sec = max(_blend(stats.cold_start_sec, seconds), 0.0)
            self._pending_load[engine] = seconds

    def observe(
        self,
        engine: str,
        chars: int,
        render_sec: float,
        qa_mode: str = config.QA_MODE_OFF,
        attempts: int = 1,
        cold: bool = False,
        audio_sec: float | None = None,
    ) -> None:
        """Обновляет статистику движка по прошедшему куску.

        Скорость движка учится **только на кусках без проверки**: в куске с
        проверкой сидит и Whisper, и повторные синтезы, и если считать их
        скоростью, движок «замедлится» от включённого QA — а это цена проверки,
        а не движка. Её ведёт отдельный множитель режима. Попытки и длительность
        аудио на скорость тоже не влияют: первые уже сидят в длительности,
        вторая нужна только для лога.
        """
        chars = max(int(chars), 0)
        render_sec = max(float(render_sec), 0.0)
        mode = str(qa_mode or config.QA_MODE_OFF)
        with self._lock:
            stats = self._stats_for(engine)
            base = self._base(stats, engine, chars)
            if mode == config.QA_MODE_OFF:
                inference = render_sec
                if cold:
                    cold_cost = self._pending_load.pop(engine, stats.cold_start_sec)
                    inference = max(inference - cold_cost, 0.0)
                if chars > 0:
                    observed = _clamp(inference / chars, MIN_RATE, MAX_RATE)
                    stats.sec_per_char = _clamp(
                        _blend(stats.sec_per_char, observed), MIN_RATE, MAX_RATE
                    )
            else:
                # Загрузку движка всё равно считаем оплаченной, даже если кусок
                # шёл через проверку: иначе она вычиталась бы из следующего.
                self._pending_load.pop(engine, None)
                if base > 0:
                    factors = stats.factors()
                    observed_factor = _clamp(
                        render_sec / base, MIN_QA_FACTOR, MAX_QA_FACTOR
                    )
                    factors[mode] = _clamp(
                        _blend(factors.get(mode, 1.0), observed_factor),
                        MIN_QA_FACTOR,
                        MAX_QA_FACTOR,
                    )
            stats.samples += 1
        logger.debug(
            "ETA: %s — %s знаков за %.2f c (QA %s, попыток %s, аудио %s c) → %.4f с/знак",
            engine, chars, render_sec, mode, attempts,
            "?" if audio_sec is None else f"{audio_sec:.2f}", self.sec_per_char(engine),
        )

    # -- оценка ----------------------------------------------------------------
    def _base(self, stats: _EngineStats, engine: str, chars: int) -> float:
        """Время инференса без проверки для уже разрешённых настроек движка."""
        return max(int(chars), 0) * stats.sec_per_char

    def sec_per_char(self, engine: str) -> float:
        """Текущая (после наблюдений) скорость движка, секунд на знак."""
        with self._lock:
            return self._stats_for(engine).sec_per_char

    def cold_start_sec(self, engine: str) -> float:
        """Текущая оценка холодной загрузки движка."""
        with self._lock:
            return self._stats_for(engine).cold_start_sec

    def qa_factor(self, engine: str, mode: str = config.QA_MODE_STRICT) -> float:
        """Множитель режима проверки: сколько времени она добавляет к синтезу."""
        with self._lock:
            stats = self._stats_for(engine)
            if str(mode or config.QA_MODE_OFF) == config.QA_MODE_OFF:
                return 1.0
            return stats.factors().get(str(mode), 1.0)

    def estimate_step(self, step: EtaStep) -> float:
        """Оценка одного куска: инференс + (холодный старт), всё × цена проверки."""
        chars = max(int(step.chars), 0)
        with self._lock:
            stats = self._stats_for(step.engine)
            total = self._base(stats, step.engine, chars)
            if step.cold:
                total += stats.cold_start_sec
            mode = str(step.qa_mode or config.QA_MODE_OFF)
            if mode != config.QA_MODE_OFF:
                total *= stats.factors().get(mode, 1.0)
        return max(total, 0.0)

    def estimate(self, steps) -> float:
        """Оценка плана целиком. Пустой план — ровно 0.

        Каждый кусок считается по своему движку, поэтому смешанный проект
        (F5 + XTTS в одном диалоге) получает сумму по составляющим, а не одно
        среднее на всех.
        """
        total = 0.0
        for step in steps or ():
            total += self.estimate_step(step)
        return max(total, 0.0)

    def snapshot(self) -> dict:
        """Что накопилось — для логов и отладки; формат не является контрактом."""
        with self._lock:
            return {
                engine: {
                    "sec_per_char": round(stats.sec_per_char, 6),
                    "cold_start_sec": round(stats.cold_start_sec, 3),
                    "samples": stats.samples,
                    "qa_factor": dict(stats.factors()),
                }
                for engine, stats in self._stats.items()
            }


_tracker: EtaTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> EtaTracker:
    """Синглтон процесса: статистика копится между задачами, а не внутри одной."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = EtaTracker()
        return _tracker


def reset() -> None:
    """Сбрасывает накопленную статистику. Нужен тестам."""
    global _tracker
    with _tracker_lock:
        _tracker = None
