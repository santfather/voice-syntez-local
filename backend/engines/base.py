"""Общий интерфейс движков синтеза и их паспорт.

Пайплайн (`audio_pipeline.py`) работает только через этот интерфейс и не знает,
F5-TTS сейчас синтезирует кусок или XTTS v2. Ручки, специфичные для конкретного
движка (`temperature` у XTTS, `cfg_strength` у F5), передаются одним словарём
`engine_params`: движок берёт из него то, что объявил в `EngineInfo.params`, а
остальное молча игнорирует — иначе общий вызов из пайплайна пришлось бы
разветвлять на каждом куске.

Здесь нет импорта torch: паспорта движков нужны и `voices_store.py` (валидация
поля `engine`), и `/api/engines`, а тянуть ради справочных данных нейросетевые
библиотеки нельзя — их импорт стоит десятки секунд и сотни мегабайт RSS.
"""

import gc
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .. import config

logger = logging.getLogger(__name__)

# Единый sample rate всех движков: и F5-TTS, и XTTS v2 отдают 24 кГц.
# Куски разных движков склеиваются в один файл напрямую, поэтому это не
# деталь реализации, а требование к интерфейсу.
SAMPLE_RATE = 24_000

ENGINE_F5 = "f5"
ENGINE_XTTS = "xtts"
ENGINE_XTTS_BANANA = "xtts-banana"

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"


class EngineBusyError(RuntimeError):
    """Выгрузка отменена: движок прямо сейчас синтезирует.

    Пайплайн синтезирует в отдельном потоке, поэтому обнулить модель из-под
    работающего инференса — значит уронить кусок на середине, а не освободить
    память. Отказ приходит вызывающему понятным текстом (роут отдаёт по нему 409).
    """


def release_torch_memory() -> None:
    """Best-effort возврат памяти Python/Torch/MPS после выгрузки модели.

    `gc.collect()` идёт раньше `empty_cache()`: пока на тензоры есть ссылки в
    циклах, драйвер их не заберёт. Отсутствие torch или MPS и ошибка очистки не
    считаются ошибкой выгрузки — ссылки на модель уже обнулены, а кеш драйвера
    ОС заберёт сама, поэтому сбой уборки только логируется.
    """
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001 — уборка мусора не должна ломать выгрузку
        logger.warning("Не удалось очистить кеш MPS после выгрузки: %s", exc)


@dataclass(frozen=True)
class EngineParam:
    """Числовая ручка движка: границы, дефолт и подпись для интерфейса.

    Границы живут рядом с движком, а не в JS: одно и то же значение приходит и из
    формы нового голоса, и из карточки слота, и из `voices.json` — проверять его
    в трёх местах по-разному означало бы три разных поведения.
    """

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    integer: bool = False
    hint: str = ""

    def clamp(self, value: Any) -> float | int:
        """Приводит значение к числу в объявленных границах; нечисло — к дефолту."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(self.default)
        number = min(max(number, self.minimum), self.maximum)
        return int(round(number)) if self.integer else round(number, 4)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "integer": self.integer,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class EngineInfo:
    """Паспорт движка: имя, назначение и набор его ручек."""

    id: str
    label: str
    description: str
    supports_accents: bool
    # Короткая справка про ресурсы и лицензию — показывается в интерфейсе.
    note: str = ""
    params: tuple[EngineParam, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "supports_accents": self.supports_accents,
            "note": self.note,
            "params": [param.to_dict() for param in self.params],
        }


ENGINE_INFOS: dict[str, EngineInfo] = {
    ENGINE_F5: EngineInfo(
        id=ENGINE_F5,
        label="F5-TTS Russian",
        description="Файнтюн под русский с разметкой ударений. Основной движок проекта.",
        supports_accents=True,
        note="Веса ~1.4 ГБ, работает офлайн.",
    ),
    ENGINE_XTTS: EngineInfo(
        id=ENGINE_XTTS,
        label="XTTS v2 (базовая)",
        description="Многоязычная модель Coqui, клонирование по 6–15 с референса.",
        supports_accents=False,
        note="Веса ~1.9 ГБ, лицензия CPML (некоммерческая). Ударения не понимает — текст идёт как есть.",
        params=(
            EngineParam(
                name="temperature",
                label="Температура — стабильность ↔ живость",
                default=config.DEFAULT_XTTS_TEMPERATURE,
                minimum=config.XTTS_TEMPERATURE_RANGE[0],
                maximum=config.XTTS_TEMPERATURE_RANGE[1],
                step=0.05,
                hint="ниже — ровнее и предсказуемее, выше — живее и рискованнее",
            ),
            EngineParam(
                name="repetition_penalty",
                label="Штраф за повторы",
                default=config.DEFAULT_XTTS_REPETITION_PENALTY,
                minimum=config.XTTS_REPETITION_PENALTY_RANGE[0],
                maximum=config.XTTS_REPETITION_PENALTY_RANGE[1],
                step=0.5,
                hint="лечит заикание и «залипание» на слоге",
            ),
        ),
    ),
    ENGINE_XTTS_BANANA: EngineInfo(
        id=ENGINE_XTTS_BANANA,
        label="XTTS v2 + русский файнтюн (banana)",
        description="Комьюнити-файнтюн под живую русскую речь: лучше держит разговорные ударения.",
        supports_accents=False,
        note=(
            "Веса ~5.2 ГБ, лицензия базовой модели (CPML, некоммерческая). Обучен преимущественно "
            "на женских голосах — мужские реплики через него могут звучать феминизированно."
        ),
        params=(
            EngineParam(
                name="temperature",
                label="Температура — стабильность ↔ живость",
                default=config.DEFAULT_XTTS_TEMPERATURE,
                minimum=config.XTTS_TEMPERATURE_RANGE[0],
                maximum=config.XTTS_TEMPERATURE_RANGE[1],
                step=0.05,
                hint="ниже — ровнее и предсказуемее, выше — живее и рискованнее",
            ),
            EngineParam(
                name="repetition_penalty",
                label="Штраф за повторы",
                default=config.DEFAULT_XTTS_REPETITION_PENALTY,
                minimum=config.XTTS_REPETITION_PENALTY_RANGE[0],
                maximum=config.XTTS_REPETITION_PENALTY_RANGE[1],
                step=0.5,
                hint="лечит заикание и «залипание» на слоге",
            ),
        ),
    ),
}


def default_engine_for_gender(gender: str) -> str:
    """Подсказка движка по полу — именно подсказка, а не правило.

    Файнтюн F5 обучен на разметке ударений и ровнее звучит на мужских голосах,
    базовая XTTS — наоборот, женские. Пользователь может выбрать любой движок
    для любого голоса: это значение только предзаполняет форму.
    """
    return ENGINE_XTTS if gender == "female" else ENGINE_F5


def engine_info(engine_id: str) -> EngineInfo:
    return ENGINE_INFOS.get(engine_id) or ENGINE_INFOS[ENGINE_F5]


def normalize_engine_params(engine_id: str, raw: dict | None) -> dict:
    """Приводит словарь ручек к границам, объявленным движком.

    Незнакомые ключи не выбрасываем: пайплайн передаёт через тот же словарь
    общие ручки (`target_rms`, `cross_fade_duration`), а движок читает из него
    только то, что ему нужно. Значения объявленных ручек при этом обязательно
    оказываются числами в допустимых границах — мусор с фронта или из
    `voices.json` не должен доходить до модели.
    """
    declared = {param.name: param for param in engine_info(engine_id).params}
    result: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        param = declared.get(key)
        result[key] = param.clamp(value) if param else value
    return result


class SynthesisEngine(ABC):
    """Один движок синтеза: тёплая модель на весь жизненный цикл процесса.

    Наследнику достаточно реализовать `load()` и `_synthesize()`: состояние
    (idle/loading/ready/failed) и приведение параметров к границам движка уже
    сделаны здесь, одинаково для всех движков.
    """

    info: EngineInfo

    # Понимает ли движок `seed`. У F5 он задаёт разброс генерации, у XTTS — сид
    # torch: без него одна и та же реплика каждый раз звучит по-новому, и
    # понравившийся вариант нельзя ни выбрать осознанно, ни повторить. Пайплайн
    # передаёт сид только таким движкам и записывает его рядом с куском.
    supports_seed: bool = False

    def __init__(self) -> None:
        self._state = STATE_IDLE
        self._last_error: str | None = None
        self._state_lock = threading.Lock()
        # Счётчик активных синтезов: пайплайн синтезирует в отдельном потоке, а
        # `unload()` обязан отказаться выгружать модель, которую прямо сейчас
        # читает инференс. Тот же лок удерживается на время `_release()`, поэтому
        # синтез, стартовавший сразу после проверки счётчика, не начнётся раньше,
        # чем модель будет освобождена: он подождёт лок и поднимет движок заново.
        self._active_lock = threading.Lock()
        self._active_synthesizes = 0
        # `time.monotonic()` последней загрузки или завершённого синтеза — точка
        # отсчёта простоя для фоновой выгрузки (см. engine_lifecycle).
        self._last_used_at: float | None = None

    # -- состояние -------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.info.id

    @property
    def supports_accents(self) -> bool:
        """Понимает ли движок `+`-ударения RUAccent (см. audio_pipeline)."""
        return self.info.supports_accents

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_loaded(self) -> bool:
        return self._state == STATE_READY

    @property
    def active_synthesizes(self) -> int:
        """Сколько синтезов идёт прямо сейчас: по этому числу `unload()` отказывает."""
        with self._active_lock:
            return self._active_synthesizes

    @property
    def last_used_at(self) -> float | None:
        """Когда движок последний раз поднимали или синтезировали (`time.monotonic()`)."""
        return self._last_used_at

    def _mark(self, state: str, error: Exception | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._last_error = None if error is None else str(error)
            if state == STATE_READY:
                # Только что поднятый движок тоже «только что использован»: иначе
                # фоновая выгрузка забрала бы его сразу после загрузки.
                self._last_used_at = time.monotonic()

    def _touch(self) -> None:
        """Отмечает конец использования: простой для политики считается от него."""
        with self._state_lock:
            self._last_used_at = time.monotonic()

    def to_dict(self) -> dict:
        return {**self.info.to_dict(), "state": self._state, "error": self._last_error}

    # -- загрузка и синтез -----------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Грузит модель (идемпотентно). Тяжёлая операция — вызывается в отдельном потоке."""

    def _release(self) -> None:
        """Освобождает ресурсы модели: подклассы обнуляют модель, тензоры и латенты.

        В базовом классе ресурсов нет, поэтому хук — no-op. Вызывается только из
        `unload()` и только когда движок свободен, поэтому подклассу не нужно
        защищаться от одновременного инференса. Хук должен быть идемпотентным:
        выгрузка незагруженного движка — нормальная ситуация.
        """

    def unload(self) -> None:
        """Выгружает модель и освобождает её память. Идемпотентно.

        Отказ, а не тихая выгрузка, если движок занят: `EngineBusyError`
        поднимается, пока счётчик активных синтезов не ноль. Лок счётчика
        удерживается до конца `_release()`, поэтому гонка «выгрузка против
        стартующей задачи» безопасна: синтез подождёт и поднимет модель заново
        (`synthesize()` сам вызывает `load()`), то есть худший исход — лишняя
        загрузка, а не падение.

        Ошибка `_release()` не глотается — о ней нужно сказать вызывающему, — но
        и не оставляет движок в подвешенном состоянии: состояние помечается
        `failed` с текстом ошибки, а повторные `unload()`/`load()` работают.
        """
        with self._active_lock:
            if self._active_synthesizes:
                raise EngineBusyError(
                    f"Движок «{self.info.label}» сейчас синтезирует "
                    f"({self._active_synthesizes} активных вызовов): выгрузка отменена, "
                    "чтобы не оборвать инференс. Дождитесь окончания задачи."
                )
            try:
                self._release()
            except Exception as exc:
                # Не глотаем: об ошибке нужно сказать вызывающему. Состояние при
                # этом согласовано — `failed` с текстом, повторные вызовы работают.
                logger.error("Не удалось выгрузить движок %s: %s", self.id, exc)
                self._mark(STATE_FAILED, exc)
                raise
            self._mark(STATE_IDLE)
        logger.info("Движок %s выгружен, память освобождена", self.id)

    @abstractmethod
    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[Any, int]:
        """Возвращает (waveform float32, sample_rate)."""

    def defaults(self) -> dict:
        """Объявленные движком ручки со значениями по умолчанию."""
        return {param.name: param.default for param in self.info.params}

    def normalize_params(self, raw: dict | None) -> dict:
        """Приводит словарь ручек к границам этого движка (см. normalize_engine_params)."""
        return normalize_engine_params(self.id, raw)

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        speed: float = config.DEFAULT_SPEED,
        **params: Any,
    ) -> tuple[Any, int]:
        """Синтез одного куска: `(waveform, sample_rate)`, sample_rate всегда SAMPLE_RATE.

        Модель поднимается здесь же, если движок выгружен: `load()` вызывается при
        `not is_loaded`, поэтому ручная или фоновая выгрузка не ломает следующий
        синтез — он просто снова платит за загрузку. Пока вызов активен, счётчик
        не ноль, и `unload()` отказывается освобождать модель (см. `EngineBusyError`).
        """
        with self._active_lock:
            self._active_synthesizes += 1
        try:
            if not self.is_loaded:
                self.load()
            merged = {**self.defaults(), **self.normalize_params(params)}
            waveform, sample_rate = self._synthesize(
                text, ref_audio_path, ref_text, float(speed), merged
            )
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f"Движок {self.id} вернул {sample_rate} Гц вместо {SAMPLE_RATE}: "
                    "куски разных движков нельзя склеить без ресемплинга"
                )
            return waveform, sample_rate
        finally:
            with self._active_lock:
                self._active_synthesizes -= 1
            self._touch()
