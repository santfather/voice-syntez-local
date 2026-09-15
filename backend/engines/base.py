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

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .. import config

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

    def _mark(self, state: str, error: Exception | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._last_error = None if error is None else str(error)

    def to_dict(self) -> dict:
        return {**self.info.to_dict(), "state": self._state, "error": self._last_error}

    # -- загрузка и синтез -----------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Грузит модель (идемпотентно). Тяжёлая операция — вызывается в отдельном потоке."""

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
        """Синтез одного куска: `(waveform, sample_rate)`, sample_rate всегда SAMPLE_RATE."""
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
