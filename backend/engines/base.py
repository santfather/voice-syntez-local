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
ENGINE_QWEN = "qwen3-tts"
ENGINE_KOKORO = "kokoro-ru"

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"

# Режимы работы движка. Это **не** режим проверки качества (`config.QA_MODES`):
# QA отвечает на вопрос «проверять ли результат», а режим — «насколько дорого и
# качественно синтезировать». Три ступени одинаковы для всех движков, потому что
# пользователю нужен один и тот же выбор в интерфейсе, а не своя лестница у
# каждой модели:
#
# * `draft` — черновик: дешевле и быстрее, качество вторично (проверить темп,
#   прикинуть длину, прогнать длинный диалог «на слух»);
# * `quality` — рабочий режим: значения, на которых движок проверен;
# * `experimental` — возможности, которые ещё не подтверждены: они включаются
#   осознанно, и результат нужно слушать, а не принимать на веру.
ENGINE_MODE_DRAFT = "draft"
ENGINE_MODE_QUALITY = "quality"
ENGINE_MODE_EXPERIMENTAL = "experimental"
ENGINE_MODES: tuple[str, ...] = (
    ENGINE_MODE_DRAFT,
    ENGINE_MODE_QUALITY,
    ENGINE_MODE_EXPERIMENTAL,
)
# Значение `mode` в `engine_params` (см. `SynthesisEngine.synthesize`). Живёт в
# том же словаре, что и числовые ручки: пайплайну не нужен второй канал, чтобы
# донести выбор до движка.
ENGINE_MODE_KEY = "mode"


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
class EngineMode:
    """Режим работы движка: подпись для интерфейса и числовые переопределения.

    `overrides` — значения **уже объявленных** ручек движка (`EngineParam.name`):
    черновик почти всегда означает «то же самое, но дешевле» (меньше шагов,
    ниже потолок длины). Пары, а не словарь: `EngineInfo` неизменяем и хешируем,
    а словарь в неизменяемой структуре — это изменяемое состояние под видом
    константы.

    Режим, которому мало числовых ручек (например, «только эмбеддинг спикера»
    вместо полного клонирования), движок разбирает сам по `params["mode"]`:
    паспорт объявляет **наличие** режима, а не способ его исполнить.
    """

    id: str
    label: str
    hint: str = ""
    overrides: tuple[tuple[str, float], ...] = ()

    def overrides_dict(self) -> dict[str, float]:
        return {name: value for name, value in self.overrides}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "hint": self.hint,
            # Переопределения отдаются интерфейсу, потому что режим — это ещё и
            # значения объявленных ручек: без них карточка показывала бы прошлый
            # режим, а модель считала бы новый.
            "overrides": self.overrides_dict(),
        }


@dataclass(frozen=True)
class EngineInfo:
    """Паспорт движка: имя, назначение, возможности и набор его ручек.

    Возможности, а не догадки вызывающего кода: пайплайн спрашивает паспорт,
    нужен ли движку референс (`supports_cloning`) и применимы ли к нему
    интонационные профили (`supports_prosody_profiles`), вместо того чтобы
    разветвляться на идентификаторах движков. Новый движок описывается здесь
    одним словарём данных, без второй реализации пайплайна.
    """

    id: str
    label: str
    description: str
    supports_accents: bool
    # Короткая справка про ресурсы и лицензию — показывается в интерфейсе.
    note: str = ""
    params: tuple[EngineParam, ...] = ()
    # Понимает ли движок `seed`. Живёт в паспорте, а не только атрибутом класса:
    # паспорт читает и прокси изолированного движка (engines/worker_engine.py),
    # который обязан знать про сид, не импортируя модуль движка (тот тянет torch
    # и веса). Источник истины один — здесь.
    supports_seed: bool = False
    # Нужен ли движку референс голоса. `False` — движок работает на собственных
    # встроенных голосах (пресетах), и пайплайн синтезирует без референса:
    # резолвер профилей не вызывается вовсе, а не «вызывается и падает».
    supports_cloning: bool = True
    # Применимы ли к движку интонационные профили (UPDATE 3). У движка без
    # клонирования профилей нет по определению: профиль — это запись голоса,
    # которой у пресетного движка не существует.
    supports_prosody_profiles: bool = True
    # Языки, объявленные моделью (коды ISO 639-1). Справочно для интерфейса и
    # валидации: сам проект синтезирует по-русски, но Kokoro-ru русский умеет
    # только один, а Qwen3-TTS — десять, и это видимая пользователю разница.
    languages: tuple[str, ...] = ("ru",)
    # Движок, на который разрешено откатиться, если файлы этого движка не
    # установлены. Пусто — отката нет, недоступность движка остаётся ошибкой:
    # молчаливая подмена одного звучания другим допустима только там, где
    # автор движка её объявил.
    fallback_engine: str = ""
    # Объявленные режимы работы. Пусто — у движка их нет, и интерфейс не
    # показывает переключатель: обещать выбор, который ничего не меняет, нельзя.
    modes: tuple[EngineMode, ...] = ()
    default_mode: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "supports_accents": self.supports_accents,
            "supports_seed": self.supports_seed,
            "supports_cloning": self.supports_cloning,
            "supports_prosody_profiles": self.supports_prosody_profiles,
            "languages": list(self.languages),
            "fallback_engine": self.fallback_engine,
            "default_mode": self.default_mode,
            "note": self.note,
            "params": [param.to_dict() for param in self.params],
            "modes": [mode.to_dict() for mode in self.modes],
        }

    def mode(self, mode_id: str | None) -> EngineMode | None:
        """Объявленный режим по id; неизвестный или пустой — `default_mode`."""
        wanted = str(mode_id or "").strip() or self.default_mode
        if not wanted:
            return None
        return next((item for item in self.modes if item.id == wanted), None)


ENGINE_INFOS: dict[str, EngineInfo] = {
    ENGINE_F5: EngineInfo(
        id=ENGINE_F5,
        label="F5-TTS Russian",
        description="Файнтюн под русский с разметкой ударений. Основной движок проекта.",
        supports_accents=True,
        supports_seed=True,
        note="Веса ~1.4 ГБ, работает офлайн.",
    ),
    ENGINE_XTTS: EngineInfo(
        id=ENGINE_XTTS,
        label="XTTS v2 (базовая)",
        description="Многоязычная модель Coqui, клонирование по 6–15 с референса.",
        supports_accents=False,
        supports_seed=True,
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
        supports_seed=True,
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
    ENGINE_QWEN: EngineInfo(
        id=ENGINE_QWEN,
        label="Qwen3-TTS 1.7B (Base)",
        description=(
            "Клонирование голоса по 3 с референса с сохранением его расшифровки. "
            "Десять языков, разметку ударений не понимает."
        ),
        supports_accents=False,
        supports_seed=True,
        languages=(
            # Порядок — как в карточке модели, а не по алфавиту: первым идёт
            # язык проекта, дальше — остальные объявленные моделью.
            "ru",
            "en",
            "zh",
            "ja",
            "ko",
            "de",
            "fr",
            "pt",
            "es",
            "it",
        ),
        # Откат разрешён только явно и только по недоступности весов: без этого
        # голос, настроенный на Qwen, падал бы на каждом куске, пока модель не
        # скачана, вместо того чтобы честно дочитать диалог основным движком.
        fallback_engine=ENGINE_F5,
        note=(
            "Веса ~4.5 ГБ плюс токенизатор ~0.4 ГБ, лицензия Apache-2.0. "
            "Отдельный пакет `qwen-tts` с версией transformers ниже проектной — "
            "ставится без зависимостей поверх шима (см. engines/qwen_engine.py). "
            "MPS официально не заявлен: устройство выбирается пробой, с откатом на CPU."
        ),
        params=(
            EngineParam(
                name="temperature",
                label="Температура — стабильность ↔ живость",
                default=config.DEFAULT_QWEN_TEMPERATURE,
                minimum=config.QWEN_TEMPERATURE_RANGE[0],
                maximum=config.QWEN_TEMPERATURE_RANGE[1],
                step=0.05,
                hint="ниже — ровнее и предсказуемее, выше — живее и рискованнее",
            ),
            EngineParam(
                name="top_p",
                label="Отбор ядра (top-p)",
                default=config.DEFAULT_QWEN_TOP_P,
                minimum=config.QWEN_TOP_P_RANGE[0],
                maximum=config.QWEN_TOP_P_RANGE[1],
                step=0.05,
                hint="ниже — уже круг вариантов, выше — разнообразнее",
            ),
            EngineParam(
                name="repetition_penalty",
                label="Штраф за повторы",
                default=config.DEFAULT_QWEN_REPETITION_PENALTY,
                minimum=config.QWEN_REPETITION_PENALTY_RANGE[0],
                maximum=config.QWEN_REPETITION_PENALTY_RANGE[1],
                step=0.05,
                hint="лечит заикание и «залипание» на слоге",
            ),
            EngineParam(
                name="max_new_tokens",
                label="Потолок длины генерации",
                default=config.DEFAULT_QWEN_MAX_NEW_TOKENS,
                minimum=config.QWEN_MAX_NEW_TOKENS_RANGE[0],
                maximum=config.QWEN_MAX_NEW_TOKENS_RANGE[1],
                step=256,
                integer=True,
                hint="страховка от «разговорившейся» модели, а не регулятор длины",
            ),
        ),
        modes=(
            EngineMode(
                id=ENGINE_MODE_DRAFT,
                label="Черновик",
                hint="ниже потолок длины: быстрее на длинных репликах, качество то же",
                overrides=(("max_new_tokens", config.QWEN_DRAFT_MAX_NEW_TOKENS),),
            ),
            EngineMode(
                id=ENGINE_MODE_QUALITY,
                label="Качество",
                hint="рабочие значения ручек — на них движок проверен",
            ),
            EngineMode(
                id=ENGINE_MODE_EXPERIMENTAL,
                label="Эксперимент",
                hint=(
                    "клонирование только по эмбеддингу спикера, без его расшифровки: "
                    "работает без ref_text, но качество клонирования ниже"
                ),
            ),
        ),
        default_mode=ENGINE_MODE_QUALITY,
    ),
    ENGINE_KOKORO: EngineInfo(
        id=ENGINE_KOKORO,
        label="Kokoro-ru (встроенные голоса)",
        description=(
            "Лёгкий пресетный движок: 82 млн параметров, синтез встроенными голосами "
            "модели — записанный референс не используется."
        ),
        supports_accents=False,
        # Просодия детерминированная: стилевой вектор выбирается по длине
        # фонемной строки, поэтому одинаковый текст всегда звучит одинаково — и
        # «запомнить удачный вариант» здесь нечего.
        supports_seed=False,
        # Встроенные голоса (sveta/masha/dima) вместо клонирования: референс не
        # нужен, а ударения расставляет RUAccent внутри произносительного словаря
        # модели, поэтому `+`-разметка пайплайна ей не требуется.
        supports_cloning=False,
        supports_prosody_profiles=False,
        note=(
            "Веса ~0.7 ГБ (два чекпоинта), лицензия весов OpenRAIL, работает на CPU "
            "(RTF ~0.1). Тёмный тембр и фрикативный призвук в ж/ш/х — известные "
            "ограничения файнтюна (см. docs/known-issues.md)."
        ),
        # Откат объявлен по той же причине, что у Qwen: голос, настроенный на
        # Kokoro, обязан дочитать диалог, пока весов нет на диске, а не падать на
        # каждой реплике.
        fallback_engine=ENGINE_F5,
        modes=(
            EngineMode(
                id=ENGINE_MODE_DRAFT,
                label="Черновик",
                hint=(
                    "фонемизация без RUAccent: ударения расставляет espeak — быстрее, "
                    "но ошибок в ударениях заметно больше"
                ),
            ),
            EngineMode(
                id=ENGINE_MODE_QUALITY,
                label="Качество",
                hint=(
                    "рабочий путь: RUAccent расставляет ударения, редукция гласных "
                    "включена"
                ),
            ),
            EngineMode(
                id=ENGINE_MODE_EXPERIMENTAL,
                label="Эксперимент",
                hint=(
                    "без редукции гласных (разметка первой версии файнтюна): звучание "
                    "ближе к написанию, на слух не проверялось"
                ),
            ),
        ),
        default_mode=ENGINE_MODE_QUALITY,
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


def clamp_params(info: EngineInfo, raw: dict | None) -> dict:
    """Ручки по границам конкретного паспорта (см. `normalize_engine_params`).

    Отдельно от поиска по id, потому что у самого движка паспорт уже есть
    (`self.info`), и повторный поиск по глобальному словарю — это лишний шанс
    разойтись: движок с незарегистрированным id получал бы границы F5.
    """
    declared = {param.name: param for param in info.params}
    result: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        param = declared.get(key)
        result[key] = param.clamp(value) if param else value
    return result


def resolve_mode_id(info: EngineInfo, raw: Any) -> str:
    """Режим по паспорту (см. `normalize_engine_mode`) — без поиска по id."""
    if not info.modes:
        return ""
    chosen = info.mode(raw)
    if chosen is None:
        if str(raw or "").strip():
            logger.warning(
                "Движок %s: неизвестный режим «%s» — работаю на значениях по умолчанию",
                info.id,
                raw,
            )
        return ""
    return chosen.id


def normalize_engine_params(engine_id: str, raw: dict | None) -> dict:
    """Приводит словарь ручек к границам, объявленным движком.

    Незнакомые ключи не выбрасываем: пайплайн передаёт через тот же словарь
    общие ручки (`target_rms`, `cross_fade_duration`), а движок читает из него
    только то, что ему нужно. Значения объявленных ручек при этом обязательно
    оказываются числами в допустимых границах — мусор с фронта или из
    `voices.json` не должен доходить до модели.
    """
    return clamp_params(engine_info(engine_id), raw)


def normalize_engine_mode(engine_id: str, raw: Any) -> str:
    """Режим работы движка: объявленный id или `""`, если режимов нет.

    Неизвестный режим не подменяется молча на дефолтный: подмена превратила бы
    «эксперимент» в «качество», и пользователь слушал бы не то, что выбрал.
    Значение приводится к пустому, а движок тогда работает на объявленных
    значениях ручек — то есть на дефолте, но без обещания конкретного режима.
    """
    return resolve_mode_id(engine_info(engine_id), raw)


def requires_reference(engine_id: str) -> bool:
    """Нужен ли движку референс голоса (см. `EngineInfo.supports_cloning`)."""
    return engine_info(engine_id).supports_cloning


def fallback_engine(engine_id: str) -> str:
    """Движок отката по недоступности, объявленный паспортом (`""` — нет отката).

    Откат на самого себя и на неизвестный движок не выдаётся: первый не меняет
    ничего, второй уронил бы синтез вместо того, чтобы его спасти.
    """
    target = engine_info(engine_id).fallback_engine
    if not target or target == engine_id or target not in ENGINE_INFOS:
        return ""
    return target


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
    #
    # Значение берётся из паспорта, а не из атрибута класса: у изолированного
    # движка (engines/worker_engine.py) нет доступа к классу модели — его импорт
    # стоит десятков секунд и сотни мегабайт. Движки, которым нужно иное значение
    # (заглушки в тестах), по-прежнему могут перекрыть его атрибутом класса.
    @property
    def supports_seed(self) -> bool:
        return self.info.supports_seed

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
        """Приводит словарь ручек к границам этого движка (см. clamp_params)."""
        return clamp_params(self.info, raw)

    def resolve_mode(self, params: dict | None = None) -> str:
        """Действующий режим: явный выбор в `params` или объявленный дефолт.

        `""` — у движка режимов нет либо выбор непонятен; тогда ручки берутся
        как есть, без пресета.
        """
        return resolve_mode_id(self.info, (params or {}).get(ENGINE_MODE_KEY))

    def _merged_params(self, params: dict) -> dict:
        """Ручки одного синтеза: дефолты → пресет режима → явные значения.

        Порядок не случаен. Режим — это пресет («то же, но дешевле»), поэтому
        явно заданная ручка обязана его переопределять: иначе выбранное
        пользователем значение молча вернулось бы к режимному.
        """
        merged: dict[str, Any] = dict(self.defaults())
        mode_id = self.resolve_mode(params)
        mode = self.info.mode(mode_id) if mode_id else None
        if mode is not None:
            merged.update(self.normalize_params(mode.overrides_dict()))
        merged.update(self.normalize_params(params))
        if mode_id:
            # Движок, которому мало числовых ручек, разбирает режим сам.
            merged[ENGINE_MODE_KEY] = mode_id
        return merged

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
            merged = self._merged_params(params)
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
