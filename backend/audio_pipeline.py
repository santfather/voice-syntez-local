"""Генерация реплик по очереди и склейка их в один аудиофайл."""

import asyncio
import logging
import math
import os
import random
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    config,
    qa_screening,
    reference_resolver,
    synthesis_trace,
    take_quality,
    warmup_context,
)
from . import model_manager
from . import short_utterance as su
from . import short_utterance_boundary as boundary_module
from .accentizer import accentuate
from .dialogue_parser import Replica
from .engines.base import SAMPLE_RATE, SynthesisEngine, fallback_engine, requires_reference
from .engines.registry import created_engines, get_engine
from .engines.worker_protocol import ERROR_AUDIO, text_fingerprint
from .pronunciation import active_rules
from .settings_resolution import (
    COMMON_FIELDS,
    effective_values,
    engine_defaults,
    resolve_synthesis_settings,
    split_values,
)
from .short_utterance import classify_utterance, count_words
from .text_preprocess import normalize, normalize_stages
from .transcribe import transcribe_audio, word_error_rate
from .voices_store import Voice, get_store

logger = logging.getLogger(__name__)

# Ограничитель пика после нормализации: выше 0 dBFS файл клипует.
_PEAK_LIMIT = 0.99


def _text_fingerprint_text(text: str) -> str:
    """Отпечаток реплики для лога: длина, хеш и короткая выжимка.

    Текст реплики — пользовательский контент, и в логе он не нужен целиком; но
    при разборе падения важно понять, на какой именно реплике оно случилось, и
    сверить это с тем, что человек видит в интерфейсе (см. worker_protocol).
    """
    fingerprint = text_fingerprint(text)
    return (
        f"{fingerprint['length']} знаков, sha1 {fingerprint['sha1']}, "
        f"«{fingerprint['preview']}»"
    )

ProgressCallback = Callable[[int, int, str], None]
# Ожидание освобождения памяти перед повторной попыткой (её ставит очередь).
WaitForMemory = Callable[[], Awaitable[None]]
# Сообщение о ходе работы (попадает в статус задачи в интерфейсе).
NoteCallback = Callable[[str], None]
# Фактические времена куска: индекс (с нуля), сколько секунд занял синтез,
# сколько вышло аудио и чем закончилась проверка. Очередь кормит этим
# статистику ETA (см. backend/eta.py) — пайплайн сам её не ведёт и про оценку
# времени ничего не знает.
ChunkTimingCallback = Callable[[int, float, float, "QaOutcome | None"], None]
# Движок поднят: id и сколько секунд заняла загрузка (холодный старт).
EngineLoadedCallback = Callable[[str, float], None]

# Чем закончился QA-цикл куска. Отдельные значения, а не флаг «получилось»: в
# интерфейсе важно различать «проверено и прошло» и «проверить не успели» —
# принятое без полной проверки аудио не должно выглядеть проверенным.
QA_PASSED = "passed"
QA_BUDGET = "budget_exhausted"
QA_ATTEMPTS = "attempts_exhausted"
QA_UNAVAILABLE = "unavailable"


@dataclass
class QaSettings:
    """Параметры проверки куска.

    Значения по умолчанию берутся из конфига, но живут отдельным объектом, а не
    читаются из `config` внутри цикла: в тестах цикл должен прогоняться на своих
    числах, а не на боевых трёхстах секундах бюджета.
    """

    wer_threshold: float = config.QA_WER_THRESHOLD
    max_attempts: int = config.QA_MAX_ATTEMPTS
    budget_sec: float = config.QA_BUDGET_SEC
    # Режим проверки: `config.QA_MODES`. По умолчанию строгий — `QaSettings()` без
    # аргументов означает прежнее поведение «каждый кусок через Whisper».
    mode: str = config.QA_MODE_STRICT

    @classmethod
    def for_mode(cls, mode: str) -> "QaSettings | None":
        """Настройки для режима из запроса; None — проверка выключена.

        «Выключено» — это `None`, а не объект с режимом `off`: так у проверки
        остаётся единственное представление «её нет», и весь остальной код
        продолжает спрашивать `settings.qa is None`.
        """
        if mode == config.QA_MODE_OFF:
            return None
        return cls(mode=mode)

    @property
    def smart(self) -> bool:
        """Сначала дешёвый отбор, в Whisper — только подозрительные куски."""
        return self.mode == config.QA_MODE_SMART


@dataclass
class QaOutcome:
    """Итог проверки одного куска: чем цикл закончился и что из него выбрано."""

    status: str
    wer: float | None
    attempts: int
    # Режим, в котором шла проверка, и вердикт дешёвого отбора. Отбора может не
    # быть: строгий режим идёт в расшифровку сразу, и тогда `screening` — None.
    mode: str = config.QA_MODE_STRICT
    screening: qa_screening.Screening | None = None
    # Расшифровка куска, если она была. Хранится вместе с итогом, потому что по
    # ней видно, **что услышал** Whisper: WER отвечает «насколько похоже», а
    # диагностике нужен ещё и текст — «Да, да, да…» по числу не читается.
    transcription: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "wer": None if self.wer is None else round(self.wer, 4),
            "attempts": self.attempts,
            "mode": self.mode,
            "screening": None if self.screening is None else self.screening.to_dict(),
            "transcription": self.transcription,
        }


@dataclass
class ShortUtteranceSettings:
    """Настройки слоя коротких реплик (§23).

    Выключено по умолчанию: стратегия включается по результатам живого benchmark'а
    (`tools/short_bench.py`), а не по предположению. `strategy = auto` берёт
    измеренную политику движка (`short_utterance.default_strategy`).
    """

    enabled: bool = config.SHORT_UTTERANCE_DEFAULT_ENABLED
    strategy: str = su.STRATEGY_AUTO
    # Сторона контекста (A/B/C/D из §30) и carrier; None — значения стратегии.
    side: str | None = None
    carrier: str | None = None
    thresholds: su.ShortThresholds = field(default_factory=su.ShortThresholds)
    max_attempts: int = config.SHORT_UTTERANCE_MAX_ATTEMPTS
    boundary_method: str = config.SHORT_UTTERANCE_BOUNDARY_METHOD

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "side": self.side,
            "carrier": self.carrier,
            "thresholds": self.thresholds.to_dict(),
            "max_attempts": self.max_attempts,
            "boundary_method": self.boundary_method,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ShortUtteranceSettings | None":
        """Настройки из запроса; `None` — слой не запрашивали вовсе.

        Пустое тело означает «как решено в приложении» (политика из конфига), а не
        «выключить»: иначе интерфейс, не передавший поле, гасил бы оптимизацию.
        """
        if raw is None:
            return None
        settings = cls(
            enabled=bool(raw.get("enabled", config.SHORT_UTTERANCE_DEFAULT_ENABLED)),
            strategy=str(raw.get("strategy") or su.STRATEGY_AUTO),
            side=raw.get("side") or None,
            carrier=(raw.get("carrier") or None),
            thresholds=su.ShortThresholds.from_dict(raw.get("thresholds")),
            max_attempts=int(raw.get("max_attempts") or config.SHORT_UTTERANCE_MAX_ATTEMPTS),
            boundary_method=str(
                raw.get("boundary_method") or config.SHORT_UTTERANCE_BOUNDARY_METHOD
            ),
        )
        if settings.strategy not in su.SELECTABLE_STRATEGIES:
            raise ValueError(f"Неизвестная стратегия короткой реплики: {settings.strategy}")
        if settings.side is not None and settings.side not in su.CONTEXT_SIDES:
            raise ValueError(f"Неизвестная сторона контекста: {settings.side}")
        if settings.boundary_method not in boundary_module.METHODS:
            raise ValueError(f"Неизвестный метод границ: {settings.boundary_method}")
        return settings


@dataclass
class SpeakerSettings:
    voice_id: str
    speed: float = config.DEFAULT_SPEED
    cfg_strength: float = config.DEFAULT_CFG_STRENGTH
    nfe_step: int = config.DEFAULT_NFE_STEP
    target_rms: float = config.DEFAULT_TARGET_RMS
    gain_db: float = config.DEFAULT_GAIN_DB
    pitch_semitones: float = config.DEFAULT_PITCH_SEMITONES
    # Пауза перед репликами этого спикера; None — брать общую из RenderSettings.
    # Имя отличается от общего `RenderSettings.pause_ms`: в одной модели запроса
    # (режим «Сплошной текст») оба поля соседствуют, и совпадающие имена слились бы.
    pause_override_ms: int | None = None
    # Ручки движка этого голоса, переопределённые на одну генерацию (у XTTS —
    # температура и штраф за повторы). То, чего здесь нет, берётся из карточки голоса.
    engine_params: dict = field(default_factory=dict)
    # Что из значений выше выбран сам этот слот. Только эти поля перекрывают
    # пресет голоса; остальные наследуются (см. settings_resolution). Пусто —
    # «слот ничего не выбирал»: так выглядят проектные спикеры, у которых ручки
    # не трогали, и это же отличает их от разовой формы, заполненной целиком.
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSettings":
        raw_pause = data.get("pause_override_ms")
        raw_params = data.get("engine_params")
        settings = cls(
            voice_id=str(data.get("voice_id", "")),
            speed=float(data.get("speed", config.DEFAULT_SPEED)),
            cfg_strength=float(data.get("cfg_strength", config.DEFAULT_CFG_STRENGTH)),
            nfe_step=int(data.get("nfe_step", config.DEFAULT_NFE_STEP)),
            target_rms=float(data.get("target_rms", config.DEFAULT_TARGET_RMS)),
            gain_db=float(data.get("gain_db", config.DEFAULT_GAIN_DB)),
            pitch_semitones=float(data.get("pitch_semitones", config.DEFAULT_PITCH_SEMITONES)),
            pause_override_ms=None if raw_pause is None else int(raw_pause),
            engine_params=dict(raw_params) if isinstance(raw_params, dict) else {},
        )
        # Ключи, которые пришли в запросе, и есть выбор этого слоя: значения,
        # досчитанные из дефолтов, в слой не попадают — иначе слот молча
        # перекрывал бы пресет голоса тем, чего пользователь не выбирал.
        settings.overrides = {
            name: value
            for name, value in data.items()
            if value is not None and (name in COMMON_FIELDS or name == "engine_params")
        }
        return settings


@dataclass
class RenderSettings:
    pause_ms: int = config.DEFAULT_PAUSE_MS
    cross_fade_duration: float = config.DEFAULT_CROSS_FADE_DURATION
    auto_accent: bool = True
    output_format: str = config.DEFAULT_OUTPUT_FORMAT
    # Проверка куска (`config.QA_MODES`) или None — обычный путь, одна попытка без
    # расшифровки. Выбор пользователя на задачу, а не свойство установки: цена
    # проверки — синтез плюс отдельный процесс Whisper на каждую попытку.
    qa: QaSettings | None = None
    # Контракт рендера: текст обязан быть уже подготовленным анализом проекта.
    # Включено у проектного рендера и выключено у разовых задач (`/api/generate`,
    # `/api/render-text`), у которых сохранённого анализа нет. Флаг существует
    # ровно для того, чтобы «посчитать текст на месте» не осталось незаметным
    # обходом обязательной подготовки (см. `render_dialogue`).
    require_prepared: bool = False
    # Оптимизация коротких реплик (§23). `None` — слой не запрашивали: работает
    # политика приложения (по умолчанию выключено, пока benchmark не покажет
    # улучшение).
    short_utterance: "ShortUtteranceSettings | None" = None
    # Прогрев коротких реплик (warmup-context): перед короткой фразой в модель
    # уходит скрытый текст, а в файл попадает только цель. `None` — политика
    # приложения (`config.WARMUP_ENABLED`), `False` — выключено для этой задачи,
    # `True` — включено. Свойство запуска, как и проверка качества.
    warmup: bool | None = None
    # Имя готового файла, заданное пользователем. Свойство запуска, а не сборки:
    # в настройках проекта оно не запоминается — иначе следующая сборка молча
    # взяла бы имя, выбранное для прошлого файла, и получила бы «-2» вместо него.
    # Пустая строка — прежнее поведение: имя из id задачи.
    output_name: str = ""


@dataclass
class RenderResult:
    output_path: Path
    duration_sec: float
    replicas_done: int
    # Границы кусков в сэмплах: по ним можно заменить одну реплику, не трогая остальное.
    segments: list[tuple[int, int]] = field(default_factory=list)
    # Сид каждого куска (`None` — движок сид не принимает). Хранится, чтобы кусок
    # можно было воспроизвести: без него понравившийся вариант не повторить.
    seeds: list[int | None] = field(default_factory=list)
    # Итог строгой проверки по индексу реплики; None — проверка не гонялась.
    qa: list[QaOutcome | None] = field(default_factory=list)
    # Диагностические метрики подготовленного куска по индексу реплики. Не
    # оценка естественности голоса: набор измерений, по которым видно, почему
    # take может звучать плохо (см. `take_quality`).
    qualities: list[take_quality.TakeQuality] = field(default_factory=list)
    # План короткой реплики по индексу: класс, стратегия, источник контекста,
    # отпечаток синтез-текста (§24). Пусто — слой не применялся.
    short_runs: dict[int, "ShortRun"] = field(default_factory=dict)
    # Прогрев коротких реплик по индексу: префикс, граница цели и откат (§15).
    warmups: dict[int, dict] = field(default_factory=dict)
    # Фактически использованный референс по индексу реплики (UPDATE 2 §10):
    # профиль, эмоция и признак отката на NEUTRAL. Нужен, чтобы карточка реплики
    # и метаданные take'а показывали **факт**, а не намерение.
    references: dict[int, dict] = field(default_factory=dict)


@dataclass
class RenderPartial:
    """Что успело получиться до прерывания рендера (creash_report).

    Состояние прикрепляется к исключению, а не заворачивает его: типы исключений
    (отмена, таймаут, ошибка движка, падение воркера) значимы для вызывающего, и
    подмена их общей обёрткой сломала бы разбор ошибок в очереди. Очередь читает
    это состояние через `getattr(exc, "partial_render", None)` и решает:
    сохранить готовые куски вариантами и продолжить с упавшей реплики или
    завершить задачу ошибкой, не потеряв сделанное.

    Куски здесь **подготовленные** (края, тембр, RMS), но не финализированные:
    нормализация громкости считается по всему треку, а трека ещё нет. При
    продолжении рендера они встают в начало сборки как есть, поэтому итоговый
    файл всё равно получает общую нормализацию.
    """

    pieces: list[np.ndarray]
    segments: list[tuple[int, int]]
    seeds: list[int | None]
    qa: list[QaOutcome | None]
    qualities: list[take_quality.TakeQuality]
    # Индекс реплики (с нуля), на которой прервалось; `None` — все реплики
    # синтезированы, а сбой случился на сборке (склейка, запись файла).
    failed_index: int | None
    error: str
    # Планы коротких реплик готовых кусков: продолжение рендера не должно терять
    # метаданные уже сделанного (см. §24). Поле с значением по умолчанию, поэтому
    # стоит последним.
    short_runs: dict[int, "ShortRun"] = field(default_factory=dict)

    @property
    def done(self) -> int:
        """Сколько реплик готово — подряд, с начала диалога."""
        return len(self.pieces)

    def to_dict(self) -> dict:
        return {
            "replicas_done": self.done,
            "failed_index": self.failed_index,
            "error": self.error,
        }


@dataclass
class ChunkVariant:
    """Сохранённый вариант куска: файл для прослушивания, сид и подпись.

    Вариант — это готовое аудио, а не рецепт: выбор между вариантами должен быть
    мгновенным, а повторный синтез с тем же сидом на MPS не гарантирует тот же
    результат. Файл лежит уже финализированным (`_finalize_track`), то есть в том
    же виде, в каком звучит в треке: варианты сравнивают на слух, и разная
    громкость решала бы выбор вместо голоса.
    """

    id: str
    path: Path
    label: str
    seed: int | None
    duration_sec: float
    # Итог строгой проверки этого варианта (`None` — проверка не гонялась).
    # Свойство самого аудио, а не реплики: у вариантов разная история проверок, и
    # при выборе варианта к реплике должна вернуться именно его отметка.
    qa: QaOutcome | None = None
    # Диагностика этого звучания (клиппинг, тишина, громкость, LUFS...). None —
    # метрик нет: так выглядят варианты, сохранённые до фазы, и это норма.
    quality: take_quality.TakeQuality | None = None
    # Метаданные короткой реплики (§24): класс, стратегия, источник контекста,
    # отпечаток синтез-текста. None — обычная реплика.
    plan: dict | None = None


class ChunkTimeoutError(RuntimeError):
    """Синтез одного куска не уложился в TTS_CHUNK_TIMEOUT_SEC — похоже на зависание."""


# Расширение недописанного аудио: `<имя>.part` рядом с целевым файлом. Имя
# содержит `.part`, поэтому остатки от прошлых запусков находятся одним правилом
# (см. `cleanup_partials` и `cache_cleanup`).
PARTIAL_SUFFIX = ".part"


class AudioFileError(RuntimeError):
    """Записанное аудио не прошло проверку: пустой, обрезанный или нечитаемый файл.

    Отдельный класс, а не общий `RuntimeError`: в очереди по нему выставляется
    свой тип ошибки (`AUDIO_ERROR`), и пользователь видит «файл записан
    некорректно», а не «движок синтеза упал» — это разные причины и разные
    советы (проверить место на диске против повторить синтез).
    """

    error_type = ERROR_AUDIO


# Сколько ждать подтверждения, что зависший воркер убит. Короткий шаг: ответ
# пользователю важнее, а `abort_current` — это `kill()` плюс ожидание кода
# возврата, то есть секунды, а не минуты.
ABORT_TIMEOUT_SEC = 15.0


class JobAbortedError(RuntimeError):
    """Задача прервана watchdog'ом (перерасход ресурсов), а не упала сама."""


class JobCancelledError(JobAbortedError):
    """Задача отменена пользователем на безопасной точке.

    Наследник `JobAbortedError`, а не соседний класс: для пайплайна это тот же
    «прерванный прогон» (см. `should_abort`), и старый код, ловящий прерывание,
    продолжает работать. Различает их вызывающий: отмена — не ошибка, а
    отдельное состояние задачи (`cancelled`), тогда как watchdog — `error`.
    """


def _pitch_shift(chunk: np.ndarray, semitones: float) -> np.ndarray:
    """Меняет высоту тона, не трогая длительность. Тяжёлый импорт — только по факту."""
    if not semitones:
        return chunk
    import librosa

    return np.asarray(
        librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=float(semitones)),
        dtype=np.float32,
    )


def _edge_bounds(chunk: np.ndarray, margin_ms: float | None = None) -> tuple[int, int] | None:
    """Границы речи в куске по энергетическому порогу; `None` — резать нечего.

    Выделено из `_trim_edge_silence` ради трейса (§15 UPDATE 2): чтобы объяснить
    пропажу окончания, нужно знать **сколько** сэмплов снято и с какой стороны, а
    не только получить обрезанный массив. Логика та же, что была: кадры по
    `EDGE_SILENCE_FRAME_MS`, порог `EDGE_SILENCE_DB`, запас с обеих сторон.

    `margin_ms` — запас вокруг речи. По умолчанию общий (30 мс); короткая реплика
    получает более широкий (`SHORT_EDGE_GUARD_MS`): у неё край и есть слово, а
    цена лишних 30 мс тишины в файле нулевая — паузу всё равно задаёт `pause_ms`.
    """
    frame = int(SAMPLE_RATE * config.EDGE_SILENCE_FRAME_MS / 1000)
    if frame <= 0 or chunk.size < 4 * frame:
        return None
    usable = chunk.size - chunk.size % frame
    frames = chunk[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loud = np.flatnonzero(rms > 10.0 ** (config.EDGE_SILENCE_DB / 20.0))
    if loud.size == 0:  # кусок целиком тихий — резать нечего
        return None
    guard_ms = config.EDGE_SILENCE_MARGIN_MS if margin_ms is None else margin_ms
    margin = int(SAMPLE_RATE * max(guard_ms, 0) / 1000)
    start = max(0, int(loud[0]) * frame - margin)
    end = min(chunk.size, (int(loud[-1]) + 1) * frame + margin)
    if end - start < 2 * frame:
        return None
    return start, end


def _trim_edge_silence(chunk: np.ndarray, margin_ms: float | None = None) -> np.ndarray:
    """Срезает тишину, которую модель оставила на краях куска.

    Модель почти всегда добавляет к сгенерированному куску немного тишины, а
    `pause_ms` добавляет паузу поверх неё. Суммарный зазор между репликами тогда
    гуляет от куска к куску и звучит неритмично. Обрезка по энергетическому
    порогу делает паузу ровно той, что задал пользователь.
    """
    bounds = _edge_bounds(chunk, margin_ms)
    if bounds is None:
        return chunk
    start, end = bounds
    return chunk[start:end]


def _short_guard_ms(text: str) -> float:
    """Запас обрезки для этого текста: расширенный для коротких реплик.

    Критерий — тот же классификатор, что и у слоя коротких реплик
    (`classify_utterance`): «край = слово» бывает ровно у коротких фраз, и
    отдельного порога заводить не нужно.
    """
    try:
        if classify_utterance(text or "").is_short:
            return max(float(config.SHORT_EDGE_GUARD_MS), float(config.EDGE_SILENCE_MARGIN_MS))
    except Exception as exc:  # noqa: BLE001 — классификация не повод менять синтез
        logger.debug("Класс реплики не определён (%s) — беру общий запас обрезки", exc)
    return float(config.EDGE_SILENCE_MARGIN_MS)


def _prepare_chunk(
    chunk: np.ndarray,
    settings: SpeakerSettings,
    *,
    trace: "synthesis_trace.ReplicaTrace | None" = None,
    text: str = "",
) -> np.ndarray:
    """Готовит кусок к склейке: края, тембр и громкость.

    Один проход с одной копией: на длинных репликах лишние копии waveform заметны
    и по памяти, и по времени. Gain применяется после нормализации RMS — иначе
    нормализация обнулила бы ручную балансировку громкости.

    `trace` (необязательный) записывает срезы аудио по стадиям и числа обрезки:
    без него «слово пропало» неотличимо от «модель его не сказала». На сам звук
    трейс не влияет — только наблюдает.

    `text` — текст реплики: по нему выбирается запас обрезки (у коротких фраз он
    шире). Пустая строка означает общий запас, то есть прежнее поведение.
    """
    raw = np.asarray(chunk, dtype=np.float32)
    guard_ms = _short_guard_ms(text)
    bounds = _edge_bounds(raw, guard_ms) if raw.size else None
    if bounds is not None:
        start, end = bounds
        trimmed = raw[start:end]
    else:
        trimmed = raw
    if trace is not None:
        trace.edge_trim_enabled = True
        trace.edge_silence_db = float(config.EDGE_SILENCE_DB)
        trace.edge_fade_ms = float(config.EDGE_FADE_MS)
        trace.pre_guard_ms = guard_ms
        trace.post_guard_ms = guard_ms
        if bounds is None:
            trace.trim(raw, 0, raw.size, note="резать нечего (короткий кусок или тишина)")
        else:
            trace.trim(raw, start, end, note=f"запас {guard_ms:.0f} мс")
        trace.stage(synthesis_trace.STAGE_TRIM, trimmed)
    result = np.array(_pitch_shift(trimmed, settings.pitch_semitones),
                      dtype=np.float32, copy=True)

    fade = int(SAMPLE_RATE * config.EDGE_FADE_MS / 1000)
    # Модель иногда оставляет на самом краю куска щелчок. Кроссфейд с соседним
    # куском такой щелчок не убирает — он его только смешивает с чужим звуком.
    if fade > 0 and result.size >= 2 * fade:
        result[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        result[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    rms = float(np.sqrt(np.mean(np.square(result)))) if result.size else 0.0
    if rms > 1e-6:
        result *= settings.target_rms / rms

    if settings.gain_db:
        result *= 10.0 ** (settings.gain_db / 20.0)

    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > _PEAK_LIMIT:
        result *= _PEAK_LIMIT / peak
    if trace is not None:
        trace.stage(synthesis_trace.STAGE_FINAL, result)
    return result


def _normalize_loudness(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    """Подтягивает громкость всего трека к target_lufs (ITU-R BS.1770).

    Нормализация кусков по RMS выравнивает их между собой, но не даёт равной
    *воспринимаемой* громкости: плотность звука у F5 и XTTS разная, поэтому на
    стыке движков слышен скачок даже при одинаковом `target_rms`. LUFS считает
    громкость целого файла и лечит именно это.
    """
    import pyloudnorm as pyln

    try:
        loudness = float(pyln.Meter(SAMPLE_RATE).integrated_loudness(audio))
    except ValueError:  # короче 400 мс — измерять нечего, Meter бросает ValueError
        return audio
    if not math.isfinite(loudness):  # цифровая тишина даёт -inf
        return audio
    gain_db = target_lufs - loudness
    if abs(gain_db) < 0.1:
        return audio
    logger.info("Громкость трека: %.1f LUFS → %.1f LUFS (%+.1f дБ)", loudness, target_lufs, gain_db)
    return audio * (10.0 ** (gain_db / 20.0))


def _limit_peaks(audio: np.ndarray, ceiling: float = _PEAK_LIMIT) -> np.ndarray:
    """Лимитер: придерживает пики выше потолка, не трогая громкость остального.

    Нужен после подъёма до целевого LUFS: без него пики уходят за 0 dBFS и файл
    клипует. Усиление считается по блокам (не по отсчётам) — иначе на резких
    пиках лимитер сам даёт щелчки; атака берётся с запасом вперёд, восстановление
    плавное.
    """
    block = int(SAMPLE_RATE * config.LIMITER_BLOCK_MS / 1000)
    if block <= 0 or audio.size < 4 * block:
        return audio
    usable = audio.size - audio.size % block
    peaks = np.abs(audio[:usable]).reshape(-1, block).max(axis=1)
    needed = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))

    attacked = needed.copy()
    for shift in range(1, max(config.LIMITER_LOOKAHEAD_BLOCKS, 1)):
        attacked[:-shift] = np.minimum(attacked[:-shift], needed[shift:])

    release_ratio = 1.0 + config.LIMITER_BLOCK_MS / config.LIMITER_RELEASE_MS
    gains = np.empty_like(attacked)
    gain = 1.0
    for index, target in enumerate(attacked):
        gain = float(target) if target < gain else min(float(target), gain * release_ratio)
        gains[index] = gain

    limited = audio[:usable] * np.repeat(gains, block)
    tail = audio[usable:] * gains[-1]
    return np.concatenate((limited, tail))


def _finalize_track(audio: np.ndarray) -> np.ndarray:
    """Финальный проход по собранному треку: воспринимаемая громкость и потолок.

    Применяется и к целиком собранному диалогу, и после замены одной реплики:
    иначе перегенерированный кусок остался бы на другой громкости, чем файл.
    """
    return _limit_peaks(_normalize_loudness(np.asarray(audio, dtype=np.float32), config.OUTPUT_LUFS))


def _resolve_voice(label: str, settings: SpeakerSettings) -> Voice:
    if not settings.voice_id:
        raise ValueError(f"Не выбран голос для «{label}»")
    voice = get_store().get(settings.voice_id)
    if voice is None:
        raise ValueError(f"Голос {settings.voice_id} для «{label}» не найден")
    if not voice.audio_path.exists():
        raise ValueError(f"Файл референса для голоса «{voice.name}» потерян ({voice.audio_file})")
    if not voice.ref_text.strip():
        raise ValueError(
            f"У голоса «{voice.name}» не заполнен референс-текст — без него синтез невозможен"
        )
    return voice


def voice_layer(voice: Voice) -> dict:
    """Слой голоса в иерархии: пресет плюс подобранные ручки движка.

    Два места хранения, а не одно, потому что у них разная жизнь: пресет — это
    общие ручки (скорость, CFG, NFE), а `engine_params` привязаны к движку и
    сбрасываются при его смене. Для резолва это один слой.
    """
    layer: dict[str, Any] = dict(voice.preset or {})
    if voice.engine_params:
        layer["engine_params"] = dict(voice.engine_params)
    return layer


def tuning_for(voice: Voice, base: SpeakerSettings, overrides: dict | None = None) -> SpeakerSettings:
    """Эффективные параметры куска: движок → пресет голоса → слот → реплика.

    Единственная точка склейки слоёв для синтеза: прослушивание, рендер проекта
    и карточка куска зовут её с разными наборами правок, но порядок и зажим
    значений считаются здесь, а не повторяются в API и на фронте.
    """
    values = effective_values(
        resolve_synthesis_settings(
            engine_defaults(voice.engine),
            voice_layer(voice),
            base.overrides,
            overrides,
        )
    )
    common, engine = split_values(values)
    settings = SpeakerSettings.from_dict(
        {**common, "voice_id": voice.id, "engine_params": engine}
    )
    # Это готовый результат, а не слой правок: если бы `overrides` здесь остались
    # заполненными, вложенный резолв принял бы все значения за выбор этого слоя.
    settings.overrides = {}
    return settings


def settings_view(
    voice: Voice, base: SpeakerSettings, overrides: dict | None = None
) -> dict:
    """Разрешённые параметры вместе с источником каждого значения.

    Нужно интерфейсу: без источника «сброшено» и «подобрано здесь» выглядели бы
    одинаково, а подпись «наследуется от голоса Анна» нечем было бы собрать.
    """
    return resolve_synthesis_settings(
        engine_defaults(voice.engine),
        voice_layer(voice),
        base.overrides,
        overrides,
    )


def _settings_for(replica: Replica, base: SpeakerSettings) -> SpeakerSettings:
    """Параметры куска по слоям, с голосом из `voice_id`.

    Оставлено для тех, кому не нужен сам голос: карточка куска и тесты. Рендер
    диалога разрешает голос один раз на диалог и зовёт `tuning_for` напрямую —
    читать `voices.json` на каждую реплику незачем.
    """
    voice = _resolve_voice(
        replica.label, replace(base, voice_id=str(replica.voice_id or base.voice_id))
    )
    return tuning_for(voice, base, replica.overrides)


def _engine_for(voice: Voice) -> SynthesisEngine:
    """Движок, выбранный для этого голоса.

    Единственное место, где работает откат по недоступности. Откат — не подмена
    «на похожий»: он срабатывает только когда файлов движка нет на диске (модель
    не скачана) и только если сам движок объявил, на кого откатываться. Иначе
    диалог падал бы на каждой реплике до первой загрузки весов, а недоступность
    выглядела бы как поломка синтеза. Каждый такой случай попадает в лог: молча
    отрендеренный другим голосом диалог — это то, что нельзя не заметить.
    """
    engine_id = str(voice.engine or "")
    fallback_id = fallback_engine(engine_id)
    if fallback_id and not model_manager.engine_available(engine_id):
        logger.warning(
            "Движок «%s» недоступен (файлы модели не установлены) — голос «%s» "
            "синтезируется движком «%s». Скачайте модель во вкладке «Модели», "
            "чтобы услышать выбранное звучание.",
            engine_id,
            voice.name,
            fallback_id,
        )
        engine_id = fallback_id
    try:
        return get_engine(engine_id)
    except ValueError as exc:
        raise ValueError(f"У голоса «{voice.name}» неизвестный движок «{voice.engine}»") from exc


def _reference_for(voice: Voice, engine: SynthesisEngine, replica: Replica):
    """Референс под действующую просодию реплики; `None` — движку он не нужен.

    Пресетный движок (см. `EngineInfo.supports_cloning`) говорит встроенным
    голосом, и референса у такого голоса просто нет. Спрашивать его у резолвера
    мало того что бессмысленно — резолвер честно упал бы, не найдя записи, и
    синтез остановился бы на движке, которому ничего не мешало работать.
    """
    if not requires_reference(engine.id):
        return None
    return reference_resolver.resolve_prosody(
        voice, engine.id, replica.prosody_effective, profile_id=replica.reference_profile_id
    )


def engine_for_id(engine_id: str) -> SynthesisEngine:
    """Движок по id, без карточки голоса.

    Нужен сравнению движков: оно перебирает движки для одного и того же голоса,
    поэтому выбирать движок «из голоса» здесь нечего. Ошибка — понятным текстом,
    а не молчаливым F5: неизвестный id должен быть виден вызывающему.
    """
    try:
        return get_engine(engine_id)
    except ValueError as exc:
        raise ValueError(f"Неизвестный движок синтеза: {engine_id}") from exc


@dataclass(frozen=True)
class TextStages:
    """Путь текста до модели по стадиям — то, что показывает preview.

    Стадии, а не один итог, потому что preprocessing непрозрачен: пользователь
    видит «В 2026 году цена выросла на 5%.» и не понимает, откуда взялось
    «две тысячи двадцать шестом». Показывая каждую ступень рядом с итогом, preview
    отвечает и на «что уйдёт в модель», и на «почему именно это». Если стадия
    ничего не изменила, её результат равен предыдущей — по этому равенству видно,
    что шаг сработал вхолостую (например, словарь без совпадений).

    `yo` — отдельная стадия восстановления «ё»: она стоит между нормализацией и
    словарём и видна пользователю, потому что замена «е» на «ё» меняет звучание, а
    не только написание. `matches` — отчёт словаря из того же прохода, что дал
    `dictionary`: preview не может показать правило, которого не было, или
    пропустить сработавшее. `accents_applied` — пытались ли вообще ставить ударения
    (RUAccent поднимается только когда его просят и когда движок понимает «+»).
    """

    original: str
    normalized: str
    yo: str
    dictionary: str
    accentized: str
    final: str
    matches: list[dict]
    supports_accents: bool
    accents_applied: bool


def preview_text(
    text: str,
    *,
    supports_accents: bool,
    auto_accent: bool,
    rules: list | None = None,
) -> TextStages:
    """Прогоняет текст по всем стадиям preprocessing и возвращает каждую.

    Единственный источник истины: и синтез (`_text_for_engine`), и preview-эндпоинт
    зовут эту функцию, поэтому «что услышит модель» не может разойтись с тем, что
    реально уходит в движок, — совпадение обеспечено конструкцией, а не дисциплиной
    двух реализаций.

    Порядок важен и повторяет план фаз 6–8: нормализация → восстановление «ё» →
    пользовательский словарь → RUAccent. Правила словаря берутся из сервиса (снимок
    в памяти, не чтение базы на каждый кусок), а флаг `supports_accents` решает,
    дойдут ли «+» из замены до движка: XTTS прочитала бы знак как отдельный символ,
    поэтому для неё словарь отдаёт замену без разметки.

    Все стадии — срезы одного прохода `normalize_stages`: отдельные прогоны однажды
    разошлись бы, и preview показал бы не то, что ушло в модель. «Ё» здесь
    детерминированное и неоднозначные омографы не трогает — их разрешает словарь.

    RUAccent получает уже развёрнутые числа: «12» должно стать «двена́дцать», а не
    остаться цифрами. Ударения — только у движков, которые их понимают
    (`supports_accents`); выключенный `auto_accent` оставляет стадию `accentized`
    равной словарю, чтобы «выключено» было видно, а не выглядело как «ничего не
    нашлось».
    """
    if rules is None:
        rules = active_rules()
    stages = normalize_stages(text, rules, supports_accents=supports_accents)
    accents_applied = bool(auto_accent and supports_accents)
    accentized = accentuate(stages.result) if accents_applied else stages.result
    return TextStages(
        original=text,
        normalized=stages.normalized,
        yo=stages.yo,
        dictionary=stages.result,
        accentized=accentized,
        final=accentized,
        matches=stages.matches,
        supports_accents=supports_accents,
        accents_applied=accents_applied,
    )


def _text_for_engine(text: str, engine: SynthesisEngine, auto_accent: bool) -> str:
    """Готовит текст куска к отправке в модель.

    Тонкая обёртка над `preview_text`: пайплайн берёт из стадий только итог, но
    проходит ровно тот же путь, что и preview. Держать здесь второй порядок шагов
    значило бы однажды показать пользователю одно, а отправить в модель другое.
    """
    return preview_text(
        text,
        supports_accents=engine.supports_accents,
        auto_accent=auto_accent,
    ).final


def replica_chars(text: str) -> int:
    """Размер куска в знаках для статистики времени.

    Одна мера и у плана, и у наблюдения (см. `backend/eta.py`): если бы оценка
    считала знаки по одному тексту, а факт — по другому, статистика уточнялась
    бы в сторону, которой нет. Знаки считаются по тексту реплики, то есть без
    стадий preprocessing: нормализация («12:30» → «двенадцать часов тридцать
    минут») удлиняет текст, но идёт до движка и у всех языков и режимов
    одинакова по порядку величины, а тащить её в оценку значило бы поднимать
    нормализатор ради числа в ETA.
    """
    return len(text or "")


async def _load_engines(
    voices: Iterable[Voice], on_loaded: EngineLoadedCallback | None = None
) -> dict[str, SynthesisEngine]:
    """Грузит движки всех голосов диалога до первой реплики.

    Модель XTTS поднимается 20–40 секунд. Если грузить её лениво, эта пауза
    придётся на середину диалога — а прогретая модель нужна до начала синтеза,
    ровно как это было с единственным F5-TTS.

    Ключ — идентификатор из карточки голоса, а не `engine.id`: именно по нему
    движок ищется на каждой реплике, и подменённый движок (тесты, будущая
    обёртка над моделью) иначе оказался бы в словаре под другим именем.
    `on_loaded` — необязательный колбэк с id и длительностью загрузки: по нему
    очередь узнаёт цену холодного старта (`backend/eta.py`).
    """
    engines: dict[str, SynthesisEngine] = {}
    for voice in voices:
        if voice.engine in engines:
            continue
        engine = _engine_for(voice)
        engines[voice.engine] = engine
        if on_loaded is not None:
            # Длительность загрузки — это и есть холодный старт движка; она
            # уходит в статистику ETA, чтобы следующий рендер знал её заранее.
            started = time.monotonic()
            await asyncio.to_thread(engine.load)
            on_loaded(voice.engine, time.monotonic() - started)
        else:
            await asyncio.to_thread(engine.load)
        logger.info("Движок %s готов (%s)", engine.id, voice.name)
    return engines


def _engine_params(
    tuning: SpeakerSettings, render: RenderSettings, seed: int | None, gender: str = ""
) -> dict:
    """Ручки движка для одного куска.

    Значения приходят уже разрешёнными (`tuning_for`): движок → пресет голоса →
    слот → реплика. Склейки здесь нет намеренно: вторая точка слияния слоёв
    однажды уже разошлась с первой и перекрывала пресет голоса дефолтами слота.
    Общие ручки пайплайна (`target_rms`, `cross_fade_duration`) добавляются
    последними: они относятся к сборке куска, и движок читает их только если они
    у него есть. `seed` передаётся только движкам, которые его принимают
    (см. `SynthesisEngine.supports_seed`), и то же значение уходит в карточку
    куска — чтобы вариант можно было воспроизвести, а не только принять.

    `gender` — не ручка движка, а вход пресетного синтеза: Kokoro-ru выбирает по
    полу встроенный голос, а карточку голоса (`Voice`) движок не видит. Ключ не
    объявлен ни одной ручкой, поэтому `clamp_params` пропускает его как есть:
    пресетные движки его читают, остальные не замечают.
    """
    return {
        **tuning.engine_params,
        "cfg_strength": tuning.cfg_strength,
        "nfe_step": tuning.nfe_step,
        "target_rms": tuning.target_rms,
        "cross_fade_duration": render.cross_fade_duration,
        "seed": seed,
        "gender": str(gender or ""),
    }


def _pause_samples(speaker: SpeakerSettings, render: RenderSettings) -> int:
    """Пауза перед репликами спикера: его личное значение важнее общего."""
    pause_ms = render.pause_ms if speaker.pause_override_ms is None else speaker.pause_override_ms
    return int(SAMPLE_RATE * max(pause_ms, 0) / 1000)


def _log_synthesis_input(
    *,
    engine: SynthesisEngine,
    voice: Voice,
    text: str,
    source_text: str | None,
    tuning: SpeakerSettings,
    settings: RenderSettings,
    position: str,
    label: str,
    seed: int | None,
    params: dict,
    reference: "reference_resolver.ResolvedReference | None" = None,
    ref_audio_path: str = "",
    ref_text: str = "",
) -> None:
    """Одна строка со всем входом синтеза — непосредственно перед вызовом движка.

    Требование §1 пакета коротких реплик: без такого следа «звучит плохо» не с чем
    связать — ни движок, ни голос, ни длину текста, ни сид, ни референс в кадре не
    видны. Строка одна и структурированная (`key=value`), поэтому её разбирает и
    benchmark, и grep при разборе жалобы.

    Тексты пишутся **отпечатком** (длина, sha1, первые слова), а не целиком:
    реплика — пользовательский контент, и в лог она не попадает
    (`worker_protocol.text_fingerprint`). Benchmark'у, который работает на своих
    фиксированных фразах, этого достаточно, а разбор инцидента получает ровно то,
    что нужно для сверки с интерфейсом.
    """
    logger.info(
        "tts.synthesis.input engine=%s voice=%s label=%s position=%s source=[%s] "
        "final=[%s] prepared=%s words=%s class=%s ref=%s ref_text=[%s] speed=%.2f "
        "seed=%s params=%s emotion=%s profile=%s fallback=%s",
        engine.id,
        tuning.voice_id,
        label,
        position,
        _text_fingerprint_text(source_text if source_text is not None else text),
        _text_fingerprint_text(text),
        source_text is not None,
        count_words(text),
        classify_utterance(text).kind,
        Path(ref_audio_path).name if ref_audio_path else voice.audio_path.name,
        _text_fingerprint_text(ref_text or ""),
        tuning.speed,
        seed,
        params,
        "" if reference is None else reference.resolved_emotion,
        "" if reference is None else reference.profile_id,
        "" if reference is None else reference.fallback_used,
    )


def _synthesize_in_thread(
    engine: SynthesisEngine, **kwargs: Any
) -> tuple[tuple[np.ndarray, int] | None, Exception | None]:
    """Вызывает движок и возвращает исключение значением, а не наружу.

    `asyncio.wait_for` ловит `TimeoutError`, а с Python 3.10 `socket.timeout` — тот же
    самый класс. Поэтому любой чужой таймаут (сетевой запрос внутри движка, таймаут
    нативной библиотеки) выглядел бы как «инференс не уложился в CHUNK_TIMEOUT_SEC»
    и уводил бы поиск причины в сторону. Исключение возвращается значением и
    поднимается уже после ожидания — с настоящим типом и текстом.
    """
    try:
        return engine.synthesize(**kwargs), None
    except Exception as exc:  # noqa: BLE001 — исключение не глотаем, а переносим наверх
        return None, exc


async def _synthesize_chunk(
    engine: SynthesisEngine,
    voice: Voice,
    text: str,
    tuning: SpeakerSettings,
    settings: RenderSettings,
    position: str,
    label: str,
    source_text: str | None = None,
    reference: "reference_resolver.ResolvedReference | None" = None,
    seed: int | None = None,
) -> tuple[np.ndarray, int | None]:
    """Синтезирует один кусок, отделяя свой таймаут от чужой ошибки движка.

    Разделение принципиальное: `TimeoutError` от `asyncio.wait_for` означает, что
    зависла именно наша задача, а всё остальное — ошибку внутри движка, которую
    нужно показать пользователю такой, какая она есть (см. `_synthesize_in_thread`).
    Общий помощник на два места — сборку диалога и перегенерацию одной реплики —
    ещё и потому, что вторая копия этой обвязки однажды уже разошлась с первой.

    Сид выбирается здесь, а не в движке: пайплайн — единственное место, которое
    знает и про поддержку сида движком, и про то, что значение надо вернуть
    наверх вместе с куском.

    `source_text` — исходная реплика до подготовки; нужна только для строки лога
    о входе синтеза (см. `_log_synthesis_input`), модели она не уходит.

    `reference` — выбранный референс (эмоциональный или нейтральный, см.
    `reference_resolver`). Без него берётся основной референс голоса: разовые
    задачи и старые вызовы продолжают работать как раньше.

    `seed` — принудительный сид. Production его не задаёт (сид выбирается
    случайно на каждую попытку), а benchmark'у просодии он нужен: сравнение
    reference-профилей обязано отличаться **только** профилем (§32).
    """
    seed = (
        seed
        if seed is not None
        else (random.randrange(config.SEED_MAX) if engine.supports_seed else None)
    )
    params = _engine_params(tuning, settings, seed, voice.gender)
    ref_audio_path = str(reference.audio_path) if reference is not None else str(voice.audio_path)
    ref_text = reference.ref_text if reference is not None else voice.ref_text
    _log_synthesis_input(
        engine=engine,
        voice=voice,
        text=text,
        source_text=source_text,
        tuning=tuning,
        settings=settings,
        position=position,
        label=label,
        seed=seed,
        params=params,
        reference=reference,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
    )
    started = time.monotonic()
    try:
        outcome, failure = await asyncio.wait_for(
            asyncio.to_thread(
                _synthesize_in_thread,
                engine,
                text=text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                speed=tuning.speed,
                **params,
            ),
            timeout=config.CHUNK_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        # Зависший инференс нельзя прервать снаружи, если модель живёт в отдельном
        # процессе: поток остался бы в нативном вызове, воркер — занятым навсегда,
        # и следующая реплика встала бы за ним в очередь. Поэтому таймаут не
        # только сообщается наверх, но и убивает процесс движка. Хук
        # необязательный (`abort_current` есть только у изолированного движка):
        # движки в процессе бэкенда о нём не знают и знать не должны.
        abort = getattr(engine, "abort_current", None)
        if callable(abort):
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(abort, f"таймаут {config.CHUNK_TIMEOUT_SEC:.0f} с"),
                    timeout=ABORT_TIMEOUT_SEC,
                )
            except Exception as abort_exc:  # noqa: BLE001 — исход всё равно таймаут
                logger.error("Не удалось прервать зависший синтез %s: %s", engine.id, abort_exc)
        logger.warning(
            "Таймаут синтеза: реплика %s не готова за %.0f c (движок %s, голос %s, %s)",
            position, config.CHUNK_TIMEOUT_SEC, engine.id, tuning.voice_id,
            _text_fingerprint_text(text),
        )
        raise ChunkTimeoutError(
            f"Реплика {position} не синтезировалась за {config.CHUNK_TIMEOUT_SEC:.0f} c "
            "— возможно, завис инференс"
        ) from exc
    if failure is not None:
        logger.error(
            "Реплика %s (%s): движок %s упал (%s), %s",
            position, label, engine.id, type(failure).__name__, _text_fingerprint_text(text),
            exc_info=failure,
        )
        raise failure
    assert outcome is not None  # при пустом failure результат синтеза всегда есть
    chunk, _ = outcome
    logger.info(
        "Реплика %s (%s, %s) — %.1f c, %.1f c аудио, сид %s",
        position, label, engine.id, time.monotonic() - started,
        len(chunk) / SAMPLE_RATE, seed,
    )
    return chunk, seed


def _nudge_tuning(
    voice: Voice, engine: SynthesisEngine, tuning: SpeakerSettings, attempt: int
) -> SpeakerSettings:
    """Параметры следующей попытки: шаг в сторону предсказуемости.

    Не перебор вслепую: шаг считается от значения, которое реально ушло в модель
    (карточка слота важнее карточки голоса — ровно как в `_engine_params`), и
    зажимается границами, объявленными самим движком. Сид здесь не трогаем: его
    выбирает `_synthesize_chunk` заново на каждой попытке, и в карточке куска
    остаётся сид той попытки, что попала в файл.
    """
    if attempt <= 0:
        return tuning

    params = dict(tuning.engine_params)
    for param in engine.info.params:
        if param.name not in ("temperature", "repetition_penalty"):
            continue
        current = params.get(param.name, voice.engine_params.get(param.name, param.default))
        step = (
            -config.QA_TEMPERATURE_STEP
            if param.name == "temperature"
            else config.QA_REPETITION_PENALTY_STEP
        )
        params[param.name] = param.clamp(float(current) + step * attempt)

    # cfg_strength есть у F5 и не объявлен в паспорте XTTS — там он просто
    # остаётся неиспользованным, как и остальные чужие ключи в `engine_params`.
    low, high = config.CFG_RANGE
    cfg = min(max(tuning.cfg_strength + config.QA_CFG_STEP * attempt, low), high)
    return replace(tuning, cfg_strength=cfg, engine_params=params)


async def _transcribe_chunk(chunk: np.ndarray) -> str:
    """Расшифровывает готовый кусок отдельным процессом Whisper.

    Через временный файл: воркер читает аудио сам и живёт в своём процессе —
    передать ему waveform напрямую нечем. Файл удаляется сразу: кусок бывает в
    несколько минут аудио, и копить такие в output/ незачем.
    """

    def run() -> str:
        # Префикс — часть контракта с очисткой кеша: `cache_cleanup` узнаёт свои
        # остатки по явному имени (`voice-syntez-`), а не по маске «похоже на
        # временный файл». Процесс может упасть между созданием и удалением, и
        # тогда этот файл останется на диске — опознаваемым.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="voice-syntez-", delete=False
        ) as handle:
            path = Path(handle.name)
        try:
            _write_audio(path, chunk, "wav")
            return transcribe_audio(path)
        finally:
            path.unlink(missing_ok=True)

    return await asyncio.to_thread(run)


async def _measure_wer(chunk: np.ndarray, expected: str) -> float:
    """Считает WER куска относительно текста, который в него отправляли.

    Расшифровка проходит ту же предобработку (`normalize`), что и текст перед
    синтезом: Whisper пишет числа цифрами («2026»), а в модель ушло «две тысячи
    двадцать шестом» — без общего шага они расходились бы в каждом числе, и
    проверка валилась бы на ровном месте.

    Словарь произношения применяется к **обеим** сторонам. Модель произносит
    замену («SQL» → «эскьюэль»), и Whisper запишет именно её; если reference
    оставить без словаря, каждая такая реплика выглядела бы ошибкой — WER 0.75
    при пороге 0.2, лишние попытки и лишний запуск Whisper на ровном месте.
    Обратная сторона той же монеты: Whisper иногда пишет услышанное как «SQL»,
    и тогда словарь нужен уже расшифровке. Один шаг на оба текста закрывает оба
    случая — ровно так же, как `normalize` закрывает числа.
    """
    recognized = await _transcribe_chunk(chunk)
    if not recognized.strip():
        return 1.0  # модель промолчала — это провал проверки, а не «сверять нечего»
    rules = active_rules()
    return word_error_rate(
        normalize(expected, pronunciation=rules, supports_accents=False),
        normalize(recognized, pronunciation=rules, supports_accents=False),
    )


@dataclass
class ShortRun:
    """Короткая реплика в работе: план для модели и посчитанный вердикт.

    `Run` — потому что это состояние одного прогона, а не настройка: план уже
    учитывает движок (разметку ударений) и стратегию, а вердикт появляется после
    синтеза и решает, повторять ли попытку.
    """

    plan: su.SynthesisPlan
    settings: ShortUtteranceSettings
    verdict: su.ShortVerdict | None = None
    boundary: boundary_module.Boundary | None = None
    fallback: str = ""
    attempts: int = 0

    @property
    def active(self) -> bool:
        return self.settings.enabled

    def to_dict(self) -> dict:
        payload = self.plan.to_dict()
        # Стадия планировщика (§25): «что сделали» рядом с «что просили»
        # (`short_utterance_strategy`). Откат виден как FALLBACK_DIRECT, а не как
        # молчаливая подмена стратегии.
        payload["short_planner_stage"] = su.planner_stage(self.plan.strategy, self.fallback)
        payload["short_utterance_fallback"] = self.fallback
        payload["short_utterance_attempts"] = self.attempts
        payload["short_utterance_verdict"] = (
            None if self.verdict is None else self.verdict.to_dict()
        )
        payload["short_utterance_boundary"] = (
            None if self.boundary is None else self.boundary.to_dict()
        )
        return payload


def short_run_for(
    replica: Replica,
    engine: SynthesisEngine,
    settings: RenderSettings,
    *,
    context: su.ShortUtteranceContext | None = None,
) -> ShortRun | None:
    """Строит план короткой реплики — или `None`, если слой не применяется.

    Слой не применяется, когда он выключен или когда реплика обычная (NORMAL):
    длинный текст идёт прежним путём без единого лишнего шага (§28). Стратегия
    `auto` превращается в измеренную политику движка здесь же — по одному месту
    на все пути синтеза.
    """
    short = settings.short_utterance
    if short is None or not short.enabled:
        return None
    text = (replica.final_text or "").strip() or replica.text
    context = context or su.ShortUtteranceContext(target_index=0, target_text=text)
    utterance = context.utterance or su.classify_utterance(text, short.thresholds)
    if not utterance.is_short:
        return None
    strategy = su.resolve_strategy(engine.id, short.strategy)
    plan = su.build_plan(
        context,
        strategy=strategy,
        supports_accents=engine.supports_accents,
        carrier=short.carrier,
        side=short.side,
    )
    if not plan.utterance_class or plan.utterance_class == su.CLASS_NORMAL:
        plan = replace(plan, utterance_class=utterance.kind)
    return ShortRun(plan=plan, settings=short)


def _short_trace_fields(trace: "synthesis_trace.ReplicaTrace", short: ShortRun | None) -> None:
    """Переносит план короткой реплики в трейс: что ушло в модель и как обрезалось."""
    if trace is None or short is None:
        return
    plan = short.plan
    trace.short_strategy = plan.strategy
    trace.tts_target_text = plan.target_text
    trace.tts_context_text = plan.context_text
    trace.tts_synthesis_text = plan.synthesis_text
    trace.short_fallback = short.fallback
    trace.short_attempts = short.attempts
    if short.boundary is not None:
        trace.pre_guard_ms = boundary_module.PAD_BEFORE_SEC * 1000
        trace.post_guard_ms = boundary_module.PAD_AFTER_SEC * 1000


async def _synthesize_attempt(
    engine: SynthesisEngine,
    voice: Voice,
    replica: Replica,
    text: str,
    tuning: SpeakerSettings,
    settings: RenderSettings,
    position: str,
    label: str,
    short: ShortRun | None,
    attempt: int,
    trace: "synthesis_trace.ReplicaTrace | None" = None,
    reference: "reference_resolver.ResolvedReference | None" = None,
    seed: int | None = None,
    warmup: "warmup_context.WarmupContext | None" = None,
) -> tuple[np.ndarray, int | None]:
    """Одна попытка: синтез плана, обрезка до цели и откат, если границы нет.

    Порядок обязателен: обрезка идёт **до** подготовки куска, потому что
    подготовка выравнивает громкость, и с необрезанным контекстом уровень задавал
    бы он, а не целевая реплика.

    `seed` — принудительный сид для benchmark'а просодии; в production не задан,
    и каждая попытка берёт случайный (см. `_synthesize_chunk`).
    """
    # План применяется, пока не случился откат. Откат закрепляется: если границы
    # не нашлось, следующие попытки синтезируют цель сразу — повторять заведомо
    # бесполезный контекстный прогон незачем.
    # Цель в «плоском» виде: подготовленный текст, а если слой строился — тот, что
    # понимает движок (без «+» у движков без ударений). Обычный путь это не
    # меняет: без слоя текстом остаётся подготовленный `final_text`.
    plain_text = text
    if short is not None and short.active:
        plain_text = short.plan.engine_target_text or text
    use_plan = short is not None and short.active and not short.fallback
    synthesis_text = short.plan.synthesis_text if use_plan else plain_text
    # Прогрев: в модель уходит «скрытый префикс + цель», из аудио остаётся только
    # цель. Префикс готовится тем же путём, что и цель (`_text_for_engine`): у F5
    # это с ударениями, у XTTS — без «+»-разметки, по паспорту движка (§8).
    warmup_prefix = (warmup.prefix_text or "") if warmup is not None else ""
    if warmup_prefix:
        prepared_prefix = await asyncio.to_thread(
            _text_for_engine, warmup_prefix, engine, settings.auto_accent
        )
        if prepared_prefix.strip():
            warmup_prefix = prepared_prefix
            synthesis_text = f"{warmup_prefix.rstrip()} {plain_text.lstrip()}"
        else:
            # Префикс целиком состоял из служебных знаков — прогрева не будет.
            warmup_prefix = ""
    needs_crop = use_plan and short.plan.needs_crop
    chunk, seed = await _synthesize_chunk(
        engine,
        voice,
        synthesis_text,
        _nudge_tuning(voice, engine, tuning, attempt - 1),
        settings,
        position=position,
        label=label,
        source_text=replica.text,
        reference=reference,
        seed=seed,
    )
    if trace is not None:
        trace.seed = seed
        trace.stage(synthesis_trace.STAGE_RAW, chunk)
        _short_trace_fields(trace, short)
    if warmup_prefix and warmup is not None:
        # Границу цели ищем **на каждой попытке** заново: новый сид даёт другую
        # длительность префикса, и прошлый timestamp к ней не относится (§14).
        boundary = await asyncio.to_thread(
            warmup_context.locate_target_start,
            chunk,
            SAMPLE_RATE,
            synthesis_text,
            plain_text,
        )
        if boundary is None:
            # Fail-safe обязателен: сомнительный результат не используем вовсе, а
            # синтезируем цель заново без прогрева (§9). Повреждённый trim хуже,
            # чем отсутствие улучшения.
            warmup.fallback = warmup_context.REASON_ALIGNMENT
            logger.info("warmup fallback replica=%s reason=%s", position, warmup.fallback)
            chunk, seed = await _synthesize_chunk(
                engine,
                voice,
                plain_text,
                _nudge_tuning(voice, engine, tuning, attempt - 1),
                settings,
                position=position,
                label=label,
                source_text=replica.text,
                reference=reference,
                seed=seed,
            )
            if trace is not None:
                trace.seed = seed
                trace.warmup_used = True
                trace.warmup_fallback = warmup.fallback
                trace.warmup_prefix_chars = len(warmup.prefix_text or "")
                trace.stage(
                    synthesis_trace.STAGE_RAW,
                    chunk,
                    note="после отката прогрева: цель отдельно",
                )
                trace.stage(
                    synthesis_trace.STAGE_CONTEXT,
                    None,
                    note="граница цели не найдена — прогрев отброшен",
                )
            return chunk, seed
        preroll = int(SAMPLE_RATE * max(config.WARMUP_PREROLL_MS, 0) / 1000)
        start = max(0, int(boundary.start_sec * SAMPLE_RATE) - preroll)
        warmup.boundary_sec = boundary.start_sec
        logger.info(
            "warmup aligned replica=%s boundary=%.2fs confidence=%s words=%s/%s",
            position, boundary.start_sec, boundary.confidence,
            boundary.matched_words, boundary.total_words,
        )
        if trace is not None:
            trace.warmup_used = True
            trace.warmup_prefix_chars = len(warmup.prefix_text or "")
            trace.warmup_boundary_sec = boundary.start_sec
            trace.stage(
                synthesis_trace.STAGE_CONTEXT,
                chunk[start:],
                note=(
                    "прогрев отрезан по границе "
                    f"{boundary.method} (уверенность {boundary.confidence:.2f}, "
                    f"запас {config.WARMUP_PREROLL_MS:.0f} мс)"
                ),
            )
        return chunk[start:], seed

    if not needs_crop:
        if trace is not None:
            # Обрезки контекста не было — это факт, а не отсутствие данных.
            trace.stage(
                synthesis_trace.STAGE_CONTEXT,
                None,
                note="контекст не добавлялся" if not use_plan else "обрезка не нужна",
            )
        return chunk, seed

    # Границу ищем только выбранным методом. «По тишине» production-ready не
    # считается: она не подтверждает, что найден именно целевой текст, и обрезать
    # по ней значит рискнуть чужим куском в файле. Если ASR границу не подтвердил,
    # честнее откатиться на прежнее поведение и показать это в диагностике.
    cropped, boundary = await asyncio.to_thread(
        boundary_module.crop_to_target,
        chunk,
        short.plan.target_text,
        method=short.settings.boundary_method,
        side=short.plan.side,
    )
    if boundary is None:
        # Надёжной границы нет: лучше прежнее поведение, чем контекст в файле.
        short.fallback = "нет надёжной границы — синтез цели отдельно"
        logger.info(
            "Реплика %s: граница цели не найдена (%s) — синтезирую без контекста",
            position, short.plan.strategy,
        )
        chunk, seed = await _synthesize_chunk(
            engine,
            voice,
            plain_text,
            _nudge_tuning(voice, engine, tuning, attempt - 1),
            settings,
            position=position,
            label=label,
            source_text=replica.text,
            reference=reference,
            seed=seed,
        )
        if trace is not None:
            trace.seed = seed
            trace.short_fallback = short.fallback
            trace.stage(synthesis_trace.STAGE_RAW, chunk, note="после отката: цель отдельно")
            trace.stage(synthesis_trace.STAGE_CONTEXT, None, note="граница не найдена — откат")
        return chunk, seed

    short.boundary = boundary
    if trace is not None:
        trace.stage(
            synthesis_trace.STAGE_CONTEXT,
            cropped,
            note=(
                f"граница {boundary.method}, уверенность {boundary.confidence:.2f}, "
                f"слов {boundary.matched_words}/{boundary.total_words}"
            ),
        )
        _short_trace_fields(trace, short)
    return cropped, seed


async def _synthesize_checked(
    engine: SynthesisEngine,
    voice: Voice,
    replica: Replica,
    tuning: SpeakerSettings,
    settings: RenderSettings,
    position: str,
    label: str,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    context: su.ShortUtteranceContext | None = None,
    trace: "synthesis_trace.ReplicaTrace | None" = None,
    reference: "reference_resolver.ResolvedReference | None" = None,
    seed: int | None = None,
    warmup: "warmup_context.WarmupContext | None" = None,
) -> tuple[np.ndarray, int | None, QaOutcome | None, ShortRun | None]:
    """Синтез куска: обычный, с проверкой (`settings.qa`) и/или коротким слоем.

    Без проверок это ровно один вызов движка. Каждая дополнительная проверка
    добавляет к попытке полный синтез (и, в строгом режиме, отдельный процесс
    Whisper), поэтому цикл ограничен и числом попыток, и временем. Из попыток
    берётся лучшая: сначала по вердикту короткой реплики (меньше причин), затем по
    WER — выбросить уже сгенерированное аудио и вернуть ошибку значило бы потерять
    работу ради формальности.

    Короткий слой встроен в тот же цикл, а не в отдельный: у него общий бюджет
    времени, общие безопасные точки отмены и тот же выбор лучшей попытки. Разница
    только в том, что он добавляет свой критерий приёмки (§19–§21).

    Smart-режим отличается только входом в расшифровку: сначала кусок проходит
    дешёвый отбор по waveform, и если брака не видно, Whisper не поднимается
    вовсе. Отбор ошибается в сторону «подозрительный» — лишний Whisper дешевле
    пропущенного брака.

    Ошибки синтеза и таймауты идут наружу, как и раньше: проверка отвечает за
    качество готового куска, а не за то, чтобы прятать зависший инференс.
    """
    # Текст куска готовит `_text_for_engine`: числа и латиница для всех движков,
    # ударения — только для тех, кто понимает «+» (XTTS его прочитала бы вслух).
    # Но если реплика пришла из проанализированного проекта, текст уже подготовлен
    # и сохранён (`final_text`) — тогда он берётся как есть: пересчитывать его
    # значило бы отправить в модель не то, что пользователь видел и подтверждал.
    if should_abort and should_abort():
        raise JobCancelledError(f"Реплика {position}: отменено до синтеза")
    prepared = (replica.final_text or "").strip()
    if prepared:
        text = replica.final_text
    else:
        text = await asyncio.to_thread(
            _text_for_engine, replica.text, engine, settings.auto_accent
        )
    # Прогрев и контекст короткого слоя — два ответа на одну проблему: если
    # прогрев применён, контекстный план не строится вовсе. Иначе в модель ушли бы
    # и префикс, и соседняя реплика, а обрезать пришлось бы дважды.
    if warmup is not None and warmup.prefix_text:
        short = None
    else:
        short = short_run_for(replica, engine, settings, context=context)
    qa = settings.qa
    use_qa = qa is not None and qa.mode != config.QA_MODE_OFF
    if trace is not None:
        # Вход синтеза фиксируется до первой попытки: даже если движок упадёт, в
        # трейсе останется, что именно ему отправили.
        trace.prepared_from_final = bool(prepared)
        trace.word_count = count_words(text)
        trace.char_count = len(text)
        trace.utterance_class = (
            short.plan.utterance_class if short is not None else classify_utterance(text).kind
        )
        trace.short_strategy_requested = (
            settings.short_utterance.strategy if settings.short_utterance else ""
        )
        # Куски: реплика, помещающаяся в лимит движка, обязана уходить одним
        # текстом (§16). Пайплайн не режет текст сам — фиксируем это фактом, чтобы
        # регресс «короткая реплика внезапно разрезана» был виден в трейсе.
        trace.text_chunks = 1
        trace.chunk_boundaries = [0, len(text)]

    if short is None and not use_qa:
        chunk, seed = await _synthesize_attempt(
            engine, voice, replica, text, tuning, settings, position, label, None, 1, trace,
            reference, seed, warmup,
        )
        return chunk, seed, None, None

    started = time.monotonic()
    limit = max(
        qa.max_attempts if use_qa else 1,
        short.settings.max_attempts if short is not None else 1,
        1,
    )
    # Лучшая попытка: ключ, сырой кусок, измеренный (подготовленный) кусок, сид,
    # итог проверки. Подготовленный хранится рядом, чтобы финальная проверка и
    # трейс смотрели на то же аудио, по которому принималось решение.
    best: tuple[tuple, np.ndarray, np.ndarray, int | None, QaOutcome | None] | None = None
    attempts = 0
    status = QA_ATTEMPTS
    screening: qa_screening.Screening | None = None
    # План текущей попытки. Он может смениться между попытками: если DIRECT не
    # сказал слово, следующая попытка идёт с контекстом того же спикера (см.
    # `_escalate_short_plan`). Так «потерянное слово» лечится не перебором сидов, а
    # сменой стратегии — и только в пределах лимита попыток (§23).
    attempt_short = short
    for attempt in range(1, limit + 1):
        # Между попытками проверки — безопасная точка: здесь не убивается поток
        # внутри инференса, а просто не начинается следующая попытка.
        if attempt > 1:
            if should_abort and should_abort():
                raise JobCancelledError(f"Реплика {position}: отменено между попытками")
            if use_qa and time.monotonic() - started >= qa.budget_sec:
                status = QA_BUDGET
                break
            # Whisper на каждой попытке — это ещё один процесс и ещё ~1.6 ГБ
            # памяти: перед повтором ждём, пока система её отпустит.
            if wait_for_memory is not None:
                await wait_for_memory()
        if on_note:
            reason = "проверка" if use_qa else "короткая реплика"
            on_note(f"Реплика {position}: {reason}, попытка {attempt} из {limit}")

        chunk, seed = await _synthesize_attempt(
            engine, voice, replica, text, tuning, settings, position, label, attempt_short,
            attempt, trace, reference, seed, warmup,
        )
        attempts = attempt
        if attempt_short is not None:
            attempt_short.attempts = attempt
        # Проверка измеряет то, что попадёт в файл, а не сырой выход модели:
        # подготовка (обрезка краёв, кроссфейд, нормализация) стоит между ними, и
        # дефект, внесённый ею, иначе не видел бы никто — «QA прошёл, а слово
        # пропало». Подготовка детерминирована, поэтому вызывающий получит из
        # этого же сырого куска ровно тот же результат.
        measured = await asyncio.to_thread(
            _prepare_chunk, chunk, tuning, text=replica.final_text or replica.text
        )
        screening = None
        qa_outcome: QaOutcome | None = None
        transcription: str | None = None
        wer: float | None = None
        qa_accepted = True
        if use_qa and qa.smart:
            # Отбор идёт в потоке: на длинном куске это автокорреляция, и в
            # event loop ей делать нечего.
            screening = await asyncio.to_thread(
                qa_screening.screen_chunk, measured, replica.text
            )
            if not screening.suspicious:
                logger.info("Реплика %s: отбор не нашёл брака — расшифровка не нужна", position)
                qa_outcome = QaOutcome(
                    status=QA_PASSED, wer=None, attempts=attempt, mode=qa.mode, screening=screening
                )
            elif on_note:
                on_note(
                    f"Реплика {position}: отбор нашёл брак "
                    f"({qa_screening.describe_reasons(screening.reasons)}) — расшифровка"
                )
        if use_qa and qa_outcome is None:
            try:
                transcription, wer = await _transcribe_and_measure(measured, replica.text)
            except Exception as exc:  # noqa: BLE001 — упавшая расшифровка не губит задачу
                logger.warning(
                    "Реплика %s: проверка недоступна (%s: %s) — принимаю без проверки",
                    position, type(exc).__name__, exc,
                )
                qa_outcome = QaOutcome(
                    status=QA_UNAVAILABLE,
                    wer=None,
                    attempts=attempt,
                    mode=qa.mode,
                    screening=screening,
                )
            else:
                # «Проверить не удалось» — не провал: попытку не повторяем только
                # из-за недоступности Whisper, но короткий вердикт продолжит работу.
                qa_accepted = wer <= qa.wer_threshold
                qa_outcome = QaOutcome(
                    status=QA_PASSED if qa_accepted else QA_ATTEMPTS,
                    wer=wer,
                    attempts=attempt,
                    mode=qa.mode,
                    screening=screening,
                    transcription=transcription or "",
                )
                if on_note:
                    on_note(f"Реплика {position}: проверка {attempt} из {limit}, WER {wer:.2f}")

        verdict = None
        if attempt_short is not None:
            verdict = await asyncio.to_thread(
                su.check_chunk,
                measured,
                attempt_short.plan.target_text,
                transcription=transcription,
                wer=wer,
            )
            attempt_short.verdict = verdict
            if on_note and not verdict.ok:
                on_note(
                    f"Реплика {position}: короткая проверка — "
                    f"{su.describe_short_reasons(list(verdict.reasons))}"
                )

        accepted = qa_accepted and (verdict.ok if verdict is not None else True)
        if accepted:
            if qa_outcome is None:
                # Проверка не гонялась вовсе: короткий слой справился сам.
                qa_outcome = None
            logger.info(
                "Реплика %s: принята (попытка %s, стратегия %s)",
                position, attempt, attempt_short.plan.strategy if attempt_short else "direct",
            )
            return chunk, seed, qa_outcome, attempt_short
        if use_qa and qa_outcome.status == QA_UNAVAILABLE and verdict is None:
            # Нечем проверять — повторять нечего.
            return chunk, seed, qa_outcome, short

        if qa_outcome is not None and qa_outcome.status == QA_PASSED:
            # Порог WER взят, но короткая проверка нашла дефект: попытку считаем
            # неудачной, иначе «Да, да, да…» уезжало бы в файл с отметкой «ок».
            qa_outcome = QaOutcome(
                status=QA_ATTEMPTS,
                wer=wer,
                attempts=attempt,
                mode=qa.mode,
                screening=screening,
                transcription=transcription or "",
            )
        key = _attempt_score(verdict, wer)
        if best is None or key > best[0]:
            best = (key, chunk, measured, seed, qa_outcome, attempt_short)
        # Провал по словам — повод сменить стратегию, а не только сид: контекст
        # того же спикера возвращает модели просодию, которой не хватает
        # односложной фразе. Смена разовая и ограничена лимитом попыток.
        if verdict is not None and not verdict.ok and attempt < limit:
            escalated = _escalate_short_plan(
                attempt_short, replica, engine, settings, context
            )
            if escalated is not None:
                escalated.attempts = attempt
                attempt_short = escalated
                if on_note:
                    on_note(
                        f"Реплика {position}: слова не все — пробую "
                        f"{escalated.plan.strategy} с контекстом того же спикера"
                    )

    assert best is not None  # первая попытка выполняется всегда
    _, chunk, measured, seed, qa_outcome, short = best
    if short is not None:
        # Финальный вердикт — по подготовленному аудио лучшей попытки: именно оно
        # уходит в файл, и «слово потеряно» должно относиться к нему.
        short.verdict = su.check_chunk(measured, short.plan.target_text)
    # Итог цикла: «бюджет исчерпан» важнее статуса последней попытки, потому что
    # именно он объясняет, почему попыток больше не было. Число попыток — общее,
    # а не той, что оказалась лучшей.
    final_status = (
        status
        if status == QA_BUDGET
        else (qa_outcome.status if qa_outcome is not None else QA_ATTEMPTS)
    )
    final_outcome = (
        None
        if qa_outcome is None
        else QaOutcome(
            status=final_status,
            wer=qa_outcome.wer,
            attempts=attempts,
            mode=qa_outcome.mode,
            screening=qa_outcome.screening,
            transcription=qa_outcome.transcription,
        )
    )
    if trace is not None:
        # Итог проверки в трейс: статус, WER и то, что услышал Whisper. Ровно здесь
        # видно расхождение «QA прошёл по сырому куску» → «в файл пошёл
        # подготовленный» (см. §18 пакета UPDATE 2).
        trace.qa_status = final_status
        trace.qa_wer = None if final_outcome is None else final_outcome.wer
        if final_outcome is not None:
            trace.qa_transcript = final_outcome.transcription
        if short is not None and short.verdict is not None:
            verdict = short.verdict
            trace.repetition_detected = su.REASON_REPETITION in verdict.reasons
            trace.qa_transcript = trace.qa_transcript or verdict.transcription
            trace.first_word_ok = verdict.first_word_ok
            trace.last_word_ok = verdict.last_word_ok
            trace.expected_words = list(verdict.expected_words)
            trace.asr_words = list(verdict.asr_words)
    logger.info(
        "Реплика %s: проверка — %s, попыток %s, стратегия %s",
        position, final_status if final_outcome is not None else "не гонялась", attempts,
        short.plan.strategy if short else "direct",
    )
    return chunk, seed, final_outcome, short


def _escalate_short_plan(
    current: ShortRun | None,
    replica: Replica,
    engine: SynthesisEngine,
    settings: RenderSettings,
    context: su.ShortUtteranceContext | None,
) -> ShortRun | None:
    """Меняет план на контекстный, если текущий синтезировал цель «в вакууме».

    Смысл шага: у короткой фразы модели не хватает просодического контекста, и она
    может не договорить окончание. Контекст того же спикера возвращает этот
    контекст — но не сам по себе, а как **ответ на измеренный провал**: слова
    проверены по расшифровке, и только если их не хватает, стратегия меняется.
    Обратный порядок (сразу контекст) даёт более длинную и дорогую реплику без
    выигрыша — это измерено (см. `docs/dialogue.md`).
    """
    if current is None or context is None:
        return None
    if current.plan.needs_crop or current.fallback:
        # Уже с контекстом или уже откатились: второй раз то же самое не поможет.
        return None
    if not (context.same_speaker_previous or context.same_speaker_next):
        # Контекста **того же спикера** нет — эскалировать некуда. Кросс-спикерный
        # сосед здесь не годится: его нельзя прочитать текущим голосом (§27).
        return None
    if current.settings.strategy not in (su.STRATEGY_AUTO, su.STRATEGY_DIRECT):
        return None
    escalated_settings = replace(current.settings, strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    plan = short_run_for(replica, engine, replace(settings, short_utterance=escalated_settings),
                         context=context)
    if plan is None or not plan.plan.needs_crop:
        return None
    return plan


def _attempt_score(verdict: su.ShortVerdict | None, wer: float | None) -> tuple:
    """Ключ выбора лучшей попытки: сначала короткий вердикт, затем WER."""
    if verdict is None:
        return (0, 0, -(wer if wer is not None else 1.0))
    reasons, negative_wer = verdict.score
    return (1 if verdict.ok else 0, reasons, negative_wer)


async def _transcribe_and_measure(chunk: np.ndarray, expected: str) -> tuple[str, float]:
    """Расшифровка куска и WER — одной операцией, чтобы текст был под рукой.

    Короткая проверка хочет не только число, но и саму расшифровку («Да, да, да…»
    видно словами). Вернуть её из `_measure_wer` дешевле, чем запускать Whisper
    второй раз ради той же записи.
    """
    recognized = await _transcribe_chunk(chunk)
    if not recognized.strip():
        return recognized, 1.0  # модель промолчала — это провал проверки
    rules = active_rules()
    wer = word_error_rate(
        normalize(expected, pronunciation=rules, supports_accents=False),
        normalize(recognized, pronunciation=rules, supports_accents=False),
    )
    return recognized, wer


async def _measure_wer(chunk: np.ndarray, expected: str) -> float:
    """Считает WER куска относительно текста, который в него отправляли.

    Расшифровка проходит ту же предобработку (`normalize`), что и текст перед
    синтезом: Whisper пишет числа цифрами («2026»), а в модель ушло «две тысячи
    двадцать шестом» — без общего шага они расходились бы в каждом числе, и
    проверка валилась бы на ровном месте.

    Словарь произношения применяется к **обеим** сторонам. Модель произносит
    замену («SQL» → «эскьюэль»), и Whisper запишет именно её; если reference
    оставить без словаря, каждая такая реплика выглядела бы ошибкой — WER 0.75
    при пороге 0.2, лишние попытки и лишний запуск Whisper на ровном месте.
    Обратная сторона той же монеты: Whisper иногда пишет услышанное как «SQL»,
    и тогда словарь нужен уже расшифровке. Один шаг на оба текста закрывает оба
    случая — ровно так же, как `normalize` закрывает числа.
    """
    _, wer = await _transcribe_and_measure(chunk, expected)
    return wer


async def render_dialogue(
    job_id: str,
    replicas: Iterable[Replica],
    speakers: dict[str, SpeakerSettings],
    settings: RenderSettings,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    on_engine_loaded: EngineLoadedCallback | None = None,
    on_chunk_timing: ChunkTimingCallback | None = None,
    resume: RenderPartial | None = None,
    llm_hints: Mapping[int, Mapping[str, object]] | None = None,
) -> RenderResult:
    """Синтезирует реплики строго по одной и склеивает их в один файл.

    Блокирующие шаги (загрузка модели, ударения, инференс, запись файла) уводятся
    в отдельный поток, чтобы не встал event loop. Инференс ограничен
    `config.CHUNK_TIMEOUT_SEC`, а между кусками проверяется `should_abort`.

    Движок выбирается у каждого голоса отдельно, поэтому в одном файле спокойно
    соседствуют куски F5-TTS и XTTS: реплики идут строго последовательно, и
    склейка для неё одинакова — все движки отдают 24 кГц.

    `on_engine_loaded` и `on_chunk_timing` — фактические времена для статистики
    ETA; без них поведение прежнее (оба колбэка необязательны).

    `resume` — продолжение прерванного рендера (creash_report): готовые куски
    встают в начало сборки, синтез идёт с первой неготовой реплики. Это не
    «второй пайплайн»: тот же цикл, тот же движок, та же склейка и та же общая
    нормализация громкости — пропускается только уже сделанная работа.
    """
    replicas = list(replicas)
    if not replicas:
        raise ValueError("Диалог пуст")

    labels = {r.voice: r.label for r in replicas}
    unknown = sorted({r.voice for r in replicas if r.voice not in speakers})
    if unknown:
        names = ", ".join(f"«{labels[key]}»" for key in unknown)
        raise ValueError(f"Не назначен голос для: {names}")

    # Контракт рендера проекта: текст обязан быть подготовлен заранее. Проверка
    # стоит до загрузки движков — незачем поднимать модель на полторы минуты ради
    # запроса, который всё равно не имеет права синтезировать. И до цикла синтеза,
    # чтобы «посчитать текст на месте» не осталось незаметным обходом: при
    # `require_prepared` реплика без `final_text` — ошибка, а не повод готовить
    # текст заново (см. `_synthesize_checked`).
    if settings.require_prepared:
        unprepared = [
            str(index + 1)
            for index, replica in enumerate(replicas)
            if not (replica.final_text or "").strip()
        ]
        if unprepared:
            tail = " и другие" if len(unprepared) > 5 else ""
            raise ValueError(
                "Текст реплик не подготовлен: реплики "
                + ", ".join(unprepared[:5])
                + tail
                + ". Выполните анализ диалога и подтвердите найденные слова."
            )

    # Контекст коротких реплик строится один раз на весь диалог: он нужен
    # соседям, а значит известен только здесь. Исходные тексты при этом не
    # меняются — план живёт рядом с репликой, а не в ней (см. short_utterance).
    short_contexts: list[su.ShortUtteranceContext] = []
    if settings.short_utterance is not None and settings.short_utterance.enabled:
        short_contexts = await asyncio.to_thread(
            su.build_contexts,
            replicas,
            thresholds=settings.short_utterance.thresholds,
            # Подсказки LLM (класс реплики и релевантные соседи) — только
            # информация для контекста: ни текст, ни план синтеза они не меняют.
            llm_hints=llm_hints,
        )

    # Прогрев коротких реплик: префикс спрашивается у локальной LLM и живёт только
    # до синтеза. Строится лениво — по реплике перед её синтезом, а не пачкой:
    # прогрев нужен только коротким, а вызов LLM стоит времени (§18).
    warmup_enabled = (
        config.WARMUP_ENABLED if settings.warmup is None else bool(settings.warmup)
    )
    if warmup_enabled:
        logger.info(
            "warmup enabled job=%s replicas=%s engines=%s",
            job_id, len(replicas), ",".join(config.WARMUP_ENGINES),
        )

    resolved: dict[str, Voice] = {}
    per_replica: dict[int, SpeakerSettings] = {}
    for index, replica in enumerate(replicas):
        base = speakers[replica.voice]
        # Голос разрешается один раз на голос, а не на реплику: это чтение
        # `voices.json`, и на длинном диалоге оно заметно. Ключ — эффективный
        # голос: у реплик одного спикера голоса могут различаться.
        voice_id = str(replica.voice_id or base.voice_id)
        voice = resolved.get(voice_id)
        if voice is None:
            voice = _resolve_voice(replica.label, replace(base, voice_id=voice_id))
            resolved[voice_id] = voice
        per_replica[index] = tuning_for(voice, base, replica.overrides)
    engines = await _load_engines(list(resolved.values()), on_loaded=on_engine_loaded)

    references: dict[int, dict] = {}
    warmups: dict[int, dict] = {}
    pieces: list[np.ndarray] = list(resume.pieces) if resume else []
    segments: list[tuple[int, int]] = list(resume.segments) if resume else []
    seeds: list[int | None] = list(resume.seeds) if resume else []
    checks: list[QaOutcome | None] = list(resume.qa) if resume else []
    qualities: list[take_quality.TakeQuality] = list(resume.qualities) if resume else []
    # План короткой реплики по индексу: он уходит в метаданные куска (§24).
    short_runs: dict[int, ShortRun] = dict(resume.short_runs) if resume else {}
    total = len(replicas)
    cursor = segments[-1][1] if segments else 0
    # Сколько реплик уже готово: их не синтезируем заново — в этом смысл
    # продолжения. Проверка «не больше, чем реплик» стоит здесь, а не у
    # вызывающего: продолжение с чужим набором реплик дало бы файл, в котором
    # куски не соответствуют тексту.
    done_before = len(pieces)
    if done_before > total:
        raise ValueError(
            f"Продолжение рендера противоречит диалогу: готово {done_before} кусков, "
            f"а реплик {total}"
        )
    # Индекс реплики, на которой прервалось: нужен очереди, чтобы пометить
    # конкретную реплику `interrupted` и назвать её в диагностике.
    current_index: int | None = None

    # Трейс стадий (§14–§15 UPDATE 2). Он включается переменной окружения и по
    # умолчанию выключен: без него `trace is None`, и ни одна ветка ниже не
    # выполняется. Включённый — записывает аудио стадий и числа обрезки, чтобы
    # «слово пропало» можно было разделить на «модель не сказала» и «пайплайн срезал».
    trace = synthesis_trace.start(job_id)

    try:
        for index, replica in enumerate(replicas, start=1):
            if index <= done_before:
                continue
            current_index = index - 1
            if should_abort and should_abort():
                # Отмена пользователя и прерывание watchdog'ом приходят одним
                # сигналом: различить их — дело вызывающего (job_queue), а пайплайн
                # обязан просто остановиться между кусками, не убивая поток внутри
                # вызова модели.
                raise JobCancelledError(f"Прервано на реплике {index} из {total}")
            if on_progress:
                on_progress(index, total, replica.label)
            settings_for_replica = per_replica[index - 1]
            voice = resolved[settings_for_replica.voice_id]
            engine = engines[voice.engine]
            # Референс выбирается под действующую просодию реплики одним резолвером
            # (UPDATE 3 §24, §51): `override → рекомендация модели → эмоция`.
            # Раньше здесь стояла эмоция, и рекомендованный профиль (`SURPRISE →
            # EXCLAMATION`) до движка не доезжал — LLM называла интонацию, а
            # синтез получал референс прежней эмоции. Гарантия «только свой голос»
            # структурная — профили лежат внутри голоса.
            reference = _reference_for(voice, engine, replica)
            references[index - 1] = reference.to_dict() if reference is not None else None
            warmup: warmup_context.WarmupContext | None = None
            if warmup_enabled:
                warmup = await asyncio.to_thread(
                    warmup_context.get_service().build,
                    target_text=replica.final_text or replica.text,
                    engine_id=engine.id,
                    previous_text=(replicas[index - 2].final_text or replicas[index - 2].text)
                    if index >= 2
                    else None,
                    next_text=(replicas[index].final_text or replicas[index].text)
                    if index < total
                    else None,
                    speaker=replica.label,
                )
                if warmup.prefix_text:
                    logger.info(
                        "warmup used replica=%s target_chars=%s prefix_chars=%s cache_hit=%s",
                        index, len(replica.final_text or replica.text),
                        len(warmup.prefix_text), warmup.cache_hit,
                    )
            replica_trace = None
            if trace is not None:
                replica_trace = trace.replica(
                    index,
                    project_id=getattr(settings, "project_id", "") or "",
                    speaker=replica.voice,
                    voice_id=voice.id,
                    engine=engine.id,
                    label=replica.label,
                    position=f"{index} из {total}",
                    source_text=replica.text or "",
                    final_text=replica.final_text or "",
                    reference_audio=(
                        Path(reference.audio_path).name if reference is not None else ""
                    ),
                    reference_text="" if reference is None else (reference.ref_text or ""),
                )
                replica_trace.emotion_detected = replica.emotion_detected
                replica_trace.emotion_override = replica.emotion_override
                replica_trace.emotion_effective = replica.emotion_effective
                # Интонация: рекомендация модели и то, что дошло до резолвера.
                # Второе может отличаться от первого — при ручном выборе (§37).
                replica_trace.prosody_profile = getattr(replica, "prosody_profile", "")
                replica_trace.prosody_effective = replica.prosody_effective
                replica_trace.prosody_confidence = getattr(replica, "prosody_confidence", None)
                replica_trace.prosody_intensity = getattr(replica, "prosody_intensity", None)
                replica_trace.prosody_pace = getattr(replica, "prosody_pace", "")
                replica_trace.dialogue_act = getattr(replica, "dialogue_act", "")
                replica_trace.context_dependency = getattr(replica, "context_dependency", "")
                if reference is not None:
                    replica_trace.reference_profile_id = reference.profile_id
                    # Ключ профиля в терминах §55 — то же значение, что эмоция профиля.
                    replica_trace.reference_profile_key = reference.resolved_emotion
                    replica_trace.reference_requested_profile = reference.requested_emotion
                    replica_trace.reference_resolved_profile = reference.resolved_emotion
                    replica_trace.reference_emotion = reference.resolved_emotion
                    replica_trace.reference_fallback_used = reference.fallback_used
                    replica_trace.reference_fallback_reason = reference.fallback_reason

            started = time.monotonic()
            chunk, seed, qa_outcome, short_run = await _synthesize_checked(
                engine,
                voice,
                replica,
                settings_for_replica,
                settings,
                position=f"{index} из {total}",
                label=replica.label,
                wait_for_memory=wait_for_memory,
                on_note=on_note,
                context=short_contexts[index - 1] if short_contexts else None,
                trace=replica_trace,
                reference=reference,
                warmup=warmup,
            )
            if warmup is not None:
                # Паспорт прогрева собирается **после** синтеза: граница и признак
                # отката появляются в ходе попытки, и снимок до неё показывал бы
                # «прогрев применён, отказов нет» даже когда границы не нашлось.
                warmups[index - 1] = warmup.to_dict()
            if short_run is not None:
                short_runs[index - 1] = short_run
            seeds.append(seed)
            checks.append(qa_outcome)
            pause = _pause_samples(settings_for_replica, settings)
            if pieces and pause:
                pieces.append(np.zeros(pause, dtype=np.float32))
                cursor += pause
            prepared = await asyncio.to_thread(
                _prepare_chunk,
                chunk,
                settings_for_replica,
                trace=replica_trace,
                text=replica.final_text or replica.text,
            )
            if replica_trace is not None:
                replica_trace.speed = settings_for_replica.speed
                replica_trace.engine_params = dict(settings_for_replica.engine_params or {})
                replica_trace.seconds = time.monotonic() - started
                # Аудио стадий уже сохранено вызовами `stage()` — здесь только
                # запись итога на диск, чтобы падение на следующей реплике не
                # потеряло уже собранные данные.
                trace.write(replica_trace)
            # Диагностика считается по подготовленному куску — по тому, что реально
            # попадёт в файл, а не по сырому выходу модели.
            qualities.append(
                await asyncio.to_thread(
                    take_quality.measure, prepared, replica.text, qa=qa_outcome, raw=chunk
                )
            )
            segments.append((cursor, cursor + prepared.size))
            cursor += prepared.size
            pieces.append(prepared)
            # Время уходит наружу после подготовки куска: это ровно тот интервал,
            # который прожил пользователь, включая подготовку, и он же ляжет в
            # статистику — иначе оценка систематически недосчитывала бы кусок.
            if on_chunk_timing:
                on_chunk_timing(
                    index - 1,
                    time.monotonic() - started,
                    prepared.size / SAMPLE_RATE,
                    qa_outcome,
                )
    except BaseException as exc:
        # Готовое не теряем: состояние уходит наверх вместе с исключением
        # (creash_report). `current_index` обнуляется ниже, когда цикл дошёл до
        # конца, — иначе сбой на сборке пометил бы `interrupted` последнюю
        # реплику, аудио которой уже синтезировано.
        exc.partial_render = RenderPartial(
            pieces=pieces,
            segments=segments,
            seeds=seeds,
            qa=checks,
            qualities=qualities,
            short_runs=short_runs,
            failed_index=current_index,
            error=str(exc),
        )
        raise
    current_index = None

    final = await asyncio.to_thread(np.concatenate, pieces)
    final = await asyncio.to_thread(_finalize_track, final)
    if trace is not None:
        # Сводка — после успешной сборки: по ней видно, у каких реплик снимались
        # края и у каких ASR не услышал первое/последнее слово.
        summary_path = await asyncio.to_thread(trace.write_summary)
        logger.info("Трейс синтеза: %s", summary_path)
    output_path = await asyncio.to_thread(
        _write_output, job_id, final, settings.output_format, settings.output_name
    )
    return RenderResult(
        output_path=output_path,
        duration_sec=len(final) / SAMPLE_RATE,
        replicas_done=total,
        segments=segments,
        seeds=seeds,
        qa=checks,
        qualities=qualities,
        short_runs=short_runs,
        references=references,
        warmups=warmups,
    )


async def synthesize_replica(
    replica: Replica,
    speaker: SpeakerSettings,
    settings: RenderSettings,
    index: int,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    context: su.ShortUtteranceContext | None = None,
) -> tuple[np.ndarray, int | None, QaOutcome | None, take_quality.TakeQuality, ShortRun | None]:
    """Синтезирует и готовит один кусок — без записи в готовый файл.

    Возвращает подготовленный кусок, сид, которым он получен, итог строгой
    проверки (`None` — проверка выключена) и диагностику take'а: всё это нужно
    вызывающему, чтобы сохранить вариант и уже его поставить в файл.
    Диагностика считается здесь, а не у вызывающего: перегруз модели виден только
    на сыром куске, который наружу не отдаётся. Пересобирать весь диалог из-за
    одной неудачной реплики — десятки секунд на каждый кусок, поэтому
    перегенерация трогает ровно один кусок.
    """
    # Параметры из маркера в тексте и правок карточки важнее карточки спикера —
    # ровно как при первой сборке, иначе перегенерация дала бы кусок, несовместимый
    # с соседями по звучанию. Голос разрешается здесь же: `tuning_for` строит
    # результат, а не слой, и по нему уже не восстановить, чей это был `voice_id`.
    voice = _resolve_voice(
        replica.label, replace(speaker, voice_id=str(replica.voice_id or speaker.voice_id))
    )
    resolved = tuning_for(voice, speaker, replica.overrides)
    engine = _engine_for(voice)
    await asyncio.to_thread(engine.load)
    reference = _reference_for(voice, engine, replica)

    chunk, seed, qa_outcome, short_run = await _synthesize_checked(
        engine,
        voice,
        replica,
        resolved,
        settings,
        position=str(index + 1),
        label=replica.label,
        wait_for_memory=wait_for_memory,
        on_note=on_note,
        should_abort=should_abort,
        context=context,
        reference=reference,
    )
    prepared = await asyncio.to_thread(
        _prepare_chunk, chunk, resolved, text=replica.final_text or replica.text
    )
    # Диагностика считается здесь, где под рукой и сырой выход модели, и
    # подготовленный кусок: перегруз подготовки не виден, и перемерить его
    # снаружи уже нечем (см. `take_quality.measure`).
    quality = await asyncio.to_thread(
        take_quality.measure, prepared, replica.text, qa=qa_outcome, raw=chunk
    )
    logger.info(
        "Реплика %s (%s) пересинтезирована: %.2f c аудио, сид %s",
        index + 1, replica.label, prepared.size / SAMPLE_RATE, seed,
    )
    return prepared, seed, qa_outcome, quality, short_run, reference


async def synthesize_take(
    engine: SynthesisEngine,
    voice: Voice,
    text: str,
    tuning: SpeakerSettings,
    settings: RenderSettings,
    *,
    label: str,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    reference: "reference_resolver.ResolvedReference | None" = None,
    seed: int | None = None,
) -> tuple[np.ndarray, int | None, QaOutcome | None]:
    """Синтезирует одну фразу вне диалога — тем же путём, что и реплику.

    Нужно сравнению движков: одну и ту же фразу и один и тот же reference читают
    разные модели, и путь до них обязан совпадать с боевым (preprocessing текста →
    ограничение инференса по времени → QA-цикл). Иначе замеренное время описывало
    бы не то, что происходит при обычной генерации.

    `reference` и `seed` — для benchmark'а reference-профилей (§30–§32): он
    сравнивает один и тот же текст и голос, меняя **только** профиль, и обязан
    зафиксировать сид. Без них берутся основной референс голоса и случайный сид,
    как раньше.
    """
    replica = Replica(voice=tuning.voice_id or voice.id, text=text, line_number=1)
    chunk, seed, qa_outcome, _short_run = await _synthesize_checked(
        engine,
        voice,
        replica,
        tuning,
        settings,
        position=label,
        label=label,
        wait_for_memory=wait_for_memory,
        on_note=on_note,
        should_abort=should_abort,
        reference=reference,
        seed=seed,
    )
    return chunk, seed, qa_outcome


def write_take(
    path: Path, chunk: np.ndarray, tuning: SpeakerSettings, text: str = ""
) -> float:
    """Записывает отдельный take wav и возвращает его длительность.

    Take готовится так же, как кусок трека (края, тембр, RMS, затем LUFS и
    лимитер): сравнение движков слушают на слух, и без выравнивания громкости
    выбор решала бы громкость, а не голос.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_chunk(chunk, tuning, text=text)
    final = _finalize_track(prepared)
    _write_audio(path, final, "wav")
    return final.size / SAMPLE_RATE


def variant_path(job_id: str, index: int, variant_id: str) -> Path:
    """Файл варианта куска.

    Отдельный от `output/{job_id}.{fmt}`, а не «кусок внутри него»: варианты
    живут своей жизнью — их прослушивают и сравнивают между собой, тогда как в
    готовый файл попадает только один из них.
    """
    return config.OUTPUT_DIR / f"{job_id}-r{index + 1}-{variant_id}.wav"


def save_chunk_variant(
    job_id: str,
    index: int,
    variant_id: str,
    chunk: np.ndarray,
    seed: int | None,
    label: str,
    qa: QaOutcome | None = None,
    quality: take_quality.TakeQuality | None = None,
    plan: dict | None = None,
) -> ChunkVariant:
    """Сохраняет кусок вариантом: файл для прослушивания + сид в карточке.

    Вариант финализируется здесь же (LUFS + лимитер). Куски в готовом файле
    нормализованы вместе, а вырезанный из файла кусок лежит на другой громкости,
    чем только что сгенерированный, — и сравнение двух вариантов на слух
    превращалось бы в сравнение громкости.
    """
    path = variant_path(job_id, index, variant_id)
    _write_audio(path, _finalize_track(np.asarray(chunk, dtype=np.float32)), "wav")
    return ChunkVariant(
        id=variant_id,
        path=path,
        label=label,
        seed=seed,
        duration_sec=chunk.size / SAMPLE_RATE,
        qa=qa,
        quality=quality,
        plan=plan,
    )


def read_variant(variant: ChunkVariant) -> np.ndarray:
    return _read_audio(variant.path, "wav")


def drop_variant(variant: ChunkVariant) -> None:
    """Удаляет файл вытесненного варианта: держать его больше не для чего."""
    variant.path.unlink(missing_ok=True)


def read_segment(source_path: Path, output_format: str, bounds: tuple[int, int]) -> np.ndarray:
    """Вырезает текущий кусок из готового файла.

    Нужно, чтобы до первой замены сохранить исходный вариант: иначе
    «перегенерировать» означает «потерять то, что было», и сравнивать новое
    будет не с чем.
    """
    audio = _read_audio(source_path, output_format)
    start, end = bounds
    return audio[start:end]


def export_project_chunks(
    project_id: str, source_path: Path, output_format: str, segments: list[tuple[int, int]]
) -> list[Path]:
    """Сохраняет куски готового файла отдельными wav — по одному на реплику.

    Проект хранит варианты отдельными файлами, а не вырезками из итогового трека:
    замена одной реплики не должна менять остальные, а прослушать реплику до
    пересборки всего файла иначе нечем. Имя уникальное: прошлые рендеры остаются
    доступными как отдельные варианты, и перезапись их файлов ломала бы историю.
    """
    directory = config.PROJECTS_OUTPUT_DIR / project_id
    directory.mkdir(parents=True, exist_ok=True)
    audio = _read_audio(source_path, output_format)
    paths: list[Path] = []
    for index, (start, end) in enumerate(segments, start=1):
        target = directory / f"r{index}-{uuid.uuid4().hex[:8]}.wav"
        _write_audio(target, np.asarray(audio[start:end], dtype=np.float32), "wav")
        paths.append(target)
    return paths


def export_partial_takes(project_id: str, pieces: Iterable[np.ndarray]) -> list[Path]:
    """Сохраняет куски **прерванного** рендера отдельными wav.

    Трека ещё нет, поэтому куски лежат в памяти подготовленными, но не
    финализированными вместе. Каждый финализируется сам по себе — ровно как
    вариант куска (см. `save_chunk_variant`): без этого сохранённое звучало бы
    тише готового файла, а сравнение с прежними вариантами превращалось бы в
    сравнение громкости.

    Имена отличаются от кусков обычного рендера (`r-part-…`): по ним видно, что
    это остаток прерванной сборки, и они не путаются с вырезками готового файла.
    """
    directory = config.PROJECTS_OUTPUT_DIR / project_id
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for piece in pieces:
        target = directory / f"r-part-{uuid.uuid4().hex[:8]}.wav"
        _write_audio(target, _finalize_track(np.asarray(piece, dtype=np.float32)), "wav")
        paths.append(target)
    return paths


def voice_engine(voice_id: str) -> str:
    """Движок голоса для карточки куска; пустая строка — голос не найден."""
    if not voice_id:
        return ""
    voice = get_store().get(voice_id)
    return voice.engine if voice is not None else ""


def tuning_parameters(tuning: SpeakerSettings) -> dict:
    """Эффективные параметры готового резолва — то, чем получен кусок.

    Отдельно от `chunk_parameters`: там параметры считаются по реплике и слоту
    (наследование), а здесь принимается уже посчитанный результат. Сравнению
    движков нужен именно второй — оно синтезирует не реплику, а тестовую фразу,
    и записывает рядом с результатом фактические ручки своего движка.
    """
    return {
        "speed": tuning.speed,
        "cfg_strength": tuning.cfg_strength,
        "nfe_step": tuning.nfe_step,
        "target_rms": tuning.target_rms,
        "gain_db": tuning.gain_db,
        "pitch_semitones": tuning.pitch_semitones,
        "pause_override_ms": tuning.pause_override_ms,
        "engine_params": dict(tuning.engine_params),
    }


def chunk_parameters(replica: Replica, speaker: SpeakerSettings) -> dict:
    """Эффективные параметры куска — то, чем он реально получен.

    Хранится рядом с вариантом, чтобы карточку можно было воспроизвести: у
    реплики свои overrides поверх карточки спикера, и «какие настройки были»
    задним числом иначе не восстановить.
    """
    return tuning_parameters(_settings_for(replica, speaker))


async def apply_variant(
    source_path: Path,
    settings: RenderSettings,
    segments: list[tuple[int, int]],
    index: int,
    variant: ChunkVariant,
) -> tuple[float, tuple[int, int]]:
    """Ставит сохранённый вариант в готовый файл.

    Возвращает длительность файла и новые границы куска. Единая точка и для
    «сгенерировать заново», и для «оставить другой вариант»: обе операции должны
    давать ровно один и тот же файл, иначе выбор варианта в интерфейсе менял бы
    звук не так, как та же сгенерированная реплика.
    """
    content = await asyncio.to_thread(read_variant, variant)
    audio = await asyncio.to_thread(_read_audio, source_path, settings.output_format)
    start, end = segments[index]
    updated = np.concatenate((audio[:start], content, audio[end:]))
    updated = await asyncio.to_thread(_finalize_track, updated)
    await asyncio.to_thread(_write_audio, source_path, updated, settings.output_format)
    return len(updated) / SAMPLE_RATE, (start, start + content.size)


def _read_audio(path: Path, output_format: str) -> np.ndarray:
    """Читает готовый файл обратно в float32-моно при SAMPLE_RATE.

    Приведение к внутреннему формату — не косметика: файл может прийти извне
    (браузерная запись в WebM/Opus почти всегда 48 кГц и часто стерео), и без
    ресемплинга со сведением в моно его длительность, темп и расстановка пауз
    уехали бы ровно во столько раз, во сколько отличается частота. Свои файлы
    проект всегда пишет моно при `SAMPLE_RATE`, поэтому для них это no-op.
    """
    if (output_format or "wav").lower() == "wav":
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    else:
        # Формат декодирует ffmpeg с явным `s16le`, а не pydub: pydub отдаёт сырые
        # сэмплы в том формате, который выбрал ffmpeg для вывода, и разрядность
        # приходится угадывать по ширине сэмпла. Для opus — а это любая запись из
        # браузера (webm/ogg) — ffmpeg выдаёт 32-битный float, и разбор его как
        # int16 давал вдвое больше сэмплов: девятисекундная реплика превращалась в
        # девятнадцать секунд шума. Именно поэтому дубли во вкладке записи звучали
        # иначе, чем браузерное превью той же записи (§24).
        from .audio_analysis import load_mono

        audio, _ = load_mono(path, output_format, target_sr=SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        # Сведение усреднением, а не выбором первого канала: иначе при записи в
        # стерео терялась бы половина сигнала (например, второй микрофон).
        audio = audio.mean(axis=1, dtype=np.float32)
    else:
        audio = audio.reshape(-1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if sample_rate and int(sample_rate) != int(SAMPLE_RATE):
        import librosa

        try:
            audio = librosa.resample(
                audio,
                orig_sr=int(sample_rate),
                target_sr=int(SAMPLE_RATE),
                res_type="soxr_hq",
            )
        except ImportError:  # soxr необязателен — kaiser_best это чистый scipy
            audio = librosa.resample(
                audio,
                orig_sr=int(sample_rate),
                target_sr=int(SAMPLE_RATE),
                res_type="kaiser_best",
            )
        audio = np.ascontiguousarray(audio, dtype=np.float32)
    return audio


def _write_audio(target: Path, audio: np.ndarray, output_format: str) -> None:
    """Пишет аудио атомарно: временный файл → проверка → переименование.

    Прямая запись в целевой файл оставляла бы «половину» при падении процесса или
    нехватке места, и эта половина выглядела бы готовым take'ом: по имени файла
    недописанное от целого не отличить, а плеер на нём либо молчит, либо падает.
    Поэтому результат сначала пишется рядом под именем `<имя>.part`, проверяется
    на читаемость и ненулевую длительность (см. `validate_audio_file`) и только
    потом занимает своё имя: `os.replace` в пределах каталога атомарен.
    """
    import soundfile as sf

    output_format = (output_format or "wav").lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    try:
        if output_format == "wav":
            # Формат задаётся явно: по расширению `.part` soundfile его не узнает.
            sf.write(partial, audio, SAMPLE_RATE, format="WAV")
            validate_audio_file(partial, expect_sample_rate=SAMPLE_RATE)
        else:
            tmp_wav = target.with_name(target.name + PARTIAL_SUFFIX + ".wav")
            try:
                sf.write(tmp_wav, audio, SAMPLE_RATE, format="WAV")
                validate_audio_file(tmp_wav, expect_sample_rate=SAMPLE_RATE)
                from pydub import AudioSegment

                AudioSegment.from_wav(tmp_wav).export(
                    partial, format=output_format, bitrate="192k"
                )
            finally:
                tmp_wav.unlink(missing_ok=True)
            # Частота у сжатого формата своя (mp3 может ресемплировать) — здесь
            # проверяется читаемость и непустая длительность, а не 24 кГц.
            validate_audio_file(partial, expect_sample_rate=None)
        os.replace(partial, target)
    except BaseException:
        # Недописанное не оставляем: иначе оно копится в каталогах и путается с
        # готовыми файлами (остатки от прошлых запусков убирает `cleanup_partials`).
        partial.unlink(missing_ok=True)
        raise


def validate_audio_file(path: Path, *, expect_sample_rate: int | None = SAMPLE_RATE) -> dict:
    """Проверяет, что файл — читаемое аудио ненулевой длительности.

    «Файл существует» и «файл слушается» — разные вещи: нулевой или обрезанный
    WAV появляется при нехватке места и падении процесса, и записывать такой
    результат готовым take'ом нельзя. Возвращает метаданные (частота, кадры,
    длительность) — они же уходят в диагностику; ошибка — `AudioFileError`.
    """
    if not path.exists():
        raise AudioFileError(f"файл не создан: {path.name}")
    if (path.suffix or "").lower() == ".wav":
        import soundfile as sf

        try:
            info = sf.info(path)
        except Exception as exc:  # причина не важна: любая значит «не читается»
            raise AudioFileError(f"файл не читается как аудио: {path.name} ({exc})") from exc
        if info.frames <= 0:
            raise AudioFileError(f"в файле нет звука: {path.name}")
        if expect_sample_rate is not None and info.samplerate != expect_sample_rate:
            raise AudioFileError(
                f"частота дискретизации {info.samplerate} Гц вместо {expect_sample_rate}: {path.name}"
            )
        # Метаданные не доказывают, что файл читается: проверяется декодирование
        # первого и последнего кадра. Это два обращения к диску вместо чтения
        # всего трека (он бывает в сотни мегабайт) и ловит файлы, у которых
        # заголовок обещает данные, которых в файле нет.
        try:
            with sf.SoundFile(path) as handle:
                handle.seek(0)
                handle.read(1)
                handle.seek(max(0, int(info.frames) - 1))
                tail = handle.read(1)
        except Exception as exc:
            raise AudioFileError(f"файл обрезан или повреждён: {path.name} ({exc})") from exc
        if np.asarray(tail).size == 0:
            # Заголовок обещал кадры, которых в файле нет: так выглядит обрыв
            # записи, если файл успел получить длину до данных.
            raise AudioFileError(f"файл обрезан: в нём нет последнего кадра ({path.name})")
        return {
            "sample_rate": info.samplerate,
            "frames": int(info.frames),
            "duration_sec": round(info.frames / info.samplerate, 4),
            "format": str(info.format),
        }

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(path)
    except Exception as exc:  # см. выше: причина не важна
        raise AudioFileError(f"файл не читается как аудио: {path.name} ({exc})") from exc
    if len(segment) <= 0:
        raise AudioFileError(f"в файле нет звука: {path.name}")
    if expect_sample_rate is not None and segment.frame_rate != expect_sample_rate:
        raise AudioFileError(
            f"частота дискретизации {segment.frame_rate} Гц вместо {expect_sample_rate}: {path.name}"
        )
    return {
        "sample_rate": segment.frame_rate,
        "frames": len(segment),
        "duration_sec": round(len(segment) / 1000, 4),
        "format": path.suffix.lstrip(".").lower() or "unknown",
    }


# Предел имени файла: длинное имя ломает не запись (её лимит больше), а скачивание
# в браузере и просмотр каталога. Обрезаем по символам, а не по байтам: кириллица
# в UTF-8 занимает по два байта, и «40 символов» иначе превратились бы в 20.
OUTPUT_NAME_MAX_CHARS = 60
# Форматы, которые пользователь может написать в имени по привычке: «диалог.wav»
# при формате wav не должно дать «диалог.wav.wav».
_KNOWN_AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".flac")


def safe_output_name(name: str) -> str:
    """Имя файла из пользовательской строки — без пути и запрещённых символов.

    Имя приходит из интерфейса и подставляется в путь, поэтому из него убирается
    всё, что может увести запись за пределы `output/` (разделители, `..`), и всё,
    что запрещено в именах файлов. Кириллица сохраняется: имя файла выбирает
    пользователь, и транслит здесь был бы неожиданностью, а не безопасностью.
    Возвращается основа без расширения; пустая строка — имени нет.
    """
    cleaned = "".join(
        char if (char.isalnum() or char in " -_().") else "_" for char in str(name or "")
    )
    # Точки уходят вместе с остальным: расширение подставляет пайплайн, а точка в
    # имени — это либо попытка задать расширение, либо скрытый файл.
    cleaned = cleaned.strip()
    cleaned = "_".join(cleaned.split())
    # Повторы подчёркиваний схлопываются: «отчёт: 2025!» — это одно имя, а не
    # «отчёт__2025». Так имя остаётся читаемым и не зависит от того, сколько
    # недопустимых символов стояло рядом.
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_-. ")
    cleaned = cleaned[:OUTPUT_NAME_MAX_CHARS].strip("_-. ")
    for suffix in _KNOWN_AUDIO_SUFFIXES:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip("_-. ")
    return cleaned


def _output_target(job_id: str, output_format: str, output_name: str = "") -> Path:
    """Путь готового файла: имя пользователя или id задачи.

    Имя пользователя не перезаписывает чужой файл: два диалога с именем «Тест»
    должны лежать рядом, а не затирать друг друга. Поэтому занятое имя получает
    числовой хвост, а не отказ. Имя из id задачи остаётся как было — оно
    уникально по построению, и повторная запись того же файла ожидаема.
    """
    stem = safe_output_name(output_name)
    if not stem:
        return config.OUTPUT_DIR / f"{job_id}.{output_format}"
    target = config.OUTPUT_DIR / f"{stem}.{output_format}"
    if target.exists():
        for number in range(2, 100):
            candidate = config.OUTPUT_DIR / f"{stem}-{number}.{output_format}"
            if not candidate.exists():
                logger.info("Имя «%s» занято — файл записан как %s", stem, candidate.name)
                return candidate
        return config.OUTPUT_DIR / f"{stem}-{uuid.uuid4().hex[:6]}.{output_format}"
    return target


def _write_output(
    job_id: str, audio: np.ndarray, output_format: str, output_name: str = ""
) -> Path:
    """Пишет готовый файл задачи и проверяет его перед объявлением готовности.

    Проверка здесь дублирует ту, что уже сделана в `_write_audio` (иначе файл не
    получил бы своё имя), и стоит намеренно: контракт «DONE значит файл звучит»
    должен проверяться на итоговом имени, а не выводиться из внутренностей
    записи — цена проверки одна операция чтения метаданных.
    """
    output_format = (output_format or "wav").lower()
    target = _output_target(job_id, output_format, output_name)
    expect = SAMPLE_RATE if output_format == "wav" else None
    _write_audio(target, audio, output_format)
    validate_audio_file(target, expect_sample_rate=expect)
    return target


def cleanup_output(ttl_hours: float = config.OUTPUT_TTL_HOURS) -> int:
    """Удаляет готовые файлы старше ttl_hours. Возвращает число удалённых.

    Подкаталоги не трогаются: в `output/projects/` лежат куски проектов, а в
    `output/benchmarks/` — результаты сравнения движков, и то и другое должно
    переживать очистку, пока пользователь с ним работает. TTL рассчитан на
    транзитные файлы задач, которые всегда лежат в корне `output/`.
    """
    if ttl_hours <= 0:
        return 0
    threshold = time.time() - ttl_hours * 3600
    removed = 0
    for path in config.OUTPUT_DIR.iterdir():
        if path.is_dir() or path.name == ".gitkeep":
            continue
        try:
            if path.stat().st_mtime < threshold:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("Не удалось удалить %s: %s", path, exc)
    if removed:
        logger.info("Очистил %s старых файлов из output/", removed)
    return removed


def abort_running_inference(reason: str = "остановка") -> list[str]:
    """Просит движки, умеющие это, прервать текущий инференс.

    Нужно при выключении приложения: зависший кусок нельзя прервать извне, и без
    этого остановка ждала бы его таймаута (`TTS_CHUNK_TIMEOUT_SEC`), а поток
    оставался бы в нативном вызове. Результат куска всё равно выбрасывается —
    задача отменена. Движки в процессе бэкенда хука не имеют (`getattr`), поэтому
    вызов безопасен и для них.
    """
    aborted: list[str] = []
    for engine_id, engine in created_engines().items():
        abort = getattr(engine, "abort_current", None)
        if not callable(abort):
            continue
        try:
            if abort(reason):
                aborted.append(engine_id)
        except Exception as exc:  # noqa: BLE001 — выключение важнее аккуратности
            logger.warning("Движок %s: не удалось прервать инференс (%s)", engine_id, exc)
    return aborted


def partial_files() -> list[Path]:
    """Недописанные аудиофайлы (`*.part`, `*.part.wav`) во всех каталогах вывода.

    Каталоги кусков и сравнений вложены в `output/`, поэтому обход идёт от корня,
    а повторы (если каталог задан отдельно, вне `output/`) убираются по
    разрешённому пути: один и тот же файл не должен попасть в отчёт дважды.
    """
    roots = dict.fromkeys(
        root.resolve() for root in (config.OUTPUT_DIR, config.PROJECTS_OUTPUT_DIR, config.BENCHMARKS_DIR)
        if root.exists()
    )
    found: dict[Path, Path] = {}
    for root in roots:
        # Отдельно заданный вложенный каталог уже обойдён вместе с `output/`.
        if any(root != other and other in root.parents for other in roots):
            continue
        for path in root.rglob(f"*{PARTIAL_SUFFIX}*"):
            if path.is_file() and PARTIAL_SUFFIX in path.name:
                found.setdefault(path.resolve(), path)
    return list(found.values())


def cleanup_partials() -> int:
    """Удаляет недописанные аудиофайлы после перезапуска приложения.

    Запись идёт через `<имя>.part` (см. `_write_audio`) и рядом лежит ровно до
    `os.replace`. Если процесс упал посреди записи, файл остаётся, и его никто
    больше не тронет: целевого имени у него нет, в базу он не попал, а по имени
    он не отличается от «просто странного» файла. Поэтому остатки убираются при
    старте — по явному признаку `.part`, а не по «похоже на временный».
    """
    removed = 0
    for path in partial_files():
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            path.unlink()
            removed += 1
            logger.info("Удалён недописанный файл %s (%.1f МБ)", path.name, size_mb)
        except OSError as exc:
            logger.warning("Не удалось удалить недописанный %s: %s", path, exc)
    return removed
