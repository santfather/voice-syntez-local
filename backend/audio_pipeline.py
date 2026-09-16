"""Генерация реплик по очереди и склейка их в один аудиофайл."""

import asyncio
import logging
import math
import random
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .accentizer import accentuate
from .dialogue_parser import Replica
from .engines.base import SAMPLE_RATE, SynthesisEngine
from .engines.registry import get_engine
from .text_preprocess import normalize
from .transcribe import transcribe_audio, word_error_rate
from .voices_store import Voice, get_store

logger = logging.getLogger(__name__)

# Ограничитель пика после нормализации: выше 0 dBFS файл клипует.
_PEAK_LIMIT = 0.99

ProgressCallback = Callable[[int, int, str], None]
# Ожидание освобождения памяти перед повторной попыткой (её ставит очередь).
WaitForMemory = Callable[[], Awaitable[None]]
# Сообщение о ходе работы (попадает в статус задачи в интерфейсе).
NoteCallback = Callable[[str], None]

# Чем закончился QA-цикл куска. Отдельные значения, а не флаг «получилось»: в
# интерфейсе важно различать «проверено и прошло» и «проверить не успели» —
# принятое без полной проверки аудио не должно выглядеть проверенным.
QA_PASSED = "passed"
QA_BUDGET = "budget_exhausted"
QA_ATTEMPTS = "attempts_exhausted"
QA_UNAVAILABLE = "unavailable"


@dataclass
class QaSettings:
    """Параметры строгой проверки куска.

    Значения по умолчанию берутся из конфига, но живут отдельным объектом, а не
    читаются из `config` внутри цикла: в тестах цикл должен прогоняться на своих
    числах, а не на боевых трёхстах секундах бюджета.
    """

    wer_threshold: float = config.QA_WER_THRESHOLD
    max_attempts: int = config.QA_MAX_ATTEMPTS
    budget_sec: float = config.QA_BUDGET_SEC


@dataclass
class QaOutcome:
    """Итог проверки одного куска: чем цикл закончился и что из него выбрано."""

    status: str
    wer: float | None
    attempts: int

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "wer": None if self.wer is None else round(self.wer, 4),
            "attempts": self.attempts,
        }


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

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSettings":
        raw_pause = data.get("pause_override_ms")
        raw_params = data.get("engine_params")
        return cls(
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


@dataclass
class RenderSettings:
    pause_ms: int = config.DEFAULT_PAUSE_MS
    cross_fade_duration: float = config.DEFAULT_CROSS_FADE_DURATION
    auto_accent: bool = True
    output_format: str = config.DEFAULT_OUTPUT_FORMAT
    # Строгая проверка куска (Фаза 7); None — обычный путь, одна попытка без
    # расшифровки. Выбор пользователя на задачу, а не свойство установки: цена
    # проверки — синтез плюс отдельный процесс Whisper на каждую попытку.
    qa: QaSettings | None = None


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


class ChunkTimeoutError(RuntimeError):
    """Синтез одного куска не уложился в TTS_CHUNK_TIMEOUT_SEC — похоже на зависание."""


class JobAbortedError(RuntimeError):
    """Задача прервана watchdog'ом (перерасход ресурсов), а не упала сама."""


def _pitch_shift(chunk: np.ndarray, semitones: float) -> np.ndarray:
    """Меняет высоту тона, не трогая длительность. Тяжёлый импорт — только по факту."""
    if not semitones:
        return chunk
    import librosa

    return np.asarray(
        librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=float(semitones)),
        dtype=np.float32,
    )


def _trim_edge_silence(chunk: np.ndarray) -> np.ndarray:
    """Срезает тишину, которую модель оставила на краях куска.

    Модель почти всегда добавляет к сгенерированному куску немного тишины, а
    `pause_ms` добавляет паузу поверх неё. Суммарный зазор между репликами тогда
    гуляет от куска к куску и звучит неритмично. Обрезка по энергетическому
    порогу делает паузу ровно той, что задал пользователь.
    """
    frame = int(SAMPLE_RATE * config.EDGE_SILENCE_FRAME_MS / 1000)
    if frame <= 0 or chunk.size < 4 * frame:
        return chunk
    usable = chunk.size - chunk.size % frame
    frames = chunk[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loud = np.flatnonzero(rms > 10.0 ** (config.EDGE_SILENCE_DB / 20.0))
    if loud.size == 0:  # кусок целиком тихий — резать нечего
        return chunk
    margin = int(SAMPLE_RATE * config.EDGE_SILENCE_MARGIN_MS / 1000)
    start = max(0, int(loud[0]) * frame - margin)
    end = min(chunk.size, (int(loud[-1]) + 1) * frame + margin)
    if end - start < 2 * frame:
        return chunk
    return chunk[start:end]


def _prepare_chunk(chunk: np.ndarray, settings: SpeakerSettings) -> np.ndarray:
    """Готовит кусок к склейке: края, тембр и громкость.

    Один проход с одной копией: на длинных репликах лишние копии waveform заметны
    и по памяти, и по времени. Gain применяется после нормализации RMS — иначе
    нормализация обнулила бы ручную балансировку громкости.
    """
    trimmed = _trim_edge_silence(np.asarray(chunk, dtype=np.float32))
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


def _settings_for(replica: Replica, base: SpeakerSettings) -> SpeakerSettings:
    """Параметры куска: значения из маркера в тексте перекрывают карточку."""
    if not replica.overrides:
        return base
    return replace(
        base,
        speed=float(replica.overrides.get("speed", base.speed)),
        cfg_strength=float(replica.overrides.get("cfg_strength", base.cfg_strength)),
        nfe_step=int(replica.overrides.get("nfe_step", base.nfe_step)),
    )


def _engine_for(voice: Voice) -> SynthesisEngine:
    """Движок, выбранный для этого голоса."""
    try:
        return get_engine(voice.engine)
    except ValueError as exc:
        raise ValueError(f"У голоса «{voice.name}» неизвестный движок «{voice.engine}»") from exc


def _text_for_engine(text: str, engine: SynthesisEngine, auto_accent: bool) -> str:
    """Готовит текст куска к отправке в модель.

    Порядок важен: сначала числа и латиница (`text_preprocess`), потом ударения.
    RUAccent ставит «+» перед гласной, и на развёрнутых числах он тоже должен
    отработать — иначе «две тысячи двадцать шестом» уйдёт без разметки. Ударения
    при этом остаются только у F5: XTTS знак «+» читает как отдельный символ.
    """
    prepared = normalize(text)
    if auto_accent and engine.supports_accents:
        return accentuate(prepared)
    return prepared


async def _load_engines(voices: Iterable[Voice]) -> dict[str, SynthesisEngine]:
    """Грузит движки всех голосов диалога до первой реплики.

    Модель XTTS поднимается 20–40 секунд. Если грузить её лениво, эта пауза
    придётся на середину диалога — а прогретая модель нужна до начала синтеза,
    ровно как это было с единственным F5-TTS.
    """
    engines: dict[str, SynthesisEngine] = {}
    for voice in voices:
        engine = _engine_for(voice)
        if engine.id in engines:
            continue
        engines[engine.id] = engine
        await asyncio.to_thread(engine.load)
        logger.info("Движок %s готов (%s)", engine.id, voice.name)
    return engines


def _engine_params(
    voice: Voice, speaker: SpeakerSettings, render: RenderSettings, seed: int | None
) -> dict:
    """Ручки движка для одного куска.

    Значения, сохранённые у голоса, — это дефолты, а карточка слота их
    переопределяет. Общие ручки пайплайна (`target_rms`, `cross_fade_duration`)
    добавляются последними: они относятся к сборке куска, и движок читает их
    только если они у него есть. `seed` передаётся только движкам, которые его
    принимают (см. `SynthesisEngine.supports_seed`), и то же значение уходит в
    карточку куска — чтобы вариант можно было воспроизвести, а не только принять.
    """
    return {
        **voice.engine_params,
        **speaker.engine_params,
        "cfg_strength": speaker.cfg_strength,
        "nfe_step": speaker.nfe_step,
        "target_rms": speaker.target_rms,
        "cross_fade_duration": render.cross_fade_duration,
        "seed": seed,
    }


def _pause_samples(speaker: SpeakerSettings, render: RenderSettings) -> int:
    """Пауза перед репликами спикера: его личное значение важнее общего."""
    pause_ms = render.pause_ms if speaker.pause_override_ms is None else speaker.pause_override_ms
    return int(SAMPLE_RATE * max(pause_ms, 0) / 1000)


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
    """
    seed = random.randrange(config.SEED_MAX) if engine.supports_seed else None
    started = time.monotonic()
    try:
        outcome, failure = await asyncio.wait_for(
            asyncio.to_thread(
                _synthesize_in_thread,
                engine,
                text=text,
                ref_audio_path=str(voice.audio_path),
                ref_text=voice.ref_text,
                speed=tuning.speed,
                **_engine_params(voice, tuning, settings, seed),
            ),
            timeout=config.CHUNK_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        logger.warning(
            "Таймаут синтеза: реплика %s не готова за %.0f c (движок %s, голос %s, %s знаков): %.200s",
            position, config.CHUNK_TIMEOUT_SEC, engine.id, tuning.voice_id, len(text), text,
        )
        raise ChunkTimeoutError(
            f"Реплика {position} не синтезировалась за {config.CHUNK_TIMEOUT_SEC:.0f} c "
            "— возможно, завис инференс"
        ) from exc
    if failure is not None:
        logger.error(
            "Реплика %s (%s): движок %s упал (%s)",
            position, label, engine.id, type(failure).__name__, exc_info=failure,
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
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
    """
    recognized = await _transcribe_chunk(chunk)
    if not recognized.strip():
        return 1.0  # модель промолчала — это провал проверки, а не «сверять нечего»
    return word_error_rate(normalize(expected), normalize(recognized))


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
) -> tuple[np.ndarray, int | None, QaOutcome | None]:
    """Синтез куска: обычный или со строгой проверкой (`settings.qa`).

    Без проверки это ровно один вызов движка. С проверкой каждая попытка — это
    полный синтез плюс отдельный процесс Whisper, поэтому цикл ограничен и числом
    попыток, и временем, а из проверенных берётся лучшая по WER, даже если порог
    так и не взят: выбросить уже сгенерированное аудио и вернуть ошибку значило бы
    потерять работу ради формальности.

    Ошибки синтеза и таймауты идут наружу, как и раньше: проверка отвечает за
    качество готового куска, а не за то, чтобы прятать зависший инференс.
    """
    # Текст куска готовит `_text_for_engine`: числа и латиница для всех движков,
    # ударения — только для тех, кто понимает «+» (XTTS его прочитала бы вслух).
    text = await asyncio.to_thread(_text_for_engine, replica.text, engine, settings.auto_accent)
    qa = settings.qa
    if qa is None:
        chunk, seed = await _synthesize_chunk(
            engine, voice, text, tuning, settings, position=position, label=label
        )
        return chunk, seed, None

    started = time.monotonic()
    limit = max(qa.max_attempts, 1)
    best: tuple[np.ndarray, int | None, float] | None = None
    attempts = 0
    status = QA_ATTEMPTS
    for attempt in range(1, limit + 1):
        if attempt > 1:
            if time.monotonic() - started >= qa.budget_sec:
                status = QA_BUDGET
                break
            # Whisper на каждой попытке — это ещё один процесс и ещё ~1.6 ГБ
            # памяти: перед повтором ждём, пока система её отпустит.
            if wait_for_memory is not None:
                await wait_for_memory()
        if on_note:
            on_note(f"Реплика {position}: строгая проверка, попытка {attempt} из {limit}")

        chunk, seed = await _synthesize_chunk(
            engine,
            voice,
            text,
            _nudge_tuning(voice, engine, tuning, attempt - 1),
            settings,
            position=position,
            label=label,
        )
        attempts = attempt
        try:
            wer = await _measure_wer(chunk, replica.text)
        except Exception as exc:  # noqa: BLE001 — упавшая расшифровка не должна губить задачу
            logger.warning(
                "Реплика %s: строгая проверка недоступна (%s: %s) — принимаю без проверки",
                position, type(exc).__name__, exc,
            )
            return chunk, seed, QaOutcome(status=QA_UNAVAILABLE, wer=None, attempts=attempt)

        if best is None or wer < best[2]:
            best = (chunk, seed, wer)
        if on_note:
            on_note(f"Реплика {position}: проверка {attempt} из {limit}, WER {wer:.2f}")
        if wer <= qa.wer_threshold:
            status = QA_PASSED
            break

    assert best is not None  # первая попытка выполняется всегда
    chunk, seed, wer = best
    logger.info(
        "Реплика %s: строгая проверка — %s, WER %.2f, попыток %s",
        position, status, wer, attempts,
    )
    return chunk, seed, QaOutcome(status=status, wer=wer, attempts=attempts)


async def render_dialogue(
    job_id: str,
    replicas: Iterable[Replica],
    speakers: dict[str, SpeakerSettings],
    settings: RenderSettings,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
) -> RenderResult:
    """Синтезирует реплики строго по одной и склеивает их в один файл.

    Блокирующие шаги (загрузка модели, ударения, инференс, запись файла) уводятся
    в отдельный поток, чтобы не встал event loop. Инференс ограничен
    `config.CHUNK_TIMEOUT_SEC`, а между кусками проверяется `should_abort`.

    Движок выбирается у каждого голоса отдельно, поэтому в одном файле спокойно
    соседствуют куски F5-TTS и XTTS: реплики идут строго последовательно, и
    склейка для неё одинакова — все движки отдают 24 кГц.
    """
    replicas = list(replicas)
    if not replicas:
        raise ValueError("Диалог пуст")

    labels = {r.voice: r.label for r in replicas}
    unknown = sorted({r.voice for r in replicas if r.voice not in speakers})
    if unknown:
        names = ", ".join(f"«{labels[key]}»" for key in unknown)
        raise ValueError(f"Не назначен голос для: {names}")

    resolved: dict[str, Voice] = {}
    for replica in replicas:
        if replica.voice not in resolved:
            resolved[replica.voice] = _resolve_voice(replica.label, speakers[replica.voice])
    engines = await _load_engines(list(resolved.values()))

    pieces: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    seeds: list[int | None] = []
    checks: list[QaOutcome | None] = []
    total = len(replicas)
    cursor = 0
    for index, replica in enumerate(replicas, start=1):
        if should_abort and should_abort():
            raise JobAbortedError(
                f"Прервано на реплике {index} из {total}: превышен лимит памяти"
            )
        if on_progress:
            on_progress(index, total, replica.label)
        settings_for_replica = _settings_for(replica, speakers[replica.voice])
        voice = resolved[replica.voice]
        engine = engines[voice.engine]

        chunk, seed, qa_outcome = await _synthesize_checked(
            engine,
            voice,
            replica,
            settings_for_replica,
            settings,
            position=f"{index} из {total}",
            label=replica.label,
            wait_for_memory=wait_for_memory,
            on_note=on_note,
        )
        seeds.append(seed)
        checks.append(qa_outcome)

        pause = _pause_samples(settings_for_replica, settings)
        if pieces and pause:
            pieces.append(np.zeros(pause, dtype=np.float32))
            cursor += pause
        prepared = await asyncio.to_thread(_prepare_chunk, chunk, settings_for_replica)
        segments.append((cursor, cursor + prepared.size))
        cursor += prepared.size
        pieces.append(prepared)

    final = await asyncio.to_thread(np.concatenate, pieces)
    final = await asyncio.to_thread(_finalize_track, final)
    output_path = await asyncio.to_thread(_write_output, job_id, final, settings.output_format)
    return RenderResult(
        output_path=output_path,
        duration_sec=len(final) / SAMPLE_RATE,
        replicas_done=total,
        segments=segments,
        seeds=seeds,
        qa=checks,
    )


async def synthesize_replica(
    replica: Replica,
    speaker: SpeakerSettings,
    settings: RenderSettings,
    index: int,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
) -> tuple[np.ndarray, int | None, QaOutcome | None]:
    """Синтезирует и готовит один кусок — без записи в готовый файл.

    Возвращает подготовленный кусок, сид, которым он получен, и итог строгой
    проверки (`None` — проверка выключена): всё это нужно вызывающему, чтобы
    сохранить вариант и уже его поставить в файл. Пересобирать весь диалог из-за
    одной неудачной реплики — десятки секунд на каждый кусок, поэтому
    перегенерация трогает ровно один кусок.
    """
    voice = _resolve_voice(replica.label, speaker)
    engine = _engine_for(voice)
    await asyncio.to_thread(engine.load)
    # Параметры из маркера в тексте важнее карточки — ровно как при первой сборке,
    # иначе перегенерация дала бы кусок, несовместимый с соседями по звучанию.
    resolved = _settings_for(replica, speaker)

    chunk, seed, qa_outcome = await _synthesize_checked(
        engine,
        voice,
        replica,
        resolved,
        settings,
        position=str(index + 1),
        label=replica.label,
        wait_for_memory=wait_for_memory,
        on_note=on_note,
    )
    prepared = await asyncio.to_thread(_prepare_chunk, chunk, resolved)
    logger.info(
        "Реплика %s (%s) пересинтезирована: %.2f c аудио, сид %s",
        index + 1, replica.label, prepared.size / SAMPLE_RATE, seed,
    )
    return prepared, seed, qa_outcome


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
    """Читает готовый файл обратно в float32-моно при SAMPLE_RATE."""
    if (output_format or "wav").lower() == "wav":
        import soundfile as sf

        data, _ = sf.read(path, dtype="float32")
        return np.asarray(data, dtype=np.float32).reshape(-1)

    from pydub import AudioSegment

    segment = AudioSegment.from_file(path, format=output_format)
    samples = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _write_audio(target: Path, audio: np.ndarray, output_format: str) -> None:
    import soundfile as sf

    output_format = (output_format or "wav").lower()
    if output_format == "wav":
        sf.write(target, audio, SAMPLE_RATE)
        return

    tmp_wav = target.with_suffix(".tmp.wav")
    sf.write(tmp_wav, audio, SAMPLE_RATE)
    try:
        from pydub import AudioSegment

        AudioSegment.from_wav(tmp_wav).export(target, format=output_format, bitrate="192k")
    finally:
        tmp_wav.unlink(missing_ok=True)


def _write_output(job_id: str, audio: np.ndarray, output_format: str) -> Path:
    output_format = (output_format or "wav").lower()
    target = config.OUTPUT_DIR / f"{job_id}.{output_format}"
    _write_audio(target, audio, output_format)
    return target


def cleanup_output(ttl_hours: float = config.OUTPUT_TTL_HOURS) -> int:
    """Удаляет готовые файлы старше ttl_hours. Возвращает число удалённых."""
    if ttl_hours <= 0:
        return 0
    threshold = time.time() - ttl_hours * 3600
    removed = 0
    for path in config.OUTPUT_DIR.iterdir():
        if not path.is_file() or path.name == ".gitkeep":
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
