"""Генерация реплик по очереди и склейка их в один аудиофайл."""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .accentizer import accentuate
from .dialogue_parser import Replica
from .tts_engine import SAMPLE_RATE, get_engine
from .voices_store import Voice, get_store

logger = logging.getLogger(__name__)

# Целевая громкость куска: реплики разных голосов не должны «прыгать» по уровню.
_TARGET_RMS = 0.1
_PEAK_LIMIT = 0.99

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SpeakerSettings:
    voice_id: str
    speed: float = config.DEFAULT_SPEED
    cfg_strength: float = config.DEFAULT_CFG_STRENGTH
    nfe_step: int = config.DEFAULT_NFE_STEP

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSettings":
        return cls(
            voice_id=str(data.get("voice_id", "")),
            speed=float(data.get("speed", config.DEFAULT_SPEED)),
            cfg_strength=float(data.get("cfg_strength", config.DEFAULT_CFG_STRENGTH)),
            nfe_step=int(data.get("nfe_step", config.DEFAULT_NFE_STEP)),
        )


@dataclass
class RenderSettings:
    pause_ms: int = config.DEFAULT_PAUSE_MS
    cross_fade_duration: float = config.DEFAULT_CROSS_FADE_DURATION
    auto_accent: bool = True
    output_format: str = config.DEFAULT_OUTPUT_FORMAT


@dataclass
class RenderResult:
    output_path: Path
    duration_sec: float
    replicas_done: int


class ChunkTimeoutError(RuntimeError):
    """Синтез одного куска не уложился в TTS_CHUNK_TIMEOUT_SEC — похоже на зависание."""


class JobAbortedError(RuntimeError):
    """Задача прервана watchdog'ом (перерасход ресурсов), а не упала сама."""


def _normalize(chunk: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    if rms > 1e-6:
        chunk = chunk * (_TARGET_RMS / rms)
    peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    if peak > _PEAK_LIMIT:
        chunk = chunk * (_PEAK_LIMIT / peak)
    return chunk.astype(np.float32)


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
    return SpeakerSettings(
        voice_id=base.voice_id,
        speed=float(replica.overrides.get("speed", base.speed)),
        cfg_strength=float(replica.overrides.get("cfg_strength", base.cfg_strength)),
        nfe_step=int(replica.overrides.get("nfe_step", base.nfe_step)),
    )


async def render_dialogue(
    job_id: str,
    replicas: Iterable[Replica],
    speakers: dict[str, SpeakerSettings],
    settings: RenderSettings,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> RenderResult:
    """Синтезирует реплики строго по одной и склеивает их в один файл.

    Блокирующие шаги (загрузка модели, ударения, инференс, запись файла) уводятся
    в отдельный поток, чтобы не встал event loop. Инференс ограничен
    `config.CHUNK_TIMEOUT_SEC`, а между кусками проверяется `should_abort`.
    """
    replicas = list(replicas)
    if not replicas:
        raise ValueError("Диалог пуст")

    engine = get_engine()
    await asyncio.to_thread(engine.load)

    labels = {r.voice: r.label for r in replicas}
    unknown = sorted({r.voice for r in replicas if r.voice not in speakers})
    if unknown:
        names = ", ".join(f"«{labels[key]}»" for key in unknown)
        raise ValueError(f"Не назначен голос для: {names}")

    resolved: dict[str, Voice] = {}
    for replica in replicas:
        if replica.voice not in resolved:
            resolved[replica.voice] = _resolve_voice(replica.label, speakers[replica.voice])

    pause_samples = int(SAMPLE_RATE * max(settings.pause_ms, 0) / 1000)
    silence = np.zeros(pause_samples, dtype=np.float32) if pause_samples else None

    pieces: list[np.ndarray] = []
    total = len(replicas)

    for index, replica in enumerate(replicas, start=1):
        if should_abort and should_abort():
            raise JobAbortedError(
                f"Прервано на реплике {index} из {total}: превышен лимит памяти"
            )
        if on_progress:
            on_progress(index, total, replica.label)
        settings_for_replica = _settings_for(replica, speakers[replica.voice])
        voice = resolved[replica.voice]

        text = replica.text
        if settings.auto_accent:
            text = await asyncio.to_thread(accentuate, text)

        started = time.monotonic()
        try:
            chunk = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.synthesize,
                    text=text,
                    ref_audio_path=str(voice.audio_path),
                    ref_text=voice.ref_text,
                    speed=settings_for_replica.speed,
                    nfe_step=settings_for_replica.nfe_step,
                    cfg_strength=settings_for_replica.cfg_strength,
                    cross_fade_duration=settings.cross_fade_duration,
                ),
                timeout=config.CHUNK_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Таймаут синтеза: реплика %s/%s не готова за %.0f c (голос %s, %s знаков): %.200s",
                index, total, config.CHUNK_TIMEOUT_SEC, settings_for_replica.voice_id,
                len(text), text,
            )
            raise ChunkTimeoutError(
                f"Реплика {index} из {total} не синтезировалась за "
                f"{config.CHUNK_TIMEOUT_SEC:.0f} c — возможно, завис инференс"
            ) from exc
        logger.info(
            "Реплика %s/%s (%s) — %.1f c, %.1f c аудио",
            index, total, replica.label, time.monotonic() - started, len(chunk) / SAMPLE_RATE,
        )

        if pieces and silence is not None:
            pieces.append(silence)
        pieces.append(await asyncio.to_thread(_normalize, chunk))

    final = await asyncio.to_thread(np.concatenate, pieces)
    output_path = await asyncio.to_thread(_write_output, job_id, final, settings.output_format)
    return RenderResult(
        output_path=output_path,
        duration_sec=len(final) / SAMPLE_RATE,
        replicas_done=total,
    )


def _write_output(job_id: str, audio: np.ndarray, output_format: str) -> Path:
    import soundfile as sf

    output_format = (output_format or "wav").lower()
    target = config.OUTPUT_DIR / f"{job_id}.{output_format}"
    if output_format == "wav":
        sf.write(target, audio, SAMPLE_RATE)
        return target

    tmp_wav = config.OUTPUT_DIR / f"{job_id}.tmp.wav"
    sf.write(tmp_wav, audio, SAMPLE_RATE)
    try:
        from pydub import AudioSegment

        AudioSegment.from_wav(tmp_wav).export(target, format=output_format, bitrate="192k")
    finally:
        tmp_wav.unlink(missing_ok=True)
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
