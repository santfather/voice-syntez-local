"""Движок F5-TTS Russian: тонкая обёртка над существующим `tts_engine.TTSEngine`.

Внутренняя логика F5 не меняется (последовательные батчи на MPS, фолбэк на CPU,
сид в допустимых границах) — здесь только приведение её вызова к общему
интерфейсу `SynthesisEngine` и объявление ручек, специфичных для F5.
"""

import logging

import numpy as np

from .. import config
from ..tts_engine import TTSEngine
from .base import (
    ENGINE_F5,
    ENGINE_INFOS,
    SAMPLE_RATE,
    SynthesisEngine,
    release_torch_memory,
)

logger = logging.getLogger(__name__)


class F5Engine(SynthesisEngine):
    """F5-TTS как один из движков синтеза."""

    info = ENGINE_INFOS[ENGINE_F5]

    def __init__(self) -> None:
        super().__init__()
        # TTSEngine — singleton и уже следит за своей загрузкой; здесь только
        # состояние для интерфейса, чтобы /api/engines отвечал единообразно.
        self._engine = TTSEngine.instance()

    @property
    def device(self) -> str | None:
        return self._engine.device

    def load(self) -> None:
        if self.is_loaded:
            return
        self._mark("loading")
        try:
            self._engine.load()
        except Exception as exc:  # noqa: BLE001 — состояние нужно отдать в /api/status
            logger.error("Не удалось загрузить F5-TTS: %s", exc)
            self._mark("failed", exc)
            raise
        self._mark("ready")

    def _release(self) -> None:
        """Обнуляет модель F5 (общий singleton `TTSEngine`) и возвращает память MPS.

        Сам `TTSEngine` остаётся тем же объектом: сбрасывается только ссылка на
        модель, поэтому `load()` после выгрузки поднимает веса заново.
        """
        self._engine.unload()
        release_torch_memory()

    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        """`cfg_strength`/`nfe_step` приходят из карточки слота, остальное — из общих настроек.

        `cross_fade_duration` и `target_rms` — ручки F5: первая сшивает батчи
        внутри куска, вторая выравнивает громкость разных текстов до нормализации.
        У XTTS аналогов нет, поэтому ключи передаются как есть и читаются только здесь.
        `seed` (когда задан) фиксирует генерацию: с ним тот же текст и тот же
        референс дают тот же результат, и вариант куска можно повторить.
        """
        waveform = self._engine.synthesize(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            speed=speed,
            seed=params.get("seed"),
            nfe_step=int(params.get("nfe_step", config.DEFAULT_NFE_STEP)),
            cfg_strength=float(params.get("cfg_strength", config.DEFAULT_CFG_STRENGTH)),
            cross_fade_duration=float(
                params.get("cross_fade_duration", config.DEFAULT_CROSS_FADE_DURATION)
            ),
            target_rms=float(params.get("target_rms", config.DEFAULT_TARGET_RMS)),
        )
        return waveform, SAMPLE_RATE
