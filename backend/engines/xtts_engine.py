"""Движок XTTS v2 (Coqui): zero-shot клонирование голоса по короткому референсу.

Две модели одного семейства живут как два независимых движка:

* `xtts` — базовая многоязычная `coqui/XTTS-v2` (веса ~1.9 ГБ);
* `xtts-banana` — русский комьюнити-файнтюн `Ftfyhh/xttsv2_banana` (веса ~5.2 ГБ).

Обе грузятся в тот же процесс, что и F5-TTS, и держатся тёплыми: по замерам три
тёплые модели вместе занимают ~1.5 ГБ RSS, то есть заметно меньше лимита процесса.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import config
from .base import (
    ENGINE_INFOS,
    ENGINE_XTTS,
    ENGINE_XTTS_BANANA,
    SAMPLE_RATE,
    SynthesisEngine,
    release_torch_memory,
)

logger = logging.getLogger(__name__)


def _install_transformers_shim() -> None:
    """Возвращает функцию, которую coqui-tts ждёт от transformers, а тот её удалил.

    `TTS/tts/layers/tortoise/autoregressive.py` импортирует
    `transformers.pytorch_utils.isin_mps_friendly`, но в transformers 5.x этой
    функции больше нет — импорт `TTS` падает с ImportError ещё до загрузки весов.
    Внутри библиотеки она используется ровно для выбора индексов и полностью
    заменяется на `torch.isin`. Шим ставится здесь, а не в `config.py`: он нужен
    только этому движку, и живёт до конца процесса.
    """
    import torch
    import transformers
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        return
    pytorch_utils.isin_mps_friendly = lambda elements, test_elements: torch.isin(
        elements, test_elements
    )
    logger.info(
        "XTTS: включён шим isin_mps_friendly (в transformers %s функция удалена)",
        transformers.__version__,
    )


def resolve_checkpoint_dir(root: Path) -> Path | None:
    """Ищет папку чекпоинта XTTS: в ней должны лежать `model.pth` и `config.json`.

    Путь не совпадает с именем репозитория: `huggingface_hub` при скачивании
    раскладывает файлы по вложенным папкам (`model_banana/v2.0.2/`), и жёстко
    прописанный путь сломался бы при любом обновлении репозитория.
    """
    if (root / "model.pth").is_file() and (root / "config.json").is_file():
        return root
    for model_file in sorted(root.rglob("model.pth")):
        if (model_file.parent / "config.json").is_file():
            return model_file.parent
    return None


class XTTSEngine(SynthesisEngine):
    """Одна модель XTTS v2: загрузка, кеш латентов референсов и инференс."""

    def __init__(self, info_id: str, checkpoint_root: Path) -> None:
        super().__init__()
        self.info = ENGINE_INFOS[info_id]
        self._checkpoint_root = Path(checkpoint_root)
        self._model = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        # Conditioning latents пересчитываются 7–8 секунд на голос, а зависят
        # только от файла референса — держим их между репликами.
        self._latents: OrderedDict[tuple, tuple] = OrderedDict()
        self._latents_lock = threading.Lock()

    @property
    def device(self) -> str | None:
        return self._device

    # -- загрузка --------------------------------------------------------------
    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            self._mark("loading")
            try:
                self._load_model(config.pick_device())
            except Exception as exc:  # noqa: BLE001 — состояние нужно отдать в /api/status
                logger.error("Не удалось загрузить XTTS (%s): %s", self.id, exc)
                self._model = None
                self._mark("failed", exc)
                raise
            self._mark("ready")

    def _release(self) -> None:
        """Обнуляет модель, устройство и кеш conditioning latents.

        Латенты — это тензоры на голос, и без их очистки выгрузка модели
        освободила бы не всю память движка. Вызывается, когда активных синтезов
        нет, поэтому локи свободны.
        """
        with self._load_lock:
            self._model = None
            self._device = None
        with self._latents_lock:
            self._latents.clear()
        release_torch_memory()

    def _load_model(self, device: str) -> None:
        config.configure_torch()
        _install_transformers_shim()

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        checkpoint_dir = resolve_checkpoint_dir(self._checkpoint_root)
        if checkpoint_dir is None:
            raise RuntimeError(
                f"Веса XTTS не найдены в {self._checkpoint_root}: нужны model.pth и config.json. "
                "Скачайте чекпоинт (см. README, раздел про движки)."
            )

        logger.info("Загружаю %s: %s на %s", self.info.label, checkpoint_dir, device)
        xtts_config = XttsConfig()
        xtts_config.load_json(str(checkpoint_dir / "config.json"))
        model = Xtts.init_from_config(xtts_config)
        model.load_checkpoint(xtts_config, checkpoint_dir=f"{checkpoint_dir}/", eval=True)
        model.to(device)
        self._model = model
        self._device = device
        logger.info("%s готова", self.info.label)

    # -- conditioning latents --------------------------------------------------
    def _conditioning(self, ref_audio_path: str) -> tuple:
        """Латент говорящего по референсу: считается один раз на файл."""
        path = Path(ref_audio_path)
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            # Файл не читается — ключ без метаданных, а понятную ошибку отдаст модель.
            key = (str(path), 0.0, 0)

        with self._latents_lock:
            cached = self._latents.get(key)
            if cached is not None:
                self._latents.move_to_end(key)
                return cached

        # Расчёт вне блокировки: очередь и так последовательная, но повторный
        # синтез той же реплики не должен ждать чужой расчёт.
        latents = self._model.get_conditioning_latents(
            audio_path=[str(path)],
            max_ref_length=config.XTTS_MAX_REF_SEC,
            gpt_cond_len=config.XTTS_COND_LEN_SEC,
            gpt_cond_chunk_len=config.XTTS_COND_LEN_SEC,
            sound_norm_refs=False,
        )
        # На CPU: инференс сам переносит их на устройство модели, а кеш не держит
        # лишние тензоры в памяти GPU.
        stored = (latents[0].cpu(), latents[1].cpu())
        with self._latents_lock:
            self._latents[key] = stored
            self._latents.move_to_end(key)
            while len(self._latents) > config.XTTS_LATENTS_CACHE_SIZE:
                self._latents.popitem(last=False)
        return stored

    # -- синтез ----------------------------------------------------------------
    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        """XTTS не использует `ref_text`, `cfg_strength` и `nfe_step`: референс ей
        нужен только как звук, а стабильность регулируется `temperature`."""
        with self._infer_lock:
            try:
                return self._infer(text, ref_audio_path, speed, params)
            except Exception as exc:
                if self._device != "mps":
                    raise
                # Часть операций на MPS может падать — перезапускаем модель на CPU.
                logger.warning("MPS упал в XTTS (%s). Перезагружаю %s на CPU.", exc, self.info.label)
                with self._load_lock:
                    self._model = None
                    self._load_model("cpu")
                return self._infer(text, ref_audio_path, speed, params)

    def _infer(
        self, text: str, ref_audio_path: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        gpt_cond_latent, speaker_embedding = self._conditioning(ref_audio_path)
        seed = params.get("seed")
        if seed is not None:
            # XTTS выбирает токены через torch.multinomial, поэтому без сида один
            # и тот же текст каждый раз звучит по-новому. Сид ставим перед
            # `inference`: с ним вариант куска воспроизводим, а без него выбор
            # «какой вариант оставить» был бы сравнением вслепую.
            import torch

            torch.manual_seed(int(seed))
        output = self._model.inference(
            text,
            config.XTTS_LANGUAGE,
            gpt_cond_latent,
            speaker_embedding,
            temperature=float(params["temperature"]),
            repetition_penalty=float(params["repetition_penalty"]),
            top_k=config.DEFAULT_XTTS_TOP_K,
            top_p=config.DEFAULT_XTTS_TOP_P,
            speed=speed,
            # Реплика диалога длиннее лимита XTTS для русского (~182 знака), а
            # обрыв внутри фразы сбросил бы интонацию — режем по предложениям.
            enable_text_splitting=True,
        )
        waveform = output["wav"]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return waveform, SAMPLE_RATE


def create_xtts_engine(engine_id: str) -> XTTSEngine:
    """Фабрика для реестра: путь к весам берётся из конфига по идентификатору движка."""
    roots = {
        ENGINE_XTTS: config.XTTS_BASE_DIR,
        ENGINE_XTTS_BANANA: config.XTTS_BANANA_DIR,
    }
    return XTTSEngine(engine_id, roots[engine_id])
