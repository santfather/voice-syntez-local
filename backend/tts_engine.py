"""Обёртка над F5-TTS: модель грузится один раз (singleton) и живёт в памяти процесса."""

import logging
import random
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# F5-TTS при каждом `infer` вызывает `seed_everything`, который пишет сид в
# os.environ["PYTHONHASHSEED"] (f5_tts/model/utils.py). При seed=None библиотека
# берёт random.randint(0, sys.maxsize) — это больше допустимого максимума
# 4294967295, и унаследовавший переменную подпроцесс падает с
# "Fatal Python error: config_init_hash_seed". Сид задаём сами, в допустимых
# границах: распределение остаётся случайным, как и раньше.
HASH_SEED_MAX = 2 ** 32


class _InlineExecutor:
    """Синхронная замена ThreadPoolExecutor: задачи выполняются сразу в текущем потоке."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 — ошибку отдаём через Future, как это делает executor
            future.set_exception(exc)
        return future


def _force_sequential_batches() -> None:
    """Отключает параллельную генерацию батчей внутри F5-TTS (нужно для MPS).

    `f5_tts.infer.utils_infer.infer_batch_process` раскидывает батчи одной реплики по
    потокам `ThreadPoolExecutor`. MPS-бэкенд torch не потокобезопасен: параллельные ядра
    из разных потоков ломают общий кэш шейдеров Metal
    (`MetalShaderLibrary::exec_unary_kernel`) и процесс падает с SIGSEGV, который
    невозможно перехватить из Python. Выполняем батчи последовательно.
    """
    from f5_tts.infer import utils_infer

    utils_infer.ThreadPoolExecutor = _InlineExecutor
    logger.info("F5-TTS: параллельная генерация батчей отключена (MPS не потокобезопасен)")


class TTSEngine:
    """Единственный экземпляр модели на процесс. Синтез — строго последовательный."""

    _instance: Optional["TTSEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.device: str | None = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -- singleton -------------------------------------------------------------
    @classmethod
    def instance(cls) -> "TTSEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- загрузка --------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_weights(self) -> tuple[Path, Path]:
        """Локальные пути к чекпоинту и словарю; при отсутствии — докачивает с HF."""
        ckpt = self._find(config.CKPT_FILE)
        vocab = self._find(config.VOCAB_FILE)
        if ckpt and vocab:
            return ckpt, vocab

        from huggingface_hub import hf_hub_download

        logger.info("Веса не найдены локально, качаю %s", config.HF_REPO_ID)
        if not ckpt:
            ckpt = Path(
                hf_hub_download(config.HF_REPO_ID, config.HF_CKPT_PATH, local_dir=config.MODELS_DIR)
            )
        if not vocab:
            vocab = Path(
                hf_hub_download(config.HF_REPO_ID, config.HF_VOCAB_PATH, local_dir=config.MODELS_DIR)
            )
        return ckpt, vocab

    @staticmethod
    def _find(expected: Path) -> Path | None:
        """Ищет файл по ожидаемому пути, затем по имени в любом месте models/."""
        if expected.exists():
            return expected
        found = next(config.MODELS_DIR.rglob(expected.name), None)
        return found

    def load(self, device: str | None = None) -> None:
        """Грузит модель (идемпотентно). Тяжёлая операция — вызывать один раз при старте."""
        with self._load_lock:
            if self._model is not None and (device is None or device == self.device):
                return

            config.configure_torch()
            from f5_tts.api import F5TTS

            device = device or config.pick_device()
            if device == "mps":
                _force_sequential_batches()
            ckpt, vocab = self._resolve_weights()
            logger.info("Загружаю F5-TTS: ckpt=%s vocab=%s device=%s", ckpt.name, vocab.name, device)
            self._model = F5TTS(
                model=config.F5_MODEL_NAME,
                ckpt_file=str(ckpt),
                vocab_file=str(vocab),
                device=device,
            )
            self.device = device
            logger.info("Модель загружена на %s", device)

    # -- синтез ----------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        speed: float = config.DEFAULT_SPEED,
        nfe_step: int = config.DEFAULT_NFE_STEP,
        cfg_strength: float = config.DEFAULT_CFG_STRENGTH,
        sway_sampling_coef: float = config.DEFAULT_SWAY_SAMPLING_COEF,
        cross_fade_duration: float = config.DEFAULT_CROSS_FADE_DURATION,
    ) -> np.ndarray:
        """Синтез одной реплики. Возвращает float32 waveform, sr = 24000."""
        if self._model is None:
            self.load()

        with self._infer_lock:  # очередь последовательная, но перестрахуемся
            try:
                return self._infer(
                    text, ref_audio_path, ref_text, speed, nfe_step, cfg_strength,
                    sway_sampling_coef, cross_fade_duration,
                )
            except Exception as exc:
                if self.device != "mps":
                    raise
                # Часть операций на MPS может падать — перезапускаем модель на CPU.
                logger.warning("MPS упал (%s). Перезагружаю модель на CPU.", exc)
                self._model = None
                self.load(device="cpu")
                return self._infer(
                    text, ref_audio_path, ref_text, speed, nfe_step, cfg_strength,
                    sway_sampling_coef, cross_fade_duration,
                )

    def _infer(
        self, text, ref_audio_path, ref_text, speed, nfe_step, cfg_strength,
        sway_sampling_coef, cross_fade_duration,
    ) -> np.ndarray:
        wav, sr, _ = self._model.infer(
            ref_file=str(ref_audio_path),
            ref_text=ref_text,
            gen_text=text,
            show_info=lambda *a, **kw: None,
            nfe_step=int(nfe_step),
            cfg_strength=float(cfg_strength),
            speed=float(speed),
            sway_sampling_coef=float(sway_sampling_coef),
            cross_fade_duration=float(cross_fade_duration),
            seed=random.randrange(HASH_SEED_MAX),
        )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Неожиданный sample rate от модели: {sr}")
        return wav


def get_engine() -> TTSEngine:
    return TTSEngine.instance()
