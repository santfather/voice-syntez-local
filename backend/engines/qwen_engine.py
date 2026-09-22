"""Движок Qwen3-TTS 1.7B (Base): клонирование голоса по короткому референсу.

Модель живёт в пакете `qwen-tts` и состоит из двух частей: `-Base` синтезирует
кодовые кадры, `-Tokenizer-12Hz` превращает их в звук. Обе скачиваются в
`models/` (см. `model_manager`), потому что проект работает офлайн.

Три особенности, ради которых этот адаптер существует отдельно:

* **Версия transformers.** `qwen-tts` требует transformers 4.57, а проект живёт
  на 5.x (Whisper и coqui). Совместимость обеспечивается установкой без
  зависимостей (`pip install --no-deps qwen-tts`) плюс шимом на функции, которые
  пятая версия удалила, — ровно тем же приёмом, что уже применён к coqui-tts
  (см. `xtts_engine._install_transformers_shim`).
* **Устройство.** MPS для этой модели официально не заявлен: `qwen-tts`
  документирует CUDA. Устройство выбирается `config.pick_device()`, dtype — по
  устройству, а падение на MPS повторяет ту же логику, что у XTTS: движок
  перезагружается на CPU и кусок синтезируется заново, а не теряется.
* **Clone-prompt.** Полный промпт референса (`create_voice_clone_prompt`) — это
  посчитанные на устройстве фичи; их пересчёт на каждую реплику стоил бы
  столько же, сколько сам синтез. Промпт кешируется по файлу референса и режиму
  клонирования, как conditioning latents у XTTS.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import config
from .base import (
    ENGINE_INFOS,
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_MODE_KEY,
    ENGINE_QWEN,
    SAMPLE_RATE,
    SynthesisEngine,
    release_torch_memory,
)

logger = logging.getLogger(__name__)

# Режим, в котором промпт строится только по эмбеддингу спикера: тогда `ref_text`
# не нужен, но качество клонирования ниже — см. паспорт движка.
_X_VECTOR_MODE = ENGINE_MODE_EXPERIMENTAL


def _install_transformers_shim() -> None:
    """Возвращает функции, которые `qwen-tts` ждёт от transformers, а тот удалил.

    Пятая версия transformers вычистила часть вспомогательных функций, которые
    импортирует код, написанный под 4.x. Импорт падает ещё до загрузки весов, и
    выглядит это как «модель не установилась». Шим ставится здесь, а не в
    `config.py`: он нужен только этому движку и живёт до конца процесса.

    Список намеренно короткий — только то, что можно восстановить без изменения
    поведения. Остальные несовместимости ловит `tools/qwen_feasibility.py`, и
    каждая добавляется сюда отдельной строкой, а не «на всякий случай».
    """
    import torch
    import transformers
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        # Используется для выбора индексов; `torch.isin` делает то же самое.
        pytorch_utils.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )
        logger.info(
            "Qwen3-TTS: включён шим isin_mps_friendly (в transformers %s функция удалена)",
            transformers.__version__,
        )


def resolve_model_dir(root: Path, *, required: tuple[str, ...]) -> Path | None:
    """Каталог, в котором лежат все обязательные файлы модели, или `None`.

    Каталог модели определяется по `config.json`: в HF-репозитории он один и
    всегда рядом с весами. Путь не предполагается равным имени репозитория —
    `huggingface_hub` раскладывает файлы по вложенным папкам, — и жёстко
    прописанный путь сломался бы при любом обновлении модели. Проверка та же,
    что у `model_manager.missing_files`, поэтому «модель установлена» и «модель
    загружается» не могут разойтись.

    Шаблоны допускаются (`*.safetensors`): крупные веса публикуются шардами, и
    единственного файла с ожидаемым именем может просто не быть.
    """
    if not root.is_dir():
        return None
    for config_file in sorted(root.rglob("config.json")):
        parent = config_file.parent
        if all(any(parent.rglob(name)) for name in required):
            return parent
    return None


class QwenTTSEngine(SynthesisEngine):
    """Одна модель Qwen3-TTS: загрузка, кеш clone-prompt и инференс."""

    info = ENGINE_INFOS[ENGINE_QWEN]

    def __init__(self, base_dir: Path | None = None) -> None:
        super().__init__()
        self._base_dir = Path(base_dir if base_dir is not None else config.QWEN_BASE_DIR)
        self._model = None
        self._device: str | None = None
        self._dtype = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        # Ключ — референс вместе с его метаданными и режимом клонирования:
        # подмена файла под тем же именем должна давать новый промпт, а не
        # звучание старого голоса.
        self._prompts: OrderedDict[tuple, object] = OrderedDict()
        self._prompts_lock = threading.Lock()

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def dtype(self) -> str:
        """dtype загруженных весов словами (`float32`/`bfloat16`) — для диагностики.

        Нужен пробам и логу: без него «на MPS было bf16 или fp32» пришлось бы
        выяснять чтением кода, а это как раз то, что измеряет `tools/qwen_feasibility.py`.
        """
        return "" if self._dtype is None else str(self._dtype).replace("torch.", "")

    # -- загрузка --------------------------------------------------------------
    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            self._mark("loading")
            device = self._pick_device()
            try:
                self._load_model(device)
            except Exception as exc:
                logger.error("Не удалось загрузить Qwen3-TTS на %s: %s", device, exc)
                self._model = None
                self._mark("failed", exc)
                raise
            self._mark("ready")

    def _pick_device(self) -> str:
        """Устройство из настройки: `auto` — MPS, если он доступен и не запрещён."""
        wanted = (config.QWEN_DEVICE or "auto").strip().lower()
        if wanted == "auto":
            wanted = "cpu" if config.qwen_force_cpu() else config.pick_device()
        if wanted == "mps" and config.qwen_force_cpu():
            # Явно выбранный MPS уступает аварийному тормозу: переменная
            # существует ровно для случая «проба ошиблась, модель падает».
            logger.warning("Qwen3-TTS: MPS запрещён настройкой — работаю на CPU")
            wanted = "cpu"
        if wanted not in ("mps", "cpu", "cuda"):
            logger.warning("Qwen3-TTS: неизвестное устройство «%s» — работаю на CPU", wanted)
            wanted = "cpu"
        return wanted

    def _load_model(self, device: str) -> None:
        torch = config.configure_torch()

        # Файлы проверяются раньше импорта пакета: пользователю, у которого нет
        # весов, нужнее «скачайте модель», чем «нет модуля qwen_tts», — а
        # порядок наоборот дал бы вторую ошибку вместо первой.
        model_dir = resolve_model_dir(
            self._base_dir, required=("config.json", "*.safetensors")
        )
        if model_dir is None:
            raise RuntimeError(
                f"Веса Qwen3-TTS не найдены в {self._base_dir}: нужен каталог модели "
                "с config.json и весами (*.safetensors). Скачайте модель во вкладке "
                "«Модели» или через `huggingface-cli download` (см. README)."
            )
        _install_transformers_shim()
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover — зависит от окружения
            raise RuntimeError(
                "Пакет `qwen-tts` не установлен. Он требует transformers 4.57, а "
                "проект живёт на 5.x, поэтому ставится без зависимостей: "
                "`./venv/bin/pip install --no-deps -r requirements-qwen.txt` "
                "(см. docs/install.md)."
            ) from exc

        # bf16 только там, где он есть: 1.7B в float32 на MPS — это ~7 ГБ и
        # заметно медленнее. На CPU bf16 у большинства операций эмулируется.
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        logger.info(
            "Загружаю %s: %s на %s (%s, attention=%s)",
            self.info.label,
            model_dir,
            device,
            dtype,
            config.QWEN_ATTN_IMPLEMENTATION,
        )
        model = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map=device,
            dtype=dtype,
            attn_implementation=config.QWEN_ATTN_IMPLEMENTATION,
        )
        self._model = model
        self._device = device
        self._dtype = dtype
        logger.info("%s готов", self.info.label)

    def _release(self) -> None:
        """Обнуляет модель, устройство и кеш промптов, затем возвращает память.

        Промпты — это тензоры на устройстве модели: без их очистки выгрузка
        освободила бы веса, но не то, что посчитано по референсам. Вызывается
        только когда активных синтезов нет, поэтому локи свободны.
        """
        with self._load_lock:
            self._model = None
            self._device = None
            self._dtype = None
        with self._prompts_lock:
            self._prompts.clear()
        release_torch_memory()

    # -- clone-prompt ----------------------------------------------------------
    def _clone_prompt(self, ref_audio_path: str, ref_text: str, x_vector_only: bool):
        """Промпт референса: считается один раз на файл и режим клонирования."""
        path = Path(ref_audio_path)
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size, bool(x_vector_only))
        except OSError:
            # Файл не читается — ключ без метаданных, а понятную ошибку отдаст модель.
            key = (str(path), 0.0, 0, bool(x_vector_only))

        with self._prompts_lock:
            cached = self._prompts.get(key)
            if cached is not None:
                self._prompts.move_to_end(key)
                return cached

        # Расчёт вне блокировки: очередь последовательная, но повторный синтез
        # той же реплики не должен ждать чужой пересчёт.
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(path),
            ref_text="" if x_vector_only else str(ref_text or ""),
            x_vector_only_mode=bool(x_vector_only),
        )
        with self._prompts_lock:
            self._prompts[key] = prompt
            self._prompts.move_to_end(key)
            while len(self._prompts) > config.QWEN_PROMPT_CACHE_SIZE:
                self._prompts.popitem(last=False)
        return prompt

    # -- синтез ----------------------------------------------------------------
    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        """Qwen не понимает `+`-ударения: текст приходит уже подготовленным пайплайном."""
        with self._infer_lock:
            try:
                return self._infer(text, ref_audio_path, ref_text, speed, params)
            except Exception as exc:
                if self._device != "mps":
                    raise
                # Часть операций на MPS может падать — повторяем на CPU, как XTTS.
                logger.warning(
                    "MPS упал в Qwen3-TTS (%s). Перезагружаю %s на CPU.", exc, self.info.label
                )
                with self._load_lock:
                    self._model = None
                    self._load_model("cpu")
                return self._infer(text, ref_audio_path, ref_text, speed, params)

    def _infer(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        """Один вызов `generate_voice_clone` плюс приведение звука к контракту.

        Приведение нужно по двум осям:

        * **sample rate.** Куски всех движков склеиваются напрямую, поэтому
          чужой sample rate — это не «чуть другой звук», а брак сборки. Если
          модель отдала не 24 кГц, звук ресемплируется, а расхождение попадает
          в лог: молчаливо приводить частоту нельзя, но и терять кусок из-за неё
          незачем.
        * **скорость.** Qwen не принимает `speed` как ручку генерации, поэтому
          замедление и ускорение делаются time-stretch'ем уже готового звука:
          он меняет длительность, но не высоту голоса.
        """
        seed = params.get("seed")
        if seed is not None:
            import torch

            # Сид ставится перед `generate`: с ним вариант куска воспроизводим,
            # а без него выбор «какой вариант оставить» был бы сравнением вслепую.
            torch.manual_seed(int(seed))

        x_vector_only = str(params.get(ENGINE_MODE_KEY) or "") == _X_VECTOR_MODE
        prompt = self._clone_prompt(ref_audio_path, ref_text, x_vector_only)
        wavs, sample_rate = self._model.generate_voice_clone(
            text=text,
            language=config.QWEN_LANGUAGE,
            voice_clone_prompt=prompt,
            temperature=float(params["temperature"]),
            top_p=float(params["top_p"]),
            repetition_penalty=float(params["repetition_penalty"]),
            max_new_tokens=int(params["max_new_tokens"]),
        )
        waveform = self._to_numpy(wavs)
        waveform = self._apply_speed(waveform, sample_rate, speed)
        waveform = self._ensure_rate(waveform, sample_rate)
        return waveform, SAMPLE_RATE

    @staticmethod
    def _to_numpy(wavs) -> np.ndarray:
        """Первый канал ответа модели в одномерный float32 — без копий, если можно."""
        first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(first, "detach"):
            first = first.detach().to("cpu").float().numpy()
        return np.asarray(first, dtype=np.float32).reshape(-1)

    @staticmethod
    def _apply_speed(waveform: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        """Замедление/ускорение без изменения высоты голоса (time-stretch)."""
        if not speed or abs(float(speed) - 1.0) < 1e-3:
            return waveform
        import librosa

        return np.asarray(
            librosa.effects.time_stretch(waveform, rate=float(speed)), dtype=np.float32
        )

    @staticmethod
    def _ensure_rate(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Приводит звук к 24 кГц, если модель отдала другую частоту."""
        if int(sample_rate) == SAMPLE_RATE:
            return waveform
        logger.warning(
            "Qwen3-TTS вернул %s Гц вместо %s — ресемплирую кусок",
            sample_rate,
            SAMPLE_RATE,
        )
        import librosa

        return np.asarray(
            librosa.resample(waveform, orig_sr=int(sample_rate), target_sr=SAMPLE_RATE),
            dtype=np.float32,
        )


def create_qwen_engine() -> QwenTTSEngine:
    """Фабрика для реестра: путь к весам берётся из конфига в момент вызова."""
    return QwenTTSEngine(config.QWEN_BASE_DIR)
