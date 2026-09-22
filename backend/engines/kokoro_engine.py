"""Движок Kokoro-ru: пресетный синтез встроенными голосами модели.

Русского в официальном Kokoro-82M нет — модель объявляет восемь языков, и ни
одного славянского. Поэтому движок работает на файнтюне `zaakirio/kokoro-ru`
(82 млн параметров, 24 кГц, CPU) и состоит из трёх частей, каждая из которых
лежит в каталоге модели:

* `kokoro-ru-v2-*.pth` — веса `KModel` из штатного пакета `kokoro`. Пакет не
  патчится: `KModel` language-blind, русская фонетика приходит снаружи;
* `ru_g2p.py` — произносительный словарь репозитория: RUAccent расставляет
  ударения и восстанавливает «ё», затем espeak-ng фонемизирует;
* `espeak-data/` — пересобранный под ударения словарь espeak-ng. Штатный
  игнорирует комбинирующий акут, и ударения теряются молча — поэтому словарь
  репозитория обязателен, а не «желателен».

Четыре особенности, ради которых этот адаптер существует отдельно:

* **Голос выбирается по полу голоса.** Модель говорит собственными голосами
  (sveta, masha, dima), клонирования у неё нет. Пол пользователь уже назвал при
  создании голоса, поэтому второй ручки «каким встроенным голосом говорить» в
  интерфейсе нет — см. `_voice_for_gender`.
* **Референс игнорируется.** `supports_cloning=False`, значит пайплайн и не
  спрашивает референс (`_reference_for`), но аргументы `ref_audio_path` и
  `ref_text` в `_synthesize` всё равно приходят: карточка голоса требует их для
  других движков. Здесь они не используются.
* **Длинный текст режется по фонемам.** Контекст `KModel` — 512 позиций, и
  строка длиннее падает `AssertionError` внутри модели. Реплика проекта (до
  ~600 знаков) в потолок не влезает, поэтому фонемная строка режется по границам
  слов и синтезируется по частям (см. `_split_phonemes`).
* **Скорость считает модель.** `KModel.forward` делит предсказанные длительности
  на `speed`, поэтому ручка «Скорость речи» уходит в инференс как есть, без
  time-stretch готового звука (в отличие от Qwen3-TTS, см. `qwen_engine`).
"""

import importlib.util
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import config
from .base import (
    ENGINE_INFOS,
    ENGINE_KOKORO,
    ENGINE_MODE_DRAFT,
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_MODE_KEY,
    SAMPLE_RATE,
    STATE_FAILED,
    STATE_LOADING,
    STATE_READY,
    SynthesisEngine,
    release_torch_memory,
)

logger = logging.getLogger(__name__)

# Имя, под которым модуль словаря загружается в текущий процесс. Своё, а не
# `ru_g2p`: одноимённый чужой модуль в `sys.path` не должен подменять словарь
# модели.
_G2P_MODULE = "kokoro_ru_g2p"

# Чекпоинт под встроенный голос. sveta и masha делят одну нарезку и различаются
# только голосовым пакетом, dima обучен отдельно — см. карточку модели.
_WEIGHTS_BY_VOICE = {
    config.KOKORO_VOICE_FEMALE: config.KOKORO_BASE_WEIGHTS,
    config.KOKORO_VOICE_OTHER: config.KOKORO_BASE_WEIGHTS,
    config.KOKORO_VOICE_MALE: config.KOKORO_DIMA_WEIGHTS,
}


def _missing_dependency(package: str, exc: ImportError) -> RuntimeError:
    """Понятная ошибка вместо «No module named ...» — с командой установки."""
    return RuntimeError(
        f"Пакет `{package}` для движка Kokoro-ru не установлен. Его зависимости "
        "опциональны и ставятся отдельно от основного набора: "
        "`./venv/bin/pip install -r requirements-kokoro.txt` (см. docs/install.md)."
    )


def _pattern_name(pattern: str) -> str:
    """Имя файла из шаблона: `voices/*.pt` и `*.pt` ищутся одинаково (см. `_match`)."""
    return Path(pattern).name


def _match(parent: Path, pattern: str) -> bool:
    """Есть ли в каталоге файл по шаблону — на любой глубине.

    Тот же приём, что в `model_manager`: `huggingface_hub` раскладывает репозиторий
    по вложенным папкам, и жёсткий относительный путь сломался бы при обновлении
    модели.
    """
    if (parent / pattern).is_file():
        return True
    return any(found.is_file() for found in parent.rglob(_pattern_name(pattern)))


def resolve_model_dir(root: Path, *, required: tuple[str, ...], anchor: str) -> Path | None:
    """Каталог, в котором лежат все обязательные файлы модели, или `None`.

    Поиск идёт по «якорю» (`config.json`): в HF-репозитории он один и всегда рядом
    с весами, поэтому именно он задаёт каталог модели, а остальные файлы
    проверяются относительно него. Набор обязательных файлов — общий с паспортом
    модели (`config.KOKORO_REQUIRED_FILES`), иначе «модель установлена» на вкладке
    «Модели» и «модель загружается» здесь могли бы разойтись.
    """
    if not root.is_dir():
        return None
    for anchor_file in sorted(root.rglob(anchor)):
        parent = anchor_file.parent
        if all(_match(parent, name) for name in required):
            return parent
    return None


def _load_g2p_module(path: Path):
    """Модуль `ru_g2p.py` из каталога модели — как модуль, а не как текст.

    Загружается по пути, а не добавлением каталога в `sys.path`: каталог модели не
    должен становиться местом, откуда в процесс попадает любой одноимённый модуль.
    Своё имя важно ещё и потому, что словарь сам находит `espeak-data/` и
    `kokoro-config.json` рядом с собой (`Path(__file__).parent`) — при загрузке по
    пути это ровно каталог модели.
    """
    spec = importlib.util.spec_from_file_location(_G2P_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить произносительный словарь Kokoro-ru: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise _missing_dependency("misaki", exc) from exc
    return module


def _split_phonemes(phonemes: str, limit: int) -> list[str]:
    """Режет фонемную строку на части не длиннее `limit`, по границам слов.

    У `KModel` жёсткий потолок контекста (`len(input_ids) + 2 <= 512`), и строка
    длиннее падает `AssertionError` внутри модели, а не ошибкой синтеза. Режем по
    пробелам: фонемная строка espeak уже разбита на слова, а склейка кусков идёт
    напрямую — разрез посреди слова был бы слышен как обрыв.
    """
    pieces: list[str] = []
    current = ""
    for word in phonemes.split(" "):
        if not word:
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
        while len(current) > limit:
            # Слово длиннее потолка: режем по символам. Артефакт на одном слове
            # лучше, чем отказ синтезировать реплику целиком.
            pieces.append(current[:limit])
            current = current[limit:]
    if current:
        pieces.append(current)
    return pieces


class KokoroEngine(SynthesisEngine):
    """Один словарь на все голоса и по одной модели на каждый чекпоинт."""

    info = ENGINE_INFOS[ENGINE_KOKORO]

    def __init__(self, base_dir: Path | None = None) -> None:
        super().__init__()
        self._base_dir = Path(base_dir if base_dir is not None else config.KOKORO_DIR)
        self._model_dir: Path | None = None
        self._g2p = None
        self._models: dict[str, object] = {}
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        # Голосовой пакет — стилевые векторы голоса; ключ включает метаданные
        # файла, потому что подмена файла под тем же именем обязана дать новое
        # звучание, а не звучание старого голоса.
        self._voices: OrderedDict[tuple, object] = OrderedDict()
        self._voices_lock = threading.Lock()

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def voice_names(self) -> tuple[str, ...]:
        """Все встроенные голоса модели — для диагностики и тестов."""
        return tuple(dict.fromkeys(_WEIGHTS_BY_VOICE))

    # -- загрузка --------------------------------------------------------------
    def load(self) -> None:
        """Поднимает произносительный словарь — общую часть для всех голосов.

        Веса приходят позже, и это не оптимизация: чекпоинт зависит от пола
        голоса, а `load()` вызывается до первой реплики, когда голос ещё не
        известен (пайплайн прогревает движки заранее, см. `_load_engines`).
        Поэтому `ready` здесь означает «готов работать»: словарь espeak-ng и
        RUAccent подняты, выбор чекпоинта — первый шаг синтеза (`_ensure_model`).
        """
        with self._load_lock:
            if self.is_loaded and self._g2p is not None:
                return
            self._mark(STATE_LOADING)
            try:
                model_dir = self._resolve_dir()
                module = _load_g2p_module(model_dir / "ru_g2p.py")
                logger.info("Загружаю %s: словарь %s", self.info.label, model_dir)
                # Размер модели омографов берётся из настройки проекта: он же
                # используется акцентуатором пайплайна (`backend/accentizer.py`),
                # и разъезжаться этим двум местам незачем.
                self._g2p = module.RuG2P(omograph_model_size=config.OMOGRAPH_MODEL_SIZE)
                self._model_dir = model_dir
            except Exception as exc:
                logger.error("Не удалось подготовить %s: %s", self.info.label, exc)
                self._g2p = None
                self._mark(STATE_FAILED, exc)
                raise
            self._mark(STATE_READY)

    def _resolve_dir(self) -> Path:
        """Каталог модели по обязательным файлам. Нет файлов — понятная ошибка."""
        model_dir = resolve_model_dir(
            self._base_dir,
            required=config.KOKORO_REQUIRED_FILES,
            anchor="config.json",
        )
        if model_dir is None:
            raise RuntimeError(
                f"Веса Kokoro-ru не найдены в {self._base_dir}: нужен каталог модели с "
                "чекпоинтами (*.pth), голосовыми пакетами (voices/*.pt) и "
                "произносительным словарём ru_g2p.py. Скачайте модель во вкладке "
                "«Модели» или через `huggingface-cli download` (см. README)."
            )
        return model_dir

    def _device_for(self) -> str:
        """Устройство модели: по умолчанию CPU (см. `config.KOKORO_DEVICE`)."""
        wanted = (config.KOKORO_DEVICE or "cpu").strip().lower()
        if wanted in ("cpu", "mps", "cuda"):
            return wanted
        logger.warning("Kokoro-ru: неизвестное устройство «%s» — работаю на CPU", wanted)
        return "cpu"

    def _ensure_model(self, weights_name: str):
        """Модель под нужный чекпоинт: поднимается один раз и живёт до выгрузки."""
        with self._load_lock:
            cached = self._models.get(weights_name)
            if cached is not None:
                return cached
            model_dir = self._model_dir or self._resolve_dir()
            weights = model_dir / weights_name
            if not weights.is_file():
                raise RuntimeError(f"Чекпоинт Kokoro-ru не найден: {weights}")
            device = self._device_for()
            try:
                from kokoro import KModel
            except ImportError as exc:
                raise _missing_dependency("kokoro", exc) from exc
            logger.info(
                "Загружаю %s: %s на %s", self.info.label, weights.name, device
            )
            try:
                model = KModel(
                    # `repo_id` не нужен для работы — он передаётся, чтобы пакет
                    # не подставлял свой репозиторий по умолчанию. Сеть не
                    # понадобится: и конфигурация, и веса заданы путями.
                    repo_id=config.KOKORO_REPO_ID,
                    config=str(model_dir / "config.json"),
                    model=str(weights),
                    # Декодер istftnet считает комплексным преобразованием, а на
                    # MPS комплексные операции поддержаны не везде: вне CPU
                    # считаем вещественно. Численно это не то же самое, поэтому
                    # устройство и остаётся осознанным выбором (по умолчанию CPU).
                    disable_complex=device != "cpu",
                ).eval()
                model.to(device)
            except Exception as exc:
                logger.error("Не удалось загрузить %s на %s: %s", self.info.label, device, exc)
                self._mark(STATE_FAILED, exc)
                raise
            self._models[weights_name] = model
            self._device = device
            logger.info("%s готов (%s)", self.info.label, weights_name)
            return model

    def _release(self) -> None:
        """Обнуляет словарь, модели и голосовые пакеты, затем возвращает память.

        Вызывается только когда активных синтезов нет, поэтому локи свободны.
        Голосовые пакеты — тензоры, и без их очистки выгрузка освободила бы веса,
        но не то, что прочитано из `voices/`.
        """
        with self._load_lock:
            self._g2p = None
            self._models.clear()
            self._model_dir = None
            self._device = None
        with self._voices_lock:
            self._voices.clear()
        release_torch_memory()

    # -- встроенные голоса -----------------------------------------------------
    @staticmethod
    def _voice_for_gender(gender: str | None) -> str:
        """Встроенный голос по полу голоса (см. `config.KOKORO_VOICE_*`).

        Мужскому голосу достаётся единственная мужская нарезка файнтюна, женскому
        и «другому» — голоса женского чекпоинта (см. карточку модели).
        """
        return {
            "female": config.KOKORO_VOICE_FEMALE,
            "male": config.KOKORO_VOICE_MALE,
        }.get(str(gender or "").strip().lower(), config.KOKORO_VOICE_OTHER)

    def _voice_pack(self, model_dir: Path, voice: str):
        """Голосовой пакет `voices/<voice>.pt` — стилевые векторы встроенного голоса."""
        path = model_dir / "voices" / f"{voice}.pt"
        try:
            stat = path.stat()
            key: tuple = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            # Файл не читается — понятную ошибку отдаст проверка ниже, а ключ
            # без метаданных не даст закешировать чужой голос.
            key = (str(path), 0.0, 0)

        with self._voices_lock:
            cached = self._voices.get(key)
            if cached is not None:
                self._voices.move_to_end(key)
                return cached

        if not path.is_file():
            raise RuntimeError(
                f"Голосовой пакет «{voice}» не найден: {path}. Перекачайте модель во "
                "вкладке «Модели» — пакеты голосов лежат в репозитории рядом с весами."
            )
        import torch

        # `weights_only=False` — так пакет читает сам репозиторий модели: файл
        # содержит не только тензоры. Источник — локальный каталог модели.
        pack = torch.load(str(path), map_location="cpu", weights_only=False)
        with self._voices_lock:
            self._voices[key] = pack
            self._voices.move_to_end(key)
            while len(self._voices) > config.KOKORO_VOICE_CACHE_SIZE:
                self._voices.popitem(last=False)
        return pack

    # -- синтез ----------------------------------------------------------------
    def _synthesize(
        self, text: str, ref_audio_path: str, ref_text: str, speed: float, params: dict
    ) -> tuple[np.ndarray, int]:
        """Синтез встроенным голосом. Референс голоса не используется намеренно."""
        with self._infer_lock:
            voice = self._voice_for_gender(params.get("gender"))
            model_dir = self._model_dir or self._resolve_dir()
            weights = _WEIGHTS_BY_VOICE.get(voice, config.KOKORO_BASE_WEIGHTS)
            model = self._ensure_model(weights)
            pack = self._voice_pack(model_dir, voice)
            phonemes = self._phonemize(text, params)
            if not phonemes.strip():
                raise RuntimeError(
                    f"Kokoro-ru не получил ни одной фонемы из текста ({len(text)} знаков)"
                )
            pieces = _split_phonemes(phonemes, config.KOKORO_MAX_PHONEMES)
            logger.info(
                "Kokoro-ru: %s фонем в %s частях, голос %s",
                len(phonemes),
                len(pieces),
                voice,
            )
            waveform = self._infer(model, pack, pieces, speed)
            # Kokoro отдаёт звук без частоты: 24 кГц заданы архитектурой модели
            # (istftnet), и это же значение требует контракт движков — куски
            # склеиваются напрямую (см. `SAMPLE_RATE`).
            return waveform, SAMPLE_RATE

    def _phonemize(self, text: str, params: dict) -> str:
        """Фонемная строка по режиму работы (см. паспорт движка).

        Режим здесь — не числовая ручка, а выбор пути через произносительный
        словарь: `draft` обходится без RUAccent (ударения расставляет espeak),
        `experimental` выключает редукцию гласных. Обе разницы задаются одним
        словарём, поэтому пересобирать его ради другого режима не нужно.
        """
        mode = str(params.get(ENGINE_MODE_KEY) or "")
        # `reduction` — открытый атрибут RuG2P и ровно та разница, которой он
        # управляет в разметке (v1 против v2 файнтюна).
        self._g2p.reduction = mode != ENGINE_MODE_EXPERIMENTAL
        if mode == ENGINE_MODE_DRAFT:
            phonemes, oov = self._g2p.phonemize_accented(text)
        else:
            phonemes, oov = self._g2p.phonemize(text)
        if oov:
            # Символ вне словаря модели отбрасывается при токенизации: слово
            # звучит неверно, и без строки в логе это выглядело бы как брак модели.
            logger.warning(
                "Kokoro-ru: символы вне словаря модели (%s) — произношение в этих "
                "местах может быть неверным",
                "".join(sorted(oov)),
            )
        return phonemes

    def _infer(self, model, pack, pieces: list[str], speed: float) -> np.ndarray:
        """По одному проходу модели на часть: у неё жёсткий потолок длины строки."""
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for piece in pieces:
                if not piece.strip():
                    continue
                # Стилевой вектор выбирается по длине фонемной строки; индекс
                # ограничен размером пакета, чтобы длинная часть не вышла за него.
                style = pack[min(len(piece), len(pack)) - 1]
                if style.dim() == 1:
                    # Пакеты Kokoro публикуются и как [510, 1, 256], и как
                    # [510, 256], а модель ждёт двумерный тензор (`ref_s[:, 128:]`).
                    style = style.unsqueeze(0)
                audio = model(piece, style, float(speed), return_output=True).audio
                chunks.append(
                    np.asarray(audio.detach().cpu().float().numpy(), dtype=np.float32).reshape(-1)
                )
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks)


def create_kokoro_engine() -> KokoroEngine:
    """Фабрика для реестра: путь к весам берётся из конфига в момент вызова."""
    return KokoroEngine(config.KOKORO_DIR)
