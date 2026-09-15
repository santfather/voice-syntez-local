"""Пути, дефолтные параметры и настройки окружения (MPS / лимиты потоков)."""

import os
from pathlib import Path

# --- Лимиты потоков для сторонних thread-пулов ---------------------------------
# OMP/OpenBLAS/MKL/VecLib/NumExpr читают эти переменные в момент загрузки своей
# нативной библиотеки, то есть при первом `import numpy` / `import torch` /
# `import librosa`. `torch.set_num_threads()` такие пулы не ограничивает: без этих
# переменных они инициализируются по числу ядер и обходят настройку torch.
# `backend/__init__.py` импортирует config первым, поэтому к моменту первого
# импорта numpy в любом подмодуле значения уже выставлены.
_THREAD_LIMIT = os.environ.get("TTS_THREAD_LIMIT", "4")
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _THREAD_LIMIT)

# --- MPS: разрешаем фолбэк неподдерживаемых операций на CPU -------------------
# Должно быть выставлено до первого импорта torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --- Пути ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", BASE_DIR / "models"))
VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR", BASE_DIR / "voices"))
OUTPUT_DIR = Path(os.environ.get("TTS_OUTPUT_DIR", BASE_DIR / "output"))
FRONTEND_DIR = BASE_DIR / "frontend"
VOICES_JSON = VOICES_DIR / "voices.json"

for _d in (MODELS_DIR, VOICES_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Веса модели --------------------------------------------------------------
# Misha24-10/F5-TTS_RUSSIAN -> F5TTS_v1_Base_accent_tune (полная разметка ударений)
HF_REPO_ID = os.environ.get("TTS_HF_REPO_ID", "Misha24-10/F5-TTS_RUSSIAN")
HF_CKPT_PATH = os.environ.get(
    "TTS_HF_CKPT_PATH", "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors"
)
HF_VOCAB_PATH = os.environ.get("TTS_HF_VOCAB_PATH", "F5TTS_v1_Base/vocab.txt")

# Локальные пути (структура папок повторяет HF-репозиторий; если файлов нет —
# будут докачаны из HF при старте)
CKPT_FILE = Path(os.environ.get("TTS_CKPT_FILE", MODELS_DIR / HF_CKPT_PATH))
VOCAB_FILE = Path(os.environ.get("TTS_VOCAB_FILE", MODELS_DIR / HF_VOCAB_PATH))

# Конфиг архитектуры F5-TTS, под который обучен чекпоинт
F5_MODEL_NAME = os.environ.get("TTS_F5_MODEL_NAME", "F5TTS_v1_Base")
VOCAB_SIZE = 10000

# --- Ресурсы ------------------------------------------------------------------
# Не занимать все P-ядра разом: модель крутится строго последовательно.
# По умолчанию совпадает с TTS_THREAD_LIMIT (см. блок лимитов потоков в начале файла).
TORCH_NUM_THREADS = int(os.environ.get("TTS_TORCH_THREADS", _THREAD_LIMIT))
# Таймаут на синтез одного куска (секунды). Не занижать: самый долгий честный
# инференс на MPS в этом проекте — ~150 с (реплика 600 знаков, nfe_step=32),
# поэтому дефолт взят с четырёхкратным запасом.
CHUNK_TIMEOUT_SEC = float(os.environ.get("TTS_CHUNK_TIMEOUT_SEC", "600"))
# Потолок памяти процесса (МБ): при устойчивом превышении watchdog прерывает
# текущую задачу (сам процесс не убивается).
MAX_RSS_MB = int(os.environ.get("TTS_MAX_RSS_MB", "4096"))
# Сколько хранить готовые файлы в output/
OUTPUT_TTL_HOURS = float(os.environ.get("TTS_OUTPUT_TTL_HOURS", "24"))
# Максимальная длина одной реплики (символов) — защита от случайных "полотен"
MAX_REPLICA_CHARS = int(os.environ.get("TTS_MAX_REPLICA_CHARS", "600"))
# Максимальная длина сплошного текста для одной задачи (символов).
# ~50 000 знаков — это примерно час готового аудио; защита от случайной
# вставки книги целиком в режиме «Сплошной текст».
MAX_TEXT_CHARS = int(os.environ.get("TTS_MAX_TEXT_CHARS", "50000"))

# --- Ударения (RUAccent) ------------------------------------------------------
# Размер модели омографов: turbo3.1 — то, что рекомендует README ruaccent.
# Крупнее только big_poetry, но она обучалась на стихах. Параметр влияет
# исключительно на выбор варианта для слов-омографов (зáмок/замóк).
OMOGRAPH_MODEL_SIZE = os.environ.get("TTS_OMOGRAPH_MODEL_SIZE", "turbo3.1")

# --- Дефолтные параметры синтеза ---------------------------------------------
DEFAULT_NFE_STEP = int(os.environ.get("TTS_NFE_STEP", "32"))
DEFAULT_CFG_STRENGTH = float(os.environ.get("TTS_CFG_STRENGTH", "2.0"))
DEFAULT_SPEED = 1.0
DEFAULT_SWAY_SAMPLING_COEF = -1.0
DEFAULT_CROSS_FADE_DURATION = 0.15
DEFAULT_PAUSE_MS = 400
DEFAULT_OUTPUT_FORMAT = "mp3"

# Фраза для прослушивания голоса во вкладке «Голоса»
DEFAULT_PREVIEW_TEXT = "Привет! Так звучит этот голос в диалоге."

# --- Слоты голосов в тексте ---------------------------------------------------
# Маркер «(1)» в диалоге = слот 1. Больше MAX_SLOT_NUMBER цифр — это уже не слот
# (защита от «в (2024) году»), а обычный текст в скобках.
DEFAULT_SLOT = 1
MAX_SLOT_NUMBER = int(os.environ.get("TTS_MAX_SLOT", "20"))

# Границы для валидации параметров, приходящих с фронта
SPEED_RANGE = (0.5, 2.0)
CFG_RANGE = (1.0, 4.0)
NFE_ALLOWED = (8, 16, 32)
PAUSE_MS_RANGE = (0, 5000)
CROSS_FADE_RANGE = (0.0, 1.0)

_torch = None


def configure_torch():
    """Импортирует torch и один раз ограничивает число CPU-потоков."""
    global _torch
    if _torch is not None:
        return _torch
    import torch

    torch.set_num_threads(TORCH_NUM_THREADS)
    _torch = torch
    return _torch


def pick_device() -> str:
    """mps, если доступен, иначе cpu."""
    torch = configure_torch()
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
