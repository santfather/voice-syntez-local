"""Пути, дефолтные параметры и настройки окружения (MPS / лимиты потоков)."""

import logging
import os
from pathlib import Path

logger = logging.getLogger("tts.config")

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

# --- MPS: потолки памяти ------------------------------------------------------
# Аудит §11.3/§11.4: без явных потолков модель может занять всю unified memory —
# на 8–16 ГБ это путь к kernel/GPU panic, а не к деградации. Значения — из
# матрицы §11.3: `HIGH = 1.0` (именно `1.0`, а не `0.0`: ноль в этой переменной
# означает «без потолка», а не «ноль памяти»), `LOW = 0.9` — доля, с которой
# аллокатор начинает отдавать драйверу освобождённое.
#
# Читает их нативная часть torch при инициализации MPS, поэтому выставлены они
# здесь — до первого импорта torch, как и фолбэк выше. `setdefault`, а не
# присваивание: пользователь вправе опустить или поднять потолок своими
# значениями, и перезапуск для этого не нужен.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")

# --- Пути ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", BASE_DIR / "models"))
VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR", BASE_DIR / "voices"))
OUTPUT_DIR = Path(os.environ.get("TTS_OUTPUT_DIR", BASE_DIR / "output"))
FRONTEND_DIR = BASE_DIR / "frontend"
VOICES_JSON = VOICES_DIR / "voices.json"

# --- Постоянные проекты (SQLite) ----------------------------------------------
# Отдельно от output/: готовые файлы задач живут по TTL, а проект должен
# переживать и перезапуск, и очистку, иначе «открыть вчерашний диалог» невозможно.
DATA_DIR = Path(os.environ.get("TTS_DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.environ.get("TTS_DB_PATH", DATA_DIR / "voice_syntez.db"))
# Куски (takes) проектов: по одному wav на реплику, чтобы их можно было
# прослушать и заменить отдельно от итогового файла.
PROJECTS_OUTPUT_DIR = OUTPUT_DIR / "projects"
# Сравнение движков: по одному wav на движок, имя — {run_id}-{engine}.wav.
# Отдельный каталог, а не файлы в корне output/: TTL-очистка проходит только по
# файлам верхнего уровня (см. audio_pipeline.cleanup_output), поэтому результат
# сравнения не пропадает посреди сессии, пока пользователь его слушает.
BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"
# Диагностические архивы (качество озвучки): проект, настройки, план коротких
# реплик, take'ы и аудио в одном ZIP. Отдельный каталог по той же причине, что и
# у сравнения движков: архив нужен пользователю после разбора, а не до TTL.
DIAGNOSTICS_DIR = Path(os.environ.get("TTS_DIAGNOSTICS_DIR", OUTPUT_DIR / "diagnostics"))
# Сколько аудио кладётся в архив. Предел нужен не ради места на диске, а ради
# того, чтобы архив открывался: полный набор take'ов длинного диалога плюс
# референсы легко переваливает за гигабайт, и такой файл уже не отправить.
DIAGNOSTICS_MAX_AUDIO_MB = float(os.environ.get("TTS_DIAGNOSTICS_MAX_AUDIO_MB", "400"))
# Сколько последних строк лога попадает в архив. Лог — главный источник причин
# («граница цели не найдена», «attempts_exhausted»), но целиком он не нужен.
DIAGNOSTICS_LOG_LINES = int(os.environ.get("TTS_DIAGNOSTICS_LOG_LINES", "600"))
# Лог приложения: run.sh пишет stdout в этот файл. Пустой путь или отсутствующий
# файл — не ошибка: архив собирается без логов, а причина указывается в README.
# --- Записи пользователя («Сам себе звукорежиссер») ----------------------------
# Отдельный каталог, а не `voices/`: там лежат референсы TTS-голосов, и смешивать
# с ними сырые записи диалога нельзя (и по смыслу, и по жизненному циклу).
RECORDINGS_DIR = Path(os.environ.get("TTS_RECORDINGS_DIR", BASE_DIR / "recordings"))
# Пределы загрузки одной записи. Ограничения — часть контракта безопасности:
# без них один upload может заполнить диск или подвесить декодер.
MAX_RECORDING_BYTES = int(os.environ.get("TTS_RECORDING_MAX_BYTES", str(64 * 1024 * 1024)))
MAX_RECORDING_SEC = float(os.environ.get("TTS_RECORDING_MAX_SEC", "300"))
# Настройки обработки записанного голоса. UI показывает уже (0.75–1.25 и ±6), а
# backend допускает более широкий диапазон: голову за голову не отвечает, но и не
# даёт выйти за пределы, где time-stretch и pitch-shift перестают звучать.
RECORDING_SPEED_RANGE = (
    float(os.environ.get("TTS_RECORDING_SPEED_MIN", "0.5")),
    float(os.environ.get("TTS_RECORDING_SPEED_MAX", "2.0")),
)
RECORDING_PITCH_RANGE = (
    float(os.environ.get("TTS_RECORDING_PITCH_MIN", "-12")),
    float(os.environ.get("TTS_RECORDING_PITCH_MAX", "12")),
)
# Формат финального файла записи. Задача фазы — единый MP3, поэтому формат не
# выбирается пользователем, а зафиксирован; WAV остаётся внутренним мастером.
RECORDING_OUTPUT_FORMAT = os.environ.get("TTS_RECORDING_FORMAT", "mp3")

LOG_PATH = Path(os.environ.get("TTS_LOG_PATH", BASE_DIR / "logs" / "voice_syntez.log"))
# Ротация журнала (F-F1). До неё файл рос без предела: на момент аудита он уже
# подходил к мегабайту, а для разбора падения нужен свежий хвост, а не начало
# месяца. Пять копий по 5 МБ — запас на редкую поломку без накопления старых логов.
LOG_MAX_BYTES = int(os.environ.get("TTS_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("TTS_LOG_BACKUP_COUNT", "5"))

# --- Пределы размера запроса (SECURITY) ---------------------------------------
# Тело запроса читается в память целиком ещё до валидации, поэтому предел нужен
# на входе, а не в обработчике: без него один запрос на гигабайты укладывает
# сервер (сервис слушает только loopback, но это снижает риск, а не отменяет
# правило). Тело без файла — это текст диалога, ему мегабайтов хватает с запасом;
# загрузка файла идёт multipart, и у неё предел выше — по самому крупному из
# загружаемых типов (запись голоса, `MAX_RECORDING_BYTES`) с запасом на обвязку.
MAX_JSON_BODY_BYTES = int(os.environ.get("TTS_MAX_JSON_BODY_BYTES", str(8 * 1024 * 1024)))
MAX_UPLOAD_BYTES = int(os.environ.get("TTS_MAX_UPLOAD_BYTES", str(96 * 1024 * 1024)))
# Архив проекта переносит take'ы и референсы, поэтому его предел выше загрузки
# одного файла. Распакованный объём ограничен отдельно: «архив-бомба» сжимается
# в мегабайты, а разворачивается в гигабайты, и предел на сжатый файл от этого не
# защищает.
MAX_ARCHIVE_BYTES = int(os.environ.get("TTS_MAX_ARCHIVE_BYTES", str(512 * 1024 * 1024)))
ARCHIVE_MAX_ENTRIES = int(os.environ.get("TTS_ARCHIVE_MAX_ENTRIES", "20000"))
ARCHIVE_MAX_UNCOMPRESSED_BYTES = int(
    os.environ.get("TTS_ARCHIVE_MAX_UNCOMPRESSED_BYTES", str(2 * 1024 * 1024 * 1024))
)

for _d in (
    RECORDINGS_DIR,
    MODELS_DIR,
    VOICES_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    PROJECTS_OUTPUT_DIR,
    BENCHMARKS_DIR,
    DIAGNOSTICS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Режим проекта: диалог со спикерами или сплошной текст одним голосом.
PROJECT_MODE_DIALOGUE = "dialogue"
PROJECT_MODE_TEXT = "text"
PROJECT_MODES = (PROJECT_MODE_DIALOGUE, PROJECT_MODE_TEXT)

# Статус проекта и статус отдельной реплики в нём.
PROJECT_STATUS_DRAFT = "draft"
PROJECT_STATUS_RENDERING = "rendering"
PROJECT_STATUS_RENDERED = "rendered"
PROJECT_STATUS_ERROR = "error"
REPLICA_STATUS_PENDING = "pending"
REPLICA_STATUS_RENDERED = "rendered"
# Реплика прямо сейчас синтезируется. Отдельный статус, а не «pending»: после
# падения приложения видно, на какой реплике процесс остановился, — иначе
# «зависшие» реплики не отличить от тех, что ещё не начинали.
REPLICA_STATUS_RENDERING = "rendering"
# Синтез реплики прерван падением воркера или перезапуском приложения: аудио нет
# (или есть не полностью), и это не «pending» — попытку уже делали, и причина
# записана в диагностике (worker_crashes, projects.last_error).
REPLICA_STATUS_INTERRUPTED = "interrupted"
REPLICA_STATUSES = (
    REPLICA_STATUS_PENDING,
    REPLICA_STATUS_RENDERING,
    REPLICA_STATUS_RENDERED,
    REPLICA_STATUS_INTERRUPTED,
)

# Причины, которые записываются в базу при восстановлении после перезапуска
# (creash_report §15). Тексты на русском: их читает пользователь в карточке
# проекта, и «rendering» без объяснения выглядел бы зависанием.
RECOVERY_RENDER_MESSAGE = (
    "Рендер прерван перезапуском приложения: очередь задач живёт в памяти, "
    "поэтому сборка не продолжилась. Запустите синтез заново."
)
RECOVERY_ANALYSIS_MESSAGE = (
    "Анализ текста прерван перезапуском приложения. Запустите анализ заново."
)
# Задача, осиротевшая при перезапуске: её запись в таблице `jobs` осталась в
# `queued`/`processing`, а самой задачи в памяти уже нет.
RECOVERY_QUEUE_MESSAGE = (
    "Задача прервана перезапуском приложения: очередь живёт в памяти, "
    "поэтому работа не продолжилась. Запустите её заново."
)

# Готовность текста — отдельная ось от статуса рендера. `PROJECT_STATUS_*`
# отвечает на «что с аудио» (идёт сборка, готово, упало), а `PROJECT_ANALYSIS_*`
# на «можно ли вообще синтезировать»: пока текст не проанализирован, рендер
# запрещён, и смешивать эти состояния значило бы потерять это различие.
PROJECT_ANALYSIS_RAW = "raw"
PROJECT_ANALYSIS_ANALYZING = "analyzing"
PROJECT_ANALYSIS_NEEDS_REVIEW = "needs_review"
PROJECT_ANALYSIS_READY = "ready"
PROJECT_ANALYSIS_ERROR = "error"
PROJECT_ANALYSIS_STATUSES = (
    PROJECT_ANALYSIS_RAW,
    PROJECT_ANALYSIS_ANALYZING,
    PROJECT_ANALYSIS_NEEDS_REVIEW,
    PROJECT_ANALYSIS_READY,
    PROJECT_ANALYSIS_ERROR,
)

# Готовность текста одной реплики: `pending` — стадии устарели или их нет,
# `done` — `final_text` актуален и его можно отправлять в модель, `error` —
# подготовить не удалось (причина остаётся в `analysis_error`).
REPLICA_ANALYSIS_PENDING = "pending"
REPLICA_ANALYSIS_DONE = "done"
REPLICA_ANALYSIS_ERROR = "error"

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

# --- Веса XTTS v2 (второй движок синтеза) -------------------------------------
# Базовая модель: coqui/XTTS-v2 — скачана целиком, чтобы первый синтез работал
# офлайн (каталог лежит в models/, а не в кеше huggingface_hub под именем хеша).
# Русский файнтюн Ftfyhh/xttsv2_banana: путь ищется рекурсивно, потому что
# huggingface_hub раскладывает репозиторий по вложенным папкам (model_banana/v2.0.2).
XTTS_BASE_DIR = Path(os.environ.get("TTS_XTTS_BASE_DIR", MODELS_DIR / "xtts_v2"))
XTTS_BANANA_DIR = Path(os.environ.get("TTS_XTTS_BANANA_DIR", MODELS_DIR / "xtts_v2_banana"))
# Русский — единственный язык, который нужен всем голосам проекта.
XTTS_LANGUAGE = os.environ.get("TTS_XTTS_LANGUAGE", "ru")
# Сколько секунд референса идёт в conditioning latents. Референс длиннее модель
# всё равно обрежет, а расчёт латентов кешируется на голос.
XTTS_COND_LEN_SEC = int(os.environ.get("TTS_XTTS_COND_LEN_SEC", "6"))
XTTS_MAX_REF_SEC = int(os.environ.get("TTS_XTTS_MAX_REF_SEC", "30"))
# Сколько голосов держать в кеше conditioning latents (по одному набору тензоров на голос).
XTTS_LATENTS_CACHE_SIZE = int(os.environ.get("TTS_XTTS_LATENTS_CACHE", "8"))

# --- Веса Qwen3-TTS (третий движок синтеза) -----------------------------------
# Две модели одного семейства, и обе нужны вместе: `-Base` синтезирует и умеет
# клонирование по 3 с референса, `-Tokenizer-12Hz` превращает её коды в звук.
# Пути локальные: `qwen-tts` умеет качать веса сам, но тогда первый синтез
# требует сети, а проект работает офлайн (см. `model_manager`).
QWEN_BASE_REPO_ID = os.environ.get(
    "TTS_QWEN_BASE_REPO_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
QWEN_TOKENIZER_REPO_ID = os.environ.get(
    "TTS_QWEN_TOKENIZER_REPO_ID", "Qwen/Qwen3-TTS-Tokenizer-12Hz"
)
QWEN_BASE_DIR = Path(os.environ.get("TTS_QWEN_BASE_DIR", MODELS_DIR / "qwen3_tts_base"))
QWEN_TOKENIZER_DIR = Path(
    os.environ.get("TTS_QWEN_TOKENIZER_DIR", MODELS_DIR / "qwen3_tts_tokenizer")
)
# Язык проекта. Qwen3-TTS принимает не код, а имя языка («Russian»), и по нему
# выбирает фонетику: «ru» она бы не поняла и ушла бы в автоопределение.
QWEN_LANGUAGE = os.environ.get("TTS_QWEN_LANGUAGE", "Russian")
# Потолок референса в секундах. Модель клонирует по 3 с; длинную запись она
# обрежет сама, но промпт считается на каждый новый файл, и платить за это
# минутами записи незачем.
QWEN_MAX_REF_SEC = int(os.environ.get("TTS_QWEN_MAX_REF_SEC", "15"))
# Сколько голосов держать в кеше clone-prompt. Промпт — это уже посчитанные
# фичи референса на устройстве модели, поэтому кеш маленький: у XTTS десяток
# наборов латентов стоит десятки мегабайт, здесь — сотни.
QWEN_PROMPT_CACHE_SIZE = int(os.environ.get("TTS_QWEN_PROMPT_CACHE", "4"))
# Принудительный отказ от MPS. Нужен и для отладки шима, и как аварийный тормоз:
# если проба устройства ошибается, пользователь должен уметь вернуться на CPU
# одной переменной, не удаляя модель.
QWEN_FORCE_CPU_ENV = "TTS_QWEN_FORCE_CPU"
# Устройство по умолчанию для Qwen, выясненное пробой. `qwen-tts` заявляет
# только CUDA, а на Apple Silicon остаётся MPS без гарантий: значение ниже —
# результат прогона `tools/qwen_feasibility.py`, а не предположение.
QWEN_DEVICE = os.environ.get("TTS_QWEN_DEVICE", "auto")
# Реализация attention. По умолчанию `eager`: flash-attention требует CUDA, а
# `sdpa` на MPS работает не для всех масок, и падение внутри модели выглядело бы
# как «движок сломался». Медленнее — но одинаково предсказуемо на обоих
# устройствах, а ускорение включается осознанно, после живой пробы.
QWEN_ATTN_IMPLEMENTATION = os.environ.get("TTS_QWEN_ATTN", "eager")

# --- Веса Kokoro-ru (четвёртый движок синтеза) --------------------------------
# Русского в официальном Kokoro-82M нет: модель объявляет восемь языков
# (`a,b,e,f,h,i,p,j,z`), и ни одного славянского среди них нет. Поэтому движок
# работает на комьюнити-файнтюне `zaakirio/kokoro-ru` — те же 82 млн параметров
# и 24 кГц, но русская фонетика. Репозиторий скачивается целиком (снапшотом):
# кроме двух чекпоинтов в нём лежат произносительный словарь `ru_g2p.py`,
# пересобранный под ударения `espeak-data/` и голосовые пакеты `voices/`, и по
# отдельным файлам этот набор не собрать.
KOKORO_REPO_ID = os.environ.get("TTS_KOKORO_REPO_ID", "zaakirio/kokoro-ru")
KOKORO_DIR = Path(os.environ.get("TTS_KOKORO_DIR", MODELS_DIR / "kokoro_ru"))
# Код языка для espeak-ng (`misaki_espeak.EspeakG2P(language=...)`), а не для
# интерфейса: русский здесь единственный, и другого файнтюн не обещает.
KOKORO_LANGUAGE = os.environ.get("TTS_KOKORO_LANGUAGE", "ru")
# Два чекпоинта одного релиза: `base` несёт женские голоса (sveta и masha
# отличаются только голосовым пакетом), `dima` — мужской. Разделение по файлам,
# а не по голосовым пакетам: пакет задаёт тембр, чекпоинт — обученную на нём
# модель.
KOKORO_BASE_WEIGHTS = os.environ.get("TTS_KOKORO_BASE_WEIGHTS", "kokoro-ru-v2-base.pth")
KOKORO_DIMA_WEIGHTS = os.environ.get("TTS_KOKORO_DIMA_WEIGHTS", "kokoro-ru-v2-dima.pth")
# Встроенный голос выбирается по полу карточки голоса: пользователь уже ответил
# на этот вопрос при создании голоса, и вторая ручка «каким из трёх встроенных
# голосов говорить» была бы вопросом про то же самое. `other` достаётся masha —
# второй женский пакет того же чекпоинта.
KOKORO_VOICE_FEMALE = os.environ.get("TTS_KOKORO_VOICE_FEMALE", "sveta")
KOKORO_VOICE_MALE = os.environ.get("TTS_KOKORO_VOICE_MALE", "dima")
KOKORO_VOICE_OTHER = os.environ.get("TTS_KOKORO_VOICE_OTHER", "masha")
# Потолок одной фонемной строки. Контекст `KModel` — 512 позиций, две из них
# служебные (`assert len(input_ids) + 2 <= context_length`), поэтому более
# длинный кусок режется по границам слов: без этого длинная реплика падала бы
# не ошибкой модели, а `AssertionError` внутри неё.
KOKORO_MAX_PHONEMES = int(os.environ.get("TTS_KOKORO_MAX_PHONEMES", "510"))
# Сколько голосовых пакетов держать в памяти. Пакет — 510×256 float32 (около
# 0.5 МБ), но читается с диска `torch.load`; кеш нужен, чтобы не платить за это
# на каждой реплике.
KOKORO_VOICE_CACHE_SIZE = int(os.environ.get("TTS_KOKORO_VOICE_CACHE", "4"))
# Устройство. По умолчанию CPU: файнтюн объявлен CPU-only и измерен на нём
# (RTF ~0.1), а 82 млн параметров на MPS не выигрывают столько, чтобы платить за
# непроверенные операции. `mps` — осознанная проба, не режим по умолчанию.
KOKORO_DEVICE = os.environ.get("TTS_KOKORO_DEVICE", "cpu")
# Обязательные файлы модели. Список один на двоих — паспорт модели
# (`model_manager`) и сам движок (`engines/kokoro_engine.py`) проверяют по нему:
# «установлено» на вкладке «Модели» и успешная загрузка обязаны совпадать, иначе
# интерфейс обещал бы рабочую модель, которую движок не находит.
KOKORO_REQUIRED_FILES = (
    "config.json",  # конфигурация `KModel`: словарь фонем и архитектура
    "kokoro-config.json",  # словарь фонем для произносительного словаря
    KOKORO_BASE_WEIGHTS,
    KOKORO_DIMA_WEIGHTS,
    "ru_g2p.py",  # произносительный словарь репозитория (ударения через RUAccent)
    # Пересобранный под ударения словарь espeak-ng: без него ударения теряются
    # молча, и словарь репозитория отказывается работать — и правильно делает.
    "ru_dict",
    "*.pt",  # голосовые пакеты `voices/{sveta,masha,dima}.pt`
)

# --- Ресурсы ------------------------------------------------------------------
# Не занимать все P-ядра разом: модель крутится строго последовательно.
# По умолчанию совпадает с TTS_THREAD_LIMIT (см. блок лимитов потоков в начале файла).
TORCH_NUM_THREADS = int(os.environ.get("TTS_TORCH_THREADS", _THREAD_LIMIT))
# Таймаут на синтез одного куска (секунды). Не занижать: самый долгий честный
# инференс на MPS в этом проекте — ~150 с (реплика 600 знаков, nfe_step=32),
# поэтому дефолт взят с четырёхкратным запасом.
CHUNK_TIMEOUT_SEC = float(os.environ.get("TTS_CHUNK_TIMEOUT_SEC", "600"))
# Таймаут декодирования аудио через ffmpeg (секунды). Референс — до 50 МБ, честное
# декодирование такого файла занимает секунды; без таймаута битый или зацикленный
# вход оставлял процесс ждать ffmpeg вечно.
FFMPEG_TIMEOUT_SEC = float(os.environ.get("TTS_FFMPEG_TIMEOUT_SEC", "120"))
# Потолок памяти процесса (МБ): при устойчивом превышении watchdog прерывает
# текущую задачу (сам процесс не убивается). Считается по RSS из psutil, а RSS
# на Apple Silicon не включает память, выделенную моделями через Metal (MPS):
# с поднятыми движками RSS остаётся десятками мегабайт при физическом следе в
# гигабайты (проверено `footprint`: F5 + обе XTTS ≈ 8.9 ГБ против 36 МБ в RSS).
# То есть лимит ловит буферы пайплайна и нативные аллокации, а вес моделей —
# нет; за перегрузку всей машины отвечает SYSTEM_MEM_THRESHOLD_PERCENT ниже.
MAX_RSS_MB = int(os.environ.get("TTS_MAX_RSS_MB", "5120"))
# Порог общей памяти системы (проценты): вторая, независимая от MAX_RSS_MB
# проверка — она видит нагрузку чужих процессов, которую свой RSS не показывает.
SYSTEM_MEM_THRESHOLD_PERCENT = int(os.environ.get("TTS_SYSTEM_MEM_THRESHOLD", "85"))
# Доля памяти Metal, которую разрешено занять одному процессу
# (`torch.mps.set_per_process_memory_fraction`). `0` — не ограничивать: тогда
# потолок задаёт только система, и на 8–16 ГБ процесс может запросить больше
# физической памяти (аудит §11.4) — это уже не деградация, а GPU panic.
#
# Доля считается от `recommended_max_memory()`, то есть от того, что Metal
# считает разумным максимумом для машины, и действует **на процесс**: у каждого
# воркера свой потолок, а сумма по процессам держится политикой выгрузки
# простаивающих движков (`TTS_ENGINE_IDLE_UNLOAD_MIN`). 0.8 оставляет системе
# запас на себя и на чужие приложения, не мешая поднять самую тяжёлую из
# поддерживаемых моделей.
MPS_MEMORY_FRACTION = float(os.environ.get("TTS_MPS_MEMORY_FRACTION", "0.8"))
# Выгрузка простаивающего движка (фаза 10): сколько минут движок может не
# участвовать в синтезе, прежде чем фоновая задача вернёт его память. `0` —
# политика выключена, выгрузка остаётся только ручной.
DEFAULT_ENGINE_IDLE_UNLOAD_MIN = 15.0
ENGINE_IDLE_UNLOAD_ENV = "TTS_ENGINE_IDLE_UNLOAD_MIN"


def engine_idle_unload_minutes() -> float:
    """Порог простоя в минутах из окружения — читается **в момент вызова**.

    Не константой на импорте: настройку меняют без перезапуска (и подменяют в
    тестах через `monkeypatch.setenv`), а `0` — это «выключено». Нечисло и
    отрицательное значение приводим к безопасному дефолту: опечатка в
    переменной окружения не должна включать выгрузку «прямо сейчас».
    """
    raw = os.environ.get(ENGINE_IDLE_UNLOAD_ENV)
    if raw is None:
        return DEFAULT_ENGINE_IDLE_UNLOAD_MIN
    try:
        minutes = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ENGINE_IDLE_UNLOAD_MIN
    return minutes if minutes > 0 else 0.0


# --- Короткие реплики (docs/dialogue.md) --------------------------------------
# Короткие реплики звучат хуже длинных: модели не хватает контекста. Слой
# исправления живёт после подготовки текста.
#
# Включён **по умолчанию** — по результатам живого benchmark'а, а не по
# предположению: на F5 стратегия `same_speaker_context` убрала реальные дефекты
# (средний WER 0.125–0.20 → 0.000, «Да.» с WER 1.0 и повтор в «Нет.» исчезли), а
# контрольные длинные фразы не изменились (WER 0.0 при обеих стратегиях). Для
# движков без измерений `auto` выбирает `direct`, то есть ничего не меняет, —
# поэтому включение по умолчанию не создаёт риска: работает только там, где
# измерено улучшение.
#
# Выключить можно явно: `TTS_SHORT_UTTERANCE=0` (или галочкой в панели рендера).
SHORT_UTTERANCE_ENV = "TTS_SHORT_UTTERANCE"
SHORT_UTTERANCE_DEFAULT_ENABLED = os.environ.get(SHORT_UTTERANCE_ENV, "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Пороги классов (§2). Слова — основная ось, знаки — страховка от одного
# длинного слова: «Здравствуйте» по словам короткое, а по звучанию уже нет.
SHORT_UTTERANCE_VERY_SHORT_WORDS = int(os.environ.get("TTS_SHORT_VERY_SHORT_WORDS", "2"))
SHORT_UTTERANCE_SHORT_WORDS = int(os.environ.get("TTS_SHORT_SHORT_WORDS", "5"))
SHORT_UTTERANCE_VERY_SHORT_CHARS = int(os.environ.get("TTS_SHORT_VERY_SHORT_CHARS", "20"))
SHORT_UTTERANCE_SHORT_CHARS = int(os.environ.get("TTS_SHORT_SHORT_CHARS", "60"))
# Ограниченный повтор короткой реплики (§19): 2–3 попытки, не больше.
SHORT_UTTERANCE_MAX_ATTEMPTS = int(os.environ.get("TTS_SHORT_MAX_ATTEMPTS", "2"))
# Нейтральный carrier для SYNTHETIC_CONTEXT (§7): минимально влияет на эмоцию,
# не меняет смысл цели и используется только если benchmark это подтвердил.
SHORT_UTTERANCE_SYNTHETIC_CARRIER = os.environ.get(
    "TTS_SHORT_SYNTHETIC_CARRIER", "Хорошо."
)
SHORT_UTTERANCE_SYNTHETIC_CARRIERS = (
    SHORT_UTTERANCE_SYNTHETIC_CARRIER,
    "Понятно.",
    "Так.",
    "Вот так.",
)
# Насколько далеко ищется реплика того же спикера для контекста (§9). Соседняя
# реплика часто принадлежит другому персонажу, поэтому «рядом» — это несколько
# реплик, но не вся сцена.
SHORT_UTTERANCE_CONTEXT_WINDOW = int(os.environ.get("TTS_SHORT_CONTEXT_WINDOW", "3"))
# Сколько реплик одного спикера разрешено склеивать в один синтез (§10).
SHORT_UTTERANCE_BATCH_MAX = int(os.environ.get("TTS_SHORT_BATCH_MAX", "3"))
# Sanity-check длительности короткого куска (§21): ожидаемая длительность
# считается от длины текста (символов в секунду), а не задаётся одним числом для
# всех реплик. Порог — доля от ожидаемой плюс абсолютный минимум.
SHORT_UTTERANCE_CHARS_PER_SEC = float(os.environ.get("TTS_SHORT_CHARS_PER_SEC", "15"))
SHORT_UTTERANCE_MIN_DURATION_RATIO = float(
    os.environ.get("TTS_SHORT_MIN_DURATION_RATIO", "0.35")
)
SHORT_UTTERANCE_MIN_DURATION_SEC = float(os.environ.get("TTS_SHORT_MIN_DURATION_SEC", "0.15"))
# Метод определения границы цели в контекстном синтезе: `asr` (таймстемпы слов,
# единственный пригодный для производства) или `silence` (только benchmark).
SHORT_UTTERANCE_BOUNDARY_METHOD = os.environ.get("TTS_SHORT_BOUNDARY", "asr")
# Измеренная политика (benchmark 2026-09-17, `tools/short_bench.py`, реальные голоса):
# у F5 контекст того же спикера убирает реальные дефекты коротких реплик — WER 0.0
# против 0.125–0.2 и 70 % прошедших короткую проверку против 50 % у direct (в том
# числе «Да.» с WER 1.0 и повтор в «Нет.»); synthetic_context даёт те же 70 %, но
# оставляет риск carrier'а в аудио. У XTTS разницы не нашлось (0.60 ok у обеих
# стратегий, WER 0.0), поэтому остаётся direct: лишний синтез без выигрыша не нужен.
# Движок без измерений (xtts-banana) тоже остаётся на direct — «нет данных» не
# повод включать экспериментальную стратегию.
SHORT_UTTERANCE_MEASURED_STRATEGIES: dict[str, str] = {
    "f5": "direct",
}
# Почему теперь direct (UPDATE 2 §24, §45). Прошлый замер сравнивал WER **сырого**
# выхода модели с текстом цели, а в файл уходил обрезанный кусок: дефект,
# внесённый обрезкой, в том замере не видел никто (QA мерил не то аудио — это
# исправлено в Phase 2). Новый замер (F5, голос REF_RU, диалог §2, трейс стадий и
# ASR по словам на сыром и финальном аудио) показал: контекстный синтез не теряет
# слов (first/last word на месте, контекст в файл не протекает), но делает ту же
# реплику на 25–50 % длиннее baseline («Дайте пройти. Пожалуйста.» 4.26 с против
# 2.82 с) и стоит вчетверо дороже: 347 с против 90 с на 10 реплик — при том, что
# DIRECT тоже не теряет ни одного слова. Пока контекст не даёт измеримого выигрыша
# в разборчивости, производственная стратегия — DIRECT; контекстные стратегии
# остаются выбираемыми вручную для экспериментов (§25, §26).


# Политика по движкам. Начинается с измеренной и переопределяется окружением:
# формат переменной "f5=direct,xtts=synthetic_context".
def _engine_strategies() -> dict[str, str]:
    raw = os.environ.get("TTS_SHORT_ENGINE_STRATEGIES", "").strip()
    result: dict[str, str] = dict(SHORT_UTTERANCE_MEASURED_STRATEGIES)
    for item in raw.split(","):
        name, _, strategy = item.partition("=")
        if name.strip() and strategy.strip():
            result[name.strip()] = strategy.strip()
    return result


SHORT_UTTERANCE_ENGINE_STRATEGIES = _engine_strategies()

# --- Просодическая маршрутизация (UPDATE 3 §25, §26, §35) --------------------
# Режим отката резолвера референсов:
#   `neutral_only` — нужного интонационного профиля нет → NEUTRAL с явной
#       пометкой. Консервативный режим и значение по умолчанию: §26 разрешает
#       откат «на соседний профиль» только после Reference Prosody Transfer
#       Benchmark, а он ещё не подтвердил ни одну цепочку (§35).
#   `chain` — таблица цепочек (`reference_resolver.FALLBACK_CHAINS`) целиком.
# Переключается по факту benchmark'а (Phase 11), а не «на глаз»: неподтверждённый
# откат — это подмена одного звучания другим без доказательства, что оно ближе.
PROSODY_FALLBACK_MODE_ENV = "TTS_PROSODY_FALLBACK_MODE"
PROSODY_FALLBACK_MODE = os.environ.get(PROSODY_FALLBACK_MODE_ENV, "neutral_only").strip()

# Темп фразы (§12) — контролируемый словарь приложения. `llm/schemas.py` держит
# свою копию осознанно: пакет `llm` не импортирует остальное приложение и
# тестируется без него. Равенство наборов проверяется тестом, поэтому разойтись
# они не могут.
PROSODY_PACES: tuple[str, ...] = ("SLOW", "NORMAL", "FAST")

# --- Изоляция синтеза в отдельном процессе (creash_report) --------------------
# Модель и инференс живут в дочернем процессе. Причина не в архитектурной
# красоте: падение нативной библиотеки (SIGABRT/SIGSEGV) или убийство процесса
# по памяти убивало весь бэкенд вместе с базой, очередью и уже готовыми
# репликами. С изоляцией падает только воркер, а бэкенд видит код возврата,
# объясняет причину и поднимает замену.
WORKER_ISOLATION_ENV = "TTS_WORKER_ISOLATION"
DEFAULT_WORKER_ISOLATION = True
# Маркер дочернего процесса: внутри воркера реестр обязан собирать настоящий
# движок, а не ещё одного воркера (иначе рекурсия).
WORKER_CHILD_ENV = "TTS_WORKER_CHILD"
# Фабрика движка для дочернего процесса в формате «модуль:функция» — шов для
# тестов изоляции: проверять падение и перезапуск на настоящей модели нельзя.
WORKER_ENGINE_FACTORY_ENV = "TTS_WORKER_ENGINE_FACTORY"
WORKER_MODULE = "backend.engines.worker_process"
# Запас к таймауту куска: сам таймаут ставит пайплайн (`CHUNK_TIMEOUT_SEC`), и
# его вердикт точнее — там известна реплика. Воркер получает таймаут чуть
# больше, иначе ответ «не уложился» приходил бы из двух мест по-разному.
WORKER_TIMEOUT_SLACK_SEC = float(os.environ.get("TTS_WORKER_TIMEOUT_SLACK_SEC", "60"))
WORKER_REQUEST_TIMEOUT_SEC = CHUNK_TIMEOUT_SEC + WORKER_TIMEOUT_SLACK_SEC
# Подъём модели в воркере: XTTS-banana грузится минутами, поэтому таймаут не
# меньше, чем у куска.
WORKER_LOAD_TIMEOUT_SEC = float(
    os.environ.get("TTS_WORKER_LOAD_TIMEOUT_SEC", str(max(CHUNK_TIMEOUT_SEC, 600.0)))
)
# Сколько ждать добровольного выхода воркера при выгрузке и выключении, прежде
# чем убить его: `shutdown` по каналу + `SIGTERM` + `SIGKILL`.
WORKER_SHUTDOWN_GRACE_SEC = float(os.environ.get("TTS_WORKER_SHUTDOWN_GRACE_SEC", "10"))
# Защита от цикла падений: столько отказов за окно переводят движок в DEGRADED —
# новые задачи по нему не стартуют, пока не истечёт пауза. Без этого «падение на
# каждой реплике» превращалось бы в бесконечный перезапуск процессов.
WORKER_CRASH_LIMIT = int(os.environ.get("TTS_WORKER_CRASH_LIMIT", "3"))
WORKER_CRASH_WINDOW_SEC = float(os.environ.get("TTS_WORKER_CRASH_WINDOW_SEC", "60"))
WORKER_DEGRADED_COOLDOWN_SEC = float(os.environ.get("TTS_WORKER_DEGRADED_COOLDOWN_SEC", "120"))
# Сколько раз задача автоматически повторяется после падения воркера. Один
# повтор — компромисс: разовый сбой (OOM от соседнего процесса) лечится сам, а
# систематический не превращается в бесконечный цикл.
WORKER_CRASH_RETRIES = int(os.environ.get("TTS_WORKER_CRASH_RETRIES", "1"))
# Потолок памяти воркера (МБ): вес моделей виден только в его процессе, поэтому
# watchdog следит за ним отдельно (см. resource_guard).
WORKER_MAX_RSS_MB = int(os.environ.get("TTS_WORKER_MAX_RSS_MB", str(MAX_RSS_MB)))


def worker_isolation_enabled() -> bool:
    """Включена ли изоляция инференса — читается **в момент вызова**.

    Функцией, а не константой: тесты выключают изоляцию на время одного теста
    (`monkeypatch.setenv`), а дочерний процесс обязан получить `False` по
    маркеру `TTS_WORKER_CHILD` независимо от настроек родителя.
    """
    if os.environ.get(WORKER_CHILD_ENV):
        return False
    raw = os.environ.get(WORKER_ISOLATION_ENV)
    if raw is None:
        return DEFAULT_WORKER_ISOLATION
    return raw.strip().lower() not in ("0", "false", "no", "off", "выкл")


# Сколько хранить готовые файлы в output/
OUTPUT_TTL_HOURS = float(os.environ.get("TTS_OUTPUT_TTL_HOURS", "24"))
# Сколько свободного места должно оставаться на диске перед рендером (§6
# «Заполнен диск»). Ноль или меньше выключает проверку — она нужна тестам и тем,
# кто уверен в своём диске. Порог в мегабайтах: гигабайты читались бы «на глаз»,
# а отказ обязан случиться заранее и с понятным текстом, а не на середине записи.
DISK_MIN_FREE_MB = int(os.environ.get("TTS_DISK_MIN_FREE_MB", "512"))
# Максимальная длина куска (символов), который уходит в модель за один прогон.
# Реплика длиннее режется по границам предложений (см. dialogue_parser).
MAX_REPLICA_CHARS = int(os.environ.get("TTS_MAX_REPLICA_CHARS", "600"))
# Стратегия нарезки текста на куски — выбирается в интерфейсе, а не средой
# окружения: это разовый выбор под задачу, а не свойство установки. Короткий
# кусок реже «уплывает» по интонации, длинный — реже рвёт мысль на границе.
CHUNK_STRATEGY_SHORT = "short"
CHUNK_STRATEGY_PARAGRAPH = "paragraph"
CHUNK_STRATEGY_DEFAULT = CHUNK_STRATEGY_PARAGRAPH
SHORT_CHUNK_CHARS = 200


def chunk_chars(strategy: str | None) -> int:
    """Лимит куска в знаках для выбранной стратегии нарезки."""
    if strategy == CHUNK_STRATEGY_SHORT:
        return SHORT_CHUNK_CHARS
    return MAX_REPLICA_CHARS


# Верхняя граница сида генерации, [0, SEED_MAX). Значение не «на глаз»: F5-TTS
# прокидывает сид в `seed_everything`, а тот пишет его в os.environ
# ["PYTHONHASHSEED"] (f5_tts/model/utils.py). При seed=None библиотека берёт
# random.randint(0, sys.maxsize) — это больше допустимого максимума 4294967295,
# и унаследовавший переменную подпроцесс падает с
# "Fatal Python error: config_init_hash_seed". Граница общая для всех движков:
# сид задаёт пайплайн, а не движок, чтобы его можно было записать рядом с куском.
SEED_MAX = 2 ** 32

# Сколько вариантов одного куска хранить. Вариант — это аудио, уже побывавшее в
# файле: без него «перегенерировать» означает потерять предыдущий результат, а
# A/B на слух без предыдущего невозможен.
MAX_REPLICA_VARIANTS = int(os.environ.get("TTS_MAX_REPLICA_VARIANTS", "3"))

# Сколько take'ов реплики хранить в базе. Это не про варианты одной задачи
# (их ограничивает MAX_REPLICA_VARIANTS), а про историю на диск: каждый рендер
# добавляет реплике вариант, и без предела `output/projects/{id}/` рос бы вместе
# с числом прогонов. Вытесняются самые старые take'ы, активный не трогается.
MAX_TAKES_PER_REPLICA = int(os.environ.get("TTS_MAX_TAKES_PER_REPLICA", "20"))

# --- Строгая проверка синтеза (QA-цикл) ---------------------------------------
# Опциональный цикл: каждый кусок расшифровывается тем же Whisper'ом, что и
# референс, и при расхождении с исходным текстом синтез повторяется. По умолчанию
# выключен: одна попытка — это полный синтез плюс отдельный процесс Whisper, и на
# 10+ репликах цена растёт кратно.
# Порог отдельный от `transcribe.MATCH_THRESHOLD` (0.6) и строже него: 60% ловят
# грубый рассинхрон «текст от другого файла», а здесь принимается готовый
# результат — распознанное должно совпасть с исходным текстом почти дословно.
QA_WER_THRESHOLD = float(os.environ.get("TTS_QA_WER_THRESHOLD", "0.15"))
# Границы цикла: останавливает то, что наступит раньше — попытки или время.
# Бюджет обязан с запасом покрывать несколько попыток: одна попытка — это синтез
# (до CHUNK_TIMEOUT_SEC) плюс подъём Whisper в отдельном процессе, то есть десятки
# секунд на каждом шаге.
QA_MAX_ATTEMPTS = int(os.environ.get("TTS_QA_MAX_ATTEMPTS", "4"))
QA_BUDGET_SEC = float(os.environ.get("TTS_QA_BUDGET_SEC", "300"))
# Шаг подстройки за попытку — не перебор вслепую, а движение в сторону
# предсказуемости внутри границ, объявленных движком (CFG_RANGE у F5, паспорт
# temperature/repetition_penalty у XTTS). Сид каждая попытка получает новый.
QA_CFG_STEP = 0.4
QA_TEMPERATURE_STEP = 0.1
QA_REPETITION_PENALTY_STEP = 1.0

# Режимы проверки. Раньше их было два — «выключено» и «строгая проверка каждого
# куска», — и на длинном диалоге второй означал десяток запусков Whisper. Smart
# добавляет третью ступень: сначала дешёвый отбор по самому аудио, и только
# подозрительные куски уходят в расшифровку (см. `qa_screening`).
QA_MODE_OFF = "off"
QA_MODE_SMART = "smart"
QA_MODE_STRICT = "strict"
QA_MODES = (QA_MODE_OFF, QA_MODE_SMART, QA_MODE_STRICT)

# --- Smart QA: дешёвый отбор подозрительных кусков -----------------------------
# Отбор считает только то, что видно в самом waveform: длину относительно текста,
# тишину, перегруз, уровень, повторы. Ошибиться в сторону «подозрительный» стоит
# одного лишнего запуска Whisper, в сторону «нормальный» — дороже: непроверенный
# кусок уедет в готовый файл. Поэтому границы широкие.
QA_SCREEN_CHARS_PER_SEC = 15.0
# Оценка длины по тексту ниже секунды — это шум: на короткой фразе «0.6 с вместо
# 0.9» не значит ничего. Поэтому ожидаемая длина не опускается ниже секунды.
QA_SCREEN_MIN_EXPECTED_SEC = 1.0
# Совсем короткий кусок (щелчок, обрыв) — признак сорвавшегося синтеза.
QA_SCREEN_MIN_DURATION_SEC = 0.2
QA_SCREEN_SHORT_RATIO = 0.35
QA_SCREEN_LONG_RATIO = 3.0
# Кусок, который больше чем на 60 % состоит из тишины, — это либо обрыв, либо
# модель «задумалась» посреди фразы.
QA_SCREEN_SILENCE_RATIO = 0.6
QA_SCREEN_SILENCE_DB = -45.0
# Границы уровня считаются до нормализации куска: тихий или перегруженный выход
# движка видно только на сыром waveform — `_prepare_chunk` его уже выровняет.
QA_SCREEN_RMS_FLOOR_DB = -45.0
QA_SCREEN_RMS_CEIL_DB = -1.0
# Повтор участка ищется корреляцией сигнала с собой при сдвиге 0.4–2 с: у
# зацикленной модели кусок повторяется почти дословно. Замер идёт только там, где
# меняется громкость: ровный тон (затянутая гласная, гудок) коррелирует с собой
# при сдвиге, кратного периоду, и без этой оговорки отбор браковал бы каждую
# длинную гласную.
QA_SCREEN_REPEAT_CORR = 0.98
QA_SCREEN_REPEAT_MIN_ENVELOPE_DB = 3.0
QA_SCREEN_REPEAT_MIN_LAG_SEC = 0.4
QA_SCREEN_REPEAT_MAX_LAG_SEC = 2.0
# Кадр замера громкости: 20 мс — коротко для интонации и длинно для дрожания
# отдельного периода.
QA_SCREEN_FRAME_SEC = 0.02

# Максимальная длина сплошного текста для одной задачи (символов).
# ~50 000 знаков — это примерно час готового аудио; защита от случайной
# вставки книги целиком в режиме «Сплошной текст».
MAX_TEXT_CHARS = int(os.environ.get("TTS_MAX_TEXT_CHARS", "50000"))

# Максимальное число реплик в диалоге. Ориентир тот же, что у MAX_TEXT_CHARS:
# 500 реплик по ~100 знаков — примерно тот же объём текста, но уже сотни
# прогонов модели, то есть часы синтеза. Диалоговый режим лимита на длину
# текста не имеет вовсе, поэтому реплики — единственная защита от гигантского
# ввода здесь.
MAX_REPLICAS = int(os.environ.get("TTS_MAX_REPLICAS", "500"))

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
# Целевая громкость куска (RMS). Одно значение и для модели (`target_rms` в F5-TTS),
# и для пост-нормализации: иначе куски приходят с разным уровнем и «скачут» на стыках.
DEFAULT_TARGET_RMS = float(os.environ.get("TTS_TARGET_RMS", "0.1"))
# Сглаживание самых краёв куска: модель иногда оставляет на границе щелчок,
# который кроссфейд не убирает, а только смешивает с соседним звуком.
EDGE_FADE_MS = float(os.environ.get("TTS_EDGE_FADE_MS", "10"))
# Балансировка голосов по громкости (дБ) — применяется после выравнивания RMS.
DEFAULT_GAIN_DB = 0.0
# Питч-шифт меняет высоту, но не пол голоса: тембр остаётся от референса.
DEFAULT_PITCH_SEMITONES = 0.0

# --- Громкость и паузы собранного трека ---------------------------------------
# Финальный проход по готовому файлу. Нормализация кусков по RMS выравнивает их
# между собой, но не даёт одинаковой *воспринимаемой* громкости: плотность звука
# у F5 и XTTS разная, и на стыке движков слышен «скачок» даже при равном RMS.
# LUFS (ITU-R BS.1770) выравнивает громкость файла целиком.
OUTPUT_LUFS = float(os.environ.get("TTS_OUTPUT_LUFS", "-16"))
# Лимитер после нормализации: без него подъём громкости до целевого LUFS
# упирается в клиппинг на пиках (потолок — тот же _PEAK_LIMIT в audio_pipeline).
# Блочный, с запасом на атаку и медленным восстановлением: усиление меняется по
# блокам, а не по отсчётам, иначе на резких пиках лимитер сам даёт щелчки.
LIMITER_BLOCK_MS = 5.0
LIMITER_LOOKAHEAD_BLOCKS = 4  # 20 мс: усиление падает раньше, чем придёт пик
LIMITER_RELEASE_MS = 60.0
# Порог тишины на краях куска (дБFS) и запас вокруг найденной речи (мс). Модель
# сама оставляет тишину на краях, а pause_ms добавляет паузу поверх — суммарный
# зазор между репликами гуляет. Обрезав края, получаем ровно заданную паузу.
EDGE_SILENCE_DB = float(os.environ.get("TTS_EDGE_SILENCE_DB", "-45"))
EDGE_SILENCE_FRAME_MS = 20.0
EDGE_SILENCE_MARGIN_MS = 30.0

# Запас вокруг найденной речи для коротких реплик (мс). Общий порог тишины
# подобран на длинном тексте, где потеря 30 мс на краю незаметна; на односложной
# фразе край — это и есть слово, поэтому у коротких кусков запас шире, а решение
# принимается отдельно (см. `audio_pipeline._edge_bounds`).
SHORT_EDGE_GUARD_MS = float(os.environ.get("TTS_SHORT_EDGE_GUARD_MS", "60"))

# --- Трейс синтеза (UPDATE 2 §14–§15) ------------------------------------------
# Диагностика «где теряется слово»: срезы аудио по стадиям и числа обрезки.
# Выключено по умолчанию — трейс пишет файлы и нужен только при разборе.
SYNTHESIS_TRACE_ENV = "TTS_SYNTHESIS_TRACE"
SYNTHESIS_TRACE_DIR_ENV = "TTS_SYNTHESIS_TRACE_DIR"

# --- Прогрев коротких реплик (warm-up prefix) ----------------------------------
# Перед короткой репликой в TTS уходит скрытый естественный текст, а в готовый
# файл попадает только цель (см. backend/warmup_context.py). Выключено по
# умолчанию до живого A/B на реальных голосах: включение функции, которая
# добавляет LLM-вызов и alignment в каждый короткий кусок, — решение по измерению,
# а не по предположению (§23, §25 пакета).
WARMUP_ENV = "TTS_WARMUP_ENABLED"
WARMUP_ENABLED = os.environ.get(WARMUP_ENV, "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)
# Короткость цели: те же две оси, что у слоя коротких реплик (знаки и слова), но
# свои пороги — прогрев дороже контекста, поэтому его область уже.
WARMUP_MAX_CHARS = int(os.environ.get("TTS_WARMUP_MAX_CHARS", "80"))
WARMUP_MAX_WORDS = int(os.environ.get("TTS_WARMUP_MAX_WORDS", "12"))
# Предел префикса: длинный контекст не улучшает вход в речь, а удлиняет синтез и
# повышает шанс, что модель «уедет» в собственный текст.
WARMUP_MAX_PREFIX_CHARS = int(os.environ.get("TTS_WARMUP_MAX_PREFIX_CHARS", "180"))
WARMUP_LLM_TIMEOUT_SEC = float(os.environ.get("TTS_WARMUP_LLM_TIMEOUT_SEC", "10"))
# Сколько токенов разрешено модели: два предложения — это десятки токенов, а не
# сотни. Ограничение спасает от «размышлений» на локальной модели.
WARMUP_MAX_TOKENS = int(os.environ.get("TTS_WARMUP_MAX_TOKENS", "96"))
WARMUP_TEMPERATURE = float(os.environ.get("TTS_WARMUP_TEMPERATURE", "0.3"))
WARMUP_SEED = int(os.environ.get("TTS_WARMUP_SEED", "0"))
WARMUP_NUM_CTX = int(os.environ.get("TTS_WARMUP_NUM_CTX", "2048"))
WARMUP_KEEP_ALIVE = os.environ.get("TTS_WARMUP_KEEP_ALIVE", "5m")
# Запас перед найденной границей цели: атаку первого фонема легко срезать «по
# границе слова». Значение измеряется на живом прогоне, а не берётся из головы.
WARMUP_PREROLL_MS = float(os.environ.get("TTS_WARMUP_PREROLL_MS", "30"))
# Уверенность alignment: доля найденных слов цели. Ниже порога граница считается
# ненадёжной, и пайплайн синтезирует цель заново без прогрева (fail-safe, §9).
WARMUP_ALIGNMENT_MIN_CONFIDENCE = float(
    os.environ.get("TTS_WARMUP_ALIGNMENT_MIN_CONFIDENCE", "0.8")
)
WARMUP_CACHE_SIZE = int(os.environ.get("TTS_WARMUP_CACHE_SIZE", "256"))
# Движки, для которых прогрев разрешён. Список, а не флаг: разные движки
# по-разному реагируют на контекст, и отключать их нужно по одному (§13).
WARMUP_ENGINES = tuple(
    item.strip()
    for item in os.environ.get("TTS_WARMUP_ENGINES", "f5").split(",")
    if item.strip()
)

# --- Дефолтные параметры XTTS v2 ----------------------------------------------
# Значения совпадают с тем, что записано в config.json базовой модели и
# рекомендовано её авторами: на них XTTS v2 стабильно и разборчиво читает русский.
DEFAULT_XTTS_TEMPERATURE = float(os.environ.get("TTS_XTTS_TEMPERATURE", "0.75"))
DEFAULT_XTTS_REPETITION_PENALTY = float(os.environ.get("TTS_XTTS_REPETITION_PENALTY", "5.0"))
DEFAULT_XTTS_TOP_K = 50
DEFAULT_XTTS_TOP_P = 0.85

# --- Дефолтные параметры Qwen3-TTS --------------------------------------------
# Стартовые значения, а не измеренные: у проекта пока нет живого прогона этой
# модели (ни весов, ни MPS-гарантий), поэтому честная формулировка — «с чего
# начинать подбор». Диапазоны заданы там, где за их пределами речь распадается:
# ниже 0.1 модель «залипает» на одном слоге, выше 1.5 — перестаёт держать текст.
DEFAULT_QWEN_TEMPERATURE = float(os.environ.get("TTS_QWEN_TEMPERATURE", "0.9"))
DEFAULT_QWEN_TOP_P = float(os.environ.get("TTS_QWEN_TOP_P", "0.9"))
DEFAULT_QWEN_REPETITION_PENALTY = float(os.environ.get("TTS_QWEN_REPETITION_PENALTY", "1.05"))
# Потолок генерации: страховка от «разговорившейся» языковой модели, а не
# регулятор длины реплики. 2048 кодовых кадров при 12 Гц — это ~170 с звука,
# то есть заведомо больше самого длинного куска (600 знаков).
DEFAULT_QWEN_MAX_NEW_TOKENS = int(os.environ.get("TTS_QWEN_MAX_NEW_TOKENS", "2048"))
# Черновик — тот же движок с меньшим потолком: короткие куски пишутся быстрее,
# а качество клонирования от потолка не зависит.
QWEN_DRAFT_MAX_NEW_TOKENS = int(os.environ.get("TTS_QWEN_DRAFT_MAX_NEW_TOKENS", "1024"))

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
GAIN_DB_RANGE = (-20.0, 20.0)
PITCH_SEMITONES_RANGE = (-12.0, 12.0)
# Ниже 0.02 кусок звучит как шёпот, выше 0.3 модель упирается в клиппинг.
TARGET_RMS_RANGE = (0.02, 0.3)
# Границы ручек XTTS v2: ниже 0.1 модель «заикается» на одном слоге, выше 1.5
# распадается в шум; штраф за повторы ниже 1.0 бессмыслен, выше 20 речь становится рваной.
XTTS_TEMPERATURE_RANGE = (0.1, 1.5)
XTTS_REPETITION_PENALTY_RANGE = (1.0, 20.0)
# Границы ручек Qwen3-TTS: это разброс языковой модели, а не диффузии, поэтому
# они уже, чем у XTTS, а штраф за повторы считается множителем лог-вероятностей
# и за 2.0 делает речь рваной (каждое слово «впервые»).
QWEN_TEMPERATURE_RANGE = (0.1, 1.5)
QWEN_TOP_P_RANGE = (0.1, 1.0)
QWEN_REPETITION_PENALTY_RANGE = (1.0, 2.0)
QWEN_MAX_NEW_TOKENS_RANGE = (256, 8192)

_torch = None


def configure_torch():
    """Импортирует torch и один раз задаёт пределы ресурсов: потоки CPU и память MPS.

    Точка входа одна на процесс: в бэкенде её зовёт `lifespan`, в воркере — сам
    движок при загрузке, поэтому потолки одинаковы и в изолированном режиме, и
    без него.
    """
    global _torch
    if _torch is not None:
        return _torch
    import torch

    torch.set_num_threads(TORCH_NUM_THREADS)
    limit_mps_memory(torch)
    _torch = torch
    return _torch


def limit_mps_memory(torch) -> None:
    """Жёсткий потолок памяти Metal и лог того, что разрешено процессу.

    Без `set_per_process_memory_fraction()` MPS может рапортовать потолок больше
    физической памяти машины (аудит §11.4): тогда единственный ограничитель —
    сама система, и это не деградация, а GPU panic. Логируется и
    `recommended_max_memory()`, который аудит требует видеть: без него непонятно,
    от чего взята доля.

    Отдельной функцией, а не строчкой в `configure_torch`, ради теста: заглушка
    torch подставляется без импорта настоящего (см. `test_engine_lifecycle`).

    Отсутствие MPS и ошибка вызова отказом старта не считаются: на CPU
    ограничивать нечего, а сервис обязан подняться — о недоступном потолке
    достаточно предупредить.
    """
    try:
        if not torch.backends.mps.is_available():
            return
        if MPS_MEMORY_FRACTION <= 0:
            logger.info("Потолок памяти MPS не задан: TTS_MPS_MEMORY_FRACTION=0")
            return
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
        recommended_gb = torch.mps.recommended_max_memory() / (1024**3)
        logger.info(
            "Память MPS: рекомендовано %.1f ГБ, процессу разрешено %.1f ГБ (доля %.2f)",
            recommended_gb,
            recommended_gb * MPS_MEMORY_FRACTION,
            MPS_MEMORY_FRACTION,
        )
    except Exception as exc:  # noqa: BLE001 — без потолка сервис жив, без старта — нет
        logger.warning("Не удалось ограничить память MPS: %s", exc)


def pick_device() -> str:
    """mps, если доступен, иначе cpu."""
    torch = configure_torch()
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def qwen_force_cpu() -> bool:
    """Запрещён ли MPS для Qwen3-TTS — читается **в момент вызова**.

    Функцией, а не константой: это аварийный переключатель. Если проба MPS
    оказалась неверной, пользователь возвращает движок на CPU переменной
    окружения и перезапускает задачу, не удаляя модель и не переустанавливая
    зависимости. Читать её на импорте значило бы требовать полного перезапуска
    сервера ради одной настройки одного движка.
    """
    return os.environ.get(QWEN_FORCE_CPU_ENV, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )
