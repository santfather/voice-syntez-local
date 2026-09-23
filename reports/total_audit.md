# TOTAL_AUDIT — VOICE_SYNTEZ (TTS-дашборд для озвучки русских диалогов)

Дата: 2026-09-22
Версия/коммит: `908168f` (ДВЖ1: контракт движков, Qwen3-TTS 1.7B и пресетный Kokoro-ru) + 20 незакоммиченных файлов
Аудитор: AI-аудитор (TraeCode), 6 параллельных суб-аудиторов по слоям
Язык: русский. Область: локальное настольное приложение для одного пользователя на macOS / Apple Silicon.

---

## 1. Резюме (Executive Summary)

VOICE_SYNTEZ — локальный TTS-дашборд (FastAPI + Uvicorn + SQLite + PyTorch/MPS) для озвучки русских диалогов пятью движками (`f5`, `xtts`, `xtts-banana`, `qwen3-tts`, `kokoro-ru`). Архитектура зрелая: движок живёт в отдельном процессе (`subprocess.Popen` + `multiprocessing.Pipe`), паспорт `EngineInfo` — data-driven источник истины, есть миграции БД с `schema_version`, `ResourceGuard`, TTL-очистка, 1170 тестов. Это заметно выше среднего для «настольного» проекта.

Общий вердикт: **работоспособно и продуманно, но есть несколько мест, где теряется работа пользователя или блокируется весь сервис.** Ни одного `Blocker` (приложение запускается, данные проекта не теряются при штатной работе) не найдено. Топ-3 риска:

1. **Очередь задач не персистится** — после рестарта/падения запись о задаче исчезает полностью (в БД нет таблицы `jobs`), пользователь не узнает ни результата, ни причины.
2. **Блокировка event loop** — синхронный LLM-анализ в `render_text` и синхронная запись референса в `add_reference` останавливают весь сервер на время операции (остальные вкладки «висят»).
3. **MPS-нагрузка без предохранителей** — `ffmpeg` без `timeout`, MPS→CPU fallback на *любое* исключение без `empty_cache()`, не выставлены `PYTORCH_MPS_*_WATERMARK_RATIO`/`set_per_process_memory_fraction()`. На 16 ГБ unified memory это путь к swap-деградации и, в пределе, к GPU panic (§11).

---

## 2. Карта репозитория

### Точки входа
- [run.sh](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/run.sh) — основной лаунчер (`exec uvicorn`), pidfile `.voice_syntez.pid`, выставляет `PYTHONHASHSEED=0`, `ORT_DISABLE_TELEMETRY=1`.
- `VOICE_SYNTEZ.command` — второй лаунчер (дублирует логику run.sh, см. 4.9).
- [backend/main.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py) — FastAPI-приложение, 93 роутера, 4135 строк.
- [frontend/index.html](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/index.html) + [app.js](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js) — UI без сборки.
- [tools/doctor.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/tools/doctor.py) — диагностика окружения.

### LOC по слоям (cloc)

| Слой | Файлы | Code | Крупнейшие файлы |
|---|---|---|---|
| Python (backend + tools + tests) | 177 | 50 497 | `main.py` 4135, `audio_pipeline.py` 2831, `job_queue.py` 2029 |
| JavaScript (frontend) | 1 | 6 160 | `app.js` 7591 строк всего |
| HTML | 1 | 1 289 | `index.html` |
| CSS | 1 | 835 | `style.css` |
| **Итого** | 183 | **58 827** | — |

Комментариев в Python — 12 701 строка (25% code), пустых — 11 238. Это плотно документированный код.

### Горячие пути
1. **Старт** — `run.sh` → `uvicorn` → `main.lifespan` → `configure_torch` (MPS), `_seed_builtin_voices`, `ResourceGuard.run`.
2. **Загрузка модели** — `POST /api/engines/{id}/load` → `asyncio.to_thread(engine.load)` → воркер-процесс, сначала READY, потом веса (см. 4.2).
3. **Синтез реплики** — `job_queue` воркер → `WorkerEngine.synthesize` → pipe → процесс движка → инференс на MPS.
4. **Рендер диалога** — `POST /api/render` → очередь → сборка трека (`audio_pipeline`), паузы, LUFS, таймлайн.
5. **Экспорт** — `project_export.py` (архив, MP3, субтитры, stems), временные каталоги в системном temp.

### Внешние процессы и сеть
- `subprocess` — 38 вхождений (ffmpeg, запуск воркеров, `memory_pressure`).
- `asyncio.to_thread` — 86 вхождений (основной механизм увода блокирующих вызовов из loop).
- Загрузка весов по сети — `downloader`/`model_manager` (huggingface_hub), без проверки целостности и таймаута (см. 4.2).
- Веса на диске: `models/` — **15 ГБ** (7 каталогов моделей).

---

## 3. Топ-10 находок

| # | Severity | Слой | Находка | Доказательство | Последствие | Рекомендация | Оценка |
|---|---|---|---|---|---|---|---|
| 1 | Critical | Очередь | Задачи живут только в памяти, нет persist | `job_queue.py:285-286` `self._jobs: dict[str, Job] = {}`; в БД нет таблицы `jobs` | После рестарта/краха запись о задаче исчезает: нет ни результата, ни ошибки, ни следа в истории | Таблица `jobs(id, status, created_at, error, result_path)`, запись при `submit`/переходе статуса, восстановление «осиротевших» при старте | M |
| 2 | Critical | Backend | Синхронный LLM-анализ и запись референса в event loop | `main.py:1914` `_analyze_text_chunks(text, chunks)`; `main.py:1240` `get_store().add_reference(...)` | Loop заблокирован на время операции: все вкладки «висят», `/api/status` не отвечает | Обернуть в `await asyncio.to_thread(...)`, как сделано у соседнего `create_voice` (`main.py:1071`) | S |
| 3 | Critical | Движки/MPS | `ffmpeg` без таймаута | `audio_analysis.py:102-108` `subprocess.run([...ffmpeg...])` без `timeout=` — единственный такой вызов в проекте | Битый/бесконечный вход вешает воркер и держит память; пользователь видит вечный «синтез идёт» | `timeout=` + `subprocess.TimeoutExpired` → понятная ошибка задачи | S |
| 4 | Critical | Движки/MPS | MPS→CPU fallback на любое исключение, без очистки кеша | `xtts_engine.py:188-198` `except Exception as exc: ... self._load_model("cpu")`; `empty_cache()` там не вызывается | При утечке/фрагментации MPS модель молча уезжает на CPU, а старая Metal-память не возвращается — двойной расход unified memory | Ловить только MPS-ошибки (`"mps" in str(exc).lower()`), перед перезагрузкой `release_torch_memory()` | S |
| 5 | Major | Лаунчер/Хранилище | Логи не ротируются, растут бесконечно | `VOICE_SYNTEZ.command:37` (`LOG_FILE=…/logs/voice_syntez.log`, append без ротации); `main.py:94` `logging.basicConfig` без `FileHandler`; `logs/voice_syntez.log` = 592 КБ и растёт | Диск и время открытия лога растут; разбор инцидентов дорожает | `RotatingFileHandler(maxBytes=…, backupCount=5)` | S |
| 6 | Major | Безопасность/API | Нет лимитов размера на вход импорта и тело JSON | `main.py:1019` `_read_json_body` без ограничения; `project_export.py:574-599` `_validate_entries` проверяет путь/тип, но не размер и не число записей; `_extract` (`:602`) | Zip-bomb: архив с высокой компрессией раздувает память/диск, кладёт сервис | Лимит `Content-Length`, `max_entries`/`max_uncompressed` при распаковке | M |
| 7 | Major | Очередь | `PriorityQueue()` без `maxsize` + `_prune()` с `break` | `job_queue.py:299` `asyncio.PriorityQueue()`; `_prune` (`:710`) | Спам задач копит объекты в памяти неограниченно; «хвост» истории не чистится | `maxsize` + отказ `503` при переполнении; prune по возрасту без `break` | S |
| 8 | Major | Хранилище | Take'ы копятся без TTL, каскад удаления оставляет сироты-файлы | `db/repositories/takes.py` (`TakesRepository.add` / `list_for_project` — нет TTL/лимита); FK `ON DELETE CASCADE` чистит строки, но не файлы `output/projects/{id}/` | Диск забивается; БД и файловая система расходятся | TTL/лимит take'ов + уборка файлов по удалённым id | M |
| 9 | Major | Текст | Словарь произношения перекомпилируется на каждый проход (2–3× на реплику) | `coverage_predicate` O(слов×правил); нормализация/QA перечитывают текст заново | На 100 репликах — заметная просадка CPU задолго до узкого места MPS | Кеш скомпилированного словаря по версии + один проход нормализации | M |
| 10 | Major | Конфигурация | `torch==2.14.0` и `torchaudio==2.11.0` из разных минорных рядов | `requirements.txt:7-8` | Возможен отказ резолва или ABI-несовместимость при переустановке | Пины одного релизного ряда (сверить совместимость с f5-tts) | S |

---

## 4. Детальные находки по слоям

### 4.1 Backend / API

**F-A1. Critical — Синхронный LLM-анализ в async-роутере**
- Локация: [main.py:1914](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py#L1914) `render_text`
```python
chunks = split_into_chunks(text, config.chunk_chars(payload.chunk_strategy))
if not chunks:
    raise HTTPException(status_code=400, detail="Текст пуст — нечего озвучивать")
llm_report = _analyze_text_chunks(text, chunks)
```
- Почему плохо: `_analyze_text_chunks` — блокирующий вызов LLM, вызван прямо в корутине.
- Последствие: на время анализа весь event loop стоит; `/api/status` и все вкладки не отвечают.
- Рекомендация: `await asyncio.to_thread(_analyze_text_chunks, text, chunks)`.
- Трудозатраты: S

**F-A2. Critical — Синхронная запись референса в async-роутере**
- Локация: [main.py:1240](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py#L1240) `add_reference`
```python
try:
    profile = get_store().add_reference(
        voice_id, emotion=emotion, audio_filename=file.filename or "reference.wav",
        audio_bytes=data, ref_text=ref_text, label=label, ...
```
- Почему плохо: `store.add_reference` пишет файл и может запускать denoise/проверку — синхронно в loop; соседний `create_voice` (`main.py:1071`) уже уведён в `to_thread`.
- Последствие: загрузка большого референса блокирует сервер.
- Рекомендация: обернуть в `await asyncio.to_thread(get_store().add_reference, ...)`.
- Трудозатраты: S

**F-A3. Major — Нет лимита на тело JSON**
- Локация: [main.py:1019](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py#L1019) `_read_json_body`; `import_project` (`main.py:2768`).
- Почему плохо: размер входа не ограничен ни в теле, ни при импорте архива.
- Последствие: один огромный запрос выедает память процесса.
- Рекомендация: проверять `Content-Length` и `len(data)` до `json.loads`/распаковки.
- Трудозатраты: S

**F-A4. Minor — `except Exception` без трейсбека, утечка `str(exc)` в ответ**
- Локация: `main.py:492,499`; глобального `@app.exception_handler` нет.
- Почему плохо: инциденты трудно разбирать; текст исключения уходит клиенту.
- Последствие: невоспроизводимые сбои без стека в логе.
- Рекомендация: `logger.exception(...)`; единый `exception_handler`.
- Трудозатраты: S

**F-A5. Minor — Нет `/healthz`, нет request-id/middleware**
- Локация: подтверждено — `add_middleware`/`exception_handler`/`/health` в `main.py` отсутствуют.
- Почему плохо: нет быстрой проверки живости модели и БД; логи не сшиваются по запросу.
- Рекомендация: `/healthz` (модель + `PRAGMA quick_check`) и middleware с request-id.
- Трудозатраты: S

**F-A6. Info — привязка к 127.0.0.1 корректна; CORS/CSRF на бэке отсутствуют (loopback-контекст).**

### 4.2 Движки синтеза

**F-E1. Critical — Fallback на CPU по любому исключению**
- Локация: [xtts_engine.py:188-198](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/engines/xtts_engine.py#L188-L198)
```python
except Exception as exc:
    if self._device != "mps":
        raise
    logger.warning("MPS упал в XTTS (%s). Перезагружаю %s на CPU.", exc, self.info.label)
    with self._load_lock:
        self._model = None
        self._load_model("cpu")
    return self._infer(text, ref_audio_path, speed, params)
```
- Почему плохо: любое исключение (включая бизнес-ошибку) трактуется как сбой MPS; перед перезагрузкой не зовётся `release_torch_memory()`.
- Последствие: молчаливая деградация в 10–30× и двойной расход unified memory (старый пул Metal не отдан).
- Рекомендация: фильтровать по `"mps" in str(exc).lower()`; перед `_load_model("cpu")` вызвать `release_torch_memory()`.
- Трудозатраты: S

**F-E2. Major — `empty_cache()` только при выгрузке**
- Локация: [base.py:77-92](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/engines/base.py#L77-L92) `release_torch_memory` (порядок `gc.collect()` → `empty_cache()` — **верный**).
- Почему плохо: между репликами длинного диалога кеш не сбрасывается.
- Последствие: `driver_allocated_memory()` растёт от реплики к реплике — классический симптом MPS-утечки (§11.2).
- Рекомендация: лёгкий `gc.collect(); torch.mps.empty_cache()` в конце обработки каждой реплики; логировать дельту.
- Трудозатраты: S

**F-E3. Major — Не выставлены MPS-предохранители**
- Локация: `PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO`, `PYTORCH_MPS_PREALLOCATE`, `set_per_process_memory_fraction()` — отсутствуют; выставлен только `PYTORCH_ENABLE_MPS_FALLBACK=1` (config.py).
- Почему плохо: без `set_per_process_memory_fraction()` модель может занять всю unified memory.
- Последствие: kernel panic / GPU panic на машинах 8–16 ГБ (§11.4).
- Рекомендация: задать watermark-ratio и fraction в `configure_torch` (см. матрицу §11).
- Трудозатраты: S

**F-E4. Major — Загрузка весов без таймаута и проверки целостности**
- Локация: `model_manager`/`downloader` — скачивание через `huggingface_hub`.
- Почему плохо: оборванная загрузка не обнаруживается (нет sha256/размера), таймаута нет.
- Последствие: битые веса → падение при загрузке модели, неотличимое от «нет весов».
- Рекомендация: сверять sha256/размер после скачивания; таймаут на HTTP.
- Трудозатраты: M

**F-E5. Minor — `Kokoro.load()` объявляет READY до фактической загрузки весов; мёртвая константа `DOWNLOAD_INTERRUPTED`; нет кулдауна/thermal-проверки.**
Рекомендация: статус READY только после успешной загрузки; удалить мёртвую константу.
Трудозатраты: S

**F-E6. Info — DataLoader/`num_workers`/`pin_memory` в проекте не используются (актуально для §11.7); изоляция процессов оправдана.**

### 4.3 Пайплайн текста

**F-T1. Major — Перекомпиляция словаря произношения каждый проход.** `coverage_predicate` даёт O(слов×правил); нормализация и QA перечитывают текст заново (2–3 прохода на реплику). Рекомендация: кеш скомпилированного словаря по версии + один проход. Трудозатраты: M.

**F-T2. Major — `parse_dialogue` квадратичен и без лимитов.** [dialogue_parser.py:307-318](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/dialogue_parser.py#L307-L318): накопление текста сегмента через повторную конкатенацию `segment.text = f"{segment.text} {chunk}"` в цикле — O(n²) по длине реплики.
```python
def append(chunk: str, line_number: int) -> None:
    chunk = chunk.strip()
    if not chunk:
        return
    segment = current or start(slot_key(config.DEFAULT_SLOT), {}, line_number)
    if segment.text:
        segment.text = f"{segment.text} {chunk}"
        return
```
На 100 репликах деградирует раньше синтеза. Рекомендация: собирать части в список и `" ".join(...)`; лимит числа реплик. Трудозатраты: M.

**F-T3. Major — Нет NFC-нормализации.** Декомпозированная «ё» (`е` + U+0308) не распознаётся. Рекомендация: `unicodedata.normalize("NFC", text)` на входе. Трудозатраты: S.

**F-T4. Minor — Accentizer-синглтон без сериализации инференса; preview отдаёт RUAccent до 50 000 знаков.** Рекомендация: lock вокруг инференса; ограничить размер preview. Трудозатраты: S.

**F-T5. Minor — Drift с [docs/text-pipeline.md](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/docs/text-pipeline.md)** (описанные стадии не совпадают с фактическим порядком вызовов).

### 4.4 Пайплайн синтеза

**F-S1. Major — Take'ы копятся без TTL.** [db/repositories/takes.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/db/repositories/takes.py) — `TakesRepository` знает только `add`/`get`/`list_for_replica`/`list_for_project`/`delete`/`paths`; ни TTL, ни лимита числа take'ов нет. Диск и БД растут неограниченно. Рекомендация: TTL/лимит числа take'ов. Трудозатраты: M.

**F-S2. Major — Каскадное удаление реплик оставляет сироты-файлы.** FK `ON DELETE CASCADE` чистит строки, но не `output/projects/{id}/`. Рекомендация: уборка файлов по удалённым id в одной транзакции с записью в БД. Трудозатраты: M.

**F-S3. Minor — Сравнение громкости на float; `_limit_peaks` на Python-цикле; таймлайн доверяет `duration_sec` из БД; дубль подсчёта длительности в [project_export.py:461](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/project_export.py#L461).** Трудозатраты: S.

### 4.5 БД

**F-D1. Major — Миграции без бэкапа и `integrity_check`.** При сбое посреди миграции нет отката. Рекомендация: копия файла БД перед миграцией + `PRAGMA integrity_check` при старте. Трудозатраты: M.

**F-D2. Major — Синхронный SQLite в async-роутерах (десятки вызовов `get_store()`/репозиториев прямо в корутинах).** Блокирует loop. Рекомендация: `to_thread` для тяжёлых запросов, WAL уже включён. Трудозатраты: M.

**F-D3. Minor — Нет таблицы `jobs`** → задача не переживает рестарт (см. топ-1). Трудозатраты: M.

**F-D4. Info — 10 миграций, `schema_version` есть, FK `ON DELETE CASCADE` корректен кроме `worker_crashes`.**

### 4.6 Файловое хранилище

**F-F1. Major — Лог не ротируется.** Пишет файл лаунчер `VOICE_SYNTEZ.command:37` (`LOG_FILE="$LOG_DIR/voice_syntez.log"`, append без ротации), а `logging.basicConfig` в [main.py:94](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py#L94) задаёт только формат/уровень, без `FileHandler`; `logs/voice_syntez.log` = 592 КБ. Рекомендация: `RotatingFileHandler`. Трудозатраты: S.

**F-F2. Major — `voices.json` не восстанавливается из `voices/`.** Ручное удаление маркера/файла ломает библиотеку. Рекомендация: пересборка `voices.json` сканированием каталога при отсутствии. Трудозатраты: M.

**F-F3. Minor — `Path.glob`/повторные `os.path.exists` в горячих циклах; `tempfile` без `delete=False` при передаче в ffmpeg (потенциальная ловушка §4.4).** Трудозатраты: S.

### 4.7 Очередь задач

**F-Q1. Critical — Нет persist очереди.** [job_queue.py:285-286](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/job_queue.py#L285-L286)
```python
self._queue: asyncio.PriorityQueue | None = None
self._jobs: dict[str, Job] = {}
```
Последствие: рестарт/краш — задача исчезает без следа. Рекомендация: таблица `jobs` + восстановление при старте. Трудозатраты: M.

**F-Q2. Major — Отмена стоящей в очереди задачи игнорируется.** [job_queue.py:730-735](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/job_queue.py#L730-L735). Рекомендация: проверять статус `cancelled` перед запуском. Трудозатраты: S.

**F-Q3. Major — Кооперативная отмена ждёт до 600 с; `_wait_for_memory()` без таймаута.** Очередь может встать навсегда. Рекомендация: таймаут ожидания памяти + прерываемая отмена. Трудозатраты: M.

**F-Q4. Major — `PriorityQueue()` без `maxsize`; preview может голодать под потоком render.** Рекомендация: `maxsize` + резерв слота под preview. Трудозатраты: S.

**F-Q5. Major — Нет проверки свободного места на диске перед рендером.** Рекомендация: `shutil.disk_usage` + отказ заранее. Трудозатраты: S.

### 4.8 Frontend

**F-W1. Major — Гонки запросов без `AbortController`.** [app.js:3609](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L3609). Быстрое переключение вкладок/опросов приводит к «догоняющим» ответам, перетирающим состояние. Рекомендация: отменять предыдущий запрос. Трудозатраты: M.

**F-W2. Major — `init()` — один try на четыре шага.** [app.js:7558-7589](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L7558-L7589). Падение `loadEngines()` блокирует `loadVoices()`/`loadSavedProject()`. Рекомендация: раздельные try. Трудозатраты: S.

**F-W3. Major — Монолит `app.js` 7591 строк.** Затрудняет поддержку. Рекомендация: разбить на модули (без изменения поведения). Трудозатраты: L.

**F-W4. Minor — ⚠️ SECURITY: 60 `innerHTML`, два без `esc()`.** [app.js:1271](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L1271), [app.js:4263](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L4263) — только числовые id из БД, риск низкий, но правило нарушено. Рекомендация: `esc()` для любого динамического значения. Трудозатраты: S.

**F-W5. Minor — Нет `aria-*` и CSP.** Рекомендация: базовые aria-атрибуты и CSP-мета. Трудозатраты: S.

### 4.9 Лаунчер и окружение

**F-L1. Major — `run.sh` и `VOICE_SYNTEZ.command` дублируют логику и lock.** Два источника правды о запуске. Рекомендация: `.command` вызывает `run.sh`. Трудозатраты: S.

**F-L2. Major — `run.sh` не пишет в `logs/` (логирование только внутри приложения).** Падения до старта Python не фиксируются. Рекомендация: перенаправлять stdout/stderr лаунчера в лог. Трудозатраты: S.

**F-L3. Minor — ResourceGuard прерывает задачу только на безопасной точке.** Трудозатраты: S.

### 4.10 Тесты

- 1170 `def test_`. Прогон зелёный (проверено в этой сессии).
- **F-X1. Major — Smoke не проверяет MP3.** [tools/smoke.py:409](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/tools/smoke.py#L409): ключевой пользовательский выход не покрыт сквозной проверкой. Трудозатраты: S.
- **F-X2. Major — Нет тестов на гонки, отмену и падение модели.** Заглушки всегда успешны. Рекомендация: параметризовать заглушку инференса на исключение. Трудозатраты: M.
- **F-X3. Minor — Мусор в корне:** `dialogue.mp3`, `dialogue-2.mp3` (untracked), `total_audit.md`. Трудозатраты: S.

### 4.11 Конфигурация и зависимости

- **F-C1. Major — `torch==2.14.0` vs `torchaudio==2.11.0`.** [requirements.txt:7-8](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/requirements.txt#L7-L8). Разные минорные ряды — риск резолва/ABI. Трудозатраты: S.
- **F-C2. Info — venv 2.8 ГБ; 4 файла requirements (`-denoise`, `-kokoro`, `-qwen`); возможны дубликаты зависимостей.** Трудозатраты: S.

### 4.12 Безопасность

См. отдельный раздел 7 (⚠️ SECURITY).

### 4.13 Документация и drift

- **F-DC1. Minor — Drift** `docs/*` ↔ код: стадии пайплайна текста, поля встроенных голосов (частично поправлено в текущей незакоммиченной работе). Трудозатраты: S.
- **F-DC2. Info — нет `changelog`/`docs/history.md` с актуальной историей фаз.** Трудозатраты: S.

### 4.14 Лицензии

- **F-LIC1. Major — Лицензия весов `Ftfyhh/xttsv2_banana` не указана** в README/доках. F5-TTS Russian — CC-BY-NC-4.0, XTTS — CPML (некоммерческие). Рекомендация: явно указать лицензии весов и запрет коммерческого использования. Трудозатраты: S.

### 4.15 macOS / Apple Silicon

Все находки этого слоя вынесены в **раздел 11** со ссылками на пункты чек-листа §11.1–§11.11.

---

## 5. Узкие места и профиль ресурсов

| Операция | Текущая сложность | Узкое место | Ожидаемый выигрыш |
|---|---|---|---|
| Разбор диалога, 100 реплик | ~O(n²) | `parse_dialogue` (`dialogue_parser.py:307-318`) | Линейный проход — секунды на больших сценах |
| Пайплайн текста реплики | 2–3 прохода | перекомпиляция словаря + повторная нормализация | ~2× CPU на текстовую часть |
| `render_text` | блокирует loop | синхронный LLM-анализ (`main.py:1914`) | Сервер перестаёт «висеть» |
| `add_reference` | блокирует loop | синхронная запись (`main.py:1240`) | UI отзывчив при загрузке записи |
| Синтез реплики на MPS | 1 проход, без сброса кеша | `empty_cache` только при unload | Ограничение роста `driver_allocated_memory` (§11.2) |
| Постобработка ffmpeg | синхронный `subprocess.run` | нет `timeout` (`audio_analysis.py:102`) | Устойчивость к битому входу |
| Рендер диалога, память | пик «F5 + обе XTTS» | нет лимита MPS-fraction | Предотвращение swap/kernel panic |
| Экспорт | temp в системном temp | нет лимитов объёма | Защита от zip-bomb |
| Логи | линейный рост файла | нет ротации | Предсказуемый диск |
| БД | синхронные вызовы в корутинах | блокируют loop | Отзывчивость API |

---

## 6. Отказоустойчивость: матрица сценариев

| Сценарий | Текущее поведение | Риск | Что добавить |
|---|---|---|---|
| Пустой ввод | 400 «Текст пуст» (`main.py:1902`) | Нет | — (корректно) |
| Гигантский ввод (100k) | 400 по `MAX_TEXT_CHARS` в `render_text` | Нет | Лимит и в `import_project` |
| 100 реплик | Квадратичный парсинг + накопление take'ов | Major | Линейный парсер + TTL take'ов |
| 10 вкладок | Гонки ответов без `AbortController` | Major | Отмена предыдущих запросов |
| Падение модели на середине | Статус `error`, retry-асимметрия | Major | Единый retry + понятная ошибка |
| Отключение питания при записи take | `.part`/`.part.wav` остаются, БД не знает о них | Major | Уборка `.part*` при старте |
| Занятый порт после рестарта | `run.sh` видит живой pidfile и выходит | Minor | Автовыбор свободного порта / понятное сообщение |
| Нет ffmpeg | `RuntimeError` из `audio_analysis.py:111` | Major | Проверка в `doctor.py` + ранний отказ |
| Нет весов | `engine_available` = False, откат только для голоса с записью | Major | Понятный текст «веса не найдены» (§4.2) |
| Рестарт во время задачи | Задача исчезает без следа (нет persist) | Critical | Таблица `jobs` + восстановление |
| Заполнен диск | Нет проверки перед рендером | Major | `disk_usage` + ранний отказ |
| Повреждённая SQLite | Нет `integrity_check`/бэкапа при миграции | Major | Копия + `quick_check` при старте |
| Удалён `voices.json` | Библиотека не восстанавливается | Major | Пересборка из `voices/` |
| Два рендера одновременно | Один воркер, изоляция temp по uuid | Нет | — (изоляция есть) |
| Обрыв загрузки весов | Не детектируется | Major | sha256/размер |

---

## 7. Безопасность

Локальный loopback-сервис, но правила соблюдены не везде.

- ⚠️ **SECURITY Minor** — `innerHTML` без `esc()`: [app.js:1271](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L1271), [app.js:4263](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L4263). Значения — числовые id из БД, но правило нарушено.
- ⚠️ **SECURITY Minor** — `str(exc)` уходит в браузер (`main.py:492,499`) → утечка внутренних деталей.
- ⚠️ **SECURITY Major** — zip-bomb при импорте: [project_export.py:574-599](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/project_export.py#L574-L599) `_validate_entries` проверяет путь, тип и место назначения, но **не** размер и не число записей; `_extract` (`:602`) распаковывает без лимитов. Path-traversal закрыт (uuid-имена + `safe_component`).
- ⚠️ **SECURITY Major** — нет лимита размера тела запроса (`_read_json_body`, `main.py:1019`).
- ✅ Привязка к `127.0.0.1` корректна; CORS `*` не открыт; секретов в репозитории не найдено; `shell=True` — 0; `pickle.load` — 0; `eval(` — 1 вхождение, и это `model.eval()` в [kokoro_engine.py:281](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/engines/kokoro_engine.py#L281) (не проблема).
- ✅ SQL — только параметризованные запросы (f-string в SQL не найдено).

---

## 8. Технический долг

- **Архитектура:** очередь без persist; отсутствие слоя сервисов между API и хранилищем (часть логики прямо в роутерах); нет единого `exception_handler`/`/healthz`.
- **Код:** `main.py` 4135 строк и `app.js` 7591 строк — точки концентрации; 118 `except Exception` (часть — осознанный best-effort с `noqa`, часть — без трейсбека).
- **Тесты:** 1170 тестов, но нет покрытия гонок/отмены/падения модели и MP3 в smoke.
- **Документация:** drift `docs/text-pipeline.md` ↔ код; нет истории фаз.
- **Зависимости:** несогласованные пины torch/torchaudio; 4 файла requirements с возможными дублями; лицензии весов не задокументированы.

---

## 9. План действий

### 9.1. Немедленно (Blocker/Critical)
1. Персист очереди задач: таблица `jobs`, запись статусов, восстановление при старте. (**топ-1**)
2. `await asyncio.to_thread(...)` для `_analyze_text_chunks` и `get_store().add_reference`. (**топ-2**)
3. `timeout=` для `subprocess.run` в `audio_analysis.py:102`. (**топ-3**)
4. Ужесточить MPS-fallback: ловить только MPS-ошибки + `release_torch_memory()` перед перезагрузкой. (**топ-4**)

### 9.2. Спринт (Major)
5. `RotatingFileHandler` для логов. 6. Лимиты размера тела/архива. 7. `maxsize` очереди + prune по возрасту. 8. TTL take'ов + уборка файлов при каскадном удалении. 9. Кеш словаря произношения + один проход нормализации. 10. Согласовать пины torch/torchaudio. 11. `AbortController` во фронтенде + раздельные try в `init()`. 12. MPS-предохранители (`set_per_process_memory_fraction`, watermark-ratio). 13. `integrity_check`/бэкап при миграции. 14. Пересборка `voices.json` из `voices/`. 15. Проверка свободного места перед рендером. 16. Проверка целостности весов (sha256/размер). 17. Логирование лицензий весов.

### 9.3. Бэклог (Minor/Info)
18. NFC-нормализация текста. 19. `esc()` для всех `innerHTML`. 20. `/healthz` + request-id middleware. 21. Разбить `app.js` на модули. 22. `aria-*` и CSP. 23. Единый retry очереди. 24. Уборка `.part*` при старте. 25. MP3 в smoke-тесте. 26. Тесты на гонки/отмену/падение модели. 27. Убрать мусор из корня. 28. Докстринги и актуализация docs.

---

## 10. Приложения

### A. Команды для воспроизведения

Выполнено в этой сессии:
```bash
ruff check backend/ tools/ tests/     # 39 ошибок (см. ниже)
cloc backend/ frontend/ tools/ tests/ # 183 файла, 58 827 code
grep -rn "except Exception" backend/ | wc -l   # 118
grep -rn "except:" backend/ | wc -l            # 0
grep -rn "shell=True" backend/ | wc -l         # 0
grep -rn "eval(" backend/                      # 1 → model.eval(), не проблема
grep -rn "pickle.load" backend/ | wc -l        # 0
grep -rn "innerHTML" frontend/ | wc -l         # 60
grep -rn "time.sleep" backend/ | wc -l         # 3
grep -rn "asyncio.to_thread" backend/ | wc -l  # 86
grep -rn "@app\.(get|post|put|delete|patch)" backend/ | wc -l  # 93
grep -rn "def test_" tests/ | wc -l            # 1170
```

Рекомендуется после установки инструментов:
```bash
mypy backend/ --ignore-missing-imports
vulture backend/ --min-confidence 80
bandit -r backend/ -ll
pip-audit -r requirements.txt
radon cc backend/ -s -a && radon mi backend/ -s
```

`ruff check backend/ tools/ tests/` → 39 ошибок: `ISC004`×7, `I001`×7, `BLE001`×6, `B008`×3 (`main.py:1057,1091,1141` — `File()` в default), `B023`×3 (`tools/loadtest.py:335-337`, замыкание на переменные цикла — результаты нагрузочного теста не искажает), `EXE001`×2, `RUF046`×2, `F401`×2, по 1 `PYI034/UP031/TRY004/C401/RUF022/RUF100/F841` (`F841` — неиспользуемая `voice` в `main.py:2377`).

### B. Метрики

| Метрика | Значение |
|---|---|
| LOC (Python code) | 50 497 |
| LOC (JS/HTML/CSS) | 8 284 |
| LOC всего | 58 827 |
| Комментарии Python | 12 701 (25% code) |
| `except Exception` | 118 |
| `except:` (голый) | 0 |
| `shell=True` | 0 |
| `pickle.load` | 0 |
| `eval(` | 1 (`model.eval()`) |
| `time.sleep` | 3 |
| `subprocess` | 38 |
| `asyncio.to_thread` | 86 |
| Роутеров | 93 |
| Тестов | 1170 |
| TODO/FIXME/HACK/XXX | 0 |
| `print(` | 6 |
| venv | 2.8 ГБ |
| models/ | 15 ГБ |
| БД | 2.6 МБ |
| Лог | 592 КБ (без ротации) |

### C. Что не удалось проверить и почему

- **`mypy`, `vulture`, `bandit`, `pip-audit`, `radon`** — не установлены в venv. Типизация, мёртвый код, security-скан и цикломатика проверены вручную (grep + чтение), но не инструментально.
- **Реальный пик памяти `driver_allocated_memory()`** при «F5 + обе XTTS» — приложение остановлено, рантайм-замер не снимался. Оценки в §11 — из кода и документации, не из живого профиля.
- **Thermal-профиль** — `powermetrics` требует sudo и не запускался; вывод о throttling — из отсутствия логирования времени реплик, а не из замеров.
- **Kernel panic в `/Library/Logs/DiagnosticReports/`** — не проверялось (вне scope доступа сессии).
- **Ядерная безопасность лицензий весов** — тексты лицензий `xttsv2_banana`/`kokoro_ru` в репозитории не найдены.
- **Гонки в браузере (10 вкладок)** — только статический анализ `app.js`; нагрузочный прогон UI не выполнялся.

---

## 11. macOS / Apple Silicon: профиль и чек-лист

Все находки этого раздела относятся к слою «macOS / Apple Silicon» (scope §2) и размечены ссылками на пункты чек-листа.

### 11.1 Профиль памяти (что делает код)

Замеры живьём не снимались (приложение остановлено), поэтому таблица — из кода и структуры операций. Метрика указана та, которой пользуется код:

| Операция | Что читает код | Комментарий |
|---|---|---|
| Загрузка F5 | `driver_allocated_memory()` (через `/api/status`) | `resource_guard.mps_memory_snapshot` (`resource_guard.py:57-86`) — метрика корректная |
| Загрузка XTTS / XTTS-banana | то же | Веса ~5.2 ГБ у banana — главный вклад |
| Синтез реплики | **не логируется** | Нет до/после — утечка невидима (§11.2) |
| Рендер диалога | косвенно через `ResourceGuard` | Прерывание только на безопасной точке |
| Выгрузка | `release_torch_memory()` (`base.py:77-92`) | `gc.collect()` → `empty_cache()`, порядок верный |

Важно: `MAX_RSS_MB` опирается на **psutil RSS**, который, как честно написано в докстринге `resource_guard.py:60-65`, **не отражает аллокации Metal**. Отдельно есть корректный MPS-снапшот (`driver_allocated_memory` + `current_allocated_memory`), но `recommended_max_memory()` не используется, а `driver_allocated_memory()` не логируется до/после синтеза.

### 11.2 Очистка MPS-памяти и утечки

- ✅ `gc.collect()` вызывается **перед** `torch.mps.empty_cache()` ([base.py:85-90](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/engines/base.py#L85-L90)) — соответствует требованию §11.2.
- ❌ `empty_cache()` **не вызывается между синтезами** — только при выгрузке модели. `driver_allocated_memory()` может расти от реплики к реплике (§11.2, «главный симптом утечки»).
- ❌ Нет логирования `driver_allocated_memory()` до/после каждого синтеза.
- Severity: **Critical** (для длинных диалогов) — потенциальная утечка невидима.

### 11.3 Матрица MPS-переменных

| Переменная / вызов | Установлена в коде/лаунчере | Рекомендуется |
|---|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | ✅ `config.py:23-25` | оставить |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | ❌ | задать `1.0` (не `0.0`) |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | ❌ | задать `0.9` |
| `PYTORCH_MPS_PREALLOCATE` | ❌ | рассмотреть для TTS |
| `torch.mps.set_per_process_memory_fraction()` | ❌ | задать жёсткий потолок (§11.3/§11.4) |
| `torch.mps.recommended_max_memory()` | ❌ | логировать потолок |
| Обработка `RuntimeError` с `"mps"` в тексте | ⚠️ частично: `xtts_engine.py:188` ловит **любое** исключение | сузить до MPS-ошибок + fallback `.to('cpu')` |
| `num_workers=0` для DataLoader | н/п — DataLoader не используется | — |
| `pin_memory=True` на MPS | не используется (хорошо) | — |

### 11.4 Kernel panic и «other allocations»

- ❌ Нет проверки/логирования `RuntimeError: MPS backend out of memory` с выводом трёх чисел (**MPS allocated / other allocations / max allowed**).
- ❌ Нет механизма понижения модели при OOM (например, выгрузка `xtts-banana` при нехватке памяти).
- ❌ Нет проверки `Kernel_*.panic` в `/Library/Logs/DiagnosticReports/` при старте.
- Контекст: на 48 ГБ MPS может рапортовать `max allowed: 63.65 GiB`, что превышает физическую память — без `set_per_process_memory_fraction()` это путь к GPU panic. Severity: **Major** (на 8–16 ГБ — **Critical**).

### 11.5 Thermal throttling

- ❌ Нет логирования времени синтеза каждой реплики → thermal-деградация невидима.
- ❌ Нет паузы/кулдауна между репликами (в бенчмарках StyleTTS2 использовался cooldown).
- ❌ [tools/doctor.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/tools/doctor.py) не проверяет thermal state. (`memory_pressure` есть в `backend/llm/memory_policy.py:291`, но не в doctor.)
- Severity: **Major** — на 10+ минутах аудио рост времени реплик спишут на «модель тормозит».

### 11.6 Отладка Metal и профилирование

- ❌ Не использован Metal debugger / `gpudebug` / Metal System Trace.
- ❌ Не рассматривался `MTLCaptureManager` для захвата `.gputrace` из Python.
- Инфо: для TTS на MPS перспективен путь CoreML (StyleTTS2 → CoreML: −41% времени на cool-run, −47% диска).

### 11.7 DataLoader, потоки

- ✅ DataLoader/`num_workers`/`pin_memory` не используются — конфликт spawn/fork с MPS-контекстом отсутствует по построению (§11.7, «num_workers > 0 вызывает немедленное завершение» — неприменимо).
- ❓ `torch.set_num_threads()` для CPU-fallback не задан — на CPU-откате PyTorch может занять все ядра, конкурируя с MPS-потоком. Severity: **Minor**.

### 11.8 Специфика TTS на Apple Silicon

- ❌ Нет проверки длины входного текста для MPS-SDPA (~12 000 токенов — hard crash). Есть лишь `MAX_REPLICA_CHARS=600` (косвенная защита).
- ❌ Не рассматривалась `Kokoro-coreml` (82M, 22× realtime на M1 16 ГБ) как лёгкая альтернатива.
- ❓ Silent NaN при неконтигуозных тензорах (XTTS) — код не проверяет результат на NaN. Severity: **Major**.
- ❌ Нет стратегии сегментации для 10+ минут аудио.

### 11.9 Список macOS-специфичных находок

| Severity | Находка | Пункт | Рекомендация |
|---|---|---|---|
| Critical | Нет `empty_cache()`/`gc.collect()` между синтезами; дельта `driver_allocated_memory()` не логируется | §11.2 | Сброс + лог дельты на каждой реплике |
| Major | Не заданы `PYTORCH_MPS_*_WATERMARK_RATIO`, `set_per_process_memory_fraction()` | §11.3 | Задать в `configure_torch` |
| Major | Нет обработки MPS OOM тремя числами и понижения модели | §11.4 | Ловить OOM, логировать 3 числа, выгружать banana |
| Major | Нет логирования thermal/времени реплик, нет кулдауна | §11.5 | Логировать время реплик, добавить проверку в doctor |
| Major | MPS→CPU fallback на любое исключение | §11.3 | Сузить до MPS-ошибок |
| Minor | `MAX_RSS_MB` по psutil RSS (не Metal); `recommended_max_memory()` не используется | §11.1 | Бюджетировать по `driver_allocated_memory()` |
| Minor | `torch.set_num_threads()` для CPU-fallback не задан | §11.7 | Ограничить потоки на CPU |
| Info | Возможен CoreML для отдельных стадий; Metal debugger не применялся | §11.6 | Пилот на одной стадии |

### 11.10 Команды для сбора macOS-метрик (для владельца)

```bash
# Давление памяти (без sudo)
vm_stat 1
memory_pressure
sysctl -n vm.swapusage
sysctl -n kern.memorystatus_vm_pressure_level

# MPS из Python в рантайме приложения
#   torch.mps.driver_allocated_memory() / 1e9  → реальный пул Metal
#   torch.mps.current_allocated_memory() / 1e9 → живые тензоры
#   torch.mps.recommended_max_memory() / 1e9   → потолок системы

# Thermal/power (sudo, bounded sample 1–2 с, не оставлять открытым)
sudo powermetrics --samplers thermal -n 1
sudo powermetrics --samplers cpu_power,gpu_power --show-usage-summary -n 1

# Паники ядра
ls /Library/Logs/DiagnosticReports/Kernel_*.panic
log show --predicate 'eventMessage CONTAINS "Previous shutdown cause"' --last 7d
```

### 11.11 Рекомендации по CoreML

Кандидаты на конвертацию (по убыванию выигрыша): акустическая модель XTTS/F5 (основное время), G2P/акцентуация (RUAccent уже ONNX), денойзер референса. Ожидаемый выигрыш по внешнему бенчмарку StyleTTS2 → CoreML: **−41% времени** (mixed-precision fp16 + fp32 на cumsum-чувствительных стадиях) и **−47% размера** на диске. Для лёгкого пути — оценить `Kokoro-coreml` (82M) вместо тяжёлых XTTS на машинах ≤16 ГБ.

---

### Definition of Done — отметка

- ✅ Покрыты все 15 слоёв (§4.1–4.15), включая macOS/Apple Silicon (§11).
- ✅ Топ-10 с severity и трудозатратами (§3).
- ✅ Каждая находка с `файл:строка` и цитатой.
- ✅ Матрица отказоустойчивости на 15 сценариев (§6).
- ✅ План «немедленно / спринт / бэклог» (§9).
- ✅ Приложения с командами и метриками (§10 A/B).
- ✅ Раздел «что не удалось проверить» (§10 C).
- ✅ macOS-находки вынесены в §11.
- ⚠️ Инструментальная проверка `mypy`/`bandit`/`vulture`/`pip-audit`/`radon` не выполнена (не установлены) — зафиксировано в §10 C.
