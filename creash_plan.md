# План: изоляция TTS-воркера и восстановление после падений

Рабочий документ к `creash_report.md`. Сначала — разведка (§35 отчёта), затем —
минимальный план по фазам. Большой рефакторинг сюда не входит: изоляция
встраивается в существующие швы, второй реализации F5/XTTS не появляется.

## 1. Разведка: как оно устроено сейчас

| Что | Где | Состояние |
| --- | --- | --- |
| Очередь задач | `backend/job_queue.py` | Один последовательный воркер (`asyncio`), приоритеты, кооперативная отмена, `abort_kind` (user/watchdog), ETA. Всё в процессе бэкенда. |
| Вызов инференса | `audio_pipeline._synthesize_in_thread` → `SynthesisEngine.synthesize` | Синхронный вызов в `asyncio.to_thread`, то есть **в процессе бэкенда**. Единственная точка входа в модель. |
| Движки | `backend/engines/base.py`, `f5_engine.py`, `xtts_engine.py`, `registry.py` | ABC `SynthesisEngine` + паспорта `ENGINE_INFOS`; ленивое создание по одному тёплому экземпляру на процесс; `unload()` с `EngineBusyError`; авто-выгрузка по простою (`engine_lifecycle.py`). |
| Изоляция процесса — уже есть | `transcribe.py` + `transcribe_worker.py`, `denoise*.py` | Whisper и шумодав запускаются отдельными процессами (`subprocess.run`, JSON в stdout, логи в stderr, `cwd=config.BASE_DIR`). Готовый образец для TTS-воркера, но одноразовый: модель нужно держать тёплой. |
| Персистентность | `backend/db/` (миграции 1–5), `store.py`, репозитории | Проекты/спикеры/реплики/варианты/произношение. `replicas.status` — свободный текст (`pending`/`rendered`), `projects.status` — `draft`/`rendering`/`rendered`/`error`, `projects.last_error` есть. |
| Запись аудио | `audio_pipeline._write_audio`, `write_take`, `export_project_chunks` | Пишет **сразу в целевой файл**; временный `.tmp.wav` появляется только на пути mp3. Валидации результата нет. |
| Отмена | `job_queue.cancel`, `request_abort`, `audio_pipeline.should_abort` | Кооперативная, между кусками. Отмена и прерывание watchdog'ом различаются (`_finish_interrupted`). |
| Память | `resource_guard.py` | RSS процесса бэкенда (`MAX_RSS_MB`) + системный порог (`SYSTEM_MEM_THRESHOLD_PERCENT`); `_wait_for_memory` держит задачу в очереди. Состояний NORMAL/WARNING/CRITICAL нет, память воркера (если он станет отдельным) не видна. |
| Старт/стоп | `main.lifespan` | `init_db`, старт очереди, `ResourceGuard`, прогрев F5 в фоне (`_warmup`), watcher простоя; на выходе — `queue.stop()` и `cancel()` фоновых задач. Восстановления «зависших» статусов после рестарта нет. |
| Health | `GET /api/status` | Состояния движков, акцентуатор, очередь, RSS/CPU/MPS. Лёгкий, модели не трогает. |

Что из этого — причина пакета: **падение инференса (SIGABRT/SIGSEGV/OOM в
нативной библиотеке) убивает весь бэкенд**, потому что модель живёт в том же
процессе. Всё остальное (потеря незавершённых реплик, «зависшие» статусы после
рестарта, частично записанный WAV) — следствия того же решения.

## 2. Целевая архитектура (минимальная)

```
FastAPI (процесс A)                          worker (процесс B, spawn через Popen)
  job_queue ── audio_pipeline ── WorkerEngine ──pipe(pickle)── worker_process
     │                              (прокси,                        │
     │                               тот же ABC)                    └── registry._create(engine_id)
     │                                                                   → F5Engine / XTTS
     └── EngineSupervisor (pid, состояние, exit code, signal,
         счётчики падений, restart, DEGRADED)
```

Ключевое решение: **изоляция встраивается в `SynthesisEngine`**. `WorkerEngine`
наследует тот же ABC и реализует `load()`/`_synthesize()`/`_release()` через IPC,
поэтому пайплайн, QA, варианты, сиды, ETA, перегенерация и выгрузка по простою
работают без изменений, а второй реализации моделей не появляется: в дочернем
процессе движок создаёт тот же `registry._create`.

Транспорт: `subprocess.Popen([sys.executable, "-m", "backend.engines.worker_process", …])`
с двумя `multiprocessing.Pipe(duplex=False)`, чьи дескрипторы передаются через
`pass_fds`. Проверено на этой машине: передача float32-массива, `poll(deadline)`
для таймаута, падение ребёнка → `EOFError` у родителя и `returncode = -6`
(SIGABRT). `Popen`, а не `fork`/`multiprocessing.Process`: на macOS `fork` после
инициализации MPS в родителе — известный источник падений, а `spawn` требует
сериализуемости цели и повторного импорта `__main__` (у нас это uvicorn).

## 3. Фазы

**Ф1. Воркер и супервизор.** `engines/worker_protocol.py` (кадры, имена запросов,
таксономия ошибок), `engine/worker_process.py` (цикл: `load`/`synthesize`/`unload`/`ping`/`shutdown`,
EOF → выход, логи в stderr), `engines/worker_engine.py` (`WorkerEngine(SynthesisEngine)`),
`engines/supervisor.py` (состояния STARTING/IDLE/BUSY/CRASHED/RESTARTING/STOPPED/DEGRADED,
pid, exit code, signal, счётчик падений, окно crash-loop, `shutdown_all`).
Флаг `TTS_WORKER_ISOLATION` (по умолчанию включён; тесты выключают), фабрика
движка через `TTS_WORKER_ENGINE_FACTORY` — тестовый шов, чтобы в дочернем процессе
не поднимать настоящую модель.

**Ф2. Падение → состояние задачи и реплики.** `replicas.status` получает
`rendering` и `interrupted`; миграция 6 — таблица `worker_crashes` (job, project,
replica, движок, pid, exit code, signal, error_type, время) — диагностика не
пишется в строку реплики. Очередь: `Job.error_type` (`CANCELLED` / `WORKER_CRASH` /
`WORKER_TIMEOUT` / `TTS_ERROR` / `WATCHDOG`), реплика перед инференсом помечается
`rendering`, при падении — `interrupted`, уже готовые куски сохраняются вариантами,
ограниченный автоповтор с продолжением с упавшей реплики (готовые куски читаются
обратно, финальная нормализация громкости применяется ко всему треку целиком).

**Ф3. Атомарная запись, валидация, восстановление.** `_write_audio` пишет в
`*.part` → проверяет файл (`soundfile.info`: длительность > 0, sample rate) →
`os.replace`. Валидация аудио перед `DONE`. При старте: `rendering` → `interrupted`
у реплик, `projects.status='rendering'` → `draft` + `last_error`, удаление
осиротевших `*.part`.

**Ф4. Память и выключение.** `memory_monitor.get_state()` → NORMAL/WARNING/CRITICAL
(инъекция для тестов), в CRITICAL задача не стартует новый кусок, сначала
best-effort выгрузка простаивающих движков, затем ожидание; `/api/status` получает
состояние воркеров и памяти, лёгкие роуты воркер не ждут. На выходе —
`supervisor.shutdown_all()` без осиротевших процессов; воркер сам выходит по EOF.

**Ф5. Документация и проверки.** README (раздел про изоляцию и recovery),
`updates.md`-стиль отчёт, Obsidian, полная регрессия `pytest`, живой сценарий с
контролируемым падением воркера на работающем сервере, smoke-прогон, финальный
отчёт по 15 пунктам §38.

## 4. Границы (чего не делаем)

* Второй пайплайн синтеза и вторая реализация F5/XTTS — не появляются.
* Падение воркера не маскируется «тихой» подменой результата: задача получает
  честный статус ошибки с типом.
* База не пересоздаётся; существующие проекты и правила произношения не трогаются.
* Один воркер очереди сохраняется.
* Таймауты и watchdog не отключаются ради «зелёных» тестов.
