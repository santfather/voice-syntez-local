# TTS Worker Isolation & Crash Recovery

## Цель

Повысить отказоустойчивость Voice_syntez при аварийном завершении нативного ML/TTS-кода.

На macOS ARM64 периодически наблюдается аварийное завершение Python:

```text
Exception Type: EXC_CRASH (SIGABRT)
Termination Reason: Namespace SIGNAL, Code 6
Abort trap: 6
```

Такой crash может происходить внутри PyTorch/MPS/Metal/TTS или другой нативной зависимости и не обязан превращаться в Python exception. Поэтому `try/except` внутри основного backend-процесса недостаточно.

Основная задача: **авария TTS inference не должна завершать backend, уничтожать очередь, повреждать проект или терять уже успешно сгенерированные реплики.**

---

# 1. Архитектура

Не выполнять тяжёлый TTS inference непосредственно в основном FastAPI/backend процессе.

Целевая схема:

```text
Application / UI
       │
       ↓
FastAPI Backend
       │
       ↓
Job Queue
       │
       ↓
TTS Worker Supervisor
       │
       └── TTS Worker Process
              │
              ├── F5-TTS
              └── XTTS
```

Допустима архитектура с отдельными worker-процессами для движков:

```text
TTS Supervisor
   ├── F5 Worker
   └── XTTS Worker
```

если это лучше соответствует существующему Engine Registry.

Главное требование:

```text
SIGABRT / SIGSEGV / native crash TTS worker
                ↓
backend продолжает работать
                ↓
текущий job корректно помечается
                ↓
worker перезапускается
                ↓
приложение остаётся пригодным для дальнейшей работы
```

---

# 2. Не переписывать существующий TTS pipeline

Сохранить существующие:

* Engine Registry;
* F5-TTS engine;
* XTTS engine;
* job queue;
* project/replica persistence;
* preprocessing;
* pronunciation pipeline;
* Smart QA;
* variants;
* regeneration;
* seed;
* существующие API contracts там, где это возможно.

Worker isolation должен быть транспортным/исполнительным слоем вокруг существующих engine implementations.

Не создавать вторую независимую реализацию F5/XTTS.

---

# 3. TTS Worker Process

Inference должен выполняться в отдельном OS process.

Не использовать только Python thread.

Причина:

```text
Thread
  └── native abort
        ↓
      весь Python process погибает
```

Нужно:

```text
Backend process
     │
     └── Worker process
              ↓
          native abort
              ↓
        погибает worker
              ↓
        backend жив
```

Использовать подходящий механизм multiprocessing/subprocess с учётом macOS.

На macOS не полагаться на небезопасное наследование уже инициализированного PyTorch/MPS state через `fork`.

Предпочитать безопасный lifecycle worker, совместимый со `spawn`.

ML/MPS initialization должен происходить внутри worker process.

---

# 4. Worker Supervisor

Реализовать supervisor, который знает:

```text
worker pid
worker state
current job
engine
start time
last heartbeat/activity
exit code
restart count
```

Минимальные состояния:

```text
STARTING
IDLE
BUSY
CRASHED
RESTARTING
STOPPED
```

При нормальной работе:

```text
IDLE
 ↓
BUSY
 ↓
IDLE
```

При crash:

```text
BUSY
 ↓
CRASHED
 ↓
RESTARTING
 ↓
STARTING
 ↓
IDLE
```

---

# 5. Обнаружение native crash

Supervisor должен отслеживать завершение worker process.

Например:

```text
exitcode == 0
```

нормальное завершение.

На Unix/macOS отрицательный exit code может соответствовать signal termination.

Для SIGABRT ожидаемо что-то эквивалентное:

```text
signal = SIGABRT
exitcode = -6
```

Не привязывать всю архитектуру только к `-6`.

Обрабатывать как worker crash также другие неожиданные завершения, включая потенциальные:

```text
SIGABRT
SIGSEGV
SIGBUS
SIGKILL
```

и любые неожиданные non-zero exits.

Сохранять диагностическую причину, если она известна.

---

# 6. Backend не должен падать вместе с worker

Обязательное требование:

```text
kill(worker, SIGABRT)
```

не должен:

* завершать FastAPI;
* повреждать SQLite;
* уничтожать проект;
* терять уже завершённые replicas;
* оставлять job навечно в `running`;
* блокировать очередь навечно.

После crash обычные лёгкие API должны продолжать отвечать.

Например:

```text
GET project
GET replicas
GET voices
GET jobs
GET health
```

---

# 7. Persistence перед inference

Перед передачей реплики worker сохранить её состояние в DB.

Пример:

```text
QUEUED
 ↓
RENDERING
 ↓
DB COMMIT
 ↓
send to worker
```

Нельзя делать:

```text
send to worker
 ↓
успех
 ↓
только теперь впервые записать состояние
```

потому что native crash может произойти раньше.

---

# 8. Crash state

Если worker погиб во время генерации реплики:

```text
replica.status = interrupted
```

или использовать эквивалентное существующее состояние.

Сохранить диагностические данные:

```text
job_id
replica_id
engine
worker_pid
exit_code
signal
started_at
interrupted_at
retry_count
error_type
```

Не сохранять огромные traceback/crash reports непосредственно в основную таблицу replica, если для этого уже существует job/error/log subsystem.

---

# 9. Не считать crash обычной ошибкой TTS

Различать:

```text
Python inference exception
```

и:

```text
worker process crashed
```

Например:

```text
TTS_ERROR
WORKER_CRASH
WORKER_TIMEOUT
CANCELLED
```

Это необходимо для диагностики.

---

# 10. Автоматический restart worker

После неожиданного завершения worker supervisor должен автоматически создать новый worker.

Последовательность:

```text
worker crash
 ↓
record crash
 ↓
mark current replica interrupted
 ↓
clean IPC resources
 ↓
spawn new worker
 ↓
initialize engine when required
 ↓
worker READY
```

Backend перезапускать не требуется.

---

# 11. Защита от restart loop

Нельзя бесконечно перезапускать worker, если одна и та же модель немедленно падает.

Добавить crash-loop protection.

Например:

```text
3 crashes within 60 seconds
```

→ временно:

```text
worker = DEGRADED
engine = temporarily unavailable
```

Конкретные значения оформить конфигурацией.

UI/API должны получить понятную ошибку:

```text
TTS worker repeatedly crashed.
Engine temporarily unavailable.
```

Не продолжать бесконечный restart loop.

---

# 12. Retry policy

Не делать бесконечный автоматический retry одной и той же реплики.

Безопасный default:

```text
native worker crash
→ replica interrupted
→ worker restart
→ пользователь может Retry
```

Если существующая архитектура уже имеет надёжную retry policy, допускается один автоматический retry.

Но обязательно ограничить:

```text
max_worker_crash_retries
```

чтобы плохой input/model не создавал:

```text
crash
restart
same job
crash
restart
same job
...
```

---

# 13. Сохранение завершённых реплик

При batch/dialogue render:

```text
Replica 1 → DONE
Replica 2 → DONE
Replica 3 → DONE
Replica 4 → CRASH
```

после crash:

```text
Replica 1 → DONE
Replica 2 → DONE
Replica 3 → DONE
Replica 4 → INTERRUPTED
```

Первые три WAV нельзя удалять или пересоздавать без необходимости.

После восстановления должна быть возможность продолжить с Replica 4.

---

# 14. Atomic audio write

Worker не должен писать непосредственно в окончательный WAV так, чтобы crash мог оставить повреждённый файл, выглядящий как валидный результат.

Использовать схему:

```text
replica_004.wav.tmp
        ↓
generate/write
        ↓
flush/close
        ↓
validate
        ↓
atomic rename
        ↓
replica_004.wav
```

Если worker погиб:

```text
replica_004.wav.tmp
```

не считается завершённым audio asset.

При startup/recovery временные файлы можно безопасно удалить.

---

# 15. Audio validation перед commit

Перед переводом replica в `DONE` проверить минимум:

* файл существует;
* файл можно открыть;
* duration > 0;
* размер > разумного минимального значения;
* audio metadata читается.

Только после этого:

```text
temp → final
DB status → DONE
```

Не помечать replica `DONE`, пока final audio asset не подтверждён.

---

# 16. Recovery после backend restart

Backend также может быть принудительно закрыт пользователем или IDE.

При startup найти stale состояния:

```text
RENDERING
RUNNING
```

для которых уже нет живого worker/job.

Перевести их в:

```text
INTERRUPTED
```

или эквивалентное recovery state.

Нельзя оставлять:

```text
rendering forever
```

---

# 17. Queue recovery

После worker crash очередь не должна навсегда останавливаться.

После восстановления worker:

```text
Worker READY
 ↓
queue dispatcher resumes
```

При этом interrupted job не должен автоматически повторяться бесконечно.

Определить явное поведение:

```text
current crashed replica → INTERRUPTED

following queued jobs
→ либо продолжаются
→ либо queue переходит в PAUSED
```

Выбрать вариант, наиболее совместимый с текущим UX.

Предпочтительно для первого безопасного варианта:

```text
worker crash
→ current job interrupted
→ worker restart
→ очередь может продолжить следующие независимые jobs
```

если это не нарушает последовательность одного dialogue render.

Для одного dialogue render допустимо остановить именно этот project render и дать пользователю `Resume`.

---

# 18. Memory Pressure Guard

Добавить отдельный контроль memory pressure перед запуском тяжёлого inference.

Цель — уменьшить вероятность MPS/native crash из-за memory pressure.

Состояния:

```text
NORMAL
WARNING
CRITICAL
```

Логику адаптировать под macOS и существующий stack.

Не хардкодить предположение, что любой SIGABRT вызван памятью.

Memory guard — профилактический механизм, а не диагноз всех crash.

---

# 19. Поведение при memory pressure

### NORMAL

Разрешить inference.

### WARNING

Не запускать новую тяжёлую операцию, если уже выполняется другая ML operation.

Допускается короткое ожидание освобождения ресурсов.

### CRITICAL

Не запускать новый TTS inference.

Попытаться:

```text
unload unused engine
release unused model references
gc.collect()
clear applicable backend caches
```

после чего повторно проверить memory pressure.

Если ресурсов всё ещё недостаточно:

```text
job remains queued / waiting_for_memory
```

а backend остаётся доступным.

---

# 20. Лёгкие API не блокировать

Memory pressure не должен делать всё приложение неработоспособным.

Даже если новый TTS inference временно запрещён, должны работать:

```text
projects
settings
voices
text editing
pronunciation
preview where it doesn't require heavy TTS model
job status
health
```

---

# 21. Model unload

Добавить безопасный механизм:

```text
unload_engine(engine_id)
```

если его ещё нет.

Он должен:

1. убедиться, что engine не выполняет inference;
2. удалить model references;
3. освободить доступные runtime resources;
4. выполнить cleanup;
5. перевести engine state в `UNLOADED`.

Следующий запрос может снова lazy-load модель.

Не unload активную модель посреди inference.

---

# 22. Не держать одновременно ненужные тяжёлые модели

Проверить сценарий:

```text
F5 loaded
XTTS loaded
Whisper loaded
RUAccent loaded
```

и определить, какие модели реально должны одновременно находиться в памяти.

Не менять архитектуру Smart QA без необходимости, но учитывать его memory footprint.

Если Whisper уже изолирован в отдельный процесс, сохранить эту изоляцию.

---

# 23. Health endpoint

Расширить существующий health/status API или добавить worker status.

Полезная диагностика:

```json
{
  "backend": "ok",
  "tts_worker": {
    "state": "idle",
    "pid": 12345,
    "engine": "f5",
    "restart_count": 1,
    "last_exit_code": -6
  }
}
```

Не обязательно сохранять именно эту schema — адаптировать к текущему API.

---

# 24. Logging

При worker crash записывать:

```text
timestamp
worker pid
engine
job id
project id
replica id
exit code
signal
memory state if available
restart attempt
```

Не логировать целиком большие пользовательские тексты.

Для текста достаточно:

```text
length
hash
короткий truncated preview
```

если preview действительно нужен для диагностики.

---

# 25. macOS-specific considerations

Проект работает на Apple Silicon.

Проверить:

```text
multiprocessing start method
PyTorch initialization
MPS initialization
Metal resources
model loading lifecycle
```

Не наследовать инициализированный MPS/PyTorch context через небезопасный process fork.

Worker должен самостоятельно импортировать/инициализировать необходимые inference resources после spawn, если этого требует используемый stack.

Не делать глобальную инициализацию тяжёлых моделей при import backend module.

---

# 26. Graceful shutdown

При штатном завершении приложения:

```text
backend shutdown
 ↓
stop accepting new jobs
 ↓
finish/cancel current work according to policy
 ↓
send worker shutdown
 ↓
wait limited timeout
 ↓
terminate if necessary
```

Не оставлять orphan Python worker processes.

---

# 27. Cancellation

Существующий job cancellation не должен сломаться после process isolation.

Если текущую TTS операцию нельзя безопасно прервать внутри native inference, допускается:

```text
cancel job
→ terminate worker
→ mark job CANCELLED
→ spawn clean worker
```

Но отличать intentional cancellation от crash:

```text
CANCELLED != WORKER_CRASH
```

---

# 28. Обязательные тесты

Добавить минимум:

```text
test_tts_worker_sigabrt_does_not_kill_backend
test_worker_is_restarted_after_crash
test_crashed_replica_marked_interrupted
test_completed_replicas_survive_worker_crash
test_project_survives_backend_restart
test_queue_continues_after_worker_restart

test_memory_pressure_blocks_new_inference
test_memory_pressure_does_not_block_light_api
test_unused_engine_can_be_unloaded

test_partial_audio_is_not_committed
test_audio_write_is_atomic
test_rendering_state_recovers_after_restart
```

---

# 29. Дополнительные обязательные тесты

Также добавить:

```text
test_worker_nonzero_exit_is_detected
test_worker_sigsegv_is_detected
test_worker_crash_records_exit_code
test_worker_crash_does_not_corrupt_sqlite

test_worker_restart_returns_to_ready
test_worker_restart_loop_is_limited
test_same_crashing_job_is_not_retried_forever

test_completed_audio_remains_after_next_replica_crash
test_interrupted_replica_can_be_retried
test_retry_creates_valid_audio

test_temp_audio_removed_after_crash
test_done_status_requires_valid_audio

test_backend_startup_marks_stale_render_as_interrupted
test_backend_startup_cleans_stale_temp_audio

test_graceful_shutdown_stops_worker
test_shutdown_does_not_leave_orphan_worker

test_cancelled_worker_job_is_not_reported_as_crash
```

---

# 30. Как тестировать настоящий SIGABRT

Не пытаться вызвать настоящий crash PyTorch/MPS в unit test.

Создать test worker/task, который намеренно выполняет:

```python
os.abort()
```

или эквивалентный controlled crash внутри **дочернего test process**.

Проверить:

```text
worker → SIGABRT
backend/test supervisor → жив
exit detected
job interrupted
worker restarted
```

Никогда не вызывать `os.abort()` в процессе самого pytest runner.

---

# 31. Integration crash test

Сценарий:

```text
start backend/supervisor
 ↓
queue Replica A
 ↓
Replica A completes
 ↓
queue Replica B
 ↓
worker intentionally SIGABRT during B
 ↓
verify backend alive
 ↓
verify A = DONE
 ↓
verify B = INTERRUPTED
 ↓
verify worker restarted
 ↓
retry B
 ↓
B = DONE
```

Это один из ключевых acceptance tests задачи.

---

# 32. Memory tests

Memory-pressure unit tests не должны требовать реально исчерпать RAM Mac.

Абстрагировать получение memory state:

```python
memory_monitor.get_state()
```

и в tests подменять:

```text
NORMAL
WARNING
CRITICAL
```

Проверить policy отдельно от реального системного мониторинга.

---

# 33. Не делать опасный recovery

После worker crash запрещено автоматически считать:

```text
temp WAV exists
```

доказательством успешного inference.

Также запрещено:

```text
status = DONE
```

только потому, что процесс успел создать файл.

Нужна завершённая atomic transaction:

```text
valid audio
+
atomic final file
+
DB completion
```

---

# 34. Definition of Done

Задача считается выполненной только если:

* TTS inference изолирован от FastAPI process;
* controlled `SIGABRT` worker не завершает backend;
* supervisor обнаруживает crash;
* crash фиксируется как отдельный тип ошибки;
* текущая replica становится `INTERRUPTED`;
* уже готовые replicas остаются `DONE`;
* partial WAV не становится готовым asset;
* запись audio выполняется atomically;
* worker автоматически восстанавливается;
* restart loop ограничен;
* одна crashing replica не создаёт бесконечный retry;
* очередь восстанавливается согласно выбранной policy;
* проект переживает backend restart;
* stale `RENDERING` восстанавливается как `INTERRUPTED`;
* memory pressure может блокировать новый inference;
* при этом lightweight API остаются доступными;
* неиспользуемый engine можно unload;
* graceful shutdown не оставляет orphan workers;
* cancellation отличается от worker crash;
* все существующие тесты продолжают проходить;
* все перечисленные crash/recovery tests проходят.

---

# 35. Порядок работы ИИ-агента

Перед изменениями:

1. изучить текущую job queue;
2. изучить Engine Registry;
3. определить, где именно сейчас вызывается `engine.synthesize()`;
4. определить lifecycle F5/XTTS;
5. проверить, какие модели инициализируются при import;
6. проверить существующий Whisper process isolation;
7. изучить persistence Job/Project/Replica;
8. изучить cancellation;
9. изучить запись WAV;
10. проверить startup/shutdown FastAPI.

После этого составить минимальный implementation plan.

Не начинать с большого рефакторинга.

---

# 36. Рекомендуемый порядок реализации

## Phase 1 — Worker isolation

Вынести TTS inference в отдельный process.

### Tests

```text
test_worker_can_synthesize
test_tts_worker_sigabrt_does_not_kill_backend
```

### Expected result

Controlled `SIGABRT` уничтожает только worker.

---

## Phase 2 — Supervisor/restart

Добавить обнаружение crash и restart.

### Tests

```text
test_worker_is_restarted_after_crash
test_worker_restart_returns_to_ready
test_worker_restart_loop_is_limited
```

### Expected result

После аварии создаётся новый пригодный к работе worker.

---

## Phase 3 — Job/Replica recovery

Связать crash с persistence.

### Tests

```text
test_crashed_replica_marked_interrupted
test_completed_replicas_survive_worker_crash
test_worker_crash_does_not_corrupt_sqlite
test_interrupted_replica_can_be_retried
```

### Expected result

Crash одной реплики не уничтожает результаты проекта.

---

## Phase 4 — Atomic audio

Перевести запись результатов на temp + validate + atomic rename.

### Tests

```text
test_partial_audio_is_not_committed
test_audio_write_is_atomic
test_done_status_requires_valid_audio
test_temp_audio_removed_after_crash
```

### Expected result

Повреждённый/частичный WAV никогда не считается готовым.

---

## Phase 5 — Startup recovery

Добавить восстановление stale jobs.

### Tests

```text
test_project_survives_backend_restart
test_rendering_state_recovers_after_restart
test_backend_startup_marks_stale_render_as_interrupted
```

### Expected result

После перезапуска приложение возвращается в консистентное состояние.

---

## Phase 6 — Queue recovery

Восстановить dispatcher после worker restart.

### Tests

```text
test_queue_continues_after_worker_restart
test_same_crashing_job_is_not_retried_forever
```

### Expected result

Одна авария не блокирует всю очередь.

---

## Phase 7 — Memory guard/unload

Добавить memory pressure abstraction и безопасный unload.

### Tests

```text
test_memory_pressure_blocks_new_inference
test_memory_pressure_does_not_block_light_api
test_unused_engine_can_be_unloaded
```

### Expected result

Тяжёлый inference не стартует при критическом memory pressure, но приложение остаётся доступным.

---

## Phase 8 — Shutdown/cancellation

Проверить lifecycle процессов.

### Tests

```text
test_graceful_shutdown_stops_worker
test_shutdown_does_not_leave_orphan_worker
test_cancelled_worker_job_is_not_reported_as_crash
```

### Expected result

Worker lifecycle полностью контролируется приложением.

---

# 37. Финальная проверка

После реализации обязательно выполнить полный regression suite проекта.

Затем выполнить controlled crash scenario:

```text
F5/XTTS worker running
        ↓
controlled SIGABRT
        ↓
backend survives
        ↓
current replica INTERRUPTED
        ↓
previous audio preserved
        ↓
worker restarted
        ↓
retry succeeds
```

После этого выполнить реальный smoke test:

```text
F5 generation
XTTS generation, если доступен
Dialogue render
Continuous render
Regenerate replica
Smart QA
Cancel job
Application restart/recovery
```

Не считать задачу выполненной только по unit tests.

---

# 38. Финальный отчёт ИИ-агента

После выполнения предоставить:

```text
1. Root architecture before/after
2. Изменённые файлы
3. Новые DB migrations
4. Worker IPC mechanism
5. Worker lifecycle
6. Crash detection mechanism
7. Restart policy
8. Retry policy
9. Memory-pressure policy
10. Atomic audio strategy
11. Startup recovery strategy
12. Результаты unit tests
13. Результаты integration crash tests
14. Результаты smoke tests
15. Известные ограничения
```

Если в процессе будет обнаружена конкретная первопричина исходного macOS `SIGABRT`, зафиксировать её отдельно, но **не удалять worker isolation только потому, что найден один конкретный crash bug**. Изоляция нужна как системная защита от native failures TTS/ML stack.
