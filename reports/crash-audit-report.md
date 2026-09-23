# CRASH_AUDIT — VOICE_SYNTEZ

**Дата аудита:** 2026-09-23
**Коммит:** `511aa3a1dc70bf657d1ae8dd14a7a22e805ece63` (2026-09-23 13:55:13 +0200, `DOCS: §9.3 п.28 …`)
**Рабочее дерево:** не чистое — `git status --short` показывает 20+ изменённых файлов (`M backend/accentizer.py`, `M backend/engines/base.py`, `M backend/tts_engine.py`, …). Аудит проводился по рабочему дереву, а не по коммиту.
**Окружение:**

| Параметр | Значение |
| --- | --- |
| macOS | 27.0 (26A428), `arm64`, Mac15,6 |
| Память | 18 ГБ (`hw.memsize` = 19 327 352 832) |
| Python | 3.11.15 (Homebrew `python@3.11` 3.11.15_1, сборка `Clang 21.0.0`) |
| Интерпретатор | `venv/bin/python` → `/opt/homebrew/Cellar/python@3.11/3.11.15_1/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python` |
| torch / torchaudio | 2.11.0 / 2.11.0 |
| numpy | 2.4.6 |
| transformers / tokenizers | 5.17.0 / 0.23.2 |
| Запуск продукта | `run.sh:137` — `exec uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"` (без `--workers`, то есть 1 воркер) |
| IDE | Trae (Electron), coalition `com.trae.app`, песочница `TOOLHOST_SANDBOX_DISABLED=false` |

**Входные данные:** `python_issue.md` (два crash-отчёта), `crash_audit.md` (ТЗ), `logs/voice_syntez.log` (5634 строки, 2026-09-17 10:43:08 → 2026-09-23 13:07:59), `~/Library/Logs/DiagnosticReports/*.ips` (55 активных + 172 в `Retired/`), код репозитория.

---

## 1. Первопричина (Root Cause)

**Первопричина: это не дефект продукта, а намеренный вызов `os.abort()` в тестовой заглушке. `EXC_CRASH (SIGABRT)` в отчётах `Python-*.ips` порождает `tests/fake_worker_engine.py` (строки 83 и 88), который выполняется в дочернем процессе-воркере, поднимаемом тестами изоляции. Продуктовый код `os.abort()` не вызывает ни в одном файле.**

Доказательства, по возрастанию силы.

### 1.1. Прямое воспроизведение с перехваченным Python-стеком

Драйвер поднимает реальный `WorkerSupervisor` (`backend/engines/supervisor.py`) с тестовой фабрикой и отправляет воркеру текст `CRASH`. Шим `sitecustomize.py` (инъекция только в окружение дочернего процесса через `PYTHONPATH`, см. `supervisor.py:622-639` — `_child_env()` передаёт `sys.path` родителя целиком) оборачивает `os.abort` и печатает стек.

Фактический вывод (stderr воркера, перенаправленный драйвером `_start_drains`):

```
=== TRACED os.abort() ===
pid=64160
sys.argv=['/Users/…/backend/engines/worker_process.py', 'f5', '10', '13']
sys.executable='/Users/…/VOICE_SYNTEZ/venv/bin/python'
cwd='/Users/…/VOICE_SYNTEZ'
in sys.modules: torch                          True      <- «тёплый» прогон
in sys.modules: numpy                          True
in sys.modules: multiprocessing                True
in sys.modules: multiprocessing.shared_memory  False
--- python stack ---
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "backend/engines/worker_process.py", line 250, in <module>
    raise SystemExit(main())
  File "backend/engines/worker_process.py", line 231, in main
    answer = handle_request(message, holder, engine_id)
  File "backend/engines/worker_process.py", line 166, in handle_request
    waveform, sample_rate = holder.engine.synthesize(
  File "backend/engines/base.py", line 840, in synthesize
    waveform, sample_rate = self._synthesize(
  File "tests/fake_worker_engine.py", line 99, in _synthesize
    self._crash()
  File "tests/fake_worker_engine.py", line 83, in _crash
    os.abort()
=== END TRACED ===
```

Итог того же прогона:

```
ERROR tts.worker.supervisor: Воркер f5: процесс воркера завершился во время синтеза
  (аварийное завершение нативной библиотеки (SIGABRT)) [pid=64160, код=-6, сигнал=SIGABRT]
WARNING tts.worker.supervisor: Воркер f5 перезапущен после падения (перезапуск №1)
DRIVER: backend жив, pid=64158
```

Родительский процесс (бэкенд) остался жив, воркер перезапустился — то есть изоляция работает штатно.

### 1.2. Форма стека в `.ips` — это ровно вызов `os.abort()` из Python-метода

`python_issue.md:47-60` (отчёт от 2026-09-23 15:11:39, pid 53891):

```
Thread 0 Crashed::  Dispatch queue: com.apple.main-thread
0  libsystem_kernel.dylib   __pthread_kill
1  libsystem_pthread.dylib  pthread_kill
2  libsystem_c.dylib        abort
3  Python                   os_abort
4  Python                   cfunction_vectorcall_NOARGS     <- builtin без аргументов
5  Python                   _PyEval_EvalFrameDefault
6  Python                   _PyEval_Vector
7  Python                   method_vectorcall               <- вызов метода `obj.m()`
8  Python                   _PyVectorcall_Call
9  Python                   _PyEval_EvalFrameDefault
10 Python                   PyEval_EvalCode
11 Python                   builtin_exec                    <- exec кода модуля
12 Python                   cfunction_vectorcall_FASTCALL_KEYWORDS
13 Python                   _PyEval_EvalFrameDefault
14 Python                   _PyEval_Vector
15 Python                   pymain_run_module               <- `python -m …`
16 Python                   Py_RunMain
17 Python                   Py_BytesMain
18 dyld                     start
```

Кадры `1.1` и `1.2` совпадают один в один: `pymain_run_module` = `python -m backend.engines.worker_process`; `builtin_exec` = исполнение кода модуля; `method_vectorcall` + `_PyVectorcall_Call` = вызов `holder.engine.synthesize(...)`, затем `self._synthesize(...)`; `cfunction_vectorcall_NOARGS` + `os_abort` = `os.abort()`.

Ключевой признак: `os_abort` вызывается **через Python-байткод** (`cfunction_vectorcall_NOARGS`), то есть это `os.abort()` из Python, а не `abort()` из C/C++/Metal. Нативный OOM MPS выглядел бы как `SIGSEGV`/`SIGBUS` + `EXC_BAD_ACCESS`, а `Py_FatalError` — без кадров `os_abort`/`cfunction_vectorcall_NOARGS`.

### 1.3. `os.abort()` есть только в тестовой заглушке

`grep -rn "os\.abort|os\._exit|faulthandler\.|Py_FatalError|signal\.signal|set_start_method|resource_tracker|torch\.multiprocessing"` по всем `*.py` репозитория даёт 4 совпадения, и все — в тестах:

```
tests/test_worker_isolation.py:9:    Падение проверяется «по-настоящему»: `os.abort()` выполняется в ребёнке…
tests/fake_worker_engine.py:10:    * `CRASH`/`краш` — процесс падает нативно (`os.abort()`, то есть SIGABRT);
tests/fake_worker_engine.py:83:            os.abort()
tests/fake_worker_engine.py:88:        os.abort()
```

В `backend/` — ноль вызовов. Заглушка выбирает поведение по тексту реплики (`tests/fake_worker_engine.py:94-114`): `crash`/`краш` → `_crash()` → `os.abort()`; `segv` → `ctypes.string_at(0)`; `boom` → `RuntimeError`; `hang` → `sleep(120)`.

### 1.4. Это задокументировано как ожидаемый шум

`docs/testing.md:40-49`:

> Тесты изоляции и восстановления (`test_worker_isolation.py`, `test_worker_recovery.py`, `test_crash_report_acceptance.py`, `test_memory_and_shutdown.py`) роняют воркер **намеренно**: `tests/fake_worker_engine.py` по маркерам `CRASH`/`краш`/`SEGV` вызывает `os.abort()` или `ctypes.string_at(0)` в дочернем процессе. macOS на каждый такой SIGABRT пишет отчёт в `~/Library/Logs/DiagnosticReports/` (архив — в `Retired/`), поэтому десятки файлов `Python-*.ips` после прогонов — **ожидаемый шум, а не дефект**: в `backend/` вызовов `os.abort()` нет ни одного, и продуктовый воркер сам себя не роняет. Если до падения реплика успела пройти нормальный синтез, в отчёте будет и `libtorch`: уборка в `synthesize()` (`release_torch_memory(light=True)`) импортирует torch в процессе воркера.

### 1.5. Почему в дампах есть `libtorch`/`libomp`/`libshm`

`backend/engines/base.py:840-857` — хвост `synthesize()` в `finally` зовёт `release_torch_memory(light=True)`, а тот (`base.py:78-110`) делает `import torch` + `torch.mps.empty_cache()`. Поэтому воркер, который до падения успел выполнить **хотя бы один нормальный синтез**, привозит в дамп `libtorch_cpu.dylib`, `libtorch_python.dylib`, `libshm.dylib`, `libomp.dylib`, `_posixshmem` — что и видно в `python_issue.md:115-136`. Это подтверждается воспроизведением: в «тёплом» прогоне `torch in sys.modules = True`, в «холодном» — `False`.

### 1.6. Корреляция по времени и по PID: падения — из pytest-сессий, а не из продукта

* Ни один из 97 разобранных ранее `.ips` (pid из заголовков) не встречается в `logs/voice_syntez.log` вообще (`0/97`); 81 из 97 не имеют активности приложения в окне ±120 с.
* Оба отчёта в `python_issue.md` (15:11:39 и 15:26:21) попадают **внутрь окна pytest-сессии**: `.pytest_cache/` создан 14:37, `v/cache/lastfailed` обновлён **15:06** (содержимое — `{}`, падений нет), `v/cache/nodeids` (142 КБ) — **15:59**.
* Продуктовый журнал в это время молчит: последняя строка `logs/voice_syntez.log` — `2026-09-23 13:07:59,068 … GET /api/recording-projects` (5634-я строка), и файл больше не изменялся (mtime 13:07). При работающем бэкенде `/api/status` пишется каждые ~4 с (см. строки 5622-5633), то есть обслуживающего бэкенда в 15:11 и 15:26 не было.
* Разбивка `.ips` по часам (по mtime) совпадает с окнами тестовых прогонов, а не с сессиями работы с продуктом:
  `09-22 21ч — 6, 22ч — 5, 23ч — 8; 09-23 00ч — 10, 01ч — 6, 02ч — 13; 12ч — 1, 13ч — 1, 15ч — 2, 16ч — 1`
  (`Retired/`, 172 файла: `09-18 08–18ч`, `09-21 11–13ч`, `09-22 15–19ч` — те же признаки).

### 1.7. Что это НЕ объясняет

Наблюдение из ТЗ «процесс жил ~178 мс, это падение на инициализации, модели даже не успели загрузиться» **не подтверждается**. Из `python_issue.md`:

| Поле | Отчёт 1 (pid 53891) | Отчёт 2 (pid 55917) |
| --- | --- | --- |
| `procLaunch` | 2026-09-23 15:11:38.6635 | 2026-09-23 15:26:20.9518 |
| `captureTime` | 2026-09-23 15:11:39.3849 | 2026-09-23 15:26:21.6301 |
| Жизнь (wall) | **721 мс** | **678 мс** |
| `procStartAbsTime` | 433 906 285 004 | 455 081 130 009 |
| `procExitAbsTime` | 433 922 167 867 | 455 095 941 978 |
| Жизнь (mach, нс) | **15.9 мс** | **14.8 мс** |

Значения «~178 мс» получить из этих двух отчётов не удалось (см. §9). Кроме того, падение явно произошло **не на инициализации**: в дампе загружены `libtorch_cpu.dylib`, `libtorch_python.dylib`, `libshm.dylib` — а в этом коде torch попадает в процесс воркера только внутри `synthesize()` (`base.py:850`). Значит, воркер успел обслужить минимум один синтез и упал на следующем — это «тёплая» сигнатура, и она воспроизведена в §3.

---

## 2. Альтернативные гипотезы и почему они отвергнуты

Гипотезы — из `crash_audit.md` §1.5. Проверялись все шесть.

| № | Гипотеза | Как проверял | Результат |
| --- | --- | --- | --- |
| 1 | `multiprocessing` + `resource_tracker` / leaked semaphore | Читал исходники stdlib: `lib/python3.11/multiprocessing/resource_tracker.py`, `popen_fork.py`, `forkserver.py`; искал `abort`/`_exit` во всём каталоге `multiprocessing/*.py`; искал `resource_tracker` в логе | **Отвергнута.** Ни один путь stdlib `multiprocessing` не вызывает `abort()`: завершение — только `os._exit(code)` (`forkserver.py:281`, `popen_fork.py:73`); `abort` в `managers.py:1101-1105` — это имя XML-RPC-метода `AsyncResult.abort`, а не сигнал; `resource_tracker.main()` (`resource_tracker.py:209-215`) игнорирует `SIGINT`/`SIGTERM` через обработчики, а не аварийно завершается. В журнале предупреждение ровно одно и при **штатном** shutdown: `voice_syntez.log:1032` — `resource_tracker: There appear to be 1 leaked semaphore objects…` (2026-09-17 11:35:54), это `UserWarning`, процесс завершился нормально |
| 2 | `faulthandler` + повторный SIGABRT | Искал `faulthandler` в репозитории (0 вхождений в `*.py`), проверял окружение (`PYTHONFAULTHANDLER` отсутствует), смотрел стек | **Отвергнута.** `faulthandler` в проекте не включён; в стеке нет кадров обработчика фатальных сигналов, стек — чистый путь `os.abort()` |
| 3 | `Py_FatalError` на инициализации | Сопоставлял форму стека | **Отвергнута.** `Py_FatalError` — C-путь (`_Py_FatalErrorFormat`) и не проходит через `cfunction_vectorcall_NOARGS`/`os_abort`; кроме того, процесс дожил до импорта torch (`libtorch_cpu.dylib` в дампе), то есть инициализация прошла |
| 4 | OpenMP-конфликт / `__kmp_abort_process` | Искал дубли `libomp` в дампах и на диске; проверял env на `KMP_DUPLICATE_LIB_OK`; пробовал `otool -L`, `brew list --versions libomp` | **Отвергнута как причина этих падений.** В обоих отчётах `libomp.dylib` ровно один (uuid `e56febf1-776c-35bb-b9a1-8c978a01425c`, путь `/Users/USER/*/libomp.dylib` — из torch wheel), кадров `libomp`/`__kmp_abort_process` в стеке нет, `KMP_DUPLICATE_LIB_OK` в окружении не выставлен. `otool`/`brew` недоступны без принятия Xcode-лицензии (см. §9). **Отдельная находка (не причина этих падений):** на диске есть **второй** `libomp.dylib` — `venv/lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib`, помимо `…/torch/lib/libomp.dylib`; риск, если оба когда-нибудь окажутся в одном процессе, вынесен в §9 |
| 5 | `torch.multiprocessing` + `fork` + MPS | Искал `torch.multiprocessing`, `set_start_method`, `multiprocessing.get_context`, `os.fork(` по репозиторию; читал `supervisor.py` | **Отвергнута.** 0 вхождений. Изоляция движков сделана **не** на `fork`/`multiprocessing.Process`, а на `subprocess.Popen` (`supervisor.py:307-317`) с явным `env`, `pass_fds` и `cwd` — то есть сознательно в обход fork+MPS |
| 6 | Явный `os.abort()` в коде или зависимости | `grep` по репозиторию и по stdlib; воспроизведение с перехватом стека | **Подтверждена** — см. §1 |

Дополнительно проверено и **не подтвердилось** как причина:

* `pip check` — единственное расхождение: `deepfilternet 0.5.6` требует `numpy<2.0` (стоит 2.4.6) и `packaging<24.0` (стоит 26.3). Это несовместимость объявленных требований, а не `abort` на старте: `deepfilternet` живёт за отдельным процессом (`backend/denoise.py:30,58` → `python -m backend.denoise_worker`), и в дампах нет ни `onnxruntime`, ни `sklearn`.
* `numpy 2.4.6` + `torch 2.11.0` — совместимы (torch 2.11 собран под numpy 2.x); `torch`/`torchaudio` — одной версии 2.11.0; `tokenizers 0.23.2` + `transformers 5.17.0` — согласованы.
* Память MPS — не причина: `driver_allocated_memory()` полностью возвращается к базовому уровню после `gc.collect()+empty_cache()` (см. §5).

---

## 3. Минимальный воспроизводимый кейс

Скрипт воспроизводит **тот самый** SIGABRT: поднимает реальный `WorkerSupervisor` с тестовой фабрикой движка и просит воркер упасть. 28 строк, падает детерминированно.

```python
"""Минимальное воспроизведение SIGABRT (EXC_CRASH) в процессе-воркере.

Запуск (из корня репозитория):
    venv/bin/python /tmp/repro_crash.py
"""
import os
import sys

BASE = "/Users/vladislavkovalenko/Projects/VOICE_SYNTEZ"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tests"))
os.environ["TTS_WORKER_ISOLATION"] = "1"
os.environ["TTS_WORKER_ENGINE_FACTORY"] = "fake_worker_engine:create"

from backend import config
from backend.engines import supervisor as sup_mod
from backend.engines.base import ENGINE_F5
from backend.engines.worker_engine import WorkerEngine

config.WORKER_REQUEST_TIMEOUT_SEC = 30.0
sup_mod._supervisor = sup_mod.WorkerSupervisor()
engine = WorkerEngine(ENGINE_F5)
try:
    engine.synthesize("CRASH", "/tmp/reference.wav", "референс")
except Exception as exc:
    print("ABORT CONFIRMED:", getattr(exc, "details", exc))
finally:
    sup_mod._supervisor.stop_all(grace=2.0)
```

Команда запуска:

```bash
cd /Users/vladislavkovalenko/Projects/VOICE_SYNTEZ
venv/bin/python /tmp/repro_crash.py
```

Фактический вывод (хвост):

```
Воркер f5: процесс воркера завершился во время синтеза
  (аварийное завершение нативной библиотеки (SIGABRT)) [pid=64727, код=-6, сигнал=SIGABRT]
Воркер f5 перезапущен после падения (перезапуск №1)
ABORT CONFIRMED: {'exit_code': -6, 'signal': 6, 'signal_name': 'SIGABRT',
                  'reason': 'аварийное завершение нативной библиотеки (SIGABRT)', 'pid': 64727}
```

Чтобы получить именно **тёплую** сигнатуру (с `libtorch` в дампе, как в `python_issue.md`), перед «CRASH» надо выполнить один нормальный синтез — это делают `tests/test_worker_isolation.py::test_tts_worker_sigabrt_does_not_kill_backend` (строка 134) и `tests/test_worker_recovery.py::test_render_resumes_after_worker_crash` / `test_crash_on_first_replica_restarts_render` (строки 126, 159); отдельный драйвер `warm`-режима использовался в §1.1.

Штатный путь до того же падения без самодельного скрипта:

```bash
# только тест SIGABRT, в обычном Terminal.app (не в песочнице Trae)
venv/bin/python -m pytest tests/test_worker_isolation.py -k sigabrt -x
```

**Варианты A/B/C/D из ТЗ (`crash_audit.md` §4.10) не воспроизводят падение:** `import torch`/`import numpy`/`import multiprocessing` сами по себе не падают (`/tmp/mps_probe.py`: `import torch` → `mps=True`, код 0), инициализация multiprocessing+MPS не падает, запуск сервера без синтеза не падает. Падает только ветка «воркеру пришёл текст `CRASH`» — то есть гипотеза «импорт-конфликт» окончательно исключена.

---

## 4. План устранения

Правки в код в рамках аудита **не вносились** (`crash_audit.md` §11). Ниже — предлагаемые изменения с точными местами.

### 4.1. Немедленные меры

1. **Сменить трактовку этих `.ips`.** Это не продуктовые падения. Практическое следствие: не заводить инцидент на `Python-*.ips` без проверки пары `(coalitionName, parentPid)` и времени против `logs/voice_syntez.log`. Готовый признак «это тест»: родитель — процесс `pytest` (или его потомок), а в `logs/voice_syntez.log` в окне ±2 мин нет ни одной строки.
2. **Запускать тесты изоляции в системном Terminal.app, не под Trae.** `docs/testing.md:51-55`: под TraeCode песочница перехватывает дочерние процессы — `.ips` не появляются вовсе, в логе видно `TRAE Sandbox Error: process crashed`, а тесты с жёстким таймаутом (`test_worker_timeout_kills_process_and_restarts`, таймаут 1 с) могут ложно падать из-за замедления спавна. Это подтверждено напрямую: под песочницей `venv/bin/python` исполняется из копии `…/.sboxcache_501/audit_trae-sandbox_*/.sandbox_exec/…/Python.app/Contents/MacOS/Python` (см. `lsof`), а в конце прогона печатается `TRAE Sandbox Error: process crashed; pid=…`.
3. **Развести «шум» и «сигнал» в `docs/testing.md`.** Дополнить раздел (после строки 55) инструкцией из п.1: как по `python_issue.md`-подобному дампу за минуту понять, что это тест, а не продукт. Ссылка на `.pytest_cache/` как на дешёвый маркер окна прогона.
4. **Помечать намеренные падения в stderr воркера.** В `tests/fake_worker_engine.py:78-88` (`_crash`) перед `os.abort()` писать одну строку вида `FAKE-WORKER-CRASH pid=… marker=…` в `sys.stderr` — она попадёт в app-лог через `_start_drains` (`supervisor.py:343-356`) и снимет любую двусмысленность при разборе будущих дампов.
5. **Не трогать `os.abort()` в `backend/`** — там его нет, и добавлять нечего.

### 4.2. Архитектурные правки

1. **Включить `faulthandler` в воркере.** Сейчас нативный `SIGABRT`/`SIGSEGV` в воркере не оставляет Python-стека — и именно поэтому исходный дамп приходится расшифровывать по форме кадров. Добавить в `backend/engines/worker_process.py` в начале `main()` (до импорта движка, рядом со строкой 38) `faulthandler.enable(file=open(<log>/worker_faulthandler.log, "a"))` и логировать `os.getpid()`, `sys.argv`, `sys.executable`, `os.getcwd()`. Это ровно тот класс падений, который иначе невидим.
2. **Поставить `sys.excepthook`/`threading.excepthook` в воркере.** Сейчас `worker_process.py:230-234` ловит `BaseException` вокруг `handle_request`, но падения вне этого блока (исключения в потоках) не логируются; `threading.excepthook` закроет вторую половину.
3. **Убрать латентный дубль OpenMP.** `venv/lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib` и `…/torch/lib/libomp.dylib` — две разные копии рантайма OpenMP в одном venv. Сегодня в дампах загружена только torch-копия, поэтому `__kmp_abort_process` не срабатывает; но при появлении в процессе воркера `sklearn` (или `onnxruntime`) риск двойной инициализации libomp становится реальным. Правильная мера — **не** `KMP_DUPLICATE_LIB_OK=TRUE` (ТЗ §8 запрещает как костыль), а архитектурная развязка: не импортировать `sklearn`/`onnxruntime` в тот же процесс, где живёт torch (для `deepfilternet` это уже сделано — `backend/denoise.py:30,58` выносит его в `backend.denoise_worker`; ту же политику надо зафиксировать явно и проверять тестом-«стражником» на пересечение множеств загруженных образов).
4. **Проверить, что батчи F5 остаются сериализованными.** `backend/tts_engine.py:24-57` подменяет `ThreadPoolExecutor` на `_InlineExecutor` для `f5_tts.infer.utils_infer`, потому что «MPS-бэкенд torch не потокобезопасен: параллельные ядра из разных потоков ломают общий кэш шейдеров Metal… и процесс падает с SIGSEGV». Это уже ровно та мера («ThreadPool executor с max_workers=1 / сериализация MPS-доступа»), которую рекомендует ТЗ §6, — её надо не переоткрывать, а покрыть тестом, чтобы будущий рефакторинг её не потерял.
5. **Решить расхождение `pip check`.** `deepfilternet 0.5.6` требует `numpy<2.0`, стоит 2.4.6; и `packaging<24.0`, стоит 26.3. Сегодня это не стреляет, потому что `deepfilternet` изолирован процессом, но объявленные требования врут, и `pip` при следующей установке может «разрешить» конфликт силой. Варианты: обновить `deepfilternet` до версии с numpy 2.x, либо зафиксировать пины и оставить изоляцию процессом (предпочтительно — изоляция уже есть).
6. **Ограничить число одновременно загруженных движков.** ТЗ §5.3 и §12 предлагают «радикально урезать до одного активного» при 18 ГБ. В проекте уже есть вкладка выгрузки и `engine_lifecycle` + `free_warm_engines` (`backend/audio_pipeline.py:1058`); проверить, что политика «F5 + обе XTTS одновременно» действительно не включается по умолчанию.

### 4.3. Долгосрочные улучшения

1. **CI-шаг «разбор crash-отчётов».** После прогона набора тестов считать `Python-*.ips`, появившиеся в окне прогона, и явно помечать их как принадлежащие тестам (архивировать/удалять), чтобы `DiagnosticReports` перестал быть свалкой: сейчас 55 активных + 172 в `Retired/`.
2. **Телеметрия MPS на постоянку.** Лог-строка `tts.mps.synthesis engine=… driver_before_mb=… driver_after_mb=… delta_mb=…` есть в коде (`backend/engines/base.py:853-857`), но в `logs/voice_syntez.log` **не встречается ни разу** — то есть текущая сборка/прогоны этой метрики не оставляют. Проверить, почему (уровень логгера, версия кода в прогонах, или путь синтеза не доходит до `finally`) — без неё нельзя отследить утечку аллокатора MPS в бою.
3. **Регламент ротации отчётов.** `Retired/` уже 172 файла; описать в `docs/operations.md` ежемесячную очистку `DiagnosticReports` (или `obsidian-memory`-стиль консолидации), чтобы окно наблюдения оставалось обозримым.
4. **Тест-«стражник» на `os.abort` в `backend/`.** Автотест, который падает, если в `backend/` появится `os.abort`, `os._exit`, `faulthandler.disable` или `set_start_method("fork")`. Дёшево, и защищает инвариант, на котором держится вся эта диагностика.

---

## 5. Профиль памяти до/после

Измерения — `/tmp/mps_probe.py` (`resource.getrusage(RUSAGE_SELF).ru_maxrss` + MPS API), команда `venv/bin/python /tmp/mps_probe.py`. Системный фон в момент замера: `memory_pressure` → free 77 %; `vm.swapusage` → `total = 6144.00M  used = 4556.31M  free = 1587.69M`; `hw.memsize` = 19 327 352 832; `kern.num_taskthreads` = 4096; `OMP_NUM_THREADS` не задан, `torch.get_num_threads()` = 6.

| Операция | RSS, МБ | `driver_allocated_memory()`, МБ | `current_allocated_memory()`, МБ | swap used, МБ | memory_pressure |
| --- | --- | --- | --- | --- | --- |
| Старт процесса | 21.2 | — (torch не импортирован) | — | 4556.31 | free 77 % |
| `import numpy` (2.4.6) | 32.7 | — | — | 4556.31 | free 77 % |
| `import torch` (2.11.0, `mps=True`) | 436.0 | 0.5 | 0.0 | 4556.31 | free 77 % |
| Аллокация 1024×1024 на MPS | 449.5 | 34.0 | 4.2 | — | — |
| `del` + `gc.collect()` + `torch.mps.empty_cache()` | 449.7 | **0.5** | **0.0** | — | — |
| Загрузка F5-TTS | не проверено (см. §9) | не проверено | не проверено | — | — |
| Загрузка XTTS base | не проверено | не проверено | не проверено | — | — |
| Загрузка XTTS banana | не проверено | не проверено | не проверено | — | — |
| Синтез 1 реплики | не проверено | не проверено | не проверено | — | — |
| Синтез 10 реплик | не проверено | не проверено | не проверено | — | — |
| Рендер диалога 50 реплик | не проверено | не проверено | не проверено | — | — |
| Выгрузка модели | не проверено | не проверено | не проверено | — | — |
| После `gc.collect()+empty_cache()` | 449.7 | 0.5 | 0.0 | — | — |

Дополнительно измерено: `torch.mps.recommended_max_memory()` = 14302.2 МБ (≈ 74 % от 18 ГБ — это и есть потолок unified memory, который видит MPS).

**Выводы по памяти:**

* `driver_allocated_memory()` полностью возвращается к базовому уровню `0.5 МБ` после `gc.collect()+empty_cache()` — **утечки аллокатора MPS в микропробе нет**. Гипотеза «MPS-OOM как причина SIGABRT» не подтверждается ещё и по форме падения: OOM дал бы `RuntimeError: MPS backend out of memory` либо `SIGSEGV`, а не `os_abort` из Python-байткода (§1.2).
* `ru_maxrss` после `import torch` (436 МБ) не падает после уборки — это нормально: RSS не возвращает освобождённые страницы ОС, уходит только MPS-аллокатор.
* Строки для реальных моделей не заполнены сознательно: их измерение требует загрузки F5/XTTS (десятки секунд и несколько ГБ) и, по `docs/testing.md:51`, должно проводиться **вне** песочницы Trae. См. §9.

---

## 6. Модель потоков/процессов (целевая схема)

### 6.1. Как есть (карта, собранная по коду)

```
run.sh:137  exec uvicorn backend.main:app --host 127.0.0.1 --port $PORT   (1 воркер, без --workers)
│
└── Главный процесс Python (бэкенд, pid 64158 в моём прогоне)
    ├── Event loop asyncio
    │   ├── HTTP API (backend/main.py)
    │   ├── backend/job_queue.py:375  async def start() → asyncio.Task (очередь задач)
    │   └── asyncio.to_thread(...) — десятки вызовов (main.py, audio_pipeline.py, job_queue.py):
    │        синхронные вещи (загрузка/выгрузка движка, экспорт, анализ) уходят
    │        в дефолтный executor петли
    ├── threading.Thread — загрузка моделей: backend/model_manager.py:787
    ├── subprocess.run — внешние утилиты: backend/denoise.py:60, backend/audio_analysis.py:105,
    │                    backend/transcribe.py:72,107
    └── subprocess.Popen — процесс(ы)-воркеры синтеза (по одному на движок):
         backend/engines/supervisor.py:307-317
           cmd = [sys.executable, "-m", config.WORKER_MODULE, engine_id, req_fd, resp_fd]
           config.WORKER_MODULE = "backend.engines.worker_process"  (config.py:521)
           env = _child_env()  (supervisor.py:622-639: PYTHONPATH = BASE_DIR + sys.path родителя)
           pass_fds=(request_recv, response_send), stdin=DEVNULL, stdout=PIPE, stderr=PIPE
         ├── Pipe(duplex=False) ×2 — multiprocessing.connection (каналы запрос/ответ)
         ├── threading.Thread ×2 — drain stdout/stderr воркера (supervisor.py:343-356)
         └── atexit.register(_stop_at_exit, _supervisor) (supervisor.py:691)

    Процесс-воркер:  python -m backend.engines.worker_process <engine> <fd> <fd>
    └── backend/engines/worker_process.py:38  from multiprocessing.connection import Connection
        ├── цикл main() (стр. 231: handle_request)
        │    └── handle_request (стр. 166) → holder.engine.synthesize(...)
        │         → backend/engines/base.py:840 → _synthesize()
        │         → finally (base.py:850) release_torch_memory(light=True) → import torch + mps.empty_cache()
        └── отдельный процесс — от deepfilternet: backend/denoise.py:30,58
            python -m backend.denoise_worker <src> <dst>
```

Что здесь важно для аудита:

* Изоляция движков сделана **на `subprocess.Popen`, а не на `fork`/`multiprocessing.Process`** — это осознанное решение (комментарий в `supervisor.py:293-295`: «Каналы создаёт родитель… поэтому воркер не наследует ничего лишнего (в том числе не наследует инициализированный MPS — он поднимает его сам)»). Для MPS+macOS это правильнее, чем `fork`: дочерний процесс не наследует состояние Metal.
* `multiprocessing` в воркере появляется только из-за `Connection` (`worker_process.py:38`) и `Pipe` — это объясняет наличие `_multiprocessing`/`libshm`/`_posixshmem` в дампах, которое ТЗ (§«наблюдения по логу») трактовало как «приложение на старте порождает дочерние процессы». Порождение процессов — не признак дефекта, а архитектура изоляции.
* `_InlineExecutor` (`backend/tts_engine.py:24-57`) уже сериализует батчи F5 внутри одной реплики — то есть MPS-доступ к модели уже сериализован, как и рекомендует ТЗ §6 («ThreadPool с max_workers=1»).

### 6.2. Целевая схема

Схема из ТЗ §6 в этом проекте уже реализована, и менять её ради падений, которых нет, не нужно. Целевое состояние — **зафиксировать текущее и закрыть три дырки**:

```
Главный процесс (uvicorn, 1 воркер)                       [есть]
 ├── Event loop (asyncio + job_queue)                     [есть]
 ├── Синхронные задачи через asyncio.to_thread            [есть]
 │    └── доступ к MPS сериализован (_InlineExecutor)      [есть, tts_engine.py:24-57]
 ├── subprocess.run для ffmpeg/анализа                    [есть]
 └── subprocess-воркеры синтеза (по одному на движок)      [есть; сознательно вместо fork]
      ├── faulthandler.enable + логирование pid/argv      [ДОБАВИТЬ — §4.2 п.1]
      ├── sys.excepthook / threading.excepthook            [ДОБАВИТЬ — §4.2 п.2]
      └── запрет импорта sklearn/onnxruntime рядом с torch [ЗАФИКСИРОВАТЬ — §4.2 п.3]
```

Никаких «никаких дочерних Python-процессов, кроме spawn» вводить не надо: изоляция процессов здесь — не источник падения, а средство выживания при падении модели (что и подтверждает §1.1: воркер умер, бэкенд остался жив).

---

## 7. Логи и телеметрия, которые нужно добавить

| Что | Куда | Зачем |
| --- | --- | --- |
| `faulthandler.enable(file=…)` | `backend/engines/worker_process.py`, начало `main()` (до импорта движка) | Нативный SIGABRT/SIGSEGV в воркере сейчас не оставляет Python-стека; именно из-за этого дамп приходится расшифровывать по форме кадров |
| `logging` при старте воркера: `os.getpid()`, `sys.argv`, `sys.executable`, `os.getcwd()`, `sys.path` | там же | Сопоставление строки журнала с `pid`/`parentPid` из `.ips` — сейчас единственная связь теряется |
| `sys.excepthook`, `threading.excepthook` | `worker_process.py` и `backend/main.py` | Падения вне `try` вокруг `handle_request` (`worker_process.py:230-234`) и исключения в потоках сейчас невидимы |
| `atexit`-хук с логом причины выхода | воркер | Отличить нормальный shutdown (`worker_process.py:240-242`) от падения |
| Снимок MPS-памяти на каждом синтезе | уже есть: `backend/engines/base.py:853-857` | Метрика объявлена, но в `logs/voice_syntez.log` **отсутствует** — проверить, почему не пишется (§4.3 п.2) |
| Строка-маркер перед `os.abort()` в заглушке | `tests/fake_worker_engine.py:78-88` | Мгновенно отличать тестовый SIGABRT от продуктового |
| Контроль `env` дочернего процесса | `supervisor.py:622-639` уже формирует окружение; добавить лог итогового `PYTHONPATH`/`sys.executable` на DEBUG | ТЗ §4.8 просит `ps eww <pid>` до падения — при жизни процесса 0.7 с это невыполнимо, а лог закрывает вопрос |

Отдельно: Trae инжектит в окружение `PYTHONSTARTUP`, `PYTHONUTF8`, `PYTHON_BASIC_REPL`, `TOOLHOST_SANDBOX_DISABLED=false`, `TRAE_SANDBOX_*` (см. Приложение A). Ни одна из этих переменных не влияет на падение (оно воспроизводится и вне IDE), но `PYTHONSTARTUP` указывает на `…/ms-python.python/pythonrc.py` — при отладке «необъяснимого» поведения в REPL это первое, что стоит проверить.

---

## 8. Что проверить при следующем запуске (smoke-протокол)

После правок из §4.2:

1. **Изоляция не сломана.**
   ```bash
   cd /Users/vladislavkovalenko/Projects/VOICE_SYNTEZ
   venv/bin/python -m pytest tests/test_worker_isolation.py tests/test_worker_recovery.py \
       tests/test_crash_report_acceptance.py tests/test_memory_and_shutdown.py -q
   ```
   Ожидание: все зелёные; в `logs/voice_syntez.log` видно `Воркер f5 запущен/остановлен` и `перезапуск №1` без падения бэкенда.
2. **Faulthandler реально пишет.** Спровоцировать `CRASH` (скрипт из §3) и убедиться, что в `worker_faulthandler.log` появился Python-стек, а в app-логе — строка с `pid` воркера.
3. **Тёплая и холодная сигнатуры воспроизводятся.** «Холодный» прогон (§3 как есть) не должен содержать `libtorch` в дампе; «тёплый» (сначала нормальный синтез, потом `CRASH`) — должен. Это контроль того, что объяснение из §1.5 не сломалось.
4. **Живое приложение не затронуто.**
   ```bash
   ./run.sh           # или VOICE_SYNTEZ.command
   ```
   Проверить: `MPS available: True`, загрузка F5, синтез одной реплики, выгрузка через вкладку 06, в логе — строка `tts.mps.synthesis … delta_mb=…`, отсутствие новых `Python-*.ips` за окно работы.
5. **Контроль «дубля libomp».**
   ```bash
   venv/bin/python - <<'PY'
   import torch, subprocess, os
   # в процессе с torch не должно импортироваться sklearn/onnxruntime
   print("torch loaded:", "torch" in __import__("sys").modules)
   PY
   grep -c "libomp" ~/Library/Logs/DiagnosticReports/Python-*.ips 2>/dev/null | tail -5
   ```
   Ожидание: в дампе — не более одного `libomp.dylib`.
6. **Метрики памяти вне песочницы.** Прогнать `/tmp/mps_probe.py` в системном Terminal.app (не в Trae) и сверить, что `driver` возвращается к базовому уровню (§5).
7. **Таймаутный тест не флакует.** `test_worker_timeout_kills_process_and_restarts` (таймаут 1 с) — прогнать 10 раз подряд в Terminal.app; под Trae ожидаемы ложные падения (`docs/testing.md:51-55`).

---

## 9. Открытые вопросы и что не удалось проверить

1. **`otool -L` по ключевым библиотекам — не проверено.** `otool` и `brew` отказывают: `You have not agreed to the Xcode license agreements`. Требуется `sudo xcodebuild -license accept` (действие пользователя, в аудите не выполнялось). Компенсировано косвенно: по `usedImages` в дампах видно, какие образы реально загружены и откуда.
2. **`brew list --versions libomp` — не проверено** по той же причине (Xcode-лицензия). Подтверждено иначе, поиском по файловой системе: `libomp` от Homebrew нет; есть только две копии внутри venv — `torch/lib/libomp.dylib` и `sklearn/.dylibs/libomp.dylib` (вторая в дампах не загружалась).
3. **Реальный расход памяти при загрузке F5/XTTS и при синтезе (строки таблицы §5) — не проверено.** Требует загрузки моделей (несколько ГБ на 18 ГБ машины) и, по `docs/testing.md:51`, замера вне песочницы Trae.
4. **Значение «~178 мс» из ТЗ — не воспроизведено.** По `procLaunch`/`captureTime` обоих отчётов получается 721 мс и 678 мс; по `procStartAbsTime`/`procExitAbsTime` — 15.9 мс и 14.8 мс. Откуда взялось 178 мс, установить не удалось (возможно, из другого отчёта или из иного поля). Соответственно, вывод ТЗ «падение на инициализации» не подтверждается: в дампах есть `libtorch_cpu.dylib`, а torch попадает в процесс воркера только внутри `synthesize()`.
5. **`argv` упавшего процесса в `.ips` отсутствует** — по формату отчёта macOS его там нет. Поэтому «это был pytest, а не продукт» доказано косвенно: формой стека (§1.2), единственным `os.abort()` в `tests/` (§1.3), документом `docs/testing.md:40-49` (§1.4), совпадением времени с окном pytest-сессии и молчанием продуктового журнала (§1.6). Прямой привязки «этот pid → команда pytest» нет.
6. **Почему `tts.mps.synthesis` отсутствует в `logs/voice_syntez.log`, хотя строка есть в `base.py:853-857`** — не проверено. Возможные причины: иная версия кода в прогонах, уровень логгера, либо синтез не доходит до `finally`. Для честной таблицы §5 в бою это нужно выяснить.
7. **Поведение под песочницей Trae.** Подтверждено, что дочерний Python исполняется из копии в `/private/tmp/.sboxcache_501/audit_trae-sandbox_*/.sandbox_exec/…` и что в конце печатается `TRAE Sandbox Error: process crashed`. Дальше (перехватывает ли Trae `SIGABRT` полностью, влияет ли перехват на коды возврата вне тестов) — не проверено.
8. **`DYLD_INSERT_LIBRARIES` / `DYLD_LIBRARY_PATH` в окружении Trae** — в снятом окружении не обнаружены (см. Приложение A). ТЗ §4.8 просит проверять их наличие; при повторном аудите стоит снимать `env` из самого IDE до запуска приложения.

---

## 10. Приложения

### A. Дампы окружения

Ключевые переменные окружения (снято в терминале Trae; полный `env | sort` — по требованию):

```
BUNDLED_DEBUGPY_PATH=/Users/…/.trae/extensions/ms-python.debugpy-2026.6.0-darwin-arm64/bundled/libs/debugpy
PYTHON_BASIC_REPL=1
PYTHONIOENCODING=utf-8
PYTHONSTARTUP=/Users/…/Library/Application Support/Trae/User/workspaceStorage/…/ms-python.python/pythonrc.py
PYTHONUTF8=1
TOOLHOST_SANDBOX_DISABLED=false
TRAE_SANDBOX_CLI_PATH=/Applications/Trae.app/Contents/Resources/app/modules/sandbox/trae-sandbox
TRAE_SANDBOX_DUMP_DIR=/Users/…/Library/Application Support/Trae/Crashpad/sandbox-pending
TRAE_SANDBOX_LOG_DIR=/Users/…/Library/Application Support/Trae/logs/20260923T091548/Modular
TRAE_SANDBOX_NEW_PERMISSION=1
TRAE_SANDBOX_SBOX_ID=TRAE-SBX-f606de0c1042fcd5a9f8899e3bd56883
TRAE_SANDBOX_SOURCE_FLAG_PATH=/var/folders/…/T/sandbox-source-flag-51f3614a-…-90f4371c1bc
TRAE_SANDBOX_STORAGE_PATH=/Users/…/Library/Application Support/Trae/ModularData/ai-agent/sandbox
TRAE_SANDBOX_TRACE_FILE=/var/folders/…/T/trae_sandbox_trace_63accfdd-….jsonl
```

Отсутствуют (проверено целевым grep): `PYTHONFAULTHANDLER`, `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`, `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`, `KMP_*`.

Критичные версии (`venv/bin/pip list`):

```
torch 2.11.0   torchaudio 2.11.0   numpy 2.4.6
transformers 5.17.0   tokenizers 0.23.2   librosa 0.11.0
onnxruntime 1.30.0   scikit-learn 1.9.1   numba 0.67.0   llvmlite 0.49.0
DeepFilterNet 0.5.6   packaging 26.3   soundfile 0.14.0
```

`venv/bin/pip check`:

```
deepfilternet 0.5.6 has requirement numpy<2.0,>=1.22, but you have numpy 2.4.6.
deepfilternet 0.5.6 has requirement packaging<24.0,>=23.0, but you have packaging 26.3.
```

Разрешение пути интерпретатора (`readlink -f venv/bin/python` и `lsof` живого процесса):

```
venv/bin/python → python3.11
readlink -f → /opt/homebrew/Cellar/python@3.11/3.11.15_1/Frameworks/Python.framework/Versions/3.11/bin/python3.11
реально исполняется (ps/lsof): /opt/homebrew/Cellar/python@3.11/3.11.15_1/Frameworks/
    Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python
```

Этот путь совпадает с `procPath` в обоих отчётах (`python_issue.md:185`, с редуцированием приватных компонентов до `*`), то есть падавший процесс — **тот же интерпретатор venv**, что и у продукта, а не «какой-то другой Python».

`otool -L` по ключевым `.so`/`.dylib` — **не снято**: без принятия Xcode-лицензии команда не выполняется (см. §9).

### B. Полный список abort-триггеров, найденных grep'ом

```
$ grep -rn "os\.abort|os\._exit|faulthandler\.|Py_FatalError|signal\.signal|set_start_method|resource_tracker|torch\.multiprocessing" \
    --include="*.py" .
tests/test_worker_isolation.py:9:    Падение проверяется «по-настоящему»: `os.abort()` выполняется в ребёнке, а тест
tests/fake_worker_engine.py:10:    * `CRASH`/`краш` — процесс падает нативно (`os.abort()`, то есть SIGABRT);
tests/fake_worker_engine.py:83:            os.abort()
tests/fake_worker_engine.py:88:        os.abort()
```

Итого: `backend/` — 0, `tools/` — 0, `frontend/` — 0, `tests/` — 4 (все в заглушке и её докстринге). `os._exit`, `faulthandler`, `Py_FatalError`, `signal.signal`, `set_start_method`, `resource_tracker`, `torch.multiprocessing` — 0 вхождений во всём репозитории.

Проверка stdlib (для гипотез 1–3):

```
$ grep -rn "abort\b\|_exit(" lib/python3.11/multiprocessing/*.py
…/multiprocessing/forkserver.py:281:                                os._exit(code)
…/multiprocessing/managers.py:1101:    _exposed_ = ('__getattribute__', 'wait', 'abort', 'reset')   # имя XML-RPC-метода
…/multiprocessing/popen_fork.py:73:                os._exit(code)

$ grep -rln "os\.abort()" lib/python3.11
lib/python3.11/test/test_subprocess.py            # только тесты самого CPython
```

Единственное предупреждение `resource_tracker` в продуктовом журнале (штатный shutdown, не падение):

```
logs/voice_syntez.log:1032: …/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker:
    There appear to be 1 leaked semaphore objects to clean up at shutdown
logs/voice_syntez.log:1033:   warnings.warn('resource_tracker: There appear to be %d '
```

### C. Ссылки на upstream-issues

Прямую ссылку на issue, соответствующую этой первопричине, приводить не нужно и нельзя: первопричина — собственный вызов `os.abort()` в тестовой заглушке репозитория, а не дефект PyTorch/CPython/libomp.

Что можно сослаться с доказательством (найдено в локальных исходниках, а не по памяти):

* **CPython bpo-33613** — упомянут в комментарии локального исходника `lib/python3.11/multiprocessing/resource_tracker.py:139`: «bpo-33613: Register a signal mask that will block the signals…», в контексте маскирования сигналов вокруг запуска дочернего процесса. Релевантно гипотезе 1 (resource_tracker) как источник деталей о его поведении.
* **`MetalShaderLibrary::exec_unary_kernel`** — упоминается в докстринге `backend/tts_engine.py:45-53` как причина нативного SIGSEGV при параллельных батчах F5 на MPS. Это внутренний символ Metal, и он объясняет, почему MPS-доступ в проекте сериализован; конкретный upstream-issue в рамках аудита **не проверялся** (нет web-доступа, ссылку не выдумываю).

Остальные ссылки (PyTorch MPS thread-safety, дубль libomp у onnxruntime+torch, numpy 2.x + torch) — **не проверены**; при необходимости их стоит собрать отдельно с web-доступом.
