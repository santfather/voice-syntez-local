# VOICE_SYNTEZ --- инструкция для ИИ-агента по развитию приложения

## 0. Назначение документа

Этот документ --- **пошаговая инструкция для ИИ-агента**, который должен
улучшать существующий проект `VOICE_SYNTEZ`, не ломая уже работающий
TTS-пайплайн.

Исходное приложение --- локальный TTS-дашборд для озвучки русских
диалогов и сплошного текста. В текущей архитектуре уже есть:

-   F5-TTS Russian, XTTS v2 и XTTS Banana;
-   выбор движка отдельно для каждого голоса;
-   клонирование голоса по reference-аудио;
-   Whisper для расшифровки reference и строгого QA;
-   RUAccent для F5;
-   preprocessing русского текста;
-   генерация диалогов и сплошного текста;
-   очередь с одним inference worker;
-   перегенерация отдельных реплик;
-   история вариантов реплики;
-   LUFS-нормализация, limiter, trimming тишины;
-   контроль памяти;
-   REST API;
-   unit/integration tests на `StubEngine`;
-   отдельный нагрузочный тест на реальных моделях.

Цель дальнейшей работы --- превратить приложение из продвинутого
TTS-дашборда в **локальную студию озвучки**, сохранив offline-first
подход, single-user сценарий и устойчивость на Apple Silicon.

------------------------------------------------------------------------

# 1. Общие правила для агента

## 1.1. Не переписывать работающую архитектуру без необходимости

Перед каждой фазой агент обязан:

1.  изучить текущую реализацию затрагиваемых модулей;
2.  изучить существующие тесты;
3.  определить backward compatibility API и форматов данных;
4.  сделать минимально необходимое изменение;
5.  добавить/обновить тесты;
6.  запустить весь существующий test suite;
7.  только после успешных тестов переходить к следующей фазе.

Не выполнять несколько крупных фаз одновременно.

------------------------------------------------------------------------

## 1.2. Главный критерий завершения фазы

Фаза считается завершённой только если одновременно выполнены:

-   реализована заявленная функциональность;
-   старое поведение не сломано;
-   написаны тесты новой функциональности;
-   все новые тесты проходят;
-   все старые тесты проходят;
-   ошибки API имеют понятный текст;
-   UI корректно обрабатывает loading/error/empty states;
-   README обновлён, если изменилось пользовательское поведение, API,
    структура проекта или переменные окружения.

Базовая команда regression:

``` bash
source venv/bin/activate
python -m pytest
```

Если фаза затрагивает реальный inference, дополнительно выполнить
соответствующие фазы:

``` bash
./venv/bin/python tools/loadtest.py
```

или их релевантное подмножество.

------------------------------------------------------------------------

## 1.3. Архитектурные ограничения

Не менять без отдельного доказанного основания:

-   localhost/offline-first модель;
-   отсутствие обязательной авторизации;
-   single-user режим;
-   единый serial inference worker;
-   отсутствие параллельного TTS inference;
-   общий интерфейс `SynthesisEngine`;
-   возможность смешивать F5 и XTTS в одном диалоге;
-   lazy loading моделей;
-   независимость основного pipeline от конкретного TTS engine.

Не добавлять cloud dependency в критический runtime path.

------------------------------------------------------------------------

# 2. Целевая архитектура

Основное направление:

``` text
Project
 ├── metadata
 ├── source_text
 ├── render_settings
 ├── speakers
 ├── scenes
 │    └── replicas
 │         ├── text
 │         ├── voice
 │         ├── synthesis_settings
 │         ├── qa
 │         └── takes
 │              ├── audio
 │              ├── seed
 │              ├── engine
 │              ├── parameters
 │              └── qa
 └── renders
```

Иерархия параметров:

``` text
Engine defaults
      ↓
Voice preset
      ↓
Project/Speaker settings
      ↓
Replica override
```

Более нижний уровень имеет приоритет.

------------------------------------------------------------------------

# ФАЗА 1. Persistent Projects + SQLite

## Цель

Перестать рассматривать `job` как единственную долгоживущую сущность.

Добавить сохраняемый **Project**, который можно закрыть и открыть
позднее без потери структуры диалога, настроек, выбранных голосов, takes
и статуса работы.

## Требования

Добавить SQLite storage.

Рекомендуемая структура:

``` text
backend/
    db/
        connection.py
        migrations.py
        repositories/
            projects.py
            replicas.py
            takes.py
```

Минимальные сущности:

### Project

-   `id`
-   `name`
-   `created_at`
-   `updated_at`
-   `source_text`
-   `mode`
-   `render_settings`
-   `status`

### Speaker

-   project id;
-   source speaker/slot;
-   assigned voice id;
-   speaker-level overrides.

### Replica

-   project id;
-   order/index;
-   text;
-   speaker;
-   resolved voice;
-   overrides;
-   selected take;
-   render status.

### Take

-   replica id;
-   audio path;
-   seed;
-   engine;
-   effective parameters;
-   duration;
-   QA metadata;
-   created_at.

## API

Добавить REST API приблизительно следующего уровня:

``` text
POST   /api/projects
GET    /api/projects
GET    /api/projects/{id}
PATCH  /api/projects/{id}
DELETE /api/projects/{id}

POST   /api/projects/{id}/parse
POST   /api/projects/{id}/render
```

Не удалять существующие `/api/generate` и `/api/render-text` на этой
фазе.

## Migration

При первом запуске база должна создаваться автоматически.

Не требовать ручного SQL от пользователя.

## Тесты фазы

Добавить:

``` text
tests/test_projects_store.py
tests/test_projects_api.py
```

Обязательно проверить:

1.  создание проекта;
2.  чтение проекта после повторного открытия DB connection;
3.  обновление source text;
4.  сохранение speakers;
5.  сохранение replicas;
6.  сохранение take;
7.  выбор active take;
8.  удаление проекта;
9.  cascade/cleanup связанных записей;
10. отсутствие повреждения `voices.json`;
11. старые `/api/generate` продолжают работать;
12. некорректный project id возвращает 404;
13. данные двух проектов не смешиваются.

### Regression

``` bash
python -m pytest
```

## Ожидаемый результат

После перезапуска backend пользователь может открыть ранее созданный
проект и получить:

-   исходный текст;
-   список реплик;
-   назначенные голоса;
-   настройки;
-   сохранённые takes;
-   выбранные варианты.

Проект больше не зависит от 24-часового TTL обычного transient output.

------------------------------------------------------------------------

# ФАЗА 2. Replica Editor

## Цель

Сделать DSL-маркеры `(1)`, `(2)`, `(1 speed=...)` advanced-режимом, а
основной workflow --- визуальным.

## UI

После parsing показывать карточки реплик.

Пример:

``` text
АРТЕМ · Replica 12
Voice: Vlad
Engine: F5

Привет! Рад тебя видеть.

[Play] [Regenerate]

Take: 2
Speed: 1.00
Pause before: 350 ms
```

Карточка должна позволять:

-   выбрать voice;
-   видеть engine;
-   менять speed;
-   менять pause;
-   менять engine-specific parameters;
-   запускать regenerate;
-   прослушивать takes;
-   выбирать active take;
-   возвращать параметр к inherited value.

## Важное условие

Текстовый DSL не удалять.

Должны существовать два представления:

``` text
Visual editor
Source / Advanced
```

Изменение source должно позволять заново выполнить parse.

## Тесты фазы

Backend:

``` text
tests/test_replica_editor_api.py
```

Frontend желательно покрыть минимальными DOM/browser tests, если в
проект вводится frontend test runner.

Проверить:

1.  parser создаёт ожидаемое количество replicas;
2.  порядок replicas стабилен;
3.  speaker/slot корректно назначается;
4.  изменение voice одной replica не меняет остальные;
5.  override speed применяется только к нужной replica;
6.  reset override возвращает inherited value;
7.  regenerate изменяет только выбранную replica;
8.  take можно выбрать без повторного synthesis;
9.  исходный DSL продолжает парситься;
10. mixed slot/name syntax продолжает работать.

### Regression

``` bash
python -m pytest
```

## Ожидаемый результат

Для обычной работы пользователю больше не требуется редактировать
`(1 cfg=... nfe=...)`.

Все параметры конкретной реплики доступны визуально, но старый DSL
остаётся совместимым.

------------------------------------------------------------------------

# ФАЗА 3. Voice Presets + Parameter Inheritance

## Цель

Настройки, подобранные пользователем в `Прослушать`, должны становиться
реальными defaults этого voice.

## Иерархия

Реализовать:

``` text
engine defaults
    < voice preset
    < project speaker
    < replica override
```

## Требования

Для каждого synthesis request вычислять `effective_settings`.

Добавить единую функцию, например:

``` python
resolve_synthesis_settings(
    engine_defaults,
    voice_settings,
    speaker_overrides,
    replica_overrides,
)
```

Не размазывать merge logic по API и frontend.

API по возможности должен возвращать:

``` json
{
  "speed": {
    "value": 0.95,
    "source": "voice"
  }
}
```

или эквивалентную информацию, позволяющую UI показать источник значения.

## UI

Показывать inherited/overridden state.

Пример:

``` text
CFG 2.3
Inherited from voice "Anna"
```

Override:

``` text
CFG 3.0
Overridden for this replica
[Reset]
```

## Тесты фазы

Добавить:

``` text
tests/test_settings_resolution.py
```

Проверить:

1.  engine default без overrides;
2.  voice override engine default;
3.  speaker override voice;
4.  replica override speaker;
5.  reset replica возвращает speaker;
6.  reset speaker возвращает voice;
7.  параметры другого engine не применяются;
8.  clamp выполняется после resolution;
9.  unknown engine parameters безопасно игнорируются;
10. preview и normal render используют одинаковый voice preset.

### Regression

``` bash
python -m pytest
```

## Ожидаемый результат

Пользователь настраивает голос один раз.

Новый диалог автоматически получает эти настройки, а локальные изменения
конкретной реплики не портят voice preset.

------------------------------------------------------------------------

# ФАЗА 4. Smart QA

## Цель

Уменьшить стоимость Whisper QA.

Вместо двух режимов сделать:

``` text
Off
Smart
Strict
```

## Strict

Сохранить текущее поведение:

``` text
synthesis → Whisper → WER → retry
```

## Smart

Сначала выполнять дешёвый audio screening.

Проверять признаки:

-   duration / text length ratio;
-   слишком короткий результат;
-   слишком длинный результат;
-   silence ratio;
-   clipping;
-   abnormal RMS/energy;
-   пустой waveform;
-   подозрительные повторяющиеся участки;
-   extreme duration per word/syllable.

Только suspicious chunks отправлять в Whisper.

## QA metadata

Для replica/take хранить:

``` json
{
  "mode": "smart",
  "screening": {
    "suspicious": true,
    "reasons": []
  },
  "wer": 0.08,
  "attempts": 1
}
```

## Тесты фазы

Добавить:

``` text
tests/test_smart_qa.py
```

Использовать синтетические waveform fixtures.

Проверить:

1.  normal audio не отправляется в ASR;
2.  silence отправляется в ASR или отклоняется по установленной
    политике;
3.  clipping определяется;
4.  abnormal duration определяется;
5.  empty audio определяется;
6.  suspicious chunk вызывает Whisper QA;
7.  normal chunk в Smart mode не вызывает Whisper;
8.  Strict всегда вызывает Whisper;
9.  Off не запускает QA;
10. retry policy не превышает budget/attempt limits;
11. лучший take сохраняется при исчерпании QA;
12. QA metadata сохраняется.

### Performance test

На фиксированном тестовом диалоге сравнить:

``` text
Off
Smart
Strict
```

Зафиксировать:

-   render time;
-   количество ASR calls;
-   число retries;
-   peak memory.

## Ожидаемый результат

Smart QA должен существенно уменьшить число Whisper запусков на
нормальном тексте.

Критерий:

-   при чистом тестовом диалоге большинство chunks не проходят через
    Whisper;
-   Strict сохраняет прежнюю семантику;
-   Smart не ухудшает стабильность pipeline.

------------------------------------------------------------------------

# ФАЗА 5. Russian Text Normalization v2

## Цель

Улучшить то, что реально отправляется в TTS.

## Рефакторинг

Разделить preprocessing:

``` text
backend/text_normalization/
    pipeline.py
    numbers.py
    dates.py
    time.py
    units.py
    money.py
    abbreviations.py
    latin.py
    morphology.py
    pronunciation.py
```

## Поддержать

### Числительные + существительные

``` text
2 км  → два километра
5 км  → пять километров
21 рубль
22 рубля
25 рублей
```

### Dates

Поддержать типичные русские формы дат.

### Time

Например:

``` text
12:30
```

не должно случайно превращаться в два независимых числа.

### Money

Обработать:

-   ₽;
-   руб.;
-   рублей;
-   `$`;
-   `€`.

### Percent

``` text
5%
```

### Ranges

``` text
5–10 км
```

### Roman numerals

Например:

``` text
XXI век
```

### Number sign

``` text
№ 15
```

### Phones / URLs / email

Определить явную политику чтения или сохранения.

Не выполнять разрушительную нормализацию.

## Тесты фазы

Существенно расширить:

``` text
tests/test_text_preprocess.py
```

или разделить на:

``` text
tests/text_normalization/
```

Минимум добавить parameterized cases для:

-   1/2/5/11/21/22/25;
-   падежей;
-   годов;
-   дат;
-   времени;
-   денег;
-   процентов;
-   дробей;
-   единиц;
-   диапазонов;
-   сокращений;
-   mixed Cyrillic/Latin;
-   URL;
-   email;
-   phone;
-   leading zero;
-   version strings;
-   IP-like strings.

Проверить идемпотентность там, где она ожидается:

``` python
normalize(normalize(text)) == normalize(text)
```

### Regression

Запустить весь test suite.

## Ожидаемый результат

Фразы типа:

``` text
Я прошёл 2 км и заплатил 25 руб.
```

не должны превращаться в грамматически очевидно неверный текст вроде:

``` text
два километров
```

Нормализатор должен улучшать TTS input, не разрушая даты, версии, URL и
технические конструкции.

------------------------------------------------------------------------

# ФАЗА 6. Pronunciation Dictionary

## Цель

Дать пользователю контролируемый способ исправлять произношение терминов
и ударений.

## Pipeline

Для F5:

``` text
raw text
 ↓
normalization
 ↓
user pronunciation dictionary
 ↓
RUAccent
 ↓
F5
```

Для XTTS:

``` text
raw text
 ↓
normalization
 ↓
compatible pronunciation replacements
 ↓
XTTS
```

Не отправлять F5 `+` stress markup в XTTS.

## Формат

Хранить пользовательский словарь в SQLite или отдельном JSON.

Пример:

``` text
OpenAI       → оупен эй-ай
PostgreSQL   → постгрес
SQL          → эскьюэль
звонит       → звон+ит
```

Поддержать:

-   exact word;
-   case-insensitive matching, где безопасно;
-   whole-word matching;
-   preview результата.

## Тесты фазы

Добавить:

``` text
tests/test_pronunciation_dictionary.py
```

Проверить:

1.  exact replacement;
2.  отсутствие replacement внутри другого слова;
3.  регистр;
4.  несколько терминов;
5.  stress markup F5;
6.  stress markup не попадает в XTTS;
7.  dictionary применяется после normalization;
8.  пользовательская запись имеет приоритет над RUAccent;
9.  пустой словарь ничего не меняет;
10. malformed entry не ломает synthesis.

## Ожидаемый результат

Пользователь может исправить произношение имени, бренда или термина один
раз, после чего правило автоматически применяется во всех проектах.

------------------------------------------------------------------------

# ФАЗА 7. Preview: «Что услышит модель»

## Цель

Сделать preprocessing прозрачным.

## UI

Добавить preview:

``` text
Исходный текст

В 2026 году цена выросла на 5%.

Что услышит модель

В две тысячи двадцать шестом году цена выросла на пять процентов.
```

Для F5 добавить отображение accentized form.

Желательно показывать стадии:

``` text
Original
Normalized
Pronunciation dictionary
Accentized
```

## API

Добавить endpoint:

``` text
POST /api/text/preview
```

Он не должен запускать TTS.

## Тесты фазы

Добавить:

``` text
tests/test_text_preview_api.py
```

Проверить:

1.  preview F5;
2.  preview XTTS;
3.  RUAccent применяется только к поддерживаемому engine;
4.  dictionary виден в preview;
5.  preview совпадает с фактическим input synthesis;
6.  endpoint не загружает TTS engine;
7.  ошибка accentizer отражается явно;
8.  исходный текст не изменяется.

## Ожидаемый результат

Пользователь до дорогостоящего synthesis точно видит текст, который
будет отправлен модели.

------------------------------------------------------------------------

# ФАЗА 8. Voice Benchmark / Engine Comparison

## Цель

Позволить сравнивать один reference на нескольких движках.

## UI

Добавить действие:

``` text
Сравнить движки
```

Пользователь задаёт тестовую фразу.

Система генерирует отдельные takes:

``` text
F5
Play
Render: 1.4 sec
WER: 0.04

XTTS
Play
Render: 7.3 sec
WER: 0.02

XTTS Banana
Play
Render: 8.1 sec
WER: 0.08
```

Не объявлять автоматически «лучший голос» только по WER.

Пользователь сам выбирает preferred engine/take.

## Тесты фазы

Добавить:

``` text
tests/test_voice_benchmark.py
```

На `StubEngine` проверить:

1.  benchmark вызывает все доступные выбранные engines;
2.  unavailable engine не рушит весь benchmark;
3.  результаты разделены;
4.  render duration сохраняется;
5.  QA metadata сохраняется;
6.  выбор preferred engine обновляет voice;
7.  reference не перезаписывается;
8.  параметры движков не смешиваются.

### Manual/live test

На реальных моделях проверить один мужской и один женский reference.

## Ожидаемый результат

Пользователь может за один workflow прослушать один голос на нескольких
моделях и вручную выбрать подходящий engine.

------------------------------------------------------------------------

# ФАЗА 9. Model Manager

## Цель

Убрать необходимость вручную управлять model files.

## UI

Добавить раздел:

``` text
Models
```

Показывать:

-   installed/not installed;
-   model size;
-   loaded/unloaded;
-   path;
-   download state;
-   error;
-   disk usage.

Пример:

``` text
F5 Russian
Installed · Loaded · 1.4 GB

XTTS v2
Not installed · 1.9 GB
[Download]

Whisper
Installed · Unloaded · 1.6 GB
```

## Backend

Создать model registry/service, который знает:

-   model id;
-   required files;
-   repository;
-   local path;
-   download strategy;
-   approximate size;
-   runtime state.

Download должен иметь progress/error state.

## Тесты фазы

Добавить:

``` text
tests/test_model_manager.py
tests/test_models_api.py
```

Сеть мокать.

Проверить:

1.  installed detection;
2.  missing required file;
3.  download success;
4.  download failure;
5.  interrupted download;
6.  повторный download не повреждает установленную модель;
7.  model path validation;
8.  disk usage;
9.  API status;
10. удаление модели запрещено, пока engine активно используется;
11. F5/XTTS registry metadata корректны.

## Ожидаемый результат

Для установки/диагностики модели пользователю не требуется вручную
запускать `snapshot_download` или разбирать структуру каталогов.

------------------------------------------------------------------------

# ФАЗА 10. Engine Unload + Memory Pressure

## Цель

Освобождать память без полного рестарта приложения.

## Interface

Расширить `SynthesisEngine`:

``` python
load()
unload()
is_loaded
synthesize(...)
```

## Политика

Добавить ручной unload и optional idle unload.

Например:

``` text
TTS_ENGINE_IDLE_UNLOAD_MIN=15
```

Не выгружать engine:

-   во время активной synthesis;
-   если он нужен текущей queued operation, если это создаёт race;
-   посреди regenerate.

После unload корректно очищать Python/Torch/MPS resources.

## Тесты фазы

Добавить:

``` text
tests/test_engine_lifecycle.py
```

Проверить:

1.  load → loaded;
2.  unload → unloaded;
3.  повторный unload безопасен;
4.  synthesize после unload вызывает reload;
5.  нельзя unload во время active use;
6.  idle policy;
7.  registry не возвращает stale object state;
8.  ошибка unload не убивает backend;
9.  очередь продолжает работать после unload/reload.

### Live memory test

На Apple Silicon:

1.  загрузить F5;
2.  записать memory metrics;
3.  загрузить XTTS;
4.  записать memory metrics;
5.  unload XTTS;
6.  проверить снижение memory pressure/footprint настолько, насколько
    позволяет runtime.

Не утверждать точное освобождение MPS памяти без измерения.

## Ожидаемый результат

Пользователь может освободить неиспользуемую модель без рестарта
backend.

------------------------------------------------------------------------

# ФАЗА 11. Priority Queue + Job Cancellation

## Цель

Сохранить single-worker inference, но улучшить интерактивность.

## Priority

Рекомендуемые классы:

``` text
0 preview
1 replica regenerate
2 normal render
3 batch/background render
```

Использовать priority queue.

Внутри одинакового priority сохранять FIFO.

## Cancellation

Добавить:

``` text
POST /api/jobs/{job_id}/cancel
```

или эквивалент.

Состояния job должны явно различать:

``` text
queued
processing
done
error
cancelled
```

Для processing cancellation использовать cooperative cancellation между
chunks/attempts.

Не пытаться небезопасно убивать поток внутри model call.

## Тесты фазы

Расширить:

``` text
tests/test_job_queue.py
```

Проверить:

1.  preview идёт раньше queued batch;
2.  regenerate идёт раньше normal render;
3.  FIFO внутри priority;
4.  queued job отменяется;
5.  cancelled job никогда не запускается;
6.  processing job отменяется на ближайшей безопасной точке;
7.  cancellation не повреждает следующий job;
8.  partial output cleanup;
9.  cancelled status сохраняется;
10. повторная cancellation идемпотентна.

## Ожидаемый результат

Длинный render не делает приложение полностью неинтерактивным на уровне
очереди.

Короткие preview/regenerate получают приоритет, а ошибочно запущенную
задачу можно отменить.

------------------------------------------------------------------------

# ФАЗА 12. Engine-aware ETA

## Цель

Сделать ETA реалистичным для F5, XTTS, cold start и QA.

## Метрики

Хранить rolling/EWMA statistics:

``` text
engine
chars
audio duration
render duration
cold/warm
qa mode
qa attempts
```

Не использовать только число replicas.

## Тесты фазы

Добавить:

``` text
tests/test_eta.py
```

Проверить:

1.  F5 и XTTS имеют независимую статистику;
2.  cold start penalty учитывается;
3.  warm start не получает cold penalty;
4.  QA multiplier учитывается;
5.  EWMA обновляется;
6.  отсутствие истории имеет разумный fallback;
7.  ETA не отрицательный;
8.  completed work уменьшает remaining ETA;
9.  mixed-engine project считается по составляющим.

## Ожидаемый результат

Status UI показывает приблизительно полезное время:

``` text
18 / 47 реплик
≈ 2 мин 40 сек
```

а не формальную оценку, одинаковую для разных engines.

------------------------------------------------------------------------

# ФАЗА 13. Timeline / Mini-DAW

## Цель

Перейти от единственного итогового player к визуальной работе с
репликами.

## UI

Добавить timeline:

``` text
00:00                                      01:42

ARTEM  ████████       █████████
MARGO          ███████         ███████
```

Каждый segment соответствует replica/take.

Клик открывает inspector:

``` text
Replica 17
Speaker: Margo
Voice: Anna

[Play]

Take 1
Take 2
Take 3

[Regenerate]

Pause before
Speed
Engine params
QA
```

## Backend

Timeline должен строиться по реальным duration/start/end, а не
приблизительным символам.

После замены take timestamps хвоста должны пересчитываться.

## Тесты фазы

Добавить:

``` text
tests/test_timeline.py
```

Проверить:

1.  start/end первой replica;
2.  pause учитывается;
3.  crossfade учитывается;
4.  замена более длинным take сдвигает хвост;
5.  замена коротким take сдвигает хвост назад;
6.  mixed engines не влияют на timestamp semantics;
7.  timeline совпадает с итоговой длительностью файла;
8.  active take соответствует timeline segment.

## Ожидаемый результат

Пользователь может визуально найти нужную реплику, прослушать её и
заменить take без поиска по длинному текстовому списку.

------------------------------------------------------------------------

# ФАЗА 14. Extended Quality Metadata

## Цель

Дополнить WER объективными диагностическими признаками.

## Для take хранить

По возможности:

-   WER;
-   clipping;
-   silence ratio;
-   duration;
-   duration/text ratio;
-   peak dBFS;
-   RMS;
-   LUFS;
-   QA attempts;
-   screening reasons.

Не превращать эти показатели в автоматическую оценку естественности
голоса.

## UI

Обычному пользователю показывать только warnings.

Подробности разместить в:

``` text
Diagnostics
```

## Тесты фазы

Добавить:

``` text
tests/test_take_quality.py
```

Проверить синтетическими fixtures:

1.  clipping;
2.  silence;
3.  normal signal;
4.  abnormal short duration;
5.  abnormal long duration;
6.  LUFS calculation;
7.  peak;
8.  metadata serialization;
9.  отсутствие optional metric не ломает API.

## Ожидаемый результат

При плохом take система может объяснить техническую причину:

``` text
Clipping detected
Unusually long duration
High silence ratio
```

а не ограничиваться WER.

------------------------------------------------------------------------

# ФАЗА 15. Project Export / Import

## Цель

Сделать проекты переносимыми и архивируемыми.

## Формат

Например:

``` text
project.ttsproject
```

ZIP-контейнер:

``` text
project.json
references/
takes/
```

Не включать model weights.

## Export audio

Добавить варианты:

-   final WAV;
-   final MP3;
-   replicas as separate WAV;
-   stems per speaker;
-   transcript JSON;
-   SRT;
-   VTT.

## Subtitles

Использовать реальные timeline timestamps.

## Тесты фазы

Добавить:

``` text
tests/test_project_export.py
```

Проверить:

1.  export → import round trip;
2.  project metadata сохраняется;
3.  speaker assignments сохраняются;
4.  takes сохраняются;
5.  missing optional take корректно обрабатывается;
6.  path traversal в ZIP запрещён;
7.  corrupt archive даёт понятную ошибку;
8.  model weights не экспортируются;
9.  SRT timestamps валидны;
10. VTT timestamps валидны;
11. stems содержат ожидаемые speaker segments.

## Ожидаемый результат

Пользователь может перенести проект на другой компьютер или сохранить
его как архив и позднее полностью восстановить.

------------------------------------------------------------------------

# ФАЗА 16. Desktop Packaging

## Цель

После стабилизации предыдущих фаз упростить запуск приложения.

Не выполнять эту фазу раньше persistence/model manager.

Возможные варианты:

-   lightweight launcher;
-   packaged Python runtime;
-   desktop wrapper.

Критерии выбора:

-   доступ к локальным моделям;
-   ffmpeg;
-   MPS;
-   понятные logs;
-   возможность обновления;
-   отсутствие cloud requirement.

## Тесты фазы

Минимум smoke tests:

1.  clean installation;
2.  first launch;
3.  model discovery;
4.  F5 synthesis;
5.  XTTS synthesis;
6.  project create/save/reopen;
7.  export;
8.  restart;
9.  path with spaces;
10. non-ASCII project/voice names.

## Ожидаемый результат

Пользователь запускает приложение без ручной активации virtualenv и
команды `./run.sh`, при этом backend остаётся локальным.

------------------------------------------------------------------------

# 3. Что намеренно НЕ делать сейчас

Не добавлять без отдельного требования:

## Cloud deployment

Проект должен оставаться полностью работоспособным localhost/offline.

## Authentication / multi-user

Текущая задача --- single-user workstation.

## Realtime streaming TTS

Основной use case --- offline render и редактирование результата.

## Parallel model inference

Не запускать несколько тяжёлых TTS inference одновременно только ради
RPS.

На Apple Silicon главный лимит --- unified memory и model footprint.

Сначала использовать:

-   priority queue;
-   cancellation;
-   lazy loading;
-   unload;
-   caching.

------------------------------------------------------------------------

# 4. Рекомендуемый порядок реализации

Строго рекомендуемый порядок:

``` text
1. Persistent Projects + SQLite
2. Replica Editor
3. Voice Presets + Inheritance
4. Smart QA
5. Text Normalization v2
6. Pronunciation Dictionary
7. Text/Input Preview
8. Voice Benchmark
9. Model Manager
10. Engine Unload
11. Priority Queue + Cancellation
12. Engine-aware ETA
13. Timeline / Mini-DAW
14. Extended Quality Metadata
15. Project Export / Import
16. Desktop Packaging
```

Не начинать Timeline до появления persistent Project/Replica/Take model.

Не начинать Desktop Packaging до стабилизации persistence и model
management.

------------------------------------------------------------------------

# 5. Обязательный шаблон работы агента для КАЖДОЙ фазы

Перед реализацией агент должен вывести краткий plan:

``` text
PHASE N

Affected modules:
- ...

Schema/API changes:
- ...

Backward compatibility risks:
- ...

Tests to add:
- ...
```

Затем реализовать фазу.

После реализации обязательно выполнить тесты.

Финальный отчёт фазы должен иметь формат:

``` text
PHASE N COMPLETE

Implemented:
- ...

Files changed:
- ...

Tests added:
- ...

Test result:
python -m pytest
X passed, 0 failed

Additional live/load tests:
...

Backward compatibility:
PASS / FAIL

Expected result verification:
PASS / FAIL

Remaining issues:
- ...
```

Если тесты не проходят, агент **не имеет права считать фазу
завершённой**.

------------------------------------------------------------------------

# 6. Definition of Done всего roadmap

Roadmap считается реализованным, когда пользователь может:

1.  создать проект;
2.  вставить диалог или текст;
3.  получить визуальный список replicas;
4.  назначить voices;
5.  автоматически использовать voice presets;
6.  изменить параметры отдельной replica;
7.  увидеть normalized/model input;
8.  использовать pronunciation dictionary;
9.  запустить Off/Smart/Strict QA;
10. сравнить voice на нескольких engines;
11. установить/проверить модели через Model Manager;
12. выгрузить неиспользуемый engine;
13. запустить длинный render;
14. отменить его;
15. запускать preview/regenerate с приоритетом;
16. видеть адекватный ETA;
17. работать с takes на timeline;
18. видеть технические QA warnings;
19. закрыть приложение;
20. после запуска восстановить проект;
21. экспортировать проект;
22. экспортировать WAV/MP3/SRT/VTT/stems;
23. импортировать проект обратно.

------------------------------------------------------------------------

# 7. Финальный regression gate

После завершения всех фаз выполнить:

``` bash
source venv/bin/activate

python -m pytest
```

Затем живые тесты минимум:

``` bash
./venv/bin/python tools/loadtest.py light
./venv/bin/python tools/loadtest.py queue
./venv/bin/python tools/loadtest.py qa
./venv/bin/python tools/loadtest.py engines
```

Провести ручной end-to-end сценарий:

``` text
Create project
→ add two voices
→ parse mixed-speaker dialogue
→ render F5 + XTTS
→ Smart QA
→ regenerate replica
→ select old take
→ restart backend
→ reopen project
→ export WAV
→ export SRT
→ export project
→ import project
→ render again
```

## Финальный ожидаемый результат

Приложение должно представлять собой устойчивую локальную TTS-студию,
где:

-   TTS engines являются заменяемыми backend-компонентами;
-   проекты и takes сохраняются;
-   пользователь работает с репликами визуально;
-   настройки наследуются предсказуемо;
-   preprocessing прозрачен;
-   русский текст нормализуется корректнее;
-   произношение можно контролировать словарём;
-   QA имеет дешёвый Smart mode и полный Strict mode;
-   модели можно устанавливать и выгружать;
-   очередь остаётся memory-safe, но становится приоритетной;
-   jobs можно отменять;
-   ETA учитывает реальную скорость движков;
-   timeline позволяет редактировать результат как аудиопроект;
-   проекты можно экспортировать и восстанавливать;
-   каждое крупное изменение защищено автоматическими тестами.

Главный принцип дальнейшей разработки:

> **Не добавлять новую сложность без измеримого улучшения workflow,
> качества или устойчивости. Каждая фаза завершается тестами и
> проверяемым пользовательским результатом.**
