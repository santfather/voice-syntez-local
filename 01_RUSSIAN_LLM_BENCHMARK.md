# Задача 1 --- Russian Linguistic LLM Benchmark

## Назначение

Создать воспроизводимый benchmark для выбора локальной LLM, которая
лучше всего подходит для подготовки **русского текста** к TTS:
разрешение омографов по контексту, восстановление `ё`, выявление
орфографических/лексических проблем, работа с именами, топонимами,
аббревиатурами, числами, иностранными словами и сложными контекстными
случаями.

Benchmark **не должен** выбирать модель по общим рейтингам. Победитель
определяется только качеством на корпусе Voice_syntez, стабильностью
JSON, скоростью и реальным потреблением памяти на целевой машине:
**MacBook Pro M3, 18 GB unified memory**.

Задача является самостоятельной и должна быть завершена **до** включения
LLM в production pipeline.

------------------------------------------------------------------------

## 1. Исходные ограничения проекта

Перед реализацией агент обязан прочитать актуальные `README.md`,
проектную базу знаний и код, связанный с:

-   `text_preprocess`;
-   восстановлением `ё`;
-   pronunciation dictionary;
-   RUAccent;
-   `audio_pipeline.preview_text(...)`;
-   Dialogue/Continuous Text preprocessing;
-   project/replica persistence;
-   engine registry;
-   memory watchdog;
-   load tests;
-   Whisper QA;
-   F5 / XTTS lifecycle.

Не создавать второй независимый preprocessing pipeline.

Текущий проект рассчитан на Apple Silicon и уже может одновременно
использовать тяжёлые компоненты: F5-TTS, XTTS, XTTS Banana, Whisper и
RUAccent. Согласно текущей документации проекта, ранее наблюдалась
системная память до \~86% при QA и до \~83% при двух TTS-движках без
Whisper. Поэтому LLM нельзя добавлять как ещё одну постоянно резидентную
модель без memory policy.

------------------------------------------------------------------------

## 2. Установка Ollama

Если Ollama отсутствует --- установить официальную актуальную версию для
macOS.

Проверить:

``` bash
ollama --version
ollama list
```

Не привязывать Python-код приложения к CLI. Benchmark и будущая
интеграция должны обращаться к локальному Ollama HTTP API через
отдельный adapter/client.

Ollama должен считаться **опциональной локальной зависимостью**.
Отсутствие Ollama не должно ломать существующий TTS.

------------------------------------------------------------------------

## 3. Обязательные модели benchmark

Скачать минимум следующие официальные модели Ollama:

``` bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3:4b-instruct-2507-q8_0
ollama pull qwen3:8b
ollama pull gemma3:4b
```

Контрольные размеры на момент постановки задачи:

  Модель                            Квантование / класс     Примерный размер
  --------------------------------- --------------------- ------------------
  `qwen3:4b-instruct-2507-q4_K_M`   Q4_K_M                          \~2.5 GB
  `qwen3:4b-instruct-2507-q8_0`     Q8_0                            \~4.3 GB
  `qwen3:8b`                        Q4_K_M                          \~5.2 GB
  `gemma3:4b`                       Q4-class                        \~3.3 GB

Основные кандидаты на production: `qwen3:8b` и
`qwen3:4b-instruct-2507-q8_0`.

`gemma3:4b` --- независимый multilingual baseline, чтобы benchmark не
был сравнением только вариантов Qwen.

Не скачивать для этой задачи `qwen3:14b`, `gemma3:12b`, 27B/30B/32B и
более крупные модели. На машине с 18 GB unified memory они создают
неоправданный риск memory pressure рядом с TTS/Whisper.

Разрешается добавить **один** русскоязычный community fine-tune как
экспериментальный кандидат, но: - он не может заменить официальные
baseline-модели; - зафиксировать точный model tag/digest, лицензию и
источник; - не объявлять его лучшим только из-за `ru` в названии; -
production-выбор только по результатам этого benchmark.

------------------------------------------------------------------------

## 4. Версионирование benchmark

Каждый прогон сохраняет:

``` text
benchmark_version
dataset_version
git_commit
timestamp
macOS version
machine / chip
total RAM
ollama version
model tag
model digest
model reported size
context setting
temperature
seed/options if supported
prompt_version
schema_version
```

Результат должен быть воспроизводимым.

Не использовать плавающий `latest`, если после выбора production-модели
можно закрепить конкретный digest/tag.

------------------------------------------------------------------------

## 5. Корпус

Создать versioned dataset, например:

``` text
benchmarks/russian_linguistics/
    dataset.v1.jsonl
    schema.json
    README.md
    expected/
```

Минимум **300 вручную проверенных случаев** в v1.

Целевой размер после стабилизации --- 500+.

Нельзя генерировать весь gold corpus самой тестируемой LLM.

Gold labels должны быть проверены человеком. Спорные случаи помечать как
ambiguous/review, а не придумывать единственный «правильный» ответ.

------------------------------------------------------------------------

## 6. Категории корпуса

Обязательно покрыть:

### 6.1 Омографы

Контекстные пары и тройки:

``` text
за́мок / замо́к
му́ка / мука́
а́тлас / атла́с
пла́чу / плачу́
сто́ит / стои́т
у́же / уже́
о́рган / орга́н
бе́лки / белки́
кру́жки / кружки́
по́лки / полки́
```

Использовать предложения, где значение определяется контекстом, а не
отдельным словом.

### 6.2 `е/ё`

Проверять восстановление `ё` только там, где оно действительно
требуется:

``` text
все / всё
осел / осёл
передохнем / передохнём
узнает / узнаёт
```

Добавить случаи, где замена `е → ё` была бы ошибкой.

### 6.3 Морфология и согласование

Слова, где форма и ударение зависят от числа, падежа, времени, лица.

### 6.4 Имена и фамилии

Русские и иностранные имена, редкие фамилии, инициалы.

Не требовать от LLM выдумывать произношение неизвестной фамилии:
правильным результатом может быть `needs_review=true`.

### 6.5 Топонимы

Россия/СНГ и иностранные географические названия.

### 6.6 Аббревиатуры и сокращения

``` text
МГУ
МЧС
РЖД
ООО
ул.
руб.
км/ч
т. д.
и т. п.
```

Ожидаемый результат должен различать буквенное чтение, слово и раскрытие
сокращения.

### 6.7 Числа

Кардинальные/порядковые, годы, даты, время, дроби, проценты, валюты,
единицы измерения, номера.

### 6.8 Латиница и англицизмы

``` text
Python
GitHub
API
JSON
OpenAI
FastAPI
macOS
```

Не считать обязательной транслитерацию, если production policy требует
review.

### 6.9 Техническая лексика

Термины из реальных сценариев проекта.

### 6.10 Диалоговый контекст

Значение целевой реплики определяется предыдущими/следующими репликами.

### 6.11 Короткие реплики

``` text
Да.
Нет.
Конечно.
Правда?
Именно.
Хорошо.
```

Проверить, умеет ли LLM определить контекстную функцию, не переписывая
target.

### 6.12 Негативные случаи

Текст, где **ничего исправлять не нужно**.

Это обязательно для измерения false positives.

------------------------------------------------------------------------

## 7. Формат задачи для LLM

LLM не должна возвращать переписанный абзац.

Вход:

``` json
{
  "replica_id": 17,
  "target_text": "Он поменял замок на входной двери.",
  "context_before": ["Ключ снова застрял."],
  "context_after": ["Теперь дверь открывается нормально."],
  "language": "ru"
}
```

Выход --- только structured annotations:

``` json
{
  "replica_id": 17,
  "items": [
    {
      "span_start": 11,
      "span_end": 16,
      "source": "замок",
      "type": "homograph",
      "meaning": "lock",
      "suggested_form": "замо́к",
      "confidence": 0.98,
      "needs_review": false
    }
  ]
}
```

Span offsets должны проверяться backend-кодом против исходной строки.

LLM не имеет права молча изменять текст вне указанных spans.

------------------------------------------------------------------------

## 8. Единый prompt

Все модели получают: - одинаковый system prompt; - одинаковую JSON
schema; - одинаковый corpus; - одинаковый объём контекста; - одинаковые
sampling settings, насколько API моделей это позволяет.

Основной benchmark запускать в максимально детерминированном режиме.

Отдельно допускается stability-run с несколькими повторами.

Prompt хранить versioned-файлом:

``` text
benchmarks/russian_linguistics/prompts/analyzer.v1.txt
```

------------------------------------------------------------------------

## 9. Метрики

Считать минимум:

``` text
homograph_accuracy
yo_precision
yo_recall
yo_f1
issue_precision
issue_recall
issue_f1
false_positive_rate
needs_review_precision
span_accuracy
valid_json_rate
schema_valid_rate
source_preservation_rate
latency_p50
latency_p95
tokens_per_second
peak_process_memory
peak_system_memory_percent
memory_pressure_events
```

Отдельно считать результаты по каждой категории.

Не сводить всё к одному непрозрачному score.

Итоговый отчёт должен показывать trade-off качество / память / скорость.

------------------------------------------------------------------------

## 10. Критические ошибки

Отдельно считать `critical_error_rate`.

Критическая ошибка: - изменение смысла; - уверенная неправильная
постановка омографа; - выдуманное слово/произношение; - изменение
числа/даты/имени; - изменение текста вне разрешённого span; - невалидный
JSON, если повторный repair также не помог; - high-confidence ответ там,
где gold требует review.

------------------------------------------------------------------------

## 11. Memory Safety для M3 18 GB

### 11.1 Основной принцип

Benchmark моделей выполняется **последовательно**, не параллельно.

Запрещено одновременно держать несколько benchmark LLM загруженными.

### 11.2 Лимиты

Ввести конфиг:

``` text
LLM_MEMORY_NORMAL_MAX_PERCENT=70
LLM_MEMORY_WARNING_PERCENT=75
LLM_MEMORY_CRITICAL_PERCENT=82
LLM_MEMORY_HARD_STOP_PERCENT=88
```

Это стартовые безопасные значения, которые агент обязан проверить
реальными замерами.

Поведение:

``` text
< 70%     NORMAL
70–75%    NORMAL, но логировать рост
75–82%    WARNING: не запускать новую тяжёлую ML-задачу
82–88%    CRITICAL: остановить dispatch тяжёлых задач, выгрузить неиспользуемые модели
>= 88%    HARD STOP: benchmark/inference не начинать; текущую безопасно завершить/прервать согласно архитектуре
```

Не пытаться «занять все 18 GB».

### 11.3 Абсолютный резерв

Цель --- сохранять минимум **3.5--4 GB системного резерва** для
macOS/UI/служебных процессов.

Если memory pressure macOS становится жёлтым/красным раньше процентных
порогов, pressure state имеет приоритет над процентом.

### 11.4 Запрет тяжёлой конкуренции

Во время benchmark LLM не запускать одновременно:

``` text
F5 synthesis
XTTS synthesis
XTTS Banana
Whisper QA/transcription
другой LLM inference
```

Лёгкие HTTP API приложения могут работать.

RUAccent и обычный deterministic preprocessing допустимы, если реальные
замеры не показывают проблему.

### 11.5 Контекст

Не использовать огромные context windows только потому, что модель их
поддерживает.

Для production-like benchmark начать с:

``` text
num_ctx = 8192
```

и отдельным тестом проверить 4096/8192/16384.

Выбирать минимальный context, достаточный для абзаца/сцены. Большой KV
cache --- лишнее давление на unified memory.

### 11.6 Keep-alive

После каждого model-run модель должна выгружаться/истекать из Ollama
перед загрузкой следующей. Проверить фактическое освобождение памяти.

------------------------------------------------------------------------

## 12. Benchmark runner

Создать CLI, например:

``` bash
python tools/benchmark_russian_llm.py \
  --dataset benchmarks/russian_linguistics/dataset.v1.jsonl \
  --models qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct-2507-q8_0 qwen3:8b gemma3:4b
```

Поддержать:

``` text
--category
--limit
--model
--repeat
--output
--resume
```

Каждый raw response сохранять отдельно от normalized metrics.

Секретов в benchmark нет; пользовательский production-текст в benchmark
artifacts не сохранять без необходимости.

------------------------------------------------------------------------

## 13. Unit tests без реальных моделей

Обычный `pytest` не должен загружать Ollama-модели.

Использовать fake/stub Ollama client.

Обязательные тесты:

``` text
test_benchmark_dataset_schema
test_benchmark_gold_spans_match_source
test_benchmark_rejects_invalid_gold_span
test_llm_response_schema_validation
test_llm_response_rejects_changed_source_span
test_llm_response_rejects_out_of_bounds_span
test_llm_response_rejects_unknown_replica
test_metrics_homograph_accuracy
test_metrics_yo_precision_recall
test_metrics_false_positive_rate
test_metrics_valid_json_rate
test_metrics_critical_error_rate
test_benchmark_models_run_sequentially
test_memory_warning_blocks_next_heavy_job
test_memory_critical_blocks_llm_start
test_hard_stop_never_starts_model
test_benchmark_resume_does_not_duplicate_cases
test_model_metadata_is_saved
test_prompt_version_is_saved
```

------------------------------------------------------------------------

## 14. Phase plan

### Phase 1 --- Infrastructure

Реализовать Ollama benchmark client, schemas, versioning, fake client.

**Tests:** schema/client/versioning tests.

**Gate:** полный существующий pytest + новые tests зелёные.

### Phase 2 --- Gold corpus v1

Создать минимум 300 проверенных русских кейсов.

**Tests:** dataset schema, span integrity, category coverage, duplicate
detection.

**Gate:** dataset validator проходит 100%.

### Phase 3 --- Metrics

Реализовать метрики по категориям и critical errors.

**Tests:** все metric tests на hand-crafted fixtures.

**Gate:** результаты вручную сверены на маленьком fixture.

### Phase 4 --- Memory guard

Реализовать последовательный model runner и memory policy.

**Tests:** mocked NORMAL/WARNING/CRITICAL/HARD STOP.

**Gate:** никакого реального исчерпания RAM в тестах.

### Phase 5 --- Real benchmark

Скачать четыре обязательные модели и выполнить полный corpus.

**Tests:** smoke по 10--20 кейсам перед full run.

**Gate:** все модели дают отчёт; нет swap/memory-pressure зависания
системы.

### Phase 6 --- Selection

Сравнить модели.

Production candidate выбирать по: 1. русская контекстная точность; 2.
critical error rate; 3. false positives; 4. JSON/schema reliability; 5.
память; 6. latency.

Не выбирать модель только по скорости.

### Phase 7 --- Regression baseline

Сохранить выбранные результаты как baseline для будущих моделей/prompt
changes.

**Gate:** будущий benchmark умеет сравнить новый run с baseline и
показать регрессии.

------------------------------------------------------------------------

## 15. Acceptance Criteria

Задача завершена, если:

1.  Ollama установлен и health-check работает.
2.  Все четыре обязательные модели доступны локально.
3.  Есть versioned corpus \>=300 русских кейсов.
4.  Gold corpus проверен и валидируется.
5.  Есть единая structured-output schema.
6.  Есть воспроизводимый runner.
7.  Все модели прогоняются последовательно.
8.  Memory guard не допускает тяжёлой конкуренции.
9.  Система не подвисает во время полного benchmark.
10. Есть метрики по каждой лингвистической категории.
11. Отдельно измеряются critical errors и false positives.
12. Выбран primary model и fallback model на основании данных.
13. Сохранены model tags/digests и prompt version.
14. Все существующие и новые automated tests проходят.
15. Выполнен реальный smoke/full benchmark.
16. Результаты добавлены в проектную документацию и базу знаний.

------------------------------------------------------------------------

## 16. Обновление документации и базы знаний --- обязательно

По окончании задачи обновить:

``` text
README.md
```

Добавить раздел: - Ollama как optional dependency; - установка; - список
benchmark-моделей; - команды запуска benchmark; - memory policy; -
выбранная primary/fallback модель; - ограничения.

Создать/обновить проектную базу знаний:

``` text
docs/knowledge/llm-russian-benchmark.md
```

Зафиксировать: - зачем нужен benchmark; - dataset version; - prompt
version; - модели + digest; - результаты; - известные слабые
категории; - memory measurements; - production recommendation; - дату и
git commit.

Также создать:

``` text
docs/architecture/ollama-memory-policy.md
```

Документ должен стать source of truth по совместному использованию
Ollama/TTS/Whisper.

Если в проекте уже существует другой каталог/формат базы знаний ---
использовать существующую структуру вместо создания дубликата.

------------------------------------------------------------------------

## 17. Финальный отчёт ИИ-агента

В конце агент обязан сообщить:

``` text
changed files
installed Ollama version
downloaded model tags/digests/sizes
dataset version/count/category distribution
prompt/schema version
results per model
critical errors
false-positive rates
latency
peak memory
memory-pressure observations
selected primary model
selected fallback model
why selected
all tests + results
real benchmark command/results
documentation updated
known limitations
```

Не переходить к production-интеграции Analyzer, пока этот benchmark не
завершён и primary/fallback model не выбраны.

------------------------------------------------------------------------

## 18. Справочные источники на момент постановки задачи

Проверить актуальность перед установкой:

-   Ollama Qwen3 tags: https://ollama.com/library/qwen3/tags
-   Qwen3 8B: https://ollama.com/library/qwen3:8b
-   Qwen3 4B Instruct Q8:
    https://ollama.com/library/qwen3:4b-instruct-2507-q8_0
-   Gemma 3: https://ollama.com/library/gemma3

Теги/размеры могут измениться. В документации проекта фиксировать
реально установленный digest.
