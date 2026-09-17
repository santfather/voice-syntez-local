# Задача 2 --- Local LLM Russian Linguistic Analyzer

## Назначение

После завершения `Russian Linguistic LLM Benchmark` встроить выбранную
локальную LLM через Ollama в Voice_syntez как **контекстный русский
лингвистический анализатор перед TTS**.

Analyzer должен улучшать подготовку русского текста к синтезу, но не
заменять существующие детерминированные механизмы.

Главный принцип:

> LLM анализирует смысл и предлагает структурированные annotations.
> Backend остаётся source of truth и сам решает, какие изменения
> допустимо применить.

LLM не должна получать право свободно переписывать `source_text` или
`final_text`.

------------------------------------------------------------------------

## 1. Dependency на задачу benchmark

Эту задачу начинать только после завершения
`Russian Linguistic LLM Benchmark`.

Не хардкодить название модели заранее.

В конфиг production попадают результаты benchmark:

``` text
LLM_PRIMARY_MODEL=<winner tag/digest>
LLM_FALLBACK_MODEL=<fallback tag/digest>
```

Если benchmark ещё не завершён --- Analyzer остаётся disabled.

Ожидаемые стартовые кандидаты benchmark: - `qwen3:8b`; -
`qwen3:4b-instruct-2507-q8_0`; - `qwen3:4b-instruct-2507-q4_K_M`; -
`gemma3:4b`.

Production-модель выбирается **по benchmark**, а не по этой инструкции.

------------------------------------------------------------------------

## 2. Место в существующей архитектуре

Не создавать второй preprocessing pipeline.

Целевая схема:

``` text
SOURCE TEXT / FILE
        ↓
DIALOGUE PARSER
        ↓
speaker / replica resolution
        ↓
existing deterministic normalization
        ↓
existing ё stage
        ↓
LOCAL LLM RUSSIAN ANALYSIS
        │
        ├── homograph resolution
        ├── suspicious ё cases
        ├── unknown / rare words
        ├── proper names / surnames
        ├── toponyms
        ├── abbreviations
        ├── foreign words
        ├── contextual pronunciation candidates
        ├── semantic phrase boundaries
        └── short-utterance context hints
        ↓
PRONUNCIATION REVIEW
        ↓
PROJECT DICTIONARY → GLOBAL DICTIONARY
        ↓
existing engine-specific finalization
        ├── F5 → RUAccent / supported stress markup
        └── XTTS → no F5 "+" markup
        ↓
final_text
        ↓
Short Utterance Strategy
        ↓
TTS
```

Если фактический pipeline проекта отличается --- сначала документировать
реальный call graph и адаптировать интеграцию без дублирования логики.

------------------------------------------------------------------------

## 3. Что LLM должна делать

### 3.1 Омографы

Определять значение по контексту:

``` text
Он открыл старый замок.
На двери сломался замок.
```

Возвращать annotation, а не новый текст.

### 3.2 `е/ё`

LLM может: - подтверждать существующий `ё` resolver; - находить
контекстные сомнения; - предлагать correction; - повышать приоритет
review при конфликте.

Не выполнять глобальную замену `е → ё`.

### 3.3 Незнакомые/редкие слова

Находить: - имена; - фамилии; - топонимы; - бренды; - иностранные
слова; - технические термины; - аббревиатуры.

Если произношение ненадёжно --- `needs_review=true`.

Не выдумывать уверенное произношение.

### 3.4 Словарь

LLM создаёт **candidate**, но не пишет автоматически в Global
Dictionary.

Пользователь может: - принять для Project Dictionary; - принять
глобально; - отредактировать; - выбрать вариант; - оставить как есть; -
отклонить.

Project Dictionary имеет приоритет над Global Dictionary.

### 3.5 Короткие реплики

Analyzer может возвращать semantic hints:

``` text
SHORT_REPLY
QUESTION
CONFIRMATION
NEGATION
SURPRISE
CONTEXT_DEPENDENT
```

и ссылки на релевантные соседние replica IDs.

Но LLM **не управляет аудио напрямую** и не добавляет synthetic carrier
в `final_text`.

Short Utterance Strategy решает это отдельно.

### 3.6 Семантические границы

Допускается annotation смысловых групп/пауз, но production-применение
пауз включать только после отдельного теста качества.

Не менять punctuation автоматически только потому, что LLM предложила.

------------------------------------------------------------------------

## 4. Structured Output Contract

Создать строгую schema, например:

``` json
{
  "schema_version": "1",
  "project_id": "p1",
  "replica_id": 17,
  "language": "ru",
  "items": [
    {
      "span_start": 11,
      "span_end": 16,
      "source": "замок",
      "type": "homograph",
      "meaning": "lock",
      "suggested_form": "замо́к",
      "confidence": 0.98,
      "needs_review": false,
      "reason_code": "CONTEXT_DISAMBIGUATION"
    }
  ],
  "utterance": {
    "class": "NORMAL",
    "context_dependency": "LOW",
    "relevant_replica_ids": []
  }
}
```

Backend обязан валидировать: - JSON; - schema version; - project/replica
ID; - span bounds; - `source == source_text[span_start:span_end]`; -
разрешённые `type`; - confidence range; - отсутствие overlapping
conflicting patches; - отсутствие изменений вне spans.

Невалидный ответ LLM не применяется.

------------------------------------------------------------------------

## 5. Никакого свободного rewrite

Запрещён production API вида:

``` text
prepare_text_with_llm(text) -> rewritten_text
```

Предпочтительный контракт:

``` text
analyze_russian_text(context) -> LinguisticAnalysis
```

и отдельно:

``` text
validate_analysis(...)
resolve_candidates(...)
apply_confirmed_pronunciation_rules(...)
```

LLM никогда не становится владельцем `final_text`.

------------------------------------------------------------------------

## 6. Контекст

Не вызывать LLM по одному слову.

Для replica передавать: - target replica; - ограниченное число
предыдущих/следующих реплик; - speaker IDs/labels без лишних
персональных данных; - при необходимости соседний абзац; - уже
подтверждённые project dictionary entries, если они релевантны.

Не передавать весь огромный проект без необходимости.

Начальная policy:

``` text
target replica
+ up to 2 previous replicas
+ up to 2 next replicas
+ hard character/token budget
```

Для длинных сцен использовать sliding context.

------------------------------------------------------------------------

## 7. Batch analysis

Не делать отдельный LLM request на каждую короткую replica, если
несколько реплик можно безопасно проанализировать одним батчем.

Batch должен: - сохранять `replica_id`; - не смешивать spans разных
реплик; - иметь ограничение по tokens; - позволять retry только
невалидной части; - не создавать огромный KV cache.

Цель --- снизить latency и дать модели контекст.

------------------------------------------------------------------------

## 8. Prompt

Prompt должен быть ориентирован на русский язык.

Требования: - отвечать только по schema; - не исправлять стиль; - не
улучшать авторский текст; - не цензурировать; - не перефразировать; - не
менять punctuation без отдельного annotation; - не придумывать
произношение при низкой уверенности; - явно использовать
`needs_review`; - различать «не знаю» и «ошибки нет»; - использовать
соседний контекст только для анализа target.

Prompt хранить versioned:

``` text
prompts/russian_linguistic_analyzer.v1.txt
```

Prompt version сохранять вместе с analysis result.

------------------------------------------------------------------------

## 9. Confidence policy

LLM confidence не считать истинной вероятностью.

Начальная production policy:

``` text
любой LLM candidate → visible/auditable
```

Автоматическое применение разрешать только для категорий, отдельно
подтверждённых benchmark/regression suite.

Для первого релиза: - омографы → review либо deterministic
cross-check; - неизвестные имена/топонимы → review; - Global Dictionary
→ всегда explicit user action; - conflict с deterministic resolver →
review; - high confidence не отменяет validation.

------------------------------------------------------------------------

## 10. Cross-check с существующими инструментами

Для `ё`, dictionary и RUAccent сохранять существующий deterministic
pipeline.

Полезный режим:

``` text
deterministic result == LLM suggestion
    → agreement

deterministic result != LLM suggestion
    → conflict / needs review
```

Не давать LLM напрямую генерировать F5-specific `+`.

Engine-specific markup формируется только существующим engine
adapter/RUAccent layer.

XTTS не должен получать F5 `+`.

------------------------------------------------------------------------

## 11. Persistence

Сохранять analysis отдельно от текста.

Минимально:

``` text
analysis_id
project_id
replica_id
source_text_hash
context_hash
model_tag
model_digest
prompt_version
schema_version
analysis_json
status
created_at
updated_at
```

Не считать analysis действительным, если изменились: - source replica; -
релевантный context; - project dictionary entry, влияющая на replica; -
analyzer prompt/schema; - выбранная model/digest; - preprocessing
version, если это меняет вход Analyzer.

------------------------------------------------------------------------

## 12. Analysis states

Интегрировать с существующим project preparation state.

Внутренний LLM substate:

``` text
DISABLED
PENDING
RUNNING
READY
NEEDS_REVIEW
FAILED
STALE
```

LLM failure не должен уничтожать проект.

Если LLM optional: - пользователь может повторить; - при разрешённом
режиме можно продолжить deterministic-only; - UI обязан явно показать,
что LLM analysis не выполнен.

Не делать silent fallback, который выглядит как успешный LLM analysis.

------------------------------------------------------------------------

## 13. Ollama adapter

Создать отдельный слой, например:

``` text
OllamaClient
LinguisticAnalyzer
LinguisticAnalysisValidator
```

Не размазывать HTTP calls по routes/UI.

Поддержать: - health; - model availability; - analyze; - timeout; -
cancellation; - unload/keep-alive policy; - structured output; - one
bounded retry для malformed JSON; - diagnostics.

Backend должен переживать недоступность Ollama.

------------------------------------------------------------------------

## 14. Memory Policy --- M3 / 18 GB

Это обязательная часть реализации.

### 14.1 Лимиты

Использовать единый Memory Pressure Guard с benchmark-задачей:

``` text
NORMAL target: < 70%
WARNING:       >= 75%
CRITICAL:      >= 82%
HARD STOP:     >= 88%
```

Оставлять ориентировочно **3.5--4 GB системного резерва**.

Порог должен учитывать не только RSS Python, но и system memory
pressure/unified memory.

Если macOS pressure хуже, чем предполагает процент, pressure state имеет
приоритет.

### 14.2 Запрет тяжёлого параллелизма

На машине 18 GB не разрешать одновременно активный inference:

``` text
LLM + F5 TTS
LLM + XTTS
LLM + XTTS Banana
LLM + Whisper
LLM + другая LLM
```

по умолчанию.

Не путать «модель загружена» и «идёт inference»: после реальных
измерений допустим более мягкий режим только если memory guard
доказывает безопасность.

Первый production release должен быть консервативным: **heavy ML
scheduler = 1 heavy inference at a time**.

Лёгкие API/UI/SQLite/preprocessing должны оставаться отзывчивыми.

### 14.3 Анализ до TTS

Предпочтительный lifecycle:

``` text
load/analyze source
↓
Ollama inference
↓
persist analysis
↓
review
↓
release/unload LLM when appropriate
↓
load/run TTS
```

Не держать LLM резидентной без причины во время длинного render.

### 14.4 Context

Production default:

``` text
num_ctx = 8192
```

Если benchmark доказал, что 4096 достаточно --- использовать 4096.

16384 разрешать только при измеренной необходимости.

Не использовать 40K/128K/256K по умолчанию.

### 14.5 Queue

LLM analysis и TTS/Whisper должны координироваться через общий
heavy-resource gate или эквивалентный scheduler.

Нельзя создавать независимые очереди, которые одновременно считают себя
вправе загрузить всю unified memory.

------------------------------------------------------------------------

## 15. UI

В Dialogue и Continuous Text добавить единый блок:

``` text
Лингвистический анализ: выкл / ожидает / анализ / готов / требует проверки / ошибка
Модель: ...
```

Показать категории: - омографы; - `ё`; - имена/термины; -
аббревиатуры; - иностранные слова; - прочие pronunciation candidates.

Для каждого candidate:

``` text
source
context snippet
suggestion
reason/type
confidence
deterministic agreement/conflict
```

Actions: - принять для проекта; - принять глобально; - изменить; -
оставить как есть; - отклонить.

Не показывать пользователю chain-of-thought. Хранить только краткие
reason codes/объяснение, предназначенное schema.

------------------------------------------------------------------------

## 16. API

Логически нужны операции:

``` text
GET  /api/llm/status
GET  /api/llm/models
POST /api/projects/{project_id}/linguistic-analysis
GET  /api/projects/{project_id}/linguistic-analysis
POST /api/projects/{project_id}/linguistic-analysis/review
```

Названия адаптировать к текущей API convention.

`POST .../render` не должен сам скрытно запускать LLM, если проект
требует review.

------------------------------------------------------------------------

## 17. Fallback

Обязательный сценарий:

``` text
Ollama unavailable
↓
existing deterministic preprocessing remains operational
```

Если проект настроен на `LLM required before render`, UI показывает
блокирующую причину.

Если режим `LLM optional`, пользователь может явно выбрать
deterministic-only.

Никакого silent semantic downgrade.

------------------------------------------------------------------------

## 18. Cache

Кэшировать analysis по:

``` text
source_text_hash
context_hash
model_digest
prompt_version
schema_version
relevant dictionary version
```

Одинаковый неизменившийся текст не должен повторно гоняться через LLM.

Изменение только одной replica должно инвалидировать её и минимально
необходимый context neighborhood, а не весь проект, если архитектура
позволяет.

------------------------------------------------------------------------

## 19. Privacy

По умолчанию Analyzer локальный.

Не отправлять текст во внешние cloud API.

В README/UI явно указать: - Ollama работает локально; - выбранная модель
локальная; - network нужен для первоначального скачивания моделей; -
production analysis не требует cloud LLM.

------------------------------------------------------------------------

## 20. Unit tests --- без реальной LLM

Обычный pytest использует fake Ollama client.

Обязательные тесты:

``` text
test_llm_disabled_keeps_existing_pipeline_working
test_ollama_unavailable_does_not_crash_backend
test_analyzer_uses_selected_model
test_analyzer_sends_russian_context
test_analyzer_returns_annotations_not_rewritten_text

test_analysis_schema_validation
test_analysis_rejects_invalid_span
test_analysis_rejects_source_mismatch
test_analysis_rejects_unknown_type
test_analysis_rejects_conflicting_overlapping_patches
test_malformed_json_is_not_applied
test_malformed_json_retry_is_bounded

test_homograph_candidate_created
test_yo_conflict_requires_review
test_unknown_name_can_require_review
test_llm_cannot_write_global_dictionary_automatically
test_project_dictionary_requires_review_action
test_project_dictionary_overrides_global

test_llm_never_emits_f5_markup_into_xtts_final
test_f5_markup_remains_engine_adapter_responsibility
test_preview_and_render_use_same_confirmed_result

test_analysis_is_cached_for_unchanged_replica
test_source_change_invalidates_analysis
test_context_change_invalidates_dependent_analysis
test_model_digest_change_invalidates_analysis
test_prompt_version_change_invalidates_analysis
test_dictionary_change_invalidates_affected_analysis

test_memory_warning_blocks_new_llm_inference
test_memory_critical_blocks_heavy_dispatch
test_hard_stop_prevents_llm_start
test_llm_and_tts_do_not_infer_concurrently
test_llm_and_whisper_do_not_infer_concurrently
test_light_api_works_while_llm_analysis_runs

test_llm_unload_releases_resource_slot
test_llm_timeout_releases_resource_slot
test_llm_failure_releases_resource_slot
test_cancelled_llm_job_releases_resource_slot

test_short_replica_receives_context_hint
test_different_speaker_context_does_not_modify_target
test_llm_analysis_does_not_modify_source_text
test_llm_analysis_does_not_modify_final_text_before_review
```

------------------------------------------------------------------------

## 21. Integration tests

С fake/stub LLM + StubEngine проверить полный flow:

``` text
project
→ parse
→ deterministic preprocessing
→ LLM analysis
→ review
→ dictionary apply
→ engine-specific finalization
→ READY
→ render
```

Обязательные:

``` text
test_dialogue_full_flow_with_llm_review
test_continuous_text_full_flow_with_llm_review
test_render_blocked_when_required_llm_review_pending
test_render_allowed_after_llm_review
test_deterministic_only_flow_when_llm_optional
test_render_engine_receives_confirmed_final_text
```

------------------------------------------------------------------------

## 22. Real smoke tests

После unit/integration tests провести реальные smoke tests с выбранной
primary model.

Набор минимум:

``` text
20 homograph cases
20 ё cases
10 names/toponyms
10 abbreviations/numbers
10 dialogue-context cases
10 short replicas
```

Проверить: - JSON validity; - latency; - memory; - no UI freeze; - no
simultaneous heavy inference; - correct invalidation; - correct
review; - F5 final; - XTTS final.

Затем один реальный F5 render после LLM analysis.

XTTS smoke --- если XTTS доступен.

------------------------------------------------------------------------

## 23. Phase plan

### Phase 1 --- Architecture & adapter

Сначала нарисовать фактический call graph текущего preprocessing и heavy
ML lifecycle.

Реализовать Ollama adapter + fake client + health/model status.

**Tests:** adapter/health/failure/timeout.

**Gate:** весь существующий pytest зелёный.

### Phase 2 --- Schema & validator

Реализовать structured annotations и строгую валидацию spans/source.

**Tests:** schema, malformed JSON, span/source/conflict tests.

**Gate:** LLM не может изменить текст вне разрешённых annotations.

### Phase 3 --- Persistence/cache/invalidation

Добавить storage без destructive DB reset.

**Tests:** cache + все invalidation cases.

**Gate:** restart приложения сохраняет valid analysis.

### Phase 4 --- Memory scheduler

Объединить LLM с heavy-resource policy TTS/Whisper.

**Tests:** mocked pressure states + concurrency guards.

**Gate:** heavy inference concurrency по умолчанию = 1.

### Phase 5 --- Russian linguistic integration

Омографы, `ё`, unknown/proper words, abbreviations.

**Tests:** category-specific fixtures.

**Gate:** candidates появляются, но не мутируют source/final silently.

### Phase 6 --- Review + dictionaries

Интегрировать Project/Global Dictionary review.

**Tests:** precedence, explicit actions, invalidation.

**Gate:** Global Dictionary никогда не меняется без explicit action.

### Phase 7 --- Dialogue/Continuous integration

Обе вкладки используют один backend Analyzer.

**Tests:** full-flow integration.

**Gate:** никакой дублированной frontend linguistic logic.

### Phase 8 --- Short utterance hints

Передавать context hints в Short Utterance subsystem, не управляя аудио
напрямую.

**Tests:** target preservation/context IDs.

**Gate:** source/final text не меняются.

### Phase 9 --- Real smoke + memory validation

Реальная primary LLM + F5; XTTS при наличии.

**Gate:** Mac остаётся отзывчивым, memory thresholds соблюдаются.

### Phase 10 --- Documentation / Knowledge Base

Обновить все документы.

**Gate:** документация соответствует фактическому коду/config/API/model
digest.

------------------------------------------------------------------------

## 24. Acceptance Criteria

Задача завершена только если:

1.  Используется победитель benchmark, а не случайно выбранная модель.
2.  Ollama optional и локальный.
3.  LLM возвращает annotations, не rewrite.
4.  Все spans валидируются.
5.  source text неизменяем.
6.  final_text формируется существующим deterministic/engine-specific
    pipeline.
7.  Омографы анализируются с контекстом.
8.  `ё` cross-check работает.
9.  Unknown/proper words попадают в review.
10. Global Dictionary не изменяется автоматически.
11. Project Dictionary имеет приоритет.
12. F5 markup остаётся engine-specific.
13. XTTS не получает F5 `+`.
14. Cache/invalidation корректны.
15. LLM/TTS/Whisper не делают тяжёлый inference параллельно по
    умолчанию.
16. WARNING/CRITICAL/HARD STOP реально блокируют новый heavy dispatch.
17. Light API остаётся отзывчивым.
18. Ollama failure не убивает backend.
19. Dialogue и Continuous используют общий Analyzer.
20. Реальные smoke tests пройдены.
21. Все существующие + новые tests зелёные.
22. README и база знаний обновлены.

------------------------------------------------------------------------

## 25. Обновление документации и базы знаний --- обязательно

Обновить:

``` text
README.md
```

Добавить: - Local LLM Analyzer; - как включить/выключить; - Ollama
setup; - фактически выбранную primary/fallback model; - model digest; -
memory policy; - ограничения; - troubleshooting; - deterministic-only
fallback.

Создать/обновить:

``` text
docs/knowledge/local-llm-russian-analyzer.md
docs/architecture/ollama-memory-policy.md
docs/architecture/text-preprocessing-pipeline.md
```

`text-preprocessing-pipeline.md` должен показывать единственный
актуальный pipeline от source до `engine.synthesize()` для: -
Dialogue; - Continuous Text; - F5; - XTTS; - LLM enabled; - LLM
disabled.

Добавить API documentation для новых endpoints.

Если проект уже имеет существующую структуру knowledge base ---
использовать её, не создавать параллельную.

Обновить changelog/roadmap, если такие файлы существуют.

------------------------------------------------------------------------

## 26. Финальный отчёт ИИ-агента

В конце обязательно предоставить:

``` text
architecture before/after
benchmark winner actually used
model tag + digest
Ollama version
changed files
DB migrations
new config
new API endpoints
prompt/schema versions
memory thresholds
measured peak memory
measured latency
cache/invalidation behavior
dictionary/review behavior
F5 behavior
XTTS behavior
short-utterance integration
unit tests + result
integration tests + result
real smoke tests + result
documentation updated
known limitations
```

Если реальные измерения показывают, что выбранная LLM вместе с текущим
TTS stack слишком тяжела для 18 GB, не ослаблять memory guard ради
прохождения задачи. Переключить production primary на benchmark-approved
fallback и задокументировать решение.

------------------------------------------------------------------------

## 27. Не делать

Запрещено:

-   позволять LLM свободно переписывать пользовательский текст;
-   автоматически добавлять все LLM suggestions в Global Dictionary;
-   использовать LLM вместо RUAccent без benchmark-доказательства;
-   отправлять F5 `+` в XTTS;
-   запускать LLM/TTS/Whisper тяжёлый inference параллельно на 18 GB по
    умолчанию;
-   использовать 40K+ context без необходимости;
-   держать несколько Ollama-моделей одновременно;
-   скрытно fallback'иться после LLM error;
-   ломать существующий deterministic-only режим;
-   делать destructive DB reset;
-   запускать реальные модели в обычном pytest;
-   считать confidence LLM истинной вероятностью;
-   объявлять задачу выполненной без реального memory smoke test.

Главная цель: **LLM добавляет контекстное понимание русского языка, а
существующий детерминированный pipeline сохраняет контроль,
воспроизводимость и безопасность текста перед синтезом.**
