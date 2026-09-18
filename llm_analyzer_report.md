# Local LLM Russian Linguistic Analyzer: финальный отчёт (Task 2)

Отчёт по §26 постановки. Числа — из живого прогона
(`output/llm-smoke/smoke-*.json`, инструмент `tools/llm_analyzer_smoke.py`) и из
автоматических тестов. Разбор и ограничения — в `llm_russian_analyzer.md`, политика
памяти — в `ollama_memory_policy.md`, benchmark — в `llm_benchmark_report.md`.

Дата: 2026-09-18.

## 1. Архитектура: было и стало

**Было:** единственный детерминированный проход подготовки реплик
(`project_analysis.prepare_replica` → `audio_pipeline.preview_text`: числа, даты,
сокращения, «ё», словарь, ударения) и review кандидатов словаря.

**Стало:** в тот же проход добавлена стадия LLM-анализа — до детерминированной
подготовки по реплике, но в одном каноническом проходе:

```text
реплика + ограниченный контекст (≤2 соседа, бюджет символов)
   → Ollama (format=json, схема текстом в prompt, think выключен у thinking-моделей)
   → строгая валидация структуры (parse_analysis, check_spans=False, check_reasons=False)
   → локализация границ backend'ом (source ищется в тексте, выдуманное отбрасывается)
   → кандидаты словаря с источником llm и сверкой с детерминированным слоем
   → review человеком → правило словаря → final_text → engine adapter → Short Utterance
```

Ни одного второго pipeline, ни одной точки, где модель пишет текст.

## 2. Выбранная модель

| | |
|---|---|
| Production primary (тихая машина) | `qwen3:8b`, digest `500a1f067a9f` |
| Fallback и практичный primary рядом с дашбордом | `qwen3:4b-instruct-2507-q8_0` (`aa7252f68dda`) / `-q4_K_M` (`0edcdef34593`) |
| Живой smoke выполнен на | `qwen3:4b-instruct-2507-q4_K_M`, num_ctx 4096 |
| Ollama | 0.34.1 (HTTP API, `127.0.0.1:11434`) |
| prompt / schema | версия 2 / версия 1 |

Почему в smoke не primary: дашборд прогревает F5 (~3–4 GB), и вместе с
пользовательскими приложениями на 18 GB остаётся 2–4 GB при жёлтом давлении macOS —
под это не проходит ни одна модель. Постановка (§26) прямо запрещает ослаблять
memory guard ради прохождения задачи и предписывает переключить production primary
на benchmark-approved fallback; это и сделано, с замерами в
`ollama_memory_policy.md` §4.1.

## 3. Живой smoke (§22)

Набор из gold-корпуса: 20 омографов, 20 «ё», 5 имён, 5 топонимов, 5 аббревиатур,
5 чисел, 10 диалоговых кейсов, 10 коротких реплик, 10 чистых текстов — 90 кейсов,
110 реплик с контекстом.

| Показатель | Значение |
|---|---|
| Реплик проанализировано | 110 из 110 |
| Вызовов модели | 110 (кеш пуст в первом проходе) |
| Время прохода | 344 с (≈3.1 с на реплику) |
| Предложений | 85 |
| Требуют решения человека | 8 |
| Конфликтов с детерминированным слоем | 0 |
| Неудачных разборов | 0 |
| Пик памяти | 78.2 % |
| Минимум свободной памяти | 3.93 GB |
| Модель выгружена после прохода | да |
| Исходный текст изменён | нет (`sources_intact: true`) |
| `final_text` изменён моделью | нет (0 из 110) |

Первый прогон дал 12 неудачных разборов из 110 — все с одной причиной:
`unknown_reason` (модель выдумала код причины). Проверка кода причины сделана
справочной для Analyzer'а (в benchmark строгость осталась: там она измеряется),
после чего повторный прогон дал **0 неудачных разборов из 110**. Это ровно тот
случай, ради которого живой smoke и нужен: на подставной модели такой отказ не
воспроизводится.

Реальный рендер F5 после анализа: маленький проект из 4 чистых реплик, статус
`done` за **86 с**, у всех 4 реплик есть take, `final_text` непустой. Путь
«review → рендер» проверен интеграционными тестами на подставной модели
(`test_render_allowed_after_llm_review`, `test_render_blocked_when_required_llm_review_pending`).

## 4. Что проверено тестами

- **115 новых тестов** Task 2 (всего в проекте 1029, все зелёные):
  `tests/test_llm_analyzer.py` (22), `tests/test_llm_analysis_cache.py` (17),
  `tests/test_llm_scheduler.py` (16), `tests/test_llm_project_analysis.py` (21) плюс
  инфраструктурные и benchmark-тесты Task 1 (53).
- Схема и валидатор: границы, `source`, типы, уверенность, перекрытия, версия схемы,
  лишние поля, битый JSON, ограниченный повтор.
- Инварианты текста: прогон с LLM и без LLM даёт идентичные `text` и `final_text`.
- Кеш: попадание при неизменном входе и инвалидация по каждому изменению (текст,
  контекст, словарь, digest модели, версии prompt/схемы).
- Память: WARNING/CRITICAL/HARD STOP, приоритет macOS pressure, один тяжёлый слот,
  остановка длинного прохода по HARD STOP, освобождение слота при ошибке/таймауте/отмене.
- Review и словари: приоритет Project над Global, явный `scope: global`, отклонение
  как память о решении, конфликт требует человека.
- Интеграция: диалог и сплошной текст используют один Analyzer; рендер ждёт review в
  режиме «LLM обязателен».

## 5. База данных и конфигурация

- Миграция 7: `llm_analyses` (разбор на реплику, хеши входа, статус, модель, ошибка).
- Миграция 8: `projects.llm_analysis_status/model/error/updated_at` — отдельная ось
  состояния анализа. Обе — аддитивные, без destructive reset: старые базы
  обновляются, данные сохраняются (проверено тестом миграции).
- Конфигурация (`LLM_*`, дублируется приставкой `TTS_`): `LLM_ANALYZER_ENABLED`,
  `LLM_PRIMARY_MODEL`, `LLM_FALLBACK_MODEL`, `LLM_ANALYZER_NUM_CTX`,
  `LLM_ANALYZER_TEMPERATURE`, `LLM_ANALYZER_SEED`, `LLM_ANALYZER_TIMEOUT_SEC`,
  `LLM_ANALYZER_MAX_REPAIRS`, `LLM_ANALYZER_CONTEXT_REPLICAS`, `LLM_ANALYZER_CONTEXT_CHARS`,
  `LLM_ANALYZER_RESPONSE_FORMAT`, `LLM_ANALYZER_THINK`, `LLM_ANALYZER_REQUIRED_FOR_RENDER`;
  память — `LLM_MEMORY_*` (Task 1), Ollama — `TTS_OLLAMA_*`.

## 6. API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/llm/status` | готовность анализатора, выбранная модель, версии |
| `GET` | `/api/llm/models` | скачанные модели и роли primary/fallback |
| `GET` | `/api/projects/{id}/linguistic-analysis` | состояние, модель, причины отказов, кандидаты, отчёт последнего прохода |
| `POST` | `/api/projects/{id}/linguistic-analysis` | запуск (тот же канонический проход, что `/analyze`) |
| `POST` | `/api/projects/{id}/pronunciation/review` | решения по предложениям (существующая ручка) |

## 7. Поведение по движкам

- **F5** получает подтверждённый `final_text` с разметкой `+` от RUAccent и словаря;
  модель в эту разметку не вмешивается (её `+` снимается).
- **XTTS** не получает `+` — как и раньше, знак снимается в слое словаря по
  `supports_accents`; LLM-предложения на это не влияют.
- Рендер берёт сохранённый `final_text`, а не пересчитывает текст заново
  (тест `test_render_uses_saved_final_text`).

## 8. Известные ограничения

1. Качество анализа — модельное: `issue_f1` выбранной модели 0.19 на gold-корпусе,
   модели почти не считают границы точно (это компенсировано поиском слова на
   backend'е, но не устранено).
2. `reason_code` выбранной 4B-модели иногда выдуман; в Analyzer'е он справочный, в
   benchmark — строгий (метрика меряет следование контракту).
3. Массовый review дорог: каждое решение пересчитывает затронутые реплики вместе с
   LLM-разбором.
4. Рядом с прогретым F5 на 18 GB при пользовательской загрузке анализ недоступен —
   это осознанный отказ guard'а, а не дефект; рабочий режим — выгрузить движок,
   проанализировать, отрендерить.
5. Класс реплики от модели — подсказка; аудио и стратегию решает Short Utterance.
6. У сплошного текста нет проекта, поэтому решения по его предложениям идут через
   общий словарь (или через вкладку диалога): ручка review привязана к проекту.
   Анализ при этом общий — один Analyzer на обе вкладки.

## 9. Изменённые файлы (Task 2)

`backend/llm/analyzer.py`, `backend/llm/integration.py`, `backend/llm/analysis_cache.py`,
`backend/llm/scheduler.py`, `backend/llm/schemas.py`, `backend/llm/prompt.py`,
`backend/llm/versioning.py` (prompt v2), `backend/db/migrations.py`,
`backend/db/repositories/llm_analyses.py`, `backend/db/repositories/projects.py`,
`backend/db/store.py`, `backend/main.py`, `backend/job_queue.py`,
`backend/audio_pipeline.py`, `backend/short_utterance.py`, `frontend/app.js`,
`frontend/index.html`, `tools/llm_analyzer_smoke.py`, `tests/test_llm_analyzer.py`,
`tests/test_llm_analysis_cache.py`, `tests/test_llm_scheduler.py`,
`tests/test_llm_project_analysis.py`, `README.md`, `llm_analyzer_plan.md`,
`llm_russian_analyzer.md`, `ollama_memory_policy.md`.
