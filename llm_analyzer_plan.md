# План встраивания LLM-анализатора русского текста (Task 2)

Документ фиксирует **фактический** call graph проекта и то, куда именно встраивается
локальный анализатор. Написан до кода интеграции: постановка Task 2 требует сначала
описать реальный pipeline, а потом адаптировать встраивание без дублирования логики.
Ссылки — `путь:строка` на момент написания (2026-09-18, после Task 1).

## 1. Pipeline «текст → движок», как он есть

```text
SOURCE TEXT
  │  dialogue_parser.parse_dialogue (dialogue_parser.py:249) → Replica (dialogue_parser.py:66)
  ▼
Проект в SQLite: ProjectsStore.parse_project (db/store.py:204)
  │  реплики: replicas (миграция 1), текст в replicas.text
  ▼
Обязательная подготовка: POST /api/projects/{id}/analyze (main.py:2339)
  │  main._analyze_project (main.py:2299)
  │  └── project_analysis.prepare_replica (project_analysis.py:151)
  │        ├── audio_pipeline.preview_text (audio_pipeline.py:663)
  │        │     └── text_normalization.normalize_stages (pipeline.py:229)
  │        │           numbers → dates → time → abbreviations → units → money →
  │        │           «ё» (pipeline.py:169 → yo_restoration.restore_yo) →
  │        │           словарь (pronunciation.apply_pronunciation_report) → latin
  │        │           + ударения только при auto_accent и supports_accents
  │        │             (audio_pipeline.py:696, accentizer.accentuate)
  │        └── pronunciation_suggest.build_suggestions (pronunciation_suggest.py:193)
  │  ProjectsStore.save_analysis (db/store.py:372) → replicas.py:215
  │     сохраняет: normalized_text, yo_text, dictionary_text, accentized_text,
  │     final_text, analysis_status, dictionary_matches, pronunciation_candidates,
  │     supports_accents, auto_accent, effective_engine_params
  ▼
Review словаря: POST /api/projects/{id}/pronunciation/review (main.py:1182)
  │  Project Dictionary (project_pronunciation_entries) имеет приоритет над
  │  Global (pronunciation_entries): pronunciation.rules_for (pronunciation.py:88)
  ▼
Рендер: POST /api/projects/{id}/render (main.py:2447) → job_queue
  │  guard: settings.require_prepared (audio_pipeline.py:1494-1507)
  │  текст для движка: audio_pipeline._text_for_engine (audio_pipeline.py:711)
  │     final_text из реплики важнее пересчёта (audio_pipeline.py:1228-1234)
  ▼
engine.synthesize(text=..., ref_audio_path=..., ref_text=..., speed=..., **params)
      (engines/base.py:388; F5: f5_engine.py:61, XTTS: xtts_engine.py:182)
  ▼
Short Utterance Strategy (audio_pipeline.py:1512 → short_utterance.build_contexts)
```

Контракты, которые нельзя нарушать:

* `final_text` принадлежит детерминированному слою и engine adapter'у; LLM его не
  пишет. Ударения (`+`) — только у движков с `EngineInfo.supports_accents` (F5), и
  знак снимается для XTTS в `text_normalization/pronunciation.py:118-121`.
* Словарь — единственная точка слияния `_rules_for_project` (main.py:2222);
  проектные правила идут раньше глобальных (`pronunciation.py:113-117`).
* Рендер без подготовки запрещён guard'ом `require_prepared`; статус подготовки
  лежит в `projects.analysis_status` (config.PROJECT_ANALYSIS_*).

## 2. Куда встраивается LLM (после «ё», до review)

```text
… детерминированная нормализация и «ё» (как есть)
  ▼
LLM ANALYSIS  ← новая стадия: только аннотации (backend/llm/analyzer.py)
  │   вход: source-текст реплики + ограниченный контекст (≤2 соседа, бюджет символов)
  │   выход: аннотации + подсказка о реплике; НИКАКОГО текста
  ▼
Сверка с детерминированным слоем (§10 Task 2)
  │   согласие / конфликт → конфликт уходит в review
  ▼
Review: существующие pronunciation_candidates + LLM-кандидаты
  │   принять в проект / глобально / отклонить (main.py:1182 — образец)
  ▼
Проектный словарь → Global Dictionary (как сейчас)
  ▼
final_text → engine adapter → Short Utterance → TTS
```

Почему именно здесь: LLM нужен контекст уже нормализованного текста (числа и
сокращения раскрыты, «ё» восстановлена), но решение по словарю принимает человек, а
ударения ставит RUAccent. Раньше — анализ работал бы по сырым числам и латинице,
позже — не успевал бы повлиять на `final_text` до рендера.

## 3. Тяжёлые ресурсы: что уже есть и чего не хватает

* Единственный воркер очереди (`job_queue.py:732`) сериализует **TTS и Whisper** между
  собой; перед задачей он ждёт память (`_wait_for_memory` job_queue.py:748 →
  `resource_guard.is_memory_critical` → `memory_monitor.classify`).
* `HeavyGate` (`llm/memory_policy.py:388`) — общий gate для LLM/TTS/Whisper — уже
  написан в Task 1, но подключён только к benchmark-runner'у (`runner.py:354`).
  **Задача фазы 4 — подключить его к очереди**: анализ берёт `hold(HEAVY_LLM)`, задача
  синтеза — `hold(HEAVY_TTS)`, Whisper — `hold(HEAVY_WHISPER)`; `external_busy` для
  процесса дашборда = «очередь выполняет тяжёлую задачу».
* Политика памяти LLM (`memory_policy.decide`, пороги 70/75/82/88, резерв 3.5–4 GB)
  решает про запуск анализа; измеренный след `qwen3:8b` — 6.17 GiB, пик 86 % (Task 1),
  поэтому в production рядом с синтезом резерв остаётся 3.5–4 GB, а при нехватке
  используется fallback-модель (решение benchmark'а).

## 4. Хранение: что добавит фаза 3

Миграция 7 (последняя сейчас — 6, `migrations.py:185`): таблица
`llm_analyses` с полями из §11 Task 2 — `analysis_id, project_id, replica_id,
source_text_hash, context_hash, model_tag, model_digest, prompt_version,
schema_version, analysis_json, status, created_at, updated_at` (+ индекс по
`(project_id, replica_id)`). Существующие таблицы и поля подготовки не трогаются:
destructive reset запрещён, старые базы обязаны обновляться миграцией.

Инвалидация опирается на уже существующие хеши/версии: смена исходного текста,
контекста, правила словаря, версии prompt/schema или digest модели делает разбор
устаревшим (`STALE`), а не удаляет его: история разборов нужна для аудита.

## 5. Фазы Task 2 в терминах этого кода

| Фаза | Что делаем | Точки в коде |
|---|---|---|
| 1 | Call graph (этот документ) + адаптер/статус | `backend/llm/analyzer.py`, `main.py:499,511` — сделано |
| 2 | Схема и валидатор | `schemas.parse_analysis(check_spans=…)`, `analyzer.locate_annotations` |
| 3 | Persistence/cache/invalidation | миграция 7, `db/repositories/llm_analyses.py`, `db/store.py` |
| 4 | Memory scheduler | `memory_policy.HeavyGate` + `job_queue.py:732`, `transcribe.py` |
| 5 | Русская лингвистика | `analyzer` → кандидаты в `pronunciation_candidates` |
| 6 | Review + словари | `main.py:1182` (образец), `pronunciation.py`, `_invalidate_for_rule` |
| 7 | Dialogue + Continuous | `main.py:2299 _analyze_project`, `frontend/app.js:2455,2526` |
| 8 | Short utterance hints | `short_utterance.py:228,303`, `audio_pipeline.py:1512` |
| 9 | Реальный smoke | primary `qwen3:8b`, F5 (+ XTTS, если доступен) |
| 10 | Документация | `README.md`, `llm_russian_analyzer.md`, `ollama_memory_policy.md` |

## 6. Правила, которые проверяются тестами

1. LLM не пишет текст: в разборе нет поля с текстом, лишнее поле — отказ (`unknown_field`).
2. Границы ищет backend: `source` модели сверяется с текстом, неверный `source`
   отбрасывается с причиной, неверные `span` исправляются (`locate_annotations`).
3. Ответ применяется целиком или не применяется: мусор вместо JSON → один повтор →
   `FAILED` с причиной.
4. Отказ виден: недоступная Ollama, таймаут, отсутствующая модель — `FAILED` + причина,
   никогда не «тихий» переход в детерминированный режим под видом успеха.
5. Разметка `+` не приходит от LLM: знак снимается, факт виден в разборе (`markup_stripped`).
6. Тяжёлый inference один: `HeavyGate` для LLM/TTS/Whisper, проверяется тестами
   конкурентности, а не только документацией.
