# UPDATE 3 — Phase 0: аудит, карта компонентов и контракты

Дата: 2026-09-18. Основание: `update_3.md` §1 (обязательный аудит до кода), §80
(Phase 0 — Audit & Contracts). Gate фазы: **нет дублирующей архитектуры**.

Главный вывод аудита: **UPDATE 2 уже реализовал ядро того, что требует UPDATE 3.**
Отдельный `EmotionAnalyzer`, второй LLM-pipeline, вторая сущность референса и
второй резолвер создавать **нельзя** — они уже есть. Работа сводится к расширению
существующих подсистем.

---

## 1. Карта компонентов: reuse / extend / migrate / deprecate

| # | §1 | Компонент (факт) | Решение |
|---|---|---|---|
| 1 | Local LLM Russian Linguistic Analyzer | `backend/llm/analyzer.py` (`LinguisticAnalyzer`, `AnalyzerSettings`, `settings_from_env`, `build_context`, `ReplicaAnalysis`), `llm/schemas.py`, `llm/prompt.py`, `llm/__init__.py` | **extend** (schema + prompt + scene batching) |
| 2 | Ollama client/runner | `backend/llm/ollama_client.py` (health/models/chat/unload/capabilities), `llm/runner.py` (benchmark) | **reuse as-is** |
| 3 | LLM schema | `llm/schemas.py`: `SCHEMA_VERSION="2"`, `analysis_json_schema()`, `ANNOTATION_TYPES`, `UTTERANCE_CLASSES`, `CONTEXT_DEPENDENCY`, `EMOTION_VALUES`, `DIALOGUE_ACT_MAX_CHARS=40`, `parse_analysis`, `ERROR_*` | **extend** (новые поля, `SCHEMA_VERSION=3`) |
| 4 | Prompt versioning | `llm/versioning.py` (`PROMPT_VERSION="3"`, `BENCHMARK_VERSION="1"`), `benchmarks/.../prompts/analyzer.{v1,v2,v3}.txt` | **extend** (v4 + поднять `PROMPT_VERSION`) |
| 5 | LLM cache/invalidation | `llm/analysis_cache.py`: ключ из 6 полей (`source_text_hash`, `context_hash`, `dictionary_hash`, `model_digest`, `prompt_version`, `schema_version`), `mark_stale` ±2 соседа, таблица `llm_analyses` (миграция 7) | **reuse as-is** (новые версии инвалидируют автоматически) |
| 6 | memory policy / HeavyGate | `llm/memory_policy.py` (4 уровня, `decide`, `must_abort_running`, `HeavyGate`/`try_acquire`), `llm/scheduler.py`, `job_queue.py:751` | **reuse as-is** (§49: второго планировщика не создавать) |
| 7 | Project / Replica persistence | `db/migrations.py` (9 миграций), `db/repositories/{projects,replicas,takes}.py` | **extend** (миграция 10: prosody-поля) |
| 8 | Dialogue parser | `backend/dialogue_parser.py` | **reuse as-is** |
| 9 | speaker mapping | `_MIGRATION_1.speakers`, `projects.py:174-211`, UI `renderVoiceConfigs` | **reuse as-is** |
| 10 | common text preprocessing | `text_normalization/pipeline.py` → `audio_pipeline.preview_text` | **reuse as-is** |
| 11 | ё-restoration | `yo_restoration.restore_yo` (в `pipeline.py:169`) | **reuse as-is** |
| 12 | pronunciation dictionary | `backend/pronunciation.py`, `pronunciation_suggest.py`, `project_pronunciation_entries` | **reuse as-is** |
| 13 | RUAccent / F5 finalization | `backend/accentizer.py`, `audio_pipeline.preview_text:779`; F5 `+`-разметка, для XTTS снимается | **reuse as-is** |
| 14 | `/api/text/preview` | `main.py:1655` → `audio_pipeline.preview_text` | **extend** (блок prosody отдельно от linguistic) |
| 15 | short-utterance layer | `backend/short_utterance.py` (5 стратегий, `build_contexts`, `_nearest_same_speaker`, `SHORT_UTTERANCE_*` в `config.py`), `short_utterance_boundary.py` | **extend** (prosody как hint + compatibility policy) |
| 16 | QA | `qa_screening.py`, `audio_pipeline` (smart/strict), `transcribe.py` | **reuse as-is** |
| 17 | voices storage | `backend/voices_store.py`: `Voice` + **`ReferenceProfile`** + `profiles: list` + legacy→NEUTRAL проекция (`reference_profiles()`/`neutral_profile()`), `voices.json` | **reuse as-is** (1:N уже есть!) |
| 18 | запись `RECORD_PHRASES` | `frontend/app.js:91-137` (11 фраз), `renderRecordPhrases:2405`, `startRecording:2443`, `finishRecording:2548`, `createVoice:2332`, `addVoiceReference:1865` | **extend** (см. §4 — главный разрыв) |
| 19 | фактическое хранение WAV | **1 запись = 1 голос = 1 файл** `voices/{voice_id}{suffix}`; профили — `voices/{voice_id}-{profile_id}{suffix}`. Конкатенации нет, батч-загрузки нет | **extend** |
| 20 | regenerate/variants/takes | `job_queue._project_take:1537`, `_regenerate:1407`, `save_replica_take`/`save_render_takes`, `MAX_REPLICA_VARIANTS=3`, метаданные take (`seed/engine/parameters/qa/quality`) | **extend** (prosody в take) |
| 21 | diagnostics | `backend/diagnostics.py` (`_voice_layer`, `_project_slice`, `_takes_slice`, `_short_reading`, `_analysis_slice`) | **extend** (prosody-поля в срез) |
| 22 | UI вкладки Dialogue | `frontend/index.html:181-593` (`#replica-cards`, `#voice-configs`, `#dialogue-preview`, `#review-panel`, `#llm-panel`), `app.js`: `renderReplicaCards:3855`, `replicaCardHtml:3933`, **`replicaEmotionHtml:4024`**, `bindReplicaCards:5190` | **extend** (просодия-дропдаун + confidence + reference details) |

### Дополнительно уже существует и переиспользуется

| Требование UPDATE 3 | Факт в коде |
|---|---|
| §23–§27 ProsodyResolver (deterministic, same-voice, quality, engine compat, fallback) | `backend/reference_resolver.py`: `resolve_reference` (явный `profile_id` → эмоция → NEUTRAL), `_usable` (quality/engine), `ResolvedReference` (requested/resolved/profile_id/audio/ref_text/fallback_used/reason), `reference_status`. Same-voice гарантирован **структурно** (профили внутри `Voice`) |
| §36 replica prosody fields | `replicas`: `emotion_detected`, `emotion_confidence`, `emotion_override`, `dialogue_act`, `context_dependency`, `reference_profile_id`, `reference_emotion`, `reference_fallback_used`; `emotion_effective` вычисляется (`replicas.py:55-57`) |
| §37 override сильнее Auto | `emotions.emotion_effective(detected, override)` |
| §10/§18 ReferenceProfile data model | `voices_store.ReferenceProfile`: `id, emotion, label, audio_file, ref_text, source, quality_status, quality_note, engine_compatibility, created_at` |
| §20/§79 миграция legacy-голоса | `Voice.reference_profiles()` проецирует старый `audio_file`/`ref_text` в NEUTRAL **на чтение** — миграция данных не нужна, старый voice_id не меняется |
| §22 quality gate | `_resolve_ref_text` (Whisper + `check_ref_text_match`) → `quality_status`/`quality_note`; `_usable` отказывает `warning` |
| §4 источник prosody — не текст | prompt v3 прямо запрещает переписывать реплику и вставлять теги; `REWRITE_FIELDS` в схеме; тест `test_llm_does_not_modify_*` |
| §46 cache key с версиями | `AnalysisKey`: `prompt_version` + `schema_version` в ключе |
| §48 LLM failure → renderable | `emotions.heuristic_emotion` + `EMOTION_NEUTRAL` по умолчанию; отказ LLM не блокирует рендер |
| §73 engine-specific | F5: `+`-разметка + `nfe_step/cfg_strength/cross_fade_duration/target_rms`; XTTS: `ref_text` **игнорируется**, `temperature/repetition_penalty/top_k/top_p/speed` |

---

## 2. Разрывы (что реально надо сделать)

| # | §UPDATE 3 | Разрыв | Сложность |
|---|---|---|---|
| G1 | §16, §18–§21, Phase 3 | **11 записей `RECORD_PHRASES` не превращаются в 11 профилей одного голоса.** Сейчас: выбрал одну фразу → записал → `POST /api/voices` → новый голос. Нет ни мультизаписи, ни маппинга `label → profile_key` | высокая |
| G2 | §10, §16 | **Таксономия.** Сейчас 5 эмоций: `NEUTRAL, QUESTION, DELIGHT, SURPRISE, FEAR`. Нужно 11: `+ CALM, NEUTRAL_QUESTION, EXCLAMATION, SAD_SYMPATHETIC, IRONIC, STRICT, ENUMERATION, EXCITED`. Схема LLM (`EMOTION_VALUES`), `emotions.py`, prompt — всё ограничено пятёркой | средняя |
| G3 | §8, §11, §12, §9 | **Нет `intensity`, `pace`, `prosody.confidence` (как отдельного поля), `recommended_profile`.** `dialogue_act` есть, но это свободная строка ≤40 символов без controlled vocabulary и валидации | средняя |
| G4 | §6, §7, §77, Phase 6 | **Один LLM-запрос на реплику.** `ProjectLlmAnalyzer.run` обходит реплики по одной (`integration.py`), контекст передаётся через `context_replicas`, но батчинга «одна сцена → один запрос → `results[]` по replica_id» нет | высокая |
| G5 | §26, §17, Phase 4 | **Fallback-таблица = только NEUTRAL.** Цепочки вида `QUESTION → NEUTRAL_QUESTION → NEUTRAL` не существует; `resolve_reference` безусловно падает в NEUTRAL | низкая |
| G6 | §28, §22 | `engine_compatibility` при создании профиля ставится `[voice.engine]` — фактическая совместимость не проверяется отдельно | низкая |
| G7 | §13, §39, §40, §57, Phase 8 | **UI:** дропдаун эмоций уже есть (`replicaEmotionHtml`), но нет отдельного блока «Интонация», процента уверенности, деталей `Reference:`/`Fallback:` и блока интерпретации в preview | средняя |
| G8 | §55, §56, §59 | **Метаданные take и diagnostics** не содержат `prosody_effective`/`reference_profile_key`/`short_strategy` в явном виде (частично лежат в `parameters`) | низкая |
| G9 | §14, §51, §52 | `context_dependency` **не используется** short-planner'ом; нет compatibility policy для same-speaker контекста с разными эмоциями | средняя |
| G10 | §30–§35, §72–§74, Phases 1, 11 | **Reference Prosody Transfer Benchmark, listening review, real dialogue benchmark** — инфраструктуры нет | высокая (нужны реальные модели + прослушивание) |
| G11 | §47 | Валидация LLM-ответа: `dialogue_act` не сверяется со словарём, `intensity`/`pace` не существуют | низкая |
| G12 | §84–§89, Phase 13 | Документация/Knowledge Base по dialogue analyzer, reference profiles, prosody, benchmark | средняя |

---

## 3. Контракты (целевые)

### 3.1 Резолвер (`reference_resolver.py`) — расширяется, не переписывается

```python
resolve_prosody(voice, engine_id, prosody_effective, dialogue_act="", intensity=None)
    -> ResolvedProsody(requested_profile, resolved_profile, profile_id,
                       audio_path, ref_text, fallback_used, fallback_reason,
                       voice_id, engine_id)
```

Обратная совместимость: существующая `resolve_reference(voice, engine_id, emotion, *, profile_id)`
остаётся точкой входа; `resolve_prosody` — надстройка, добавляющая fallback-цепочку.
Инвариант `resolved.voice_id == replica.voice_id` обеспечен структурно.

### 3.2 Схема LLM — расширяется до v3

```jsonc
// utterance (существующее) + новое
{
  "class": "NORMAL",                       // как есть
  "context_dependency": "HIGH",            // как есть
  "relevant_replica_ids": [11, 13],        // как есть
  "emotion": "QUESTION",                   // как есть (расширяется словарь)
  "emotion_confidence": 0.94,              // как есть
  "dialogue_act": "QUESTION",              // как есть + controlled vocabulary
  "prosody": {                             // НОВОЕ
    "profile": "QUESTION",
    "recommended_profile": "QUESTION",
    "intensity": 0.55,
    "pace": "NORMAL",
    "confidence": 0.94
  }
}
```

`SCHEMA_VERSION: "2" → "3"`, `PROMPT_VERSION: "3" → "4"` — старые разборы
инвалидируются кешем автоматически (ключ уже содержит обе версии).

### 3.3 Батчинг (Phase 6)

```jsonc
// один запрос на окно/сцену
{ "replicas": [ {"replica_id": 12, "prosody": {...}}, {"replica_id": 13, "prosody": {...}} ] }
```

Требуется новая схема ответа плюс окно разбиения (`context_replicas` уже есть как
размер окна) и привязка результата к `replica_id` с отбраковкой неизвестных id.

### 3.4 БД — миграция 10 (аддитивная)

```sql
ALTER TABLE replicas ADD COLUMN prosody_profile TEXT;        -- recommended (LLM)
ALTER TABLE replicas ADD COLUMN prosody_intensity REAL;
ALTER TABLE replicas ADD COLUMN prosody_pace TEXT;
ALTER TABLE replicas ADD COLUMN prosody_confidence REAL;
ALTER TABLE replicas ADD COLUMN reference_profile_key TEXT;
ALTER TABLE replicas ADD COLUMN reference_fallback_reason TEXT;
```

`prosody_override`/`prosody_effective` **не дублируют** `emotion_override`/`emotion_effective`:
решение — или переиспользовать существующие поля (алиас в API), или добавить явные
(`prosody_override`), но не оба варианта сразу. Требует решения (см. вопрос 2).

`voices.json` — **формат не меняется**: `ReferenceProfile.emotion` становится
`profile_key` (расширенный словарь), legacy-проекция сохраняется.

---

## 4. §21: фактическое хранение `RECORD_PHRASES` — ответ

Вопрос задачи («11 отдельных WAV или один объединённый?») в коде имеет ответ
**«ни то, ни другое»**:

- `RECORD_PHRASES` (`app.js:91-137`) — это лишь список **подсказок текста** для записи;
- UI записывает **одну выбранную** фразу за раз (`renderRecordPhrases:2405`,
  `startRecording:2443` → `finishRecording:2548`), шлёт **один** `POST /api/voices`
  (`createVoice:2332`) → создаётся **один голос** и **один** WAV
  (`voices_store.py:374-377`);
- конкатенации нет (все `concat` в проекте — сборка итогового трека, не референса);
- батч-загрузки нескольких файлов нет (`List[UploadFile]` не используется).

Следствие для Phase 3: **отдельные оригинальные записи сейчас не существуют** —
их нужно получить. Значит Phase 3 = новый поток записи «все фразы → один голос»,
а не «прикрепить существующие WAV». Это UX-решение (см. вопрос 1).

---

## 5. Перепланированные фазы (с учётом уже сделанного)

| Phase | Статус | Что осталось |
|---|---|---|
| 0 Audit & Contracts | **выполнено** | этот документ |
| 1 Reference benchmark infra | todo | корпус 30 реплик, матрица, метрики, шаблон listening review |
| 2 ReferenceProfile Data Model | **сделано** | добавить `profile_key`-словарь и quality/engine-нюансы |
| 3 RECORD_PHRASES → Profiles | todo (G1) | мультизапись 11 фраз в один голос + маппинг label→profile_key |
| 4 ProsodyResolver | частично (G5, G6) | fallback-цепочка, `resolve_prosody`, engine-compat |
| 5 Dialogue Analyzer Schema | **выполнено** (G2, G3) | словарь профилей, intensity/pace/confidence, controlled `dialogue_act`, schema v3 |
| 6 Scene-aware LLM Analysis | **выполнено** (G4) | окна/батчинг, `results[]`, валидация, fallback; gate «нет one-request-per-replica» пройден |
| 7 Persistence & Invalidation | частично (G8) | миграция 10, stale-логика при смене текста/голоса |
| 8 Dialogue UI | частично (G7) | блок «Интонация», confidence, reference details, override |
| 9 TTS Integration | частично | проброс `ResolvedProsody` до движка (точки уже найдены: `audio_pipeline.py:986-987 → 1005-1013`) |
| 10 Regenerate & Variants | частично | prosody-aware regenerate, метаданные take |
| 11 Real-model Benchmark | todo (G10) | требует F5/XTTS/Whisper + прослушивание |
| 12 Regression & Hardening | todo | полный pytest, LLM off, memory pressure |
| 13 Documentation & KB | todo (G12) | README, API docs, data model docs, KB |

---

## 6. Gate Phase 0

- Дублирующей архитектуры не создаём: `ReferenceProfile`, `resolve_reference`,
  `LinguisticAnalyzer`, `HeavyGate`, `AnalysisCache`, short-utterance — **переиспользуются**.
- Новый код добавляется в: `emotions.py` (словарь), `schemas.py` + prompt v4 (поля),
  `integration.py` + окно (батчинг), `reference_resolver.py` (fallback-цепочка),
  `db/migrations.py` (миграция 10), `app.js` (UI), `benchmarks/` (новый benchmark).
- Старые голоса и проекты продолжают работать: `voices.json` не меняет формат,
  миграция 10 аддитивная, legacy→NEUTRAL проекция сохраняется.
