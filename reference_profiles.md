# Reference Profiles и просодия: база знаний (UPDATE 3)

Документ описывает, как один голос получает **несколько интонационных записей**,
как из этих записей выбирается референс под реплику и как метаданные интонации
живут рядом с текстом, не попадая в него. Отчёт по реализации — `update_3_report.md`,
результаты benchmark'а — `reference_prosody_benchmark.md`, анализатор —
`llm_russian_analyzer.md` (раздел 9).

Дата: 2026-09-19. Код: `backend/voices_store.py`, `backend/emotions.py`,
`backend/reference_resolver.py`, `backend/db/migrations.py`, `backend/main.py`,
`frontend/app.js`.

---

## Часть I. Reference Profiles (§82)

### 1. Модель данных: Voice 1:N ReferenceProfile

```text
Voice (voices.json)
  ├── audio_file, ref_text, engine, gender, ...     ← legacy-референс
  ├── profiles: [ReferenceProfile, ...]              ← интонационные записи
  └── reference_profiles()                            ← проекция на чтение

ReferenceProfile (backend/voices_store.py:135)
  ├── id                        uuid hex[:12] (или "{voice_id}-neutral")
  ├── voice_id                  восстанавливается из вложенности в Voice
  ├── emotion / profile_key     ключ профиля (словарь emotions.PROFILE_KEYS)
  ├── label                     человекочитаемая подпись
  ├── audio_file, ref_text      запись и её расшифровка
  ├── source                    "legacy" | "record"
  ├── quality_status, quality_note
  ├── transcription_status
  ├── duration_sec
  ├── engine_compatibility: [engine_id, ...]
  ├── source_record_phrase_id   какая фраза RECORD_PHRASES дала профиль
  ├── is_default, enabled, enabled_for_auto
  └── created_at, updated_at
```

`profile_key` — свойство-синоним `emotion` (`voices_store.py:197`). Отдельного
метода `profile_by_key` в коде нет.

### 2. Schema и статусы

Голос хранится в `voices/voices.json`; профили лежат **внутри** записи голоса —
в SQLite их нет. Поэтому 1:N «голос → профили» обеспечивается структурно, а не
связью по внешнему ключу.

Статусы качества (`voices_store.py:112-121`):

| Поле | Значения |
|---|---|
| `quality_status` | `ok` / `warning` / `unknown` |
| `transcription_status` | `ok` / `warning` / `from_form` / `unknown` |
| `source` | `legacy` (старый референс) / `record` (записан в приложении) |

### 3. Migration (совместимость со старыми голосами)

**Отдельной миграции данных для профилей нет.** Голос, сохранённый до появления
профилей, читается как нейтральный через проекцию на чтение:

- `Voice.reference_profiles()` (`voices_store.py:297-315`) добавляет
  `neutral_profile()` первым, если его `audio_file` ещё не встречается среди
  профилей;
- `Voice.neutral_profile()` (`voices_store.py:317-342`) строит профиль с
  `id = f"{self.id}-neutral"`, `label = "Основной"`, `source = "legacy"`,
  `audio_file`/`ref_text` из старого референса и `engine_compatibility=[engine]`;
- `voices.json` при этом **не переписывается** — это проекция «на чтение».

БД от этого не зависит: миграции 9 и 10 добавляют репликам колонки, которые лишь
**ссылаются** на профили (`reference_profile_id`, `reference_profile_key`). Старые
голоса и проекты продолжают работать без миграции данных; `voice_id` не меняется,
один профиль не превращается в отдельный голос.

Профили, записанные поверх старого голоса, живут рядом: legacy-референс остаётся
нейтральным «Основным», новые — отдельными записями.

### 4. RECORD_PHRASES mapping

Список фраз для начитки объявлен **во фронтенде** (`frontend/app.js:91-137`, ровно
11 фраз); маппинг «подпись → ключ профиля» — `RECORD_PHRASE_PROFILES`
(`frontend/app.js:145-157`):

| Подпись фразы | Ключ профиля |
|---|---|
| Вопрос и утверждение | `NEUTRAL_QUESTION` |
| Восклицания, много шипящих | `EXCLAMATION` |
| Спокойная просьба (короче) | `CALM` |
| Только вопросы | `QUESTION` |
| Сложные сочетания согласных | `NEUTRAL` |
| Радость, восторг | `DELIGHT` |
| Огорчение, сочувствие | `SAD_SYMPATHETIC` |
| Лёгкая ирония | `IRONIC` |
| Строгий тон, короткий приказ | `STRICT` |
| Перечисление, ровный ритм | `ENUMERATION` |
| Быстрая, взволнованная речь | `EXCITED` |

Идентификатор фразы — её индекс (`recordPhraseId`, `frontend/app.js:163`).
Backend не знает списка фраз: он принимает form-поле `source_record_phrase_id`
(`main.py:1163`) и хранит его в профиле. **Повторная запись той же фразы заменяет
прежний профиль** этого голоса, а не создаёт дубль (`voices_store.py:601-605,
676-691`); старый файл удаляется.

### 5. Quality requirements

Профиль попадает в автоматику, только если прошёл проверки:

- **Сверка «референс ↔ расшифровка»** — `transcribe.check_ref_text_match`,
  порог `MATCH_THRESHOLD = 0.6` (`backend/transcribe.py:42`). Расхождение даёт
  `quality_status = warning` и `quality_note`; такой профиль `_usable` отвергает.
- **Пол записи** — `audio_analysis.check_gender_mismatch`, диапазоны
  `MALE_RANGE = (85, 155)` / `FEMALE_RANGE = (165, 255)` Гц
  (`backend/audio_analysis.py:47-48`).
- **Перегрузка** — клиппинг при `CLIPPING_LEVEL = 0.99` и доле выше
  `CLIPPING_MIN_RATIO = 0.001` (`backend/audio_analysis.py:63-64`).

`quality_status`/`quality_note` — единственный «билет» профиля в автоматику:
`_usable` отказывает по `warning` с человекочитаемой причиной.

### 6. Engine compatibility

`engine_compatibility` — **список** id движков (`voices_store.py:184`):

- пустой список = «профиль годится для любого движка голоса»
  (`reference_resolver.py:243-254`);
- значения — `f5`, `xtts` и т.п.;
- проверяет `engine_compatible(profile, engine_id)`, вызывает её `_usable`
  (`reference_resolver.py:238-239`) с причиной
  `"профиль не проверен для движка {engine_id}"`;
- **нейтральный профиль проверку движка не проходит** — для NEUTRAL вызывается
  `check_engine=False` (`reference_resolver.py:210-214, 344`).

Следствие: профиль, подтверждённый на F5, автоматика не возьмёт на XTTS, пока он
не помечен совместимым и с XTTS.

### 7. Fallback rules

Режим отката задаётся `PROSODY_FALLBACK_MODE` (`backend/config.py:316-317`):

| Режим | Что делает |
|---|---|
| `neutral_only` (дефолт) | нет нужного профиля → нейтральный профиль **того же** голоса |
| `chain` | идёт по цепочке родственных интонаций (`FALLBACK_CHAINS`), всегда заканчиваясь NEUTRAL |

`fallback_chain()` (`reference_resolver.py:126-142`) гарантированно завершается
`NEUTRAL`; неизвестный режим трактуется консервативно. Причины (русские строки,
`_reason`, `reference_resolver.py:372-389`):

- `"интонационный профиль"` — профиль найден;
- `"референса для интонации {X} нет — взят нейтральный"` — сработал откат;
- `"профиль {X} отсутствует — использован ближайший {Y}"` — цепочка (`chain`);
- `"нейтральный референс"` — цель и была NEUTRAL;
- к причине добавляется отказ в скобках, например
  `"… (профиль не подтверждён для автоматического выбора)"`.

Причины отказа `_usable` (`reference_resolver.py:198-240`): `"профиль выключен"`,
`"у профиля нет файла"`, `"файл референса отсутствует"`, `quality_note` (или
`"референс не прошёл проверку"`), `"профиль не подтверждён для автоматического
выбора"`, `"профиль не проверен для движка {engine_id}"`. Если не нашлось даже
нейтрального — `ReferenceUnavailableError`, синтез не молчит.

### 8. Same-voice invariant

Инвариант **структурный**: профили лежат внутри `Voice`, а резолвер обращается к
`voice.reference_profiles()` (`reference_resolver.py:301`). Взять референс чужого
голоса нельзя — такого пути в коде нет; `voice_id` в результате
(`ResolvedReference.voice_id`) — страховка от ошибки вызывающего. Покрыто тестом
`test_fallback_never_leaves_the_voice` (`tests/test_prosody_resolver.py`).

---

## Часть II. Prosody (§83)

### 9. Четыре состояния интонации

| Спецификация | Как есть в коде |
|---|---|
| `prosody_detected` | колонка `prosody_profile` — **рекомендация** модели/эвристики |
| `prosody_override` | существующая `emotion_override` — ручной выбор человека |
| `prosody_effective` | **вычисляется**, не хранится: `override → recommended → emotion` |
| — | `emotion_detected` / `emotion_confidence` — сырой разбор LLM |

`prosody_effective` считает `emotions.prosody_effective`
(`backend/emotions.py:163-184`), реплика делегирует ей
(`backend/dialogue_parser.py:110-117`). Отдельных колонок `prosody_detected` /
`prosody_override` **не существует** — это осознанное решение, чтобы не дублировать
`emotion_override`/`emotion_effective` (комментарий `migrations.py:279-283`).

### 10. Поля реплики

Колонки (миграция 10, `backend/db/migrations.py:290-297`; миграция 9 — эмоции):

| Поле | Тип | Смысл |
|---|---|---|
| `prosody_profile` | TEXT NOT NULL DEFAULT '' | рекомендуемый профиль |
| `prosody_intensity` | REAL NULL | насколько выраженно (0…1); `NULL` — «модель не сказала» |
| `prosody_pace` | TEXT | `SLOW` / `NORMAL` / `FAST` |
| `prosody_confidence` | REAL NULL | уверенность просодии |
| `dialogue_act` | TEXT | речевой акт (controlled, ≤ 40 символов) |
| `context_dependency` | TEXT | `LOW` / `MEDIUM` / `HIGH` |
| `reference_profile_key` | TEXT | ключ фактически выбранного профиля |
| `reference_fallback_reason` | TEXT | причина отката словами |

Словари: `PROSODY_PACES = ("SLOW", "NORMAL", "FAST")` (`config.py:323`),
`CONTEXT_DEPENDENCIES = ("LOW", "MEDIUM", "HIGH")` (`llm/schemas.py:105`).

### 11. Словарь профилей

`backend/emotions.py`:

- `PROFILE_KEYS` (11) — записываемые профили: `NEUTRAL`, `CALM`, `QUESTION`,
  `NEUTRAL_QUESTION`, `EXCLAMATION`, `DELIGHT`, `SAD_SYMPATHETIC`, `IRONIC`,
  `STRICT`, `ENUMERATION`, `EXCITED`;
- `EMOTIONS` (13) = `PROFILE_KEYS` + `SURPRISE`, `FEAR` — эмоции, у которых своей
  записи может не быть;
- `SELECTABLE_EMOTIONS` (14) = `AUTO` + `EMOTIONS` — то, что видит пользователь;
- `PROSODY_REGISTER_*` + `prosody_compatible` — совместимость интонаций для
  same-speaker контекста коротких реплик;
- `heuristic_emotion(text)` — детерминированный fallback без LLM.

Правило выбора: ручной `override` всегда сильнее рекомендации; `AUTO` — это не
эмоция, а «реши сам».

### 12. Resolver

`resolve_prosody(...)` (`reference_resolver.py:392-429`) — надстройка над
`resolve_reference(...)` (`275-369`), добавляющая fallback-цепочку. Результат —
`ResolvedReference` (`145-195`) с полями `voice_id`, `requested_emotion`,
`resolved_emotion`, `profile_id`, `audio_path`, `ref_text`, `fallback_used`,
`reason`, `quality_status`, `engine_id`, `fallback_reason`, `dialogue_act`,
`intensity`, `candidates_tried`. `to_dict()` отдаёт их под именами §24:
`requested_profile`, `resolved_profile`, `reference_profile_id`,
`reference_emotion`, `reference_fallback_used`, `reference_fallback_reason`,
`reference_quality`, `reference_reason`, `reference_audio`, `reference_text`,
`engine_id`, `dialogue_act`, `intensity`, `candidates_tried`.

Сортировка кандидатов (`_candidates_for`): сначала прошедшие проверку, затем не
`is_default`, затем по `created_at` — выбор детерминирован.

### 13. Просодия не входит в TTS text

Интонация — **метаданные**, а не слово в тексте. Ни одно поле просодии/эмоции не
попадает в `source_text`, `normalized_text`, `dictionary_text`, `accentized_text`
или `final_text`:

- докстринг `backend/emotions.py:1-11`;
- комментарий миграции 9 (`migrations.py:258-263`) и миграции 10
  (`migrations.py:275-289`);
- `backend/dialogue_parser.py:82-101` — поля не уходят в движок как слово;
- `REWRITE_FIELDS` в `backend/llm/schemas.py:270-285` — ответ модели с новым
  текстом отвергается целиком.

Метаданные едут в движок отдельно от текста (`main.py:_prosody_metadata`,
`2132-2147`), а `regenerate` берёт **сохранённый** текст, меняя только интонацию.

### 14. Что видит пользователь (API)

Блок `prosody` в ответе реплики (`main.py:2338-2355`): `profile`, `profile_title`,
`intensity`, `pace`, `confidence`, `override`, `effective`, `effective_title`,
`source`, `dialogue_act`, `context_dependency`. Блок `reference`
(`main.py:2356-2363`): `profile_id`, `profile_key`, `emotion`, `fallback_used`,
`fallback_reason`.

Ручное управление: `emotion_override` и `reference_profile_id` в
`PATCH /api/projects/{id}/replicas/{index}` и
`POST /api/projects/{id}/replicas/{index}/regenerate`. Подтверждение профиля для
автоматики — `PATCH /api/voices/{voice_id}/references/{profile_id}` с
`{"enabled_for_auto": true|false}` (галочка «авто» в карточке голоса).

### 15. Границы

- Профиль по умолчанию **не** участвует в автоматике: `enabled_for_auto = false`.
  Пока профиль не подтверждён прослушиванием, автоматика его не берёт, но вручную
  выбрать можно (см. `reference_prosody_benchmark.md`).
- `intensity`/`pace`/`confidence` — информация, а не ручки движка: они не меняют
  `nfe_step`, `cfg_strength` или скорость синтеза.
- Решение о **синтезе короткой реплики** принимает отдельный слой; интонация для
  него — подсказка.
