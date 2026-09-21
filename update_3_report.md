# UPDATE 3 — отчёт: Russian Dialogue Analyzer, Reference Profiles и автоматическая маршрутизация просодии

Отчёт по пакету `update_3.md` (шаблон §90). Карта решений и разрывов — в
`dialogue_prosody_plan.md`; факты реализации, измерения и ограничения — здесь.
Отдельно стоящая база знаний: `reference_profiles.md` (§82–§83),
`reference_prosody_benchmark.md` (§84), `llm_russian_analyzer.md` (§81).

**Принципиальная граница этого отчёта:** всё, что не измерено, помечено как не
измеренное. Прогонов переноса просодии (F5/XTTS) и прослушивания не было —
поэтому §17–§21 отвечают «не выполнено», а не дают красивых чисел.

---

## 1. Summary

UPDATE 3 надстроен над Local LLM Russian Linguistic Analyzer, а не продублирован:
тот же анализатор, та же схема ответа, тот же кеш и тот же пайплайн — добавлен
сценарный режим и просодическая маршрутизация.

1. **Russian Dialogue Analyzer.** Анализатор читает диалог **окнами сцены**
   (`plan_windows`), один запрос — на окно, а не на реплику. Смысл реплики
   («Правда?» после хорошей и после плохой новости) решается по сцене, а не по
   одной строке.
2. **Reference Profile.** Один голос → N записей (`Voice 1:N ReferenceProfile`).
   Записи по 11 фразам `RECORD_PHRASES` становятся 11 профилями **одного**
   голоса, каждый — со своей интонацией, расшифровкой и качеством.
3. **Просодия как метаданные.** Модель предлагает интонацию
   (`prosody.profile`/`recommended_profile`), интенсивность, темп и уверенность.
   Ни одно из этих полей не попадает в текст синтеза.
4. **ProsodyResolver.** Единственное место, которое выбирает файл референса:
   явный профиль → интонация → NEUTRAL. LLM называет намерение, backend выбирает
   конкретный reference — детерминированно.
5. **Авто и безопасность.** Автоматическая маршрутизация берёт только
   подтверждённые профили (`enabled_for_auto`); по умолчанию подтверждённых нет,
   поэтому Auto уводит в NEUTRAL, а профили остаются доступны вручную.
6. **Откат.** Таблица `FALLBACK_CHAINS`; производственный режим —
   `neutral_only` (откат сразу на NEUTRAL). Режим `chain` включается только после
   подтверждения benchmark'ом.
7. **Совместимость.** Все поля аддитивны, миграция данных не нужна: старые голоса
   читаются как NEUTRAL-профиль, старые проекты — как «модель ничего не
   рекомендовала».

## 2. Changed files

| Файл | Что |
|---|---|
| `backend/emotions.py` | два словаря (`PROFILE_KEYS` 11 / `EMOTIONS` 13), `prosody_effective`, регистры просодии, `prosody_compatible` |
| `backend/reference_resolver.py` | `resolve_reference`, `resolve_prosody`, `ResolvedReference`, `FALLBACK_CHAINS`, `engine_compatible` |
| `backend/voices_store.py` | `ReferenceProfile`, `reference_profiles()`, `neutral_profile()`, проекция legacy-референса |
| `backend/llm/schemas.py` | схема v3: `ProsodyHint`, поля просодии у `UtteranceHint`, `_sanitize_prosody` |
| `backend/llm/analyzer.py` | сценарный режим: `build_context`, `plan_windows`, `analyze_window`, `window_prompt` |
| `backend/llm/integration.py` | `ProjectLlmAnalyzer.run(scene=…)`, окна сцены, `replica_emotions` |
| `backend/llm/prompt.py`, `versioning.py` | prompt v4 и оконный v5, `PROMPT_VERSION = "4"` |
| `backend/llm/settings_store.py`, `fake_client.py` | настройки и поддельный клиент под сцену |
| `backend/dialogue_parser.py` | просодические поля реплики, `prosody_effective` |
| `backend/audio_pipeline.py` | референс через `resolve_prosody`, просодия в метаданных варианта |
| `backend/short_utterance.py` | учёт регистра просодии при выборе контекста |
| `backend/synthesis_trace.py` | просодия в трейсе стадий |
| `backend/job_queue.py` | паспорт просодии в метаданных варианта |
| `backend/db/migrations.py` | миграция 10 |
| `backend/db/repositories/replicas.py`, `backend/db/store.py` | колонки просодии, сохранение |
| `backend/main.py` | просодия в ответе реплики, `reference_status`, CRUD профилей, флаг `enabled_for_auto` |
| `frontend/app.js`, `frontend/index.html` | `RECORD_PHRASE_PROFILES`, карточка просодии, dropdown, чекбокс «авто» |
| `backend/prosody_benchmark.py` (новый) | матрица сравнения профилей (инфраструктура) |
| `tools/prosody_benchmark.py` (новый) | CLI прогона benchmark'а |
| `benchmarks/prosody/` (новый) | корпус 36 реплик, 9 категорий |
| `benchmarks/russian_linguistics/prompts/analyzer.v4.txt`, `analyzer.v5.txt` (новые) | одиночный и оконный prompt'ы |
| `tests/test_prosody_resolver.py`, `test_prosody_benchmark.py`, `test_prosody_benchmark_cli.py` (новые) | тесты UPDATE 3 |
| `README.md`, `llm_russian_analyzer.md`, `reference_profiles.md`, `reference_prosody_benchmark.md`, `dialogue_prosody_plan.md` | документация и база знаний |

## 3. DB/storage migrations

**Миграция 10** ([migrations.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/db/migrations.py#L290-L297)),
аддитивная, к таблице `replicas`:

```sql
prosody_profile, prosody_pace,
reference_profile_key, reference_fallback_reason,   -- TEXT NOT NULL DEFAULT ''
prosody_intensity, prosody_confidence               -- REAL (NULL = «модель не сказала»)
```

Осознанно **не** заведены колонки-дубликаты:

- `prosody_override` — ручной выбор уже хранится в `emotion_override` (миграция 9);
  вторая колонка стала бы вторым источником правды;
- `prosody_effective` — вычисляется правилом `override → recommended → эмоция`
  (`emotions.prosody_effective`), хранение дубликата разошлось бы с правилом.

`prosody_intensity`/`prosody_confidence` допускают `NULL`: «модель не сказала»
честно отличается от `0.0` (измеренная тишина). Отдельной DB-миграции под
Reference Profiles **нет** — профили живут внутри `voices.json`, а не в таблицах.

## 4. ReferenceProfile implementation

Модель — `Voice 1:N ReferenceProfile`, профили хранятся внутри записи голоса
(`voices.json`), а не отдельной таблицей. Это делает гарантию «только свой
голос» **структурной**: резолвер физически не может обратиться к чужой записи.

Поля профиля: `id`, `emotion`, `audio_file`, `ref_text`, `quality_status`
(`ok`/`warning`/`unknown`), `quality_note`, `transcription_status`, `source`
(`legacy`/`record`), `engine_compatibility`, `enabled`, `is_default`,
`enabled_for_auto`, `source_record_phrase_id`, `created_at`.

- **Миграция на чтение.** Старые голоса без профилей не переписываются:
  `Voice.reference_profiles()` поднимает legacy `audio_file`/`ref_text` в
  NEUTRAL-профиль `{voice.id}-neutral` (`label = «Основной»`),
  `neutral_profile()` возвращает его же для резолвера.
- **Совместимость движка** (`reference_resolver.engine_compatible`): пустой список
  — «любой движок голоса». NEUTRAL-профиль (сам голос) проверкой движка не
  ограничивается: иначе смена движка голоса запретила бы синтез там, где раньше
  всё работало.
- **same-voice invariant** — структурная (см. выше); проверка `voice_id` в
  результате — страховка от ошибки вызывающего кода, а не основной механизм.

Полное описание — `reference_profiles.md`, часть I.

## 5. RECORD_PHRASES migration

11 фраз `RECORD_PHRASES` ([app.js](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L91-L137))
превращены в 11 профилей **одного** голоса через карту `RECORD_PHRASE_PROFILES`
([app.js](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/frontend/app.js#L145-L157)):

| Фраза (подпись) | Ключ профиля |
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

Позиция фразы в списке — устойчивый идентификатор (`source_record_phrase_id`),
по нему backend понимает, какой профиль заменяет повторная запись: «одна фраза —
один профиль». Правило закреплено тестом (`tests/test_record_phrases.py`).

Под фразы `SURPRISE`/`FEAR` (из UPDATE 2) отдельной записи нет: они остаются
различимыми эмоциями модели, но резолвер уводит их в ближайший записываемый
профиль с явным откатом (§17).

## 6. Russian Dialogue Analyzer changes

Тот же `LinguisticAnalyzer`, но с оконным режимом — параллельный
`EmotionAnalyzer` не создавался.

- **Сцена вместо реплики.** `ProjectLlmAnalyzer.run(scene=True)` режет диалог
  `plan_windows(size=DEFAULT_SCENE_REPLICAS=6, overlap=DEFAULT_CONTEXT_REPLICAS=2)`
  — окна по 6 реплик с перекрытием 2, шаг 4. Ответ на перекрывающуюся реплику
  берётся из первого окна: у каждой реплики ровно один разбор.
- **`analyze_window`** ([analyzer.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/llm/analyzer.py#L674-L792)):
  один запрос на окно; контекст — всё окно, а не ±2 соседа. Не разобранная или не
  прошедшая проверку реплика получает `FAILED` с причиной, а не тихо пропускается.
- **Сплошной текст** идёт прежним путём — по одной «реплике»-фрагменту с
  `build_context` (±2 соседа, бюджет 4000 символов): его фрагменты не сцена с
  говорящими, и оконный режим менял бы смысл запроса.
- **Память.** Слот `HEAVY_LLM` берётся на весь проход; перед каждым окном
  проверяется HARD STOP — частичный разбор с причиной лучше swap на всей машине.

## 7. New LLM schema

Схема v3 добавила блок `prosody` в `UtteranceHint`
([schemas.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/llm/schemas.py#L348-L381)):

```
ProsodyHint: profile, recommended_profile, intensity, pace, confidence
```

- `profile` — что модель **распознала** (семантика, включая `SURPRISE`/`FEAR`);
- `recommended_profile` — ближайший **записываемый** профиль голоса (§10): модель
  сама принимает решение о замене (`SURPRISE → EXCLAMATION`), и резолверу не
  нужно выводить его из цепочки;
- `intensity`, `pace`, `confidence` — метаданные, не параметры синтеза.

`_sanitize_prosody` приводит ответ к безопасному: неизвестный `dialogue_act` →
`OTHER`, `intensity`/`confidence` вне 0..1 → «не сказала», неизвестный
`recommended_profile` отбрасывается. `prosody_detected`/`prosody_override`/
`prosody_effective` из §83 — это не хранимые колонки, а **имена** трёх состояний
интонации; см. `reference_profiles.md`, часть II.

## 8. Prompt/schema versions

| Что | Значение | Источник |
|---|---|---|
| Schema version | `3` | `backend/llm/schemas.py`, `SCHEMA_VERSION` |
| Prompt version | `4` | `backend/llm/versioning.py`, `PROMPT_VERSION` |
| Prompt (одиночная реплика) | `analyzer.v4.txt` | `benchmarks/russian_linguistics/prompts/` |
| Prompt (окно сцены) | `analyzer.v5.txt` | там же |

Обе версии — и schema, и prompt — входят в ключ кеша, поэтому их смена
инвалидирует сохранённый разбор. Смена prompt'а или схемы — единственный способ
инвалидации; ручная смена интонации модель повторно не вызывает.

## 9. ProsodyResolver

Единственное место выбора референса
([reference_resolver.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/reference_resolver.py#L275-L429)):

1. явный выбор пользователя (`profile_id`) — если профиль принадлежит **этому**
   голосу, прошёл качество и совместим с движком (флаг `enabled_for_auto` здесь не
   проверяется — человек вправе выбрать неподтверждённый профиль вручную);
2. цепочка отката под запрошенную интонацию;
3. NEUTRAL того же голоса — с пометкой `fallback_used`, без отказа;
4. иначе — ошибка: без референса синтезировать нельзя.

`resolve_prosody` — та же логика плюс диагностический контекст (`dialogue_act`,
`intensity`); LLM файл не выбирает, она называет только `prosody_effective`.
`ResolvedReference.to_dict()` отдаёт поля и под именами UPDATE 2, и под именами
§24 (`requested_profile`/`resolved_profile`), не заводя второй сущности.

## 10. Fallback policy

- Таблица `FALLBACK_CHAINS` — централизованная, покрыта тестом на полноту: под
  каждый профиль, последний элемент всегда NEUTRAL.
- Режимы (`config.PROSODY_FALLBACK_MODE`):
  - **`neutral_only` (по умолчанию)** — откат сразу на NEUTRAL. Консервативный
    производственный режим: §26 разрешает «соседний профиль» только после
    подтверждения benchmark'ом, а подтверждения нет.
  - **`chain`** — включает таблицу целиком; это режим, который проверяет
    benchmark (Phase 11).
- Неизвестное значение режима не подменяется молча: логируется и трактуется как
  `neutral_only`.
- Причина отката называется словами (`_reason`, `reference_fallback_reason`) —
  пользователь видит, почему интонационный референс не взяли.

## 11. Dialogue UI

- **Карточка реплики.** Блок просодии: что распознано (`prosody.profile`),
  уверенность, связь с контекстом, насыщенность, темп, действующий профиль
  (`prosody.effective`) и источник; dropdown ручной смены интонации
  (`data-role="prosody"` → `emotion_override`).
- **Карточка голоса.** Список `reference_profiles` с чекбоксом «авто»
  (`data-role="reference-auto"` → `enabled_for_auto`, `PATCH
  /api/voices/{id}/references/{profile_id}`).
- **Мастер записи.** 11 фраз `RECORD_PHRASES` с прогрессом «осталось фраз N из
  11»; каждая запись становится профилем голоса.

## 12. Regenerate/variants changes

Метаданные варианта несут паспорт просодии: `prosody_profile`,
`prosody_confidence`, `prosody_intensity`, `prosody_pace`
(`_prosody_metadata` в [main.py](file:///Users/vladislavkovalenko/Projects/VOICE_SYNTEZ/backend/main.py#L2132-L2144)).
Пересинтез (`regenerate`) идёт тем же маршрутом через `resolve_prosody`, поэтому
смена интонации или профиля меняет reference варианта, а не только его звук.
Смена ручного выбора (`emotion_override`) не вызывает модель повторно и не меняет
текст — только следующий синтез берёт другой референс.

## 13. Short-utterance integration

Слой коротких реплик учитывает просодию при выборе контекста: контекст берётся
только у реплики того же спикера **и** совместимого регистра
(`emotions.prosody_compatible`, регистры `calm`/`high`/`marked`). Спокойная фраза и
крик не склеиваются в одно звучание; вопрос и спокойный ответ остаются в одном
регистре (это одна интонационная линия). Без анализа (LLM выключена) проверка
пропускает: отказ функциональности ради политики недопустим.

## 14. Unit test results

Новые unit-тесты (без моделей, на стабах):

- `tests/test_prosody_resolver.py` — **20** тестов: полнота таблицы отката,
  консервативный режим, явный профиль, `warning`-профиль, выключенный профиль,
  неподтверждённый профиль (ждёт benchmark, но доступен вручную), совместимость с
  движком, «резолвер не покидает голос», детерминизм, имена полей §24.
- `tests/test_prosody_benchmark.py` — **17** тестов: корпус, матрица, метрики,
  формат отчёта.
- `tests/test_prosody_benchmark_cli.py` — **7** тестов: CLI, аргументы, запись
  отчёта.

Обновлены: `test_emotion_layer.py`, `test_llm_analyzer.py`,
`test_llm_project_analysis.py`, `test_record_phrases.py`, `test_short_utterance.py`.

## 15. Integration test results

`tests/test_llm_project_analysis.py` и `test_record_phrases.py` проходят оконный
путь: импорт диалога → окна сцены → разбор → `_apply_emotions`/`replica_emotions`
→ сохранение просодии у реплики. Проверяются: один разбор на реплику при
перекрытии окон, `FAILED` с причиной для неразобранной реплики, проекция legacy
голоса в NEUTRAL-профиль, «одна фраза = один профиль».

## 16. Full pytest result

```text
1211 passed, 1 warning
```

Предупреждение — `DeprecationWarning: 'audioop' is deprecated` из pydub, не
относится к изменениям. Полный прогон — с отключённым внешним LLM-конфигом:

```bash
TTS_LLM_SETTINGS_PATH=/tmp/nonexistent-llm.json ./venv/bin/python -m pytest -q
```

## 17. F5 reference benchmark result

**Не выполнено.** Инфраструктура готова (`backend/prosody_benchmark.py`,
`tools/prosody_benchmark.py`, корпус `benchmarks/prosody/` — 36 реплик), но
матрица переноса просодии на F5 не прогонялась: у сохранённых голосов нет ни
одного записанного профиля (`profiles: []` у всех шести голосов), а подделывать
эмоциональный референс из нейтрального §47 прямо запрещает. См.
`reference_prosody_benchmark.md`.

## 18. XTTS benchmark result

**Не выполнено** — по той же причине, что §17.

## 19. Real dialogue benchmark

**Не выполнен.** §74 требует прогона на реальном диалоге; для этого нужны
записанные профили и включённые для Auto референсы, которых пока нет.

## 20. Listening review

**Не выполнен.** Автоматический ASR не заменяет прослушивание, и автор изменений
не может выполнить субъективную оценку за пользователя. Форма подготовлена в
`reference_prosody_benchmark.md` (§9).

## 21. Performance/memory results

Отдельных замеров производительности UPDATE 3 **не проводилось**. Архитектурно
сценарный режим снижает число запросов к модели (один запрос на 6 реплик вместо
шести), а слот `HEAVY_LLM` берётся на весь проход и проверяется перед каждым окном
(защита от HARD STOP). Точные числа не измерены — не выдаются за измерения.

## 22. Documentation updated

- **README** (UPDATE 3): новый раздел «Интонация, Reference Profiles и Auto»,
  подраздел «Модель данных» с диаграммой `Voice 1─N ReferenceProfile` /
  `Project ─ Replica`; API-строки профилей; «Интонация и референс в ответе API»;
  переписанный блок «Тесты» (1211 тестов, таблица «кто что поднимает»); env
  просодии; раздел «Что не входит в проект» приведён в соответствие (анализатор
  только **предлагает** интонацию, решает человек).
- `llm_russian_analyzer.md` — раздел §9 «Russian Dialogue Analyzer (UPDATE 3)»;
  контракт обновлён до схемы v3 / prompt v4.
- `dialogue_prosody_plan.md` — карта reuse/extend/migrate и разрывы G1–G12.

## 23. Knowledge Base updated

- `reference_profiles.md` — §82 (модель `Voice 1:N ReferenceProfile`, schema,
  migration, `RECORD_PHRASES` mapping, quality requirements, engine compatibility,
  fallback rules, same-voice invariant) и §83 (четыре состояния интонации, поля
  реплики, resolver, «просодия не входит в TTS text»).
- `reference_prosody_benchmark.md` — §84: hardware (Apple M3 Pro, arm64, 18 GB),
  engine (не измерено для F5/XTTS/Banana), voices, profiles, seeds, targets,
  automatic metrics, listening conclusions, профили для Auto (ни одного), known
  limitations, как прогнать. Неподтверждённые выводы не записаны.
- `llm_russian_analyzer.md` — §81.

## 24. Known limitations

1. **Перенос просодии не подтверждён.** Ни один профиль не включён для Auto
   (`enabled_for_auto == False` у всех). Практическое следствие: Auto уводит
   реплику в NEUTRAL, пока пользователь сам не подтвердит профиль вручную.
2. **У существующих голосов нет записанных профилей** (`profiles: []`) — запись
   11 фраз пользователем ещё не выполнена.
3. **Listening review не проведён** (§20).
4. **`SURPRISE`/`FEAR`** не имеют собственной записи — до записи звучат ближайшим
   записываемым профилем с явной пометкой отката.
5. **XTTS на просодии не проверялся**; сравнение движков по переносу интонации
   отсутствует.
6. **Режим `chain`** остаётся не производственным: он включается только по факту
   подтверждения benchmark'ом.
7. **Эмоциональный референс может отличаться тембрально** от нейтрального: это
   другая запись, и «похожесть на себя» проверяется только прослушиванием.

## 25. Recommended next step

1. **Записать 11 фраз** `RECORD_PHRASES` на реальном голосе — так появляются 11
   профилей и предмет для benchmark'а.
2. **Прогнать матрицу** `tools/prosody_benchmark.py` на F5 и XTTS: один и тот же
   текст и сид, меняется только профиль. Автоматика ответит «текст не сломался»;
   перенос интонации оценивает человек.
3. **Прослушать** и заполнить форму §34/§9 в `reference_prosody_benchmark.md`.
4. **После подтверждения** включить успешные профили (`enabled_for_auto`) и
   заполнить §10–§11 отчёта benchmark'а фактическими данными.
5. **Прогнать real dialogue benchmark** §74 на диалоге с автоматической
   маршрутизацией, когда профили подтверждены.
