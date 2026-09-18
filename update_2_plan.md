# UPDATE 2 — план и аудит (Phase 0/1)

Рабочий документ по пакету `update_2.md` («Emotion-Aware Dialogue + Reliable Short
Replica Synthesis»). Здесь зафиксированы **факты текущего кода** (Phase 0) и план
фаз; по мере выполнения пункты получают ссылки на отчёты, тесты и коммиты.
Итоговый отчёт по шаблону §53 — в `update_2_report.md`.

## §1. Что уже есть и переиспользуется (Phase 0)

Второго пайплайна нет и не будет: и диалог, и сплошной текст, и короткие реплики,
и (новые) эмоции идут одним путём `_synthesize_checked` → `_prepare_chunk` →
`render_dialogue`. Ни один слой не переписывает `final_text`.

| Механизм | Где | Что важно для UPDATE 2 |
| --- | --- | --- |
| Разбор диалога | `backend/dialogue_parser.py:249` `parse_dialogue` | реплика = одна строка/поворот спикера; при `max_replica_chars` режется **по границам предложений** (`_split_long_replicas:303`) |
| Нарезка на куски | `split_into_chunks:381` | используется для реплик длиннее лимита и для сплошного текста; короткая реплика куском быть не может |
| Один вызов движка на реплику | `audio_pipeline._synthesize_checked:1196` | `text = replica.final_text` (или `_text_for_engine`), `short_run_for` строит план короткой реплики |
| Короткий слой | `short_utterance.py`, `_synthesize_attempt:1119` | контекст + обрезка по ASR-таймстемпам (`short_utterance_boundary.py:140`), паддинги 60/100 мс (`:49`), откат на DIRECT (`:1173`) |
| QA | `qa_screening.screen_chunk`, `_transcribe_and_measure:1418`, `su.check_chunk:737` | WER и причины считаются **по сырому выходу модели**, до `_prepare_chunk` |
| Подготовка куска | `_prepare_chunk:441` | порядок: `_trim_edge_silence:416` → pitch → fade `EDGE_FADE_MS=10` → RMS `target_rms` → gain → пик-лимит |
| Обрезка тишины | `_trim_edge_silence:416` | кадр 20 мс, порог `EDGE_SILENCE_DB=-45`, запас `EDGE_SILENCE_MARGIN_MS=30`; применяется к **сырому** выходу до нормализации |
| Диагностика take | `take_quality.measure:213` | считается по **подготовленному** куску (и `raw` — только для клиппинга) |
| Склейка | `render_dialogue:1605` | пауза + `np.concatenate`, без перекрытий; `_finalize_track:526` — LUFS + лимитер |
| Варианты реплик | `takes` (БД), `save_replica_take`, `select_take`, API `takes/{id}` | метаданные take'а — `parameters` (в т.ч. план короткой реплики), `qa`, `quality`, `seed`, `engine` |
| LLM-анализатор | `backend/llm/*` | structured output, кэш по хешу входа, `think`, primary/fallback, `utterance_hints` для короткого слоя |
| Review произношения | `pronunciation_candidates` + `POST /api/pronunciation` | единственный законный путь правки текста — подтверждение пользователем |
| Миграции | `backend/db/migrations*` (аддитивные) | старые голоса/проекты обязаны работать |
| Память | `resource_guard`, `memory_monitor`, `llm/memory_policy.py` | LLM и Whisper не живут одновременно с TTS без нужды |
| Диагностика | `backend/diagnostics.py` | архив качества уже собирает takes/settings/лог/аудио — расширяется эмоциями и трейсом |

## §2. Кандидаты причины обрывов коротких реплик (Phase 1, до измерений)

Гипотезы, а не факты. Проверяются инструментацией (§14–§15 пакета):

1. **`_trim_edge_silence` на сыром уровне.** Порог −45 dBFS применяется к выходу
   модели **до** выравнивания громкости: тихий финальный согласный (или breathy
   attack) кадрами по 20 мс может уйти под порог и быть срезанным вместе с
   запасом 30 мс. Для длинной реплики потеря 30–80 мс незаметна, для «Да.» —
   это заметная доля слова.
2. **Микрофейд 10 мс** (`_prepare_chunk:452`): не удаляет, но ослабляет атаку
   взрывного/финального щелевого; на односложных фразах это слышно.
3. **QA смотрит не на то.** WER и причины считаются по сырому куску
   (`_synthesize_checked:1308`), а в файл идёт подготовленный: потеря, внесённая
   trim/fade, в отчёте не видна — «QA прошёл, а слово пропало».
4. **Модель.** F5 на очень коротком тексте может сама не договорить финальный
   слог; тогда trim не виноват, и лечится это стратегией синтеза (atomic DIRECT,
   контекст, повтор), а не постобработкой.
5. **Контекстный слой.** `same_speaker_context` + обрезка по ASR: если граница
   найдена неточно, обрезается первый/последний фонем. Паддинги 60/100 мс есть,
   но benchmark'а «сколько режет» не было.

Разделить 1–3 и 4 можно только сравнением `01_raw_engine` ↔ `03_post_edge_trim` ↔
`04_final_replica` вместе с числами среза по стадиям — это и делает трейс.

## §3. План фаз

| Фаза | Что | Гейт |
| --- | --- | --- |
| 0 | аудит (этот документ) | контракты зафиксированы |
| 1 | `backend/synthesis_trace.py`, чекпоинты 01–04, `tools/short_trace.py`, живой прогон регрессионного диалога | **причина названа числами**: где физически теряется начало/конец |
| 2 | atomic short replica (инвариант + тест), boundary-safe trim с guard, `ShortUtteranceQA`, bounded retry | DIRECT-озвучка коротких фраз полная без синтетического контекста |
| 3 | `ReferenceProfile`, миграция старых голосов в NEUTRAL, `EmotionReferenceResolver` (same-voice, NEUTRAL fallback) | старые голоса и проекты работают |
| 4 | LLM scene/emotion в существующей схеме, кэш, prompt safety, deterministic fallback | LLM выключен → приложение работает |
| 5 | UI эмоций, regenerate с эмоцией, метаданные вариантов, API | ручной выбор важнее авто |
| 6 | `ShortDialogueSynthesisPlanner` (CONTEXT_ASSISTED/SAME_SPEAKER_BATCH) — **только если benchmark фазы 2 покажет необходимость** | нет naive crop |
| 7 | real-model benchmark (F5, ≥2 голоса, несколько сидов, XTTS если есть) + listening review | результаты записаны |
| 8 | README/Knowledge Base + `update_2_report.md` | документация = код |

## §4. Контракты, которые нельзя ломать

* `source_text`/`normalized_text`/`dictionary_text`/`accentized_text`/`final_text`
  не содержат служебных тегов: эмоция — metadata.
* Правка текста — только через review/подтверждение (`pronunciation_candidates`),
  включая опечатку `быает → бывает`.
* Один audio synthesis unit — один speaker, один voice, совместимый движок.
* Cross-speaker текст может быть *семантическим* контекстом для LLM, но не
  аудио-целью.
* Границы контекстного синтеза — только ASR/forced alignment, без «по проценту
  символов/слов/длительности».
* QA не судит эмоцию через Whisper.
* Миграции аддитивные, старые API/проекты/голоса продолжают работать.
