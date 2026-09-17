# План: качество коротких реплик (short_phrasases.md)

Рабочий документ к `short_phrasases.md`. Сначала — разведка существующих швов и
решение о том, куда именно встраивается слой коротких реплик, затем фазы.

## 1. Разведка: что уже есть и что переиспользуется

| Что нужно по документу | Что есть в проекте | Решение |
| --- | --- | --- |
| Логирование входа синтеза перед `engine.synthesize()` (§1) | `_synthesize_chunk` — единственная точка вызова движка | Структурная строка лога `tts.synthesis.input` со всеми полями §1; парсится benchmark'ом |
| Единый классификатор реплики (§2) | нет | `backend/short_utterance.py`: `classify_utterance` → `UtteranceClass` (very_short/short/normal), пороги в `config` |
| Отдельный runtime-слой контекста (§3, §12–§15) | `Replica.final_text` — подготовленный текст, `_synthesize_checked` берёт его как есть | Слой строится **после** подготовки и не трогает `final_text`: `ShortUtteranceContext` → `SynthesisPlan(tts_target_text, tts_context_text, tts_synthesis_text, strategy)` |
| Стратегии DIRECT/PUNCTUATION/SAME_SPEAKER_CONTEXT/SYNTHETIC_CONTEXT/BATCH_AND_CROP (§4–§10) | нет | Все пять — чистые функции построения synthesis-текста + benchmark; в продакшене включается только выигравшая |
| Надёжные границы для обрезки (§11, §31) | Whisper уже есть: `transcribe.py` + `transcribe_worker.py` (JSON в stdout) | Добавить режим `--words` с таймстемпами слов (та же зависимость, без новых); дешёвый вариант — поиск паузы — только для benchmark |
| Short-specific QA (§20, §21) | `qa_screening.screen_chunk` (пусто/тишина/длина/перегруз/повтор), `_measure_wer` через Whisper | `short_utterance.check_chunk`: те же признаки + специфичные (недостающие/лишние слова, повтор слова) и порог длительности от длины текста |
| Ограниченный retry (§19) | QA-цикл с `_nudge_tuning`; сид выбирается заново на каждую попытку | Отдельная политика для very_short/short: до `SHORT_UTTERANCE_MAX_ATTEMPTS` попыток с новым сидом, лучший результат по вердикту |
| Метаданные реплики (§24) | `takes.parameters` (JSON) и `take_quality` | Добавить в параметры куска: `utterance_class`, `short_utterance_strategy`, `synthesis_text_hash`, `context_source`, `attempt_count` |
| UI (§23) | настройки рендера (`RenderSettings` + форма) | Поля `enabled` / пороги / стратегия (`auto`/`direct`/`context`) в настройках рендера, «Дополнительно» |
| Benchmark (§1, §25, §26) | `tools/memory_check.py` (в процессе), `tools/loadtest.py` (через API), `backend/benchmark.py` (сравнение движков) | `tools/short_bench.py` — в процессе, реальные модели, отчёт JSON + WAV для прослушивания |

## 2. Архитектура слоя

```
Prepared Replica (final_text, разметка ударений, словарь)
        │
        ├── classify_utterance(final_text) → UtteranceClass
        │
        ├── NORMAL → существующий путь (ничего не меняется)
        │
        └── SHORT / VERY_SHORT (если включено)
                │
                ├── ShortUtteranceContext (target/previous/next/тот же спикер)
                │
                ├── strategy → SynthesisPlan
                │     tts_target_text   = final_text (не меняется)
                │     tts_context_text  = контекст (или "")
                │     tts_synthesis_text= то, что реально уйдёт в движок
                │
                ├── engine-specific подготовка synthesis-текста
                │     F5: ударения и «+» сохраняются и в контексте
                │     XTTS: «+» не добавляются
                │
                ├── engine.synthesize(tts_synthesis_text)
                │
                ├── обрезка цели по надёжной границе (ASR-таймстемпы);
                │   нет надёжной границы → fallback DIRECT (без контекста)
                │
                └── short QA → при провале повтор с новым сидом (ограниченно)
```

## 3. Фазы

**Ф1. Классификатор, контекст, стратегии, план.** `backend/short_utterance.py`
(чистый модуль без torch): `UtteranceClass`, `classify_utterance`,
`ShortUtteranceContext`, `build_contexts`, `SynthesisPlan`, стратегии, отпечаток
синтез-текста; пороги и стратегия — в `config`; структурный лог входа синтеза
(§1). Тесты §27 для детектора, контекста и стратегий.

**Ф2. Short QA и ограниченный retry.** `check_chunk` (пусто/тишина/длительность/
повтор/лишние и недостающие слова по расшифровке — расшифровка инъектируется),
`SHORT_UTTERANCE_MAX_ATTEMPTS`, политика повтора с новым сидом в
`_synthesize_checked`. Тесты §27 (retry/QA).

**Ф3. Benchmark и границы.** Режим `--words` в воркере распознавания и
`transcribe_words`, определение границы цели (ASR-таймстемпы и пауза),
`tools/short_bench.py`: фразы §1 × голоса × стратегии × сиды, метрики (ASR-текст,
совпадение, длительность, доля тишины, время генерации), WAV для listening A/B,
отчёт JSON/Markdown.

**Ф4. Интеграция.** `RenderSettings` + API + форма (включение, пороги,
стратегия), проход плана по репликам в `render_dialogue` (и в перегенерации из
payload), F5/XTTS-специфичная подготовка synthesis-текста, метаданные куска,
регистрация стратегий в `/api/status`/настройках. Тесты §27–§28 (регрессия
длинного текста, `test_normal_utterance_bypasses_short_optimizer`).

**Ф5. Живой benchmark и решение.** Прогон на реальных F5 (и XTTS, если
доступен): A = DIRECT, B = стратегии; слушательные файлы для A/B; выбор
производственной политики **по измерениям**, README/API-доки/отчёт, Obsidian.

## 4. Границы (чего не делаем)

* `source_text` и `final_text` не мутируются: контекст живёт только в runtime.
* NORMAL-реплики идут прежним путём без единого лишнего шага.
* Новая тяжёлая зависимость не добавляется: границы — существующим Whisper.
* Контекст разных спикеров не синтезируется чужим голосом.
* Обрезка «примерно по времени» не внедряется: нет надёжной границы → DIRECT.
* Production-стратегия включается только по результатам benchmark; до него —
  выключено.
