# Short Utterance TTS Quality

## Задача

Улучшить качество озвучки очень коротких реплик, например:

```text
Привет!
Как дела?
Да.
Нет.
Спасибо!
Что?
Почему?
Хорошо.
Ясно.
```

Наблюдаемое поведение:

```text
длинные предложения / абзацы → хорошее качество
короткие реплики → заметно хуже
```

Необходимо реализовать отдельную стратегию обработки коротких реплик, не ухудшая существующую генерацию длинного текста.

---

# 1. Сначала провести диагностику

Не предполагать заранее, что причина только в F5-TTS, ударениях или punctuation.

Добавить воспроизводимый benchmark.

Создать набор:

```text
Привет!
Как дела?
Спасибо.
Да.
Нет.
Хорошо.
Я понял.
Что случилось?
Почему?
До встречи!
```

И несколько контрольных длинных фраз.

Для каждой фразы логировать непосредственно перед `engine.synthesize()`:

```text
engine
voice_id
source_text
final_text
text_length
word_count
reference_audio
reference_text
speed
seed
effective engine params
```

Проверить отдельно F5 и XTTS, если оба доступны.

---

# 2. Ввести Short Utterance Detector

Добавить единый helper, например:

```python
classify_utterance(text) -> UtteranceClass
```

Минимальные классы:

```text
VERY_SHORT
SHORT
NORMAL
```

Не привязывать определение только к количеству символов.

Учитывать как минимум:

```text
word_count
character_count
```

Начальные ориентиры для эксперимента:

```text
VERY_SHORT:
    <= 2 words

SHORT:
    3–5 words

NORMAL:
    > 5 words
```

Пороговые значения должны быть конфигурируемыми.

Не менять NORMAL pipeline без необходимости.

---

# 3. Главное решение: Context-Assisted Synthesis

Для короткой реплики TTS должен иметь возможность получить больше языкового/просодического контекста, чем непосредственно озвучиваемая строка.

Пример dialogue:

```text
Анна: Привет!
Борис: Как дела?
Анна: Отлично. Сегодня наконец-то закончила работу над проектом.
```

Для:

```text
Как дела?
```

движку может быть полезен контекст соседних реплик.

Но нельзя просто передать соседний текст как часть обычной реплики, потому что он попадёт в итоговый WAV.

Нужен специальный механизм.

---

# 4. Реализовать несколько стратегий и benchmark

Не внедрять один hack сразу как единственно правильный.

Добавить экспериментальные стратегии:

```text
DIRECT
PUNCTUATION
SAME_SPEAKER_CONTEXT
SYNTHETIC_CONTEXT
BATCH_AND_CROP
```

После benchmark оставить только реально полезные стратегии.

---

# 5. DIRECT

Текущая генерация:

```text
"Привет!"
        ↓
TTS
        ↓
audio
```

Она остаётся baseline.

Все улучшения сравнивать с ней.

---

# 6. PUNCTUATION strategy

Проверить влияние нормализованной punctuation.

Например:

```text
Привет!
Привет.
Привет…
```

и:

```text
Как дела?
Как дела?..
```

Но не делать автоматическую замену `!` → `.` или `?` → `.` без доказанного улучшения.

Пунктуация несёт интонацию.

Задача benchmark — определить реальное поведение каждого engine.

---

# 7. Synthetic Context

Для короткой фразы можно экспериментально добавить нейтральный контекст:

```text
target:
Привет!

synthesis input:
Хорошо. Привет! Хорошо.
```

или другой нейтральный carrier.

После синтеза оставить только audio target-фразы.

ВАЖНО:

не использовать случайные слова без benchmark.

Synthetic carrier должен:

* минимально влиять на эмоциональную окраску;
* хорошо работать с русским языком;
* не менять смысл target;
* использоваться только если качество реально выше.

---

# 8. Dialogue Context

Для Dialogue предпочтительнее использовать настоящий контекст проекта.

Например:

```text
previous:
— Ты уже пришёл?

target:
— Да.

next:
— Тогда начинаем.
```

Контекст:

```text
Ты уже пришёл? Да. Тогда начинаем.
```

может дать модели значительно больше информации о просодии слова `Да.`.

Однако учитывать speakers.

Если соседняя реплика принадлежит другому голосу, нельзя без проверки заставлять текущий voice произносить весь dialogue context.

Поэтому отдельно исследовать:

```text
same-speaker context
cross-speaker textual context
synthetic neutral context
```

Не предполагать, что cross-speaker synthesis безопасен.

---

# 9. Preferred approach: same-speaker context

Если рядом существует предыдущая или следующая реплика того же speaker, использовать её как candidate context.

Пример:

```text
Анна: Я только что вернулась домой.
Борис: Правда?
Анна: Да. Очень устала.
```

Для:

```text
Да. Очень устала.
```

контекст Анны может включать её предыдущую реплику.

При этом сохранять distinction:

```text
context_text
target_text
```

---

# 10. Batch short adjacent replicas

Исследовать особенно важный вариант.

Если подряд идут несколько коротких реплик ОДНОГО speaker и между ними нет реплик другого speaker:

```text
Привет!
Как дела?
Всё хорошо?
```

вместо трёх независимых inference:

```text
TTS("Привет!")
TTS("Как дела?")
TTS("Всё хорошо?")
```

попробовать:

```text
TTS("Привет! Как дела? Всё хорошо?")
```

Это потенциально может дать модели значительно более естественный контекст.

После этого разделить audio обратно на логические replicas.

Не объединять реплики разных speakers.

---

# 11. Не делать naive audio crop

Разделение результата:

```text
Привет! | Как дела? | Всё хорошо?
```

нельзя делать по заранее рассчитанной длительности текста.

Нужны реальные границы.

Исследовать:

```text
silence detection
forced alignment
Whisper timestamps
word timestamps
```

Если в проекте уже есть Whisper/QA infrastructure, сначала проверить возможность повторного использования существующей инфраструктуры.

Не добавлять новую тяжёлую dependency без необходимости.

---

# 12. Short Utterance Context Window

Создать структуру:

```python
ShortUtteranceContext(
    target_replica_id,
    target_text,
    previous_text,
    next_text,
    same_speaker_previous,
    same_speaker_next,
    strategy,
)
```

Context builder не должен изменять исходный текст проекта.

---

# 13. Не записывать synthetic context в final_text

Очень важно сохранить семантику существующего preprocessing pipeline.

Если:

```text
final_text = "Привет!"
```

то `final_text` должен оставаться:

```text
Привет!
```

а не:

```text
Хорошо. Привет! Хорошо.
```

Добавить отдельные runtime поля:

```text
tts_target_text
tts_context_text
tts_synthesis_text
short_utterance_strategy
```

Таким образом:

```text
prepared final_text
        ↓
Short Utterance Strategy
        ↓
tts_synthesis_text
        ↓
engine
```

---

# 14. Не нарушить правило prepared dialogue

Short Utterance Strategy должна работать ПОСЛЕ:

```text
dialogue parsing
↓
speaker detection
↓
voice mapping
↓
normalization
↓
ё
↓
dictionary
↓
stress/accent processing
↓
final_text
```

То есть нельзя обходить уже подготовленный pronunciation pipeline.

Контекст также должен проходить корректную engine-specific preparation.

---

# 15. Особое внимание F5 stress markup

Если F5 получает ударения:

```text
final_text
```

с `+`, context-assisted input должен сохранить корректную разметку.

Нельзя получить ситуацию:

```text
target → accentized
context → raw
```

Весь реально передаваемый F5 текст должен соответствовать требованиям F5 pipeline.

Для XTTS не добавлять F5-specific `+`.

---

# 16. Reference text/audio

Отдельно проверить гипотезу, что проблема коротких фраз связана не с target text, а с conditioning/reference.

Benchmark:

```text
same target
same seed
same voice
same engine
```

при разных reference samples.

Особенно проверить:

```text
"Привет!"
"Как дела?"
"Да."
```

Некоторые reference samples могут давать значительно более стабильную просодию коротких utterances.

---

# 17. Short Voice Reference Benchmark

Для каждого voice preset дать возможность прогнать:

```text
Привет!
Как дела?
Спасибо.
Да.
Нет.
Что случилось?
```

и получить benchmark variants.

Не менять автоматически reference audio пользователя.

Цель — определить, связан ли дефект с конкретным reference.

---

# 18. Seed stability

Для коротких фраз влияние seed может быть непропорционально большим.

Провести benchmark:

```text
same short phrase
same voice
same params

seed 1
seed 2
seed 3
seed 4
seed 5
```

Сравнить стабильность.

Не решать проблему генерацией десятков вариантов.

Допустимо впоследствии реализовать ограниченный механизм:

```text
short phrase
→ generate N candidates
→ QA
→ select acceptable candidate
```

только если benchmark покажет необходимость.

---

# 19. Short Utterance Retry

Для VERY_SHORT допускается отдельная policy.

Например:

```text
attempt 1
↓
Smart QA
↓
bad transcription / invalid duration
↓
attempt 2 with different seed
```

Ограничить:

```text
max_short_utterance_attempts
```

например 2–3.

Не создавать бесконечную генерацию.

---

# 20. QA для коротких фраз

Обычный WER может плохо работать на:

```text
Да.
Нет.
Что?
```

Одна ошибка означает огромный WER.

Добавить short-specific QA.

Проверять:

```text
expected normalized text
ASR transcription
duration
empty/silent audio
unexpected repetition
unexpected extra words
```

Особенно обнаруживать случаи:

```text
target:
"Да."

audio transcription:
"Да, да, да..."
```

или:

```text
target:
"Привет!"

audio:
silence / unintelligible output
```

---

# 21. Минимальная duration sanity check

Добавить sanity check короткого audio.

Не задавать один жёсткий minimum duration для всех реплик.

Но обнаруживать явно аномальные результаты:

```text
0 ms
50 ms
почти полностью silence
```

Duration threshold должен учитывать количество phonemes/слов либо быть достаточно консервативным.

---

# 22. Не растягивать короткие фразы искусственно

Не делать глобально:

```text
"Да." → "Да-а-а."
```

или:

```text
"Привет!" → "Привет, привет!"
```

Это изменяет произносимый текст.

Любой carrier/context должен существовать только на synthesis layer и не менять target semantics.

---

# 23. UI

Для пользователя основная логика должна работать автоматически.

В Settings/Advanced можно добавить:

```text
Short phrase optimization

[✓] Enable

Very short: ≤ 2 words
Short: ≤ 5 words

Strategy:
Auto
Direct
Context-assisted
```

Основной рекомендуемый режим:

```text
Auto
```

---

# 24. Replica metadata

Сохранять диагностически:

```text
utterance_class
short_utterance_strategy
synthesis_text_hash
context_source
attempt_count
seed
```

Не обязательно сохранять synthetic context навечно, если архитектура этого не требует, но synthesis должен быть воспроизводим.

---

# 25. Ключевой A/B benchmark

Создать automated/manual benchmark:

```text
A = current DIRECT
B = context-assisted
```

Фразы:

```text
Привет!
Как дела?
Спасибо!
Да.
Нет.
Хорошо.
Я понял.
Что?
Почему?
До встречи!
```

Голоса:

```text
минимум 2 F5 voices
минимум 1 XTTS voice, если XTTS доступен
```

Для каждого:

```text
same voice
same reference
same engine params
same seed where applicable
```

---

# 26. Не использовать субъективную метрику как единственный критерий

Автоматически собрать:

```text
ASR result
text match
duration
silence ratio
generation time
strategy
seed
```

Но естественность оценить также реальным listening A/B.

Не придумывать "naturalness score", если модель/метрика реально не существует.

---

# 27. Обязательные тесты

Добавить:

```text
test_short_utterance_detector

test_one_word_phrase_is_very_short
test_two_word_phrase_is_very_short
test_long_sentence_is_normal

test_short_strategy_does_not_modify_source_text
test_short_strategy_does_not_modify_final_text

test_short_strategy_uses_prepared_text
test_f5_short_context_preserves_accent_markup
test_xtts_short_context_has_no_f5_markup

test_same_speaker_context_can_be_selected
test_different_speaker_is_not_merged_as_same_speaker

test_adjacent_short_same_speaker_replicas_can_be_grouped
test_adjacent_different_speakers_are_not_grouped

test_short_strategy_can_fallback_to_direct
test_normal_text_uses_existing_pipeline

test_short_retry_is_limited
test_short_failed_qa_can_retry
test_short_successful_qa_does_not_retry

test_short_audio_empty_result_is_rejected
test_short_audio_repetition_is_detected

test_short_optimization_does_not_change_long_text_output
```

---

# 28. Regression requirement

Особенно важно:

**не ухудшить длинные фразы ради коротких.**

До изменений зафиксировать baseline нескольких длинных фраз.

После реализации проверить:

```text
NORMAL utterance
→ same preprocessing
→ same engine parameters
→ same synthesis path
```

если short optimization не применяется.

Добавить:

```text
test_normal_utterance_bypasses_short_optimizer
```

---

# 29. Реализация по фазам

## Phase 1 — Benchmark

Сначала никаких production heuristics.

Создать short-phrase benchmark и определить:

```text
какие engines страдают
какие voices страдают
влияние reference
влияние seed
влияние punctuation
влияние context
```

### Expected result

Есть воспроизводимый отчёт, показывающий, при каких условиях качество деградирует.

---

## Phase 2 — Detector

Добавить `UtteranceClass`.

### Tests

```text
test_short_utterance_detector
test_one_word_phrase_is_very_short
test_two_word_phrase_is_very_short
test_long_sentence_is_normal
```

### Expected result

Короткие фразы надёжно определяются, NORMAL pipeline не изменён.

---

## Phase 3 — Context Builder

Добавить context-assisted synthesis abstraction.

### Tests

```text
test_short_strategy_does_not_modify_final_text
test_short_strategy_uses_prepared_text
test_same_speaker_context_can_be_selected
test_different_speaker_is_not_merged_as_same_speaker
```

### Expected result

Можно построить дополнительный synthesis context без изменения текста проекта.

---

## Phase 4 — Engine-specific experiments

Провести реальные A/B для F5 и XTTS.

Не включать strategy глобально до результатов benchmark.

### Expected result

Для каждого engine определена отдельная оптимальная short-phrase policy.

Допускается:

```text
F5 → context-assisted
XTTS → direct
```

или наоборот.

Не требовать одной стратегии для всех engines.

---

## Phase 5 — QA + Retry

Добавить short-specific QA.

### Tests

```text
test_short_failed_qa_can_retry
test_short_successful_qa_does_not_retry
test_short_retry_is_limited
test_short_audio_empty_result_is_rejected
```

### Expected result

Очевидно неудачная короткая генерация может быть автоматически повторена ограниченное количество раз.

---

## Phase 6 — Dialogue integration

Интегрировать оптимизацию после Dialogue Preparation Pipeline.

### Tests

```text
test_short_strategy_uses_prepared_text
test_f5_short_context_preserves_accent_markup
test_xtts_short_context_has_no_f5_markup
test_normal_utterance_bypasses_short_optimizer
```

### Expected result

Pronunciation/dictionary/stress остаются единым source of truth.

---

# 30. Очень важный эксперимент: contextual prefix/suffix

ИИ-агент должен отдельно исследовать этот вариант.

Для:

```text
Привет!
```

сравнить:

```text
A:
Привет!

B:
Хорошо. Привет!

C:
Привет! Хорошо.

D:
Хорошо. Привет! Хорошо.
```

Для:

```text
Как дела?
```

аналогично.

Сравнить не только pronunciation, но:

```text
attack начала фразы
intonation
tempo
voice stability
ending
artifacts
```

Если prefix значительно улучшает начало короткой фразы, это сильный сигнал, что проблема связана с отсутствием preceding context.

---

# 31. Но carrier cropping не внедрять без надёжного alignment

Если лучший результат получается:

```text
Хорошо. Привет!
```

не пытаться вырезать `Хорошо.` приблизительно по времени.

Production implementation разрешён только при надёжном определении границы.

Если alignment недостаточно стабилен:

```text
fallback → DIRECT
```

лучше, чем случайно оставить часть carrier в итоговом WAV.

---

# 32. Альтернативная стратегия — contextual batching

Для Dialogue особенно тщательно проверить batching.

Если структура позволяет:

```text
Speaker A:
Привет!
Как дела?
Давно не виделись.
```

предпочтительный эксперимент:

```text
Привет! Как дела? Давно не виделись.
```

Это лучше synthetic carrier, потому что модель получает настоящий авторский текст.

Но metadata должны сохранить три логические replicas.

---

# 33. Acceptance Criteria

Задача считается выполненной только если:

1. существует воспроизводимый short-phrase benchmark;
2. определено, является проблема engine-specific или общей;
3. короткие utterances автоматически классифицируются;
4. long-form pipeline не изменён;
5. `source_text` и `final_text` не мутируются synthetic context;
6. context-assisted synthesis является отдельным runtime layer;
7. F5/XTTS могут иметь разные policies;
8. short-specific QA обнаруживает очевидные failures;
9. retry ограничен;
10. соседние разные speakers не объединяются;
11. подготовленные ударения/словарь не обходятся;
12. проведён listening A/B;
13. выбранная production strategy показала реальное улучшение на benchmark;
14. все regression tests проходят.

---

# 34. Что НЕ делать

Не начинать задачу с:

```text
увеличим паузы
заменим все ! на .
добавим многоточия
снизим speed
добавим пробелы
повторим слово
```

как с универсального решения.

Такие изменения допустимы только как benchmark variants.

Также не создавать отдельный preprocessing pipeline специально для коротких реплик.

Правильная архитектура:

```text
Prepared Replica
      ↓
final_text
      ↓
Utterance Classifier
      ↓
NORMAL ─────────────→ existing synthesis
      │
      └ SHORT
          ↓
   Short Utterance Strategy
          ↓
   engine-specific synthesis
          ↓
          QA
          ↓
        audio
```

Главный принцип: **короткая фраза должна получить больше полезного контекста для генерации, но пользователь в итоговом WAV должен услышать только исходную подготовленную реплику.**
