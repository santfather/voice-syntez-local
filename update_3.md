# UPDATE 3 --- Russian Dialogue Analyzer + Voice Reference Profiles + Automatic Prosody Routing

## 0. Цель задачи

Расширить уже существующий **Local LLM Russian Linguistic Analyzer** до
более высокого уровня:

``` text
Russian Dialogue Analyzer
```

Новый анализатор должен сохранить все существующие функции
лингвистического анализа русского текста и дополнительно анализировать
каждую реплику диалога с точки зрения:

-   смысла в контексте сцены;
-   типа высказывания;
-   эмоции;
-   требуемой интонации;
-   степени эмоциональной интенсивности;
-   зависимости от соседних реплик;
-   пригодности контекста для short-utterance synthesis.

Одновременно существующие записи одного голоса, сделанные по
`RECORD_PHRASES`, необходимо превратить в **Reference Profiles** этого
же `voice_id`.

Итоговый пользовательский сценарий:

``` text
Пользователь вставляет диалог
        ↓
Parser разбивает его на реплики
        ↓
существующий linguistic preprocessing
        ↓
Russian Dialogue Analyzer анализирует сцену
        ↓
для каждой replica появляется Auto-интонация
        ↓
ProsodyResolver выбирает reference profile того же голоса
        ↓
пользователь при необходимости меняет профиль dropdown'ом
        ↓
Short Dialogue Planner выбирает стратегию синтеза
        ↓
F5 / XTTS получает final_text + правильный reference
        ↓
QA
        ↓
готовая реплика
```

Ключевой принцип:

> LLM определяет смысл и prosody intent.\
> Backend выбирает конкретный reference.\
> TTS получает чистый произносимый текст.\
> Пользователь всегда может переопределить автоматический выбор.

------------------------------------------------------------------------

# 1. Сначала провести аудит текущей реализации

Перед написанием кода агент обязан изучить актуальный проект и не
дублировать уже реализованные подсистемы.

Найти и документировать:

1.  Local LLM Russian Linguistic Analyzer.
2.  Ollama client/runner.
3.  текущую LLM schema.
4.  prompt versioning.
5.  LLM cache/invalidation.
6.  memory policy / HeavyGate.
7.  Project / Replica persistence.
8.  Dialogue parser.
9.  speaker mapping.
10. common text preprocessing.
11. ё-restoration.
12. pronunciation dictionary.
13. RUAccent/F5 finalization.
14. `/api/text/preview`.
15. short-utterance layer.
16. QA.
17. voices storage.
18. запись `RECORD_PHRASES`.
19. фактическое хранение WAV каждого recorded phrase.
20. regenerate/variants/takes.
21. diagnostics.
22. UI вкладки Dialogue.

До реализации составить краткую внутреннюю карту:

``` text
current component
→ reused as-is
→ extended
→ migrated
→ deprecated
```

Не создавать параллельный `EmotionAnalyzer`, если существующий LLM
Analyzer можно расширить.

------------------------------------------------------------------------

# 2. Сохранить существующие гарантии

Новая функция не должна ломать:

-   Continuous Text;
-   существующий pronunciation analysis;
-   Dictionary;
-   `ё`;
-   RUAccent;
-   F5-specific `+`;
-   XTTS;
-   Projects;
-   existing voices;
-   existing jobs;
-   regenerate;
-   variants;
-   strict/smart QA;
-   short utterance optimization;
-   preview;
-   memory guard;
-   работу без Ollama.

Все старые проекты и голоса должны открываться после миграции.

------------------------------------------------------------------------

# 3. Целевая архитектура

``` text
SOURCE DIALOGUE
      ↓
Dialogue Parser
      ↓
Speaker Mapping
      ↓
Deterministic Russian Preprocessing
      ↓
Existing Linguistic Analysis
      ↓
Russian Dialogue Analyzer
      ├── linguistic annotations
      ├── semantic meaning
      ├── dialogue act
      ├── emotion
      ├── prosody intent
      ├── intensity
      └── context dependency
      ↓
Prepared Replica
      ↓
Manual Prosody Override
      ↓
ProsodyResolver
      ↓
ReferenceProfileResolver
      ↓
Short Dialogue Synthesis Planner
      ↓
Engine-specific finalization
      ↓
F5 / XTTS
      ↓
RAW AUDIO
      ↓
Postprocessing
      ↓
QA
      ↓
Replica Take
      ↓
Dialogue Merge
```

------------------------------------------------------------------------

# 4. Не превращать emotion в текстовые TTS-теги

Запрещено отправлять в F5/XTTS:

``` text
[восторг] Мы победили!
[вопрос] Это проблема?
[ирония] Ну конечно.
<emotion="fear">Кто там?</emotion>
```

Не добавлять служебные маркеры в:

``` text
source_text
normalized_text
yo_text
dictionary_text
accentized_text
final_text
```

Допустимо показывать UI:

``` text
[Вопрос] Это проблема?
```

но `[Вопрос]` не является частью TTS input.

Правильно:

``` text
final_text = "Это проблема?"

prosody_detected = QUESTION
prosody_override = null
prosody_effective = QUESTION
reference_profile_id = "..."
```

------------------------------------------------------------------------

# 5. Расширить Local LLM Russian Linguistic Analyzer

Не создавать второй независимый LLM pipeline.

Текущий анализатор расширить до:

``` text
Russian Dialogue Analyzer
```

Он должен уметь работать в двух режимах:

``` text
LINGUISTIC
DIALOGUE
```

или через единую schema с optional dialogue fields.

Для Continuous Text prosody-анализ может отсутствовать или работать
только при необходимости.

Для Dialogue --- prosody fields обязательны в structured result, если
LLM analysis успешно выполнен.

------------------------------------------------------------------------

# 6. LLM должна анализировать сцену, а не изолированную реплику

Нельзя определять emotion только по:

``` text
текущему предложению
?
!
```

Например:

``` text
— Мы выиграли миллион.
— Правда?
```

и:

``` text
— Я опять забыл документы.
— Правда?
```

семантически различаются.

Поэтому LLM должна получать context window.

Минимально:

``` text
previous replicas
current replica
next replicas
speaker IDs
```

Для короткой сцены можно анализировать всю сцену одним request.

Для длинного диалога --- sliding semantic windows.

------------------------------------------------------------------------

# 7. Не делать один LLM request на каждую replica

Это важно для:

-   latency;
-   памяти;
-   стабильности;
-   согласованности анализа сцены.

Предпочтительно:

``` text
one scene/window
→ one LLM request
→ results[] for multiple replica_id
```

Пример:

``` json
{
  "replicas": [
    {
      "replica_id": 12,
      "prosody": {}
    },
    {
      "replica_id": 13,
      "prosody": {}
    }
  ]
}
```

------------------------------------------------------------------------

# 8. Structured output Russian Dialogue Analyzer

Расширить существующую schema.

Рекомендуемая логическая структура:

``` json
{
  "replica_id": 12,
  "linguistic_issues": [],
  "dialogue": {
    "dialogue_act": "QUESTION",
    "context_dependency": "HIGH"
  },
  "prosody": {
    "emotion": "QUESTION",
    "recommended_profile": "QUESTION",
    "intensity": 0.55,
    "pace": "NORMAL",
    "confidence": 0.94
  }
}
```

Точные имена привести к текущему style проекта.

------------------------------------------------------------------------

# 9. Dialogue Act

Ввести отдельное понятие:

``` text
dialogue_act
```

Это не то же самое, что emotion.

Начальный controlled vocabulary:

``` text
STATEMENT
QUESTION
ANSWER
REQUEST
COMMAND
REACTION
COMPLIMENT
REFUSAL
AGREEMENT
DISAGREEMENT
GREETING
FAREWELL
WARNING
EXCLAMATION
ENUMERATION
OTHER
```

Не нужно превращать его в огромную ontology.

Он нужен для contextual reasoning и диагностики.

------------------------------------------------------------------------

# 10. Prosody Profile

Production-набор должен соответствовать реально записанным reference
profiles.

Начальный набор:

``` text
NEUTRAL
CALM
QUESTION
NEUTRAL_QUESTION
EXCLAMATION
DELIGHT
SAD_SYMPATHETIC
IRONIC
STRICT
ENUMERATION
EXCITED
```

Архитектура должна позволять позднее добавить:

``` text
SURPRISE
FEAR
ANGRY
WHISPER
CONFIDENT
SOFT
```

без большой миграции.

------------------------------------------------------------------------

# 11. Intensity --- отдельное поле

Не плодить:

``` text
SLIGHT_DELIGHT
MEDIUM_DELIGHT
STRONG_DELIGHT
```

Использовать:

``` text
profile = DELIGHT
intensity = 0.0 .. 1.0
```

В первой production-версии `intensity` может быть:

-   metadata;
-   UI hint;
-   benchmark field.

Не обязательно сразу пытаться преобразовать intensity в engine params.

Запрещено автоматически менять CFG/temperature/speed на основании
intensity без benchmark.

------------------------------------------------------------------------

# 12. Pace --- только semantic hint

Допустимые значения:

``` text
SLOW
NORMAL
FAST
```

Пока это semantic metadata.

Не превращать автоматически:

``` text
FAST → speed=1.4
```

без отдельного benchmark.

Для профиля `EXCITED` reference audio уже может содержать ускоренную
манеру речи.

------------------------------------------------------------------------

# 13. Confidence

LLM должна возвращать:

``` text
confidence = 0..1
```

Но confidence самой LLM не считать доказательством правильности.

Использовать его для UI:

``` text
Авто: Ирония · 91%
```

или:

``` text
Авто: Нейтрально · низкая уверенность
```

Низкая confidence не должна блокировать render.

------------------------------------------------------------------------

# 14. Context Dependency

Добавить:

``` text
LOW
MEDIUM
HIGH
```

Пример:

``` text
"Здравствуйте."
→ LOW

"Правда?"
→ HIGH

"Это проблема?"
→ HIGH
```

Это поле можно использовать Short Dialogue Planner как дополнительный
hint.

Но LLM не должна непосредственно решать, как резать WAV.

------------------------------------------------------------------------

# 15. Существующие RECORD_PHRASES

В проекте уже используются следующие записи:

``` javascript
const RECORD_PHRASES = [
  {
    label: 'Вопрос и утверждение',
    text: 'Добрый вечер. Мы договаривались встретиться у метро, но я вас так и не увидел. Вы точно получили моё сообщение?',
  },
  {
    label: 'Восклицания, много шипящих',
    text: 'Осторожно, здесь очень скользко! Я чуть не упал, когда выбегал из подъезда. И это уже третий раз за неделю.',
  },
  {
    label: 'Спокойная просьба (короче)',
    text: 'Дайте пройти, пожалуйста. Я вас совсем не знаю, и мне нечего вам сказать.',
  },
  {
    label: 'Только вопросы',
    text: 'Ты уверен, что мы правильно свернули? Кажется, этот поворот был раньше. Может, спросим дорогу у кого-нибудь?',
  },
  {
    label: 'Сложные сочетания согласных',
    text: 'Съешь ещё этих мягких французских булочек, а потом расскажешь, как прошла твоя поездка в Ярославль.',
  },
  {
    label: 'Радость, восторг',
    text: 'Ты представляешь, у нас всё получилось! Ребёнок сам застегнул все пуговицы, а потом ещё и спел мне целую песню про самолёт.',
  },
  {
    label: 'Огорчение, сочувствие',
    text: 'Мне так жаль, что всё так обернулось. Она ждала этот день, а теперь идёт домой одна и не знает, что делать дальше.',
  },
  {
    label: 'Лёгкая ирония',
    text: 'Ну конечно, именно сегодня лифт снова сломан. Как будто он специально ждёт, пока актёр из соседней квартиры опять устроит репетицию.',
  },
  {
    label: 'Строгий тон, короткий приказ',
    text: 'Немедленно отдайте мне ключи от квартиры. Я всё сказал предельно ясно, и вы прекрасно понимаете, о чём идёт речь.',
  },
  {
    label: 'Перечисление, ровный ритм',
    text: 'На столе лежали ключи, кошелёк, блокнот, ручка и ещё какие-то бумаги, которые я так и не успел разобрать.',
  },
  {
    label: 'Быстрая, взволнованная речь',
    text: 'Скорее, мы опаздываем! Автобус уже подъезжает, а нам ещё нужно забрать вещи, запереть дверь и найти, куда делись ключи.',
  },
];
```

------------------------------------------------------------------------

# 16. RECORD_PHRASES → Reference Profiles

Не считать 11 записей 11 отдельными голосами.

Если все записаны Марией:

``` text
voice_id = MARIA
```

для всех.

Но каждая запись становится отдельным:

``` text
ReferenceProfile
```

Рекомендуемый mapping:

``` text
Вопрос и утверждение
→ NEUTRAL_QUESTION

Восклицания, много шипящих
→ EXCLAMATION

Спокойная просьба (короче)
→ CALM

Только вопросы
→ QUESTION

Сложные сочетания согласных
→ NEUTRAL

Радость, восторг
→ DELIGHT

Огорчение, сочувствие
→ SAD_SYMPATHETIC

Лёгкая ирония
→ IRONIC

Строгий тон, короткий приказ
→ STRICT

Перечисление, ровный ритм
→ ENUMERATION

Быстрая, взволнованная речь
→ EXCITED
```

Это initial mapping.

Перед окончательной фиксацией проверить его через benchmark.

------------------------------------------------------------------------

# 17. Не выдумывать SURPRISE/FEAR profile

Если в текущих 11 записях нет специально записанного:

``` text
SURPRISE
FEAR
```

не помечать произвольно существующий WAV как `FEAR`.

LLM может определить:

``` text
semantic emotion = FEAR
```

а `ProsodyResolver` должен подобрать ближайший разрешённый reference
fallback.

Например:

``` text
FEAR
→ EXCITED
```

только если benchmark подтвердил, что это приемлемо.

Иначе:

``` text
FEAR
→ NEUTRAL
```

с явным:

``` text
fallback_used = true
```

------------------------------------------------------------------------

# 18. Data model ReferenceProfile

Рекомендуемая сущность:

``` text
ReferenceProfile

id
voice_id
profile_key
label
audio_path
ref_text

quality_status
transcription_status
duration_sec

engine_compatibility

source_record_phrase_id
is_default
enabled

created_at
updated_at
```

При необходимости:

``` text
metadata_json
```

------------------------------------------------------------------------

# 19. Voice → ReferenceProfile relation

``` text
Voice 1:N ReferenceProfile
```

Например:

``` text
Maria
 ├── neutral
 ├── calm
 ├── question
 ├── neutral_question
 ├── exclamation
 ├── delight
 ├── sad_sympathetic
 ├── ironic
 ├── strict
 ├── enumeration
 └── excited
```

Не копировать настройки voice для каждого profile.

------------------------------------------------------------------------

# 20. Миграция существующих voices

Существующий голос с:

``` text
ref_audio
ref_text
```

должен продолжить работать.

При migration:

``` text
existing ref
→ ReferenceProfile(profile_key=NEUTRAL)
```

или создать compatibility layer.

Требования:

-   старый voice ID не меняется;
-   существующие projects открываются;
-   старые jobs/takes читаются;
-   API не падает;
-   fallback на старый ref возможен.

------------------------------------------------------------------------

# 21. Проверить фактическое хранение RECORD_PHRASES

Это обязательная audit-задача.

Нужно определить:

``` text
11 отдельных WAV
```

или:

``` text
один объединённый WAV
```

или другой формат.

Если записи уже отдельные:

``` text
reuse files
→ attach as ReferenceProfiles
```

Если сейчас они объединяются:

необходимо сохранить отдельные оригинальные recordings или сегменты как
отдельные reference assets.

Не строить emotion routing поверх одного длинного WAV без
доказательства, что движок получает нужный эмоциональный участок.

------------------------------------------------------------------------

# 22. Reference quality

Каждый ReferenceProfile должен пройти существующую проверку:

``` text
audio validity
duration
transcription/ref_text match
quality warnings
```

Профиль с плохим quality status не должен автоматически выигрывать
только из-за нужной emotion.

------------------------------------------------------------------------

# 23. ProsodyResolver

Создать отдельный deterministic backend component:

``` python
resolve_prosody(
    voice_id,
    engine_id,
    prosody_effective,
    dialogue_act,
    intensity,
) -> ResolvedProsody
```

LLM не должна выбирать filesystem path.

------------------------------------------------------------------------

# 24. ResolvedProsody

Результат:

``` text
requested_profile
resolved_profile

reference_profile_id
ref_audio
ref_text

fallback_used
fallback_reason

voice_id
engine_id
```

------------------------------------------------------------------------

# 25. Resolver priority

Пример:

``` text
requested exact profile
        ↓
profile exists?
        ↓ yes
quality acceptable?
        ↓ yes
engine compatible?
        ↓ yes
use it
```

Иначе:

``` text
configured fallback
→ NEUTRAL
```

Никаких случайных выборов.

------------------------------------------------------------------------

# 26. Resolver fallback table

Fallback table должна быть:

-   централизованной;
-   конфигурируемой;
-   документированной;
-   покрытой тестами.

Начальный безопасный вариант:

``` text
NEUTRAL → NEUTRAL
CALM → CALM → NEUTRAL
QUESTION → QUESTION → NEUTRAL_QUESTION → NEUTRAL
NEUTRAL_QUESTION → NEUTRAL_QUESTION → QUESTION → NEUTRAL
EXCLAMATION → EXCLAMATION → EXCITED → NEUTRAL
DELIGHT → DELIGHT → EXCLAMATION → NEUTRAL
SAD_SYMPATHETIC → SAD_SYMPATHETIC → CALM → NEUTRAL
IRONIC → IRONIC → NEUTRAL
STRICT → STRICT → NEUTRAL
ENUMERATION → ENUMERATION → NEUTRAL
EXCITED → EXCITED → EXCLAMATION → NEUTRAL
```

Но fallback beyond NEUTRAL должен быть подтверждён benchmark.

До benchmark допустим более консервативный:

``` text
missing exact profile → NEUTRAL
```

------------------------------------------------------------------------

# 27. Не использовать reference другого voice

Абсолютный invariant:

``` text
resolved_reference.voice_id == replica.voice_id
```

Если у Марии нет `IRONIC`, нельзя взять `IRONIC` у Артёма.

Fallback только внутри того же voice.

------------------------------------------------------------------------

# 28. Engine compatibility

ReferenceProfile может иметь:

``` text
engine_compatibility
```

Например:

``` text
["f5", "xtts"]
```

или:

``` text
["f5"]
```

Resolver должен учитывать фактический engine выбранного voice.

------------------------------------------------------------------------

# 29. F5 и XTTS benchmark отдельно

Нельзя предполагать, что prosody transfer одинаков.

Из текущего проекта уже известно, что short-utterance strategies ведут
себя по-разному на F5 и XTTS.

Поэтому Reference Profile benchmark обязательно разделять:

``` text
F5
XTTS
XTTS Banana, если реально используется
```

------------------------------------------------------------------------

# 30. Reference Prosody Transfer Benchmark --- обязательная фаза до production routing

Создать отдельный benchmark.

Цель:

> проверить, насколько смена reference profile реально меняет просодию
> target, не ухудшая текст и voice identity.

------------------------------------------------------------------------

# 31. Benchmark target corpus

Минимум 30 реплик.

Категории:

## Neutral

``` text
Я пришёл немного раньше.
Сегодня было довольно тепло.
```

## Question

``` text
Ты уверен?
Это правда?
Что произошло?
```

## Delight

``` text
У нас получилось!
Это потрясающе!
```

## Sad/Sympathetic

``` text
Мне очень жаль.
Я понимаю, как тебе тяжело.
```

## Irony

``` text
Ну конечно.
Очень вовремя.
```

## Strict

``` text
Стойте здесь.
Отдайте ключи.
```

## Enumeration

``` text
Нужны ключи, документы, телефон и деньги.
```

## Excited

``` text
Скорее, мы опаздываем!
Он уже здесь!
```

## Ambiguous/context-dependent

``` text
Правда?
Вот как.
Ну конечно.
```

------------------------------------------------------------------------

# 32. Benchmark matrix

Для одного target:

фиксировать:

``` text
voice
engine
seed
speed
engine params
final_text
```

Менять только:

``` text
ReferenceProfile
```

Сравнить минимум:

``` text
NEUTRAL
QUESTION
DELIGHT
IRONIC
STRICT
EXCITED
```

где доступны.

------------------------------------------------------------------------

# 33. Benchmark metrics

Автоматические:

``` text
ASR transcript
WER
first_word_ok
last_word_ok
extra_words
repetition
duration
generation_time
QA status
```

Metadata:

``` text
voice_id
engine
profile
seed
reference_profile_id
```

------------------------------------------------------------------------

# 34. Listening review benchmark

Для prosody обязательно ручное прослушивание.

Поля:

``` text
target_text
expected_intent
reference_profile

text_complete: pass/fail
clarity: pass/fail
voice_identity: pass/fail
prosody_match: pass/fail
overacting: yes/no
artifacts: yes/no
notes
```

Не пытаться заменить это Whisper.

------------------------------------------------------------------------

# 35. Benchmark gate

Production automatic routing разрешить только после ответа:

``` text
какие profiles реально дают устойчивый перенос на F5?
какие на XTTS?
какие ухудшают дикцию?
какие нестабильны?
```

Если профиль не доказал пользу:

``` text
enabled_for_auto = false
```

но можно оставить его доступным экспериментально/manual.

------------------------------------------------------------------------

# 36. Replica prosody fields

Добавить к persistent replica:

``` text
prosody_detected
prosody_confidence

prosody_override
prosody_effective

dialogue_act
context_dependency
intensity
pace

reference_profile_id
reference_profile_key

reference_fallback_used
reference_fallback_reason
```

------------------------------------------------------------------------

# 37. Auto + manual override

Правило:

``` python
prosody_effective = (
    prosody_override
    if prosody_override is not None
    else prosody_detected
)
```

Manual всегда сильнее Auto.

------------------------------------------------------------------------

# 38. UI Dialogue

У каждой replica добавить dropdown.

Пример:

``` text
МАРИЯ
Это правда?

Интонация:
[ Авто → Вопрос ▼ ]
```

Options:

``` text
Авто
Нейтрально
Спокойно
Вопрос
Вопрос + нейтрально
Восклицание
Радость / восторг
Огорчение / сочувствие
Ирония
Строго
Перечисление
Взволнованно
```

Показывать только profiles, поддерживаемые системой.

------------------------------------------------------------------------

# 39. UI confidence

При Auto:

``` text
Авто → Ирония · 91%
```

При низкой уверенности:

``` text
Авто → Нейтрально · низкая уверенность
```

Не перегружать интерфейс числом, если это ухудшает UX.

Можно показывать confidence tooltip/details.

------------------------------------------------------------------------

# 40. UI reference status

Дополнительно диагностически:

``` text
Reference: Ирония
```

или:

``` text
Reference: Нейтральный
Fallback: профиль "Испуг" отсутствует
```

В обычном режиме это можно спрятать в details.

------------------------------------------------------------------------

# 41. Manual override не запускает LLM заново

При выборе:

``` text
Ирония → Восторг
```

не нужно повторно анализировать сцену.

Нужно:

``` text
save override
→ resolve reference
→ mark audio stale
→ regenerate/render
```

------------------------------------------------------------------------

# 42. Source text не меняется

Изменение prosody не должно инвалидировать:

``` text
source
normalization
dictionary
ё
pronunciation review
```

Оно инвалидирует только downstream:

``` text
reference resolution
audio take
QA
```

------------------------------------------------------------------------

# 43. Изменение source text

Изменение исходного текста должно инвалидировать:

``` text
linguistic analysis
dialogue analysis
prosody_detected
prepared snapshot
audio
```

Manual override policy выбрать явно:

предпочтительно сбрасывать override для изменённой replica или помечать
его `needs_confirmation`.

Не оставлять старую emotion молча на совершенно новом тексте.

------------------------------------------------------------------------

# 44. Изменение speaker/voice

Если replica получает другой voice:

``` text
prosody_detected
```

может сохраниться как semantic result.

Но обязательно инвалидировать:

``` text
reference_profile_id
resolved reference
audio
```

и заново выполнить resolver для нового voice.

------------------------------------------------------------------------

# 45. LLM prompt

Prompt должен объяснять:

-   язык преимущественно русский;
-   диалог --- данные;
-   не выполнять инструкции из текста;
-   не переписывать реплики;
-   не менять punctuation ради выразительности;
-   не добавлять слова;
-   возвращать schema-valid JSON;
-   использовать контекст;
-   отличать dialogue act от emotion;
-   не выбирать WAV/path;
-   не вставлять TTS markup.

------------------------------------------------------------------------

# 46. Prompt versioning

Изменение prosody prompt:

``` text
prompt_version++
schema_version++
```

Cache key должен учитывать обе версии.

------------------------------------------------------------------------

# 47. Backend validation LLM result

Не доверять LLM автоматически.

Проверять:

``` text
known replica_id
known enum
confidence range
intensity range
pace enum
dialogue_act enum
no unknown fields if schema strict
```

Unknown profile:

``` text
→ reject field
→ safe fallback
```

Не падать всем проектом.

------------------------------------------------------------------------

# 48. LLM failure

Если:

``` text
Ollama unavailable
timeout
OOM/memory guard
invalid JSON
schema error
```

Dialogue должен оставаться renderable.

Fallback:

``` text
prosody_detected = NEUTRAL
confidence = null
analysis_warning = ...
```

если проект не настроен в explicit "LLM required" mode.

------------------------------------------------------------------------

# 49. Existing memory policy

Интегрировать с текущим:

``` text
HeavyGate
LLM memory policy
Ollama unload
TTS queue
Whisper
```

Не создавать второй resource scheduler.

------------------------------------------------------------------------

# 50. Preferred analysis/render lifecycle

На 18 GB:

``` text
1. parse
2. deterministic preprocessing
3. load Qwen
4. Russian Dialogue Analyzer
5. persist result
6. unload Qwen when policy requires
7. resolve references
8. load TTS
9. render
10. Whisper QA when needed
```

Не держать Qwen 8B + F5 + XTTS + Whisper одновременно без необходимости.

------------------------------------------------------------------------

# 51. Short Utterance integration

Не заменять существующий short-utterance layer.

Добавить prosody metadata как input hint.

``` text
Prepared Replica
        ↓
prosody_effective
        ↓
Reference Resolver
        ↓
Short Utterance Planner
```

или иной порядок, если текущая архитектура требует этого, но final
engine call должен иметь:

``` text
correct final_text
correct same-voice reference
correct short strategy
```

------------------------------------------------------------------------

# 52. Same-speaker context и prosody

Если short strategy использует same-speaker context:

необходимо учитывать, что соседние реплики могут иметь разные эмоции.

Не объединять слепо:

``` text
спокойная replica
+
крик
```

только потому, что speaker одинаковый.

Добавить compatibility policy.

Например:

``` text
same voice
same engine
prosody compatible
```

------------------------------------------------------------------------

# 53. Не позволять LLM решать audio cropping

LLM может сказать:

``` text
context_dependency = HIGH
```

Но не должна возвращать:

``` text
crop from 0.72 sec to 1.43 sec
```

Audio alignment/cropping остаётся deterministic/audio subsystem concern.

------------------------------------------------------------------------

# 54. Regenerate

При regenerate replica UI должен позволять:

``` text
prosody profile
seed
допустимые engine params
```

Изменение prosody:

``` text
→ new take
```

Старый take сохранить как variant по существующим правилам.

------------------------------------------------------------------------

# 55. Variant metadata

Каждый variant/take должен хранить:

``` text
seed
engine
prosody_effective
reference_profile_id
reference_profile_key
fallback_used
short_strategy
qa
duration
```

Чтобы можно было понять, почему два варианта звучат по-разному.

------------------------------------------------------------------------

# 56. Diagnostics

В diagnostics для replica добавить:

``` text
prosody_detected
prosody_confidence
prosody_override
prosody_effective

dialogue_act
context_dependency
intensity
pace

requested_reference_profile
resolved_reference_profile
fallback_used
fallback_reason

reference_audio identity
reference_text

short_strategy
seed
engine
```

Не хранить лишние binary data в обычном log.

------------------------------------------------------------------------

# 57. Preview

Расширить «Что услышит модель» аккуратно.

Не смешивать linguistic preview с prosody.

Можно добавить отдельный блок:

``` text
Интерпретация диалога

Тип: вопрос
Интонация: вопросительная
Уверенность: высокая
Reference: "Только вопросы"
Fallback: нет
```

Сам `final_text` должен оставаться чистым.

------------------------------------------------------------------------

# 58. API

Сначала расширять существующие project endpoints.

Логически нужны:

``` text
GET dialogue analysis
PATCH replica prosody override
GET voice reference profiles
POST/PUT reference profile mapping
render
regenerate
```

Не создавать десятки узких endpoints без необходимости.

------------------------------------------------------------------------

# 59. Suggested API representation

Replica:

``` json
{
  "id": "...",
  "text": "Это проблема?",
  "prosody": {
    "detected": "QUESTION",
    "confidence": 0.95,
    "override": null,
    "effective": "QUESTION",
    "dialogue_act": "QUESTION",
    "context_dependency": "HIGH",
    "intensity": 0.55
  },
  "reference": {
    "profile_id": "...",
    "profile_key": "QUESTION",
    "fallback_used": false
  }
}
```

------------------------------------------------------------------------

# 60. Unit tests --- ReferenceProfile

Обязательные:

``` text
test_voice_can_have_multiple_reference_profiles
test_reference_profile_requires_voice
test_reference_profile_key_is_valid
test_existing_voice_reference_migrates_to_neutral
test_old_voice_id_is_preserved_after_migration
test_old_project_loads_after_reference_profile_migration

test_reference_profile_ref_text_is_preserved
test_reference_profile_quality_status_is_preserved
test_disabled_reference_profile_is_not_auto_selected
test_bad_quality_reference_is_not_auto_selected
```

------------------------------------------------------------------------

# 61. Unit tests --- ProsodyResolver

``` text
test_exact_profile_is_selected
test_reference_is_always_same_voice
test_other_voice_profile_is_never_selected

test_question_falls_back_to_neutral_question
test_missing_profile_falls_back_to_neutral
test_fallback_is_recorded
test_fallback_reason_is_recorded

test_disabled_profile_is_skipped
test_incompatible_engine_profile_is_skipped
test_bad_quality_profile_is_skipped

test_resolver_is_deterministic
```

------------------------------------------------------------------------

# 62. Unit tests --- Russian Dialogue Analyzer

``` text
test_dialogue_analyzer_extends_existing_linguistic_analysis
test_dialogue_analysis_returns_results_by_replica_id
test_dialogue_analysis_uses_neighbor_context

test_question_mark_alone_does_not_force_semantic_emotion
test_same_text_can_receive_different_prosody_in_different_context

test_dialogue_act_is_valid_enum
test_prosody_profile_is_valid_enum
test_confidence_is_bounded
test_intensity_is_bounded
test_pace_is_valid

test_llm_does_not_modify_source_text
test_llm_does_not_modify_final_text
test_llm_does_not_insert_emotion_tags
test_llm_does_not_insert_engine_markup

test_unknown_profile_is_rejected
test_unknown_replica_id_is_rejected
test_malformed_json_falls_back_safely
test_llm_timeout_falls_back_safely
test_ollama_unavailable_falls_back_safely
```

------------------------------------------------------------------------

# 63. Context tests

Обязательные semantic fixtures:

``` text
— Мы выиграли миллион.
— Правда?
```

Ожидается, что analyzer может отличить от:

``` text
— Я опять забыл документы.
— Правда?
```

Тест не должен требовать конкретной вероятностной LLM эмоции в unit
test.

Использовать mocked structured result для contract test.

Real LLM behaviour проверять benchmark/integration smoke.

------------------------------------------------------------------------

# 64. Unit tests --- override/invalidation

``` text
test_manual_prosody_override_wins
test_auto_prosody_used_without_override

test_manual_override_does_not_rerun_llm
test_manual_override_marks_audio_stale
test_manual_override_does_not_invalidate_dictionary

test_source_change_invalidates_detected_prosody
test_voice_change_invalidates_reference_resolution
test_voice_change_preserves_semantic_prosody_when_valid

test_new_voice_resolves_reference_again
```

------------------------------------------------------------------------

# 65. Unit tests --- engine boundary

Использовать spy engine.

Проверить:

``` text
test_tts_receives_clean_text_without_prosody_tags
test_tts_receives_resolved_reference_audio
test_tts_receives_resolved_reference_text
test_tts_reference_belongs_to_replica_voice

test_f5_receives_f5_stress_markup
test_xtts_does_not_receive_f5_stress_markup

test_prosody_override_changes_reference_not_source_text
```

------------------------------------------------------------------------

# 66. Unit tests --- short utterance integration

``` text
test_short_strategy_preserves_prosody_effective
test_short_strategy_uses_same_voice_reference
test_same_speaker_context_checks_prosody_compatibility
test_different_speakers_never_share_audio_reference
test_short_fallback_does_not_change_prosody_override
```

------------------------------------------------------------------------

# 67. Unit tests --- variants/regenerate

``` text
test_variant_records_prosody
test_variant_records_reference_profile
test_variant_records_fallback
test_regenerate_can_change_prosody
test_regenerate_keeps_old_take
test_regenerate_does_not_modify_neighbor_replica
```

------------------------------------------------------------------------

# 68. API tests

``` text
test_api_returns_replica_prosody
test_api_returns_reference_profile
test_api_patch_prosody_override
test_api_clear_prosody_override
test_api_lists_voice_reference_profiles

test_api_old_voice_payload_still_works
test_api_old_project_still_works
test_api_render_works_without_llm
```

------------------------------------------------------------------------

# 69. UI tests

Если frontend test infrastructure есть:

``` text
test_dialogue_replica_shows_auto_prosody
test_prosody_dropdown_lists_profiles
test_manual_selection_updates_override
test_clear_override_returns_to_auto
test_fallback_warning_is_visible
```

Если автоматизированных frontend tests нет --- добавить минимальные
DOM/unit tests или formal manual smoke checklist.

------------------------------------------------------------------------

# 70. Integration test --- Maria Reference Profiles

Создать fixture/test voice с несколькими reference profiles.

Не использовать реальные персональные production assets в unit tests.

Stub assets:

``` text
neutral.wav
question.wav
delight.wav
ironic.wav
```

Flow:

``` text
create voice
→ add profiles
→ create dialogue
→ parse
→ mock LLM analysis
→ assign Maria
→ resolve profiles
→ render with stub engine
→ inspect engine calls
```

------------------------------------------------------------------------

# 71. Critical integration assertions

``` text
one voice has many profiles
replica voice remains Maria

QUESTION replica uses Maria QUESTION
IRONIC replica uses Maria IRONIC
missing profile uses Maria NEUTRAL

never uses another voice
final_text unchanged
manual override wins
```

------------------------------------------------------------------------

# 72. Real LLM smoke test

С установленным Ollama:

``` text
qwen3:8b
```

Прогнать несколько сцен.

Сохранить:

``` text
raw structured output
validated output
latency
model
prompt version
schema version
memory state
```

Не включать gold answer в prompt.

------------------------------------------------------------------------

# 73. Real Reference Prosody Transfer Benchmark

Это обязательный real-model test.

Прогнать минимум:

``` text
1 F5 voice with all available profiles
30 targets
several profiles
fixed seed per comparison
```

Если есть второй F5 voice --- повторить.

XTTS --- отдельный benchmark.

------------------------------------------------------------------------

# 74. Real dialogue benchmark

Использовать реальную сцену:

``` text
АРТЁМ: Красивая.
МАРГАРИТА: Дайте пройти. Пожалуйста.
АРТЕМ: Можем пройти вместе.
МАРГАРИТА: Вы меня с кем-то спутали.
АРТЁМ: Такую не спутаешь.

МАРГАРИТА: Ты опоздал.
АРТЁМ: Всякое бывает. Оставь номер.
МАРГАРИТА: Мне тридцать девять, мальчик.
АРТЁМ: Это проблема?
МАРГАРИТА: Вижу, что не для тебя.
```

Проверить:

``` text
prosody analysis
reference routing
short strategy
text completeness
voice identity
manual override
regenerate
```

------------------------------------------------------------------------

# 75. Не hardcode emotion этого fixture

Fixture используется для regression.

Но production code не должен содержать:

``` python
if text == "Это проблема?":
    profile = QUESTION
```

Все решения проходят через analyzer/resolver.

------------------------------------------------------------------------

# 76. Regression tests существующих функций

После реализации обязательно прогнать полный текущий pytest.

Отдельно проверить:

``` text
Continuous Text
Dictionary
pronunciation suggestions
ё
RUAccent
preview
Projects
Timeline
short utterances
Smart QA
Strict QA
regenerate
variants
F5
XTTS
LLM disabled
LLM enabled
memory guard
```

------------------------------------------------------------------------

# 77. Performance tests

Замерить:

``` text
dialogue analysis latency
LLM request count
average replicas/request
peak memory
render latency before/after
```

Не допустить N LLM calls на N replicas.

------------------------------------------------------------------------

# 78. Cache tests

``` text
test_same_scene_reuses_dialogue_analysis_cache
test_prompt_version_invalidates_cache
test_schema_version_invalidates_cache
test_source_change_invalidates_cache
test_speaker_mapping_change_invalidates_relevant_cache
test_manual_override_does_not_invalidate_llm_cache
```

------------------------------------------------------------------------

# 79. Migration tests

Если меняется storage:

``` text
test_migration_is_idempotent
test_migration_preserves_voice_ids
test_migration_preserves_ref_audio
test_migration_preserves_ref_text
test_migration_preserves_projects
test_migration_can_run_twice
```

Перед destructive migration обязательно backup.

------------------------------------------------------------------------

# 80. Implementation phases

## Phase 0 --- Audit & Contracts

-   изучить текущий код;
-   описать contracts;
-   определить storage RECORD_PHRASES;
-   определить migration path;
-   определить existing LLM schema extension.

Gate:

``` text
нет дублирующей архитектуры
```

------------------------------------------------------------------------

## Phase 1 --- Reference Prosody Benchmark Infrastructure

До production routing:

-   benchmark corpus;
-   matrix;
-   automatic text QA;
-   listening review template;
-   report generation.

Gate:

``` text
можно объективно сравнить reference profiles
```

------------------------------------------------------------------------

## Phase 2 --- ReferenceProfile Data Model

-   storage;
-   CRUD;
-   migration;
-   quality fields;
-   compatibility;
-   existing voice fallback.

Gate:

``` text
старые voices/projects работают
```

------------------------------------------------------------------------

## Phase 3 --- RECORD_PHRASES Migration/Mapping

-   каждая запись связана с voice;
-   mapping label → profile;
-   отдельные WAV/ref_text;
-   quality validation.

Gate:

``` text
Maria имеет несколько profiles под одним voice_id
```

------------------------------------------------------------------------

## Phase 4 --- ProsodyResolver

-   exact profile;
-   fallback;
-   quality;
-   engine compatibility;
-   same-voice invariant;
-   diagnostics.

Gate:

``` text
resolver полностью deterministic
```

------------------------------------------------------------------------

## Phase 5 --- Russian Dialogue Analyzer Schema

-   dialogue_act;
-   prosody;
-   intensity;
-   pace;
-   confidence;
-   context_dependency;
-   prompt/schema version.

Gate:

``` text
существующий linguistic analyzer не сломан
```

------------------------------------------------------------------------

## Phase 6 --- Scene-aware LLM Analysis

-   context windows;
-   batching;
-   multi-replica response;
-   validation;
-   fallback;
-   cache;
-   HeavyGate.

Gate:

``` text
нет one-request-per-replica
```

------------------------------------------------------------------------

## Phase 7 --- Persistence & Invalidation

-   replica fields;
-   overrides;
-   effective values;
-   stale logic;
-   source/voice changes.

Gate:

``` text
state transitions воспроизводимы
```

------------------------------------------------------------------------

## Phase 8 --- Dialogue UI

-   Auto;
-   dropdown;
-   confidence;
-   fallback;
-   reference details;
-   manual override;
-   clear override.

Gate:

``` text
пользователь может полностью контролировать Auto
```

------------------------------------------------------------------------

## Phase 9 --- TTS Integration

-   resolver at engine boundary;
-   clean final_text;
-   reference profile;
-   F5/XTTS separation;
-   short layer integration.

Gate:

``` text
engine spy tests проходят
```

------------------------------------------------------------------------

## Phase 10 --- Regenerate & Variants

-   prosody-aware regenerate;
-   take metadata;
-   old take preserved;
-   comparison.

Gate:

``` text
смена интонации одной replica не трогает соседние
```

------------------------------------------------------------------------

## Phase 11 --- Real-model Benchmark

-   F5 prosody transfer;
-   XTTS separately;
-   dialogue fixture;
-   listening review;
-   performance/memory.

Gate:

``` text
Auto profiles разрешены только для подтверждённых profiles
```

------------------------------------------------------------------------

## Phase 12 --- Regression & Production Hardening

-   full pytest;
-   API;
-   UI;
-   migrations;
-   LLM off;
-   memory pressure;
-   diagnostics.

Gate:

``` text
нет regression существующего функционала
```

------------------------------------------------------------------------

## Phase 13 --- Documentation & Knowledge Base

Только после подтверждения фактического поведения.

------------------------------------------------------------------------

# 81. Обязательное обновление Knowledge Base

После реализации обновить библиотеку знаний проекта.

Добавить отдельный раздел:

``` text
Russian Dialogue Analyzer
```

Описать:

-   назначение;
-   архитектуру;
-   schema;
-   prompt version;
-   context windows;
-   batching;
-   fallback;
-   cache;
-   failure modes.

------------------------------------------------------------------------

# 82. Knowledge Base --- Reference Profiles

Документировать:

``` text
Voice 1:N ReferenceProfile
```

Добавить:

-   schema;
-   migration;
-   RECORD_PHRASES mapping;
-   quality requirements;
-   engine compatibility;
-   fallback rules;
-   same-voice invariant.

------------------------------------------------------------------------

# 83. Knowledge Base --- Prosody

Документировать:

``` text
prosody_detected
prosody_override
prosody_effective
```

и:

``` text
dialogue_act
intensity
pace
context_dependency
```

Объяснить, что prosody metadata не входит в TTS text.

------------------------------------------------------------------------

# 84. Knowledge Base --- Benchmark Results

Сохранить:

``` text
reference_prosody_benchmark.md
```

или эквивалент.

Указать:

-   hardware;
-   engine;
-   voices;
-   profiles;
-   seeds;
-   targets;
-   automatic metrics;
-   listening conclusions;
-   profiles enabled for Auto;
-   profiles disabled for Auto;
-   known limitations.

Не писать неподтверждённые выводы.

------------------------------------------------------------------------

# 85. Обновление README

README должен объяснять пользователю:

1.  что такое Reference Profile;
2.  почему один голос имеет несколько интонационных записей;
3.  как RECORD_PHRASES используются после записи;
4.  что делает Auto;
5.  что анализирует LLM;
6.  как изменить интонацию вручную;
7.  что такое fallback;
8.  что произойдёт без Ollama;
9.  как regenerate работает с prosody;
10. какие движки лучше/хуже переносят prosody согласно фактическому
    benchmark.

------------------------------------------------------------------------

# 86. Обновление API docs

Документировать:

-   новые replica fields;
-   ReferenceProfile representation;
-   override;
-   resolver result;
-   fallback fields;
-   migration compatibility.

------------------------------------------------------------------------

# 87. Обновление data model docs

Добавить diagram:

``` text
Voice
  1
  |
  N
ReferenceProfile

Project
  |
Replica
  ├ prosody analysis
  ├ override
  ├ resolved reference
  └ Takes
```

------------------------------------------------------------------------

# 88. Обновление testing docs

Добавить команды:

``` text
unit tests
integration tests
LLM smoke
reference prosody benchmark
real dialogue benchmark
full regression
```

Указать, какие тесты:

``` text
mock/stub
```

а какие реально поднимают:

``` text
Ollama
F5
XTTS
Whisper
```

------------------------------------------------------------------------

# 89. Обновление configuration docs

Если появляются env/config:

документировать их.

Не вводить десятки tuning variables без необходимости.

Особенно описать:

``` text
Auto prosody enabled
allowed profiles
fallback policy
benchmark-only/debug flags
```

------------------------------------------------------------------------

# 90. Финальный отчёт агента

После реализации агент обязан выдать:

``` text
1. Summary.
2. Changed files.
3. DB/storage migrations.
4. ReferenceProfile implementation.
5. RECORD_PHRASES migration.
6. Russian Dialogue Analyzer changes.
7. New LLM schema.
8. Prompt/schema versions.
9. ProsodyResolver.
10. Fallback policy.
11. Dialogue UI.
12. Regenerate/variants changes.
13. Short-utterance integration.
14. Unit test results.
15. Integration test results.
16. Full pytest result.
17. F5 reference benchmark result.
18. XTTS benchmark result.
19. Real dialogue benchmark.
20. Listening review.
21. Performance/memory results.
22. Documentation updated.
23. Knowledge Base updated.
24. Known limitations.
25. Recommended next step.
```

------------------------------------------------------------------------

# 91. Definition of Done

Задача завершена только если:

-   существующий Local LLM Analyzer расширен, а не продублирован;
-   LLM анализирует dialogue context;
-   результаты привязаны к replica_id;
-   есть dialogue_act;
-   есть prosody metadata;
-   есть confidence;
-   есть manual override;
-   source/final text не загрязняется тегами;
-   один voice поддерживает много ReferenceProfiles;
-   существующие RECORD_PHRASES реально используются;
-   один profile не превращается в отдельный voice;
-   resolver deterministic;
-   reference всегда принадлежит тому же voice;
-   missing profile имеет явный fallback;
-   bad profile не выбирается автоматически;
-   engine compatibility учитывается;
-   старые voices/projects мигрируют;
-   LLM failure не блокирует обычный render;
-   HeavyGate/memory policy переиспользуются;
-   нет одного LLM request на каждую replica;
-   short-utterance layer не сломан;
-   regenerate поддерживает смену prosody;
-   variants сохраняют prosody/reference metadata;
-   benchmark переноса reference выполнен;
-   F5 и XTTS оценены отдельно;
-   listening review выполнен;
-   full regression suite проходит;
-   README обновлён;
-   API docs обновлены;
-   data model docs обновлены;
-   testing docs обновлены;
-   Knowledge Base обновлена.

------------------------------------------------------------------------

# 92. Что нельзя считать выполнением задачи

Не принимать реализацию, если агент сделал только:

``` text
dropdown эмоций
```

без ReferenceProfile routing.

Не принимать:

``` text
LLM → [восторг] текст → TTS
```

Не принимать:

``` text
? → QUESTION
! → DELIGHT
```

как основной semantic analyzer.

Не принимать выбор чужого reference.

Не принимать hardcoded emotions для benchmark dialogue.

Не принимать один LLM request на каждую replica.

Не принимать automatic engine parameter changes без benchmark.

Не принимать утверждение:

``` text
"эмоции работают"
```

без Reference Prosody Transfer Benchmark и listening review.

------------------------------------------------------------------------

# 93. Итоговый принцип системы

``` text
ТЕКСТ
    ↓
Что сказано?
    ↓
Russian Linguistic Analysis

КОНТЕКСТ
    ↓
Что имеется в виду?
    ↓
Russian Dialogue Analyzer

PROSODY INTENT
    ↓
Как это должно звучать?
    ↓
ProsodyResolver

VOICE
    ↓
Какая запись этого же голоса лучше подходит?
    ↓
ReferenceProfileResolver

SYNTHESIS
    ↓
Как надёжно синтезировать именно эту replica?
    ↓
Short Dialogue Planner + Engine

QUALITY
    ↓
Всё ли произнесено правильно?
    ↓
QA
```

LLM отвечает прежде всего за:

``` text
понимание
```

Backend --- за:

``` text
детерминированные решения и состояние
```

Reference Profiles --- за:

``` text
просодическое conditioning
```

TTS --- за:

``` text
генерацию аудио
```

QA --- за:

``` text
проверку результата
```

Эти обязанности не смешивать.
