Исправить workflow вкладки «Озвучка диалога» так, чтобы генерация аудио была невозможна до завершения анализа и подготовки текста.

Критерий результата: один и тот же текст при одинаковом голосе, движке и effective settings должен поступать в TTS в одинаково подготовленном виде независимо от режима «Сплошной текст» или «Озвучка диалога». Нельзя считать задачу завершённой только потому, что появился новый экран анализа: необходимо найти и устранить фактическую причину разницы качества.

Работай с существующей архитектурой проекта. Не создавай второй независимый preprocessing pipeline. Существующие audio_pipeline.preview_text(...), normalize_stages, pronunciation dictionary, RUAccent, project/replica model и engine registry должны переиспользоваться.

1. Новый обязательный workflow

Для диалога реализовать state machine:

RAW
 ↓
ANALYZING
 ↓
NEEDS_REVIEW   # есть неоднозначные/неподтверждённые pronunciation candidates
 ↓
READY
 ↓
RENDERING
 ↓
DONE

Допускается ERROR для ошибок анализа/рендера.

Правила

RAW, ANALYZING, NEEDS_REVIEW → рендер запрещён.

READY → можно запускать TTS.

изменение исходного текста → анализ инвалидируется, состояние возвращается в RAW;

повторный parse, изменивший набор/текст реплик → анализ соответствующих реплик инвалидируется;

изменение speaker → voice mapping или engine должно инвалидировать engine-dependent подготовку реплик;

изменение pronunciation dictionary, влияющее на проект, должно помечать затронутые реплики как требующие повторной подготовки;

рендер никогда не должен молча анализировать сырой текст как обход state machine.

UI должен явно показывать текущий статус и причину, почему кнопка генерации недоступна.

2. Порядок подготовки диалога

Обязательный порядок:

source/file
  ↓
parse dialogue
  ↓
detect speaker markers / slots
  ↓
create/update replicas
  ↓
normalize replica text
  ↓
restore ё
  ↓
find pronunciation candidates
  ↓
dictionary review
  ↓
apply pronunciation dictionary
  ↓
apply engine-specific accent processing
  ↓
save final_text
  ↓
READY
  ↓
TTS

Важно

Speaker/slot markers (ИВАН:, ARTEM(1):, (1), inline params и т.п.) должны быть разобраны до лингвистического preprocessing. Служебная разметка не должна попадать в текст, отправляемый в normalization/RUAccent/TTS.

Не анализировать весь диалог одной строкой после удаления маркеров. После parse анализировать каждую реплику отдельно, сохраняя speaker/slot/voice/engine context.

3. Данные Replica

Проверить текущую SQLite/project schema и добавить недостающие поля. Не дублировать уже существующие данные.

Для каждой реплики должны быть доступны как минимум:

source_text
normalized_text
yo_text
dictionary_text
accentized_text
final_text

speaker / speaker_marker
voice_id
engine

analysis_status
analysis_version
analysis_error

Дополнительно сохранить полезную диагностическую информацию:

dictionary_matches
pronunciation_candidates
supports_accents
auto_accent
effective_engine_params

Если стадии уже хранятся иначе, сохранить текущую модель и расширить её минимально. Не делать миграцию, ломающую существующие проекты: нужна нормальная SQLite migration/backward compatibility.

final_text является замороженным результатом подготовки конкретной версии реплики.

4. Единственный источник preprocessing

Не создавать отдельные функции вроде:

prepare_dialogue_text()
prepare_continuous_text()

с разной логикой нормализации.

Сделать один canonical preprocessing path, используемый:

/api/text/preview;

подготовкой диалога;

подготовкой сплошного текста;

непосредственно перед TTS как проверкой/получением подготовленного результата там, где это ещё необходимо;

regeneration отдельной реплики.

Существующий порядок стадий сохранить:

normalization
→ ё
→ pronunciation dictionary
→ RUAccent (только если engine это поддерживает и auto_accent включён)
→ final

Preview и фактический input модели должны формироваться одной реализацией.

5. Pronunciation review

Использовать существующий механизм /api/pronunciation/suggestions.

После анализа показать кандидаты, требующие решения пользователя.

Пример UX:

Найдено 7 слов для проверки

договор  → догов+ор
звонит   → звон+ит
замок    → з+амок / зам+ок
OpenAI   → оупен эй-ай

Нельзя

Не добавлять автоматически все suggestions в глобальный словарь.

Особенно нельзя глобально фиксировать омограф без контекста.

Нужны два scope словаря

project
global

Приоритет:

project dictionary
→ global dictionary
→ automatic processing

Если текущая модель pronunciation dictionary не поддерживает project scope — расширить её.

UI review должен позволять:

принять предложенный вариант только для проекта;

принять глобально;

отредактировать pronunciation;

выбрать вариант неоднозначного слова;

пропустить предложение;

после подтверждения повторно пересчитать затронутые реплики.

Не блокировать READY из-за слов, которые пользователь явно отметил как «пропустить/оставить как есть».

6. RUAccent и движки

Учитывать паспорт движка supports_accents.

Для F5 при включённом auto_accent применять RUAccent после normalization/ё/dictionary.

Для XTTS/F5-несовместимого движка не отправлять F5 stress markup только ради того, чтобы стадия анализа выглядела одинаково.

Если voice/engine меняется после анализа, пересчитать engine-dependent accentized_text/final_text.

Если RUAccent недоступен/упал:

не скрывать ошибку;

проект не должен притворяться полностью подготовленным с ударениями;

UI должен показать понятную причину;

если пользователь сознательно отключил auto-accent, это не ошибка.

7. Вкладка «Озвучка диалога»

Перестроить основной UX.

Шаг A — источник

Пользователь:

вставляет диалог;

либо прикрепляет поддерживаемый текстовый файл.

Использовать существующие ограничения размера текста и parser.

Шаг B — «Анализировать диалог»

Отдельная основная кнопка.

После запуска:

parse;

определить speakers/slots;

создать/обновить replicas;

провести preprocessing;

найти pronunciation candidates;

перейти в NEEDS_REVIEW либо READY.

Показать progress анализа.

Шаг C — review

Визуальный редактор реплик должен показывать минимум:

speaker;

назначенный voice;

engine;

исходный текст;

подготовленный final_text по раскрытию;

предупреждения;

pronunciation candidates;

действие «Что услышит модель».

Шаг D — генерация

Кнопка «Сгенерировать аудио» активна только при READY.

Рядом показывать краткое подтверждение, например:

Подготовлено: 38 реплик
Голоса: 3
Словарь: 6 применённых правил
Ударения: F5 — включены

8. Render contract

Это критическая часть.

После READY dialogue render должен брать подготовленный текст реплик, а не снова строить независимый текстовый pipeline из сырого dialogue_text.

Концептуально:

assert project.status == "READY"

for replica in project.replicas:
    synthesize(
        text=replica.final_text,
        voice=resolved_voice,
        engine=resolved_engine,
        ...
    )

Фактическую реализацию адаптировать под существующий код.

Добавить server-side guard: прямой API-вызов render для неподготовленного проекта должен вернуть корректную 4xx ошибку, даже если UI такую кнопку не показывает.

Legacy /api/generate нельзя оставлять скрытым обходом нового правила. Либо:

перевести его на новый prepare/render contract;

либо явно сделать legacy/deprecated endpoint с безопасным поведением;

либо удалить после проверки всех callers/tests.

Не ломать API без необходимости.

9. Regenerate одной реплики

При regeneration готовой реплики использовать тот же сохранённый final_text, если source/voice/engine/dictionary context не менялись.

Если пользователь редактирует текст конкретной реплики:

replica → RAW / NEEDS_ANALYSIS
project → not READY

Повторно анализировать можно только изменённую реплику, если остальные не затронуты.

После изменения project/global dictionary определить затронутые реплики и пересчитать их; не требуется без причины прогонять весь большой проект.

10. Обязательное расследование разницы качества Dialogue vs Continuous

Не предполагать заранее, что проблема только в ударениях.

Сделать regression/debug comparison:

SAME SOURCE TEXT
SAME VOICE
SAME ENGINE
SAME REFERENCE AUDIO
SAME REFERENCE TEXT
SAME EFFECTIVE SETTINGS
SAME AUTO_ACCENT

для:

Continuous text
vs
Dialogue replica

Непосредственно перед engine.synthesize() уметь диагностически сравнить:

engine
voice_id
reference_audio identity/path
reference_text
source_text
normalized/final text
speed
engine params
seed
chunk boundaries
pause/crossfade where relevant
QA mode

Логи не должны содержать лишние персональные данные и не должны бесконтрольно печатать огромные тексты; для production/debug использовать разумное truncation/hash там, где подходит.

Проверить отдельно

Совпадает ли final_text.

Совпадают ли effective voice settings.

Совпадает ли reference audio/ref text.

Нет ли двойного preprocessing в dialogue.

Нет ли потери ё/ударений при parse/serialization.

Не попадают ли markers/inline params в TTS text.

Одинаково ли chunking обрабатывает короткую реплику.

Нет ли разных defaults F5/XTTS между вкладками.

Не переопределяются ли параметры voice/project/replica в dialogue.

Не отличается ли путь вызова engine или подготовка reference.

Не портит ли качество postprocessing/crossfade/normalization; сравнить также raw replica WAV до финальной склейки.

Если final_text и engine input полностью совпадают, искать дефект дальше по synthesis/audio pipeline. Не закрывать задачу изменением UI.

11. Parity tests — обязательны

Добавить unit/integration tests минимум:

test_dialogue_cannot_render_before_analysis
test_dialogue_analysis_creates_replicas
test_speaker_markers_removed_before_preprocessing
test_inline_slot_metadata_not_sent_to_tts
test_edit_source_invalidates_analysis
test_edit_replica_invalidates_only_required_analysis
test_voice_engine_change_invalidates_engine_dependent_analysis

test_pronunciation_candidates_require_review
test_pronunciation_skip_allows_ready
test_project_dictionary_overrides_global
test_global_dictionary_used_when_no_project_override
test_dictionary_change_reanalyzes_affected_replicas

test_f5_final_contains_expected_accent_markup
test_xtts_final_does_not_receive_f5_accent_markup
test_disabled_auto_accent_does_not_block_ready
test_accentizer_failure_is_visible

test_render_uses_saved_final_text
test_regenerate_uses_same_final_text
test_preview_final_equals_render_input

test_same_text_continuous_vs_dialogue_normalized_text
test_same_text_continuous_vs_dialogue_dictionary_text
test_same_text_continuous_vs_dialogue_final_text
test_same_voice_continuous_vs_dialogue_effective_settings
test_same_voice_continuous_vs_dialogue_reference_input
test_same_voice_continuous_vs_dialogue_engine_input

Для parity test использовать deterministic stub/spy engine, который записывает фактически полученные аргументы synthesize(). Не сравнивать только HTTP payload — сравнивать input на границе engine.

Существующие тесты не удалять ради прохождения новых.

12. API

Имена endpoint можно адаптировать к текущему API, но логически нужны операции:

POST /api/projects/{project_id}/analyze
GET  /api/projects/{project_id}/analysis
POST /api/projects/{project_id}/pronunciation/review
POST /api/projects/{project_id}/render

Если существующий /parse уже является частью project flow, не дублировать parse без причины: можно расширить orchestration endpoint analyze.

Ответ анализа должен содержать:

{
  "status": "needs_review",
  "replicas_total": 38,
  "replicas_analyzed": 38,
  "speakers": [],
  "candidates": [],
  "warnings": [],
  "errors": []
}

Конкретную schema согласовать с существующими Pydantic models.

13. Файлы диалога

Если UI уже принимает файл — провести его через тот же source_text → analyze flow, что и вставленный текст.

Не создавать отдельный parser/preprocessor только для upload.

Для .txt/.md:

file bytes
→ decode/validate
→ source_text
→ common dialogue analyze

При ошибке кодировки/пустом файле/слишком большом файле вернуть понятную ошибку до анализа.

14. Производительность

Анализ текста не должен запускать TTS model.

/api/text/preview уже задуман как дешёвая операция; сохранить это свойство.

RUAccent может грузиться по необходимости, но F5/XTTS synthesis weights не должны загружаться только ради анализа.

Не запускать Whisper для обычного preprocessing диалога. Whisper/Smart QA относится к проверке синтезированного аудио, а не к подготовке исходного текста.

Для большого диалога не делать N одинаковых инициализаций моделей/словарей на N реплик.

15. Не смешивать эту задачу с лишними изменениями

В рамках этого этапа не заниматься:

cloud deployment;

multi-user/auth;

real-time streaming;

параллельным TTS;

новым TTS engine;

редизайном всего приложения;

полной заменой parser;

полной заменой pronunciation/RUAccent subsystem.

Допускаются только изменения, необходимые для обязательной подготовки диалога, parity и исправления обнаруженной причины деградации качества.

16. Definition of Done

Задача завершена только если одновременно выполнено всё:

диалог нельзя синтезировать до анализа;

parser корректно выделяет speakers/slots до linguistic preprocessing;

пользователь видит найденных speakers и replicas;

pronunciation candidates проходят review;

поддерживается project-level pronunciation override;

F5 получает корректно подготовленный текст с нужными ударениями;

XTTS не получает несовместимую F5-разметку;

final_text сохранён для каждой подготовленной реплики;

render использует сохранённый подготовленный текст;

изменение source/voice/engine/dictionary корректно инвалидирует анализ;

preview показывает тот же final, который реально поступит в engine;

одинаковая реплика в Continuous и Dialogue имеет одинаковый preprocessing result при одинаковом context;

effective settings и reference inputs между режимами проверены;

найдена и исправлена реальная причина существенной разницы качества либо тестами доказано, на каком последующем уровне pipeline она возникает;

все новые и существующие tests проходят;

README/API documentation обновлены под новый workflow.

17. Порядок работы ИИ-агента

Сначала прочитать текущий README и релевантный код.

Найти текущие реализации:

project/replica persistence;

dialogue parser;

audio_pipeline.preview_text;

normalize_stages;

pronunciation dictionary/suggestions;

RUAccent;

/api/text/preview;

project parse/render;

continuous render;

dialogue render;

regeneration;

frontend dialogue tab.

До изменений нарисовать фактический call graph двух путей:

Continuous → engine.synthesize;

Dialogue → engine.synthesize.

Зафиксировать различия.

Добавить parity tests/spy engine, которые воспроизводят проблему или хотя бы фиксируют фактические engine inputs.

Реализовать state machine и persistence анализа.

Реализовать pronunciation review/project dictionary.

Перевести dialogue render на prepared final_text.

Исправить обнаруженные различия между Continuous и Dialogue, которые не обусловлены намеренно разными функциями режима.

Обновить UI.

Прогнать unit + integration tests.

Провести один реальный smoke test F5:

одинаковая короткая русская фраза;

тот же voice;

Continuous и Dialogue;

сохранить raw replica outputs для слухового A/B.

Если доступен XTTS — повторить parity smoke test для XTTS.

Обновить README с новым пользовательским workflow и API.

В финальном отчёте перечислить:

root cause разницы качества;

изменённые файлы;

migrations;

новые endpoints/schema;

новые tests;

результаты test suite;

результаты Continuous vs Dialogue parity;

известные ограничения.

18. Правила реализации

Не маскировать ошибку fallback'ом, если он снова позволяет синтезировать сырой текст.

Не копировать preprocessing между frontend/backend.

Backend — источник истины для статуса подготовки и final_text.

Не доверять disabled-кнопке UI: guards обязательны на API.

Не менять пользовательский source text при preview/analyze; хранить производные стадии отдельно.

Не уничтожать вручную введённые пользователем ударения/произношение.

Не применять автоматическую замену неоднозначного омографа глобально без подтверждения.

Не делать destructive DB reset ради новой schema.

Не удалять существующую поддержку вариантов/seed/QA/regeneration.

Сохранять single-worker TTS architecture; эта задача не требует параллельного inference.

При обнаружении причины качества сначала закрепить её regression test, затем исправлять.

Не считать визуальное сходство двух вкладок доказательством parity. Истина — аргументы на границе engine.synthesize() и raw audio до общей финальной сборки.