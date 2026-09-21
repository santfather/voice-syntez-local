# Фаза 2 «Сам себе звукорежиссер» — отчёт

Отчёт по пакету `PHASE_2_SAM_SEBE_ZVUKOREZHISSER.md` (шаблоны §40, §46).

## 1. Добавленные файлы

| Файл | Что |
|---|---|
| `backend/recording_store.py` | проекты записи, записываемые голоса, дубли: JSON на диске, атомарная запись, безопасные имена |
| `backend/recording_audio.py` | декод браузерной записи в PCM, time-stretch, pitch shift, denoise-адаптер, trim, уровень, кодирование WAV/MP3 |
| `backend/recording_pipeline.py` | монтаж: активные дубли → паузы → LUFS/лимитер → мастер → MP3, состояние рендера |
| `tests/test_recording_pipeline.py` | blocking-интеграционные тесты Test A–G (2 и 3 голоса, порядок, дубли, блокировка, декод MP3) |
| `tests/test_recording_store.py`, `tests/test_recording_audio.py`, `tests/test_recording_api.py` | unit/API-покрытие хранилища, DSP и эндпоинтов |
| `sam_sebe_zvukorezhisser_report.md` | этот отчёт |

## 2. Изменённые файлы

| Файл | Что |
|---|---|
| `backend/config.py` | `RECORDINGS_DIR`, лимиты загрузки и диапазоны обработки, формат финального файла |
| `backend/main.py` | namespace `/api/recording-projects…` (15 маршрутов) и модели запросов |
| `frontend/index.html`, `frontend/app.js`, `frontend/style.css` | новая вкладка `04 · Сам себе звукорежиссер` с пятью панелями |
| `tests/conftest.py` | `RECORDINGS_DIR` уводится в `tmp_path`, singleton'ы хранилища и состояний монтажа сбрасываются |
| `README.md` | раздел про вкладку, API, структуру, переменные окружения, тесты и troubleshooting |

## 3. Storage schema проекта записи

```text
recordings/
  projects.json                  индекс: id, имя, статус, обновлён, сколько записано
  {project_id}/
    project.json                 полное состояние проекта
    raw/     r0001_take01.wav    сырые дубли (не изменяются никогда)
    processed/  {take}-{hash}.wav  кэш обработанных версий (по ключу настроек)
    previews/                    превью обработки
    output/  dialogue-master.wav + dialogue.mp3
```

`project.json`:

```text
id, name, dialogue_text, status, created_at, updated_at, last_error
replicas[]        {index, speaker, text}          # из общего парсера, порядок исходный
role_voices       {speaker: profile_id}
voice_profiles[]  {id, name, speed, pitch_semitones, denoise, created_at}
takes[]           {id, replica_index, voice_profile_id, raw_file, duration_sec,
                   created_at, level_warning}
active_takes      {replica_index: take_id}       # решение пользователя, не свойство файла
recording_order   dialogue | roles               # влияет только на показ
render_settings   {pause_ms, …}
```

## 4. Формат raw master

Браузерная запись (WebM/Opus, MP4/AAC и т. п.) декодируется в **WAV, моно,
float32 при 24 кГц** — тот же `SAMPLE_RATE`, что у остального пайплайна, — и
именно в этом виде сохраняется как raw. Монтаж над WebM/Opus не выполняется.
Проверяется не имя файла, а успешность декодирования; расширение — только грубый
фильтр (§23).

## 5. Time stretch

`librosa.effects.time_stretch` (phase vocoder) — та же библиотека, что уже
используется проектом для pitch shift, поэтому новых зависимостей нет. Обычный
resample не применяется: он меняет и длительность, и высоту. Тест проверяет это
прямо: доминирующая частота до и после растяжения совпадает, а при `speed=1.0`
`librosa` не вызывается вовсе.

## 6. Pitch shift

Существующий `audio_pipeline._pitch_shift` (librosa `pitch_shift`): высота
меняется, длительность сохраняется, raw не затрагивается. Диапазон UI `−6…+6`
полутонов, backend допускает `−12…+12` (`TTS_RECORDING_PITCH_MIN/MAX`).

## 7. DeepFilterNet

Существующий `backend/denoise.py` (`is_available()` + `clean_bytes()`), тот же
subsystem, что очищает референсы TTS. Очистка идёт над копией записи: raw master
остаётся нетронутым. Если DeepFilterNet не установлен или падает, обработка
продолжается без очистки, а причина попадает в предупреждения (тумблер
показывает это пользователю). Тесты проверяют, что при выключенной очистке
`denoise` не вызывается вообще.

## 8. Edge trim

Существующая обрезка `audio_pipeline._trim_edge_silence` с запасом вокруг речи
(`_short_guard_ms` для коротких реплик, иначе `EDGE_SILENCE_MARGIN_MS`). Порядок:
обрезка **до** растяжения — после time-stretch границы речи размываются.

## 9. LUFS и limiter

Существующий финальный проход `audio_pipeline._finalize_track` (LUFS-нормализация
+ лимитер) применяется к **собранному диалогу целиком**, а не к каждой роли
отдельно: воспринимаемая громкость контролируется один раз по всему треку (§29).

## 10. Новые API endpoints

Все перечислены в README (раздел API): список/CRUD проектов, `parse`,
`voice-profiles` (создание/правка), `roles/{speaker}`, дубли
(`replicas/{index}/takes`, `takes/{take_id}/audio|select`, удаление), `preview` и
`previews/{name}/audio`, `render`, `render/status`, `audio`.

Монтаж запускается в потоке (`run_in_executor`), состояние опрашивается отдельным
эндпоинтом — тяжёлая обработка в HTTP-запросе не выполняется (§22).

## 11. Frontend APIs для микрофона

`navigator.mediaDevices.getUserMedia({audio})`, `enumerateDevices()` для выбора
входа, `MediaRecorder` (webm/opus), `AudioContext` + `AnalyserNode` для level
meter, `URL.createObjectURL` для немедленного прослушивания; при отказе доступа
(`NotAllowedError`) и отсутствии устройства (`NotFoundError`) показывается
понятное сообщение. Опциональный отсчёт 0/1/2/3 секунды перед записью
(по умолчанию 1).

## 12. Дефекты backend, найденные тестами, и их исправление

Автоматические тесты поймали три дефекта, которых чтение кода не показало; все три
исправлены до коммита.

| Дефект | Проявление | Исправление |
|---|---|---|
| `POST /api/recording-projects/{id}/takes/{take_id}/preview` звал `store.safe_component(take_id)`, хотя `safe_component` — функция модуля `recording_store`, а не метод `RecordingStore` | `AttributeError` → 500 на каждом запросе превью: прослушать обработанный дубль было нельзя вовсе | `recording_store.safe_component(take_id)` в `backend/main.py` |
| `audio_pipeline._read_audio` не приводил файл к внутреннему формату (`sf.read` + `reshape(-1)`: ни ресемплинга, ни сведения каналов) | браузерная запись (WebM/Opus, почти всегда 48 кГц, часто стерео) читалась как поток на родной частоте: одна секунда записи давала `duration_sec = 2.0`, вместе с ней уезжали темп, паузы и длительности дублей | `_read_audio` сводит каналы усреднением и ресемплирует к `SAMPLE_RATE` (`librosa.resample`, `soxr_hq` с откатом на `kaiser_best`) |
| `POST …/render` уходил в поток, **не** помечая состояние запущенным: окно между ответом 202 и первым оператором рабочего потока отдавало прошлый монтаж | первый же опрос `render/status` показывал «готово» с прежней длительностью — «сменил дубль, а файл не изменился». В тесте это дало плавающее падение Test E при случайном порядке; интерфейс прячет окно оптимистичной локальной отметкой «монтаж запущен», но контракт API оставался неверным для любого другого клиента (перезагрузка страницы, вторая вкладка, скрипт) | `recording_pipeline.begin_render()` объявляет запуск **до** `run_in_executor` и под тем же замком, что и проверка «уже идёт»; `abort_render()` снимает вечное «running», если поток не удалось запустить; эндпоинт отдаёт 409 на `RenderBusyError` |

Все три дефекта закреплены тестами:
`tests/test_recording_api.py::test_preview_returns_url_and_decodable_wav`,
`tests/test_recording_api.py::test_preview_speed_changes_duration`,
`tests/test_recording_audio.py::test_decode_normalizes_stereo_and_foreign_sample_rate`,
`tests/test_recording_pipeline.py::test_render_status_shows_new_run_not_previous_one`,
`tests/test_recording_pipeline.py::test_begin_render_refuses_second_start`,
`tests/test_recording_pipeline.py::test_render_of_missing_project_does_not_stay_running`.
Первые три до исправления были помечены `xfail` с текстом причины; после — проходят
как обычные тесты, `xfail`-маркеров в recording-наборах не осталось. Третий дефект
нашёлся не чтением, а полным прогоном со случайным порядком тестов — то есть тем
самым regression suite, которого требует §45.

Приведение формата живёт в общем `_read_audio`, но для TTS-пайплайна это no-op:
свои файлы проект всегда пишет моно при `SAMPLE_RATE`. Отсутствие регресса
подтверждает полный прогон (§13).

## 13. Результат полного pytest

```text
./venv/bin/python -m pytest -q            (порядок тестов перемешивает pytest-randomly)

собрано тестов: 1288
passed:         1288
failed:         0
errors:         0
skipped:        0
xfailed:        0
время:          166.0 с
exit code:      0

Recording-набор отдельно — 43 теста:
  tests/test_recording_store.py     8
  tests/test_recording_audio.py    14
  tests/test_recording_api.py      10
  tests/test_recording_pipeline.py 11   (Test A–G из §42 + Test I «финальный проход» + 3 регресса на запуск монтажа)
```

Первый полный прогон после подключения frontend дал одно плавающее падение
(`test_active_take_change_is_used_in_next_render`, Test E): разница длительностей
двух монтажей оказалась ровно `0.0`. Причина — не DSP, а гонка «202 раньше
состояния» (§12, третий дефект); после исправления полный прогон и три повторных
прогона recording-набора зелёные.

## 14. E2E smoke test

```text
REAL MICROPHONE E2E: NOT VERIFIED
```

Реального микрофона в агентской среде нет, поэтому ручной сценарий §36 (запись
двух ролей голосом, перезапись дублей, denoise, preview, reload) **не выполнялся**.
Вместо него обязательный автоматический набор §42 выполнен на подготовленных
fixture-аудиофайлах:

| Тест | Что проверено | Результат |
|---|---|---|
| A — два голоса | A→B→A→B по доминирующей частоте сегментов, паузы, один MP3 | PASS |
| B/C — три голоса и запись по ролям | A→B→C→A→B→C при записи блоками по ролям | PASS |
| D — обработка голосов | скорость/высота применяются только к своим голосам | PASS |
| E — смена дубля | следующий рендер использует новый дубль, разница равна разнице длительностей | PASS |
| F — незаписанная реплика | 409 с индексами, файла не появляется | PASS |
| G — декод MP3 | контейнер, каналы, частота, длительность, сэмплы | PASS |

## 15. Известные ограничения

1. **Реальный микрофон не проверен** (см. §14): UI записи покрыт только чтением
   кода и синтаксической проверкой; ошибки устройств проверены логикой, но не
   живой записью.
2. **Гибридный режим** (часть ролей — запись, часть — TTS) не реализован; для
   него хранилище уже разделяет активный дубль и сборку.
3. **Кроссфейд** между записанными репликами равен нулю: реплики сшиваются по
   сэмплам после обрезки краёв, пауза — цифровая тишина. «Существующего helper'а»
   для перекрытия реплик в проекте нет: `cross_fade_duration`
   (`config.DEFAULT_CROSS_FADE_DURATION`) — это внутренняя ручка F5 для сшивания
   кусков одной реплики внутри движка (`tts_engine`/`_prepare_chunk`), а не
   overlap на таймлайне, поэтому переиспользовать её на сборке нечего. §28
   («кроссфейд минимальный», «не съедать атаку следующего голоса») выполнен
   буквально: перекрытия нет, атака следующей реплики не может быть съедена.
4. **Формат финального файла зафиксирован** (`mp3`): задача фазы — единый MP3,
   поэтому выбор формата в панели «Монтаж» показывает один вариант.
5. **Denoise** уменьшает постоянный шум и гул, но не удаляет речь других людей и
   громкие отдельные звуки — это указано в подписи тумблера, а не обещано как
   «шумоподавление вообще».
6. **Пауза — общая** для всего диалога: индивидуальной паузы на реплику в этой
   фазе нет (§32).
7. **Диапазоны ручек в UI уже backend'а**: ползунки дают `0.75–1.25` и `−6…+6`
   полутонов — ровно как в §20/§21; backend принимает `0.5–2.0` и `−12…+12`
   (`RECORDING_SPEED_RANGE`/`RECORDING_PITCH_RANGE`), чтобы API и профиль,
   сохранённый не из интерфейса, не упирались в потолок UI.
8. **Обработка идёт в одном потоке** (`run_in_executor`), прогресс читается
   опросом `render/status` раз в секунду: длинный диалог (десятки реплик)
   собирается последовательно, без параллельного DSP по репликам.

## 16. Backward compatibility

* Существующие вкладки (`01 · Голоса`, `02 · Озвучка диалога`, `03 · Сплошной
  текст`) не изменены: новая вкладка добавлена рядом, TTS-очередь, голоса,
  варианты и API не тронуты.
* Вкладки «Словарь» и «Модели» сохранили разметку, id и логику; изменены только
  два номера в подписях — `04 → 05` и `05 → 06` (кнопка и `eyebrow`), иначе
  после вставки новой вкладки под номером `04` (§1, §33) в панели вкладок
  оказались бы два «04» подряд.
* Записи не попадают в `voices/` и не уходят в F5/XTTS: recorded render — это
  сборка готовых waveform.
* Новые переменные окружения имеют значения по умолчанию, поэтому запуск без них
  даёт прежнее поведение TTS и пустой (но рабочий) раздел записей.


## 17. Definition of Done (§46)

```text
FINAL OUTPUT
- MP3 render: PASS
- 2 voices dialogue: PASS
- 3 voices dialogue: PASS
- role-grouped recording order restoration: PASS
- MP3 decode verification: PASS

TESTS
- full pytest: 1288 passed, 0 failed, 0 errors, 0 skipped, 0 xfailed (exit code 0)
- recording tests: 43 — tests/test_recording_store.py (8) + tests/test_recording_audio.py (14)
  + tests/test_recording_api.py (10) + tests/test_recording_pipeline.py (11: Test A–G, Test I, 3 регресса)
- failures: 0
- skipped: 0
- real microphone E2E: NOT VERIFIED (REAL MICROPHONE E2E: NOT VERIFIED)

KNOWLEDGE BASE
- README updated: YES
- API section updated: YES
- project structure updated: YES
- troubleshooting updated: YES
```
