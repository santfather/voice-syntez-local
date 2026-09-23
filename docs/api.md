# Справочник API

Полный справочник HTTP-интерфейса дашборда: маршруты FastAPI из `backend/main.py` с назначением, параметрами и заметными кодами ответа. Маршруты сгруппированы по темам; ниже таблиц — общие правила формата ответов и примеры `curl`. Синтез, перегенерация и монтаж записи идут через одну очередь, поэтому приоритеты и отмена описаны в разделе [«Статус и разовые задачи»](#статус-и-разовые-задачи).

## Содержание

- [Проекты и рендер](#проекты-и-рендер)
- [Голоса](#голоса)
- [Словарь произношения](#словарь-произношения)
- [Запись микрофона](#запись-микрофона)
- [LLM-анализ](#llm-анализ)
- [Модели и кеш](#модели-и-кеш)
- [Диагностика](#диагностика)
- [Статус и разовые задачи](#статус-и-разовые-задачи)
- [Формат ответов и общие правила](#формат-ответов-и-общие-правила)
- [Интонация и референс в ответе API](#интонация-и-референс-в-ответе-api)
- [Проекты против задач](#проекты-против-задач)
- [Примеры запросов](#примеры-запросов)

## Проекты и рендер

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/api/projects` | Создать проект → сам проект |
| `GET` | `/api/projects` | Список проектов: имя, статус, число реплик, время последней правки |
| `GET` | `/api/projects/{id}` | Проект целиком: исходный текст, настройки сборки, спикеры, реплики с вариантами |
| `PATCH` | `/api/projects/{id}` | Изменить имя, текст, режим, настройки сборки и/или назначения голосов по спикерам |
| `DELETE` | `/api/projects/{id}` | Удалить проект вместе с его записями и файлами кусков |
| `GET` | `/api/projects/{id}/timeline` | Таймлайн: длительность, дорожки спикеров и сегменты реплик с `start_sec`/`end_sec` по активным take'ам (только чтение) |
| `POST` | `/api/projects/{id}/parse` | Разобрать исходный текст проекта в реплики (необязательное тело: `chunk_strategy`) |
| `POST` | `/api/projects/{id}/analyze` | **Обязательная подготовка диалога**: разбор (если реплик нет или `parse: true`) и подготовка реплик — стадии текста, кандидаты в словарь, эффективные параметры. Тело: `chunk_strategy?`, `parse?`, `indexes?` (точечный пересчёт), `auto_accent?`. Подробный отчёт по стадиям подготовки; 400 на неизвестный номер реплики; модели синтеза **не поднимаются** |
| `GET` | `/api/projects/{id}/analysis` | Состояние подготовки без пересчёта: `status`, версия, причина, счётчики реплик, кандидаты, предупреждения (только чтение) |
| `POST` | `/api/projects/{id}/render` | Поставить рендер проекта → `job_id` (приоритет 2; `background: true` — приоритет 3). **409**, пока текст не подготовлен (`/analyze`); синтез идёт по сохранённому `final_text`, а не по сырому тексту реплик |
| `PATCH` | `/api/projects/{id}/replicas/{index}` | Править одну реплику: свой голос, эмоцию (`emotion_override`, `null` — «Авто»), явный референс (`reference_profile_id`), отдельные параметры (`overrides`) или сброс правок |
| `POST` | `/api/projects/{id}/replicas/{index}/regenerate` | Пересинтезировать одну реплику проекта, не трогая остальные → `job_id` (приоритет 1). Тело (необязательно): `emotion_override`, `reference_profile_id` — эмоция применяется до синтеза |
| `POST` | `/api/projects/{id}/replicas/{index}/takes/{take_id}` | Сделать вариант реплики активным — без пересинтеза |
| `GET` | `/api/projects/{id}/replicas/{index}/takes/{take_id}/audio` | Файл варианта реплики проекта — для прослушивания (всегда wav) |
| `POST` | `/api/projects/{id}/export` | Скачать проект архивом `.ttsproject`: метаданные, спикеры, реплики, take'ы и референсы используемых голосов (веса моделей не входят) |
| `GET` | `/api/projects/{id}/export/audio` | Итоговый трек: `?format=wav` (по умолчанию) или `mp3`; собирается из активных take'ов, длительность совпадает с таймлайном |
| `GET` | `/api/projects/{id}/export/replicas` | ZIP с отдельными WAV реплик — по файлу на каждую звучащую реплику |
| `GET` | `/api/projects/{id}/export/stems` | ZIP со stems по спикерам: на месте чужих реплик тишина, дорожки синхронны с итоговым треком |
| `GET` | `/api/projects/{id}/export/transcript` | Транскрипт JSON: реплика, спикер, голос, текст и реальные `start_sec`/`end_sec` |
| `GET` | `/api/projects/{id}/export/subtitles` | Субтитры по таймстемпам таймлайна: `?format=srt` (по умолчанию) или `vtt` |
| `POST` | `/api/projects/import` | Импорт архива `.ttsproject` (multipart, поле `file`) → 201 и карточка нового проекта; 400 с понятным текстом на битый или чужой архив |

## Голоса

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/voices` | Список голосов |
| `POST` | `/api/voices` | Создать голос (multipart: `name`, `gender`, `ref_text`, `file`, `engine`, `verify_ref_text`; `file` необязателен — его не требует движок со встроенными голосами) |
| `POST` | `/api/voices/analyze` | Проверить запись до сохранения (multipart: `file`, `gender`) → тон, длительность, перегрузка, полоса частот |
| `POST` | `/api/voices/transcribe` | Распознать речь в записи (multipart: `file`) → `ref_text` + длительности |
| `PATCH` | `/api/voices/{voice_id}` | Сменить движок голоса, его ручки и/или пресет (`preset`), не трогая референс |
| `POST` | `/api/voices/{id}/references` | Добавить голосу референс-профиль (multipart: `file`, `emotion`, `ref_text`, `label`, `verify_ref_text`, `source_record_phrase_id`, `enabled`, `is_default`, `enabled_for_auto`) — отдельный профиль того же голоса; `emotion` не может быть `AUTO`. Ответ содержит `profile` и `reference_profiles` |
| `PATCH` | `/api/voices/{id}/references/{profile_id}` | Сменить `emotion`, `label`, `enabled`, `is_default` или `enabled_for_auto` профиля, не перезаписывая запись. Именно этим включает/выключается галочка «авто» |
| `DELETE` | `/api/voices/{id}/references/{profile_id}` | Удалить профиль вместе с его файлом (основной нейтральный не удаляется) |
| `DELETE` | `/api/voices/{voice_id}` | Удалить голос |
| `POST` | `/api/voices/{id}/benchmark` | Сравнить голос на движках одной фразой → `benchmark_id` + `job_id` (202; `engines` пустой — все объявленные, неизвестный движок — 400; приоритет 3 — фоновая работа) |
| `GET` | `/api/benchmarks/{benchmark_id}` | Ход и результаты сравнения: строка на движок (`status`, `render_sec`, `duration_sec`, `qa`, `params`, `audio_url`) |
| `GET` | `/api/benchmarks/{benchmark_id}/{engine}/audio` | Файл результата одного движка — для прослушивания (всегда wav) |
| `POST` | `/api/benchmarks/{benchmark_id}/select` | Сделать выбранный в сравнении движок движком голоса (меняет только `engine`) |

## Словарь произношения

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/pronunciation` | Словарь произношения: все правила (включая выключенные) и их число. С `?project_id=` — правила проекта (у них приоритет над глобальными) |
| `POST` | `/api/pronunciation` | Добавить правило; повторный источник обновляет его, а не создаёт дубль. Поле `project_id` кладёт правило в словарь проекта; ответ содержит `affected` — реплики, у которых подготовка устарела |
| `PATCH` | `/api/pronunciation/{id}` | Изменить правило (в том числе `enabled` — не удаляя из словаря); `?project_id=` выбирает область |
| `DELETE` | `/api/pronunciation/{id}` | Удалить правило (`?project_id=` — из словаря проекта); удаление тоже устаревает затронутые реплики |
| `POST` | `/api/pronunciation/preview` | Стадии обработки текста и сработавшие правила — без синтеза и без записи в базу; возвращает `original`, `normalized`, `yo`, `result` и `matches` |
| `POST` | `/api/pronunciation/suggestions` | Кандидаты в словарь из текста: ё-омографы, омографы ударения и редкие слова → `{candidates, considered}` (тело `{text, engine?}`; ничего не пишет, не синтезирует и не поднимает TTS-движок) |
| `POST` | `/api/projects/{id}/pronunciation/review` | Решение по предложенному слову: `scope: project` — принять в словарь проекта, `scope: global` — в общий, пустая замена — пропустить. Затронутые реплики помечаются устаревшими и пересчитываются; в ответе — правило, список затронутых проектов и новое состояние подготовки |

## Запись микрофона

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/recording-projects` | Список проектов записи (вкладка «Сам себе звукорежиссер») |
| `POST` | `/api/recording-projects` | Создать проект записи: `{name, dialogue_text}`; текст сразу разбирается общим парсером |
| `GET` | `/api/recording-projects/{id}` | Проект записи: роли, дубли со ссылками, готовность и состояние монтажа |
| `PATCH` | `/api/recording-projects/{id}` | Правка проекта: имя, текст (переразбирается), порядок записи, настройки сборки |
| `DELETE` | `/api/recording-projects/{id}` | Удалить проект записи вместе с файлами |
| `POST` | `/api/recording-projects/{id}/parse` | Повторно разобрать сохранённый текст диалога |
| `POST` | `/api/recording-projects/{id}/voice-profiles` | Создать записываемый голос: `{name, speed, pitch_semitones, denoise}` |
| `PATCH` | `/api/recording-projects/{id}/voice-profiles/{profile_id}` | Изменить настройки обработки голоса (raw-записи не трогаются) |
| `PUT` | `/api/recording-projects/{id}/roles/{speaker}` | Назначить роли записываемый голос (`profile_id: null` — снять) |
| `POST` | `/api/recording-projects/{id}/replicas/{index}/takes` | Загрузить дубль реплики (multipart, поле `file`) → `{take, project}`; новый дубль сразу становится активным, запись ограничена `TTS_RECORDING_MAX_SEC` (по умолчанию 300 с) |
| `GET` | `/api/recording-projects/{id}/replicas/{index}/takes` | Список дублей реплики |
| `GET` | `/api/recording-projects/{id}/takes/{take_id}/audio` | Сырой дубль (WAV) — для сравнения дублей без обработки |
| `DELETE` | `/api/recording-projects/{id}/replicas/{index}/takes/{take_id}` | Удалить дубль |
| `POST` | `/api/recording-projects/{id}/replicas/{index}/takes/{take_id}/select` | Сделать дубль активным |
| `POST` | `/api/recording-projects/{id}/takes/{take_id}/preview` | Превью обработки дубля (`speed`, `pitch_semitones`, `denoise`) → `{preview_id, url, settings, warnings}`; незаданные поля берутся у голоса, указанные — применяются только к этому превью |
| `GET` | `/api/recording-projects/{id}/previews/{name}/audio` | Аудио превью обработки |
| `POST` | `/api/recording-projects/{id}/render` | Собрать диалог (202); необязательное тело `pause_ms` перебивает сохранённую настройку сборки. **409** с `detail.missing`, если есть незаписанные реплики; 409, если монтаж уже идёт |
| `GET` | `/api/recording-projects/{id}/render/status` | Состояние монтажа: `status` (`idle`/`running`/`done`/`error`), `progress`, `message`, `error`, `duration_sec`, `warnings`, `url` и метки времени |
| `GET` | `/api/recording-projects/{id}/audio` | Готовый диалог одним файлом (`?download=true` — с именем файла): по умолчанию MP3 (`TTS_RECORDING_FORMAT`) |

## LLM-анализ

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/llm/status` | Готовность локального анализатора: включён ли, доступна ли Ollama, какая модель выбрана (primary/fallback), версии prompt/schema. Работает и при выключенной Ollama |
| `GET` | `/api/llm/models` | Скачанные модели Ollama с ролями primary/fallback |
| `POST` | `/api/llm/settings` | Включить/выключить анализатор и сменить модель → сохранённые настройки (пишет `data/llm_settings.json`, анализатор пересоздаётся сразу) |
| `GET` | `/api/projects/{id}/linguistic-analysis` | Состояние LLM-анализа проекта (`DISABLED/PENDING/RUNNING/READY/NEEDS_REVIEW/FAILED/STALE`), модель, причина отказа и кандидаты LLM с согласием/конфликтом с детерминированным слоем. Только чтение |
| `POST` | `/api/projects/{id}/linguistic-analysis` | Запустить лингвистический анализ — **тот же** канонический проход, что `/analyze` (второго pipeline нет), плюс состояние LLM в ответе |

## Модели и кеш

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/api/engines` | Паспорта движков: подписи, описания, поддержка ударений, клонирования и профилей просодии, языки, объявленный откат, режимы работы, встроенные голоса (`builtin_voices`) и границы ручек (из этого строится UI) |
| `POST` | `/api/engines/{id}/unload` | Выгрузить движок и вернуть память без перезапуска (404 на неизвестный; 409, если движок синтезирует или очередь занята; повторная выгрузка — no-op) |
| `POST` | `/api/engines/{id}/load` | Поднять выгруженный движок заранее, без синтеза (404 на неизвестный; 500 с причиной, если модель не поднялась) |
| `GET` | `/api/models` | Модели: установка, размер, путь, состояние движка и скачивания + сводка по диску. В сеть не ходит |
| `GET` | `/api/models/{id}` | Одна модель — для опроса прогресса скачивания |
| `POST` | `/api/models/{id}/download` | Запустить скачивание модели в фоне (202; установленная — идемпотентный no-op, кеш HF — 400) |
| `DELETE` | `/api/models/{id}` | Удалить файлы модели (409, пока движок занят; кеш HF — 400) |
| `GET` | `/api/cache` | Занятое место по трём категориям своего кеша (+ `temp_files_count`); ничего не удаляет |
| `POST` | `/api/cache/clear` | Очистить выбранные категории → `freed_mb`/`freed_files`/`total_mb` (400 на неизвестную и пустой выбор) |
| `POST` | `/api/reset` | Сброс вкладки → `scope` (`cache`/`full`) + `tab` (`dialogue`/`text`), для диалога `project_id`; в ответе `cache`/`project_deleted`/`diagnostics_deleted`/`llm_settings_reset` (400 на неизвестные `scope`/`tab`, 409 при занятой очереди на `full`) |

## Диагностика

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/api/projects/{id}/diagnostics` | Собрать архив диагностики проекта (тело: `job_id`, `include_references`, `max_audio_mb`) → имя, размер, путь, предупреждения и список архивов проекта. Ничего не меняет; без `job_id` описывает проект, с ним — ещё и тот рендер (`file_name`) |
| `GET` | `/api/projects/{id}/diagnostics` | Собранные архивы диагностики проекта — от свежих к старым |
| `GET` | `/api/projects/{id}/diagnostics/{name}` | Скачать архив диагностики (имя проверяется: `..`, подкаталоги и чужие расширения отклоняются) |
| `DELETE` | `/api/projects/{id}/diagnostics/{name}` | Удалить архив диагностики |
| `GET` | `/api/diagnostics/worker` | Диагностика процесса синтеза: включена ли изоляция, состояние воркеров (pid, перезапуски, последний отказ) и последние падения с кодом, сигналом и репликой (`?limit=`, `?project_id=`). Читает память и базу — модель не ждёт |

## Статус и разовые задачи

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/` | Дашборд |
| `GET` | `/healthz` | Живость сервиса для внешнего сторожа: `{"status": "ok", "database": "ok", "engines": {…}, "request_id": "…"}`, если база открывается и `PRAGMA quick_check` её не ругает, а среди созданных движков нет упавших (`failed`); иначе — 503 и `status: "degraded"` с причиной. Модель не ждёт и движки не поднимает: `idle` у неподнятого движка — норма. Краткая версия `/api/status`; в схему OpenAPI не входит |
| `GET` | `/api/status` | Готовность каждого движка и акцентуатора, устройство, размер очереди, RSS, CPU, `memory_state` (NORMAL/WARNING/CRITICAL с причиной), `workers[]` (состояние процессов синтеза) и `mps_memory` (счётчики MPS или `null`). Модель не ждёт |
| `POST` | `/api/parse` | Разобрать диалог → реплики + список спикеров (400 при слишком длинном куске) |
| `POST` | `/api/text/preview` | Что услышит модель: стадии `original → normalized → yo → dictionary → accentized → final`, сработавшие правила, движок и состояние RUAccent — без синтеза, загрузки движка и записи в базу. Текст ограничен `TTS_MAX_PREVIEW_CHARS` (по умолчанию 2000), а не общим `TTS_MAX_TEXT_CHARS`: стадии считаются синхронно и панель ждёт их глазами |
| `POST` | `/api/preview` | Прослушать голос: синтез одной фразы → `job_id` (приоритет 0) |
| `POST` | `/api/generate` | Поставить задачу по диалогу → `job_id` (приоритет 2; `background: true` — приоритет 3) |
| `POST` | `/api/render-text` | Поставить задачу по сплошному тексту (один голос) → `job_id` (приоритет 2; `background: true` — приоритет 3) |
| `GET` | `/api/jobs/{job_id}` | Статус, прогресс, ETA (`eta_sec` + готовая строка `eta_text`); у ошибки — `error_type` (`WORKER_CRASH`, `WORKER_TIMEOUT`, `TTS_ERROR`, `AUDIO_ERROR`, `CANCELLED`, `WATCHDOG`, `INTERRUPTED`) и `error_title`; у готовой задачи ещё и `replicas` — куски с их местом в файле, сидом, отметкой проверки качества и списком вариантов |
| `POST` | `/api/jobs/{job_id}/cancel` | Отменить задачу: ожидающую — сразу (`cancelled`), идущую — на ближайшей безопасной точке (200, `{"job": …}`; 404 на неизвестный id, повторная отмена идемпотентна) |
| `GET` | `/api/jobs/{job_id}/audio` | Готовый файл (`?download=true` — как вложение) |
| `POST` | `/api/jobs/{job_id}/replicas/{index}/regenerate` | Пересинтезировать одну реплику готового файла, не трогая остальные (приоритет 1) |
| `POST` | `/api/jobs/{job_id}/replicas/{index}/variants/{variant_id}` | Поставить в файл сохранённый вариант реплики (без пересинтеза) |
| `GET` | `/api/jobs/{job_id}/replicas/{index}/variants/{variant_id}/audio` | Файл варианта — для прослушивания (всегда wav) |

## Формат ответов и общие правила

`engine_params` — ручки движка, переопределяющие настройки голоса на одну задачу; объявленные
границы зажимаются, незнакомые ключи движок игнорирует. Отдельный ключ `mode` (`draft` /
`quality` / `experimental`) выбирает режим работы движка: пресет режима накладывается на дефолты,
а явно переданные ручки побеждают; у движков без объявленных режимов ключ ничего не меняет.
Пока задача пересинтезирует реплику
или ставит вариант, `status` остаётся прежним (`done`), а индекс занятой реплики лежит в
отдельном поле `regenerating_replica` (`null` в покое); при неудаче текст ошибки — в `regen_error`
(отмена пересборки тоже пишет туда `"Отменено"`, а сам рендер остаётся `done`).
Необязательное поле `background: true` у рендер-запросов (`/api/generate`, `/api/render-text`,
`/api/projects/{id}/render`) опускает задачу в приоритет 3: её пропустят вперёд прослушивание и
перегенерация. У задачи есть поле `cancel_requested` — `true` между запросом отмены и безопасной
точкой; статусы: `queued`, `processing`, `done`, `error`, `cancelled` (см. [«Очередь и отмена»](operations.md)).
У реплики в `replicas` есть `seed` (текущее звучание) и `variants` — список
`{id, label, seed, duration_sec, active, quality, audio_url}`; выбор и постановка варианта идут через
ту же очередь, что и синтез, поэтому одновременно движутся только по одному. Поле `qa` —
итог проверки качества этого звучания (`{status, wer, attempts, mode, screening}`) или `null`,
если проверка не гонялась. Поле `quality` — диагностические метрики этого звучания
(`{duration_sec, chars, clipping, clipping_ratio, silence_ratio, duration_per_char, peak_dbfs,
rms_dbfs, lufs, wer, qa_attempts, qa_mode, screening_reasons, warnings}`) или `null`, если
метрик нет: так выглядят take'ы, сохранённые до появления диагностики. Каждое поле внутри
необязательно (`null` — «не измерено»); `warnings` — список `{code, text}` с русскими
формулировками, `screening_reasons` — коды причин `qa_screening`. Оценки естественности голоса
среди них нет: метрики не сворачиваются в балл и take'ы по ним не ранжируются (см. «Диагностика take'а» в [диалоге](dialogue.md)). `quality` есть у take'ов проекта (`GET /api/projects/{id}`), у вариантов
задачи и у активного звучания реплики в `GET /api/jobs/{id}`. Режим задаётся строкой `qa: "off" | "smart" | "strict"` в
`/api/generate`, `/api/render-text` и `/api/projects/{id}/render`; в проекте он лежит среди
настроек сборки, поэтому незаданное поле в запросе сохранённый режим не сбрасывает. Старое
булево значение принимается как есть (`true` → `strict`, `false` → `off`), неизвестное — 422.
Числа цикла берутся из `TTS_QA_*`, границы отбора — из `QA_SCREEN_*`
(см. [«Проверка качества синтеза»](dialogue.md)). У `smart` в `screening` видно, что решил отбор: `suspicious`
и причины (`empty`, `too_short`, `duration_short`, `duration_long`, `silence`, `clipping`,
`level`, `repeat`), а `wer` у принятого без расшифровки куска — `null`.

**Тело запроса ограничено по размеру.** JSON-запросы — 8 МБ (`TTS_MAX_JSON_BODY_BYTES`),
multipart-загрузки (референс голоса, запись) — 96 МБ (`TTS_MAX_UPLOAD_BYTES`): превышение даёт
413 до того, как тело прочитано целиком. Предел проверяется и по объявленному `Content-Length`,
и по фактически прочитанным байтам, поэтому чанковая отправка его не обходит. Импорт
`.ttsproject` ограничен ещё и отдельно: размер архива, число записей и суммарный распакованный
объём (последний считается по заголовкам ZIP, без распаковки).

**У каждого ответа есть `x-request-id`.** Клиент может прислать свой заголовок `x-request-id`
(валидное значение — до 64 знаков из `A-Za-z0-9._-`), иначе сервер сгенерирует своё; значение
попадает в тот же заголовок ответа, в строку журнала (`logs/voice_syntez.log` — в квадратных
скобках после уровня) и в поле `request_id` ответов `/healthz` и ответа об ошибке. Так жалоба
«вот этот запрос» связывается с нужными строками журнала.

**Необработанное исключение — всегда 500 с общим текстом.** Ошибка, до которой не дошло своего
обработчика, логируется на сервере с трейсбеком, а наружу уходит только
`{"detail": "Внутренняя ошибка сервера. Подробности — в журнале приложения.", "request_id": "…"}`:
тип исключения, путь и внутренние детали клиенту не отдаются. Ошибки с понятным пользователю
смыслом (недоступный движок, занятый ресурс, битый архив) по-прежнему приходят со своим текстом и
своим кодом.

## Интонация и референс в ответе API

У реплики проекта (`GET /api/projects/{id}` и `GET /api/jobs/{id}`) есть два блока
метаданных интонации — они **не** входят в текст и не меняют `final_text`:

```jsonc
"prosody": {
  "profile": "QUESTION",          // рекомендация анализатора (prosody_profile)
  "profile_title": "Вопрос",
  "intensity": 0.55,              // null — «модель не сказала»
  "pace": "NORMAL",               // SLOW | NORMAL | FAST
  "confidence": 0.94,             // null, если неизвестно
  "override": null,               // ручной выбор (= emotion_override), null — «Авто»
  "effective": "QUESTION",        // override → recommended → NEUTRAL
  "effective_title": "Вопрос",
  "source": "llm",                // llm | heuristic | none
  "dialogue_act": "QUESTION",
  "context_dependency": "HIGH"    // LOW | MEDIUM | HIGH
},
"reference": {
  "profile_id": "…-question",     // фактически выбранный профиль
  "profile_key": "QUESTION",
  "emotion": "QUESTION",
  "fallback_used": false,         // true — откат на другой профиль
  "fallback_reason": ""           // причина словами, если откат был
}
```

- **Override.** `emotion_override` (и явный `reference_profile_id`) задаются в
  `PATCH /api/projects/{id}/replicas/{index}` и в теле
  `POST /api/projects/{id}/replicas/{index}/regenerate`. `null` в `emotion_override`
  означает «Авто». Правка — это metadata: повторный анализ не нужен.
- **Результат резолвера.** `reference` — проекция `ResolvedReference`: `profile_id`,
  `profile_key`/`emotion`, `fallback_used`, `fallback_reason`; в метаданных take он
  сохраняется как `reference_profile_id`/`reference_emotion`/`reference_fallback_used`.
- **Fallback-поля.** `fallback_used` и `fallback_reason` заполняются, когда нужного
  профиля нет и взят другой **того же** голоса; причина — готовая русская строка
  (например «референса для интонации IRONIC нет — взят нейтральный»).
- **Совместимость и миграции.** Формат `voices.json` не менялся: профили добавлены
  внутри записи голоса, старый одиночный референс проецируется в нейтральный профиль
  на чтение. Поля интонации у реплик добавлены аддитивными миграциями 9 и 10;
  проекты и голоса, сохранённые раньше, читаются без пересчёта.
- **ReferenceProfile в API голоса.** `GET /api/voices` отдаёт у каждого голоса
  `reference_profiles[]` с полями `id`, `profile_key`, `emotion`, `emotion_title`,
  `label`, `has_audio`, `source`, `quality_status`, `quality_note`,
  `transcription_status`, `duration_sec`, `engine_compatibility`, `is_default`,
  `enabled`, `enabled_for_auto`, `created_at`.
- **Встроенные голоса движков.** У голоса пресетного движка (`supports_cloning: false`) файла
  референса нет вовсе: `has_audio` у него `false`, `ref_text` пуст, `reference_profiles` пуст,
  а `POST /api/voices` создаёт такую карточку **без** `file`. Голоса, объявленные движком
  (`builtin_voices` в `GET /api/engines`), приложение заводит в списке само, как только веса
  движка скачаны; удалённую пользователем карточку оно заново не создаёт.

## Проекты против задач

Задача (`/api/generate`, `/api/render-text`) — разовая: файл готов, а её карточка живёт только
в памяти процесса и исчезает при перезапуске. **Проект** — сохраняемая сущность: исходный текст,
настройки сборки, назначенные голоса, разобранные реплики и варианты их звучания лежат в SQLite
(`data/voice_syntez.db`) и переживают перезапуск backend. Файлы кусков проекта пишутся отдельно
от транзитного `output/` — в `output/projects/{id}/` — и не удаляются по 24-часовому TTL.
У каждого take там же лежат `qa` (вердикт проверки) и `quality` (диагностические метрики,
колонка `takes.quality`, миграция 4); записи, сделанные до её появления, читаются с
`quality: null`.

Порядок работы: создать проект → положить текст → `PATCH` с назначением голосов спикерам →
`parse` → **`analyze`** → `render`. Разбор и состав реплик — отдельный шаг: правка текста не запускает синтез
сама по себе. Голоса, уже назначенные спикерам, при повторном `parse` сохраняются, а у реплик,
чей текст и спикер не менялись, остаются их варианты и выбранный take — правится одна строка,
а не весь диалог. Возвращает проект целиком каждый изменяющий запрос: интерфейсу не нужно
досбирать состояние из нескольких ответов.

**Синтез возможен только по подготовленному тексту.** Обязательный шаг `analyze` считает для
каждой реплики стадии подготовки (нормализация → восстановление «ё» → словарь произношения →
ударения) и сохраняет итог в `final_text`; рендер берёт именно его и ничего не пересчитывает.
Подробности — в разделе [«Обязательная подготовка диалога»](dialogue.md).

Статусы проекта: `draft` → `rendering` → `rendered` (или `error` с текстом в `last_error`).
Статус `rendered` ставится после того, как куски записаны вариантами реплик: открытый следом
проект уже показывает этот рендер, а не пустую историю.

Правки реплик идут точечным `PATCH`: непришедшее поле значит «не трогать», а `null` — «вернуть
наследуемое у спикера» (голос, параметр, отдельную ручку движка). Ответ содержит только карточку
правленой реплики, поэтому смена голоса или скорости одной строки не задевает соседние —
сравнение вариантов на слух не должно пересобирать весь диалог. Параметры реплики ложатся поверх
карточки спикера, карточка спикера — поверх пресета голоса, пресет — поверх дефолтов движка
(см. [«Иерархия настроек»](synthesis.md)), а `reset_overrides` разом снимает все правки строки.

В ответе у реплик и у спикеров есть `settings` — разрешённые значения вместе с источником
(`engine` / `voice` / `speaker` / `replica`) и значением, к которому вернёт сброс. По ним
интерфейс и подписывает, что унаследовано, а что задано здесь; своей копии правил склейки у
фронтенда нет.

В интерфейсе с проектом работает вкладка «Озвучка диалога»: Панель 1 — редактор реплик,
Панель 4 — сборка файла. `/api/generate` и `/api/render-text` остаются для разовых задач,
скриптов и нагрузочного теста.

## Примеры запросов

```bash
curl -X POST http://localhost:8000/api/voices \
  -F "name=Иван" -F "gender=male" -F "ref_text=Текст из референса." -F "file=@ivan.wav" \
  -F "engine=f5"

# Перевести готовый голос на XTTS и настроить его ручки, не перезаписывая референс:
curl -X PATCH http://localhost:8000/api/voices/<id> \
  -H "Content-Type: application/json" \
  -d '{"engine": "xtts", "engine_params": {"temperature": 0.8, "repetition_penalty": 6.0}}'

# Записать подобранное в «Прослушать» пресетом голоса (пустой объект — сбросить пресет):
curl -X PATCH http://localhost:8000/api/voices/<id> \
  -H "Content-Type: application/json" \
  -d '{"preset": {"speed": 0.95, "cfg_strength": 2.3}}'

curl http://localhost:8000/api/engines

# Словарь произношения: добавить правило, посмотреть список и проверить текст.
curl -X POST http://localhost:8000/api/pronunciation \
  -H "Content-Type: application/json" \
  -d '{"source": "OpenAI", "target": "оупен эй-ай", "note": "название продукта"}'

# Правило с ударением: «+» дойдёт до F5 и будет вырезан для XTTS.
curl -X POST http://localhost:8000/api/pronunciation \
  -H "Content-Type: application/json" \
  -d '{"source": "звонит", "target": "звон+ит"}'

curl http://localhost:8000/api/pronunciation

# Выключить правило, не удаляя его:
curl -X PATCH http://localhost:8000/api/pronunciation/<id> \
  -H "Content-Type: application/json" -d '{"enabled": false}'

# Что услышит конкретный движок: original → normalized → yo → result
# плюс список сработавших правил.
curl -X POST http://localhost:8000/api/pronunciation/preview \
  -H "Content-Type: application/json" \
  -d '{"text": "Мы используем PostgreSQL и SQL.", "engine": "xtts"}'

# Кандидаты в словарь: неоднозначные «е/ё», омографы ударения и редкие слова.
# Ничего не пишется и не синтезируется; подтверждение — обычный POST выше.
curl -X POST http://localhost:8000/api/pronunciation/suggestions \
  -H "Content-Type: application/json" \
  -d '{"text": "Мы все пойдем в замок, прошел дождь.", "engine": "f5"}'
# → {"candidates": [{"word": "все", "target": "всё", "kind": "yo_homograph", …}, …],
#    "considered": 8}

# Что услышит модель: все стадии preprocessing, без синтеза. Голос задаёт движок,
# engine — запасной вариант, если голоса нет; project_id + replica_index берут
# текст и голос конкретной реплики проекта.
curl -X POST http://localhost:8000/api/text/preview \
  -H "Content-Type: application/json" \
  -d '{"text": "В 2026 году цена выросла на 5%.", "engine": "f5"}'
# → {"original": …, "normalized": …, "yo": …, "dictionary": …,
#    "accentized": …, "final": …, "matches": [...], …}

curl -X POST http://localhost:8000/api/text/preview \
  -H "Content-Type: application/json" \
  -d '{"project_id": "<project_id>", "replica_index": 12}'

curl -X DELETE http://localhost:8000/api/pronunciation/<id>

# Проект: создать, назначить голоса спикерам, разобрать текст и запустить рендер.
curl -X POST http://localhost:8000/api/projects -H "Content-Type: application/json" -d '{
  "name": "Пилот", "mode": "dialogue",
  "source_text": "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."
}'

curl -X PATCH http://localhost:8000/api/projects/<project_id> -H "Content-Type: application/json" -d '{
  "speakers": {"ИВАН": {"voice_id": "<id-f5>"}, "МАРГО": {"voice_id": "<id-xtts>"}}
}'

curl -X POST http://localhost:8000/api/projects/<project_id>/parse
curl -X POST http://localhost:8000/api/projects/<project_id>/render \
  -H "Content-Type: application/json" -d '{"output_format": "mp3", "qa": "smart"}'
curl http://localhost:8000/api/projects/<project_id>   # реплики, их варианты и выбранные take

# Редактор реплик: свой голос и параметры ровно у одной строки, и сброс к спикеру.
curl -X PATCH http://localhost:8000/api/projects/<project_id>/replicas/12 \
  -H "Content-Type: application/json" \
  -d '{"voice_id": "<id-xtts>", "overrides": {"speed": 1.1, "pause_override_ms": 350}}'
curl -X PATCH http://localhost:8000/api/projects/<project_id>/replicas/12 \
  -H "Content-Type: application/json" -d '{"overrides": {"speed": null}}'
curl -X POST http://localhost:8000/api/projects/<project_id>/replicas/12/regenerate  # новый take
curl -X POST http://localhost:8000/api/projects/<project_id>/replicas/12/takes/3      # выбрать take

# Таймлайн: где какая реплика звучит в готовом файле. Границы считаются по активным
# take'ам при каждом запросе, поэтому замена take сразу пересчитывает хвост; запрос
# ничего не синтезирует и не пишет. 404 — проект не найден.
curl http://localhost:8000/api/projects/<project_id>/timeline
# → {"project_id": "…", "duration_sec": 102.4, "settings": {"pause_ms": 350, "cross_fade_duration": 0.15},
#    "speakers": [{"key": "#1", "label": "АРТЕМ", "voice_id": "…", "voice_name": "Vlad"}],
#    "replicas": [{"index": 0, "speaker": "#1", "speaker_label": "АРТЕМ", "engine": "f5",
#                  "start_sec": 0.0, "end_sec": 3.2, "duration_sec": 3.2, "pause_ms": 0,
#                  "take_id": 12, "take_label": "исходный", "has_audio": true,
#                  "status": "rendered", "qa": null, "text": "…"}, …]}
#
# `pause_ms` — эффективная пауза реплики (личная правка важнее паузы спикера, та — общей).
# Перед первой репликой она записана, но в файл не вставляется: сборка ставит паузу только
# перед каждым следующим куском. Реплика без take идёт с `duration_sec: 0`,
# `has_audio: false`, `take_id: null` и не двигает курсор.

# Экспорт проекта и аудио. Ни один из этих запросов не синтезирует: файл собирается
# из готовых take'ов по таймлайну, поэтому его длительность совпадает с `/timeline`.
curl -o project.ttsproject http://localhost:8000/api/projects/<project_id>/export
curl -o dialogue.wav  "http://localhost:8000/api/projects/<project_id>/export/audio?format=wav"
curl -o dialogue.mp3  "http://localhost:8000/api/projects/<project_id>/export/audio?format=mp3"
curl -o replicas.zip  http://localhost:8000/api/projects/<project_id>/export/replicas
curl -o stems.zip     http://localhost:8000/api/projects/<project_id>/export/stems
curl -o transcript.json http://localhost:8000/api/projects/<project_id>/export/transcript
curl -o subtitles.srt "http://localhost:8000/api/projects/<project_id>/export/subtitles?format=srt"
curl -o subtitles.vtt "http://localhost:8000/api/projects/<project_id>/export/subtitles?format=vtt"

# Импорт архива: создаётся новый проект, существующие не перезаписываются.
# Голоса сопоставляются по имени; отсутствующие создаются из референсов архива.
curl -X POST http://localhost:8000/api/projects/import -F "file=@project.ttsproject"
# → 201 и карточка проекта (то же, что отдаёт GET /api/projects/<id>)
# Битый архив, отсутствие project.json или чужая версия формата → 400 с текстом ошибки.
# Слишком большой архив, архив с чрезмерным числом записей или распакованным объёмом
# («архив-бомба») отклоняется до распаковки — 413/400.

# Диалог, где у спикеров разные движки (движок берётся из карточки голоса):
curl -X POST http://localhost:8000/api/generate -H "Content-Type: application/json" -d '{
  "dialogue_text": "Иван: Привет!\nМария: Привет, как дела?",
  "speakers": {
    "Иван":  {"voice_id": "<id-f5>"},
    "Мария": {"voice_id": "<id-xtts>", "engine_params": {"temperature": 0.8}}
  },
  "pause_ms": 400, "auto_accent": true, "output_format": "mp3"
}'

curl http://localhost:8000/api/jobs/<job_id>
curl -o dialogue.mp3 "http://localhost:8000/api/jobs/<job_id>/audio?download=true"

# Отмена: ожидающая задача становится `cancelled` сразу, идущая прерывается на
# ближайшей безопасной точке (между репликами). Повторный вызов — не ошибка.
curl -X POST http://localhost:8000/api/jobs/<job_id>/cancel
# → {"job": {"job_id": "…", "status": "cancelled", "cancel_requested": true, "message": "…"}}

# Фоновый рендер: уходит в приоритет 3, пропуская вперёд прослушивание и перегенерацию.
curl -X POST http://localhost:8000/api/projects/<project_id>/render \
  -H "Content-Type: application/json" -d '{"background": true}'

# Сравнение голоса на движках: одна фраза, один reference, отдельный take на движок.
# engines можно не указывать — тогда сравниваются все объявленные движки.
curl -X POST http://localhost:8000/api/voices/<voice_id>/benchmark \
  -H "Content-Type: application/json" \
  -d '{"text": "Сегодня хорошая погода, и мы можем спокойно обсудить дело.",
       "engines": ["f5", "xtts", "xtts-banana"], "qa": "smart"}'
# → {"benchmark_id": "…", "job_id": "…", "engines": ["f5", "xtts", "xtts-banana"]}

curl http://localhost:8000/api/benchmarks/<benchmark_id>          # ход и строки результатов
curl -o f5.wav http://localhost:8000/api/benchmarks/<benchmark_id>/f5/audio
curl -o xtts.wav http://localhost:8000/api/benchmarks/<benchmark_id>/xtts/audio

# Выбрать движок вручную: меняется только engine у голоса (reference и пресет не трогаются).
curl -X POST http://localhost:8000/api/benchmarks/<benchmark_id>/select \
  -H "Content-Type: application/json" -d '{"engine": "xtts"}'

# Менеджер моделей: что установлено, что поднято, сколько занимает диск.
curl http://localhost:8000/api/models
# → {"models": [{"id": "f5", "installed": true, "loaded": false, "size_bytes": 1350000000,
#                "path": "…/models", "download": {"state": "idle", …}}, …],
#    "disk": {"models_bytes": …, "free_bytes": …, "total_bytes": …}}

curl http://localhost:8000/api/models/xtts          # одна модель: прогресс и ошибка скачивания
curl -X POST http://localhost:8000/api/models/xtts/download   # 202; качает в фоне
curl -X DELETE http://localhost:8000/api/models/xtts          # 409, пока движок занят
# → {"detail": "Модель «XTTS v2 (базовая)» сейчас используется движком. …"}

# Выгрузка неиспользуемого движка: память возвращается без перезапуска backend.
curl -X POST http://localhost:8000/api/engines/xtts/unload
# → {"engine": "xtts", "state": "idle", "unloaded": true,
#    "message": "Движок «XTTS v2 (базовая)» выгружен: память освобождена."}
# Занят синтезом или очередью — 409 с текстом, что именно мешает:
# → {"detail": "В очереди есть задача: движок может понадобиться следующей реплике. …"}

# Вернуть движок обратно, не запуская синтез:
curl -X POST http://localhost:8000/api/engines/xtts/load
# → {"engine": "xtts", "state": "ready", "loaded": true, "message": "Движок «XTTS v2 (базовая)» поднят."}

# Память MPS (Apple Silicon) в статусе: null, если torch/MPS недоступны.
curl http://localhost:8000/api/status | python -c "import json,sys; print(json.load(sys.stdin)['mps_memory'])"
# → {'current_allocated_mb': 1240.5, 'driver_allocated_mb': 2100.0}

# Очистка кеша приложения: что занимают три категории и сколько освободит их очистка.
curl http://localhost:8000/api/cache
# → {"output_mb": 128.4, "benchmarks_mb": 21.0, "temp_files_mb": 4.2, "temp_files_count": 3}

curl -X POST http://localhost:8000/api/cache/clear \
  -H "Content-Type: application/json" \
  -d '{"targets": ["output", "benchmarks", "temp_files"]}'
# → {"freed_mb": {"output": 128.4, "benchmarks": 21.0, "temp_files": 4.2},
#    "freed_files": {"output": 12, "benchmarks": 3, "temp_files": 3}, "total_mb": 153.6}

# Неизвестная категория и пустой список — 400 с понятным текстом; «голый список» тоже принимается:
curl -X POST http://localhost:8000/api/cache/clear -H "Content-Type: application/json" -d '["output"]'
# → {"freed_mb": {"output": 0.0}, "freed_files": {"output": 0}, "total_mb": 0.0}

# Сброс вкладки. «Только кеш» — тот же кеш, что и выше, без данных вкладки:
curl -X POST http://localhost:8000/api/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "cache", "tab": "text"}'
# → {"scope": "cache", "tab": "text", "cache": {"freed_mb": {...}, "freed_files": {...}, "total_mb": 153.6},
#    "project_deleted": null, "diagnostics_deleted": 0, "llm_settings_reset": false}

# «Полный сброс вкладки»: кеш, открытый проект диалога (если он открыт) и настройки анализатора.
# 409, если в очереди есть задача; 400 на неизвестные scope/tab.
curl -X POST http://localhost:8000/api/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "full", "tab": "dialogue", "project_id": "p_12ab34"}'
# → {"scope": "full", "tab": "dialogue", "cache": {"total_mb": 0.0},
#    "project_deleted": "p_12ab34", "diagnostics_deleted": 2, "llm_settings_reset": true}
```

## См. также

- [Вкладки интерфейса](guide-tabs.md) — что делает каждая панель и какой маршрут вызывает.
- [Озвучка диалога](dialogue.md) — проекты, подготовка реплик, QA и таймлайн.
- [Запись микрофона](recording.md) — вкладка «Сам себе звукорежиссер» целиком.
- [Лингвистический анализ](llm-analysis.md) — `/api/llm/*` и `linguistic-analysis`.
- [Эксплуатация](operations.md) — очередь, отмена, кеш и восстановление.
