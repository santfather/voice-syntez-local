# Reference Prosody Transfer Benchmark — результаты и состояние

Документ фиксирует, **что измерено**, а что нет, по benchmark'у переноса просодии
(UPDATE 3 §29–§35, §72–§74). Корпус и инструмент — в `benchmarks/prosody/`,
база знаний по профилям и просодии — в `reference_profiles.md`.

Дата: 2026-09-19.

## Главное правило этого документа

**Неподтверждённые выводы здесь не пишутся.** Там, где измерение не проводилось,
стоит «не измерено», а не предположение. Прямое следствие: **ни один профиль не
разрешён для автоматического выбора** (`enabled_for_auto = false` у всех), пока
перенос не подтверждён прослушиванием (§35).

## 1. Что подтверждено

| Что | Статус | Чем подтверждено |
|---|---|---|
| Корпус реплик v1 готов и валиден | да | `tools/prosody_benchmark.py --check` (без моделей) |
| Матрица и отчёты строятся без моделей | да | `--plan`, тесты `tests/test_prosody_benchmark*.py` на `StubEngine` |
| Резолвер детерминирован и не выходит за голос | да | `tests/test_prosody_resolver.py` (в т.ч. `test_fallback_never_leaves_the_voice`) |
| Gate §35 соблюдён на живом приложении | да | прогон API через `httpx.ASGITransport` на боевом `main.app` (11 проверок) |
| Профиль по умолчанию **не** участвует в автоматике | да | `enabled_for_auto = false`; `_usable(check_auto=...)` |
| Возврат к нейтральному — штатный путь, а не ошибка | да | `reference_fallback_used` / `reference_fallback_reason` в ответе реплики |
| Полный прогон матрицы на реальных моделях (F5 / XTTS) | **не измерено** | профилей в `voices.json` нет, `output/benchmarks/prosody/` пуст |
| Listening review | **не выполнено** | требует человека; заключений нет |
| Real dialogue benchmark (§74) | **не выполнен** | — |

Что именно проверял живой прогон gate §35 (без синтеза):

1. профиль с `enabled_for_auto = false` не выбирается автоматикой;
2. тот же профиль доступен при явном `reference_profile_id`;
3. автоматика отказывает с причиной словами
   («профиль не подтверждён для автоматического выбора»);
4. подтверждённый профиль (`enabled_for_auto = true`) берётся автоматикой;
5. снятие подтверждения снова закрывает автоматику;
6. NEUTRAL гейтом не затронут.

## 2. Hardware

Из паспорта прогонов (`benchmarks/russian_linguistics/results/full/*/run.json`):

| Параметр | Значение |
|---|---|
| Машина | Apple M3 Pro, `arm64` |
| ОЗУ | 18.0 GB |
| ОС | macOS, Darwin 25.6.0 |
| Ускоритель инференса | PyTorch MPS (с фолбэком на CPU) |

Именно на этой конфигурации снимались все предыдущие реальные замеры проекта
(короткие реплики, LLM-benchmark); других машин в замерах нет.

## 3. Engine

Benchmark разделяется по движкам **обязательно** (§29) — перенос просодии на F5 и
XTTS нельзя считать одинаковым. Известно из UPDATE 2: движки по-разному ведут себя
даже на коротких репликах, поэтому переносить выводы с одного на другой нельзя.

| Движок | Прогон переноса просодии |
|---|---|
| F5-TTS (v1 Base accent-tune, MPS) | не измерено |
| XTTS v2 (базовая) | не измерено |
| XTTS v2 Banana | не измерено |

## 4. Voices

Доступные голоса (`voices/voices.json`), пригодные для матрицы:

| id | Имя | Пол | Движок |
|---|---|---|---|
| `ff79a6feebaf` | Мари 2 | female | f5 |
| `0ce60e31e3e0` | Влад | male | f5 |
| `d3c6fc2b4815` | Мария | female | xtts |
| `78e40f408fe3` | BananaTest | female | xtts-banana |
| `70ddd8f1acb8` | REF_RU (тест) | male | f5 |

**Reference Profiles у них сейчас нет** (`profiles: []` у каждого), поэтому матрицу
физически не на чем строить: сначала нужно начитать профили (`RECORD_PHRASES`,
`reference_profiles.md` §4), затем прогнать benchmark.

## 5. Profiles

Фиксируется всё, кроме профиля (§32):

```text
voice, engine, seed, speed, engine params, final_text
```

Меняется только `ReferenceProfile`. Сравниваются, где доступны (§32):
`NEUTRAL`, `QUESTION`, `DELIGHT`, `IRONIC`, `STRICT`, `EXCITED`.

Статус переноса по профилям — **не измерено** (таблица §6 пуста не случайно).

## 6. Seeds

Сид один на всю матрицу (`DEFAULT_SEED = 0`, `--seed`) и входит в метаданные
каждой ячейки. Числа ниже не приводятся: прогона не было.

## 7. Targets (корпус)

`benchmarks/prosody/corpus.v1.jsonl` — **36 реплик**, валидны при ≥30 (§31):

| Категория | Реплик |
|---|---|
| `neutral` | 4 |
| `question` | 4 |
| `delight` | 4 |
| `sad_sympathetic` | 4 |
| `irony` | 4 |
| `strict` | 4 |
| `enumeration` | 4 |
| `excited` | 4 |
| `ambiguous` | 4 |

Поля кейса: `id`, `category`, `text`, `expected_intent`, `ambiguous`, `note`.
Словарь `expected_intent` — боевой (`backend/emotions.py`), своего корпус не
вводит. Реплики написаны вручную, ни одна модель корпус не порождала.

## 8. Automatic metrics

Считаются по каждой ячейке (текст + расшифровка ASR; WER берётся у существующего
`transcribe.word_error_rate`, второй формулы нет):

```text
transcript, wer, first_word_ok, last_word_ok,
extra_words, repetition, duration_sec, generation_sec, text_qa
```

Метаданные ячейки: `voice_id`, `engine`, `profile`, `seed`,
`reference_profile_id`, `reference_audio`, `case_id`, `category`,
`expected_intent`, `ambiguous`.

Результаты — `output/benchmarks/prosody/<run_id>/`: WAV на ячейку, `report.json`,
`report.md` (Δ метрик к NEUTRAL по профилям и категориям) и `review.md`
(шаблон прослушивания). **Чисел этих метрик для реальных моделей пока нет.**

## 9. Listening conclusions

**Заключений нет.** ASR не заменяет прослушивание (§34), а прослушивание не
выполнялось. Форма отзыва (`review.md`) уже содержит нужные поля:

```text
target_text, expected_intent, reference_profile,
text_complete, clarity, voice_identity, prosody_match, overacting, artifacts, notes
```

Для реплик `ambiguous: true` `prosody_match` оценивается с учётом подсказки из
`note` (§34). Заполненного `review.md` нет ни одного.

## 10. Profiles enabled for Auto

**Ни одного.** Список разрешённых профилей пуст.

## 11. Profiles disabled for Auto

**Все** существующие профили (у голосов их пока нет; по умолчанию
`enabled_for_auto = false`, §35). Правило: профиль, чья польза не доказана
прослушиванием, остаётся доступным **вручную** (у реплики), но автоматика его не
берёт. Пока профилей нет, автоматика работает на NEUTRAL — это ожидаемое
поведение, а не сбой.

## 12. Real dialogue benchmark (§74)

**Не выполнен.** Сцена из §74 должна проверять: `prosody analysis`,
`reference routing`, `short strategy`, `text completeness`, `voice identity`,
`manual override`, `regenerate`. Fixture диалога существует и покрыт
regression-тестом (`tests/test_regression_dialogue.py`), но прогон на реальных
моделях с профилями не делался.

## 13. Known limitations

- Перенос просодии **не измерен ни на одном движке** — production routing работает
  консервативно (`PROSODY_FALLBACK_MODE=neutral_only`), и это осознанно.
- Эмоциональный профиль может менять тембр: проверка качества сверяет запись с
  расшифровкой, но не «похожесть на себя»; это ловится только прослушиванием.
- У `SURPRISE` и `FEAR` нет своих фраз в `RECORD_PHRASES` — до отдельной записи
  они озвучиваются нейтральным референсом.
- Выводы нельзя переносить между движками, между голосами и с коротких реплик —
  каждый перенос проверяется заново.
- Числа `duration_sec`/`generation_sec` зависят от машины и текущей загрузки; на
  другом hardware их нельзя сравнивать напрямую.

## 14. Как прогнать

```bash
./venv/bin/python tools/prosody_benchmark.py --check                  # корпус, без моделей
./venv/bin/python tools/prosody_benchmark.py --plan --voice <id>      # матрица, без моделей
./venv/bin/python tools/prosody_benchmark.py --voice <id>             # F5/XTTS по движку голоса
./venv/bin/python tools/prosody_benchmark.py --voice <id> --engine xtts
./venv/bin/python -m pytest tests/test_prosody_benchmark.py -q
```

После прогона человек слушает `review.md`, смотрит `report.md` и только затем
подтверждает профили галочкой «авто» (§35). До этого документ обязан оставаться с
пустыми §6, §8 и §9 — это и есть честное состояние.
