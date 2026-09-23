"""Схема базы и её версии.

Схема создаётся сама при первом обращении: требовать от пользователя ручного
SQL нельзя — приложение локальное и ставится распаковкой архива. Миграции
пронумерованы, потому что база уже может существовать у того, кто обновился:
версия в `schema_version` говорит, что именно нужно досоздать.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Продолжение имени копии базы перед миграцией (F-D1). Копия лежит рядом с
# оригиналом, поэтому её видно глазом и легко вернуть вручную.
BACKUP_SUFFIX = ".bak"

# Проект: то, что пользователь открывает и закрывает. Текст хранится целиком,
# чтобы диалог не приходилось склеивать обратно из реплик — иначе при повторном
# разборе потерялись бы пустые строки и разметка, которую пользователь написал.
_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    source_text     TEXT NOT NULL DEFAULT '',
    mode            TEXT NOT NULL DEFAULT 'dialogue',
    render_settings TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'draft',
    job_id          TEXT,
    last_error      TEXT
);

-- Спикер проекта: слот или имя из текста плюс назначенный ему голос.
-- Назначение живёт отдельно от реплик: повторный разбор текста не должен
-- сбрасывать голоса, которые пользователь уже выбрал.
CREATE TABLE IF NOT EXISTS speakers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    label      TEXT NOT NULL,
    voice_id   TEXT NOT NULL DEFAULT '',
    overrides  TEXT NOT NULL DEFAULT '{}',
    UNIQUE (project_id, key)
);

CREATE TABLE IF NOT EXISTS replicas (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    text             TEXT NOT NULL,
    speaker          TEXT NOT NULL,
    voice_id         TEXT NOT NULL DEFAULT '',
    overrides        TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    selected_take_id INTEGER,
    UNIQUE (project_id, idx)
);

-- Вариант (take) реплики: готовое аудио плюс то, чем оно получено. Отдельный
-- файл на реплику, а не вырезка из итогового трека: варианты сравнивают на слух,
-- и после замены одного из них остальные не должны меняться.
CREATE TABLE IF NOT EXISTS takes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    replica_id  INTEGER NOT NULL REFERENCES replicas(id) ON DELETE CASCADE,
    label       TEXT NOT NULL DEFAULT '',
    audio_path  TEXT NOT NULL,
    seed        INTEGER,
    engine      TEXT NOT NULL DEFAULT '',
    parameters  TEXT NOT NULL DEFAULT '{}',
    duration_sec REAL NOT NULL DEFAULT 0,
    qa          TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_replicas_project ON replicas(project_id, idx);
CREATE INDEX IF NOT EXISTS idx_speakers_project ON speakers(project_id);
CREATE INDEX IF NOT EXISTS idx_takes_replica ON takes(replica_id, id);
"""

# Собственный голос реплики поверх голоса спикера. NULL — реплика наследует
# голос спикера, и тогда смена голоса спикера её тоже меняет; непустое значение —
# пользователь выбрал голос именно этой реплике. Без отдельной колонки отличить
# «унаследовано» от «выбрано вручную» нельзя: у реплики лежит уже вычисленное
# значение, и сброс правки было бы некуда вернуть.
_MIGRATION_2 = """
ALTER TABLE replicas ADD COLUMN voice_override TEXT;
"""

# Словарь произношения. Хранится глобально, без привязки к проекту: правило
# «SQL читается как эскьюэль» пользователь задаёт один раз и ожидает его во всех
# диалогах сразу. Уникальность по источнику и режиму регистра — это то, по чему
# повторное добавление обновляет правило, а не плодит дубли (см. pronunciation.py).
_MIGRATION_3 = """
CREATE TABLE IF NOT EXISTS pronunciation_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT NOT NULL,
    target         TEXT NOT NULL,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    whole_word     INTEGER NOT NULL DEFAULT 1,
    enabled        INTEGER NOT NULL DEFAULT 1,
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (source, case_sensitive)
);

CREATE INDEX IF NOT EXISTS idx_pronunciation_enabled ON pronunciation_entries(enabled);
"""

# Диагностические метрики take'а: WER, клиппинг, доля тишины, длительность,
# секунды на знак, peak/RMS/LUFS, попытки и причины отбора. Отдельной колонкой, а
# не внутри `qa`: это измерения самого готового куска, а не вердикт проверки, и
# они есть даже у take, который в Whisper не отправляли. JSON, потому что набор
# полей растёт от фазы к фазе, а читается он всегда целиком. У записей, сделанных
# до этой миграции, колонка пуста — они читаются как `quality: null`.
_MIGRATION_4 = """
ALTER TABLE takes ADD COLUMN quality TEXT;
"""

# Обязательная подготовка диалога перед синтезом: состояние анализа у проекта и
# стадии подготовки каждой реплики.
#
# Состояние анализа живёт отдельной колонкой, а не в `projects.status`: `status`
# отвечает за рендер («идёт сборка», «готово»), а `analysis_status` — за готовность
# текста («сырой», «анализируется», «нужно подтверждение», «готов»). Смешав их,
# нельзя было бы отличить «рендер идёт» от «текст ещё не подготовлен», а именно
# это различие и запрещает запускать синтез по сырому тексту.
#
# Стадии реплики хранятся колонками, а не одним JSON: по ним интерфейс показывает
# «что услышит модель» по шагам, а рендер обязан брать ровно `final_text`.
# `source_text` отдельной колонкой не заводится — это уже существующий
# `replicas.text`; вторая копия стала бы вторым источником правды, который рано
# или поздно разойдётся с первым. Наружу реплика отдаёт и `text`, и `source_text`.
#
# Словарь произношения уровня проекта — отдельной таблицей, а не колонкой
# `project_id` в глобальной: в SQLite NULL-ы в UNIQUE не конфликтуют между собой,
# и глобальные правила перестали бы быть уникальными по источнику. Отдельная
# таблица оставляет глобальному словарю его инвариант, а приоритет
# «проект → глобальный → автоматика» задаётся порядком слияния при чтении.
_MIGRATION_5 = """
ALTER TABLE projects ADD COLUMN analysis_status TEXT NOT NULL DEFAULT 'raw';
ALTER TABLE projects ADD COLUMN analysis_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE projects ADD COLUMN analysis_error TEXT;
ALTER TABLE projects ADD COLUMN analysis_started_at TEXT;
ALTER TABLE projects ADD COLUMN analysis_finished_at TEXT;

ALTER TABLE replicas ADD COLUMN normalized_text TEXT;
ALTER TABLE replicas ADD COLUMN yo_text TEXT;
ALTER TABLE replicas ADD COLUMN dictionary_text TEXT;
ALTER TABLE replicas ADD COLUMN accentized_text TEXT;
ALTER TABLE replicas ADD COLUMN final_text TEXT;
ALTER TABLE replicas ADD COLUMN analysis_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE replicas ADD COLUMN analysis_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE replicas ADD COLUMN analysis_error TEXT;
ALTER TABLE replicas ADD COLUMN dictionary_matches TEXT NOT NULL DEFAULT '[]';
ALTER TABLE replicas ADD COLUMN pronunciation_candidates TEXT NOT NULL DEFAULT '[]';
ALTER TABLE replicas ADD COLUMN supports_accents INTEGER NOT NULL DEFAULT 1;
ALTER TABLE replicas ADD COLUMN auto_accent INTEGER NOT NULL DEFAULT 1;
ALTER TABLE replicas ADD COLUMN effective_engine_params TEXT NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_replicas_analysis ON replicas(project_id, analysis_status);

CREATE TABLE IF NOT EXISTS project_pronunciation_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source         TEXT NOT NULL,
    target         TEXT NOT NULL,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    whole_word     INTEGER NOT NULL DEFAULT 1,
    enabled        INTEGER NOT NULL DEFAULT 1,
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (project_id, source, case_sensitive)
);

CREATE INDEX IF NOT EXISTS idx_project_pronunciation
    ON project_pronunciation_entries(project_id, enabled);
"""

# Падения процесса-синтеза (creash_report). Отдельная таблица, а не поля реплики:
# падение относится к процессу и движку, случается вне проекта (разовые задачи) и
# ценно историей — «падало три раза на этой машине» и «упало на этой реплике»
# разные вопросы. Тексты реплик сюда не пишутся: только индекс, движок, причина и
# коды — диагностика не должна превращаться во второй архив пользовательского
# текста (см. worker_protocol.text_fingerprint для логов).
_MIGRATION_6 = """
CREATE TABLE IF NOT EXISTS worker_crashes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    job_id         TEXT NOT NULL DEFAULT '',
    project_id     TEXT,
    replica_id     INTEGER,
    replica_index  INTEGER,
    engine         TEXT NOT NULL DEFAULT '',
    error_type     TEXT NOT NULL DEFAULT '',
    message        TEXT NOT NULL DEFAULT '',
    pid            INTEGER,
    exit_code      INTEGER,
    signal         INTEGER,
    signal_name    TEXT,
    reason         TEXT NOT NULL DEFAULT '',
    retry_count    INTEGER NOT NULL DEFAULT 0,
    attempt        INTEGER NOT NULL DEFAULT 1,
    started_at     TEXT,
    interrupted_at TEXT,
    memory_percent REAL
);

CREATE INDEX IF NOT EXISTS idx_worker_crashes_created ON worker_crashes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_crashes_job ON worker_crashes(job_id);
-- Осиротевшие `*.part` (недописанное аудио) ищутся по каталогам, а не по базе;
-- индекс нужен только для отчёта «что падало у этой реплики».
CREATE INDEX IF NOT EXISTS idx_worker_crashes_replica ON worker_crashes(project_id, replica_index);
"""

# Разборы локальной LLM живут отдельно от текста реплики: текст — пользовательский
# и неизменяемый, разбор — производная, которую можно пересчитать и которая обязана
# устаревать при изменении входа. Хеши входа хранятся рядом, чтобы «устарел или нет»
# решалось сравнением, а не догадкой по времени.
_MIGRATION_7 = """
CREATE TABLE IF NOT EXISTS llm_analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id       TEXT NOT NULL UNIQUE,
    project_id        TEXT NOT NULL,
    replica_id        INTEGER NOT NULL,
    replica_index     INTEGER NOT NULL,
    source_text_hash  TEXT NOT NULL DEFAULT '',
    context_hash      TEXT NOT NULL DEFAULT '',
    dictionary_hash   TEXT NOT NULL DEFAULT '',
    model_tag         TEXT NOT NULL DEFAULT '',
    model_digest      TEXT NOT NULL DEFAULT '',
    prompt_version    TEXT NOT NULL DEFAULT '',
    schema_version    TEXT NOT NULL DEFAULT '',
    analysis_json     TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL DEFAULT 'PENDING',
    error             TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    -- Ключ — место реплики в проекте: при повторном разборе диалога id меняются,
    -- а индекс остаётся тем же местом в тексте, и разбор не должен «теряться».
    UNIQUE(project_id, replica_index)
);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_project ON llm_analyses(project_id, replica_index);
CREATE INDEX IF NOT EXISTS idx_llm_analyses_status ON llm_analyses(status);
"""

# Подстатус LLM-анализа живёт рядом с подготовкой текста, но отдельной осью: без него
# интерфейс не смог бы отличить «подготовлено детерминированно» от «LLM ещё не
# смотрела» и «LLM нашла спорное». Модель и digest хранятся здесь же: результат
# анализа нельзя трактовать, не зная, чей он.
_MIGRATION_8 = """
ALTER TABLE projects ADD COLUMN llm_analysis_status TEXT NOT NULL DEFAULT 'DISABLED';
ALTER TABLE projects ADD COLUMN llm_analysis_model TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN llm_analysis_error TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN llm_analysis_updated_at TEXT;
"""

# Эмоция — метаданные реплики, а не текст (UPDATE 2 §4, §6): ни одно из этих
# полей не попадает ни в `text`, ни в `final_text`. `emotion_effective` не
# хранится отдельной колонкой: это `override → detected → NEUTRAL`, и второй
# источник правды разошёлся бы с первым при правке одного только override.
# `reference_*` — что **фактически** ушло в движок у последнего синтеза: по ним
# в карточке видно, взялся эмоциональный референс или случился откат на NEUTRAL.
_MIGRATION_9 = """
ALTER TABLE replicas ADD COLUMN emotion_detected TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN emotion_confidence REAL NOT NULL DEFAULT 0;
ALTER TABLE replicas ADD COLUMN emotion_override TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN dialogue_act TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN context_dependency TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN reference_profile_id TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN reference_emotion TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN reference_fallback_used INTEGER NOT NULL DEFAULT 0;
"""

# Просодия реплики (UPDATE 3 §8, §24, §36). Колонки аддитивные: старые проекты
# читаются как «модель ничего не рекомендовала», и резолвер сам уводит реплику в
# NEUTRAL — миграции данных не нужно.
#
# `prosody_override` здесь **нет** намеренно: ручной выбор пользователя уже
# хранится в `emotion_override`, и второй колонки-дубликата не заводим — иначе
# два источника правды разошлись бы при правке одного из них. `prosody_effective`
# тоже не хранится: это `override → recommended → эмоция` (см.
# `emotions.prosody_effective`), и второй источник правды здесь так же опасен.
#
# `prosody_intensity`/`prosody_confidence` допускают NULL — это «модель не
# сказала», и оно честно отличается от 0.0 (`0.0` — измеренная тишина).
# `reference_profile_key` — ключ **взятого** профиля (расширенный словарь §10),
# `reference_fallback_reason` — словами, почему случился откат: оба описывают
# факт последнего синтеза и нужны карточке и метаданным варианта (§55).
_MIGRATION_10 = """
ALTER TABLE replicas ADD COLUMN prosody_profile TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN prosody_intensity REAL;
ALTER TABLE replicas ADD COLUMN prosody_pace TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN prosody_confidence REAL;
ALTER TABLE replicas ADD COLUMN reference_profile_key TEXT NOT NULL DEFAULT '';
ALTER TABLE replicas ADD COLUMN reference_fallback_reason TEXT NOT NULL DEFAULT '';
"""

# Задачи очереди (F-Q1): до этой таблицы задача жила только в памяти процесса, и
# после рестарта или краха от неё не оставалось ничего — ни статуса, ни причины.
# Отдельная таблица, а не поля проекта: задача бывает и разовой (без проекта),
# а `projects.job_id` хранит лишь последнюю. Хранится снимок того, что показывает
# `Job.to_dict()`: интерфейс и диагностика читают одно и то же.
#
# Восстановление задач не делается: очередь и воркеры живут в процессе, и
# «продолжить» синтез после рестарта нельзя. Смысл записи — в том, что
# осиротевшая задача при старте получает внятный статус и причину, а не исчезает
# бесследно (см. `recovery.recover_after_restart`).
_MIGRATION_11 = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'queued',
    total_replicas  INTEGER NOT NULL DEFAULT 0,
    current_replica INTEGER NOT NULL DEFAULT 0,
    current_voice   TEXT NOT NULL DEFAULT '',
    output_format   TEXT NOT NULL DEFAULT 'wav',
    message         TEXT NOT NULL DEFAULT '',
    error           TEXT,
    error_type      TEXT NOT NULL DEFAULT '',
    output_path     TEXT,
    duration_sec    REAL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _MIGRATION_1),
    (2, _MIGRATION_2),
    (3, _MIGRATION_3),
    (4, _MIGRATION_4),
    (5, _MIGRATION_5),
    (6, _MIGRATION_6),
    (7, _MIGRATION_7),
    (8, _MIGRATION_8),
    (9, _MIGRATION_9),
    (10, _MIGRATION_10),
    (11, _MIGRATION_11),
)


class DatabaseCorruptedError(RuntimeError):
    """Файл базы повреждён: `PRAGMA integrity_check` не подтвердил целостность.

    Отдельный класс, а не общий `RuntimeError`: вызывающий должен отличать «база
    бита» от «миграция не применилась» — советы разные (восстановить из копии
    против сообщить об ошибке схемы), и смешивать их в одном тексте нельзя.
    """


def schema_version(connection: sqlite3.Connection) -> int:
    """Версия схемы в базе; `0` — таблицы версий ещё нет."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT version FROM schema_version").fetchone()
    return int(row[0]) if row else 0


def check_integrity(connection: sqlite3.Connection) -> None:
    """Отказывает, если SQLite считает файл повреждённым (F-D1).

    Полный `integrity_check`, а не `quick_check`: второй пропускает битые
    индексы, а именно они ломают выборки при чтении. Проверка стоит до миграций —
    применять схему к повреждённому файлу значит дописывать поверх мусора и
    терять следы того, что испортилось. Плата — чтение всей базы, но она
    происходит один раз за запуск (кеш `_initialized_path` в connection.py).
    """
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise DatabaseCorruptedError(f"База повреждена: {exc}") from exc
    problems = [str(row[0]) for row in rows if str(row[0]).strip().lower() != "ok"]
    if problems:
        raise DatabaseCorruptedError("База повреждена: " + "; ".join(problems[:5]))


def backup_before_migration(connection: sqlite3.Connection, path: Path) -> Path | None:
    """Копия файла базы перед миграцией. `None` — копировать нечего (F-D1).

    Копия кладётся только у существующей базы, которую действительно ждёт хотя бы
    одна миграция: на первом запуске файла ещё нет, а когда схема уже последней
    версии — терять нечего. Снимок делается средствами SQLite (`backup`), а не
    копированием файла: в режиме WAL часть подтверждённых данных лежит в `-wal`,
    и простое копирование дало бы старый или битый снимок.
    """
    if not path.exists():
        return None
    current = schema_version(connection)
    if current <= 0 or not any(version > current for version, _ in MIGRATIONS):
        return None
    target = path.with_name(path.name + BACKUP_SUFFIX)
    target.unlink(missing_ok=True)
    destination = sqlite3.connect(target)
    try:
        connection.backup(destination)
        destination.commit()
    finally:
        destination.close()
    logger.info("Копия базы перед миграцией: %s", target)
    return target


def apply_migrations(connection: sqlite3.Connection, path: Path | None = None) -> int:
    """Доводит схему до последней версии. Возвращает итоговую версию.

    `path` — файл базы: по нему перед миграцией кладётся копия (см.
    `backup_before_migration`). Без пути (так зовут миграции тесты и снимки в
    памяти) копия не делается, а всё остальное работает как раньше.
    """
    check_integrity(connection)
    current = schema_version(connection)
    if path is not None:
        backup_before_migration(connection, path)
    for version, script in MIGRATIONS:
        if version <= current:
            continue
        connection.executescript(script)
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        logger.info("Схема базы обновлена до версии %s", version)
        current = version
    return current
