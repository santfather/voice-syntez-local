"""Диагностический архив: собрать всё, чем объясняется качество озвучки.

Задача, ради которой этот модуль существует: «короткие реплики звучат плохо, и
непонятно почему». Ответить на неё из интерфейса нельзя — причина живёт в
сочетании данных, которые лежат в разных местах: исходный и подготовленный текст
в базе, настройки сборки в карточке проекта, план короткой реплики в параметрах
take'а, результат проверки в `qa`, откат слоя в логе, версия модели в окружении.
Поэтому архив — не «дамп базы», а **набор согласованных срезов одного рендера**,
собранных так, чтобы их можно было прочитать подряд.

Границы решения:

* Архив **не меняет** ни проект, ни базу: только читает. Сборка диагностики не
  должна становиться источником новых дефектов.
* Аудио кладётся файлами, а не в base64 внутри JSON: файл нужен для прослушивания
  в плеере, а JSON — для чтения глазами. Смешивать их в одном формате значит
  получить архив, который не открывается ни тем, ни другим.
* Аудио ограничено по объёму (`config.DIAGNOSTICS_MAX_AUDIO_MB`), и порядок
  включения — от самого важного к наименее важному: готовый трек, затем take'ы
  реплик, затем референсы голосов. Что не вошло, перечисляется в `manifest.json`
  и в README: молча усечённый архив читался бы как «здесь всё».
* Абсолютные пути не попадают в JSON-срезы: вместо них — имя файла внутри архива.
  Архив собирают, чтобы передать человеку или модели, и путь с именем домашнего
  каталога в такой передаче не нужен.

Второй способ собрать те же сведения (например, читать их прямо из базы при
разборе) разошёлся бы с этим на первой же правке схемы, поэтому срезы строятся
здесь и только здесь, а API их лишь отдаёт.
"""

import json
import logging
import zipfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config, resource_guard
from . import short_utterance as su
from .accentizer import Accentizer
from .db.store import get_projects_store
from .engines.base import ENGINE_INFOS
from .engines.registry import created_engines
from .engines.supervisor import get_supervisor
from .llm import analyzer as llm_analyzer
from .llm import memory_policy as llm_memory
from .llm import versioning
from .project_export import safe_name
from .voices_store import get_store as get_voices_store

logger = logging.getLogger(__name__)

# Версия формата архива. Растёт, когда меняется смысл файлов внутри: читающий
# (человек или скрипт) должен видеть, к какой версии относятся поля.
DIAGNOSTICS_VERSION = 1
ARCHIVE_SUFFIX = ".zip"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"

PROJECT_NAME = "project.json"
SETTINGS_NAME = "settings.json"
TAKES_NAME = "takes.json"
ANALYSIS_NAME = "analysis.json"
ENVIRONMENT_NAME = "environment.json"
LOG_NAME = "logs/voice_syntez.log"

FINAL_DIR = "audio/final"
TAKE_DIR = "audio/takes"
REFERENCE_DIR = "audio/references"

# Сколько архивов одного проекта оставлять. Диагностика собирается по конкретному
# случаю, и десять последних версий — это история правок, а не склад.
KEEP_PER_PROJECT = 10


class DiagnosticsError(ValueError):
    """Архив собрать нельзя — API отдаёт это текстом 400."""


@dataclass
class DiagnosticsBundle:
    """Готовый архив: путь, состав и то, что в него не поместилось."""

    name: str
    path: Path
    size_mb: float
    entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size_mb": self.size_mb,
            "entries": self.entries,
            "warnings": self.warnings,
            "version": DIAGNOSTICS_VERSION,
        }


# --- срезы --------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone().strftime("%Y%m%d-%H%M%S")


def _voice_layer(voice_id: str) -> dict:
    """Голос так, как он важен для разбора: без пути и без самого файла.

    Идентификатор и имя нужны, чтобы связать реплику с голосом; `engine`,
    `ref_text` и предупреждения — чтобы объяснить дефект («узкополосный
    референс», «расшифровка не совпала»). Путь к файлу в `voices/` не входит:
    это адрес на конкретной машине, а не свойство записи.
    """
    if not voice_id:
        return {}
    try:
        voice = get_voices_store().get(voice_id)
    except Exception as exc:  # noqa: BLE001 — голос мог быть удалён
        logger.debug("Голос %s не прочитан: %s", voice_id, exc)
        return {"id": voice_id, "missing": True}
    if voice is None:
        return {"id": voice_id, "missing": True}
    return {
        "id": voice.id,
        "name": voice.name,
        "engine": voice.engine,
        "gender": voice.gender,
        "ref_text": voice.ref_text,
        "engine_params": dict(voice.engine_params or {}),
        "preset": dict(voice.preset or {}),
        "f0_hz": voice.f0_hz,
        "gender_warning": voice.gender_warning,
        "band_warning": voice.band_warning,
        "ref_text_warning": voice.ref_text_warning,
        "reference_file": voice.audio_file,
    }


def _reference_file(voice_id: str) -> Path | None:
    """Файл референса голоса, если он есть и лежит внутри `voices/`."""
    if not voice_id:
        return None
    try:
        voice = get_voices_store().get(voice_id)
    except Exception:  # noqa: BLE001 — см. `_voice_layer`
        return None
    if voice is None:
        return None
    path = voice.audio_path
    try:
        inside = config.VOICES_DIR.resolve() in path.resolve().parents
    except OSError:
        return None
    return path if inside and path.exists() else None


def _project_slice(project: dict) -> dict:
    """Проект: исходник, стадии подготовки, спикеры, состояние анализа.

    Реплики идут в порядке диалога, и у каждой — все стадии по отдельности.
    Именно их сравнение отвечает на половину вопросов о качестве: «что написали»
    (`source_text`), «что стало после нормализации» и «что реально читала модель»
    (`final_text`). Свести их в одну строку значило бы потерять ответ.
    """
    replicas = []
    for replica in project.get("replicas") or []:
        replicas.append(
            {
                "index": replica.get("index"),
                "speaker": replica.get("speaker"),
                "label": replica.get("label"),
                "source_text": replica.get("source_text"),
                "normalized_text": replica.get("normalized_text"),
                "yo_text": replica.get("yo_text"),
                "dictionary_text": replica.get("dictionary_text"),
                "accentized_text": replica.get("accentized_text"),
                "final_text": replica.get("final_text"),
                "analysis_status": replica.get("analysis_status"),
                "analysis_version": replica.get("analysis_version"),
                "analysis_error": replica.get("analysis_error"),
                "dictionary_matches": replica.get("dictionary_matches") or [],
                "pronunciation_candidates": replica.get("pronunciation_candidates") or [],
                "supports_accents": replica.get("supports_accents"),
                "auto_accent": replica.get("auto_accent"),
                "voice_id": replica.get("voice_id"),
                "voice_override": replica.get("voice_override"),
                "overrides": replica.get("overrides") or {},
                "effective_engine_params": replica.get("effective_engine_params") or {},
                "status": replica.get("status"),
                "selected_take_id": replica.get("selected_take_id"),
                "voices": _voice_layer(
                    str(replica.get("voice_override") or replica.get("voice_id") or "")
                ),
            }
        )
    speakers = []
    for speaker in project.get("speakers") or []:
        speakers.append(
            {
                "key": speaker.get("key"),
                "label": speaker.get("label"),
                "voice_id": speaker.get("voice_id"),
                "overrides": speaker.get("overrides") or {},
                "voices": _voice_layer(str(speaker.get("voice_id") or "")),
            }
        )
    return {
        "id": project.get("id"),
        "name": project.get("name"),
        "mode": project.get("mode"),
        "created_at": project.get("created_at"),
        "updated_at": project.get("updated_at"),
        "status": project.get("status"),
        "job_id": project.get("job_id"),
        "last_error": project.get("last_error"),
        "source_text": project.get("source_text"),
        "analysis_status": project.get("analysis_status"),
        "analysis_error": project.get("analysis_error"),
        "analysis_version": project.get("analysis_version"),
        "analysis_started_at": project.get("analysis_started_at"),
        "analysis_finished_at": project.get("analysis_finished_at"),
        "llm_analysis_status": project.get("llm_analysis_status"),
        "llm_analysis_model": project.get("llm_analysis_model"),
        "llm_analysis_error": project.get("llm_analysis_error"),
        "llm_analysis_updated_at": project.get("llm_analysis_updated_at"),
        "render_settings": project.get("render_settings") or {},
        "speakers": speakers,
        "replicas": replicas,
    }


def _short_policy() -> dict:
    """Политика коротких реплик, действующая в приложении.

    Не то же самое, что настройки конкретного рендера: здесь — политика по
    умолчанию и измеренные стратегии по движкам. Без неё непонятно, было ли
    «Авто» в настройках рендера осмысленным выбором или слоем, который для этого
    движка ничего не делает.
    """
    return {
        "enabled_by_default": config.SHORT_UTTERANCE_DEFAULT_ENABLED,
        "strategies_by_engine": {
            engine_id: su.default_strategy(engine_id) for engine_id in ENGINE_INFOS
        },
        "available_strategies": list(su.SELECTABLE_STRATEGIES),
        "thresholds": su.ShortThresholds().to_dict(),
        "boundary_method_by_default": config.SHORT_UTTERANCE_BOUNDARY_METHOD,
        "max_attempts_by_default": config.SHORT_UTTERANCE_MAX_ATTEMPTS,
    }


def _settings_slice(project: dict, render_settings: dict | None) -> dict:
    """Настройки, применимые к рендеру: эффективные плюс источники их значений.

    `render_settings` — то, что посчитал API для конкретного запуска (сохранённое
    в проекте плюс тело запроса). Сохранённое в проекте идёт рядом, отдельным
    ключом: разница между ними и есть ответ на «почему получилось не то, что я
    выбирал».
    """
    return {
        "effective": render_settings or {},
        "saved_in_project": project.get("render_settings") or {},
        "short_utterance_policy": _short_policy(),
        "qa": {
            "default_mode": config.QA_MODE_OFF,
            "modes": list(config.QA_MODES),
            # Пороги дешёвого отбора: по ним видно, почему кусок признан
            # подозрительным и ушёл в Whisper (или не ушёл).
            "screening": {
                "chars_per_sec": config.QA_SCREEN_CHARS_PER_SEC,
                "min_expected_sec": config.QA_SCREEN_MIN_EXPECTED_SEC,
                "min_duration_sec": config.QA_SCREEN_MIN_DURATION_SEC,
                "short_ratio": config.QA_SCREEN_SHORT_RATIO,
                "long_ratio": config.QA_SCREEN_LONG_RATIO,
                "silence_ratio": config.QA_SCREEN_SILENCE_RATIO,
                "silence_db": config.QA_SCREEN_SILENCE_DB,
                "rms_floor_db": config.QA_SCREEN_RMS_FLOOR_DB,
                "rms_ceil_db": config.QA_SCREEN_RMS_CEIL_DB,
                "repeat_corr": config.QA_SCREEN_REPEAT_CORR,
                "frame_sec": config.QA_SCREEN_FRAME_SEC,
            },
        },
        "chunking": {
            "default_strategy": config.CHUNK_STRATEGY_DEFAULT,
            "strategies": [config.CHUNK_STRATEGY_PARAGRAPH, config.CHUNK_STRATEGY_SHORT],
            "chars_by_strategy": {
                strategy: config.chunk_chars(strategy)
                for strategy in (config.CHUNK_STRATEGY_PARAGRAPH, config.CHUNK_STRATEGY_SHORT)
            },
        },
        "accentizer": _accentizer_state(),
        "output": {
            "default_format": config.DEFAULT_OUTPUT_FORMAT,
            "default_pause_ms": config.DEFAULT_PAUSE_MS,
            "pause_range": list(config.PAUSE_MS_RANGE),
            "default_cross_fade": config.DEFAULT_CROSS_FADE_DURATION,
            "cross_fade_range": list(config.CROSS_FADE_RANGE),
        },
    }


def _accentizer_state() -> dict:
    """Состояние расстановки ударений: загружена ли модель и не было ли отказа.

    Только флаги, без загрузки: `instance()` модель не поднимает, а состояние
    нужно ровно затем, чтобы отличить «ударения не расставлены» от «устали
    расставлять».
    """
    try:
        accentizer = Accentizer.instance()
        return {
            "loaded": accentizer.is_loaded,
            "state": accentizer.state,
            "error": accentizer.last_error,
        }
    except Exception as exc:  # noqa: BLE001 — состояние вторично
        return {"error": str(exc)}


def _take_stream(replica: dict, take: dict) -> dict:
    return {
        "take_id": take.get("id"),
        "label": take.get("label"),
        "seed": take.get("seed"),
        "engine": take.get("engine"),
        "duration_sec": take.get("duration_sec"),
        "created_at": take.get("created_at"),
        "active": int(take.get("id") or 0) == int(replica.get("selected_take_id") or 0),
        # Параметры куска: сюда же попадает план короткой реплики
        # (`short_utterance_*`, `tts_synthesis_text`) — то, чем этот take получен.
        "parameters": take.get("parameters") or {},
        "qa": take.get("qa"),
        "quality": take.get("quality"),
    }


def _takes_slice(project: dict, store) -> tuple[dict, list[tuple[str, Path]]]:
    """Take'ы всех реплик плюс файлы, которые к ним есть на диске.

    Файл берётся из строки take'а, но путь в JSON не попадает: он заменяется
    именем внутри архива. Связь «строка ↔ файл» восстанавливается по `take_id`.
    """
    replicas = []
    files: list[tuple[str, Path]] = []
    for replica in project.get("replicas") or []:
        index = int(replica.get("index") or 0)
        takes = []
        for take in replica.get("takes") or []:
            item = _take_stream(replica, take)
            path = Path(str(take.get("audio_path") or ""))
            name = f"{TAKE_DIR}/r{index}-take{take.get('id')}{path.suffix or '.wav'}"
            if path.exists():
                item["audio"] = name
                files.append((name, path))
            else:
                # Файл мог быть удалён (проект чистили) — это факт для разбора,
                # а не ошибка сборки: остальные данные о take'е остаются ценными.
                item["audio"] = None
                item["audio_missing"] = True
            takes.append(item)
        replicas.append(
            {
                "index": index,
                "speaker": replica.get("speaker"),
                "final_text": replica.get("final_text"),
                "short": _short_reading(replica),
                "takes": takes,
            }
        )
    return {"replicas": replicas}, files


def _short_reading(replica: dict) -> dict:
    """Коротко: была ли реплика короткой и что решил слой.

    Читается из параметров активного take'а, потому что только там лежит решение,
    принятое при синтезе. Реплика без take'ов получает `applied: false`: решения
    ещё не было, и пустой объект читался бы как «слой не сработал».
    """
    selected = replica.get("selected_take_id")
    plan: dict = {}
    for take in replica.get("takes") or []:
        if int(take.get("id") or 0) == int(selected or 0):
            plan = (take.get("parameters") or {})
            break
    words = len(str(replica.get("final_text") or replica.get("text") or "").split())
    return {
        "words": words,
        "applied": bool(plan.get("short_utterance_strategy")),
        "strategy_requested": plan.get("short_utterance_strategy_requested"),
        "strategy": plan.get("short_utterance_strategy"),
        "context_source": plan.get("short_utterance_context_source"),
        "class": plan.get("short_utterance_class"),
        "attempts": plan.get("short_utterance_attempts"),
        "fallback": plan.get("short_utterance_fallback"),
        "synthesis_text": plan.get("tts_synthesis_text"),
    }


def _analysis_slice(project: dict, store) -> dict:
    """Сохранённые разборы локальной LLM плюс кандидаты на произношение.

    Разбор кладётся сырым (`analysis_json` из базы разобран в объект): это
    предложение модели, а не применённый текст, и подменять его пересказом в
    диагностике нельзя. Рядом — версии входа (хеши, модель, prompt, схема): по ним
    видно, к какому тексту разбор относится и не устарел ли он.
    """
    try:
        rows = store.llm_analyses(str(project.get("id")))
    except Exception as exc:  # noqa: BLE001 — анализа могло не быть
        logger.warning("Разборы LLM не прочитаны: %s", exc)
        return {"analyses": [], "candidates": [], "error": str(exc)}
    analyses = []
    for row in rows:
        raw = row.get("analysis_json")
        parsed: object = None
        if raw:
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                # Повреждённая строка — повод показать её как есть, а не сорвать
                # сборку архива: остальные срезы от неё не зависят.
                parsed = {"raw": str(raw)}
        analyses.append(
            {
                "replica_index": int(row.get("replica_index") or 0),
                "status": row.get("status"),
                "error": row.get("error") or "",
                "model_tag": row.get("model_tag"),
                "model_digest": row.get("model_digest"),
                "prompt_version": row.get("prompt_version"),
                "schema_version": row.get("schema_version"),
                "source_text_hash": row.get("source_text_hash"),
                "context_hash": row.get("context_hash"),
                "dictionary_hash": row.get("dictionary_hash"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "analysis": parsed,
            }
        )
    # Кандидаты берутся из подготовки реплик: там они лежат уже сверенными с
    # детерминированным слоем, и это ровно тот список, который видел пользователь.
    candidates = [
        {**candidate, "replica_index": int(replica.get("index") or 0)}
        for replica in project.get("replicas") or []
        for candidate in (replica.get("pronunciation_candidates") or [])
    ]
    return {"analyses": analyses, "candidates": candidates}


def _environment_slice() -> dict:
    """Окружение: машина, версии, движки, воркеры, Ollama, состояние памяти.

    Собирается best-effort: недоступная метрика (нет Ollama, мёртвый воркер) —
    это тоже результат разбора, но не повод не собрать архив.
    """
    data: dict = {
        "collected_at": _now().isoformat(timespec="seconds"),
        "git_commit": versioning.git_commit(),
        "machine": versioning.machine_info(),
    }
    try:
        data["engines"] = [engine.to_dict() for engine in created_engines().values()]
    except Exception as exc:  # noqa: BLE001
        data["engines"] = []
        data["engines_error"] = str(exc)
    try:
        data["workers"] = get_supervisor().status()
    except Exception as exc:  # noqa: BLE001
        data["workers_error"] = str(exc)
    try:
        data["device"] = config.pick_device()
        data["worker_isolation"] = config.worker_isolation_enabled()
        data["memory_state"] = resource_guard.memory_state()
        data["memory_snapshot"] = resource_guard.snapshot()
        data["mps_memory"] = resource_guard.mps_memory_snapshot()
    except Exception as exc:  # noqa: BLE001
        data["runtime_error"] = str(exc)
    try:
        data["llm"] = llm_analyzer.get_analyzer().status()
    except Exception as exc:  # noqa: BLE001 — Ollama необязательна
        data["llm_error"] = str(exc)
    # Пороги памяти для модели: именно они решают, дойдёт ли анализ до конца.
    # Пороги синтеза лежат рядом, в `memory_state.thresholds` (их считает
    # memory_monitor), а здесь — политика тяжёлых задач (§11.2 Task 2).
    try:
        data["llm_memory_policy"] = llm_memory.thresholds_from_env().to_dict()
    except Exception as exc:  # noqa: BLE001
        data["llm_memory_policy_error"] = str(exc)
    return data


def _log_slice(log_path: Path | None, limit: int) -> tuple[str, list[str]]:
    """Хвост лога приложения. Пусто — строка о причине, а не пустой файл."""
    warnings: list[str] = []
    path = Path(log_path) if log_path else config.LOG_PATH
    if not path.exists():
        warnings.append(
            f"лог {path} не найден: архив собран без логов "
            "(причина дефекта в журнале рендера осталась вне архива)"
        )
        return "", warnings
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            lines = stream.readlines()
    except OSError as exc:
        warnings.append(f"лог {path} не прочитан: {exc}")
        return "", warnings
    tail = lines[-limit:] if limit > 0 else lines
    header = (
        f"# хвост лога {path.name}: последние {len(tail)} строк из {len(lines)}\n"
    )
    return header + "".join(tail), warnings


# --- README архива ------------------------------------------------------------
def _readme(bundle_name: str, project: dict, warnings: list[str], included: dict) -> str:
    """Инструкция к архиву: что внутри и куда смотреть.

    README — часть формата, а не украшение: архив читает человек (или модель) через
    день после рендера, и без объяснения, где лежит решение слоя коротких реплик,
    он превращается в набор JSON.
    """
    lines = [
        f"# Диагностика озвучки: {project.get('name')}",
        "",
        f"Архив: `{bundle_name}`",
        f"Формат: версия {DIAGNOSTICS_VERSION}",
        f"Собран: {_now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "Архив собран сервером Voice Synthesizer локально и никуда не отправлялся:",
        "это срез одного проекта (и, если он был указан, одного рендера) для разбора",
        "качества. Он ничего не меняет в проекте и базе.",
        "",
        "## Что внутри",
        "",
        "| Файл | Что это |",
        "| --- | --- |",
        (
            f"| `{PROJECT_NAME}` | Исходный текст, все стадии подготовки каждой реплики "
            "(`source_text` → `final_text`), спикеры и голоса, состояние анализа |"
        ),
        (
            f"| `{SETTINGS_NAME}` | Настройки, применимые к сборке: эффективные, "
            "сохранённые в проекте, политика коротких реплик и пороги проверки качества |"
        ),
        (
            f"| `{TAKES_NAME}` | По каждой реплике — варианты: движок, сид, длительность, "
            "параметры куска (включая план короткой реплики), QA и метрики качества |"
        ),
        (
            f"| `{ANALYSIS_NAME}` | Сохранённые разборы локального лингвистического "
            "анализатора и кандидаты на произношение |"
        ),
        (
            f"| `{ENVIRONMENT_NAME}` | Машина, версия коммита, движки, воркеры, "
            "память, состояние Ollama |"
        ),
        f"| `{LOG_NAME}` | Хвост журнала сервера — там видно причины отказов слоя |",
        f"| `{FINAL_DIR}/` | Готовый трек рендера |",
        f"| `{TAKE_DIR}/` | Аудио вариантов реплик (сопоставляется по `take_id`) |",
        f"| `{REFERENCE_DIR}/` | Референсы голосов, которыми озвучен проект |",
        "",
        "## Если короткая реплика звучит плохо — порядок чтения",
        "",
        (
            f"1. `{TAKES_NAME}` → `replicas[].short`: была ли реплика короткой, какая "
            "стратегия применена, что запрошено и **был ли откат** (`fallback`). "
            "Пустой `strategy` при `applied: false` означает, что слой не сработал."
        ),
        (
            "2. Там же `takes[].parameters.tts_synthesis_text` — текст, который реально "
            "ушёл в модель, и `attempts` — сколько было попыток."
        ),
        (
            "3. `takes[].qa` (WER расшифровки) и `takes[].quality` (клиппинг, тишина, "
            "LUFS, доля пауз) — расходятся ли слова с текстом и не режет ли звук."
        ),
        (
            f"4. `{PROJECT_NAME}` → `source_text` против `final_text`: не потерялись ли "
            "слова и ударения при подготовке."
        ),
        (
            f"5. `{LOG_NAME}` → поиск по «граница цели не найдена», "
            "«attempts_exhausted», «откат»: причина отказа слоя пишется туда текстом."
        ),
        (
            f"6. `{SETTINGS_NAME}` и `{ENVIRONMENT_NAME}`: включён ли слой, какая "
            "стратегия измерена для движка, хватало ли памяти на модель."
        ),
        "",
    ]
    if included.get("skipped"):
        lines += [
            "## Что не поместилось",
            "",
            f"Аудио ограничено {included.get('limit_mb')} МБ на архив; не вошли:",
            "",
            *[f"* `{name}`" for name in included["skipped"]],
            "",
            "Остальные срезы описывают эти файлы полностью — отсутствует только звук.",
            "",
        ]
    if warnings:
        lines += ["## Предупреждения сборки", "", *[f"* {item}" for item in warnings], ""]
    return "\n".join(lines)


# --- сборка -------------------------------------------------------------------
def _archive_prefix(project: dict) -> str:
    """Общий префикс имён архивов одного проекта.

    В имени есть и название, и начало идентификатора: имя читает человек, а по
    идентификатору архив находится среди чужих. Без идентификатора два проекта с
    одним названием («Тест») показывали бы друг другу архивы и вытесняли бы их
    при ротации — имена проектов не уникальны, и делать по ним вид, что уникальны,
    нельзя.
    """
    name = safe_name(str(project.get("name") or "project"))
    project_id = str(project.get("id") or "")[:8]
    return f"diagnostics-{name}-{project_id}-" if project_id else f"diagnostics-{name}-"


def _unique_name(project: dict, moment: datetime) -> str:
    stem = f"{_archive_prefix(project)}{_stamp(moment)}"
    target = config.DIAGNOSTICS_DIR / f"{stem}{ARCHIVE_SUFFIX}"
    if not target.exists():
        return target.name
    for number in range(2, 100):
        candidate = config.DIAGNOSTICS_DIR / f"{stem}-{number}{ARCHIVE_SUFFIX}"
        if not candidate.exists():
            return candidate.name
    return f"{stem}-{moment.strftime('%f')}{ARCHIVE_SUFFIX}"


def _fit_audio(
    items: list[tuple[str, Path]], limit_mb: float
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Укладывает аудио в бюджет, начиная с самого важного.

    Порядок задаёт вызывающий: готовый трек, затем take'ы, затем референсы. Файл
    сверх бюджета пропускается, но следующие за ним всё ещё могут войти: короткие
    take'ы важнее длинного референса, и «после первого превышения остановиться»
    выбросило бы именно их.
    """
    limit_bytes = int(limit_mb * 1024 * 1024) if limit_mb > 0 else 0
    included: list[tuple[str, Path]] = []
    skipped: list[str] = []
    used = 0
    for name, path in items:
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append(f"{name} (файл недоступен)")
            continue
        if limit_bytes and used + size > limit_bytes:
            skipped.append(f"{name} ({size / (1024 * 1024):.1f} МБ)")
            continue
        used += size
        included.append((name, path))
    return included, skipped


def create_diagnostics_archive(
    project: dict,
    *,
    render_settings: dict | None = None,
    job_output: Path | None = None,
    include_references: bool = True,
    max_audio_mb: float | None = None,
    log_lines: int | None = None,
    log_path: Path | None = None,
) -> DiagnosticsBundle:
    """Собирает архив диагностики проекта и возвращает его описание.

    `job_output` — файл конкретного рендера: именно его пользователь слушал,
    когда решил, что звучит плохо. Без него архив описывает проект целиком, но
    без готового трека.
    """
    if not project:
        raise DiagnosticsError("Проект не найден — диагностику собирать не из чего")
    store = get_projects_store()
    moment = _now()
    warnings: list[str] = []

    project_slice = _project_slice(project)
    settings_slice = _settings_slice(project, render_settings)
    takes_slice, take_files = _takes_slice(project, store)
    analysis_slice = _analysis_slice(project, store)
    environment = _environment_slice()

    audio_queue: list[tuple[str, Path]] = []
    if job_output is not None and Path(job_output).exists():
        audio_queue.append((f"{FINAL_DIR}/{Path(job_output).name}", Path(job_output)))
    elif job_output is not None:
        warnings.append(f"готовый файл рендера {Path(job_output).name} уже удалён (TTL)")
    audio_queue.extend(take_files)
    if include_references:
        seen: set[str] = set()
        for replica in project.get("replicas") or []:
            voice_id = str(replica.get("voice_override") or replica.get("voice_id") or "")
            for speaker in project.get("speakers") or []:
                if speaker.get("key") == replica.get("speaker"):
                    voice_id = str(speaker.get("voice_id") or voice_id)
                    break
            if not voice_id or voice_id in seen:
                continue
            seen.add(voice_id)
            path = _reference_file(voice_id)
            if path is not None:
                audio_queue.append((f"{REFERENCE_DIR}/{voice_id}{path.suffix}", path))

    limit_mb = config.DIAGNOSTICS_MAX_AUDIO_MB if max_audio_mb is None else max_audio_mb
    included_audio, skipped = _fit_audio(audio_queue, limit_mb)
    if skipped:
        warnings.append(
            f"в архив не вошли {len(skipped)} аудиофайлов (предел {limit_mb} МБ)"
        )
    # Связь «take ↔ файл»: чего нет в архиве, не должно выглядеть как лежащее там.
    archive_names = {name for name, _ in included_audio}
    for take in takes_slice["replicas"]:
        for item in take.get("takes") or []:
            if item.get("audio") and item["audio"] not in archive_names:
                item["audio_in_bundle"] = False

    log_text, log_warnings = _log_slice(log_path, log_lines or config.DIAGNOSTICS_LOG_LINES)
    warnings.extend(log_warnings)

    files: dict[str, bytes] = {
        PROJECT_NAME: _dump(project_slice),
        SETTINGS_NAME: _dump(settings_slice),
        TAKES_NAME: _dump(takes_slice),
        ANALYSIS_NAME: _dump(analysis_slice),
        ENVIRONMENT_NAME: _dump(environment),
    }
    if log_text:
        files[LOG_NAME] = log_text.encode("utf-8", errors="replace")

    name = _unique_name(project, moment)
    included = {
        "limit_mb": limit_mb,
        "skipped": skipped,
        "audio": [item[0] for item in included_audio],
    }
    files[README_NAME] = _readme(name, project, warnings, included).encode("utf-8")
    files[MANIFEST_NAME] = _dump(
        {
            "version": DIAGNOSTICS_VERSION,
            "name": name,
            "collected_at": moment.isoformat(timespec="seconds"),
            "project_id": project.get("id"),
            "project_name": project.get("name"),
            "git_commit": environment.get("git_commit", ""),
            "files": sorted(files) + [item[0] for item in included_audio],
            "audio_limit_mb": limit_mb,
            "audio_skipped": skipped,
            "warnings": warnings,
        }
    )

    # Каталог создаётся здесь, а не только при импорте конфига: путь может быть
    # переопределён (`TTS_DIAGNOSTICS_DIR`), а сборка архива не должна зависеть от
    # того, успел ли кто-то создать каталог заранее.
    config.DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.DIAGNOSTICS_DIR / name
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for entry, blob in files.items():
                archive.writestr(entry, blob)
            for entry, path in included_audio:
                archive.write(path, entry)
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise DiagnosticsError(f"архив не собран: {exc}") from exc

    size_mb = round(target.stat().st_size / (1024 * 1024), 2)
    logger.info(
        "Диагностика проекта %s: %s (%.2f МБ, %s файлов аудио)",
        project.get("id"),
        name,
        size_mb,
        len(included_audio),
    )
    _prune(project)
    return DiagnosticsBundle(
        name=name,
        path=target,
        size_mb=size_mb,
        entries=sorted(files) + [item[0] for item in included_audio],
        warnings=warnings,
    )


def _dump(payload: object) -> bytes:
    """JSON с `ensure_ascii=False`: срезы читает человек, кириллица — не escape-коды."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def describe_render_settings(settings: object) -> dict:
    """Настройки рендера в виде, пригодном для JSON.

    `asdict` разворачивает и вложенные настройки (проверка качества, короткие
    реплики), поэтому в архиве оказываются именно те значения, с которыми шёл
    синтез, а не их пересказ. Словарь возвращается как есть: диагностика не
    должна падать из-за того, что ей передали не датакласс.
    """
    if settings is None:
        return {}
    if isinstance(settings, dict):
        return dict(settings)
    if is_dataclass(settings) and not isinstance(settings, type):
        return asdict(settings)
    return {"value": str(settings)}


def _prune(project: dict) -> None:
    """Оставляет последние `KEEP_PER_PROJECT` архивов этого проекта."""
    prefix = _archive_prefix(project)
    try:
        ours = sorted(
            (
                path
                for path in config.DIAGNOSTICS_DIR.glob(f"{prefix}*{ARCHIVE_SUFFIX}")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError as exc:
        logger.debug("Каталог диагностики не прочитан: %s", exc)
        return
    for stale in ours[KEEP_PER_PROJECT:]:
        try:
            stale.unlink()
            logger.info("Старый архив диагностики удалён: %s", stale.name)
        except OSError as exc:
            logger.debug("Архив %s не удалён: %s", stale.name, exc)


# --- доступ к готовым архивам -------------------------------------------------
def archive_prefix(project: dict) -> str:
    """Префикс имён архивов проекта — нужен и проверке принадлежности в API."""
    return _archive_prefix(project)


def resolve_archive(name: str, *, project: dict | None = None) -> Path:
    """Путь архива по имени — только внутри каталога диагностики.

    Имя приходит из URL, поэтому проверка здесь, а не в обработчике: `..`,
    подкаталоги и чужие расширения отсекаются до любого обращения к диску. Если
    передан проект, проверяется и принадлежность: файл соседнего проекта по имени
    открывать нельзя, даже когда имя угадано.
    """
    candidate = (config.DIAGNOSTICS_DIR / str(name)).resolve()
    root = config.DIAGNOSTICS_DIR.resolve()
    if candidate.parent != root or candidate.suffix != ARCHIVE_SUFFIX:
        raise DiagnosticsError("Некорректное имя архива")
    if project is not None and not candidate.name.startswith(_archive_prefix(project)):
        raise DiagnosticsError("Архив не найден")
    if not candidate.is_file():
        raise DiagnosticsError("Архив не найден")
    return candidate


def list_archives(project: dict) -> list[dict]:
    """Собранные архивы проекта — от свежих к старым."""
    prefix = _archive_prefix(project)
    try:
        paths = [
            path
            for path in config.DIAGNOSTICS_DIR.glob(f"{prefix}*{ARCHIVE_SUFFIX}")
            if path.is_file()
        ]
    except OSError:
        return []
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        {
            "name": path.name,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        }
        for path in paths
    ]


def discard_archive(name: str, *, project: dict | None = None) -> None:
    """Удаляет архив (пользователь разобрался или архив не нужен)."""
    resolve_archive(name, project=project).unlink()


def clear_archives(project: dict) -> int:
    """Удаляет все архивы проекта; возвращает число удалённых."""
    removed = 0
    for item in list_archives(project):
        try:
            (config.DIAGNOSTICS_DIR / item["name"]).unlink()
            removed += 1
        except OSError as exc:
            logger.debug("Архив %s не удалён: %s", item["name"], exc)
    return removed


__all__ = [
    "DIAGNOSTICS_VERSION",
    "archive_prefix",
    "DiagnosticsBundle",
    "DiagnosticsError",
    "clear_archives",
    "create_diagnostics_archive",
    "describe_render_settings",
    "discard_archive",
    "list_archives",
    "resolve_archive",
]
