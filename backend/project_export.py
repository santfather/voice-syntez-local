"""Перенос проекта между машинами: архив `.ttsproject`, аудио и субтитры.

Проект живёт в SQLite и в файлах (`output/projects/{id}/`, `voices/`), поэтому
«сохранить диалог» — это не одна копия файла, а согласованный набор: метаданные,
спикеры с назначенными голосами, реплики, take'ы и референсы тех голосов, без
которых проект не зазвучит. Архив собирает всё это в один ZIP с генерируемыми
именами (пользовательские имена в путях не участвуют), а импорт разворачивает его
обратно, сопоставляя голоса по имени.

Веса моделей в архив не попадают никогда: он собирается обходом известных файлов
проекта и голосов, а не каталога `models/`. Это не оптимизация, а свойство
формата: архив должен переносить диалог, а не гигабайты чекпоинтов.

Безопасность импорта — часть контракта, а не деталь: чужой архив может содержать
`../`, абсолютные пути или ссылки, поэтому распаковка идёт только после проверки
каждой записи, во временный каталог, и проект создаётся лишь тогда, когда
`project.json` прочитан и признан совместимым.

Аудио-экспорт опирается на таймлайн (`backend/timeline.py`): итоговый трек
собирается из активных take'ов с теми же паузами, что и рендер, stems кладутся в
общую шкалу времени, а SRT/VTT берут реальные `start_sec`/`end_sec`. Второй способ
считать время разошёлся бы с файлом на первой же замене take.
"""

import io
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from . import audio_pipeline, config, timeline
from .audio_pipeline import RenderSettings
from .db.store import get_projects_store
from .engines.base import ENGINE_INFOS, SAMPLE_RATE, requires_reference
from .voices_store import ALLOWED_AUDIO_SUFFIXES
from .voices_store import get_store as get_voices_store

logger = logging.getLogger(__name__)

# Формат контейнера. Версия растёт, когда меняется смысл полей: импорт архива
# «из будущего» должен отказать понятным текстом, а не прочитать половину.
FORMAT_NAME = "ttsproject"
FORMAT_VERSION = 1
MANIFEST_NAME = "project.json"
ARCHIVE_SUFFIX = ".ttsproject"
REFERENCE_DIR = "references"
TAKE_DIR = "takes"

# Префикс временного каталога: по нему `discard` понимает, что каталог наш, и
# удаляет его целиком после отдачи файла.
_WORK_PREFIX = "tts-export-"
_IMPORT_PREFIX = "tts-import-"


class ProjectExportError(ValueError):
    """Архив или запрос экспорта некорректны — API отдаёт это текстом 400."""


# --- общие помощники ----------------------------------------------------------
def _workdir(prefix: str = _WORK_PREFIX) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def discard(path: Path) -> None:
    """Убирает временный каталог отданного файла (вызывается после ответа)."""
    work = Path(path).parent
    if work.name.startswith(_WORK_PREFIX):
        shutil.rmtree(work, ignore_errors=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    """Имя файла для Content-Disposition: без разделителей пути и спецсимволов."""
    cleaned = "".join(
        char if char.isalnum() or char in " -_()." else "_" for char in str(name or "")
    ).strip()
    cleaned = cleaned.replace(" ", "_").strip("._")
    return cleaned[:64] or "project"


def _inside(base: Path, relative: object) -> Path | None:
    """Путь внутри `base` или None, если запись ведёт наружу.

    Проверяется и сам манифест: `file`/`reference` в `project.json` — такие же
    данные из чужого архива, как и имена записей, и доверять им нельзя.
    """
    if not relative:
        return None
    pure = PurePosixPath(str(relative))
    if pure.is_absolute() or ".." in pure.parts or "\\" in str(relative):
        return None
    candidate = (base / pure).resolve()
    root = base.resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


# --- экспорт: сборка манифеста -------------------------------------------------
def _used_voice_ids(project: dict) -> list[str]:
    """Голоса, без которых проект не зазвучит: назначенные спикерам и репликам."""
    used: list[str] = []
    for speaker in project.get("speakers") or []:
        voice_id = str(speaker.get("voice_id") or "")
        if voice_id and voice_id not in used:
            used.append(voice_id)
    for replica in project.get("replicas") or []:
        voice_id = str(replica.get("voice_override") or "")
        if voice_id and voice_id not in used:
            used.append(voice_id)
    return used


def _reference_name(path: Path, number: int) -> str | None:
    """Имя референса внутри архива: своё, по номеру, с исходным форматом."""
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        logger.warning("Референс %s не попадает в архив: формат %s не поддерживается", path, suffix)
        return None
    return f"{REFERENCE_DIR}/ref-{number:02d}{suffix}"


def _voice_catalog(project: dict, archive: zipfile.ZipFile) -> tuple[dict[str, str], list[dict]]:
    """Описания используемых голосов плюс файлы их референсов в архиве.

    Возвращается и карта `voice_id → ключ манифеста`: спикеры и реплики ссылаются
    на голос ключом, потому что при импорте идентификаторы пересоздаются (голос
    сопоставляется по имени, а не по id с чужой машины).
    """
    store = get_voices_store()
    keys: dict[str, str] = {}
    entries: list[dict] = []
    for voice_id in _used_voice_ids(project):
        key = f"v{len(entries) + 1}"
        entry = {
            "key": key,
            "id": voice_id,
            "name": "",
            "gender": "other",
            "ref_text": "",
            "engine": "",
            "engine_params": {},
            "preset": {},
            "reference": None,
        }
        voice = store.get(voice_id)
        if voice is None:
            logger.warning("Голос %s не найден в voices.json — едет без описания", voice_id)
        else:
            entry.update(
                {
                    "name": voice.name,
                    "gender": voice.gender,
                    "ref_text": voice.ref_text,
                    "engine": voice.engine,
                    "engine_params": dict(voice.engine_params or {}),
                    "preset": dict(voice.preset or {}),
                }
            )
            # Голос встроенного движка референса не имеет вовсе (см.
            # `EngineInfo.builtin_voices`): это не потерянный файл, и в архив ему
            # ехать нечем — предупреждать здесь не о чем.
            if voice.audio_file:
                reference = voice.audio_path
                name = _reference_name(reference, len(entries) + 1) if reference.is_file() else None
                if name is None:
                    logger.warning(
                        "Референс голоса «%s» потерян — архив поедет без него", voice.name
                    )
                else:
                    archive.write(reference, name)
                    entry["reference"] = name
        keys[voice_id] = key
        entries.append(entry)
    return keys, entries


def _take_entries(project: dict, archive: zipfile.ZipFile) -> list[dict]:
    """Take'ы проекта: файл в архив (если он есть) и всё, чем он получен.

    Отсутствующий файл — не ошибка: запись в базе может пережить файл, и архив
    обязан собираться, помечая такой take недоступным. Иначе один потерянный wav
    делал бы неэкспортируемым весь проект.
    """
    entries: list[dict] = []
    for replica in project.get("replicas") or []:
        index = int(replica["index"])
        active = timeline.active_take(replica)
        active_id = None if active is None else int(active["id"])
        for take in replica.get("takes") or []:
            entry = {
                "replica_index": index,
                "label": str(take.get("label") or ""),
                "file": None,
                "available": False,
                "selected": int(take["id"]) == active_id,
                "seed": take.get("seed"),
                "engine": str(take.get("engine") or ""),
                "parameters": dict(take.get("parameters") or {}),
                "duration_sec": float(take.get("duration_sec") or 0.0),
                "qa": take.get("qa"),
                "quality": take.get("quality"),
            }
            path = Path(str(take.get("audio_path") or ""))
            if path.is_file():
                suffix = path.suffix.lower()
                if suffix not in ALLOWED_AUDIO_SUFFIXES:
                    suffix = ".wav"
                name = f"{TAKE_DIR}/r{index + 1:03d}-t{len(entries) + 1:03d}{suffix}"
                archive.write(path, name)
                entry["file"] = name
                entry["available"] = True
            else:
                logger.warning(
                    "Take %s реплики %s: файл %s недоступен — в архив не попадёт",
                    take.get("id"), index + 1, path,
                )
            entries.append(entry)
    return entries


def _manifest(project: dict, voice_keys: dict[str, str], voices: list[dict], takes: list[dict]) -> dict:
    def voice_key(value: object) -> str | None:
        key = voice_keys.get(str(value or ""))
        return key

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "exported_at": _now(),
        "project": {
            "name": project.get("name") or "",
            "mode": project.get("mode") or config.PROJECT_MODE_DIALOGUE,
            "source_text": project.get("source_text") or "",
            "render_settings": dict(project.get("render_settings") or {}),
            "status": project.get("status") or config.PROJECT_STATUS_DRAFT,
        },
        "voices": voices,
        "speakers": [
            {
                "key": str(speaker.get("key") or ""),
                "label": str(speaker.get("label") or speaker.get("key") or ""),
                "voice_key": voice_key(speaker.get("voice_id")),
                "overrides": dict(speaker.get("overrides") or {}),
            }
            for speaker in project.get("speakers") or []
        ],
        "replicas": [
            {
                "index": int(replica["index"]),
                "text": str(replica.get("text") or ""),
                "speaker": str(replica.get("speaker") or ""),
                "voice_key": voice_key(replica.get("voice_id")),
                "voice_override_key": voice_key(replica.get("voice_override")),
                "overrides": dict(replica.get("overrides") or {}),
                "status": str(replica.get("status") or config.REPLICA_STATUS_PENDING),
            }
            for replica in project.get("replicas") or []
        ],
        "takes": takes,
    }


def export_project_archive(project: dict) -> Path:
    """Собирает `.ttsproject` во временном каталоге и возвращает путь к архиву."""
    work = _workdir()
    target = work / f"{safe_name(project.get('name'))}{ARCHIVE_SUFFIX}"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        voice_keys, voices = _voice_catalog(project, archive)
        takes = _take_entries(project, archive)
        manifest = _manifest(project, voice_keys, voices, takes)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info(
        "Проект %s: архив собран (%s голосов, %s take'ов)",
        project.get("id"), len(voices), len(takes),
    )
    return target


# --- экспорт: аудио, транскрипт, субтитры --------------------------------------
def render_settings_for(project: dict) -> RenderSettings:
    """Настройки сборки из карточки проекта — те же, что у рендера.

    Читаются только те поля, что влияют на время: пауза и кроссфейд. Проверка
    качества и автоударения касаются синтеза, а экспорт ничего не синтезирует.
    """
    saved = project.get("render_settings") or {}
    pause_ms = int(saved.get("pause_ms", config.DEFAULT_PAUSE_MS))
    pause_ms = min(max(pause_ms, config.PAUSE_MS_RANGE[0]), config.PAUSE_MS_RANGE[1])
    cross_fade = float(saved.get("cross_fade_duration", config.DEFAULT_CROSS_FADE_DURATION))
    cross_fade = min(max(cross_fade, config.CROSS_FADE_RANGE[0]), config.CROSS_FADE_RANGE[1])
    return RenderSettings(pause_ms=pause_ms, cross_fade_duration=cross_fade)


def project_timings(project: dict) -> tuple[RenderSettings, list[timeline.ReplicaTiming]]:
    """Таймлайн проекта — тот же расчёт, что отдаёт `GET /timeline`."""
    render = render_settings_for(project)
    timings = timeline.replica_timings(
        project.get("replicas") or [],
        render,
        speakers=timeline.speaker_pause_overrides(project.get("speakers") or []),
    )
    return render, timings


def _replicas_by_index(project: dict) -> dict[int, dict]:
    return {int(replica["index"]): replica for replica in project.get("replicas") or []}


def _fit(audio: np.ndarray, expected_sec: float) -> np.ndarray:
    """Приводит кусок к длительности из карточки take'а.

    Таймлайн считает время по `duration_sec`, а файл может отличаться на считаные
    сэмплы (перезапись вручную, чужой рендер). Экспорт обязан повторять таймлайн,
    поэтому расхождение выравнивается — с предупреждением, чтобы это не осталось
    незамеченным.
    """
    expected = round(max(float(expected_sec or 0.0), 0.0) * SAMPLE_RATE)
    if expected <= 0 or audio.size == expected:
        return audio
    logger.warning(
        "Экспорт: длительность take'а %s сэмплов вместо %s — выравниваю по таймлайну",
        audio.size, expected,
    )
    if audio.size > expected:
        return audio[:expected]
    return np.pad(audio, (0, expected - audio.size))


def _read_take(take: dict, duration_sec: float) -> np.ndarray | None:
    path = Path(str(take.get("audio_path") or ""))
    if not path.is_file():
        return None
    try:
        audio = audio_pipeline._read_audio(path, path.suffix.lstrip(".") or "wav")
    except Exception as exc:  # noqa: BLE001 — битый файл одного take не рушит экспорт
        logger.warning("Экспорт: не удалось прочитать %s (%s)", path, exc)
        return None
    return _fit(np.asarray(audio, dtype=np.float32).reshape(-1), duration_sec)


def _piece(timings: list[timeline.ReplicaTiming], project: dict) -> list[tuple[timeline.ReplicaTiming, np.ndarray]]:
    """Звучащие реплики с их аудио — в порядке таймлайна."""
    by_index = _replicas_by_index(project)
    result: list[tuple[timeline.ReplicaTiming, np.ndarray]] = []
    for timing in timings:
        if not timing.has_audio:
            continue
        replica = by_index.get(timing.index)
        take = None if replica is None else timeline.active_take(replica)
        if take is None:
            continue
        audio = _read_take(take, timing.duration_sec)
        if audio is None or not audio.size:
            continue
        result.append((timing, audio))
    return result


def _assembled(pieces: list[tuple[timeline.ReplicaTiming, np.ndarray]]) -> list[np.ndarray]:
    """Куски с паузами перед ними — в порядке таймлайна."""
    result: list[np.ndarray] = []
    for timing, audio in pieces:
        pause = int(SAMPLE_RATE * max(timing.pause_ms, 0) / 1000)
        if result and pause:
            result.append(np.zeros(pause, dtype=np.float32))
        result.append(audio)
    return result


def build_final_track(project: dict) -> tuple[np.ndarray, list[timeline.ReplicaTiming]]:
    """Итоговый трек из активных take'ов — по правилам сборки рендера.

    Паузы берутся из таймлайна (личная правка реплики важнее паузы спикера, та —
    общей), склейка — без перекрытия, финальный проход по громкости применяется ко
    всему треку, как в `render_dialogue`. Поэтому длительность совпадает с
    таймлайном, а звучание — с рендером.
    """
    _, timings = project_timings(project)
    pieces = _assembled(_piece(timings, project))
    if not pieces:
        return np.zeros(0, dtype=np.float32), timings
    return audio_pipeline._finalize_track(np.concatenate(pieces)), timings


def export_final_audio(project: dict, output_format: str) -> Path:
    """Собирает финальный трек проекта и пишет его файлом в заданном формате.

    Пустое звучание — ошибка, а не пустой файл: рендер трека без активных take'ов
    означал бы успешный экспорт тишины.
    """
    if output_format not in ("wav", "mp3"):
        raise ProjectExportError(
            f"Неизвестный формат экспорта: {output_format or '(пусто)'}. Доступны: wav, mp3"
        )
    audio, timings = build_final_track(project)
    if not audio.size:
        raise ProjectExportError("В проекте нет готового звучания — сначала соберите трек")
    work = _workdir()
    target = work / f"dialogue.{output_format}"
    audio_pipeline._write_audio(target, audio, output_format)
    logger.info(
        "Проект %s: экспорт %s, %.1f с (таймлайн %.1f с)",
        project.get("id"), output_format,
        audio.size / SAMPLE_RATE, timeline.timeline_duration(timings),
    )
    return target


def export_replicas_archive(project: dict) -> Path:
    """ZIP с отдельными WAV активных take'ов — по одному на звучащую реплику."""
    _, timings = project_timings(project)
    pieces = _piece(timings, project)
    if not pieces:
        raise ProjectExportError("В проекте нет готового звучания — сначала соберите трек")
    work = _workdir()
    target = work / "replicas.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for timing, audio in pieces:
            name = f"r{timing.index + 1:03d}.wav"
            wav_path = work / name
            audio_pipeline._write_audio(wav_path, audio, "wav")
            archive.write(wav_path, name)
            wav_path.unlink(missing_ok=True)
    return target


def speaker_keys(project: dict) -> list[str]:
    """Спикеры проекта в устойчивом порядке: дорожки stems идут по нему."""
    keys: list[str] = []
    for speaker in project.get("speakers") or []:
        key = str(speaker.get("key") or "")
        if key and key not in keys:
            keys.append(key)
    for replica in project.get("replicas") or []:
        key = str(replica.get("speaker") or "")
        if key and key not in keys:
            keys.append(key)
    return keys


def export_stems_archive(project: dict) -> Path:
    """ZIP со stems по спикерам в общей шкале времени.

    Дорожка каждого спикера — тот же итоговый трек, но чужие реплики заменены
    тишиной. Поэтому дорожки синхронны между собой и с итоговым файлом: место
    реплики задаётся её `start_sec`, а не порядком внутри дорожки. Складывать
    только «свои» куски подряд было бы дешевле, но тогда stems нельзя наложить
    друг на друга.
    """
    _, timings = project_timings(project)
    assembled = _piece(timings, project)
    if not assembled:
        raise ProjectExportError("В проекте нет готового звучания — сначала соберите трек")
    # Длина — ровно та же, что у собранного трека: те же куски и паузы, но без
    # финального прохода по громкости, который на разложение на дорожки не влияет.
    total = sum(audio.size for _, audio in assembled) + sum(
        int(SAMPLE_RATE * max(timing.pause_ms, 0) / 1000)
        for position, (timing, _) in enumerate(assembled)
        if position
    )
    tracks: dict[str, np.ndarray] = {key: np.zeros(total, dtype=np.float32) for key in speaker_keys(project)}
    by_index = _replicas_by_index(project)
    for timing, audio in assembled:
        replica = by_index.get(timing.index)
        key = "" if replica is None else str(replica.get("speaker") or "")
        track = tracks.setdefault(key, np.zeros(total, dtype=np.float32))
        start = round(timing.start_sec * SAMPLE_RATE)
        end = min(start + audio.size, total)
        if end > start:
            track[start:end] += audio[: end - start]

    work = _workdir()
    target = work / "stems.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for number, key in enumerate(speaker_keys(project), start=1):
            name = f"stem-{number:02d}.wav"
            wav_path = work / name
            audio_pipeline._write_audio(wav_path, tracks.get(key, np.zeros(total, dtype=np.float32)), "wav")
            archive.write(wav_path, name)
            wav_path.unlink(missing_ok=True)
    return target


def transcript_payload(project: dict) -> dict:
    """Транскрипт: реплика, спикер, голос, текст и реальные границы звучания."""
    _, timings = project_timings(project)
    by_index = _replicas_by_index(project)
    voices = get_voices_store()
    names: dict[str, str] = {}
    segments: list[dict] = []
    for timing in timings:
        if not timing.has_audio:
            continue
        replica = by_index.get(timing.index) or {}
        voice_id = str(replica.get("voice_id") or "")
        if voice_id not in names:
            voice = voices.get(voice_id)
            names[voice_id] = voice.name if voice is not None else ""
        segments.append(
            {
                "index": timing.index,
                "speaker": str(replica.get("speaker") or ""),
                "speaker_label": str(replica.get("label") or replica.get("speaker") or ""),
                "voice_id": voice_id,
                "voice_name": names[voice_id],
                "text": str(replica.get("text") or ""),
                "start_sec": round(timing.start_sec, timeline.ROUND_DIGITS),
                "end_sec": round(timing.end_sec, timeline.ROUND_DIGITS),
                "duration_sec": round(timing.duration_sec, timeline.ROUND_DIGITS),
            }
        )
    return {
        "project_id": str(project.get("id") or ""),
        "name": str(project.get("name") or ""),
        "duration_sec": round(timeline.timeline_duration(timings), timeline.ROUND_DIGITS),
        "segments": segments,
    }


def export_transcript(project: dict) -> Path:
    work = _workdir()
    target = work / "transcript.json"
    target.write_text(
        json.dumps(transcript_payload(project), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _timestamp(milliseconds: int, separator: str) -> str:
    hours, rest = divmod(max(int(milliseconds), 0), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def subtitles_text(project: dict, file_format: str) -> str:
    """SRT или VTT по реальным таймстемпам таймлайна.

    Реплики без звука в субтитры не попадают: у них нет ни начала, ни конца, а
    выдумывать им место значило бы рассинхронизировать файл с дорожкой.
    """
    if file_format not in ("srt", "vtt"):
        raise ProjectExportError(
            f"Неизвестный формат субтитров: {file_format or '(пусто)'}. Доступны: srt, vtt"
        )
    _, timings = project_timings(project)
    by_index = _replicas_by_index(project)
    separator = "," if file_format == "srt" else "."
    blocks: list[str] = []
    for number, timing in enumerate((item for item in timings if item.has_audio), start=1):
        replica = by_index.get(timing.index) or {}
        start = _timestamp(round(timing.start_sec * 1000), separator)
        end = _timestamp(round(timing.end_sec * 1000), separator)
        text = str(replica.get("text") or "").strip()
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    if file_format == "vtt":
        return "WEBVTT\n\n" + "\n".join(blocks)
    return "\n".join(blocks)


def export_subtitles(project: dict, file_format: str) -> Path:
    work = _workdir()
    target = work / f"subtitles.{file_format}"
    target.write_text(subtitles_text(project, file_format), encoding="utf-8")
    return target


# --- импорт --------------------------------------------------------------------
def _mb(value: int) -> int:
    """Размер в мегабайтах — для текста отказа, где байты ничего не говорят."""
    return value // (1024 * 1024)


def _validate_entries(archive: zipfile.ZipFile, work: Path) -> None:
    """Проверяет каждую запись до распаковки: путь, тип, место назначения, размер.

    Символические ссылки отклоняются отдельно: ZIP умеет хранить их как обычную
    запись с флагом, и распаковка такой ссылки создала бы файл вне каталога уже
    на следующем шаге.

    Число записей и их суммарный размер ограничены здесь, до `_extract`: сжатый
    «архив-бомба» весит мегабайты, а разворачивается в гигабайты, поэтому предел
    на сжатый файл от него не спасает. Обе величины берутся из заголовков ZIP, без
    распаковки, так что отклонение происходит до записи файлов на диск.
    """
    infos = archive.infolist()
    if not infos:
        raise ProjectExportError("Архив пуст")
    if len(infos) > config.ARCHIVE_MAX_ENTRIES:
        raise ProjectExportError(
            f"В архиве слишком много записей: {len(infos)} "
            f"(предел {config.ARCHIVE_MAX_ENTRIES})"
        )
    total_uncompressed = 0
    for info in infos:
        total_uncompressed += info.file_size
        if total_uncompressed > config.ARCHIVE_MAX_UNCOMPRESSED_BYTES:
            raise ProjectExportError(
                "Распакованный архив больше "
                f"{_mb(config.ARCHIVE_MAX_UNCOMPRESSED_BYTES)} МБ — файл повреждён "
                "или это не архив проекта"
            )
        name = info.filename
        if not name or name.startswith("/") or "\\" in name or (len(name) > 1 and name[1] == ":"):
            raise ProjectExportError(f"Недопустимый путь в архиве: {name}")
        parts = PurePosixPath(name).parts
        if any(part == ".." for part in parts):
            raise ProjectExportError(f"Недопустимый путь в архиве: {name}")
        if info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ProjectExportError(f"В архиве есть символическая ссылка: {name}")
        if name != MANIFEST_NAME and not (parts and parts[0] in (REFERENCE_DIR, TAKE_DIR) and len(parts) > 1):
            raise ProjectExportError(f"Неожиданная запись в архиве: {name}")
        if _inside(work, name) is None:
            raise ProjectExportError(f"Запись ведёт за пределы каталога распаковки: {name}")


def _extract(archive: zipfile.ZipFile, work: Path) -> None:
    """Распаковывает записи вручную: файлы пишем сами, ссылки создать нечем."""
    for info in archive.infolist():
        if info.is_dir():
            continue
        target = _inside(work, info.filename)
        if target is None:  # уже проверено, но путь не должен зависеть от порядка
            raise ProjectExportError(f"Недопустимый путь в архиве: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _read_manifest(work: Path) -> dict:
    path = work / MANIFEST_NAME
    if not path.is_file():
        raise ProjectExportError(
            "В архиве нет project.json — это не архив проекта (.ttsproject)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ProjectExportError(f"Файл project.json повреждён: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ProjectExportError("Это не архив проекта: незнакомый формат контейнера")
    version = payload.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise ProjectExportError("В архиве не указана версия формата — файл повреждён")
    if version > FORMAT_VERSION:
        raise ProjectExportError(
            f"Версия формата архива {version} новее поддерживаемой {FORMAT_VERSION}. "
            "Обновите приложение."
        )
    return payload


def _create_voice(store, entry: dict, work: Path):
    """Недостающий голос из архива: по имени, полу, движку и референсу.

    Голос без референса или без его расшифровки создать нельзя — синтез по такому
    голосу всё равно не пойдёт, поэтому он пропускается с предупреждением, а
    проект остаётся импортированным (спикер просто без голоса). Исключение —
    движок со встроенными голосами: референса у его голоса нет по устройству, и
    голос восстанавливается по имени, полу и движку (`EngineInfo.builtin_voices`).
    """
    name = str(entry.get("name") or "").strip()
    ref_text = str(entry.get("ref_text") or "").strip()
    reference = _inside(work, entry.get("reference"))
    engine = str(entry.get("engine") or "")
    if engine not in ENGINE_INFOS:
        engine = ""
    builtin = bool(engine) and not requires_reference(engine)
    if not builtin and (not name or not ref_text or reference is None or not reference.is_file()):
        logger.warning(
            "Импорт: голос «%s» пропущен — нет референса или его расшифровки",
            name or entry.get("key"),
        )
        return None
    suffix = ".wav"
    audio_bytes = b""
    if not builtin:
        suffix = reference.suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            suffix = ".wav"
        audio_bytes = reference.read_bytes()
    try:
        voice = store.create(
            name=name,
            gender=str(entry.get("gender") or "other"),
            ref_text=ref_text,
            audio_filename="" if builtin else f"import{suffix}",
            audio_bytes=audio_bytes,
            verify_ref_text=False,
            engine=engine,
        )
    except (ValueError, OSError) as exc:
        logger.warning("Импорт: голос «%s» не создан (%s)", name, exc)
        return None
    try:
        store.update(
            voice.id,
            engine=engine or None,
            engine_params=dict(entry.get("engine_params") or {}),
            preset=dict(entry.get("preset") or {}),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Импорт: настройки голоса «%s» не применены (%s)", name, exc)
    return store.get(voice.id)


def _restore_voices(entries: list[dict], work: Path) -> dict[str, str]:
    """Карта ключ манифеста → id голоса на этой машине.

    Существующие голоса не перезаписываются: совпадение по имени означает, что
    пользователь уже настроил этот голос здесь, и импорт не должен откатывать его
    движок или пресет.
    """
    store = get_voices_store()
    known = {voice.name.strip().casefold(): voice for voice in store.list()}
    mapping: dict[str, str] = {}
    for entry in entries:
        key = str(entry.get("key") or "")
        if not key:
            continue
        name = str(entry.get("name") or "").strip()
        voice = known.get(name.casefold()) if name else None
        if voice is None and name:
            voice = _create_voice(store, entry, work)
            if voice is not None:
                known[voice.name.strip().casefold()] = voice
        mapping[key] = voice.id if voice is not None else ""
    return mapping


def _copy_takes(manifest: dict, work: Path, project_id: str) -> dict[int, list[dict]]:
    """Переносит файлы take'ов в каталог проекта и собирает записи для базы."""
    directory = config.PROJECTS_OUTPUT_DIR / project_id
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[dict]] = {}
    for entry in manifest.get("takes") or []:
        try:
            index = int(entry.get("replica_index"))
        except (TypeError, ValueError):
            continue
        source = _inside(work, entry.get("file"))
        if source is None or not source.is_file():
            logger.warning(
                "Импорт: take реплики %s недоступен — пропускаю", index + 1
            )
            continue
        suffix = source.suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            suffix = ".wav"
        target = directory / f"r{index + 1}-{uuid.uuid4().hex[:8]}{suffix}"
        shutil.copyfile(source, target)
        result.setdefault(index, []).append(
            {
                "audio_path": str(target),
                "label": str(entry.get("label") or ""),
                "seed": entry.get("seed"),
                "engine": str(entry.get("engine") or ""),
                "parameters": dict(entry.get("parameters") or {}),
                "duration_sec": float(entry.get("duration_sec") or 0.0),
                "qa": entry.get("qa"),
                "quality": entry.get("quality"),
                "selected": bool(entry.get("selected")),
            }
        )
    return result


def _restore_speakers(manifest: dict, voice_map: dict[str, str]) -> dict[str, dict]:
    speakers: dict[str, dict] = {}
    for entry in manifest.get("speakers") or []:
        key = str(entry.get("key") or "")
        if not key:
            continue
        speakers[key] = {
            "label": str(entry.get("label") or key),
            "voice_id": voice_map.get(str(entry.get("voice_key") or ""), ""),
            "overrides": dict(entry.get("overrides") or {}),
        }
    return speakers


def _restore_replicas(
    manifest: dict, voice_map: dict[str, str], speakers: dict[str, dict]
) -> list[dict]:
    replicas: list[dict] = []
    for entry in manifest.get("replicas") or []:
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        key = str(entry.get("speaker") or "")
        # Спикер мог не попасть в список (архив правили руками): создаём его по
        # реплике, иначе она осталась бы без голоса и без дорожки на таймлайне.
        speakers.setdefault(key, {"label": key, "voice_id": "", "overrides": {}})
        override_key = str(entry.get("voice_override_key") or "")
        override = voice_map.get(override_key) if override_key else None
        effective = voice_map.get(str(entry.get("voice_key") or ""), "")
        replicas.append(
            {
                "index": index,
                "text": str(entry.get("text") or ""),
                "speaker": key,
                "voice_id": effective or str(speakers[key].get("voice_id") or ""),
                "voice_override": override or None,
                "overrides": dict(entry.get("overrides") or {}),
                "status": str(entry.get("status") or config.REPLICA_STATUS_PENDING),
            }
        )
    replicas.sort(key=lambda item: item["index"])
    return replicas


def _restore(manifest: dict, work: Path) -> dict:
    meta = manifest.get("project") or {}
    name = str(meta.get("name") or "").strip() or "Импортированный проект"
    mode = str(meta.get("mode") or config.PROJECT_MODE_DIALOGUE)
    if mode not in config.PROJECT_MODES:
        mode = config.PROJECT_MODE_DIALOGUE
    render_settings = meta.get("render_settings")
    render_settings = dict(render_settings) if isinstance(render_settings, dict) else {}
    status = str(meta.get("status") or config.PROJECT_STATUS_DRAFT)

    store = get_projects_store()
    project = store.create_project(
        name=name,
        source_text=str(meta.get("source_text") or ""),
        mode=mode,
        render_settings=render_settings,
    )
    project_id = str(project["id"])
    try:
        voice_map = _restore_voices(manifest.get("voices") or [], work)
        speakers = _restore_speakers(manifest, voice_map)
        replicas = _restore_replicas(manifest, voice_map, speakers)
        takes = _copy_takes(manifest, work, project_id)
        store.import_project_content(
            project_id,
            speakers=speakers,
            replicas=replicas,
            takes=takes,
            status=status,
        )
    except Exception:
        # Половина проекта хуже отсутствия проекта: недособранный диалог ещё и
        # занял бы id, а его файлы остались бы мусором в output/projects/.
        store.delete_project(project_id)
        raise
    logger.info(
        "Проект %s импортирован: «%s», %s реплик, %s take'ов",
        project_id, name, len(replicas), sum(len(items) for items in takes.values()),
    )
    return store.get_project(project_id)  # type: ignore[return-value]


def import_archive(data: bytes) -> dict:
    """Разворачивает `.ttsproject` в новый проект и возвращает его карточку."""
    if not data:
        raise ProjectExportError("Пустой файл архива")
    # Предел на сжатый файл — здесь, а не только на входе запроса: `import_archive`
    # зовётся и в обход FastAPI (тесты, скрипты), и проверка обязана быть в самом
    # формате, а не в транспорте.
    if len(data) > config.MAX_ARCHIVE_BYTES:
        raise ProjectExportError(
            f"Архив больше {_mb(config.MAX_ARCHIVE_BYTES)} МБ — "
            "это не архив проекта или файл повреждён"
        )
    stream = io.BytesIO(data)
    if not zipfile.is_zipfile(stream):
        raise ProjectExportError(
            "Файл не является ZIP-архивом проекта (.ttsproject)"
        )
    with tempfile.TemporaryDirectory(prefix=_IMPORT_PREFIX) as raw:
        work = Path(raw)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                _validate_entries(archive, work)
                _extract(archive, work)
        except zipfile.BadZipFile as exc:
            raise ProjectExportError("Архив повреждён и не читается как ZIP") from exc
        manifest = _read_manifest(work)
        return _restore(manifest, work)
