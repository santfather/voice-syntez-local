"""Монтаж записанного диалога: дубли → обработка → паузы → LUFS → MP3 (§15, §41).

Главный результат режима — **один цельный файл** с диалогом в исходном порядке
реплик. Порядок записи (по диалогу или по ролям) на монтаж не влияет: он
определяется только `index` реплики из общего разбора (§4, Test C).

Пайплайн сознательно тонкий: он не синтезирует, не парсит и не редактирует
записи — только собирает уже обработанные дубли в мастер и применяет к мастеру
существующие финальные проходы проекта (LUFS + лимитер). Тяжёлая обработка
запускается в потоке, а не внутри HTTP-запроса (§22).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audio_pipeline, config, recording_audio, recording_store
from .engines.base import SAMPLE_RATE
from .recording_store import RecordedTake, RecordingProject

logger = logging.getLogger(__name__)

# Формат финального файла: задача фазы — один MP3 (§41). Внутренний мастер
# остаётся WAV: многократное MP3 → decode → MP3 между этапами запрещено.
MASTER_NAME = "dialogue-master.wav"
FINAL_NAME = "dialogue.mp3"


class RecordingRenderError(ValueError):
    """Монтаж невозможен — API отдаёт это текстом 400 с причиной."""


class RenderBusyError(RuntimeError):
    """Монтаж этого проекта уже идёт: второй запуск означал бы две записи в один файл."""


@dataclass
class RenderState:
    """Состояние монтажа: интерфейс опрашивает его после запуска (§22)."""

    status: str = "idle"  # idle | running | done | error
    progress: float = 0.0
    message: str = ""
    error: str = ""
    output_path: str = ""
    url: str = ""
    duration_sec: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    warnings: list = field(default_factory=list)


_states: dict[str, RenderState] = {}
_lock = threading.Lock()


def render_state(project_id: str) -> RenderState:
    with _lock:
        return _states.setdefault(project_id, RenderState())


def _started_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def begin_render(project_id: str) -> RenderState:
    """Помечает монтаж запущенным **до** того, как он уйдёт в поток.

    Между `POST /render` (202) и первым оператором рабочего потока есть окно, в
    котором состояние ещё хранит прошлый монтаж. Клиент, опрашивающий
    `render/status` (тест или интерфейс), в этом окне видит «готово» с прежней
    длительностью и решает, что новый монтаж уже прошёл, — то есть выбор другого
    дубля выглядел бы как «ничего не изменилось». Поэтому запуск объявляется здесь
    и под тем же замком, что и проверка «уже идёт»: два одновременных запуска
    писали бы в один и тот же output-файл.
    """
    with _lock:
        state = _states.setdefault(project_id, RenderState())
        if state.status == "running":
            raise RenderBusyError("Монтаж уже идёт")
        state.status = "running"
        state.progress = 0.0
        state.message = "монтаж запущен"
        state.error = ""
        state.warnings = []
        state.started_at = _started_at()
        state.finished_at = ""
        return state


def abort_render(project_id: str, error: str) -> RenderState:
    """Монтаж не удалось даже запустить (например, executor уже закрыт).

    Без этого состояние осталось бы «running» навсегда, и кнопка сборки в
    интерфейсе больше не разблокировалась бы.
    """
    with _lock:
        state = _states.setdefault(project_id, RenderState())
        state.status = "error"
        state.error = error
        state.message = "монтаж не запустился"
        state.finished_at = _started_at()
        return state


def missing_replicas(project: RecordingProject) -> list[int]:
    """Индексы незаписанных реплик — блокирующее условие монтажа (§19, Test F)."""
    return project.missing_replicas()


def processed_take(
    project: RecordingProject,
    take: RecordedTake,
    *,
    text: str = "",
) -> tuple[np.ndarray, list[str]]:
    """Обработанный дубль по настройкам его записываемого голоса.

    Всегда начинается с **сырого** файла: обработанные версии — кэш, а не вход
    (§16). Настройки берутся у профиля, назначенного роли реплики; если профиль
    не назначен, дубль обрабатывается настройками по умолчанию (это не ошибка:
    пользователь мог записать до создания профиля).
    """
    profile = project.profile(take.voice_profile_id) if take.voice_profile_id else None
    store = recording_store.get_store()
    raw_path = store.raw_path(project.id, take)
    if not raw_path.exists():
        raise RecordingRenderError(f"Сырой дубль {take.raw_file} не найден")
    raw = recording_audio.decode(raw_path.read_bytes(), raw_path.suffix or ".wav")
    return recording_audio.process(
        raw,
        speed=profile.speed if profile else 1.0,
        pitch_semitones=profile.pitch_semitones if profile else 0.0,
        denoise_enabled=bool(profile.denoise) if profile else False,
        text=text,
    )


def pause_samples(pause_ms: float) -> int:
    return int(SAMPLE_RATE * max(float(pause_ms), 0) / 1000)


def assemble(
    project: RecordingProject,
    *,
    pause_ms: float | None = None,
    on_progress=None,
) -> tuple[Path, Path, float, list[str]]:
    """Собирает диалог: возвращает (мастер WAV, финальный MP3, длительность, предупреждения).

    Пауза вставляется **между** репликами и только между ними: тишина внутри
    записей уже срезана обрезкой краёв, поэтому реальный зазор равен `pause_ms`,
    а не «тишина записи + пауза» (§27).
    """
    total = len(project.replicas)
    if not total:
        raise RecordingRenderError("Диалог не разобран — нечего собирать")
    missing = missing_replicas(project)
    if missing:
        names = ", ".join(str(index + 1) for index in missing[:10])
        raise RecordingRenderError(
            f"Не записаны реплики: {names}. Соберите диалог без пропусков."
        )

    settings = dict(project.render_settings or {})
    pause = float(settings.get("pause_ms", config.DEFAULT_PAUSE_MS) if pause_ms is None else pause_ms)
    pieces: list[np.ndarray] = []
    warnings: list[str] = []
    for position, replica in enumerate(project.replicas):
        index = int(replica["index"])
        take = project.active_take(index)
        if take is None:  # проверено выше, но типобезопасность не помешает
            raise RecordingRenderError(f"Реплика {index + 1}: нет активного дубля")
        chunk, chunk_warnings = processed_take(project, take, text=str(replica.get("text") or ""))
        warnings.extend(f"реплика {index + 1}: {item}" for item in chunk_warnings)
        if position and pause > 0:
            pieces.append(np.zeros(pause_samples(pause), dtype=np.float32))
        pieces.append(chunk)
        if on_progress:
            on_progress((position + 1) / total, f"реплика {index + 1} из {total}")

    master = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    # Финальные проходы проекта: воспринимаемая громкость и защита от клиппинга
    # считаются по всему диалогу, а не по каждой роли отдельно (§29).
    master = audio_pipeline._finalize_track(master)

    store = recording_store.get_store()
    output_dir = store.output_dir(project.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / MASTER_NAME
    audio_pipeline._write_audio(master_path, master, "wav")
    final_format = config.RECORDING_OUTPUT_FORMAT or "mp3"
    final_name = FINAL_NAME if final_format == "mp3" else f"dialogue.{final_format}"
    final_path = output_dir / final_name
    # MP3 кодируется **один раз** из финального мастера: повторные перекодирования
    # между этапами накапливали бы потери кодека (§41).
    audio_pipeline._write_audio(final_path, master, final_format)
    duration = float(master.size) / float(SAMPLE_RATE)
    logger.info(
        "Проект записи %s: диалог собран (%.2f с, %s)",
        project.id, duration, final_path.name,
    )
    return master_path, final_path, duration, warnings


def render(project_id: str, *, pause_ms: float | None = None) -> RenderState:
    """Синхронный монтаж: вызывается из потока, состояние видно через `render_state`."""
    state = render_state(project_id)
    store = recording_store.get_store()
    try:
        project = store.require_project(project_id)
    except KeyError as exc:
        # Состояние обязано перестать быть «running»: запуск уже объявлен
        # (`begin_render`), и без этого кнопка сборки осталась бы заблокированной
        # до перезапуска процесса — проекта-то уже нет.
        abort_render(project_id, "Проект записи не найден")
        raise RecordingRenderError("Проект записи не найден") from exc

    with _lock:
        state.status = "running"
        state.progress = 0.0
        state.error = ""
        state.message = "собираю диалог"
        state.started_at = _started_at()

    def progress(ratio: float, message: str) -> None:
        with _lock:
            state.progress = round(ratio, 4)
            state.message = message

    try:
        master_path, final_path, duration, warnings = assemble(
            project, pause_ms=pause_ms, on_progress=progress
        )
    except Exception as exc:  # noqa: BLE001 — состояние обязано быть видимым
        with _lock:
            state.status = "error"
            state.error = str(exc)
            state.message = "монтаж не удался"
            state.finished_at = _started_at()
        store.update_project(project_id, status=recording_store.PROJECT_STATUS_ERROR, last_error=str(exc))
        logger.warning("Проект записи %s: монтаж не удался (%s)", project_id, exc)
        return state

    with _lock:
        state.status = "done"
        state.progress = 1.0
        state.message = f"готово: {duration:.1f} с"
        state.output_path = str(final_path)
        state.url = f"/api/recording-projects/{project_id}/audio"
        state.duration_sec = round(duration, 3)
        state.warnings = warnings
        state.finished_at = _started_at()
    store.update_project(
        project_id,
        status=recording_store.PROJECT_STATUS_RENDERED,
        last_error="",
    )
    del master_path
    return state


def reset_states() -> None:
    """Сброс состояний монтажа — для тестов."""
    with _lock:
        _states.clear()


__all__ = [
    "FINAL_NAME",
    "MASTER_NAME",
    "RecordingRenderError",
    "RenderBusyError",
    "RenderState",
    "abort_render",
    "assemble",
    "begin_render",
    "missing_replicas",
    "processed_take",
    "render",
    "render_state",
    "reset_states",
]
