"""Таймлайн проекта: где какая реплика звучит в итоговом файле.

Здесь единственная правда о времени. Границы считаются по **реальным
длительностям активных take'ов**, а не по числу знаков: длина куска зависит от
движка, скорости, пауз по краям и обрезки тишины (`_prepare_chunk`), и оценка
«по символам» разошлась бы с файлом на первом же рендере.

Правила совпадают со сборкой трека (`audio_pipeline.render_dialogue`):

* перед каждой репликой, кроме первой, вставляется пауза её спикера
  (`_pause_samples`: личный `pause_override_ms` важнее общего `pause_ms`);
* границы реплики — `[начало паузы, начало паузы + длительность куска)`;
* курсор идёт по порядку реплик и накапливается в секундах.

Хранилища `start_sec`/`end_sec` в базе нет намеренно: оно разошлось бы с
файлом после первой же замены take. Таймлайн считается заново на каждый
запрос, поэтому замена звучания пересчитывает хвост сама собой.

`cross_fade_duration` на длину трека не влияет: это ручка F5 для сшивания
батчей **внутри** куска (см. `engines/f5_engine`), а не склейка реплик.
Склейка реплик — `np.concatenate` без перекрытия, и `_finalize_track` длину не
меняет, поэтому менять границы от кроссфейда нечему. Это зафиксировано тестом.

Смешанные движки на семантику времени не влияют: все движки отдают 24 кГц, а
длительность берётся из take'а, которым реплика звучит сейчас.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio_pipeline import RenderSettings, SpeakerSettings

# Точность отдачи в API: миллисекунды. Считается всё в float, округляется
# только сериализация — иначе ошибка копилась бы на каждом шаге.
ROUND_DIGITS = 3


@dataclass
class ReplicaTiming:
    """Место одной реплики во времени и то, чем она звучит.

    `has_audio = False` — у реплики ещё нет активного take (или файл потерян).
    Такая реплика остаётся в списке с нулевой длительностью и не двигает
    курсор: файла с ней пока не существует, и сдвигать ею чужие границы значило
    бы обещать звук, которого нет.
    """

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    pause_ms: int
    has_audio: bool
    take_id: int | None = None
    take_label: str = ""
    engine: str = ""


def active_take(replica: dict) -> dict | None:
    """Активный вариант реплики: выбранный take, иначе — последний.

    Файл собирается из выбранного варианта, поэтому и время берётся у него.
    Фолбэк на последний вариант нужен для проектов, у которых take есть, а
    `selected_take_id` пуст: показывать такую реплику без звука было бы неправдой.
    """
    takes = list(replica.get("takes") or [])
    if not takes:
        return None
    selected = replica.get("selected_take_id")
    if selected is not None:
        for take in takes:
            if int(take["id"]) == int(selected):
                return take
    return takes[-1]


def _pause_override(replica: dict, speakers: dict[str, int | None] | None) -> int | None:
    """Личная пауза реплики, если она её выбрала.

    Приоритет тот же, что при сборке (`audio_pipeline.tuning_for`): правка
    реплики (`overrides`) перекрывает паузу спикера, а её отсутствие — «общая для
    диалога». Пауза спикера приходит готовой картой `speakers`, потому что
    лежит в слое своего спикера, а не в реплике.
    """
    own = SpeakerSettings.from_dict(replica.get("overrides") or {}).pause_override_ms
    if own is not None:
        return own
    return (speakers or {}).get(str(replica.get("speaker") or ""))


def _pause_ms(override: int | None, render: RenderSettings) -> int:
    """Эффективная пауза перед репликой — тем же правилом, что и в файле.

    Перед первой репликой паузы не бывает (сборка её не вставляет), но в ответе
    значение остаётся: это настройка реплики, которую показывает инспектор, и
    подменять её нулём значило бы показывать не то, чем реплика читается.
    """
    return int(render.pause_ms if override is None else override)


def _exists(path: object) -> bool:
    return Path(str(path)).exists()


def speaker_pause_overrides(speakers: list[dict]) -> dict[str, int | None]:
    """Личные паузы спикеров проекта — слой между пресетом и правкой реплики."""
    result: dict[str, int | None] = {}
    for speaker in speakers:
        settings = SpeakerSettings.from_dict(speaker.get("overrides") or {})
        result[str(speaker["key"])] = settings.pause_override_ms
    return result


def replica_timings(
    replicas: list[dict],
    render: RenderSettings,
    has_file: Callable[[object], bool] | None = None,
    speakers: dict[str, int | None] | None = None,
) -> list[ReplicaTiming]:
    """Границы всех реплик подряд, по реальным длительностям их take'ов.

    Реплика без take не двигает курсор, но остаётся в списке: её надо видеть на
    таймлайне, чтобы сгенерировать. `has_file` — проверка существования файла
    take'а (по умолчанию `Path.exists`); `speakers` — личные паузы спикеров
    (см. `speaker_pause_overrides`): без них правка паузы слота не доехала бы до
    границ, и таймлайн разошёлся бы с файлом.
    """
    check_file = _exists if has_file is None else has_file

    timings: list[ReplicaTiming] = []
    cursor = 0.0
    has_pieces = False
    for replica in replicas:
        index = int(replica["index"])
        override = _pause_override(replica, speakers)
        pause_ms = _pause_ms(override, render)
        take = active_take(replica)
        duration = 0.0
        has_audio = False
        if take is not None:
            try:
                has_audio = bool(check_file(take.get("audio_path")))
            except OSError:  # недоступный путь — считаем, что звука нет
                has_audio = False
            if has_audio:
                duration = max(float(take.get("duration_sec") or 0.0), 0.0)

        # Пауза повторяет цикл сборки: только перед репликой, которая звучит, и
        # только если до неё уже был звук. `np.zeros(0)` в файл не попадает, и
        # нулевая длительность курсор не двигает.
        if has_pieces and pause_ms > 0:
            cursor += pause_ms / 1000.0

        start = cursor
        cursor += duration
        if has_audio:
            has_pieces = True
        timings.append(
            ReplicaTiming(
                index=index,
                start_sec=start,
                end_sec=cursor,
                duration_sec=cursor - start,
                pause_ms=pause_ms,
                has_audio=has_audio,
                take_id=None if take is None else int(take["id"]),
                take_label="" if take is None else str(take.get("label") or ""),
                engine="" if take is None else str(take.get("engine") or ""),
            )
        )
    return timings


def timeline_duration(timings: list[ReplicaTiming]) -> float:
    """Длительность файла по таймлайну.

    Это конец последней реплики, **у которой есть звук**: пауза перед репликой
    без take в файл не попадает (вставлять её не перед чем), и хвостовой паузы
    сборка не добавляет вовсе. Для файла без звука — ноль.
    """
    return max((item.end_sec for item in timings if item.has_audio), default=0.0)


def timeline_view(
    speakers: list[dict],
    segment: dict,
    timings: list[ReplicaTiming],
    render: RenderSettings,
    duration_sec: float,
) -> dict:
    """Ответ таймлайна: дорожки спикеров и сегменты реплик.

    `segment` — карточка реплики для интерфейса (голос, движок, текст, QA,
    настройки): таймлайн не должен быть вторым источником правды о реплике,
    поэтому реплика отдаётся ровно в той же форме, что и `GET /api/projects/{id}`.
    """
    segments: list[dict] = []
    for timing in timings:
        card = segment(timing.index)
        segments.append(
            {
                **card,
                "start_sec": round(timing.start_sec, ROUND_DIGITS),
                "end_sec": round(timing.end_sec, ROUND_DIGITS),
                "duration_sec": round(timing.duration_sec, ROUND_DIGITS),
                "pause_ms": timing.pause_ms,
                "has_audio": timing.has_audio,
                "take_id": timing.take_id,
                "take_label": timing.take_label,
                "engine": card.get("engine") or timing.engine,
                "status": "rendered" if timing.has_audio else "pending",
            }
        )
    return {
        "duration_sec": round(duration_sec, ROUND_DIGITS),
        "settings": {
            "pause_ms": int(render.pause_ms),
            "cross_fade_duration": float(render.cross_fade_duration),
        },
        "speakers": speakers,
        "replicas": segments,
    }
