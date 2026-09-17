"""Прерванный рендер: частичное состояние и продолжение с упавшей реплики.

Проверяется механика, на которой стоит восстановление после падения воркера
(creash_report): пайплайн прикрепляет к исключению готовые куски, а повторный
прогон с `resume` не пересоздаёт их, а продолжает синтез с той реплики, на
которой прервалось. Модель не поднимается — движок подменяется заглушкой,
которая падает по маркеру в тексте.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from conftest import (  # noqa: F401 — sine нужен фикстуре voices
    F5_VOICE,
    StubEngine,
    sine,
)

from backend import audio_pipeline
from backend.audio_pipeline import RenderPartial, RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines import worker_protocol as proto

MARKER = "падение"
FIRST = "Первая реплика диалога"
SECOND = f"Вторая реплика: {MARKER} процесса"
THIRD = "Третья реплика диалога"


class FailingEngine(StubEngine):
    """Заглушка, падающая на реплике с маркером — как упавший процесс воркера."""

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        if MARKER in text.lower():
            raise proto.WorkerCrashError(
                "процесс воркера завершился во время синтеза",
                engine="stub",
                details={"exit_code": -6, "signal": 6, "signal_name": "SIGABRT"},
            )
        return super()._synthesize(text, ref_audio_path, ref_text, speed, params)


def _replicas(texts: tuple[str, ...] = (FIRST, SECOND, THIRD)) -> list[Replica]:
    return [
        Replica(voice=F5_VOICE, text=text, line_number=index + 1)
        for index, text in enumerate(texts)
    ]


def _speakers() -> dict[str, SpeakerSettings]:
    return {F5_VOICE: SpeakerSettings(voice_id=F5_VOICE)}


def _render(replicas, monkeypatch, settings=None, resume=None, job_id="job-1"):
    return asyncio.run(
        audio_pipeline.render_dialogue(
            job_id=job_id,
            replicas=replicas,
            speakers=_speakers(),
            settings=settings or RenderSettings(),
            resume=resume,
        )
    )


def _interrupt(monkeypatch, replicas=None, settings=None) -> RenderPartial:
    """Первый прогон с падающим движком — до состояния «прервано»."""
    failing = FailingEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: failing)
    with pytest.raises(proto.WorkerCrashError) as caught:
        _render(replicas or _replicas(), monkeypatch, settings)
    return caught.value.partial_render


def test_partial_state_is_attached_to_error(voices, monkeypatch):
    """Падение на второй реплике: в исключении — первый готовый кусок."""
    failing = FailingEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: failing)

    with pytest.raises(proto.WorkerCrashError) as caught:
        _render(_replicas(), monkeypatch)

    partial = caught.value.partial_render
    assert isinstance(partial, RenderPartial)
    assert partial.done == 1
    assert partial.failed_index == 1
    assert len(partial.segments) == 1
    assert len(partial.seeds) == 1
    assert len(partial.qualities) == 1
    assert partial.pieces[0].size == partial.segments[0][1] - partial.segments[0][0]
    assert partial.error.startswith("процесс воркера завершился")
    # Первая реплика синтезирована ровно один раз: частичное состояние — это
    # сделанная работа, а не список «что пробовали».
    assert [call["text"] for call in failing.calls] == [FIRST]


def test_resume_continues_from_failed_replica(voices, monkeypatch):
    """Продолжение синтезирует только неготовые реплики и склеивает всё вместе."""
    partial = _interrupt(monkeypatch)

    # «Замена воркера»: тот же движок, но уже без сбоя.
    fixed = StubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: fixed)
    result = _render(_replicas(), monkeypatch, resume=partial, job_id="job-2")

    assert result.replicas_done == 3
    assert len(result.segments) == 3
    assert len(result.seeds) == 3
    # Синтезированы только вторая и третья реплики: первая взята из продолжения.
    assert [call["text"] for call in fixed.calls] == [SECOND, THIRD]
    # Первый кусок в файле — тот же самый сигнал, что был готов до падения:
    # продолжение не пересоздаёт аудио, готовый кусок встаёт в сборку как есть.
    # Побитового совпадения тут быть не может: громкость нормализуется по всему
    # треку (`_finalize_track`), поэтому сравнивается форма сигнала, а не уровень.
    audio = audio_pipeline._read_audio(result.output_path, "wav")
    start, end = result.segments[0]
    expected = np.asarray(partial.pieces[0], dtype=np.float32)
    assert end - start == expected.size
    segment = audio[start:end]
    denominator = float(np.sqrt(np.mean(segment**2)) * np.sqrt(np.mean(expected**2)))
    correlation = float(np.mean(segment * expected) / denominator) if denominator else 1.0
    assert correlation > 0.999, "готовый кусок в сборке не совпал с сохранённым"
    # Итог длиннее готового префикса: остальные реплики действительно добавлены.
    assert result.duration_sec > expected.size / audio_pipeline.SAMPLE_RATE


def test_resume_keeps_pause_between_ready_and_new_piece(voices, monkeypatch):
    """Пауза между готовой и новой репликой сохраняется — склейка не «слипается»."""
    settings = RenderSettings(pause_ms=500)
    partial = _interrupt(monkeypatch, replicas=_replicas((FIRST, SECOND)), settings=settings)

    fixed = StubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: fixed)
    result = _render(
        _replicas((FIRST, THIRD)), monkeypatch, settings=settings, resume=partial, job_id="job-3"
    )
    gap = result.segments[1][0] - result.segments[0][1]
    assert gap == int(audio_pipeline.SAMPLE_RATE * 0.5)


def test_resume_rejects_more_pieces_than_replicas(voices, monkeypatch):
    """Продолжение с чужим набором реплик — ошибка, а не файл не по тексту."""
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: StubEngine())
    too_many = RenderPartial(
        pieces=[np.zeros(10, dtype=np.float32)] * 5,
        segments=[(0, 10)] * 5,
        seeds=[1] * 5,
        qa=[None] * 5,
        qualities=[],
        failed_index=0,
        error="тест",
    )
    with pytest.raises(ValueError, match="Продолжение рендера противоречит диалогу"):
        _render(_replicas(), monkeypatch, resume=too_many)


def test_crash_on_first_replica_leaves_empty_prefix(voices, monkeypatch):
    """Падение на первой реплике: продолжать нечего, но состояние всё равно есть."""
    partial = _interrupt(monkeypatch, replicas=_replicas((SECOND, THIRD)))
    assert partial.done == 0
    assert partial.failed_index == 0
    assert partial.pieces == []
    assert partial.segments == []


def test_cancellation_also_keeps_ready_pieces(voices, monkeypatch):
    """Отмена между репликами: готовое тоже не теряется (тем же механизмом)."""
    engine = StubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)
    calls = {"count": 0}

    def should_abort() -> bool:
        calls["count"] += 1
        return calls["count"] > 1  # первая реплика синтезируется, вторая — нет

    with pytest.raises(audio_pipeline.JobCancelledError) as caught:
        asyncio.run(
            audio_pipeline.render_dialogue(
                job_id="job-4",
                replicas=_replicas(),
                speakers=_speakers(),
                settings=RenderSettings(),
                should_abort=should_abort,
            )
        )
    partial = caught.value.partial_render
    assert partial.done == 1
    assert partial.failed_index == 1
