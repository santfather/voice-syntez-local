"""Трейс синтеза: где теряется начало и конец короткой реплики (UPDATE 2 §13–§15).

Тесты проверяют сам инструмент, а не качество звука: что он включается только по
переменной окружения, что пишет срез аудио на каждой стадии, что числа обрезки
соответствуют реально снятым сэмплам и что выключенный трейс не меняет ни одного
байта готового файла. Настоящий разбор «модель не сказала / пайплайн срезал»
делается живым прогоном (`tools/short_trace.py`), а здесь фиксируется контракт
инструмента — иначе первый же прогон на моделях пришлось бы отлаживать вслепую.
"""

import asyncio
import json

import numpy as np
import pytest
import soundfile as sf
from conftest import sine

from backend import audio_pipeline, synthesis_trace
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines.base import SAMPLE_RATE

TEXT = "Красивая."


@pytest.fixture
def trace_env(monkeypatch, workspace):
    """Трейс включён и пишет в tmp_path — рабочие каталоги тест не трогает."""
    root = workspace / "trace"
    monkeypatch.setenv(synthesis_trace.TRACE_ENV, "1")
    monkeypatch.setenv(synthesis_trace.TRACE_DIR_ENV, str(root))
    return root


def _speaker(voice) -> SpeakerSettings:
    return SpeakerSettings.from_dict({"voice_id": voice.id})


def _render(job_id: str, text: str = TEXT):
    """Один рендер реплики через настоящий пайплайн с движком-заглушкой."""
    return asyncio.run(
        audio_pipeline.render_dialogue(
            job_id,
            [Replica(voice="ИВАН", text=text, line_number=1)],
            {"ИВАН": _speaker(_voice_holder["voice"])},
            RenderSettings(output_format="wav"),
        )
    )


# Голос, которым пользуются тесты. Фикстура `fake_store` создаёт его в tmp_path и
# подменяет хранилище пайплайна; здесь он просто доступен по имени.
_voice_holder: dict = {}


@pytest.fixture(autouse=True)
def remember_voice(fake_store, stub, voice):
    _voice_holder["voice"] = voice
    yield
    _voice_holder.clear()


def test_trace_is_off_by_default(monkeypatch, workspace):
    """Без переменной окружения трейс не создаётся и ничего не пишет."""
    monkeypatch.delenv(synthesis_trace.TRACE_ENV, raising=False)
    assert synthesis_trace.enabled() is False
    assert synthesis_trace.start("job-off") is None
    assert not synthes_trace_dir(workspace).exists()


def synthes_trace_dir(workspace):
    return workspace / "output" / "trace"


def test_trace_records_stages_and_numbers(trace_env):
    """Включённый трейс пишет стадии, файлы стадий и числа обрезки."""
    _render("job-trace")

    root = trace_env / "job-trace"
    records = [
        json.loads(line)
        for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["source_text"] == TEXT
    assert record["engine"] == "stub"
    assert record["text_chunks"] == 1
    assert record["chunk_boundaries"] == [0, len(TEXT)]
    assert record["reference_audio"] == "ref.wav"
    # Стадии: сырой выход модели, обрезка контекста (не применялась) и финал.
    stages = record["stages"]
    assert stages[synthesis_trace.STAGE_RAW]["applied"] is True
    assert stages[synthesis_trace.STAGE_CONTEXT]["applied"] is False
    assert stages[synthesis_trace.STAGE_TRIM]["applied"] is True
    assert stages[synthesis_trace.STAGE_FINAL]["applied"] is True
    # Числа обрезки согласованы с размерами стадий.
    trim = stages[synthesis_trace.STAGE_TRIM]
    assert stages[synthesis_trace.STAGE_RAW]["samples"] - trim["dropped_start"] - trim[
        "dropped_end"
    ] == trim["samples"]
    # Файлы стадий на месте и читаются как аудио нужной длины.
    for stage in (
        synthesis_trace.STAGE_RAW,
        synthesis_trace.STAGE_TRIM,
        synthesis_trace.STAGE_FINAL,
    ):
        path = root / f"r001/{stage}.wav"
        assert path.is_file(), stage
        data, rate = sf.read(path)
        assert rate == SAMPLE_RATE
        assert data.size == stages[stage]["samples"]
    # Сводка прогона: по ней разбор начинается.
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["replicas"] == 1
    assert summary["trimmed"] == 1
    assert summary["job_id"] == "job-trace"


def test_trace_does_not_change_the_audio(trace_env, voice):
    """Трейс — наблюдатель: подготовка куска с ним и без него даёт тот же звук.

    Сравниваются именно результаты `_prepare_chunk` на одном и том же входе:
    два полных рендера сравнивать нельзя — заглушка намеренно делает каждый вызов
    чуть длиннее предыдущего, и различие в длине было бы её свойством, а не
    эффектом трейса.
    """
    chunk = sine(0.4, amplitude=0.35)
    settings = SpeakerSettings.from_dict({"voice_id": voice.id})
    clean = audio_pipeline._prepare_chunk(chunk, settings)

    trace = synthesis_trace.start("job-observer")
    record = trace.replica(1, source_text="Красивая.")
    observed = audio_pipeline._prepare_chunk(chunk, settings, trace=record)

    assert np.array_equal(clean, observed)
    # Файл финальной стадии — ровно то, что вернула подготовка.
    written, rate = sf.read(trace_env / "job-observer" / "r001" / "04_final_replica.wav")
    assert rate == SAMPLE_RATE
    assert np.allclose(written, clean, atol=1 / 32768)  # PCM_16 на записи


def test_trace_prunes_old_runs(trace_env):
    """Старые прогоны вытесняются: трейс — инструмент разбора, а не склад."""
    for index in range(synthesis_trace.KEEP_RUNS + 3):
        _render(f"job-{index:02d}")
    runs = sorted(path.name for path in trace_env.iterdir() if path.is_dir())
    assert len(runs) == synthesis_trace.KEEP_RUNS
    # Живыми остаются последние прогоны, а не первые.
    assert "job-00" not in runs


def test_edge_bounds_reports_what_is_cut():
    """Обрезка краёв возвращает границы, а трейс считает по ним снятые сэмплы."""
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    speech = sine(0.3, amplitude=0.4)
    chunk = np.concatenate((silence, speech, silence))
    bounds = audio_pipeline._edge_bounds(chunk)
    assert bounds is not None
    start, end = bounds
    assert start > 0 and end < chunk.size
    # Запас с обеих сторон: речь не обрезается «по порогу», а получает поля.
    assert start <= int(SAMPLE_RATE * 0.5)
    assert end >= int(SAMPLE_RATE * 0.8)
    # Короткий кусок не режется вовсе — это защита от потери односложной фразы.
    assert audio_pipeline._edge_bounds(sine(0.05)) is None


def test_trace_record_carries_emotion_and_reference_fields(trace_env):
    """Поля эмоции и референса есть в записи уже сейчас — их заполнит слой эмоций."""
    trace = synthesis_trace.start("job-fields")
    record = trace.replica(
        1,
        source_text="Да.",
        emotion_detected="QUESTION",
        emotion_effective="QUESTION",
        reference_profile_id="profile-1",
    )
    payload = record.to_dict()
    assert payload["emotion_detected"] == "QUESTION"
    assert payload["emotion_effective"] == "QUESTION"
    assert payload["reference_profile_id"] == "profile-1"
    assert payload["reference_fallback_used"] is False
    # Служебная ссылка на прогон в JSON не попадает.
    assert "_trace" not in payload
