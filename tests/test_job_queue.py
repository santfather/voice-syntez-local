"""Очередь задач: сборка, перегенерация реплики, история вариантов и сиды."""

import asyncio
import contextlib
import time

import pytest

from backend import audio_pipeline, config
from backend.audio_pipeline import QaSettings, RenderSettings, SpeakerSettings, read_variant
from backend.dialogue_parser import Replica
from backend.job_queue import Job, JobPayload, JobQueue, JobStatus


def _payload(count: int = 3, pause_ms: int = 300, qa: QaSettings | None = None) -> JobPayload:
    return JobPayload(
        replicas=[
            Replica(voice="#1", text=f"Реплика номер {index}", line_number=index)
            for index in range(1, count + 1)
        ],
        speakers={"#1": SpeakerSettings(voice_id="voice1")},
        settings=RenderSettings(pause_ms=pause_ms, output_format="wav", qa=qa),
    )


async def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@contextlib.asynccontextmanager
async def _queue():
    queue = JobQueue()
    await queue.start()
    try:
        yield queue
    finally:
        await queue.stop()


def _run(scenario) -> None:
    asyncio.run(scenario())


def test_render_job_records_segments_and_seeds(stub, fake_store):
    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            assert job.error is None
            assert job.output_path is not None and job.output_path.exists()
            assert job.payload is not None
            assert len(job.segments) == 3
            assert len(job.seeds) == 3
            assert all(isinstance(seed, int) for seed in job.seeds)
            # Сид выбирает пайплайн и отдаёт его движку — иначе вариант не повторить.
            assert [call["params"]["seed"] for call in stub.calls] == job.seeds
            assert queue.active_seed(job, 0) == job.seeds[0]

    _run(scenario)


def test_regenerate_keeps_previous_variant_and_shifts_tail(stub, fake_store):
    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            original_duration = job.duration_sec
            tail_before = job.segments[2]

            queue.submit_regenerate(job.id, 0)
            assert job.regenerating == 0  # флаг ставится сразу, ещё до старта воркера
            assert await _wait_until(
                lambda: job.regenerating is None and len(job.variants.get(0, [])) == 2
            ), job.regen_error
            assert job.regen_error is None

            variants = job.variants[0]
            assert [variant.label for variant in variants] == ["исходный", "вариант 1"]
            assert job.active_variant[0] == "v1"
            # Исходный вариант сохранил сид первой сборки, новый — свой.
            assert variants[0].seed == job.seeds[0]
            assert queue.active_seed(job, 0) == variants[1].seed
            assert len(stub.calls) == 4  # пересинтезирована ровно одна реплика

            delta = read_variant(variants[1]).size - read_variant(variants[0]).size
            assert job.segments[0][1] - job.segments[0][0] == read_variant(variants[1]).size
            assert job.segments[2] == (tail_before[0] + delta, tail_before[1] + delta)
            assert job.duration_sec == pytest.approx(original_duration + delta / 24000, abs=0.01)

    _run(scenario)


def test_select_saved_variant_does_not_synthesize(stub, fake_store):
    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            queue.submit_regenerate(job.id, 0)
            assert await _wait_until(lambda: job.regenerating is None and len(job.variants[0]) == 2)
            calls_after_regen = len(stub.calls)
            regenerated = list(job.segments)

            queue.submit_variant(job.id, 0, "v0")
            assert await _wait_until(lambda: job.active_variant[0] == "v0" and job.regenerating is None)
            # Выбор варианта — это готовое аудио: модели он не касается.
            assert len(stub.calls) == calls_after_regen
            assert job.segments[0][1] - job.segments[0][0] == read_variant(job.variants[0][0]).size
            assert job.segments[0] != regenerated[0]
            with pytest.raises(ValueError, match="уже стоит"):
                queue.submit_variant(job.id, 0, "v0")

    _run(scenario)


def test_variants_are_evicted_by_limit(monkeypatch, stub, fake_store):
    monkeypatch.setattr(config, "MAX_REPLICA_VARIANTS", 2)

    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_payload(count=1))
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            for _ in range(3):
                queue.submit_regenerate(job.id, 0)
                assert await _wait_until(lambda: job.regenerating is None)
                await asyncio.sleep(0)  # дать воркеру закончить запись файла варианта

            assert [variant.id for variant in job.variants[0]] == ["v2", "v3"]
            assert job.active_variant[0] == "v3"
            # Вытесненные варианты удаляются вместе с файлами: держать их не для чего.
            assert not (config.OUTPUT_DIR / f"{job.id}-r1-v0.wav").exists()
            assert not (config.OUTPUT_DIR / f"{job.id}-r1-v1.wav").exists()
            assert (config.OUTPUT_DIR / f"{job.id}-r1-v3.wav").exists()

    _run(scenario)


def test_submit_and_regenerate_guards(stub, fake_store):
    async def scenario():
        queue = JobQueue()
        with pytest.raises(RuntimeError, match="не запущена"):
            queue.submit(_payload())

        async with _queue() as started:
            job = started.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error

            with pytest.raises(ValueError, match="не найдена"):
                started.submit_regenerate("нет-такой", 0)
            with pytest.raises(ValueError, match="Такой реплики"):
                started.submit_regenerate(job.id, 99)
            with pytest.raises(KeyError):
                started.find_variant(job, 0, "v99")

            busy = Job(id="busy", status=JobStatus.DONE, regenerating=0)
            busy.output_path = job.output_path
            busy.payload = job.payload
            started._jobs["busy"] = busy
            with pytest.raises(ValueError, match="уже пересобирается"):
                started.submit_regenerate("busy", 0)

            unfinished = Job(id="unfinished", status=JobStatus.QUEUED)
            started._jobs["unfinished"] = unfinished
            with pytest.raises(ValueError, match="готового файла"):
                started.submit_regenerate("unfinished", 0)

    _run(scenario)


def test_shift_segments_moves_following_chunks():
    segments = [(0, 10), (20, 30), (40, 50)]
    JobQueue._shift_segments(segments, 1, (20, 40))
    assert segments == [(0, 10), (20, 40), (50, 60)]

    untouched = [(0, 10), (20, 30)]
    JobQueue._shift_segments(untouched, 0, (0, 10))  # длина не изменилась — сдвига нет
    assert untouched == [(0, 10), (20, 30)]


def test_active_seed_falls_back_to_original_and_handles_missing_index():
    queue = JobQueue()
    job = Job(id="job", status=JobStatus.DONE, seeds=[11, None])
    assert queue.active_seed(job, 0) == 11
    assert queue.active_seed(job, 1) is None
    assert queue.active_seed(job, 5) is None


def test_qa_outcome_is_stored_and_follows_variants(monkeypatch, stub, fake_store):
    async def fake(chunk):
        return "Реплика номер 1"

    monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", fake)

    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_payload(count=2, qa=QaSettings(wer_threshold=1.0)))
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            # Отметка есть у каждой реплики и относится к её текущему звучанию.
            assert [outcome.status for outcome in job.qa] == [
                audio_pipeline.QA_PASSED,
                audio_pipeline.QA_PASSED,
            ]
            assert queue.active_qa(job, 0) is job.qa[0]

            queue.submit_regenerate(job.id, 0)
            assert await _wait_until(
                lambda: job.regenerating is None and len(job.variants.get(0, [])) == 2
            ), job.regen_error
            # Перегенерация проходит через ту же проверку: новая отметка — у нового
            # варианта, а исходный вариант сохранил ту, с которой был сгенерирован.
            assert queue.active_qa(job, 0) is job.variants[0][1].qa
            assert job.variants[0][0].qa is job.qa[0]

    _run(scenario)
