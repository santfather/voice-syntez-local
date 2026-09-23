"""Приёмочные тесты по списку падений воркера (§28–§29 пакета надёжности).

Имена тестов совпадают с именами из списка требований, чтобы он проверялся
дословно, а не «похожими» сценариями. Поведение, которое уже покрыто подробными
тестами (`test_worker_isolation.py`, `test_worker_recovery.py`,
`test_render_recovery.py`, `test_recovery.py`, `test_memory_and_shutdown.py`),
здесь проверяется одной-двумя точными проверками: это трассировка требований, а не
второй набор сценариев.

Модели не поднимаются: в воркере работает заглушка, которая падает по маркеру в
тексте (`tests/fake_worker_engine.py`).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from conftest import F5_VOICE, sine

from backend import audio_pipeline, config, job_queue, main, memory_monitor, recovery
from backend.audio_pipeline import AudioFileError, RenderSettings, SpeakerSettings
from backend.db.store import get_projects_store
from backend.dialogue_parser import Replica
from backend.engines import registry
from backend.engines import supervisor as supervisor_module
from backend.engines import worker_protocol as proto
from backend.job_queue import JobPayload, JobQueue, JobStatus

FACTORY = "fake_worker_engine:create"
FIRST = "Первая реплика диалога"
CRASH_TEXT = "Реплика краш процесса"
THIRD = "Третья реплика диалога"
HANG_TEXT = "Реплика висни процесса"

JOB_TIMEOUT_SEC = 60.0


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Изоляция включена, в воркере — лёгкая заглушка."""
    monkeypatch.setenv(config.WORKER_ISOLATION_ENV, "1")
    monkeypatch.setenv(config.WORKER_ENGINE_FACTORY_ENV, FACTORY)
    monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(config, "WORKER_LOAD_TIMEOUT_SEC", 20.0)
    monkeypatch.setattr(config, "WORKER_SHUTDOWN_GRACE_SEC", 3.0)
    fresh = supervisor_module.WorkerSupervisor()
    monkeypatch.setattr(supervisor_module, "_supervisor", fresh)
    monkeypatch.setattr(registry, "_instances", {})
    yield SimpleNamespace(supervisor=fresh, path=tmp_path)
    fresh.begin_shutdown()
    fresh.stop_all(grace=1.0)
    fresh.reset_after_stop()


def _run(scenario):
    return asyncio.run(scenario())


async def _wait_until(predicate, timeout: float = JOB_TIMEOUT_SEC) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def _engine():
    return registry.get_engine("f5")


def _synth(engine, text: str):
    return engine.synthesize(text, "/tmp/reference.wav", "референс")


def _replicas(texts=(FIRST, CRASH_TEXT, THIRD)) -> list[Replica]:
    return [
        Replica(voice=F5_VOICE, text=text, line_number=index + 1)
        for index, text in enumerate(texts)
    ]


def _payload(replicas=None, project_id=None) -> JobPayload:
    replicas = replicas or _replicas()
    return JobPayload(
        replicas=replicas,
        speakers={replica.voice: SpeakerSettings(voice_id=F5_VOICE) for replica in replicas},
        settings=RenderSettings(pause_ms=0),
        project_id=project_id,
    )


def _project(texts=(FIRST, CRASH_TEXT, THIRD)) -> tuple[str, list[Replica]]:
    store = get_projects_store()
    project = store.create_project(
        "Приёмка",
        source_text="".join(f"ИВАН: {text}\n" for text in texts),
        mode="dialogue",
    )
    parsed = store.parse_project(project["id"], None)
    replicas = [
        Replica(voice=row["speaker"], text=row["text"], line_number=row["index"] + 1)
        for row in parsed["replicas"]
    ]
    return project["id"], replicas


async def _render(queue, payload) -> object:
    job = queue.submit(payload)
    await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})
    return job


# --- Phase 2: supervisor / restart --------------------------------------------
def test_worker_is_restarted_after_crash(worker_env, voices):
    engine = _engine()
    _synth(engine, FIRST)
    stale_pid = engine.worker_pid
    with pytest.raises(proto.WorkerCrashError):
        _synth(engine, CRASH_TEXT)
    assert engine.worker_pid is not None
    assert engine.worker_pid != stale_pid
    assert engine.worker.state == proto.WORKER_STATE_IDLE
    engine.unload()


def test_worker_restart_returns_to_ready(worker_env, voices):
    engine = _engine()
    with pytest.raises(proto.WorkerCrashError):
        _synth(engine, CRASH_TEXT)
    wave, sample_rate = _synth(engine, FIRST)
    assert wave.size > 0
    assert sample_rate == audio_pipeline.SAMPLE_RATE
    assert engine.is_loaded
    engine.unload()


def test_worker_restart_loop_is_limited(worker_env, voices, monkeypatch):
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 2)
    engine = _engine()
    for _ in range(2):
        with pytest.raises(proto.WorkerCrashError):
            _synth(engine, CRASH_TEXT)
    assert engine.worker.state == proto.WORKER_STATE_DEGRADED

    # В DEGRADED новые процессы не поднимаются: цикл падений ограничен.
    with pytest.raises(proto.WorkerUnavailableError):
        _synth(engine, FIRST)
    assert engine.worker.process is None
    engine.unload()


# --- Phase 3: job/replica recovery --------------------------------------------
def test_crashed_replica_marked_interrupted(worker_env, voices, monkeypatch):
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project()

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(replicas, project_id))
        finally:
            await queue.stop()

    job = _run(scenario)
    assert job.status is JobStatus.ERROR
    by_index = {row["index"]: row for row in store.get_project(project_id)["replicas"]}
    assert by_index[1]["status"] == config.REPLICA_STATUS_INTERRUPTED
    assert by_index[2]["status"] == config.REPLICA_STATUS_PENDING


def test_completed_replicas_survive_worker_crash(worker_env, voices, monkeypatch):
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project()

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(replicas, project_id))
        finally:
            await queue.stop()

    _run(scenario)
    first = store.get_project(project_id)["replicas"][0]
    assert first["status"] == config.REPLICA_STATUS_RENDERED
    take_path = Path(first["takes"][0]["audio_path"])
    assert take_path.exists()
    # Файл не пустой и читается: «сохранён» значит «звучит», а не «есть на диске».
    assert audio_pipeline.validate_audio_file(take_path)["duration_sec"] > 0


def test_worker_crash_does_not_corrupt_sqlite(worker_env, voices, monkeypatch):
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project()

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(replicas, project_id))
        finally:
            await queue.stop()

    _run(scenario)
    connection = sqlite3.connect(config.DB_PATH)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    # И база отвечает тем же, что до падения: проект, реплики и диагностика.
    assert store.get_project(project_id) is not None
    assert len(store.list_worker_crashes(limit=10)) == 1


def test_interrupted_replica_can_be_retried(worker_env, voices, monkeypatch):
    """Повтор после прерывания: тот же проект доезжает до готовности."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project()

    async def first_attempt():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(replicas, project_id))
        finally:
            await queue.stop()

    assert _run(first_attempt).status is JobStatus.ERROR

    # «Починенный» движок: повтор той же задачи проходит целиком.
    fixed = _replicas((FIRST, "Вторая реплика процесса", THIRD))
    async def second_attempt():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(fixed, project_id))
        finally:
            await queue.stop()

    job = _run(second_attempt)
    assert job.status is JobStatus.DONE, job.error
    by_index = {row["index"]: row for row in store.get_project(project_id)["replicas"]}
    assert by_index[0]["status"] == config.REPLICA_STATUS_RENDERED
    assert by_index[1]["status"] == config.REPLICA_STATUS_RENDERED


# --- Phase 4: atomic audio ----------------------------------------------------
def test_audio_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "take.wav"
    real_write = sf.write

    def broken(path, data, samplerate, **kwargs):
        real_write(path, data, samplerate, **kwargs)
        raise OSError("нет места")

    monkeypatch.setattr(sf, "write", broken)
    with pytest.raises(OSError):
        audio_pipeline._write_audio(target, sine(0.2), "wav")
    assert not target.exists()


def test_partial_audio_is_not_committed(tmp_path):
    target = tmp_path / "empty.wav"
    with pytest.raises(AudioFileError):
        audio_pipeline._write_audio(target, np.zeros(0, dtype=np.float32), "wav")
    assert not target.exists()


def test_done_status_requires_valid_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    path = audio_pipeline._write_output("job-x", sine(0.2), "wav")
    meta = audio_pipeline.validate_audio_file(path)
    assert meta["frames"] > 0

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    with pytest.raises(AudioFileError):
        audio_pipeline.validate_audio_file(broken)


def test_temp_audio_removed_after_crash(workspace):
    leftovers = [workspace / "output" / "job.wav.part", workspace / "output" / "job.mp3.part.wav"]
    for path in leftovers:
        path.write_bytes(b"half")
    assert audio_pipeline.cleanup_partials() == 2
    assert all(not path.exists() for path in leftovers)


# --- Phase 5: startup recovery -------------------------------------------------
def test_backend_startup_marks_stale_render_as_interrupted(workspace):
    store = get_projects_store()
    project = store.create_project("Старт", source_text=f"ИВАН: {FIRST}", mode="dialogue")
    store.parse_project(project["id"], None)
    store.update_project(project["id"], status=config.PROJECT_STATUS_RENDERING, job_id="job-1")
    store.set_replica_status(project["id"], 0, config.REPLICA_STATUS_RENDERING)

    report = recovery.recover_after_restart()

    assert report.projects and report.replicas == 1
    fresh = store.get_project(project["id"])
    assert fresh["status"] == config.PROJECT_STATUS_DRAFT
    assert fresh["replicas"][0]["status"] == config.REPLICA_STATUS_INTERRUPTED


def test_rendering_state_recovers_after_restart(workspace):
    """После перезапуска состояние «идёт» не остаётся ни у проекта, ни у анализа."""
    store = get_projects_store()
    project = store.create_project("Анализ", source_text=f"ИВАН: {FIRST}", mode="dialogue")
    store.parse_project(project["id"], None)
    store.update_project(
        project["id"],
        status=config.PROJECT_STATUS_RENDERING,
        analysis_status=config.PROJECT_ANALYSIS_ANALYZING,
    )

    recovery.recover_after_restart()

    fresh = store.get_project(project["id"])
    assert fresh["status"] == config.PROJECT_STATUS_DRAFT
    assert fresh["analysis_status"] == config.PROJECT_ANALYSIS_RAW
    assert fresh["last_error"] == config.RECOVERY_RENDER_MESSAGE


def test_project_survives_backend_restart(workspace):
    """Проект и его готовые куски читаются после восстановления без изменений."""
    from backend import audio_pipeline as pipeline

    store = get_projects_store()
    project = store.create_project("Живой", source_text=f"ИВАН: {FIRST}", mode="dialogue")
    store.parse_project(project["id"], None)
    take = pipeline.config.PROJECTS_OUTPUT_DIR / project["id"] / "r1.wav"
    take.parent.mkdir(parents=True, exist_ok=True)
    sf.write(take, sine(0.2), audio_pipeline.SAMPLE_RATE)
    store.save_render_takes(
        project["id"], "job-1", [{"index": 0, "audio_path": str(take), "duration_sec": 0.2}]
    )

    recovery.recover_after_restart()

    fresh = store.get_project(project["id"])
    assert fresh["name"] == "Живой"
    assert fresh["replicas"][0]["status"] == config.REPLICA_STATUS_RENDERED
    assert Path(fresh["replicas"][0]["takes"][0]["audio_path"]).exists()


# --- Phase 6: queue recovery ---------------------------------------------------
def test_queue_continues_after_worker_restart(worker_env, voices, monkeypatch):
    """Одна авария не блокирует очередь: следующая задача выполняется."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            crashed = await _render(queue, _payload(_replicas((FIRST, CRASH_TEXT))))
            healthy = await _render(queue, _payload(_replicas((FIRST, THIRD))))
            return crashed, healthy
        finally:
            await queue.stop()

    crashed, healthy = _run(scenario)
    assert crashed.status is JobStatus.ERROR
    assert healthy.status is JobStatus.DONE, healthy.error
    assert len(healthy.segments) == 2


def test_same_crashing_job_is_not_retried_forever(worker_env, voices, monkeypatch):
    """Число попыток ограничено, и падение не превращается в цикл."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 1)
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project()

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            return await _render(queue, _payload(replicas, project_id))
        finally:
            await queue.stop()

    job = _run(scenario)
    assert job.status is JobStatus.ERROR
    # Первая попытка + один повтор: ровно две записи в диагностике, не больше.
    crashes = store.list_worker_crashes(limit=20)
    assert len(crashes) == 2
    assert sorted(crash["attempt"] for crash in crashes) == [1, 2]


# --- Phase 7: memory guard / unload -------------------------------------------
def test_memory_pressure_blocks_new_inference(stub, voices, monkeypatch):
    """В CRITICAL задача не стартует, пока память не освободится."""
    pressure = {"critical": True}
    monkeypatch.setattr(
        job_queue.resource_guard, "is_memory_critical", lambda: pressure["critical"]
    )
    monkeypatch.setattr(
        job_queue.resource_guard,
        "memory_state",
        lambda: {"state": memory_monitor.STATE_CRITICAL, "reason": "память системы занята на 93%"},
    )
    monkeypatch.setattr(job_queue, "MEMORY_WAIT_RETRY_SEC", 0.01)

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(_replicas((FIRST,))))
            await asyncio.sleep(0.3)
            started_under_pressure = job.status is JobStatus.PROCESSING
            pressure["critical"] = False  # память «освободилась»
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            return started_under_pressure, job
        finally:
            await queue.stop()

    started_under_pressure, job = _run(scenario)
    assert started_under_pressure is False, "задача стартовала при критической памяти"
    assert job.status is JobStatus.DONE


def test_memory_pressure_does_not_block_light_api(monkeypatch):
    """Даже в CRITICAL лёгкие роуты отвечают: они не ждут ни модели, ни памяти."""
    monkeypatch.setattr(
        memory_monitor,
        "get_state",
        lambda sampler=None: {
            "state": memory_monitor.STATE_CRITICAL,
            "reason": "процесс синтеза занимает 9000 МБ",
            "rss_mb": 200.0,
            "workers_rss_mb": 9000.0,
            "worker_peak_rss_mb": 9000.0,
            "system_percent": 90.0,
            "workers": [],
            "thresholds": {},
        },
    )

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return (
                await client.get("/api/status"),
                await client.get("/api/diagnostics/worker"),
            )

    status, diagnostics = _run(scenario)
    assert status.status_code == 200
    assert status.json()["memory_state"]["state"] == memory_monitor.STATE_CRITICAL
    assert diagnostics.status_code == 200


def test_unused_engine_can_be_unloaded(worker_env, voices):
    """Простаивающий движок выгружается, следующий синтез поднимает его заново."""
    engine = _engine()
    _synth(engine, FIRST)
    engine.unload()
    assert engine.worker.pid is None
    assert engine.is_loaded is False

    wave, _ = _synth(engine, FIRST)
    assert wave.size > 0
    engine.unload()


# --- Phase 8: shutdown / cancellation -----------------------------------------
def test_graceful_shutdown_stops_worker(worker_env, voices):
    engine = _engine()
    _synth(engine, FIRST)
    assert engine.worker.alive()
    worker_env.supervisor.begin_shutdown()
    assert worker_env.supervisor.stop_all() == 1
    assert engine.worker.pid is None
    assert engine.worker.state == proto.WORKER_STATE_STOPPED


def test_shutdown_does_not_leave_orphan_worker(worker_env, voices):
    engine = _engine()
    _synth(engine, FIRST)
    process = engine.worker.process
    worker_env.supervisor.begin_shutdown()
    worker_env.supervisor.stop_all()
    assert process.poll() is not None, "процесс воркера пережил приложение"


def test_cancelled_worker_job_is_not_reported_as_crash(worker_env, voices, monkeypatch):
    """Отмена пользователем — не падение процесса: тип CANCELLED, воркер жив."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 0)
    store = get_projects_store()
    # Вторая реплика «тормозит» (заглушка спит три секунды): отмена успевает
    # прийти между кусками детерминированно, а не гонкой с мгновенным синтезом.
    texts = (FIRST, "Реплика номер два slow", THIRD, "Реплика номер четыре")

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(_replicas(texts)))
            assert await _wait_until(lambda: job.current_replica >= 1)
            queue.cancel(job.id)
            assert await _wait_until(
                lambda: job.status in {JobStatus.CANCELLED, JobStatus.DONE}, timeout=20.0
            )
            return job
        finally:
            await queue.stop()

    job = _run(scenario)
    assert job.status is JobStatus.CANCELLED
    assert job.error_type == proto.ERROR_CANCELLED
    # Процесс синтеза жив, отказов нет, диагностика не засорена «падением».
    engine = _engine()
    assert engine.worker.alive()
    assert engine.worker.failures == 0
    assert store.list_worker_crashes(limit=5) == []
    engine.unload()
