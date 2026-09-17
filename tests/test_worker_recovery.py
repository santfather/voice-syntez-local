"""Восстановление после падения воркера на уровне очереди (creash_report §12–§14).

Проверяется то, что видит пользователь: задача продолжается с упавшей реплики,
готовые реплики не пересоздаются и не теряются, прерванная помечается
`interrupted`, а причина падения с кодом и сигналом остаётся в диагностике.

Модель не поднимается: воркер запускается с лёгкой фабрикой-заглушкой, которая
падает нативно по маркеру в тексте (`tests/fake_worker_engine.py`).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import F5_VOICE, sine  # noqa: F401 — sine нужен фикстуре voices

from backend import config
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.db.store import get_projects_store
from backend.dialogue_parser import Replica
from backend.engines import registry
from backend.engines import supervisor as supervisor_module
from backend.engines import worker_protocol as proto
from backend.engines.base import ENGINE_F5
from backend.job_queue import PARTIAL_TAKE_LABEL, JobPayload, JobQueue, JobStatus

FACTORY = "fake_worker_engine:create"
CRASH_TEXT = "Вторая реплика краш процесса"
FIRST = "Первая реплика диалога"
THIRD = "Третья реплика диалога"

# Насколько долго ждать завершения задачи: воркер стартует заново и грузит
# «модель»-заглушку, но настоящих весов здесь нет.
JOB_TIMEOUT_SEC = 60.0


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Изоляция включена, движок в воркере — лёгкая заглушка с журналом вызовов."""
    monkeypatch.setenv(config.WORKER_ISOLATION_ENV, "1")
    monkeypatch.setenv(config.WORKER_ENGINE_FACTORY_ENV, FACTORY)
    monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(config, "WORKER_LOAD_TIMEOUT_SEC", 20.0)
    monkeypatch.setattr(config, "WORKER_SHUTDOWN_GRACE_SEC", 3.0)
    fresh = supervisor_module.WorkerSupervisor()
    monkeypatch.setattr(supervisor_module, "_supervisor", fresh)
    # Реестр кеширует прокси: без сброса тест мог бы получить движок, подписанный
    # на супервизор предыдущего теста.
    monkeypatch.setattr(registry, "_instances", {})
    call_log = tmp_path / "calls.log"
    monkeypatch.setenv("TTS_FAKE_CALL_LOG", str(call_log))
    yield SimpleNamespace(supervisor=fresh, call_log=call_log, path=tmp_path)
    fresh.stop_all(grace=1.0)
    fresh.reset_after_stop()


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _wait_until(predicate, timeout: float = JOB_TIMEOUT_SEC) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def _payload(replicas: list[Replica], project_id: str | None = None) -> JobPayload:
    """Задача рендера: голос назначается каждому ключу спикера этих реплик."""
    return JobPayload(
        replicas=replicas,
        speakers={
            replica.voice: SpeakerSettings(voice_id=F5_VOICE) for replica in replicas
        },
        settings=RenderSettings(pause_ms=0),
        project_id=project_id,
    )


def _replicas(texts: tuple[str, ...] = (FIRST, CRASH_TEXT, THIRD)) -> list[Replica]:
    return [
        Replica(voice=F5_VOICE, text=text, line_number=index + 1)
        for index, text in enumerate(texts)
    ]


def _plain(text: str) -> str:
    """Текст без знаков ударения: до движка он доходит уже с разметкой RUAccent."""
    return (text or "").replace("+", "")


def _calls(worker_env) -> list[str]:
    """Тексты, ушедшие в модель (без разметки ударений), в порядке вызовов."""
    if not worker_env.call_log.exists():
        return []
    return [
        _plain(line.split("\t", 1)[1])
        for line in worker_env.call_log.read_text(encoding="utf-8").splitlines()
        if "\t" in line
    ]


def _project_with_replicas() -> tuple[str, list[Replica]]:
    """Проект с разобранными репликами — без анализа: он здесь не проверяется."""
    store = get_projects_store()
    project = store.create_project(
        "Тест восстановления",
        source_text=f"ИВАН: {FIRST}\nИВАН: {CRASH_TEXT}\nИВАН: {THIRD}",
        mode="dialogue",
    )
    parsed = store.parse_project(project["id"], None)
    replicas = [
        Replica(voice=row["speaker"], text=row["text"], line_number=row["index"] + 1)
        for row in parsed["replicas"]
    ]
    return project["id"], replicas


# --- продолжение после разового падения ---------------------------------------
def test_render_resumes_after_worker_crash(worker_env, voices, monkeypatch):
    """Разовое падение воркера: задача доезжает до конца, начиная с упавшей реплики."""
    monkeypatch.setenv("TTS_FAKE_CRASH_ONCE", str(worker_env.path / "crash.once"))

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(_replicas()))
            assert await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR})
            assert job.status is JobStatus.DONE, job.error
            assert job.error is None
            assert len(job.segments) == 3
            assert job.duration_sec and job.duration_sec > 0
            # О падении не забыли: тип отказа и диагностика остались, хотя задача
            # и завершилась успешно — иначе нестабильность движка была бы не видна.
            assert job.error_type == proto.ERROR_WORKER_CRASH
            assert job.message.startswith("Готово")
        finally:
            await queue.stop()

    _run(scenario)

    calls = _calls(worker_env)
    assert sum(text == FIRST for text in calls) == 1, "готовая реплика пересоздана"
    assert sum(text == CRASH_TEXT for text in calls) == 2, "упавшая реплика не повторена"
    assert sum(text == THIRD for text in calls) == 1
    handle = worker_env.supervisor.handle(ENGINE_F5)
    assert handle.failures == 1
    assert handle.restarts >= 1
    assert handle.last_failure["signal_name"] == "SIGABRT"


def test_crash_on_first_replica_restarts_render(worker_env, voices, monkeypatch):
    """Падение на первой реплике: повторяем сначала — продолжать нечего."""
    monkeypatch.setenv("TTS_FAKE_CRASH_ONCE", str(worker_env.path / "crash.once"))

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(_replicas((CRASH_TEXT, FIRST))))
            assert await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR})
            assert job.status is JobStatus.DONE, job.error
            assert len(job.segments) == 2
        finally:
            await queue.stop()

    _run(scenario)
    calls = _calls(worker_env)
    # Падение на первой реплике: первая попытка оборвалась сразу, повтор прошёл
    # весь диалог целиком.
    assert calls == [CRASH_TEXT, CRASH_TEXT, FIRST]


# --- исчерпание попыток: диагностика и сохранённое готовое ----------------------
def test_giving_up_records_crash_and_marks_replica_interrupted(worker_env, voices, monkeypatch):
    """Систематическое падение: задача — ошибка, готовое сохранено, реплика помечена."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 1)
    # Порог DEGRADED поднят: здесь проверяется исчерпание попыток, а не защита от
    # цикла падений (у неё отдельный тест).
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 10)
    store = get_projects_store()
    project_id, replicas = _project_with_replicas()

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(replicas, project_id=project_id))
            assert await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR})
            assert job.status is JobStatus.ERROR
            assert job.error_type == proto.ERROR_WORKER_CRASH
            assert "SIGABRT" in (job.error or "")
        finally:
            await queue.stop()

    _run(scenario)

    project = store.get_project(project_id)
    assert project["status"] == config.PROJECT_STATUS_ERROR
    assert "SIGABRT" in (project["last_error"] or "")
    by_index = {row["index"]: row for row in project["replicas"]}
    # Первая реплика синтезирована до падения — её кусок сохранён вариантом и
    # стал текущим звучанием; вторая прервана и помечена именно так.
    assert by_index[0]["status"] == config.REPLICA_STATUS_RENDERED
    takes = by_index[0]["takes"]
    assert len(takes) == 1
    assert PARTIAL_TAKE_LABEL in takes[0]["label"]
    assert Path(takes[0]["audio_path"]).exists()
    assert by_index[1]["status"] == config.REPLICA_STATUS_INTERRUPTED
    assert by_index[1]["takes"] == []
    assert by_index[2]["status"] == config.REPLICA_STATUS_PENDING

    # Диагностика: по записи на попытку, с кодом возврата, сигналом и репликой.
    crashes = store.list_worker_crashes(limit=10)
    assert len(crashes) == 2
    for crash in crashes:
        assert crash["job_id"] != ""
        assert crash["project_id"] == project_id
        assert crash["replica_index"] == 1
        assert crash["engine"] == ENGINE_F5
        assert crash["error_type"] == proto.ERROR_WORKER_CRASH
        assert crash["exit_code"] == -6
        assert crash["signal"] == 6
        assert crash["signal_name"] == "SIGABRT"
    assert [crash["attempt"] for crash in crashes] == [2, 1]
    assert crashes[0]["retry_count"] == 1


def test_engine_error_is_not_retried_and_keeps_project_consistent(
    worker_env, voices, monkeypatch
):
    """Ошибка модели — не падение процесса: повтора нет, тип ошибки другой."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 3)
    store = get_projects_store()
    project_id, _ = _project_with_replicas()
    replicas = _replicas((FIRST, "Третья реплика бум процесса", THIRD))

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(replicas, project_id=project_id))
            assert await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR})
            assert job.status is JobStatus.ERROR
            assert job.error_type == proto.ERROR_TTS
            assert "бум" in (job.error or "") or "RuntimeError" in (job.error or "")
        finally:
            await queue.stop()

    _run(scenario)
    # Повтора не было: движок жив, и «повторять то же самое» смысла не имеет.
    assert len(_calls(worker_env)) == 2
    project = store.get_project(project_id)
    by_index = {row["index"]: row for row in project["replicas"]}
    assert by_index[0]["status"] == config.REPLICA_STATUS_RENDERED
    # Реплика, на которой прервалась ошибка модели, тоже помечена: статус
    # «rendering» навсегда выглядел бы как «синтез идёт».
    assert by_index[1]["status"] == config.REPLICA_STATUS_INTERRUPTED
    # Диагностика падений пуста: процесс не падал.
    assert store.list_worker_crashes(limit=10) == []


def test_status_exposes_worker_and_diagnostics(worker_env, voices):
    """Расширенный health: состояние воркера и последние падения — на одном роуте."""
    from backend import main

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(_replicas((FIRST, CRASH_TEXT, THIRD))))
            await _wait_until(lambda: job.status in {JobStatus.DONE, JobStatus.ERROR})
        finally:
            await queue.stop()
        return await main.worker_diagnostics()

    payload = asyncio.run(scenario())
    assert payload["isolation"] is True
    assert payload["workers"][0]["engine"] == ENGINE_F5
    assert payload["crashes"], "падение обязано остаться в диагностике"
    assert payload["crashes"][0]["error_type"] == proto.ERROR_WORKER_CRASH
