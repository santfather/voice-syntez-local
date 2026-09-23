"""Память (NORMAL/WARNING/CRITICAL), лёгкий API и безопасное завершение.

Проверяется поведение под давлением памяти и при выключении: очередь не начинает
новую тяжёлую работу в критическом состоянии и сначала пробует вернуть память
выгрузкой простаивающих движков; лёгкие роуты отвечают, даже когда воркер мёртв;
остановка сервера не оставляет ни осиротевших процессов, ни «вечного» статуса
реплики и сохраняет уже готовые куски.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from conftest import (  # noqa: F401 — sine нужен фикстуре voices
    F5_VOICE,
    StubEngine,
    sine,
)

from backend import config, job_queue, main, memory_monitor, resource_guard
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.db.store import get_projects_store
from backend.dialogue_parser import Replica
from backend.engines import registry
from backend.engines import supervisor as supervisor_module
from backend.engines import worker_protocol as proto
from backend.engines.base import STATE_READY
from backend.job_queue import JobPayload, JobQueue, JobStatus

FACTORY = "fake_worker_engine:create"
CRASH_TEXT = "Реплика краш процесса"
HANG_TEXT = "Реплика висни процесса"
FIRST = "Первая реплика диалога"


# --- классификация памяти ------------------------------------------------------
def test_classify_reports_normal_warning_and_critical():
    """Три состояния различаются по числам, а не по «на глаз»."""
    normal = memory_monitor.classify(rss_mb=100, workers_rss_mb=100, system_percent=40)
    assert normal[0] == memory_monitor.STATE_NORMAL

    warning = memory_monitor.classify(rss_mb=100, workers_rss_mb=100, system_percent=82)
    assert warning[0] == memory_monitor.STATE_WARNING
    assert "близко к порогу" in warning[1]

    system_critical = memory_monitor.classify(rss_mb=100, workers_rss_mb=100, system_percent=91)
    assert system_critical[0] == memory_monitor.STATE_CRITICAL
    assert "память системы занята" in system_critical[1]


def test_classify_counts_backend_and_worker_limits():
    """Свой потолок и потолок воркеров проверяются отдельно от системного."""
    backend = memory_monitor.classify(
        rss_mb=6000, workers_rss_mb=100, system_percent=40, max_rss_mb=5000
    )
    assert backend[0] == memory_monitor.STATE_CRITICAL
    assert "бэкенд занимает" in backend[1]

    workers = memory_monitor.classify(
        rss_mb=100, workers_rss_mb=6000, system_percent=40, worker_max_rss_mb=5000
    )
    assert workers[0] == memory_monitor.STATE_CRITICAL
    # Модели живут в воркерах: без этого слагаемого watchdog их не увидел бы.
    assert "процессы синтеза занимают" in workers[1]

    warning = memory_monitor.classify(
        rss_mb=100, workers_rss_mb=4500, system_percent=40, worker_max_rss_mb=5000
    )
    assert warning[0] == memory_monitor.STATE_WARNING


def test_get_state_uses_injected_sampler():
    """Сбор чисел инъектируется: тест не зависит от того, что ещё запущено рядом."""
    state = memory_monitor.get_state(
        lambda: {"rss_mb": 120.5, "workers_rss_mb": 2600.0, "system_percent": 55.0}
    )
    assert state["state"] == memory_monitor.STATE_NORMAL
    assert state["workers_rss_mb"] == 2600.0
    assert state["thresholds"]["worker_max_rss_mb"] == config.WORKER_MAX_RSS_MB


def test_broken_sampler_does_not_block_work():
    """Сломанная метрика не останавливает синтез: нет данных — значит не мешаем."""

    def broken():
        raise RuntimeError("psutil недоступен")

    state = memory_monitor.get_state(broken)
    assert state["state"] == memory_monitor.STATE_NORMAL
    assert state["rss_mb"] == 0.0


def test_default_sampler_sums_worker_memory(monkeypatch):
    """Сводка по умолчанию включает RSS процессов синтеза (там живут модели)."""
    import psutil

    supervisor = supervisor_module.WorkerSupervisor()
    handle = supervisor.handle("f5")
    handle.process = SimpleNamespace(pid=4242, poll=lambda: None)
    monkeypatch.setattr(supervisor_module, "_supervisor", supervisor)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=300 * 1024 * 1024)
        ),
    )

    state = memory_monitor.sample()
    assert state["workers_rss_mb"] == pytest.approx(300.0, abs=1.0)
    assert state["workers"][0]["pid"] == 4242
    # Фейковый процесс не умеет `wait`/`terminate`: убираем его до уборки теста,
    # иначе супервизор попытается остановить несуществующий процесс.
    handle.process = None
    handle.state = proto.WORKER_STATE_STOPPED
    supervisor.reset_after_stop()


def test_memory_state_delegates_to_monitor(monkeypatch):
    """`/api/status` и очередь видят одно и то же состояние."""
    monkeypatch.setattr(
        memory_monitor,
        "get_state",
        lambda sampler=None: {"state": memory_monitor.STATE_WARNING, "reason": "тест"},
    )
    state = resource_guard.memory_state()
    assert state["state"] == memory_monitor.STATE_WARNING
    assert resource_guard.is_memory_critical() is False


def test_full_status_payload_has_memory_state(monkeypatch):
    """Расширенный health: состояние памяти в `/api/status` с причиной."""
    monkeypatch.setattr(
        resource_guard,
        "memory_state",
        lambda: {"state": memory_monitor.STATE_NORMAL, "reason": "памяти достаточно"},
    )
    payload = asyncio.run(main.status())
    assert payload["memory_state"]["reason"] == "памяти достаточно"
    assert "workers" in payload and "worker_isolation" in payload


def test_mps_oom_is_recognised_by_text_and_numbers_are_parsed():
    """Нехватка памяти Metal опознаётся по тексту — и типом из воркера тоже (§11.4)."""
    real = RuntimeError(
        "MPS backend out of memory (MPS allocated: 5.43 GB, other allocations: 1.34 GB, "
        "max allowed: 8.00 GB). Tried to allocate 1.00 GB on private pool."
    )
    assert resource_guard.is_mps_oom(real) is True
    assert resource_guard.mps_oom_numbers(real) == {
        "allocated": "5.43 GB",
        "other": "1.34 GB",
        "max_allowed": "8.00 GB",
    }

    # В изолированном режиме тот же текст приходит чужим классом: распознавание
    # обязано работать по тексту, а не по типу исключения.
    foreign = proto.TtsEngineError(str(real), engine="f5")
    assert resource_guard.is_mps_oom(foreign) is True
    assert resource_guard.mps_oom_numbers(foreign)["max_allowed"] == "8.00 GB"


def test_non_mps_failures_are_not_treated_as_oom():
    """Обычный отказ движка не повод выгружать тёплые движки и повторять реплику."""
    assert resource_guard.is_mps_oom(RuntimeError("CUDA out of memory")) is False
    assert resource_guard.is_mps_oom(RuntimeError("движок вернул пустой ответ")) is False
    # Формулировку задаёт torch: если числа не разобрались, это не ошибка —
    # вызывающий просто пишет текст исключения как есть.
    assert resource_guard.mps_oom_numbers(RuntimeError("MPS out of memory")) is None


# --- очередь и память ----------------------------------------------------------
def _payload(replicas=None, project_id=None) -> JobPayload:
    replicas = replicas or [Replica(voice=F5_VOICE, text=FIRST, line_number=1)]
    return JobPayload(
        replicas=replicas,
        speakers={replica.voice: SpeakerSettings(voice_id=F5_VOICE) for replica in replicas},
        settings=RenderSettings(pause_ms=0),
        project_id=project_id,
    )


async def _wait_until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


def test_queue_waits_in_critical_memory_and_explains_why(stub, voices, monkeypatch, caplog):
    """В критическом состоянии задача ждёт с причиной, а потом выполняется."""
    calls = {"count": 0}

    def critical() -> bool:
        calls["count"] += 1
        return calls["count"] <= 2  # дважды «критично», затем память в порядке

    monkeypatch.setattr(job_queue.resource_guard, "is_memory_critical", critical)
    monkeypatch.setattr(
        job_queue.resource_guard,
        "memory_state",
        lambda: {"state": memory_monitor.STATE_CRITICAL, "reason": "память системы занята на 91%"},
    )
    monkeypatch.setattr(job_queue, "MEMORY_WAIT_RETRY_SEC", 0.01)

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            return job
        finally:
            await queue.stop()

    with caplog.at_level("WARNING"):
        job = asyncio.run(scenario())
    assert calls["count"] >= 3
    assert job.error is None
    # Причина ожидания названа, а не просто «подождите».
    assert any("память системы занята на 91%" in record.getMessage()
               for record in caplog.records)


def test_memory_wait_gives_up_instead_of_waiting_forever(stub, voices, monkeypatch):
    """F-Q3. Бюджет ожидания исчерпан — задача завершается ошибкой, а не ждёт вечно.

    Память, которая не освободилась, оставляла задачу «ожидающей» навсегда:
    очередь стояла, а пользователь видел только «ожидание».
    """
    monkeypatch.setattr(job_queue.resource_guard, "is_memory_critical", lambda: True)
    monkeypatch.setattr(
        job_queue.resource_guard,
        "memory_state",
        lambda: {"state": memory_monitor.STATE_CRITICAL, "reason": "память системы занята на 93%"},
    )
    monkeypatch.setattr(job_queue, "MEMORY_WAIT_RETRY_SEC", 0.01)
    monkeypatch.setattr(job_queue, "MEMORY_WAIT_LIMIT_SEC", 0.05)

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.ERROR), job.status
            return job
        finally:
            await queue.stop()

    job = asyncio.run(scenario())
    assert job.error_type == proto.ERROR_WATCHDOG
    assert "Не дождалась памяти" in (job.error or "")
    assert job.message == "Прервано: нет памяти"
    # До движка дело не дошло: задача так и не началась.
    assert stub.calls == []


def test_memory_wait_wakes_up_on_cancel(monkeypatch):
    """F-Q3. Отмена заканчивает ожидание памяти сразу, а не по концу паузы.

    Пауза между проверками — секунды, и без пробуждения по отмене очередь
    отвечала бы на неё только по её истечении.
    """
    monkeypatch.setattr(job_queue.resource_guard, "is_memory_critical", lambda: True)
    monkeypatch.setattr(job_queue, "MEMORY_WAIT_RETRY_SEC", 30.0)

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload())
            waiter = asyncio.create_task(queue._wait_for_memory(job.id))
            await asyncio.sleep(0.1)  # пауза ожидания уже началась (30 с)
            started = time.monotonic()
            queue.cancel(job.id)
            assert await asyncio.wait_for(waiter, timeout=5.0) is True
            return time.monotonic() - started
        finally:
            await queue.stop()

    assert asyncio.run(scenario()) < 5.0, "ожидание не прервалось отменой"


def test_critical_memory_unloads_idle_engines_first(stub, monkeypatch):
    """Перед ожиданием очередь возвращает память выгрузкой простаивающего движка."""
    loaded = StubEngine()
    loaded._mark(STATE_READY)
    monkeypatch.setattr(job_queue, "created_engines", lambda: {"f5": loaded})
    calls = {"count": 0}

    def critical() -> bool:
        calls["count"] += 1
        return calls["count"] == 1  # критично ровно до выгрузки

    monkeypatch.setattr(job_queue.resource_guard, "is_memory_critical", critical)

    asyncio.run(JobQueue()._wait_for_memory("задача-без-проекта"))
    assert loaded.is_loaded is False, "простаивающий движок не выгружен"
    assert loaded.active_synthesizes == 0


def test_queue_does_not_unload_engine_during_render(stub, monkeypatch):
    """Во время рендера выгрузка не делается: модель нужна следующей реплике."""
    loaded = StubEngine()
    loaded._mark(STATE_READY)
    monkeypatch.setattr(job_queue, "created_engines", lambda: {"f5": loaded})

    async def scenario():
        queue = JobQueue()
        queue._current_job_id = "job-1"  # имитация идущего рендера
        return queue._free_memory_for_job()

    assert asyncio.run(scenario()) == []
    assert loaded.is_loaded is True


def test_watchdog_uses_memory_state(monkeypatch):
    """Watchdog прерывает задачу по состоянию памяти, а не по «одному RSS».

    Проверяется связка: критическое состояние, удержавшееся несколько проверок,
    вызывает `on_breach` — иначе задача продолжала бы давить на память, которую
    заняла сама же.
    """
    events: list[str] = []
    monkeypatch.setattr(resource_guard, "CHECK_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(
        resource_guard.memory_monitor,
        "get_state",
        lambda sampler=None: {
            "state": memory_monitor.STATE_CRITICAL,
            "reason": "процессы синтеза занимают 7000 МБ",
        },
    )
    guard = resource_guard.ResourceGuard(lambda: events.append("breach") or "прервано")

    async def scenario():
        task = asyncio.create_task(guard.run())
        for _ in range(200):
            if events:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert events == ["breach"]


# --- лёгкий API и безопасная выгрузка ------------------------------------------
@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Изоляция включена, в воркере — лёгкая заглушка с журналом вызовов."""
    monkeypatch.setenv(config.WORKER_ISOLATION_ENV, "1")
    monkeypatch.setenv(config.WORKER_ENGINE_FACTORY_ENV, FACTORY)
    monkeypatch.setenv("TTS_FAKE_CALL_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(config, "WORKER_LOAD_TIMEOUT_SEC", 20.0)
    monkeypatch.setattr(config, "WORKER_SHUTDOWN_GRACE_SEC", 3.0)
    fresh = supervisor_module.WorkerSupervisor()
    monkeypatch.setattr(supervisor_module, "_supervisor", fresh)
    monkeypatch.setattr(registry, "_instances", {})
    yield SimpleNamespace(supervisor=fresh, path=tmp_path)
    fresh.stop_all(grace=1.0)
    fresh.reset_after_stop()


def _synth(engine, text: str):
    return engine.synthesize(text, "/tmp/reference.wav", "референс")


def test_light_api_answers_while_worker_is_dead(worker_env, voices):
    """Мёртвый воркер не мешает ни `/api/status`, ни диагностике."""
    engine = registry.get_engine("f5")
    _synth(engine, FIRST)
    process = engine.worker.process
    assert process is not None
    process.kill()  # падение мимо супервизора: именно так его и не заметить
    process.wait(timeout=10)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return (
                await client.get("/api/status"),
                await client.get("/api/diagnostics/worker"),
            )

    status, diagnostics = asyncio.run(scenario())
    assert status.status_code == 200
    assert status.json()["workers"][0]["alive"] is False
    assert diagnostics.status_code == 200
    assert diagnostics.json()["isolation"] is True
    engine.unload()


def test_unload_is_safe_after_crash(worker_env, voices):
    """Выгрузка после падения воркера — не ошибка: выгружать уже нечего."""
    engine = registry.get_engine("f5")
    _synth(engine, FIRST)
    with pytest.raises(proto.WorkerCrashError):
        _synth(engine, CRASH_TEXT)

    engine.unload()  # не должно бросить
    assert engine.worker.state in (proto.WORKER_STATE_STOPPED, proto.WORKER_STATE_IDLE)

    # И движок поднимается заново обычным путём.
    _, sample_rate = _synth(engine, FIRST)
    assert sample_rate == 24000
    engine.unload()


def test_degraded_engine_fails_fast_without_extra_attempts(worker_env, voices, monkeypatch):
    """После серии падений задача падает быстро и объяснимо, очередь остаётся живой."""
    monkeypatch.setattr(config, "WORKER_CRASH_RETRIES", 5)
    engine = registry.get_engine("f5")

    for _ in range(config.WORKER_CRASH_LIMIT):
        with pytest.raises(proto.WorkerCrashError):
            _synth(engine, CRASH_TEXT)
    assert engine.worker.state == proto.WORKER_STATE_DEGRADED
    failures_before = engine.worker.failures

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload())
            assert await _wait_until(lambda: job.status is JobStatus.ERROR)
            return job
        finally:
            await queue.stop()

    job = asyncio.run(scenario())
    assert job.error_type == proto.ERROR_WORKER_CRASH
    assert "временно отключён" in (job.error or "")
    # Повторов не было: движок отключён, и «пробовать ещё» значило бы крутить цикл.
    assert engine.worker.failures == failures_before


def test_shutdown_marks_replica_interrupted_and_keeps_ready_pieces(worker_env, voices):
    """Остановка сервера: готовое сохранено, реплика помечена прерванной, сирот нет."""
    store = get_projects_store()
    project = store.create_project(
        "Остановка", source_text=f"ИВАН: {FIRST}\nИВАН: {HANG_TEXT}", mode="dialogue"
    )
    parsed = store.parse_project(project["id"], None)
    replicas = [
        Replica(voice=row["speaker"], text=row["text"], line_number=row["index"] + 1)
        for row in parsed["replicas"]
    ]

    async def scenario():
        queue = JobQueue()
        await queue.start()
        try:
            job = queue.submit(_payload(replicas, project_id=project["id"]))
            # Ждём, пока воркер уйдёт во вторую реплику (она «висит»), и выключаемся
            # так же, как это делает lifespan: сначала помечаем выключение, потом
            # останавливаем очередь (она прервёт текущий инференс).
            assert await _wait_until(lambda: job.current_replica == 1)
            worker_env.supervisor.begin_shutdown()
            await queue.stop()
            return job
        finally:
            await queue.stop()

    job = asyncio.run(scenario())
    assert job.error_type == proto.ERROR_INTERRUPTED
    assert "остановка сервера" in (job.error or "")

    fresh = store.get_project(project["id"])
    by_index = {row["index"]: row for row in fresh["replicas"]}
    # Первая реплика синтезирована и сохранена, вторая прервана выключением.
    assert by_index[0]["status"] == config.REPLICA_STATUS_RENDERED
    assert by_index[0]["takes"], "готовый кусок потерян при выключении"
    assert Path(by_index[0]["takes"][0]["audio_path"]).exists()
    assert by_index[1]["status"] == config.REPLICA_STATUS_INTERRUPTED

    # Путь выключения приложения (см. lifespan): воркеры останавливаются, и
    # осиротевших процессов не остаётся.
    worker_env.supervisor.stop_all()
    for item in worker_env.supervisor.status():
        assert item["state"] == proto.WORKER_STATE_STOPPED
        assert item["alive"] is False
        assert item["failures"] == 0, "остановка приложения не должна считаться падением"
