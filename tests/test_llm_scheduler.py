"""Планировщик тяжёлых задач (Task 2, фаза 4): память и один inference за раз.

Проверяется ровно то, что в постановке названо обязательным: WARNING/CRITICAL/HARD
STOP блокируют новую тяжёлую работу, LLM не идёт одновременно с синтезом или
Whisper, слот освобождается при любой ошибке, а лёгкие API продолжают отвечать.

Модели не поднимаются: датчики памяти подставные, слот — общий `HeavyGate` из
Task 1, синтез представлен держателем слота, а не реальным воркером.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from conftest import StubEngine  # noqa: F401 — нужен фикстуре движков

from backend import job_queue as job_queue_module
from backend.llm import memory_policy as memory
from backend.llm import scheduler as scheduler_module
from backend.llm.scheduler import HeavyBlockedError, HeavyScheduler

GREEN = {"system_percent": 50.0, "available_gb": 12.0, "pressure": "green"}


def _scheduler(sample: dict | None = None, *, gate=None, external_busy=None) -> HeavyScheduler:
    return HeavyScheduler(
        gate=gate if gate is not None else memory.HeavyGate(),
        sensor=lambda: dict(sample or GREEN),
        external_busy=external_busy,
    )


def test_memory_warning_blocks_new_llm_inference():
    """WARNING: новая тяжёлая задача не начинается, причина видна вызывающему."""
    scheduler = _scheduler({"system_percent": 78.0, "available_gb": 9.0, "pressure": "green"})
    decision = scheduler.check(memory.HEAVY_LLM)
    assert decision.allowed is False
    assert decision.memory_level == memory.LEVEL_WARNING
    assert "WARNING" in decision.memory_reason or "занята" in decision.memory_reason
    with pytest.raises(HeavyBlockedError) as exc, scheduler.hold(memory.HEAVY_LLM):
        raise AssertionError("слот не должен выдаваться")
    assert exc.value.decision.memory_level == memory.LEVEL_WARNING
    assert scheduler.busy_reason() == "", "отказ не занимает слот"


def test_memory_critical_blocks_heavy_dispatch():
    """CRITICAL: не начинается ни LLM, ни синтез — и рекомендуется выгрузка."""
    scheduler = _scheduler({"system_percent": 84.0, "available_gb": 9.0, "pressure": "green"})
    for kind in (memory.HEAVY_LLM, memory.HEAVY_TTS, memory.HEAVY_WHISPER):
        decision = scheduler.check(kind)
        assert decision.allowed is False, kind
        assert decision.memory_level == memory.LEVEL_CRITICAL, kind
    decision = memory.decide_current(
        thresholds=None, sensor=lambda: {"system_percent": 84.0, "available_gb": 9.0}
    )
    assert decision.unload_recommended is True


def test_hard_stop_prevents_llm_start():
    """HARD STOP: запуск запрещён, и это видно по уровню решения."""
    scheduler = _scheduler({"system_percent": 91.0, "available_gb": 2.0, "pressure": "green"})
    with pytest.raises(HeavyBlockedError) as exc, scheduler.hold(memory.HEAVY_LLM):
        raise AssertionError("HARD STOP не должен пускать запуск")
    assert exc.value.decision.memory_level == memory.LEVEL_HARD_STOP


def test_macos_pressure_blocks_even_with_low_percent():
    """Красное давление macOS важнее процента: запуск запрещён при 45 %."""
    scheduler = _scheduler({"system_percent": 45.0, "available_gb": 12.0, "pressure": "red"})
    assert scheduler.check(memory.HEAVY_LLM).allowed is False


def test_llm_and_tts_do_not_infer_concurrently():
    """LLM и синтез не идут одновременно: слот один на всех."""
    scheduler = _scheduler()
    with scheduler.hold(memory.HEAVY_LLM, owner="анализ"):
        assert scheduler.check(memory.HEAVY_TTS).allowed is False
        assert "занят" in scheduler.check(memory.HEAVY_TTS).reason
        # Планировщик отказывает до захвата и объясняет причину — это его контракт.
        with pytest.raises(HeavyBlockedError), scheduler.hold(memory.HEAVY_TTS, owner="синтез"):
            raise AssertionError("синтез не должен получить слот")
    # Обратный порядок: синтез держит слот — анализ ждёт.
    with scheduler.hold(memory.HEAVY_TTS, owner="синтез"):
        assert scheduler.check(memory.HEAVY_LLM).allowed is False
    assert scheduler.busy_reason() == ""


def test_llm_and_whisper_do_not_infer_concurrently():
    """Whisper — такой же тяжёлый inference: с LLM одновременно не выполняется."""
    scheduler = _scheduler()
    with scheduler.hold(memory.HEAVY_WHISPER, owner="распознавание"):
        assert scheduler.check(memory.HEAVY_LLM).allowed is False
        with pytest.raises(HeavyBlockedError), scheduler.hold(memory.HEAVY_LLM):
            raise AssertionError("LLM не должен получить слот")
    assert scheduler.busy_reason() == ""


def test_external_busy_is_visible_to_llm():
    """Занятость вне процесса (воркер синтеза) тоже блокирует LLM-анализ."""
    scheduler = _scheduler(external_busy=lambda: "воркер синтеза выполняет реплику")
    decision = scheduler.check(memory.HEAVY_LLM)
    assert decision.allowed is False
    assert "воркер синтеза" in decision.reason


def test_slot_is_released_on_error_timeout_and_cancel():
    """Ошибка, таймаут или отмена внутри задачи не оставляют слот занятым."""
    scheduler = _scheduler()
    with (
        pytest.raises(ValueError, match="boom"),
        scheduler.hold(memory.HEAVY_LLM, owner="ошибка"),
    ):
        raise ValueError("boom")
    assert scheduler.busy_reason() == ""

    with (
        pytest.raises(TimeoutError),
        scheduler.hold(memory.HEAVY_LLM, owner="таймаут"),
    ):
        raise TimeoutError("Ollama не ответила")
    assert scheduler.busy_reason() == ""

    class Cancelled(Exception):
        pass

    with pytest.raises(Cancelled), scheduler.hold(memory.HEAVY_LLM, owner="отмена"):
        raise Cancelled()
    assert scheduler.busy_reason() == ""
    # После освобождения следующая тяжёлая задача снова проходит.
    with scheduler.hold(memory.HEAVY_LLM):
        assert scheduler.busy_reason() != ""


def test_light_work_keeps_running_while_llm_holds_slot():
    """Лёгкая работа (та же ветка, что API/SQLite) не блокируется слотом."""
    scheduler = _scheduler()
    done: list[str] = []

    def light_job() -> None:
        done.append("light")

    with scheduler.hold(memory.HEAVY_LLM, owner="анализ"):
        thread = threading.Thread(target=light_job)
        thread.start()
        thread.join(timeout=2)
        assert done == ["light"], "лёгкая работа не должна ждать тяжёлый слот"


def test_queue_waits_for_heavy_slot_and_reports_it(monkeypatch):
    """Очередь синтеза ждёт свободный слот и показывает причину ожидания."""
    # Ожидание в жизни длится секунды; в тесте — доли секунды.
    monkeypatch.setattr(job_queue_module, "MEMORY_WAIT_RETRY_SEC", 0.02)
    gate = memory.HeavyGate()
    queue = job_queue_module.JobQueue()
    job_id = "job-wait"
    queue._jobs[job_id] = job_queue_module.Job(
        id=job_id,
        status=job_queue_module.JobStatus.QUEUED,
        payload=None,  # type: ignore[arg-type]
        created_at="",
    )
    ticket = gate.try_acquire(memory.HEAVY_LLM, owner="анализ")
    monkeypatch.setattr(memory, "get_gate", lambda: gate)

    async def scenario() -> bool:
        task = asyncio.create_task(queue._wait_for_heavy_slot(job_id))
        await asyncio.sleep(0.05)
        assert queue._jobs[job_id].message.startswith("Ожидание:")
        assert not task.done(), "задача не должна начинаться при занятом слоте"
        gate.release(ticket)
        return await asyncio.wait_for(task, timeout=3)

    assert asyncio.run(scenario()) is True


def test_queue_light_api_answers_while_llm_analysis_runs(monkeypatch):
    """Лёгкие API отвечают, пока идёт LLM-анализ: слот держит только тяжёлое."""
    import asyncio as aio

    from test_projects_api import _client

    gate = memory.HeavyGate()

    async def scenario():
        async with _client(monkeypatch) as client:
            with gate.hold(memory.HEAVY_LLM, owner="анализ"):
                response = await client.get("/api/status")
                assert response.status_code == 200
                assert response.json()["engines"] is not None

    aio.run(scenario())


def test_scheduler_singleton_shares_gate_with_queue():
    """Планировщик и очередь используют один и тот же gate, а не свои копии."""
    scheduler_module.reset_scheduler()
    try:
        scheduler = scheduler_module.get_scheduler()
        assert scheduler.gate is memory.get_gate()
        with scheduler.hold(memory.HEAVY_LLM, owner="анализ"):
            assert memory.get_gate().busy() != ""
    finally:
        scheduler_module.reset_scheduler()
    assert memory.get_gate().busy() == ""


def test_scheduler_decision_is_serializable():
    """Решение планировщика уходит в ответ API и в лог — значит, сериализуемо."""
    scheduler = _scheduler()
    payload = scheduler.check(memory.HEAVY_LLM).to_dict()
    assert payload["allowed"] is True
    assert payload["kind"] == memory.HEAVY_LLM
    assert "memory_level" in payload


def test_scheduler_does_not_block_light_deterministic_work():
    """Детерминированная подготовка текста не занимает слот и не ждёт его."""
    scheduler = _scheduler({"system_percent": 95.0, "available_gb": 1.0, "pressure": "red"})
    # Планировщик запрещает тяжёлое…
    assert scheduler.check(memory.HEAVY_LLM).allowed is False
    # …но обычная работа продолжается: она сюда вообще не обращается.
    from backend.text_normalization import normalize

    started = time.monotonic()
    assert normalize("В 2026 году") != ""
    assert time.monotonic() - started < 5.0


def test_queue_dispatches_light_task_while_gate_is_busy(monkeypatch):
    """Лёгкая задача очереди не ждёт тяжёлый слот: иначе UI зависал бы за анализом.

    Выбор ранее сохранённого варианта — запись в базу. Держать её за LLM-анализом
    значит задерживать отзывчивое действие интерфейса без причины.
    """
    gate = memory.HeavyGate()
    monkeypatch.setattr(memory, "get_gate", lambda: gate)
    monkeypatch.setattr(job_queue_module, "MEMORY_WAIT_RETRY_SEC", 0.02)
    queue = job_queue_module.JobQueue()
    handled: list[str] = []

    async def fake_select(task) -> None:
        handled.append(task.job_id)

    monkeypatch.setattr(queue, "_select_variant", fake_select)
    ticket = gate.try_acquire(memory.HEAVY_LLM, owner="анализ")

    async def scenario() -> None:
        await queue.start()
        job = queue._jobs.setdefault(
            "j-light",
            job_queue_module.Job(
                id="j-light",
                status=job_queue_module.JobStatus.QUEUED,
                payload=None,  # type: ignore[arg-type]
                created_at="",
            ),
        )
        assert job is not None
        await queue._queue.put(
            (0, 0, job_queue_module.SelectVariantTask(job_id="j-light", index=0, variant_id="v1"))
        )
        for _ in range(100):
            if handled:
                break
            await asyncio.sleep(0.02)
        await queue.stop()

    try:
        asyncio.run(scenario())
    finally:
        gate.release(ticket)
    assert handled == ["j-light"], "лёгкая задача обязана выполниться при занятом слоте"


def test_queue_holds_slot_while_rendering(monkeypatch):
    """Тяжёлая задача очереди занимает слот и не отпускает его до конца работы."""
    gate = memory.HeavyGate()
    monkeypatch.setattr(memory, "get_gate", lambda: gate)
    queue = job_queue_module.JobQueue()
    observed: list[str] = []

    async def fake_render(task) -> None:
        observed.append(gate.busy())

    monkeypatch.setattr(queue, "_render", fake_render)

    async def scenario() -> None:
        await queue.start()
        queue._jobs["j-heavy"] = job_queue_module.Job(
            id="j-heavy",
            status=job_queue_module.JobStatus.QUEUED,
            payload=None,  # type: ignore[arg-type]
            created_at="",
        )
        await queue._queue.put(
            (
                0,
                0,
                job_queue_module.RenderTask(job_id="j-heavy", payload=None),  # type: ignore[arg-type]
            )
        )
        for _ in range(200):
            if observed:
                break
            await asyncio.sleep(0.02)
        await queue.stop()

    asyncio.run(scenario())
    assert observed and "job:j-heavy" in observed[0], "синтез обязан держать тяжёлый слот"
    assert gate.busy() == "", "после задачи слот освобождён"
