"""Engine-aware ETA: статистика по движкам, холодный старт, проверка и остаток.

Модуль `backend/eta.py` проверяется сам по себе (он для этого и сделан без
обращений к моделям), а затем — тем же путём, что и в бою: настоящая очередь на
`StubEngine` с разной скоростью у разных движков. Реальные модели не
поднимаются: задержку изображает `seconds_per_char` заглушки.
"""

import asyncio
import contextlib
import threading
import time

import httpx
import pytest
import soundfile as sf
from conftest import StubEngine, sine

from backend import audio_pipeline, eta, main
from backend.audio_pipeline import QaSettings, RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines.base import ENGINE_F5, ENGINE_XTTS, SAMPLE_RATE, STATE_READY
from backend.job_queue import JobPayload, JobQueue, JobStatus
from backend.voices_store import Voice

F5 = ENGINE_F5
XTTS = ENGINE_XTTS
UNKNOWN = "неизвестный-движок"


@pytest.fixture(autouse=True)
def clean_tracker():
    """Свежая статистика на каждый тест.

    Синглтон живёт в процессе и намеренно копится между задачами; в тестах это
    означало бы, что результат зависит от порядка их выполнения.
    """
    eta.reset()
    yield
    eta.reset()


def _step(engine: str = F5, chars: int = 100, qa: str = "off", cold: bool = False):
    return eta.EtaStep(engine=engine, chars=chars, qa_mode=qa, cold=cold)


# --- 1. Независимая статистика движков ----------------------------------------


def test_engines_keep_independent_statistics():
    tracker = eta.EtaTracker()
    # F5 учится быть быстрым, XTTS — медленным; дефолты разные уже сами по себе.
    for _ in range(6):
        tracker.observe(F5, chars=100, render_sec=1.0)
        tracker.observe(XTTS, chars=100, render_sec=4.0)

    f5 = tracker.estimate([_step(engine=F5, chars=100)])
    xtts = tracker.estimate([_step(engine=XTTS, chars=100)])
    assert f5 == pytest.approx(1.0, rel=0.2)
    assert xtts == pytest.approx(4.0, rel=0.2)
    assert xtts > f5 * 2

    # Наблюдение одного движка не сдвигает оценку другого.
    before = tracker.estimate([_step(engine=F5, chars=100)])
    for _ in range(10):
        tracker.observe(XTTS, chars=100, render_sec=9.0)
    assert tracker.estimate([_step(engine=F5, chars=100)]) == pytest.approx(before)


def test_unknown_engine_has_its_own_fallback():
    tracker = eta.EtaTracker()
    tracker.observe(F5, chars=100, render_sec=5.0)
    estimate = tracker.estimate([_step(engine=UNKNOWN, chars=100)])
    assert estimate > 0
    # Чужой истории у него нет: это общий дефолт, а не скорость F5.
    assert estimate == pytest.approx(100 * eta.DEFAULT_SEC_PER_CHAR[""])


# --- 2-3. Холодный и тёплый старт ---------------------------------------------


def test_cold_start_adds_model_load_cost():
    tracker = eta.EtaTracker()
    warm = tracker.estimate([_step(cold=False)])
    cold = tracker.estimate([_step(cold=True)])
    assert cold > warm
    assert cold - warm == pytest.approx(eta.DEFAULT_COLD_START_SEC[F5])

    # Надбавка одна на движок: холодный только первый кусок, остальные тёплые.
    plan = [_step(cold=True), _step(cold=False), _step(cold=False)]
    assert tracker.estimate(plan) == pytest.approx(
        tracker.estimate([_step()]) * 3 + eta.DEFAULT_COLD_START_SEC[F5]
    )


def test_warm_start_gets_no_cold_penalty():
    tracker = eta.EtaTracker()
    tracker.observe(F5, chars=100, render_sec=1.0)
    warm = tracker.estimate([_step(cold=False)])
    assert warm == pytest.approx(1.0, rel=0.2)
    assert warm < tracker.estimate([_step(cold=True)])

    # Оценка тёплого куска не зависит от того, сколько раз движок поднимали.
    tracker.note_load(F5, 20.0)
    assert tracker.estimate([_step(cold=False)]) == pytest.approx(warm)


def test_load_cost_is_subtracted_from_first_observation():
    tracker = eta.EtaTracker()
    tracker.note_load(F5, 5.0)
    tracker.observe(F5, chars=100, render_sec=6.0, cold=True)
    # 6 c = 5 c загрузки + 1 c инференса, а не 6 c на 100 знаков.
    assert tracker.estimate([_step(chars=100, cold=False)]) == pytest.approx(1.0, rel=0.2)


# --- 4. Множитель проверки ----------------------------------------------------


def test_qa_multiplier_is_engine_specific_and_learned():
    # Дефолты: strict дороже smart, smart дороже off — и это константы модуля,
    # которые затем перебиваются наблюдениями.
    defaults = eta.EtaTracker()
    assert defaults.qa_factor(F5, "strict") > defaults.qa_factor(F5, "smart") > 1.0
    assert defaults.estimate([_step(qa="strict")]) > defaults.estimate([_step(qa="off")])

    # Наблюдение strict'а меняет только цену strict'а у своего движка.
    tracker = eta.EtaTracker()
    tracker.observe(F5, chars=100, render_sec=1.0)
    off_before = tracker.estimate([_step(qa="off")])
    strict_before = tracker.estimate([_step(qa="strict")])
    tracker.observe(F5, chars=100, render_sec=4.0, qa_mode="strict", attempts=2)
    assert tracker.estimate([_step(qa="strict")]) > strict_before
    assert tracker.estimate([_step(qa="off")]) == pytest.approx(off_before)
    # Цена проверки у другого движка своя: наблюдение F5 её не трогает.
    assert eta.EtaTracker().qa_factor(XTTS, "strict") == pytest.approx(
        eta.DEFAULT_QA_FACTOR["strict"]
    )


def test_qa_factor_never_below_one():
    tracker = eta.EtaTracker()
    for _ in range(20):
        # Проверка не может быть быстрее самого синтеза, каким бы шумным ни был факт.
        tracker.observe(F5, chars=100, render_sec=0.01, qa_mode="strict")
    assert tracker.qa_factor(F5, "strict") >= 1.0
    assert tracker.estimate([_step(qa="strict")]) >= tracker.estimate([_step(qa="off")])


# --- 5. EWMA ------------------------------------------------------------------


def test_ewma_converges_to_observed_value():
    tracker = eta.EtaTracker()
    baseline = tracker.estimate([_step(chars=100)])
    assert baseline == pytest.approx(100 * eta.DEFAULT_SEC_PER_CHAR[F5])  # пока дефолт

    for _ in range(60):
        tracker.observe(F5, chars=100, render_sec=2.0)

    estimate = tracker.estimate([_step(chars=100)])
    assert estimate == pytest.approx(2.0, rel=0.1)
    assert estimate != pytest.approx(baseline)
    assert tracker.snapshot()[F5]["samples"] == 60

    # Между дефолтом и сходимостью оценка двигается постепенно, а не прыжком:
    # коэффициент EWMA — не 1.0 и не 0.0.
    first = eta.EtaTracker()
    first.observe(F5, chars=100, render_sec=2.0)
    assert first.estimate([_step(chars=100)]) == pytest.approx(
        (1 - eta.EWMA_ALPHA) * baseline + eta.EWMA_ALPHA * 2.0
    )


# --- 6. Fallback без истории --------------------------------------------------


def test_fallback_without_history_is_sane_and_engine_specific():
    tracker = eta.EtaTracker()
    f5 = tracker.estimate([_step(engine=F5, chars=200)])
    xtts = tracker.estimate([_step(engine=XTTS, chars=200)])
    for value in (f5, xtts):
        assert 0.0 < value < 3600.0
    # F5 в этом проекте заметно быстрее XTTS — дефолты это различие сохраняют.
    assert f5 < xtts
    # Оценка ничего не «наблюдала» — истории нет, только дефолты.
    assert all(stats["samples"] == 0 for stats in tracker.snapshot().values())


# --- 7. ETA не отрицательный --------------------------------------------------


def test_eta_is_never_negative():
    tracker = eta.EtaTracker()
    assert tracker.estimate([]) == 0.0
    assert tracker.estimate(None) == 0.0

    # Шумные наблюдения: отрицательное время, нули, гигантские значения.
    for render_sec in (-5.0, 0.0, -0.001, 1e6, -1e6):
        tracker.observe(F5, chars=100, render_sec=render_sec, cold=True)
        assert tracker.estimate([_step(), _step(engine=XTTS), _step(qa="strict")]) >= 0.0
    assert tracker.estimate([_step(chars=0)]) >= 0.0
    assert tracker.sec_per_char(F5) > 0


def test_format_eta_russian_forms_and_zero():
    assert eta.format_eta(0) == "0 сек"
    assert eta.format_eta(-3) == "0 сек"
    assert eta.format_eta(1) == "1 сек"
    assert eta.format_eta(2) == "2 сек"
    assert eta.format_eta(5) == "5 сек"
    assert eta.format_eta(11) == "11 сек"
    assert eta.format_eta(21) == "21 сек"
    assert eta.format_eta(60) == "1 мин"  # ровная минута — без «0 сек»
    assert eta.format_eta(3600) == "1 ч"
    assert eta.format_eta(160) == "2 мин 40 сек"
    assert eta.format_eta(3852) == "1 ч 4 мин 12 сек"
    assert eta.format_eta(float("nan")) == "0 сек"
    assert eta.format_eta(None) == "0 сек"


# --- 8-9. Остаток и смешанный проект ------------------------------------------


def test_completed_work_reduces_remaining_estimate():
    tracker = eta.EtaTracker()
    plan = [_step(chars=150) for _ in range(4)]
    initial = tracker.estimate(plan)

    tracker.observe(F5, chars=150, render_sec=1.4, cold=plan[0].cold)
    remaining = tracker.estimate(plan[1:])
    assert remaining < initial
    # Выполнен кусок — в остатке на один кусок меньше.
    assert remaining == pytest.approx(tracker.estimate([_step(chars=150)] * 3))
    assert tracker.estimate(plan[4:]) == 0.0


def test_mixed_project_sums_by_engine():
    tracker = eta.EtaTracker()
    tracker.observe(F5, chars=100, render_sec=1.0)
    tracker.observe(XTTS, chars=100, render_sec=4.0)
    tracker.observe(XTTS, chars=100, render_sec=4.0)
    tracker.observe(XTTS, chars=100, render_sec=4.0)
    tracker.observe(XTTS, chars=100, render_sec=4.0)
    tracker.observe(XTTS, chars=100, render_sec=4.0)
    mixed = [
        _step(engine=F5, chars=100),
        _step(engine=XTTS, chars=100),
        _step(engine=F5, chars=100),
    ]
    per_chunk = sum(tracker.estimate([step]) for step in mixed)
    assert tracker.estimate(mixed) == pytest.approx(per_chunk)
    # Сумма по кускам равна сумме оценок каждого куска: F5 + XTTS + F5.
    assert tracker.estimate(mixed) == pytest.approx(1.0 + 4.0 + 1.0, rel=0.1)
    # И это не одно среднее на все реплики: XTTS-кусок заметно дороже F5.
    assert tracker.estimate([mixed[1]]) > 2 * tracker.estimate([mixed[0]])


# --- Интеграция с очередью и REST ---------------------------------------------


def _payload(count: int = 3, qa: QaSettings | None = None) -> JobPayload:
    return JobPayload(
        replicas=[
            Replica(voice="#1", text=f"Реплика номер {index}", line_number=index)
            for index in range(1, count + 1)
        ],
        speakers={"#1": SpeakerSettings(voice_id="voice1")},
        settings=RenderSettings(pause_ms=0, output_format="wav", qa=qa),
    )


def _engines(monkeypatch, engines: dict[str, StubEngine]) -> None:
    """Свой движок на каждый id: разницу скоростей видно и в плане, и в оценке."""
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engines[engine_id])
    monkeypatch.setattr(audio_pipeline, "engine_for_id", lambda engine_id: engines[engine_id])
    monkeypatch.setattr(audio_pipeline, "_engine_for", lambda voice: engines[voice.engine])


def _voice(workspace, voice_id: str, engine: str) -> Voice:
    path = workspace / "voices" / f"{voice_id}.wav"
    sf.write(path, sine(2.0, 180.0), SAMPLE_RATE)
    return Voice(
        id=voice_id,
        name=f"Голос {voice_id}",
        gender="male",
        ref_text="Привет, это тест",
        audio_file=path.name,
        engine=engine,
    )


def _store(monkeypatch, voices: dict[str, Voice]) -> None:
    class Store:
        def get(self, voice_id: str):
            return voices.get(voice_id)

    monkeypatch.setattr(audio_pipeline, "get_store", lambda: Store())


def _mixed_payload(count: int = 4) -> JobPayload:
    """Диалог, где реплики идут разными движками: F5, XTTS, F5, XTTS."""
    payload = _payload(count=count)
    payload.replicas = [
        Replica(
            voice="#1" if index % 2 else "#2",
            text=f"Реплика номер {index}",
            line_number=index,
        )
        for index in range(1, count + 1)
    ]
    payload.speakers = {
        "#1": SpeakerSettings(voice_id="voice1"),
        "#2": SpeakerSettings(voice_id="voice2"),
    }
    return payload


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


class _GatedEngine(StubEngine):
    """Заглушка, которая запирает каждый синтез до разрешения теста.

    У каждого вызова свой номер и своё событие `entered`: тест видит, что
    воркер вошёл именно в этот кусок, — и только тогда читает статус задачи.
    Без шлюза заглушка проходит весь диалог быстрее, чем тест успевает
    заглянуть в промежуточное состояние.
    """

    def __init__(self, gate: "_Gate", seconds_per_char: float = 0.02) -> None:
        super().__init__(seconds_per_char)
        self._gate = gate

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        self._gate.arrive(text)
        return super()._synthesize(text, ref_audio_path, ref_text, speed, params)


class _Gate:
    """События входа в куски и разрешение продолжить."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        # Ключ — номер вызова, а не текст: у соседних реплик текст может совпасть,
        # и событие «вошли в реплику 1» сработало бы и на второй такой же.
        self.entered: dict[int, threading.Event] = {}
        self.release = threading.Event()
        self.open_flag = False

    def arrive(self, text: str) -> None:
        position = len(self.calls) + 1  # как «реплика N» у пайплайна: с единицы
        self.calls.append(text)
        self.entered.setdefault(position, threading.Event()).set()
        if self.open_flag:
            return
        assert self.release.wait(10.0), f"тест не отпустил синтез №{position}"

    def wait_entered(self, position: int, timeout: float = 10.0) -> bool:
        event = self.entered.setdefault(position, threading.Event())
        return event.wait(timeout)

    def release_one(self) -> None:
        self.release.set()
        self.release.clear()

    def open(self) -> None:
        self.open_flag = True
        self.release.set()


def _mixed_queue(monkeypatch, workspace):
    """Очередь на двух заглушках: F5 быстрый, XTTS медленный."""
    engines = {F5: StubEngine(seconds_per_char=0.02), XTTS: StubEngine(0.1)}
    _engines(monkeypatch, engines)
    _store(
        monkeypatch,
        {
            "voice1": _voice(workspace, "voice1", F5),
            "voice2": _voice(workspace, "voice2", XTTS),
        },
    )
    return engines


def test_queue_shows_eta_grows_down_and_resets(monkeypatch, workspace):
    """ETA появляется во время рендера, уменьшается по ходу и обнуляется в конце.

    Заглушка придерживается на каждом куске: без задержки рендер заканчивается
    быстрее, чем тест успевает увидеть промежуточный статус, и «ETA уменьшается»
    нечем было бы проверить. Порядок реплик — F5, XTTS, F5, XTTS.
    """
    engines = _mixed_queue(monkeypatch, workspace)
    gate = _Gate()
    engines[F5] = _GatedEngine(gate, seconds_per_char=0.02)
    engines[XTTS] = _GatedEngine(gate, seconds_per_char=0.1)
    _engines(monkeypatch, engines)

    async def scenario():
        async with _queue() as queue:
            job = queue.submit(_mixed_payload(count=4))
            try:
                # План посчитан, первый кусок ещё не начался — это первичная оценка.
                assert await asyncio.to_thread(gate.wait_entered, 1)
                initial = job.eta_sec
                assert initial is not None and initial > 0
                assert job.to_dict()["eta_text"] is not None

                # Отпускаем F5-кусок и упираемся в XTTS. Остаток меньше
                # первичной оценки: выполненный кусок из него ушёл.
                gate.release_one()
                assert await asyncio.to_thread(gate.wait_entered, 2)
                observed = job.eta_sec
                assert observed is not None and observed > 0
                assert observed < initial, "выполненный кусок не уменьшил остаток"
            finally:
                gate.open()
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            assert job.eta_sec == 0.0
            assert job.to_dict()["eta_text"] is None
            return job

    job = asyncio.run(scenario())
    assert job.duration_sec and job.duration_sec > 0


def test_eta_estimate_drops_after_observed_chunk():
    """Оценка остатка после куска меньше оценки всего плана.

    Отдельно от очереди сравнением двух чисел одного плана — до синтеза и после
    первого куска: выполненная работа обязана уменьшать остаток.
    """
    tracker = eta.get_tracker()
    plan = [
        _step(engine=F5, chars=100, cold=True),
        _step(engine=XTTS, chars=100, cold=True),
        _step(engine=F5, chars=100),
    ]
    initial = tracker.estimate(plan)
    tracker.observe(F5, chars=100, render_sec=2.0, cold=True)
    remaining = tracker.estimate(plan[1:])
    assert 0 < remaining < initial


def test_singleton_accumulates_data_between_jobs(monkeypatch, workspace):
    """Статистика живёт в процессе: второй рендер опирается на первый."""
    engines = _mixed_queue(monkeypatch, workspace)
    calls = [0]

    async def scenario():
        async with _queue() as queue:
            for _ in range(2):
                job = queue.submit(_mixed_payload(count=2))
                done = _wait_until(lambda job=job: job.status is JobStatus.DONE)
                assert await done, job.error
                calls[0] += len(engines[F5].calls) + len(engines[XTTS].calls)
                engines[F5].calls.clear()
                engines[XTTS].calls.clear()

    asyncio.run(scenario())
    snapshot = eta.get_tracker().snapshot()
    # Общий синглтон: наблюдения обеих задач в одной истории, и она не пуста.
    assert set(snapshot) == {F5, XTTS}
    assert snapshot[F5]["samples"] >= 2
    assert snapshot[XTTS]["samples"] >= 2
    assert calls[0] > 0


def test_job_status_route_carries_eta_text(monkeypatch, fake_store):
    async def scenario():
        queue = JobQueue()
        await queue.start()
        monkeypatch.setattr(main, "get_queue", lambda: queue)
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                job = queue.submit(_payload(count=2))
                response = await client.get(f"/api/jobs/{job.id}")
                assert response.status_code == 200
                body = response.json()
                assert "eta_sec" in body  # поле осталось: обратная совместимость
                assert body["eta_text"] is None  # в очереди ждать ещё нечего

                queue._jobs[job.id].eta_sec = 160.0
                body = (await client.get(f"/api/jobs/{job.id}")).json()
                assert body["eta_sec"] == 160.0
                assert body["eta_text"] == "2 мин 40 сек"
        finally:
            await queue.stop()

    asyncio.run(scenario())


def test_singleton_tracker_is_shared_and_resettable():
    first = eta.get_tracker()
    first.observe(F5, chars=100, render_sec=1.0)
    assert eta.get_tracker() is first
    assert eta.get_tracker().snapshot()[F5]["samples"] == 1
    eta.reset()
    assert eta.get_tracker() is not first
    assert eta.get_tracker().snapshot() == {}


def test_observer_is_thread_safe():
    """Синтез идёт в отдельном потоке, статус читается из event loop."""
    tracker = eta.EtaTracker()
    plan = [_step(chars=50), _step(engine=XTTS, chars=50)]

    def worker() -> None:
        for _ in range(200):
            tracker.observe(F5, chars=50, render_sec=0.5)
            tracker.observe(XTTS, chars=50, render_sec=2.0, cold=True)
            tracker.estimate(plan)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert tracker.snapshot()[F5]["samples"] == 800
    assert tracker.estimate(plan) > 0


def test_render_plan_marks_first_chunk_cold_per_engine(monkeypatch, workspace):
    """Холодный старт — ровно первый кусок каждого ещё не поднятого движка."""
    engines = {F5: StubEngine(), XTTS: StubEngine()}
    _engines(monkeypatch, engines)
    monkeypatch.setattr(
        "backend.job_queue.created_engine", lambda engine_id: engines.get(engine_id)
    )
    _store(
        monkeypatch,
        {
            "voice1": _voice(workspace, "voice1", F5),
            "voice2": _voice(workspace, "voice2", XTTS),
        },
    )
    payload = _payload(count=1)
    payload.replicas = [
        Replica(voice="#1", text="раз", line_number=1),
        Replica(voice="#2", text="два", line_number=2),
        Replica(voice="#1", text="три", line_number=3),
    ]
    payload.speakers = {
        "#1": SpeakerSettings(voice_id="voice1"),
        "#2": SpeakerSettings(voice_id="voice2"),
    }
    queue = JobQueue()
    plan = queue._render_plan(payload)
    assert [(step.engine, step.cold) for step in plan] == [
        (F5, True), (XTTS, True), (F5, False)
    ]
    assert [step.chars for step in plan] == [3, 3, 3]

    # Прогретый движок холодным не считается: загрузку за него уже заплатили.
    engines[F5]._mark(STATE_READY)
    plan = queue._render_plan(payload)
    assert [(step.engine, step.cold) for step in plan] == [
        (F5, False), (XTTS, True), (F5, False)
    ]


def test_queue_plan_uses_replica_engine_override(monkeypatch, workspace):
    """Правка голоса у реплики меняет и движок в плане ETA, а не только звук."""
    engines = {F5: StubEngine(), XTTS: StubEngine()}
    _engines(monkeypatch, engines)
    _store(
        monkeypatch,
        {
            "voice1": _voice(workspace, "voice1", F5),
            "voice2": _voice(workspace, "voice2", XTTS),
        },
    )
    payload = _payload(count=1)
    payload.replicas = [
        Replica(voice="#1", text="раз", line_number=1, voice_id="voice2"),
        Replica(voice="#1", text="два", line_number=2),
    ]
    plan = JobQueue()._render_plan(payload)
    assert [step.engine for step in plan] == [XTTS, F5]
