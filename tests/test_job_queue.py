"""Очередь задач: сборка, перегенерация реплики, история вариантов и сиды.

Отдельный блок в конце — фаза 11: приоритеты и отмена. Приоритеты проверяются с
«занятым» воркером: пока первая задача держится у движка, остальные копятся в
очереди, и порядок их выхода виден однозначно — без гонок на скорость заглушки.
Отмена проверяется на настоящем пайплайне: движок-заглушка пишет вызовы, поэтому
видно, начался ли синтез вообще и на каком куске остановился.
"""

import asyncio
import contextlib
import threading
import time

import httpx
import pytest
from conftest import StubEngine

from backend import audio_pipeline, config, main
from backend.audio_pipeline import (
    QaSettings,
    RenderSettings,
    SpeakerSettings,
    read_variant,
)
from backend.dialogue_parser import Replica
from backend.job_queue import (
    PRIORITY_BACKGROUND,
    PRIORITY_PREVIEW,
    PRIORITY_RENDER,
    Job,
    JobPayload,
    JobQueue,
    JobStatus,
)


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


# --- ФАЗА 11: приоритетная очередь и отмена ------------------------------------
# Порядок задач проверяется на «занятом» воркере: пока первая задача держится у
# движка, остальные лежат в очереди, и их выход виден однозначно. Ждать по времени
# не нужно: заглушка сама сообщает, что вошла в синтез, и держит его, пока тест не
# отпустит. Отмена проверяется на том же пути: заглушка пишет вызовы, поэтому видно,
# начался ли синтез вообще и на каком куске он остановился.
class _Gate:
    """Ручной шлюз перед синтезом: тест сам решает, когда модель что-то увидит.

    `pass_through(n)` пропускает первые n вызовов и запирает следующий: так
    проверяется отмена между кусками — часть синтеза успевает состояться, а
    следующий кусок до движка не доходит.
    """

    def __init__(self) -> None:
        self.armed = False
        self.limit = 0
        self.passed = 0
        self.blocked = 0
        self.calls: list[str] = []
        self.only: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self, text: str) -> None:
        # `entered` отмечает приход в синтез всегда: по нему тест понимает, что
        # воркер занят задачей, даже когда шлюз открыт.
        self.entered.set()
        if not self.armed:
            self.passed += 1
            return
        # `only` адресует остановку конкретному тексту: у перегенерации до синтеза
        # ещё есть внутренний вызов — вырезание текущего куска в вариант.
        if self.only is not None and text != self.only:
            self.passed += 1
            return
        if self.passed >= self.limit:
            self.blocked += 1
            assert self.release.wait(10.0), "тест не отпустил синтез"
        self.passed += 1

    def pass_through(self, count: int = 0, only: str | None = None) -> None:
        """Пропустить `count` синтезов (можно — только с текстом `only`) и запереть следующий."""
        self.limit = count
        self.only = only
        self.passed = 0
        self.blocked = 0
        self.entered.clear()
        self.release.clear()
        self.armed = True

    def open(self) -> None:
        """Отпустить все синтезы и больше не задерживать."""
        self.armed = False
        self.release.set()


def _gated_engine() -> tuple[StubEngine, _Gate]:
    """Заглушка, которая спрашивает `_Gate` перед каждым синтезом."""
    engine = StubEngine()
    gate = _Gate()
    original = engine._synthesize

    def gated(text, ref_audio_path, ref_text, speed, params):
        gate.calls.append(text)
        gate.wait(text)
        return original(text, ref_audio_path, ref_text, speed, params)

    engine._synthesize = gated
    return engine, gate


def _use_engine(monkeypatch, engine: StubEngine) -> None:
    """Подменяет движок очереди на заглушку со шлюзом."""
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda _engine_id: engine)
    monkeypatch.setattr(audio_pipeline, "engine_for_id", lambda _engine_id: engine)
    monkeypatch.setattr(audio_pipeline, "_engine_for", lambda _voice: engine)


def _single(text: str) -> JobPayload:
    """Одна реплика с заданным текстом.

    Текст в заглушке нормализуется («Реплика номер 1» → «Реплика номер один»),
    поэтому порядок задач проверяется по словам, а не по цифрам.
    """
    return JobPayload(
        replicas=[Replica(voice="#1", text=text, line_number=1)],
        speakers={"#1": SpeakerSettings(voice_id="voice1")},
        settings=RenderSettings(pause_ms=0, output_format="wav"),
    )


def _chunks(*texts: str) -> JobPayload:
    """Несколько реплик в одной задаче — чтобы отмена была видна между кусками."""
    return JobPayload(
        replicas=[
            Replica(voice="#1", text=text, line_number=index)
            for index, text in enumerate(texts, start=1)
        ],
        speakers={"#1": SpeakerSettings(voice_id="voice1")},
        settings=RenderSettings(pause_ms=0, output_format="wav"),
    )


def _terminated(queue: JobQueue, job: Job) -> bool:
    """Задача закончилась (успехом, ошибкой или отменой) — воркер свободен."""
    return job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)


def test_preview_outruns_queued_batch(monkeypatch, stub, fake_store):
    """1. preview (приоритет 0) обгоняет обычный рендер (2), ждавший раньше."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        async with _queue() as queue:
            blocker = queue.submit(_single("Держу"))
            assert await _wait_until(gate.entered.is_set), "воркер не занят"
            batch = queue.submit(_single("Рендер"), priority=PRIORITY_RENDER)
            preview = queue.submit(_single("Прослушивание"), priority=PRIORITY_PREVIEW)
            await asyncio.sleep(0.2)  # дать воркеру разобрать очередь

            gate.open()
            assert await _wait_until(
                lambda: _terminated(queue, preview) and _terminated(queue, batch)
            )
            assert [job.status for job in (blocker, preview, batch)] == [JobStatus.DONE] * 3
            # preview был поставлен позже batch, но прошёл раньше него.
            assert gate.calls == ["Держу", "Прослушивание", "Рендер"]

    _run(scenario)


def test_regenerate_outruns_normal_render(monkeypatch, stub, fake_store):
    """2. перегенерация реплики (1) обгоняет обычный рендер (2)."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)
        async with _queue() as queue:
            done = queue.submit(_single("Исходная"))
            assert await _wait_until(lambda: done.status is JobStatus.DONE), done.error

            gate.calls.clear()
            queue.submit(_single("Держу"))
            assert await _wait_until(lambda: gate.blocked == 1), "воркер не занят"
            render = queue.submit(_single("Рендер"), priority=PRIORITY_RENDER)
            queue.submit_regenerate(done.id, 0)
            await asyncio.sleep(0.2)

            gate.open()
            assert await _wait_until(
                lambda: _terminated(queue, render) and done.regenerating is None
            )
            # Перегенерация поставлена позже рендера, но обошла его.
            assert gate.calls == ["Держу", "Исходная", "Рендер"]
            assert render.status is JobStatus.DONE
            assert done.status is JobStatus.DONE  # сам рендер остаётся готовым
            assert len(done.variants.get(0, [])) == 2  # исходный + новый

    _run(scenario)


def test_fifo_inside_one_priority(monkeypatch, stub, fake_store):
    """3. Внутри одного приоритета порядок строгий: кто раньше встал, тот раньше вышел."""
    engine, gate = _gated_engine()
    texts = ["Первый", "Второй", "Третий"]

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)
        async with _queue() as queue:
            # Первый задача занимает воркер, вторая запирается на входе, третья
            # ждёт в очереди — и выходит строго после второй.
            first = queue.submit(_single(texts[0]), priority=PRIORITY_RENDER)
            second = queue.submit(_single(texts[1]), priority=PRIORITY_RENDER)
            third = queue.submit(_single(texts[2]), priority=PRIORITY_RENDER)
            assert await _wait_until(lambda: gate.blocked == 1), "вторая задача не дошла"
            assert third.status is JobStatus.QUEUED
            assert first.status is JobStatus.DONE

            gate.open()
            assert await _wait_until(
                lambda: _terminated(queue, first)
                and _terminated(queue, second)
                and _terminated(queue, third)
            )
            assert gate.calls == texts

    _run(scenario)


def test_queued_job_is_cancelled_before_start(monkeypatch, stub, fake_store):
    """4-5. Ожидающая задача отменяется сразу и никогда не доходит до движка."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        async with _queue() as queue:
            blocker = queue.submit(_single("Держу"))
            assert await _wait_until(gate.entered.is_set), "воркер не занят"
            waiting = queue.submit(_single("Ожидание"), priority=PRIORITY_RENDER)
            await asyncio.sleep(0.2)

            cancelled = queue.cancel(waiting.id)
            assert cancelled is waiting
            assert waiting.status is JobStatus.CANCELLED
            assert "до запуска" in waiting.message
            assert waiting.cancel_requested is True
            # Отменённая задача не исчезает: её видно тем же `get`.
            assert queue.get(waiting.id) is waiting

            gate.open()
            assert await _wait_until(lambda: _terminated(queue, blocker))
            await asyncio.sleep(0.1)  # воркер доходит до отменённой задачи в очереди
            assert waiting.status is JobStatus.CANCELLED
            assert gate.calls == ["Держу"]  # движок её так и не увидел

    _run(scenario)


def test_processing_job_is_cancelled_at_safe_point(monkeypatch, stub, fake_store):
    """6. Идущая задача отменяется между кусками: синтез успел частично, поток цел."""
    engine, gate = _gated_engine()
    texts = ["Первый кусок", "Второй кусок", "Третий кусок"]

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)  # первый кусок проходит, второй запирается
        async with _queue() as queue:
            job = queue.submit(_chunks(*texts), priority=PRIORITY_RENDER)
            assert await _wait_until(lambda: gate.blocked == 1), "второй кусок не дошёл"
            assert job.current_replica >= 1, "первый кусок не готов"

            queue.cancel(job.id)
            assert job.cancel_requested is True  # отмена видна сразу, до безопасной точки
            assert job.status is JobStatus.PROCESSING
            assert "безопасной точке" in job.message

            gate.open()
            assert await _wait_until(lambda: job.status is JobStatus.CANCELLED)
            # Прервались между кусками, а не внутри вызова модели: второй кусок
            # начался и завершился, третий не начинался.
            assert gate.calls == texts[:2]
            assert queue.current_job_id is None
            assert queue.is_busy() is False

    _run(scenario)


def test_cancelled_job_does_not_damage_next_one(monkeypatch, stub, fake_store):
    """7. После отмены воркер в чистом состоянии: следующая задача проходит целиком."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)  # первый кусок проходит, на втором задачу отменяем
        async with _queue() as queue:
            first = queue.submit(_chunks("Отменяемая раз", "Отменяемая два", "Отменяемая три"))
            assert await _wait_until(lambda: gate.blocked == 1), "второй кусок не дошёл"
            assert first.current_replica >= 1, "первый кусок не готов"
            queue.cancel(first.id)
            gate.open()
            assert await _wait_until(lambda: first.status is JobStatus.CANCELLED)

            # Следующая задача не наследует ни отмену, ни флаг прерывания.
            second = queue.submit(_payload())
            assert await _wait_until(lambda: second.status is JobStatus.DONE), second.error
            assert second.error is None
            assert second.cancel_requested is False
            assert len(second.segments) == 3
            assert second.output_path is not None and second.output_path.exists()
            assert queue.current_job_id is None
            assert queue.get(first.id) is first  # отменённая задача никуда не исчезла
            assert gate.calls == ["Отменяемая раз", "Отменяемая два",
                                  "Реплика номер один",
                                  "Реплика номер два", "Реплика номер три"]

    _run(scenario)


def test_cancelled_render_leaves_no_partial_output(monkeypatch, stub, fake_store):
    """8. Недописанный вывод задачи вычищается, транзитных файлов не остаётся."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)
        async with _queue() as queue:
            job = queue.submit(_chunks("Раз", "Два", "Три"), priority=PRIORITY_RENDER)
            assert await _wait_until(lambda: gate.blocked == 1)
            expected = config.OUTPUT_DIR / f"{job.id}.wav"

            queue.cancel(job.id)
            gate.open()
            assert await _wait_until(lambda: job.status is JobStatus.CANCELLED)
            assert not expected.exists()
            assert not list(config.OUTPUT_DIR.glob(f"{job.id}*"))
            assert job.output_path is None

    _run(scenario)


def test_cancelled_take_drops_partial_and_keeps_previous_sound(monkeypatch, stub, fake_store):
    """Частичный take удаляется: исходный файл и прежние варианты остаются целы."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        async with _queue() as queue:
            job = queue.submit(_single("Исходная"))
            assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
            original = list(job.segments)
            before = sorted(config.OUTPUT_DIR.glob(f"{job.id}*"))

            # Первый вызов перегенерации — вырезание текущего куска в вариант
            # «исходный» (`_keep_current`). Шлюз держит именно его, а отмена
            # приходит, когда этот вариант уже на диске: так проверяется, что
            # отменённый пересинтез не оставляет за собой ни аудио, ни истории.
            gate.pass_through(0, only="Исходная")
            queue.submit_regenerate(job.id, 0)
            assert await _wait_until(lambda: gate.blocked == 1), "синтез take не начался"
            assert job.cancel_requested is False
            queue.cancel(job.id)
            # Отмена видна сразу, ещё до безопасной точки: задача остаётся
            # готовым рендером, но пересборка помечена отменённой.
            assert job.cancel_requested is True
            assert job.regen_error == "Отменено"
            gate.open()
            assert await _wait_until(lambda: job.regenerating is None)

            # Рендер остаётся готовым: отмена не превращает готовый файл в ошибку.
            assert job.status is JobStatus.DONE
            assert job.regen_error == "Отменено"
            assert job.variants.get(0, []) == []
            assert job.segments == original
            assert job.output_path is not None and job.output_path.exists()
            # Транзитных файлов от отменённой перегенерации не осталось: даже
            # «исходный» вариант, уже записанный на диск, удалён вместе с историей,
            # которой не было, а готовый файл не тронут.
            assert sorted(config.OUTPUT_DIR.glob(f"{job.id}*")) == before
            assert (config.OUTPUT_DIR / f"{job.id}.wav").exists()

    _run(scenario)


def test_repeated_cancel_is_idempotent(monkeypatch, stub, fake_store):
    """10. Повторная отмена — не ошибка; завершённую задачу отменять нечего."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        async with _queue() as queue:
            blocker = queue.submit(_single("Держу"))
            assert await _wait_until(gate.entered.is_set)
            queued = queue.submit(_single("Ожидание"), priority=PRIORITY_RENDER)
            await asyncio.sleep(0.2)

            first = queue.cancel(queued.id)
            second = queue.cancel(queued.id)
            assert first is queued and second is queued
            assert queued.status is JobStatus.CANCELLED
            assert second.to_dict() == first.to_dict()

            gate.open()
            assert await _wait_until(lambda: blocker.status is JobStatus.DONE)

            # Готовую задачу отмена не портит: статус и файл остаются на месте.
            path = blocker.output_path
            result = queue.cancel(blocker.id)
            assert result is blocker
            assert blocker.status is JobStatus.DONE
            assert blocker.error is None
            assert path is not None and path.exists()

            assert queue.cancel("нет-такой") is None  # неизвестный id — не задача

    _run(scenario)


def test_cancelled_job_does_not_keep_engine_busy(monkeypatch, stub, fake_store):
    """Отменённая задача не висит в `is_busy()`: выгрузка движка не залипает."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        async with _queue() as queue:
            # Пока воркер занят первой задачей, вторая точно остаётся в очереди.
            blocker = queue.submit(_single("Держу"))
            assert await _wait_until(gate.entered.is_set)
            waiting = queue.submit(_single("Ожидание"), priority=PRIORITY_RENDER)
            await asyncio.sleep(0.2)
            assert waiting.status is JobStatus.QUEUED
            assert queue.is_busy() is True

            queue.cancel(waiting.id)
            assert waiting.status is JobStatus.CANCELLED
            # «Занято» остаётся только из-за первой, реально идущей задачи.
            assert queue.is_busy() is True
            gate.open()
            assert await _wait_until(lambda: blocker.status is JobStatus.DONE)
            assert queue.is_busy() is False

    _run(scenario)


def test_cancel_does_not_touch_previous_takes_of_project(stub, fake_store, monkeypatch):
    """Отмена рендера проекта не портит уже сохранённые куски реплик."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        queue = JobQueue()
        await queue.start()
        monkeypatch.setattr(main, "get_queue", lambda: queue)
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                project = (
                    await client.post(
                        "/api/projects",
                        json={
                            "name": "Отмена",
                            "source_text": "ИВАН: Раз.\nИВАН: Два.",
                            "mode": "dialogue",
                        },
                    )
                ).json()
                await client.patch(
                    f"/api/projects/{project['id']}",
                    json={"speakers": {"ИВАН": {"voice_id": "voice1"}}},
                )
                await client.post(f"/api/projects/{project['id']}/parse", json={})

                # Первый рендер доводим до конца — у проекта появляются куски.
                accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
                first_id = accepted.json()["job_id"]
                assert await _wait_until(
                    lambda: queue.get(first_id).status is JobStatus.DONE
                ), queue.get(first_id).error
                project = (await client.get(f"/api/projects/{project['id']}")).json()
                takes = [replica["takes"] for replica in project["replicas"]]
                assert all(items for items in takes)

                # Второй рендер отменяем: статус проекта возвращается в согласованный
                # `draft`, а сохранённые куски остаются на месте.
                gate.pass_through(0)
                second = await client.post(f"/api/projects/{project['id']}/render", json={})
                second_id = second.json()["job_id"]
                assert await _wait_until(gate.entered.is_set)
                cancelled = await client.post(f"/api/jobs/{second_id}/cancel")
                assert cancelled.status_code == 200, cancelled.text
                gate.open()
                assert await _wait_until(
                    lambda: queue.get(second_id).status is JobStatus.CANCELLED
                )
                # Статус проекта пишется в том же обработчике отдельной записью в
                # базу — даём воркеру её закончить.
                await asyncio.sleep(0.1)

                project = (await client.get(f"/api/projects/{project['id']}")).json()
                assert project["status"] == config.PROJECT_STATUS_DRAFT
                assert [replica["takes"] for replica in project["replicas"]] == takes
        finally:
            gate.open()
            await queue.stop()

    _run(scenario)


def test_cancel_route_statuses_and_404(stub, fake_store, monkeypatch):
    """9. Статус `cancelled` сохраняется и виден через REST; неизвестный id — 404."""
    engine, gate = _gated_engine()

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        queue = JobQueue()
        await queue.start()
        monkeypatch.setattr(main, "get_queue", lambda: queue)
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Ожидающая задача: отмена сразу переводит её в `cancelled`.
                blocker = queue.submit(_single("Держу"))
                assert await _wait_until(gate.entered.is_set)
                waiting = queue.submit(_single("Ожидание"), priority=PRIORITY_RENDER)
                await asyncio.sleep(0.2)

                response = await client.post(f"/api/jobs/{waiting.id}/cancel")
                assert response.status_code == 200, response.text
                body = response.json()["job"]
                assert body["status"] == "cancelled"
                assert body["cancel_requested"] is True
                assert body["finished_at"] is not None

                # Отменённая задача не исчезает: её видно тем же GET, что и раньше.
                status = await client.get(f"/api/jobs/{waiting.id}")
                assert status.status_code == 200
                assert status.json()["status"] == "cancelled"

                again = await client.post(f"/api/jobs/{waiting.id}/cancel")
                assert again.status_code == 200
                assert again.json()["job"]["status"] == "cancelled"

                gate.open()
                assert await _wait_until(lambda: blocker.status is JobStatus.DONE)
                done = await client.post(f"/api/jobs/{blocker.id}/cancel")
                assert done.status_code == 200
                assert done.json()["job"]["status"] == "done"  # идемпотентно

                missing = await client.post("/api/jobs/нет-такой/cancel")
                assert missing.status_code == 404
                assert "не найдена" in missing.json()["detail"]
        finally:
            gate.open()
            await queue.stop()

    _run(scenario)


def test_background_render_goes_to_lowest_priority(monkeypatch, stub, fake_store):
    """`background=True` опускает рендер в приоритет 3, не ломая обычный запуск."""
    seen: list[int] = []
    engine, gate = _gated_engine()
    original = JobQueue.submit

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(0)
        async with _queue() as queue:
            def recording(payload, priority=PRIORITY_RENDER):
                seen.append(priority)
                return original(queue, payload, priority=priority)

            monkeypatch.setattr(queue, "submit", recording)
            blocker = queue.submit(_single("Держу"))
            assert await _wait_until(gate.entered.is_set)
            # Фоновый рендер ждёт; обычный — обгоняет его, хотя поставлен позже.
            background = queue.submit(_payload(count=1), priority=PRIORITY_BACKGROUND)
            batch = queue.submit(_payload(count=1), priority=PRIORITY_RENDER)
            await asyncio.sleep(0.2)

            gate.open()
            assert await _wait_until(
                lambda: _terminated(queue, batch) and _terminated(queue, background)
            )
            assert blocker.status is JobStatus.DONE
            assert seen == [PRIORITY_RENDER, PRIORITY_BACKGROUND, PRIORITY_RENDER]

    _run(scenario)


def test_render_requests_deliver_background_and_priority(stub, fake_store, monkeypatch):
    """Поле `background` у рендер-запросов доезжает до очереди приоритетом 3."""
    seen: list[tuple[str, int]] = []

    async def scenario():
        queue = JobQueue()
        await queue.start()
        original = queue.submit

        def recording(payload, priority=PRIORITY_RENDER):
            seen.append((payload.replicas[0].text, priority))
            return original(payload, priority=priority)

        monkeypatch.setattr(queue, "submit", recording)
        monkeypatch.setattr(main, "get_queue", lambda: queue)
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Старый вызов без поля `background` остаётся обычным рендером.
                plain = await client.post(
                    "/api/render-text",
                    json={"voice_id": "voice1", "text": "Обычный текст", "output_format": "wav"},
                )
                assert plain.status_code == 202, plain.text
                background_text = await client.post(
                    "/api/render-text",
                    json={
                        "voice_id": "voice1",
                        "text": "Фоновый текст",
                        "output_format": "wav",
                        "background": True,
                    },
                )
                assert background_text.status_code == 202, background_text.text
                dialogue = await client.post(
                    "/api/generate",
                    json={
                        "dialogue_text": "ИВАН: Фоновая реплика",
                        "speakers": {"ИВАН": {"voice_id": "voice1"}},
                        "output_format": "wav",
                        "background": True,
                    },
                )
                assert dialogue.status_code == 202, dialogue.text
                await asyncio.sleep(0.3)
        finally:
            await queue.stop()
        assert seen == [
            ("Обычный текст", PRIORITY_RENDER),
            ("Фоновый текст", PRIORITY_BACKGROUND),
            ("Фоновая реплика", PRIORITY_BACKGROUND),
        ]

    _run(scenario)


def test_old_api_paths_still_work_with_priorities(stub, fake_store, monkeypatch):
    """Приоритеты не сломали старые сценарии: generate, render-text и preview доходят."""
    async def scenario():
        queue = JobQueue()
        await queue.start()
        monkeypatch.setattr(main, "get_queue", lambda: queue)
        transport = httpx.ASGITransport(app=main.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                dialogue = await client.post(
                    "/api/generate",
                    json={
                        "dialogue_text": "ИВАН: Первая реплика.\nМАРГО: Вторая реплика.",
                        "speakers": {
                            "ИВАН": {"voice_id": "voice1"},
                            "МАРГО": {"voice_id": "voice1"},
                        },
                        "output_format": "wav",
                    },
                )
                assert dialogue.status_code == 202, dialogue.text
                job = queue.get(dialogue.json()["job_id"])
                assert await _wait_until(lambda: job.status is JobStatus.DONE), job.error
                assert job.output_path is not None and job.output_path.exists()

                preview = await client.post(
                    "/api/preview", json={"voice_id": "voice1", "text": "Проверка"}
                )
                assert preview.status_code == 202, preview.text
                preview_job = queue.get(preview.json()["job_id"])
                assert await _wait_until(lambda: preview_job.status is JobStatus.DONE)

                text = await client.post(
                    "/api/render-text",
                    json={"voice_id": "voice1", "text": "Сплошной текст", "output_format": "wav"},
                )
                assert text.status_code == 202, text.text
                text_job = queue.get(text.json()["job_id"])
                assert await _wait_until(lambda: text_job.status is JobStatus.DONE), text_job.error
        finally:
            await queue.stop()

    _run(scenario)


def test_watchdog_interrupt_is_an_error_not_a_cancellation(monkeypatch, stub, fake_store):
    """Прерывание watchdog'ом — ошибка с причиной, а не тихое «отменено».

    Сигнал в пайплайн один (`cancel_requested`), и до этой проверки прерывание
    по памяти выглядело для пользователя как его собственная отмена: причина
    «превышен лимит памяти» терялась, а проект возвращался в черновик. Причина
    важнее факта — иначе непонятно, почему длинный рендер остановился сам.
    """
    engine, gate = _gated_engine()
    texts = ["Первый кусок", "Второй кусок", "Третий кусок"]

    async def scenario():
        _use_engine(monkeypatch, engine)
        gate.pass_through(1)
        async with _queue() as queue:
            job = queue.submit(_chunks(*texts), priority=PRIORITY_RENDER)
            assert await _wait_until(lambda: gate.blocked == 1), "второй кусок не дошёл"

            assert queue.request_abort("превышен лимит памяти") == job.id
            gate.open()
            assert await _wait_until(lambda: job.status is not JobStatus.PROCESSING)

            assert job.status is JobStatus.ERROR
            assert job.error == "превышен лимит памяти"
            assert job.message == "Прервано"
            assert "превышен лимит памяти" in job.message or job.error
            # Частичный файл убран, воркер свободен, следующая задача не задета.
            assert job.output_path is None or not job.output_path.exists()
            assert queue.current_job_id is None
            assert queue.is_busy() is False

            follow_up = queue.submit(_payload(count=2))
            assert await _wait_until(lambda: follow_up.status is JobStatus.DONE), follow_up.error
            assert follow_up.cancel_requested is False

    _run(scenario)
