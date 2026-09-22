"""Сравнение голоса на нескольких движках (ФАЗА 8).

Проверяется, что сравнение идёт через единственный воркер очереди, читает один и
тот же reference разными моделями, не смешивает ручки движков, не рушится от
недоступной модели и не переписывает reference при выборе движка. Модели
подменены заглушкой `StubEngine`: реальные веса в тестах не поднимаются.
"""

import asyncio
import contextlib
import os
import threading
import time

import httpx
import pytest
from conftest import StubEngine

from backend import audio_pipeline, benchmark, config, main
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines.base import EngineInfo, EngineParam
from backend.job_queue import JobPayload, JobQueue
from backend.voices_store import Voice, check_engine

TEXT = "Тестовая фраза для сравнения движков."
# Все объявленные в паспортах движки: сравнение по умолчанию берёт именно их,
# и список обязан совпадать с `ENGINE_INFOS`, иначе «сравнить все» молча
# пропускало бы движок.
ALL_ENGINES = ["f5", "xtts", "xtts-banana", "qwen3-tts", "kokoro-ru"]


@pytest.fixture(autouse=True)
def clean_registry():
    """Реестр запусков живёт в памяти модуля — каждый тест начинает с пустого."""
    benchmark.reset_registry()
    yield
    benchmark.reset_registry()


class _Store:
    """Хранилище голосов для роутов: без `voices.json`, но с той же правкой движка."""

    def __init__(self, *voices: Voice) -> None:
        self._voices = {voice.id: voice for voice in voices}

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)

    def update(
        self,
        voice_id: str,
        engine: str | None = None,
        engine_params: dict | None = None,
        preset: dict | None = None,
    ) -> Voice:
        voice = self._voices.get(voice_id)
        if voice is None:
            raise KeyError(voice_id)
        if engine:
            voice.engine = check_engine(engine)
            # Ручки прошлого движка к новому не относятся — как в боевом хранилище.
            if engine_params is None:
                voice.engine_params = {}
        if preset is not None:
            voice.preset = dict(preset)
        return voice


@pytest.fixture
def store(voice, monkeypatch):
    """Один голос для роутов и для пайплайна: подменяем оба хранилища."""
    holder = _Store(voice)
    monkeypatch.setattr(main, "get_store", lambda: holder)
    monkeypatch.setattr(audio_pipeline, "get_store", lambda: holder)
    return voice


@contextlib.asynccontextmanager
async def _client(monkeypatch):
    """Очередь и ASGI-клиент в одном event loop: воркер нельзя запускать в другом."""
    queue = JobQueue()
    await queue.start()
    monkeypatch.setattr(main, "get_queue", lambda: queue)
    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, queue
    finally:
        await queue.stop()


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def _start(client, *, voice_id: str = "voice1", **overrides) -> dict:
    payload = {"text": TEXT, **overrides}
    response = await client.post(f"/api/voices/{voice_id}/benchmark", json=payload)
    assert response.status_code == 202, response.text
    return response.json()


async def _completed(client, benchmark_id: str) -> dict:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        response = await client.get(f"/api/benchmarks/{benchmark_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"сравнение {benchmark_id} не завершилось")


def _render_payload(count: int = 2) -> JobPayload:
    return JobPayload(
        replicas=[
            Replica(voice="#1", text=f"Реплика {index}", line_number=index)
            for index in range(1, count + 1)
        ],
        speakers={"#1": SpeakerSettings(voice_id="voice1")},
        settings=RenderSettings(pause_ms=0, output_format="wav"),
    )


def _engine_map(*engines: StubEngine) -> dict:
    return {engine.info.id: engine for engine in engines}


# --- 1, 3: все выбранные движки и раздельные результаты ------------------------
def test_benchmark_runs_every_selected_engine(stub, store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=ALL_ENGINES)
            assert started["engines"] == ALL_ENGINES
            data = await _completed(client, started["benchmark_id"])

            assert data["status"] == "done"
            assert data["voice_id"] == "voice1"
            assert [item["engine"] for item in data["results"]] == ALL_ENGINES
            assert all(item["status"] == "done" for item in data["results"])
            # Строка на движок — один вызов модели, без повторов.
            assert len(stub.calls) == len(ALL_ENGINES)
            # Reference и фраза у всех движков одни и те же.
            assert {call["ref_audio_path"] for call in stub.calls} == {str(store.audio_path)}
            assert {call["ref_text"] for call in stub.calls} == {store.ref_text}
            assert len({call["text"] for call in stub.calls}) == 1
            assert TEXT in stub.calls[0]["text"]
            # Проверка не включалась — отметки нет.
            assert all(item["qa"] is None for item in data["results"])

    _run(scenario)


def test_default_engines_are_all_declared(stub, store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client)
            assert started["engines"] == ALL_ENGINES

    _run(scenario)


def _plain_engine(engine_id: str, seconds_per_char: float = 0.02) -> StubEngine:
    engine = StubEngine(seconds_per_char=seconds_per_char)
    engine.info = EngineInfo(
        id=engine_id,
        label=engine_id,
        description="",
        supports_accents=False,
    )
    return engine


def test_results_are_separated_per_engine(store, monkeypatch):
    """Каждый движок пишет свой файл: результаты не перепутаны между строками."""
    stubs = _engine_map(
        _plain_engine("f5", 0.02),
        _plain_engine("xtts", 0.04),
        _plain_engine("xtts-banana", 0.06),
    )
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: stubs[engine_id])

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=list(stubs))
            data = await _completed(client, started["benchmark_id"])

            paths = [
                benchmark.take_path(data["benchmark_id"], item["engine"])
                for item in data["results"]
            ]
            assert all(path.exists() for path in paths)
            assert len({path.name for path in paths}) == 3
            # Разная длина аудио у заглушек: файлы не могли достаться чужой строке.
            durations = [item["duration_sec"] for item in data["results"]]
            assert durations == sorted(durations)
            assert len({path.stat().st_size for path in paths}) == 3
            assert all(len(engine.calls) == 1 for engine in stubs.values())
            # Ссылка на файл ведёт именно к этому движку.
            for item in data["results"]:
                assert item["audio_url"].endswith(f"/{item['engine']}/audio")
                response = await client.get(item["audio_url"])
                assert response.status_code == 200
                assert response.headers["content-type"] == "audio/wav"
                assert response.content == benchmark.take_path(
                    data["benchmark_id"], item["engine"]
                ).read_bytes()

    _run(scenario)


# --- 2: недоступный движок не рушит сравнение ---------------------------------
class _LoadFailStub(StubEngine):
    def load(self) -> None:
        raise RuntimeError("веса не найдены")


class _SynthFailStub(StubEngine):
    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        raise RuntimeError("не хватило памяти")


def test_unavailable_engine_does_not_break_benchmark(store, monkeypatch):
    good = StubEngine()
    good.info = EngineInfo(id="f5", label="F5", description="", supports_accents=False)
    broken_load = _LoadFailStub()
    broken_load.info = EngineInfo(id="xtts", label="XTTS", description="", supports_accents=False)
    broken_synth = _SynthFailStub()
    broken_synth.info = EngineInfo(
        id="xtts-banana", label="Banana", description="", supports_accents=False
    )
    stubs = _engine_map(good, broken_load, broken_synth)
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: stubs[engine_id])

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=list(stubs))
            data = await _completed(client, started["benchmark_id"])
            # Сравнение состоялось: упавшие движки описаны, остальные на месте.
            assert data["status"] == "done"
            results = {item["engine"]: item for item in data["results"]}
            assert results["f5"]["status"] == "done"
            assert results["f5"]["audio_url"] is not None
            assert results["xtts"]["status"] == "error"
            assert "модель не поднялась" in results["xtts"]["error"]
            assert results["xtts"]["audio_url"] is None
            assert results["xtts-banana"]["status"] == "error"
            assert "синтез не удался" in results["xtts-banana"]["error"]
            assert len(good.calls) == 1

    _run(scenario)


# --- 4: время синтеза измеряется и не включает загрузку модели ----------------
class _SlowStub(StubEngine):
    def __init__(self, load_sec: float, synth_sec: float) -> None:
        super().__init__()
        self.load_sec = load_sec
        self.synth_sec = synth_sec

    def load(self) -> None:
        time.sleep(self.load_sec)
        super().load()

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        time.sleep(self.synth_sec)
        return super()._synthesize(text, ref_audio_path, ref_text, speed, params)


def test_render_sec_is_measured_without_model_load(store, monkeypatch):
    engine = _SlowStub(load_sec=0.4, synth_sec=0.12)
    engine.info = EngineInfo(id="f5", label="F5", description="", supports_accents=False)
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5"])
            data = await _completed(client, started["benchmark_id"])
            item = data["results"][0]
            assert item["status"] == "done"
            assert item["render_sec"] is not None
            # Синтез шёл не меньше своей задержки, но загрузка модели в замер не попала.
            assert item["render_sec"] >= 0.08
            assert item["render_sec"] < 0.35
            assert item["duration_sec"] > 0

    _run(scenario)


# --- 5: метаданные проверки сохраняются ---------------------------------------
def test_qa_metadata_is_saved(stub, store, monkeypatch):
    async def fake(chunk):
        return TEXT

    monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", fake)

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5"], qa="strict")
            data = await _completed(client, started["benchmark_id"])
            qa = data["results"][0]["qa"]
            assert qa is not None
            assert qa["status"] == audio_pipeline.QA_PASSED
            assert qa["mode"] == "strict"
            assert qa["wer"] is not None and qa["wer"] <= 0.15
            assert qa["attempts"] == 1

    _run(scenario)


# --- 6, 7: выбор движка и неизменный reference --------------------------------
def test_select_engine_updates_only_engine(stub, store, monkeypatch):
    voice = store
    voice.preset = {"speed": 1.1, "cfg_strength": 2.4}

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5", "xtts"])
            await _completed(client, started["benchmark_id"])
            calls_before_select = len(stub.calls)

            response = await client.post(
                f"/api/benchmarks/{started['benchmark_id']}/select", json={"engine": "xtts"}
            )
            assert response.status_code == 200, response.text
            updated = response.json()
            assert updated["engine"] == "xtts"
            # Всё остальное у голоса осталось как было.
            assert updated["ref_text"] == voice.ref_text
            assert updated["audio_file"] == voice.audio_file
            assert updated["preset"] == {"speed": 1.1, "cfg_strength": 2.4}
            # Выбор движка — это запись в voices.json, а не ещё один синтез.
            assert len(stub.calls) == calls_before_select

    _run(scenario)


def test_reference_is_not_overwritten(stub, store, monkeypatch):
    voice = store
    before_bytes = voice.audio_path.read_bytes()
    before_text = voice.ref_text

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5", "xtts"])
            await _completed(client, started["benchmark_id"])
            assert voice.ref_text == before_text
            assert voice.audio_file == "ref.wav"
            assert voice.audio_path.read_bytes() == before_bytes

            response = await client.post(
                f"/api/benchmarks/{started['benchmark_id']}/select", json={"engine": "xtts"}
            )
            assert response.status_code == 200
            assert voice.ref_text == before_text
            assert voice.audio_file == "ref.wav"
            assert voice.audio_path.read_bytes() == before_bytes

    _run(scenario)


# --- 8: параметры движков не смешиваются --------------------------------------
def _param_engine(engine_id: str, label: str, params: tuple) -> StubEngine:
    engine = StubEngine()
    engine.info = EngineInfo(
        id=engine_id,
        label=label,
        description="",
        supports_accents=False,
        params=params,
    )
    return engine


def test_engine_params_do_not_mix(store, monkeypatch):
    voice = store
    # Пресет общий для всех движков, ручки движка — только для своего.
    voice.preset = {"speed": 1.2}
    voice.engine_params = {"temperature": 0.4, "repetition_penalty": 1.7}
    f5 = _param_engine("f5", "F5", ())
    xtts = _param_engine(
        "xtts",
        "XTTS",
        (
            EngineParam("temperature", "Температура", 0.75, 0.05, 1.5, 0.05),
            EngineParam("repetition_penalty", "Штраф", 5.0, 1.0, 20.0, 0.5),
        ),
    )
    stubs = _engine_map(f5, xtts)
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: stubs[engine_id])

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5", "xtts"])
            data = await _completed(client, started["benchmark_id"])
            results = {item["engine"]: item for item in data["results"]}

            f5_params = f5.calls[0]["params"]
            xtts_params = xtts.calls[0]["params"]
            # Ручки XTTS не доехали до F5: у него их нет в паспорте.
            assert "temperature" not in f5_params
            assert "repetition_penalty" not in f5_params
            assert results["f5"]["params"]["engine_params"] == {}
            # И наоборот: объявленные ручки XTTS применились именно к нему.
            assert xtts_params["temperature"] == 0.4
            assert xtts_params["repetition_penalty"] == 1.7
            assert results["xtts"]["params"]["engine_params"] == {
                "temperature": 0.4,
                "repetition_penalty": 1.7,
            }
            # Пресет голоса — общий: сравнение идёт на одних настройках.
            assert f5.calls[0]["speed"] == 1.2
            assert xtts.calls[0]["speed"] == 1.2

    _run(scenario)


# --- очередь, статусы и ошибки запросов ---------------------------------------
class _GatedStub(StubEngine):
    """Заглушка, задерживающая первый синтез: им занят единственный воркер."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.started = threading.Event()

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        if not self.started.is_set():
            self.started.set()
            assert self.gate.wait(timeout=10.0)
        return super()._synthesize(text, ref_audio_path, ref_text, speed, params)


def test_benchmark_waits_for_busy_worker(store, monkeypatch):
    gated = _GatedStub()
    gated.info = EngineInfo(id="f5", label="F5", description="", supports_accents=False)
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: gated)

    async def scenario():
        async with _client(monkeypatch) as (client, queue):
            queue.submit(_render_payload())  # обычная задача занимает воркер
            assert await _wait_until(gated.started.is_set)

            started = await _start(client, engines=["f5"])
            await asyncio.sleep(0.15)
            run = benchmark.get_run(started["benchmark_id"])
            # Синтез в обход очереди не начался: запуск ждёт своей очереди.
            assert run is not None
            assert run.status == benchmark.RUN_QUEUED
            assert run.results == []
            assert len(gated.calls) == 0
            job = (await client.get(f"/api/jobs/{started['job_id']}")).json()
            assert job["status"] == "queued"

            gated.gate.set()
            data = await _completed(client, started["benchmark_id"])
            assert data["status"] == "done"
            assert len(data["results"]) == 1
            assert await _wait_until(
                lambda: (queue.get(started["job_id"]) is not None)
                and queue.get(started["job_id"]).message == "Сравнение готово"
            )

    _run(scenario)


def test_benchmark_status_matches_job(stub, store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=ALL_ENGINES)
            data = await _completed(client, started["benchmark_id"])
            job = (await client.get(f"/api/jobs/{started['job_id']}")).json()
            assert job["status"] == "done" and data["status"] == "done"
            assert job["total_replicas"] == len(data["engines"]) == len(data["results"])
            assert job["finished_at"] == data["finished_at"]

    _run(scenario)


def test_benchmark_request_errors_are_clear(stub, store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            missing = await client.post("/api/voices/нет-такой/benchmark", json={"text": TEXT})
            assert missing.status_code == 404
            assert missing.json()["detail"] == "Голос не найден"

            empty = await client.post("/api/voices/voice1/benchmark", json={"text": "   "})
            assert empty.status_code == 400
            assert "Пустой текст" in empty.json()["detail"]

            unknown = await client.post(
                "/api/voices/voice1/benchmark", json={"text": TEXT, "engines": ["f5", "нет-такой"]}
            )
            assert unknown.status_code == 400
            assert "Неизвестный движок" in unknown.json()["detail"]
            assert "нет-такой" in unknown.json()["detail"]

            too_long = await client.post(
                "/api/voices/voice1/benchmark",
                json={"text": "а" * (config.MAX_REPLICA_CHARS + 1)},
            )
            assert too_long.status_code == 400

            assert (await client.get("/api/benchmarks/нет-такой")).status_code == 404
            assert (await client.get("/api/benchmarks/нет-такой/f5/audio")).status_code == 404
            assert (
                await client.post("/api/benchmarks/нет-такой/select", json={"engine": "f5"})
            ).status_code == 404

    _run(scenario)


def test_select_validates_engine_and_result(stub, store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5"])
            data = await _completed(client, started["benchmark_id"])
            assert data["results"][0]["status"] == "done"

            unknown = await client.post(
                f"/api/benchmarks/{started['benchmark_id']}/select", json={"engine": "нет-такой"}
            )
            assert unknown.status_code == 400
            assert "Неизвестный движок" in unknown.json()["detail"]

            # Движок объявлен, но в этом сравнении не участвовал — выбирать нечего.
            absent = await client.post(
                f"/api/benchmarks/{started['benchmark_id']}/select", json={"engine": "xtts"}
            )
            assert absent.status_code == 400
            assert "не дал результата" in absent.json()["detail"]

            assert (await client.get(f"/api/benchmarks/{started['benchmark_id']}/nope/audio")).status_code == 404

    _run(scenario)


def test_benchmark_takes_survive_output_cleanup(stub, store, monkeypatch):
    """TTL-очистка output/ не трогает результаты сравнения."""
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            started = await _start(client, engines=["f5", "xtts"])
            data = await _completed(client, started["benchmark_id"])
            paths = [
                benchmark.take_path(data["benchmark_id"], item["engine"])
                for item in data["results"]
            ]
            stale = config.OUTPUT_DIR / "old-job.wav"
            stale.write_bytes(b"stale")
            old = time.time() - 48 * 3600
            os.utime(stale, (old, old))
            for path in paths:
                os.utime(path, (old, old))

            removed = audio_pipeline.cleanup_output(ttl_hours=24)
            assert removed >= 1
            assert not stale.exists()
            assert all(path.exists() for path in paths)

    _run(scenario)


def test_registry_prunes_old_runs_but_keeps_active(monkeypatch):
    monkeypatch.setattr(benchmark, "MAX_KEPT_BENCHMARKS", 2)

    def _register(run_id: str, status: str) -> benchmark.BenchmarkRun:
        run = benchmark.BenchmarkRun(
            id=run_id, voice_id="voice1", text=TEXT, engines=["f5"], status=status
        )
        return benchmark.register(run)

    _register("done1", benchmark.RUN_DONE)
    _register("done2", benchmark.RUN_DONE)
    active = _register("active", benchmark.RUN_RUNNING)
    _register("done3", benchmark.RUN_DONE)

    runs = [run.id for run in benchmark.list_runs()]
    assert len(runs) == 2
    assert active.id in runs
    assert benchmark.get_run("done1") is None
