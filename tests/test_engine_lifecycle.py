"""Жизненный цикл движков (ФАЗА 10): load/unload, простой и REST выгрузки.

Реальные модели не поднимаются: везде `StubEngine` из conftest, каталоги уведены
в `tmp_path`. Проверяется настоящее поведение базового класса, реестра, очереди и
роутов — подменена только сама модель.
"""

import asyncio
import contextlib
import logging
import sys
import threading
import time
import types

import httpx
import pytest
from conftest import STUB_ENGINE_ID, StubEngine
from test_api import _client, _generate, _wait

from backend import config, engine_lifecycle, main, resource_guard
from backend.engines import base as engines_base
from backend.engines import registry
from backend.engines.base import (
    ENGINE_XTTS,
    STATE_FAILED,
    STATE_IDLE,
    STATE_READY,
    EngineBusyError,
)
from backend.job_queue import Job, JobQueue, JobStatus

SAMPLE_RATE = 24_000


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    """Реестр движков пуст на входе: тесты не должны видеть чужие экземпляры."""
    monkeypatch.setattr(registry, "_instances", {})


@pytest.fixture(autouse=True)
def no_real_torch(monkeypatch):
    """`config._torch` — заглушка без MPS: `/api/status` не импортирует настоящий torch.

    Иначе первый же запрос статуса тянул бы металл на секунды, а `mps_memory`
    зависел бы от того, успел ли его импортировать предыдущий тест.
    """
    monkeypatch.setattr(
        config,
        "_torch",
        types.SimpleNamespace(
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        ),
    )


@contextlib.asynccontextmanager
async def _plain_client():
    """Клиент без очереди: роутам выгрузки и `/api/status` она не нужна."""
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


# --- 1-3. load / unload / повторный unload ------------------------------------
class _RecordingEngine(StubEngine):
    """Заглушка, считающая вызовы `_release`: по ним видно, что хук дошёл."""

    def __init__(self) -> None:
        super().__init__()
        self.releases = 0

    def _release(self) -> None:
        self.releases += 1


def test_load_then_unload_and_repeated_unload_is_safe():
    engine = _RecordingEngine()
    assert engine.state == STATE_IDLE and not engine.is_loaded

    engine.load()
    assert engine.is_loaded and engine.state == STATE_READY

    engine.unload()
    assert not engine.is_loaded and engine.state == STATE_IDLE
    assert engine.releases == 1

    # Повторная выгрузка незагруженного движка — безопасный no-op.
    engine.unload()
    assert engine.state == STATE_IDLE
    assert engine.releases == 2


# --- 4. synthesize после unload снова поднимает модель -------------------------
def test_synthesize_after_unload_reloads_model():
    engine = StubEngine()
    engine.load()
    engine.unload()
    assert engine.loads == 1

    waveform, rate = engine.synthesize("Привет", "/tmp/ref.wav", "Привет")

    assert engine.loads == 2  # synthesize сам вызвал load()
    assert engine.is_loaded and engine.state == STATE_READY
    assert rate == SAMPLE_RATE
    assert len(waveform) > 0
    assert engine.active_synthesizes == 0


# --- 5. нельзя выгрузить во время активного синтеза ---------------------------
class _BlockingEngine(StubEngine):
    """Синтез, который висит, пока тест не разрешит: окно для попытки unload."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.finish = threading.Event()

    def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
        self.started.set()
        assert self.finish.wait(5.0), "синтез не дождался разрешения"
        return super()._synthesize(text, ref_audio_path, ref_text, speed, params)


def test_unload_refused_while_synthesis_is_active():
    engine = _BlockingEngine()
    engine.load()
    outcome: dict = {}

    def synthesize():
        outcome["value"] = engine.synthesize("Привет", "/tmp/ref.wav", "Привет")

    thread = threading.Thread(target=synthesize, daemon=True)
    thread.start()
    assert engine.started.wait(5.0)
    assert engine.active_synthesizes == 1

    with pytest.raises(EngineBusyError, match="синтезирует"):
        engine.unload()

    # Модель осталась поднятой, а синтез дошёл до конца.
    assert engine.is_loaded
    engine.finish.set()
    thread.join(5.0)
    assert not thread.is_alive()
    assert outcome["value"] is not None
    assert engine.active_synthesizes == 0

    engine.unload()
    assert not engine.is_loaded


def test_unload_route_refuses_while_synthesizing(monkeypatch):
    engine = _BlockingEngine()
    engine.load()
    monkeypatch.setitem(registry._instances, ENGINE_XTTS, engine)
    thread = threading.Thread(
        target=lambda: engine.synthesize("Привет", "/tmp/ref.wav", "Привет"), daemon=True
    )
    thread.start()
    try:
        assert engine.started.wait(5.0)

        async def scenario():
            async with _plain_client() as client:
                response = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
                assert response.status_code == 409, response.text
                assert "синтезирует" in response.json()["detail"]
                assert engine.is_loaded

        _run(scenario)
    finally:
        engine.finish.set()
        thread.join(5.0)


# --- 6. политика простоя ------------------------------------------------------
def test_idle_policy_decision_is_pure_and_explicit():
    now = 1000.0
    base = {
        "policy_minutes": 15.0,
        "last_used_at": now - 60,
        "now": now,
        "loaded": True,
        "busy": False,
        "queue_busy": False,
    }

    assert engine_lifecycle.decide_idle_unload(**base) == (
        False, engine_lifecycle.REASON_TOO_EARLY
    )

    idle = {**base, "last_used_at": now - 15 * 60}
    assert engine_lifecycle.decide_idle_unload(**idle) == (True, engine_lifecycle.REASON_IDLE)
    assert engine_lifecycle.decide_idle_unload(
        **{**idle, "policy_minutes": 0.0}
    ) == (False, engine_lifecycle.REASON_DISABLED)
    assert engine_lifecycle.decide_idle_unload(
        **{**idle, "busy": True}
    ) == (False, engine_lifecycle.REASON_BUSY)
    assert engine_lifecycle.decide_idle_unload(
        **{**idle, "queue_busy": True}
    ) == (False, engine_lifecycle.REASON_QUEUE_BUSY)
    assert engine_lifecycle.decide_idle_unload(
        **{**idle, "loaded": False}
    ) == (False, engine_lifecycle.REASON_NOT_LOADED)


def test_idle_minutes_are_read_from_env_at_call_time(monkeypatch):
    monkeypatch.delenv(config.ENGINE_IDLE_UNLOAD_ENV, raising=False)
    assert config.engine_idle_unload_minutes() == config.DEFAULT_ENGINE_IDLE_UNLOAD_MIN

    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "0")
    assert config.engine_idle_unload_minutes() == 0.0

    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "2.5")
    assert config.engine_idle_unload_minutes() == 2.5

    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "мусор")
    assert config.engine_idle_unload_minutes() == config.DEFAULT_ENGINE_IDLE_UNLOAD_MIN


def test_idle_watcher_unloads_only_after_threshold(monkeypatch):
    engine = StubEngine()
    engine.load()
    monkeypatch.setattr(engine, "_last_used_at", 1000.0)
    clock = {"now": 1000.0}
    watcher = engine_lifecycle.IdleUnloadWatcher(
        engines=lambda: {STUB_ENGINE_ID: engine}, now=lambda: clock["now"]
    )
    # Очередь не поднимаем: решение по простою проверяется отдельно от её состояния.
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: False)
    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "15")

    clock["now"] = 1000.0 + 14 * 60
    assert asyncio.run(watcher.sweep()) == []
    assert engine.is_loaded

    clock["now"] = 1000.0 + 15 * 60
    assert asyncio.run(watcher.sweep()) == [STUB_ENGINE_ID]
    assert not engine.is_loaded


def test_idle_policy_disabled_never_unloads(monkeypatch):
    engine = StubEngine()
    engine.load()
    monkeypatch.setattr(engine, "_last_used_at", 0.0)
    watcher = engine_lifecycle.IdleUnloadWatcher(
        engines=lambda: {STUB_ENGINE_ID: engine}, now=lambda: 10 ** 9
    )
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: False)
    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "0")

    assert asyncio.run(watcher.sweep()) == []
    assert engine.is_loaded


def test_idle_watcher_skips_busy_queue(monkeypatch):
    engine = StubEngine()
    engine.load()
    monkeypatch.setattr(engine, "_last_used_at", 0.0)
    watcher = engine_lifecycle.IdleUnloadWatcher(
        engines=lambda: {STUB_ENGINE_ID: engine}, now=lambda: 10 ** 9
    )
    monkeypatch.setenv(config.ENGINE_IDLE_UNLOAD_ENV, "15")
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: True)

    assert asyncio.run(watcher.sweep()) == []
    assert engine.is_loaded


def test_queue_is_busy_covers_queued_processing_and_regenerate():
    queue = JobQueue()
    assert queue.is_busy() is False

    queue._jobs["a"] = Job(id="a", status=JobStatus.QUEUED)
    assert queue.is_busy() is True

    queue._jobs["a"].status = JobStatus.DONE
    assert queue.is_busy() is False

    queue._jobs["b"] = Job(id="b", status=JobStatus.PROCESSING)
    assert queue.is_busy() is True

    queue._jobs["b"].status = JobStatus.DONE
    queue._jobs["c"] = Job(id="c", regenerating=0)
    assert queue.is_busy() is True  # перегенерация ещё в очереди

    queue._jobs["c"].regenerating = None
    queue._current_job_id = "c"
    assert queue.is_busy() is True


# --- 7. реестр не отдаёт устаревшее состояние ----------------------------------
def test_registry_does_not_cache_stale_state(monkeypatch):
    engine = StubEngine()
    registry._instances[STUB_ENGINE_ID] = engine
    engine.load()
    assert registry.created_engines()[STUB_ENGINE_ID].is_loaded is True
    assert registry.created_engine(STUB_ENGINE_ID) is engine

    engine.unload()

    snapshot = registry.created_engines()[STUB_ENGINE_ID]
    assert snapshot.is_loaded is False
    assert snapshot.state == STATE_IDLE
    assert snapshot.to_dict()["state"] == STATE_IDLE

    async def scenario():
        async with _plain_client() as client:
            body = (await client.get("/api/status")).json()
            entry = next(item for item in body["engines"] if item["id"] == STUB_ENGINE_ID)
            assert entry["state"] == STATE_IDLE

    _run(scenario)


# --- 8. ошибка выгрузки не убивает backend ------------------------------------
class _BrokenReleaseEngine(StubEngine):
    def _release(self) -> None:
        raise RuntimeError("металл не отвечает")


def test_release_error_is_logged_and_engine_recovers(caplog):
    engine = _BrokenReleaseEngine()
    engine.load()

    with (
        caplog.at_level(logging.ERROR, logger="backend.engines.base"),
        pytest.raises(RuntimeError, match="металл"),
    ):
        engine.unload()

    assert engine.state == STATE_FAILED
    assert not engine.is_loaded
    assert engine.last_error is not None and "металл" in engine.last_error
    assert "Не удалось выгрузить" in caplog.text

    # Состояние согласовано: движок снова поднимается и синтезирует.
    engine.load()
    assert engine.is_loaded
    waveform, _ = engine.synthesize("Привет", "/tmp/ref.wav", "Привет")
    assert len(waveform) > 0


def test_unload_route_reports_release_error_and_service_survives(monkeypatch):
    engine = _BrokenReleaseEngine()
    engine.load()
    monkeypatch.setitem(registry._instances, ENGINE_XTTS, engine)

    async def scenario():
        async with _plain_client() as client:
            response = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
            assert response.status_code == 500, response.text
            assert "Не удалось выгрузить" in response.json()["detail"]

            # Сервис жив: следующий запрос отвечает, движок можно поднять снова.
            status = await client.get("/api/status")
            assert status.status_code == 200
            loaded = await client.post(f"/api/engines/{ENGINE_XTTS}/load")
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["loaded"] is True

    _run(scenario)


# --- 9. очередь продолжает работать после unload/reload ------------------------
def test_queue_keeps_working_after_unload_and_reload(stub, fake_store, monkeypatch):
    monkeypatch.setitem(registry._instances, ENGINE_XTTS, stub)

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            first_id = await _generate(client)
            await _wait(client, first_id, lambda item: item["status"] == "done")
            assert stub.is_loaded

            unloaded = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
            assert unloaded.status_code == 200, unloaded.text
            assert unloaded.json()["unloaded"] is True
            assert not stub.is_loaded
            loads_after_unload = stub.loads

            # Следующий рендер сам поднимает модель: выгрузка не ломает очередь.
            second_id = await _generate(client)
            data = await _wait(client, second_id, lambda item: item["status"] == "done")
            assert stub.loads > loads_after_unload
            assert stub.is_loaded
            audio = await client.get(f"/api/jobs/{second_id}/audio")
            assert audio.status_code == 200
            assert len(audio.content) > 0
            assert data["status"] == "done"

    _run(scenario)


# --- REST: 200 / 404 / 409 и /api/status без MPS -------------------------------
def test_engine_unload_and_load_routes(stub, monkeypatch):
    monkeypatch.setitem(registry._instances, ENGINE_XTTS, stub)
    stub.load()

    async def scenario():
        async with _plain_client() as client:
            unloaded = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
            assert unloaded.status_code == 200, unloaded.text
            body = unloaded.json()
            assert body["unloaded"] is True and body["state"] == STATE_IDLE
            assert "память освобождена" in body["message"]
            assert not stub.is_loaded

            loaded = await client.post(f"/api/engines/{ENGINE_XTTS}/load")
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["loaded"] is True and stub.is_loaded

            # Повторная выгрузка уже выгруженного — успешный no-op.
            again = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
            assert again.status_code == 200
            again_body = again.json()
            assert again_body["unloaded"] is True
            assert again.json()["state"] == STATE_IDLE

    _run(scenario)


def test_engine_lifecycle_routes_reject_unknown_engine():
    async def scenario():
        async with _plain_client() as client:
            for url in ("/api/engines/нет-такого/unload", "/api/engines/нет-такого/load"):
                response = await client.post(url)
                assert response.status_code == 404, (url, response.text)
                assert "не найден" in response.json()["detail"]

    _run(scenario)


def test_unload_route_refuses_while_queue_busy(stub, monkeypatch):
    monkeypatch.setitem(registry._instances, ENGINE_XTTS, stub)
    stub.load()
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: True)

    async def scenario():
        async with _plain_client() as client:
            response = await client.post(f"/api/engines/{ENGINE_XTTS}/unload")
            assert response.status_code == 409, response.text
            assert "очереди" in response.json()["detail"]
            assert stub.is_loaded  # модель не тронута

    _run(scenario)


def test_unload_of_never_created_engine_is_noop(monkeypatch):
    monkeypatch.setenv("TTS_XTTS_BASE_DIR", "/нет/такого/каталога")
    created: list[str] = []
    monkeypatch.setattr(
        registry, "_create", lambda engine_id: created.append(engine_id)
    )

    async def scenario():
        async with _plain_client() as client:
            response = await client.post("/api/engines/xtts/unload")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["unloaded"] is False and body["state"] == STATE_IDLE
            assert "не поднят" in body["message"]

    _run(scenario)
    assert created == []  # выгрузка не поднимает модель ради самой себя


def test_mps_memory_snapshot_reads_torch_and_swallows_errors(monkeypatch):
    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_allocated_memory() -> int:
            return 128 * 1024 * 1024

        @staticmethod
        def driver_allocated_memory() -> int:
            return 256 * 1024 * 1024

    fake = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=_FakeMps), mps=_FakeMps
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resource_guard.mps_memory_snapshot() == {
        "current_allocated_mb": 128.0,
        "driver_allocated_mb": 256.0,
    }

    class _BrokenMps:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("Metal недоступен")

    broken = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=_BrokenMps), mps=_BrokenMps
    )
    monkeypatch.setitem(sys.modules, "torch", broken)
    assert resource_guard.mps_memory_snapshot() is None


def test_status_does_not_fail_without_mps(monkeypatch):
    """Без импортированного torch метрика MPS — `None`, а `/api/status` отвечает 200.

    `config._torch` — заглушка без MPS (autouse-фикстура выше), поэтому роут не
    импортирует настоящий torch и результат не зависит от порядка прогона.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    async def scenario():
        async with _plain_client() as client:
            response = await client.get("/api/status")
            assert response.status_code == 200, response.text
            body = response.json()
            assert "mps_memory" in body
            assert body["mps_memory"] is None

    _run(scenario)


# --- измеримость на живом сервисе не тормозит выгрузку -------------------------
def test_unload_is_quick_and_frees_engine_time():
    """Выгрузка не должна ждать `gc`/MPS-очистки дольше разумного в тестах."""
    engine = StubEngine()
    engine.load()
    before = time.monotonic()
    engine.unload()
    assert time.monotonic() - before < 5.0
    assert engine.last_used_at is not None


# --- Фаза 3 (§11.2/§11.3/§11.4): уборка памяти, потолок Metal, принудительная выгрузка
def _fake_torch(mps, *, threads: list[int] | None = None) -> types.SimpleNamespace:
    """Заглушка torch: MPS-счётчики и `set_num_threads` без импорта настоящего."""
    def set_num_threads(value: int) -> None:
        if threads is not None:
            threads.append(value)

    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=mps),
        mps=mps,
        set_num_threads=set_num_threads,
    )


def test_mps_driver_allocated_mb_is_none_without_torch_or_mps(monkeypatch):
    """Метрика берётся только у живого MPS: без torch и без Metal — `None`.

    torch метрикой не импортируется намеренно (читается `sys.modules`): иначе
    каждая реплика платила бы секунды за первый импорт ради одной строки лога.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert engines_base.mps_driver_allocated_mb() is None

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_NoMps))
    assert engines_base.mps_driver_allocated_mb() is None


def test_mps_driver_allocated_mb_reads_counter_and_swallows_errors(monkeypatch):
    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def driver_allocated_memory() -> int:
            return 512 * 1024 * 1024

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_FakeMps))
    assert engines_base.mps_driver_allocated_mb() == 512.0

    class _BrokenMps:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("Metal недоступен")

    # Сбой чтения — не повод ронять синтез: метрика нужна логу, а не решению.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_BrokenMps))
    assert engines_base.mps_driver_allocated_mb() is None


def test_synthesize_cleans_mps_cache_in_tail_and_logs_delta(monkeypatch, caplog):
    """Хвост синтеза чистит кеш Metal и пишет дельту — по ней виден рост памяти (§11.2)."""
    releases: list[dict] = []
    monkeypatch.setattr(
        engines_base, "release_torch_memory", lambda **kwargs: releases.append(kwargs)
    )
    readings = iter([100.0, 130.0])
    monkeypatch.setattr(engines_base, "mps_driver_allocated_mb", lambda: next(readings))

    engine = StubEngine()
    with caplog.at_level(logging.INFO, logger="backend.engines.base"):
        engine.synthesize("Привет", "/tmp/ref.wav", "Привет")

    assert len(releases) == 1, "уборка обязана идти на каждом синтезе, а не только при выгрузке"
    # Уборка именно лёгкая: полная коллекция на каждой реплике дороже самого синтеза.
    assert releases == [{"light": True}]
    assert engine.active_synthesizes == 0
    line = next(
        record.getMessage()
        for record in caplog.records
        if "tts.mps.synthesis" in record.getMessage()
    )
    assert "driver_before_mb=100.0" in line
    assert "driver_after_mb=130.0" in line
    assert "delta_mb=+30.0" in line


def test_light_release_collects_only_young_generation(monkeypatch):
    """Хвост реплики собирает молодое поколение, выгрузка — все: цена разная (§3.1).

    Полная коллекция на разросшемся процессе стоит ~0.2 с (замер на этом проекте),
    и на каждой реплике она съедала бы время синтеза; модель при этом остаётся в
    памяти, а ссылки на неё живут в старших поколениях, то есть полный сбор в
    хвосте не нужен.
    """
    generations: list[int] = []

    def fake_collect(generation: int = 2) -> int:
        generations.append(generation)
        return 0

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(engines_base.gc, "collect", fake_collect)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_NoMps))

    engines_base.release_torch_memory(light=True)
    engines_base.release_torch_memory()  # выгрузка: ссылки на модель уже обнулены

    assert generations == [0, 2]


def test_synthesize_releases_memory_even_when_engine_fails(monkeypatch):
    """Упавший инференс тоже убирает за собой и снимает счётчик занятости."""

    class _FailingEngine(StubEngine):
        def _synthesize(self, *_args, **_kwargs):
            raise RuntimeError("инференс упал")

    releases: list[str] = []
    monkeypatch.setattr(
        engines_base, "release_torch_memory", lambda **kwargs: releases.append("clean")
    )
    engine = _FailingEngine()

    with pytest.raises(RuntimeError, match="инференс упал"):
        engine.synthesize("Привет", "/tmp/ref.wav", "Привет")

    assert releases == ["clean"]
    assert engine.active_synthesizes == 0


def test_limit_mps_memory_sets_fraction_and_logs_recommended(monkeypatch, caplog):
    """Доля памяти выставляется явно, и в лог уходит потолок, от которого она взята."""
    calls: list[float] = []

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_per_process_memory_fraction(value: float) -> None:
            calls.append(value)

        @staticmethod
        def recommended_max_memory() -> int:
            return 16 * 1024**3

    monkeypatch.setattr(config, "MPS_MEMORY_FRACTION", 0.75)
    with caplog.at_level(logging.INFO, logger="tts.config"):
        config.limit_mps_memory(_fake_torch(_FakeMps))

    assert calls == [0.75]
    assert "рекомендовано 16.0 ГБ" in caplog.text
    assert "разрешено 12.0 ГБ" in caplog.text


def test_limit_mps_memory_is_optional_and_never_breaks_start(monkeypatch, caplog):
    """Нет MPS, доля `0` или сбой вызова — сервис поднимается, а не падает (§11.3)."""
    caplog.set_level(logging.INFO, logger="tts.config")

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    config.limit_mps_memory(_fake_torch(_NoMps))  # без MPS ограничивать нечего

    calls: list[float] = []

    class _FailingMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_per_process_memory_fraction(value: float) -> None:
            calls.append(value)

        @staticmethod
        def recommended_max_memory() -> int:
            raise RuntimeError("MPS не отвечает")

    # Доля нулевая — потолок не выставляем вовсе: `0.0` в этом API означает «всё»,
    # а не «ничего», и молчаливое «разрешить всю unified memory» тут недопустимо.
    monkeypatch.setattr(config, "MPS_MEMORY_FRACTION", 0.0)
    config.limit_mps_memory(_fake_torch(_FailingMps))
    assert calls == []
    assert "не задан" in caplog.text

    # Сбой на самом вызове — предупреждение, а не отказ старта.
    monkeypatch.setattr(config, "MPS_MEMORY_FRACTION", 0.5)
    config.limit_mps_memory(_fake_torch(_FailingMps))
    assert calls == [0.5]
    assert "Не удалось ограничить память MPS" in caplog.text


def test_configure_torch_sets_threads_from_config(monkeypatch):
    """Потоки CPU ограничиваются значением проекта — на CPU-фолбэке иначе все ядра (§11.7)."""
    threads: list[int] = []
    calls: list[float] = []

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_per_process_memory_fraction(value: float) -> None:
            calls.append(value)

        @staticmethod
        def recommended_max_memory() -> int:
            return 8 * 1024**3

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_FakeMps, threads=threads))
    monkeypatch.setattr(config, "_torch", None)
    monkeypatch.setattr(config, "TORCH_NUM_THREADS", 3)

    assert config.configure_torch().set_num_threads is not None
    assert threads == [3]
    # Тот же вызов задаёт и потолок Metal: оба ограничения — часть одной настройки.
    assert calls == [config.MPS_MEMORY_FRACTION]


def test_free_warm_engines_unloads_only_idle_neighbours(monkeypatch):
    """Принудительная уборка трогает тёплые свободные движки и не трогает остальные."""
    warm = StubEngine()
    warm.load()
    busy = StubEngine()
    busy.load()
    busy._active_synthesizes = 1
    cold = StubEngine()  # не поднят: освобождать нечего
    requester = StubEngine()
    requester.load()
    monkeypatch.setattr(
        engine_lifecycle,
        "created_engines",
        lambda: {"warm": warm, "busy": busy, "cold": cold, "requester": requester},
    )

    assert engine_lifecycle.free_warm_engines(exclude="requester") == ["warm"]
    assert not warm.is_loaded
    assert busy.is_loaded  # занятый движок не выгружаем: `unload()` отказал бы
    assert requester.is_loaded  # ради него память и освобождали
    assert not cold.is_loaded


def test_free_warm_engines_survives_failed_unload(monkeypatch):
    """Отказ выгрузки одного движка не мешает убрать остальные и не поднимается наверх."""
    broken = StubEngine()
    broken.load()
    healthy = StubEngine()
    healthy.load()

    def refuse() -> None:
        raise EngineBusyError("движок занят")

    monkeypatch.setattr(broken, "unload", refuse)
    monkeypatch.setattr(
        engine_lifecycle, "created_engines", lambda: {"broken": broken, "healthy": healthy}
    )

    assert engine_lifecycle.free_warm_engines() == ["healthy"]
    assert healthy.is_loaded is False
