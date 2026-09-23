"""Изоляция синтеза: процесс-воркер, падение, перезапуск, выгрузка, health.

Тесты не поднимают настоящую модель: супервизор запускает дочерний процесс с
лёгкой фабрикой движка (`tests/fake_worker_engine.py`), которая умеет падать по
тексту реплики. Это единственный способ проверить SIGABRT и восстановление
детерминированно — и ровно поэтому фабрика вынесена в отдельный модуль: в
дочернем процессе не должно оказаться ни torch, ни весов.

Падение проверяется «по-настоящему»: `os.abort()` выполняется в ребёнке, а тест
(процесс бэкенда) обязан остаться жив и увидеть код возврата -6.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

from backend import audio_pipeline, config
from backend.audio_pipeline import ChunkTimeoutError, RenderSettings, SpeakerSettings
from backend.engines import registry
from backend.engines import supervisor as supervisor_module
from backend.engines import worker_protocol as proto
from backend.engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    ENGINE_XTTS,
    SAMPLE_RATE,
    EngineInfo,
    SynthesisEngine,
)
from backend.engines.supervisor import get_supervisor
from backend.engines.worker_engine import WorkerEngine
from backend.voices_store import Voice

FACTORY = "fake_worker_engine:create"


@pytest.fixture
def isolated(monkeypatch):
    """Включает изоляцию с лёгкой фабрикой движка и короткими таймаутами.

    Таймауты короткие, потому что проверяется поведение, а не боевые 600 секунд:
    ожидание ответа и пауза после серии падений в тестах измеряются секундами.

    Супервизор подменяется свежим на каждый тест: в бою счётчик падений и окно
    crash-loop живут на весь процесс (в этом и смысл защиты от цикла), а в
    тестах накопленные отказы соседних тестов сделали бы утверждения
    зависимыми от порядка запуска.
    """
    monkeypatch.setenv(config.WORKER_ISOLATION_ENV, "1")
    monkeypatch.setenv(config.WORKER_ENGINE_FACTORY_ENV, FACTORY)
    monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(config, "WORKER_LOAD_TIMEOUT_SEC", 20.0)
    monkeypatch.setattr(config, "WORKER_SHUTDOWN_GRACE_SEC", 3.0)
    fresh = supervisor_module.WorkerSupervisor()
    monkeypatch.setattr(supervisor_module, "_supervisor", fresh)
    yield fresh
    fresh.stop_all(grace=1.0)
    fresh.reset_after_stop()


def _synth(engine: WorkerEngine, text: str):
    """Синтез одной реплики с фиктивным референсом: тесты проверяют воркер, а не голос."""
    return engine.synthesize(text, "/tmp/reference.wav", "референс")


def _make_engine(engine_id: str = ENGINE_F5) -> WorkerEngine:
    engine = WorkerEngine(engine_id)
    return engine


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _pid_gone(process) -> bool:
    return process is None or process.poll() is not None


# --- базовое: инференс действительно в другом процессе -------------------------
def test_worker_can_synthesize(isolated):
    engine = _make_engine()
    try:
        wave, sample_rate = _synth(engine, "Привет, это проверка воркера")
        assert sample_rate == SAMPLE_RATE
        assert wave.dtype.name == "float32"
        assert wave.size > 0
        # Процесс синтеза — не тот, в котором живёт бэкенд: в этом весь смысл
        # изоляции (падение модели не должно уносить приложение).
        assert engine.worker_pid is not None
        assert engine.worker_pid != os.getpid()
        assert engine.is_loaded
    finally:
        engine.unload()


def test_worker_roundtrip_does_not_import_torch_in_backend(isolated):
    """Инференс в воркере не тянет torch в процесс бэкенда.

    Проверка сравнивает состояние «до и после», а не требует отсутствия torch:
    в полном прогоне тестов torch уже мог быть импортирован другим тестом, и
    ложное падение здесь говорило бы не о изоляции, а о порядке тестов.
    """
    before = "torch" in sys.modules
    engine = _make_engine()
    try:
        _synth(engine, "Проверка импорта")
        assert ("torch" in sys.modules) == before
    finally:
        engine.unload()


def test_worker_ping_reports_state_and_pid(isolated):
    supervisor = get_supervisor()
    answer = supervisor.request_ping(ENGINE_F5)
    assert answer["ok"] is True
    assert answer["engine"] == ENGINE_F5
    assert answer["pid"] != os.getpid()
    # До первого синтеза модель не поднимается: пинг не должен стоить загрузки.
    assert answer["created"] is False


# --- падения ------------------------------------------------------------------
def test_tts_worker_sigabrt_does_not_kill_backend(isolated):
    """Падение воркера SIGABRT: бэкенд жив, отказ описан, замена поднята."""
    engine = _make_engine()
    try:
        with pytest.raises(proto.WorkerCrashError) as caught:
            _synth(engine, "CRASH")
        failure = caught.value
        assert failure.error_type == proto.ERROR_WORKER_CRASH
        assert failure.details["signal"] == 6
        assert failure.details["signal_name"] == "SIGABRT"
        assert failure.details["exit_code"] == -6

        # Замена поднимается сразу после падения: следующая реплика должна
        # работать, а не обнаруживать мёртвый процесс.
        assert _wait_for(lambda: engine.worker.pid is not None and engine.worker.alive())
        assert engine.worker.state == proto.WORKER_STATE_IDLE
        wave, _ = _synth(engine, "После падения")
        assert wave.size > 0
        assert engine.worker.failures == 1
        assert engine.worker.restarts >= 1
    finally:
        engine.unload()


def test_tts_worker_sigsegv_is_classified_as_crash(isolated):
    engine = _make_engine()
    try:
        with pytest.raises(proto.WorkerCrashError) as caught:
            _synth(engine, "SEGV")
        assert caught.value.details["signal"] == 11
        assert "памяти" in caught.value.details["reason"]
    finally:
        engine.unload()


def test_engine_error_keeps_worker_alive(isolated):
    """Ошибка модели — не падение процесса: воркер жив и следующий запрос проходит."""
    engine = _make_engine()
    try:
        _synth(engine, "Разогрев")
        pid_before = engine.worker_pid
        with pytest.raises(proto.TtsEngineError) as caught:
            _synth(engine, "BOOM")
        assert "модель сломалась внутри воркера" in str(caught.value)
        assert caught.value.error_type == proto.ERROR_TTS
        assert engine.worker_pid == pid_before
        assert engine.worker.alive()
        assert engine.worker.failures == 0
        _synth(engine, "И ещё раз")
    finally:
        engine.unload()


def test_worker_timeout_kills_process_and_restarts(isolated, monkeypatch):
    """Зависший воркер убивается по таймауту, и поднимается замена.

    Холодный спавн вынесен за бюджет таймаута намеренно: под песочницей TraeCode
    первый подъём воркера (импорт torch, ~2 с) не укладывается в секунду, и
    проверка ловила бы скорость импорта, а не реакцию супервизора. Поэтому
    сначала воркер прогревается с щедрым лимитом фикстуры (10 с), и только
    потом бюджет ужимается до 1 с — и «HANG» проверяет именно таймаут. Перед
    последней репликой бюджет возвращается: её выполняет уже **новая**, только
    что поднятая замена, и ужимать её спавн снова было бы той же ошибкой.
    """
    engine = _make_engine()
    generous = config.WORKER_REQUEST_TIMEOUT_SEC
    try:
        _synth(engine, "Разогрев")  # спавн завершён, воркер ответил
        monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 1.0)
        stale = engine.worker.process
        with pytest.raises(proto.WorkerTimeoutError) as caught:
            _synth(engine, "HANG")
        assert caught.value.error_type == proto.ERROR_WORKER_TIMEOUT
        assert "не ответил за 1 с" in str(caught.value)
        assert _pid_gone(stale), "зависший процесс обязан быть убит, а не ждать вечно"
        assert _wait_for(lambda: engine.worker.alive() and engine.worker.process is not stale)
        monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", generous)
        _synth(engine, "После таймаута")
    finally:
        engine.unload()


def test_abort_current_stops_stuck_worker_without_double_count(isolated):
    """Хук таймаута пайплайна: висящий запрос возвращает отказ, а не второй сбой."""
    engine = _make_engine()
    outcome: dict = {}

    def run() -> None:
        try:
            _synth(engine, "HANG")
            outcome["result"] = "ok"
        except Exception as exc:  # noqa: BLE001 — тип проверяется в тесте
            outcome["error"] = exc

    try:
        _synth(engine, "Разогрев")
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert _wait_for(lambda: engine.worker.state == proto.WORKER_STATE_BUSY)
        assert engine.abort_current("тест") is True
        thread.join(timeout=20)
        assert isinstance(outcome.get("error"), proto.WorkerTimeoutError)
        # Отказ один: и таймаут, и висящий поток описывают одно падение.
        assert engine.worker.failures == 1
    finally:
        engine.unload()


def test_supervisor_degrades_after_crash_loop_and_recovers(isolated, monkeypatch):
    monkeypatch.setattr(config, "WORKER_CRASH_LIMIT", 3)
    monkeypatch.setattr(config, "WORKER_DEGRADED_COOLDOWN_SEC", 0.3)
    engine = _make_engine()
    try:
        for _ in range(3):
            with pytest.raises(proto.WorkerCrashError):
                _synth(engine, "CRASH")
        handle = engine.worker
        assert handle.state == proto.WORKER_STATE_DEGRADED
        assert handle.last_failure["state"] == proto.WORKER_STATE_DEGRADED

        # В режиме DEGRADED новые процессы не поднимаются: иначе цикл падений
        # превратился бы в бесконечный перезапуск.
        with pytest.raises(proto.WorkerUnavailableError) as caught:
            _synth(engine, "Привет")
        assert caught.value.details["degraded"] is True
        assert "временно отключён" in str(caught.value)

        # Пауза истекла — одна попытка, и движок снова работает.
        time.sleep(0.4)
        wave, _ = _synth(engine, "После паузы")
        assert wave.size > 0
    finally:
        engine.unload()


# --- выгрузка, выключение, отсутствие сирот -----------------------------------
def test_unload_stops_worker_process(isolated):
    engine = _make_engine()
    _synth(engine, "Разогрев")
    stale = engine.worker.process
    assert stale is not None
    engine.unload()
    assert engine.worker.state == proto.WORKER_STATE_STOPPED
    assert engine.worker.pid is None
    assert _pid_gone(stale)

    # Следующий синтез поднимает новый процесс: выгрузка не ломает движок.
    _synth(engine, "Заново")
    assert engine.worker.pid != stale.pid
    engine.unload()


def test_shutdown_all_leaves_no_orphans(isolated):
    first = _make_engine(ENGINE_F5)
    second = _make_engine(ENGINE_XTTS)
    _synth(first, "Первый движок")
    _synth(second, "Второй движок")
    processes = [first.worker.process, second.worker.process]
    assert all(process is not None for process in processes)

    stopped = get_supervisor().stop_all()
    assert stopped == 2
    assert all(_pid_gone(process) for process in processes)
    assert {handle["state"] for handle in get_supervisor().status()} == {
        proto.WORKER_STATE_STOPPED
    }
    get_supervisor().reset_after_stop()


def test_worker_exits_when_parent_channel_closes(isolated):
    """Смерть бэкенда не оставляет воркер с моделью в памяти.

    Отдельного «присмотра» за родителем нет: канал закрывается вместе с
    процессом, ребёнок получает EOF и выходит (в тесте это эмулируется закрытием
    каналов).
    """
    engine = _make_engine()
    _synth(engine, "Разогрев")
    handle = engine.worker
    process = handle.process
    assert process is not None

    handle.request.close()
    handle.response.close()
    assert _wait_for(lambda: process.poll() is not None, timeout=10), (
        "воркер обязан выйти, когда родитель перестал отвечать"
    )
    with handle.state_lock:
        handle.process = None


# --- реестр, паспорт, health --------------------------------------------------
def test_registry_returns_worker_engine_when_isolated(isolated, monkeypatch):
    monkeypatch.setattr(registry, "_instances", {})
    engine = registry.get_engine(ENGINE_F5)
    assert isinstance(engine, WorkerEngine)
    # Настоящий движок собирается тем же реестром, но в другом процессе: второй
    # реализации F5/XTTS в проекте не появляется.
    assert registry.create_local_engine(ENGINE_F5).info.id == ENGINE_F5


def test_passport_supports_seed_is_the_single_source(isolated):
    for engine_id in (ENGINE_F5, ENGINE_XTTS, "xtts-banana"):
        assert ENGINE_INFOS[engine_id].supports_seed is True

    class WithSeed(SynthesisEngine):
        info = EngineInfo(
            id="seed", label="С сидом", description="", supports_accents=False,
            supports_seed=True,
        )

        def load(self) -> None:  # pragma: no cover — не вызывается
            pass

        def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
            raise NotImplementedError  # pragma: no cover

    class WithoutSeed(WithSeed):
        info = EngineInfo(
            id="no-seed", label="Без сида", description="", supports_accents=False
        )

    assert WithSeed().supports_seed is True
    assert WithoutSeed().supports_seed is False


def test_supervisor_status_describes_worker(isolated):
    engine = _make_engine()
    try:
        _synth(engine, "Разогрев")
        status = {item["engine"]: item for item in get_supervisor().status()}
        entry = status[ENGINE_F5]
        assert entry["state"] == proto.WORKER_STATE_IDLE
        assert entry["pid"] == engine.worker_pid
        assert entry["alive"] is True
        assert entry["restarts"] == 0
        assert entry["last_failure"] is None
        assert entry.get("started_at")
    finally:
        engine.unload()


# --- связка с пайплайном: таймаут убивает воркер --------------------------------
def test_pipeline_timeout_aborts_isolated_worker(isolated, monkeypatch, workspace):
    """Таймаут куска в пайплайне обязан убить зависший процесс воркера.

    Иначе поток остался бы в нативном вызове, а воркер — занятым навсегда: все
    следующие реплики встали бы за ним в очередь. Таймаут пайплайна короче
    таймаута воркера (у него точнее вердикт — известна реплика).
    """
    import soundfile as sf
    from conftest import sine

    sf.write(workspace / "voices" / "ref.wav", sine(1.0), SAMPLE_RATE)
    voice = Voice(
        id="voice-1", name="Голос", gender="male", ref_text="референс",
        audio_file="ref.wav", engine=ENGINE_F5,
    )
    monkeypatch.setattr(config, "CHUNK_TIMEOUT_SEC", 0.5)
    monkeypatch.setattr(config, "WORKER_REQUEST_TIMEOUT_SEC", 30.0)
    engine = _make_engine()
    tuning = SpeakerSettings(voice_id=voice.id)
    try:
        _synth(engine, "Разогрев")
        stale = engine.worker.process
        with pytest.raises(ChunkTimeoutError):
            asyncio.run(
                audio_pipeline._synthesize_chunk(
                    engine, voice, "HANG", tuning, RenderSettings(), "1", "Тест"
                )
            )
        assert _pid_gone(stale)
        assert engine.worker.failures == 1
    finally:
        engine.unload()
