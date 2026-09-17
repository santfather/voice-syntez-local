"""Политика памяти и тяжёлый gate (§11 Task 1).

Проверяются mocked-состояния NORMAL/WARNING/CRITICAL/HARD STOP: реальное
исчерпание памяти в тестах недопустимо (Gate фазы 4), поэтому пороги подаются
числами, а датчики — подставными функциями.
"""

from __future__ import annotations

import pytest

from backend.llm import memory_policy as p


def _decide(percent, available, pressure=p.PRESSURE_GREEN, size=None, **overrides):
    return p.decide(
        system_percent=percent,
        available_gb=available,
        pressure=pressure,
        model_size_gb=size,
        thresholds=p.MemoryThresholds(**overrides) if overrides else None,
    )


def test_memory_thresholds_map_to_four_levels():
    """Границы §11.2: <70 NORMAL, 75 WARNING, 82 CRITICAL, 88 HARD STOP."""
    assert _decide(50.0, 9.0).level == p.LEVEL_NORMAL
    assert _decide(69.9, 9.0).level == p.LEVEL_NORMAL
    growth = _decide(70.0, 9.0)
    assert growth.level == p.LEVEL_NORMAL
    assert growth.log_growth is True and growth.allow_heavy_start is True
    assert _decide(75.0, 9.0).level == p.LEVEL_WARNING
    assert _decide(81.9, 9.0).level == p.LEVEL_WARNING
    assert _decide(82.0, 9.0).level == p.LEVEL_CRITICAL
    assert _decide(87.9, 9.0).level == p.LEVEL_CRITICAL
    assert _decide(88.0, 9.0).level == p.LEVEL_HARD_STOP


def test_memory_warning_allows_nothing_heavy():
    """WARNING («не запускать новую тяжёлую задачу») запрещает и LLM, и TTS."""
    warning = _decide(76.0, 9.0)
    assert warning.allow_heavy_start is False
    assert warning.allow_llm_start is False
    # Выгружать в WARNING ещё нечего: CRITICAL и HARD STOP — вот где освобождают.
    assert warning.unload_recommended is False


def test_memory_critical_and_hard_stop_recommend_unload():
    critical = _decide(84.0, 9.0)
    hard_stop = _decide(90.0, 9.0)
    assert critical.level == p.LEVEL_CRITICAL
    assert hard_stop.level == p.LEVEL_HARD_STOP
    for decision in (critical, hard_stop):
        assert decision.allow_llm_start is False
        assert decision.unload_recommended is True


def test_memory_pressure_outranks_percent():
    """Жёлтое/красное давление macOS важнее процента (§11.3)."""
    yellow = _decide(40.0, 9.0, pressure=p.PRESSURE_YELLOW)
    assert yellow.level == p.LEVEL_WARNING
    assert yellow.allow_llm_start is False
    red = _decide(40.0, 9.0, pressure=p.PRESSURE_RED)
    assert red.level == p.LEVEL_HARD_STOP
    assert red.allow_llm_start is False


def test_memory_reserve_is_absolute():
    """Резерв 3.5--4 GB под macOS: ниже него — отказ, даже при низком проценте."""
    assert _decide(50.0, 3.4).level == p.LEVEL_CRITICAL
    assert _decide(50.0, 3.8).level == p.LEVEL_WARNING
    assert _decide(50.0, 4.5).level == p.LEVEL_NORMAL


def test_memory_model_must_fit_with_reserve():
    """Модель обязана влезать вместе с резервом, а не «в притык»."""
    assert _decide(50.0, 6.0, size=4.3).level == p.LEVEL_WARNING
    assert _decide(50.0, 4.0, size=5.2).level == p.LEVEL_CRITICAL
    assert _decide(50.0, 10.0, size=5.2).level == p.LEVEL_NORMAL


def test_memory_thresholds_from_env():
    """Пороги читаются из окружения — их можно поднять под конкретную машину."""
    limits = p.thresholds_from_env(
        {
            "LLM_MEMORY_WARNING_PERCENT": "70",
            "LLM_MEMORY_HARD_STOP_PERCENT": "85",
            "LLM_MEMORY_RESERVE_GB": "3.0",
            "LLM_MEMORY_BROKEN": "не число",
        }
    )
    assert limits.warning_percent == 70.0
    assert limits.hard_stop_percent == 85.0
    assert limits.reserve_gb == 3.0
    assert limits.critical_percent == p.MemoryThresholds().critical_percent


def test_memory_decision_serializes_for_report():
    """Решение обязано быть в отчёте: без него непонятно, почему модель пропущена."""
    payload = _decide(84.0, 9.0, size=5.2).to_dict()
    assert payload["level"] == p.LEVEL_CRITICAL
    assert payload["thresholds"]["hard_stop_percent"] == 88.0
    assert payload["allow_llm_start"] is False
    assert "занята" in payload["reason"]


def test_memory_sensor_failure_is_normal_not_critical():
    """Сломанный датчик — не повод останавливать прогон."""
    decision = p.decide_current(sensor=lambda: (_ for _ in ()).throw(RuntimeError("нет датчика")))
    assert decision.level == p.LEVEL_NORMAL


def test_heavy_gate_allows_only_one_job():
    """Тяжёлые задачи не конкурируют: вторая не начинается."""
    gate = p.HeavyGate()
    assert gate.busy() == ""
    ticket = gate.try_acquire(p.HEAVY_LLM, owner="benchmark")
    assert "benchmark" in gate.busy()
    with pytest.raises(p.HeavyResourceBusy):
        gate.try_acquire(p.HEAVY_TTS, owner="tts")
    gate.release(ticket)
    assert gate.busy() == ""
    # После освобождения следующая тяжёлая задача проходит.
    with gate.hold(p.HEAVY_WHISPER, owner="qa"):
        assert "whisper" in gate.busy()
    assert gate.busy() == ""


def test_heavy_gate_releases_after_error():
    """Исключение внутри задачи не должно оставить gate занятым навсегда."""
    gate = p.HeavyGate()
    with pytest.raises(ValueError, match="boom"), gate.hold(p.HEAVY_LLM, owner="benchmark"):
        raise ValueError("boom")
    assert gate.busy() == ""


def test_heavy_gate_consults_external_state_and_validates_kind():
    """Внешняя занятость (очередь TTS) и неизвестный вид задачи."""
    busy_gate = p.HeavyGate(external_busy=lambda: "очередь синтеза занята")
    assert "очередь" in busy_gate.busy()
    with pytest.raises(p.HeavyResourceBusy):
        busy_gate.try_acquire(p.HEAVY_LLM)
    broken = p.HeavyGate(external_busy=lambda: (_ for _ in ()).throw(RuntimeError("нет API")))
    assert "не удалось" in broken.busy()
    with pytest.raises(ValueError, match="неизвестный вид"):
        p.HeavyGate().try_acquire("тренировка")
