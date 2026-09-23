"""Планировщик тяжёлых задач: один inference за раз и только при живой памяти.

Task 2 §14 требует двух вещей, которые нельзя развести по разным местам:

1. **Память.** LLM-анализ не начинается при WARNING/CRITICAL/HARD STOP, резерв
   3.5--4 GB под macOS сохраняется, давление macOS важнее процента.
2. **Отсутствие тяжёлого параллелизма.** Одновременно выполняется ровно один
   тяжёлый inference: LLM-анализ, синтез F5/XTTS или Whisper. На 18 GB unified
   memory вторая тяжёлая модель означает компрессию и swap, то есть зависание всей
   машины, а не «чуть медленнее».

Почему единый планировщик, а не проверки по месту: очередь синтеза, LLM-анализ и
распознавание — три разных пути запуска. Если каждый будет сам решать, «можно ли
начинать», они рано или поздно решат это одновременно. Здесь же решение одно:
`check()` отвечает на вопрос «можно ли», `hold()` выдаёт слот и гарантированно
возвращает его в `finally` — и при ошибке, и при таймауте, и при отмене.

Лёгкие операции (API, UI, SQLite, детерминированная подготовка текста) слот не
занимают и не блокируются: планировщик про них не знает.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import memory_policy as memory

logger = logging.getLogger("tts.llm.scheduler")


class HeavyBlockedError(RuntimeError):
    """Тяжёлую задачу нельзя начать: память или занятый слот.

    Отдельный тип, а не общий RuntimeError: вызывающий код обязан различать
    «сейчас нельзя» (повторить позже, показать причину) и настоящую ошибку
    выполнения.
    """

    def __init__(self, reason: str, decision: ScheduleDecision) -> None:
        super().__init__(reason)
        self.decision = decision


@dataclass(frozen=True)
class ScheduleDecision:
    """Ответ планировщика: можно ли начинать и почему."""

    allowed: bool
    reason: str
    kind: str
    memory_level: str = ""
    memory_reason: str = ""
    holder: str = ""
    model_size_gb: float | None = None
    # Прерывать ли **уже идущую** работу, а не только запрещать новую. Поле
    # повторяет решение `memory_policy` (HARD STOP или исчерпанный резерв), чтобы
    # вызывающий слой не выводил это условие сам — иначе первый же пропущенный
    # случай (резерв под порогом, но не HARD STOP) продолжил бы проход.
    must_abort_running: bool = False

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "kind": self.kind,
            "memory_level": self.memory_level,
            "memory_reason": self.memory_reason,
            "holder": self.holder,
            "model_size_gb": self.model_size_gb,
            "must_abort_running": self.must_abort_running,
        }


@dataclass
class HeavyScheduler:
    """Единая точка решения о запуске тяжёлой задачи.

    `external_busy` нужен, когда тяжёлую работу выполняет кто-то вне процесса
    (например, отдельный процесс-воркер синтеза): планировщик обязан видеть эту
    занятость, а не считать себя единственным потребителем памяти.
    """

    gate: memory.HeavyGate | None = None
    thresholds: memory.MemoryThresholds | None = None
    sensor: memory.Sensor | None = None
    external_busy: Callable[[], str] | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.gate is None:
            self.gate = memory.HeavyGate(external_busy=self.external_busy)
        elif self.external_busy is not None:
            # Внешняя занятость должна быть видна и у готового gate (общий
            # singleton из Task 1).
            self.gate = memory.HeavyGate(external_busy=self.external_busy)

    # -- решение ---------------------------------------------------------------
    def check(
        self, kind: str = memory.HEAVY_LLM, *, model_size_gb: float | None = None
    ) -> ScheduleDecision:
        """Можно ли начинать тяжёлую задачу прямо сейчас."""
        decision = memory.decide_current(
            model_size_gb=model_size_gb, thresholds=self.thresholds, sensor=self.sensor
        )
        holder = self.gate.busy() if self.gate is not None else ""
        if holder:
            return ScheduleDecision(
                allowed=False,
                reason=f"тяжёлый ресурс занят: {holder}",
                kind=kind,
                memory_level=decision.level,
                memory_reason=decision.reason,
                holder=holder,
                model_size_gb=model_size_gb,
                must_abort_running=decision.must_abort_running,
            )
        if kind == memory.HEAVY_LLM:
            allowed = decision.allow_llm_start
        else:
            # Синтез и Whisper — такая же тяжёлая работа: WARNING уже запрещает
            # начинать новую (§11.2 Task 1, §14.2 Task 2).
            allowed = decision.allow_heavy_start
        return ScheduleDecision(
            allowed=allowed,
            reason=decision.reason if allowed else f"память: {decision.reason}",
            kind=kind,
            memory_level=decision.level,
            memory_reason=decision.reason,
            holder=holder,
            model_size_gb=model_size_gb,
            must_abort_running=decision.must_abort_running,
        )

    def busy_reason(self) -> str:
        """Кто занимает тяжёлый слот (пусто — свободен)."""
        return self.gate.busy() if self.gate is not None else ""

    # -- слот ------------------------------------------------------------------
    @contextmanager
    def hold(
        self,
        kind: str = memory.HEAVY_LLM,
        *,
        model_size_gb: float | None = None,
        owner: str = "",
    ) -> Iterator[memory.GateTicket]:
        """Занимает тяжёлый слот на время работы.

        Решение принимается **до** захвата, а слот освобождается в `finally`:
        ошибка, таймаут или отмена внутри задачи не должны оставить ресурс
        занятым навсегда — иначе следующая тяжёлая задача не начнётся никогда.
        """
        decision = self.check(kind, model_size_gb=model_size_gb)
        if not decision.allowed:
            raise HeavyBlockedError(decision.reason, decision)
        try:
            ticket = self.gate.try_acquire(kind, owner=owner)  # type: ignore[union-attr]
        except memory.HeavyResourceBusy as exc:
            # Гонка: слот заняли между проверкой и захватом. Это не ошибка логики,
            # но начинать всё равно нельзя.
            raise HeavyBlockedError(str(exc), self.check(kind, model_size_gb=model_size_gb)) from exc
        try:
            yield ticket
        finally:
            self.gate.release(ticket)  # type: ignore[union-attr]


_scheduler: HeavyScheduler | None = None


def get_scheduler() -> HeavyScheduler:
    """Singleton планировщика: общий gate с очередью синтеза и Whisper."""
    global _scheduler
    if _scheduler is None:
        _scheduler = HeavyScheduler(gate=memory.get_gate())
    return _scheduler


def reset_scheduler() -> None:
    """Сброс singleton'а и общего gate — для тестов и смены настроек."""
    global _scheduler
    _scheduler = None
    memory.reset_gate()
