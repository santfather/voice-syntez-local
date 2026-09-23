"""Политика памяти для LLM-задач на M3 18 GB (§11 постановки Task 1).

Машина здесь одна и известная: 18 GB unified memory, где LLM, TTS, Whisper и
система делят одну и ту же физическую память. Поэтому у LLM-задач отдельная
политика с четырьмя состояниями и абсолютным резервом под macOS, а не «пока не
упадёт».

Почему не «занять всё, что свободно»:

* macOS при нехватке памяти уходит в компрессию и swap, и тогда зависает не
  benchmark, а вся машина — включая сессию, в которой этот benchmark запущен;
* замер на границе памяти даёт неверные числа: модель, которой не хватило
  страниц, считается медленной, хотя она просто ждала swap;
* 3.5--4 GB резерва нужны системе и UI: без них пользователь не увидит ни
  прогресса, ни отчёта.

Priority pressure над процентом — не украшение: на macOS `memory_pressure` может
быть жёлтым при формально свободной памяти (например, система уже сбрасывает
файловый кеш). В этом состоянии новая тяжёлая задача вредит, даже если процент
ниже порога.

Тяжёлые ресурсы не конкурируют (§11.4): LLM-инференс, синтез F5/XTTS и Whisper
выполняются по одному. За это отвечает `HeavyGate` — в одном процессе, а для
внешних потребителей (очередь TTS) — через `external_busy`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

logger = logging.getLogger("tts.llm.memory")

# --- состояния -----------------------------------------------------------------
LEVEL_NORMAL = "NORMAL"
LEVEL_WARNING = "WARNING"
LEVEL_CRITICAL = "CRITICAL"
LEVEL_HARD_STOP = "HARD_STOP"
LEVELS: tuple[str, ...] = (LEVEL_NORMAL, LEVEL_WARNING, LEVEL_CRITICAL, LEVEL_HARD_STOP)

# --- давление памяти macOS -----------------------------------------------------
PRESSURE_GREEN = "green"
PRESSURE_YELLOW = "yellow"
PRESSURE_RED = "red"
PRESSURE_UNKNOWN = "unknown"
# `sysctl kern.memorystatus_vm_pressure_level`: 1 — normal, 2 — warning, 4 — critical.
_PRESSURE_BY_LEVEL = {1: PRESSURE_GREEN, 2: PRESSURE_YELLOW, 4: PRESSURE_RED}

# --- тяжёлые задачи ------------------------------------------------------------
HEAVY_LLM = "llm"
HEAVY_TTS = "tts"
HEAVY_WHISPER = "whisper"
HEAVY_KINDS: tuple[str, ...] = (HEAVY_LLM, HEAVY_TTS, HEAVY_WHISPER)


@dataclass(frozen=True)
class MemoryThresholds:
    """Пороги политики. Стартовые значения из §11.2, проверяются реальными замерами."""

    normal_max_percent: float = 70.0
    warning_percent: float = 75.0
    critical_percent: float = 82.0
    hard_stop_percent: float = 88.0
    # Целевой резерв под macOS/UI и жёсткий минимум, ниже которого уже поздно.
    reserve_gb: float = 4.0
    reserve_min_gb: float = 3.5

    def to_dict(self) -> dict:
        return {
            "normal_max_percent": self.normal_max_percent,
            "warning_percent": self.warning_percent,
            "critical_percent": self.critical_percent,
            "hard_stop_percent": self.hard_stop_percent,
            "reserve_gb": self.reserve_gb,
            "reserve_min_gb": self.reserve_min_gb,
        }


def thresholds_from_env(env: Mapping[str, str] | None = None) -> MemoryThresholds:
    """Пороги из окружения: их можно поднять/опустить, не правя код.

    Переменные названы как в постановке (`LLM_MEMORY_*`), чтобы значения в отчёте
    можно было сверить с документацией один в один.
    """
    source = os.environ if env is None else env
    base = MemoryThresholds()
    values = {
        "normal_max_percent": _env_float(source, "LLM_MEMORY_NORMAL_MAX_PERCENT"),
        "warning_percent": _env_float(source, "LLM_MEMORY_WARNING_PERCENT"),
        "critical_percent": _env_float(source, "LLM_MEMORY_CRITICAL_PERCENT"),
        "hard_stop_percent": _env_float(source, "LLM_MEMORY_HARD_STOP_PERCENT"),
        "reserve_gb": _env_float(source, "LLM_MEMORY_RESERVE_GB"),
        "reserve_min_gb": _env_float(source, "LLM_MEMORY_RESERVE_MIN_GB"),
    }
    return replace(base, **{key: value for key, value in values.items() if value is not None})


def _env_float(source: Mapping[str, str], name: str) -> float | None:
    raw = source.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("Переменная %s = %r не число — беру значение по умолчанию", name, raw)
        return None


@dataclass(frozen=True)
class MemoryDecision:
    """Решение политики: можно ли начинать тяжёлую задачу и что с уже загруженным."""

    level: str
    reason: str
    allow_heavy_start: bool
    allow_llm_start: bool
    unload_recommended: bool
    log_growth: bool
    system_percent: float
    available_gb: float
    pressure: str
    model_size_gb: float | None = None
    thresholds: MemoryThresholds = field(default_factory=MemoryThresholds)
    # Датчики прочитались. Если нет — решение не применяется, и это видно в отчёте.
    sensor_ok: bool = True
    # Прерывать ли **уже идущую** тяжёлую задачу. Это не то же самое, что «не
    # начинать новую»: постановка (§11.2) различает CRITICAL («остановить dispatch
    # новых задач») и HARD STOP («текущую безопасно завершить/прервать»). Поэтому
    # прерываем работу только тогда, когда есть жёсткий факт: HARD STOP или
    # исчерпанный абсолютный резерв. Процентный CRITICAL сам по себе идущий
    # benchmark не убивает — иначе один замер посередине модели даёт обрезанные
    # данные вместо результата.
    must_abort_running: bool = False

    def to_dict(self) -> dict:
        """Плоское представление решения о памяти для отчёта."""
        return {
            "level": self.level,
            "reason": self.reason,
            "allow_heavy_start": self.allow_heavy_start,
            "allow_llm_start": self.allow_llm_start,
            "unload_recommended": self.unload_recommended,
            "log_growth": self.log_growth,
            "system_percent": self.system_percent,
            "available_gb": self.available_gb,
            "pressure": self.pressure,
            "model_size_gb": self.model_size_gb,
            "sensor_ok": self.sensor_ok,
            "must_abort_running": self.must_abort_running,
            "thresholds": self.thresholds.to_dict(),
        }


def decide(
    *,
    system_percent: float,
    available_gb: float,
    pressure: str = PRESSURE_UNKNOWN,
    model_size_gb: float | None = None,
    thresholds: MemoryThresholds | None = None,
    sensor_ok: bool = True,
) -> MemoryDecision:
    """Чистая классификация: без psutil, sysctl и глобального состояния.

    Порядок проверок — от самого опасного к самому мягкому. Давление macOS
    проверяется раньше процента: жёлтое давление при 60 % — это всё равно «не
    начинать», потому что система уже экономит память.

    `sensor_ok=False` — датчики не прочитались. Политика в этом случае **не
    применяется** (как и в `memory_monitor` для синтеза): сломанная метрика не
    доказательство нехватки памяти, а остановка прогона из-за неё была бы хуже
    отсутствия проверки. Флаг остаётся в отчёте, поэтому такой прогон видно.
    """
    limits = thresholds or MemoryThresholds()
    if not sensor_ok:
        return MemoryDecision(
            level=LEVEL_NORMAL,
            reason="датчики памяти недоступны — политика не применяется",
            allow_heavy_start=True,
            allow_llm_start=True,
            unload_recommended=False,
            log_growth=False,
            system_percent=system_percent,
            available_gb=available_gb,
            pressure=pressure,
            model_size_gb=model_size_gb,
            thresholds=limits,
            sensor_ok=False,
        )
    level = LEVEL_NORMAL
    reasons: list[str] = []

    def raise_to(candidate: str, reason: str) -> None:
        nonlocal level
        if LEVELS.index(candidate) > LEVELS.index(level):
            level = candidate
        reasons.append(reason)

    if pressure == PRESSURE_RED:
        raise_to(LEVEL_HARD_STOP, "macOS memory pressure красное")
    elif pressure == PRESSURE_YELLOW:
        raise_to(LEVEL_WARNING, "macOS memory pressure жёлтое")

    if available_gb < limits.reserve_min_gb:
        raise_to(
            LEVEL_CRITICAL,
            f"свободно {available_gb:.1f} GB — меньше жёсткого резерва {limits.reserve_min_gb:.1f} GB",
        )
    elif available_gb < limits.reserve_gb:
        raise_to(
            LEVEL_WARNING,
            f"свободно {available_gb:.1f} GB — меньше целевого резерва {limits.reserve_gb:.1f} GB",
        )

    if model_size_gb:
        if available_gb < model_size_gb:
            raise_to(
                LEVEL_CRITICAL,
                f"модель {model_size_gb:.1f} GB не помещается в свободные {available_gb:.1f} GB",
            )
        elif available_gb < model_size_gb + limits.reserve_min_gb:
            raise_to(
                LEVEL_WARNING,
                f"модель {model_size_gb:.1f} GB оставит меньше {limits.reserve_min_gb:.1f} GB резерва",
            )

    if system_percent >= limits.hard_stop_percent:
        raise_to(
            LEVEL_HARD_STOP,
            f"память занята на {system_percent:.0f}% (HARD STOP с {limits.hard_stop_percent:.0f}%)",
        )
    elif system_percent >= limits.critical_percent:
        raise_to(
            LEVEL_CRITICAL,
            f"память занята на {system_percent:.0f}% (CRITICAL с {limits.critical_percent:.0f}%)",
        )
    elif system_percent >= limits.warning_percent:
        raise_to(
            LEVEL_WARNING,
            f"память занята на {system_percent:.0f}% (WARNING с {limits.warning_percent:.0f}%)",
        )
    elif system_percent >= limits.normal_max_percent:
        reasons.append(
            f"память занята на {system_percent:.0f}% — рост отмечен, задача разрешена"
        )

    if not reasons:
        reasons.append(f"память занята на {system_percent:.0f}%, резерв {available_gb:.1f} GB — норма")

    return MemoryDecision(
        level=level,
        reason="; ".join(reasons),
        must_abort_running=(
            level == LEVEL_HARD_STOP or available_gb < limits.reserve_min_gb
        ),
        # WARNING — «новую тяжёлую задачу не запускать» (§11.2), поэтому запуск
        # разрешён только в NORMAL. Это осознанно строго: лучше отложить прогон,
        # чем получить числа на машине, которая уходит в swap.
        allow_heavy_start=level == LEVEL_NORMAL,
        allow_llm_start=level == LEVEL_NORMAL,
        unload_recommended=level in (LEVEL_CRITICAL, LEVEL_HARD_STOP),
        log_growth=system_percent >= limits.normal_max_percent,
        system_percent=system_percent,
        available_gb=available_gb,
        pressure=pressure,
        model_size_gb=model_size_gb,
        thresholds=limits,
    )


# --- живые датчики -------------------------------------------------------------
Sensor = Callable[[], dict]


def system_percent() -> float:
    import psutil

    return round(float(psutil.virtual_memory().percent), 1)


def available_gb() -> float:
    import psutil

    return round(float(psutil.virtual_memory().available) / (1024**3), 2)


def memory_pressure(timeout_sec: float = 2.0) -> str:
    """Давление памяти macOS через sysctl. Не macOS или ошибка — `unknown`.

    Через CLI, а не через сторонний модуль: это одна команда один раз на модель, и
    ради неё не стоит добавлять зависимость.
    """
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Не удалось прочитать memory pressure: %s", exc)
        return PRESSURE_UNKNOWN
    if completed.returncode != 0:
        return PRESSURE_UNKNOWN
    try:
        return _PRESSURE_BY_LEVEL.get(int(completed.stdout.strip()), PRESSURE_UNKNOWN)
    except ValueError:
        return PRESSURE_UNKNOWN


def read_sensors() -> dict:
    """Текущие показания: процент, свободные гигабайты и давление."""
    return {
        "system_percent": system_percent(),
        "available_gb": available_gb(),
        "pressure": memory_pressure(),
    }


def bypass_decision(reason: str, *, thresholds: MemoryThresholds | None = None) -> MemoryDecision:
    """Решение «проверка памяти не применяется».

    Нужно там, где тяжёлая модель не поднимается вовсе — например, `--dry-run`
    benchmark'а (подставной клиент) или проверка gold. Гонять такую проверку через
    политику памяти значило бы делать её результат зависимым от того, что ещё
    запущено на машине, а модели при этом никто не грузит.
    """
    return MemoryDecision(
        level=LEVEL_NORMAL,
        reason=reason,
        allow_heavy_start=True,
        allow_llm_start=True,
        unload_recommended=False,
        log_growth=False,
        system_percent=0.0,
        available_gb=0.0,
        pressure=PRESSURE_UNKNOWN,
        thresholds=thresholds or MemoryThresholds(),
        sensor_ok=False,
    )


def decide_current(
    *,
    model_size_gb: float | None = None,
    thresholds: MemoryThresholds | None = None,
    sensor: Sensor | None = None,
) -> MemoryDecision:
    """Решение по живым датчикам. Ошибка сбора не должна ронять прогон."""
    collect = sensor or read_sensors
    try:
        data = collect()
    except Exception as exc:  # noqa: BLE001 — метрика вторична
        logger.warning("Не удалось прочитать память: %s", exc)
        return decide(
            system_percent=0.0,
            available_gb=0.0,
            pressure=PRESSURE_UNKNOWN,
            model_size_gb=model_size_gb,
            thresholds=thresholds,
            sensor_ok=False,
        )
    return decide(
        system_percent=float(data.get("system_percent") or 0.0),
        available_gb=float(data.get("available_gb") or 0.0),
        pressure=str(data.get("pressure") or PRESSURE_UNKNOWN),
        model_size_gb=model_size_gb,
        thresholds=thresholds,
    )


# --- тяжёлая конкуренция -------------------------------------------------------
@dataclass(frozen=True)
class GateTicket:
    kind: str
    owner: str = ""


class HeavyResourceBusy(RuntimeError):
    """Тяжёлая задача уже выполняется — вторая не начинается."""


class HeavyGate:
    """Одновременно выполняется не более одной тяжёлой ML-задачи (§11.4).

    Внутри процесса — честная блокировка; снаружи (очередь TTS, Whisper в
    отдельном процессе) — через `external_busy`, который вызывающий передаёт сам.
    Так gate не притворяется, что видит чужие процессы, и не «разрешает»
    конкурентный запуск по незнанию.
    """

    def __init__(self, external_busy: Callable[[], str] | None = None) -> None:
        self._external_busy = external_busy
        self._lock = threading.Lock()
        self._current: GateTicket | None = None

    def busy(self) -> str:
        """Описание занятости: пустая строка — можно начинать."""
        with self._lock:
            if self._current is not None:
                return f"уже выполняется {self._current.kind} ({self._current.owner or 'без имени'})"
        return self._external_reason()

    def _external_reason(self) -> str:
        """Занятость, о которой знает только внешний потребитель (чужой процесс)."""
        if self._external_busy is None:
            return ""
        try:
            return str(self._external_busy() or "")
        except Exception as exc:  # noqa: BLE001 — проверка не повод падать
            return f"не удалось проверить внешнюю занятость: {exc}"

    def try_acquire(self, kind: str, owner: str = "") -> GateTicket:
        if kind not in HEAVY_KINDS:
            raise ValueError(f"неизвестный вид тяжёлой задачи: {kind}")
        with self._lock:
            # Обе проверки — под тем же локом, что и захват. Иначе между «слот
            # свободен» и «слот занят мной» вклинивается второй претендент, и один
            # из двоих получает `HeavyResourceBusy` уже после того, как решил, что
            # можно начинать (у очереди это ожидание слота, см. `_run_heavy`).
            if self._current is not None:
                raise HeavyResourceBusy(f"уже выполняется {self._current.kind}")
            reason = self._external_reason()
            if reason:
                raise HeavyResourceBusy(reason)
            self._current = GateTicket(kind=kind, owner=owner)
            return self._current

    def release(self, ticket: GateTicket) -> None:
        with self._lock:
            if self._current == ticket:
                self._current = None

    @contextmanager
    def hold(self, kind: str, owner: str = "") -> Iterator[GateTicket]:
        ticket = self.try_acquire(kind, owner)
        try:
            yield ticket
        finally:
            self.release(ticket)


_gate: HeavyGate | None = None
_gate_lock = threading.Lock()


def get_gate() -> HeavyGate:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = HeavyGate()
        return _gate


def reset_gate() -> None:
    """Сбрасывает общий gate (нужно тестам и повторному запуску в процессе)."""
    global _gate
    with _gate_lock:
        _gate = None
