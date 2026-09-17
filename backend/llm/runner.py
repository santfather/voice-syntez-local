"""Последовательный прогон моделей по корпусу (§11–§12 постановки Task 1).

Три вещи, ради которых этот модуль существует отдельно от CLI:

1. **Модели идут строго по очереди.** Перед каждой моделью выгружаются все
   остальные, после — выгружается сама, и освобождение проверяется по `/api/ps`.
   Две модели одновременно на 18 GB unified memory — это swap и недостоверные
   замеры, а не «быстрее».
2. **Память проверяется до запуска, а не после падения.** Решение принимает
   `memory_policy`, и запрет — это отказ в запуске с причиной в отчёте, а не
   тихое продолжение (никакого silent fallback).
3. **Прогресс не теряется.** Каждый сырой ответ дописывается в JSONL сразу после
   получения, поэтому упавший или прерванный прогон продолжается с места
   остановки (`--resume`), а сырые ответы лежат отдельно от метрик: любую цифру
   можно перепроверить руками.

Сырые ответы и метрики лежат в разных файлах (`raw.jsonl` и `metrics.json`):
метрики — это производная, и при смене формулы их нужно пересчитать, не трогая
ответы модели.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import memory_policy as memory
from . import metrics as metrics_mod
from . import schemas as s
from . import versioning
from .fake_client import FakeOllamaClient
from .ollama_client import (
    OllamaModel,
    OllamaModelMissingError,
)
from .prompt import PromptTemplate, load_prompt

logger = logging.getLogger("tts.llm.runner")

DEFAULT_NUM_CTX = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0
DEFAULT_KEEP_ALIVE = "5m"
# Сколько раз повторять запрос, если ответ не прошёл валидацию. Один повтор — это
# «repair» из §10: невалидный JSON критичен только тогда, когда и он не помог.
DEFAULT_MAX_REPAIRS = 1
RAW_FILE = "raw.jsonl"
METRICS_FILE = "metrics.json"
RUN_FILE = "run.json"
REPAIR_INSTRUCTION = (
    "Предыдущий ответ не прошёл проверку схемы. Верни ТОЛЬКО JSON-объект по схеме, "
    "без пояснений и без markdown. Поля span_start/span_end/source обязаны точно "
    "соответствовать тексту реплики."
)


class BenchmarkError(RuntimeError):
    """Ошибка прогона, из-за которой продолжать нельзя."""


@dataclass
class CaseRun:
    """Один сырой ответ модели на один кейс."""

    case_id: str
    category: str
    repeat: int
    raw_text: str
    latency_sec: float
    tokens_per_second: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    repair_attempted: bool = False
    error: str = ""
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "repeat": self.repeat,
            "raw_text": self.raw_text,
            "latency_sec": round(self.latency_sec, 4),
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "repair_attempted": self.repair_attempted,
            "error": self.error,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CaseRun:
        return cls(
            case_id=str(raw.get("case_id") or ""),
            category=str(raw.get("category") or ""),
            repeat=int(raw.get("repeat") or 0),
            raw_text=str(raw.get("raw_text") or ""),
            latency_sec=float(raw.get("latency_sec") or 0.0),
            tokens_per_second=raw.get("tokens_per_second"),
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            repair_attempted=bool(raw.get("repair_attempted", False)),
            error=str(raw.get("error") or ""),
            options=dict(raw.get("options") or {}),
        )


@dataclass
class ModelRunResult:
    """Итог прогона одной модели: метрики, память и служебные факты."""

    model: str
    status: str
    reason: str = ""
    cases: int = 0
    planned_cases: int = 0
    reports: list[CaseRun] = field(default_factory=list)
    unloaded_before: list[str] = field(default_factory=list)
    unload_confirmed: bool = True
    metrics: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "cases": self.cases,
            "planned_cases": self.planned_cases,
            "unloaded_before": list(self.unloaded_before),
            "unload_confirmed": self.unload_confirmed,
            "metrics": self.metrics,
            "memory": self.memory,
        }


def model_slug(tag: str) -> str:
    """Имя каталога для результатов: `qwen3:4b-instruct` → `qwen3-4b-instruct`."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-") or "model"


class BenchmarkRunner:
    """Прогоняет список моделей по списку кейсов, по одной модели за раз."""

    def __init__(
        self,
        *,
        client,
        cases: Sequence[s.DatasetCase],
        output_dir: Path,
        prompt: PromptTemplate | None = None,
        options: dict | None = None,
        gate: memory.HeavyGate | None = None,
        thresholds: memory.MemoryThresholds | None = None,
        sensor: memory.Sensor | None = None,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        dataset_version: str = "1",
        on_progress: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.cases = list(cases)
        self.output_dir = Path(output_dir)
        self.prompt = prompt or load_prompt()
        self.options = {
            "num_ctx": DEFAULT_NUM_CTX,
            "temperature": DEFAULT_TEMPERATURE,
            "seed": DEFAULT_SEED,
            **dict(options or {}),
        }
        self.gate = gate or memory.HeavyGate()
        self.thresholds = thresholds or memory.thresholds_from_env()
        self.sensor = sensor
        self.max_repairs = max_repairs
        self.dataset_version = dataset_version
        self.on_progress = on_progress or (lambda message: logger.info("%s", message))
        self.clock = clock

    # -- планирование -----------------------------------------------------------
    def select_cases(
        self, *, categories: Iterable[str] | None = None, limit: int | None = None
    ) -> list[s.DatasetCase]:
        """Кейсы прогона: фильтр по категориям и ограничение объёма (для smoke)."""
        selected = list(self.cases)
        if categories:
            allowed = set(categories)
            unknown = allowed - set(s.CATEGORIES)
            if unknown:
                raise BenchmarkError(f"неизвестные категории: {sorted(unknown)}")
            selected = [case for case in selected if case.category in allowed]
        if limit is not None:
            selected = selected[: max(int(limit), 0)]
        return selected

    def run(
        self,
        models: Sequence[str],
        *,
        categories: Iterable[str] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        resume: bool = False,
        cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """Прогоняет модели последовательно и возвращает отчёт по всем."""
        status = self.client.health()
        if not status.available:
            raise BenchmarkError(f"Ollama недоступна: {status.error or 'нет соединения'}")

        selected = self.select_cases(categories=categories, limit=limit)
        if not selected:
            raise BenchmarkError("не выбрано ни одного кейса")
        started_at = time.time()
        results: dict[str, ModelRunResult] = {}
        stopped_reason = ""
        for model in models:
            result = self._run_model(
                model, selected, repeat=repeat, resume=resume, cancel=cancel
            )
            results[model] = result
            if result.status == "hard_stop":
                stopped_reason = result.reason
                break
        report = {
            "models": {tag: item.to_dict() for tag, item in results.items()},
            "cases": len(selected),
            "repeat": repeat,
            "started_at": started_at,
            "duration_sec": round(time.time() - started_at, 2),
            "stopped_reason": stopped_reason,
            "options": dict(self.options),
            "prompt_version": self.prompt.version,
            "dataset_version": self.dataset_version,
            "schema_version": s.SCHEMA_VERSION,
            "ollama_version": status.version,
        }
        self._write_json(self.output_dir / "benchmark.json", report)
        return report

    # -- одна модель ------------------------------------------------------------
    def _run_model(
        self,
        model: str,
        cases: Sequence[s.DatasetCase],
        *,
        repeat: int,
        resume: bool,
        cancel: Callable[[], bool] | None,
    ) -> ModelRunResult:
        model_dir = self.output_dir / model_slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        raw_path = model_dir / RAW_FILE
        try:
            info: OllamaModel = self.client.require_model(model)
        except OllamaModelMissingError as exc:
            # Не «пропускаем тихо»: модель обязана быть скачана, а причина отказа
            # остаётся в отчёте, чтобы прогон нельзя было принять за полный.
            self.on_progress(f"{model}: пропуск — {exc}")
            return ModelRunResult(model=model, status="missing", reason=str(exc))

        decision = memory.decide_current(
            model_size_gb=info.size_gb, thresholds=self.thresholds, sensor=self.sensor
        )
        self.on_progress(f"{model}: память {decision.level} — {decision.reason}")
        if not decision.allow_llm_start:
            # HARD STOP останавливает весь benchmark: если памяти нет сейчас, её не
            # станет и для следующей модели. WARNING/CRITICAL — пропуск этой.
            status = "hard_stop" if decision.level == memory.LEVEL_HARD_STOP else "blocked"
            result = ModelRunResult(model=model, status=status, reason=decision.reason)
            # CRITICAL/HARD STOP требуют ещё и освободить уже загруженное (§11.2):
            # «не запускать новое» без выгрузки старого оставляет память занятой.
            if decision.unload_recommended:
                result.unloaded_before = self._unload_everything()
            result.metrics = {"memory": decision.to_dict()}
            return result

        unloaded = []
        try:
            unloaded = self.client.unload_others(keep=model)
        except Exception as exc:  # noqa: BLE001 — выгрузка вторична, но о ней сообщаем
            logger.warning("Не удалось выгрузить другие модели перед %s: %s", model, exc)
        if unloaded:
            self.on_progress(f"{model}: выгружены перед запуском: {', '.join(unloaded)}")

        done = self._completed_cases(raw_path) if resume else set()
        if done:
            self.on_progress(f"{model}: продолжаю прогон, уже готово кейсов: {len(done)}")
        reports: list[CaseRun] = []
        memory_samples: list[dict] = []
        pressure_events = 0
        peak_rss = 0.0
        peak_percent = 0.0
        cancelled = False
        gate_busy = ""
        try:
            # Тяжёлая задача одна на процесс и на машину: TTS/Whisper/другой LLM
            # одновременно с benchmark не выполняются (§11.4).
            with self.gate.hold(memory.HEAVY_LLM, owner=f"benchmark:{model}"):
                for case in cases:
                    if cancel is not None and cancel():
                        cancelled = True
                        break
                    if case.id in done:
                        continue
                    for repeat_index in range(max(int(repeat), 1)):
                        run = self._run_case(model, case, repeat_index, cancel=cancel)
                        reports.append(run)
                        self._append_jsonl(raw_path, run.to_dict())
                    sample = self._sample_memory()
                    memory_samples.append(sample)
                    peak_rss = max(peak_rss, float(sample.get("rss_mb") or 0.0))
                    peak_percent = max(peak_percent, float(sample.get("system_percent") or 0.0))
                    if sample.get("pressure") in (memory.PRESSURE_YELLOW, memory.PRESSURE_RED):
                        pressure_events += 1
                    # Модель уже загружена: если память ушла в CRITICAL/HARD STOP,
                    # дальше не идём — иначе прогон начнёт влиять на свои же замеры.
                    if sample.get("level") in (memory.LEVEL_CRITICAL, memory.LEVEL_HARD_STOP):
                        self.on_progress(
                            f"{model}: прогон остановлен на кейсе {case.id} — "
                            + f"{sample.get('reason')}"
                        )
                        break
        except memory.HeavyResourceBusy as exc:
            # Не «подождать и попробовать»: занятый тяжёлый ресурс — это решение
            # пользователя, а не повод запустить конкурентный инференс.
            gate_busy = str(exc)
            self.on_progress(f"{model}: тяжёлый ресурс занят — {gate_busy}")

        if gate_busy and not reports:
            return ModelRunResult(
                model=model,
                status="blocked",
                reason=f"тяжёлый ресурс занят: {gate_busy}",
                planned_cases=len(cases),
            )

        confirmed = self._unload_model(model)
        measured = self._combine_reports(model, cases, raw_path, reports)
        attempted_ids = {run.case_id for run in measured}
        attempted = [case for case in cases if case.id in attempted_ids]
        memory_summary = {
            "peak_process_memory": round(peak_rss, 1),
            "peak_system_memory_percent": round(peak_percent, 1),
            "memory_pressure_events": pressure_events,
            "samples": len(memory_samples),
            "decision": decision.to_dict(),
        }
        # Метрики считаются по всем ответам модели (в том числе дописанным в
        # прошлом запуске), но только по тем кейсам, которые она действительно
        # получила: непройденные из-за отмены кейсы — это не ошибка модели.
        case_by_id = {case.id: case for case in cases}
        metrics = metrics_mod.score_run(
            attempted,
            [self._prediction(run, case_by_id=case_by_id) for run in measured],
            memory=memory_summary,
        )
        metrics["planned_cases"] = len(cases)
        # Паспорт пишется до метрик: метрики ссылаются на него, и файл метрик
        # обязан быть самодостаточным (с версиями и digest модели внутри).
        self._write_metadata(model, info, metrics, model_dir)
        self._write_json(model_dir / METRICS_FILE, metrics)
        if cancelled:
            status = "cancelled"
        else:
            status = "ok" if confirmed else "ok_unload_failed"
        self.on_progress(
            f"{model}: {status}, кейсов {len(measured)}, "
            f"issue_f1={metrics['metrics'].get('issue_f1')}"
        )
        result = ModelRunResult(
            model=model,
            status=status,
            cases=len(measured),
            planned_cases=len(cases),
            reports=measured,
            unloaded_before=unloaded,
            unload_confirmed=confirmed,
            metrics=metrics,
            memory=memory_summary,
        )
        return result

    # -- один кейс --------------------------------------------------------------
    def _run_case(
        self,
        model: str,
        case: s.DatasetCase,
        repeat_index: int,
        *,
        cancel: Callable[[], bool] | None,
    ) -> CaseRun:
        payload = case.prompt_payload()
        messages = self.prompt.messages(payload)
        started = self.clock()
        error = ""
        repair_attempted = False
        try:
            chat = self._chat(model, messages, cancel=cancel)
            raw_text = chat.text
            analysis, errors = s.parse_analysis(
                raw_text,
                expected_replica_id=case.replica_id,
                target_text=case.target_text,
            )
            repairs = 0
            while analysis is None and repairs < self.max_repairs:
                repair_attempted = True
                repairs += 1
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": raw_text[:2000]},
                    {"role": "user", "content": REPAIR_INSTRUCTION},
                ]
                chat = self._chat(model, repair_messages, cancel=cancel)
                raw_text = chat.text
                analysis, errors = s.parse_analysis(
                    raw_text,
                    expected_replica_id=case.replica_id,
                    target_text=case.target_text,
                )
            if analysis is None:
                error = ",".join(errors)
        except Exception as exc:  # noqa: BLE001 — сбой одного кейса не роняет прогон
            logger.warning("Кейс %s на %s: %s", case.id, model, exc)
            return CaseRun(
                case_id=case.id,
                category=case.category,
                repeat=repeat_index,
                raw_text="",
                latency_sec=self.clock() - started,
                error=f"{type(exc).__name__}: {exc}",
                options=dict(self.options),
            )
        return CaseRun(
            case_id=case.id,
            category=case.category,
            repeat=repeat_index,
            raw_text=raw_text,
            latency_sec=self.clock() - started,
            tokens_per_second=chat.tokens_per_second,
            prompt_tokens=chat.prompt_eval_count,
            completion_tokens=chat.eval_count,
            repair_attempted=repair_attempted,
            error=error,
            options=dict(self.options),
        )

    def _chat(self, model: str, messages: list[dict], *, cancel: Callable[[], bool] | None):
        """Один вызов модели с JSON-схемой и коротким keep-alive."""
        return self.client.chat(
            model,
            messages,
            schema=s.analysis_json_schema(),
            options=dict(self.options),
            keep_alive=DEFAULT_KEEP_ALIVE,
            cancel=cancel,
        )

    # -- вспомогательное --------------------------------------------------------
    def _prediction(self, run: CaseRun, *, case_by_id: dict[str, s.DatasetCase]):
        case = case_by_id.get(run.case_id)
        analysis = None
        errors: tuple[str, ...] = ()
        if run.raw_text and case is not None:
            analysis, problem_list = s.parse_analysis(
                run.raw_text,
                expected_replica_id=case.replica_id,
                target_text=case.target_text,
            )
            errors = tuple(problem_list)
        if not run.raw_text and run.error:
            errors = (run.error,)
        return metrics_mod.Prediction(
            case_id=run.case_id,
            category=run.category,
            raw_text=run.raw_text,
            analysis=analysis,
            errors=errors,
            latency_sec=run.latency_sec,
            tokens_per_second=run.tokens_per_second,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            repair_attempted=run.repair_attempted,
        )

    def _sample_memory(self) -> dict:
        decision = memory.decide_current(
            thresholds=self.thresholds, sensor=self.sensor
        )
        payload = decision.to_dict()
        if self.sensor is None:
            payload["rss_mb"] = _process_rss_mb()
        payload["level"] = decision.level
        payload["reason"] = decision.reason
        return payload

    def _combine_reports(
        self,
        model: str,
        cases: Sequence[s.DatasetCase],
        raw_path: Path,
        fresh: Sequence[CaseRun],
    ) -> list[CaseRun]:
        """Все ответы модели: дописанные сейчас и уже лежавшие в raw.jsonl.

        При `--resume` метрики обязаны считаться по всему прогону, а не только по
        новым кейсам, иначе повторный запуск покажет другую цифру на том же ответе.
        """
        stored: dict[tuple[str, int], CaseRun] = {}
        for run in self._read_jsonl(raw_path):
            stored[(run.case_id, run.repeat)] = run
        for run in fresh:
            stored[(run.case_id, run.repeat)] = run
        order = {case.id: index for index, case in enumerate(cases)}
        return sorted(
            stored.values(), key=lambda run: (order.get(run.case_id, len(order)), run.repeat)
        )

    def _completed_cases(self, raw_path: Path) -> set[str]:
        return {run.case_id for run in self._read_jsonl(raw_path)}

    def _unload_everything(self) -> list[str]:
        """Выгружает все модели: так требует CRITICAL/HARD STOP (§11.2)."""
        try:
            unloaded = self.client.unload_others(keep=None)
        except Exception as exc:  # noqa: BLE001 — не смогли выгрузить, сообщаем
            logger.warning("Не удалось выгрузить модели: %s", exc)
            return []
        if unloaded:
            self.on_progress(f"выгружены модели: {', '.join(unloaded)}")
        return unloaded

    def _unload_model(self, model: str) -> bool:
        """Выгружает модель и проверяет освобождение по списку загруженных."""
        try:
            self.client.unload(model)
            still_loaded = self.client.loaded_models(exclude=None)
        except Exception as exc:  # noqa: BLE001 — проверим факт, но не упадём
            logger.warning("Не удалось выгрузить %s: %s", model, exc)
            return False
        if model in still_loaded:
            self.on_progress(f"{model}: модель осталась в памяти после выгрузки")
            return False
        return True

    def _write_metadata(
        self, model: str, info: OllamaModel, metrics_report: dict, model_dir: Path
    ) -> None:
        try:
            ollama_version = self.client.version()
        except Exception:  # noqa: BLE001 — версия не повод падать
            ollama_version = ""
        metadata = versioning.build_run_metadata(
            model_tag=model,
            dataset_version=self.dataset_version,
            prompt_version=self.prompt.version,
            schema_version=s.SCHEMA_VERSION,
            ollama_version=ollama_version,
            model_digest=info.digest,
            model_size_gb=info.size_gb,
            context=int(self.options.get("num_ctx", DEFAULT_NUM_CTX)),
            temperature=float(self.options.get("temperature", DEFAULT_TEMPERATURE)),
            seed=self.options.get("seed"),
            options=self.options,
        )
        versioning.write_run_metadata(model_dir / RUN_FILE, metadata)
        metrics_report["metadata"] = metadata.to_dict()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> list[CaseRun]:
        if not path.exists():
            return []
        runs: list[CaseRun] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                logger.warning("Битая строка в %s — пропускаю", path)
                continue
            if isinstance(raw, dict):
                runs.append(CaseRun.from_dict(raw))
        return runs


def _process_rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001 — метрика не повод падать
        return 0.0


def dry_run_client(cases: Sequence[s.DatasetCase], **kwargs) -> FakeOllamaClient:
    """Клиент для `--dry-run`: отвечает gold-аннотациями без моделей и сети."""
    gold = {
        case.replica_id: [item.to_dict() for item in case.expected] for case in cases
    }
    return FakeOllamaClient(gold_by_replica=gold, **kwargs)
