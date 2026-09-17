"""Версии прогона: без них benchmark невоспроизводим (§4).

Результат «модель A лучше модели B» ничего не значит, если неизвестно, какой был
prompt, какая ревизия корпуса, какой digest модели и сколько было контекста.
Поэтому каждый прогон сохраняет полный набор версий и параметров, а не только
итоговые метрики.

Digest важен отдельно от тега: `qwen3:8b` со временем может указывать на другую
сборку, и сравнение «до/после» окажется сравнением двух разных моделей. Поэтому в
отчёте фиксируется и тег, и digest, и размер — то, что видно в `ollama list`.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("tts.llm.versioning")

BENCHMARK_VERSION = "1"
PROMPT_VERSION = "1"
# Версия схемы живёт в `schemas.SCHEMA_VERSION`; здесь — только для отчёта.

GIT_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class RunMetadata:
    """Паспорт прогона: чем именно получены числа."""

    model_tag: str
    benchmark_version: str = BENCHMARK_VERSION
    dataset_version: str = "1"
    prompt_version: str = PROMPT_VERSION
    schema_version: str = "1"
    git_commit: str = ""
    timestamp: str = ""
    os_version: str = ""
    machine: str = ""
    chip: str = ""
    total_ram_gb: float = 0.0
    ollama_version: str = ""
    model_digest: str = ""
    model_size_gb: float = 0.0
    context: int = 8192
    temperature: float = 0.0
    seed: int | None = 0
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "benchmark_version": self.benchmark_version,
            "dataset_version": self.dataset_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "os_version": self.os_version,
            "machine": self.machine,
            "chip": self.chip,
            "total_ram_gb": self.total_ram_gb,
            "ollama_version": self.ollama_version,
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "model_size_gb": self.model_size_gb,
            "context": self.context,
            "temperature": self.temperature,
            "seed": self.seed,
            "options": dict(self.options),
        }
        return payload


def git_commit(short: bool = True) -> str:
    """Коммит репозитория на момент прогона; без git — пустая строка.

    Ошибку не поднимаем: benchmark должен работать и из распакованного архива,
    где `.git` нет.
    """
    command = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git недоступен: %s", exc)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def chip_name() -> str:
    """Имя процессора/чипа: на macOS `sysctl` точнее, чем `platform`."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SEC,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def machine_info() -> dict:
    """ОС, чип, объём памяти — то, к чему относятся замеры."""
    total_ram_gb = 0.0
    try:
        import psutil

        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception as exc:  # noqa: BLE001 — метрика не повод падать
        logger.debug("Не удалось прочитать объём памяти: %s", exc)
    return {
        "os_version": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "chip": chip_name(),
        "total_ram_gb": total_ram_gb,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_run_metadata(
    *,
    model_tag: str,
    dataset_version: str = "1",
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = "1",
    ollama_version: str = "",
    model_digest: str = "",
    model_size_gb: float = 0.0,
    context: int = 8192,
    temperature: float = 0.0,
    seed: int | None = 0,
    options: dict | None = None,
    include_machine: bool = True,
) -> RunMetadata:
    """Собирает паспорт прогона.

    `include_machine=False` нужен тестам: там машина не важна, а вызовы `sysctl`
    и psutil в каждом тесте — лишнее.
    """
    info = machine_info() if include_machine else {
        "os_version": "", "machine": "", "chip": "", "total_ram_gb": 0.0
    }
    return RunMetadata(
        model_tag=model_tag,
        dataset_version=dataset_version,
        prompt_version=prompt_version,
        schema_version=schema_version,
        git_commit=git_commit(),
        timestamp=now_iso(),
        model_digest=model_digest,
        model_size_gb=model_size_gb,
        ollama_version=ollama_version,
        context=context,
        temperature=temperature,
        seed=seed,
        options=dict(options or {}),
        **info,
    )


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    """Паспорт лежит рядом с результатами: отчёт без него не читается."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def baseline_path(output_dir: Path, dataset_version: str = "1") -> Path:
    """Путь baseline-файла: с ним сравниваются будущие прогоны (§14, Phase 7)."""
    return output_dir / f"baseline.v{dataset_version}.json"


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """Сравнивает прогон с baseline и возвращает регрессии/улучшения.

    Сравниваются метрики, которые есть в обоих отчётах: так будущий прогон
    (другая модель, другой prompt) показывает, что именно изменилось, а не
    «стало хуже вообще».
    """
    result: dict = {"regressions": {}, "improvements": {}, "unchanged": []}
    current_metrics = current.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    for name, old_value in baseline_metrics.items():
        if name not in current_metrics:
            continue
        new_value = current_metrics[name]
        if not isinstance(old_value, (int, float)) or not isinstance(new_value, (int, float)):
            if old_value != new_value:
                result["regressions"][name] = {"baseline": old_value, "current": new_value}
            else:
                result["unchanged"].append(name)
            continue
        delta = round(float(new_value) - float(old_value), 4)
        # Для «чем больше, тем лучше» и «чем меньше, тем лучше» знак разный:
        # правило сравнения задаётся в metrics.HIGHER_IS_BETTER.
        from .metrics import HIGHER_IS_BETTER

        better = delta > 0 if HIGHER_IS_BETTER.get(name, True) else delta < 0
        if abs(delta) < 1e-9:
            result["unchanged"].append(name)
        elif better:
            result["improvements"][name] = {"baseline": old_value, "current": new_value, "delta": delta}
        else:
            result["regressions"][name] = {"baseline": old_value, "current": new_value, "delta": delta}
    return result
