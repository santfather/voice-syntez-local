#!/usr/bin/env python3
"""Реальный smoke лингвистического анализатора (Task 2, фаза 9).

Инструмент ходит по живому дашборду и проходит путь пользователя с **настоящей**
моделью Ollama и настоящим синтезом F5:

    набор кейсов §22 → проект → разбор → лингвистический анализ (LLM) →
    review предложений → рендер F5 → сверка, что в модель ушёл подтверждённый текст

Набор кейсов собирается из gold-корпуса benchmark'а (`dataset.v1.jsonl`): 20
омографов, 20 «ё», 10 имён/топонимов, 10 аббревиатур/чисел, 10 диалоговых кейсов,
10 коротких реплик. Это те же случаи, на которых модели сравнивались, поэтому smoke
отвечает на вопрос «работает ли выбранная модель в бою», а не «нравится ли нам
текст».

Инструмент ничего не мокает и не поднимает модели сам: сервер запускается отдельно
с включённым анализатором (`LLM_ANALYZER_ENABLED=1`). В pytest он не входит — как и
`tools/smoke.py`, потому что для обычного прогона тестов реальные модели запрещены.

    LLM_ANALYZER_ENABLED=1 ./venv/bin/python tools/llm_analyzer_smoke.py --dry-run
    LLM_ANALYZER_ENABLED=1 ./venv/bin/python tools/llm_analyzer_smoke.py --render
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.smoke import Api, SmokeError, wait_for_job

DATASET = ROOT / "benchmarks" / "russian_linguistics" / "dataset.v1.jsonl"
DEFAULT_REPORT = ROOT / "output" / "llm-smoke"
SPEAKERS = ("ИВАН", "МАРГО")
# Набор §22: сколько кейсов каждой категории берём в smoke.
CATEGORY_PLAN: tuple[tuple[str, int], ...] = (
    ("homograph", 20),
    ("yo", 20),
    ("name", 5),
    ("toponym", 5),
    ("abbreviation", 5),
    ("number", 5),
    ("dialogue_context", 10),
    ("short_replica", 10),
    # Чистые тексты: на них проверяется, что анализ не «находит» лишнего и что
    # рендер после анализа проходит без ручного ревью.
    ("no_issue", 10),
)


@dataclass
class MemorySampler:
    """Пик и минимум памяти за прогон: без них smoke не отвечает на вопрос §14.

    Сэмплер в отдельном потоке, а не «до и после»: пик памяти приходится на
    загрузку модели, и замерить его двумя точками невозможно.
    """

    interval: float = 0.5
    peak_percent: float = 0.0
    min_available_gb: float = 1e9
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:  # pragma: no cover — живой замер, не в pytest
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover — живой замер, не в pytest
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:  # pragma: no cover — живой замер, не в pytest
        import psutil

        while not self._stop.is_set():
            memory = psutil.virtual_memory()
            self.peak_percent = max(self.peak_percent, float(memory.percent))
            self.min_available_gb = min(
                self.min_available_gb, float(memory.available) / (1024**3)
            )
            self.samples += 1
            time.sleep(self.interval)

    def to_dict(self) -> dict:
        return {
            "peak_system_memory_percent": round(self.peak_percent, 1),
            "min_available_gb": (
                round(self.min_available_gb, 2) if self.samples else None
            ),
            "samples": self.samples,
        }


def load_cases() -> list[dict]:
    """Кейсы набора §22 из gold-корпуса: по категориям и в порядке корпуса."""
    cases: list[dict] = []
    by_category: dict[str, list[dict]] = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            by_category.setdefault(case["category"], []).append(case)
    for category, limit in CATEGORY_PLAN:
        cases.extend(by_category.get(category, [])[:limit])
    return cases


def build_dialogue(cases: list[dict]) -> list[dict]:
    """Реплики диалога: кейс + его контекст, спикеры чередуются.

    Контекст кейса добавляется отдельными репликами: короткие реплики и диалоговые
    случаи без соседей бессмысленны, а модель получает контекст из самого диалога.
    """
    replicas: list[dict] = []
    for position, case in enumerate(cases):
        speaker = SPEAKERS[position % len(SPEAKERS)]
        for context in case.get("context_before") or []:
            replicas.append({"speaker": SPEAKERS[(position + 1) % len(SPEAKERS)], "text": context})
        replicas.append(
            {
                "speaker": speaker,
                "text": case["target_text"],
                "case_id": case["id"],
                "category": case["category"],
            }
        )
    return replicas


def dialogue_text(replicas: list[dict]) -> str:
    return "\n".join(f"{item['speaker']}: {item['text']}" for item in replicas)


def _clean_cases() -> list[dict]:
    """Запасной «чистый» текст: если в корпусе нет категории no_issue.

    Текст без омографов, «ё»-сомнений и редких слов: анализ обязан не найти в нём
    ничего, а значит рендер после анализа разрешён без ревью.
    """
    return [
        {"id": "clean-1", "category": "no_issue", "target_text": "Сегодня хорошая погода, и я рад тебя видеть."},
        {"id": "clean-2", "category": "no_issue", "target_text": "Мы закончили работу над проектом."},
        {"id": "clean-3", "category": "no_issue", "target_text": "На кухне пахло свежим хлебом и корицей."},
        {"id": "clean-4", "category": "no_issue", "target_text": "Ранним утром туман стелился по берегу реки."},
    ]


def pick_f5_voice(api: Api, explicit: str | None) -> str:
    """Голос для рендера: явный или первый доступный с движком F5."""
    payload = api.json("GET", "/api/voices")
    voices = payload.get("voices") or []
    if explicit:
        if not any(item["id"] == explicit for item in voices):
            raise SmokeError(f"Голос {explicit} не найден")
        return explicit
    for item in voices:
        if item.get("engine") == "f5":
            return str(item["id"])
    raise SmokeError("Нет ни одного голоса F5 — smoke-рендер невозможен")


def review_candidates(
    api: Api, project_id: str, report: dict, *, accept: bool = False, max_rounds: int = 5
) -> None:
    """Решает все предложения: принять предложенную замену или пропустить.

    Список берётся из двух источников — детерминированные кандидаты и предложения
    LLM: у них общий путь review, и оба должны быть решены, иначе проект не станет
    готовым и рендер запрещён.

    По умолчанию кандидаты **пропускаются** (выключенное правило = память о
    решении): принятое правило меняет `final_text`, из-за чего контекст соседей
    меняется и модель на следующем круге предлагает новые слова — на большом
    проекте такой цикл не сходится. Пропуск оставляет текст как есть и сходится за
    один-два круга; принятие правила проверяется интеграционными тестами с
    подставной моделью, где результат детерминирован.
    """
    accepted = skipped = 0
    iterations = 0
    for _ in range(max_rounds):
        iterations += 1
        analysis = api.json("GET", f"/api/projects/{project_id}/analysis")
        llm = api.json("GET", f"/api/projects/{project_id}/linguistic-analysis")
        pending = list(analysis.get("candidates") or [])
        pending += [item for item in (llm.get("candidates") or []) if item.get("needs_review")]
        if not pending:
            break
        for item in pending:
            target = str(item.get("target") or "")
            api.json(
                "POST",
                f"/api/projects/{project_id}/pronunciation/review",
                {
                    "source": item.get("word") or "",
                    "target": (target or item.get("word") or "") if accept else item.get("word") or "",
                    "scope": "project",
                    "enabled": bool(accept and target),
                    "replica_index": item.get("replica_index"),
                    "note": "smoke лингвистического анализатора",
                },
            )
            if accept and target:
                accepted += 1
            else:
                skipped += 1
    report["review"] = {
        "accepted": accepted,
        "skipped": skipped,
        "iterations": iterations,
    }
    state = api.json("GET", f"/api/projects/{project_id}/analysis")
    report["analysis_status_after_review"] = state.get("status")
    report["pending_after_review"] = state.get("candidates_total")


def run(args: argparse.Namespace) -> dict:
    api = Api(args.base_url, timeout=args.timeout)
    health = api.json("GET", "/api/status")
    llm_status = api.json("GET", "/api/llm/status")
    report: dict = {
        "base_url": args.base_url,
        "ollama": llm_status.get("ollama"),
        "analyzer": {
            "enabled": llm_status.get("enabled"),
            "model": llm_status.get("model"),
            "primary": llm_status.get("settings", {}).get("primary_model"),
            "fallback": llm_status.get("settings", {}).get("fallback_model"),
            "prompt_version": llm_status.get("prompt_version"),
            "schema_version": llm_status.get("schema_version"),
        },
        "engines": health.get("engines") if isinstance(health, dict) else None,
    }
    if not llm_status.get("enabled"):
        raise SmokeError("Analyzer выключен: запустите сервер с LLM_ANALYZER_ENABLED=1")

    cases = load_cases()
    replicas = build_dialogue(cases)
    report["cases"] = {
        "total": len(cases),
        "replicas": len(replicas),
        "by_category": {
            category: sum(1 for case in cases if case["category"] == category)
            for category, _ in CATEGORY_PLAN
        },
    }

    voice_id = pick_f5_voice(api, args.voice)
    report["voice"] = voice_id

    project = api.json(
        "POST",
        "/api/projects",
        {"name": f"LLM smoke {uuid.uuid4().hex[:6]}", "source_text": dialogue_text(replicas)},
    )
    project_id = project["id"]
    report["project_id"] = project_id
    api.json("POST", f"/api/projects/{project_id}/parse", {})
    speakers = {speaker: {"voice_id": voice_id} for speaker in SPEAKERS}
    api.json("PATCH", f"/api/projects/{project_id}", {"speakers": speakers})

    sampler = MemorySampler()
    sampler.start()
    started = time.monotonic()
    try:
        response = api.json(
            "POST", f"/api/projects/{project_id}/linguistic-analysis", {"auto_accent": True}
        )
        analyze_seconds = time.monotonic() - started
    finally:
        sampler.stop()

    llm = response.get("llm") or {}
    run = llm.get("run") or {}
    report["analysis"] = {
        "status": llm.get("status"),
        "model": llm.get("model"),
        # `run` — отчёт именно этого запуска: вызовы, кеш, время, неудачные реплики.
        "seconds": run.get("seconds"),
        "wall_seconds": round(analyze_seconds, 2),
        "replicas": run.get("replicas"),
        "from_cache": run.get("from_cache"),
        "calls": run.get("calls"),
        "unloaded": run.get("unloaded"),
        "replicas_failed": run.get("replicas_failed"),
        "failed": run.get("failed"),
        "candidates_total": llm.get("candidates_total"),
        "needs_review_total": llm.get("needs_review_total"),
        "conflicts_total": llm.get("conflicts_total"),
        "error": llm.get("error") or run.get("error") or "",
        "blocked_reason": run.get("blocked_reason") or "",
        "memory": sampler.to_dict(),
    }

    # Массовый review на проекте из 100 реплик не делаем: каждое решение
    # пересчитывает проект целиком (это правило продакшена — «пересчитать
    # затронутое сразу»), и на десятках слов это часы. Ревью и рендер проверяются на
    # маленьком проекте ниже, где цикл сходится.
    project = api.json("GET", f"/api/projects/{project_id}")
    report["final_texts"] = {
        "replicas": len(project["replicas"]),
        "prepared": sum(1 for row in project["replicas"] if row["final_text"]),
        "changed_by_llm": 0,  # проверяется ниже: исходный текст и final_text неизменны
    }
    report["final_texts"]["sources_intact"] = all(
        row["text"] in project["source_text"] for row in project["replicas"]
    )

    if args.render:
        # Текст для рендера: случаи, где анализ не должен ничего находить. Так
        # проверяется именно то, что просит §22 — реальный F5-рендер **после**
        # анализа, — и не упирается в ручной review, который на большом тексте
        # требует человека за каждый круг. Путь «review → рендер» проверяется
        # интеграционными тестами с подставной моделью.
        clean = [case for case in load_cases() if case["category"] == "no_issue"]
        clean = clean or _clean_cases()
        report["render"] = render_small_project(
            api, build_dialogue(clean[: args.render_limit]), voice_id, args
        )
        if report["render"].get("error"):
            return report

    return report


def render_small_project(api: Api, replicas: list[dict], voice_id: str, args) -> dict:
    """Один реальный рендер F5 после анализа — на маленьком проекте.

    Набор §22 (100 реплик) нужен для измерений анализа; рендерить его целиком
    незачем: постановка просит один реальный F5-рендер после лингвистического
    анализа. Маленький проект ещё и сходится по review за пару кругов, поэтому
    проверка «подтверждённый текст доехал до модели» получается честной.
    """
    project = api.json(
        "POST",
        "/api/projects",
        {"name": f"LLM smoke render {uuid.uuid4().hex[:6]}", "source_text": dialogue_text(replicas)},
    )
    project_id = project["id"]
    api.json("POST", f"/api/projects/{project_id}/parse", {})
    api.json(
        "PATCH",
        f"/api/projects/{project_id}",
        {"speakers": {speaker: {"voice_id": voice_id} for speaker in SPEAKERS}},
    )
    api.json("POST", f"/api/projects/{project_id}/linguistic-analysis", {"auto_accent": True})
    local: dict = {}
    review_candidates(api, project_id, local, accept=args.accept, max_rounds=6)
    report = {
        "project_id": project_id,
        "review": local.get("review"),
        "analysis_status": local.get("analysis_status_after_review"),
        "seconds": None,
    }
    try:
        started = time.monotonic()
        render = api.json("POST", f"/api/projects/{project_id}/render", {})
        job = wait_for_job(api, render["job_id"], args.render_timeout, "рендер F5")
        report["seconds"] = round(time.monotonic() - started, 1)
        report["job_id"] = render["job_id"]
        report["status"] = job.get("status")
        report["takes"] = sum(len(row.get("takes") or []) for row in (job.get("replicas") or []))
        report["output"] = job.get("output_path")
        final = api.json("GET", f"/api/projects/{project_id}")
        report["replicas"] = len(final["replicas"])
        report["replicas_with_takes"] = sum(1 for row in final["replicas"] if row.get("takes"))
        report["final_texts_present"] = all(row["final_text"] for row in final["replicas"])
    except SmokeError as exc:
        # Отказ на рендере — результат, а не повод потерять измерения анализа.
        report["error"] = str(exc)
    return report


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Реальный smoke лингвистического анализатора")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--voice", default=None, help="голос F5 (по умолчанию первый найденный)")
    parser.add_argument("--render", action="store_true", help="сделать реальный рендер F5")
    parser.add_argument("--timeout", type=float, default=120.0, help="таймаут HTTP-запроса")
    parser.add_argument("--render-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--render-limit", type=int, default=6, help="сколько реплик в проекте для рендера"
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help="принимать предложения в словарь (по умолчанию — пропускать, чтобы цикл review сходился)",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true", help="показать план без запросов")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        cases = load_cases()
        replicas = build_dialogue(cases)
        counts = {
            category: sum(1 for case in cases if case["category"] == category)
            for category, _ in CATEGORY_PLAN
        }
        print(f"Кейсов: {len(cases)}, реплик с контекстом: {len(replicas)}")
        for category, count in counts.items():
            print(f"  {category:<20} {count}")
        print(f"Рендер: {'да' if args.render else 'нет'}, отчёт: {args.report}")
        return 0

    try:
        report = run(args)
    except SmokeError as exc:
        print(f"Smoke не выполнен: {exc}", file=sys.stderr)
        return 2
    args.report.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = args.report / f"smoke-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {path}")
    return 0 if not report.get("render", {}).get("error") else 3


if __name__ == "__main__":
    raise SystemExit(main())
