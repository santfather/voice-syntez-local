#!/usr/bin/env python3
"""Живая A/B-проверка коротких реплик: слой выключен против включённого.

Отвечает на жалобу «короткие фразы проглатывают окончания»: рендерит один и тот же
короткий диалог дважды — с выключенным и включённым слоем коротких реплик — и
сравнивает **измеренные** величины: длительность куска и WER из Smart QA (Whisper
сверяет то, что услышал, с текстом реплики).

Модели поднимает сервер: инструмент ничего не мокает и не запускает сам. Требуется
живой дашборд с голосом F5.

    ./venv/bin/python tools/short_ab.py --voice <id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm_analyzer_smoke import review_candidates
from tools.smoke import Api, SmokeError, wait_for_job

# Короткие фразы из постановки плюс одна длинная — контроль, что слой её не трогает.
SHORT_PHRASES = ("Да.", "Спасибо.", "Нет.", "Как дела?")
LONG_PHRASE = "Сегодня хорошая погода, и я решил прогуляться по набережной до заката."
SPEAKER = "ИВАН"


def _build_project(api: Api, voice_id: str, text: str) -> str:
    project = api.json(
        "POST",
        "/api/projects",
        {"name": f"short A/B {uuid.uuid4().hex[:6]}", "source_text": text},
    )
    project_id = project["id"]
    api.json("POST", f"/api/projects/{project_id}/parse", {})
    api.json(
        "PATCH",
        f"/api/projects/{project_id}",
        {"speakers": {SPEAKER: {"voice_id": voice_id}}},
    )
    # Назначение голоса устаревает подготовку (текст зависит от движка): без анализа
    # рендер запрещён. Для коротких реплик достаточно детерминированного прохода, а
    # найденные слова решаем «оставить как есть» — текст не меняется, и A/B остаётся
    # честным сравнением одного и того же текста.
    api.json("POST", f"/api/projects/{project_id}/analyze", {"auto_accent": True})
    review_candidates(api, project_id, {}, accept=False, max_rounds=4)
    return project_id


def _render(api: Api, project_id: str, *, short_enabled: bool, timeout: float) -> dict:
    started = time.monotonic()
    job = api.json(
        "POST",
        f"/api/projects/{project_id}/render",
        {
            "qa": "smart",
            "short_utterance": {
                "enabled": short_enabled,
                "strategy": "auto" if short_enabled else "direct",
            },
        },
    )
    result = wait_for_job(api, job["job_id"], timeout, "рендер")
    project = api.json("GET", f"/api/projects/{project_id}")
    takes = []
    for row in project["replicas"]:
        take = (row.get("takes") or [None])[0] or {}
        takes.append(
            {
                "index": int(row["index"]),
                "text": row["text"],
                "final_text": row["final_text"],
                "duration_sec": take.get("duration_sec"),
                "qa_wer": (take.get("qa") or {}).get("wer") if take.get("qa") else None,
                "qa_status": (take.get("qa") or {}).get("status") if take.get("qa") else None,
                "short": (take.get("parameters") or {}).get("short_utterance"),
            }
        )
    return {
        "job_id": job["job_id"],
        "status": result.get("status"),
        "wall_seconds": round(time.monotonic() - started, 1),
        "takes": takes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B коротких реплик на живом сервере")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--report", type=Path, default=ROOT / "output" / "short-ab")
    args = parser.parse_args(argv)

    api = Api(args.base_url, timeout=args.timeout)
    text = "\n".join(f"{SPEAKER}: {phrase}" for phrase in (*SHORT_PHRASES, LONG_PHRASE))
    report: dict = {"phrases": list(SHORT_PHRASES), "long": LONG_PHRASE, "voice": args.voice}
    try:
        off_id = _build_project(api, args.voice, text)
        report["off"] = _render(api, off_id, short_enabled=False, timeout=args.timeout)
        on_id = _build_project(api, args.voice, text)
        report["on"] = _render(api, on_id, short_enabled=True, timeout=args.timeout)
    except SmokeError as exc:
        print(f"A/B не выполнен: {exc}", file=sys.stderr)
        return 2

    args.report.mkdir(parents=True, exist_ok=True)
    path = args.report / f"short-ab-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'фраза':<50} {'выкл: с/WER':<22} {'вкл: с/WER':<22} стратегия")
    for left, right in zip(report["off"]["takes"], report["on"]["takes"]):
        short = (right.get("short") or {}).get("strategy") or "—"
        l = f"{left['duration_sec']}/{left['qa_wer']}"
        r = f"{right['duration_sec']}/{right['qa_wer']}"
        print(f"{left['text'][:48]:<50} {l:<22} {r:<22} {short}")
    print(f"\nОтчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
