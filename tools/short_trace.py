#!/usr/bin/env python3
"""Живой разбор обрывов коротких реплик: где именно теряется начало и конец.

Инструмент для §13–§15 пакета UPDATE 2. Он **не** второй пайплайн: проект
разбирается и готовится теми же функциями, что и в приложении
(`store.parse_project`, `project_analysis.prepare_replica`, `audio_pipeline.
render_dialogue`), а трейс лишь наблюдает за стадиями.

Что делает:

1. берёт регрессионный диалог (`tests/fixtures/short_dialogue_ru.txt`) или свой
   текст, создаёт проект, готовит текст и рендерит его настоящей моделью;
2. по каждому куску сравнивает стадии `01_raw_engine` → `03_post_edge_trim` →
   `04_final_replica` по числам: сколько сэмплов снято обрезкой тишины и **какого
   уровня был снятый край** (слово или тишина);
3. с `--asr` дополнительно расшифровывает сырой и финальный WAV по словам и
   проверяет, на месте ли первое и последнее ожидаемое слово.

Ответ, который выдаёт инструмент: «модель не сказала» (слова нет уже в сыром
выходе) или «пайплайн срезал» (слово есть в сыром и пропало в финальном).

    ./venv/bin/python tools/short_trace.py --list-voices
    ./venv/bin/python tools/short_trace.py --voice <id> --asr
    ./venv/bin/python tools/short_trace.py --voice <id> --short --strategy same_speaker_context
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import (
    audio_pipeline,
    config,
    project_analysis,
    synthesis_trace,
)
from backend.audio_pipeline import (
    QaSettings,
    RenderSettings,
    SpeakerSettings,
)
from backend.db.connection import init_db
from backend.db.store import get_projects_store
from backend.dialogue_parser import Replica
from backend.engines.base import SAMPLE_RATE
from backend.voices_store import get_store as get_voices_store

FIXTURE = ROOT / "tests" / "fixtures" / "short_dialogue_ru.txt"


def _unload_engines() -> None:
    """Выгружает поднятые движки: трейс уже собран, модели больше не нужны.

    Whisper для расшифровки — отдельный процесс на ~1.6 ГБ, а TTS-модель занимает
    несколько гигабайт. Держать их одновременно на машине пользователя незачем.
    """
    from backend.engines.registry import created_engines

    for engine in created_engines().values():
        if not engine.is_loaded:
            continue
        try:
            engine.unload()
            print(f"Движок {engine.id} выгружен — память возвращена")
        except Exception as exc:  # noqa: BLE001 — выгрузка не повод падать
            print(f"Движок {engine.id} не выгружен: {exc}")


def _voices(engine: str | None = None):
    voices = [voice for voice in get_voices_store().list() if not voice.is_demo]
    if engine:
        voices = [voice for voice in voices if voice.engine == engine]
    return voices


def _pick_voice(engine: str | None, wanted: str | None):
    store = get_voices_store()
    if wanted:
        voice = store.get(wanted)
        if voice is None:
            raise SystemExit(f"Голос {wanted} не найден")
        return voice
    voices = _voices(engine)
    if not voices:
        raise SystemExit("Нет ни одного сохранённого голоса — загрузите голос в дашборде")
    # Самый свежий: обычно это тот голос, которым пользователь и работает.
    return max(voices, key=lambda item: item.created_at)


def _prepare(project_id: str, by_speaker: dict[str, object], auto_accent: bool) -> dict:
    """Детерминированная подготовка текста — та же, что и в приложении.

    Вызывается `project_analysis.prepare_replica` — единственная каноническая
    подготовка: инструмент не имеет права считать текст по-своему, иначе он
    измерял бы не тот вход, который уходит в модель у пользователя.
    """
    store = get_projects_store()
    project = store.get_project(project_id)
    preparations = []
    for row in project["replicas"]:
        index = int(row["index"])
        replica = Replica(
            voice=row["speaker"],
            text=row["text"],
            line_number=index + 1,
            overrides=row["overrides"],
            voice_id=row["voice_override"],
        )
        voice = by_speaker.get(row["speaker"])
        speaker = SpeakerSettings(voice_id=voice.id if voice else "")
        preparations.append(
            project_analysis.prepare_replica(
                replica, speaker, voice, index=index, rules=[], auto_accent=auto_accent
            )
        )
    summary = project_analysis.summarize(preparations)
    store.save_analysis(project_id, summary)
    return store.get_project(project_id)


def _replicas(project: dict) -> list[Replica]:
    return [
        Replica(
            voice=row["speaker"],
            text=row["text"],
            line_number=int(row["index"]) + 1,
            overrides=row["overrides"],
            voice_id=row["voice_override"],
            final_text=row["final_text"],
        )
        for row in project["replicas"]
    ]


def _rms_dbfs(audio: np.ndarray) -> float | None:
    data = np.asarray(audio, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(data))))
    return None if rms <= 1e-9 else round(20.0 * float(np.log10(rms)), 2)


def _words(text: str) -> list[str]:
    """Слова для сверки: без ударений, регистра и «ё» (Whisper пишет «е»)."""
    import re

    clean = re.sub(r"\+", "", text or "").lower().replace("ё", "е")
    return [word for word in re.findall(r"[а-яa-z0-9]+", clean) if word]


def _asr_words(path: Path) -> list[str]:
    """Расшифровка WAV по словам (Whisper в отдельном процессе)."""
    from backend import transcribe

    stamps = transcribe.transcribe_words(path)
    return [_normalize(word.word) for word in stamps]


def _normalize(word: str) -> str:
    return str(word).lower().replace("ё", "е").strip(".,!?;:…—-\"'")


def _word_check(expected: str, heard: list[str]) -> dict:
    """Первое/последнее слово и полнота: есть ли ожидаемые слова в расшифровке."""
    target = _words(expected)
    if not target:
        return {}
    heard_set = set(heard)
    missing = [word for word in target if word not in heard_set]
    return {
        "expected_words": target,
        "asr_words": heard,
        "missing_words": missing,
        "first_word_ok": target[0] in heard_set,
        "last_word_ok": target[-1] in heard_set,
        "complete": not missing,
    }


def _analyze(trace_dir: Path, *, asr: bool, limit: int | None = None) -> dict:
    """Сводит трейс в отчёт: стадии, обрезка и (по желанию) проверка слов."""
    records = [
        json.loads(line)
        for line in (trace_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    replicas = []
    for record in records:
        stages = record.get("stages", {})
        trim = stages.get(synthesis_trace.STAGE_TRIM, {})
        item = {
            "index": record["index"],
            "final_text": record.get("final_text") or record.get("source_text"),
            "raw_sec": round(stages.get(synthesis_trace.STAGE_RAW, {}).get("samples", 0) / SAMPLE_RATE, 3),
            "trim_sec": round(trim.get("samples", 0) / SAMPLE_RATE, 3),
            "final_sec": round(stages.get(synthesis_trace.STAGE_FINAL, {}).get("samples", 0) / SAMPLE_RATE, 3),
            "dropped_start_ms": round(trim.get("dropped_start", 0) / SAMPLE_RATE * 1000, 1),
            "dropped_end_ms": round(trim.get("dropped_end", 0) / SAMPLE_RATE * 1000, 1),
            "dropped_tail_rms_dbfs": trim.get("dropped_tail_rms_dbfs"),
            "dropped_head_rms_dbfs": trim.get("dropped_head_rms_dbfs"),
            "qa_status": record.get("qa_status"),
            "qa_wer": record.get("qa_wer"),
            "short_strategy": record.get("short_strategy"),
            "short_fallback": record.get("short_fallback"),
            "asr_raw": None,
            "asr_final": None,
        }
        if asr:
            for label, name in (("asr_raw", synthesis_trace.STAGE_RAW), ("asr_final", synthesis_trace.STAGE_FINAL)):
                path = trace_dir / f"r{record['index']:03d}" / f"{name}.wav"
                if not path.is_file():
                    continue
                heard = _asr_words(path)
                item[label] = _word_check(item["final_text"], heard)
        replicas.append(item)
        if limit and len(replicas) >= limit:
            break
    return {
        "trace_dir": str(trace_dir),
        "replicas": replicas,
        "verdict": _verdict(replicas, asr=asr),
    }


def _verdict(replicas: list[dict], *, asr: bool) -> dict:
    """Итог разбора: срезал ли пайплайн звучащий край и где пропали слова.

    Это и есть тот ответ, ради которого инструмент существует: «модель не
    сказала» и «пайплайн срезал» — разные диагнозы и разные исправления.
    """
    threshold = float(config.EDGE_SILENCE_DB) + 6.0
    cut_speech: list[dict] = []
    model_dropped: list[dict] = []
    pipeline_dropped: list[dict] = []
    for item in replicas:
        for side, level in (
            ("start", item["dropped_head_rms_dbfs"]),
            ("end", item["dropped_tail_rms_dbfs"]),
        ):
            if level is not None and level > threshold:
                cut_speech.append({"index": item["index"], "side": side, "rms_dbfs": level})
        raw, final = item.get("asr_raw"), item.get("asr_final")
        if not raw or not final:
            continue
        if not raw.get("complete", True):
            model_dropped.append(
                {"index": item["index"], "missing": raw.get("missing_words", [])}
            )
        elif not final.get("complete", True):
            pipeline_dropped.append(
                {"index": item["index"], "missing": final.get("missing_words", [])}
            )
    return {
        "cut_audible_edge": cut_speech,
        "model_missing_words": model_dropped,
        "pipeline_missing_words": pipeline_dropped,
        "asr_used": asr,
    }


def _print(report: dict) -> None:
    print(f"\nТрейс: {report['trace_dir']}")
    header = f"{'№':>3} {'текст':<34} {'raw':>6} {'trim':>6} {'final':>6} {'−нач':>6} {'−кон':>6} {'хвост':>7}"
    print(header)
    for item in report["replicas"]:
        text = (item["final_text"] or "")[:33]
        tail = item["dropped_tail_rms_dbfs"]
        print(
            f"{item['index']:>3} {text:<34} {item['raw_sec']:>6.2f} {item['trim_sec']:>6.2f} "
            f"{item['final_sec']:>6.2f} {item['dropped_start_ms']:>6.0f} {item['dropped_end_ms']:>6.0f} "
            f"{(tail if tail is not None else 0):>7.1f}"
        )
        if item.get("asr_raw") or item.get("asr_final"):
            raw = (item.get("asr_raw") or {}).get("complete")
            final = (item.get("asr_final") or {}).get("complete")
            print(
                f"      ASR: raw_complete={raw} final_complete={final} "
                f"raw={item.get('asr_raw', {}).get('asr_words')} "
                f"final={item.get('asr_final', {}).get('asr_words')}"
            )
    verdict = report["verdict"]
    print("\nВердикт:")
    if verdict["cut_audible_edge"]:
        for item in verdict["cut_audible_edge"]:
            print(
                f"  ! обрезка сняла звучащий край: реплика {item['index']}, "
                f"{item['side']}, RMS {item['rms_dbfs']} dBFS"
            )
    else:
        print("  обрезка краёв снимала только тишину (выше порога речи ничего не срезано)")
    if verdict["asr_used"]:
        print(f"  модель не сказала слова: {verdict['model_missing_words'] or '—'}")
        print(f"  пайплайн потерял слова: {verdict['pipeline_missing_words'] or '—'}")
    else:
        print("  слова не проверялись (--asr не задан)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", help="id голоса (по умолчанию — самый свежий сохранённый)")
    parser.add_argument("--engine", default="f5", help="движок для автовыбора голоса (по умолчанию f5)")
    parser.add_argument("--text", help="свой текст вместо регрессионного диалога")
    parser.add_argument("--list-voices", action="store_true", help="показать голоса и выйти")
    parser.add_argument("--short", action="store_true", help="включить слой коротких реплик")
    parser.add_argument(
        "--strategy", default="auto", help="стратегия короткого слоя (auto/direct/same_speaker_context/…)"
    )
    parser.add_argument(
        "--qa",
        default=config.QA_MODE_OFF,
        choices=list(config.QA_MODES),
        help="режим проверки качества во время рендера (по умолчанию off: слушаем сам пайплайн)",
    )
    parser.add_argument("--asr", action="store_true", help="расшифровать raw/final по словам (Whisper)")
    parser.add_argument("--pause-ms", type=int, default=400)
    parser.add_argument("--out", help="куда записать отчёт JSON (по умолчанию — рядом с трейсом)")
    parser.add_argument(
        "--trace-dir",
        help="разобрать уже собранный трейс вместо нового рендера (удобно для повторного --asr)",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="не выгружать модель перед расшифровкой (по умолчанию выгружается)",
    )
    args = parser.parse_args()

    if args.list_voices:
        for voice in _voices():
            print(f"{voice.id}  {voice.name}  {voice.engine}  {voice.created_at}")
        return 0

    if args.trace_dir:
        trace_dir = Path(args.trace_dir).expanduser()
        if not (trace_dir / "trace.jsonl").is_file():
            raise SystemExit(f"В {trace_dir} нет trace.jsonl")
        if args.asr and not args.keep_model:
            _unload_engines()
        report = _analyze(trace_dir, asr=args.asr)
        report["trace_dir"] = str(trace_dir)
        _print(report)
        target = Path(args.out) if args.out else trace_dir / "report.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт: {target}")
        return 0

    init_db()
    voice = _pick_voice(args.engine or None, args.voice)
    text = Path(args.text).read_text(encoding="utf-8") if args.text else FIXTURE.read_text(encoding="utf-8")
    print(f"Голос: {voice.name} ({voice.id}, {voice.engine})")

    store = get_projects_store()
    project = store.create_project(
        f"short trace {uuid.uuid4().hex[:6]}", source_text=text, mode=config.PROJECT_MODE_DIALOGUE
    )
    store.parse_project(project["id"])
    store.update_project(
        project["id"],
        speakers={
            speaker["key"]: {"voice_id": voice.id}
            for speaker in store.get_project(project["id"])["speakers"]
        },
    )
    project = _prepare(
        project["id"],
        {speaker["key"]: voice for speaker in store.get_project(project["id"])["speakers"]},
        True,
    )
    print(
        f"Проект: {project['id']} «{project['name']}» — "
        f"{len(project['replicas'])} реплик, подготовка {project['analysis_status']}"
    )

    settings = RenderSettings(
        pause_ms=args.pause_ms,
        output_format="wav",
        auto_accent=True,
        qa=QaSettings.for_mode(args.qa),
        require_prepared=True,
        short_utterance=(
            audio_pipeline.ShortUtteranceSettings(enabled=True, strategy=args.strategy)
            if args.short
            else None
        ),
    )
    speakers = {
        speaker["key"]: SpeakerSettings.from_dict({"voice_id": speaker["voice_id"]})
        for speaker in project["speakers"]
    }
    job_id = f"trace-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    result = asyncio.run(
        audio_pipeline.render_dialogue(
            job_id, _replicas(project), speakers, settings
        )
    )
    print(f"Рендер: {result.duration_sec:.1f} с аудио за {time.monotonic() - started:.1f} с ({result.output_path})")

    trace_dir = Path(config.OUTPUT_DIR) / "trace" / job_id
    if not trace_dir.is_dir():
        raise SystemExit(
            "Трейс не найден: запустите инструмент с TTS_SYNTHESIS_TRACE=1 "
            f"(ожидался каталог {trace_dir})"
        )
    if args.asr and not args.keep_model:
        # Whisper для расшифровки — отдельный процесс на ~1.6 ГБ; модель уже
        # сделала свою работу, и держать их одновременно незачем.
        _unload_engines()
    report = _analyze(trace_dir, asr=args.asr)
    report["job_id"] = job_id
    report["project_id"] = project["id"]
    report["voice"] = {"id": voice.id, "name": voice.name, "engine": voice.engine}
    report["settings"] = {
        "short_utterance": None if settings.short_utterance is None else settings.short_utterance.to_dict(),
        "qa": args.qa,
        "pause_ms": args.pause_ms,
    }
    _print(report)
    target = Path(args.out) if args.out else trace_dir / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {target}")
    print(f"Аудио стадий: {trace_dir}/rNNN/01_raw_engine.wav … 04_final_replica.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
