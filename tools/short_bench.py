"""Benchmark коротких реплик (short_phrasases.md §1, §25, §26, §30).

Инструмент запускается **в процессе** и поднимает настоящие модели: дашборд перед
прогоном лучше остановить — два процесса с моделями не влезут в память Apple
Silicon. Модели при этом живут в изолированных воркерах (см. README), поэтому
память самого инструмента остаётся небольшой.

Что делает прогон:

1. берёт набор коротких фраз (§1) и контрольные длинные;
2. синтезирует каждую несколькими стратегиями (§4) на нескольких голосах с
   **одним и тем же сидом** — это и есть контролируемое A/B: A = `direct`,
   B = контекст;
3. для стратегий с контекстом режет результат до целевой реплики по надёжной
   границе и честно помечает откат на `direct`, если границы нет (§31);
4. считает объективные метрики: длительность, долю тишины, уровень, LUFS, время
   генерации, ASR-расшифровку, WER и короткий вердикт (§20, §26);
5. складывает WAV для прослушивания и пишет отчёт JSON + Markdown.

Слушательное A/B делает человек: инструмент готовит материал и объективные числа,
но «естественность» не выдумывает (§26).

Запуск:

    ./venv/bin/python tools/short_bench.py --plan          # что будет сгенерировано
    ./venv/bin/python tools/short_bench.py                # прогон (реальные модели)
    ./venv/bin/python tools/short_bench.py --seeds 5 --strategies direct   # §18
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Инструмент запускается из корня репозитория и как `tools/short_bench.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import audio_pipeline, config, qa_screening, take_quality
from backend import short_utterance as su
from backend import short_utterance_boundary as boundary_module
from backend.audio_pipeline import SpeakerSettings
from backend.engines.registry import get_engine
from backend.text_preprocess import normalize
from backend.transcribe import transcribe_audio, word_error_rate
from backend.voices_store import get_store

logger = logging.getLogger("short_bench")

# Фразы из §1: ровно тот набор, по которому заявлена проблема.
SHORT_PHRASES = (
    "Привет!",
    "Как дела?",
    "Спасибо.",
    "Да.",
    "Нет.",
    "Хорошо.",
    "Я понял.",
    "Что случилось?",
    "Почему?",
    "До встречи!",
)
# Контрольные длинные фразы: они обязаны звучать не хуже, чем до изменения (§28).
LONG_PHRASES = (
    "Сегодня хорошая погода, и я рад тебя видеть.",
    "Мы закончили работу над проектом и теперь можем немного отдохнуть.",
    "Он подошёл к окну, посмотрел на улицу и улыбнулся своим мыслям.",
)
# Нейтральная реплика «того же спикера» для benchmark'а: в реальном проекте на её
# месте была бы предыдущая реплика персонажа (§9). Фиксированная — чтобы сравнение
# стратегий было честным.
SAME_SPEAKER_CONTEXT = "Я только что вернулась домой."

DEFAULT_STRATEGIES = (
    su.STRATEGY_DIRECT,
    su.STRATEGY_PUNCTUATION,
    su.STRATEGY_SAME_SPEAKER_CONTEXT,
    su.STRATEGY_SYNTHETIC_CONTEXT,
)


@dataclass
class Variant:
    """Один прогон: фраза × голос × стратегия × сид."""

    phrase: str
    voice_id: str
    voice_name: str
    engine: str
    strategy: str
    seed: int
    synthesis_text: str
    utterance_class: str
    fallback: str = ""
    boundary: dict | None = None
    duration_sec: float = 0.0
    silence_ratio: float = 0.0
    rms_dbfs: float = 0.0
    peak_dbfs: float = 0.0
    lufs: float | None = None
    generation_sec: float = 0.0
    transcription: str = ""
    wer: float | None = None
    verdict: dict | None = None
    audio_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "engine": self.engine,
            "strategy": self.strategy,
            "seed": self.seed,
            "synthesis_text": self.synthesis_text,
            "utterance_class": self.utterance_class,
            "fallback": self.fallback,
            "boundary": self.boundary,
            "duration_sec": round(self.duration_sec, 3),
            "silence_ratio": round(self.silence_ratio, 3),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "lufs": None if self.lufs is None else round(self.lufs, 2),
            "generation_sec": round(self.generation_sec, 2),
            "transcription": self.transcription,
            "wer": None if self.wer is None else round(self.wer, 3),
            "verdict": self.verdict,
            "audio_path": self.audio_path,
            "error": self.error,
        }


@dataclass
class Plan:
    """Что будет сгенерировано: считается до подъёма моделей."""

    phrases: list[str]
    variants: list[Variant] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phrases": self.phrases,
            "variants": [
                {
                    "phrase": item.phrase,
                    "voice_id": item.voice_id,
                    "engine": item.engine,
                    "strategy": item.strategy,
                    "seed": item.seed,
                    "synthesis_text": item.synthesis_text,
                }
                for item in self.variants
            ],
        }


def _slug(text: str) -> str:
    """Короткое имя файла из фразы: буквы и цифры, остальное — подчёркивание."""
    cleaned = "".join(char if char.isalnum() else "_" for char in text.lower())
    return cleaned.strip("_")[:40] or "phrase"


def build_plan(
    *,
    voices: list,
    strategies: tuple[str, ...],
    seeds: int,
    phrases: tuple[str, ...],
    long_phrases: tuple[str, ...],
    carrier: str | None = None,
    side: str | None = None,
    with_long: bool = True,
) -> Plan:
    """План прогона: фраза × голоса × стратегии × сиды.

    Длинные контрольные фразы идут только через `direct`: их задача — показать,
    что обычный путь не изменился, а не сравнить стратегии на длинном тексте.
    """
    plan = Plan(phrases=list(phrases))
    for phrase in phrases:
        context = su.ShortUtteranceContext(
            target_index=0,
            target_text=phrase,
            speaker="bench",
            same_speaker_previous=SAME_SPEAKER_CONTEXT,
            same_speaker_next=SAME_SPEAKER_CONTEXT,
            utterance=su.classify_utterance(phrase),
        )
        for voice in voices:
            for strategy in strategies:
                for seed in range(1, seeds + 1):
                    item = _variant_for(
                        phrase=phrase,
                        voice=voice,
                        strategy=strategy,
                        seed=seed,
                        context=context,
                        carrier=carrier,
                        side=side,
                    )
                    plan.variants.append(item)
    if with_long:
        for phrase in long_phrases:
            for voice in voices:
                context = su.ShortUtteranceContext(
                    target_index=0, target_text=phrase, speaker="bench"
                )
                plan.variants.append(
                    _variant_for(
                        phrase=phrase,
                        voice=voice,
                        strategy=su.STRATEGY_DIRECT,
                        seed=1,
                        context=context,
                    )
                )
    return plan


def _variant_for(
    *,
    phrase: str,
    voice,
    strategy: str,
    seed: int,
    context: su.ShortUtteranceContext,
    carrier: str | None = None,
    side: str | None = None,
) -> Variant:
    plan = su.build_plan(
        context,
        strategy=strategy,
        supports_accents=voice.engine == "f5",
        carrier=carrier,
        side=side,
    )
    return Variant(
        phrase=phrase,
        voice_id=voice.id,
        voice_name=voice.name,
        engine=voice.engine,
        strategy=strategy,
        seed=seed,
        synthesis_text=plan.synthesis_text,
        utterance_class=plan.utterance_class,
    )


def synthesize_variant(
    variant: Variant,
    voice,
    *,
    out_dir: Path,
    tuning: SpeakerSettings | None = None,
    boundary_method: str = boundary_module.METHOD_ASR,
    with_asr: bool = True,
    asr_provider=None,
) -> Variant:
    """Синтезирует один вариант и считает его метрики. Модель уже должна быть поднята.

    Порядок важен и повторяет боевой: синтез → обрезка до цели → подготовка куска
    → метрики. Громкость выравнивается **после** обрезки: иначе carrier задавал бы
    уровень целевой реплики.
    """
    from backend.engines.base import SAMPLE_RATE

    engine = get_engine(voice.engine)
    settings = tuning or SpeakerSettings(voice_id=voice.id)
    params = {
        **engine.defaults(),
        "cfg_strength": settings.cfg_strength,
        "nfe_step": settings.nfe_step,
        "target_rms": settings.target_rms,
        "cross_fade_duration": config.DEFAULT_CROSS_FADE_DURATION,
        "seed": variant.seed,
    }
    started = time.monotonic()
    try:
        raw, sample_rate = engine.synthesize(
            variant.synthesis_text,
            str(voice.audio_path),
            voice.ref_text,
            settings.speed,
            **params,
        )
    except Exception as exc:  # noqa: BLE001 — сбой одного варианта не рушит прогон
        variant.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Вариант не сгенерирован (%s, %s): %s", variant.phrase, variant.strategy, exc)
        return variant
    variant.generation_sec = time.monotonic() - started
    if sample_rate != SAMPLE_RATE:
        variant.error = f"движок вернул {sample_rate} Гц"
        return variant

    audio = np.asarray(raw, dtype=np.float32)
    # Обрезка нужна только там, где синтез-текст шире цели.
    plan_target = variant.phrase
    if variant.synthesis_text != plan_target:
        cropped, found = boundary_module.crop_to_target(
            audio,
            plan_target,
            method=boundary_method,
            side=_side_of(variant.synthesis_text, plan_target),
            asr=asr_provider,
        )
        variant.boundary = found.to_dict() if found else None
        if found is None:
            # Границы нет — честно помечаем и синтезируем цель отдельно, как в бою.
            variant.fallback = "нет надёжной границы"
            audio, sample_rate = engine.synthesize(
                plan_target,
                str(voice.audio_path),
                voice.ref_text,
                settings.speed,
                **params,
            )
            audio = np.asarray(audio, dtype=np.float32)
        else:
            audio = cropped

    prepared = audio_pipeline._prepare_chunk(audio, settings)
    final = audio_pipeline._finalize_track(prepared)
    variant.duration_sec = final.size / SAMPLE_RATE
    variant.silence_ratio = _silence_ratio(final)
    quality = take_quality.measure(final, variant.phrase)
    variant.rms_dbfs = quality.rms_dbfs
    variant.peak_dbfs = quality.peak_dbfs
    variant.lufs = quality.lufs

    target_dir = out_dir / "audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / (
        f"{_slug(variant.phrase)}__{variant.engine}__{variant.voice_id}"
        f"__{variant.strategy}__s{variant.seed}.wav"
    )
    audio_pipeline._write_audio(path, final, "wav")
    variant.audio_path = str(path)

    if with_asr:
        try:
            variant.transcription = transcribe_audio(path)
            rules = None
            variant.wer = word_error_rate(
                normalize(variant.phrase, pronunciation=rules, supports_accents=False),
                normalize(variant.transcription, pronunciation=rules, supports_accents=False),
            )
        except Exception as exc:  # noqa: BLE001 — распознавание может быть недоступно
            logger.warning("Расшифровка недоступна (%s): %s", variant.phrase, exc)
    verdict = su.check_chunk(
        final,
        variant.phrase,
        transcription=variant.transcription or None,
        wer=variant.wer,
    )
    variant.verdict = verdict.to_dict()
    return variant


def _side_of(synthesis_text: str, target: str) -> str:
    """Где в синтез-тексте стоит цель: для выбора правильного вхождения (§30)."""
    if not target or target not in synthesis_text:
        return su.SIDE_PREFIX
    if synthesis_text.startswith(target):
        return su.SIDE_SUFFIX
    if synthesis_text.endswith(target):
        return su.SIDE_PREFIX
    return su.SIDE_BOTH


def _silence_ratio(audio: np.ndarray) -> float:
    """Доля кадров ниже порога тишины — «почти тишина» видно числом."""
    from backend.engines.base import SAMPLE_RATE

    frame = max(1, int(SAMPLE_RATE * config.QA_SCREEN_FRAME_SEC))
    frames = qa_screening._frame_rms(np.asarray(audio, dtype=np.float32), frame)
    if frames.size == 0:
        return 0.0
    quiet = qa_screening._to_db(frames) < config.QA_SCREEN_SILENCE_DB
    return float(np.mean(quiet))


def summarize(variants: list[Variant]) -> dict:
    """Сводка по стратегиям и голосам: средние метрики и доля прошедших вердикт."""
    summary: dict = {"by_engine_strategy": {}, "by_voice": {}, "by_phrase": {}}

    def bucket(key: str, item: Variant, target: dict) -> None:
        entry = target.setdefault(
            key,
            {"count": 0, "ok": 0, "wer": [], "duration": [], "silence": [], "generation": []},
        )
        entry["count"] += 1
        verdict = item.verdict or {}
        if verdict.get("ok"):
            entry["ok"] += 1
        if item.wer is not None:
            entry["wer"].append(item.wer)
        entry["duration"].append(item.duration_sec)
        entry["silence"].append(item.silence_ratio)
        entry["generation"].append(item.generation_sec)

    for item in variants:
        if item.error:
            continue
        bucket(f"{item.engine} / {item.strategy}", item, summary["by_engine_strategy"])
        bucket(f"{item.engine} / {item.voice_name}", item, summary["by_voice"])
        bucket(f"{item.phrase} / {item.strategy}", item, summary["by_phrase"])

    def finalize(entry: dict) -> dict:
        count = max(entry["count"], 1)
        return {
            "count": entry["count"],
            "ok_share": round(entry["ok"] / count, 3),
            "wer_mean": round(sum(entry["wer"]) / len(entry["wer"]), 3) if entry["wer"] else None,
            "duration_mean": round(sum(entry["duration"]) / count, 3),
            "silence_mean": round(sum(entry["silence"]) / count, 3),
            "generation_mean": round(sum(entry["generation"]) / count, 2),
        }

    for name in ("by_engine_strategy", "by_voice", "by_phrase"):
        summary[name] = {key: finalize(value) for key, value in summary[name].items()}
    return summary


def render_markdown(report: dict) -> str:
    """Отчёт в Markdown: сводки, выводы и как слушать A/B."""
    lines: list[str] = []
    lines.append("# Benchmark коротких реплик")
    lines.append("")
    lines.append(f"- Дата: {report['created_at']}")
    lines.append(f"- Движки и голоса: {', '.join(report['voices'])}")
    lines.append(f"- Стратегии: {', '.join(report['strategies'])}")
    lines.append(f"- Сиды: {', '.join(str(seed) for seed in report['seeds'])}")
    lines.append(f"- Метод границ: {report['boundary_method']}")
    lines.append(f"- Вариантов: {report['variants_total']} (ошибок: {report['errors']})")
    lines.append("")
    lines.append("## Сводка по движку и стратегии")
    lines.append("")
    lines.append("| Движок / стратегия | N | Вердикт ok | WER | Длительность, с | Тишина | Генерация, с |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, value in report["summary"]["by_engine_strategy"].items():
        wer = "—" if value["wer_mean"] is None else f"{value['wer_mean']:.3f}"
        lines.append(
            f"| {key} | {value['count']} | {value['ok_share']:.2f} | {wer} | "
            f"{value['duration_mean']:.2f} | {value['silence_mean']:.2f} | "
            f"{value['generation_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Сводка по голосам (влияние reference, §16)")
    lines.append("")
    lines.append("| Движок / голос | N | Вердикт ok | WER | Длительность, с | Тишина |")
    lines.append("|---|---|---|---|---|---|")
    for key, value in report["summary"]["by_voice"].items():
        wer = "—" if value["wer_mean"] is None else f"{value['wer_mean']:.3f}"
        lines.append(
            f"| {key} | {value['count']} | {value['ok_share']:.2f} | {wer} | "
            f"{value['duration_mean']:.2f} | {value['silence_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Выводы (по измеренным числам)")
    lines.append("")
    for line in report["conclusions"]:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Слушательное A/B")
    lines.append("")
    lines.append("Файлы лежат рядом с отчётом: `audio/<фраза>__<движок>__<голос>__<стратегия>__s<сид>.wav`.")
    lines.append("Сравнивать нужно варианты одной фразы и одного голоса с одинаковым сидом:")
    lines.append("`direct` — базовая генерация, остальные — экспериментальные стратегии.")
    lines.append("Слушательная оценка — за человеком: инструмент даёт материал и объективные метрики, а")
    lines.append("«naturalness score» не выдумывает (§26).")
    lines.append("")
    return "\n".join(lines)


def conclude(report: dict, summary: dict) -> list[str]:
    """Автоматические выводы: что лучше по каждой метрике и что включать (§29, Phase 4).

    Выводы формулируются по числам и с оговоркой: если разница в пределах шума,
    честнее оставить DIRECT, чем включать стратегию «на всякий случай».
    """
    conclusions: list[str] = []
    by_engine: dict[str, dict[str, dict]] = {}
    for key, value in summary["by_engine_strategy"].items():
        engine, _, strategy = key.partition(" / ")
        by_engine.setdefault(engine, {})[strategy] = value
    for engine, strategies in sorted(by_engine.items()):
        base = strategies.get(su.STRATEGY_DIRECT)
        if base is None:
            continue
        best = None
        for strategy, value in strategies.items():
            if strategy == su.STRATEGY_DIRECT:
                continue
            better_ok = value["ok_share"] > base["ok_share"] + 0.02
            better_wer = (
                value["wer_mean"] is not None
                and base["wer_mean"] is not None
                and value["wer_mean"] < base["wer_mean"] - 0.02
            )
            if (better_ok or better_wer) and (
                best is None or value["ok_share"] > strategies[best]["ok_share"]
            ):
                best = strategy
        if best is None:
            conclusions.append(
                f"{engine}: ни одна стратегия не улучшила `direct` по числам — "
                "производственной остаётся `direct`"
            )
        else:
            value = strategies[best]
            conclusions.append(
                f"{engine}: лучше `direct` выглядит `{best}` "
                f"(вердикт ok {value['ok_share']:.2f} против {base['ok_share']:.2f}, "
                f"WER {value['wer_mean']} против {base['wer_mean']}) — проверить на слух и, "
                f"если подтвердится, включить `TTS_SHORT_ENGINE_STRATEGIES={engine}={best}`"
            )
    conclusions.append(
        "Слушательное A/B обязательно: числа ловят пустое/растянутое/повторённое аудио, "
        "но не «живость» интонации"
    )
    return conclusions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark коротких реплик")
    parser.add_argument("--voices", default="", help="id голосов через запятую (по умолчанию авто)")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--seeds", type=int, default=1, help="сколько сидов на вариант (§18)")
    parser.add_argument("--max-phrases", type=int, default=0, help="ограничить число коротких фраз")
    parser.add_argument("--no-long", action="store_true", help="без контрольных длинных фраз")
    parser.add_argument("--boundary", default=boundary_module.METHOD_ASR,
                        choices=list(boundary_module.METHODS))
    parser.add_argument("--carrier", default=None, help="carrier для synthetic_context")
    parser.add_argument("--side", default=None, choices=[None, *su.CONTEXT_SIDES])
    parser.add_argument("--no-asr", action="store_true", help="без расшифровки (быстрее)")
    parser.add_argument("--out", default="", help="каталог отчёта (по умолчанию output/short-bench)")
    parser.add_argument("--plan", action="store_true", help="только план, без моделей")
    parser.add_argument("--json", action="store_true", help="напечатать путь к JSON-отчёту")
    return parser.parse_args(argv)


def pick_voices(requested: str) -> list:
    """Голоса для прогона: явные или автоматически — 2 F5 и 1 XTTS, если есть.

    Документ требует минимум два голоса F5 и один XTTS (§25): дефект может быть
    связан с конкретным reference, и на одном голосе это не увидеть.
    """
    store = get_store()
    voices = store.list()
    if requested:
        wanted = [item.strip() for item in requested.split(",") if item.strip()]
        selected = [voice for voice in voices if voice.id in wanted]
        missing = [item for item in wanted if item not in {voice.id for voice in selected}]
        if missing:
            raise SystemExit(f"Голоса не найдены: {', '.join(missing)}")
        return selected
    f5 = [voice for voice in voices if voice.engine == "f5"][:2]
    xtts = [voice for voice in voices if voice.engine.startswith("xtts")][:1]
    return f5 + xtts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")
    args = parse_args(argv)
    short_phrases = SHORT_PHRASES[: args.max_phrases] if args.max_phrases else SHORT_PHRASES
    strategies = tuple(item.strip() for item in args.strategies.split(",") if item.strip())
    for strategy in strategies:
        if strategy not in su.STRATEGIES:
            raise SystemExit(f"Неизвестная стратегия: {strategy}")
    voices = pick_voices(args.voices)
    if not voices:
        raise SystemExit("Нет голосов: создайте хотя бы один в дашборде")

    plan = build_plan(
        voices=voices,
        strategies=strategies,
        seeds=max(args.seeds, 1),
        phrases=short_phrases,
        long_phrases=LONG_PHRASES,
        carrier=args.carrier,
        side=args.side,
        with_long=not args.no_long,
    )
    print(f"Голоса: {', '.join(f'{voice.name} ({voice.engine})' for voice in voices)}")
    print(f"Фраз: {len(plan.phrases)} коротких + {0 if args.no_long else len(LONG_PHRASES)} длинных")
    print(f"Стратегии: {', '.join(strategies)}; сидов: {args.seeds}")
    print(f"Вариантов к генерации: {len(plan.variants)}")
    if args.plan:
        for item in plan.variants[:40]:
            print(
                f"  {item.engine:12} {item.voice_name:14} {item.strategy:22} "
                f"«{item.phrase}» → «{item.synthesis_text}»"
            )
        if len(plan.variants) > 40:
            print(f"  … ещё {len(plan.variants) - 40}")
        return 0

    out_dir = Path(args.out) if args.out else config.OUTPUT_DIR / "short-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Поднимаю модели (это может занять минуты)…")
    engines = sorted({voice.engine for voice in voices})
    for engine_id in engines:
        get_engine(engine_id).load()
        print(f"  {engine_id}: готов ({get_engine(engine_id).id})")

    variants: list[Variant] = []
    total = len(plan.variants)
    for index, variant in enumerate(plan.variants, start=1):
        voice = next(item for item in voices if item.id == variant.voice_id)
        print(
            f"[{index}/{total}] {variant.engine} {variant.voice_name} {variant.strategy} "
            f"«{variant.phrase}»",
            flush=True,
        )
        synthesize_variant(
            variant,
            voice,
            out_dir=out_dir,
            boundary_method=args.boundary,
            with_asr=not args.no_asr,
        )
        variants.append(variant)

    summary = summarize(variants)
    errors = [item for item in variants if item.error]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voices": [f"{voice.name} ({voice.engine}, {voice.id})" for voice in voices],
        "strategies": list(strategies),
        "seeds": list(range(1, max(args.seeds, 1) + 1)),
        "boundary_method": args.boundary,
        "variants_total": len(variants),
        "errors": len(errors),
        "plan": plan.to_dict(),
        "variants": [item.to_dict() for item in variants],
        "summary": summary,
        "conclusions": [],
    }
    report["conclusions"] = conclude(report, summary)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Отчёт: {out_dir / 'report.md'}")
    print(f"JSON:  {out_dir / 'report.json'}")
    if args.json:
        print(str(out_dir / "report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
