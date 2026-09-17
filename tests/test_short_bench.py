"""Benchmark коротких реплик: план, сводка, выводы и отчёт (§1, §25, §26).

Модели не поднимаются: проверяются планирование, метрики и формулировка выводов —
то, что определяет, будет ли прогон сопоставимым, а решение — основанным на числах.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import config
from backend import short_utterance as su
from backend.voices_store import Voice
from tools import short_bench as bench


class _FakeStore:
    def __init__(self, voices: list[Voice]) -> None:
        self._voices = voices

    def list(self) -> list[Voice]:
        return list(self._voices)


def _voice(voice_id: str, name: str, engine: str) -> Voice:
    return Voice(
        id=voice_id, name=name, gender="female", ref_text="референс",
        audio_file="ref.wav", engine=engine,
    )


@pytest.fixture
def fake_voices(monkeypatch):
    voices = [
        _voice("f5-a", "Ф5 первая", "f5"),
        _voice("f5-b", "Ф5 вторая", "f5"),
        _voice("xtts-a", "Икс", "xtts"),
    ]
    monkeypatch.setattr(bench, "get_store", lambda: _FakeStore(voices))
    return voices


def test_plan_covers_phrases_voices_strategies_and_seeds(fake_voices):
    plan = bench.build_plan(
        voices=fake_voices[:2],
        strategies=(su.STRATEGY_DIRECT, su.STRATEGY_SYNTHETIC_CONTEXT),
        seeds=2,
        phrases=("Привет!", "Да."),
        long_phrases=(),
        with_long=False,
    )
    assert len(plan.variants) == 2 * 2 * 2 * 2  # фразы × голоса × стратегии × сиды
    keys = {(item.phrase, item.voice_id, item.strategy, item.seed) for item in plan.variants}
    assert len(keys) == len(plan.variants), "варианты не должны повторяться"
    direct = next(item for item in plan.variants if item.strategy == su.STRATEGY_DIRECT)
    assert direct.synthesis_text == direct.phrase
    context = next(
        item for item in plan.variants if item.strategy == su.STRATEGY_SYNTHETIC_CONTEXT
    )
    assert context.synthesis_text == "Хорошо. Привет! Хорошо."
    assert context.utterance_class == su.CLASS_VERY_SHORT


def test_plan_puts_long_phrases_only_through_direct(fake_voices):
    """Длинные контрольные фразы проверяют регрессию, а не стратегии (§28)."""
    plan = bench.build_plan(
        voices=fake_voices[:1],
        strategies=(su.STRATEGY_DIRECT, su.STRATEGY_SYNTHETIC_CONTEXT),
        seeds=1,
        phrases=("Да.",),
        long_phrases=bench.LONG_PHRASES[:1],
        with_long=True,
    )
    long_items = [item for item in plan.variants if item.phrase == bench.LONG_PHRASES[0]]
    assert long_items
    assert {item.strategy for item in long_items} == {su.STRATEGY_DIRECT}


def test_plan_for_f5_keeps_accents_and_xtts_does_not(fake_voices):
    """В плане учитывается движок голоса: F5 получает «+», XTTS — нет (§15)."""
    plan = bench.build_plan(
        voices=fake_voices,
        strategies=(su.STRATEGY_SAME_SPEAKER_CONTEXT,),
        seeds=1,
        phrases=("Да.",),
        long_phrases=(),
        with_long=False,
    )
    by_engine = {item.engine: item.synthesis_text for item in plan.variants}
    assert by_engine["f5"] == "Я только что вернулась домой. Да."
    assert by_engine["xtts"] == "Я только что вернулась домой. Да."


def _variant(
    *,
    engine: str = "f5",
    strategy: str = su.STRATEGY_DIRECT,
    ok: bool = True,
    wer: float | None = 0.0,
    duration: float = 0.5,
    silence: float = 0.1,
    generation: float = 1.0,
    phrase: str = "Да.",
    voice: str = "Ф5",
    error: str = "",
) -> bench.Variant:
    item = bench.Variant(
        phrase=phrase,
        voice_id="f5-a",
        voice_name=voice,
        engine=engine,
        strategy=strategy,
        seed=1,
        synthesis_text=phrase,
        utterance_class=su.CLASS_VERY_SHORT,
    )
    item.verdict = {"ok": ok, "reasons": [] if ok else ["repetition"]}
    item.wer = wer
    item.duration_sec = duration
    item.silence_ratio = silence
    item.generation_sec = generation
    item.error = error
    return item


def test_summary_counts_ok_share_wer_and_means():
    variants = [
        _variant(strategy=su.STRATEGY_DIRECT, ok=True, wer=0.0),
        _variant(strategy=su.STRATEGY_DIRECT, ok=False, wer=0.5, duration=1.5),
        _variant(strategy=su.STRATEGY_SYNTHETIC_CONTEXT, ok=True, wer=0.0, generation=3.0),
        _variant(strategy=su.STRATEGY_DIRECT, error="сбой", duration=0.0),
    ]
    summary = bench.summarize(variants)
    direct = summary["by_engine_strategy"]["f5 / direct"]
    assert direct["count"] == 2, "ошибочный вариант в сводку не попадает"
    assert direct["ok_share"] == pytest.approx(0.5)
    assert direct["wer_mean"] == pytest.approx(0.25)
    assert direct["duration_mean"] == pytest.approx(1.0)
    synthetic = summary["by_engine_strategy"]["f5 / synthetic_context"]
    assert synthetic["ok_share"] == 1.0
    assert summary["by_voice"], "сводка по голосам нужна для §16"


def test_conclusions_recommend_only_on_measured_improvement():
    """Вывод формулируется по числам: улучшение — рекомендация, шум — нет."""
    summary = {
        "by_engine_strategy": {
            "f5 / direct": {"count": 10, "ok_share": 0.5, "wer_mean": 0.4,
                            "duration_mean": 0.5, "silence_mean": 0.1, "generation_mean": 1.0},
            "f5 / synthetic_context": {"count": 10, "ok_share": 0.9, "wer_mean": 0.1,
                                       "duration_mean": 0.5, "silence_mean": 0.1,
                                       "generation_mean": 3.0},
        },
        "by_voice": {},
        "by_phrase": {},
    }
    conclusions = bench.conclude({}, summary)
    line = next(item for item in conclusions if item.startswith("f5:"))
    assert "synthetic_context" in line
    assert "TTS_SHORT_ENGINE_STRATEGIES=f5=synthetic_context" in line


def test_conclusions_keep_direct_when_nothing_improves():
    summary = {
        "by_engine_strategy": {
            "xtts / direct": {"count": 10, "ok_share": 0.9, "wer_mean": 0.1,
                              "duration_mean": 0.5, "silence_mean": 0.1, "generation_mean": 5.0},
            "xtts / punctuation": {"count": 10, "ok_share": 0.9, "wer_mean": 0.12,
                                   "duration_mean": 0.5, "silence_mean": 0.1,
                                   "generation_mean": 5.0},
        },
        "by_voice": {},
        "by_phrase": {},
    }
    conclusions = bench.conclude({}, summary)
    line = next(item for item in conclusions if item.startswith("xtts:"))
    assert "остаётся `direct`" in line


def test_markdown_report_contains_tables_and_listening_section():
    report = {
        "created_at": "2026-09-17T00:00:00+00:00",
        "voices": ["Ф5 (f5, f5-a)"],
        "strategies": [su.STRATEGY_DIRECT],
        "seeds": [1],
        "boundary_method": "asr",
        "variants_total": 1,
        "errors": 0,
        "summary": {
            "by_engine_strategy": {
                "f5 / direct": {"count": 1, "ok_share": 1.0, "wer_mean": 0.0,
                                "duration_mean": 0.5, "silence_mean": 0.1,
                                "generation_mean": 1.0},
            },
            "by_voice": {},
            "by_phrase": {},
        },
        "conclusions": ["f5: всё хорошо"],
        "variants": [],
    }
    markdown = bench.render_markdown(report)
    assert "# Benchmark коротких реплик" in markdown
    assert "| f5 / direct |" in markdown
    assert "Слушательное A/B" in markdown
    assert "naturalness score" in markdown, "оговорка о субъективной оценке обязательна"


def test_side_detection_matches_context_position():
    assert bench._side_of("Хорошо. Привет! Хорошо.", "Привет!") == su.SIDE_BOTH
    assert bench._side_of("Я вернулась. Да.", "Да.") == su.SIDE_PREFIX
    assert bench._side_of("Да. Я вернулась.", "Да.") == su.SIDE_SUFFIX
    assert bench._side_of("Совсем другой текст", "Да.") == su.SIDE_PREFIX


def test_report_json_is_serializable(tmp_path):
    report = {"variants": [_variant().to_dict()], "nested": {"boundary": None}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["variants"][0]["strategy"] == su.STRATEGY_DIRECT
    assert loaded["variants"][0]["verdict"]["ok"] is True


def test_plan_mode_prints_without_models(fake_voices, capsys, monkeypatch):
    """`--plan` не поднимает модели и показывает, что будет сгенерировано."""
    called: list[str] = []
    monkeypatch.setattr(bench, "get_engine", lambda engine_id: called.append(engine_id))
    code = bench.main(["--plan", "--max-phrases", "2", "--seeds", "1"])
    output = capsys.readouterr().out
    assert code == 0
    assert called == [], "план не должен поднимать движки"
    assert "Вариантов к генерации" in output
    assert "Хорошо. Привет! Хорошо." in output


def test_unknown_strategy_is_rejected(fake_voices):
    with pytest.raises(SystemExit, match="Неизвестная стратегия"):
        bench.main(["--plan", "--strategies", "магия"])


def test_voice_selection_prefers_two_f5_and_one_xtts(fake_voices):
    voices = bench.pick_voices("")
    assert [voice.id for voice in voices] == ["f5-a", "f5-b", "xtts-a"]
    assert bench.pick_voices("f5-a")[0].id == "f5-a"
    with pytest.raises(SystemExit, match="Голоса не найдены"):
        bench.pick_voices("нет-такого")


def test_pick_voices_raises_without_any_voice(monkeypatch):
    monkeypatch.setattr(bench, "get_store", lambda: _FakeStore([]))
    assert bench.pick_voices("") == []


def test_default_output_dir_respects_config(monkeypatch, tmp_path):
    """Отчёт по умолчанию ложится в output/short-bench (а не в зашитый путь)."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    args = bench.parse_args([])
    assert args.out == ""
    assert (Path(args.out) if args.out else config.OUTPUT_DIR / "short-bench") == (
        tmp_path / "short-bench"
    )
