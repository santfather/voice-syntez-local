"""CLI benchmark'а просодии: проверка корпуса, план матрицы и прогон (§30–§35).

Модели не поднимаются: движок передаётся тестом (`StubEngine`), распознавание —
заглушкой. Проверяется то, что определяет сопоставимость прогона: какие колонки
попали в матрицу, с каким сидом, и что отчёт оказался на диске.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf
from conftest import STUB_ENGINE_ID, StubEngine, sine

from backend import config, emotions
from backend import prosody_benchmark as pb
from backend.engines.base import SAMPLE_RATE
from backend.voices_store import ReferenceProfile, Voice
from tools import prosody_benchmark as cli

CORPUS = Path(__file__).resolve().parent.parent / "benchmarks" / "prosody" / "corpus.v1.jsonl"


class _FakeStore:
    def __init__(self, voices: list[Voice]) -> None:
        self._voices = voices

    def list(self) -> list[Voice]:
        return list(self._voices)


def _voice(voice_id: str = "voice-a", **overrides) -> Voice:
    voice = Voice(
        id=voice_id,
        name="Мария",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="base.wav",
        engine=STUB_ENGINE_ID,
    )
    for key, value in overrides.items():
        setattr(voice, key, value)
    return voice


def _with_question(workspace) -> Voice:
    """Голос с записанным вопросным референсом: из чего строить матрицу."""
    sf.write(workspace / "voices" / "base.wav", sine(0.4, 180.0), SAMPLE_RATE)
    sf.write(workspace / "voices" / "question.wav", sine(0.4, 200.0), SAMPLE_RATE)
    voice = _voice()
    voice.profiles = [
        ReferenceProfile(
            id="q",
            emotion=emotions.EMOTION_QUESTION,
            audio_file="question.wav",
            ref_text="Ты уверен?",
            quality_status="ok",
        )
    ]
    return voice


# --- корпус и разбор аргументов -------------------------------------------------
def test_check_validates_the_corpus_without_models(capsys):
    """`--check` не поднимает модель и печатает состав корпуса (§31)."""
    args = cli.parse_args(["--check"])
    assert args.corpus == cli.DEFAULT_CORPUS
    assert cli.run(args) == 0

    out = capsys.readouterr().out
    assert "неоднозначных" in out
    for category in pb.CATEGORIES:
        assert category in out


def test_check_fails_on_a_broken_corpus(tmp_path, capsys):
    """Битый корпус — отказ до прогона, а не «прогон с пропусками»."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "a", "category": "нет-такой", "text": "Да.", "expected_intent": "NEUTRAL"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        cli.run(cli.parse_args(["--corpus", str(path), "--check"]))
    err = capsys.readouterr().err
    assert "корпус:" in err and "неизвестная категория" in err

    # Нечитаемая строка тоже называется номером строки, а не стектрейсом.
    path.write_text("{не json}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="строка 1"):
        cli.run(cli.parse_args(["--corpus", str(path), "--check"]))


def test_plan_shows_the_matrix_without_models(workspace, capsys):
    """План виден до подъёма модели: колонки, сид и сама ячейка (§32)."""
    voice = _with_question(workspace)
    args = cli.parse_args(
        ["--plan", "--voice", voice.id, "--limit", "1", "--seed", "7", "--profiles", "NEUTRAL,QUESTION"]
    )
    assert cli.run(args, store=_FakeStore([voice])) == 0

    out = capsys.readouterr().out
    assert "NEUTRAL, QUESTION" in out
    assert "сид: 7" in out
    assert "neutral-001" in out and "question.wav" in out
    assert "Я пришёл немного раньше." not in out  # тексты в плане не печатаются


def test_plan_requires_a_neutral_reference(workspace):
    """Без нейтрального референса сравнивать не с чем — отказ с объяснением."""
    voice = _voice(audio_file="missing.wav")
    args = cli.parse_args(["--plan", "--voice", voice.id])
    with pytest.raises(SystemExit, match="нейтрального референса"):
        cli.run(args, store=_FakeStore([voice]))


def test_plan_refuses_unknown_profiles_and_voices():
    """Опечатка в профиле или голосе называется, а не игнорируется."""
    with pytest.raises(SystemExit, match="Неизвестные профили"):
        cli.parse_profiles("NEUTRAL,ВОСТОРГ")

    with pytest.raises(SystemExit, match="Список профилей пуст"):
        cli.parse_profiles("  ,  ")

    with pytest.raises(SystemExit, match="Укажите --voice"):
        cli.pick_voice("", store=_FakeStore([_voice("a"), _voice("b")]))

    with pytest.raises(SystemExit, match="не найден"):
        cli.pick_voice("нет-такого", store=_FakeStore([_voice("a")]))

    with pytest.raises(SystemExit, match="Нет голосов"):
        cli.pick_voice("", store=_FakeStore([]))


# --- прогон ---------------------------------------------------------------------
def test_run_writes_reports_and_audio(workspace, capsys):
    """Прогон CLI пишет отчёт, форму прослушивания и WAV для каждой ячейки."""
    voice = _with_question(workspace)
    engine = StubEngine()
    args = cli.parse_args(
        [
            "--voice", voice.id,
            "--limit", "1",
            "--profiles", "NEUTRAL,QUESTION",
            "--seed", "3",
        ]
    )
    # Распознавание подменяется: Whisper в тестах не поднимается (и в бою он
    # подставляется в `run` по умолчанию — параметр передаётся явно).
    assert cli.run(args, store=_FakeStore([voice]), engine=engine, transcribe=lambda path: "") == 0

    out_dir = config.BENCHMARKS_DIR / "prosody"
    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    report = json.loads((run_dirs[0] / "report.json").read_text(encoding="utf-8"))
    assert report["cells_total"] == 2
    assert report["seed"] == 3
    assert report["profiles"] == [emotions.EMOTION_NEUTRAL, emotions.EMOTION_QUESTION]
    # Один вызов модели на ячейку, сид задан прогоном — сравнение контролируемое.
    assert len(engine.calls) == 2
    assert {call["params"]["seed"] for call in engine.calls} == {3}
    assert (run_dirs[0] / "report.md").exists() and (run_dirs[0] / "review.md").exists()
    assert len(list((run_dirs[0] / "audio").iterdir())) == 2
    assert "Готово" in capsys.readouterr().out


def test_run_without_asr_skips_recognition(workspace):
    """`--no-asr` не зовёт распознавание, но ячейки и файлы на месте."""
    voice = _with_question(workspace)
    calls: list[Path] = []
    args = cli.parse_args(["--voice", voice.id, "--limit", "1", "--no-asr"])

    def transcribe(path: Path) -> str:
        calls.append(path)
        return "Я пришёл немного раньше."

    assert cli.run(args, store=_FakeStore([voice]), engine=StubEngine(), transcribe=transcribe) == 0
    assert calls == []
