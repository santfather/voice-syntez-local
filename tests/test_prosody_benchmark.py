"""Reference prosody benchmark: корпус, матрица, метрики и отчёт (UPDATE 3 §30–§35).

Benchmark отвечает на один вопрос — «меняет ли смена reference-профиля просодию
target, не ломая текст» — и потому тесты идут по границам, где легко получить
правдоподобный, но неверный ответ:

* **корпус** — реплики на месте, категории не пусты, словарь намерений не разошёлся
  с боевым (§31);
* **матрица** — сравниваются только доступные профили, и каждая ячейка отличается от
  соседней **только** профилем (§32);
* **метрики** — WER, слова и повторы считаются, а не выглядят посчитанными (§33);
* **отчёт** — числа попадают в файлы, а перенос интонации остаётся человеку (§34).

Модели не поднимаются: синтез идёт через `StubEngine`, распознавание — заглушкой.
"""

import asyncio
import json
from pathlib import Path

import pytest
import soundfile as sf
from conftest import STUB_ENGINE_ID, StubEngine, sine

from backend import config, emotions
from backend import prosody_benchmark as pb
from backend.audio_pipeline import JobCancelledError
from backend.engines.base import SAMPLE_RATE
from backend.voices_store import ReferenceProfile, Voice

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "benchmarks" / "prosody"
CORPUS = CORPUS_DIR / "corpus.v1.jsonl"

FILES = ("base.wav", "question.wav", "delight.wav")


def _files(workspace) -> None:
    """Файлы референсов: без них профиль существует, но не пригоден (§22)."""
    for name in FILES:
        sf.write(workspace / "voices" / name, sine(0.4, 180.0), SAMPLE_RATE)


def _profile(profile_id: str, emotion: str, audio: str, **overrides) -> ReferenceProfile:
    fields = {
        "id": profile_id,
        "emotion": emotion,
        "audio_file": audio,
        "ref_text": "Ты уверен?",
        "quality_status": "ok",
    }
    fields.update(overrides)
    return ReferenceProfile(**fields)


def _voice(profiles=None, **overrides) -> Voice:
    voice = Voice(
        id="voice-a",
        name="Мария",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="base.wav",
        engine=STUB_ENGINE_ID,
    )
    voice.profiles = list(profiles or [])
    for key, value in overrides.items():
        setattr(voice, key, value)
    return voice


def _case(case_id: str = "c1", text: str = "Ты уверен?") -> pb.CorpusCase:
    return pb.CorpusCase(
        id=case_id, category=pb.CATEGORY_QUESTION, text=text, expected_intent="QUESTION"
    )


# --- §31. Корпус ---------------------------------------------------------------
def test_corpus_v1_is_complete_and_has_no_problems():
    """Корпус проходит собственную проверку: категории не пусты, реплик хватает."""
    cases = pb.load_corpus(CORPUS)
    assert pb.corpus_problems(cases) == []
    assert len(cases) >= pb.MIN_CORPUS_CASES
    counts = pb.category_counts(cases)
    assert set(counts) == set(pb.CATEGORIES)
    assert all(count > 0 for count in counts.values())
    assert any(case.ambiguous for case in cases)


def test_corpus_files_and_schema_match_code():
    """Схема корпуса не расходится с кодом: словари берутся из одного источника."""
    for name in ("corpus.v1.jsonl", "schema.json", "README.md"):
        assert (CORPUS_DIR / name).exists(), name

    schema = json.loads((CORPUS_DIR / "schema.json").read_text(encoding="utf-8"))
    assert schema["required"] == ["id", "category", "text", "expected_intent"]
    assert tuple(schema["properties"]["category"]["enum"]) == pb.CATEGORIES
    assert tuple(schema["properties"]["expected_intent"]["enum"]) == emotions.EMOTIONS


def test_load_corpus_names_the_broken_line(tmp_path):
    """Ошибка в данных называется строкой: корпус правят руками, и опечатка — норма."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "a", "category": "neutral", "text": "Да.", "expected_intent": "NEUTRAL"}\n'
        "не json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="строка 2"):
        pb.load_corpus(path)

    path.write_text('{"id": "a", "category": "neutral"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="нет полей"):
        pb.load_corpus(path)

    with pytest.raises(ValueError, match="не найден"):
        pb.load_corpus(tmp_path / "нет-такого.jsonl")


def test_corpus_problems_report_every_defect_at_once():
    """Проверка возвращает список проблем, а не первую: `--check` печатает их все."""
    cases = [
        _case("dup"),
        _case("dup"),
        pb.CorpusCase(id="bad-cat", category="нет-такой", text="Текст.", expected_intent="QUESTION"),
        pb.CorpusCase(id="empty", category=pb.CATEGORY_NEUTRAL, text="", expected_intent="NEUTRAL"),
        pb.CorpusCase(
            id="bad-intent",
            category=pb.CATEGORY_NEUTRAL,
            text="Текст.",
            expected_intent="ВОСТОРГ",
        ),
    ]
    problems = pb.corpus_problems(cases)
    joined = "\n".join(problems)
    assert "повтор id: dup" in joined
    assert "неизвестная категория" in joined
    assert "пустой текст" in joined
    assert "вне словаря эмоций" in joined
    # Мало реплик и пустые категории — тоже проблемы, а не «почти готово».
    assert str(pb.MIN_CORPUS_CASES) in joined
    assert f"категория «{pb.CATEGORY_IRONY}» пуста" in joined


# --- §33. Метрики --------------------------------------------------------------
def test_metrics_of_clean_transcript():
    metrics = pb.measure_metrics("Ты уверен?", "ты уверен", duration_sec=1.5, generation_sec=2.0)
    assert metrics.wer == 0
    assert metrics.first_word_ok and metrics.last_word_ok
    assert metrics.extra_words == 0
    assert metrics.repetition == ""
    assert metrics.text_qa == pb.TEXT_QA_PASSED
    assert metrics.to_dict()["duration_sec"] == 1.5


def test_metrics_mark_silence_as_failure_and_losses_as_review():
    """WER усредняет, поэтому дефекты называются отдельными статусами (§33)."""
    empty = pb.measure_metrics("Ты уверен?", "")
    assert empty.text_qa == pb.TEXT_QA_FAILED and empty.wer == 1.0

    # На длинной реплике потерянное слово не вытягивает WER выше порога — и статус
    # обязан сказать «текст цел, но послушай», иначе дефект скрылся бы за средним.
    text = "Я пришёл немного раньше и подождал тебя в коридоре у окна"
    lost_first = pb.measure_metrics(
        text, "пришёл немного раньше и подождал тебя в коридоре у окна"
    )
    assert lost_first.wer <= config.QA_WER_THRESHOLD
    assert lost_first.text_qa == pb.TEXT_QA_REVIEW and not lost_first.first_word_ok

    lost_last = pb.measure_metrics(text, "я пришёл немного раньше и подождал тебя в коридоре у")
    assert lost_last.wer <= config.QA_WER_THRESHOLD
    assert lost_last.text_qa == pb.TEXT_QA_REVIEW and not lost_last.last_word_ok


def test_text_qa_flags_extra_words_and_repetition_as_review():
    """Лишний хвост и повтор — повод послушать, а не молчаливый автобрак (§33)."""
    base = {"transcript": "текст", "wer": 0.02, "first_word_ok": True, "last_word_ok": True}
    assert (
        pb.text_qa_status(**base, extra_words=pb.MAX_EXTRA_WORDS, repetition="")
        == pb.TEXT_QA_PASSED
    )
    assert (
        pb.text_qa_status(**base, extra_words=pb.MAX_EXTRA_WORDS + 1, repetition="")
        == pb.TEXT_QA_REVIEW
    )
    assert (
        pb.text_qa_status(**base, extra_words=0, repetition="слово «да» подряд 3 раза")
        == pb.TEXT_QA_REVIEW
    )
    assert pb.text_qa_status(**base, extra_words=0, repetition="") == pb.TEXT_QA_PASSED


def test_metrics_fail_on_wer_above_threshold():
    text = "Мы встретимся завтра утром"
    transcript = "мы встретимся завтра вечером после работы"
    metrics = pb.measure_metrics(text, transcript)
    assert metrics.wer > config.QA_WER_THRESHOLD
    assert metrics.text_qa == pb.TEXT_QA_FAILED


def test_detect_repetition_finds_word_runs_and_repeated_ngrams():
    assert "«да»" in pb.detect_repetition(["да", "да", "да", "нет"])
    assert pb.detect_repetition(["это", "правда", "это", "правда"])
    assert pb.detect_repetition(["мы", "встретимся", "завтра", "утром"]) == ""
    # Два раза подряд — ещё речь, а не дефект: порог задан явно.
    assert pb.detect_repetition(["да", "да"]) == ""


# --- §32. Матрица --------------------------------------------------------------
def test_available_resolutions_keeps_only_usable_profiles(workspace):
    """Колонка появляется, только если профиль реально доступен (§32)."""
    _files(workspace)
    voice = _voice(
        [
            _profile("q", emotions.EMOTION_QUESTION, "question.wav"),
            _profile("d", emotions.EMOTION_DELIGHT, "delight.wav", enabled=False),
            _profile("i", emotions.EMOTION_IRONIC, "delight.wav", quality_status="warning"),
        ]
    )
    resolutions = pb.available_resolutions(voice, STUB_ENGINE_ID)
    assert set(resolutions) == {emotions.EMOTION_NEUTRAL, emotions.EMOTION_QUESTION}
    assert resolutions[emotions.EMOTION_NEUTRAL].profile_id == "voice-a-neutral"
    assert resolutions[emotions.EMOTION_QUESTION].audio_path.name == "question.wav"


def test_plan_matrix_puts_all_profiles_of_one_case_together(workspace):
    """Порядок ячеек — «все профили одной реплики подряд»: так их и слушают (§34)."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])
    resolutions = pb.available_resolutions(voice, STUB_ENGINE_ID)
    assert pb.matrix_profiles(resolutions) == (
        emotions.EMOTION_NEUTRAL,
        emotions.EMOTION_QUESTION,
    )
    cells = pb.plan_matrix([_case("c1"), _case("c2")], resolutions)
    assert [(cell.case.id, cell.profile_emotion) for cell in cells] == [
        ("c1", emotions.EMOTION_NEUTRAL),
        ("c1", emotions.EMOTION_QUESTION),
        ("c2", emotions.EMOTION_NEUTRAL),
        ("c2", emotions.EMOTION_QUESTION),
    ]


# --- Выполнение матрицы --------------------------------------------------------
def _run(coroutine) -> None:
    asyncio.run(coroutine)


def _matrix(workspace, voice, cases, *, engine=None, profiles=None, **kwargs):
    """Прогон маленькой матрицы: корпус и профили задаются тестом."""
    cases = list(cases)
    resolutions = pb.available_resolutions(voice, STUB_ENGINE_ID, profiles or pb.MATRIX_PROFILES)
    cells = pb.plan_matrix(cases, resolutions)
    run = pb.make_run(
        voice,
        STUB_ENGINE_ID,
        cases_total=len(cases),
        profiles=pb.matrix_profiles(resolutions),
        seed=kwargs.pop("seed", 42),
    )
    out_dir = workspace / "prosody" / run.id
    asyncio.run(
        pb.run_matrix(
            run,
            voice,
            engine or StubEngine(),
            cells,
            out_dir=out_dir,
            auto_accent=False,
            **kwargs,
        )
    )
    return run, out_dir


def test_run_matrix_uses_one_seed_and_the_cell_reference(workspace):
    """Ячейки отличаются только профилем: сид и ручки у них общие (§32)."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])
    engine = StubEngine()

    def transcribe(path: Path) -> str:
        return "Ты уверен?"

    run, out_dir = _matrix(workspace, voice, [_case()], engine=engine, transcribe=transcribe)

    assert len(run.cells) == 2
    assert all(cell.status == pb.CELL_DONE for cell in run.cells)
    # Один вызов движка на ячейку: QA-цикл выключен и сиды не перебираются.
    assert len(engine.calls) == 2
    assert {call["params"]["seed"] for call in engine.calls} == {42}
    assert {call["text"] for call in engine.calls} == {"Ты уверен?"}
    assert [Path(call["ref_audio_path"]).name for call in engine.calls] == [
        "base.wav",
        "question.wav",
    ]
    # Профиль ячейки записан и в отчёте, и в имени файла — сверить можно без кода.
    assert [cell.reference_profile_id for cell in run.cells] == ["voice-a-neutral", "q"]
    assert all((out_dir / "audio" / cell.audio_file).exists() for cell in run.cells)
    assert run.cells[0].audio_file != run.cells[1].audio_file
    assert all(cell.metrics.text_qa == pb.TEXT_QA_PASSED for cell in run.cells)
    assert all(cell.metrics.wer == 0 for cell in run.cells)


def test_run_matrix_survives_a_failed_cell(workspace):
    """Сбой одной ячейки не уносит прогон: остальные замеры остаются (§33)."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])

    class _FailingRef(StubEngine):
        def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
            if ref_audio_path.endswith("question.wav"):
                raise RuntimeError("не хватило памяти")
            return super()._synthesize(text, ref_audio_path, ref_text, speed, params)

    run, _out_dir = _matrix(
        workspace, voice, [_case()], engine=_FailingRef(), transcribe=lambda path: "Ты уверен?"
    )
    by_profile = {cell.profile_emotion: cell for cell in run.cells}
    assert by_profile[emotions.EMOTION_NEUTRAL].status == pb.CELL_DONE
    assert by_profile[emotions.EMOTION_QUESTION].status == pb.CELL_ERROR
    assert "не хватило памяти" in by_profile[emotions.EMOTION_QUESTION].error
    assert run.finished_at is not None


def test_run_matrix_stops_on_abort(workspace):
    """Отмена проверяется до синтеза: прогон не продолжается «через силу»."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])
    engine = StubEngine()

    with pytest.raises(JobCancelledError):
        _matrix(
            workspace,
            voice,
            [_case()],
            engine=engine,
            transcribe=None,
            should_abort=lambda: True,
        )
    assert engine.calls == []


def test_run_matrix_without_asr_keeps_audio_and_marks_text_failed(workspace):
    """Без ASR аудио всё равно сохраняется: замер времени терять нельзя (§33)."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])
    run, out_dir = _matrix(workspace, voice, [_case()], transcribe=None)
    assert all(cell.status == pb.CELL_DONE for cell in run.cells)
    assert all((out_dir / "audio" / cell.audio_file).exists() for cell in run.cells)
    assert all(cell.metrics.text_qa == pb.TEXT_QA_FAILED for cell in run.cells)


# --- §33, §34. Отчёт ------------------------------------------------------------
def test_report_summarizes_profiles_against_neutral(workspace):
    """Сводка сравнивает профиль с нейтральным: без базы «лучше» не читается."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])
    run, out_dir = _matrix(workspace, voice, [_case()], transcribe=lambda path: "Ты уверен?")
    report = pb.build_report(run)

    assert report["benchmark"] == "reference_prosody"
    assert report["version"] == pb.BENCHMARK_VERSION
    assert report["cells_total"] == 2
    neutral = report["summary"]["by_profile"][emotions.EMOTION_NEUTRAL]
    question = report["summary"]["by_profile"][emotions.EMOTION_QUESTION]
    assert neutral["wer_delta_vs_neutral"] == 0
    assert question["text_ok_delta_vs_neutral"] == 0
    assert report["summary"]["by_category"][pb.CATEGORY_QUESTION]["cells"] == 2

    markdown = pb.format_report(report)
    assert "Сводка по профилям" in markdown
    assert "`c1`" in markdown and "QUESTION" in markdown

    review = pb.render_review(report)
    # Форма §34: колонки есть, оценки ставит человек — они пусты.
    for column in (
        "target_text",
        "expected_intent",
        "reference_profile",
        "text_complete",
        "clarity",
        "voice_identity",
        "prosody_match",
        "overacting",
        "artifacts",
        "notes",
    ):
        assert column in review
    assert review.count("|  |  |  |  |  |  |  |") == 2

    pb.write_report(out_dir, report)
    assert json.loads((out_dir / "report.json").read_text(encoding="utf-8"))["cells_total"] == 2
    assert (out_dir / "report.md").read_text(encoding="utf-8").startswith("# Reference Prosody")
    assert (out_dir / "review.md").read_text(encoding="utf-8").startswith("# Listening review")


def test_report_lists_generation_errors(workspace):
    """Ошибки попадают в отчёт отдельным списком, а не исчезают из вида."""
    _files(workspace)
    voice = _voice([_profile("q", emotions.EMOTION_QUESTION, "question.wav")])

    class _AllFail(StubEngine):
        def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
            raise RuntimeError("модель недоступна")

    run, _out_dir = _matrix(workspace, voice, [_case()], engine=_AllFail(), transcribe=None)
    report = pb.build_report(run)
    assert len(report["errors"]) == 2
    assert "Ошибки генерации" in pb.format_report(report)
    # Форма прослушивания пуста: слушать нечего, и это честно видно.
    assert "Ты уверен?" not in pb.render_review(report).split("## Как читать колонки")[0]
