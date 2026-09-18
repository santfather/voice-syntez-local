"""Надёжность коротких реплик: атомарность, края, QA по словам (UPDATE 2 §16–§24).

Тесты отвечают на конкретные пункты пакета и намеренно не поднимают моделей:
движок-заглушка пишет в журнал ровно то, что ушло в синтез, поэтому здесь
проверяются контракты, а не качество звука.

Главное, что фиксируется:

* короткая реплика, помещающаяся в лимит, уходит **одним** вызовом движка (§16);
* проверка качества измеряет **подготовленный** кусок, а не сырой выход модели —
  иначе дефект, внесённый обрезкой краёв, не видит никто (§18, §21);
* у короткой реплики запас обрезки шире, и тихое окончание не срезается (§19);
* вердикт называет потерянное первое и последнее слово отдельно (§21);
* разные спикеры никогда не попадают в один вызов синтеза (§27).
"""

import asyncio
import json

import numpy as np
import pytest
from conftest import sine

from backend import audio_pipeline, config, synthesis_trace
from backend import short_utterance as su
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica, parse_dialogue
from backend.engines.base import SAMPLE_RATE

SPEAKER = "ИВАН"


def _render(texts: list[str], *, short: bool = False, job_id: str = "job-short"):
    replicas = [
        Replica(voice=SPEAKER, text=text, line_number=index + 1)
        for index, text in enumerate(texts)
    ]
    settings = RenderSettings(
        output_format="wav",
        short_utterance=(
            audio_pipeline.ShortUtteranceSettings(enabled=True, strategy="direct")
            if short
            else None
        ),
    )
    return asyncio.run(
        audio_pipeline.render_dialogue(
            job_id,
            replicas,
            {SPEAKER: SpeakerSettings.from_dict({"voice_id": HOLDER["voice"].id})},
            settings,
        )
    )


HOLDER: dict = {}


@pytest.fixture(autouse=True)
def remember(stub, fake_store, voice):
    HOLDER["voice"] = voice
    yield
    HOLDER.clear()


# --- §16. Атомарность короткой реплики ----------------------------------------
def test_two_sentences_in_short_replica_stay_one_synthesis_call(stub):
    """«Дайте пройти. Пожалуйста.» — одна реплика и один вызов движка."""
    _render(["Дайте пройти. Пожалуйста."])
    assert len(stub.calls) == 1
    assert stub.calls[0]["text"].count("Дайте") == 1
    assert "Пожалуйста" in stub.calls[0]["text"]


def test_parser_does_not_split_replica_that_fits_limit():
    """Разбор диалога не режет реплику, помещающуюся в лимит, даже по предложениям."""
    parsed = parse_dialogue(
        "ИВАН: Дайте пройти. Пожалуйста.\nМАРГО: Всякое бывает. Оставь номер.",
        config.chunk_chars(config.CHUNK_STRATEGY_DEFAULT),
    )
    assert [replica.text for replica in parsed.replicas] == [
        "Дайте пройти. Пожалуйста.",
        "Всякое бывает. Оставь номер.",
    ]


def test_short_dialogue_calls_engine_once_per_replica(stub):
    """Число вызовов движка равно числу реплик: внутренних кусков нет."""
    result = _render(["Красивая.", "Ты опоздал.", "Дайте пройти. Пожалуйста."])
    assert len(stub.calls) == 3
    assert result.replicas_done == 3


def test_different_speakers_are_never_batched(stub):
    """Соседние реплики разных спикеров не смешиваются в один синтез (§27)."""
    replicas = [
        Replica(voice="ИВАН", text="Красивая.", line_number=1),
        Replica(voice="МАРГО", text="Дайте пройти.", line_number=2),
    ]
    voice_id = HOLDER["voice"].id
    settings = RenderSettings(
        output_format="wav",
        short_utterance=audio_pipeline.ShortUtteranceSettings(
            enabled=True, strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT
        ),
    )
    asyncio.run(
        audio_pipeline.render_dialogue(
            "job-speakers",
            replicas,
            {
                "ИВАН": SpeakerSettings.from_dict({"voice_id": voice_id}),
                "МАРГО": SpeakerSettings.from_dict({"voice_id": voice_id}),
            },
            settings,
        )
    )
    assert len(stub.calls) == 2
    for call in stub.calls:
        assert "Красивая" not in call["text"] or "пройти" not in call["text"]


# --- §19–§20. Края: запас и кроссфейд -----------------------------------------
def test_short_trim_fits_inside_wider_guard():
    """У короткой реплики запас обрезки шире, чем у длинной реплики."""
    assert audio_pipeline._short_guard_ms("Да.") == config.SHORT_EDGE_GUARD_MS
    assert audio_pipeline._short_guard_ms("Да.") > audio_pipeline._short_guard_ms(
        "Сегодня хорошая погода, и я решил прогуляться по набережной до заката."
    )
    long_text = "Сегодня хорошая погода, и я решил прогуляться по набережной до заката."
    assert audio_pipeline._short_guard_ms(long_text) == config.EDGE_SILENCE_MARGIN_MS


def test_edge_trim_does_not_cut_short_attack_and_ending():
    """Тихая атака и тихое окончание односложной фразы остаются в куске.

    Атака и окончание здесь тише тела (−26 dBFS против −8), но выше порога тишины:
    именно такой край и терялся бы, если бы запас обрезки был общим, а не
    расширенным для короткой реплики.
    """
    quiet = 0.05
    attack = (sine(0.06, amplitude=quiet) * np.linspace(0.3, 1.0, int(SAMPLE_RATE * 0.06))).astype(
        np.float32
    )
    body = sine(0.3, amplitude=0.4)
    ending = (sine(0.06, amplitude=quiet) * np.linspace(1.0, 0.3, int(SAMPLE_RATE * 0.06))).astype(
        np.float32
    )
    pad = int(SAMPLE_RATE * 0.3)
    chunk = np.concatenate((np.zeros(pad, dtype=np.float32), attack, body, ending,
                            np.zeros(pad, dtype=np.float32)))
    settings = SpeakerSettings.from_dict({"voice_id": "voice1"})
    prepared = audio_pipeline._prepare_chunk(chunk, settings, text="Стой!")
    guard = int(SAMPLE_RATE * config.SHORT_EDGE_GUARD_MS / 1000)
    # Вся фраза целиком: атака + тело + окончание остались в куске.
    assert prepared.size >= int(SAMPLE_RATE * (0.06 + 0.3 + 0.06))
    # Атака звучит: её уровень сравним с телом (с точностью до самой тихой фразы).
    body_level = float(np.max(np.abs(prepared[guard + int(SAMPLE_RATE * 0.1): guard + int(SAMPLE_RATE * 0.2)])))
    attack_level = float(np.max(np.abs(prepared[guard: guard + int(SAMPLE_RATE * 0.06)])))
    assert attack_level > 0.1 * body_level
    # Окончание тоже: последние 60 мс перед запасом — это тихий, но звучащий хвост.
    tail = prepared[prepared.size - guard - int(SAMPLE_RATE * 0.06): prepared.size - guard]
    assert float(np.max(np.abs(tail))) > 0.1 * body_level


def test_edge_fade_does_not_silence_short_boundary():
    """Кроссфейд 10 мс смягчает край, но не превращает слово в тишину."""
    chunk = sine(0.3, amplitude=0.4)
    settings = SpeakerSettings.from_dict({"voice_id": "voice1"})
    prepared = audio_pipeline._prepare_chunk(chunk, settings, text="Да.")
    fade = int(SAMPLE_RATE * config.EDGE_FADE_MS / 1000)
    assert fade > 0
    # Середина слова не тронута, края приглушены, но не обнулены целиком.
    assert abs(float(np.mean(np.abs(prepared[fade:-fade])))) > 0.05
    assert float(np.max(np.abs(prepared[:fade]))) > 0
    assert float(np.max(np.abs(prepared[-fade:]))) > 0


def test_edge_bounds_accepts_explicit_guard():
    """Запас — параметр: короткая реплика просит больше, длинная меньше."""
    silence = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)
    chunk = np.concatenate((silence, sine(0.3, amplitude=0.4), silence))
    narrow = audio_pipeline._edge_bounds(chunk, 10.0)
    wide = audio_pipeline._edge_bounds(chunk, 120.0)
    assert narrow is not None and wide is not None
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


# --- §21. QA по словам --------------------------------------------------------
def _perfect(text: str) -> np.ndarray:
    """Кусок, который «звучит» как надо: длительность по тексту, без дефектов."""
    seconds = su.expected_duration(len(text)) * 1.2
    return sine(max(seconds, 0.3), amplitude=0.4)


@pytest.mark.parametrize(
    ("expected", "transcription", "first_ok", "last_ok"),
    [
        ("Дайте пройти пожалуйста", "Дайте пройти пожалуйста", True, True),
        ("Дайте пройти пожалуйста", "Дайте пройти", True, False),
        ("Красивая", "Красивая", True, True),
        ("Это проблема", "Проблема", False, True),
        # Порядок проверяется отдельно (`word_order_ok`), а наличие слов — здесь.
        ("Ты опоздал", "Опоздал ты", True, True),
    ],
)
def test_short_qa_reports_first_and_last_word(expected, transcription, first_ok, last_ok):
    verdict = su.check_chunk(_perfect(expected), expected, transcription=transcription, wer=0.0)
    assert verdict.first_word_ok is first_ok
    assert verdict.last_word_ok is last_ok
    assert verdict.expected_words == tuple(su._words(expected))
    assert verdict.asr_words == tuple(su._words(transcription))


def test_short_qa_fails_on_missing_last_word():
    """Потеря последнего слова — провал, а не «WER в пределах порога»."""
    verdict = su.check_chunk(
        _perfect("Дайте пройти пожалуйста"),
        "Дайте пройти пожалуйста",
        transcription="Дайте пройти",
        wer=0.34,
    )
    assert verdict.ok is False
    assert su.REASON_MISSING_WORDS in verdict.reasons
    assert su.REASON_END_TRUNCATION in verdict.reasons
    assert verdict.missing_words == ("пожалуйста",)


def test_short_qa_fails_on_missing_first_word():
    verdict = su.check_chunk(
        _perfect("Это проблема"),
        "Это проблема",
        transcription="Проблема",
        wer=0.5,
    )
    assert verdict.ok is False
    assert su.REASON_START_TRUNCATION in verdict.reasons


def test_short_qa_detects_wrong_word_order():
    """Перестановка слов — отдельная причина, а не «часть слов не слышна»."""
    verdict = su.check_chunk(
        _perfect("Ты опоздал"),
        "Ты опоздал",
        transcription="Опоздал ты",
        wer=1.0,
    )
    assert verdict.word_order_ok is False
    assert su.REASON_WORD_ORDER in verdict.reasons


def test_short_qa_one_word_phrase_needs_the_word():
    """Для однословной реплики единственное слово обязано совпасть."""
    verdict = su.check_chunk(_perfect("Красивая"), "Красивая", transcription="Красиво", wer=1.0)
    assert verdict.ok is False
    assert su.REASON_MISSING_WORDS in verdict.reasons
    # «Потерян конец» для одного слова — не отдельный диагноз: это и есть потеря слова.
    assert su.REASON_END_TRUNCATION not in verdict.reasons

    good = su.check_chunk(_perfect("Красивая"), "Красивая", transcription="Красивая", wer=0.0)
    assert good.ok is True
    assert good.first_word_ok is True and good.last_word_ok is True


def test_short_qa_rejects_empty_and_repetition():
    empty = su.check_chunk(np.zeros(0, dtype=np.float32), "Да.", transcription="", wer=1.0)
    assert empty.ok is False
    assert su.REASON_EMPTY in empty.reasons
    repeated = su.check_chunk(
        _perfect("Да"), "Да", transcription="Да да да да", wer=0.75
    )
    assert su.REASON_REPETITION in repeated.reasons


# --- §18/§21. QA смотрит на подготовленный кусок ------------------------------
def test_qa_measures_prepared_audio_not_raw(stub, monkeypatch):
    """Вердикт и проверка считаются по подготовленному куску, который пойдёт в файл."""
    seen: dict[str, np.ndarray] = {}

    def fake_check(chunk, expected, **kwargs):
        seen["chunk"] = np.asarray(chunk, dtype=np.float32).copy()
        return su.ShortVerdict(ok=True)

    monkeypatch.setattr(audio_pipeline.su, "check_chunk", fake_check)
    # Сырой «выход модели» с тишиной по краям: подготовка и срежет края, и
    # выровняет громкость — то есть подготовленный сигнал заведомо не равен сырому.
    raw_marker = np.concatenate(
        (
            np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32),
            sine(0.4, amplitude=0.4),
            np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32),
        )
    )

    original_attempt = audio_pipeline._synthesize_attempt

    async def attempt_with_marker(*args, **kwargs):
        _raw, seed = await original_attempt(*args, **kwargs)
        # Сырой выход подменяем на известный кусок: так видно, что именно ушло в QA.
        return raw_marker, seed

    monkeypatch.setattr(audio_pipeline, "_synthesize_attempt", attempt_with_marker)
    settings = SpeakerSettings.from_dict({"voice_id": HOLDER["voice"].id})
    expected = audio_pipeline._prepare_chunk(raw_marker, settings, text="Красивая.")

    asyncio.run(
        audio_pipeline.render_dialogue(
            "job-qa-prepared",
            [Replica(voice=SPEAKER, text="Красивая.", line_number=1)],
            {SPEAKER: settings},
            RenderSettings(
                output_format="wav",
                short_utterance=audio_pipeline.ShortUtteranceSettings(
                    enabled=True, strategy="direct"
                ),
            ),
        )
    )
    assert "chunk" in seen
    # Подготовленный кусок отличается от сырого (края срезаны, громкость выровнена)
    # и совпадает с тем, что пайплайн положит в файл.
    assert seen["chunk"].size != raw_marker.size
    assert np.allclose(seen["chunk"], expected)


# --- §15. Трейс: сравнение стадий --------------------------------------------
def test_trace_exposes_raw_and_final_for_comparison(monkeypatch, workspace):
    """В режиме разбора доступны и сырой, и финальный WAV одной реплики."""
    monkeypatch.setenv(synthesis_trace.TRACE_ENV, "1")
    monkeypatch.setenv(synthesis_trace.TRACE_DIR_ENV, str(workspace / "trace"))
    _render(["Красивая."], job_id="job-compare")
    root = workspace / "trace" / "job-compare" / "r001"
    raw = root / f"{synthesis_trace.STAGE_RAW}.wav"
    final = root / f"{synthesis_trace.STAGE_FINAL}.wav"
    assert raw.is_file() and final.is_file()
    record = json.loads(
        (workspace / "trace" / "job-compare" / "trace.jsonl").read_text(encoding="utf-8")
    )
    stages = record["stages"]
    # Длительности стадий записаны и убывают от сырого к финальному.
    assert stages[synthesis_trace.STAGE_RAW]["samples"] > 0
    assert stages[synthesis_trace.STAGE_FINAL]["samples"] > 0
    assert stages[synthesis_trace.STAGE_TRIM]["samples"] <= stages[synthesis_trace.STAGE_RAW][
        "samples"
    ]


def test_normal_utterance_keeps_existing_pipeline(stub):
    """Длинная реплика идёт прежним путём: ни обрезки контекста, ни короткого слоя."""
    long_text = "Сегодня хорошая погода, и я решил прогуляться по набережной до заката."
    _render([long_text], short=True)
    assert len(stub.calls) == 1
    assert stub.calls[0]["text"] == long_text or long_text in stub.calls[0]["text"]
