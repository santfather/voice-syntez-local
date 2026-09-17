"""Слой коротких реплик: классификатор, контекст, стратегии и план.

Модели не поднимаются: здесь только текст и правила, поэтому проверяется ровно
то, что обещает `short_phrasases.md` — короткая реплика распознаётся, контекст
строится, подготовленный текст не меняется, план попадает в движок целиком
(включая разметку ударений для F5 и её отсутствие для XTTS).
"""

from __future__ import annotations

import pytest
from conftest import F5_VOICE, XTTS_VOICE  # noqa: F401 — общие идентификаторы голосов

from backend import audio_pipeline, config
from backend import short_utterance as su
from backend.dialogue_parser import Replica


def _replicas(*pairs: tuple[str, str], final: dict[int, str] | None = None) -> list[Replica]:
    """Реплики «спикер, текст»; `final` подставляет подготовленный текст по индексу."""
    prepared = final or {}
    return [
        Replica(
            voice=speaker,
            text=text,
            line_number=index + 1,
            final_text=prepared.get(index),
        )
        for index, (speaker, text) in enumerate(pairs)
    ]


# --- Phase 2: детектор ---------------------------------------------------------
def test_short_utterance_detector():
    """Классификатор отвечает классом, числом слов и знаков."""
    very_short = su.classify_utterance("Да.")
    assert very_short.kind == su.CLASS_VERY_SHORT
    assert very_short.is_short
    assert (very_short.word_count, very_short.char_count) == (1, 3)

    short = su.classify_utterance("Спасибо, всё хорошо.")
    assert short.kind == su.CLASS_SHORT

    normal = su.classify_utterance("Сегодня хорошая погода, и я рад тебя видеть.")
    assert normal.kind == su.CLASS_NORMAL
    assert normal.is_short is False
    # Слова — основная ось, знаки только уточняют.
    assert su.classify_utterance("Международный авиационно-космический салон").kind == (
        su.CLASS_SHORT
    )


def test_one_word_phrase_is_very_short():
    for text in ("Да.", "Нет.", "Привет!", "Спасибо!", "Что?", "Почему?", "Хорошо.", "Ясно."):
        assert su.classify_utterance(text).kind == su.CLASS_VERY_SHORT, text


def test_two_word_phrase_is_very_short():
    for text in ("Как дела?", "Я понял.", "Всё хорошо?", "До встречи!"):
        assert su.classify_utterance(text).kind == su.CLASS_VERY_SHORT, text


def test_long_sentence_is_normal():
    for text in (
        "Сегодня хорошая погода, и я рад тебя видеть.",
        "Мы закончили работу над проектом и теперь можем отдохнуть.",
        "Он подошёл к окну, посмотрел на улицу и улыбнулся своим мыслям.",
    ):
        assert su.classify_utterance(text).kind == su.CLASS_NORMAL, text


def test_thresholds_are_configurable():
    """Пороги настраиваются, но мусор не превращает всё в «короткое»."""
    strict = su.ShortThresholds(very_short_words=0, short_words=1, very_short_chars=1, short_chars=2)
    assert su.classify_utterance("Привет!", strict).kind == su.CLASS_NORMAL

    loose = su.ShortThresholds(very_short_words=5, short_words=9)
    assert su.classify_utterance("Я понял тебя сейчас", loose).kind == su.CLASS_VERY_SHORT

    # Из запроса приходят только положительные числа; остальное — по умолчанию.
    parsed = su.ShortThresholds.from_dict(
        {"very_short_words": 3, "short_words": "мусор", "very_short_chars": -5}
    )
    assert parsed.very_short_words == 3
    assert parsed.short_words == config.SHORT_UTTERANCE_SHORT_WORDS
    assert parsed.very_short_chars == config.SHORT_UTTERANCE_VERY_SHORT_CHARS


def test_word_count_ignores_punctuation_and_splits_hyphen():
    assert su.count_words("Привет, мир!") == 2
    assert su.count_words("что-то") == 2
    assert su.count_words("...") == 0


def test_accent_mark_does_not_split_words():
    """«+» — разметка ударения: подготовленный текст считается как произносимый.

    Иначе реплика, подготовленная для F5, выглядела бы вдвое длиннее и не попала
    бы в короткую обработку ровно там, где она нужна.
    """
    assert su.count_words("Прив+ет!") == 1
    assert su.count_words("Д+а.") == 1
    assert su.count_chars("Д+а.") == 3
    assert su.classify_utterance("Д+а.").kind == su.CLASS_VERY_SHORT
    assert su.classify_utterance("Я верн+улась дом+ой.").kind == su.CLASS_SHORT


# --- Phase 3: контекст ---------------------------------------------------------
def test_same_speaker_context_can_be_selected():
    replicas = _replicas(
        ("Анна", "Я только что вернулась домой."),
        ("Борис", "Правда?"),
        ("Анна", "Да."),
    )
    contexts = su.build_contexts(replicas)
    target = contexts[2]
    assert target.same_speaker_previous == "Я только что вернулась домой."
    assert target.same_speaker_next is None
    # Кросс-спикерный сосед сохраняется отдельно: benchmark проверяет его гипотезу,
    # но производственная стратегия его не берёт.
    assert target.previous_text == "Правда?"
    assert target.speaker == "Анна"

    plan = su.build_plan(target, strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    assert plan.synthesis_text == "Я только что вернулась домой. Да."
    assert plan.context_source == su.SOURCE_SAME_SPEAKER_PREVIOUS
    assert plan.needs_crop is True


def test_different_speaker_is_not_merged_as_same_speaker():
    replicas = _replicas(("Анна", "Привет!"), ("Борис", "Как дела?"))
    contexts = su.build_contexts(replicas)
    assert contexts[1].same_speaker_previous is None
    assert contexts[1].same_speaker_next is None
    # Контекста того же спикера нет — стратегия честно откатывается на DIRECT.
    plan = su.build_plan(contexts[1], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    assert plan.strategy == su.STRATEGY_DIRECT
    assert plan.synthesis_text == "Как дела?"
    assert plan.context_source == ""
    assert "того же спикера" in plan.note


def test_next_same_speaker_is_used_when_no_previous():
    replicas = _replicas(("Борис", "Привет!"), ("Анна", "Да."), ("Анна", "Очень устала."))
    contexts = su.build_contexts(replicas)
    plan = su.build_plan(contexts[1], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    assert plan.synthesis_text == "Очень устала. Да."
    assert plan.context_source == su.SOURCE_SAME_SPEAKER_NEXT


def test_adjacent_short_same_speaker_replicas_can_be_grouped():
    replicas = _replicas(
        ("Анна", "Привет!"),
        ("Анна", "Как дела?"),
        ("Анна", "Всё хорошо?"),
    )
    contexts = su.build_contexts(replicas)
    batches = su.build_batches(contexts)
    assert set(batches) == {0, 1, 2}
    batch = batches[0]
    assert batch.indexes == (0, 1, 2)
    assert batch.text == "Привет! Как дела? Всё хорошо?"
    assert batch.speaker == "Анна"


def test_adjacent_different_speakers_are_not_grouped():
    replicas = _replicas(
        ("Анна", "Привет!"),
        ("Борис", "Как дела?"),
        ("Анна", "Хорошо."),
    )
    contexts = su.build_contexts(replicas)
    batches = su.build_batches(contexts)
    assert batches[0].indexes == (0,)
    assert batches[1].indexes == (1,)
    assert batches[2].indexes == (2,)
    # Даже при явном разрешении кросс-спикерной склейки (только для benchmark)
    # производственное правило остаётся правилом по умолчанию.
    assert su.build_batches(contexts, allow_cross_speaker=True)[0].indexes == (0, 1, 2)


def test_long_replicas_break_the_group():
    replicas = _replicas(
        ("Анна", "Привет!"),
        ("Анна", "Сегодня хорошая погода, и я рад тебя видеть."),
        ("Анна", "Как дела?"),
    )
    contexts = su.build_contexts(replicas)
    batches = su.build_batches(contexts)
    assert batches[0].indexes == (0,)
    assert batches[2].indexes == (2,)


def test_batch_limit_is_respected():
    replicas = _replicas(*[("Анна", f"Да{i}.") for i in range(5)])
    contexts = su.build_contexts(replicas)
    batches = su.build_batches(contexts, max_replicas=2)
    assert batches[0].indexes == (0, 1)
    assert batches[2].indexes == (2, 3)
    assert batches[4].indexes == (4,)


# --- Phase 3/6: подготовленный текст не меняется --------------------------------
def test_short_strategy_does_not_modify_source_text():
    replicas = _replicas(("Анна", "Да."), ("Анна", "Я вернулась."))
    contexts = su.build_contexts(replicas)
    source_before = [replica.text for replica in replicas]

    su.build_plan(contexts[0], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    su.build_plan(contexts[0], strategy=su.STRATEGY_SYNTHETIC_CONTEXT)
    su.build_batches(contexts)

    assert [replica.text for replica in replicas] == source_before


def test_short_strategy_does_not_modify_final_text():
    """`final_text` — то, что подтвердил пользователь, и он не меняется."""
    replicas = _replicas(
        ("Анна", "Да."),
        ("Анна", "Я вернулась домой."),
        final={0: "Д+а.", 1: "Я верн+улась дом+ой."},
    )
    contexts = su.build_contexts(replicas)
    before = [replica.final_text for replica in replicas]

    plan = su.build_plan(contexts[0], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    assert plan.target_text == "Д+а."
    assert plan.synthesis_text == "Я верн+улась дом+ой. Д+а."

    for replica in replicas:
        assert replica.final_text in before
    assert [replica.final_text for replica in replicas] == before


def test_short_strategy_uses_prepared_text():
    """Контекст и цель берутся из подготовленного текста, а не из исходного."""
    replicas = _replicas(
        ("Анна", "да"),
        ("Анна", "я вернулась домой"),
        final={0: "Д+а.", 1: "Я верн+улась дом+ой."},
    )
    contexts = su.build_contexts(replicas)
    plan = su.build_plan(contexts[0], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    assert plan.target_text == "Д+а."
    assert "Я верн+улась дом+ой." in plan.synthesis_text


def test_f5_short_context_preserves_accent_markup():
    """F5 получает разметку ударений и в контексте — иначе половина входа «сырая»."""
    replicas = _replicas(
        ("Анна", "Да."),
        ("Анна", "Я вернулась домой."),
        final={0: "Д+а.", 1: "Я верн+улась дом+ой."},
    )
    contexts = su.build_contexts(replicas)
    plan = su.build_plan(
        contexts[0], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT, supports_accents=True
    )
    assert plan.synthesis_text == "Я верн+улась дом+ой. Д+а."
    assert plan.synthesis_text.count("+") == 3  # два в контексте + один в цели
    # Нейтральный carrier ударений не несёт — он и не должен.
    synthetic = su.build_plan(contexts[0], strategy=su.STRATEGY_SYNTHETIC_CONTEXT)
    assert synthetic.synthesis_text == "Хорошо. Д+а. Хорошо."


def test_xtts_short_context_has_no_f5_markup():
    """XTTS не должна получить «+»: она прочитала бы знак вслух (§15)."""
    replicas = _replicas(
        ("Анна", "Да."),
        ("Анна", "Я вернулась домой."),
        # Контекст подготовлен для F5 (с «+»), а цель синтезируется XTTS.
        final={0: "Да.", 1: "Я верн+улась дом+ой."},
    )
    contexts = su.build_contexts(replicas)
    plan = su.build_plan(
        contexts[0], strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT, supports_accents=False
    )
    assert "+" not in plan.synthesis_text
    assert plan.synthesis_text == "Я вернулась домой. Да."
    assert su.strip_accents("Я верн+улась дом+ой.") == "Я вернулась домой."


# --- стратегии и откаты ---------------------------------------------------------
def test_short_strategy_can_fallback_to_direct():
    """Нет надёжной границы — работаем как раньше, а не режем «примерно»."""
    replicas = _replicas(("Анна", "Да."), ("Анна", "Я вернулась домой."))
    contexts = su.build_contexts(replicas)
    plan = su.build_plan(
        contexts[0],
        strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT,
        allow_context=False,
    )
    assert plan.strategy == su.STRATEGY_DIRECT
    assert plan.synthesis_text == plan.target_text == "Да."
    assert plan.needs_crop is False
    assert "надёжной границы" in plan.note


def test_direct_strategy_is_baseline():
    replicas = _replicas(("Анна", "Привет!"))
    plan = su.build_plan(su.build_contexts(replicas)[0])
    assert plan.strategy == su.STRATEGY_DIRECT
    assert plan.synthesis_text == "Привет!"
    assert plan.needs_crop is False


def test_punctuation_strategy_normalizes_without_changing_intonation():
    replicas = _replicas(("Анна", "Привет !!!"))
    plan = su.build_plan(su.build_contexts(replicas)[0], strategy=su.STRATEGY_PUNCTUATION)
    assert plan.synthesis_text == "Привет!"
    # «!» не превращается в «.» без доказанного улучшения (§6).
    assert plan.synthesis_text.endswith("!")


def test_punctuation_variants_are_experiments_only():
    assert su.apply_punctuation_variant("Привет!", su.PUNCTUATION_PERIOD) == "Привет."
    assert su.apply_punctuation_variant("Привет!", su.PUNCTUATION_ELLIPSIS) == "Привет…"
    assert su.apply_punctuation_variant("Привет!", su.PUNCTUATION_AS_IS) == "Привет!"


def test_context_sides_are_selectable():
    """A/B/C/D из §30: только цель, префикс, суффикс, оба."""
    replicas = _replicas(("Анна", "Да."), ("Анна", "Я вернулась."))
    context = su.build_contexts(replicas)[0]
    strategy = su.STRATEGY_SAME_SPEAKER_CONTEXT
    assert (
        su.build_plan(context, strategy=strategy, side=su.SIDE_PREFIX).synthesis_text
        == "Я вернулась. Да."
    )
    assert (
        su.build_plan(context, strategy=strategy, side=su.SIDE_SUFFIX).synthesis_text
        == "Да. Я вернулась."
    )
    assert (
        su.build_plan(context, strategy=strategy, side=su.SIDE_BOTH).synthesis_text
        == "Я вернулась. Да. Я вернулась."
    )
    # По умолчанию: сосед — префикс, carrier — с двух сторон.
    assert su.build_plan(context, strategy=strategy).side == su.SIDE_PREFIX
    assert su.build_plan(context, strategy=su.STRATEGY_SYNTHETIC_CONTEXT).side == su.SIDE_BOTH


def test_plan_metadata_is_recorded():
    """§24: класс, стратегия, источник контекста и хеш синтез-текста."""
    replicas = _replicas(("Анна", "Да."), ("Анна", "Я вернулась."))
    plan = su.build_plan(su.build_contexts(replicas)[0], strategy=su.STRATEGY_SYNTHETIC_CONTEXT)
    payload = plan.to_dict()
    assert payload["tts_target_text"] == "Да."
    assert payload["tts_synthesis_text"] == "Хорошо. Да. Хорошо."
    assert payload["short_utterance_strategy"] == su.STRATEGY_SYNTHETIC_CONTEXT
    assert payload["short_utterance_context_source"] == su.SOURCE_SYNTHETIC
    assert payload["short_utterance_class"] == su.CLASS_VERY_SHORT
    assert payload["synthesis_text_hash"] == plan.synthesis_hash
    assert len(plan.synthesis_hash) == 8
    # Хеш различает разные synthesis-тексты и повторяется при том же.
    other = su.build_plan(su.build_contexts(replicas)[0], strategy=su.STRATEGY_DIRECT)
    assert other.synthesis_hash != plan.synthesis_hash


def test_context_does_not_leak_empty_plan_for_empty_utterance():
    replicas = _replicas(("Анна", ""))
    plan = su.build_plan(su.build_contexts(replicas)[0], strategy=su.STRATEGY_SYNTHETIC_CONTEXT)
    assert plan.synthesis_text == ""
    assert plan.needs_crop is False


def test_unknown_strategy_is_rejected():
    assert su.resolve_strategy("f5", su.STRATEGY_AUTO) == su.STRATEGY_DIRECT
    assert su.resolve_strategy("f5", su.STRATEGY_PUNCTUATION) == su.STRATEGY_PUNCTUATION
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        su.resolve_strategy("f5", "магия")


def test_production_policy_is_direct_until_benchmark(monkeypatch):
    """До измерений производственная политика — DIRECT (§29, Phase 4)."""
    assert su.default_strategy("f5") == su.STRATEGY_DIRECT
    monkeypatch.setitem(config.SHORT_UTTERANCE_ENGINE_STRATEGIES, "f5", su.STRATEGY_PUNCTUATION)
    assert su.default_strategy("f5") == su.STRATEGY_PUNCTUATION
    assert su.resolve_strategy("f5", su.STRATEGY_AUTO) == su.STRATEGY_PUNCTUATION
    assert su.default_strategy("xtts") == su.STRATEGY_DIRECT


def test_synthesis_input_is_logged_with_full_context(caplog, voices):
    """§1: перед вызовом движка в лог уходит весь вход синтеза (без текста целиком)."""
    import asyncio

    from conftest import StubEngine

    engine = StubEngine()
    engine.load()
    voice = voices[0]
    tuning = audio_pipeline.SpeakerSettings(voice_id=voice.id)
    with caplog.at_level("INFO", logger="backend.audio_pipeline"):
        asyncio.run(
            audio_pipeline._synthesize_chunk(
                engine,
                voice,
                "Прив+ет!",
                tuning,
                audio_pipeline.RenderSettings(),
                position="1",
                label="Анна",
                source_text="Привет!",
            )
        )
    line = next(
        record.getMessage() for record in caplog.records if "tts.synthesis.input" in record.getMessage()
    )
    for expected in (
        "engine=stub",
        "voice=",
        "label=Анна",
        "position=1",
        "prepared=True",
        "words=1",
        "class=very_short",
        "ref=",
        "speed=",
        "seed=",
        "params=",
    ):
        assert expected in line, (expected, line)
    # Текст реплики попадает в лог отпечатком: длина, sha1 и короткая выжимка.
    assert "Прив+ет!" in line
    assert "sha1" in line


# --- Phase 5: short QA и ограниченный повтор ------------------------------------
def _tone(seconds: float, amplitude: float = 0.2):
    import numpy as np

    timeline = np.arange(int(audio_pipeline.SAMPLE_RATE * seconds), dtype=np.float32)
    timeline = timeline / audio_pipeline.SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 180.0 * timeline)).astype(np.float32)


def test_short_audio_empty_result_is_rejected():
    """Пустое аудио — провал, а не «короткая реплика»."""
    import numpy as np

    verdict = su.check_chunk(np.zeros(0, dtype=np.float32), "Да.")
    assert verdict.ok is False
    assert su.REASON_EMPTY in verdict.reasons
    assert "пустое аудио" in su.describe_short_reasons(list(verdict.reasons))


def test_short_audio_silence_and_tiny_duration_are_rejected():
    import numpy as np

    silence = su.check_chunk(np.zeros(audio_pipeline.SAMPLE_RATE // 2, dtype=np.float32), "Да.")
    assert silence.ok is False
    assert su.REASON_SILENCE in silence.reasons

    tiny = su.check_chunk(_tone(0.05), "Да.")
    assert tiny.ok is False
    assert su.REASON_DURATION_SHORT in tiny.reasons


def test_short_audio_repetition_is_detected():
    """«Да, да, да…» — главный дефект короткой реплики, и он виден по расшифровке."""
    verdict = su.check_chunk(_tone(1.2), "Да.", transcription="Да, да, да!")
    assert verdict.ok is False
    assert su.REASON_REPETITION in verdict.reasons
    assert su.REASON_DURATION_LONG in verdict.reasons


def test_short_audio_missing_and_extra_words_are_detected():
    missing = su.check_chunk(_tone(0.4), "До встречи!", transcription="Встречи!")
    assert su.REASON_MISSING_WORDS in missing.reasons

    extra = su.check_chunk(_tone(0.4), "Да.", transcription="Да, конечно, хорошо")
    assert su.REASON_EXTRA_WORDS in extra.reasons


def test_short_audio_normal_result_passes():
    assert su.check_chunk(_tone(0.45), "Да.").ok is True
    assert su.check_chunk(_tone(0.45), "Да.", transcription="Да.").ok is True
    # Ожидаемая длительность считается от текста, а не задана одним числом.
    assert su.expected_duration(3) < su.expected_duration(12)


def test_short_long_audio_for_tiny_phrase_is_rejected():
    """Растянутое аудио для «Да.» — тоже дефект, а не «богатая интонация»."""
    verdict = su.check_chunk(_tone(3.0), "Да.", transcription="Да.")
    assert su.REASON_DURATION_LONG in verdict.reasons


def test_short_failed_qa_can_retry():
    failed = su.check_chunk(_tone(3.0), "Да.", transcription="Да.")
    assert su.should_retry_short(attempt=1, max_attempts=2, verdict=failed) is True


def test_short_successful_qa_does_not_retry():
    passed = su.check_chunk(_tone(0.45), "Да.")
    assert su.should_retry_short(attempt=1, max_attempts=3, verdict=passed) is False


def test_short_retry_is_limited():
    """Повтор ограничен: генерация «десятков вариантов» — не решение (§18)."""
    failed = su.check_chunk(_tone(3.0), "Да.", transcription="Да.")
    assert su.should_retry_short(attempt=1, max_attempts=2, verdict=failed) is True
    assert su.should_retry_short(attempt=2, max_attempts=2, verdict=failed) is False
    assert su.should_retry_short(attempt=5, max_attempts=2, verdict=failed) is False
    # Без вердикта повтора нет: проверка не гонялась — повторять нечего.
    assert su.should_retry_short(attempt=1, max_attempts=3, verdict=None) is False
    # Нулевой лимит не превращается в бесконечный цикл.
    assert su.should_retry_short(attempt=0, max_attempts=0, verdict=failed) is False


def test_best_attempt_is_chosen_by_reasons_then_wer():
    """Из попыток берётся лучшая: меньше причин, затем ниже WER."""
    good = su.ShortVerdict(ok=True, reasons=(), wer=0.0)
    bad_two = su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION, su.REASON_DURATION_LONG))
    bad_one = su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION,))
    assert su.best_verdict([bad_two, good, bad_one]) == 1
    assert su.best_verdict([bad_two, bad_one]) == 1  # у одной причины приоритет
    close_a = su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION,), wer=0.5)
    close_b = su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION,), wer=0.1)
    assert su.best_verdict([close_a, close_b]) == 1
