"""Разбор диалога: спикеры, слоты, параметры в маркерах и нарезка длинных реплик."""

from backend import config
from backend.dialogue_parser import parse_dialogue, slot_key, split_into_chunks, voice_label


def test_named_speakers_split():
    parsed = parse_dialogue("ИВАН: Привет!\nМАРГО: И тебе привет.")
    assert [replica.voice for replica in parsed.replicas] == ["ИВАН", "МАРГО"]
    assert [replica.text for replica in parsed.replicas] == ["Привет!", "И тебе привет."]
    assert [voice.label for voice in parsed.voices] == ["ИВАН", "МАРГО"]
    assert [voice.slot for voice in parsed.voices] == [None, None]


def test_slot_markers_switch_voice_mid_text():
    parsed = parse_dialogue("(1) Привет! (2) Пока!")
    assert [replica.voice for replica in parsed.replicas] == ["#1", "#2"]
    assert [replica.text for replica in parsed.replicas] == ["Привет!", "Пока!"]
    assert slot_key(3) == "#3"
    assert voice_label("#3") == "Слот 3"
    assert voice_label("ИВАН") == "ИВАН"


def test_text_without_markers_goes_to_default_slot():
    parsed = parse_dialogue("Просто текст без спикеров.")
    assert [replica.voice for replica in parsed.replicas] == [slot_key(config.DEFAULT_SLOT)]


def test_marker_params_are_clamped_to_limits():
    parsed = parse_dialogue("(1 speed=9 cfg=0.1 nfe=16) Текст")
    overrides = parsed.replicas[0].overrides
    assert overrides["speed"] == config.SPEED_RANGE[1]
    assert overrides["cfg_strength"] == config.CFG_RANGE[0]
    assert overrides["nfe_step"] == 16
    assert parsed.override_count == 1


def test_disallowed_nfe_is_ignored():
    parsed = parse_dialogue("(1 nfe=13) Текст")
    assert "nfe_step" not in parsed.replicas[0].overrides


def test_time_and_year_are_not_markers_or_slots():
    parsed = parse_dialogue("(1) Встреча в 12:30 в (2024) году.")
    assert len(parsed.replicas) == 1
    assert "12:30" in parsed.replicas[0].text
    assert "(2024)" in parsed.replicas[0].text


def test_long_replica_is_cut_on_sentence_bounds():
    first = "Первое предложение ровно в сорок знаков."
    second = "Второе предложение тоже в сорок знаков."
    text = f"{first} {second}"
    assert len(text) > 60

    parsed = parse_dialogue(text, max_replica_chars=60)
    assert len(parsed.replicas) == 2
    # Разрез строго на границе предложения, а не по счётчику символов.
    assert parsed.replicas[0].text.endswith(".")
    assert parsed.replicas[1].text.endswith(".")
    assert parsed.replicas[0].text + " " + parsed.replicas[1].text == text
    assert {replica.line_number for replica in parsed.replicas} == {1}


def test_split_into_chunks_keeps_every_piece_within_limit():
    text = " ".join(f"Предложение номер {index} с некоторым текстом." for index in range(40))
    chunks = split_into_chunks(text, 120)
    assert chunks
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_short_paragraphs_are_merged_into_one_chunk():
    text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
    # Абзацы короткие: резать их по одному — сотни крошечных кусков на книге.
    assert split_into_chunks(text, 600) == ["Первый абзац. Второй абзац. Третий абзац."]


def test_empty_input_gives_empty_list():
    assert parse_dialogue("   \n  ").replicas == []
    assert split_into_chunks("") == []
