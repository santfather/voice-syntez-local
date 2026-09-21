"""Сборка трека: подготовка куска, громкость файла, лимитер, варианты и паузы."""

import asyncio

import numpy as np
import pytest
import soundfile as sf
from conftest import dominant_hz, sine

from backend import audio_pipeline, config, warmup_context
from backend.audio_pipeline import (
    RenderSettings,
    SpeakerSettings,
    _finalize_track,
    _limit_peaks,
    _normalize_loudness,
    _pause_samples,
    _prepare_chunk,
    _settings_for,
    _text_for_engine,
    _trim_edge_silence,
    apply_variant,
    drop_variant,
    read_segment,
    read_variant,
    render_dialogue,
    save_chunk_variant,
    variant_path,
)
from backend.dialogue_parser import Replica
from backend.engines.base import SAMPLE_RATE, STATE_READY, EngineInfo, SynthesisEngine
from backend.transcribe import word_error_rate


def _speaker(voice_id: str = "voice1", **overrides) -> SpeakerSettings:
    return SpeakerSettings(voice_id=voice_id, **overrides)


def _replica(text: str = "Привет, это тест", voice: str = "#1") -> Replica:
    return Replica(voice=voice, text=text, line_number=1)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _lufs(audio: np.ndarray) -> float:
    import pyloudnorm as pyln

    return float(pyln.Meter(SAMPLE_RATE).integrated_loudness(audio))


# --- подготовка куска ---------------------------------------------------------
def test_trim_edge_silence_removes_model_padding():
    chunk = np.concatenate([np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32),
                            sine(1.0, 300.0),
                            np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)])
    trimmed = _trim_edge_silence(chunk)
    assert trimmed.size < chunk.size
    # Осталась речь плюс небольшой запас вокруг неё — пауза задаётся отдельно.
    assert trimmed.size == pytest.approx(SAMPLE_RATE * 1.0, abs=0.2 * SAMPLE_RATE)


def test_trim_edge_silence_keeps_fully_quiet_chunk():
    quiet = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert _trim_edge_silence(quiet).size == quiet.size


def test_prepare_chunk_normalizes_rms_and_fades_edges():
    prepared = _prepare_chunk(sine(2.0, 300.0, amplitude=0.02), _speaker(target_rms=0.1))
    assert _rms(prepared) == pytest.approx(0.1, abs=0.002)
    # Края сглажены: щелчок модели кроссфейд не убирает, а fade убирает.
    fade = int(SAMPLE_RATE * config.EDGE_FADE_MS / 1000)
    assert abs(float(prepared[0])) < 1e-3
    assert abs(float(prepared[-1])) < 1e-3
    assert np.max(np.abs(prepared[fade:-fade])) > 0.05


def test_prepare_chunk_applies_gain_after_normalization():
    quiet = _prepare_chunk(sine(2.0, 300.0, amplitude=0.02), _speaker(target_rms=0.05))
    loud = _prepare_chunk(
        sine(2.0, 300.0, amplitude=0.02), _speaker(target_rms=0.05, gain_db=6.0)
    )
    assert _rms(loud) / _rms(quiet) == pytest.approx(10 ** (6.0 / 20.0), rel=0.01)


def test_prepare_chunk_limits_peak_above_ceiling():
    prepared = _prepare_chunk(sine(2.0, 300.0, amplitude=0.02), _speaker(target_rms=0.3, gain_db=20.0))
    assert float(np.max(np.abs(prepared))) <= audio_pipeline._PEAK_LIMIT + 1e-6


# --- громкость файла и лимитер ------------------------------------------------
def test_normalize_loudness_hits_target():
    rng = np.random.default_rng(1)
    audio = (0.02 * rng.standard_normal(SAMPLE_RATE * 4)).astype(np.float32)
    assert _lufs(audio) < -30
    measured = _lufs(_normalize_loudness(audio, -16.0))
    assert measured == pytest.approx(-16.0, abs=0.3)


def test_normalize_loudness_skips_too_short_audio():
    short = sine(0.2, 300.0)
    assert np.array_equal(_normalize_loudness(short, -16.0), short)


def test_limit_peaks_holds_ceiling_and_leaves_quiet_audio_alone():
    loud = _limit_peaks((1.5 * sine(1.0, 220.0)).astype(np.float32))
    assert float(np.max(np.abs(loud))) <= audio_pipeline._PEAK_LIMIT + 1e-6

    quiet = sine(1.0, 220.0, amplitude=0.2)
    assert np.allclose(_limit_peaks(quiet), quiet)


def test_finalize_track_sets_loudness_and_ceiling():
    rng = np.random.default_rng(2)
    audio = (0.02 * rng.standard_normal(SAMPLE_RATE * 4)).astype(np.float32)
    finalized = _finalize_track(audio)
    assert _lufs(finalized) == pytest.approx(config.OUTPUT_LUFS, abs=0.3)
    assert float(np.max(np.abs(finalized))) <= audio_pipeline._PEAK_LIMIT + 1e-6


# --- варианты куска -----------------------------------------------------------
def test_chunk_variant_roundtrip_and_drop(workspace):
    variant = save_chunk_variant("job1", 0, "v1", sine(1.0, 400.0), 4242, "вариант 1")
    assert variant.path == variant_path("job1", 0, "v1")
    assert variant.path.exists()
    assert variant.seed == 4242
    assert variant.duration_sec == pytest.approx(1.0, abs=0.01)

    content = read_variant(variant)
    assert content.size == pytest.approx(int(SAMPLE_RATE * 1.0), abs=SAMPLE_RATE * 0.01)

    drop_variant(variant)
    assert not variant.path.exists()


def test_read_segment_returns_exact_bounds(workspace):
    path = workspace / "output" / "job.wav"
    audio = np.concatenate([sine(0.5, 200.0), sine(0.5, 800.0)])
    sf.write(path, audio, SAMPLE_RATE)

    bounds = (int(SAMPLE_RATE * 0.5), SAMPLE_RATE)
    piece = read_segment(path, "wav", bounds)
    assert piece.size == bounds[1] - bounds[0]
    assert dominant_hz(piece) == pytest.approx(800.0, abs=5.0)


def test_apply_variant_replaces_only_target_segment(workspace):
    path = workspace / "output" / "job.wav"
    parts = [sine(0.5, 200.0), sine(0.5, 800.0), sine(0.5, 1500.0)]
    audio = np.concatenate(parts)
    sf.write(path, audio, SAMPLE_RATE)
    segments = []
    cursor = 0
    for part in parts:
        segments.append((cursor, cursor + part.size))
        cursor += part.size

    variant = save_chunk_variant("job", 1, "v1", sine(0.4, 3000.0), 7, "вариант 1")
    duration, bounds = asyncio.run(
        apply_variant(path, RenderSettings(output_format="wav"), segments, 1, variant)
    )

    updated = sf.read(path, dtype="float32")[0]
    assert duration == pytest.approx(updated.size / SAMPLE_RATE)
    assert bounds == (segments[1][0], segments[1][0] + read_variant(variant).size)
    assert len(updated) == audio.size - parts[1].size + read_variant(variant).size

    replaced = updated[bounds[0]:bounds[1]]
    assert dominant_hz(replaced) == pytest.approx(3000.0, abs=5.0)
    # Соседи не тронуты по содержимому: это те же куски, что и были в файле.
    assert dominant_hz(updated[:segments[0][1]]) == pytest.approx(200.0, abs=5.0)
    assert dominant_hz(updated[bounds[1]:]) == pytest.approx(1500.0, abs=5.0)


# --- параметры и текст --------------------------------------------------------
def test_pause_samples_prefers_speaker_override():
    assert _pause_samples(_speaker(), RenderSettings(pause_ms=400)) == pytest.approx(
        int(SAMPLE_RATE * 0.4)
    )
    assert _pause_samples(_speaker(pause_override_ms=0), RenderSettings(pause_ms=400)) == 0


def test_settings_for_applies_marker_overrides(fake_store):
    replica = Replica(voice="#1", text="Текст", line_number=1, overrides={"speed": 1.5})
    slot = SpeakerSettings.from_dict({"voice_id": "voice1", "cfg_strength": 2.5})
    tuned = _settings_for(replica, slot)
    # Правка реплики (маркер в тексте) важнее слота, а нетронутая ручка — из слота.
    assert tuned.speed == 1.5 and tuned.cfg_strength == 2.5
    assert _settings_for(_replica(), slot).speed == 1.0


def test_text_for_engine_accents_only_for_supporting_engines(monkeypatch):
    monkeypatch.setattr(audio_pipeline, "accentuate", lambda text: f"[{text}]")

    class _Engine(SynthesisEngine):
        def load(self) -> None:
            self._mark(STATE_READY)

        def _synthesize(self, text, ref_audio_path, ref_text, speed, params):
            raise AssertionError("в этом тесте синтез не нужен")

    class AccentEngine(_Engine):
        info = EngineInfo(
            id="accents", label="С ударениями", description="", supports_accents=True
        )

    class PlainEngine(_Engine):
        info = EngineInfo(
            id="plain", label="Без ударений", description="", supports_accents=False
        )

    # Числа разворачиваются до ударений: «+» должен встать уже в готовое слово.
    assert _text_for_engine("12 попыток", AccentEngine(), True) == "[двенадцать попыток]"
    assert _text_for_engine("12 попыток", PlainEngine(), True) == "двенадцать попыток"
    assert _text_for_engine("12 попыток", AccentEngine(), False) == "двенадцать попыток"


# --- сборка диалога -----------------------------------------------------------
def _render(replicas, speakers, settings=None, **kwargs):
    return asyncio.run(
        render_dialogue(
            job_id="job",
            replicas=replicas,
            speakers=speakers,
            settings=settings or RenderSettings(pause_ms=300, output_format="wav"),
            **kwargs,
        )
    )


def test_render_dialogue_pauses_are_exactly_pause_ms(stub, fake_store):
    replicas = [_replica("Первая реплика"), _replica("Вторая реплика"), _replica("Третья реплика")]
    result = _render(replicas, {"#1": _speaker()})

    assert result.output_path.exists()
    assert result.replicas_done == 3
    assert len(result.segments) == 3
    assert len(result.seeds) == 3
    assert all(isinstance(seed, int) for seed in result.seeds)
    assert stub.loads == 1  # движок поднят один раз на весь диалог

    audio = sf.read(result.output_path, dtype="float32")[0]
    assert result.duration_sec == pytest.approx(audio.size / SAMPLE_RATE)
    silence = int(SAMPLE_RATE * 0.3)
    for index in range(2):
        gap = audio[result.segments[index][1]:result.segments[index + 1][0]]
        assert gap.size == silence  # ровно заданная пауза, без тишины модели поверх
        assert not np.any(gap)
    assert _lufs(audio) == pytest.approx(config.OUTPUT_LUFS, abs=0.3)


def test_render_dialogue_normalizes_text_per_engine(stub, fake_store):
    _render([_replica("В 2026 году 12 попыток")], {"#1": _speaker()})
    assert stub.calls[0]["text"] == "В две тысячи двадцать шестом году двенадцать попыток"


def test_render_dialogue_rejects_empty_and_unknown_voice(stub, fake_store):
    with pytest.raises(ValueError, match="Диалог пуст"):
        _render([], {})
    with pytest.raises(ValueError, match="Не назначен голос"):
        _render([_replica()], {"#2": _speaker()})
    with pytest.raises(ValueError, match="не найден"):
        _render([_replica()], {"#1": _speaker(voice_id="нет-такого")})


def test_render_dialogue_aborts_between_replicas(stub, fake_store):
    calls = {"count": 0}

    def should_abort() -> bool:
        calls["count"] += 1
        return calls["count"] > 1  # первая реплика успевает, дальше — стоп

    with pytest.raises(audio_pipeline.JobAbortedError):
        _render([_replica("Первая"), _replica("Вторая")], {"#1": _speaker()}, should_abort=should_abort)


def test_synthesize_replica_returns_seed_and_prepared_chunk(stub, fake_store):
    prepared, seed, qa, quality, short_run, reference = asyncio.run(
        audio_pipeline.synthesize_replica(_replica(), _speaker(), RenderSettings(), 0)
    )
    assert isinstance(seed, int)
    assert qa is None  # проверка выключена — отметки нет
    assert short_run is None  # слой коротких реплик не запрашивали
    # Референс едет вместе с куском: пересинтез обязан записать в метаданные
    # варианта, каким профилем получено это звучание (UPDATE 2 §12).
    assert reference.profile_id.endswith("-neutral")
    assert reference.resolved_emotion == "NEUTRAL"
    assert prepared.dtype == np.float32
    assert _rms(prepared) == pytest.approx(config.DEFAULT_TARGET_RMS, abs=0.002)
    # Диагностика едет вместе с куском: у вызывающего сырого выхода модели нет.
    assert quality.duration_sec == pytest.approx(prepared.size / SAMPLE_RATE, abs=1e-6)
    assert quality.qa_attempts is None


# --- строгая проверка куска (Фаза 7) ------------------------------------------
def _qa_render(**overrides) -> RenderSettings:
    """Сборка со включённой проверкой: бюджет и порог задаёт сам тест."""
    return RenderSettings(
        pause_ms=0, output_format="wav", qa=audio_pipeline.QaSettings(**overrides)
    )


def _answers(monkeypatch, *texts) -> list[np.ndarray]:
    """Подменяет Whisper: отдаёт расшифровки по порядку попыток и пишет куски.

    Куски записываются, чтобы отличить «вернули лучшую попытку» от «вернули
    последнюю»: у заглушки длина зависит от номера вызова.
    """
    chunks: list[np.ndarray] = []

    async def fake(chunk: np.ndarray) -> str:
        chunks.append(chunk)
        return texts[min(len(chunks) - 1, len(texts) - 1)]

    monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", fake)
    return chunks


def test_qa_passes_on_first_attempt(stub, fake_store, monkeypatch):
    _answers(monkeypatch, "Привет это тест")
    result = _render([_replica()], {"#1": _speaker()}, _qa_render())

    assert len(stub.calls) == 1  # порог взят — повторять нечего
    # Расшифровка теперь едет вместе с итогом проверки: по ней видно, **что
    # услышал** Whisper, а не только насколько это похоже на текст (§14 UPDATE 2).
    assert result.qa == [
        audio_pipeline.QaOutcome(
            status=audio_pipeline.QA_PASSED,
            wer=0.0,
            attempts=1,
            transcription="Привет это тест",
        )
    ]


def test_qa_retries_and_moves_tuning_within_passport(stub, fake_store, monkeypatch):
    _answers(monkeypatch, "совсем другой текст", "Привет это тест")
    result = _render(
        [_replica()], {"#1": _speaker()}, _qa_render(wer_threshold=0.5, max_attempts=3)
    )

    assert len(stub.calls) == 2
    assert result.qa[0].status == audio_pipeline.QA_PASSED
    assert result.qa[0].attempts == 2
    # Подстройка — объявленный шаг в границах движка, а не произвольное значение.
    first, second = (call["params"]["cfg_strength"] for call in stub.calls)
    assert second == pytest.approx(first + config.QA_CFG_STEP)
    assert config.CFG_RANGE[0] <= second <= config.CFG_RANGE[1]


def test_qa_budget_stops_cycle_before_retry(stub, fake_store, monkeypatch):
    _answers(monkeypatch, "совсем другой текст", "Привет это тест")
    # Нулевой бюджет — вырожденный, но это самая короткая проверка границы: первая
    # попытка выполняется всегда, вторая уже не начинается.
    result = _render(
        [_replica()], {"#1": _speaker()}, _qa_render(wer_threshold=0.1, budget_sec=0.0)
    )

    assert len(stub.calls) == 1
    assert result.qa[0].status == audio_pipeline.QA_BUDGET
    assert result.qa[0].wer == 1.0
    assert result.qa[0].attempts == 1


def test_qa_returns_best_attempt_not_last(stub, fake_store, monkeypatch):
    chunks = _answers(monkeypatch, "а б в", "и это тест", "совсем другой текст")
    result = _render(
        [_replica()], {"#1": _speaker()}, _qa_render(wer_threshold=0.15, max_attempts=3)
    )

    outcome = result.qa[0]
    assert len(stub.calls) == 3
    assert outcome.status == audio_pipeline.QA_ATTEMPTS
    assert outcome.wer == pytest.approx(1 / 3)  # ближе всех вторая попытка
    # В файл попала именно она: длина куска — длина второй попытки, а не третьей.
    assert result.segments[0][1] - result.segments[0][0] == chunks[1].size


def test_qa_waits_for_memory_only_before_retry(stub, fake_store, monkeypatch):
    _answers(monkeypatch, "совсем другой текст", "Привет это тест")
    waits: list[int] = []

    async def wait_for_memory() -> None:
        waits.append(len(stub.calls))

    _render(
        [_replica()],
        {"#1": _speaker()},
        _qa_render(wer_threshold=0.5, max_attempts=3),
        wait_for_memory=wait_for_memory,
    )

    assert len(stub.calls) == 2
    assert waits == [1]  # Whisper поднимается заново только перед повтором


def test_qa_without_transcription_accepts_chunk_unchecked(stub, fake_store, monkeypatch):
    async def broken(chunk: np.ndarray) -> str:
        raise RuntimeError("распознавание не удалось")

    monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", broken)
    result = _render([_replica()], {"#1": _speaker()}, _qa_render())

    # Упавшая расшифровка — не повод терять уже сгенерированное аудио.
    assert len(stub.calls) == 1
    assert result.qa[0].status == audio_pipeline.QA_UNAVAILABLE
    assert result.qa[0].wer is None


def test_render_without_qa_never_transcribes(stub, fake_store, monkeypatch):
    async def forbidden(chunk: np.ndarray) -> str:
        raise AssertionError("без включённой проверки расшифровка не запускается")

    monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", forbidden)
    result = _render([_replica("Первая реплика"), _replica("Вторая реплика")], {"#1": _speaker()})

    assert result.qa == [None, None]


def test_word_error_rate_counts_words_not_letters():
    assert word_error_rate("привет это тест", "привет это тест") == 0.0
    assert word_error_rate("привет это тест", "и это тест") == pytest.approx(1 / 3)
    assert word_error_rate("привет это тест", "привет это тест совсем") == pytest.approx(1 / 3)
    assert word_error_rate("привет это тест", "") == 1.0
    # «ё» и «е» для проверки одно слово: иначе строгий порог падал бы на ровном месте.
    assert word_error_rate("всё хорошо", "все хорошо") == 0.0
    # Перестановка — это замены, а не «половина текста не совпала»: так считает
    # расстояние Левенштейна, и по нему же принимается решение о повторе.
    assert word_error_rate("а б в г", "г в б а") == 1.0


# --- прогрев коротких реплик --------------------------------------------------
# Прогрев — скрытый текст перед целью. Тесты следят за тремя вещами: он уходит в
# движок, из готового файла он вырезан, а метаданные и проверка видят только цель.
WARMUP_PREFIX = "Тише, сейчас начнём."
WARMUP_TARGET = "Да."


class _FakeWarmupService:
    """Сборщик префикса без LLM: ответ задан заранее, вызовы считаются."""

    def __init__(self, prefix: str | None = WARMUP_PREFIX) -> None:
        self.prefix = prefix
        self.calls: list[dict] = []

    def build(
        self,
        *,
        target_text,
        engine_id,
        previous_text=None,
        next_text=None,
        speaker="",
        enabled=None,
    ) -> warmup_context.WarmupContext:
        self.calls.append({"target_text": target_text, "engine_id": engine_id})
        return warmup_context.WarmupContext(
            target_text=target_text,
            prefix_text=self.prefix,
            enabled=bool(self.prefix),
            reason=(
                warmup_context.REASON_OK if self.prefix else warmup_context.REASON_EMPTY_RESPONSE
            ),
        )


def _warmup_boundary(start_sec: float = 0.5) -> warmup_context.TargetBoundary:
    """Граница цели так, как её вернул бы alignment по таймстемпам слов."""
    return warmup_context.TargetBoundary(
        start_sec=start_sec, confidence=1.0, method="asr", matched_words=1, total_words=1
    )


def _warmup_render(service, monkeypatch, **overrides):
    """Рендер одной короткой реплики с подставленным сервисом прогрева."""
    monkeypatch.setattr(audio_pipeline.warmup_context, "get_service", lambda: service)
    return _render(
        [_replica(WARMUP_TARGET)],
        {"#1": _speaker()},
        RenderSettings(pause_ms=0, output_format="wav", warmup=True, **overrides),
    )


def test_warmup_sends_prefix_and_target_to_engine(stub, fake_store, monkeypatch):
    """Без полного «префикс + цель» модель не получит нужный вход в речь."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: _warmup_boundary()
    )

    _warmup_render(service, monkeypatch)

    assert service.calls[0]["target_text"] == WARMUP_TARGET
    assert len(stub.calls) == 1
    sent = stub.calls[0]["text"]
    assert sent.startswith(WARMUP_PREFIX)
    # Цель — суффикс синтез-текста: именно по этому суффиксу ищется граница.
    assert sent.endswith(WARMUP_TARGET)
    assert sent == f"{WARMUP_PREFIX} {WARMUP_TARGET}"


def test_warmup_crop_removes_prefix_audio(stub, fake_store, monkeypatch):
    """В файле должна остаться цель: иначе пользователь услышит чужой текст."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context,
        "locate_target_start",
        lambda *a, **k: _warmup_boundary(0.5),
    )

    result = _warmup_render(service, monkeypatch)

    assert len(stub.calls) == 1
    full_sec = 0.4 + 0.02 * len(stub.calls[0]["text"])
    # Срезано ровно до границы с запасом `WARMUP_PREROLL_MS`, а не «примерно».
    assert result.duration_sec == pytest.approx(
        full_sec - 0.5 + config.WARMUP_PREROLL_MS / 1000, abs=0.05
    )
    assert result.duration_sec < full_sec


def test_warmup_metadata_reports_boundary(stub, fake_store, monkeypatch):
    """Диагностике нужна фактическая граница: без неё непонятно, как резалось аудио."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context,
        "locate_target_start",
        lambda *a, **k: _warmup_boundary(0.5),
    )

    result = _warmup_render(service, monkeypatch)

    assert result.warmups[0]["warmup_boundary_sec"] == pytest.approx(0.5)


def test_warmup_alignment_failure_resynthesizes_target_only(stub, fake_store, monkeypatch):
    """Ненайденная граница — не «оставить как есть», а честный повтор только цели."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: None
    )

    _warmup_render(service, monkeypatch)

    assert len(stub.calls) == 2
    assert WARMUP_PREFIX in stub.calls[0]["text"]
    assert stub.calls[1]["text"] == WARMUP_TARGET


def test_warmup_metadata_reports_alignment_fallback(stub, fake_store, monkeypatch):
    """Откат обязан быть виден в метаданных: иначе деградация прогрева незаметна."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: None
    )

    result = _warmup_render(service, monkeypatch)

    assert result.warmups[0]["warmup_fallback"] is True


def test_qa_measures_target_without_warmup_prefix(stub, fake_store, monkeypatch):
    """Проверка сравнивает цель с целью: префикс в ожидаемом тексте уронил бы WER."""
    service = _FakeWarmupService()
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: _warmup_boundary()
    )
    measured: list[str] = []

    async def fake_measure(chunk: np.ndarray, expected: str) -> tuple[str, float]:
        measured.append(expected)
        return WARMUP_TARGET, 0.0

    monkeypatch.setattr(audio_pipeline, "_transcribe_and_measure", fake_measure)

    result = _warmup_render(
        service, monkeypatch, qa=audio_pipeline.QaSettings.for_mode("strict")
    )

    assert measured == [WARMUP_TARGET]
    assert all(WARMUP_PREFIX not in expected for expected in measured)
    assert result.qa[0].status == audio_pipeline.QA_PASSED


def test_warmup_off_keeps_plain_synthesis(stub, fake_store, monkeypatch):
    """Выключенный прогрев не должен ни звать сервис, ни менять текст для движка."""
    service = _FakeWarmupService()
    monkeypatch.setattr(audio_pipeline.warmup_context, "get_service", lambda: service)
    monkeypatch.setattr(config, "WARMUP_ENABLED", True)

    # Явный `warmup=False` сильнее включённой политики приложения.
    result = _render(
        [_replica(WARMUP_TARGET)],
        {"#1": _speaker()},
        RenderSettings(pause_ms=0, output_format="wav", warmup=False),
    )
    assert len(stub.calls) == 1
    assert stub.calls[0]["text"] == WARMUP_TARGET
    assert result.warmups == {}
    assert service.calls == []

    # Политика приложения выключена, `warmup=None` — поведение прежнее.
    stub.calls.clear()
    monkeypatch.setattr(config, "WARMUP_ENABLED", False)
    result = _render(
        [_replica(WARMUP_TARGET)],
        {"#1": _speaker()},
        RenderSettings(pause_ms=0, output_format="wav"),
    )
    assert len(stub.calls) == 1
    assert stub.calls[0]["text"] == WARMUP_TARGET
    assert result.warmups == {}
    assert service.calls == []


def test_warmup_prefix_passes_accentizer_on_f5(accent_stub, fake_store, monkeypatch):
    """F5 понимает «+»: префикс обязан пройти accentizer так же, как цель."""
    service = _FakeWarmupService(prefix="Тише начнём")
    monkeypatch.setattr(audio_pipeline.warmup_context, "get_service", lambda: service)
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: _warmup_boundary()
    )
    monkeypatch.setattr(audio_pipeline, "accentuate", lambda text: f"[{text}]")

    _warmup_render(service, monkeypatch)

    sent = accent_stub.calls[0]["text"]
    assert "[Тише начнём]" in sent
    assert sent.endswith(f"[{WARMUP_TARGET}]")


def test_warmup_prefix_skips_accentizer_on_plain_engine(stub, fake_store, monkeypatch):
    """Движок без ударений не должен получить «+»-разметку: он прочитал бы её вслух."""
    service = _FakeWarmupService(prefix="Тише начнём")
    monkeypatch.setattr(audio_pipeline.warmup_context, "get_service", lambda: service)
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: _warmup_boundary()
    )
    monkeypatch.setattr(audio_pipeline, "accentuate", lambda text: f"[{text}]")

    _warmup_render(service, monkeypatch)

    sent = stub.calls[0]["text"]
    assert "[" not in sent
    assert sent.startswith("Тише начнём")
    assert sent.endswith(WARMUP_TARGET)


def test_warmup_takes_priority_over_short_layer(stub, fake_store, monkeypatch):
    """Два слоя не складываются: в движок идёт прогрев, а не контекст короткой реплики."""
    service = _FakeWarmupService()
    monkeypatch.setattr(audio_pipeline.warmup_context, "get_service", lambda: service)
    monkeypatch.setattr(
        audio_pipeline.warmup_context, "locate_target_start", lambda *a, **k: _warmup_boundary()
    )
    settings = RenderSettings(
        pause_ms=0,
        output_format="wav",
        warmup=True,
        short_utterance=audio_pipeline.ShortUtteranceSettings(enabled=True),
    )

    _render([_replica(WARMUP_TARGET)], {"#1": _speaker()}, settings)

    # Даже если короткий слой построил свой план, первый вход модели — префикс прогрева.
    assert stub.calls[0]["text"] == f"{WARMUP_PREFIX} {WARMUP_TARGET}"
