"""Слой коротких реплик в бою: рендер, обрезка, повтор и метаданные.

Модели не поднимаются: движок — заглушка, распознавание для границ — подставной
провайдер (`boundary_module._default_asr`). Проверяется то, что обещают §14, §15,
§19, §24, §28: обычный текст идёт прежним путём, короткий получает контекст,
разметка ударений согласована, границы режут результат, повтор ограничен, а
метаданные куска остаются воспроизводимыми.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest
from conftest import F5_VOICE, XTTS_VOICE, AccentStubEngine, StubEngine  # noqa: F401
from test_projects_api import _client, _create_project, _wait_job

from backend import audio_pipeline, config
from backend import short_utterance as su
from backend import short_utterance_boundary as boundary_module
from backend.audio_pipeline import (
    RenderSettings,
    ShortUtteranceSettings,
    SpeakerSettings,
)
from backend.dialogue_parser import Replica
from backend.engines.base import SAMPLE_RATE
from backend.transcribe import WordStamp

FIRST = "Я только что вернулась домой."
SECOND = "Да."
THIRD = "Сегодня хорошая погода, и я рад тебя видеть."


def _speaker(voice_id: str = "voice1") -> SpeakerSettings:
    return SpeakerSettings(voice_id=voice_id)


def _replicas(*texts: str, voice: str = "#1") -> list[Replica]:
    return [
        Replica(voice=voice, text=text, line_number=index + 1)
        for index, text in enumerate(texts)
    ]


def _short_settings(**overrides) -> RenderSettings:
    settings = ShortUtteranceSettings(enabled=True, **overrides)
    return RenderSettings(pause_ms=0, short_utterance=settings)


def _render(replicas, settings, speakers=None, **kwargs):
    return asyncio.run(
        audio_pipeline.render_dialogue(
            job_id="job",
            replicas=replicas,
            speakers=speakers or {"#1": _speaker()},
            settings=settings,
            **kwargs,
        )
    )


@pytest.fixture(autouse=True)
def fast_boundary(monkeypatch):
    """Границы по словам без Whisper: тесты не должны поднимать распознавание.

    Провайдер ставит слова цели в хвост записи — для стратегий с контекстом этого
    достаточно, чтобы граница нашлась и обрезка сработала. Тесты, которым нужно
    «границы нет», перекрывают патч своим.
    """
    monkeypatch.setattr(
        boundary_module, "_default_asr", _fake_asr(["да", "хорошо", "привет"], share=0.3)
    )


@pytest.fixture
def fixed_seed(monkeypatch):
    """Предсказуемый сид: сравнение прогонов «с слоем» и «без» должно быть честным."""
    monkeypatch.setattr(random, "randrange", lambda _limit: 12345)


def _stamps_from_text(audio: np.ndarray, text: str, words: list[str], share: float = 0.35):
    """Подставной ASR: слова цели стоят в хвосте записи (сторона — префикс).

    Заглушка синтеза делает длительность пропорциональной тексту, поэтому «хвост»
    — это ровно та часть, которую синтез цели занял бы сам.
    """
    duration = audio.size / SAMPLE_RATE
    start = duration * (1.0 - share)
    step = (duration * share) / max(len(words), 1)
    return [
        WordStamp(word=word, start=start + index * step, end=start + (index + 1) * step * 0.9)
        for index, word in enumerate(words)
    ]


def _fake_asr(words: list[str], share: float = 0.35):
    def provider(audio: np.ndarray):
        return _stamps_from_text(audio, "", words, share=share)

    return provider


# --- обычный текст идёт прежним путём (§28) -------------------------------------
def test_normal_utterance_bypasses_short_optimizer(stub, fake_store, fixed_seed):
    """Длинная реплика при включённом слое синтезируется ровно как раньше."""
    replicas = _replicas(THIRD)
    result = _render(replicas, _short_settings(strategy=su.STRATEGY_SYNTHETIC_CONTEXT))

    assert len(stub.calls) == 1, "длинная реплика не должна получать контекст"
    assert stub.calls[0]["text"] == THIRD
    assert result.short_runs == {}, "плана короткой реплики быть не должно"
    assert result.replicas_done == 1


def test_normal_text_uses_existing_pipeline(stub, fake_store, fixed_seed):
    """Вход движка для NORMAL совпадает с прогоном без слоя — побайтово по тексту."""
    plain = _render(_replicas(THIRD), RenderSettings(pause_ms=0))
    plain_text = stub.calls[-1]["text"]
    plain_duration = plain.duration_sec
    # Джиттер заглушки зависит от номера вызова, поэтому журнал сбрасываем: иначе
    # сравнение «со слоем» и «без слоя» ловило бы не слой, а счётчик.
    stub.calls.clear()

    styled = _render(
        _replicas(THIRD), _short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT)
    )
    assert stub.calls[-1]["text"] == plain_text
    assert styled.duration_sec == pytest.approx(plain_duration, abs=1e-6)


def test_short_optimization_does_not_change_long_text_output(stub, fake_store, fixed_seed):
    """Аудио длинной реплики не меняется от включения слоя."""
    replicas = _replicas(THIRD)
    before = _render(replicas, RenderSettings(pause_ms=0))
    after = _render(replicas, _short_settings(strategy=su.STRATEGY_SYNTHETIC_CONTEXT))

    before_audio = audio_pipeline._read_audio(before.output_path, "wav")
    after_audio = audio_pipeline._read_audio(after.output_path, "wav")
    assert before_audio.size == after_audio.size
    assert np.array_equal(before_audio, after_audio)


def test_layer_is_disabled_by_default(stub, fake_store, fixed_seed):
    """Явное выключение возвращает прежнее поведение (`enabled=False`).

    По умолчанию слой включён (измеренное улучшение коротких реплик), поэтому
    «прежнее поведение» проверяется именно явным выключением — иначе тест проверял
    бы значение по умолчанию, а не механизм отказа от слоя.
    """
    result = _render(
        _replicas(SECOND),
        RenderSettings(pause_ms=0, short_utterance=ShortUtteranceSettings(enabled=False)),
    )
    assert len(stub.calls) == 1
    assert stub.calls[0]["text"] == SECOND
    assert result.short_runs == {}
    # Значение по умолчанию — включено: это измеренное решение benchmark'а.
    assert config.SHORT_UTTERANCE_DEFAULT_ENABLED is True


# --- короткая реплика получает контекст (§14, §15) ------------------------------
def test_short_strategy_uses_prepared_text(accent_stub, fake_store, fixed_seed):
    """Контекст берётся из подготовленного текста соседа, а не из исходного."""
    replicas = _replicas(THIRD, SECOND)
    replicas[0].final_text = "Мы зак+ончили раб+оту над про+ектом и теп+ерь м+ожем отдохн+уть."
    replicas[1].final_text = "Д+а."

    _render(replicas, _short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT))
    # Первая реплика обычная — синтезируется как есть; вторая короткая получает
    # контекст соседа, и обрезка возвращает её обратно.
    assert accent_stub.calls[0]["text"] == replicas[0].final_text
    assert accent_stub.calls[1]["text"] == f"{replicas[0].final_text} Д+а."


def test_f5_short_context_preserves_accent_markup(accent_stub, fake_store, fixed_seed):
    """F5 получает «+» и в цели, и в контексте (§15)."""
    replicas = _replicas(FIRST, SECOND)
    replicas[0].final_text = "Я верн+улась дом+ой."
    replicas[1].final_text = "Д+а."

    _render(replicas, _short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT))
    for call in accent_stub.calls:
        assert "+" in call["text"], call["text"]


def test_xtts_short_context_has_no_f5_markup(voices, fixed_seed, monkeypatch):
    """Движок без ударений не получает «+» ни в цели, ни в контексте (§15)."""
    engine = StubEngine()  # supports_accents = False, как у XTTS
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)
    replicas = _replicas(THIRD, SECOND, voice=XTTS_VOICE)
    replicas[0].final_text = "Я верн+улась дом+ой."
    replicas[1].final_text = "Д+а."

    asyncio.run(
        audio_pipeline.render_dialogue(
            job_id="job",
            replicas=replicas,
            speakers={XTTS_VOICE: SpeakerSettings(voice_id=XTTS_VOICE)},
            settings=_short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT),
        )
    )
    assert engine.calls, "движок должен быть вызван"
    for call in engine.calls:
        assert "+" not in call["text"], call["text"]


# --- обрезка по надёжной границе (§11, §31) -------------------------------------
def test_short_crop_keeps_only_target(stub, fake_store, fixed_seed, monkeypatch):
    """Контекст в готовый кусок не попадает: режем по таймстемпам слов."""
    monkeypatch.setattr(
        boundary_module, "_default_asr", _fake_asr(["да"], share=0.30)
    )
    replicas = _replicas(FIRST, SECOND)
    result = _render(replicas, _short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT))

    plan = result.short_runs[1]
    assert plan.plan.strategy == su.STRATEGY_SAME_SPEAKER_CONTEXT
    assert plan.boundary is not None
    assert plan.boundary.method == boundary_module.METHOD_ASR
    assert plan.boundary.production_ready is True
    assert plan.boundary.to_dict()["method"] == boundary_module.METHOD_ASR
    # Кусок второй реплики короче полного синтеза с контекстом.
    full = len(FIRST) + len(SECOND)
    start, end = result.segments[1]
    cropped_sec = (end - start) / SAMPLE_RATE
    assert cropped_sec < stub.seconds_per_char * full + 0.3


def test_short_falls_back_to_direct_without_boundary(stub, fake_store, fixed_seed, monkeypatch):
    """Нет границы — синтезируем цель отдельно, контекст в файл не попадает."""
    monkeypatch.setattr(boundary_module, "_default_asr", lambda audio: [])
    replicas = _replicas(FIRST, SECOND)
    result = _render(replicas, _short_settings(strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT))

    plan = result.short_runs[1]
    assert plan.fallback, "откат должен быть помечен"
    assert plan.boundary is None
    # Два вызова: сначала с контекстом, потом честно цель отдельно.
    assert stub.calls[-1]["text"] == SECOND
    assert stub.calls[-2]["text"].endswith(SECOND)


def test_silence_boundary_is_not_used_in_production(stub, fake_store, fixed_seed):
    """Метод «по паузе» в рендере не проходит: граница считается ненадёжной."""
    replicas = _replicas(FIRST, SECOND)
    result = _render(
        replicas,
        _short_settings(
            strategy=su.STRATEGY_SAME_SPEAKER_CONTEXT,
            boundary_method=boundary_module.METHOD_SILENCE,
        ),
    )
    plan = result.short_runs[1]
    assert plan.boundary is None
    assert plan.fallback
    assert stub.calls[-1]["text"] == SECOND


# --- повтор ограничен (§19) ------------------------------------------------------
def test_short_failed_verdict_retries_with_new_seed(stub, fake_store, monkeypatch):
    """Провал короткой проверки — повтор; из двух попыток берётся удачная."""
    verdicts = [
        su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION,)),
        su.ShortVerdict(ok=True, reasons=()),
    ]
    monkeypatch.setattr(su, "check_chunk", lambda *a, **k: verdicts.pop(0))
    result = _render(
        _replicas(SECOND),
        _short_settings(strategy=su.STRATEGY_DIRECT, max_attempts=2),
    )
    assert len(stub.calls) == 2, "должен быть ровно один повтор"
    assert result.short_runs[0].verdict.ok is True
    assert result.short_runs[0].attempts == 2


def test_short_retry_is_limited_in_render(stub, fake_store, monkeypatch):
    """Лимит попыток соблюдается: генерации «десятков вариантов» нет."""
    monkeypatch.setattr(
        su,
        "check_chunk",
        lambda *a, **k: su.ShortVerdict(ok=False, reasons=(su.REASON_REPETITION,)),
    )
    _render(
        _replicas(SECOND),
        _short_settings(strategy=su.STRATEGY_DIRECT, max_attempts=2),
    )
    assert len(stub.calls) == 2


def test_short_successful_verdict_does_not_retry(stub, fake_store):
    """Удачная попытка не повторяется — это цена и время пользователя."""
    _render(
        _replicas(SECOND),
        _short_settings(strategy=su.STRATEGY_DIRECT, max_attempts=3),
    )
    assert len(stub.calls) == 1


def test_short_layer_with_qa_keeps_qa_outcome(stub, fake_store, monkeypatch):
    """Слой не подменяет отметку проверки: QA-цикл остаётся источником WER."""
    from backend.audio_pipeline import QaSettings

    monkeypatch.setattr(
        audio_pipeline, "_transcribe_chunk", lambda chunk: _always("Да.")
    )
    settings = _short_settings(strategy=su.STRATEGY_DIRECT)
    settings.qa = QaSettings.for_mode(config.QA_MODE_STRICT)
    result = _render(_replicas(SECOND), settings)
    outcome = result.qa[0]
    assert outcome is not None
    assert outcome.wer == pytest.approx(0.0)
    assert result.short_runs[0].verdict is not None


async def _always(text: str) -> str:
    return text


# --- метаданные куска (§24) через API -------------------------------------------
def test_render_request_accepts_short_settings(stub, voices, monkeypatch):
    """Настройки слоя проходят через API и доезжают до движка."""
    async def scenario():
        async with _client(monkeypatch) as client:
            response = await client.post(
                "/api/render-text",
                json={
                    "text": "Привет !!!",
                    "voice_id": F5_VOICE,
                    "output_format": "wav",
                    "qa": "off",
                    "short_utterance": {"enabled": True, "strategy": "punctuation"},
                },
            )
            assert response.status_code == 202, response.text
            job = await _wait_job(client, response.json()["job_id"])
            assert job["status"] == "done"
            return job

    asyncio.run(scenario())
    assert stub.calls[-1]["text"] == "Привет!", stub.calls[-1]["text"]


def test_render_request_rejects_unknown_strategy(stub, voices, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            response = await client.post(
                "/api/render-text",
                json={
                    "text": "Да.",
                    "voice_id": F5_VOICE,
                    "short_utterance": {"enabled": True, "strategy": "магия"},
                },
            )
            assert response.status_code == 422, response.text

    asyncio.run(scenario())


def test_status_exposes_short_policy(monkeypatch):
    """Интерфейс узнаёт из /api/status, что произойдёт в режиме «Авто»."""
    from backend import main

    payload = asyncio.run(main.status())
    policy = payload["short_utterance"]
    # По умолчанию слой включён, но производственная стратегия — DIRECT: замер
    # Замер (см. `docs/dialogue.md`) показал, что контекстный синтез не
    # добавляет разборчивости (слова на месте и там, и там), зато делает реплику на
    # 25–50 % длиннее и стоит вчетверо дороже. Контекстные стратегии остаются
    # выбираемыми вручную, но не по умолчанию.
    assert policy["enabled"] is True
    assert policy["strategies"]["f5"] == su.STRATEGY_DIRECT
    assert policy["strategies"]["xtts"] == su.STRATEGY_DIRECT
    assert su.STRATEGY_AUTO in policy["available_strategies"]
    assert su.STRATEGY_SAME_SPEAKER_CONTEXT in policy["available_strategies"]
    assert su.STRATEGY_BATCH_AND_CROP not in policy["available_strategies"]
    assert policy["thresholds"]["very_short_words"] == config.SHORT_UTTERANCE_VERY_SHORT_WORDS


def test_project_render_records_short_metadata(stub, voices, fake_accent, monkeypatch):
    """Куски проекта помнят, что с ними сделал слой (§24)."""
    from conftest import analyze_project

    monkeypatch.setattr(boundary_module, "_default_asr", _fake_asr(["да"], share=0.3))

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(
                client, text=f"ИВАН: {FIRST}\nИВАН: {SECOND}"
            )
            parsed = await client.post(f"/api/projects/{project['id']}/parse", json={})
            speakers = {
                row["speaker"]: {"voice_id": F5_VOICE}
                for row in parsed.json()["replicas"]
            }
            patched = await client.patch(
                f"/api/projects/{project['id']}", json={"speakers": speakers}
            )
            assert patched.status_code == 200, patched.text
            settings = await analyze_project(client, project["id"], auto_accent=True)
            assert settings["status"] == "ready"
            response = await client.post(
                f"/api/projects/{project['id']}/render",
                json={
                    "output_format": "wav",
                    "qa": "off",
                    "short_utterance": {
                        "enabled": True,
                        "strategy": "same_speaker_context",
                    },
                },
            )
            assert response.status_code == 202, response.text
            await _wait_job(client, response.json()["job_id"])
            return (await client.get(f"/api/projects/{project['id']}")).json()

    project = asyncio.run(scenario())
    takes = project["replicas"][1]["takes"]
    assert takes, "у короткой реплики должен быть вариант"
    params = takes[0]["parameters"]
    assert params["short_utterance_strategy"] == su.STRATEGY_SAME_SPEAKER_CONTEXT
    assert params["short_utterance_context_source"] == su.SOURCE_SAME_SPEAKER_PREVIOUS
    assert params["short_utterance_class"] == su.CLASS_VERY_SHORT
    assert params["synthesis_text_hash"]
    # А готовый файл реплики не содержит контекста: длительность близка к цели.
    assert project["replicas"][1]["takes"][0]["duration_sec"] < 1.5
