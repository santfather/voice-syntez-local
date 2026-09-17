"""Smart QA: дешёвый отбор подозрительных кусков и режимы проверки.

Проверки идут на синтетических waveform: отбор смотрит только на само аудио, и
поднимать ради него модель незачем. Часть тестов — интеграционные, на движке-
заглушке: важно не только то, что отбор различает куски, но и что в Smart-режиме
нормальный кусок действительно не доходит до Whisper, а в Strict-режиме доходит
каждый.
"""

import asyncio

import numpy as np
import pytest
from conftest import analyze_project, sine
from test_api import _client, _generate, _wait
from test_audio_pipeline import _answers, _qa_render, _render, _replica, _speaker
from test_projects_api import _client as _projects_client
from test_projects_api import _create_project, _wait_job

from backend import audio_pipeline, config, main, qa_screening
from backend.audio_pipeline import QaSettings, RenderSettings
from backend.engines.base import SAMPLE_RATE

# Текст длиной 42 знака: ожидаемая длительность по нему — около 2.8 c.
TEXT = "Сегодня хорошая погода, и я рад тебя видеть"


def _speech(seconds: float, amplitude: float = 0.3, seed: int = 1) -> np.ndarray:
    """Речеподобный кусок: шум с гуляющей громкостью, а не ровный тон.

    Громкость меняется специально: у ровного тона отбор не ищет повторы (затянутая
    гласная коррелирует с собой при сдвиге, кратного периоду), и такой кусок
    проверял бы не то.
    """
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * np.arange(n) / SAMPLE_RATE)
    return (amplitude * rng.standard_normal(n) * envelope).astype(np.float32)


def _smart_render(**overrides) -> RenderSettings:
    """Сборка в режиме Smart: отбор включён, числа задаёт сам тест."""
    return RenderSettings(
        pause_ms=0,
        output_format="wav",
        qa=QaSettings(mode=config.QA_MODE_SMART, **overrides),
    )


def _force_screening(monkeypatch, *, suspicious: bool, reasons: list[str] | None = None) -> None:
    """Задаёт вердикт отбора: waveform в тестах не обязан быть подозрительным."""
    verdict = qa_screening.Screening(
        suspicious=suspicious,
        reasons=list(reasons or (["silence"] if suspicious else [])),
    )
    monkeypatch.setattr(qa_screening, "screen_chunk", lambda chunk, text: verdict)


# --- отбор по waveform ---------------------------------------------------------
def test_normal_audio_is_not_suspicious():
    verdict = qa_screening.screen_chunk(_speech(2.8), TEXT)
    assert verdict.suspicious is False
    assert verdict.reasons == []


def test_silence_is_flagged_with_reason():
    verdict = qa_screening.screen_chunk(np.zeros(int(SAMPLE_RATE * 2.8), dtype=np.float32), TEXT)
    assert verdict.suspicious is True
    assert qa_screening.REASON_SILENCE in verdict.reasons


def test_clipping_is_detected():
    clipped = np.clip(_speech(2.8) * 8.0, -1.0, 1.0).astype(np.float32)
    verdict = qa_screening.screen_chunk(clipped, TEXT)
    assert qa_screening.REASON_CLIPPING in verdict.reasons


def test_abnormal_duration_is_detected_in_both_directions():
    short = qa_screening.screen_chunk(_speech(0.5), TEXT)
    assert qa_screening.REASON_DURATION_SHORT in short.reasons
    # Совсем короткий кусок — отдельный признак: он не «короче текста», а сорвался.
    tiny = qa_screening.screen_chunk(_speech(0.05), TEXT)
    assert qa_screening.REASON_TOO_SHORT in tiny.reasons
    long = qa_screening.screen_chunk(_speech(12.0), TEXT)
    assert qa_screening.REASON_DURATION_LONG in long.reasons


def test_abnormal_level_is_detected():
    quiet = qa_screening.screen_chunk(_speech(2.8, amplitude=0.002), TEXT)
    assert qa_screening.REASON_LEVEL in quiet.reasons


def test_empty_audio_is_detected():
    verdict = qa_screening.screen_chunk(np.zeros(0, dtype=np.float32), TEXT)
    assert verdict.suspicious is True
    # Измерять в пустом waveform нечего: вердикт выносится сразу и без других причин.
    assert verdict.reasons == [qa_screening.REASON_EMPTY]


def test_repeated_section_is_detected():
    half = _speech(1.5)
    verdict = qa_screening.screen_chunk(np.concatenate([half, half]), TEXT)
    assert qa_screening.REASON_REPEAT in verdict.reasons


def test_steady_tone_is_not_taken_for_a_repeat():
    """Ровный тон коррелирует с собой при сдвиге, кратного периоду, — это не повтор."""
    verdict = qa_screening.screen_chunk(sine(2.8, 120.0), TEXT)
    assert qa_screening.REASON_REPEAT not in verdict.reasons


# --- режимы в пайплайне --------------------------------------------------------
def test_smart_does_not_transcribe_a_normal_chunk(stub, fake_store, monkeypatch):
    transcribed = _answers(monkeypatch, "Привет, это тест")
    result = _render([_replica()], {"#1": _speaker()}, _smart_render())

    assert transcribed == []  # до Whisper кусок не дошёл
    assert len(stub.calls) == 1
    outcome = result.qa[0]
    assert outcome.status == audio_pipeline.QA_PASSED
    assert outcome.wer is None  # расшифровки не было, и это видно
    assert outcome.mode == config.QA_MODE_SMART
    assert outcome.screening.suspicious is False


def test_smart_sends_a_suspicious_chunk_to_whisper(stub, fake_store, monkeypatch):
    # Заглушка отдаёт 0.4 c на любую реплику: для длинного текста это заведомо
    # короче ожидаемого, и отбор обязан отправить кусок в расшифровку.
    stub.seconds_per_char = 0.0
    transcribed = _answers(monkeypatch, TEXT)
    result = _render([_replica(TEXT)], {"#1": _speaker()}, _smart_render())

    assert len(transcribed) == 1
    outcome = result.qa[0]
    assert outcome.status == audio_pipeline.QA_PASSED
    assert outcome.screening.suspicious is True
    assert qa_screening.REASON_DURATION_SHORT in outcome.screening.reasons


def test_smart_retries_within_attempt_limit(stub, fake_store, monkeypatch):
    _force_screening(monkeypatch, suspicious=True)
    transcribed = _answers(monkeypatch, "совсем другой текст")  # порог не берётся никогда
    result = _render([_replica()], {"#1": _speaker()}, _smart_render(max_attempts=3))

    assert len(stub.calls) == 3
    assert len(transcribed) == 3
    outcome = result.qa[0]
    assert outcome.status == audio_pipeline.QA_ATTEMPTS
    assert outcome.attempts == 3


def test_smart_stops_on_budget_before_retry(stub, fake_store, monkeypatch):
    _force_screening(monkeypatch, suspicious=True)
    _answers(monkeypatch, "совсем другой текст")
    result = _render([_replica()], {"#1": _speaker()}, _smart_render(budget_sec=0.0))

    assert len(stub.calls) == 1  # первая попытка выполняется всегда, вторая уже нет
    assert result.qa[0].status == audio_pipeline.QA_BUDGET
    assert result.qa[0].attempts == 1


def test_smart_keeps_best_take_when_qa_exhausted(stub, fake_store, monkeypatch):
    _force_screening(monkeypatch, suspicious=True)
    chunks = _answers(monkeypatch, "а б в", "и это тест", "совсем другой текст")
    result = _render([_replica()], {"#1": _speaker()}, _smart_render(max_attempts=3))

    outcome = result.qa[0]
    assert outcome.status == audio_pipeline.QA_ATTEMPTS
    assert outcome.wer == pytest.approx(1 / 3)  # ближе всех вторая попытка
    # В файл попала именно она, а не последняя: длина совпадает со вторым куском.
    assert result.segments[0][1] - result.segments[0][0] == chunks[1].size


def test_strict_transcribes_every_chunk(stub, fake_store, monkeypatch):
    transcribed = _answers(monkeypatch, "Привет, это тест")
    replicas = [_replica(), _replica()]
    result = _render(replicas, {"#1": _speaker()}, _qa_render())

    assert len(transcribed) == 2
    assert [outcome.mode for outcome in result.qa] == [config.QA_MODE_STRICT] * 2
    # Отбора в строгом режиме нет — и в метаданных это видно.
    assert [outcome.screening for outcome in result.qa] == [None, None]


def test_off_mode_never_runs_qa(stub, fake_store, monkeypatch):
    transcribed = _answers(monkeypatch, "Привет, это тест")

    plain = _render(
        [_replica()], {"#1": _speaker()}, RenderSettings(pause_ms=0, output_format="wav")
    )
    explicit = _render(
        [_replica()],
        {"#1": _speaker()},
        RenderSettings(
            pause_ms=0, output_format="wav", qa=QaSettings(mode=config.QA_MODE_OFF)
        ),
    )

    assert transcribed == []
    assert plain.qa == [None]
    assert explicit.qa == [None]
    assert QaSettings.for_mode(config.QA_MODE_OFF) is None


def test_smart_uses_fewer_transcriptions_than_strict_on_a_clean_dialogue(
    stub, fake_store, monkeypatch
):
    """Критерий фазы: на чистом диалоге Smart почти не поднимает Whisper."""
    transcribed = _answers(monkeypatch, "Привет, это тест")
    replicas = [_replica(), _replica(), _replica()]

    smart = _render(replicas, {"#1": _speaker()}, _smart_render())
    after_smart = len(transcribed)
    strict = _render(replicas, {"#1": _speaker()}, _qa_render())

    assert after_smart == 0  # Smart обошёлся отбором
    assert len(transcribed) - after_smart == 3  # Strict расшифровал каждый кусок
    assert all(outcome.status == audio_pipeline.QA_PASSED for outcome in smart.qa)
    assert all(outcome.status == audio_pipeline.QA_PASSED for outcome in strict.qa)


# --- метаданные ---------------------------------------------------------------
def test_smart_qa_metadata_is_saved_with_take(stub, fake_store, monkeypatch):
    transcribed = _answers(monkeypatch, "Привет, это тест")

    async def scenario():
        async with _projects_client(monkeypatch) as client:
            project = await _create_project(client)
            await client.patch(
                f"/api/projects/{project['id']}",
                json={
                    "speakers": {
                        "ИВАН": {"voice_id": "voice1"},
                        "МАРГО": {"voice_id": "voice1"},
                    }
                },
            )
            await client.post(f"/api/projects/{project['id']}/parse", json={})
            await analyze_project(client, project["id"])
            accepted = await client.post(
                f"/api/projects/{project['id']}/render",
                json={"output_format": "wav", "qa": "smart"},
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])

            reopened = (await client.get(f"/api/projects/{project['id']}")).json()
            take = reopened["replicas"][0]["takes"][0]
            assert take["qa"]["mode"] == config.QA_MODE_SMART
            assert take["qa"]["screening"] == {"suspicious": False, "reasons": []}
            assert take["qa"]["status"] == audio_pipeline.QA_PASSED

    _run(scenario)
    assert transcribed == []


def test_project_render_keeps_saved_qa_mode_when_field_omitted():
    """Рендер без тела запроса не должен сбрасывать режим, выбранный в прошлый раз."""
    smart, remembered = main._resolved_render_settings(
        {"render_settings": {"qa": config.QA_MODE_SMART}}, main.ProjectRenderRequest()
    )
    assert smart.qa is not None
    assert smart.qa.mode == config.QA_MODE_SMART
    assert remembered["qa"] == config.QA_MODE_SMART

    # Проекты, собранные до появления Smart, хранят булево: true — строгая проверка.
    legacy, _ = main._resolved_render_settings(
        {"render_settings": {"qa": True}}, main.ProjectRenderRequest()
    )
    assert legacy.qa.mode == config.QA_MODE_STRICT


def test_qa_mode_accepts_legacy_boolean_and_rejects_unknown(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            broken = await client.post(
                "/api/generate", json={"dialogue_text": "ИВАН: раз", "qa": "fast"}
            )
            assert broken.status_code == 422

            _answers(monkeypatch, "Первая реплика.", "Вторая реплика.")
            # Старый клиент присылает булево: true — это прежняя строгая проверка.
            legacy = await _generate(client, qa=True)
            data = await _wait(client, legacy, lambda item: item["status"] == "done")
            assert data["replicas"][0]["qa"]["mode"] == config.QA_MODE_STRICT
            assert data["replicas"][0]["qa"]["screening"] is None

            off = await _generate(client, qa="off")
            data = await _wait(client, off, lambda item: item["status"] == "done")
            assert data["replicas"][0]["qa"] is None

    _run(scenario)


def _run(scenario) -> None:
    asyncio.run(scenario())
