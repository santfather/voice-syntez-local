"""Диагностические метрики take'а: измерения по waveform и их доезд до API.

Проверки идут на синтетических кусках: метрики считаются по самому аудио, и
поднимать ради них модель незачем. Интеграционная часть — на движке-заглушке:
важно не только то, что числа считаются, но и что они доезжают до take'ов
проекта, до вариантов задачи и что старая запись без метрик не ломает API.
"""

import asyncio
import json
import math
import time

import numpy as np
import pytest
from conftest import analyze_project, sine
from test_projects_api import _client, _create_project, _wait_job

from backend import audio_analysis, config, qa_screening, take_quality
from backend.audio_pipeline import QaOutcome
from backend.db.store import get_projects_store
from backend.engines.base import SAMPLE_RATE

# Текст длиной 43 знака: ожидаемая длительность по нему — около 2.9 c.
TEXT = "Сегодня хорошая погода, и я рад тебя видеть"


def _speech(seconds: float, amplitude: float = 0.3, seed: int = 1) -> np.ndarray:
    """Речеподобный кусок: шум с гуляющей громкостью, а не ровный тон."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * np.arange(n) / SAMPLE_RATE)
    return (amplitude * rng.standard_normal(n) * envelope).astype(np.float32)


def _clipped(seconds: float = 2.8) -> np.ndarray:
    return np.clip(_speech(seconds) * 8.0, -1.0, 1.0).astype(np.float32)


def _measure(chunk: np.ndarray, text: str = TEXT, qa: QaOutcome | None = None):
    return take_quality.measure(chunk, text, qa=qa)


def _codes(quality: take_quality.TakeQuality) -> list[str]:
    return [item["code"] for item in quality.warnings]


def _text_of(quality: take_quality.TakeQuality, code: str) -> str:
    return next(item["text"] for item in quality.warnings if item["code"] == code)


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _rendered_project(client) -> tuple[dict, str]:
    """Проект из двух реплик, доведённый до готового рендера."""
    project = await _create_project(client)
    await client.patch(
        f"/api/projects/{project['id']}",
        json={"speakers": {"ИВАН": {"voice_id": "voice1"}, "МАРГО": {"voice_id": "voice1"}}},
    )
    await client.post(f"/api/projects/{project['id']}/parse", json={})
    await analyze_project(client, project["id"])
    accepted = await client.post(
        f"/api/projects/{project['id']}/render", json={"output_format": "wav"}
    )
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]
    await _wait_job(client, job_id)
    return project, job_id


# --- измерения по waveform -----------------------------------------------------
def test_clipping_is_detected_with_ratio_and_warning():
    quality = _measure(_clipped())
    assert quality.clipping is True
    assert quality.clipping_ratio >= audio_analysis.CLIPPING_MIN_RATIO
    assert "clipping" in _codes(quality)
    assert _text_of(quality, "clipping") == "Обнаружен клиппинг"


def test_normal_signal_is_not_clipped():
    quality = _measure(_speech(2.8))
    assert quality.clipping is False
    assert quality.clipping_ratio < audio_analysis.CLIPPING_MIN_RATIO


def test_silence_is_measured_and_warned():
    quality = _measure(np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32))
    assert quality.silence_ratio == 1.0
    assert _text_of(quality, "silence") == "Высокая доля тишины"
    # Тишина — это ещё и уровень ниже пола, а не «нормальная громкость».
    assert _text_of(quality, "level") == "Низкий уровень"


def test_normal_signal_has_no_warnings():
    quality = _measure(_speech(2.8))
    assert quality.warnings == []
    assert quality.screening_reasons == []
    assert quality.silence_ratio is not None and quality.silence_ratio < 0.1
    assert quality.peak_dbfs is not None
    assert quality.rms_dbfs is not None
    assert quality.lufs is not None


def test_abnormal_short_duration_is_detected():
    short = _measure(_speech(0.5))
    assert "duration_short" in _codes(short)
    assert _text_of(short, "duration_short") == "Реплика короче своего текста"
    # Совсем короткий кусок — отдельный признак: он не «короче текста», а сорвался.
    assert "too_short" in _codes(_measure(_speech(0.05)))


def test_abnormal_long_duration_is_detected():
    quality = _measure(_speech(12.0))
    assert "duration_long" in _codes(quality)
    assert _text_of(quality, "duration_long") == "Необычно длинная реплика"


def test_empty_chunk_gets_single_empty_warning():
    quality = _measure(np.zeros(0, dtype=np.float32))
    assert _codes(quality) == ["empty"]
    assert _text_of(quality, "empty") == "Пустое аудио"
    assert quality.duration_sec == 0.0


def test_duration_per_char_is_computed():
    quality = _measure(_speech(2.8))
    assert quality.chars == len(TEXT)
    assert quality.duration_per_char == pytest.approx(quality.duration_sec / len(TEXT))
    # Без текста «секунд на знак» не существует — это None, а не деление на ноль.
    assert _measure(_speech(1.0), text="").duration_per_char is None


# --- громкость -----------------------------------------------------------------
def test_lufs_matches_pyloudnorm():
    import pyloudnorm as pyln

    chunk = _speech(3.0, amplitude=0.5)
    expected = float(pyln.Meter(SAMPLE_RATE).integrated_loudness(chunk))
    assert _measure(chunk).lufs == pytest.approx(expected, abs=1e-4)


def test_lufs_is_none_on_short_chunk_and_silence():
    # Короче 400 мс pyloudnorm измерять не умеет: None вместо ValueError.
    assert _measure(_speech(0.05)).lufs is None
    assert _measure(_speech(0.3)).lufs is None
    # Цифровая тишина даёт -inf: наружу должен уйти None, а не бесконечность.
    silence = _measure(np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32))
    assert silence.lufs is None
    assert math.isfinite(silence.rms_dbfs) and math.isfinite(silence.peak_dbfs)


def test_peak_dbfs_reflects_amplitude():
    quality = _measure(sine(2.0, 220.0, amplitude=0.5))
    assert quality.peak_dbfs == pytest.approx(20 * math.log10(0.5), abs=0.05)


# --- метаданные QA переносятся, а не пересчитываются ---------------------------
def test_qa_fields_are_carried_over():
    from backend import qa_screening

    outcome = QaOutcome(
        status="passed",
        wer=0.125,
        attempts=2,
        mode=config.QA_MODE_SMART,
        screening=qa_screening.Screening(suspicious=False, reasons=[]),
    )
    quality = _measure(_speech(2.8), qa=outcome)
    assert quality.wer == 0.125
    assert quality.qa_attempts == 2
    assert quality.qa_mode == config.QA_MODE_SMART
    assert quality.screening_reasons == []

    # Без QA причины считаются по подготовленному куску — своими кодами не обходимся.
    assert "duration_long" in _measure(_speech(12.0)).screening_reasons
    # Без проверки поля остаются пустыми, а не нулями.
    plain = _measure(_speech(2.8))
    assert (plain.wer, plain.qa_attempts, plain.qa_mode) == (None, None, None)


# --- сериализация --------------------------------------------------------------
def test_metadata_round_trip_and_missing_fields():
    quality = _measure(
        _speech(2.8),
        qa=QaOutcome(status="passed", wer=0.125, attempts=2, mode=config.QA_MODE_SMART),
    )
    data = quality.to_dict()
    # allow_nan=False: ни NaN, ни бесконечностей в метаданных быть не должно.
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    restored = take_quality.TakeQuality.from_dict(data)
    assert restored is not None and restored.to_dict() == data

    # Отсутствующие поля читаются как None/пустые списки — это норма, а не сбой.
    partial = take_quality.TakeQuality.from_dict({"duration_sec": 1.5, "chars": 10})
    assert partial is not None
    assert (partial.duration_sec, partial.chars) == (1.5, 10)
    assert partial.lufs is None
    assert partial.silence_ratio is None
    assert partial.duration_per_char is None
    assert partial.warnings == []
    assert partial.screening_reasons == []

    # Нечего читать — значит, метрик нет.
    assert take_quality.TakeQuality.from_dict(None) is None
    assert take_quality.TakeQuality.from_dict({}) is None


def test_serialization_never_emits_non_finite_numbers():
    broken = take_quality.TakeQuality(
        duration_sec=1.0,
        lufs=float("inf"),
        peak_dbfs=float("nan"),
        rms_dbfs=float("-inf"),
        clipping_ratio=float("nan"),
    )
    data = broken.to_dict()
    assert data["lufs"] is None
    assert data["peak_dbfs"] is None
    assert data["rms_dbfs"] is None
    assert data["clipping_ratio"] is None
    json.dumps(data, allow_nan=False)


# --- доезд до API --------------------------------------------------------------
def test_quality_reaches_project_takes_and_job_api(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            project, job_id = await _rendered_project(client)

            reopened = (await client.get(f"/api/projects/{project['id']}")).json()
            take = reopened["replicas"][0]["takes"][0]
            quality = take["quality"]
            assert quality is not None
            for key in (
                "duration_sec",
                "chars",
                "clipping",
                "clipping_ratio",
                "silence_ratio",
                "duration_per_char",
                "peak_dbfs",
                "rms_dbfs",
                "lufs",
                "wer",
                "qa_attempts",
                "qa_mode",
                "screening_reasons",
                "warnings",
            ):
                assert key in quality
            assert quality["duration_sec"] > 0
            assert quality["warnings"] == []
            # Существующие поля take'а не потеряны и не переименованы.
            assert take["qa"] is None
            assert take["duration_sec"] > 0

            # Метрики видны и в статусе готовой задачи.
            job = (await client.get(f"/api/jobs/{job_id}")).json()
            assert job["replicas"][0]["quality"] is not None
            assert job["replicas"][0]["quality"]["chars"] > 0

            # Пересинтез реплики проекта добавляет take с метриками, а не без них.
            accepted = await client.post(
                f"/api/projects/{project['id']}/replicas/0/regenerate"
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])
            after = (await client.get(f"/api/projects/{project['id']}")).json()
            takes = after["replicas"][0]["takes"]
            assert len(takes) == 2
            assert all(item["quality"] is not None for item in takes)

    _run(scenario)


def test_quality_reaches_job_variants(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            _project, job_id = await _rendered_project(client)
            accepted = await client.post(f"/api/jobs/{job_id}/replicas/0/regenerate")
            assert accepted.status_code == 202, accepted.text

            deadline = time.monotonic() + 30.0
            variants: list[dict] = []
            while time.monotonic() < deadline:
                data = (await client.get(f"/api/jobs/{job_id}")).json()
                variants = data["replicas"][0]["variants"]
                if len(variants) >= 2 and data["regenerating_replica"] is None:
                    break
                await asyncio.sleep(0.02)
            assert len(variants) == 2, variants
            # И «исходный», и новый вариант приходят со своей диагностикой.
            for variant in variants:
                assert variant["quality"] is not None
                assert "warnings" in variant["quality"]
                assert variant["quality"]["duration_sec"] > 0

    _run(scenario)


def test_migration_4_adds_quality_column_to_existing_database(tmp_path):
    """База прошлых фаз доводится до колонки `quality` без пересоздания."""
    import sqlite3

    from backend.db import migrations

    connection = sqlite3.connect(tmp_path / "old.db")
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for version, script in migrations.MIGRATIONS:
        if version <= 3:
            connection.executescript(script)
            connection.execute("DELETE FROM schema_version")
            connection.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    assert "quality" not in [row[1] for row in connection.execute("PRAGMA table_info(takes)")]

    assert migrations.apply_migrations(connection) == migrations.MIGRATIONS[-1][0]
    assert "quality" in [row[1] for row in connection.execute("PRAGMA table_info(takes)")]
    # Повторный запуск идемпотентен: колонка не добавляется второй раз.
    assert migrations.apply_migrations(connection) == migrations.MIGRATIONS[-1][0]


def test_take_without_quality_is_served_as_null(workspace, stub, fake_store, monkeypatch):
    """Take, записанный до фазы 14, отдаётся с `quality: null` и не ломает карточку."""

    async def scenario():
        async with _client(monkeypatch) as client:
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
            # Старый код клал take без колонки quality: ключа в словаре нет вовсе.
            get_projects_store().save_render_takes(
                project["id"],
                "old-job",
                [
                    {
                        "index": 0,
                        "audio_path": str(workspace / "r0.wav"),
                        "duration_sec": 1.0,
                        "qa": None,
                    }
                ],
            )

            reopened = (await client.get(f"/api/projects/{project['id']}")).json()
            take = reopened["replicas"][0]["takes"][0]
            assert take["quality"] is None
            assert reopened["replicas"][0]["selected_take_id"] == take["id"]
            # Отсутствие метрик не ошибка: остальные поля на месте.
            assert take["duration_sec"] == 1.0
            assert take["audio_url"].endswith(f"/takes/{take['id']}/audio")

    _run(scenario)


def test_clipping_of_raw_model_output_survives_normalization():
    """Перегруз модели виден в диагностике, хотя подготовка его уже сняла.

    `_prepare_chunk` выравнивает громкость и подрезает пик, поэтому в готовом
    take перегруза нет — и без сырого куска предупреждение о клиппинге не
    появилось бы никогда, кроме случая, когда его добавили уже после подготовки.
    """
    text = "Сегодня хорошая погода, и я рад тебя видеть"
    raw = np.clip(sine(2.8, amplitude=3.0), -1.0, 1.0).astype(np.float32)
    # Подготовленный кусок: тише единицы, то есть клиппинга в нём нет.
    prepared = (raw * 0.2).astype(np.float32)

    assert audio_analysis.clipping_ratio(prepared) == 0.0
    without_raw = take_quality.measure(prepared, text)
    with_raw = take_quality.measure(prepared, text, raw=raw)

    assert without_raw.clipping is False
    assert with_raw.clipping is True
    assert with_raw.clipping_ratio > 0.0
    assert any(item["code"] == qa_screening.REASON_CLIPPING for item in with_raw.warnings)
