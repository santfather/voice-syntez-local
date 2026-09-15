"""REST API дашборда: разбор текста, постановка задач, варианты и файлы реплик."""

import asyncio
import contextlib
import time

import httpx
import pytest

from backend import config, main
from backend.job_queue import JobQueue

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."


@contextlib.asynccontextmanager
async def _client(monkeypatch):
    """Очередь и ASGI-клиент в одном event loop: воркер нельзя запускать в другом."""
    queue = JobQueue()
    await queue.start()
    monkeypatch.setattr(main, "get_queue", lambda: queue)
    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, queue
    finally:
        await queue.stop()


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _generate(client, **overrides) -> str:
    payload = {
        "dialogue_text": DIALOGUE,
        "speakers": {"ИВАН": {"voice_id": "voice1"}, "МАРГО": {"voice_id": "voice1"}},
        "output_format": "wav",
        **overrides,
    }
    response = await client.post("/api/generate", json=payload)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


async def _wait(client, job_id: str, done, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if done(data):
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"задача {job_id} не достигла нужного состояния")


def test_engines_endpoint_lists_passports(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            response = await client.get("/api/engines")
            assert response.status_code == 200
            engines = {engine["id"]: engine for engine in response.json()["engines"]}
            assert {"f5", "xtts", "xtts-banana"} <= set(engines)
            assert {param["name"] for param in engines["xtts"]["params"]} == {
                "temperature",
                "repetition_penalty",
            }

    _run(scenario)


def test_parse_endpoint_reports_replicas_and_slots(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            response = await client.post(
                "/api/parse", json={"dialogue_text": DIALOGUE, "chunk_strategy": "short"}
            )
            assert response.status_code == 200
            data = response.json()
            assert [replica["label"] for replica in data["replicas"]] == ["ИВАН", "МАРГО"]
            assert [voice["key"] for voice in data["voices"]] == ["ИВАН", "МАРГО"]

            # Стратегия нарезки описана Literal: опечатка — ошибка запроса, а не тихий дефолт.
            bad = await client.post(
                "/api/parse", json={"dialogue_text": DIALOGUE, "chunk_strategy": "long"}
            )
            assert bad.status_code == 422

    _run(scenario)


def test_generate_validates_request(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            empty = await client.post("/api/generate", json={"dialogue_text": "   "})
            assert empty.status_code == 400
            assert "пуст" in empty.json()["detail"]

            no_voice = await client.post(
                "/api/generate",
                json={"dialogue_text": DIALOGUE, "speakers": {"ИВАН": {"voice_id": "voice1"}}},
            )
            assert no_voice.status_code == 400
            assert "Не назначен голос" in no_voice.json()["detail"]

            missing = await client.get("/api/jobs/нет-такой")
            assert missing.status_code == 404

    _run(scenario)


def test_job_flow_with_variant_history(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            job_id = await _generate(client)
            data = await _wait(client, job_id, lambda item: item["status"] == "done")
            assert data["audio_url"] == f"/api/jobs/{job_id}/audio"
            assert len(data["replicas"]) == 2
            assert all(isinstance(replica["seed"], int) for replica in data["replicas"])
            assert all(replica["variants"] == [] for replica in data["replicas"])

            regenerate = await client.post(f"/api/jobs/{job_id}/replicas/0/regenerate")
            assert regenerate.status_code == 202
            data = await _wait(
                client,
                job_id,
                lambda item: item["regenerating_replica"] is None
                and len(item["replicas"][0]["variants"]) == 2,
            )
            variants = data["replicas"][0]["variants"]
            assert [variant["label"] for variant in variants] == ["исходный", "вариант 1"]
            assert [variant["active"] for variant in variants] == [False, True]
            assert data["replicas"][0]["seed"] == variants[1]["seed"]

            for variant in variants:
                audio = await client.get(variant["audio_url"])
                assert audio.status_code == 200
                assert audio.headers["content-type"] == "audio/wav"
                assert len(audio.content) > 0

            back = await client.post(f"/api/jobs/{job_id}/replicas/0/variants/v0")
            assert back.status_code == 202
            data = await _wait(
                client,
                job_id,
                lambda item: item["regenerating_replica"] is None
                and item["replicas"][0]["variants"][0]["active"] is True,
            )
            assert data["replicas"][0]["seed"] == variants[0]["seed"]

            same = await client.post(f"/api/jobs/{job_id}/replicas/0/variants/v0")
            assert same.status_code == 400  # этот вариант уже стоит в файле
            unknown = await client.post(f"/api/jobs/{job_id}/replicas/0/variants/v99")
            assert unknown.status_code == 404
            no_job = await client.post("/api/jobs/нет-такой/replicas/0/variants/v0")
            assert no_job.status_code == 404

            audio = await client.get(f"/api/jobs/{job_id}/audio")
            assert audio.status_code == 200
            assert audio.headers["content-type"] == "audio/wav"

    _run(scenario)


def test_regenerate_validates_job_and_index(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            assert (await client.post("/api/jobs/нет-такой/replicas/0/regenerate")).status_code == 404

            job_id = await _generate(client)
            await _wait(client, job_id, lambda item: item["status"] == "done")
            out_of_range = await client.post(f"/api/jobs/{job_id}/replicas/99/regenerate")
            assert out_of_range.status_code == 400

    _run(scenario)


def test_render_text_endpoint(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            empty = await client.post("/api/render-text", json={"voice_id": "voice1", "text": " "})
            assert empty.status_code == 400

            too_long = await client.post(
                "/api/render-text",
                json={"voice_id": "voice1", "text": "а" * (config.MAX_TEXT_CHARS + 1)},
            )
            assert too_long.status_code == 400

            accepted = await client.post(
                "/api/render-text",
                json={"voice_id": "voice1", "text": "Сплошной текст для озвучки.", "output_format": "wav"},
            )
            assert accepted.status_code == 202
            assert accepted.json()["total_replicas"] == 1

    _run(scenario)
