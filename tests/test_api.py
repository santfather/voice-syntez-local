"""REST API дашборда: разбор текста, постановка задач, варианты и файлы реплик."""

import asyncio
import contextlib
import time

import httpx

from backend import audio_pipeline, config, engine_lifecycle, main
from backend.engines.base import ENGINE_KOKORO
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


def test_generate_route_is_marked_deprecated():
    """`/api/generate` — legacy: обход обязательной подготовки виден в OpenAPI.

    Роут берёт сырой диалог и не требует проекта, поэтому ни `final_text`, ни
    состояния проекта, ни `require_prepared` у него нет. Фронтенд им не пользуется,
    но он остаётся для скриптов и разовых прогонов — значит, должен быть помечен, а
    не выглядеть обычной точкой входа (docs/api.md).
    """
    assert main.app.openapi()["paths"]["/api/generate"]["post"]["deprecated"] is True


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


def test_builtin_voices_appear_without_any_recording(monkeypatch):
    """Встроенные голоса движка видны в списке, как только у него есть веса.

    Ни записи, ни подложенного файла для них не нужно: движок говорит своим
    голосом по полу карточки (`EngineInfo.builtin_voices`). Проверяется весь путь
    — от доступности весов до карточки в списке и её удаления.
    """

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            # «Веса на диске»: только это условие делает встроенные голоса
            # доступными, и задаётся оно менеджером моделей, а не хранилищем.
            monkeypatch.setattr(
                main.model_manager,
                "engine_available",
                lambda engine_id: engine_id == ENGINE_KOKORO,
            )

            listed = (await client.get("/api/voices")).json()["voices"]
            builtin = [voice for voice in listed if voice["engine"] == ENGINE_KOKORO]
            assert [voice["name"] for voice in builtin] == ["Света", "Дима", "Маша"]
            # Референса нет — и это честно сказано интерфейсу: иначе он предложил бы
            # записать голос движку, которому записывать нечего.
            assert all(voice["has_audio"] is False for voice in builtin)
            assert all(voice["ref_text"] == "" for voice in builtin)

            # Второй запрос список не удваивает: заведение идемпотентно по движку.
            again = (await client.get("/api/voices")).json()["voices"]
            assert len(again) == len(listed)

            # Своя карточка встроенного голоса создаётся без файла вовсе.
            created = await client.post(
                "/api/voices",
                data={"name": "Мой Кокоро", "gender": "male", "engine": ENGINE_KOKORO},
            )
            assert created.status_code == 201, created.text
            assert created.json()["engine"] == ENGINE_KOKORO
            assert created.json()["has_audio"] is False

            # Клонирующему движку запись по-прежнему обязательна — отказ понятный.
            refused = await client.post(
                "/api/voices", data={"name": "Без записи", "gender": "male", "engine": "f5"}
            )
            assert refused.status_code == 400, refused.text
            assert "запись" in refused.json()["detail"]

            # Удалённый встроенный голос не возвращается при следующем запросе:
            # иначе удалить его было бы нельзя вовсе.
            removed = builtin[0]["id"]
            deleted = await client.delete(f"/api/voices/{removed}")
            assert deleted.status_code == 200, deleted.text
            after = (await client.get("/api/voices")).json()["voices"]
            assert removed not in {voice["id"] for voice in after}

    _run(scenario)


def test_voice_delete_refused_while_queue_is_busy(monkeypatch):
    """Голос не удаляется под работающей задачей: синтез читает его референс с диска.

    Правило то же, что у выгрузки движка и сброса приложения (`queue_busy_now`):
    иначе удаление оборвало бы рендер на середине, а ошибка выглядела бы
    случайной. Как только очередь освободилась, тот же голос удаляется штатно —
    проверка запрещает удаление на время задачи, а не навсегда.
    """
    busy = {"value": True}
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: busy["value"])
    monkeypatch.setattr(
        main.model_manager,
        "engine_available",
        lambda engine_id: engine_id == ENGINE_KOKORO,
    )

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            listed = (await client.get("/api/voices")).json()["voices"]
            voice_id = next(
                voice["id"] for voice in listed if voice["engine"] == ENGINE_KOKORO
            )

            refused = await client.delete(f"/api/voices/{voice_id}")
            assert refused.status_code == 409, refused.text
            assert "очеред" in refused.json()["detail"].lower()
            # Голос и его файлы на месте: отказ не сделал половину работы.
            still_there = (await client.get("/api/voices")).json()["voices"]
            assert voice_id in {voice["id"] for voice in still_there}

            busy["value"] = False
            deleted = await client.delete(f"/api/voices/{voice_id}")
            assert deleted.status_code == 200, deleted.text

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
            # Без тумблера строгой проверки отметок нет: это и есть поведение по умолчанию.
            assert all(replica["qa"] is None for replica in data["replicas"])

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


def test_generate_with_qa_reports_status_per_replica(stub, fake_store, monkeypatch):
    async def scenario():
        # Реплики синтезируются по очереди, поэтому расшифровки идут в том же порядке.
        answers = iter(["Первая реплика", "Вторая реплика"])

        async def fake(chunk):
            return next(answers, "")

        monkeypatch.setattr(audio_pipeline, "_transcribe_chunk", fake)
        async with _client(monkeypatch) as (client, _queue):
            job_id = await _generate(client, qa=True)
            data = await _wait(client, job_id, lambda item: item["status"] == "done")
            assert [replica["qa"]["status"] for replica in data["replicas"]] == ["passed", "passed"]
            assert [replica["qa"]["attempts"] for replica in data["replicas"]] == [1, 1]

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


def test_render_text_analysis_does_not_block_other_requests(stub, fake_store, monkeypatch):
    """Лингвистический анализ идёт в отдельном потоке, а не в event loop.

    Ручка ожидания ответа LLM — секунды. Синхронный вызов держал бы в это время
    весь сервер, включая `/api/status`, по которому лаунчер определяет готовность.
    """

    def slow_analysis(text, chunks):
        time.sleep(1.0)
        return {"status": "ready", "candidates_total": 0}

    monkeypatch.setattr(main, "_analyze_text_chunks", slow_analysis)

    async def scenario() -> None:
        async with _client(monkeypatch) as (client, _queue):
            render = asyncio.create_task(
                client.post(
                    "/api/render-text",
                    json={"voice_id": "voice1", "text": "Сплошной текст для анализа."},
                )
            )
            # Анализ длится секунду: если он выполняется в loop, к этому моменту
            # запрос уже завершится, а loop — простоял всё это время.
            await asyncio.sleep(0.2)
            assert not render.done(), "анализ держал event loop вместо отдельного потока"
            status = await client.get("/api/status")
            assert status.status_code == 200
            assert (await render).status_code == 202

    asyncio.run(scenario())
