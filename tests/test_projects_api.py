"""REST API проектов: создание, разбор в реплики, рендер и восстановление.

Тесты поднимают приложение целиком (ASGI-транспорт) с очередью в том же event
loop, но с движком-заглушкой: проверяется работа с проектом, а не модель.
"""

import asyncio
import contextlib
import time
import types
from pathlib import Path

import httpx
from conftest import analyze_project

from backend import config, main
from backend.job_queue import JobQueue, JobStatus

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."


@contextlib.asynccontextmanager
async def _client(monkeypatch):
    queue = JobQueue()
    await queue.start()
    monkeypatch.setattr(main, "get_queue", lambda: queue)
    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        await queue.stop()


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _wait_job(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"задача {job_id} не завершилась")


async def _create_project(client, name: str = "Тест", text: str = DIALOGUE) -> dict:
    response = await client.post(
        "/api/projects", json={"name": name, "source_text": text, "mode": "dialogue"}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_project_crud_flow(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(client)
            assert project["status"] == config.PROJECT_STATUS_DRAFT
            assert project["replicas_count"] == 0

            listed = await client.get("/api/projects")
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()["projects"]] == [project["id"]]

            patched = await client.patch(
                f"/api/projects/{project['id']}",
                json={"name": "Другое имя", "source_text": DIALOGUE + "\nИВАН: Третья."},
            )
            assert patched.status_code == 200
            assert patched.json()["name"] == "Другое имя"

            deleted = await client.delete(f"/api/projects/{project['id']}")
            assert deleted.status_code == 200
            assert (await client.get(f"/api/projects/{project['id']}")).status_code == 404

    _run(scenario)


def test_unknown_project_returns_404(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            for method, url in (
                ("get", "/api/projects/нет-такого"),
                ("patch", "/api/projects/нет-такого"),
                ("delete", "/api/projects/нет-такого"),
                ("post", "/api/projects/нет-такого/parse"),
                ("post", "/api/projects/нет-такого/render"),
            ):
                # Тело передаётся только POST и PATCH: у GET и DELETE аргумента
                # json нет, и это ограничение httpx, а не API.
                kwargs = {"json": {}} if method in ("post", "patch") else {}
                response = await getattr(client, method)(url, **kwargs)
                assert response.status_code == 404, (method, url, response.text)
                assert "не найден" in response.json()["detail"]

    _run(scenario)


def test_parse_creates_replicas_and_keeps_voices(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(client)
            parsed = await client.post(f"/api/projects/{project['id']}/parse", json={})
            assert parsed.status_code == 200
            data = parsed.json()
            assert [item["label"] for item in data["replicas"]] == ["ИВАН", "МАРГО"]
            assert [item["index"] for item in data["replicas"]] == [0, 1]

            assigned = await client.patch(
                f"/api/projects/{project['id']}",
                json={
                    "speakers": {
                        "ИВАН": {"voice_id": "voice1"},
                        "МАРГО": {"voice_id": "voice1"},
                    }
                },
            )
            assert assigned.status_code == 200
            assert [item["voice_id"] for item in assigned.json()["replicas"]] == [
                "voice1",
                "voice1",
            ]

            again = await client.post(
                f"/api/projects/{project['id']}/parse", json={"chunk_strategy": "short"}
            )
            assert again.status_code == 200
            assert [item["voice_id"] for item in again.json()["replicas"]] == [
                "voice1",
                "voice1",
            ]

            # Опечатка в стратегии — ошибка запроса, а не тихий дефолт.
            bad = await client.post(
                f"/api/projects/{project['id']}/parse", json={"chunk_strategy": "long"}
            )
            assert bad.status_code == 422

    _run(scenario)


def test_render_saves_takes_and_survives_reopen(stub, fake_store, monkeypatch):
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
            # Рендер возможен только по подготовленному тексту (см. /analyze).
            await analyze_project(client, project["id"])

            accepted = await client.post(
                f"/api/projects/{project['id']}/render", json={"output_format": "wav"}
            )
            assert accepted.status_code == 202, accepted.text
            job_id = accepted.json()["job_id"]
            await _wait_job(client, job_id)

            # Проект «переоткрывается» новым GET: данные берутся из SQLite, а не
            # из памяти очереди — именно это и должно переживать перезапуск.
            reopened = (await client.get(f"/api/projects/{project['id']}")).json()
            assert reopened["status"] == config.PROJECT_STATUS_RENDERED
            assert reopened["job_id"] == job_id
            assert len(reopened["replicas"]) == 2
            for replica in reopened["replicas"]:
                assert replica["status"] == config.REPLICA_STATUS_RENDERED
                assert len(replica["takes"]) == 1
                take = replica["takes"][0]
                assert take["id"] == replica["selected_take_id"]
                assert take["engine"] == "stub"
                assert take["seed"] is not None
                assert take["duration_sec"] > 0
                assert take["parameters"]["speed"] == config.DEFAULT_SPEED

            # Реплики в списке — с индексом и подписью: по ним UI строит карточки.
            assert [item["index"] for item in reopened["replicas"]] == [0, 1]

            listed = (await client.get("/api/projects")).json()["projects"]
            assert listed[0]["status"] == config.PROJECT_STATUS_RENDERED
            assert listed[0]["replicas_count"] == 2

    _run(scenario)


def test_render_requires_replicas_and_voices(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(client)
            empty = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert empty.status_code == 400
            assert "реплик" in empty.json()["detail"]

            await client.post(f"/api/projects/{project['id']}/parse", json={})
            no_voice = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert no_voice.status_code == 400
            assert "Не назначен голос" in no_voice.json()["detail"]
            assert "ИВАН" in no_voice.json()["detail"]

    _run(scenario)


def test_delete_project_removes_take_files(stub, fake_store, monkeypatch):
    """Удаление проекта не оставляет осиротевших файлов кусков на диске."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(client)
            await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": "voice1"}, "МАРГО": {"voice_id": "voice1"}}},
            )
            await client.post(f"/api/projects/{project['id']}/parse", json={})
            await analyze_project(client, project["id"])
            accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
            await _wait_job(client, accepted.json()["job_id"])

            data = (await client.get(f"/api/projects/{project['id']}")).json()
            files = [replica["takes"][0]["audio_path"] for replica in data["replicas"]]
            assert len(files) == 2
            for path in files:
                assert (await asyncio.to_thread(Path(path).exists)) is True

            assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 200
            for path in files:
                assert (await asyncio.to_thread(Path(path).exists)) is False

    _run(scenario)


def test_generate_endpoint_still_works(stub, fake_store, monkeypatch):
    """Старый путь без проектов не сломан: он остаётся рабочим в этой фазе."""
    async def scenario():
        async with _client(monkeypatch) as client:
            response = await client.post(
                "/api/generate",
                json={
                    "dialogue_text": DIALOGUE,
                    "speakers": {
                        "ИВАН": {"voice_id": "voice1"},
                        "МАРГО": {"voice_id": "voice1"},
                    },
                    "output_format": "wav",
                },
            )
            assert response.status_code == 202
            data = await _wait_job(client, response.json()["job_id"])
            assert len(data["replicas"]) == 2

    _run(scenario)


# --- прогрев коротких реплик в API рендера ------------------------------------
class _CaptureQueue:
    """Очередь-перехватчик: запоминает JobPayload, но рендер не запускает.

    Тест проверяет, что флаг прогрева доехал от тела запроса до настроек сборки;
    сам синтез здесь ни при чём, и поднимать его значило бы проверять не то.
    """

    def __init__(self) -> None:
        self.payloads: list[object] = []

    def submit(self, payload, priority=None):
        self.payloads.append(payload)
        return types.SimpleNamespace(
            id="job-capture",
            status=JobStatus.QUEUED,
            total_replicas=len(payload.replicas),
        )


async def _prepared_project(client, name: str = "Тест") -> dict:
    """Проект, готовый к рендеру: голоса назначены, текст разобран и подтверждён."""
    project = await _create_project(client, name=name)
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
    return project


def test_render_request_carries_warmup_flag(stub, fake_store, monkeypatch):
    """Флаг прогрева из тела запроса обязан дойти до RenderSettings, а не потеряться."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_project(client)
            queue = _CaptureQueue()
            monkeypatch.setattr(main, "get_queue", lambda: queue)

            accepted = await client.post(
                f"/api/projects/{project['id']}/render",
                json={"warmup_short_replicas": True, "output_format": "wav"},
            )
            assert accepted.status_code == 202, accepted.text
            assert queue.payloads[0].settings.warmup is True

            # Настройка — часть сборки проекта: повторный запуск без тела запроса
            # должен звучать так же, а не «по дефолтам».
            saved = (await client.get(f"/api/projects/{project['id']}")).json()["render_settings"]
            assert saved["warmup_short_replicas"] is True

    _run(scenario)


def test_render_without_warmup_field_uses_saved_or_default(stub, fake_store, monkeypatch):
    """Отсутствие поля не ломает запрос: берётся сохранённое значение, иначе None."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_project(client)
            fresh = await _prepared_project(client, name="Свежий")
            queue = _CaptureQueue()
            monkeypatch.setattr(main, "get_queue", lambda: queue)

            first = await client.post(
                f"/api/projects/{project['id']}/render",
                json={"warmup_short_replicas": True, "output_format": "wav"},
            )
            assert first.status_code == 202, first.text
            # Поле не пришло: запрос проходит и использует запомненное значение.
            second = await client.post(
                f"/api/projects/{project['id']}/render", json={"output_format": "wav"}
            )
            assert second.status_code == 202, second.text
            assert queue.payloads[1].settings.warmup is True

            # Свежий проект ничего не помнит: None означает «политика приложения»,
            # и это не то же самое, что явное False.
            assert (
                (await client.get(f"/api/projects/{fresh['id']}")).json()["render_settings"].get(
                    "warmup_short_replicas"
                )
                is None
            )
            third = await client.post(
                f"/api/projects/{fresh['id']}/render", json={"output_format": "wav"}
            )
            assert third.status_code == 202, third.text
            assert queue.payloads[-1].settings.warmup is None

    _run(scenario)
