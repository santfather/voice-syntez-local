"""API записей пользователя: разбор, дубли, превью и монтаж (§35).

Тесты идут через настоящее приложение (ASGI-транспорт) и настоящий HTTP-слой
FastAPI: проверяются коды ответов, форма тела и то, что состояние действительно
оказалось на диске. Записи — синтетический WAV, TTS и Whisper не поднимаются;
DeepFilterNet в этих сценариях не включается вовсе.
"""

from __future__ import annotations

import asyncio
import io
import time

import pytest
import soundfile as sf
from conftest import sine
from test_projects_api import _client

from backend import config, recording_store
from backend.engines.base import SAMPLE_RATE

DIALOGUE = "Анна: Первая реплика.\nИгорь: Вторая реплика."


def _wav_bytes(seconds: float = 0.6, freq: float = 220.0) -> bytes:
    """WAV-байты для multipart-загрузки — как запись из браузера, только без микрофона."""
    buffer = io.BytesIO()
    sf.write(buffer, sine(seconds, freq), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _create(client, text: str = DIALOGUE, name: str = "Запись") -> dict:
    """Создаёт проект записи с текстом — почти все сценарии начинаются с этого."""
    response = await client.post(
        "/api/recording-projects", json={"name": name, "dialogue_text": text}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _upload(client, project_id: str, index: int, seconds: float = 0.6) -> dict:
    """Загружает дубль реплики multipart-ом и возвращает тело ответа."""
    response = await client.post(
        f"/api/recording-projects/{project_id}/replicas/{index}/takes",
        files={"file": ("take.wav", _wav_bytes(seconds), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _wait_render(client, project_id: str, timeout: float = 60.0) -> dict:
    """Ждёт завершения монтажа: поток не должен переживать тест."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/recording-projects/{project_id}/render/status")).json()
        if data["status"] in ("done", "error"):
            return data
        await asyncio.sleep(0.05)
    raise AssertionError("монтаж не завершился")


def _duration(content: bytes) -> float:
    """Длительность WAV из тела ответа: так проверяется то, что реально скачает плеер."""
    data, sample_rate = sf.read(io.BytesIO(content))
    return len(data) / sample_rate


# --- 20. Создание и разбор ---------------------------------------------------
def test_create_with_text_parses_replicas_and_parse_is_idempotent(monkeypatch):
    """Текст сразу даёт реплики и роли, а повторный разбор ничего не ломает и не дублирует."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            assert [item["index"] for item in project["replicas"]] == [0, 1]
            assert [item["speaker"] for item in project["replicas"]] == ["Анна", "Игорь"]
            assert project["speakers"] == ["Анна", "Игорь"]
            assert [role["speaker"] for role in project["roles"]] == ["Анна", "Игорь"]
            assert project["roles"][0]["replicas"] == [0]
            assert project["roles"][1]["replicas"] == [1]

            first = await client.post(f"/api/recording-projects/{project['id']}/parse")
            second = await client.post(f"/api/recording-projects/{project['id']}/parse")
            assert first.status_code == second.status_code == 200
            assert first.json()["replicas"] == project["replicas"]
            assert second.json()["replicas"] == project["replicas"]
            assert second.json()["speakers"] == ["Анна", "Игорь"]

            # Дубль, записанный до повторного разбора, обязан пережить его: индексы
            # реплик те же, значит и привязка дубля не должна потеряться.
            take = (await _upload(client, project["id"], 0))["take"]
            after = (
                await client.post(f"/api/recording-projects/{project['id']}/parse")
            ).json()
            assert [item["id"] for item in after["takes"]] == [take["id"]]
            assert after["active_takes"]["0"] == take["id"]

    _run(scenario)


# --- 21. Загрузка дубля ------------------------------------------------------
def test_upload_take_writes_raw_wav_and_keeps_previous_take(monkeypatch):
    """Первый дубль ложится в raw/ как WAV, второй не стирает его и становится активным."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            index = project["replicas"][0]["index"]

            first = await _upload(client, project["id"], index, seconds=0.6)
            take = first["take"]
            assert take["duration_sec"] == pytest.approx(0.6, abs=0.05)

            raw = recording_store.get_store().raw_dir(project["id"])
            saved = raw / take["raw_file"]
            assert saved.exists(), "сырой дубль не появился в raw/"
            assert saved.read_bytes()[:4] == b"RIFF", "в raw/ должен лежать декодированный WAV"

            second = await _upload(client, project["id"], index, seconds=0.9)
            body = second["project"]
            assert second["take"]["id"] != take["id"]
            assert {item["id"] for item in body["takes"]} == {take["id"], second["take"]["id"]}
            assert len(list(raw.glob("*.wav"))) == 2, "второй дубль перезаписал первый"
            # Активен новый дубль, старый остаётся доступным для сравнения.
            assert body["active_takes"][str(index)] == second["take"]["id"]
            assert [item["active"] for item in body["takes"]] == [False, True]

    _run(scenario)


# --- 22. Выбор и удаление дублей ---------------------------------------------
def test_select_and_delete_takes_switch_active_and_clear_replica(monkeypatch):
    """Активный дубль переключается, при удалении активного выбирается оставшийся."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            index = project["replicas"][0]["index"]
            first = (await _upload(client, project["id"], index))["take"]
            second = (await _upload(client, project["id"], index))["take"]

            selected = await client.post(
                f"/api/recording-projects/{project['id']}/replicas/{index}"
                f"/takes/{first['id']}/select"
            )
            assert selected.status_code == 200, selected.text
            assert selected.json()["active_takes"][str(index)] == first["id"]

            # Удаляем активный: реплика остаётся записанной за счёт второго дубля.
            deleted = await client.delete(
                f"/api/recording-projects/{project['id']}/replicas/{index}"
                f"/takes/{first['id']}"
            )
            assert deleted.status_code == 200, deleted.text
            body = deleted.json()
            assert body["active_takes"][str(index)] == second["id"]
            raw = recording_store.get_store().raw_dir(project["id"])
            assert not (raw / first["raw_file"]).exists()

            # Последний дубль: реплика снова «не записана» и блокирует монтаж.
            last = await client.delete(
                f"/api/recording-projects/{project['id']}/replicas/{index}"
                f"/takes/{second['id']}"
            )
            body = last.json()
            assert str(index) not in body["active_takes"]
            assert index in body["readiness"]["missing"]
            assert not (raw / second["raw_file"]).exists()

    _run(scenario)


# --- 23. Монтаж --------------------------------------------------------------
def test_render_is_blocked_by_missing_replicas_then_accepted(monkeypatch):
    """Незаписанная реплика — 409 с перечнем индексов; после записи всех — 202."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            # Записана только первая реплика — вторая блокирует монтаж.
            await _upload(client, project["id"], 0)
            blocked = await client.post(f"/api/recording-projects/{project['id']}/render", json={})
            assert blocked.status_code == 409, blocked.text
            detail = blocked.json()["detail"]
            assert detail["missing"] == [1], "в теле должен быть индекс незаписанной реплики"

            for replica in project["replicas"][1:]:
                await _upload(client, project["id"], replica["index"])
            accepted = await client.post(
                f"/api/recording-projects/{project['id']}/render", json={"pause_ms": 150}
            )
            assert accepted.status_code == 202, accepted.text
            state = await _wait_render(client, project["id"])
            assert state["status"] == "done", state

    _run(scenario)


# --- 24. Невалидные параметры ------------------------------------------------
def test_invalid_speed_pitch_and_oversized_upload_are_rejected(monkeypatch):
    """Настройки вне диапазона и слишком большая запись — ошибка клиента, не 500."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            url = f"/api/recording-projects/{project['id']}/voice-profiles"

            too_fast = await client.post(url, json={"name": "Голос", "speed": 5.0})
            assert too_fast.status_code == 400, too_fast.text
            too_high = await client.post(url, json={"name": "Голос", "pitch_semitones": 50})
            assert too_high.status_code == 400, too_high.text

            # Лимит размера: запись проверяется до декодирования.
            monkeypatch.setattr(config, "MAX_RECORDING_BYTES", 512)
            too_big = await client.post(
                f"/api/recording-projects/{project['id']}/replicas/0/takes",
                files={"file": ("take.wav", _wav_bytes(0.6), "audio/wav")},
            )
            assert too_big.status_code == 400, too_big.text

    _run(scenario)


# --- 25. Превью --------------------------------------------------------------
def test_preview_returns_url_and_decodable_wav(monkeypatch):
    """Превью отдаёт ссылку, а по ссылке — читаемый WAV, а не пустышку."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            take = (await _upload(client, project["id"], 0, seconds=1.0))["take"]

            response = await client.post(
                f"/api/recording-projects/{project['id']}/takes/{take['id']}/preview",
                json={"speed": 1.0},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["url"], "без url интерфейсу нечего проигрывать"

            audio = await client.get(body["url"])
            assert audio.status_code == 200, audio.text
            assert _duration(audio.content) > 0, "превью декодируется в пустую дорожку"

    _run(scenario)


def test_preview_speed_changes_duration(monkeypatch):
    """Превью считается с запрошенными настройками: 1.2 обязано быть короче 1.0."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            take = (await _upload(client, project["id"], 0, seconds=1.0))["take"]
            url = f"/api/recording-projects/{project['id']}/takes/{take['id']}/preview"

            normal = await client.post(url, json={"speed": 1.0})
            fast = await client.post(url, json={"speed": 1.2})
            assert normal.status_code == 200 and fast.status_code == 200

            normal_len = _duration((await client.get(normal.json()["url"])).content)
            fast_len = _duration((await client.get(fast.json()["url"])).content)
            assert fast_len < normal_len * 0.95, "скорость не применилась к превью"

    _run(scenario)


# --- 26. Перезагрузка с диска ------------------------------------------------
def test_project_reload_keeps_takes_and_role_assignments(monkeypatch):
    """После сброса хранилища проект читается с диска: дубли и роли на месте."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            profile = (
                await client.post(
                    f"/api/recording-projects/{project['id']}/voice-profiles",
                    json={"name": "Низкий", "speed": 1.1, "pitch_semitones": -2.0},
                )
            ).json()["voice_profiles"][0]
            assigned = await client.put(
                f"/api/recording-projects/{project['id']}/roles/Анна",
                json={"profile_id": profile["id"]},
            )
            assert assigned.status_code == 200, assigned.text
            take = (await _upload(client, project["id"], 0))["take"]
            assert take["voice_profile_id"] == profile["id"], "дубль не унаследовал голос роли"

            # Сброс singleton'а равносилен перезапуску backend: следующий запрос
            # читает проект заново с диска.
            recording_store.reset_store()
            reloaded = await client.get(f"/api/recording-projects/{project['id']}")
            assert reloaded.status_code == 200, reloaded.text
            body = reloaded.json()
            assert [item["id"] for item in body["takes"]] == [take["id"]]
            assert body["active_takes"]["0"] == take["id"]
            assert body["role_voices"] == {"Анна": profile["id"]}
            assert body["voice_profiles"][0]["speed"] == pytest.approx(1.1)

            # И независимый объект хранилища видит то же самое состояние.
            from_disk = recording_store.RecordingStore(config.RECORDINGS_DIR).require_project(
                project["id"]
            )
            assert from_disk.active_take(0).id == take["id"]

    _run(scenario)


# --- 27. Список и удаление ---------------------------------------------------
def test_project_appears_in_list_and_delete_removes_it(monkeypatch):
    """Проект виден в списке, DELETE убирает его и каталог, повторный GET — 404."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            listed = await client.get("/api/recording-projects")
            assert listed.status_code == 200
            assert project["id"] in [item["id"] for item in listed.json()["projects"]]

            deleted = await client.delete(f"/api/recording-projects/{project['id']}")
            assert deleted.status_code == 200, deleted.text
            assert (await client.get(f"/api/recording-projects/{project['id']}")).status_code == 404

            listed = await client.get("/api/recording-projects")
            assert project["id"] not in [item["id"] for item in listed.json()["projects"]]
            assert not recording_store.get_store().project_dir(project["id"]).exists()

    _run(scenario)


# --- 28. Сырой дубль по ссылке -----------------------------------------------
def test_take_audio_returns_raw_wav_and_unknown_take_is_404(monkeypatch):
    """Аудио дубля — сырой WAV; несуществующий или чужой take_id не отдаётся."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create(client)
            take = (await _upload(client, project["id"], 0))["take"]

            response = await client.get(
                f"/api/recording-projects/{project['id']}/takes/{take['id']}/audio"
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("audio/wav")
            assert response.content[:4] == b"RIFF"

            missing = await client.get(
                f"/api/recording-projects/{project['id']}/takes/deadbeefdead/audio"
            )
            assert missing.status_code == 404

            # «Чужой» дубль: id существует, но принадлежит другому проекту.
            other = await _create(client, name="Другой")
            foreign = await client.get(
                f"/api/recording-projects/{other['id']}/takes/{take['id']}/audio"
            )
            assert foreign.status_code == 404

    _run(scenario)
