"""Редактор реплик: правки одной реплики, варианты и пересинтез.

Проверяется визуальный режим фазы 2: реплика правится по одной (голос, скорость,
сброс к наследуемому), у неё есть варианты звучания, а исходный DSL продолжает
разбираться как раньше. Модель подменена заглушкой: тест про API, наследование
параметров и изоляцию правок, а не про синтез.
"""

import asyncio
import contextlib
import sqlite3
import time
from pathlib import Path

import httpx
import pytest
import soundfile as sf

from backend import audio_pipeline, config, main
from backend.db import connection as db_connection
from backend.db.migrations import MIGRATIONS, _MIGRATION_1
from backend.engines.base import ENGINE_XTTS, SAMPLE_RATE
from backend.job_queue import JobQueue
from backend.voices_store import Voice
from conftest import STUB_ENGINE_ID, sine

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика.\nИВАН: Третья реплика."
# Слоты и имена в одном тексте: «(1)» без имени, «МАРГО:», метка «АРТЕМ(2):» и
# снова слот посреди строки.
MIXED_DIALOGUE = "(1) Без имени.\nМАРГО: С именем.\nАРТЕМ(2): Метка со слотом.\n(2) Снова слот два."
DSL_DIALOGUE = "(1 speed=1.2 cfg=2.5 nfe=16) Быстрее и выразительнее.\nИВАН: Обычная реплика."


class _Store:
    """Хранилище из нескольких голосов: правке реплики есть куда переключаться."""

    def __init__(self, *voices: Voice) -> None:
        self._voices = {voice.id: voice for voice in voices}

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)


@pytest.fixture
def voices(workspace, voice, monkeypatch):
    """Два голоса разных движков: у карточки должно быть что показать в «Engine»."""
    sf.write(workspace / "voices" / "ref2.wav", sine(2.0, 240.0), SAMPLE_RATE)
    second = Voice(
        id="voice2",
        name="Второй",
        gender="female",
        ref_text="Второй тест",
        audio_file="ref2.wav",
        engine=ENGINE_XTTS,
    )
    monkeypatch.setattr(audio_pipeline, "get_store", lambda: _Store(voice, second))
    return voice, second


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
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"задача {job_id} не завершилась")


async def _project(client, text: str, speakers: dict | None = None) -> dict:
    """Проект с разобранным текстом и назначенными голосами."""
    created = await client.post(
        "/api/projects", json={"name": "Редактор", "source_text": text, "mode": "dialogue"}
    )
    assert created.status_code == 201, created.text
    project = created.json()
    if speakers:
        patched = await client.patch(
            f"/api/projects/{project['id']}",
            json={"speakers": {key: {"voice_id": value} for key, value in speakers.items()}},
        )
        assert patched.status_code == 200, patched.text
    parsed = await client.post(f"/api/projects/{project['id']}/parse", json={})
    assert parsed.status_code == 200, parsed.text
    return parsed.json()


async def _render(client, project_id: str) -> dict:
    accepted = await client.post(f"/api/projects/{project_id}/render", json={})
    assert accepted.status_code == 202, accepted.text
    await _wait_job(client, accepted.json()["job_id"])
    return (await client.get(f"/api/projects/{project_id}")).json()


def _replica(project: dict, index: int) -> dict:
    return next(item for item in project["replicas"] if item["index"] == index)


# --- 1, 2: разбор текста и порядок реплик -------------------------------------
def test_parse_creates_replicas_in_source_order(voices, stub, monkeypatch):
    """Число реплик — по тексту, порядок — по тексту и стабильный при повторе."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(client, DIALOGUE)
            assert [item["index"] for item in project["replicas"]] == [0, 1, 2]
            assert [item["text"] for item in project["replicas"]] == [
                "Первая реплика.",
                "Вторая реплика.",
                "Третья реплика.",
            ]
            assert [item["speaker"] for item in project["replicas"]] == ["ИВАН", "МАРГО", "ИВАН"]

            # Повторный разбор не должен ни менять порядок, ни терять реплики.
            again = await client.post(f"/api/projects/{project['id']}/parse", json={})
            assert [item["text"] for item in again.json()["replicas"]] == [
                item["text"] for item in project["replicas"]
            ]

    _run(scenario)


# --- 3, 10: слоты, имена и смешанный синтаксис --------------------------------
def test_slots_and_names_are_assigned(voices, stub, monkeypatch):
    """Слоты и имена разбираются вместе, и каждому назначается свой голос."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(
                client,
                MIXED_DIALOGUE,
                speakers={"#1": "voice1", "МАРГО": "voice1", "#2": "voice2"},
            )
            assert [item["speaker"] for item in project["replicas"]] == [
                "#1",
                "МАРГО",
                "#2",
                "#2",
            ]
            assert [item["label"] for item in project["replicas"]] == [
                "Слот 1",
                "МАРГО",
                "Слот 2",
                "Слот 2",
            ]
            assert [item["voice_id"] for item in project["replicas"]] == [
                "voice1",
                "voice1",
                "voice2",
                "voice2",
            ]
            # Движок виден в карточке: у слота 2 голос на другом движке.
            assert [item["engine"] for item in project["replicas"]] == [
                STUB_ENGINE_ID,
                STUB_ENGINE_ID,
                ENGINE_XTTS,
                ENGINE_XTTS,
            ]

    _run(scenario)


# --- 4: правка голоса одной реплики -------------------------------------------
def test_voice_change_keeps_other_replicas(voices, stub, monkeypatch):
    """Свой голос реплики не задевает соседей и снимается сбросом."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(client, DIALOGUE, speakers={"ИВАН": "voice1", "МАРГО": "voice1"})
            patch = f"/api/projects/{project['id']}/replicas/1"

            changed = await client.patch(patch, json={"voice_id": "voice2"})
            assert changed.status_code == 200, changed.text
            updated = changed.json()["replica"]
            assert updated["voice_id"] == "voice2"
            assert updated["voice_override"] == "voice2"
            # Голос спикера остаётся доступен — к нему вернёт сброс.
            assert updated["inherited_voice_id"] == "voice1"
            assert updated["engine"] == ENGINE_XTTS

            project_after = (await client.get(f"/api/projects/{project['id']}")).json()
            assert [_replica(project_after, index)["voice_id"] for index in range(3)] == [
                "voice1",
                "voice2",
                "voice1",
            ]
            assert [_replica(project_after, index)["voice_override"] for index in range(3)] == [
                None,
                "voice2",
                None,
            ]

            # Сброс: `null` в поле голоса — «снова как у спикера», а не «без голоса».
            reset = await client.patch(patch, json={"voice_id": None})
            assert reset.json()["replica"]["voice_id"] == "voice1"
            assert reset.json()["replica"]["voice_override"] is None

    _run(scenario)


# --- 5: скорость применяется только к нужной реплике --------------------------
def test_speed_override_applies_to_one_replica(voices, stub, monkeypatch):
    """Правка скорости слышна в синтезе ровно у одной реплики."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(
                client, DIALOGUE, speakers={"ИВАН": "voice1", "МАРГО": "voice1"}
            )
            patch = f"/api/projects/{project['id']}/replicas/1"

            changed = await client.patch(patch, json={"overrides": {"speed": 1.4}})
            assert changed.status_code == 200, changed.text
            assert changed.json()["replica"]["overrides"]["speed"] == 1.4

            project_after = (await client.get(f"/api/projects/{project['id']}")).json()
            assert [_replica(project_after, index)["overrides"] for index in range(3)] == [
                {},
                {"speed": 1.4},
                {},
            ]

            stub.calls.clear()
            await _render(client, project["id"])
            assert [call["speed"] for call in stub.calls] == [
                config.DEFAULT_SPEED,
                1.4,
                config.DEFAULT_SPEED,
            ]

    _run(scenario)


# --- 6: сброс правки возвращает наследуемое значение --------------------------
def test_reset_override_returns_inherited_value(voices, stub, monkeypatch):
    """Сброс правки возвращает значение карточки спикера, а не дефолт движка."""

    async def scenario():
        async with _client(monkeypatch) as client:
            created = await client.post(
                "/api/projects",
                json={"name": "Наследование", "source_text": DIALOGUE, "mode": "dialogue"},
            )
            project_id = created.json()["id"]
            # Скорость задана у спикера: именно к ней должен вернуть сброс правки.
            await client.patch(
                f"/api/projects/{project_id}",
                json={
                    "speakers": {
                        "ИВАН": {"voice_id": "voice1", "overrides": {"speed": 1.1}},
                        "МАРГО": {"voice_id": "voice1", "overrides": {"speed": 1.1}},
                    }
                },
            )
            await client.post(f"/api/projects/{project_id}/parse", json={})

            patch = f"/api/projects/{project_id}/replicas/1"
            await client.patch(patch, json={"overrides": {"speed": 1.5}})
            stub.calls.clear()
            await _render(client, project_id)
            assert [call["speed"] for call in stub.calls] == [1.1, 1.5, 1.1]

            reset = await client.patch(patch, json={"overrides": {"speed": None}})
            assert reset.status_code == 200, reset.text
            assert reset.json()["replica"]["overrides"] == {}

            stub.calls.clear()
            await _render(client, project_id)
            assert [call["speed"] for call in stub.calls] == [1.1, 1.1, 1.1]

            # Сброс всех правок разом — то же самое, что сброс по одному.
            await client.patch(patch, json={"overrides": {"speed": 1.5, "gain_db": 3}})
            cleared = await client.patch(patch, json={"reset_overrides": True})
            assert cleared.json()["replica"]["overrides"] == {}

    _run(scenario)


# --- 7, 8: пересинтез реплики и выбор варианта --------------------------------
def test_regenerate_touches_one_replica(voices, stub, monkeypatch):
    """Пересинтез добавляет вариант выбранной реплике, соседние не меняются."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(
                client, DIALOGUE, speakers={"ИВАН": "voice1", "МАРГО": "voice1"}
            )
            rendered = await _render(client, project["id"])
            before = [_replica(rendered, index) for index in range(3)]
            assert [len(item["takes"]) for item in before] == [1, 1, 1]
            assert [item["takes"][0]["active"] for item in before] == [True, True, True]

            accepted = await client.post(
                f"/api/projects/{project['id']}/replicas/1/regenerate"
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])

            after = (await client.get(f"/api/projects/{project['id']}")).json()
            assert [len(_replica(after, index)["takes"]) for index in range(3)] == [1, 2, 1]
            for index in (0, 2):
                # Соседние реплики: тот же активный вариант и тот же список целиком.
                assert _replica(after, index)["takes"] == before[index]["takes"]
                assert _replica(after, index)["selected_take_id"] == before[index]["selected_take_id"]

            # У пересинтезированной реплики новый вариант стал активным, прежний
            # остался в списке — есть с чем сравнить на слух.
            regenerated = _replica(after, 1)
            assert len(regenerated["takes"]) == 2
            active = next(item for item in regenerated["takes"] if item["active"])
            assert active["id"] == regenerated["selected_take_id"]
            assert active["id"] != before[1]["takes"][0]["id"]
            assert [item["active"] for item in regenerated["takes"]] == [False, True]
            # Файл варианта отдаётся на прослушивание.
            assert (await client.get(active["audio_url"])).status_code == 200

            # Плеер аудио доступен и по ссылке из карточки: интерфейс не строит URL сам.
            for item in regenerated["takes"]:
                assert item["audio_url"].endswith(f"/takes/{item['id']}/audio")
                assert (await client.get(item["audio_url"])).status_code == 200

    _run(scenario)


def test_take_can_be_selected_without_synthesis(voices, stub, monkeypatch):
    """Возврат к прежнему звучанию — выбор готового файла, а не новый синтез."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(
                client, DIALOGUE, speakers={"ИВАН": "voice1", "МАРГО": "voice1"}
            )
            rendered = await _render(client, project["id"])
            original = _replica(rendered, 1)["takes"][0]["id"]

            accepted = await client.post(f"/api/projects/{project['id']}/replicas/1/regenerate")
            await _wait_job(client, accepted.json()["job_id"])
            synthesized = len(stub.calls)

            selected = await client.post(
                f"/api/projects/{project['id']}/replicas/1/takes/{original}"
            )
            assert selected.status_code == 200, selected.text
            replica = selected.json()["replica"]
            assert replica["selected_take_id"] == original
            assert [item["active"] for item in replica["takes"]] == [True, False]
            # Синтеза не было: выбор варианта — это копирование готового файла.
            assert len(stub.calls) == synthesized

            unknown = await client.post(
                f"/api/projects/{project['id']}/replicas/1/takes/99999"
            )
            assert unknown.status_code == 404

    _run(scenario)


# --- 9: исходный DSL продолжает работать ---------------------------------------
def test_source_dsl_still_parses(voices, stub, monkeypatch):
    """Маркеры `(N speed=...)` остаются рабочим способом задать параметры."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(client, DSL_DIALOGUE, speakers={"#1": "voice1", "ИВАН": "voice1"})
            assert [item["text"] for item in project["replicas"]] == [
                "Быстрее и выразительнее.",
                "Обычная реплика.",
            ]
            assert _replica(project, 0)["overrides"] == {
                "speed": 1.2,
                "cfg_strength": 2.5,
                "nfe_step": 16,
            }
            assert _replica(project, 1)["overrides"] == {}

            stub.calls.clear()
            await _render(client, project["id"])
            assert [call["speed"] for call in stub.calls] == [1.2, config.DEFAULT_SPEED]
            assert stub.calls[0]["params"]["cfg_strength"] == 2.5
            assert stub.calls[0]["params"]["nfe_step"] == 16

    _run(scenario)


# --- ошибки запроса ------------------------------------------------------------
def test_unknown_replica_and_take_return_404(voices, stub, monkeypatch):
    """Отсутствие реплики или варианта — ошибка запроса, а не пустая карточка."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(client, DIALOGUE, speakers={"ИВАН": "voice1"})
            base = f"/api/projects/{project['id']}"
            assert (await client.patch(f"{base}/replicas/7", json={"overrides": {"speed": 1.2}})).status_code == 404
            assert (await client.post(f"{base}/replicas/7/regenerate")).status_code == 404
            assert (await client.post(f"{base}/replicas/0/takes/1")).status_code == 404
            assert (await client.get(f"{base}/replicas/0/takes/1/audio")).status_code == 404
            # Реплика без голоса: пересинтез не с чего начинать.
            assert (await client.post(f"{base}/replicas/1/regenerate")).status_code == 400

    _run(scenario)


# --- миграция схемы ------------------------------------------------------------
def test_migration_adds_voice_override_to_existing_db(workspace, monkeypatch):
    """База первой фазы обновляется до v2: колонка появляется, не теряя данные."""
    path = config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(path)
    try:
        legacy.executescript(_MIGRATION_1)
        legacy.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        legacy.execute("INSERT INTO schema_version (version) VALUES (1)")
        legacy.execute(
            "INSERT INTO projects (id, name, created_at, updated_at) VALUES ('p1', 'Старый', '', '')"
        )
        legacy.commit()
    finally:
        legacy.close()

    # Кеш «схема уже проверена» относится к другому пути — сбрасываем, иначе
    # миграции не выполнятся вовсе.
    monkeypatch.setattr(db_connection, "_initialized_path", None)
    db_connection.init_db()

    probe = sqlite3.connect(path)
    try:
        columns = {row[1] for row in probe.execute("PRAGMA table_info(replicas)")}
        version = probe.execute("SELECT version FROM schema_version").fetchone()[0]
        name = probe.execute("SELECT name FROM projects WHERE id = 'p1'").fetchone()[0]
    finally:
        probe.close()
    assert "voice_override" in columns
    assert version == MIGRATIONS[-1][0]
    assert name == "Старый"


def test_project_take_files_are_kept_in_project_dir(voices, stub, workspace, monkeypatch):
    """Файл варианта лежит в каталоге проекта и удаляется вместе с ним."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project(
                client, DIALOGUE, speakers={"ИВАН": "voice1", "МАРГО": "voice1"}
            )
            await _render(client, project["id"])
            accepted = await client.post(f"/api/projects/{project['id']}/replicas/0/regenerate")
            await _wait_job(client, accepted.json()["job_id"])
            data = (await client.get(f"/api/projects/{project['id']}")).json()
            paths = [
                take["audio_path"]
                for replica in data["replicas"]
                for take in replica["takes"]
            ]
            assert len(paths) == 4
            project_dir = Path(audio_pipeline.config.PROJECTS_OUTPUT_DIR) / project["id"]
            for raw in paths:
                assert Path(raw).parent == project_dir
                assert Path(raw).exists()

            await client.delete(f"/api/projects/{project['id']}")
            for raw in paths:
                assert not Path(raw).exists()

    _run(scenario)
