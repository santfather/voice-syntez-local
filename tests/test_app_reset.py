"""Сброс приложения: `POST /api/reset`.

Проверяется граница, ради которой сброс разделён на ступени: `cache` освобождает
только транзитные файлы и не трогает вкладку, `full` добавляет к ним вкладку —
открытый проект диалога (вместе с архивами его диагностики) либо текст и настройки
сплошного текста. Неприкосновенными обязаны остаться записанные голоса, словарь
произношения и проекты, которых пользователь не открывал, — это и есть критерий
готовности через API.

Прогон не трогает рабочие файлы: каталоги уведены в `tmp_path` фикстурой
`workspace`, а путь к настройкам анализатора — переменной окружения, иначе сброс
удалил бы `data/llm_settings.json` самой машины.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import httpx
import pytest

from backend import cache_cleanup, config, diagnostics, engine_lifecycle, main
from backend.llm import settings_store
from backend.pronunciation import get_store as get_pronunciation_store

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."
MB = 1024 * 1024


def write(path: Path, size: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


@pytest.fixture
def llm_settings(tmp_path, monkeypatch):
    """Файл настроек анализатора — в `tmp_path`: сброс его удаляет."""
    path = tmp_path / "llm_settings.json"
    path.write_text('{"enabled": true}', encoding="utf-8")
    monkeypatch.setenv(settings_store.SETTINGS_ENV, str(path))
    return path


def archive_paths(project: dict, count: int = 1) -> list[Path]:
    """Файлы архивов проекта — по тому же префиксу, что строит `diagnostics`."""
    prefix = diagnostics._archive_prefix(project)
    return [
        write(config.DIAGNOSTICS_DIR / f"{prefix}20260101-00000{index}.zip")
        for index in range(count)
    ]


@contextlib.asynccontextmanager
async def _client():
    """ASGI-клиент без запуска очереди: сброс в ней не нуждается."""
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _create_project(client, name: str = "Первый") -> dict:
    response = await client.post(
        "/api/projects", json={"name": name, "source_text": DIALOGUE, "mode": "dialogue"}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _reset(client, **body) -> dict:
    response = await client.post("/api/reset", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_cache_scope_frees_files_and_keeps_open_project(workspace):
    """Первая ступень: транзитные файлы уходят, проект остаётся на месте."""
    write(config.OUTPUT_DIR / "job1.mp3", 2 * MB)
    write(config.BENCHMARKS_DIR / "abc123-f5.wav", 3 * MB)

    async def scenario():
        async with _client() as client:
            project = await _create_project(client)
            report = await _reset(client, scope="cache", tab="dialogue", project_id=project["id"])
            assert set(report) == {
                "scope", "tab", "cache", "project_deleted",
                "diagnostics_deleted", "llm_settings_reset",
            }
            assert report["scope"] == "cache"
            assert report["tab"] == "dialogue"
            assert report["cache"]["total_mb"] == 5.0
            # Ступень кеша не удаляет ни данных, ни настроек: в отчёте это видно
            # как «ничего не произошло», а не как молчание.
            assert report["project_deleted"] is None
            assert report["diagnostics_deleted"] == 0
            assert report["llm_settings_reset"] is False
            assert (await client.get(f"/api/projects/{project['id']}")).status_code == 200

    _run(scenario)
    assert not (config.OUTPUT_DIR / "job1.mp3").exists()
    assert not (config.BENCHMARKS_DIR / "abc123-f5.wav").exists()


def test_full_reset_dialogue_deletes_only_open_project(workspace, llm_settings):
    """Вторая ступень: уходит открытый проект вместе с архивами, соседний цел."""
    async def scenario():
        async with _client() as client:
            opened = await _create_project(client, name="Открытый")
            other = await _create_project(client, name="Соседний")
            opened_archives = archive_paths(opened, count=2)
            other_archives = archive_paths(other)
            # Данные проекта: каталог с take'ами и транзитный файл кеша.
            takes = write(config.PROJECTS_OUTPUT_DIR / opened["id"] / "take-1.wav")
            cached = write(config.OUTPUT_DIR / "job1.mp3", 2 * MB)

            report = await _reset(client, scope="full", tab="dialogue", project_id=opened["id"])
            assert report["project_deleted"] == opened["id"]
            assert report["diagnostics_deleted"] == len(opened_archives)
            assert report["llm_settings_reset"] is True
            assert report["cache"]["total_mb"] == 2.0

            assert (await client.get(f"/api/projects/{opened['id']}")).status_code == 404
            assert (await client.get(f"/api/projects/{other['id']}")).status_code == 200

        assert not takes.exists()
        assert not cached.exists()
        for path in opened_archives:
            assert not path.exists(), "архивы удалённого проекта обязаны уйти с ним"
        for path in other_archives:
            assert path.is_file(), "архивы чужого проекта сброс не трогает"

    _run(scenario)
    assert not llm_settings.exists(), "сброшенные настройки — это удалённый файл"


def test_full_reset_dialogue_without_open_project_is_not_an_error(workspace, llm_settings):
    """Сбрасывать нечего — это нормальный случай, а не 404 и не 500."""

    async def scenario():
        async with _client() as client:
            report = await _reset(client, scope="full", tab="dialogue")
            assert report["scope"] == "full"
            assert report["project_deleted"] is None
            assert report["diagnostics_deleted"] == 0
            assert report["llm_settings_reset"] is True

            # Повторный сброс: файла настроек уже нет — «сброшено» сменилось на
            # честное «сбрасывать было нечего».
            again = await _reset(client, scope="full", tab="dialogue")
            assert again["llm_settings_reset"] is False

    _run(scenario)
    assert not llm_settings.exists()


def test_full_reset_text_keeps_project_voices_and_dictionary(workspace, llm_settings):
    """Критерий пользователя: голоса и словарь целы, проект не тронут."""
    voices_json = write(config.VOICES_JSON, 256)
    reference = write(config.VOICES_DIR / "ref.wav", 512)
    store = get_pronunciation_store()
    store.create("SQL", "эскьюэль")

    async def scenario():
        async with _client() as client:
            project = await _create_project(client)
            archives = archive_paths(project)
            write(config.OUTPUT_DIR / "job1.mp3", MB)

            report = await _reset(client, scope="full", tab="text", project_id=project["id"])
            # Вкладка сплошного текста проект не удаляет: он принадлежит диалогу.
            assert report["project_deleted"] is None
            assert report["diagnostics_deleted"] == 0
            assert report["llm_settings_reset"] is True
            assert report["cache"]["total_mb"] == 1.0
            assert (await client.get(f"/api/projects/{project['id']}")).status_code == 200
            for path in archives:
                assert path.is_file()

    _run(scenario)
    assert voices_json.is_file()
    assert reference.is_file()
    assert [entry["source"] for entry in store.list_entries()] == ["SQL"]
    assert not llm_settings.exists()


def test_unknown_scope_and_tab_are_400(workspace):
    """Расхождение клиента и модуля — 400, и в тексте перечислены допустимые."""

    async def scenario():
        async with _client() as client:
            scope = await client.post("/api/reset", json={"scope": "everything", "tab": "text"})
            assert scope.status_code == 400, scope.text
            detail = scope.json()["detail"]
            for allowed in ("cache", "full"):
                assert allowed in detail

            tab = await client.post("/api/reset", json={"scope": "full", "tab": "voices"})
            assert tab.status_code == 400, tab.text
            assert "voices" in tab.json()["detail"]

            missing = await client.post("/api/reset", json={"scope": "full"})
            assert missing.status_code == 400, missing.text

    _run(scenario)


def test_full_reset_refused_while_queue_busy(workspace, llm_settings, monkeypatch):
    """Пока рендер в очереди, полный сброс удалял бы реплики из-под него — 409."""
    monkeypatch.setattr(engine_lifecycle, "queue_busy_now", lambda: True)

    async def scenario():
        async with _client() as client:
            project = await _create_project(client)
            response = await client.post(
                "/api/reset",
                json={"scope": "full", "tab": "dialogue", "project_id": project["id"]},
            )
            assert response.status_code == 409, response.text
            assert "очеред" in response.json()["detail"].lower()
            # Отказ ничего не удалил: и проект, и файл настроек на месте.
            assert (await client.get(f"/api/projects/{project['id']}")).status_code == 200
            assert llm_settings.is_file()

            # Ступень кеша занятости не боится: она не трогает ни текст, ни take'ы.
            allowed = await client.post("/api/reset", json={"scope": "cache", "tab": "dialogue"})
            assert allowed.status_code == 200, allowed.text

    _run(scenario)
    assert llm_settings.is_file()


def test_repeated_full_reset_is_idempotent(workspace, llm_settings):
    """Повторный сброс — тот же ответ по смыслу: удалять больше нечего."""
    write(config.OUTPUT_DIR / "job1.mp3", MB)

    async def scenario():
        async with _client() as client:
            project = await _create_project(client)
            first = await _reset(client, scope="full", tab="dialogue", project_id=project["id"])
            second = await _reset(client, scope="full", tab="dialogue", project_id=project["id"])
            assert first["cache"]["total_mb"] == 1.0
            assert second["cache"]["total_mb"] == 0.0
            assert second["project_deleted"] is None
            assert second["llm_settings_reset"] is False

    _run(scenario)


def test_cache_scope_accepts_all_cleanup_targets(workspace):
    """Ступень кеша чистит ровно то же, что ручная очистка вкладки «Модели»."""

    async def scenario():
        async with _client() as client:
            report = await _reset(client, scope="cache")
            assert sorted(report["cache"]["freed_mb"]) == sorted(cache_cleanup.TARGETS)
            assert report["cache"]["total_mb"] == 0.0

    _run(scenario)
