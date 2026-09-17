"""REST очистки кеша приложения (Задача 3): `GET /api/cache`, `POST /api/cache/clear`.

Проверяется контракт роутов: инвентарь по категориям, форма ответа очистки,
понятные 400 на неизвестную категорию и пустой выбор, «голый список» в теле и
сброс реестра сравнения движков после очистки `benchmarks`. Эндпоинт не должен
поднимать модели: движки в этих тестах не создаются вовсе.

Каталоги уведены в `tmp_path` фикстурой `workspace`; системный temp — тоже,
поэтому прогон не трогает рабочие файлы машины.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path

import httpx
import pytest

from backend import benchmark, cache_cleanup, config, main

MB = 1024 * 1024


def write(path: Path, size: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


@pytest.fixture(autouse=True)
def clean_registry():
    """Реестр сравнений живёт в памяти модуля — каждый тест начинает с пустого."""
    benchmark.reset_registry()
    yield
    benchmark.reset_registry()


@pytest.fixture
def system_temp(workspace, monkeypatch):
    temp_dir = workspace / "system-temp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
    return temp_dir


@pytest.fixture
def files(workspace):
    """Файлы во всех трёх категориях: 2 МБ output, 3 МБ benchmarks, 1 МБ temp."""
    write(config.OUTPUT_DIR / "job1.mp3", 2 * MB)
    write(config.BENCHMARKS_DIR / "abc123-f5.wav", 3 * MB)
    write(config.PROJECTS_OUTPUT_DIR / "project1" / "take-1.wav.tmp.wav", 1 * MB)
    return workspace


@contextlib.asynccontextmanager
async def _client():
    """ASGI-клиент без запуска очереди: роуты кеша в ней не нуждаются."""
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


def test_cache_inventory_endpoint_reports_categories(files):
    async def scenario():
        async with _client() as client:
            response = await client.get("/api/cache")
            assert response.status_code == 200, response.text
            body = response.json()
            assert set(body) == {
                "output_mb", "benchmarks_mb", "temp_files_mb", "temp_files_count",
            }
            assert body["output_mb"] == 2.0
            assert body["benchmarks_mb"] == 3.0
            assert body["temp_files_mb"] == 1.0
            assert body["temp_files_count"] == 1
            assert all(value >= 0 for value in body.values())

    _run(scenario)


def test_inventory_endpoint_does_not_delete_or_create_engines(files, monkeypatch):
    def explode(engine_id):
        raise AssertionError(f"инвентарь не должен создавать движок {engine_id}")

    monkeypatch.setattr("backend.engines.registry.get_engine", explode)

    async def scenario():
        async with _client() as client:
            await client.get("/api/cache")

    _run(scenario)
    assert (config.OUTPUT_DIR / "job1.mp3").is_file()
    assert (config.BENCHMARKS_DIR / "abc123-f5.wav").is_file()


def test_clear_endpoint_object_body_returns_freed_report(files):
    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/cache/clear",
                json={"targets": ["output", "benchmarks", "temp_files"]},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert set(body) == {"freed_mb", "freed_files", "total_mb"}
            assert body["freed_mb"] == {
                "output": 2.0, "benchmarks": 3.0, "temp_files": 1.0,
            }
            assert body["freed_files"] == {"output": 1, "benchmarks": 1, "temp_files": 1}
            assert body["total_mb"] == 6.0

    _run(scenario)
    assert not (config.OUTPUT_DIR / "job1.mp3").exists()
    assert not (config.BENCHMARKS_DIR / "abc123-f5.wav").exists()
    assert not (
        config.PROJECTS_OUTPUT_DIR / "project1" / "take-1.wav.tmp.wav"
    ).exists()


def test_clear_endpoint_accepts_bare_list(files):
    """Набросок эндпоинта в постановке принимает «голый список» — он работает."""

    async def scenario():
        async with _client() as client:
            response = await client.post("/api/cache/clear", json=["output"])
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["freed_mb"] == {"output": 2.0}
            assert body["freed_files"] == {"output": 1}

    _run(scenario)
    assert not (config.OUTPUT_DIR / "job1.mp3").exists()
    # Другие категории не тронуты: список означал ровно одну категорию.
    assert (config.BENCHMARKS_DIR / "abc123-f5.wav").is_file()


def test_clear_endpoint_resets_benchmark_registry(files):
    benchmark.register(
        benchmark.BenchmarkRun(id="abc123", voice_id="voice1", text="Фраза", engines=["f5"])
    )
    assert benchmark.get_run("abc123") is not None

    async def scenario():
        async with _client() as client:
            response = await client.post("/api/cache/clear", json={"targets": ["benchmarks"]})
            assert response.status_code == 200, response.text
            # Интерфейс не должен получить битую ссылку после очистки.
            run = await client.get("/api/benchmarks/abc123")
            assert run.status_code == 404, run.text

    _run(scenario)
    assert benchmark.list_runs() == []


def test_unknown_category_is_400(files):
    async def scenario():
        async with _client() as client:
            response = await client.post("/api/cache/clear", json={"targets": ["models"]})
            assert response.status_code == 400, response.text
            detail = response.json()["detail"]
            assert "Неизвестная категория" in detail and "models" in detail
            # Доступные категории перечислены: ошибка говорит, что делать.
            for target in cache_cleanup.TARGETS:
                assert target in detail

            listed = await client.post("/api/cache/clear", json=["data", "voices"])
            assert listed.status_code == 400
            assert "data" in listed.json()["detail"]

    _run(scenario)
    assert (config.OUTPUT_DIR / "job1.mp3").is_file()


def test_empty_selection_is_400(files):
    async def scenario():
        async with _client() as client:
            for payload in ({"targets": []}, [], {}):
                response = await client.post("/api/cache/clear", json=payload)
                assert response.status_code == 400, (payload, response.text)
                assert "не выбрано" in response.json()["detail"].lower()

    _run(scenario)
    assert (config.OUTPUT_DIR / "job1.mp3").is_file()


def test_broken_body_is_400(files):
    async def scenario():
        async with _client() as client:
            empty = await client.post(
                "/api/cache/clear", content=b"", headers={"Content-Type": "application/json"}
            )
            assert empty.status_code == 400, empty.text
            assert "пуст" in empty.json()["detail"].lower()

            broken = await client.post(
                "/api/cache/clear", content=b"{not json", headers={"Content-Type": "application/json"}
            )
            assert broken.status_code == 400, broken.text
            assert "JSON" in broken.json()["detail"]

            wrong_type = await client.post("/api/cache/clear", json={"targets": "output"})
            assert wrong_type.status_code == 400, wrong_type.text
            assert "списком" in wrong_type.json()["detail"]

            not_strings = await client.post("/api/cache/clear", json={"targets": [1, 2]})
            assert not_strings.status_code == 400, not_strings.text

    _run(scenario)
    assert (config.OUTPUT_DIR / "job1.mp3").is_file()


def test_repeated_clear_is_idempotent(files):
    async def scenario():
        async with _client() as client:
            first = await client.post("/api/cache/clear", json={"targets": ["output"]})
            second = await client.post("/api/cache/clear", json={"targets": ["output"]})
            assert first.status_code == second.status_code == 200
            assert first.json()["total_mb"] == 2.0
            assert second.json() == {
                "freed_mb": {"output": 0.0},
                "freed_files": {"output": 0},
                "total_mb": 0.0,
            }
            # Инвентарь после очистки согласован с тем, что реально освободилось.
            assert (await client.get("/api/cache")).json()["output_mb"] == 0.0

    _run(scenario)


def test_clear_and_inventory_with_nothing_to_clean(workspace):
    """Пустые каталоги — не ошибка: нули по всем категориям и никаких исключений."""

    async def scenario():
        async with _client() as client:
            assert (await client.get("/api/cache")).json()["output_mb"] == 0.0
            response = await client.post(
                "/api/cache/clear", json={"targets": list(cache_cleanup.TARGETS)}
            )
            assert response.status_code == 200, response.text
            assert response.json()["total_mb"] == 0.0

    _run(scenario)


def test_clear_keeps_data_voices_and_models(files, workspace, monkeypatch):
    """Критерий готовности через API: все три чекбокса — и данные проекта целы."""
    data_dir = workspace / "data"
    models_dir = workspace / "models"
    monkeypatch.setattr(config, "DB_PATH", data_dir / "voice_syntez.db")
    from backend.pronunciation import get_store as get_pronunciation_store

    get_pronunciation_store().create("SQL", "эскьюэль")
    db_path = data_dir / "voice_syntez.db"
    voices_json = write(config.VOICES_JSON, 256)
    reference = write(config.VOICES_DIR / "ref.wav", 512)
    weights = write(models_dir / "xtts_v2" / "model.pth", 1024)
    hf_cache = write(models_dir / "huggingface" / "hub" / "blob.bin", 1024)

    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/cache/clear", json={"targets": list(cache_cleanup.TARGETS)}
            )
            assert response.status_code == 200, response.text
            assert response.json()["total_mb"] > 0

    _run(scenario)
    for path in (db_path, voices_json, reference, weights, hf_cache):
        assert path.is_file(), f"{path} обязан пережить очистку через API"
    assert get_pronunciation_store().list_entries()[0]["target"] == "эскьюэль"
