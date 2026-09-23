"""REST менеджера моделей (ФАЗА 9): список, прогресс, скачивание и удаление.

Сеть не трогается: точка скачивания подменена заглушкой, которая пишет файлы в
`tmp_path`. Веса для этого теста тоже не нужны — каталог моделей уведён в tmp.
"""

import asyncio
import contextlib
import time

import httpx
import pytest
from conftest import StubEngine

from backend import config, main, model_manager
from backend.engines import registry

F5_ID = "f5"
XTTS_ID = "xtts"
BANANA_ID = "xtts-banana"
QWEN_BASE_ID = "qwen3-tts-base"
QWEN_TOKENIZER_ID = "qwen3-tts-tokenizer"
KOKORO_ID = "kokoro-ru"


def write(path, size: int = 64):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class FakeDownloader:
    """Заглушка `_hf_download`: кладёт файл на место, в сеть не ходит."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo_id: str, filename: str, local_dir):
        self.calls.append((repo_id, filename))
        if self.fail_on == filename:
            raise RuntimeError("HTTP 503: сервис недоступен")
        target = local_dir / filename
        write(target)
        return target


@contextlib.asynccontextmanager
async def _client():
    """ASGI-клиент без запуска очереди: роуты моделей в ней не нуждаются."""
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def models_dir(workspace, monkeypatch):
    root = workspace / "models"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", root)
    monkeypatch.setattr(config, "XTTS_BASE_DIR", root / "xtts_v2")
    monkeypatch.setattr(config, "XTTS_BANANA_DIR", root / "xtts_v2_banana")
    monkeypatch.setattr(config, "CKPT_FILE", root / config.HF_CKPT_PATH)
    monkeypatch.setattr(config, "VOCAB_FILE", root / config.HF_VOCAB_PATH)
    return root


@pytest.fixture(autouse=True)
def no_created_engines(monkeypatch):
    monkeypatch.setattr(registry, "_instances", {})


@pytest.fixture
def downloader(monkeypatch):
    fake = FakeDownloader()
    monkeypatch.setattr(model_manager, "_hf_download", fake)
    return fake


def _run(scenario) -> None:
    asyncio.run(scenario())


def install_f5(root) -> None:
    write(root / config.HF_CKPT_PATH)
    write(root / config.HF_VOCAB_PATH)


# --- 9. API status ------------------------------------------------------------
def test_models_endpoint_lists_state_and_disk(models_dir, downloader):
    install_f5(models_dir)

    async def scenario():
        async with _client() as client:
            response = await client.get("/api/models")
            assert response.status_code == 200, response.text
            body = response.json()
            assert {"models", "disk"} <= set(body)
            models = {item["id"]: item for item in body["models"]}
            assert {
                F5_ID,
                XTTS_ID,
                BANANA_ID,
                QWEN_BASE_ID,
                QWEN_TOKENIZER_ID,
                KOKORO_ID,
                "whisper",
            } == set(models)
            assert models[F5_ID]["installed"] is True
            assert models[XTTS_ID]["installed"] is False
            assert models[XTTS_ID]["missing_files"] == ["model.pth", "config.json"]
            assert models[XTTS_ID]["size_bytes"] == 0
            assert models[F5_ID]["loaded"] is False
            assert models["whisper"]["kind"] == "cache"
            assert set(body["disk"]) == {"models_bytes", "free_bytes", "total_bytes"}
            assert body["disk"]["free_bytes"] > 0

    _run(scenario)


def test_models_endpoint_reports_exists_and_size(models_dir, downloader):
    write(models_dir / "xtts_v2" / "model.pth", size=1000)
    write(models_dir / "xtts_v2" / "config.json", size=200)

    async def scenario():
        async with _client() as client:
            response = await client.get(f"/api/models/{XTTS_ID}")
            assert response.status_code == 200
            body = response.json()
            assert body["installed"] is True
            assert body["size_bytes"] == 1200
            assert body["missing_files"] == []
            assert body["path"] == str(config.XTTS_BASE_DIR)

    _run(scenario)


def test_unknown_model_id_is_404(models_dir, downloader):
    async def scenario():
        async with _client() as client:
            for method, url in (
                ("get", "/api/models/no-such-model"),
                ("post", "/api/models/no-such-model/download"),
                ("delete", "/api/models/no-such-model"),
            ):
                response = await getattr(client, method)(url)
                assert response.status_code == 404, (method, response.text)
                assert "не найдена" in response.json()["detail"]

    _run(scenario)


# --- 3. download success через API --------------------------------------------
async def _wait_done(client, model_id: str, timeout: float = 5.0) -> dict:
    """Опрос прогресса, как это делает вкладка: до `done` или до таймаута."""
    deadline = time.monotonic() + timeout
    body = {}
    while time.monotonic() < deadline:
        body = (await client.get(f"/api/models/{model_id}")).json()
        if body["download"]["state"] == "done":
            return body
        await asyncio.sleep(0.01)
    return body


def test_download_route_returns_202_and_finishes(models_dir, downloader):
    async def scenario():
        async with _client() as client:
            response = await client.post(f"/api/models/{XTTS_ID}/download")
            assert response.status_code == 202, response.text
            assert response.json()["download"]["state"] == "downloading"
            # Опрос прогресса тем же роутом, которым его опрашивает вкладка.
            final = await _wait_done(client, XTTS_ID)
            assert final["installed"] is True
            assert final["download"]["progress"] == 1.0
            assert {call[1] for call in downloader.calls} == {"model.pth", "config.json"}

    _run(scenario)


def test_download_failure_shows_error_text(models_dir, monkeypatch):
    monkeypatch.setattr(model_manager, "_hf_download", FakeDownloader(fail_on="model.pth"))

    async def scenario():
        async with _client() as client:
            await client.post(f"/api/models/{XTTS_ID}/download")
            body = (await client.get(f"/api/models/{XTTS_ID}")).json()
            assert body["download"]["state"] == "error"
            assert body["download"]["error"]
            assert body["installed"] is False

    _run(scenario)


def test_download_of_installed_model_keeps_files(models_dir, monkeypatch):
    install_f5(models_dir)
    before = (models_dir / config.HF_CKPT_PATH).stat().st_mtime_ns

    def explode(*_args, **_kwargs):
        raise AssertionError("установленная модель не должна скачиваться заново")

    monkeypatch.setattr(model_manager, "_hf_download", explode)

    async def scenario():
        async with _client() as client:
            response = await client.post(f"/api/models/{F5_ID}/download")
            assert response.status_code == 202
            assert response.json()["installed"] is True
            assert response.json()["download"]["state"] == "done"
            assert (models_dir / config.HF_CKPT_PATH).stat().st_mtime_ns == before

    _run(scenario)


def test_cache_model_download_is_400(models_dir, downloader):
    async def scenario():
        async with _client() as client:
            response = await client.post("/api/models/whisper/download")
            assert response.status_code == 400, response.text
            assert "кеш" in response.json()["detail"]

    _run(scenario)


# --- 10. удаление: запрет при занятом движке -----------------------------------
class _LoadedEngine(StubEngine):
    def load(self) -> None:
        self._mark("ready")


def test_delete_allowed_when_engine_idle(models_dir, downloader):
    """У модели свой каталог (XTTS), движок выгружен — удаление проходит."""
    write(config.XTTS_BASE_DIR / "model.pth")
    write(config.XTTS_BASE_DIR / "config.json")

    async def scenario():
        async with _client() as client:
            response = await client.delete(f"/api/models/{XTTS_ID}")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["deleted"] == XTTS_ID
            assert body["freed_bytes"] > 0
            assert body["state"]["installed"] is False
            assert not config.XTTS_BASE_DIR.exists()

    _run(scenario)


def test_delete_of_f5_is_refused_because_its_dir_is_shared(models_dir, downloader):
    """F5 лежит в корне `models/`: её удаление стёрло бы веса остальных моделей."""
    install_f5(models_dir)
    write(config.XTTS_BASE_DIR / "model.pth")

    async def scenario():
        async with _client() as client:
            response = await client.delete(f"/api/models/{F5_ID}")
            assert response.status_code == 400, response.text
            detail = response.json()["detail"]
            assert "каталоге моделей" in detail
            assert "вручную" in detail
            # Чужие веса целы — отказ случился до удаления.
            assert (config.XTTS_BASE_DIR / "model.pth").is_file()
            assert (models_dir / config.HF_CKPT_PATH).is_file()
            listed = (await client.get(f"/api/models/{F5_ID}")).json()
            assert listed["installed"] is True

    _run(scenario)


def test_delete_refused_while_engine_loaded(models_dir, downloader, monkeypatch):
    write(config.XTTS_BASE_DIR / "model.pth")
    write(config.XTTS_BASE_DIR / "config.json")
    engine = _LoadedEngine()
    engine.info = registry.ENGINE_INFOS[XTTS_ID]
    engine.load()
    monkeypatch.setitem(registry._instances, XTTS_ID, engine)

    async def scenario():
        async with _client() as client:
            response = await client.delete(f"/api/models/{XTTS_ID}")
            assert response.status_code == 409, response.text
            detail = response.json()["detail"]
            assert "используется движком" in detail
            assert "очереди" in detail
            # Файлы целы: отказ случился до удаления.
            assert (config.XTTS_BASE_DIR / "model.pth").is_file()
            listed = (await client.get(f"/api/models/{XTTS_ID}")).json()
            assert listed["installed"] is True
            assert listed["engine_in_use"] is True

    _run(scenario)


def test_delete_refused_while_queue_uses_engine(models_dir, downloader, monkeypatch):
    write(config.XTTS_BASE_DIR / "model.pth")
    write(config.XTTS_BASE_DIR / "config.json")

    class _Queue:
        current_job_id = "job-1"

        def get(self, _job_id):
            job = type("J", (), {})()
            job.payload = type("P", (), {})()
            job.payload.replicas = [type("R", (), {"voice": "voice1"})()]
            return job

    class _Store:
        def get(self, _voice_id):
            return type("V", (), {"engine": XTTS_ID})()

    monkeypatch.setattr("backend.job_queue.get_queue", lambda: _Queue())
    monkeypatch.setattr("backend.voices_store.get_store", lambda: _Store())

    async def scenario():
        async with _client() as client:
            response = await client.delete(f"/api/models/{XTTS_ID}")
            assert response.status_code == 409, response.text
            assert (config.XTTS_BASE_DIR / "model.pth").is_file()

    _run(scenario)


def test_cache_model_delete_is_400(models_dir, downloader):
    async def scenario():
        async with _client() as client:
            response = await client.delete("/api/models/whisper")
            assert response.status_code == 400, response.text
            assert "кеш" in response.json()["detail"]

    _run(scenario)


# --- корректность состояния ---------------------------------------------------
def test_loaded_flag_comes_from_created_engines(models_dir, downloader, monkeypatch):
    write(config.XTTS_BASE_DIR / "model.pth")
    write(config.XTTS_BASE_DIR / "config.json")
    engine = _LoadedEngine()
    engine.info = registry.ENGINE_INFOS[XTTS_ID]
    engine.load()
    monkeypatch.setitem(registry._instances, XTTS_ID, engine)

    def fake_create(engine_id):
        raise AssertionError(f"движок {engine_id} не должен создаваться ради списка моделей")

    monkeypatch.setattr(registry, "_create", fake_create)

    async def scenario():
        async with _client() as client:
            body = (await client.get("/api/models")).json()
            models = {item["id"]: item for item in body["models"]}
            assert models[XTTS_ID]["loaded"] is True
            assert models[F5_ID]["loaded"] is False

    _run(scenario)


def test_invalid_model_path_is_400_and_files_untouched(models_dir, workspace, downloader, monkeypatch):
    outside = workspace / "outside"
    outside.mkdir()
    victim = write(outside / "model.pth")
    monkeypatch.setattr(config, "XTTS_BASE_DIR", outside)

    async def scenario():
        async with _client() as client:
            response = await client.delete(f"/api/models/{XTTS_ID}")
            assert response.status_code == 400, response.text
            assert "пределы каталога моделей" in response.json()["detail"]
            assert victim.is_file()

            downloading = await client.post(f"/api/models/{XTTS_ID}/download")
            assert downloading.status_code == 400
            assert downloader.calls == []

    _run(scenario)
