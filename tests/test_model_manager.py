"""Реестр моделей (ФАЗА 9): определение установки, скачивание, удаление, диск.

Реальная сеть здесь не трогается никогда: единственная точка скачивания
(`model_manager._hf_download`) подменяется заглушкой, которая пишет файлы в
`tmp_path` и может падать по команде. Модели тоже не поднимаются: `loaded`
берётся из реестра движков, а движки в тестах не создаются.
"""

import threading
import time
from pathlib import Path

import pytest
from conftest import StubEngine

from backend import config, model_manager
from backend.engines import registry
from backend.model_manager import (
    DOWNLOAD_DONE,
    DOWNLOAD_DOWNLOADING,
    DOWNLOAD_ERROR,
    KIND_CACHE,
    KIND_LOCAL,
    ModelBusyError,
    ModelNotDownloadableError,
    ModelPathError,
    get_manager,
    model_specs,
    validate_model_path,
)

F5_ID = "f5"
XTTS_ID = "xtts"
BANANA_ID = "xtts-banana"


# --- окружение ----------------------------------------------------------------
@pytest.fixture
def models_dir(workspace, monkeypatch):
    """Каталог моделей в tmp_path: рабочие веса проекта тесты не трогают."""
    root = workspace / "models"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", root)
    monkeypatch.setattr(config, "XTTS_BASE_DIR", root / "xtts_v2")
    monkeypatch.setattr(config, "XTTS_BANANA_DIR", root / "xtts_v2_banana")
    monkeypatch.setattr(
        config,
        "CKPT_FILE",
        root / config.HF_CKPT_PATH,
    )
    monkeypatch.setattr(config, "VOCAB_FILE", root / config.HF_VOCAB_PATH)
    return root


@pytest.fixture(autouse=True)
def no_created_engines(monkeypatch):
    """Реестр движков пуст: `loaded` не должен подтягиваться из чужих тестов."""
    monkeypatch.setattr(registry, "_instances", {})
    return {}


def write(path: Path, size: int = 128) -> Path:
    """Файл-заглушка вместо гигабайтных весов: размер важен, содержимое нет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def install_f5(root: Path) -> None:
    write(root / config.HF_CKPT_PATH)
    write(root / config.HF_VOCAB_PATH)


def install_xtts(root: Path, spec_id: str = XTTS_ID) -> None:
    """Ставит XTTS. Для banana намеренно вкладывает файлы в подпапку — как HF."""
    base = model_manager.model_spec(spec_id).local_dir
    target = base / "model_banana" / "v2.0.2" if spec_id == BANANA_ID else base
    write(target / "model.pth")
    write(target / "config.json")


class FakeDownloader:
    """Замена `_hf_download`: пишет файл, ведёт журнал и умеет падать."""

    def __init__(self, fail_on: str | None = None, delay: float = 0.0) -> None:
        self.fail_on = fail_on
        self.delay = delay
        self.calls: list[tuple[str, str, Path]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, repo_id: str, filename: str, local_dir: Path) -> Path:
        self.calls.append((repo_id, filename, Path(local_dir)))
        self.started.set()
        if self.delay:
            self.release.wait(timeout=self.delay)
        if self.fail_on == filename:
            raise RuntimeError("HTTP 503: сервис недоступен")
        target = Path(local_dir) / filename
        write(target)
        return target


@pytest.fixture
def downloader(monkeypatch):
    fake = FakeDownloader()
    monkeypatch.setattr(model_manager, "_hf_download", fake)
    return fake


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Ждёт фоновое скачивание: поток завершится заведомо быстрее таймаута."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def state(model_id: str) -> dict:
    return get_manager().get(model_id)


# --- 1. installed detection ---------------------------------------------------
def test_installed_detection_reads_required_files(models_dir, downloader):
    install_f5(models_dir)
    install_xtts(models_dir)

    f5 = state(F5_ID)
    assert f5["installed"] is True
    assert f5["missing_files"] == []
    assert f5["size_bytes"] > 0
    assert f5["kind"] == KIND_LOCAL
    # Ни одного обращения к сети: установка определяется только файлами.
    assert downloader.calls == []

    xtts = state(XTTS_ID)
    assert xtts["installed"] is True
    assert xtts["missing_files"] == []


def test_installed_detection_is_offline_for_every_model(models_dir, monkeypatch):
    """Реестр не ходит в сеть: любое скачивание в этом тесте — падение."""
    def explode(*_args, **_kwargs):
        raise AssertionError("состояние модели не должно требовать сети")

    monkeypatch.setattr(model_manager, "_hf_download", explode)
    install_f5(models_dir)
    install_xtts(models_dir)
    install_xtts(models_dir, BANANA_ID)

    for spec in model_specs():
        result = state(spec.id)
        assert "installed" in result and "missing_files" in result


# --- 2. missing required file -------------------------------------------------
def test_missing_required_file_is_reported(models_dir, downloader):
    install_f5(models_dir)
    (models_dir / config.HF_VOCAB_PATH).unlink()

    f5 = state(F5_ID)
    assert f5["installed"] is False
    assert f5["missing_files"] == [config.HF_VOCAB_PATH]


def test_banana_checkpoint_is_found_recursively(models_dir, downloader):
    """Чекпоинт banana лежит во вложенной папке — установка должна это видеть."""
    install_xtts(models_dir, BANANA_ID)

    banana = state(BANANA_ID)
    assert banana["installed"] is True
    assert banana["missing_files"] == []
    assert Path(banana["path"]) == Path(config.XTTS_BANANA_DIR)


# --- 11. метаданные реестра ---------------------------------------------------
def test_registry_metadata_matches_config(models_dir):
    specs = {spec.id: spec for spec in model_specs()}
    assert len(specs) == 4  # f5, xtts, xtts-banana и вспомогательный whisper

    f5 = specs[F5_ID]
    assert f5.repo_id == config.HF_REPO_ID
    assert set(f5.required_files) == {config.HF_CKPT_PATH, config.HF_VOCAB_PATH}
    assert f5.local_dir == config.MODELS_DIR
    assert f5.engine_id == "f5"

    base = specs[XTTS_ID]
    assert base.repo_id == "coqui/XTTS-v2"
    assert base.required_files == ("model.pth", "config.json")
    assert base.local_dir == config.XTTS_BASE_DIR
    assert base.engine_id == XTTS_ID

    banana = specs[BANANA_ID]
    assert banana.repo_id == "Ftfyhh/xttsv2_banana"
    assert banana.required_files == ("model.pth", "config.json")
    assert banana.local_dir == config.XTTS_BANANA_DIR
    assert banana.engine_id == BANANA_ID

    whisper = specs["whisper"]
    assert whisper.repo_id == "openai/whisper-large-v3-turbo"
    assert whisper.kind == KIND_CACHE
    assert whisper.engine_id is None


def test_registry_paths_are_read_at_call_time(models_dir, workspace, monkeypatch):
    """Пути берутся из config при вызове: подмена каталога меняет реестр."""
    moved = workspace / "other-models"
    moved.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", moved)

    assert model_manager.model_spec(F5_ID).local_dir == moved
    assert model_manager.model_spec("whisper") is not None


# --- 3. download success ------------------------------------------------------
def test_download_success_reaches_done(models_dir, monkeypatch):
    fake = FakeDownloader()
    monkeypatch.setattr(model_manager, "_hf_download", fake)

    get_manager().download(F5_ID)
    assert wait_until(lambda: state(F5_ID)["installed"])

    f5 = state(F5_ID)
    assert f5["download"]["state"] == DOWNLOAD_DONE
    assert f5["download"]["progress"] == 1.0
    assert f5["missing_files"] == []
    assert {call[1] for call in fake.calls} == set(f5["required_files"])
    # Скачанные файлы действительно лежат в каталоге модели.
    assert (models_dir / config.HF_CKPT_PATH).is_file()


def test_download_reports_downloading_state(models_dir, monkeypatch):
    """Пока идёт скачивание, состояние — `downloading`, а не «уже готово»."""
    fake = FakeDownloader(delay=1.0)
    monkeypatch.setattr(model_manager, "_hf_download", fake)

    get_manager().download(XTTS_ID)
    assert state(XTTS_ID)["download"]["state"] == DOWNLOAD_DOWNLOADING
    fake.release.set()
    assert wait_until(lambda: state(XTTS_ID)["download"]["state"] == DOWNLOAD_DONE)


# --- 4. download failure ------------------------------------------------------
def test_download_failure_sets_error_and_process_survives(models_dir, monkeypatch):
    fake = FakeDownloader(fail_on="model.pth")
    monkeypatch.setattr(model_manager, "_hf_download", fake)

    get_manager().download(XTTS_ID)
    assert wait_until(lambda: state(XTTS_ID)["download"]["state"] == DOWNLOAD_ERROR)

    xtts = state(XTTS_ID)
    assert xtts["installed"] is False
    assert xtts["download"]["error"]
    # Приложение живо и продолжает отдавать список моделей.
    assert len(get_manager().models()) == len(model_specs())


def test_download_failure_message_explains_what_happened(models_dir, monkeypatch):
    fake = FakeDownloader(fail_on="config.json")
    monkeypatch.setattr(model_manager, "_hf_download", fake)

    get_manager().download(XTTS_ID)
    assert wait_until(lambda: state(XTTS_ID)["download"]["state"] == DOWNLOAD_ERROR)
    error = state(XTTS_ID)["download"]["error"]
    assert "Не удалось скачать" in error
    assert "503" in error or "недоступен" in error


# --- 5. interrupted download --------------------------------------------------
def test_interrupted_download_is_not_installed_and_can_resume(models_dir, monkeypatch):
    """Прерванное скачивание: модель не установлена, повтор разрешён и доходит."""
    # Первое скачивание рвётся на втором файле: один файл остался на диске.
    broken = FakeDownloader(fail_on=config.HF_VOCAB_PATH)
    monkeypatch.setattr(model_manager, "_hf_download", broken)
    get_manager().download(F5_ID)
    assert wait_until(lambda: state(F5_ID)["download"]["state"] == DOWNLOAD_ERROR)

    partial = state(F5_ID)
    assert partial["installed"] is False
    assert config.HF_VOCAB_PATH in partial["missing_files"]
    assert partial["download"]["state"] in (DOWNLOAD_ERROR, "interrupted")

    # Повторный запуск докачивает недостающее и доводит до `done`.
    good = FakeDownloader()
    monkeypatch.setattr(model_manager, "_hf_download", good)
    get_manager().download(F5_ID)
    assert wait_until(
        lambda: state(F5_ID)["download"]["state"] == DOWNLOAD_DONE
        and state(F5_ID)["installed"]
    )

    finished = state(F5_ID)
    assert finished["download"]["state"] == DOWNLOAD_DONE
    assert finished["missing_files"] == []
    # Первый (уже лежащий) файл повторно не качался — докачивалось недостающее.
    assert {call[1] for call in good.calls} == {config.HF_VOCAB_PATH}


# --- 6. повторный download не повреждает установленную модель ------------------
def test_repeat_download_keeps_installed_files_intact(models_dir, monkeypatch):
    install_f5(models_dir)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (
            models_dir / config.HF_CKPT_PATH,
            models_dir / config.HF_VOCAB_PATH,
        )
    }

    def explode(*_args, **_kwargs):
        raise AssertionError("повторное скачивание установленной модели недопустимо")

    monkeypatch.setattr(model_manager, "_hf_download", explode)
    result = get_manager().download(F5_ID)

    assert result["installed"] is True
    assert result["download"]["state"] == DOWNLOAD_DONE
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in before}
    assert after == before


def test_repeat_download_of_downloading_model_does_not_restart(models_dir, monkeypatch):
    fake = FakeDownloader(delay=1.0)
    monkeypatch.setattr(model_manager, "_hf_download", fake)

    get_manager().download(XTTS_ID)
    assert fake.started.wait(timeout=2.0)
    get_manager().download(XTTS_ID)  # второй POST, пока идёт первый
    fake.release.set()
    assert wait_until(lambda: state(XTTS_ID)["download"]["state"] == DOWNLOAD_DONE)
    assert len(fake.calls) == len(model_manager.model_spec(XTTS_ID).required_files)


# --- 7. model path validation -------------------------------------------------
def test_path_outside_models_dir_is_rejected(models_dir, workspace, monkeypatch, downloader):
    outside = workspace / "outside"
    outside.mkdir()
    victim = write(outside / "model.pth")
    monkeypatch.setattr(config, "XTTS_BASE_DIR", outside)
    monkeypatch.setattr(
        config, "XTTS_BANANA_DIR", outside / ".." / "outside"
    )

    with pytest.raises(ModelPathError, match="пределы каталога моделей"):
        validate_model_path(model_manager.model_spec(XTTS_ID))
    with pytest.raises(ModelPathError):
        get_manager().download(XTTS_ID)
    with pytest.raises(ModelPathError):
        get_manager().delete(XTTS_ID)

    # Ничего не удалено и ничего не скачано за пределами каталога моделей.
    assert victim.is_file()
    assert downloader.calls == []


def test_parent_traversal_is_rejected(models_dir, workspace, monkeypatch):
    traversal = config.MODELS_DIR / ".." / ".." / "etc"
    monkeypatch.setattr(config, "XTTS_BASE_DIR", traversal)

    with pytest.raises(ModelPathError):
        validate_model_path(model_manager.model_spec(XTTS_ID))


def test_state_of_invalid_path_is_reported_not_raised(models_dir, workspace, monkeypatch):
    """Неверная настройка пути не ломает список моделей — она видна в состоянии."""
    outside = workspace / "outside"
    outside.mkdir()
    monkeypatch.setattr(config, "XTTS_BASE_DIR", outside)

    xtts = state(XTTS_ID)
    assert xtts["installed"] is False
    assert xtts["download"]["state"] == DOWNLOAD_ERROR
    assert "пределы каталога моделей" in xtts["download"]["error"]


def test_delete_removes_only_model_directory(models_dir, downloader):
    install_xtts(models_dir)
    neighbour = write(models_dir / "F5TTS_v1_Base" / "vocab.txt")

    result = get_manager().delete(XTTS_ID)
    assert result["deleted"] == XTTS_ID
    assert not config.XTTS_BASE_DIR.exists()
    assert neighbour.is_file()
    assert state(XTTS_ID)["installed"] is False


def test_cache_model_cannot_be_deleted_or_downloaded(models_dir, downloader):
    with pytest.raises(ModelNotDownloadableError):
        get_manager().download("whisper")
    with pytest.raises(ModelNotDownloadableError):
        get_manager().delete("whisper")
    assert downloader.calls == []


# --- 8. disk usage ------------------------------------------------------------
def test_disk_usage_counts_installed_models(models_dir, downloader):
    install_xtts(models_dir)
    states = get_manager().states()
    report = get_manager().disk_report(states)

    expected = sum(
        path.stat().st_size
        for path in (config.XTTS_BASE_DIR / "model.pth", config.XTTS_BASE_DIR / "config.json")
    )
    by_id = {state.spec.id: state.to_dict() for state in states}
    assert by_id[XTTS_ID]["size_bytes"] == expected
    # Моделей, которых нет в tmp_path, в счёте нет: их файлов здесь не существует.
    assert by_id[F5_ID]["size_bytes"] == 0
    # Сводка — сумма по всем показанным моделям (включая кешированные).
    assert report["models_bytes"] == sum(item["size_bytes"] for item in by_id.values())
    assert report["free_bytes"] > 0
    assert report["total_bytes"] > report["free_bytes"]


# --- 10. удаление запрещено, пока движок занят ---------------------------------
class _LoadedEngine(StubEngine):
    """Движок, помеченный как поднятый: файлы модели сейчас читает модель."""

    def load(self) -> None:
        self._mark("ready")


def test_delete_is_refused_while_engine_is_loaded(models_dir, downloader, monkeypatch):
    install_xtts(models_dir)
    engine = _LoadedEngine()
    engine.info = registry.ENGINE_INFOS[XTTS_ID]
    engine.load()
    monkeypatch.setitem(registry._instances, XTTS_ID, engine)

    assert state(XTTS_ID)["loaded"] is True
    assert state(XTTS_ID)["engine_in_use"] is True

    with pytest.raises(ModelBusyError, match="используется движком"):
        get_manager().delete(XTTS_ID)

    # Файлы на месте: отказ случился до любого касания диска.
    assert (config.XTTS_BASE_DIR / "model.pth").is_file()
    assert state(XTTS_ID)["installed"] is True


def test_delete_is_refused_while_queue_uses_engine(models_dir, downloader, monkeypatch):
    """Очередь занята задачей с этим движком — удалять веса тоже нельзя."""
    install_xtts(models_dir)

    class _Queue:
        current_job_id = "job-1"

        def get(self, _job_id):
            class _Job:
                payload = type("P", (), {"replicas": [type("R", (), {"voice": "voice1"})()]})()

            return _Job()

    class _Store:
        def get(self, _voice_id):
            return type("V", (), {"engine": XTTS_ID})()

    monkeypatch.setattr("backend.job_queue.get_queue", lambda: _Queue())
    monkeypatch.setattr("backend.voices_store.get_store", lambda: _Store())

    assert state(XTTS_ID)["engine_in_use"] is True
    with pytest.raises(ModelBusyError):
        get_manager().delete(XTTS_ID)
    assert (config.XTTS_BASE_DIR / "model.pth").is_file()


def test_loaded_state_comes_from_created_engines_without_creating_them(
    models_dir, downloader, monkeypatch
):
    """`loaded` берётся из уже созданных движков: реестр моделей их не поднимает."""
    install_xtts(models_dir)
    created: list[str] = []

    def fake_create(engine_id):
        created.append(engine_id)
        raise AssertionError("менеджер моделей не должен создавать движки")

    monkeypatch.setattr(registry, "_create", fake_create)
    assert state(XTTS_ID)["loaded"] is False
    assert state(F5_ID)["loaded"] is False
    assert created == []


def test_engine_status_defaults_to_unloaded_for_helper_model(models_dir, downloader):
    whisper = state("whisper")
    assert whisper["loaded"] is False
    assert whisper["engine_in_use"] is False
    assert whisper["engine_id"] is None
