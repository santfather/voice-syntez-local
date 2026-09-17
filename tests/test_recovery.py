"""Атомарная запись, проверка аудио и восстановление после перезапуска.

Проверяется контракт «готовый файл действительно звучит»: запись идёт через
`*.part`, результат проверяется до переименования, а состояния «идёт» из прошлого
запуска при старте переводятся в прерванные — иначе проект навсегда остался бы в
`rendering`, а недописанный WAV лежал бы рядом с целыми.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import soundfile as sf
from conftest import sine

from backend import audio_pipeline, config, recovery
from backend.audio_pipeline import (
    PARTIAL_SUFFIX,
    AudioFileError,
    _write_audio,
    _write_output,
    validate_audio_file,
)
from backend.db.store import get_projects_store
from backend.engines import worker_protocol as proto


def _wav(path, seconds: float = 0.2) -> np.ndarray:
    data = sine(seconds)
    sf.write(path, data, audio_pipeline.SAMPLE_RATE)
    return data


# --- атомарная запись и проверка -----------------------------------------------
def test_write_is_atomic_and_leaves_no_partial_on_failure(tmp_path, monkeypatch):
    """Сбой посреди записи не оставляет ни «половины», ни временного файла."""
    target = tmp_path / "job.wav"
    real_write = sf.write

    def broken_write(path, data, samplerate, **kwargs):
        real_write(path, data, samplerate, **kwargs)  # часть данных уже на диске
        raise OSError("диск переполнен")

    monkeypatch.setattr(sf, "write", broken_write)
    with pytest.raises(OSError, match="диск переполнен"):
        _write_audio(target, sine(0.2), "wav")

    assert not target.exists(), "недописанный файл занял готовое имя"
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}*")) == [], "остался временный файл"


def test_write_rejects_empty_audio(tmp_path):
    """Пустая запись — ошибка, а не «готовый» нулевой WAV."""
    target = tmp_path / "empty.wav"
    with pytest.raises(AudioFileError, match="нет звука"):
        _write_audio(target, np.zeros(0, dtype=np.float32), "wav")
    assert not target.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}*")) == []


def test_write_output_validates_ready_file(tmp_path, monkeypatch, workspace):
    """Готовый файл задачи проверяется на итоговом имени (контракт DONE)."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    path = _write_output("job-1", sine(0.3), "wav")
    assert path.exists()
    meta = validate_audio_file(path)
    assert meta["sample_rate"] == audio_pipeline.SAMPLE_RATE
    assert meta["duration_sec"] > 0


def test_validate_reports_metadata(tmp_path):
    """Проверка возвращает метаданные: ими же объясняется готовая запись."""
    path = tmp_path / "one.wav"
    _wav(path, seconds=0.25)
    meta = validate_audio_file(path)
    assert meta["sample_rate"] == audio_pipeline.SAMPLE_RATE
    assert meta["duration_sec"] == pytest.approx(0.25, abs=0.01)
    assert meta["format"] == "WAV"


def test_broken_file_is_rejected(tmp_path):
    """Побитый файл (обрыв внутри заголовка) — ошибка, а не «готовое аудио»."""
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")  # ровно заголовок без данных

    with pytest.raises(AudioFileError, match="не читается как аудио|нет звука|обрезан"):
        validate_audio_file(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(AudioFileError, match="не создан"):
        validate_audio_file(tmp_path / "нет.wav")


def test_compressed_write_is_atomic_and_validated(tmp_path):
    """Сжатый формат тоже пишется атомарно и проверяется после конвертации."""
    target = tmp_path / "job.mp3"
    try:
        _write_audio(target, sine(0.3), "mp3")
    except Exception as exc:  # noqa: BLE001 — без ffmpeg mp3 не собрать, это не наш случай
        pytest.skip(f"конвертация недоступна: {exc}")
    assert target.exists()
    assert validate_audio_file(target, expect_sample_rate=None)["frames"] > 0
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}*")) == []


def test_cleanup_partials_removes_only_partials(workspace):
    """Убираются только остатки `.part`: готовые файлы не трогаются."""
    output = workspace / "output"
    (output / "projects" / "p1").mkdir(parents=True, exist_ok=True)
    good = output / "job.wav"
    _wav(good)
    (output / "job.mp3.part").write_bytes(b"half")
    (output / "projects" / "p1" / "r1.wav.part").write_bytes(b"half")
    (output / "job.mp3.part.wav").write_bytes(b"half")

    assert sorted(path.name for path in audio_pipeline.partial_files()) == [
        "job.mp3.part",
        "job.mp3.part.wav",
        "r1.wav.part",
    ]
    assert audio_pipeline.cleanup_partials() == 3
    assert good.exists(), "готовый файл удалён вместе с остатками"
    assert audio_pipeline.partial_files() == []


def test_cache_cleanup_counts_partial_files(workspace):
    """Ручная очистка кеша видит остатки `.part`, а не считает их готовыми файлами."""
    from backend import cache_cleanup

    (workspace / "output" / "job.wav.part").write_bytes(b"half")
    _wav(workspace / "output" / "job.wav")

    inventory = cache_cleanup.inventory()
    assert inventory["temp_files_count"] == 1
    # Готовый файл остаётся в своей категории и в temp не попадает.
    assert inventory["output_mb"] > 0
    assert inventory["output_mb"] < 1  # короткий файл: счётчик меряет его, а не весь каталог


# --- восстановление после перезапуска ------------------------------------------
def _interrupted_project() -> tuple[str, int, int]:
    """Проект, у которого рендер оборвался перезапуском приложения.

    Три реплики: первая готова (take на диске), вторая была в работе, третья не
    начиналась — ровно сценарий §13 отчёта.
    """
    store = get_projects_store()
    project = store.create_project(
        "Прерванный",
        source_text="ИВАН: Первая.\nИВАН: Вторая.\nИВАН: Третья.",
        mode="dialogue",
    )
    parsed = store.parse_project(project["id"], None)
    first = parsed["replicas"][0]["index"]
    second = parsed["replicas"][1]["index"]
    take = config.PROJECTS_OUTPUT_DIR / project["id"] / "r1-ready.wav"
    take.parent.mkdir(parents=True, exist_ok=True)
    _wav(take)
    store.save_render_takes(
        project["id"],
        "job-8",
        [{"index": first, "audio_path": str(take), "label": "рендер job-8", "duration_sec": 0.2}],
    )
    store.update_project(
        project["id"],
        status=config.PROJECT_STATUS_RENDERING,
        job_id="job-9",
        analysis_status=config.PROJECT_ANALYSIS_ANALYZING,
        analysis_started_at="t",
    )
    store.set_replica_status(project["id"], second, config.REPLICA_STATUS_RENDERING)
    return project["id"], first, second


def test_startup_recovery_interrupts_stale_state(workspace):
    """Состояния «идёт» становятся прерванными с причиной и диагностикой."""
    store = get_projects_store()
    project_id, first, second = _interrupted_project()
    part = workspace / "output" / "stale.wav.part"
    part.write_bytes(b"half")

    report = recovery.recover_after_restart()

    assert report.to_dict() == {
        "projects": 1,
        "replicas": 1,
        "analyses": 1,
        "partial_files": 1,
        "crashes": 1,
    }
    project = store.get_project(project_id)
    assert project["status"] == config.PROJECT_STATUS_DRAFT
    assert project["last_error"] == config.RECOVERY_RENDER_MESSAGE
    assert project["analysis_status"] == config.PROJECT_ANALYSIS_RAW
    assert project["analysis_error"] == config.RECOVERY_ANALYSIS_MESSAGE

    replicas = {row["index"]: row for row in project["replicas"]}
    # Готовая реплика осталась готовой вместе с вариантом, прерванная помечена
    # прерванной, неначатая — не тронута.
    assert replicas[first]["status"] == config.REPLICA_STATUS_RENDERED
    assert replicas[first]["takes"], "готовый вариант пропал при восстановлении"
    assert replicas[second]["status"] == config.REPLICA_STATUS_INTERRUPTED
    assert replicas[first + 2]["status"] == config.REPLICA_STATUS_PENDING
    assert not part.exists()

    crashes = store.list_worker_crashes(limit=5)
    assert len(crashes) == 1
    assert crashes[0]["error_type"] == proto.ERROR_INTERRUPTED
    assert crashes[0]["job_id"] == "job-9"
    assert crashes[0]["project_id"] == project_id


def test_recovery_is_idempotent(workspace):
    """Второй прогон на чистой базе ничего не восстанавливает."""
    _interrupted_project()
    first = recovery.recover_after_restart()
    second = recovery.recover_after_restart()
    assert first.projects and first.replicas
    assert second.to_dict() == {
        "projects": 0,
        "replicas": 0,
        "analyses": 0,
        "partial_files": 0,
        "crashes": 0,
    }
    assert second.empty is True


class _BrokenStore:
    """Хранилище, которое не отвечает: приложение всё равно должно подняться."""

    def interrupt_stale_state(self) -> dict:
        raise RuntimeError("база недоступна")

    def record_worker_crash(self, **fields) -> dict:  # pragma: no cover — не вызывается
        raise AssertionError("диагностика не должна писаться при сбое восстановления")


def test_recovery_survives_broken_store(workspace, monkeypatch):
    """Сбой восстановления не мешает приложению подняться."""
    monkeypatch.setattr(recovery, "get_projects_store", lambda: _BrokenStore())
    report = recovery.recover_after_restart()
    assert report.projects == []
    assert report.replicas == 0


def test_lifespan_calls_recovery_before_queue():
    """Восстановление вызывается при старте приложения и до старта очереди.

    Проверяется по исходнику: поднять настоящий lifespan в тесте нельзя (лок
    экземпляра, проверка порта, прогрев модели), а связка «восстановили, потом
    начали брать задачи» — именно то, что легко потерять при правке `lifespan`.
    """
    from backend import main

    source = inspect.getsource(main.lifespan)
    assert "recovery.recover_after_restart()" in source
    assert source.index("recovery.recover_after_restart()") < source.index("get_queue().start()")
