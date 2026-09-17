"""Очистка собственного кеша приложения (Задача 3): логика `backend/cache_cleanup.py`.

Проверяется главное свойство функции: она освобождает место в трёх своих
категориях и **физически не может** дотянуться до того, что потерять нельзя, —
`data/voice_syntez.db` со словарём произношения, `voices/` с референсами,
`models/` с весами и take'ов проектов. Сентинелы расставлены по всем этим путям.

Системный каталог временных файлов уведён в `tmp_path` (`tempfile.gettempdir`):
тест не должен трогать настоящий temp машины и, наоборот, не должен зависеть от
того, что в нём уже лежит. Веса и модели не поднимаются — это работа с диском.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from backend import benchmark, cache_cleanup, config
from backend.pronunciation import get_store as get_pronunciation_store

MB = 1024 * 1024


def write(path: Path, size: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def touch_old(path: Path, age_sec: float = cache_cleanup.TEMP_MIN_AGE_SEC + 600) -> Path:
    """Создаёт каталог (если нужно) и состаривает запись: порог — по mtime, без ожидания."""
    path.mkdir(parents=True, exist_ok=True)
    return age(path, age_sec)


def age(path: Path, age_sec: float = cache_cleanup.TEMP_MIN_AGE_SEC + 600) -> Path:
    """Состаривает уже существующую запись (файл, каталог или ссылку)."""
    stamp = time.time() - age_sec
    os.utime(path, (stamp, stamp))
    return path


def mb(size: int) -> float:
    return round(size / MB, 2)


@pytest.fixture
def system_temp(workspace, monkeypatch):
    """Системный temp внутри `tmp_path`: чужие файлы машины не трогаются."""
    temp_dir = workspace / "system-temp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
    return temp_dir


@pytest.fixture
def layout(workspace, system_temp):
    """Рабочие каталоги проекта и файлы в них — как на настоящем диске.

    `workspace` уже уводит `OUTPUT_DIR`/`VOICES_DIR`/`DB_PATH` в `tmp_path`;
    здесь добавляются `data/` и `models/` со своими файлами-сентинелами.
    """
    output = config.OUTPUT_DIR
    benchmarks = config.BENCHMARKS_DIR
    projects = config.PROJECTS_OUTPUT_DIR
    data_dir = workspace / "data"
    models_dir = workspace / "models"
    voices_dir = config.VOICES_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "output": output,
        "benchmarks": benchmarks,
        "projects": projects,
        "data": data_dir,
        "models": models_dir,
        "voices": voices_dir,
    }
    # Транзитные готовые файлы: их и должна уносить категория `output`.
    write(output / "job1.mp3", 2 * MB)
    write(output / "job2.wav", 1 * MB)
    write(output / ".gitkeep", 4)
    # Результаты сравнения движков: своя категория со сбросом реестра.
    write(benchmarks / "abc123-f5.wav", 3 * MB)
    write(benchmarks / "abc123-xtts.wav", 2 * MB)
    write(benchmarks / ".gitkeep", 4)
    # Take'ы проекта — данные проекта, категорией `output` не трогаются.
    write(projects / "project1" / "take-1.wav", 1500_000)
    write(projects / "project1" / "take-2.wav", 1500_000)
    # Промежуточные файлы самого приложения.
    write(output / "job1.mp3.tmp.wav", 512_000)
    write(projects / "project1" / "take-1.wav.tmp.wav", 256_000)
    return paths


# --- 1. очистка output --------------------------------------------------------
def test_clear_output_frees_space_and_reports_counts(layout):
    result = cache_cleanup.clear(["output"])

    assert result["freed_files"] == {"output": 2}
    assert result["freed_mb"]["output"] == mb(3 * MB)
    assert result["total_mb"] == mb(3 * MB)
    assert not (layout["output"] / "job1.mp3").exists()
    assert not (layout["output"] / "job2.wav").exists()
    # Промежуточный `.tmp.wav` — своя категория, а не `output`: без этого его
    # размер посчитался бы дважды и интерфейс обещал бы лишнее.
    assert (layout["output"] / "job1.mp3.tmp.wav").is_file()


def test_inventory_matches_categories_without_double_counting(layout):
    report = cache_cleanup.inventory()

    assert set(report) == {
        "output_mb", "benchmarks_mb", "temp_files_mb", "temp_files_count",
    }
    # output: job1 + job2 (3 МБ); `.tmp.wav` — в temp_files, take'ы — в projects.
    assert report["output_mb"] == mb(3 * MB)
    assert report["benchmarks_mb"] == mb(5 * MB)
    assert report["temp_files_mb"] == mb(512_000 + 256_000)
    # Считаются файлы, а не каталоги: project1 — это каталог, а не единица учёта.
    assert report["temp_files_count"] == 2
    assert all(value >= 0 for value in report.values())


def test_inventory_is_consistent_with_categories_it_clears(layout):
    report = cache_cleanup.inventory()
    cleared = cache_cleanup.clear(cache_cleanup.TARGETS)

    for target, key in (
        ("output", "output_mb"),
        ("benchmarks", "benchmarks_mb"),
        ("temp_files", "temp_files_mb"),
    ):
        assert cleared["freed_mb"][target] == report[key]
    assert cleared["freed_files"]["temp_files"] == report["temp_files_count"]
    assert cleared["total_mb"] == mb(3 * MB + 5 * MB + 768_000)
    # После очистки инвентарь честный ноль, а не «примерно ноль».
    assert cache_cleanup.inventory() == {
        "output_mb": 0.0, "benchmarks_mb": 0.0,
        "temp_files_mb": 0.0, "temp_files_count": 0,
    }


# --- 2. benchmarks: файлы и реестр -------------------------------------------
def test_clear_benchmarks_removes_files_and_resets_registry(layout):
    run = benchmark.register(
        benchmark.BenchmarkRun(id="abc123", voice_id="voice1", text="Фраза", engines=["f5"])
    )
    assert benchmark.list_runs() == [run]

    result = cache_cleanup.clear(["benchmarks"])

    assert result["freed_files"] == {"benchmarks": 2}
    assert result["freed_mb"]["benchmarks"] == mb(5 * MB)
    assert not (layout["benchmarks"] / "abc123-f5.wav").exists()
    assert not (layout["benchmarks"] / "abc123-xtts.wav").exists()
    # Без сброса интерфейс показывал бы строку сравнения со ссылкой на удалённый
    # файл: `/api/benchmarks/abc123/f5/audio` стал бы 404.
    assert benchmark.get_run("abc123") is None
    assert benchmark.list_runs() == []


def test_output_clear_does_not_touch_benchmark_registry(layout):
    benchmark.register(
        benchmark.BenchmarkRun(id="keep", voice_id="voice1", text="Фраза", engines=[])
    )

    cache_cleanup.clear(["output"])

    assert benchmark.get_run("keep") is not None
    benchmark.reset_registry()


# --- 3. temp_files: свои остатки и только свои --------------------------------
def test_clear_temp_files_removes_tmp_wav_and_old_our_dirs(layout, system_temp):
    old_dir = system_temp / "tts-export-abcd"
    write(old_dir / "audio.wav", 4096)
    age(old_dir)  # содержимое написано, и только теперь каталог «старый»
    old_file = age(write(system_temp / "voice-syntez-old123.wav", 2048))
    fresh_dir = system_temp / "voice-denoise-fresh"
    write(fresh_dir / "source.wav", 8192)  # только что создан и ещё работает

    result = cache_cleanup.clear(["temp_files"])

    assert not (layout["output"] / "job1.mp3.tmp.wav").exists()
    assert not (layout["projects"] / "project1" / "take-1.wav.tmp.wav").exists()
    assert not old_file.exists()
    assert not old_dir.exists()
    # Активная операция (моложе порога) остаётся: удалить её на ходу нельзя.
    assert fresh_dir.is_dir()
    assert (fresh_dir / "source.wav").is_file()
    # 2 файла .tmp.wav + 1 файл в старом каталоге + 1 старый файл в temp.
    assert result["freed_files"]["temp_files"] == 4


def test_clear_temp_files_keeps_foreign_entries_in_system_temp(layout, system_temp):
    foreign_dir = touch_old(system_temp / "somebody-elses-workdir")
    write(foreign_dir / "important.bin", 4096)
    foreign_file = age(write(system_temp / "tmpXYZ123.wav", 1024))
    lookalike = touch_old(system_temp / "tts-exportish-note")
    write(lookalike / "payload.txt", 16)

    cache_cleanup.clear(["temp_files"])

    assert foreign_dir.is_dir() and (foreign_dir / "important.bin").is_file()
    assert foreign_file.is_file()
    assert lookalike.is_dir() and (lookalike / "payload.txt").is_file()


def test_symlink_is_never_followed_by_temp_cleanup(layout, system_temp):
    """Символическая ссылка с нашим префиксом не должна увести удаление из temp."""
    victim = layout["output"] / "not-ours"
    write(victim / "data.wav", 4096)
    links = []
    for name, target, is_dir in (
        ("tts-export-link", victim, True),
        ("voice-syntez-link.wav", victim / "data.wav", False),
    ):
        link = system_temp / name
        try:
            link.symlink_to(target, target_is_directory=is_dir)
        except (OSError, NotImplementedError):  # pragma: no cover — ФС без симлинков
            pytest.skip("символические ссылки недоступны")
        age(link)
        links.append(link)

    cache_cleanup.clear(["temp_files"])

    assert victim.is_dir() and (victim / "data.wav").is_file()
    assert all(link.is_symlink() for link in links)


# --- 4. безопасность: база, словарь, голоса, модели ---------------------------
def test_all_three_targets_keep_database_dictionary_voices_and_models(layout, monkeypatch):
    data_dir = layout["data"]
    models_dir = layout["models"]
    voices_dir = layout["voices"]

    # База — ровно та, что названа в критерии готовности: `data/voice_syntez.db`.
    # `workspace` уводит её в tmp_path, поэтому здесь путь называется целиком, и
    # словарь произношения проверяется живьём, а не «где-то в tmp».
    monkeypatch.setattr(config, "DB_PATH", data_dir / "voice_syntez.db")

    # Сентинелы: файлы, потеря которых невосстановима.
    db_path = data_dir / "voice_syntez.db"
    pronunciation = get_pronunciation_store().create("SQL", "эскьюэль", note="тест")
    assert db_path.is_file()
    voices_json = write(voices_dir / "voices.json", 512)
    reference = write(voices_dir / "ref.wav", 4096)
    hf_cache = write(models_dir / "huggingface" / "models--openai--whisper" / "model.bin", 2048)
    ckpt = write(models_dir / "F5TTS_v1_Base_accent_tune" / "model_last_inference.safetensors", MB)
    db_before = db_path.read_bytes()

    result = cache_cleanup.clear(cache_cleanup.TARGETS)

    # Освобождение состоялось — иначе тест «ничего не тронуто» ничего не значит.
    assert result["total_mb"] > 0
    for path in (db_path, voices_json, reference, hf_cache, ckpt):
        assert path.is_file(), f"{path} обязан пережить очистку всех трёх категорий"
    assert db_path.read_bytes() == db_before
    # База открывается, и правило словаря на месте: файл цел не только по размеру.
    assert get_pronunciation_store().get(pronunciation["id"])["target"] == "эскьюэль"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        entries = connection.execute("SELECT source, target FROM pronunciation_entries").fetchall()
    assert entries == [("SQL", "эскьюэль")]


def test_all_categories_leave_data_and_models_outside_their_roots(layout):
    """Проверка от обратного: ни один корень категории не ведёт в данные проекта.

    Это тот же тест безопасности, но без удаления: если однажды кто-то расширит
    список корней, проверка упадёт раньше, чем файл пользователя пропадёт.
    """
    data_dir = layout["data"].resolve()
    models_dir = layout["models"].resolve()
    voices_dir = layout["voices"].resolve()

    for target in cache_cleanup.TARGETS:
        for root in cache_cleanup._roots(target):
            real = root.resolve()
            assert not real.is_relative_to(data_dir)
            assert not real.is_relative_to(models_dir)
            assert not real.is_relative_to(voices_dir)


# --- 5-6. подкаталоги и .gitkeep ----------------------------------------------
def test_output_clear_keeps_project_takes_and_benchmarks(layout):
    cache_cleanup.clear(["output"])

    assert (layout["projects"] / "project1" / "take-1.wav").is_file()
    assert (layout["projects"] / "project1" / "take-2.wav").is_file()
    assert (layout["benchmarks"] / "abc123-f5.wav").is_file()


def test_gitkeep_survives_every_category(layout):
    cache_cleanup.clear(cache_cleanup.TARGETS)

    assert (layout["output"] / ".gitkeep").is_file()
    assert (layout["benchmarks"] / ".gitkeep").is_file()


# --- 7-8. ошибки входа, идемпотентность, отсутствующие каталоги ---------------
def test_unknown_target_is_rejected_without_deleting_anything(layout):
    with pytest.raises(ValueError) as error:
        cache_cleanup.clear(["models"])

    assert "models" in str(error.value)
    assert (layout["output"] / "job1.mp3").is_file()
    assert layout["models"].is_dir()


def test_repeated_clear_is_idempotent_and_missing_dirs_are_zero(layout):
    first = cache_cleanup.clear(cache_cleanup.TARGETS)
    assert first["total_mb"] > 0

    second = cache_cleanup.clear(cache_cleanup.TARGETS)

    assert second["total_mb"] == 0.0
    assert second["freed_mb"] == {"output": 0.0, "benchmarks": 0.0, "temp_files": 0.0}
    assert second["freed_files"] == {"output": 0, "benchmarks": 0, "temp_files": 0}
    assert cache_cleanup.clear([])["total_mb"] == 0.0


def test_missing_directories_are_zero_not_an_error(workspace):
    absent = workspace / "no-such-output"
    config.OUTPUT_DIR = absent
    config.BENCHMARKS_DIR = absent / "benchmarks"
    config.PROJECTS_OUTPUT_DIR = absent / "projects"

    assert cache_cleanup.inventory()["output_mb"] == 0.0
    result = cache_cleanup.clear(cache_cleanup.TARGETS)

    assert result["total_mb"] == 0.0
    assert result["freed_files"] == {"output": 0, "benchmarks": 0, "temp_files": 0}


def test_real_system_temp_is_never_scanned(layout, monkeypatch):
    """Настоящий системный temp машины очисткой не обходится.

    Фикстура `system_temp` подменяет `gettempdir`, и без этой проверки никто бы не
    заметил, что очистка однажды начнёт смотреть в реальный `/tmp`. Каталог с нашим
    префиксом здесь лежит в `output/` — вне системного temp — и должен остаться.
    """
    real_temp = Path(tempfile.gettempdir()).resolve()
    fake = layout["output"] / "our-temp-for-test"
    write(fake / "voice-syntez-leftover.wav", 2048)
    age(fake)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake))

    result = cache_cleanup.clear(["temp_files"])

    assert fake.is_dir(), "запись вне настоящего системного temp не трогается"
    assert (fake / "voice-syntez-leftover.wav").is_file()
    # Освободились только промежуточные `.tmp.wav` из output/: запись в `fake`
    # не посчитана и не удалена.
    assert result["freed_files"]["temp_files"] == 2
    assert real_temp.is_dir()


def test_unreadable_file_does_not_abort_the_rest(layout, monkeypatch):
    """Ошибка на одном файле логируется, остальные всё равно удаляются."""
    real_unlink = Path.unlink
    blocked = layout["output"] / "job1.mp3"

    def flaky_unlink(self, *args, **kwargs):
        if self == blocked:
            raise PermissionError("файл занят")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    result = cache_cleanup.clear(["output"])

    assert blocked.is_file(), "неудачное удаление не считается освобождённым"
    assert result["freed_files"]["output"] == 1
    assert result["freed_mb"]["output"] == mb(1 * MB)
    assert not (layout["output"] / "job2.wav").exists()
