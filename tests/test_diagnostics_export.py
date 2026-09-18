"""Диагностика качества озвучки: архив с текстом, настройками, планом и аудио.

Тесты идут через приложение (ASGI-транспорт) и движок-заглушку: модели не
поднимаются, а проверяется то, ради чего архив существует — что по нему можно
ответить на вопрос «почему эта реплика звучит плохо». Поэтому проверяется не
факт наличия ZIP, а состав срезов: исходный и подготовленный текст, эффективные
настройки рендера, решение слоя коротких реплик, аудио реплик и готового трека,
хвост журнала и отсутствие в JSON путей с домашним каталогом.

Отдельно проверяется имя готового файла: пользователь задаёт его руками, значит
оно проходит через путь на диске — санитайз и поведение при совпадении имён
проверяются как контракт, а не как деталь.
"""

import asyncio
import contextlib
import io
import json
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
import soundfile as sf
from conftest import analyze_project, sine

from backend import config, diagnostics, main
from backend.audio_pipeline import safe_output_name
from backend.engines.base import ENGINE_F5, SAMPLE_RATE
from backend.job_queue import JobQueue
from backend.voices_store import get_store as get_voices_store

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."
# Короткий диалог: на нём слой коротких реплик строит план, и в архив попадает
# то, ради чего он собирается — решение слоя и текст, ушедший в модель.
SHORT_DIALOGUE = "ИВАН: Да.\nМАРГО: Ещё."


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


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Ollama в тестах не опрашивается: диагностика не должна зависеть от неё.

    Патчится только `status()` живого анализатора: он ходит по HTTP (health,
    список моделей). Подменять `get_analyzer` целиком нельзя — анализатор нужен и
    обязательной подготовке текста, и тест проверял бы не тот путь.
    """
    monkeypatch.setattr(
        diagnostics.llm_analyzer.LinguisticAnalyzer,
        "status",
        lambda self: {"enabled": False, "state": "disabled"},
    )


def _make_voice(name: str = "Иван"):
    store = get_voices_store()
    return store.create(
        name=name,
        gender="male",
        ref_text="Привет, это тест",
        audio_filename="ref.wav",
        audio_bytes=_wav_bytes(sine(2.0, 180.0)),
        verify_ref_text=False,
        engine=ENGINE_F5,
    )


def _wav_bytes(audio) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


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


async def _rendered_project(
    client,
    voice_id: str,
    output_name: str = "",
    *,
    text: str = DIALOGUE,
    short_layer: bool = False,
) -> tuple[dict, dict]:
    """Проект, доведённый до звука: возвращаются проект и статус готовой задачи."""
    created = await client.post(
        "/api/projects", json={"name": "Тест", "source_text": text, "mode": "dialogue"}
    )
    assert created.status_code == 201, created.text
    project = created.json()
    parsed = await client.post(f"/api/projects/{project['id']}/parse", json={})
    assert parsed.status_code == 200, parsed.text
    speakers = {
        item["speaker"]: {"voice_id": voice_id} for item in parsed.json()["replicas"]
    }
    patched = await client.patch(f"/api/projects/{project['id']}", json={"speakers": speakers})
    assert patched.status_code == 200, patched.text
    await analyze_project(client, project["id"])
    render_body: dict = {
        "output_format": "wav",
        "pause_ms": 400,
        "output_name": output_name,
    }
    if short_layer:
        # Без выключения «Авто» слой коротких реплик на заглушке ничего не делает:
        # измеренная политика для неизвестного движка — direct, то есть прежний путь.
        render_body["short_utterance"] = {"enabled": True, "strategy": "auto"}
    accepted = await client.post(
        f"/api/projects/{project['id']}/render", json=render_body
    )
    assert accepted.status_code == 202, accepted.text
    job = await _wait_job(client, accepted.json()["job_id"])
    response = await client.get(f"/api/projects/{project['id']}")
    assert response.status_code == 200
    return response.json(), job


def _archive(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def _read_json(data: bytes, name: str) -> dict:
    with _archive(data) as archive:
        return json.loads(archive.read(name))


# --- 1. состав архива ---------------------------------------------------------
def test_diagnostics_archive_carries_settings_text_and_audio(stub, workspace, monkeypatch):
    """Архив отвечает на «почему звучит плохо»: текст, настройки, план, аудио, журнал."""
    log = workspace / "voice_syntez.log"
    log.write_text(
        "2025-01-01 10:00:00 INFO tts.audio_pipeline: Реплика 1 из 2: граница цели не найдена\n",
        encoding="utf-8",
    )

    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, job = await _rendered_project(client, voice.id, output_name="разбор-1")

            created = await client.post(
                f"/api/projects/{project['id']}/diagnostics",
                json={"job_id": job["job_id"], "include_references": True},
            )
            assert created.status_code == 201, created.text
            payload = created.json()
            bundle = payload["bundle"]
            assert bundle["name"].startswith("diagnostics-Тест-")
            assert Path(payload["path"]).is_file()
            assert bundle["warnings"] == []

            downloaded = await client.get(payload["download_url"])
            assert downloaded.status_code == 200, downloaded.text
            assert downloaded.headers["content-type"] == "application/zip"
            # Имя в заголовке закодировано по RFC 5987: кириллицу браузер обязан
            # получить именно так, иначе сохранит файл с «%D0%A2…».
            assert quote(bundle["name"]) in downloaded.headers["content-disposition"]
            data = downloaded.content

            names = _archive(data).namelist()
            for required in (
                diagnostics.README_NAME,
                diagnostics.MANIFEST_NAME,
                diagnostics.PROJECT_NAME,
                diagnostics.SETTINGS_NAME,
                diagnostics.TAKES_NAME,
                diagnostics.ANALYSIS_NAME,
                diagnostics.ENVIRONMENT_NAME,
                diagnostics.LOG_NAME,
            ):
                assert required in names, f"в архиве нет {required}"
            # Готовый трек называется так, как его назвал пользователь.
            assert f"{diagnostics.FINAL_DIR}/разбор-1.wav" in names
            assert any(name.startswith(f"{diagnostics.TAKE_DIR}/") for name in names)
            assert any(name.startswith(f"{diagnostics.REFERENCE_DIR}/") for name in names)

            # README — инструкция, а не заглушка: в нём есть порядок разбора.
            readme = _archive(data).read(diagnostics.README_NAME).decode("utf-8")
            assert "короткая реплика звучит плохо" in readme
            assert diagnostics.TAKES_NAME in readme

            # Текст: и исходный, и подготовленный, по каждой реплике по отдельности.
            project_slice = _read_json(data, diagnostics.PROJECT_NAME)
            assert project_slice["source_text"] == DIALOGUE
            assert project_slice["replicas"][0]["source_text"] == "Первая реплика."
            assert project_slice["replicas"][0]["final_text"]
            # Путь к файлу голоса не уезжает в архив: только его имя.
            voice_layer = project_slice["replicas"][0]["voices"]
            assert voice_layer["reference_file"].endswith(".wav")
            assert "/" not in voice_layer["reference_file"]
            # Ни один срез не должен нести пути этой машины.
            assert str(workspace) not in json.dumps(project_slice, ensure_ascii=False)
            assert "/Users/" not in json.dumps(project_slice, ensure_ascii=False)

            # Настройки: эффективные (с именем файла) и политика коротких реплик.
            settings = _read_json(data, diagnostics.SETTINGS_NAME)
            assert settings["effective"]["output_format"] == "wav"
            assert settings["effective"]["output_name"] == "разбор-1"
            assert settings["short_utterance_policy"]["strategies_by_engine"]
            assert settings["qa"]["modes"] == list(config.QA_MODES)

            # Take'ы: движок, сид, параметры куска и решение слоя коротких реплик.
            takes = _read_json(data, diagnostics.TAKES_NAME)
            first = takes["replicas"][0]["takes"][0]
            assert first["engine"] == ENGINE_F5
            # Параметры куска — то, чем он получен: движок, сид и ручки синтеза.
            assert first["parameters"]["speed"]
            assert first["seed"] is not None
            assert first["audio"] in names
            # Обычная реплика: слой коротких реплик к ней не применялся, и это
            # видно в архиве явно, а не отсутствием поля.
            assert takes["replicas"][0]["short"]["applied"] is False

            # Окружение: чем синтезировали и хватало ли памяти.
            environment = _read_json(data, diagnostics.ENVIRONMENT_NAME)
            assert environment["device"]
            assert "memory_state" in environment
            assert "llm_memory_policy" in environment

            # Журнал: причина отказа слоя — из него, а не из пересказа.
            log_text = _archive(data).read(diagnostics.LOG_NAME).decode("utf-8")
            assert "граница цели не найдена" in log_text

            # Список архивов проекта и удаление — тот же файл, что скачали.
            listed = await client.get(f"/api/projects/{project['id']}/diagnostics")
            assert listed.status_code == 200
            assert [item["name"] for item in listed.json()["archives"]] == [bundle["name"]]

            removed = await client.delete(
                f"/api/projects/{project['id']}/diagnostics/{bundle['name']}"
            )
            assert removed.status_code == 200
            assert not Path(payload["path"]).exists()

    _run(scenario)


def test_diagnostics_archive_without_job_uses_project_settings(stub, monkeypatch):
    """Без `job_id` архив собирается по проекту: настройки — те, с которыми он соберётся сейчас."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, _ = await _rendered_project(client, voice.id)

            created = await client.post(
                f"/api/projects/{project['id']}/diagnostics", json={}
            )
            assert created.status_code == 201, created.text
            payload = created.json()

            settings = _read_json(
                Path(payload["path"]).read_bytes(), diagnostics.SETTINGS_NAME
            )
            # Сохранённое в проекте и эффективное совпадают, а имя файла пустое:
            # задавать его в архив нечем — рендера с именем не было.
            assert settings["effective"]["output_format"] == "wav"
            assert settings["effective"]["output_name"] == ""
            assert settings["saved_in_project"]["output_format"] == "wav"
            # Готового трека в архиве нет, и это не ошибка сборки.
            assert not any(
                name.startswith(f"{diagnostics.FINAL_DIR}/")
                for name in _archive(Path(payload["path"]).read_bytes()).namelist()
            )

    _run(scenario)


def test_diagnostics_budget_keeps_json_slices(stub, monkeypatch):
    """При жёстком пределе аудио архив всё равно пригоден: срезы на месте, причина названа."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, job = await _rendered_project(client, voice.id)

            created = await client.post(
                f"/api/projects/{project['id']}/diagnostics",
                json={"job_id": job["job_id"], "max_audio_mb": 0.0001},
            )
            assert created.status_code == 201, created.text
            payload = created.json()
            warnings = payload["bundle"]["warnings"]
            assert any("аудио" in item for item in warnings), warnings

            data = Path(payload["path"]).read_bytes()
            names = _archive(data).namelist()
            assert not [name for name in names if name.startswith("audio/")]
            assert diagnostics.TAKES_NAME in names
            manifest = _read_json(data, diagnostics.MANIFEST_NAME)
            assert manifest["audio_skipped"]
            assert manifest["warnings"] == warnings
            readme = _archive(data).read(diagnostics.README_NAME).decode("utf-8")
            assert "Что не поместилось" in readme

    _run(scenario)


def test_diagnostics_missing_log_is_reported_not_silent(stub, workspace, monkeypatch):
    """Нет журнала — архив собирается, но README и манифест говорят об этом прямо."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, _ = await _rendered_project(client, voice.id)

            created = await client.post(f"/api/projects/{project['id']}/diagnostics", json={})
            assert created.status_code == 201, created.text
            payload = created.json()
            warnings = payload["bundle"]["warnings"]
            assert any("лог" in item for item in warnings), warnings
            data = Path(payload["path"]).read_bytes()
            assert diagnostics.LOG_NAME not in _archive(data).namelist()
            readme = _archive(data).read(diagnostics.README_NAME).decode("utf-8")
            assert "Предупреждения сборки" in readme

    _run(scenario)


# --- 2. имя готового файла ----------------------------------------------------
def test_named_render_uses_user_file_name(stub, monkeypatch):
    """Заданное имя становится именем файла и видно интерфейсу в статусе задачи."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, job = await _rendered_project(client, voice.id, output_name="Мой диалог")

            assert job["file_name"] == "Мой_диалог.wav"
            assert (config.OUTPUT_DIR / "Мой_диалог.wav").is_file()
            # Проект запоминает настройки сборки, но не имя файла: оно относится к
            # запуску, а не к диалогу.
            assert "output_name" not in (project["render_settings"] or {})

            # Второй рендер с тем же именем не перезаписывает первый файл.
            accepted = await client.post(
                f"/api/projects/{project['id']}/render",
                json={"output_format": "wav", "output_name": "Мой диалог"},
            )
            assert accepted.status_code == 202, accepted.text
            second = await _wait_job(client, accepted.json()["job_id"])
            assert second["file_name"] == "Мой_диалог-2.wav"
            assert (config.OUTPUT_DIR / "Мой_диалог-2.wav").is_file()
            assert (config.OUTPUT_DIR / "Мой_диалог.wav").is_file()

    _run(scenario)


def test_render_without_name_keeps_job_id_file(stub, monkeypatch):
    """Пустое поле — прежнее поведение: имя из номера задачи, а не «-2»."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            _, job = await _rendered_project(client, voice.id)
            assert job["file_name"] == f"{job['job_id']}.wav"

    _run(scenario)


def test_render_text_accepts_output_name(stub, monkeypatch):
    """Сплошной текст: то же имя файла, что и у диалога — путь один."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            response = await client.post(
                "/api/render-text",
                json={
                    "text": "Первое предложение. Второе предложение.",
                    "voice_id": voice.id,
                    "output_format": "wav",
                    "output_name": "глава 1",
                },
            )
            assert response.status_code == 202, response.text
            job = await _wait_job(client, response.json()["job_id"])
            assert job["file_name"] == "глава_1.wav"
            assert (config.OUTPUT_DIR / "глава_1.wav").is_file()

    _run(scenario)


# --- 3. санитайз имени --------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("Мой диалог", "Мой_диалог"),
        ("  с пробелами  ", "с_пробелами"),
        # Расширение подставляет пайплайн, а не пользователь: иначе вышло бы .wav.wav.
        ("диалог.wav", "диалог"),
        ("диалог.MP3", "диалог"),
        # Путь не должен уводить запись из output/: разделители и точки уходят.
        ("../../etc/passwd", "etc_passwd"),
        ("a/b", "a_b"),
        ("..", ""),
        ("...", ""),
        ("отчёт: 2025!", "отчёт_2025"),
        ("скрытый/файл", "скрытый_файл"),
    ],
)
def test_safe_output_name(raw, expected):
    assert safe_output_name(raw) == expected


def test_safe_output_name_is_bounded():
    """Длинное имя обрезается: имя файла нужно для скачивания и просмотра каталога."""
    long_name = "а" * 200
    cleaned = safe_output_name(long_name)
    assert len(cleaned) == 60
    assert not cleaned.endswith("-")
    # Кириллица остаётся кириллицей: транслит здесь был бы неожиданностью.
    assert safe_output_name("диалог") == "диалог"


def test_resolve_archive_rejects_paths_outside_diagnostics_dir():
    """Имя архива из URL не может вывести за каталог диагностики."""
    for bad in ("../data/voice_syntez.db", "sub/archive.zip", "archive.tar", "archive"):
        with pytest.raises(diagnostics.DiagnosticsError):
            diagnostics.resolve_archive(bad)


# --- 4. слой коротких реплик в архиве ------------------------------------------
def test_diagnostics_archive_shows_short_utterance_decision(stub, workspace, monkeypatch):
    """Ради этого архив и собирается: видно решение слоя и текст, ушедший в модель."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, job = await _rendered_project(
                client, voice.id, text=SHORT_DIALOGUE, short_layer=True
            )

            created = await client.post(
                f"/api/projects/{project['id']}/diagnostics",
                json={"job_id": job["job_id"]},
            )
            assert created.status_code == 201, created.text
            data = Path(created.json()["path"]).read_bytes()

            takes = _read_json(data, diagnostics.TAKES_NAME)
            short = takes["replicas"][0]["short"]
            assert short["applied"] is True
            assert short["strategy"], short
            assert short["synthesis_text"]
            assert short["words"] <= 2
            first = takes["replicas"][0]["takes"][0]
            assert first["parameters"]["short_utterance_strategy"] == short["strategy"]
            assert first["parameters"]["tts_synthesis_text"] == short["synthesis_text"]
            # Откат слоя виден отдельным полем, а не выводится из лога: пустая
            # строка означает «плана не было», а не «откат не случился».
            assert "short_utterance_fallback" in first["parameters"]
            # Настройки рендера помнят, что слой запрашивали.
            settings = _read_json(data, diagnostics.SETTINGS_NAME)
            assert settings["effective"]["short_utterance"]["enabled"] is True

    _run(scenario)


def test_diagnostics_archives_are_scoped_to_project(stub, monkeypatch):
    """Два проекта с одним названием не видят архивы друг друга.

    Имена проектов не уникальны, поэтому привязка идёт по идентификатору: иначе
    список архивов показывал бы чужой разбор, а ротация вытесняла бы чужие файлы.
    """
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            first, _ = await _rendered_project(client, voice.id)
            second, _ = await _rendered_project(client, voice.id)
            assert first["name"] == second["name"] == "Тест"
            assert first["id"] != second["id"]

            created = await client.post(
                f"/api/projects/{first['id']}/diagnostics", json={}
            )
            assert created.status_code == 201, created.text
            name = created.json()["bundle"]["name"]
            # В имени есть начало идентификатора: по нему архив и находится.
            assert first["id"][:8] in name

            listed = await client.get(f"/api/projects/{second['id']}/diagnostics")
            assert listed.status_code == 200
            assert listed.json()["archives"] == []
            own = await client.get(f"/api/projects/{first['id']}/diagnostics")
            assert [item["name"] for item in own.json()["archives"]] == [name]
            # Через «соседний» проект тот же файл не открывается и не удаляется:
            # принадлежность архива проверяется вместе с путём.
            wrong_download = await client.get(
                f"/api/projects/{second['id']}/diagnostics/{name}"
            )
            assert wrong_download.status_code == 404
            wrong = await client.delete(
                f"/api/projects/{second['id']}/diagnostics/{name}"
            )
            assert wrong.status_code == 404
            assert (config.DIAGNOSTICS_DIR / name).is_file()
            # Свой проект удаляет архив как обычно.
            assert (
                await client.delete(f"/api/projects/{first['id']}/diagnostics/{name}")
            ).status_code == 200
            assert not (config.DIAGNOSTICS_DIR / name).exists()

    _run(scenario)


def test_diagnostics_with_unknown_job_falls_back_to_project(stub, monkeypatch):
    """Исчезнувшая задача (перезапуск приложения) не мешает собрать архив."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice()
            project, _ = await _rendered_project(client, voice.id)

            created = await client.post(
                f"/api/projects/{project['id']}/diagnostics",
                json={"job_id": "job-которого-уже-нет"},
            )
            assert created.status_code == 201, created.text
            payload = created.json()
            assert Path(payload["path"]).is_file()
            settings = _read_json(
                Path(payload["path"]).read_bytes(), diagnostics.SETTINGS_NAME
            )
            # Настройки взяты из проекта, а не выдуманы: архив всё равно пригоден.
            assert settings["effective"]["output_format"] == "wav"

    _run(scenario)
