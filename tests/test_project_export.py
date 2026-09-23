"""Экспорт и импорт проекта: архив `.ttsproject`, аудио и субтитры.

Тесты идут через приложение целиком (ASGI-транспорт) с движком-заглушкой: модель
не поднимается, а проверяется формат архива, безопасность распаковки, round trip и
совпадение экспортируемого времени с таймлайном. Голоса создаются в настоящем
`voices.json` (`tmp_path`), поэтому сценарий «на новой машине» воспроизводится
удалением голоса, а не подменой хранилища.
"""

import asyncio
import contextlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from conftest import analyze_project, sine

from backend import config, main, project_export
from backend.engines.base import ENGINE_F5, ENGINE_KOKORO, SAMPLE_RATE
from backend.job_queue import JobQueue
from backend.voices_store import get_store as get_voices_store

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."
TIMESTAMP = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})([,.])(\d{3}) --> "
    r"(\d{2}):(\d{2}):(\d{2})([,.])(\d{3})$"
)


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


def _wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def _make_voice(name: str, freq: float = 180.0):
    return get_voices_store().create(
        name=name,
        gender="male",
        ref_text="Привет, это тест",
        audio_filename="ref.wav",
        audio_bytes=_wav_bytes(sine(2.0, freq)),
        verify_ref_text=False,
        engine=ENGINE_F5,
    )


async def _create_project(client, name: str = "Тест", text: str = DIALOGUE) -> dict:
    response = await client.post(
        "/api/projects", json={"name": name, "source_text": text, "mode": "dialogue"}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _assign(client, project_id: str, voice_id: str) -> dict:
    parsed = await client.post(f"/api/projects/{project_id}/parse", json={})
    assert parsed.status_code == 200, parsed.text
    speakers = {
        item["speaker"]: {"voice_id": voice_id} for item in parsed.json()["replicas"]
    }
    response = await client.patch(
        f"/api/projects/{project_id}", json={"speakers": speakers}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _rendered_project(client, voice_id: str, text: str = DIALOGUE) -> dict:
    project = await _create_project(client, text=text)
    await _assign(client, project["id"], voice_id)
    await analyze_project(client, project["id"])
    accepted = await client.post(
        f"/api/projects/{project['id']}/render",
        json={"output_format": "wav", "pause_ms": 400},
    )
    assert accepted.status_code == 202, accepted.text
    await _wait_job(client, accepted.json()["job_id"])
    response = await client.get(f"/api/projects/{project['id']}")
    assert response.status_code == 200
    return response.json()


async def _export_archive(client, project_id: str) -> bytes:
    response = await client.post(f"/api/projects/{project_id}/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    return response.content


async def _import(client, data: bytes, filename: str = "project.ttsproject"):
    return await client.post(
        "/api/projects/import", files={"file": (filename, data, "application/zip")}
    )


async def _timeline(client, project_id: str) -> dict:
    response = await client.get(f"/api/projects/{project_id}/timeline")
    assert response.status_code == 200, response.text
    return response.json()


def _zip_names(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.namelist()


def _manifest(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read(project_export.MANIFEST_NAME))


def _rewrite(data: bytes, mutate) -> bytes:
    """Тот же архив с правленым `project.json` — для проверок версии формата."""
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                raw = source.read(info.filename)
                if info.filename == project_export.MANIFEST_NAME:
                    payload = json.loads(raw)
                    mutate(payload)
                    raw = json.dumps(payload, ensure_ascii=False).encode()
                target.writestr(info.filename, raw)
    return buffer.getvalue()


def _stamp(hours: str, minutes: str, seconds: str, millis: str) -> int:
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


# --- 1-4. round trip: проект, метаданные, голоса, take'ы ----------------------
def test_export_import_round_trip_restores_project(stub, monkeypatch):
    """Архив → импорт: проект, спикеры, реплики, голоса и take'ы восстановлены."""
    async def scenario():
        async with _client(monkeypatch) as client:
            ivan = _make_voice("Иван")
            margo = _make_voice("Марго", freq=240.0)
            project = await _rendered_project(client, ivan.id)
            # Свой голос второй реплике: проверяем, что override едет отдельно от
            # назначения спикера.
            patched = await client.patch(
                f"/api/projects/{project['id']}/replicas/1", json={"voice_id": margo.id}
            )
            assert patched.status_code == 200, patched.text
            project = (await client.get(f"/api/projects/{project['id']}")).json()

            data = await _export_archive(client, project["id"])
            assert project_export.MANIFEST_NAME in _zip_names(data)

            response = await _import(client, data)
            assert response.status_code == 201, response.text
            imported = response.json()

            assert imported["id"] != project["id"]
            assert imported["name"] == project["name"]
            assert imported["mode"] == project["mode"]
            assert imported["source_text"] == project["source_text"]
            assert imported["render_settings"] == project["render_settings"]
            assert imported["status"] == config.PROJECT_STATUS_RENDERED
            assert imported["replicas_count"] == 2

            assert [item["label"] for item in imported["speakers"]] == ["ИВАН", "МАРГО"]
            assert [item["voice_id"] for item in imported["speakers"]] == [
                ivan.id, ivan.id
            ]
            assert [item["text"] for item in imported["replicas"]] == [
                "Первая реплика.", "Вторая реплика."
            ]
            assert imported["replicas"][0]["voice_override"] is None
            assert imported["replicas"][1]["voice_override"] == margo.id

            for index, replica in enumerate(imported["replicas"]):
                assert len(replica["takes"]) == 1
                take = replica["takes"][0]
                original = project["replicas"][index]["takes"][0]
                assert take["id"] == replica["selected_take_id"]
                assert take["engine"] == ENGINE_F5
                assert take["duration_sec"] == pytest.approx(original["duration_sec"])
                assert take["parameters"] == original["parameters"]
                # файл на месте и играется через API
                assert Path(take["audio_path"]).exists()
                audio = await client.get(
                    f"/api/projects/{imported['id']}/replicas/{index}/takes/{take['id']}/audio"
                )
                assert audio.status_code == 200, audio.text
                assert len(audio.content) > 44  # wav-заголовок и данные

            reopened = await client.get(f"/api/projects/{imported['id']}")
            assert reopened.status_code == 200
            assert reopened.json()["replicas_count"] == 2
            assert reopened.json()["status"] == config.PROJECT_STATUS_RENDERED

            # Импортированный проект полностью рабочий: таймлайн считается, а
            # рендер запускается и добавляет репликам новые take'ы.
            timeline_data = await _timeline(client, imported["id"])
            assert timeline_data["duration_sec"] > 0
            # Подготовка после импорта выполняется заново, и это правильно: текст
            # готовится под словарь и движки **этой** машины, а архив переносит
            # проект, а не результаты подготовки с другого компьютера.
            await analyze_project(client, imported["id"])
            accepted = await client.post(
                f"/api/projects/{imported['id']}/render",
                json={"output_format": "wav", "pause_ms": 400},
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])
            rerendered = (await client.get(f"/api/projects/{imported['id']}")).json()
            assert rerendered["status"] == config.PROJECT_STATUS_RENDERED
            assert [len(item["takes"]) for item in rerendered["replicas"]] == [2, 2]

    _run(scenario)


# --- 3. голоса на «новой машине» ----------------------------------------------
def test_builtin_voice_travels_without_a_reference(stub, monkeypatch):
    """Голос встроенного движка едет в архив без референса и восстанавливается.

    Отсутствие записи у такого голоса — не потерянный файл, а устройство движка
    (`EngineInfo.builtin_voices`): импорт обязан завести карточку по имени, полу и
    движку, а не пропустить её как голос без референса.
    """

    async def scenario():
        async with _client(monkeypatch) as client:
            voice = get_voices_store().create(
                name="Света",
                gender="female",
                ref_text="",
                audio_filename="",
                audio_bytes=b"",
                engine=ENGINE_KOKORO,
            )
            project = await _create_project(client)
            await _assign(client, project["id"], voice.id)

            data = await _export_archive(client, project["id"])
            entries = _manifest(data)["voices"]
            assert [entry["name"] for entry in entries] == ["Света"]
            assert entries[0]["reference"] is None
            # Референсов в архиве нет вовсе: переносить нечего, и пустой каталог не
            # создаётся — иначе «нет записи» выглядело бы как потерянная.
            assert [name for name in _zip_names(data) if name.startswith("references/")] == []

            # «Новая машина»: голоса здесь нет — ни записи, ни референса.
            assert get_voices_store().delete(voice.id) is True

            response = await _import(client, data)
            assert response.status_code == 201, response.text
            created = get_voices_store().list()
            assert [item.name for item in created] == ["Света"]
            assert [item.engine for item in created] == [ENGINE_KOKORO]
            assert created[0].audio_file == ""
            assert [item["voice_id"] for item in response.json()["speakers"]] == [
                created[0].id, created[0].id
            ]

    _run(scenario)


def test_voice_is_recreated_on_new_machine_and_reused_again(stub, monkeypatch):
    """Голос сопоставляется по имени: на новой машине создаётся, на своей — нет."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            data = await _export_archive(client, project["id"])

            # «Новая машина»: голоса здесь нет — ни записи, ни референса.
            assert get_voices_store().delete(voice.id) is True
            assert get_voices_store().list() == []

            response = await _import(client, data)
            assert response.status_code == 201, response.text
            imported = response.json()
            created = get_voices_store().list()
            assert [item.name for item in created] == ["Иван"]
            assert [item.ref_text for item in created] == ["Привет, это тест"]
            assert [item.engine for item in created] == [ENGINE_F5]
            assert created[0].audio_path.exists()
            assert [item["voice_id"] for item in imported["speakers"]] == [
                created[0].id, created[0].id
            ]
            assert imported["replicas"][0]["voice_override"] is None

            # Повторный импорт на этой же машине: голос уже есть — новый не плодим.
            again = await _import(client, data)
            assert again.status_code == 201, again.text
            assert [item.name for item in get_voices_store().list()] == ["Иван"]
            assert [item["voice_id"] for item in again.json()["speakers"]] == [
                created[0].id, created[0].id
            ]

    _run(scenario)


# --- 5. отсутствующий необязательный take --------------------------------------
def test_missing_take_does_not_break_export_or_import(stub, monkeypatch):
    """Запись take'а без файла помечается недоступной, остальное восстанавливается."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            lost = Path(project["replicas"][0]["takes"][0]["audio_path"])
            lost.unlink()

            data = await _export_archive(client, project["id"])
            manifest = _manifest(data)
            assert [item["available"] for item in manifest["takes"]] == [False, True]
            assert next(item["file"] for item in manifest["takes"]) is None

            response = await _import(client, data)
            assert response.status_code == 201, response.text
            imported = response.json()
            assert [len(item["takes"]) for item in imported["replicas"]] == [0, 1]
            take = imported["replicas"][1]["takes"][0]
            assert Path(take["audio_path"]).exists()

    _run(scenario)


# --- 6. path traversal ---------------------------------------------------------
def test_path_traversal_in_zip_is_rejected(stub, monkeypatch):
    """`../`, абсолютный путь и символическая ссылка отклоняются до распаковки."""
    async def scenario():
        async with _client(monkeypatch) as client:
            before = {item["id"] for item in (await client.get("/api/projects")).json()["projects"]}
            manifest = json.dumps({"format": "ttsproject", "format_version": 1})
            for name in ("../evil.txt", "/tmp/tts-export-evil.txt"):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr(project_export.MANIFEST_NAME, manifest)
                    archive.writestr(name, "evil")
                response = await _import(client, buffer.getvalue())
                assert response.status_code == 400, (name, response.text)
                assert "путь" in response.json()["detail"].lower()

            link = io.BytesIO()
            with zipfile.ZipFile(link, "w") as archive:
                archive.writestr(project_export.MANIFEST_NAME, manifest)
                info = zipfile.ZipInfo("takes/link.wav")
                info.external_attr = (0o120777 << 16) | 0o777
                archive.writestr(info, "/etc/passwd")
            response = await _import(client, link.getvalue())
            assert response.status_code == 400, response.text
            assert "ссылк" in response.json()["detail"].lower()

            assert not Path("/tmp/tts-export-evil.txt").exists()
            after = {item["id"] for item in (await client.get("/api/projects")).json()["projects"]}
            assert after == before  # ни одного проекта не создано

    _run(scenario)


# --- 7. битый архив ------------------------------------------------------------
def test_corrupt_archive_gives_clear_error(stub, monkeypatch):
    """Не ZIP, ZIP без манифеста и чужая версия формата — понятная ошибка, не трейсбек."""
    async def scenario():
        async with _client(monkeypatch) as client:
            before = {item["id"] for item in (await client.get("/api/projects")).json()["projects"]}

            response = await _import(client, "это вовсе не архив".encode())
            assert response.status_code == 400
            assert "ZIP" in response.json()["detail"]
            assert response.json()["detail"] == (
                "Файл не является ZIP-архивом проекта (.ttsproject)"
            )

            empty = io.BytesIO()
            with zipfile.ZipFile(empty, "w") as archive:
                archive.writestr("takes/r001-t001.wav", b"x")
            response = await _import(client, empty.getvalue())
            assert response.status_code == 400
            assert "project.json" in response.json()["detail"]

            assert {
                item["id"] for item in (await client.get("/api/projects")).json()["projects"]
            } == before

            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            data = await _export_archive(client, project["id"])
            with_other_version = _rewrite(data, lambda payload: payload.update({"format_version": 99}))
            response = await _import(client, with_other_version)
            assert response.status_code == 400
            assert "99" in response.json()["detail"]
            assert "версия" in response.json()["detail"].lower()

            # Неудачный импорт не оставил ни проекта, ни его файлов.
            ids = {item["id"] for item in (await client.get("/api/projects")).json()["projects"]}
            assert ids == before | {project["id"]}

    _run(scenario)


# --- 8. веса моделей не экспортируются -----------------------------------------
def test_model_weights_are_never_exported(stub, monkeypatch):
    """В архиве нет ни весов, ни путей, похожих на каталог моделей."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            names = _zip_names(await _export_archive(client, project["id"]))

            assert names
            for name in names:
                parts = [part.lower() for part in name.split("/")]
                assert "models" not in parts
                assert not name.lower().endswith(
                    (".pth", ".safetensors", ".ckpt", ".pt", ".bin")
                )
                assert Path(name).name.lower() != "vocab.json"
            assert any(name.startswith(f"{project_export.REFERENCE_DIR}/") for name in names)
            assert any(name.startswith(f"{project_export.TAKE_DIR}/") for name in names)

    _run(scenario)


# --- экспорт аудио: длительность совпадает с таймлайном ------------------------
def test_final_audio_length_matches_timeline(stub, monkeypatch):
    """Собранный итоговый трек звучит ровно столько, сколько обещает таймлайн."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            timeline_data = await _timeline(client, project["id"])

            wav = await client.get(f"/api/projects/{project['id']}/export/audio?format=wav")
            assert wav.status_code == 200, wav.text
            assert wav.headers["content-type"].startswith("audio/wav")
            audio, rate = sf.read(io.BytesIO(wav.content), dtype="float32")
            assert rate == SAMPLE_RATE
            assert len(audio) / rate == pytest.approx(timeline_data["duration_sec"], abs=1e-3)

            # Сумма реплик и паузы таймлайна — ровно то же число.
            durations = [item["duration_sec"] for item in timeline_data["replicas"]]
            pause = timeline_data["settings"]["pause_ms"] / 1000
            assert len(audio) / rate == pytest.approx(sum(durations) + pause, abs=1e-3)

            mp3 = await client.get(f"/api/projects/{project['id']}/export/audio?format=mp3")
            assert mp3.status_code == 200, mp3.text
            assert mp3.headers["content-type"].startswith("audio/mpeg")
            assert len(mp3.content) > 1000

            bad = await client.get(
                f"/api/projects/{project['id']}/export/audio?format=flac"
            )
            assert bad.status_code == 400
            assert "flac" in bad.json()["detail"]

    _run(scenario)


# --- 9. SRT ---------------------------------------------------------------------
def test_srt_timestamps_are_valid(stub, monkeypatch):
    """SRT: `HH:MM:SS,mmm`, монотонные и равные границам таймлайна."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            timeline_data = await _timeline(client, project["id"])

            response = await client.get(
                f"/api/projects/{project['id']}/export/subtitles?format=srt"
            )
            assert response.status_code == 200, response.text
            blocks = [block for block in response.text.strip().split("\n\n") if block.strip()]
            sounding = [item for item in timeline_data["replicas"] if item["has_audio"]]
            assert len(blocks) == len(sounding) == 2

            previous_end = -1
            for number, block in enumerate(blocks, start=1):
                lines = block.split("\n")
                assert lines[0] == str(number)
                match = TIMESTAMP.match(lines[1])
                assert match, lines[1]
                assert match.group(4) == ","
                assert match.group(9) == ","
                start = _stamp(*match.groups()[:3], match.group(5))
                end = _stamp(*match.groups()[5:8], match.group(10))
                assert start < end
                assert start >= previous_end
                previous_end = end
                segment = sounding[number - 1]
                assert start == round(segment["start_sec"] * 1000)
                assert end == round(segment["end_sec"] * 1000)
                assert lines[2] == segment["text"]

            bad = await client.get(
                f"/api/projects/{project['id']}/export/subtitles?format=txt"
            )
            assert bad.status_code == 400
            assert "txt" in bad.json()["detail"]

    _run(scenario)


# --- 10. VTT --------------------------------------------------------------------
def test_vtt_timestamps_are_valid(stub, monkeypatch):
    """VTT: заголовок `WEBVTT`, формат `HH:MM:SS.mmm` и те же границы."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            timeline_data = await _timeline(client, project["id"])

            response = await client.get(
                f"/api/projects/{project['id']}/export/subtitles?format=vtt"
            )
            assert response.status_code == 200, response.text
            assert response.text.startswith("WEBVTT\n\n")
            assert response.headers["content-type"].startswith("text/vtt")

            blocks = [block for block in response.text[len("WEBVTT\n\n"):].strip().split("\n\n")]
            sounding = [item for item in timeline_data["replicas"] if item["has_audio"]]
            assert len(blocks) == len(sounding) == 2
            for number, block in enumerate(blocks, start=1):
                lines = block.split("\n")
                assert lines[0] == str(number)
                match = TIMESTAMP.match(lines[1])
                assert match, lines[1]
                assert match.group(4) == "."
                assert match.group(9) == "."
                start = _stamp(*match.groups()[:3], match.group(5))
                end = _stamp(*match.groups()[5:8], match.group(10))
                assert start == round(sounding[number - 1]["start_sec"] * 1000)
                assert end == round(sounding[number - 1]["end_sec"] * 1000)

    _run(scenario)


# --- транскрипт -----------------------------------------------------------------
def test_transcript_matches_timeline(stub, monkeypatch):
    """Транскрипт: реплика, спикер, голос, текст и реальные start/end."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            timeline_data = await _timeline(client, project["id"])

            response = await client.get(
                f"/api/projects/{project['id']}/export/transcript"
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["project_id"] == project["id"]
            assert payload["duration_sec"] == timeline_data["duration_sec"]
            assert [item["text"] for item in payload["segments"]] == [
                "Первая реплика.", "Вторая реплика."
            ]
            assert [item["speaker_label"] for item in payload["segments"]] == [
                "ИВАН", "МАРГО"
            ]
            assert {item["voice_name"] for item in payload["segments"]} == {"Иван"}
            for segment, timing in zip(payload["segments"], timeline_data["replicas"]):
                assert segment["start_sec"] == timing["start_sec"]
                assert segment["end_sec"] == timing["end_sec"]

    _run(scenario)


# --- 11. stems ------------------------------------------------------------------
def test_stems_contain_only_their_speaker_and_stay_synchronous(stub, monkeypatch):
    """Stems: у каждого спикера звучит только его реплика, дорожки одной длины."""
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            timeline_data = await _timeline(client, project["id"])

            response = await client.get(
                f"/api/projects/{project['id']}/export/stems"
            )
            assert response.status_code == 200, response.text
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = sorted(archive.namelist())
                assert names == ["stem-01.wav", "stem-02.wav"]
                tracks = {}
                for name in names:
                    audio, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32")
                    assert rate == SAMPLE_RATE
                    tracks[name] = audio

            lengths = {len(audio) for audio in tracks.values()}
            assert len(lengths) == 1
            total = next(iter(lengths))
            # Дорожки синхронны с итоговым треком: та же длина, что у таймлайна.
            assert total == pytest.approx(timeline_data["duration_sec"] * SAMPLE_RATE, abs=24)

            first, second = timeline_data["replicas"]

            def energy(audio: np.ndarray, segment: dict) -> float:
                start = round(segment["start_sec"] * SAMPLE_RATE)
                end = round(segment["end_sec"] * SAMPLE_RATE)
                return float(np.abs(audio[start:end]).sum())

            assert energy(tracks["stem-01.wav"], first) > 0
            assert energy(tracks["stem-02.wav"], second) > 0
            # Чужая реплика — ровная тишина: иначе дорожки нельзя наложить.
            assert energy(tracks["stem-01.wav"], second) == 0
            assert energy(tracks["stem-02.wav"], first) == 0

    _run(scenario)


# --- реплики отдельными WAV -----------------------------------------------------
def test_replica_export_contains_every_sounding_replica(stub, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            response = await client.get(
                f"/api/projects/{project['id']}/export/replicas"
            )
            assert response.status_code == 200, response.text
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = sorted(archive.namelist())
                assert names == ["r001.wav", "r002.wav"]
                for name in names:
                    audio, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32")
                    assert rate == SAMPLE_RATE and audio.size > 0

    _run(scenario)


# --- повторный импорт и чужие id ------------------------------------------------
def test_repeated_import_creates_second_project_without_touching_first(stub, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _rendered_project(client, voice.id)
            data = await _export_archive(client, project["id"])

            first = (await _import(client, data)).json()
            second = (await _import(client, data)).json()
            assert len({project["id"], first["id"], second["id"]}) == 3
            assert first["replicas_count"] == second["replicas_count"] == 2

            listed = (await client.get("/api/projects")).json()["projects"]
            assert len(listed) == 3
            # Исходный проект не тронут: его take'ы на месте.
            still = (await client.get(f"/api/projects/{project['id']}")).json()
            assert still["status"] == config.PROJECT_STATUS_RENDERED
            assert [len(item["takes"]) for item in still["replicas"]] == [1, 1]
            assert len(get_voices_store().list()) == 1

    _run(scenario)


# --- 404 на неизвестный проект и пустой экспорт ---------------------------------
def test_export_endpoints_return_404_for_unknown_project(stub, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            for url in (
                "export",
                "export/audio",
                "export/replicas",
                "export/stems",
                "export/transcript",
                "export/subtitles",
            ):
                response = await client.post(f"/api/projects/нет-такого/{url}") \
                    if url == "export" else await client.get(f"/api/projects/нет-такого/{url}")
                assert response.status_code == 404, (url, response.text)
                assert "не найден" in response.json()["detail"]

    _run(scenario)


def test_audio_export_without_takes_is_a_clear_error(stub, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            voice = _make_voice("Иван")
            project = await _create_project(client)
            await _assign(client, project["id"], voice.id)
            response = await client.get(
                f"/api/projects/{project['id']}/export/audio?format=wav"
            )
            assert response.status_code == 400
            assert "звучани" in response.json()["detail"]

    _run(scenario)


def test_import_rejects_junk_without_creating_project(stub, monkeypatch):
    """Мусорный файл и архив без версии — 400, и в проектах ничего не появилось."""
    async def scenario():
        async with _client(monkeypatch) as client:
            for data in (
                b"",
                b"PK\x03\x04 not really a zip",
            ):
                response = await _import(client, data)
                assert response.status_code == 400, response.text
            assert (await client.get("/api/projects")).json()["projects"] == []

    _run(scenario)
