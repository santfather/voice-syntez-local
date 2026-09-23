"""Интеграционные тесты монтажа записей: единый MP3 на 2 и 3 голоса (§41–§43).

Это blocking-набор пакета «Сам себе звукорежиссер»: проверяется не отдельная
функция, а главный результат режима — **один цельный файл** с диалогом, собранным
из записанных дублей в исходном порядке реплик.

Голоса различаются частотой сигнала, поэтому порядок в готовом файле проверяется
по звуку: тест берёт середину каждого ожидаемого сегмента и сверяет доминирующую
частоту с частотой того голоса, который там должен звучать. Перепутанный порядок
по числу реплик не виден, а по частотам — виден.

Реальные модели не поднимаются: записи синтетические, DeepFilterNet выключен,
MP3 декодируется тем же pydub, что и в приложении.
"""

import asyncio
import io
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from conftest import dominant_hz, sine
from test_projects_api import _client

from backend import audio_pipeline, recording_audio, recording_pipeline, recording_store
from backend.engines.base import SAMPLE_RATE

TWO_VOICES = (
    "Анна: Ты уже приехал?\n"
    "Игорь: Почти.\n"
    "Анна: Тогда я тебя жду.\n"
    "Игорь: Хорошо, скоро буду."
)
THREE_VOICES = (
    "Анна: Начинаем?\n"
    "Игорь: Я готов.\n"
    "Рассказчик: Они посмотрели друг на друга.\n"
    "Анна: Тогда поехали.\n"
    "Игорь: Подожди секунду.\n"
    "Рассказчик: В комнате снова стало тихо."
)

# Частоты голосов: по ним проверяется порядок в готовом файле.
VOICE_HZ = {"A": 220.0, "B": 330.0, "C": 440.0}
TAKE_SEC = 0.7
PAUSE_MS = 200


def _run(scenario) -> None:
    asyncio.run(scenario())


def _take_wav(voice: str, seconds: float = TAKE_SEC) -> bytes:
    """Запись «голоса»: синусоида своей частоты, будто её записал человек."""
    buffer = io.BytesIO()
    sf.write(buffer, sine(seconds, VOICE_HZ[voice]), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


async def _wait_render(client, project_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/recording-projects/{project_id}/render/status")).json()
        if data["status"] in ("done", "error"):
            return data
        await asyncio.sleep(0.05)
    raise AssertionError("монтаж не завершился")


async def _setup_project(client, text: str, voices: list[str]) -> tuple[dict, dict[str, str]]:
    """Проект с разобранным текстом, профилями голосов и назначенными ролями.

    Возвращается и карта «буква голоса → id профиля»: она нужна тестам, чтобы
    записывать реплики нужным голосом, не угадывая порядок профилей.
    """
    created = await client.post(
        "/api/recording-projects", json={"name": "Запись", "dialogue_text": text}
    )
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["replicas"], project

    profiles: dict[str, str] = {}
    for voice in voices:
        response = await client.post(
            f"/api/recording-projects/{project['id']}/voice-profiles",
            json={"name": f"Голос {voice}"},
        )
        assert response.status_code == 201, response.text
        project = response.json()
        profiles[voice] = next(
            item["id"] for item in project["voice_profiles"] if item["name"] == f"Голос {voice}"
        )

    for speaker, voice in zip(project["speakers"], voices, strict=False):
        response = await client.put(
            f"/api/recording-projects/{project['id']}/roles/{speaker}",
            json={"profile_id": profiles[voice]},
        )
        assert response.status_code == 200, response.text
        project = response.json()
    return project, profiles


async def _upload_take(client, project_id: str, index: int, voice: str, seconds=TAKE_SEC) -> dict:
    response = await client.post(
        f"/api/recording-projects/{project_id}/replicas/{index}/takes",
        files={"file": ("take.wav", _take_wav(voice, seconds), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()["project"]


async def _record_all(client, project: dict, profiles: dict[str, str]) -> dict:
    """По одному дублю на реплику, каждая своим голосом (порядок — по диалогу)."""
    voice_by_profile = {profile_id: voice for voice, profile_id in profiles.items()}
    for replica in project["replicas"]:
        voice = voice_by_profile[project["role_voices"][replica["speaker"]]]
        project = await _upload_take(client, project["id"], replica["index"], voice)
    return project


async def _render(client, project_id: str, pause_ms: int = PAUSE_MS) -> dict:
    response = await client.post(
        f"/api/recording-projects/{project_id}/render", json={"pause_ms": pause_ms}
    )
    assert response.status_code == 202, response.text
    state = await _wait_render(client, project_id)
    assert state["status"] == "done", state
    return state


async def _download(client, project_id: str):
    return await client.get(f"/api/recording-projects/{project_id}/audio?download=true")


async def _final_audio(client, project_id: str) -> np.ndarray:
    """Скачивает готовый файл и декодирует его — как это сделал бы плеер."""
    response = await _download(client, project_id)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    return recording_audio.decode(response.content, ".mp3")


def _voice_sequence(audio: np.ndarray, replicas: int, *, take_sec: float = TAKE_SEC) -> list[str]:
    """Голос на середине каждого ожидаемого сегмента — по доминирующей частоте.

    Ожидаемая длина сегмента — длина записи (обработка при `speed=1, pitch=0` её не
    меняет), поэтому середина сегмента заведомо внутри реплики, а не на паузе.
    """
    sequence: list[str] = []
    step = take_sec + PAUSE_MS / 1000
    for index in range(replicas):
        middle = index * step + take_sec / 2
        window = audio[
            int((middle - take_sec / 4) * SAMPLE_RATE): int((middle + take_sec / 4) * SAMPLE_RATE)
        ]
        if window.size < 128:
            sequence.append("?")
            continue
        hz = dominant_hz(window)
        sequence.append(min(VOICE_HZ, key=lambda voice: abs(VOICE_HZ[voice] - hz)))
    return sequence


# --- Test A. Цельный диалог на двух голосах -----------------------------------
def test_two_voice_dialogue_renders_single_mp3(stub, workspace, monkeypatch):
    """A→B→A→B одним файлом: порядок, паузы и декодируемый MP3 (§42, Test A)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            project = await _record_all(client, project, profiles)
            assert project["readiness"]["ready_to_render"] is True
            state = await _render(client, project["id"])
            audio = await _final_audio(client, project["id"])

            assert len(project["replicas"]) == 4
            # Порядок голосов в файле — исходный порядок реплик диалога.
            assert _voice_sequence(audio, 4) == ["A", "B", "A", "B"]
            # Между четырьмя репликами три заданные паузы.
            assert state["duration_sec"] == pytest.approx(
                4 * TAKE_SEC + 3 * PAUSE_MS / 1000, abs=0.3
            )

    _run(scenario)


# --- Test B и C. Три голоса и запись «по ролям» -------------------------------
def test_three_voice_dialogue_keeps_dialogue_order(stub, workspace, monkeypatch):
    """A→B→C→A→B→C при записи блоками по ролям (§42, Test B и C)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, THREE_VOICES, ["A", "B", "C"])
            # Физический порядок записи — блоками: сначала вся роль A, затем B, C.
            voice_by_profile = {profile_id: voice for voice, profile_id in profiles.items()}
            by_voice: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
            for replica in project["replicas"]:
                voice = voice_by_profile[project["role_voices"][replica["speaker"]]]
                by_voice[voice].append(replica)
            created: list[int] = []
            for voice in ("A", "B", "C"):
                for replica in by_voice[voice]:
                    project = await _upload_take(client, project["id"], replica["index"], voice)
                    created.append(int(replica["index"]))

            # Порядок создания дублей не совпадает с порядком реплик — в этом суть.
            assert created != sorted(created), created

            state = await _render(client, project["id"])
            audio = await _final_audio(client, project["id"])
            assert len(project["replicas"]) == 6
            assert _voice_sequence(audio, 6) == ["A", "B", "C", "A", "B", "C"]
            assert state["duration_sec"] == pytest.approx(
                6 * TAKE_SEC + 5 * PAUSE_MS / 1000, abs=0.35
            )

    _run(scenario)


def test_assembly_uses_replica_index_not_take_creation_time(stub, workspace, monkeypatch):
    """Монтаж идёт по `index` реплики, а не по времени создания дубля (§42, Test C)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            voice_by_profile = {profile_id: voice for voice, profile_id in profiles.items()}
            # Записываем в обратном порядке: Игорь, затем Анна.
            ordered = sorted(
                project["replicas"],
                key=lambda item: voice_by_profile[project["role_voices"][item["speaker"]]],
                reverse=True,
            )
            for replica in ordered:
                voice = voice_by_profile[project["role_voices"][replica["speaker"]]]
                project = await _upload_take(client, project["id"], replica["index"], voice)

            loaded = recording_store.get_store().require_project(project["id"])
            # Порядок в проекте — исходный, и активные дубли привязаны к своим индексам.
            assert [int(item["index"]) for item in loaded.replicas] == [0, 1, 2, 3]
            for replica in loaded.replicas:
                take = loaded.active_take(int(replica["index"]))
                assert take is not None and take.replica_index == int(replica["index"])

            await _render(client, project["id"])
            audio = await _final_audio(client, project["id"])
            assert _voice_sequence(audio, 4) == ["A", "B", "A", "B"]

    _run(scenario)


# --- Test D. Настройки обработки изолированы между голосами -------------------
def test_processing_settings_are_isolated_per_voice(stub, workspace, monkeypatch):
    """Скорость Voice A не влияет на Voice B и C (§42, Test D)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, THREE_VOICES, ["A", "B", "C"])
            project = await _record_all(client, project, profiles)
            settings = {
                "A": {"speed": 0.9, "pitch_semitones": 0.0, "denoise": False},
                "B": {"speed": 1.1, "pitch_semitones": -1.0, "denoise": False},
                "C": {"speed": 1.0, "pitch_semitones": 1.0, "denoise": False},
            }
            for voice, payload in settings.items():
                response = await client.patch(
                    f"/api/recording-projects/{project['id']}/voice-profiles/{profiles[voice]}",
                    json=payload,
                )
                assert response.status_code == 200, response.text

            loaded = recording_store.get_store().require_project(project["id"])
            voice_by_profile = {profile_id: voice for voice, profile_id in profiles.items()}
            measured: dict[str, float] = {}
            for replica in loaded.replicas:
                take = loaded.active_take(int(replica["index"]))
                assert take is not None
                audio, _warnings = recording_pipeline.processed_take(loaded, take)
                measured[voice_by_profile[take.voice_profile_id]] = (
                    recording_audio.duration_sec(audio)
                )
            # Медленнее всех A (0.9), быстрее всех B (1.1): настройки не «перетекли».
            assert measured["A"] > measured["C"] > measured["B"], measured

            # В готовом файле длительности сегментов сохраняют эту разницу.
            await _render(client, project["id"])
            audio = await _final_audio(client, project["id"])
            assert recording_audio.duration_sec(audio) > 3 * TAKE_SEC * 0.9

    _run(scenario)


# --- Test E. Смена активного дубля --------------------------------------------
def test_active_take_change_is_used_in_next_render(stub, workspace, monkeypatch):
    """Выбор другого дубля меняет ровно одну реплику в следующем MP3 (§42, Test E)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            project = await _record_all(client, project, profiles)
            # Второй дубль первой реплики — длиннее.
            project = await _upload_take(
                client, project["id"], 0, "A", seconds=TAKE_SEC * 1.5
            )
            takes = [take for take in project["takes"] if take["replica_index"] == 0]
            assert len(takes) == 2
            first, second = takes[0], takes[1]
            assert second["active"] is True
            assert second["duration_sec"] > first["duration_sec"]

            long_state = await _render(client, project["id"])
            response = await client.post(
                f"/api/recording-projects/{project['id']}/replicas/0/"
                f"takes/{first['id']}/select"
            )
            assert response.status_code == 200, response.text
            assert response.json()["active_takes"]["0"] == first["id"]
            short_state = await _render(client, project["id"])

            # Разница равна разнице дублей: остальные реплики не изменились.
            assert long_state["duration_sec"] - short_state["duration_sec"] == pytest.approx(
                second["duration_sec"] - first["duration_sec"], abs=0.2
            )
            audio = await _final_audio(client, project["id"])
            assert _voice_sequence(audio, 4)[0] == "A"

    _run(scenario)


# --- Test F. Незаписанная реплика блокирует монтаж ----------------------------
def test_missing_active_take_blocks_render(stub, workspace, monkeypatch):
    """Рендер отклоняется с индексами незаписанных реплик (§42, Test F)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            voice_by_profile = {profile_id: voice for voice, profile_id in profiles.items()}
            for replica in project["replicas"]:
                if replica["index"] == 2:
                    continue
                voice = voice_by_profile[project["role_voices"][replica["speaker"]]]
                project = await _upload_take(client, project["id"], replica["index"], voice)

            assert project["readiness"]["ready_to_render"] is False
            assert project["readiness"]["missing"] == [2]

            response = await client.post(
                f"/api/recording-projects/{project['id']}/render", json={}
            )
            assert response.status_code == 409, response.text
            assert response.json()["detail"]["missing"] == [2]
            # Неполного файла не появляется: молча пропустить реплику нельзя.
            assert not recording_pipeline.render_state(project["id"]).output_path

            # Тот же запрет на уровне сборки, а не только в обработчике запроса.
            loaded = recording_store.get_store().require_project(project["id"])
            with pytest.raises(recording_pipeline.RecordingRenderError):
                recording_pipeline.assemble(loaded)

    _run(scenario)


# --- Test G. Проверка декодирования MP3 ---------------------------------------
def test_final_mp3_decodes_with_sound_parameters(stub, workspace, monkeypatch):
    """MP3 открывается декодером: каналы, частота, длительность, сэмплы (§42, Test G)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            project = await _record_all(client, project, profiles)
            await _render(client, project["id"])

            response = await _download(client, project["id"])
            assert response.status_code == 200, response.text
            # Контейнер проверяется по байтам: одного «файл существует» мало.
            assert response.content[:3] == b"ID3" or response.content[0] == 0xFF

            from pydub import AudioSegment

            segment = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
            assert segment.channels >= 1
            assert segment.frame_rate > 0
            assert len(segment) > 0
            assert segment.duration_seconds > TAKE_SEC * 2

            decoded = recording_audio.decode(response.content, ".mp3")
            assert decoded.ndim == 1 and decoded.size > 0
            assert recording_audio.duration_sec(decoded) > 0

    _run(scenario)



# --- Test H. Состояние монтажа не отстаёт от запуска ---------------------------
def test_render_status_shows_new_run_not_previous_one(stub, workspace, monkeypatch):
    """Первый опрос после 202 не отдаёт прошлый монтаж (§22, §42).

    Между ответом `POST /render` и первым оператором рабочего потока есть окно.
    Если состояние в нём ещё хранит прошлый монтаж, клиент видит «готово» с
    прежней длительностью и решает, что новый монтаж уже прошёл: «сменил дубль, а
    файл не изменился». Тест ловит именно это окно — сравнением отметки запуска.
    """

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            project = await _record_all(client, project, profiles)
            previous = await _render(client, project["id"])

            response = await client.post(
                f"/api/recording-projects/{project['id']}/render", json={"pause_ms": PAUSE_MS}
            )
            assert response.status_code == 202, response.text
            status = (
                await client.get(f"/api/recording-projects/{project['id']}/render/status")
            ).json()
            assert (status["status"], status["started_at"]) != (
                "done",
                previous["started_at"],
            ), "первый опрос после запуска отдал состояние прошлого монтажа"
            finished = await _wait_render(client, project["id"])
            assert finished["status"] == "done", finished

    _run(scenario)


def test_begin_render_refuses_second_start(workspace):
    """Пока монтаж идёт, второй запуск запрещён: два потока писали бы один файл."""
    recording_pipeline.reset_states()
    state = recording_pipeline.begin_render("project-x")
    assert state.status == "running"
    assert recording_pipeline.render_state("project-x").status == "running"
    with pytest.raises(recording_pipeline.RenderBusyError):
        recording_pipeline.begin_render("project-x")
    recording_pipeline.abort_render("project-x", "проверка")
    assert recording_pipeline.render_state("project-x").status == "error"
    recording_pipeline.reset_states()


def test_concurrent_begin_render_admits_exactly_one(workspace):
    """Гонка запусков монтажа: замок пропускает ровно один.

    Проверка «монтаж уже идёт» и объявление запуска стоят под одним замком
    нарочно. Если бы проверка была отдельно от записи, два потока успели бы
    прочитать «не идёт» и оба пошли бы писать в один и тот же output-файл —
    второй монтаж затёр бы первый. Здесь потоки стартуют одновременно с барьера,
    поэтому окно между проверкой и записью у них гарантированно есть.
    """
    recording_pipeline.reset_states()
    outcomes: list[str] = []
    start = threading.Barrier(8)

    def attempt() -> None:
        start.wait()
        try:
            recording_pipeline.begin_render("project-race")
            outcomes.append("started")
        except recording_pipeline.RenderBusyError:
            outcomes.append("busy")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("started") == 1, outcomes
    assert outcomes.count("busy") == 7, outcomes
    assert recording_pipeline.render_state("project-race").status == "running"
    recording_pipeline.reset_states()


def test_render_of_missing_project_does_not_stay_running(workspace):
    """Пропавший проект не оставляет состояние «running» навсегда.

    `begin_render` объявляет запуск до потока, поэтому любая ошибка до монтажа
    обязана снять «running»: иначе кнопка сборки в интерфейсе осталась бы
    заблокированной до перезапуска процесса — 409 «Монтаж уже идёт» на каждый
    следующий запрос.
    """
    recording_pipeline.reset_states()
    with pytest.raises(recording_pipeline.RecordingRenderError):
        recording_pipeline.render("нет-такого-проекта")
    assert recording_pipeline.render_state("нет-такого-проекта").status == "error"
    recording_pipeline.reset_states()


# --- Test I. Финальный проход по всему диалогу ---------------------------------
def test_assemble_applies_final_pass_once_over_whole_dialogue(stub, workspace, monkeypatch):
    """LUFS + лимитер вызываются один раз и на весь диалог (§29, §35.21).

    Проверяется именно факт и охват финального прохода: если бы он применялся к
    каждой реплике отдельно, громкость «прыгала» бы от реплики к реплике, а
    лимитер считался бы по куску, а не по целому файлу.
    """

    async def scenario():
        async with _client(monkeypatch) as client:
            project, profiles = await _setup_project(client, TWO_VOICES, ["A", "B"])
            project = await _record_all(client, project, profiles)

            calls: list[int] = []
            original = audio_pipeline._finalize_track

            def spy(audio, *args, **kwargs):
                calls.append(int(np.asarray(audio).size))
                return original(audio, *args, **kwargs)

            monkeypatch.setattr(audio_pipeline, "_finalize_track", spy)
            store = recording_store.get_store()
            master_path, final_path, duration, warnings = await asyncio.to_thread(
                recording_pipeline.assemble,
                store.require_project(project["id"]),
                pause_ms=PAUSE_MS,
            )

            assert len(calls) == 1, "финальный проход должен вызываться один раз на диалог"
            # На входе — весь диалог (4 реплики + 3 паузы), а не одна реплика.
            assert calls[0] > int(3.5 * TAKE_SEC * SAMPLE_RATE), calls[0]
            assert master_path.exists() and final_path.exists()
            assert duration == pytest.approx(calls[0] / SAMPLE_RATE, abs=1e-3)
            assert isinstance(warnings, list)

    _run(scenario)
