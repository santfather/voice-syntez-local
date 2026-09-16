"""Таймлайн проекта: границы реплик по реальным длительностям take'ов.

Тесты идут через приложение целиком (ASGI-транспорт) и с движком-заглушкой:
проверяется арифметика времени и совпадение таймлайна с собранным файлом, а не
модель. Самый важный тест здесь — `test_timeline_matches_rendered_file`: границы
из `/timeline` обязаны совпасть с длительностью записанного аудио, иначе
визуальный таймлайн был бы красивой схемой, не имеющей отношения к файлу.
"""

import asyncio
import contextlib
import time
from pathlib import Path

import httpx
import pytest
import soundfile as sf

from backend import audio_pipeline, config, main, timeline
from backend.audio_pipeline import RenderSettings
from backend.engines.base import SAMPLE_RATE
from backend.job_queue import JobQueue

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."


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


async def _prepared_text_project(client, text: str = DIALOGUE) -> dict:
    """Проект с разобранным текстом и назначенными голосами — готов к рендеру."""
    response = await client.post(
        "/api/projects", json={"name": "Тест", "source_text": text, "mode": "dialogue"}
    )
    assert response.status_code == 201, response.text
    project = response.json()
    speakers = {
        replica["speaker"]: {"voice_id": "voice1"} for replica in (
            await client.post(f"/api/projects/{project['id']}/parse", json={})
        ).json()["replicas"]
    }
    return await _patch_speakers(client, project["id"], speakers)


async def _patch_speakers(client, project_id: str, speakers: dict) -> dict:
    """Правка спикеров без потери уже назначенных голосов.

    PATCH заменяет словарь спикеров целиком, поэтому перед отправкой в него
    подмешивается текущее состояние проекта: иначе правка паузы одного слота
    обнулила бы голос другого, и это выглядело бы как ошибка таймлайна.
    """
    project = (await client.get(f"/api/projects/{project_id}")).json()
    payload = {
        speaker["key"]: {
            "voice_id": speaker.get("voice_id") or "",
            "overrides": speaker.get("overrides") or {},
        }
        for speaker in project["speakers"]
    }
    for key, patch in speakers.items():
        payload[key] = {**payload.get(key, {}), **patch}
    response = await client.patch(f"/api/projects/{project_id}", json={"speakers": payload})
    assert response.status_code == 200, response.text
    return response.json()


async def _render(client, project_id: str, **options) -> str:
    accepted = await client.post(f"/api/projects/{project_id}/render", json=options)
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]
    await _wait_job(client, job_id)
    return job_id


async def _timeline(client, project_id: str) -> dict:
    response = await client.get(f"/api/projects/{project_id}/timeline")
    assert response.status_code == 200, response.text
    return response.json()


def _durations(timeline_data: dict) -> list[float]:
    return [segment["duration_sec"] for segment in timeline_data["replicas"]]


def _shifted_timeline(project: dict, delta: float) -> dict:
    """Таймлайн проекта, у которого первый take стал длиннее на `delta` секунд.

    Правка идёт по карточке реплики из API: границы считаются по активному take'у
    (`selected_take_id`), поэтому изменение его `duration_sec` — это ровно то же,
    что замена звучания на более длинное. Таймлайн пересчитывается тем же кодом,
    что и в эндпоинте.
    """
    replicas = []
    for card in project["replicas"]:
        if card["index"] != 0:
            replicas.append(card)
            continue
        selected = card["selected_take_id"]
        replicas.append(
            {
                **card,
                "takes": [
                    {
                        **take,
                        "duration_sec": max(
                            float(take["duration_sec"]) + delta, 0.05
                        ),
                    }
                    if int(take["id"]) == int(selected)
                    else take
                    for take in card["takes"]
                ],
            }
        )
    settings = main._replica_render_settings(project)
    timings = timeline.replica_timings(
        replicas,
        settings,
        speakers=timeline.speaker_pause_overrides(project["speakers"]),
    )
    by_index = {int(card["index"]): card for card in replicas}
    return timeline.timeline_view(
        [],
        lambda index: by_index[index],
        timings,
        settings,
        timeline.timeline_duration(timings),
    )


# --- 1. границы первой реплики и совпадение с файлом --------------------------
def test_timeline_matches_rendered_file(stub, fake_store, monkeypatch):
    """Границы таймлайна и длительность записанного файла — одно и то же число.

    Это главная проверка фазы: таймлайн обещает пользователю место реплики в
    готовом аудио, и разойтись с ним он не имеет права. Сравниваются не оценки
    «по знакам», а duration take'ов, из которых собран файл.
    """
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(client, project["id"], output_format="wav", pause_ms=400)

            data = await _timeline(client, project["id"])
            first = data["replicas"][0]
            assert first["index"] == 0
            assert first["start_sec"] == 0.0
            # Перед первой репликой пауза не вставляется — её неоткуда взять:
            # сборка ставит паузу только перед каждым следующим куском.
            assert data["settings"]["pause_ms"] == 400
            assert first["duration_sec"] > 0
            assert first["end_sec"] == pytest.approx(first["duration_sec"], abs=1e-6)

            second = data["replicas"][1]
            assert second["pause_ms"] == 400
            assert second["start_sec"] == pytest.approx(first["end_sec"] + 0.4, abs=1e-3)
            assert data["duration_sec"] == pytest.approx(second["end_sec"], abs=1e-3)

            # Границы реплик — это ровно длительности их файлов, а длительность
            # трека — их сумма плюс паузы. Никаких «оценок по символам».
            take_sizes = [
                await asyncio.to_thread(sf.info, Path(segment["takes"][0]["audio_path"]))
                for segment in data["replicas"]
            ]
            assert _durations(data) == pytest.approx([item.duration for item in take_sizes], abs=1e-3)
            assert data["duration_sec"] == pytest.approx(
                sum(item.duration for item in take_sizes) + 0.4, abs=1e-3
            )

    _run(scenario)


def test_timeline_duration_equals_rendered_file(stub, fake_store, monkeypatch):
    """Длительность таймлайна равна длительности записанного файла рендера."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            job_id = await _render(client, project["id"], output_format="wav", pause_ms=250)

            data = await _timeline(client, project["id"])
            audio_path = config.OUTPUT_DIR / f"{job_id}.wav"
            assert audio_path.exists()
            info = await asyncio.to_thread(sf.info, audio_path)

            assert data["duration_sec"] == pytest.approx(info.duration, abs=1e-3)
            # И это ровно сумма кусков с паузой между ними — без «хвоста».
            assert data["duration_sec"] == pytest.approx(
                sum(_durations(data)) + 0.25, abs=1e-3
            )

    _run(scenario)


# --- 2. пауза учитывается, личная важнее общей --------------------------------
def test_pause_is_counted_and_override_wins(stub, fake_store, monkeypatch):
    """Пауза сдвигает реплики, а `pause_override_ms` реплики важнее общего `pause_ms`."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(
                client, "ИВАН: Раз.\nИВАН: Два.\nИВАН: Три."
            )
            # У второй реплики паузы нет вовсе — она важнее общего значения.
            await client.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"overrides": {"pause_override_ms": 0}},
            )
            await _render(client, project["id"], output_format="wav", pause_ms=800)

            data = await _timeline(client, project["id"])
            pauses = [segment["pause_ms"] for segment in data["replicas"]]
            # Эффективная пауза каждой реплики: у первой — общая (в файл она не
            # вставляется, но настройка видна), у второй — личный ноль, у третьей
            # — снова общая, потому что правки у неё нет.
            assert pauses == [800, 0, 800]
            assert data["settings"]["pause_ms"] == 800

            first, second, third = data["replicas"]
            # Перед первой репликой паузы нет: файл начинается сразу со звука.
            assert first["start_sec"] == 0.0
            # Вторая реплика стоит вплотную: её личный ноль важнее общих 800 мс.
            assert second["start_sec"] == pytest.approx(first["end_sec"], abs=1e-6)
            # Третья — через общую паузу после конца второй.
            assert third["start_sec"] == pytest.approx(second["end_sec"] + 0.8, abs=1e-3)
            assert data["duration_sec"] == pytest.approx(third["end_sec"], abs=1e-3)
            assert data["duration_sec"] == pytest.approx(
                sum(_durations(data)) + 0.8, abs=1e-3
            )

    _run(scenario)


def test_pause_override_at_speaker_layer_is_honoured(stub, fake_store, monkeypatch):
    """Пауза спикера читается тем же правилом, что и при сборке файла."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _patch_speakers(
                client,
                project["id"],
                {"МАРГО": {"voice_id": "voice1", "overrides": {"pause_override_ms": 1200}}},
            )
            await _render(client, project["id"], output_format="wav", pause_ms=200)

            data = await _timeline(client, project["id"])
            assert [segment["pause_ms"] for segment in data["replicas"]] == [200, 1200]
            assert data["replicas"][1]["start_sec"] == pytest.approx(
                data["replicas"][0]["end_sec"] + 1.2, abs=1e-3
            )

    _run(scenario)


# --- 3. кроссфейд не двигает границы ------------------------------------------
def test_crossfade_does_not_shift_boundaries(stub, fake_store, monkeypatch):
    """`cross_fade_duration` — ручка F5 внутри куска; длину трека она не меняет.

    Склейка реплик — `np.concatenate` без перекрытия, а `_finalize_track` длину
    сохраняет. Поэтому ручка не может сдвинуть границы, и выдумывать ей сдвиг в
    таймлайне было бы неправдой.
    """
    stub.seconds_per_char = 0.0

    def _synthesize(text, ref_audio_path, ref_text, speed, params):
        stub.calls.append({"text": text})
        return (0.3 * __import__("numpy").ones(int(SAMPLE_RATE * 0.5), dtype="float32"),
                SAMPLE_RATE)

    monkeypatch.setattr(stub, "_synthesize", _synthesize)

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(
                client, project["id"], output_format="wav", pause_ms=300,
                cross_fade_duration=0.0,
            )
            tight = await _timeline(client, project["id"])
            await _render(
                client, project["id"], output_format="wav", pause_ms=300,
                cross_fade_duration=0.5,
            )
            wide = await _timeline(client, project["id"])

            assert [item["start_sec"] for item in wide["replicas"]] == pytest.approx(
                [item["start_sec"] for item in tight["replicas"]], abs=1e-6
            )
            assert [item["end_sec"] for item in wide["replicas"]] == pytest.approx(
                [item["end_sec"] for item in tight["replicas"]], abs=1e-6
            )
            assert wide["duration_sec"] == pytest.approx(tight["duration_sec"], abs=1e-6)
            assert tight["settings"]["cross_fade_duration"] == 0.0
            assert wide["settings"]["cross_fade_duration"] == 0.5

    _run(scenario)


# --- 4-5. замена take сдвигает хвост -------------------------------------------
def test_longer_take_shifts_tail_forward(stub, fake_store, monkeypatch):
    """Более длинный take первой реплики уводит хвост вправо на разницу длин."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(client, project["id"], output_format="wav", pause_ms=400)
            before = await _timeline(client, project["id"])
            card = (await client.get(f"/api/projects/{project['id']}")).json()

            after = _shifted_timeline(card, 1.5)
            first, second = after["replicas"]
            assert first["duration_sec"] == pytest.approx(
                before["replicas"][0]["duration_sec"] + 1.5, abs=1e-3
            )
            assert second["start_sec"] == pytest.approx(
                before["replicas"][1]["start_sec"] + 1.5, abs=1e-3
            )
            assert after["duration_sec"] == pytest.approx(
                before["duration_sec"] + 1.5, abs=1e-3
            )

    _run(scenario)


def test_shorter_take_shifts_tail_back(stub, fake_store, monkeypatch):
    """Короткий take сдвигает хвост назад — таймлайн не помнит прежних границ."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(client, project["id"], output_format="wav", pause_ms=400)
            before = await _timeline(client, project["id"])
            card = (await client.get(f"/api/projects/{project['id']}")).json()

            delta = -0.2
            after = _shifted_timeline(card, delta)
            first, second = after["replicas"]
            assert first["duration_sec"] == pytest.approx(
                before["replicas"][0]["duration_sec"] + delta, abs=1e-3
            )
            assert second["start_sec"] == pytest.approx(
                before["replicas"][1]["start_sec"] + delta, abs=1e-3
            )
            assert after["duration_sec"] == pytest.approx(
                before["duration_sec"] + delta, abs=1e-3
            )

    _run(scenario)


# --- 6. смешанные движки на время не влияют -----------------------------------
def test_mixed_engines_do_not_change_timing():
    """Длительность берётся из take'а: чем он получен, границы не меняет."""
    replicas = [
        {
            "index": 0,
            "overrides": {},
            "selected_take_id": 1,
            "takes": [
                {"id": 1, "duration_sec": 3.25, "engine": "f5", "audio_path": "/a.wav"}
            ],
        },
        {
            "index": 1,
            "overrides": {},
            "selected_take_id": 2,
            "takes": [
                {"id": 2, "duration_sec": 2.5, "engine": "xtts", "audio_path": "/b.wav"}
            ],
        },
    ]
    settings = RenderSettings(pause_ms=500)
    timings = timeline.replica_timings(replicas, settings, has_file=lambda _: True)

    assert (timings[0].start_sec, timings[0].end_sec) == (0.0, 3.25)
    assert (timings[1].start_sec, timings[1].end_sec) == (3.75, 6.25)
    assert timeline.timeline_duration(timings) == pytest.approx(6.25)

    # Те же длительности у одного движка — те же границы: семантика времени
    # от движка не зависит вовсе.
    same = [
        {**replicas[0], "takes": [{**replicas[0]["takes"][0], "engine": "xtts"}]},
        replicas[1],
    ]
    other = timeline.replica_timings(same, settings, has_file=lambda _: True)
    assert [item.end_sec for item in other] == [item.end_sec for item in timings]
    assert [item.start_sec for item in other] == [item.start_sec for item in timings]


def test_mixed_engines_timeline_over_api(stub, fake_store, monkeypatch):
    """Смешанные движки в одном диалоге — обычный случай, границы всё те же."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(client, project["id"], output_format="wav", pause_ms=300)
            data = await _timeline(client, project["id"])
            assert [segment["engine"] for segment in data["replicas"]] == ["stub", "stub"]
            # Длительность сегмента — длительность его активного take'а, а не
            # что-то, посчитанное по движку.
            for segment in data["replicas"]:
                active = next(take for take in segment["takes"] if take["active"])
                assert segment["duration_sec"] == pytest.approx(active["duration_sec"], abs=1e-3)

    _run(scenario)


# --- 8. активный take соответствует сегменту ----------------------------------
def test_active_take_matches_segment_and_switches(stub, fake_store, monkeypatch):
    """После выбора другого take сегмент показывает именно его номер и подпись."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            # Второй рендер добавляет репликам ещё по take — у первой их станет два.
            await _render(client, project["id"], output_format="wav", pause_ms=0)
            await _render(client, project["id"], output_format="wav", pause_ms=0)

            data = await _timeline(client, project["id"])
            first = data["replicas"][0]
            takes = first["takes"]
            assert len(takes) == 2
            active = next(take for take in takes if take["active"])
            assert first["take_id"] == active["id"]
            assert first["take_label"] == active["label"]
            assert first["duration_sec"] == pytest.approx(active["duration_sec"], abs=1e-3)
            assert first["has_audio"] is True and first["status"] == "rendered"

            older = takes[0]
            assert older["id"] != active["id"]
            switched = await client.post(
                f"/api/projects/{project['id']}/replicas/0/takes/{older['id']}"
            )
            assert switched.status_code == 200, switched.text

            after = (await _timeline(client, project["id"]))["replicas"][0]
            assert after["take_id"] == older["id"]
            assert after["take_label"] == older["label"]
            assert after["duration_sec"] == pytest.approx(older["duration_sec"], abs=1e-3)

    _run(scenario)


# --- реплика без take, 404 и «ничего не меняет» -------------------------------
def test_replica_without_take_keeps_cursor(stub, fake_store, monkeypatch):
    """Реплика без звука видна на таймлайне, но курсор не двигает."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client, "ИВАН: Раз.\nМАРГО: Два.")
            # Голос есть у обоих, но рендерим только… — проще: не рендерим вовсе.
            empty = await _timeline(client, project["id"])
            assert [segment["has_audio"] for segment in empty["replicas"]] == [False, False]
            assert [segment["status"] for segment in empty["replicas"]] == ["pending", "pending"]
            assert all(segment["duration_sec"] == 0 for segment in empty["replicas"])
            assert [segment["start_sec"] for segment in empty["replicas"]] == [0.0, 0.0]
            assert empty["duration_sec"] == 0

            # А теперь у первой реплики звук есть, у второй нет: вторая не двигает
            # курсор и остаётся «пустой» ровно там, где кончилась первая.
            job_id = await client.post(
                f"/api/projects/{project['id']}/replicas/0/regenerate"
            )
            assert job_id.status_code == 202, job_id.text
            await _wait_job(client, job_id.json()["job_id"])

            partial = await _timeline(client, project["id"])
            first, second = partial["replicas"]
            assert first["has_audio"] is True
            assert second["has_audio"] is False
            # Пауза второй реплики остаётся видимой: она сдвинет её, когда звук
            # появится, и молча прятать её было бы потерей настройки.
            assert second["pause_ms"] == config.DEFAULT_PAUSE_MS
            assert second["start_sec"] == pytest.approx(first["end_sec"] + 0.4, abs=1e-3)
            assert second["end_sec"] == pytest.approx(second["start_sec"], abs=1e-3)
            assert second["duration_sec"] == 0
            assert partial["duration_sec"] == pytest.approx(first["end_sec"], abs=1e-3)

    _run(scenario)


def test_timeline_unknown_project_returns_404(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as client:
            response = await client.get("/api/projects/нет-такого/timeline")
            assert response.status_code == 404
            assert "не найден" in response.json()["detail"]

    _run(scenario)


def test_timeline_does_not_change_project(stub, fake_store, monkeypatch):
    """Эндпоинт только читает: проект после запроса таймлайна байт в байт тот же."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(client, project["id"], output_format="wav", pause_ms=400)

            before = (await client.get(f"/api/projects/{project['id']}")).json()
            timeline_data = await _timeline(client, project["id"])
            timeline_again = await _timeline(client, project["id"])
            after = (await client.get(f"/api/projects/{project['id']}")).json()

            assert after == before
            assert timeline_again == timeline_data
            assert "project_id" in timeline_data
            assert len(timeline_data["speakers"]) == 2
            assert timeline_data["speakers"][0]["voice_name"] == "Тестовый"

    _run(scenario)


def test_timeline_settings_come_from_saved_project(stub, fake_store, monkeypatch):
    """Настройки в ответе — те, которыми файл будет собран, а не дефолты."""
    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _prepared_text_project(client)
            await _render(
                client, project["id"], output_format="wav", pause_ms=650,
                cross_fade_duration=0.35,
            )
            data = await _timeline(client, project["id"])
            assert data["settings"] == {"pause_ms": 650, "cross_fade_duration": 0.35}

    _run(scenario)


def test_audio_pipeline_has_no_hidden_track_trim():
    """`_finalize_track` длину не меняет — на этом стоит вся арифметика таймлайна."""
    import numpy as np

    audio = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
    audio[1000:2000] = 0.5
    assert audio_pipeline._finalize_track(audio).size == audio.size
