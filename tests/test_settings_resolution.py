"""Иерархия параметров синтеза: движок → пресет голоса → слот → реплика.

Проверяется единая точка резолва (`settings_resolution`) и то, что ею пользуются
оба пути — прослушивание голоса и рендер диалога. Иначе «настроил голос один раз»
работало бы в одном месте и не работало в другом, а карточка показывала бы не то
значение, которое уходит в модель.
"""

import asyncio
import contextlib
import json
import time
from dataclasses import asdict

import httpx
import pytest
from conftest import analyze_project

from backend import config, main
from backend.audio_pipeline import RenderSettings, SpeakerSettings, render_dialogue
from backend.dialogue_parser import Replica
from backend.engines.base import ENGINE_F5, ENGINE_XTTS
from backend.job_queue import JobQueue
from backend.settings_resolution import (
    COMMON_FIELDS,
    SOURCE_ENGINE,
    SOURCE_REPLICA,
    SOURCE_SPEAKER,
    SOURCE_VOICE,
    engine_defaults,
    resolve_synthesis_settings,
)

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика.\nИВАН: Третья реплика."


def _resolve(voice_settings=None, speaker_overrides=None, replica_overrides=None, engine=ENGINE_F5):
    return resolve_synthesis_settings(
        engine_defaults(engine), voice_settings, speaker_overrides, replica_overrides
    )


def _slot(voice) -> SpeakerSettings:
    """Слот проекта: голос без единой правки — так выглядит новый диалог."""
    return SpeakerSettings.from_dict({"voice_id": voice.id})


def _replica(text: str = "Фраза") -> Replica:
    return Replica(voice="#1", text=text, line_number=1)


# --- 1: нижний слой ------------------------------------------------------------
def test_engine_defaults_without_overrides():
    """Без выбора пользователя значения берутся из паспорта движка."""
    resolved = _resolve()
    assert resolved["speed"] == {"value": config.DEFAULT_SPEED, "source": SOURCE_ENGINE}
    assert resolved["nfe_step"]["value"] == config.DEFAULT_NFE_STEP
    assert resolved["cfg_strength"]["value"] == config.DEFAULT_CFG_STRENGTH
    # Пауза не задана никем: её берёт общая настройка диалога, а не ноль.
    assert resolved["pause_override_ms"]["value"] is None
    # Перекрывать нечего — про сброс в интерфейсе нечего и подписывать.
    assert "inherited" not in resolved["speed"]


# --- 2: пресет голоса ----------------------------------------------------------
def test_voice_preset_overrides_engine_default():
    """Подобранное в «Прослушать» значение становится значением голоса."""
    resolved = _resolve({"speed": 1.2, "nfe_step": 16})
    assert resolved["speed"] == {
        "value": 1.2,
        "source": SOURCE_VOICE,
        "inherited": config.DEFAULT_SPEED,
    }
    assert resolved["nfe_step"]["value"] == 16
    # Остальные ручки пресет не трогал — они остаются дефолтами движка.
    assert resolved["cfg_strength"]["source"] == SOURCE_ENGINE


# --- 3: слот проекта ----------------------------------------------------------
def test_speaker_overrides_voice_preset():
    resolved = _resolve({"speed": 1.2}, {"speed": 0.9})
    assert resolved["speed"] == {"value": 0.9, "source": SOURCE_SPEAKER, "inherited": 1.2}


# --- 4: правка реплики --------------------------------------------------------
def test_replica_overrides_speaker():
    resolved = _resolve({"speed": 1.2}, {"speed": 0.9}, {"speed": 1.4})
    assert resolved["speed"] == {"value": 1.4, "source": SOURCE_REPLICA, "inherited": 0.9}


# --- 5: сброс правки реплики --------------------------------------------------
@pytest.mark.parametrize("empty", [{}, None, {"speed": None}])
def test_reset_replica_returns_speaker_value(empty):
    """Сброс правки — это снятие ключа, и он возвращает значение слота."""
    layered = _resolve({"speed": 1.2, "cfg_strength": 2.7}, {"speed": 0.9}, {"speed": 1.4})
    reset = _resolve({"speed": 1.2, "cfg_strength": 2.7}, {"speed": 0.9}, empty)
    assert layered["speed"]["value"] == 1.4
    # Подпись «сбросить → …» в карточке берётся отсюда же.
    assert layered["speed"]["inherited"] == 0.9
    assert reset["speed"] == {"value": 0.9, "source": SOURCE_SPEAKER, "inherited": 1.2}


# --- 6: сброс правки слота ----------------------------------------------------
def test_reset_speaker_returns_voice_preset():
    layered = _resolve({"speed": 1.2}, {"speed": 0.9})
    reset = _resolve({"speed": 1.2}, {})
    assert layered["speed"]["inherited"] == 1.2
    assert reset["speed"] == {
        "value": 1.2,
        "source": SOURCE_VOICE,
        "inherited": config.DEFAULT_SPEED,
    }


# --- 7: ручки чужого движка ---------------------------------------------------
def test_parameters_of_other_engine_are_not_applied():
    """Температура XTTS не уходит в F5, а NFE не подменяет ручку XTTS."""
    on_f5 = _resolve({"engine_params": {"temperature": 0.5}})
    assert "temperature" not in on_f5  # у F5 такой ручки нет

    on_xtts = _resolve(
        {"engine_params": {"temperature": 0.5, "nfe_step": 16}}, engine=ENGINE_XTTS
    )
    assert on_xtts["temperature"] == {
        "value": 0.5,
        "source": SOURCE_VOICE,
        "inherited": config.DEFAULT_XTTS_TEMPERATURE,
        "engine_param": True,
    }
    assert on_xtts["nfe_step"]["value"] == config.DEFAULT_NFE_STEP
    assert on_xtts["nfe_step"]["source"] == SOURCE_ENGINE


# --- 8: зажим после выбора слоя -----------------------------------------------
def test_clamp_is_applied_after_resolution():
    # Пресет вне границ зажат, но слот всё равно важнее — зажим не превращает
    # значение в выбор этого слоя.
    resolved = _resolve({"speed": 5.0}, {"speed": 1.5})
    assert resolved["speed"]["value"] == 1.5
    assert resolved["speed"]["inherited"] == config.SPEED_RANGE[1]

    # NFE ходит только по разрешённым ступеням: 10 ближе к 8, чем к 16.
    assert _resolve({"nfe_step": 10})["nfe_step"]["value"] == 8
    # Ручка движка зажимается его собственными границами из паспорта.
    xtts = _resolve({"engine_params": {"temperature": 99.0}}, engine=ENGINE_XTTS)
    assert xtts["temperature"]["value"] == config.XTTS_TEMPERATURE_RANGE[1]
    # Нечисло не роняет резолв: подставляется дефолт движка.
    assert _resolve({"speed": "быстро"})["speed"]["value"] == config.SPEED_RANGE[0]


# --- 9: незнакомые ключи ------------------------------------------------------
def test_unknown_parameters_are_ignored():
    resolved = _resolve({"engine_params": {"nope": 1}, "gap": 42})
    assert "nope" not in resolved and "gap" not in resolved
    assert set(resolved) == set(COMMON_FIELDS)


# --- 10: прослушивание и диалог считают одинаково ------------------------------
@pytest.fixture
def preset_voice(fake_store):
    """Голос с пресетом — то, что пользователь подобрал в «Прослушать»."""
    fake_store.preset = {"speed": 1.3, "cfg_strength": 2.6, "nfe_step": 16}
    return fake_store


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


def test_preview_and_render_use_the_same_voice_preset(stub, preset_voice, monkeypatch):
    """Прослушивание и диалог берут настройки голоса из одной точки.

    Запрос прослушивания посылает только голос и фразу: незаданные поля должны
    прийти из пресета, а не из дефолтов движка — иначе подобранное в «Прослушать»
    звучало бы в диалоге по-другому.
    """

    async def scenario():
        async with _client(monkeypatch) as client:
            accepted = await client.post(
                "/api/preview", json={"voice_id": preset_voice.id, "text": "Фраза"}
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])

    asyncio.run(scenario())
    heard = stub.calls[-1]

    asyncio.run(
        render_dialogue(
            job_id="job",
            replicas=[_replica()],
            speakers={"#1": _slot(preset_voice)},
            settings=RenderSettings(output_format="wav"),
        )
    )
    rendered = stub.calls[-1]

    assert heard["speed"] == rendered["speed"] == 1.3
    assert heard["params"]["cfg_strength"] == rendered["params"]["cfg_strength"] == 2.6
    assert heard["params"]["nfe_step"] == rendered["params"]["nfe_step"] == 16


# --- пресет голоса в новом диалоге --------------------------------------------
def _write_voices(*voices) -> None:
    """Кладёт голоса в voices.json — так их видит боевое хранилище."""
    config.VOICES_JSON.write_text(
        json.dumps({"voices": [asdict(voice) for voice in voices]}, ensure_ascii=False),
        encoding="utf-8",
    )


async def _project(client, text: str, speakers: dict[str, str]) -> dict:
    created = await client.post(
        "/api/projects", json={"name": "Пресет", "source_text": text, "mode": "dialogue"}
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    patched = await client.patch(
        f"/api/projects/{project_id}",
        json={"speakers": {key: {"voice_id": value} for key, value in speakers.items()}},
    )
    assert patched.status_code == 200, patched.text
    parsed = await client.post(f"/api/projects/{project_id}/parse", json={})
    assert parsed.status_code == 200, parsed.text
    return parsed.json()


def test_voice_preset_reaches_new_project(workspace, voice, stub, monkeypatch):
    """Настроенный один раз голос достаётся новому диалогу без правок в карточках.

    Заодно проверяется то, на чём стоит интерфейс: у каждого значения есть
    источник («наследуется от голоса» / «правка этой реплики») и значение, к
    которому вернёт сброс.
    """
    _write_voices(voice)

    async def scenario():
        async with _client(monkeypatch) as client:
            saved = await client.patch(
                f"/api/voices/{voice.id}",
                json={"preset": {"speed": 1.3, "cfg_strength": 2.6, "nfe_step": 16}},
            )
            assert saved.status_code == 200, saved.text
            assert saved.json()["preset"] == {"speed": 1.3, "cfg_strength": 2.6, "nfe_step": 16}

            project = await _project(client, DIALOGUE, {"ИВАН": voice.id, "МАРГО": voice.id})
            settings = project["replicas"][0]["settings"]
            assert settings["speed"] == {
                "value": 1.3,
                "source": SOURCE_VOICE,
                "inherited": config.DEFAULT_SPEED,
            }
            assert settings["cfg_strength"]["value"] == 2.6
            assert settings["nfe_step"]["value"] == 16

            patch = f"/api/projects/{project['id']}/replicas/0"
            changed = await client.patch(patch, json={"overrides": {"speed": 1.6}})
            local = changed.json()["replica"]["settings"]
            assert local["speed"] == {"value": 1.6, "source": SOURCE_REPLICA, "inherited": 1.3}
            # Правка одной ручки не задевает остальные: они по-прежнему от голоса.
            assert local["cfg_strength"] == {
                "value": 2.6,
                "source": SOURCE_VOICE,
                "inherited": config.DEFAULT_CFG_STRENGTH,
            }

            reset = await client.patch(patch, json={"overrides": {"speed": None}})
            back = reset.json()["replica"]["settings"]
            assert back["speed"]["value"] == 1.3
            assert back["speed"]["source"] == SOURCE_VOICE

            stub.calls.clear()
            await analyze_project(client, project["id"])
            accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
            await _wait_job(client, accepted.json()["job_id"])

    asyncio.run(scenario())
    assert [call["speed"] for call in stub.calls] == [1.3, 1.3, 1.3]
    assert stub.calls[0]["params"]["cfg_strength"] == 2.6
    assert stub.calls[0]["params"]["nfe_step"] == 16


def test_voice_preset_is_not_touched_by_replica_edits(workspace, voice, stub, monkeypatch):
    """Правка реплики меняет реплику, а не сам голос."""
    _write_voices(voice)

    async def scenario():
        async with _client(monkeypatch) as client:
            await client.patch(
                f"/api/voices/{voice.id}", json={"preset": {"speed": 1.3}}
            )
            project = await _project(client, DIALOGUE, {"ИВАН": voice.id, "МАРГО": voice.id})
            await client.patch(
                f"/api/projects/{project['id']}/replicas/0", json={"overrides": {"speed": 0.7}}
            )
            # Новый диалог тем же голосом: правка прошлой реплики к нему не относится.
            fresh = await _project(client, DIALOGUE, {"ИВАН": voice.id, "МАРГО": voice.id})
            assert [replica["settings"]["speed"]["value"] for replica in fresh["replicas"]] == [
                1.3,
                1.3,
                1.3,
            ]
            # Голос остался с пресетом: сброс правки реплики возвращает именно его.
            voices = (await client.get("/api/voices")).json()["voices"]
            assert next(item for item in voices if item["id"] == voice.id)["preset"] == {"speed": 1.3}

    asyncio.run(scenario())


def test_reset_preset_returns_to_engine_defaults(workspace, voice, stub, monkeypatch):
    """Пустой пресет — «сбросить подобранное», а не «оставить как было»."""
    _write_voices(voice)

    async def scenario():
        async with _client(monkeypatch) as client:
            await client.patch(f"/api/voices/{voice.id}", json={"preset": {"speed": 1.3}})
            cleared = await client.patch(f"/api/voices/{voice.id}", json={"preset": {}})
            assert cleared.status_code == 200, cleared.text
            assert cleared.json()["preset"] == {}

            project = await _project(client, DIALOGUE, {"ИВАН": voice.id, "МАРГО": voice.id})
            settings = project["replicas"][0]["settings"]
            assert settings["speed"] == {"value": config.DEFAULT_SPEED, "source": SOURCE_ENGINE}

    asyncio.run(scenario())


def test_speaker_panel_shows_its_own_edits_as_speaker_layer(workspace, voice, stub, monkeypatch):
    """Правка слота помечена слоем спикера, а не слоем реплики.

    Панель слотов — это и есть слой спикера: если подписать его правку «правка
    этой реплики», пользователь ждал бы сброса к голосу, а кнопка вернула бы
    значение, унаследованное от голоса, — то есть подпись и поведение разошлись.
    """
    _write_voices(voice)

    async def scenario():
        async with _client(monkeypatch) as client:
            await client.patch(f"/api/voices/{voice.id}", json={"preset": {"speed": 1.3}})
            project = await _project(client, DIALOGUE, {"ИВАН": voice.id, "МАРГО": voice.id})

            patched = await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": voice.id, "overrides": {"speed": 1.6}}}},
            )
            assert patched.status_code == 200, patched.text
            payload = patched.json()
            slot = next(item for item in payload["speakers"] if item["key"] == "ИВАН")
            assert slot["settings"]["speed"] == {
                "value": 1.6,
                "source": SOURCE_SPEAKER,
                "inherited": 1.3,
            }
            # Реплика участника видит ту же правку как наследуемую от участника.
            replica = next(item for item in payload["replicas"] if item["speaker"] == "ИВАН")
            assert replica["settings"]["speed"] == {
                "value": 1.6,
                "source": SOURCE_SPEAKER,
                "inherited": 1.3,
            }

            cleared = await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": voice.id, "overrides": {}}}},
            )
            slot = next(item for item in cleared.json()["speakers"] if item["key"] == "ИВАН")
            assert slot["settings"]["speed"] == {
                "value": 1.3,
                "source": SOURCE_VOICE,
                "inherited": config.DEFAULT_SPEED,
            }

    asyncio.run(scenario())
