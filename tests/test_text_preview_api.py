"""Preview «что услышит модель»: стадии preprocessing и REST-эндпоинт.

Тесты повторяют восемь пунктов плана фазы 7. Главный из них — пятый: preview
обязан совпадать с тем, что реально уходит в движок. Поэтому показанный
эндпоинтом текст сравнивается с `stub.calls[...]["text"]` настоящего рендера, а не
с повторно посчитанной формулой: иначе тест проверял бы вторую реализацию вместо
отсутствия второй реализации.

Модели не поднимаются: движок — заглушка, голоса и база живут в `tmp_path`
(фикстура `workspace`), RUAccent подменён предсказуемой разметкой.
"""

import asyncio
import contextlib

import httpx
import pytest
import soundfile as sf
from conftest import StubEngine, sine

from backend import audio_pipeline, config, main
from backend.accentizer import STATE_FAILED, Accentizer
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines import registry
from backend.engines.base import ENGINE_F5, ENGINE_XTTS, SAMPLE_RATE, EngineInfo
from backend.pronunciation import get_store as get_pronunciation_store
from backend.voices_store import Voice

DIALOGUE = "ИВАН: В 2026 году цена выросла на 5%."
REPLICA_TEXT = "В 2026 году цена выросла на 5%."
F5_VOICE = "voice-f5"
XTTS_VOICE = "voice-xtts"


class AccentStubEngine(StubEngine):
    """Заглушка F5: говорит, что понимает «+»-ударения, как настоящий движок."""

    info = EngineInfo(
        id="stub-accents",
        label="Заглушка с ударениями",
        description="Тестовый движок вместо F5-TTS — модель не поднимается.",
        supports_accents=True,
    )


class _VoiceStore:
    """Хранилище из нескольких голосов: preview есть из чего выбирать движок."""

    def __init__(self, *voices: Voice) -> None:
        self._voices = {voice.id: voice for voice in voices}

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)


@pytest.fixture
def voices(workspace, monkeypatch):
    """Два голоса разных движков — по одному на каждую ветку ударений."""
    sf.write(workspace / "voices" / "f5.wav", sine(2.0, 180.0), SAMPLE_RATE)
    sf.write(workspace / "voices" / "xtts.wav", sine(2.0, 240.0), SAMPLE_RATE)
    f5 = Voice(
        id=F5_VOICE,
        name="Ф5",
        gender="male",
        ref_text="Привет, это тест",
        audio_file="f5.wav",
        engine=ENGINE_F5,
    )
    xtts = Voice(
        id=XTTS_VOICE,
        name="Икс",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="xtts.wav",
        engine=ENGINE_XTTS,
    )
    store = _VoiceStore(f5, xtts)
    # Эндпоинт разрешает голос через `main`, синтез — через `audio_pipeline`:
    # в бою это один и тот же singleton, в тесте подменяем оба, чтобы они не
    # разошлись и preview не оказался «про другой голос».
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(audio_pipeline, "get_store", lambda: store)
    return f5, xtts


@pytest.fixture
def fake_accent(monkeypatch):
    """Предсказуемая «RUAccent»: маркер вместо настоящей модели ударений."""

    def accent(text: str) -> str:
        return f"[{text}]"

    monkeypatch.setattr(audio_pipeline, "accentuate", accent)


@pytest.fixture
def accent_stub(monkeypatch):
    """Движок рендера, понимающий ударения (как F5): без него их не проверить."""
    engine = AccentStubEngine()
    monkeypatch.setattr(audio_pipeline, "get_engine", lambda engine_id: engine)
    return engine


@contextlib.asynccontextmanager
async def _client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _preview(client, **payload):
    return await client.post("/api/text/preview", json=payload)


async def _project(client, text: str = DIALOGUE) -> dict:
    """Проект с разобранными репликами — источник текста и голоса для preview."""
    created = await client.post(
        "/api/projects", json={"name": "Preview", "source_text": text, "mode": "dialogue"}
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    parsed = await client.post(f"/api/projects/{project_id}/parse", json={})
    assert parsed.status_code == 200, parsed.text
    return {"id": project_id, "replicas": parsed.json()["replicas"]}


def _failing_accentizer(monkeypatch, message: str) -> Accentizer:
    """RUAccent, который не поднимается: так проверяется явная ошибка в ответе."""
    accentizer = Accentizer()

    def fail() -> bool:
        accentizer._state = STATE_FAILED
        accentizer._last_error = message
        return False

    monkeypatch.setattr(accentizer, "load", fail)
    monkeypatch.setattr(Accentizer, "instance", classmethod(lambda cls: accentizer))
    return accentizer


# --- 1. preview F5 ---------------------------------------------------------------
def test_preview_f5_shows_all_stages(voices, fake_accent):
    async def scenario():
        async with _client() as client:
            response = await _preview(client, text=REPLICA_TEXT, voice_id=F5_VOICE)
            assert response.status_code == 200, response.text
            data = response.json()

            assert data["engine"] == ENGINE_F5
            assert data["engine_label"]
            assert data["supports_accents"] is True
            assert data["accents_applied"] is True
            assert data["original"] == REPLICA_TEXT
            # Числа развёрнуты ещё до словаря и ударений — общей нормализацией.
            assert "две тысячи двадцать шестом" in data["normalized"]
            assert "пять процентов" in data["normalized"]
            # Ни одного правила словаря: стадия словаря равна нормализации.
            assert data["dictionary"] == data["normalized"]
            assert data["matches"] == []
            # Ударения — отдельная стадия, у F5 она отличима от словаря.
            assert data["accentized"] == f"[{data['dictionary']}]"
            assert data["final"] == data["accentized"]
            assert set(data["accentizer"]) == {"state", "error"}

    _run(scenario)


# --- 2. preview XTTS -------------------------------------------------------------
def test_preview_xtts_skips_accent_stage(voices, fake_accent):
    async def scenario():
        async with _client() as client:
            response = await _preview(client, text="12 рублей и 5%", voice_id=XTTS_VOICE)
            assert response.status_code == 200, response.text
            data = response.json()

            assert data["engine"] == ENGINE_XTTS
            assert data["supports_accents"] is False
            assert data["accents_applied"] is False
            assert "12" not in data["normalized"]
            # Стадия ударений не выполнялась: маркер подставного RUAccent не появился.
            assert data["accentized"] == data["dictionary"] == data["normalized"]
            assert data["final"] == data["dictionary"]
            assert "[" not in data["final"]

    _run(scenario)


# --- 3. RUAccent только у поддерживающего движка ---------------------------------
def test_ruaccent_applies_only_to_supporting_engine(voices, fake_accent):
    # «+» ставится перед ударной гласной — «цен+а», а не «це+на»: иначе это не
    # разметка ударения, и XTTS получила бы «+» как обычный символ текста.
    get_pronunciation_store().create(source="цена", target="цен+а")

    async def scenario():
        async with _client() as client:
            f5 = (await _preview(client, text="цена растёт", voice_id=F5_VOICE)).json()
            xtts = (await _preview(client, text="цена растёт", voice_id=XTTS_VOICE)).json()

            assert "цен+а" in f5["dictionary"]
            assert f5["final"] == f"[{f5['dictionary']}]"
            assert "+" in f5["final"]
            # XTTS прочитала бы «+» вслух: для неё замена приходит без разметки,
            # и стадия ударений не выполняется вовсе.
            assert "+" not in xtts["dictionary"]
            assert "+" not in xtts["final"]
            assert xtts["final"] == xtts["dictionary"]

    _run(scenario)


# --- 4. правило словаря видно в preview ------------------------------------------
def test_dictionary_rule_is_visible_in_preview(voices, fake_accent):
    get_pronunciation_store().create(source="SQL", target="эскьюэль")

    async def scenario():
        async with _client() as client:
            response = await _preview(
                client, text="SQL и 12 рублей", voice_id=F5_VOICE, auto_accent=False
            )
            assert response.status_code == 200, response.text
            data = response.json()

            assert data["matches"] == [{"source": "SQL", "target": "эскьюэль", "count": 1}]
            assert "эскьюэль" in data["dictionary"]
            assert data["dictionary"] != data["normalized"]
            # Ударения выключены — стадия равна словарю, и это видно в ответе.
            assert data["accents_applied"] is False
            assert data["accentized"] == data["dictionary"]
            assert data["final"] == data["dictionary"]
            assert data["original"] == "SQL и 12 рублей"

    _run(scenario)


# --- 5. preview совпадает с фактическим input синтеза ----------------------------
def test_preview_matches_synthesis_input_for_f5(voices, fake_accent, accent_stub):
    """Ключевой тест: показанное preview — ровно то, что получил движок."""

    async def scenario():
        async with _client() as client:
            data = (
                await _preview(client, text=REPLICA_TEXT, voice_id=F5_VOICE)
            ).json()
            await audio_pipeline.render_dialogue(
                job_id="preview-f5",
                replicas=[Replica(voice="#1", text=REPLICA_TEXT, line_number=1)],
                speakers={"#1": SpeakerSettings(voice_id=F5_VOICE)},
                settings=RenderSettings(pause_ms=0, output_format="wav"),
            )
            assert accent_stub.calls[0]["text"] == data["final"]
            assert "+" in data["final"] or "[" in data["final"]

    _run(scenario)


def test_preview_matches_synthesis_input_for_xtts(voices, fake_accent, stub):
    """Тот же тест для движка без ударений: preview не обещает лишнего."""
    get_pronunciation_store().create(source="SQL", target="эскьюэль")

    async def scenario():
        async with _client() as client:
            data = (
                await _preview(client, text="SQL и 12 рублей", voice_id=XTTS_VOICE)
            ).json()
            await audio_pipeline.render_dialogue(
                job_id="preview-xtts",
                replicas=[Replica(voice="#1", text="SQL и 12 рублей", line_number=1)],
                speakers={"#1": SpeakerSettings(voice_id=XTTS_VOICE)},
                settings=RenderSettings(pause_ms=0, output_format="wav"),
            )
            assert stub.calls[0]["text"] == data["final"]
            assert "+" not in stub.calls[0]["text"]

    _run(scenario)


# --- 6. эндпоинт не поднимает TTS-движок -----------------------------------------
def test_preview_does_not_create_or_load_engine(voices, fake_accent, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("preview не должен создавать или грузить TTS-движок")

    monkeypatch.setattr(registry, "get_engine", explode)
    monkeypatch.setattr(registry, "_create", explode)
    monkeypatch.setattr(audio_pipeline, "get_engine", explode)
    before = set(registry.created_engines())

    async def scenario():
        async with _client() as client:
            for payload in (
                {"text": "Привет", "voice_id": F5_VOICE},
                {"text": "Привет", "engine": ENGINE_XTTS},
            ):
                response = await _preview(client, **payload)
                assert response.status_code == 200, response.text
            assert set(registry.created_engines()) == before

    _run(scenario)


# --- 7. ошибка accentizer отражается явно ----------------------------------------
def test_accentizer_failure_is_reported_explicitly(voices, monkeypatch):
    _failing_accentizer(monkeypatch, "модель ударений не найдена")

    async def scenario():
        async with _client() as client:
            response = await _preview(client, text=REPLICA_TEXT, voice_id=F5_VOICE)
            # Отказ RUAccent — не ошибка запроса: preview обязан показать текст
            # без ударений и причину рядом со стадией.
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["accentizer"]["state"] == STATE_FAILED
            assert "не найдена" in data["accentizer"]["error"]
            assert data["accents_applied"] is True
            assert data["final"] == data["dictionary"]
            assert "+" not in data["final"]

    _run(scenario)


# --- 8. исходный текст не изменяется ---------------------------------------------
def test_preview_does_not_change_source_text(voices, fake_accent):
    get_pronunciation_store().create(source="SQL", target="эскьюэль")

    async def scenario():
        async with _client() as client:
            project = await _project(client)
            await client.patch(
                f"/api/projects/{project['id']}/replicas/0", json={"voice_id": F5_VOICE}
            )
            before = (await client.get(f"/api/projects/{project['id']}")).json()

            direct = await _preview(client, text="SQL и 12 рублей", voice_id=F5_VOICE)
            assert direct.status_code == 200
            assert direct.json()["original"] == "SQL и 12 рублей"

            replica = await _preview(
                client, project_id=project["id"], replica_index=0
            )
            assert replica.status_code == 200, replica.text
            assert replica.json()["original"] == REPLICA_TEXT

            after = (await client.get(f"/api/projects/{project['id']}")).json()
            assert after["source_text"] == before["source_text"]
            assert after["replicas"] == before["replicas"]
            # Словарь тоже не пополнился «на всякий случай».
            assert (await client.get("/api/pronunciation")).json()["count"] == 1

    _run(scenario)


# --- проект и реплика как источник текста и голоса -------------------------------
def test_preview_takes_text_and_voice_from_replica(voices, fake_accent):
    async def scenario():
        async with _client() as client:
            project = await _project(client, "ИВАН: цена растёт")
            # Голос спикера наследуется репликой без собственного выбора.
            patched = await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": F5_VOICE}}},
            )
            assert patched.status_code == 200, patched.text
            inherited = (
                await _preview(client, project_id=project["id"], replica_index=0)
            ).json()
            assert inherited["engine"] == ENGINE_F5
            assert inherited["supports_accents"] is True
            assert inherited["original"] == "цена растёт"

            # Собственный голос реплики перекрывает голос спикера.
            await client.patch(
                f"/api/projects/{project['id']}/replicas/0", json={"voice_id": XTTS_VOICE}
            )
            overridden = (
                await _preview(client, project_id=project["id"], replica_index=0)
            ).json()
            assert overridden["engine"] == ENGINE_XTTS
            assert overridden["supports_accents"] is False

    _run(scenario)


# --- понятные ошибки на некорректный вход ---------------------------------------
def test_preview_rejects_bad_input(voices, fake_accent):
    async def scenario():
        async with _client() as client:
            project = await _project(client)
            cases = [
                ({"text": "   "}, 400),
                ({}, 400),
                ({"text": "Привет"}, 400),
                ({"text": "Привет", "engine": "нет-такого"}, 400),
                ({"text": "Привет", "voice_id": "нет-голоса"}, 404),
                ({"text": "Привет", "replica_index": 0}, 400),
                ({"project_id": project["id"]}, 400),
                ({"text": "Привет", "project_id": "нет-проекта", "replica_index": 0}, 404),
                ({"project_id": project["id"], "replica_index": 99}, 404),
                ({"text": "x" * (config.MAX_TEXT_CHARS + 1)}, 400),
            ]
            for payload, code in cases:
                response = await _preview(client, **payload)
                assert response.status_code == code, (payload, response.text)
                assert response.json()["detail"]

    _run(scenario)


# --- фаза 6 не сломана -----------------------------------------------------------
def test_pronunciation_preview_still_works(voices):
    get_pronunciation_store().create(source="SQL", target="эскьюэль")

    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/pronunciation/preview",
                json={"text": "SQL и 12 рублей", "engine": ENGINE_F5},
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["original"] == "SQL и 12 рублей"
            assert "эскьюэль" in data["result"]
            assert data["matches"][0]["source"] == "SQL"

    _run(scenario)
