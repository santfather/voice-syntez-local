"""Эмоция реплики и референс-профили голоса (UPDATE 2 §3–§12, §40).

Тесты идут по трём границам, которые легко перепутать:

* **эмоция не текст** — ни одно поле эмоции не появляется в произносимой строке,
  а ручной выбор не запускает повторный анализ;
* **референс только своего голоса** — профили живут внутри голоса, и отсутствие
  эмоционального профиля даёт нейтральный откат, а не отказ синтеза;
* **LLM необязательна** — модель выключена, недоступна или ответила мусором, а
  синтез продолжает работать на безопасном значении.

Настоящие модели не поднимаются: LLM подменяется подставным клиентом пакета
(`llm/fake_client.py`), движок — заглушкой conftest.
"""

import asyncio
import json

import pytest
import soundfile as sf
from conftest import analyze_project, sine

# Фикстура анализа сцены живёт в тесте проекта: здесь она переиспользуется как есть,
# чтобы окружение (fake-клиент + планировщик памяти) было ровно одно.
from test_llm_project_analysis import llm_env as llm_scene_env  # noqa: F401
from test_projects_api import _client

from backend import config, emotions
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines.base import ENGINE_F5, SAMPLE_RATE
from backend.llm import analyzer as llm_analyzer
from backend.llm import schemas as llm_schemas
from backend.reference_resolver import ReferenceUnavailableError, resolve_reference
from backend.voices_store import ReferenceProfile, Voice

DIALOGUE = "АРТЁМ: Красивая.\nМАРГО: Это проблема?"


# --- §3.1. Словарь эмоций -----------------------------------------------------
def test_emotion_enum_contains_required_values():
    """Минимальный production-набор на месте, и у каждой эмоции есть подпись."""
    assert set(emotions.EMOTIONS) == {
        emotions.EMOTION_NEUTRAL,
        emotions.EMOTION_QUESTION,
        emotions.EMOTION_DELIGHT,
        emotions.EMOTION_SURPRISE,
        emotions.EMOTION_FEAR,
    }
    assert emotions.EMOTION_AUTO not in emotions.EMOTIONS
    for value in emotions.SELECTABLE_EMOTIONS:
        assert emotions.EMOTION_TITLES[value]
    # Словарь LLM и словарь приложения — один и тот же набор значений.
    assert set(llm_schemas.EMOTION_VALUES) == set(emotions.EMOTIONS)


def test_manual_override_wins_over_detected():
    assert emotions.emotion_effective("QUESTION", "SURPRISE") == "SURPRISE"
    assert emotions.emotion_effective("QUESTION", "") == "QUESTION"
    assert emotions.emotion_effective("", "") == "NEUTRAL"
    # AUTO — не эмоция, а способ её выбрать: override «Авто» не блокирует detected.
    assert emotions.emotion_effective("QUESTION", "AUTO") == "QUESTION"
    # Мусор из запроса или ответа модели не становится новым значением словаря.
    assert emotions.emotion_effective("восторг", None) == "NEUTRAL"
    assert emotions.normalize_emotion("delight") == "DELIGHT"


def test_llm_unavailable_falls_back_safely():
    """Без модели эмоция берётся эвристикой, и это видно по уверенности."""
    assert emotions.heuristic_emotion("Это проблема?").emotion == emotions.EMOTION_QUESTION
    assert emotions.heuristic_emotion("Мы победили!").emotion == emotions.EMOTION_DELIGHT
    assert emotions.heuristic_emotion("Мне страшно!").emotion == emotions.EMOTION_FEAR
    guess = emotions.heuristic_emotion("Красивая.")
    assert guess.emotion == emotions.EMOTION_NEUTRAL
    # Уверенность подсказки ниже, чем у модели: выдавать её за анализ нельзя.
    assert guess.confidence <= 0.5


# --- §4. Эмоция не попадает в текст ------------------------------------------
def test_emotion_metadata_does_not_enter_tts_text(stub, fake_store, monkeypatch):
    """В движок уходит подготовленный текст, а не эмоция и не её теги."""
    voice_id = fake_store.id
    replica = Replica(
        voice="ИВАН",
        text="Это проблема?",
        line_number=1,
        final_text="Это проблема?",
        emotion_detected=emotions.EMOTION_QUESTION,
    )
    settings = RenderSettings(output_format="wav")
    asyncio.run(
        audio_pipeline_render(
            "job-emotion-text",
            [replica],
            {"ИВАН": SpeakerSettings.from_dict({"voice_id": voice_id})},
            settings,
        )
    )
    assert len(stub.calls) == 1
    sent = stub.calls[0]["text"]
    assert sent == "Это проблема?"
    for marker in ("[", "]", "emotion", "QUESTION", "восторг"):
        assert marker not in sent


def audio_pipeline_render(job_id, replicas, speakers, settings):
    from backend import audio_pipeline

    return audio_pipeline.render_dialogue(job_id, replicas, speakers, settings)


# --- §8–§10. Референс-профили -------------------------------------------------
def _voice(profiles=None, **overrides) -> Voice:
    voice = Voice(
        id="voice-a",
        name="А",
        gender="male",
        ref_text="Привет, это тест",
        audio_file="voice-a.wav",
        engine=ENGINE_F5,
    )
    voice.profiles = list(profiles or [])
    for key, value in overrides.items():
        setattr(voice, key, value)
    return voice


@pytest.fixture
def reference_files(workspace):
    """Файлы референсов в tmp-каталоге: резолвер проверяет их наличие."""
    for name in ("voice-a.wav", "voice-b.wav", "question.wav", "delight.wav", "bad.wav"):
        sf.write(workspace / "voices" / name, sine(0.4, 180.0), SAMPLE_RATE)
    return workspace / "voices"


def test_existing_voice_ref_migrates_to_neutral():
    """Старый голос без профилей читается как NEUTRAL — без миграции данных."""
    voice = _voice()
    profiles = voice.reference_profiles()
    assert len(profiles) == 1
    assert profiles[0].emotion == emotions.EMOTION_NEUTRAL
    # Даже когда рядом уже есть эмоциональные профили, основной референс остаётся
    # доступным: «нет референса для восторга» не значит «нет референса вообще».
    voice.profiles = [
        ReferenceProfile(id="p1", emotion="QUESTION", audio_file="question.wav", ref_text="Да?")
    ]
    assert [item.emotion for item in voice.reference_profiles()] == [
        emotions.EMOTION_NEUTRAL,
        emotions.EMOTION_QUESTION,
    ]


def test_question_and_delight_reference_can_be_selected(reference_files):
    voice = _voice(
        [
            ReferenceProfile(
                id="q1", emotion="QUESTION", audio_file="question.wav",
                ref_text="Ты уверен?", quality_status="ok",
            ),
            ReferenceProfile(
                id="d1", emotion="DELIGHT", audio_file="delight.wav",
                ref_text="Ура!", quality_status="ok",
            ),
        ]
    )
    question = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_QUESTION)
    assert question.profile_id == "q1"
    assert question.fallback_used is False
    delight = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_DELIGHT)
    assert delight.profile_id == "d1"


def test_surprise_and_fear_without_reference_use_neutral(reference_files):
    voice = _voice()
    for emotion in (emotions.EMOTION_SURPRISE, emotions.EMOTION_FEAR, emotions.EMOTION_DELIGHT):
        resolved = resolve_reference(voice, ENGINE_F5, emotion)
        assert resolved.resolved_emotion == emotions.EMOTION_NEUTRAL
        assert resolved.fallback_used is True
        assert resolved.profile_id.endswith("-neutral")
        assert resolved.reason  # причина называется словами, а не молчанием


def test_bad_emotion_reference_falls_back_to_neutral(reference_files):
    """Референс с расходящейся расшифровкой не предпочитается нейтральному (§38)."""
    voice = _voice(
        [
            ReferenceProfile(
                id="bad", emotion="QUESTION", audio_file="bad.wav", ref_text="Что-то другое",
                quality_status="warning", quality_note="расшифровка не совпала с записью",
            )
        ]
    )
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_QUESTION)
    assert resolved.resolved_emotion == emotions.EMOTION_NEUTRAL
    assert resolved.fallback_used is True
    assert "расшифровка" in resolved.reason


def test_emotion_reference_never_uses_other_voice(reference_files):
    """Профиль другого голоса недостижим: он лежит в записи своего голоса."""
    first = _voice(
        [ReferenceProfile(id="q1", emotion="QUESTION", audio_file="question.wav", ref_text="Да?")]
    )
    second = Voice(
        id="voice-b", name="Б", gender="female", ref_text="Привет",
        audio_file="voice-b.wav", engine=ENGINE_F5,
    )
    resolved = resolve_reference(first, ENGINE_F5, emotions.EMOTION_QUESTION)
    assert resolved.voice_id == first.id
    # У второго голоса своего эмоционального профиля нет — и он его не получает.
    assert resolve_reference(second, ENGINE_F5, emotions.EMOTION_QUESTION).resolved_emotion == (
        emotions.EMOTION_NEUTRAL
    )
    # Явно запрошенный чужой профиль не подставляется молча.
    assert (
        resolve_reference(second, ENGINE_F5, emotions.EMOTION_QUESTION, profile_id="q1").profile_id
        != "q1"
    )


def test_reference_profile_belongs_to_voice(reference_files):
    voice = _voice(
        [ReferenceProfile(id="q1", emotion="QUESTION", audio_file="question.wav", ref_text="Да?")]
    )
    assert voice.profile("q1") is not None
    assert voice.profile("чужой") is None


def test_missing_reference_file_is_a_named_error(reference_files):
    """Голос без файлов вообще — это ошибка с причиной, а не тихий синтез без референса."""
    voice = _voice(audio_file="нет-файла.wav")
    voice.profiles = []
    with pytest.raises(ReferenceUnavailableError) as error:
        resolve_reference(voice, ENGINE_F5, emotions.EMOTION_NEUTRAL)
    assert "референс" in str(error.value)


# --- §32, §40. Схема и разбор ответа LLM -------------------------------------
def test_llm_emotion_result_is_schema_validated():
    """Выдуманная эмоция — ошибка ответа, а не новое значение словаря."""
    payload = {
        "schema_version": llm_schemas.SCHEMA_VERSION,
        "replica_id": 1,
        "items": [],
        "utterance": {"class": "NORMAL", "emotion": "ЯРОСТЬ"},
    }
    analysis, errors = llm_schemas.parse_analysis(json.dumps(payload, ensure_ascii=False))
    assert analysis is None
    assert llm_schemas.ERROR_UNKNOWN_EMOTION in errors

    payload["utterance"] = {"class": "NORMAL", "emotion": "QUESTION", "emotion_confidence": 0.9}
    analysis, errors = llm_schemas.parse_analysis(json.dumps(payload, ensure_ascii=False))
    assert errors == []
    assert analysis is not None
    assert analysis.utterance.emotion == "QUESTION"
    assert analysis.utterance.emotion_confidence == 0.9


def test_emotion_schema_has_no_text_field():
    """В схеме ответа нет поля для текста: эмоцию нельзя выразить тегом (§4)."""
    schema = llm_schemas.analysis_json_schema()
    assert "emotion" in schema["properties"]["utterance"]["properties"]
    assert "emotion" in schema["properties"]["utterance"]["required"]
    # Ни в одном объекте схемы нет поля, куда можно положить переписанный текст.
    assert set(schema["properties"]) == {"schema_version", "replica_id", "items", "utterance"}
    assert "text" not in schema["properties"]["utterance"]["properties"]
    assert "text" not in schema["properties"]["items"]["items"]["properties"]


def test_llm_emotion_does_not_modify_final_text(
    llm_scene_env, fake_store, monkeypatch  # noqa: F811 — импортированная фикстура
):
    """Анализ добавляет эмоцию, но текст реплики остаётся тем же."""
    answer = json.dumps(
        {
            "schema_version": llm_schemas.SCHEMA_VERSION,
            "replica_id": 1,
            "items": [],
            "utterance": {
                "class": "SHORT_REPLY",
                "context_dependency": "HIGH",
                "emotion": "QUESTION",
                "emotion_confidence": 0.9,
                "dialogue_act": "вопрос",
            },
        },
        ensure_ascii=False,
    )
    client = llm_scene_env(lambda case: answer)

    async def scenario():
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            response = await http.get(f"/api/projects/{project['id']}")
            assert response.status_code == 200, response.text
            return response.json(), client

    payload, client = asyncio.run(scenario())
    texts = [row["final_text"] for row in payload["replicas"]]
    # Текст не переписан и не содержит служебных пометок.
    assert texts == ["Красивая.", "Это проблема?"]
    for text in texts:
        assert "emotion" not in text.lower()
    assert client.calls


def test_llm_emotion_uses_scene_context(llm_scene_env):  # noqa: F811 — импортированная фикстура
    """Модель получает соседние реплики: без сцены «Это проблема?» не читается."""
    client = llm_scene_env(
        lambda case: json.dumps(
            {
                "schema_version": llm_schemas.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [],
                "utterance": {"class": "NORMAL", "emotion": "NEUTRAL"},
            },
            ensure_ascii=False,
        )
    )

    analyzer = llm_analyzer.get_analyzer()
    analyzer.analyze_replica(
        replica_id=2,
        target_text="Это проблема?",
        before=["Мне тридцать девять, мальчик."],
        after=[],
    )
    assert client.calls
    content = client.calls[0]["messages"][1]["content"]
    # В user-сообщении есть строка-инструкция, поэтому JSON достаётся по первой
    # фигурной скобке — ровно так же, как это делает сам анализатор.
    payload = json.loads(content[content.index("{"):])
    assert payload["context_before"] == ["Мне тридцать девять, мальчик."]
    assert payload["target_text"] == "Это проблема?"


# --- §11, §12, §36. API и regenerate -----------------------------------------
def test_emotion_override_survives_api_and_does_not_touch_text(stub, fake_store, monkeypatch):
    async def scenario():
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            patched = await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"emotion_override": "SURPRISE"},
            )
            assert patched.status_code == 200, patched.text
            replica = patched.json()["replica"]
            assert replica["emotion_override"] == "SURPRISE"
            assert replica["emotion_effective"] == "SURPRISE"
            assert replica["emotion"]["effective_title"] == "Удивление"
            # Ручной выбор не требует повторного анализа и не меняет текст.
            assert replica["analysis_status"] == config.REPLICA_ANALYSIS_DONE
            assert replica["final_text"] == "Это проблема?"

            # Снятие override возвращает «Авто».
            cleared = await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"emotion_override": None},
            )
            assert cleared.json()["replica"]["emotion_override"] == ""
            assert cleared.json()["replica"]["emotion"]["source"] in ("llm", "heuristic", "none")

    asyncio.run(scenario())


def test_regenerate_can_change_emotion_without_touching_neighbors(stub, fake_store, monkeypatch):
    """Пересинтез одной реплики с новой эмоцией: соседи и их take'ы не меняются."""
    async def scenario():
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            accepted = await http.post(
                f"/api/projects/{project['id']}/render",
                json={"output_format": "wav", "output_name": "эмоции"},
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(http, accepted.json()["job_id"])
            before = (await http.get(f"/api/projects/{project['id']}")).json()
            neighbor_takes = len(before["replicas"][0]["takes"])

            response = await http.post(
                f"/api/projects/{project['id']}/replicas/1/regenerate",
                json={"emotion_override": "QUESTION"},
            )
            assert response.status_code == 202, response.text
            await _wait_job(http, response.json()["job_id"])

            after = (await http.get(f"/api/projects/{project['id']}")).json()
            assert len(after["replicas"][0]["takes"]) == neighbor_takes
            assert after["replicas"][1]["emotion_override"] == "QUESTION"
            # Вариант помнит, с каким референсом и эмоцией он получен (§12).
            take = after["replicas"][1]["takes"][-1]
            parameters = take["parameters"]
            assert parameters["reference_emotion"]
            assert parameters["reference_profile_id"]
            assert "reference_fallback_used" in parameters

    asyncio.run(scenario())


# --- helpers ------------------------------------------------------------------
async def _project(client, voice_id: str) -> dict:
    """Проект с двумя репликами и назначенным голосом из tmp-хранилища."""
    created = await client.post(
        "/api/projects", json={"name": "Эмоции", "source_text": DIALOGUE, "mode": "dialogue"}
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
    return project


async def _wait_job(client, job_id: str, timeout: float = 30.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"задача {job_id} не завершилась")
