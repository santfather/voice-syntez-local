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
import io
import json
import sqlite3
from pathlib import Path

import pytest
import soundfile as sf
from conftest import STUB_ENGINE_ID, analyze_project, sine

# Фикстура анализа сцены живёт в тесте проекта: здесь она переиспользуется как есть,
# чтобы окружение (fake-клиент + планировщик памяти) было ровно одно.
from test_llm_project_analysis import llm_env as llm_scene_env  # noqa: F401
from test_projects_api import _client

from backend import audio_pipeline, config, emotions
from backend.audio_pipeline import RenderSettings, SpeakerSettings
from backend.dialogue_parser import Replica
from backend.engines.base import ENGINE_F5, SAMPLE_RATE
from backend.llm import analyzer as llm_analyzer
from backend.llm import schemas as llm_schemas
from backend.reference_resolver import ReferenceUnavailableError, resolve_reference
from backend.voices_store import ReferenceProfile, Voice
from backend.voices_store import get_store as get_voices_store

DIALOGUE = "АРТЁМ: Красивая.\nМАРГО: Это проблема?"


# --- §3.1. Словарь эмоций -----------------------------------------------------
def test_emotion_enum_contains_required_values():
    """Production-набор профилей на месте, и у каждой эмоции есть подпись."""
    # 11 интонаций §10 — по одной на каждую записываемую фразу: из них состоит
    # словарь профилей, и только они могут быть ключом записанного референса.
    assert set(emotions.PROFILE_KEYS) == {
        emotions.EMOTION_NEUTRAL,
        emotions.EMOTION_CALM,
        emotions.EMOTION_QUESTION,
        emotions.EMOTION_NEUTRAL_QUESTION,
        emotions.EMOTION_EXCLAMATION,
        emotions.EMOTION_DELIGHT,
        emotions.EMOTION_SAD_SYMPATHETIC,
        emotions.EMOTION_IRONIC,
        emotions.EMOTION_STRICT,
        emotions.EMOTION_ENUMERATION,
        emotions.EMOTION_EXCITED,
    }
    # Семантика шире записи: испуг и удивление различимы, но профиля под них нет —
    # резолвер уводит их в ближайший разрешённый профиль с явным откатом.
    assert set(emotions.EMOTIONS) == set(emotions.PROFILE_KEYS) | {
        emotions.EMOTION_SURPRISE,
        emotions.EMOTION_FEAR,
    }
    assert emotions.EMOTION_AUTO not in emotions.EMOTIONS
    for value in emotions.SELECTABLE_EMOTIONS:
        assert emotions.EMOTION_TITLES[value]
    # Словарь LLM и словарь приложения — один и тот же набор значений.
    assert set(llm_schemas.EMOTION_VALUES) == set(emotions.EMOTIONS)


def test_profile_key_is_limited_to_recordable_emotions():
    """Ключ профиля — только из набора §10, а семантика узнаётся шире него."""
    assert emotions.is_profile_key(emotions.EMOTION_EXCITED)
    assert emotions.normalize_profile_key("excited") == emotions.EMOTION_EXCITED
    # Под испуг и удивление записи нет: создать такой профиль нельзя.
    assert not emotions.is_profile_key(emotions.EMOTION_FEAR)
    assert not emotions.is_profile_key(emotions.EMOTION_SURPRISE)
    assert emotions.normalize_profile_key(emotions.EMOTION_FEAR) == emotions.EMOTION_NEUTRAL
    assert emotions.normalize_profile_key("ЯРОСТЬ") == emotions.EMOTION_NEUTRAL
    # При этом семантическое значение остаётся известным эмоцией.
    assert emotions.is_emotion(emotions.EMOTION_FEAR)


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


# --- §16. Мастер записи: фразы становятся профилями одного голоса --------------
def _reference_wav(seconds: float = 1.0) -> bytes:
    """WAV-запись для профиля: мастер пишет в профили именно файл записи."""
    buffer = io.BytesIO()
    sf.write(buffer, sine(seconds, 180.0), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def _maria() -> Voice:
    return get_voices_store().create(
        name="Мария",
        gender="female",
        ref_text="Привет, это тест",
        audio_filename="maria.wav",
        audio_bytes=_reference_wav(),
        verify_ref_text=False,
        engine=ENGINE_F5,
    )


def _record_phrase(voice_id: str, key: str, phrase_id: str):
    return get_voices_store().add_reference(
        voice_id,
        emotion=key,
        audio_filename=f"phrase-{phrase_id}.wav",
        audio_bytes=_reference_wav(),
        ref_text=f"Фраза {phrase_id}",
        label=f"Фраза {phrase_id}",
        verify_ref_text=False,
        source_record_phrase_id=phrase_id,
    )


def test_eleven_recorded_phrases_stay_profiles_of_one_voice():
    """Gate §16: одиннадцать записей Марии — один голос и одиннадцать профилей.

    Проверяется на хранилище, а не на интерфейсе: именно здесь решается, станет ли
    запись новым голосом или профилем существующего. Если бы каждая фраза создавала
    голос, «Марий» стало бы одиннадцать, а интонацию реплики выбирать было бы не из
    чего — у каждого голоса оказался бы ровно один референс.
    """
    voice = _maria()
    for index, key in enumerate(emotions.PROFILE_KEYS):
        _record_phrase(voice.id, key, str(index))

    assert len(get_voices_store().list()) == 1, "профили не создают новых голосов"
    saved = get_voices_store().get(voice.id)
    assert saved is not None
    assert {item.emotion for item in saved.profiles} == set(emotions.PROFILE_KEYS)
    # Все профили принадлежат этому голосу: чужой референс не может быть выбран
    # даже случайно — он лежит внутри записи голоса, а не в общем списке.
    assert {item.voice_id for item in saved.profiles} == {voice.id}
    assert {item.source_record_phrase_id for item in saved.profiles} == {
        str(index) for index in range(len(emotions.PROFILE_KEYS))
    }
    assert all(item.audio_path.exists() for item in saved.profiles)


def test_rerecording_same_phrase_replaces_profile_instead_of_duplicating():
    """Одна фраза — один профиль (§16): повторная запись заменяет прежнюю."""
    voice = _maria()
    first = _record_phrase(voice.id, emotions.EMOTION_IRONIC, "7")
    second = _record_phrase(voice.id, emotions.EMOTION_IRONIC, "7")

    saved = get_voices_store().get(voice.id)
    assert saved is not None
    assert [item.id for item in saved.profiles] == [second.id], (
        "дубликат одной интонации заставил бы резолвер выбирать самую раннюю запись"
    )
    assert not first.audio_path.exists(), "файл вытесненной записи удаляется, а не копится"


def test_recorded_profile_is_not_trusted_for_auto_until_confirmed():
    """Свежая запись в автоматику не идёт: сначала benchmark и прослушивание (§35)."""
    voice = _maria()
    profile = _record_phrase(voice.id, emotions.EMOTION_IRONIC, "7")
    assert profile.enabled_for_auto is False
    # Флаг переживает перезапись `voices.json`: подтверждение — это данные профиля,
    # а не состояние процесса, и после перезапуска оно обязано сохраниться.
    saved = get_voices_store().get(voice.id)
    assert saved is not None
    assert [item.enabled_for_auto for item in saved.profiles] == [False]


def test_profile_confirmation_can_be_granted_and_revoked():
    """Подтверждение профиля ставится и снимается флагом (§35)."""
    voice = _maria()
    profile = _record_phrase(voice.id, emotions.EMOTION_IRONIC, "7")
    store = get_voices_store()

    assert store.update_reference(voice.id, profile.id, enabled_for_auto=True).enabled_for_auto
    saved = store.get(voice.id)
    assert saved is not None
    assert [item.enabled_for_auto for item in saved.profiles] == [True]
    # Соседние поля не задеты: подтверждение не переписывает саму запись.
    assert saved.profiles[0].audio_file == profile.audio_file

    assert not store.update_reference(voice.id, profile.id, enabled_for_auto=False).enabled_for_auto


def test_old_profile_json_gets_no_auto_rights():
    """Профиль из старого `voices.json` читается неподтверждённым (§35)."""
    profile = ReferenceProfile.from_dict(
        {"id": "p1", "emotion": "QUESTION", "audio_file": "q.wav"}
    )
    assert profile is not None
    assert profile.enabled_for_auto is False
    # Явно подтверждённый профиль читается как есть.
    confirmed = ReferenceProfile.from_dict(
        {
            "id": "p2",
            "emotion": "QUESTION",
            "audio_file": "q.wav",
            "enabled_for_auto": True,
        }
    )
    assert confirmed is not None and confirmed.enabled_for_auto is True


def test_confirmation_flag_is_editable_through_the_api(monkeypatch):
    """Флаг подтверждения ходит через API: из интерфейса его иначе не поставить (§35)."""
    voice = _maria()
    profile = _record_phrase(voice.id, emotions.EMOTION_IRONIC, "7")

    async def scenario() -> tuple[dict, dict]:
        async with _client(monkeypatch) as http:
            granted = await http.patch(
                f"/api/voices/{voice.id}/references/{profile.id}",
                json={"enabled_for_auto": True},
            )
            assert granted.status_code == 200, granted.text
            listed = await http.get("/api/voices")
            assert listed.status_code == 200, listed.text
            return granted.json(), listed.json()

    granted, listed = asyncio.run(scenario())
    assert granted["profile"]["enabled_for_auto"] is True
    # Ответ PATCH'а — не единственное место с флагом: список голосов его тоже
    # отдаёт, и именно оттуда интерфейс рисует галочку «авто».
    saved = next(item for item in listed["voices"] if item["id"] == voice.id)
    row = next(item for item in saved["reference_profiles"] if item["id"] == profile.id)
    assert row["enabled_for_auto"] is True


def test_semantic_emotion_cannot_be_recorded_as_profile():
    """SURPRISE, FEAR и AUTO профилем не записываются (§17): отказ, а не NEUTRAL."""
    voice = _maria()
    for key in (emotions.EMOTION_SURPRISE, emotions.EMOTION_FEAR, emotions.EMOTION_AUTO):
        with pytest.raises(ValueError):
            _record_phrase(voice.id, key, "0")
    saved = get_voices_store().get(voice.id)
    assert saved is not None
    assert saved.profiles == [], "отвергнутая запись не должна оставлять профиль"


def test_question_and_delight_reference_can_be_selected(reference_files):
    voice = _voice(
        [
            ReferenceProfile(
                id="q1", emotion="QUESTION", audio_file="question.wav",
                ref_text="Ты уверен?", quality_status="ok", enabled_for_auto=True,
            ),
            ReferenceProfile(
                id="d1", emotion="DELIGHT", audio_file="delight.wav",
                ref_text="Ура!", quality_status="ok", enabled_for_auto=True,
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
        [
            ReferenceProfile(
                id="q1", emotion="QUESTION", audio_file="question.wav",
                ref_text="Да?", enabled_for_auto=True,
            )
        ]
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


# --- §70, §71. Диалог одного голоса с несколькими профилями --------------------
def _dialogue_replica(
    text: str, line_number: int, profile: str, *, profile_id: str = ""
) -> Replica:
    """Реплика Марии: текст, рекомендованный профиль и, если нужно, явный выбор."""
    return Replica(
        voice="МАРИЯ",
        text=text,
        line_number=line_number,
        final_text=text,
        prosody_profile=profile,
        reference_profile_id=profile_id,
    )


def test_dialogue_routes_each_replica_to_its_confirmed_profile(stub, workspace, monkeypatch):
    """Один голос — много профилей: каждая реплика берёт свой, и только свой (§70, §71).

    DELIGHT записан, но не подтверждён: автоматика его не берёт (§35), и реплика
    честно уходит на сам голос. Четвёртая реплика выбирает тот же профиль вручную —
    ручной выбор флагом не ограничен, иначе неподтверждённую запись нельзя было бы
    даже прослушать.

    Голос собран здесь напрямую, а не через хранилище: `create` проверяет движок по
    настоящему реестру, а в unit-тесте работает заглушка (§70 — без production-моделей).
    """
    for name in ("maria.wav", "question.wav", "ironic.wav", "delight.wav"):
        sf.write(workspace / "voices" / name, sine(0.4, 180.0), SAMPLE_RATE)

    def profile(profile_id: str, emotion: str, audio: str, *, auto: bool) -> ReferenceProfile:
        return ReferenceProfile(
            id=profile_id,
            voice_id="maria",
            emotion=emotion,
            audio_file=audio,
            ref_text="Ты уверен?",
            quality_status="ok",
            enabled_for_auto=auto,
        )

    voice = Voice(
        id="maria",
        name="Мария",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="maria.wav",
        engine=STUB_ENGINE_ID,
    )
    voice.profiles = [
        profile("m-question", emotions.EMOTION_QUESTION, "question.wav", auto=True),
        profile("m-ironic", emotions.EMOTION_IRONIC, "ironic.wav", auto=True),
        profile("m-delight", emotions.EMOTION_DELIGHT, "delight.wav", auto=False),
    ]

    class Store:
        def get(self, voice_id: str) -> Voice | None:
            return voice if voice_id == voice.id else None

    monkeypatch.setattr(audio_pipeline, "get_store", lambda: Store())

    replicas = [
        _dialogue_replica("Ты уверен?", 1, emotions.EMOTION_QUESTION),
        _dialogue_replica("Ну конечно.", 2, emotions.EMOTION_IRONIC),
        _dialogue_replica("Ура!", 3, emotions.EMOTION_DELIGHT),
        _dialogue_replica("Я в восторге.", 4, emotions.EMOTION_QUESTION, profile_id="m-delight"),
    ]
    asyncio.run(
        audio_pipeline_render(
            "job-maria",
            replicas,
            {"МАРИЯ": SpeakerSettings.from_dict({"voice_id": voice.id})},
            RenderSettings(output_format="wav"),
        )
    )
    assert [Path(call["ref_audio_path"]).name for call in stub.calls] == [
        "question.wav",
        "ironic.wav",
        "maria.wav",  # неподтверждённый профиль не взят — синтез идёт самим голосом
        "delight.wav",  # явный выбор реплики работает и без подтверждения
    ]
    # Ни один вызов не ушёл к чужому голосу: у всех реплик один и тот же `voice_id`.
    assert {Path(call["ref_audio_path"]).parent for call in stub.calls} == {config.VOICES_DIR}
    # Текст до движка доходит без изменений: профиль — метаданные, а не слова (§4).
    assert [call["text"] for call in stub.calls] == [item.final_text for item in replicas]


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


# --- §8–§13, §47. Просодия: словари, диапазоны и безопасный откат -------------
def _prosody_answer(
    *, emotion: str = "NEUTRAL", dialogue_act: str = "", **prosody
) -> str:
    """Ответ модели с блоком просодии; остальное — минимально валидное.

    `emotion` отдельно от `prosody.profile`: первое — строгое поле реплики, второе
    — маршрутизация, и правила у них разные (§47).
    """
    utterance: dict = {"class": "NORMAL", "emotion": emotion, "prosody": prosody}
    if dialogue_act:
        utterance["dialogue_act"] = dialogue_act
    return json.dumps(
        {
            "schema_version": llm_schemas.SCHEMA_VERSION,
            "replica_id": 1,
            "items": [],
            "utterance": utterance,
        },
        ensure_ascii=False,
    )


def test_prosody_vocabulary_matches_application():
    """Словари LLM и приложения не разошлись: акты, профили, темпы (§9–§12)."""
    assert llm_schemas.DIALOGUE_ACT_OTHER in llm_schemas.DIALOGUE_ACTS
    assert len(set(llm_schemas.DIALOGUE_ACTS)) == len(llm_schemas.DIALOGUE_ACTS)
    # Записываемых профилей ровно 11 — те же, что в `emotions.PROFILE_KEYS`.
    assert set(llm_schemas.PROFILE_VALUES) == set(emotions.PROFILE_KEYS)
    assert set(llm_schemas.PROSODY_PACES) == {"SLOW", "NORMAL", "FAST"}


def test_prosody_block_is_parsed_and_kept():
    """Корректная просодия доходит до разбора без потерь (§8)."""
    analysis, errors = llm_schemas.parse_analysis(
        _prosody_answer(
            emotion="IRONIC",
            profile="IRONIC",
            recommended_profile="IRONIC",
            intensity=0.55,
            pace="SLOW",
            confidence=0.91,
        )
    )
    assert errors == []
    prosody = analysis.utterance.prosody
    assert (prosody.profile, prosody.recommended_profile) == ("IRONIC", "IRONIC")
    assert prosody.intensity == 0.55
    assert prosody.pace == "SLOW"
    assert prosody.confidence == 0.91


def test_prosody_profile_is_valid_enum():
    """Профиль из словаря эмоций принимается, выдуманный — отбрасывается."""
    analysis, errors = llm_schemas.parse_analysis(
        _prosody_answer(emotion="SURPRISE", profile="SURPRISE")
    )
    assert errors == []
    assert analysis.utterance.prosody.profile == "SURPRISE"

    analysis, _ = llm_schemas.parse_analysis(
        _prosody_answer(emotion="NEUTRAL", profile="ЯРОСТЬ")
    )
    assert analysis.utterance.prosody.profile == ""


def test_unknown_profile_is_rejected():
    """Неизвестный профиль не подменяется похожим и не роняет разбор (§47)."""
    analysis, errors = llm_schemas.parse_analysis(
        _prosody_answer(
            emotion="NEUTRAL",
            profile="ЯРОСТЬ",
            recommended_profile="ЯРОСТЬ",
            intensity=0.5,
            pace="NORMAL",
            confidence=0.7,
        )
    )
    assert errors == []
    assert analysis is not None
    prosody = analysis.utterance.prosody
    assert prosody.profile == "" and prosody.recommended_profile == ""
    # Остальные поля не тронуты: отказ поля не отменяет весь блок.
    assert prosody.intensity == 0.5 and prosody.pace == "NORMAL"


def test_recommended_profile_is_limited_to_recordable_values():
    """Замена должна быть записываемым профилем: под SURPRISE и FEAR записи нет."""
    analysis, _ = llm_schemas.parse_analysis(
        _prosody_answer(
            emotion="SURPRISE", profile="SURPRISE", recommended_profile="EXCLAMATION"
        )
    )
    assert analysis.utterance.prosody.recommended_profile == "EXCLAMATION"

    analysis, _ = llm_schemas.parse_analysis(
        _prosody_answer(emotion="FEAR", profile="FEAR", recommended_profile="FEAR")
    )
    assert analysis.utterance.prosody.recommended_profile == ""


def test_dialogue_act_is_valid_enum():
    """Речевой акт — из словаря §9; свободная строка становится OTHER."""
    analysis, errors = llm_schemas.parse_analysis(
        _prosody_answer(emotion="QUESTION", dialogue_act="QUESTION")
    )
    assert errors == []
    assert analysis.utterance.dialogue_act == "QUESTION"

    analysis, _ = llm_schemas.parse_analysis(
        _prosody_answer(emotion="QUESTION", dialogue_act="вопрос")
    )
    assert analysis.utterance.dialogue_act == llm_schemas.DIALOGUE_ACT_OTHER


def test_confidence_is_bounded():
    """Уверенность вне 0..1 — не число, а выдумка: поле отбрасывается (§13)."""
    for value, expected in ((0.0, 0.0), (1.0, 1.0), (1.4, None), (-0.1, None)):
        analysis, errors = llm_schemas.parse_analysis(
            _prosody_answer(emotion="NEUTRAL", confidence=value)
        )
        assert errors == []
        assert analysis.utterance.prosody.confidence == expected, value


def test_intensity_is_bounded():
    """Интенсивность вне 0..1 отбрасывается: это метаданные, а не параметр (§11)."""
    for value, expected in ((0.0, 0.0), (0.55, 0.55), (1.0, 1.0), (2.0, None), (-1.0, None)):
        analysis, errors = llm_schemas.parse_analysis(
            _prosody_answer(emotion="DELIGHT", intensity=value)
        )
        assert errors == []
        assert analysis.utterance.prosody.intensity == expected, value


def test_pace_is_valid():
    """Темп — только SLOW/NORMAL/FAST; остальное отбрасывается (§12)."""
    for value, expected in (("SLOW", "SLOW"), ("NORMAL", "NORMAL"), ("FAST", "FAST"), ("БЫСТРО", "")):
        analysis, errors = llm_schemas.parse_analysis(
            _prosody_answer(emotion="EXCITED", pace=value)
        )
        assert errors == []
        assert analysis.utterance.prosody.pace == expected, value


def test_unknown_prosody_field_is_rejected():
    """Лишнее поле внутри просодии — ошибка ответа: схема строгая (§47)."""
    _, errors = llm_schemas.parse_analysis(
        _prosody_answer(emotion="NEUTRAL", profile="NEUTRAL", text="переписанный текст")
    )
    assert errors == [llm_schemas.ERROR_UNKNOWN_FIELD]


def test_prosody_schema_declares_routing_enums():
    """Схема отвечает теми же словарями, по которым backend проверяет ответ."""
    utterance = llm_schemas.analysis_json_schema()["properties"]["utterance"]["properties"]
    prosody = utterance["prosody"]["properties"]
    assert tuple(prosody["profile"]["enum"]) == llm_schemas.EMOTION_VALUES
    assert tuple(prosody["recommended_profile"]["enum"]) == llm_schemas.PROFILE_VALUES
    assert tuple(prosody["pace"]["enum"]) == llm_schemas.PROSODY_PACES
    assert tuple(utterance["dialogue_act"]["enum"]) == llm_schemas.DIALOGUE_ACTS


# --- §36, §37, Phase 7. Персистенция и инвалидация просодии -------------------
def test_prosody_effective_prefers_override_over_recommendation():
    """Действующая интонация: ручной выбор → рекомендация модели → эмоция (§37)."""
    assert emotions.prosody_effective("QUESTION", "EXCITED", "IRONIC") == "EXCITED"
    assert emotions.prosody_effective("QUESTION", "", "IRONIC") == "IRONIC"
    # Без рекомендации работает прежнее правило по эмоции — состояние «до UPDATE 3»
    # читается так же, как читалось, и старые проекты не меняют звучания.
    assert emotions.prosody_effective("QUESTION", "", "") == "QUESTION"
    assert emotions.prosody_effective("", "", "") == emotions.EMOTION_NEUTRAL
    # «Авто» — способ выбрать, а не значение: рекомендация остаётся главной.
    assert emotions.prosody_effective("QUESTION", "AUTO", "IRONIC") == "IRONIC"
    # Мусор не становится профилем: он откатывается к эмоции, а не к пустоте.
    assert emotions.prosody_effective("QUESTION", "ЯРОСТЬ", "") == "QUESTION"


def _scene_prosody_answer(case: dict) -> str:
    """Ответ окна: у каждой реплики — записываемый профиль и метаданные (§8)."""

    def entry(item: dict) -> dict:
        return {
            "schema_version": llm_schemas.SCHEMA_VERSION,
            "replica_id": item["replica_id"],
            "items": [],
            "utterance": {
                "class": "NORMAL",
                "context_dependency": "HIGH",
                "emotion": "IRONIC",
                "emotion_confidence": 0.8,
                "dialogue_act": "STATEMENT",
                "prosody": {
                    "profile": "IRONIC",
                    "recommended_profile": "IRONIC",
                    "intensity": 0.42,
                    "pace": "SLOW",
                    "confidence": 0.77,
                },
            },
        }

    replicas = case.get("replicas")
    if isinstance(replicas, list):
        return json.dumps(
            {
                "schema_version": llm_schemas.SCHEMA_VERSION,
                "replicas": [entry(item) for item in replicas],
            },
            ensure_ascii=False,
        )
    return json.dumps(entry(case), ensure_ascii=False)


def test_prosody_is_persisted_and_reaches_the_card(
    llm_scene_env, fake_store, monkeypatch  # noqa: F811 — импортированная фикстура
):
    """Разбор сцены сохраняет рекомендацию, метаданные и действующий профиль (§36)."""
    llm_scene_env(_scene_prosody_answer, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            response = await http.get(f"/api/projects/{project['id']}")
            assert response.status_code == 200, response.text
            return response.json()

    payload = asyncio.run(scenario())
    assert payload["replicas"]
    for replica in payload["replicas"]:
        prosody = replica["prosody"]
        # Рекомендация и метаданные — как их назвала модель, без пересчёта.
        assert prosody["profile"] == "IRONIC"
        assert prosody["intensity"] == 0.42
        assert prosody["pace"] == "SLOW"
        assert prosody["confidence"] == 0.77
        assert prosody["profile_title"] == emotions.EMOTION_TITLES["IRONIC"]
        # Ручного выбора нет — действует рекомендация модели, и она же объяснима.
        assert prosody["effective"] == "IRONIC"
        assert prosody["effective_title"] == emotions.EMOTION_TITLES["IRONIC"]
        assert prosody["dialogue_act"] == "STATEMENT"
        assert prosody["context_dependency"] == "HIGH"
        # Интонация не попала в произносимый текст.
        assert "IRONIC" not in replica["final_text"]


def test_missing_prosody_numbers_stay_unknown_not_zero(
    llm_scene_env, fake_store, monkeypatch  # noqa: F811 — импортированная фикстура
):
    """«Модель не сказала» отличается от измеренного 0.0 (§11, §13)."""

    def answer(case: dict) -> str:
        def entry(item: dict) -> dict:
            return {
                "schema_version": llm_schemas.SCHEMA_VERSION,
                "replica_id": item["replica_id"],
                "items": [],
                "utterance": {
                    "class": "NORMAL",
                    "emotion": "CALM",
                    "prosody": {"profile": "CALM", "recommended_profile": "CALM"},
                },
            }

        replicas = case.get("replicas")
        if isinstance(replicas, list):
            return json.dumps(
                {
                    "schema_version": llm_schemas.SCHEMA_VERSION,
                    "replicas": [entry(item) for item in replicas],
                },
                ensure_ascii=False,
            )
        return json.dumps(entry(case), ensure_ascii=False)

    llm_scene_env(answer, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            return (await http.get(f"/api/projects/{project['id']}")).json()

    payload = asyncio.run(scenario())
    for replica in payload["replicas"]:
        prosody = replica["prosody"]
        assert prosody["profile"] == "CALM"
        # Пустое значение остаётся неизвестным: 0.0 означало бы «модель измерила».
        assert prosody["intensity"] is None
        assert prosody["confidence"] is None
        assert prosody["pace"] == ""


def test_override_outranks_recommendation_and_clearing_restores_it(
    llm_scene_env, fake_store, monkeypatch  # noqa: F811 — импортированная фикстура
):
    """Ручной выбор сильнее рекомендации, а его снятие возвращает «Авто» (§37)."""
    llm_scene_env(_scene_prosody_answer, handles_window=True)

    async def scenario() -> tuple[dict, dict, dict]:
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            patched = await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"emotion_override": "EXCITED"},
            )
            assert patched.status_code == 200, patched.text
            overridden = patched.json()["replica"]
            cleared = await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"emotion_override": None},
            )
            assert cleared.status_code == 200, cleared.text
            return overridden, cleared.json()["replica"], (
                await http.get(f"/api/projects/{project['id']}")
            ).json()

    overridden, cleared, payload = asyncio.run(scenario())
    # Рекомендация модели не потеряна — она лишь уступает ручному выбору.
    assert overridden["prosody"]["profile"] == "IRONIC"
    assert overridden["prosody"]["effective"] == "EXCITED"
    assert overridden["prosody"]["intensity"] == 0.42
    # Правка эмоции не пересчитывает текст и не требует повторного анализа.
    assert overridden["analysis_status"] == config.REPLICA_ANALYSIS_DONE
    assert cleared["emotion_override"] == ""
    assert cleared["prosody"]["effective"] == "IRONIC"
    # Соседняя реплика ручной правкой не затронута.
    assert payload["replicas"][0]["prosody"]["effective"] == "IRONIC"


def test_text_change_clears_recommendation_but_keeps_manual_choice(
    llm_scene_env, fake_store, monkeypatch  # noqa: F811 — импортированная фикстура
):
    """Правка текста стирает рекомендацию просодии, но не выбор пользователя (§36)."""
    llm_scene_env(_scene_prosody_answer, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"emotion_override": "EXCITED"},
            )
            edited = await http.patch(
                f"/api/projects/{project['id']}/replicas/1",
                json={"text": "Совсем другая фраза."},
            )
            assert edited.status_code == 200, edited.text
            return edited.json()["replica"]

    replica = asyncio.run(scenario())
    # Рекомендация выведена по тексту, которого больше нет, — она обнулена целиком.
    assert replica["prosody"]["profile"] == ""
    assert replica["prosody"]["intensity"] is None
    assert replica["prosody"]["pace"] == ""
    assert replica["prosody"]["confidence"] is None
    # Ручной выбор принадлежит пользователю: анализ его не отменяет.
    assert replica["emotion_override"] == "EXCITED"
    assert replica["prosody"]["effective"] == "EXCITED"


def test_analysis_without_prosody_keeps_saved_recommendation():
    """Разбор без блока просодии — не то же самое, что явное «профиля нет» (§8)."""
    from backend.db.store import get_projects_store

    store = get_projects_store()
    project = store.create_project(
        name="Просодия", source_text="АРТЁМ: Привет.", mode="dialogue"
    )
    store.parse_project(project["id"])
    store.set_replica_emotion(
        project["id"],
        0,
        detected="IRONIC",
        confidence=0.8,
        dialogue_act="STATEMENT",
        prosody={"profile": "IRONIC", "intensity": 0.4, "pace": "SLOW", "confidence": 0.7},
    )
    # Второй разбор просодии не содержит (None) — сохранённая рекомендация цела.
    store.set_replica_emotion(project["id"], 0, detected="NEUTRAL", confidence=0.5)
    replica = store.get_project(project["id"])["replicas"][0]
    assert replica["prosody_profile"] == "IRONIC"
    assert replica["prosody_intensity"] == 0.4
    assert replica["prosody_pace"] == "SLOW"
    assert replica["prosody_confidence"] == 0.7
    # Явная пустая рекомендация, наоборот, стирает профиль: модель сказала «нет».
    store.set_replica_emotion(
        project["id"], 0, detected="NEUTRAL", confidence=0.5, prosody={"profile": ""}
    )
    replica = store.get_project(project["id"])["replicas"][0]
    assert replica["prosody_profile"] == ""
    # Метаданные при этом остались: они описывают разбор, а не профиль.
    assert replica["prosody_intensity"] == 0.4


def test_migration_10_adds_prosody_columns_without_data_loss(tmp_path):
    """Миграция просодии аддитивна: старые реплики и проекты читаются как раньше."""
    from backend.db import migrations

    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    # База «до UPDATE 3»: схема без просодии.
    for version, script in migrations.MIGRATIONS:
        if version >= 10:
            break
        connection.executescript(script)
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    connection.execute("DELETE FROM schema_version")
    connection.execute("INSERT INTO schema_version (version) VALUES (9)")
    connection.execute(
        "INSERT INTO projects (id, name, source_text, mode, render_settings, status,"
        " created_at, updated_at) VALUES ('p1', 'Старый', 'текст', 'dialogue', '{}', 'raw',"
        " '2026-01-01', '2026-01-01')"
    )
    connection.execute(
        "INSERT INTO replicas (project_id, idx, text, speaker, voice_id, overrides, status)"
        " VALUES ('p1', 0, 'Привет.', 'А', '', '{}', 'pending')"
    )
    connection.commit()
    assert migrations.apply_migrations(connection) == migrations.MIGRATIONS[-1][0]
    # Идемпотентность: повторный прогон ничего не меняет.
    assert migrations.apply_migrations(connection) == migrations.MIGRATIONS[-1][0]
    columns = {row[1] for row in connection.execute("PRAGMA table_info(replicas)")}
    assert {
        "prosody_profile",
        "prosody_intensity",
        "prosody_pace",
        "prosody_confidence",
        "reference_profile_key",
        "reference_fallback_reason",
    } <= columns
    row = connection.execute("SELECT * FROM replicas WHERE project_id = 'p1'").fetchone()
    assert row["text"] == "Привет."
    # Старая реплика читается как «модель ничего не сказала», а не как 0.0.
    assert row["prosody_profile"] == ""
    assert row["prosody_intensity"] is None
    from backend.db.repositories.replicas import row_to_replica

    payload = row_to_replica(row)
    assert payload["prosody_effective"] == emotions.EMOTION_NEUTRAL
    assert payload["prosody_intensity"] is None


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


# --- Phase 9. Просодия доходит до границы движка ------------------------------
def test_recommended_profile_reaches_the_engine_boundary(
    llm_scene_env, fake_store, stub, monkeypatch  # noqa: F811 — импортированные фикстуры
):
    """До резолвера доезжает рекомендация модели, а не эмоция (§24, §51).

    SURPRISE и EXCLAMATION разведены намеренно: если бы границу движка пересекала
    эмоция, резолвер получил бы SURPRISE и откат был бы её откатом. Тест
    проверяет маршрут просодии, а не совпадение двух одинаковых значений.
    """

    def answer(case: dict) -> str:
        def entry(item: dict) -> dict:
            return {
                "schema_version": llm_schemas.SCHEMA_VERSION,
                "replica_id": item["replica_id"],
                "items": [],
                "utterance": {
                    "class": "NORMAL",
                    "context_dependency": "LOW",
                    "emotion": "SURPRISE",
                    "emotion_confidence": 0.7,
                    "dialogue_act": "EXCLAMATION",
                    "prosody": {
                        "profile": "SURPRISE",
                        "recommended_profile": "EXCLAMATION",
                        "intensity": 0.8,
                        "pace": "FAST",
                        "confidence": 0.9,
                    },
                },
            }

        replicas = case.get("replicas")
        if isinstance(replicas, list):
            return json.dumps(
                {
                    "schema_version": llm_schemas.SCHEMA_VERSION,
                    "replicas": [entry(item) for item in replicas],
                },
                ensure_ascii=False,
            )
        return json.dumps(entry(case), ensure_ascii=False)

    llm_scene_env(answer, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as http:
            project = await _project(http, fake_store.id)
            await analyze_project(http, project["id"])
            accepted = await http.post(
                f"/api/projects/{project['id']}/render",
                json={"output_format": "wav", "output_name": "просодия"},
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(http, accepted.json()["job_id"])
            return (await http.get(f"/api/projects/{project['id']}")).json()

    project = asyncio.run(scenario())
    replica = project["replicas"][0]
    # Карточка показывает и рекомендацию, и действующий профиль.
    assert replica["prosody"]["profile"] == "EXCLAMATION"
    assert replica["prosody"]["effective"] == "EXCLAMATION"
    # Эмоция осталась семантической: подменять её рекомендацией нельзя (§10).
    assert replica["emotion_effective"] == "SURPRISE"

    parameters = replica["takes"][-1]["parameters"]
    # Просодия, а не эмоция: иначе здесь стояло бы SURPRISE.
    assert parameters["prosody_effective"] == "EXCLAMATION"
    # Записанного профиля EXCLAMATION у голоса нет: откат назван, а не скрыт (§17).
    assert parameters["prosody_resolved"] == emotions.EMOTION_NEUTRAL
    assert parameters["reference_fallback_used"] is True


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
