"""Маршрутизация движка в пайплайне: откат по недоступности и пресетные движки.

Отдельный модуль, потому что проверяется не синтез как таковой, а два решения,
которые пайплайн принимает **до** модели: каким движком синтезировать голос
(`_engine_for`) и нужен ли этому движку референс (`_reference_for`). Оба решения
принимаются по паспорту движка, а не по его идентификатору, поэтому здесь не
поднимается ни одна модель — вместо них обычная заглушка `StubEngine`.

Откат проверяется на Qwen3-TTS: это единственный движок, который его объявил
(`EngineInfo.fallback_engine`). Голос, настроенный на модель, весов которой на
диске нет, обязан дочитать диалог основным движком и сказать об этом в лог —
молчаливая подмена одного звучания другим недопустима.
"""

import asyncio

import pytest
import soundfile as sf
from conftest import STUB_ENGINE_ID, StubEngine, sine

from backend import audio_pipeline, model_manager, reference_resolver
from backend.audio_pipeline import RenderSettings, SpeakerSettings, _engine_for, _reference_for
from backend.dialogue_parser import Replica
from backend.engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    ENGINE_QWEN,
    ENGINE_XTTS,
    SAMPLE_RATE,
    EngineInfo,
)
from backend.voices_store import Voice


def _voice(engine: str, voice_id: str = "voice-qwen") -> Voice:
    """Голос на выбранном движке. Файл референса создаёт фикстура `workspace`."""
    return Voice(
        id=voice_id,
        name="Квен",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="ref.wav",
        engine=engine,
    )


@pytest.fixture
def qwen_voice(workspace, monkeypatch):
    """Голос на Qwen с настоящим файлом референса и своим хранилищем."""
    sf.write(workspace / "voices" / "ref.wav", sine(2.0, 200.0), SAMPLE_RATE)
    voice = _voice(ENGINE_QWEN)

    class Store:
        def get(self, voice_id: str) -> Voice | None:
            return voice if voice_id == voice.id else None

    monkeypatch.setattr(audio_pipeline, "get_store", lambda: Store())
    return voice


@pytest.fixture
def routed_engines(monkeypatch):
    """Реестр движков, отдающий заглушку под тот id, который у него спросили.

    Одна заглушка на все случаи не годится: смысл этих тестов — какой именно
    движок попросил пайплайн. Поэтому здесь и журнал запросов, и разные заглушки
    с настоящими паспортами (от паспорта зависит и откат, и нужен ли референс).
    """
    requested: list[str] = []
    engines: dict[str, StubEngine] = {}

    def fake_get_engine(engine_id: str):
        requested.append(engine_id)
        if engine_id not in engines:

            class _Engine(StubEngine):
                info = ENGINE_INFOS.get(engine_id, ENGINE_INFOS[ENGINE_F5])

            engines[engine_id] = _Engine()
        return engines[engine_id]

    monkeypatch.setattr(audio_pipeline, "get_engine", fake_get_engine)
    return requested, engines


def _replica(text: str = "Привет, это тест") -> Replica:
    return Replica(voice="#1", text=text, line_number=1)


# --- выбор движка: откат по недоступности ------------------------------------
def test_unavailable_engine_falls_back_to_declared_one(
    qwen_voice, routed_engines, monkeypatch, caplog
):
    """Файлов Qwen нет — синтезируем объявленным откатом, а не падаем."""
    requested, engines = routed_engines
    monkeypatch.setattr(
        model_manager, "engine_available", lambda engine_id: engine_id != ENGINE_QWEN
    )

    with caplog.at_level("WARNING"):
        engine = _engine_for(qwen_voice)

    assert requested == [ENGINE_F5]
    assert engine is engines[ENGINE_F5]
    # Откат обязан быть слышен: иначе пользователь решит, что его голос звучит
    # выбранным движком, и будет искать причину в референсе.
    warning = next(item for item in caplog.records if item.levelname == "WARNING")
    assert qwen_voice.name in warning.getMessage()
    assert ENGINE_QWEN in warning.getMessage()
    assert ENGINE_F5 in warning.getMessage()


def test_available_engine_is_used_as_chosen(qwen_voice, routed_engines, monkeypatch):
    """Веса на месте — выбранный движок и работает, никакого отката."""
    requested, engines = routed_engines
    monkeypatch.setattr(model_manager, "engine_available", lambda engine_id: True)

    engine = _engine_for(qwen_voice)

    assert requested == [ENGINE_QWEN]
    assert engine is engines[ENGINE_QWEN]


def test_engine_without_declared_fallback_is_not_substituted(
    workspace, routed_engines, monkeypatch
):
    """У XTTS отката нет — недоступность остаётся его собственной заботой.

    Проверяется именно отсутствие подмены: `engine_available` здесь врёт, что
    модели нет, и если бы пайплайн откатывал по одному лишь признаку
    недоступности, голос молча зазвучал бы чужим движком.
    """
    requested, _ = routed_engines
    monkeypatch.setattr(model_manager, "engine_available", lambda engine_id: False)

    _engine_for(_voice(ENGINE_XTTS, voice_id="voice-xtts"))

    assert requested == [ENGINE_XTTS]


def test_unknown_engine_is_reported_with_voice_name(workspace, monkeypatch):
    """Опечатка в движке — ошибка с именем голоса, а не падение реестра.

    Здесь намеренно настоящий `get_engine`: сообщение «неизвестный движок»
    рождается в реестре, и подменять его заглушкой значило бы проверять не то.
    """
    monkeypatch.setattr(model_manager, "engine_available", lambda engine_id: True)
    voice = _voice("нет-такого", voice_id="voice-typo")

    with pytest.raises(ValueError) as error:
        _engine_for(voice)

    assert voice.name in str(error.value)
    assert "нет-такого" in str(error.value)


FALLBACK_PRESET_ID = "stub-preset-fallback"

FALLBACK_PRESET_INFO = EngineInfo(
    id=FALLBACK_PRESET_ID,
    label="Заглушка-пресет с откатом",
    description="Движок на встроенных голосах, объявивший откат по недоступности.",
    supports_accents=False,
    supports_cloning=False,
    supports_prosody_profiles=False,
    fallback_engine=STUB_ENGINE_ID,
)


def test_fallback_does_not_substitute_a_voice_without_reference(
    workspace, routed_engines, monkeypatch, caplog
):
    """Голос без референса не откатывается на клонирующий движок.

    У карточки встроенного голоса записи нет вовсе, и подмена движка попросила бы
    клонирующую модель говорить по файлу, которого никогда не существовало: вместо
    «скачайте модель» пользователь получил бы «файл референса потерян». Поэтому
    откат здесь не срабатывает вовсе.
    """
    requested, _ = routed_engines
    monkeypatch.setitem(ENGINE_INFOS, FALLBACK_PRESET_ID, FALLBACK_PRESET_INFO)
    monkeypatch.setattr(model_manager, "engine_available", lambda engine_id: False)
    voice = Voice(
        id="voice-builtin",
        name="Света",
        gender="female",
        ref_text="",
        audio_file="",
        engine=FALLBACK_PRESET_ID,
    )

    with caplog.at_level("WARNING"):
        _engine_for(voice)

    assert requested == [FALLBACK_PRESET_ID]
    assert caplog.records == []


# --- референс: нужен ли он движку ---------------------------------------------
PRESET_ENGINE_ID = "stub-preset"

PRESET_INFO = EngineInfo(
    id=PRESET_ENGINE_ID,
    label="Заглушка-пресет",
    description="Движок на встроенных голосах — без клонирования по референсу.",
    supports_accents=False,
    supports_cloning=False,
    supports_prosody_profiles=False,
)


class PresetStubEngine(StubEngine):
    """Движок со встроенными голосами: референс ему не нужен (будущий Kokoro-ru)."""

    info = PRESET_INFO


class QwenStubEngine(StubEngine):
    """Заглушка с паспортом Qwen: проверяется, что референс спрашивают именно у неё."""

    info = ENGINE_INFOS[ENGINE_QWEN]


@pytest.fixture
def preset_engine(monkeypatch):
    """Пресетный движок, объявленный в реестре паспортов.

    Паспорт регистрируется на время теста, потому что `engine_info` считает
    незарегистрированный id опечаткой и отдаёт возможности F5: движок «без
    клонирования» существует для пайплайна только тогда, когда он объявлен.
    """
    monkeypatch.setitem(ENGINE_INFOS, PRESET_ENGINE_ID, PRESET_INFO)
    return PresetStubEngine()


def test_preset_engine_does_not_ask_the_resolver_for_reference(preset_engine, monkeypatch):
    """У движка без клонирования референса нет — резолвер не вызывается вовсе.

    Резолвер, у которого нет записи, поднимает `ReferenceUnavailableError`.
    Спрашивать его — значит останавливать синтез на движке, которому ничего не
    мешало работать, поэтому отказ здесь жёсткий: вызов считается ошибкой теста.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("резолвер профилей не должен вызываться для пресетного движка")

    monkeypatch.setattr(reference_resolver, "resolve_prosody", forbidden)
    voice = _voice(PRESET_ENGINE_ID)

    assert _reference_for(voice, preset_engine, _replica()) is None


def test_cloning_engine_resolves_reference_for_its_engine_id(workspace, monkeypatch):
    """У движка с клонированием референс выбирается под его идентификатор."""
    seen: list[tuple] = []

    def fake_resolve(voice, engine_id, prosody_effective, **kwargs):
        seen.append((voice.id, engine_id, prosody_effective, kwargs.get("profile_id")))
        return "ссылка"

    monkeypatch.setattr(reference_resolver, "resolve_prosody", fake_resolve)
    voice = _voice(ENGINE_QWEN)

    assert _reference_for(voice, QwenStubEngine(), _replica()) == "ссылка"
    assert seen == [(voice.id, ENGINE_QWEN, "NEUTRAL", "")]


def test_replica_reference_choice_is_passed_through(workspace, monkeypatch):
    """Явный профиль реплики доезжает до резолвера, а не теряется по пути."""
    seen: list[dict] = []

    def fake_resolve(voice, engine_id, prosody_effective, **kwargs):
        seen.append({"prosody": prosody_effective, "profile_id": kwargs.get("profile_id")})
        return "ссылка"

    monkeypatch.setattr(reference_resolver, "resolve_prosody", fake_resolve)
    replica = Replica(
        voice="#1", text="Привет", line_number=1, reference_profile_id="profile-7"
    )

    _reference_for(_voice(ENGINE_QWEN), QwenStubEngine(), replica)

    assert seen == [{"prosody": "NEUTRAL", "profile_id": "profile-7"}]


# --- сквозной рендер -----------------------------------------------------------
def test_render_with_missing_engine_weights_still_produces_audio(
    qwen_voice, routed_engines, monkeypatch
):
    """Диалог на голосе без скачанных весов дочитывается, а не обрывается.

    Это и есть смысл отката: проект обязан довести длинный диалог до файла,
    даже если модель ещё не скачана, — иначе пользователь узнаёт о
    недоступности на середине рендера.
    """
    requested, engines = routed_engines
    monkeypatch.setattr(
        model_manager, "engine_available", lambda engine_id: engine_id != ENGINE_QWEN
    )

    result = asyncio.run(
        audio_pipeline.render_dialogue(
            job_id="job",
            replicas=[_replica("Первая реплика"), _replica("Вторая реплика")],
            speakers={"#1": SpeakerSettings(voice_id=qwen_voice.id)},
            settings=RenderSettings(pause_ms=0, output_format="wav"),
        )
    )

    assert result.replicas_done == 2
    assert result.output_path.exists()
    assert requested == [ENGINE_F5]  # Qwen не поднимался ни разу
    assert engines[ENGINE_F5].loads == 1  # и поднят один раз на весь диалог
    audio = sf.read(result.output_path, dtype="float32")[0]
    assert audio.size > 0


def test_stub_engine_ids_stay_untouched_by_routing(workspace, routed_engines):
    """Голос на движке без моделей в реестре проходит маршрут без отката.

    Заглушка из `conftest` не имеет ни одной `ModelSpec`, поэтому `engine_available`
    для неё честно отвечает «доступен» — на этом стоит вся остальная тестовая
    сюита, и проверять это здесь стоит хотя бы один раз явно.
    """
    requested, _ = routed_engines

    _engine_for(_voice(STUB_ENGINE_ID, voice_id="voice-stub"))

    assert requested == [STUB_ENGINE_ID]
