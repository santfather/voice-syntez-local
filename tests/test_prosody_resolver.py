"""ProsodyResolver: детерминированный выбор референса под интонацию (UPDATE 3 §23–§28).

Резолвер — единственное место, которое отвечает на вопрос «какой файл и какая
расшифровка уйдут в движок». Тесты идут по четырём границам, которые легко
перепутать:

* **порядок решений** — явный профиль → цепочка отката → NEUTRAL, без случайных
  выборов (§25);
* **таблица отката** — централизованная, конфигурируемая и заканчивается NEUTRAL,
  иначе синтез остался бы без референса (§26);
* **годность профиля** — качество и совместимость с движком проверяются до выбора
  (§22, §28); нейтральный референс голоса движком не ограничен;
* **один голос** — откат не выходит за пределы записи реплики (§27).

Модели не поднимаются: движок не нужен вовсе — резолвер работает с файлами.
"""

import soundfile as sf
from conftest import sine

from backend import config, emotions
from backend.engines.base import ENGINE_F5, ENGINE_XTTS, SAMPLE_RATE
from backend.reference_resolver import (
    FALLBACK_CHAINS,
    FALLBACK_MODE_CHAIN,
    FALLBACK_MODE_NEUTRAL_ONLY,
    ReferenceUnavailableError,
    engine_compatible,
    fallback_chain,
    fallback_mode,
    resolve_prosody,
    resolve_reference,
)
from backend.voices_store import ReferenceProfile, Voice

# Файлы референсов: резолвер проверяет их наличие, поэтому «профиль есть» и
# «профиль пригоден» — разные состояния, и на диске они выглядят по-разному.
FILES = ("voice-a.wav", "voice-b.wav", "calm.wav", "neutral-question.wav", "bad.wav")


def _files(workspace) -> None:
    for name in FILES:
        sf.write(workspace / "voices" / name, sine(0.4, 180.0), SAMPLE_RATE)


def _voice(profiles=None, **overrides) -> Voice:
    voice = Voice(
        id="voice-a",
        name="Мария",
        gender="female",
        ref_text="Привет, это тест",
        audio_file="voice-a.wav",
        engine=ENGINE_F5,
    )
    voice.profiles = list(profiles or [])
    for key, value in overrides.items():
        setattr(voice, key, value)
    return voice


def _profile(profile_id: str, emotion: str, audio: str, **overrides) -> ReferenceProfile:
    # Профиль из этой фабрики — уже подтверждённый (§35): иначе он не попал бы в
    # автоматический выбор, и тесты про порядок решений проверяли бы гейт, а не
    # порядок. Сам гейт проверяется отдельно — на профиле без флага.
    fields = {
        "id": profile_id,
        "emotion": emotion,
        "audio_file": audio,
        "ref_text": "Ты уверен?",
        "quality_status": "ok",
        "enabled_for_auto": True,
    }
    fields.update(overrides)
    return ReferenceProfile(**fields)


# --- §26. Таблица отката ------------------------------------------------------
def test_fallback_table_covers_every_profile_and_ends_with_neutral():
    """Таблица отката централизована и не оставляет интонацию без референса (§26)."""
    # Каждая записываемая интонация обязана иметь цепочку: иначе откат пришлось бы
    # выводить на месте, и он разошёлся бы между вызовами.
    for emotion in emotions.PROFILE_KEYS:
        chain = FALLBACK_CHAINS[emotion]
        assert chain[0] == emotion
        assert chain[-1] == emotions.EMOTION_NEUTRAL
        assert len(set(chain)) == len(chain)
        # Звенья — только записываемые профили: под SURPRISE и FEAR записи нет, и
        # цепочка, начинающаяся с них, обещала бы файл, которого не существует.
        assert set(chain) <= set(emotions.PROFILE_KEYS)
    # Семантические эмоции тоже разрешаются: испуг и удивление узнаются моделью, а
    # референс для них берётся ближайший записываемый — с явным откатом.
    for emotion in (emotions.EMOTION_SURPRISE, emotions.EMOTION_FEAR):
        chain = FALLBACK_CHAINS[emotion]
        assert chain[0] in emotions.PROFILE_KEYS
        assert chain[-1] == emotions.EMOTION_NEUTRAL


def test_fallback_is_configurable_and_conservative_by_default():
    """До benchmark откат идёт сразу в NEUTRAL, цепочка включается настройкой (§26)."""
    # По умолчанию — консервативный производственный режим: «соседний профиль»
    # звучит похоже, но это не подтверждено, и подменять им запись нельзя.
    assert config.PROSODY_FALLBACK_MODE == FALLBACK_MODE_NEUTRAL_ONLY
    assert fallback_chain("QUESTION") == ("QUESTION", emotions.EMOTION_NEUTRAL)
    # Включаемый режим отдаёт полную цепочку из §26.
    assert fallback_chain("QUESTION", mode=FALLBACK_MODE_CHAIN) == (
        "QUESTION",
        "NEUTRAL_QUESTION",
        emotions.EMOTION_NEUTRAL,
    )
    # Неизвестный режим не расширяет выбор молча: он трактуется как консервативный.
    assert fallback_mode("какой-то режим") == FALLBACK_MODE_NEUTRAL_ONLY


def test_unknown_emotion_gets_a_safe_chain():
    """Выдуманная эмоция не создаёт новое звено: она решается как NEUTRAL."""
    assert fallback_chain("ЯРОСТЬ") == (emotions.EMOTION_NEUTRAL,)
    assert fallback_chain("", mode=FALLBACK_MODE_CHAIN) == (emotions.EMOTION_NEUTRAL,)


# --- §25. Порядок решений -----------------------------------------------------
def test_exact_profile_is_used_without_fallback(workspace):
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "calm.wav")])
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM)
    assert resolved.profile_id == "c1"
    assert resolved.fallback_used is False
    assert resolved.resolved_emotion == emotions.EMOTION_CALM


def test_missing_profile_falls_back_to_neutral_conservatively(workspace):
    """Нет точного профиля — берётся нейтральный, и откат назван причиной (§26)."""
    _files(workspace)
    voice = _voice(
        [
            _profile("c1", emotions.EMOTION_CALM, "calm.wav"),
            _profile("q1", "NEUTRAL_QUESTION", "neutral-question.wav"),
        ]
    )
    resolved = resolve_reference(voice, ENGINE_F5, "QUESTION")
    assert resolved.resolved_emotion == emotions.EMOTION_NEUTRAL
    assert resolved.fallback_used is True
    assert resolved.fallback_reason
    assert "QUESTION" in resolved.fallback_reason


def test_chain_mode_prefers_the_nearest_profile(workspace):
    """С включённой цепочкой откат идёт на ближайшую интонацию, а не в NEUTRAL."""
    _files(workspace)
    voice = _voice([_profile("q1", "NEUTRAL_QUESTION", "neutral-question.wav")])
    resolved = resolve_reference(
        voice, ENGINE_F5, "QUESTION", fallback=FALLBACK_MODE_CHAIN
    )
    assert resolved.resolved_emotion == "NEUTRAL_QUESTION"
    assert resolved.profile_id == "q1"
    assert resolved.fallback_used is True


def test_explicit_profile_is_checked_before_the_chain(workspace):
    """Явный выбор пользователя сильнее интонации, но обязан принадлежать голосу."""
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "calm.wav")])
    resolved = resolve_reference(
        voice, ENGINE_F5, emotions.EMOTION_CALM, profile_id="c1"
    )
    assert resolved.profile_id == "c1"
    assert resolved.reason == "выбран явно"
    # Чужой профиль не подставляется: резолвер возвращается к обычному порядку.
    assert (
        resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM, profile_id="чужой").profile_id
        == "c1"
    )


# --- §22, §28. Годность профиля ----------------------------------------------
def test_warning_profile_is_skipped_in_favour_of_neutral(workspace):
    """Референс с расходящейся расшифровкой не предпочитается нейтральному (§22)."""
    _files(workspace)
    voice = _voice(
        [
            _profile(
                "bad",
                emotions.EMOTION_CALM,
                "bad.wav",
                quality_status="warning",
                quality_note="расшифровка не совпала с записью",
            )
        ]
    )
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM)
    assert resolved.resolved_emotion == emotions.EMOTION_NEUTRAL
    assert resolved.fallback_used is True
    assert "расшифровка" in resolved.fallback_reason


def test_disabled_profile_is_not_chosen(workspace):
    """Выключенный профиль остаётся в голосе, но автоматика его не берёт (§35)."""
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "calm.wav", enabled=False)])
    assert resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM).resolved_emotion == (
        emotions.EMOTION_NEUTRAL
    )


def test_unconfirmed_profile_waits_for_the_benchmark(workspace):
    """Неподтверждённый профиль автоматика не берёт: сначала benchmark и прослушивание (§35)."""
    _files(workspace)
    voice = _voice(
        [_profile("c1", emotions.EMOTION_CALM, "calm.wav", enabled_for_auto=False)]
    )
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM)
    # Профиль есть и пригоден, но в автоматику не идёт — синтез не ломается,
    # а честно откатывается на сам голос.
    assert resolved.resolved_emotion == emotions.EMOTION_NEUTRAL
    assert resolved.fallback_used is True
    assert "подтвержд" in resolved.fallback_reason
    assert resolved.profile_id.endswith("-neutral")


def test_unconfirmed_profile_is_still_available_by_explicit_choice(workspace):
    """Ручной выбор флагом не ограничен: иначе профиль нельзя и прослушать (§35)."""
    _files(workspace)
    voice = _voice(
        [_profile("c1", emotions.EMOTION_CALM, "calm.wav", enabled_for_auto=False)]
    )
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM, profile_id="c1")
    assert resolved.profile_id == "c1"
    assert resolved.reason == "выбран явно"
    assert resolved.resolved_emotion == emotions.EMOTION_CALM


def test_missing_audio_file_is_skipped(workspace):
    """Профиль без файла на диске не годится: в движок уйдёт существующий файл."""
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "нет-файла.wav")])
    assert resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM).resolved_emotion == (
        emotions.EMOTION_NEUTRAL
    )


def test_engine_compatibility_is_respected(workspace):
    """Профиль, проверенный только на XTTS, не берётся движком F5 (§28)."""
    _files(workspace)
    voice = _voice(
        [
            _profile(
                "c1", emotions.EMOTION_CALM, "calm.wav", engine_compatibility=[ENGINE_XTTS]
            )
        ]
    )
    on_f5 = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM)
    assert on_f5.resolved_emotion == emotions.EMOTION_NEUTRAL
    assert "f5" in on_f5.fallback_reason
    # На своём движке тот же профиль годится.
    on_xtts = resolve_reference(voice, ENGINE_XTTS, emotions.EMOTION_CALM)
    assert on_xtts.profile_id == "c1"


def test_voice_reference_itself_is_not_limited_by_engine(workspace):
    """Основной референс голоса — это сам голос, а не «профиль под движок» (§28)."""
    _files(workspace)
    voice = _voice(engine=ENGINE_XTTS)
    resolved = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_NEUTRAL)
    assert resolved.profile_id.endswith("-neutral")
    assert resolved.audio_path.name == "voice-a.wav"


def test_empty_compatibility_means_any_engine(workspace):
    """Старый или загруженный руками профиль без поля совместимости не теряется."""
    _files(workspace)
    profile = _profile("c1", emotions.EMOTION_CALM, "calm.wav", engine_compatibility=[])
    assert engine_compatible(profile, ENGINE_F5) is True
    assert engine_compatible(profile, "") is True


# --- §27. Один голос ---------------------------------------------------------
def test_fallback_never_leaves_the_voice(workspace):
    """Откат ищет интонацию только внутри записи этого голоса (§27)."""
    _files(workspace)
    first = _voice([_profile("q1", "NEUTRAL_QUESTION", "neutral-question.wav")])
    second = Voice(
        id="voice-b",
        name="Артём",
        gender="male",
        ref_text="Привет",
        audio_file="voice-b.wav",
        engine=ENGINE_F5,
    )
    second.profiles = [_profile("q2", "NEUTRAL_QUESTION", "neutral-question.wav")]
    # У первого голоса есть ближайший профиль — он его и получает.
    assert (
        resolve_reference(first, ENGINE_F5, "QUESTION", fallback=FALLBACK_MODE_CHAIN).profile_id
        == "q1"
    )
    # Второй голос не видит чужую запись и берёт свою.
    assert (
        resolve_reference(second, ENGINE_F5, "QUESTION", fallback=FALLBACK_MODE_CHAIN).profile_id
        == "q2"
    )
    # Голос без профилей вообще откатывается на собственный нейтральный референс.
    third = Voice(
        id="voice-c",
        name="Иван",
        gender="male",
        ref_text="Привет",
        audio_file="voice-b.wav",
        engine=ENGINE_F5,
    )
    resolved = resolve_reference(third, ENGINE_F5, "QUESTION", fallback=FALLBACK_MODE_CHAIN)
    assert resolved.voice_id == "voice-c"
    assert resolved.profile_id == "voice-c-neutral"
    assert resolved.profile_id not in ("q1", "q2")


def test_voice_without_any_reference_is_a_named_error(workspace):
    """Синтез без референса невозможен, и причина называется, а не молчит."""
    voice = _voice(audio_file="нет-файла.wav")
    voice.profiles = []
    for mode in (FALLBACK_MODE_NEUTRAL_ONLY, FALLBACK_MODE_CHAIN):
        try:
            resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM, fallback=mode)
        except ReferenceUnavailableError as exc:
            assert "референс" in str(exc)
        else:  # pragma: no cover — ошибка обязана подниматься
            raise AssertionError(f"режим {mode}: ожидалась ошибка")


# --- §23, §24. Детерминизм и диагностика --------------------------------------
def test_resolver_is_deterministic(workspace):
    """Один и тот же запрос даёт один и тот же файл — сколько бы раз его ни задать."""
    _files(workspace)
    voice = _voice(
        [
            _profile("c1", emotions.EMOTION_CALM, "calm.wav"),
            _profile(
                "c2", emotions.EMOTION_CALM, "neutral-question.wav", is_default=True
            ),
        ]
    )
    calls = [
        resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM, fallback=FALLBACK_MODE_CHAIN)
        for _ in range(5)
    ]
    assert len({item.profile_id for item in calls}) == 1
    # Профиль по умолчанию голоса предпочитается более раннему: выбор объясним.
    assert calls[0].profile_id == "c2"


def test_resolve_prosody_returns_diagnostics_without_changing_the_choice(workspace):
    """`resolve_prosody` — та же маршрутизация плюс контекст сцены для диагностики."""
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "calm.wav")])
    plain = resolve_reference(voice, ENGINE_F5, emotions.EMOTION_CALM)
    enriched = resolve_prosody(
        voice, ENGINE_F5, emotions.EMOTION_CALM, "STATEMENT", 0.4
    )
    assert enriched.profile_id == plain.profile_id
    assert enriched.dialogue_act == "STATEMENT"
    assert enriched.intensity == 0.4
    # Диагностика не подменяет решение: без контекста результат тот же самый.
    assert resolve_prosody(voice, ENGINE_F5, emotions.EMOTION_CALM).to_dict() == plain.to_dict()


def test_result_uses_resolved_prosody_field_names(workspace):
    """`to_dict` отдаёт имена §24 — второй сущности для тех же данных нет."""
    _files(workspace)
    voice = _voice([_profile("c1", emotions.EMOTION_CALM, "calm.wav")])
    data = resolve_prosody(voice, ENGINE_F5, emotions.EMOTION_CALM, "STATEMENT", 0.25).to_dict()
    assert data["requested_profile"] == emotions.EMOTION_CALM
    assert data["resolved_profile"] == emotions.EMOTION_CALM
    assert data["reference_profile_id"] == "c1"
    assert data["reference_audio"] == "calm.wav"
    assert data["reference_text"]
    assert data["reference_fallback_used"] is False
    assert data["reference_fallback_reason"] == ""
    assert data["voice_id"] == "voice-a"
    assert data["engine_id"] == ENGINE_F5
    # Путь в движок резолвер не отдаёт наружу: он остаётся деталью хранения.
    assert "audio_path" not in data
