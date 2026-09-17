"""Предложения для словаря: источники кандидатов, фильтры и поток подтверждения.

Тесты повторяют критерий готовности из плана: неоднозначное слово показывается в
панели предложений; после подтверждения правило появляется в словаре как обычное;
повторный прогон **другого** текста с тем же словом его больше не предлагает;
отклонение создаёт выключенное правило и не влияет на синтез.

Модели не поднимаются: RUAccent заменён предсказуемой разметкой (или заглушкой
акцентизатора там, где нужен настоящий `score`), список омографов подменён
небольшим словарём, база уведена в `tmp_path` фикстурой `workspace`.
"""

import asyncio
import contextlib

import httpx
import pytest

from backend import audio_pipeline, main
from backend import pronunciation_suggest as suggestions
from backend.engines import registry
from backend.engines.base import ENGINE_F5, ENGINE_XTTS
from backend.pronunciation import get_store
from backend.text_normalization import PronunciationRule, normalize

# Небольшой управляемый список омографов ударения вместо настоящего словаря
# ruaccent: тест не должен зависеть от установленной версии библиотеки.
OMOGRAPHS = {"замок": ["з+амок", "зам+ок"], "звонок": ["зв+онок", "звон+ок"]}


@pytest.fixture
def fake_accent(monkeypatch):
    """Предсказуемый «RUAccent»: размечает «замок» так, будто выбрал первый вариант."""

    def accent(text: str) -> str:
        return text.replace("замок", "з+амок")

    monkeypatch.setattr(audio_pipeline, "accentuate", accent)


@pytest.fixture
def no_accent(monkeypatch):
    """«RUAccent недоступен»: текст возвращается как есть, без «+»."""
    monkeypatch.setattr(audio_pipeline, "accentuate", lambda text: text)


@pytest.fixture
def fake_omographs(monkeypatch):
    monkeypatch.setattr(suggestions, "load_omographs", lambda: dict(OMOGRAPHS))


@contextlib.asynccontextmanager
async def _client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


async def _candidates(client, text: str, engine: str | None = None) -> dict:
    payload: dict = {"text": text}
    if engine:
        payload["engine"] = engine
    response = await client.post("/api/pronunciation/suggestions", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _words(report: dict) -> list[str]:
    return [item["word"] for item in report["candidates"]]


# --- источники кандидатов -----------------------------------------------------
def test_finds_ambiguous_yo_word(workspace, fake_accent, fake_omographs):
    report = suggestions.build_suggestions("Мы все пойдем домой.", supports_accents=True)
    candidate = next(item for item in report.candidates if item.word == "все")
    assert candidate.kind == "yo_homograph"
    assert candidate.target == "всё"
    # Альтернатива — «оставить как есть»: решение принимает пользователь.
    assert candidate.alternatives == ["все"]
    assert candidate.confidence is None  # уверенность не выдумывается
    assert candidate.reason


def test_stress_homograph_offers_ruaccent_choice(workspace, fake_accent, fake_omographs):
    report = suggestions.build_suggestions("Старый замок стоит у реки.", supports_accents=True)
    candidate = next(item for item in report.candidates if item.word == "замок")
    assert candidate.kind == "stress_homograph"
    assert candidate.target == "з+амок"  # именно то, что «выбрал» RUAccent
    assert candidate.alternatives == ["зам+ок"]


def test_stress_homograph_falls_back_to_first_variant(workspace, no_accent, fake_omographs):
    report = suggestions.build_suggestions("Старый замок стоит у реки.", supports_accents=True)
    candidate = next(item for item in report.candidates if item.word == "замок")
    # RUAccent не размечал текст (ударения выключены/недоступен) — берём первый вариант.
    assert candidate.target == "з+амок"


def test_rare_words_go_last_and_have_no_invented_target(workspace, fake_accent, fake_omographs):
    text = "Мы все видим высоконагруженный замок."
    report = suggestions.build_suggestions(text, supports_accents=True)
    kinds = [item.kind for item in report.candidates]
    assert kinds[-1] == "rare"
    rare = report.candidates[-1]
    assert rare.word == "высоконагруженный"
    assert rare.target == ""  # ударение за пользователя не выдумываем
    assert "редкое" in rare.reason


def test_sources_are_ordered_yo_then_stress_then_rare(workspace, fake_accent, fake_omographs):
    text = "Мы все видим высоконагруженный замок."
    report = suggestions.build_suggestions(text, supports_accents=True)
    ranks = {item.word: suggestions._KIND_RANK[item.kind] for item in report.candidates}
    assert ranks["все"] < ranks["замок"] < ranks["высоконагруженный"]


# --- низкая уверенность модели ё-фикации --------------------------------------
class _FakeAccentizer:
    """Подставной акцентизатор: только те методы, которые читает источник."""

    def __init__(self, scores, yo_forms=None, loaded=True):
        self._scores = scores
        self._yo_forms = yo_forms or {}
        self.is_loaded = loaded

    def yo_homograph_scores(self, text):
        return [dict(item) for item in self._scores]

    def yo_form(self, word):
        return self._yo_forms.get(word)


def test_low_confidence_prediction_adds_score(workspace, fake_accent, fake_omographs):
    accentizer = _FakeAccentizer([{"word": "небо", "score": 0.51, "entity": "YO"}])
    report = suggestions.build_suggestions(
        "Посмотри на небо.", supports_accents=True, accentizer=accentizer
    )
    candidate = next(item for item in report.candidates if item.word == "небо")
    assert candidate.confidence == pytest.approx(0.51)
    assert "не уверена" in candidate.reason


def test_low_confidence_word_outside_ambiguous_list_is_its_own_kind(workspace, fake_accent):
    accentizer = _FakeAccentizer(
        [{"word": "тест", "score": 0.4, "entity": "YO"}], {"тест": "тёст"}
    )
    report = suggestions.build_suggestions(
        "Это простой тест.", supports_accents=True, accentizer=accentizer
    )
    candidate = next(item for item in report.candidates if item.word == "тест")
    assert candidate.kind == "yo_low_confidence"
    assert candidate.target == "тёст"


def test_confident_prediction_is_not_a_candidate(workspace, fake_accent, fake_omographs):
    accentizer = _FakeAccentizer([{"word": "небо", "score": 0.99, "entity": "YO"}])
    report = suggestions.build_suggestions(
        "Посмотри на небо.", supports_accents=True, accentizer=accentizer
    )
    candidate = next(item for item in report.candidates if item.word == "небо")
    # Уверенность высокая — источник предложений молчит, остаётся только ё-омограф.
    assert candidate.kind == "yo_homograph"
    assert candidate.confidence is None


def test_unloaded_model_skips_source(workspace, fake_accent, fake_omographs):
    accentizer = _FakeAccentizer([{"word": "небо", "score": 0.1}], loaded=False)
    report = suggestions.build_suggestions(
        "Посмотри на небо.", supports_accents=True, accentizer=accentizer
    )
    candidate = next(item for item in report.candidates if item.word == "небо")
    assert candidate.confidence is None


# --- фильтры ------------------------------------------------------------------
def test_covered_by_enabled_rule_is_not_offered(workspace, fake_accent, fake_omographs):
    rules = [PronunciationRule("все", "всё")]
    report = suggestions.build_suggestions(
        "Мы все пойдем.", rules=rules, supports_accents=True
    )
    assert "все" not in _words(report.to_dict())


def test_covered_by_disabled_rule_is_not_offered(workspace, fake_accent, fake_omographs):
    # Выключенное правило — память об отклонённом предложении: слово не возвращается.
    rules = [PronunciationRule("все", "всё", enabled=False)]
    report = suggestions.build_suggestions(
        "Мы все пойдем.", rules=rules, supports_accents=True
    )
    assert "все" not in _words(report.to_dict())


def test_words_with_stress_markup_and_latin_are_skipped(workspace, fake_accent, fake_omographs):
    text = "все+ замок SQL 1.2.3"
    report = suggestions.build_suggestions(text, supports_accents=True)
    words = _words(report.to_dict())
    assert "замок" in words
    assert "все" not in words  # слово с ручным «+» кандидатом не считается
    assert all(not word.isascii() for word in words)


def test_limit_is_respected(workspace, fake_accent, fake_omographs):
    text = "Мы все видим высоконагруженный замок и звонок."
    report = suggestions.build_suggestions(text, supports_accents=True, limit=1)
    assert len(report.candidates) == 1


def test_considered_counts_words(workspace, fake_accent, fake_omographs):
    report = suggestions.build_suggestions("Мы все пойдем домой.", supports_accents=True)
    assert report.considered == 4


# --- REST API: ничего не пишет, ничего не поднимает ---------------------------
def test_suggestions_endpoint_does_not_write_to_database(workspace, fake_accent, fake_omographs):
    async def scenario():
        async with _client() as client:
            before = (await client.get("/api/pronunciation")).json()["count"]
            report = await _candidates(client, "Мы все пойдем домой.")
            assert "все" in _words(report)
            assert report["considered"] > 0
            after = (await client.get("/api/pronunciation")).json()["count"]
            assert after == before == 0

    _run(scenario)


def test_suggestions_endpoint_does_not_create_engine(workspace, fake_accent, fake_omographs, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("предложения не должны создавать или грузить TTS-движок")

    monkeypatch.setattr(registry, "get_engine", explode)
    monkeypatch.setattr(registry, "_create", explode)
    monkeypatch.setattr(audio_pipeline, "get_engine", explode)
    before = set(registry.created_engines())

    async def scenario():
        async with _client() as client:
            report = await _candidates(client, "Мы все пойдем домой.", engine=ENGINE_XTTS)
            assert "все" in _words(report)
            assert set(registry.created_engines()) == before

    _run(scenario)


def test_empty_and_candidate_free_text_give_empty_list(workspace, fake_accent, fake_omographs):
    async def scenario():
        async with _client() as client:
            empty = await _candidates(client, "   ")
            assert empty == {"candidates": [], "considered": 0}
            # «ой» и «ах» — не кандидаты: короткие, однозначные, не в словаре.
            plain = await _candidates(client, "ой ах")
            assert plain["candidates"] == []
            assert plain["considered"] == 2

    _run(scenario)


def test_suggestions_endpoint_rejects_unknown_engine(workspace, fake_accent, fake_omographs):
    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/pronunciation/suggestions",
                json={"text": "Мы все пойдем.", "engine": "нет-такого"},
            )
            assert response.status_code == 400
            assert "Неизвестный движок" in response.json()["detail"]

    _run(scenario)


# --- поток подтверждения и отклонения -----------------------------------------
def test_confirmation_creates_rule_and_stops_repeating(workspace, fake_accent, fake_omographs):
    """Критерий готовности: подтвердил один раз — на другом тексте уже не предлагают."""

    async def scenario():
        async with _client() as client:
            first = await _candidates(client, "Надо все успеть.")
            assert "все" in _words(first)

            created = await client.post(
                "/api/pronunciation",
                json={"source": "все", "target": "всё", "note": "подтверждено в предложениях"},
            )
            assert created.status_code == 201, created.text

            listing = (await client.get("/api/pronunciation")).json()
            assert [entry["source"] for entry in listing["entries"]] == ["все"]
            assert listing["entries"][0]["enabled"] is True

            second = await _candidates(client, "Мы все придем позже.")
            assert "все" not in _words(second)

    _run(scenario)


def test_rejection_creates_disabled_rule_and_does_not_affect_synthesis(
    workspace, fake_accent, fake_omographs
):
    async def scenario():
        async with _client() as client:
            first = await _candidates(client, "Надо все успеть.")
            assert "все" in _words(first)

            rejected = await client.post(
                "/api/pronunciation",
                json={
                    "source": "все",
                    "target": "всё",
                    "enabled": False,
                    "note": "отклонено в предложениях",
                },
            )
            assert rejected.status_code == 201, rejected.text
            assert rejected.json()["enabled"] is False
            assert rejected.json()["note"] == "отклонено в предложениях"

            # Больше не предлагается — правило осталось памятью об отказе.
            second = await _candidates(client, "Мы все придем позже.")
            assert "все" not in _words(second)

            # Но и на синтез не влияет: правило выключено.
            assert get_store().active_rules() == []
            assert normalize("Мы все видим.", get_store().active_rules()) == "Мы все видим."

    _run(scenario)


def test_engine_parameter_is_accepted(workspace, fake_accent, fake_omographs):
    """Движок влияет только на флаг ударений и не мешает находить кандидатов."""

    async def scenario():
        async with _client() as client:
            f5 = await _candidates(client, "Старый замок.", engine=ENGINE_F5)
            xtts = await _candidates(client, "Старый замок.", engine=ENGINE_XTTS)
            assert "замок" in _words(f5)
            assert "замок" in _words(xtts)

    _run(scenario)
