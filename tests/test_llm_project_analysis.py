"""LLM-анализ в подготовке проекта (Task 2, фазы 5 и 7).

Проверяется главное обещание задачи в терминах реального API: анализ **добавляет
предложения**, но не меняет ни исходный текст, ни `final_text`; при выключенном
Analyzer'е детерминированный pipeline работает ровно как раньше; при отказе (память,
Ollama) это видно в подстатусе, а не превращается в «тихий» успех; повторный проход
не гоняет модель заново, а изменение текста — гоняет.

Модель не поднимается: клиент Ollama подставной, датчик памяти — заглушка, движки
синтеза — `StubEngine` из conftest.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import (  # noqa: F401 — sine нужен фикстуре voices
    StubEngine,
    analyze_project,
    sine,
)
from test_projects_api import _client, _create_project
from test_text_preview_api import F5_VOICE

from backend.llm import analyzer as llm
from backend.llm import integration as llm_integration
from backend.llm import memory_policy as memory
from backend.llm import scheduler as scheduler_module
from backend.llm import schemas as s
from backend.llm.fake_client import FakeOllamaClient
from backend.llm.ollama_client import OllamaModel

DIALOGUE = (
    "ИВАН: Мы гуляли вокруг старого замка на холме.\n"
    "МАРГО: Он поменял замок на входной двери."
)
MODEL = OllamaModel(tag="qwen3:8b", digest="digest-primary", size_bytes=5 * 1024**3)
GREEN = {"system_percent": 45.0, "available_gb": 12.0, "pressure": "green"}


def _run(scenario) -> None:
    asyncio.run(scenario())


def _answer(case: dict, *, meaning: str = "строение, крепость") -> str:
    """Ответ «модели»: омограф в целевой реплике с правильным словом и границами."""
    text = case["target_text"]
    word = "замка" if "замка" in text else ("замок" if "замок" in text else "")
    items = []
    if word:
        start = text.index(word)
        items.append(
            {
                "span_start": start,
                "span_end": start + len(word),
                "source": word,
                "type": "homograph",
                "meaning": meaning,
                "suggested_form": "замок",
                "confidence": 0.9,
                "needs_review": False,
            }
        )
    return json.dumps(
        {
            "schema_version": s.SCHEMA_VERSION,
            "replica_id": case["replica_id"],
            "items": items,
            "utterance": {"class": "NORMAL", "context_dependency": "LOW"},
        },
        ensure_ascii=False,
    )


@pytest.fixture
def llm_env(monkeypatch):
    """Включает Analyzer с подставным клиентом и подставным датчиком памяти."""

    def install(
        responder=None,
        *,
        enabled: bool = True,
        required_for_render: bool = False,
        sensor: dict | None = None,
        context_replicas: int = 2,
    ) -> FakeOllamaClient:
        client = FakeOllamaClient(models=[MODEL], responder=responder or _answer, capabilities={})
        settings = llm.AnalyzerSettings(
            enabled=enabled,
            required_for_render=required_for_render,
            context_replicas=context_replicas,
        )
        analyzer = llm.LinguisticAnalyzer(settings=settings, client=client)
        monkeypatch.setattr(llm, "_analyzer", analyzer)
        scheduler = scheduler_module.HeavyScheduler(
            gate=memory.HeavyGate(), sensor=lambda: dict(sensor or GREEN)
        )
        monkeypatch.setattr(scheduler_module, "_scheduler", scheduler)
        return client

    yield install
    # Singleton'ы сбрасываются: иначе подставной анализатор утёк бы в другие тесты.
    llm.reset_analyzer()
    scheduler_module.reset_scheduler()


async def _project_with_voices(client, text: str = DIALOGUE) -> dict:
    project = await _create_project(client, text=text)
    speakers = {
        replica["speaker"]: F5_VOICE
        for replica in (await client.post(f"/api/projects/{project['id']}/parse", json={})).json()[
            "replicas"
        ]
    }
    patched = await client.patch(
        f"/api/projects/{project['id']}",
        json={"speakers": {key: {"voice_id": value} for key, value in speakers.items()}},
    )
    assert patched.status_code == 200, patched.text
    return patched.json()


def test_llm_disabled_keeps_existing_pipeline_working(
    llm_env, voices, stub, monkeypatch
):
    """Выключенный Analyzer: подготовка как раньше, LLM-кандидатов нет."""
    llm_env(enabled=False)

    async def scenario() -> dict:
        async with _client(monkeypatch) as client:
            project = await _project_with_voices(client)
            response = await client.post(f"/api/projects/{project['id']}/analyze", json={})
            assert response.status_code == 200, response.text
            state = (
                await client.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            full = (await client.get(f"/api/projects/{project['id']}")).json()
            return {"state": state, "full": full}

    result = asyncio.run(scenario())
    assert result["state"]["status"] == llm.STATUS_DISABLED
    assert result["state"]["candidates_total"] == 0
    assert result["full"]["llm_analysis_status"] == llm.STATUS_DISABLED
    assert all(row["analysis_status"] == "done" for row in result["full"]["replicas"])
    assert all(row["final_text"] for row in result["full"]["replicas"]), (
        "детерминированный текст на месте"
    )


def test_llm_analysis_adds_candidates_without_changing_text(
    llm_env, voices, stub, monkeypatch
):
    """Анализ добавляет кандидатов, но `final_text` и исходный текст не меняются."""

    async def collect(enabled: bool) -> dict:
        client = llm_env(enabled=enabled)

        async def scenario():
            async with _client(monkeypatch) as api:
                project = await _project_with_voices(api)
                assert (
                    await api.post(f"/api/projects/{project['id']}/analyze", json={})
                ).status_code == 200
                full = (await api.get(f"/api/projects/{project['id']}")).json()
                state = (
                    await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
                ).json()
                return {
                    "replicas": [
                        (row["index"], row["text"], row["final_text"]) for row in full["replicas"]
                    ],
                    "state": state,
                    "calls": len(client.calls),
                }

        return await scenario()

    with_llm = asyncio.run(collect(True))
    without_llm = asyncio.run(collect(False))

    # Текст не изменился ни в источнике, ни в подготовке — это главное требование.
    assert with_llm["replicas"] == without_llm["replicas"]
    assert with_llm["calls"] > 0, "модель должна была получить реплики"
    assert without_llm["calls"] == 0

    state = with_llm["state"]
    assert state["status"] in {llm.STATUS_READY, llm.STATUS_NEEDS_REVIEW}
    assert state["model"] == "qwen3:8b"
    assert state["candidates_total"] > 0
    candidate = state["candidates"][0]
    assert candidate["source"] == llm_integration.LLM_SOURCE
    assert candidate["type"] == "homograph"
    assert candidate["meaning"] == "строение, крепость"
    assert candidate["agreement"] in {
        llm_integration.AGREEMENT_AGREE,
        llm_integration.AGREEMENT_CONFLICT,
        llm_integration.AGREEMENT_NONE,
    }


def test_llm_analysis_is_cached_between_runs(llm_env, voices, stub, monkeypatch):
    """Повторный анализ того же текста не гоняет модель заново."""
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            first_calls = len(client.calls)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return {"first": first_calls, "second": len(client.calls)}

    calls = asyncio.run(scenario())
    assert calls["first"] > 0
    assert calls["second"] == calls["first"], "кеш обязан вернуть разбор без модели"


def test_source_change_invalidates_and_reanalyzes(llm_env, voices, stub, monkeypatch):
    """Правка текста реплики заставляет модель прочитать её заново."""
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            first = len(client.calls)
            replicas = (await api.get(f"/api/projects/{project['id']}")).json()["replicas"]
            await api.patch(
                f"/api/projects/{project['id']}/replicas/0",
                json={"text": replicas[0]["text"] + " И ещё немного текста."},
            )
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return {"first": first, "second": len(client.calls)}

    calls = asyncio.run(scenario())
    assert calls["second"] > calls["first"], "изменённая реплика должна анализироваться снова"


def test_memory_warning_is_visible_and_does_not_break_preparation(
    llm_env, voices, stub, monkeypatch
):
    """Память не даёт начать анализ: подстатус FAILED, детерминированный текст готов."""
    client = llm_env(sensor={"system_percent": 78.0, "available_gb": 9.0, "pressure": "green"})

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            response = await api.post(f"/api/projects/{project['id']}/analyze", json={})
            assert response.status_code == 200, response.text
            state = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            full = (await api.get(f"/api/projects/{project['id']}")).json()
            return {"state": state, "full": full}

    result = asyncio.run(scenario())
    assert client.calls == [], "при WARNING модель не должна вызываться"
    assert result["state"]["status"] == llm.STATUS_FAILED
    assert "память" in result["state"]["error"] or "WARNING" in result["state"]["error"]
    assert result["state"]["candidates_total"] == 0
    assert all(row["final_text"] for row in result["full"]["replicas"])


def test_render_blocked_when_required_analysis_failed(
    llm_env, voices, stub, monkeypatch
):
    """Режим «LLM обязателен»: без успешного анализа рендер не запускается."""
    llm_env(sensor={"system_percent": 95.0, "available_gb": 1.0, "pressure": "red"})

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (await api.post(f"/api/projects/{project['id']}/render", json={})).json()

    # Флаг обязательности ставим в настройках анализатора после установки окружения.
    analyzer = llm.get_analyzer()
    llm._analyzer = llm.LinguisticAnalyzer(
        settings=llm.AnalyzerSettings(enabled=True, required_for_render=True),
        client=analyzer.client,
    )
    result = asyncio.run(scenario())
    assert "detail" in result
    assert "обязателен" in result["detail"]
    assert "FAILED" in result["detail"]


def test_linguistic_analysis_endpoint_runs_canonical_pass(llm_env, voices, stub, monkeypatch):
    """`POST .../linguistic-analysis` — тот же проход, что `/analyze`, плюс состояние LLM."""
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            response = await api.post(
                f"/api/projects/{project['id']}/linguistic-analysis", json={}
            )
            assert response.status_code == 200, response.text
            return response.json()

    payload = asyncio.run(scenario())
    assert payload["llm"]["status"] in {llm.STATUS_READY, llm.STATUS_NEEDS_REVIEW}
    assert payload["llm"]["candidates_total"] > 0
    # Тот же канонический проход, что и `/analyze`: слова с омографами дают
    # детерминированные кандидаты, поэтому подготовка ждёт review, а не «ready».
    assert payload["status"] == "needs_review"
    assert len(client.calls) > 0


def test_deterministic_agreement_and_conflict():
    """Сверка с детерминированным слоем: согласие, конфликт и «своего мнения нет»."""
    from dataclasses import dataclass

    @dataclass
    class _Prep:
        source_text: str = "Он поменял замок на входной двери."
        yo_text: str = ""
        dictionary_matches: list = None  # type: ignore[assignment]

    note = s.Annotation(
        span_start=11,
        span_end=16,
        source="замок",
        type=s.TYPE_HOMOGRAPH,
        meaning="запор",
        suggested_form="замо́к",
        confidence=0.9,
    )

    # Словарь даёт то же чтение — согласие.
    agree = _Prep(dictionary_matches=[{"source": "замок", "target": "замо́к"}])
    assert llm_integration.deterministic_opinion(note, agree)[0] == llm_integration.AGREEMENT_AGREE

    # Словарь даёт другое чтение — конфликт, и он должен уйти в review.
    conflict = _Prep(dictionary_matches=[{"source": "замок", "target": "за́мок"}])
    assert (
        llm_integration.deterministic_opinion(note, conflict)[0]
        == llm_integration.AGREEMENT_CONFLICT
    )

    # Словарь слова не знает — «своего мнения нет», а не согласие.
    silent = _Prep(dictionary_matches=[])
    assert llm_integration.deterministic_opinion(note, silent)[0] == llm_integration.AGREEMENT_NONE


def test_llm_candidates_include_dropped_annotations_as_review(llm_env, voices, stub, monkeypatch):
    """Отброшенное предложение модели видно в review, а не исчезает молча."""
    client = llm_env(
        responder=lambda case: json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [
                    {
                        "span_start": 0,
                        "span_end": 5,
                        "source": "выдумка",
                        "type": "term",
                        "confidence": 0.9,
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())
    assert len(client.calls) > 0
    assert state["status"] == llm.STATUS_NEEDS_REVIEW
    dropped = [item for item in state["candidates"] if item["kind"].endswith("_dropped")]
    assert dropped, "отброшенная аннотация должна попасть в кандидатов"
    assert dropped[0]["reason_code"] == "source_not_found"
    assert dropped[0]["needs_review"] is True


def test_analyze_keeps_working_when_llm_module_disabled(llm_env, voices, stub, monkeypatch):
    """Классический путь `/analyze` без LLM остаётся полностью рабочим (регрессия)."""
    llm_env(enabled=False)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (await api.get(f"/api/projects/{project['id']}/analysis")).json()

    state = asyncio.run(scenario())
    # Детерминированный слой по-прежнему находит свои кандидаты («замок» в тексте):
    # выключенная LLM ничего не ломает и ничего не подменяет.
    assert state["status"] == "needs_review"
    assert state["candidates_total"] > 0
    assert state["llm_status"] == llm.STATUS_DISABLED
    assert state["llm_candidates_total"] == 0
    assert state["candidates_total"] >= 0


def test_llm_candidate_review_creates_project_rule_and_reanalyzes(
    llm_env, voices, stub, monkeypatch
):
    """Принятие предложения LLM создаёт правило проекта и пересчитывает реплики.

    Это единственный способ, которым предложение модели попадает в словарь: сама
    модель писать в словарь не может (§3.4, §27 Task 2).
    """
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            state = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            candidate = next(
                item for item in state["candidates"] if item["source"] == "llm" and item["word"]
            )
            review = await api.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={
                    "source": candidate["word"],
                    "target": "зам+ок",
                    "scope": "project",
                    "replica_index": candidate["replica_index"],
                    "note": "принято из предложения LLM",
                },
            )
            assert review.status_code == 200, review.text
            entries = (await api.get("/api/pronunciation", params={"project_id": project["id"]})).json()
            full = (await api.get(f"/api/projects/{project['id']}")).json()
            return {
                "review": review.json(),
                "entries": entries,
                "full": full,
                "word": candidate["word"],
            }

    result = asyncio.run(scenario())
    entry = result["review"]["entry"]
    # Источник — ровно форма из текста: правило словаря сопоставляется по границам
    # слова, поэтому «замка» и «замок» — разные записи, и подменять одну другой нельзя.
    assert entry["source"] == result["word"]
    assert entry["target"] == "зам+ок"
    assert entry["enabled"] is True
    assert entry["project_id"] == result["full"]["id"]
    # Правило проекта видно в списке правил проекта: словарь один, второго нет.
    assert any(item["source"] == result["word"] for item in result["entries"]["entries"])
    assert len(client.calls) > 0


def test_llm_candidate_review_global_scope_is_explicit(llm_env, voices, stub, monkeypatch):
    """Глобальный словарь меняется только при явном `scope: global`."""
    llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            project_review = await api.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={"source": "замок", "target": "зам+ок", "scope": "project"},
            )
            global_review = await api.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={"source": "термин", "target": "т+ермин", "scope": "global"},
            )
            return {
                "project": project_review.json()["entry"],
                "global": global_review.json()["entry"],
            }

    result = asyncio.run(scenario())
    assert result["project"]["project_id"] is not None
    assert result["global"].get("project_id") in (None, ""), (
        "глобальное правило не должно быть привязано к проекту"
    )


def test_llm_candidates_respect_rejected_dictionary_memory(llm_env, voices, stub, monkeypatch):
    """Отклонённое слово не предлагается снова, но остаётся видимым как решённое."""
    llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            before = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            word = next(item["word"] for item in before["candidates"] if item["source"] == "llm")
            # Пользователь отказался от слова: выключенное правило = память решения.
            # Источник — ровно та форма, что предложила модель: правило словаря
            # сопоставляется по границам слова, и «замок» не покроет «замка».
            await api.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={"source": word, "target": "", "scope": "project"},
            )
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            state = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            return {"state": state, "word": word}

    result = asyncio.run(scenario())
    state, word = result["state"], result["word"]
    words = [item for item in state["candidates"] if item["source"] == "llm" and item["word"]]
    assert words, "кандидат остаётся в разборе как диагностика"
    rejected = [item for item in words if item["word"] == word]
    assert rejected, "отклонённое слово должно остаться видимым"
    assert all(item["resolved_by"] == "rejected" for item in rejected)
    assert all(item["needs_review"] is False for item in rejected), (
        "повторно спрашивать про отклонённое слово нельзя"
    )
    assert state["needs_review_total"] == 0


def test_continuous_text_uses_same_analyzer(llm_env, voices, stub, monkeypatch):
    """Сплошной текст анализируется тем же Analyzer'ом и не создаёт проект.

    Второй вкладке нужен тот же разбор, что и диалогу: иначе «Сплошной текст» тихо
    оставался бы без лингвистического анализа, а постановка требует общего пути.
    При этом текст **не меняется**: аннотации только показываются, применять их без
    решения человека нельзя.
    """
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            response = await api.post(
                "/api/render-text",
                json={
                    "text": "Мы гуляли вокруг старого замка на холме.",
                    "voice_id": F5_VOICE,
                    "engine": "f5",
                },
            )
            assert response.status_code == 202, response.text
            projects = (await api.get("/api/projects")).json()["projects"]
            return {"body": response.json(), "projects": projects}

    result = asyncio.run(scenario())
    report = result["body"]["llm"]
    assert report["status"] in {llm.STATUS_READY, llm.STATUS_NEEDS_REVIEW}
    assert report["candidates_total"] >= 1
    assert report["candidates"][0]["source"] == "llm"
    assert result["projects"] == [], "сплошной текст не создаёт проект"
    assert len(client.calls) > 0


def test_continuous_text_skips_analysis_when_disabled(llm_env, voices, stub, monkeypatch):
    """Выключенный Analyzer: сплошной текст рендерится как раньше, без модели."""
    client = llm_env(enabled=False)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            response = await api.post(
                "/api/render-text",
                json={"text": "Сегодня хорошая погода.", "voice_id": F5_VOICE, "engine": "f5"},
            )
            assert response.status_code == 202, response.text
            return response.json()

    body = asyncio.run(scenario())
    assert body["llm"]["status"] == llm.STATUS_DISABLED
    assert client.calls == []
    assert body["job_id"]


def test_short_replica_receives_llm_context_hint(llm_env, voices, stub, monkeypatch):
    """Подсказка LLM доходит до контекста короткой реплики, не меняя её текст.

    Класс реплики и релевантные соседи — информация для слоя коротких реплик;
    решения по аудио, стратегии и текст остаются за ним.
    """
    from backend.dialogue_parser import Replica
    from backend.short_utterance import build_contexts

    def responder(case):
        return json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [],
                "utterance": {
                    "class": "CONFIRMATION",
                    "context_dependency": "HIGH",
                    "relevant_replica_ids": [1, 2],
                },
            },
            ensure_ascii=False,
        )

    llm_env(responder=responder)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            state = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            full = (await api.get(f"/api/projects/{project['id']}")).json()
            return {"state": state, "full": full}

    result = asyncio.run(scenario())
    assert result["state"]["status"] in {llm.STATUS_READY, llm.STATUS_NEEDS_REVIEW}

    # Подсказки читаются из сохранённых разборов проекта.
    from backend.db.store import get_projects_store

    hints = get_projects_store().llm_utterance_hints(result["full"]["id"])
    assert hints, "подсказки должны читаться из базы"
    assert hints[0]["class"] == "CONFIRMATION"
    assert hints[0]["context_dependency"] == "HIGH"

    # Контекст короткой реплики получает подсказку и сохраняет цель неизменной.
    replicas = [
        Replica(voice="ИВАН", text=row["text"], line_number=row["index"] + 1,
                final_text=row["final_text"])
        for row in result["full"]["replicas"]
    ]
    contexts = build_contexts(replicas, llm_hints=hints)
    assert contexts[0].llm_class == "CONFIRMATION"
    assert contexts[0].llm_relevant == (1, 2)
    assert contexts[0].target_text == result["full"]["replicas"][0]["final_text"]
    # Тексты соседей и план не изменились: подсказка — только информация.
    assert contexts[0].previous_text is None
    assert contexts[0].utterance is not None

    # Отрицательные и «свои» id отбрасываются: подсказка не должна ссылаться на себя.
    dirty = build_contexts(
        replicas, llm_hints={0: {"class": "NEGATION", "relevant_replica_ids": [-1, 0, 1, "мусор"]}}
    )
    assert dirty[0].llm_relevant == (1,)
    assert dirty[0].llm_class == "NEGATION"


def test_job_payload_hints_do_not_change_render_settings():
    """Подсказки не попадают в настройки рендера: аудио они не меняют."""
    from backend import job_queue as job_queue_module
    from backend.audio_pipeline import RenderSettings, SpeakerSettings
    from backend.dialogue_parser import Replica
    from backend.job_queue import JobPayload

    payload = JobPayload(
        replicas=[Replica(voice="ИВАН", text="Да.", line_number=1, final_text="Да.")],
        speakers={"ИВАН": SpeakerSettings(voice_id="voice-f5")},
        settings=RenderSettings(),
        project_id="",
    )
    before = (payload.settings.pause_ms, payload.settings.auto_accent, payload.settings.qa)
    # Пустой project_id — подсказок нет, и никакого обращения к базе не происходит.
    assert job_queue_module.JobQueue._llm_hints(payload) == {}
    after = (payload.settings.pause_ms, payload.settings.auto_accent, payload.settings.qa)
    assert after == before, "подсказки не меняют настройки рендера"
    assert payload.project_id == ""
    assert payload.replicas[0].final_text == "Да."
