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
