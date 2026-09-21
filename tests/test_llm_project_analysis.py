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
    """Ответ «модели»: омограф в целевой реплике с правильным словом и границами.

    Диалог приходит окном (§7), поэтому один и тот же ответ умеет обе формы: на
    одиночный кейс — объект разбора, на окно — конверт `replicas[]`.
    """
    replicas = case.get("replicas")
    if isinstance(replicas, list):
        return json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replicas": [json.loads(_answer(item, meaning=meaning)) for item in replicas],
            },
            ensure_ascii=False,
        )
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


def _window_aware(responder):
    """Оборачивает ответ на одну реплику в ответ на окно (§7).

    Тесты описывают разбор одной реплики, а сцена уходит окном: обёртка избавляет
    каждый responder от второго формата ответа.
    """

    def answer(case: dict) -> str:
        replicas = case.get("replicas")
        if not isinstance(replicas, list):
            return responder(case)
        return json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replicas": [json.loads(responder(item)) for item in replicas],
            },
            ensure_ascii=False,
        )

    return answer


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
        scene_replicas: int = 6,
        handles_window: bool = False,
    ) -> FakeOllamaClient:
        """Ставит подставной Analyzer.

        `handles_window=True` — responder сам читает окно (`case["replicas"]`):
        нужен там, где ответ зависит от соседей по сцене (§52), а не только от
        собственного текста реплики.
        """
        answer = responder or _answer
        client = FakeOllamaClient(
            models=[MODEL],
            responder=answer if handles_window else _window_aware(answer),
            capabilities={},
        )
        settings = llm.AnalyzerSettings(
            enabled=enabled,
            required_for_render=required_for_render,
            context_replicas=context_replicas,
            scene_replicas=scene_replicas,
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
    """Повторный анализ того же текста не гоняет модель заново (§78).

    Сцена неизменна — значит, и разбор годен: второй проход обязан взять его из
    кеша целиком. Это ещё и проверка того, что ключ кеша описан **сценой** (текст,
    контекст, говорящие), а не временем создания.
    """
    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            first_calls = len(client.calls)
            first = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()["run"]
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            second = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()["run"]
            return {
                "first": first,
                "second": second,
                "calls": len(client.calls),
                "first_calls": first_calls,
            }

    calls = asyncio.run(scenario())
    assert calls["first"]["calls"] == 1, "сцена читается одним запросом"
    assert calls["first"]["from_cache"] == 0
    assert calls["calls"] == calls["first_calls"], "кеш обязан вернуть разбор без модели"
    assert calls["second"]["calls"] == 0
    assert calls["second"]["from_cache"] == calls["first"]["replicas"]


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


def test_analysis_pass_stops_on_hard_stop_memory():
    """HARD STOP во время прохода останавливает анализ и выгружает модель (§14).

    Проход длинный: десятки реплик, модель загружена, KV-кэш растёт. Проверка только
    «до начала» не спасает — память уходит в HARD STOP уже в середине, и продолжать
    значит уйти в swap на всей машине.
    """
    from backend.llm import integration as integration_module
    from backend.llm import memory_policy as memory
    from backend.llm import scheduler as scheduler_module
    from backend.llm.analyzer import LinguisticAnalyzer
    from backend.llm.fake_client import FakeOllamaClient
    from backend.llm.ollama_client import OllamaModel

    client = FakeOllamaClient(
        models=[OllamaModel(tag="qwen3:8b", digest="d1", size_bytes=5 * 1024**3)],
        responder=lambda case: json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [],
                "utterance": {"class": "NORMAL"},
            },
            ensure_ascii=False,
        ),
    )
    green = {"system_percent": 50.0, "available_gb": 12.0, "pressure": "green"}
    hard = {"system_percent": 90.0, "available_gb": 2.0, "pressure": "green"}
    # Первая зелёная — вход в слот; вторая — проверка перед первой репликой;
    # дальше HARD STOP: анализ обязан остановиться, не дойдя до второй реплики.
    samples = [dict(green), dict(green), dict(hard)]
    state = {"index": 0}

    def sensor() -> dict:
        index = min(state["index"], len(samples) - 1)
        state["index"] += 1
        return samples[index]

    analyzer = LinguisticAnalyzer(settings=llm.AnalyzerSettings(enabled=True), client=client)
    runner = integration_module.ProjectLlmAnalyzer(
        analyzer=analyzer,
        scheduler=scheduler_module.HeavyScheduler(
            gate=memory.HeavyGate(), sensor=sensor
        ),
        lookup=lambda index, key: None,
        save=lambda index, analysis: None,
    )
    outcome = runner.run(
        project_id="p1",
        replicas=[{"index": 0, "text": "Да."}, {"index": 1, "text": "Нет."}, {"index": 2, "text": "Ок."}],
    )
    assert outcome.status == llm.STATUS_FAILED
    assert "HARD STOP" in outcome.error
    assert outcome.calls == 1, "после HARD STOP модель больше не спрашивают"
    assert client.unloaded == ["qwen3:8b"], "модель обязана быть выгружена при остановке"


def test_failed_replica_is_stored_and_visible(llm_env, voices, stub, monkeypatch):
    """Неудачный разбор реплики сохраняется и виден с причиной, а не исчезает.

    Иначе проект показывает «ошибка», а пользователь не знает, какая реплика не
    разобрана и почему — и не может решить: повторить или продолжать без неё.
    """
    from backend.llm.ollama_client import OllamaTimeoutError

    client = llm_env()
    calls = {"n": 0}

    def flaky(case):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OllamaTimeoutError("модель не ответила за 90 с")
        return _answer(case)

    client._responder = flaky

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())
    assert state["replicas_failed"] >= 1
    assert state["failed"], "причина отказа должна быть в состоянии"
    assert "Timeout" in state["failed"][0]["error"] or "не ответила" in state["failed"][0]["error"]
    assert state["status"] == llm.STATUS_FAILED


def test_render_blocked_when_required_llm_review_pending(llm_env, voices, stub, monkeypatch):
    """Режим «LLM обязателен»: пока предложения не решены, рендер запрещён.

    Это второй случай блокировки (первый — неудачный анализ): анализ прошёл, но
    требует решения человека, и тихо рендерить «как получилось» нельзя.
    """
    def responder(case: dict) -> str:
        text = case["target_text"]
        word = "старого" if "старого" in text else text.split()[0]
        start = text.index(word)
        return json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [
                    {
                        "span_start": start,
                        "span_end": start + len(word),
                        "source": word,
                        "type": "ambiguous",
                        "meaning": "не уверен в чтении",
                        "confidence": 0.4,
                        "needs_review": True,
                    }
                ],
                "utterance": {"class": "NORMAL"},
            },
            ensure_ascii=False,
        )

    llm_env(responder=responder, required_for_render=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            # Решаем только детерминированные предложения: подготовка текста
            # становится готовой, а предложение LLM остаётся нерешённым — именно
            # этот случай и проверяет режим «LLM обязателен».
            for _ in range(4):
                state = (await api.get(f"/api/projects/{project['id']}/analysis")).json()
                pending = [
                    item
                    for item in (state.get("candidates") or [])
                    if item.get("source") != "llm"
                ]
                if not pending:
                    break
                for item in pending:
                    await api.post(
                        f"/api/projects/{project['id']}/pronunciation/review",
                        json={
                            "source": item.get("word") or "",
                            "target": item.get("word") or "",
                            "scope": "project",
                            "enabled": False,
                            "replica_index": item.get("replica_index"),
                        },
                    )
            render = await api.post(f"/api/projects/{project['id']}/render", json={})
            return {
                "status": render.status_code,
                "body": render.json(),
                "llm": (
                    await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
                ).json(),
            }

    result = asyncio.run(scenario())
    assert result["llm"]["status"] == llm.STATUS_NEEDS_REVIEW, "предложение LLM ждёт решения"
    assert result["status"] == 409
    assert "обязателен" in result["body"]["detail"]


def test_render_allowed_after_llm_review(llm_env, voices, stub, monkeypatch):
    """После решения предложений рендер разрешён, и синтез берёт `final_text`.

    Полный путь §21 на подставной модели: анализ → review → готовность → рендер.
    """
    llm_env(required_for_render=True)
    clean = "ИВАН: Сегодня хорошая погода.\nМАРГО: И я рад тебя видеть."

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api, text=clean)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            final_state = (
                await api.get(f"/api/projects/{project['id']}/analysis")
            ).json()
            llm_final = (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()
            render = await api.post(f"/api/projects/{project['id']}/render", json={})
            full = (await api.get(f"/api/projects/{project['id']}")).json()
            return {
                "analysis": final_state,
                "llm": llm_final,
                "render_status": render.status_code,
                "render": render.json() if render.status_code == 202 else {},
                "final_texts": [row["final_text"] for row in full["replicas"]],
            }

    result = asyncio.run(scenario())
    assert result["analysis"]["status"] == "ready", "после review проект готов"
    assert result["llm"]["status"] == "READY"
    assert result["render_status"] == 202, result["render"]
    assert all(result["final_texts"]), "рендер идёт по подготовленному тексту"


# --- Фаза 6: сцена — один запрос на окно (§6, §7, §77) -------------------------
# Gate фазы: «нет one-request-per-replica». Проверяется не только счётчиком
# вызовов, но и тем, что в запрос уходит сцена с говорящими, а ответ раскладывается
# по `replica_id`, а не по порядку.
FOUR_LINES = (
    "ИВАН: Мы гуляли вокруг старого замка на холме.\n"
    "МАРГО: Он поменял замок на входной двери.\n"
    "ИВАН: Замок был старый и скрипел.\n"
    "МАРГО: Надо поменять замок весной."
)


def _scene_response(case: dict) -> str:
    """Ответ модели на окно: по объекту на каждую реплику, без аннотаций."""
    return json.dumps(
        {
            "schema_version": s.SCHEMA_VERSION,
            "replicas": [
                {"replica_id": item["replica_id"], "items": []} for item in case["replicas"]
            ],
        },
        ensure_ascii=False,
    )


def test_dialogue_analysis_uses_neighbor_context(llm_env, voices, stub, monkeypatch):
    """Диалог уходит моделью **сценой**: один запрос, все реплики с говорящими (§6, §7).

    Смысл реплики в диалоге читается по сцене, а не по одной строке. Поэтому §77
    требует не только «мало запросов», но и того, что в запрос попали соседи и кто
    их произносит.
    """
    captured: list[dict] = []

    def responder(case: dict) -> str:
        captured.append(case)
        return _scene_response(case)

    client = llm_env(responder=responder, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())

    assert len(client.calls) == 1, "сцена — один запрос, а не запрос на реплику (§77)"
    scene = captured[0]["replicas"]
    assert [item["speaker"] for item in scene] == ["ИВАН", "МАРГО"]
    assert scene[0]["target_text"].startswith("Мы гуляли")
    assert scene[1]["target_text"].startswith("Он поменял")

    run = state["run"]
    assert run["calls"] == 1
    assert run["replicas"] == 2, "разбор получила каждая реплика окна"
    assert run["replicas_per_call"] == 2.0, "две реплики — один вызов модели"


def test_dialogue_analysis_returns_results_by_replica_id(llm_env, voices, stub, monkeypatch):
    """Разборы раскладываются по `replica_id`, а не по порядку в ответе (§62).

    Модель вправе вернуть окно в любом порядке. При позиционной привязке реплики
    получили бы чужие аннотации — и заметить это было бы негде: обе аннотации «про
    этот текст».
    """

    def responder(case: dict) -> str:
        entries = []
        for item in reversed(case["replicas"]):
            text = item["target_text"]
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
                        "meaning": "строение, крепость",
                        "suggested_form": "замок",
                        "confidence": 0.9,
                        "needs_review": False,
                    }
                )
            entries.append({"replica_id": item["replica_id"], "items": items})
        return json.dumps(
            {"schema_version": s.SCHEMA_VERSION, "replicas": entries}, ensure_ascii=False
        )

    llm_env(responder=responder, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())
    words: dict[int, list[str]] = {}
    for candidate in state["candidates"]:
        words.setdefault(candidate["replica_index"], []).append(candidate["word"])
    assert words == {0: ["замка"], 1: ["замок"]}


def test_long_dialogue_is_read_in_overlapping_windows(llm_env, voices, stub, monkeypatch):
    """Длинный диалог режется на окна с перехлёстом: соседи остаются видимыми (§6).

    Окна перекрываются, иначе реплика на границе читалась бы без второй стороны, а
    разбор перекрывающейся реплики берётся из первого окна — ровно один на реплику.
    """
    captured: list[dict] = []

    def responder(case: dict) -> str:
        captured.append(case)
        return _scene_response(case)

    client = llm_env(responder=responder, handles_window=True, scene_replicas=3)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api, text=FOUR_LINES)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())

    assert len(client.calls) == 2, "четыре реплики по три в окне — два запроса"
    assert [len(call["replicas"]) for call in captured] == [3, 3]
    first_ids = [item["replica_id"] for item in captured[0]["replicas"]]
    second_ids = [item["replica_id"] for item in captured[1]["replicas"]]
    assert second_ids[0] == first_ids[1], "окна перекрываются: граница видит соседей"

    run = state["run"]
    assert run["replicas"] == 4
    assert run["from_cache"] == 0
    assert run["calls"] == 2
    assert run["replicas_per_call"] == 2.0
    assert state["replicas_failed"] == 0


def test_same_text_can_receive_different_prosody_in_different_context(
    llm_env, voices, stub, monkeypatch
):
    """Одно и то же «Правда?» получает разную интонацию по сцене (§52, §62).

    В этом и смысл оконного анализа: без соседей реплика была бы неразличима, и
    модель вынуждена была бы угадывать по знаку вопроса.
    """
    from backend.db.store import get_projects_store

    GOOD = "ИВАН: Мы выиграли миллион.\nМАРГО: Правда?"
    BAD = "ИВАН: Я опять забыл документы.\nМАРГО: Правда?"

    def responder(case: dict) -> str:
        scene = case["replicas"]
        delight = "миллион" in scene[0]["target_text"]
        entries = []
        for item in scene:
            utterance = {"class": "NORMAL", "context_dependency": "LOW"}
            if item["target_text"].strip().startswith("Правда"):
                utterance = {
                    "class": "QUESTION",
                    "context_dependency": "HIGH",
                    "emotion": "DELIGHT" if delight else "SAD_SYMPATHETIC",
                    "emotion_confidence": 0.8,
                    "dialogue_act": "QUESTION",
                }
            entries.append(
                {"replica_id": item["replica_id"], "items": [], "utterance": utterance}
            )
        return json.dumps(
            {"schema_version": s.SCHEMA_VERSION, "replicas": entries}, ensure_ascii=False
        )

    llm_env(responder=responder, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            found = {}
            for name, text in (("good", GOOD), ("bad", BAD)):
                project = await _project_with_voices(api, text=text)
                await api.post(f"/api/projects/{project['id']}/analyze", json={})
                found[name] = project["id"]
            return found

    projects = asyncio.run(scenario())
    store = get_projects_store()
    good = llm_integration.replica_emotions(store.llm_analyses(projects["good"]))
    bad = llm_integration.replica_emotions(store.llm_analyses(projects["bad"]))
    assert good[1]["emotion"] == "DELIGHT"
    assert bad[1]["emotion"] == "SAD_SYMPATHETIC"


def test_unknown_replica_id_is_rejected(llm_env, voices, stub, monkeypatch):
    """Чужой `replica_id` не привязывается «по порядку», а пропуск виден (§47, §62).

    Модель, вернувшая выдуманный id, не получает за это чужие тексты: разбор
    отбрасывается, а реплика, о которой ответа не было, честно уходит в FAILED.
    """

    def responder(case: dict) -> str:
        real = [item["replica_id"] for item in case["replicas"]]
        entries = [{"replica_id": real[0], "items": []}, {"replica_id": 999, "items": []}]
        return json.dumps(
            {"schema_version": s.SCHEMA_VERSION, "replicas": entries}, ensure_ascii=False
        )

    llm_env(responder=responder, handles_window=True)

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return (
                await api.get(f"/api/projects/{project['id']}/linguistic-analysis")
            ).json()

    state = asyncio.run(scenario())
    assert state["candidates_total"] == 0, "чужой id не превращается в разбор"
    assert state["replicas_failed"] == 1
    assert "не разобрана" in state["failed"][0]["error"]
    assert state["status"] == llm.STATUS_FAILED


def test_window_prompt_version_invalidates_scene_cache(llm_env, voices, stub, monkeypatch):
    """Смена версии оконного prompt'а инвалидирует разбор сцены (§46, §78).

    Разбор получен по другим условиям задачи — отдавать его как актуальный нельзя,
    хотя текст, модель и словарь не менялись.
    """
    from dataclasses import replace

    client = llm_env()

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            project = await _project_with_voices(api)
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            first = len(client.calls)
            analyzer = llm.get_analyzer()
            llm._analyzer = llm.LinguisticAnalyzer(
                settings=analyzer.settings,
                client=analyzer.client,
                window_prompt=replace(
                    analyzer.window_prompt,
                    version=str(int(analyzer.window_prompt.version) + 1),
                ),
            )
            await api.post(f"/api/projects/{project['id']}/analyze", json={})
            return {"first": first, "second": len(client.calls)}

    result = asyncio.run(scenario())
    assert result["second"] > result["first"], "новый prompt — новый разбор, кеш не подходит"


def test_speaker_change_invalidates_scene_cache():
    """Смена говорящего меняет сцену — разбор этой сцены кешем не отдаётся (§78).

    `speaker` входит в окно, поэтому переименование говорящего — это другой вход,
    а не тот же самый: «Правда?» от разных людей читается по-разному.
    """
    from backend.llm import analysis_cache as cache

    client = FakeOllamaClient(
        models=[MODEL], responder=_window_aware(_answer), capabilities={}
    )
    analyzer = llm.LinguisticAnalyzer(
        settings=llm.AnalyzerSettings(enabled=True, scene_replicas=6), client=client
    )
    saved: dict = {}

    def lookup(index, key):
        analysis = saved.get(index)
        return analysis if analysis is not None and cache.key_for(analysis) == key else None

    def save(index, analysis):
        saved[index] = analysis

    runner = llm_integration.ProjectLlmAnalyzer(
        analyzer=analyzer,
        scheduler=scheduler_module.HeavyScheduler(
            gate=memory.HeavyGate(), sensor=lambda: dict(GREEN)
        ),
        lookup=lookup,
        save=save,
        unload_after=False,
    )
    pair = [
        {
            "index": 0,
            "text": "Мы гуляли вокруг старого замка на холме.",
            "replica_id": 7,
            "speaker": "Аня",
        },
        {
            "index": 1,
            "text": "Он поменял замок на входной двери.",
            "replica_id": 8,
            "speaker": "Борис",
        },
    ]

    first = runner.run(project_id="scene", replicas=pair, scene=True)
    assert first.calls == 1 and first.from_cache == 0
    assert len(client.calls) == 1

    same = runner.run(project_id="scene", replicas=pair, scene=True)
    assert same.calls == 0, "та же сцена берётся из кеша"
    assert same.from_cache == 2
    assert len(client.calls) == 1

    renamed = [{**pair[0], "speaker": "Вера"}, pair[1]]
    changed = runner.run(project_id="scene", replicas=renamed, scene=True)
    assert changed.calls == 1, "говорящий — часть сцены, старый разбор не годится"
    assert len(client.calls) == 2

