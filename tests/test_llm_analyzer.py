"""LinguisticAnalyzer (Task 2, фаза 1): адаптер, статус, контекст и границы.

Обычный pytest не поднимает ни LLM, ни TTS: клиент Ollama подменяется
`FakeOllamaClient`, как и в benchmark'е. Проверяется главное правило задачи —
модель предлагает аннотации, а backend остаётся источником истины: он сам ищет
слово в тексте, сам решает, что принять, и никогда не берёт текст из ответа.

Отдельная проверка — «отказ виден»: недоступная Ollama, мусор вместо JSON или
попытка вернуть переписанный текст дают `failed` с причиной, а не тихий успех.
"""

from __future__ import annotations

import json

from conftest import StubEngine  # noqa: F401 — нужен фикстуре движков

from backend.llm import analyzer as llm
from backend.llm import schemas as s
from backend.llm.fake_client import FakeOllamaClient
from backend.llm.ollama_client import OllamaModel

MODELS = [
    OllamaModel(tag="qwen3:8b", digest="digest-primary", size_bytes=5 * 1024**3),
    OllamaModel(tag="qwen3:4b-instruct-2507-q8_0", digest="digest-fallback", size_bytes=4 * 1024**3),
]
CASTLE = "Мы гуляли вокруг старого замка на холме."


def _settings(**overrides) -> llm.AnalyzerSettings:
    base = llm.AnalyzerSettings(enabled=True)
    return llm.AnalyzerSettings(**{**base.__dict__, **overrides})


def _response(items: list[dict], *, replica_id: int = 1, extra: dict | None = None) -> str:
    payload = {
        "schema_version": s.SCHEMA_VERSION,
        "replica_id": replica_id,
        "items": items,
        "utterance": {"class": "NORMAL", "context_dependency": "LOW"},
    }
    payload.update(extra or {})
    return json.dumps(payload, ensure_ascii=False)


def _analyzer(responder, *, settings: llm.AnalyzerSettings | None = None, **client_kwargs):
    client = FakeOllamaClient(
        models=MODELS, responder=responder, capabilities={"qwen3:8b": ("completion", "thinking")},
        **client_kwargs,
    )
    return llm.LinguisticAnalyzer(settings=settings or _settings(), client=client), client


def test_analyzer_disabled_by_default():
    """Analyzer выключен, пока пользователь его не включил: это отдельная модель."""
    settings = llm.settings_from_env({})
    assert settings.enabled is False
    assert settings.primary_model == "qwen3:8b"
    assert settings.fallback_model == "qwen3:4b-instruct-2507-q8_0"

    client = FakeOllamaClient(models=MODELS)
    analyzer = llm.LinguisticAnalyzer(settings=settings, client=client)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_DISABLED
    assert result.items == ()
    assert client.calls == [], "выключенный Analyzer не должен ходить в модель"
    status = analyzer.status()
    assert status["enabled"] is False
    assert status["state"] == llm.STATUS_DISABLED
    assert "выключен" in status["reason"]


def test_analyzer_settings_from_env():
    """Имена переменных — как в постановке; проектная приставка TTS_ тоже работает."""
    settings = llm.settings_from_env(
        {
            "LLM_ANALYZER_ENABLED": "1",
            "LLM_PRIMARY_MODEL": "qwen3:8b",
            "LLM_FALLBACK_MODEL": "qwen3:4b-instruct-2507-q4_K_M",
            "TTS_LLM_ANALYZER_CONTEXT_REPLICAS": "1",
            "LLM_ANALYZER_TIMEOUT_SEC": "45",
            "LLM_ANALYZER_CONTEXT_CHARS": "не число",
        }
    )
    assert settings.enabled is True
    assert settings.primary_model == "qwen3:8b"
    assert settings.fallback_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert settings.context_replicas == 1
    assert settings.timeout_sec == 45.0
    assert settings.context_chars == llm.DEFAULT_CONTEXT_CHARS


def test_analyzer_uses_selected_model_and_reports_it():
    """Primary, если скачана; fallback — как состояние, видимое в результате."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение, крепость",
                    "confidence": 0.9,
                    "needs_review": False,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, client = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert client.calls[0]["model"] == "qwen3:8b"
    assert result.model_tag == "qwen3:8b"
    assert result.model_digest == "digest-primary"
    assert result.prompt_version == analyzer.prompt.version

    # Без primary выбирается fallback, и это тоже видно в результате.
    fallback_client = FakeOllamaClient(
        models=[MODELS[1]], responder=responder, capabilities={}
    )
    fallback = llm.LinguisticAnalyzer(settings=_settings(), client=fallback_client)
    assert fallback.selected_model() == "qwen3:4b-instruct-2507-q8_0"
    result = fallback.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.model_tag == "qwen3:4b-instruct-2507-q8_0"
    assert result.model_digest == "digest-fallback"


def test_analyzer_sends_russian_context_within_policy():
    """Контекст: цель + до двух соседей с каждой стороны и бюджет символов (§6)."""
    captured: dict = {}

    def responder(case):
        captured.update(case)
        return _response([], replica_id=case["replica_id"])

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(
        replica_id=7,
        target_text=CASTLE,
        before=["Реплика 1", "Реплика 2", "Реплика 3"],
        after=["Реплика 4", "Реплика 5", "Реплика 6"],
    )
    assert captured["target_text"] == CASTLE
    assert captured["context_before"] == ["Реплика 2", "Реплика 3"]
    assert captured["context_after"] == ["Реплика 4", "Реплика 5"]
    assert captured["language"] == "ru"
    assert result.source_hash and result.context_hash
    assert result.status == llm.STATUS_READY


def test_analyzer_context_budget_trims_far_neighbours():
    """Бюджет обрезает дальних соседей, целевую реплику — никогда."""
    context = llm.build_context(
        "цель", ["далёкий" * 50, "близкий"], ["следующий"], limit=2, budget_chars=12
    )
    assert context["target_text"] == "цель"
    assert context["context_before"] == ["близкий"], "дальний сосед не влез в бюджет"
    assert context["context_after"] == []

    context = llm.build_context("цель", [], [], limit=2, budget_chars=1)
    assert context["target_text"] == "цель", "цель не обрезается даже при нулевом бюджете"


def test_context_hash_changes_with_neighbours():
    """Хеш контекста меняется от соседей: анализ зависит не только от цели (§11)."""
    first = llm.build_context(CASTLE, ["до"], ["после"])
    second = llm.build_context(CASTLE, ["другое"], ["после"])
    assert llm.context_hash(first) != llm.context_hash(second)


def test_analyzer_returns_annotations_not_rewritten_text():
    """Ответ с полем переписанного текста не применяется: это не аннотация."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение",
                    "confidence": 0.9,
                    "needs_review": False,
                }
            ],
            replica_id=case["replica_id"],
            extra={"final_text": "Мы гуляли вокруг старого замка."},
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_FAILED
    assert "unknown_field" in result.error
    assert result.items == ()
    # Текст реплики нигде не подменён: в разборе вообще нет поля с текстом.
    payload = result.to_dict()
    assert "final_text" not in json.dumps(payload)
    assert CASTLE not in json.dumps(payload, ensure_ascii=False)


def test_analysis_locates_source_when_model_span_is_wrong():
    """Границы ищет backend: benchmark показал, что модель считает символы неверно."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 3,
                    "span_end": 9,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение, крепость",
                    "confidence": 0.9,
                    "needs_review": False,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_READY
    item = result.items[0]
    assert (item.span_start, item.span_end) == (25, 30)
    assert item.source == "замка"
    assert CASTLE[item.span_start : item.span_end] == item.source


def test_analysis_drops_invented_source_and_requires_review():
    """Слова нет в тексте — аннотация отбрасывается с причиной, разбор в review."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 0,
                    "span_end": 6,
                    "source": "зáмок",
                    "type": "homograph",
                    "meaning": "строение",
                    "confidence": 0.9,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.items == ()
    assert result.status == llm.STATUS_NEEDS_REVIEW
    assert result.dropped[0]["reason"] == llm.DROP_SOURCE_NOT_FOUND


def test_analysis_strips_engine_specific_stress_markup():
    """`+` — разметка F5, её не должна формировать LLM (§10, §27 Task 2)."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение",
                    "suggested_form": "зам+ок",
                    "confidence": 0.9,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.items[0].suggested_form == "замок"
    assert "+" not in json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.markup_stripped == 1
    assert result.status == llm.STATUS_NEEDS_REVIEW


def test_malformed_json_is_not_applied_and_retry_is_bounded():
    """Мусор вместо JSON: ровно один повтор, затем `failed` с причиной (§13)."""
    calls = {"count": 0}

    def garbage(case):
        calls["count"] += 1
        return "конечно! вот результат:"

    analyzer, client = _analyzer(garbage, settings=_settings(max_repairs=1))
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert calls["count"] == 2, "один запрос и ровно один повтор"
    assert len(client.calls) == 2
    assert result.status == llm.STATUS_FAILED
    assert "not_json" in result.error
    assert result.items == ()

    # Повторы ограничены настройкой: с нулём повторов запрос один.
    single, single_client = _analyzer(garbage, settings=_settings(max_repairs=0))
    single.analyze_replica(replica_id=1, target_text=CASTLE)
    assert len(single_client.calls) == 1


def test_malformed_json_repair_can_succeed():
    """Повтор после мусора: валидный ответ применяется и помечается как repair."""
    calls = {"count": 0}

    def flaky(case):
        calls["count"] += 1
        if calls["count"] == 1:
            return "не JSON"
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение",
                    "confidence": 0.9,
                    "needs_review": False,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(flaky)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_READY
    assert result.repaired is True
    assert result.items[0].source == "замка"


def test_ollama_unavailable_is_a_visible_failure_not_a_crash():
    """Ollama недоступна: backend работает, анализ — `failed` с причиной (§13, §17)."""
    analyzer, _ = _analyzer(
        lambda case: _response([], replica_id=case["replica_id"]), available=False
    )
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_FAILED
    assert "OllamaUnavailableError" in result.error
    status = analyzer.status()
    assert status["ollama"]["available"] is False
    assert status["ollama"]["error"]


def test_analyzer_status_and_models_roles():
    """Статус отвечает, готов ли анализ: включён, доступна Ollama, какая модель."""
    analyzer, _ = _analyzer(lambda case: _response([], replica_id=case["replica_id"]))
    status = analyzer.status()
    assert status["enabled"] is True
    assert status["model"] == "qwen3:8b"
    assert status["model_available"] is True
    assert status["prompt_version"] == analyzer.prompt.version
    assert status["schema_version"] == s.SCHEMA_VERSION
    assert status["ollama"]["available"] is True

    roles = {item["tag"]: item["role"] for item in analyzer.models()}
    assert roles["qwen3:8b"] == "primary"
    assert roles["qwen3:4b-instruct-2507-q8_0"] == "fallback"


def test_analyzer_status_survives_missing_model():
    """Модели нет локально: статус говорит об этом, а не падает."""
    client = FakeOllamaClient(models=[OllamaModel(tag="gemma3:4b")])
    analyzer = llm.LinguisticAnalyzer(settings=_settings(), client=client)
    status = analyzer.status()
    assert status["ollama"]["available"] is True
    assert status["model_available"] is False
    assert "не скачана" in status["reason"]
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_FAILED
    assert "не скачана" in result.error


def test_analyzer_timeout_is_a_visible_failure():
    """Таймаут — тоже видимая ошибка анализа, а не зависание и не тихий успех."""
    analyzer, _ = _analyzer(
        lambda case: _response([], replica_id=case["replica_id"]),
        settings=_settings(timeout_sec=0.5),
    )

    from backend.llm.ollama_client import OllamaTimeoutError

    def boom(*args, **kwargs):
        raise OllamaTimeoutError("Ollama не ответила за 0.5 с")

    analyzer.client.chat = boom
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_FAILED
    assert "OllamaTimeoutError" in result.error


def test_analyzer_disables_thinking_for_capable_model():
    """`think=auto`: у модели с thinking скрытое рассуждение выключается."""
    analyzer, client = _analyzer(
        lambda case: _response([], replica_id=case["replica_id"])
    )
    analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert client.calls[0]["think"] is False


def test_llm_status_and_models_endpoints(monkeypatch):
    """Ручки статуса работают и без Ollama, и с ней: дашборд не зависит от модели."""
    import asyncio

    from test_projects_api import _client

    async def scenario():
        async with _client(monkeypatch) as client:
            # По умолчанию Analyzer выключен и Ollama может быть недоступна.
            response = await client.get("/api/llm/status")
            assert response.status_code == 200
            payload = response.json()
            assert payload["enabled"] is False
            assert payload["schema_version"] == s.SCHEMA_VERSION

            models = await client.get("/api/llm/models")
            assert models.status_code == 200
            assert "models" in models.json()

            # С включённым Analyzer и подставным клиентом статус показывает модель.
            fake, _ = _analyzer(
                lambda case: _response([], replica_id=case["replica_id"])
            )
            monkeypatch.setattr(llm, "_analyzer", fake)
            payload = (await client.get("/api/llm/status")).json()
            assert payload["enabled"] is True
            assert payload["model"] == "qwen3:8b"
            assert payload["model_available"] is True
            roles = {item["tag"]: item["role"] for item in (await client.get("/api/llm/models")).json()["models"]}
            assert roles["qwen3:8b"] == "primary"

    asyncio.run(scenario())


def test_analysis_drops_unknown_type_with_reason():
    """Неизвестный тип — аннотация не принимается, но и не роняет разбор."""
    def responder(case):
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "магия",
                    "confidence": 0.9,
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    # Неизвестный тип ловит ещё схема: ответ не применяется целиком (§4 Task 2).
    assert result.status == llm.STATUS_FAILED
    assert "unknown_type" in result.error
    assert result.items == ()


def test_conflicting_overlapping_patches_are_resolved_not_applied_twice():
    """Две аннотации на одно место: принимается одна, вторая отбрасывается с причиной."""
    items, dropped = llm.locate_annotations(
        [
            s.Annotation(
                span_start=25,
                span_end=30,
                source="замка",
                type=s.TYPE_HOMOGRAPH,
                meaning="строение",
                confidence=0.9,
            ),
            s.Annotation(
                span_start=26,
                span_end=30,
                source="амка",
                type=s.TYPE_STRESS,
                meaning="ударение",
                confidence=0.5,
            ),
        ],
        CASTLE,
    )
    assert len(items) == 1
    assert items[0].source == "замка"
    assert dropped[0]["reason"] == llm.DROP_OVERLAP


def test_locate_annotations_prefers_nearest_occurrence():
    """Повтор слова в тексте: берётся вхождение, ближайшее к подсказке модели."""
    text = "Замок старый. Он открыл замок ключом."
    items, dropped = llm.locate_annotations(
        [
            s.Annotation(
                span_start=26,
                span_end=31,
                source="замок",
                type=s.TYPE_HOMOGRAPH,
                meaning="запор",
                confidence=0.9,
            )
        ],
        text,
    )
    assert dropped == ()
    assert items[0].span_start == 24, "подсказка указывает на второе вхождение"
    # Первое вхождение («Замок» в начале) не должно выбираться: подсказка дальше.
    assert text.lower().index("замок") == 0
    assert text[items[0].span_start : items[0].span_end] == "замок"


def test_unknown_reason_code_is_informational_not_fatal():
    """Выдуманный `reason_code` не отвергает разбор: это пояснение, а не решение.

    Постановка (§4) требует строго проверять JSON, версии, id, границы, `source`,
    типы, уверенность и перекрытия; код причины — короткое пояснение. Модель иногда
    выдумывает код, и терять из-за этого целую реплику нельзя: живой smoke так
    потерял 12 из 110 разборов. В benchmark строгость осталась: там метрика меряет
    следование контракту.
    """
    def responder(case):
        return _response(
            [
                {
                    "span_start": 25,
                    "span_end": 30,
                    "source": "замка",
                    "type": "homograph",
                    "meaning": "строение",
                    "confidence": 0.9,
                    "needs_review": False,
                    "reason_code": "HOMOGRAPH_CONTEXT",  # такого кода нет в схеме
                }
            ],
            replica_id=case["replica_id"],
        )

    analyzer, _ = _analyzer(responder)
    result = analyzer.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.status == llm.STATUS_READY
    assert result.items[0].source == "замка"
    assert result.items[0].reason_code == "", "неизвестный код сохраняется пустым"

    # Строгий режим (benchmark) по-прежнему отвергает такой ответ.
    payload = _response(
        [
            {
                "span_start": 25,
                "span_end": 30,
                "source": "замка",
                "type": "homograph",
                "reason_code": "HOMOGRAPH_CONTEXT",
            }
        ],
        replica_id=1,
    )
    strict, errors = s.parse_analysis(payload, expected_replica_id=1, target_text=CASTLE)
    assert strict is None
    assert s.ERROR_UNKNOWN_REASON in errors
