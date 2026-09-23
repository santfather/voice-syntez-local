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


def test_inference_params_are_recorded_in_the_analysis():
    """Параметры запроса записываются в разбор: они — часть входа, а не оформление.

    По этому хешу (`inference_hash`) кеш решает, что разбор сделан при других
    настройках, и не отдаёт его как актуальный (`analysis_cache.AnalysisKey`).
    """
    def responder(case):
        return _response([], replica_id=case["replica_id"])

    base, _ = _analyzer(responder)
    result = base.analyze_replica(replica_id=1, target_text=CASTLE)
    assert result.inference_hash == llm.inference_hash(base.settings)

    # Любой из трёх параметров меняет хеш: иначе прежний разбор выглядел бы
    # действительным при других настройках.
    for override in ({"temperature": 0.7}, {"num_ctx": 4096}, {"seed": 42}):
        other, _ = _analyzer(responder, settings=_settings(**override))
        assert llm.inference_hash(other.settings) != result.inference_hash, override
        assert other.analyze_replica(replica_id=1, target_text=CASTLE).inference_hash != (
            result.inference_hash
        ), override


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


def test_non_numeric_annotation_fields_reject_response_instead_of_crashing():
    """Мусор в типах полей аннотации — негодный ответ, а не исключение.

    Валидный JSON вроде `{"span_start": "abc"}` или `{"confidence": {"x": 1}}`
    бросал `ValueError`/`TypeError` из `Annotation.from_dict`. Вызывающий слой ловит
    только `OllamaError`, поэтому исключение уносило **весь** проход анализа — все
    реплики вместо одной. Теперь такой ответ отвергается как обычно, с кодом ошибки.
    """
    bad_spans = _response(
        [{"span_start": "abc", "span_end": 30, "source": "замка", "type": "homograph"}],
        replica_id=1,
    )
    analysis, errors = s.parse_analysis(bad_spans, expected_replica_id=1, target_text=CASTLE)
    assert analysis is None
    assert s.ERROR_FIELD_TYPES in errors

    bad_confidence = _response(
        [
            {
                "span_start": 25,
                "span_end": 30,
                "source": "замка",
                "type": "homograph",
                "confidence": {"x": 1},
            }
        ],
        replica_id=1,
    )
    analysis, errors = s.parse_analysis(bad_confidence, expected_replica_id=1, target_text=CASTLE)
    assert analysis is None
    assert s.ERROR_FIELD_TYPES in errors


def test_window_with_garbage_annotation_drops_only_its_replica():
    """В окне негодная аннотация стоит разбора одной реплики, а не всего запроса (§47)."""
    raw = json.dumps(
        {
            "schema_version": s.SCHEMA_VERSION,
            "replicas": [
                {"replica_id": 1, "items": []},
                {
                    "replica_id": 2,
                    "items": [
                        {"span_start": "abc", "span_end": 5, "source": "тест", "type": "homograph"}
                    ],
                },
            ],
        },
        ensure_ascii=False,
    )
    window, errors = s.parse_window_analysis(
        raw, replica_ids=[1, 2], target_texts={1: "Первый текст.", 2: "Второй текст."}
    )
    assert errors == []
    assert [analysis.replica_id for analysis in window.replicas] == [1]
    assert [entry["replica_id"] for entry in window.rejected] == [2]
    assert s.ERROR_FIELD_TYPES in window.rejected[0]["errors"]


def test_garbage_utterance_falls_back_and_keeps_the_analysis():
    """Подсказка о реплике — метаданные: её мусорный тип не отвергает разбор (§47)."""
    payload = _response(
        [
            {
                "span_start": 25,
                "span_end": 30,
                "source": "замка",
                "type": "homograph",
                "confidence": 0.9,
            }
        ],
        replica_id=1,
        extra={"utterance": {"class": "NORMAL", "context_dependency": "LOW", "emotion_confidence": {"x": 1}}},
    )
    analysis, errors = s.parse_analysis(payload, expected_replica_id=1, target_text=CASTLE)
    assert errors == []
    assert analysis is not None
    assert analysis.items[0].source == "замка"
    assert analysis.utterance.emotion == "", "непрочитанная эмоция — «модель не сказала»"


def test_settings_toggle_enables_analyzer_and_survives_restart(tmp_path, monkeypatch):
    """Галочка в интерфейсе включает анализ и переживает перезапуск.

    Без этого пользователь видел «анализатор выключен (LLM_ANALYZER_ENABLED=0)» и
    не мог ничего сделать из приложения: настройка жила только в окружении.
    """
    from backend.llm import settings_store

    path = tmp_path / "llm_settings.json"
    monkeypatch.setenv(settings_store.SETTINGS_ENV, str(path))

    # По умолчанию (окружение без переменных) анализатор выключен.
    assert llm.settings_from_env({}).enabled is False

    saved = settings_store.save_settings(
        {"enabled": True, "primary_model": "qwen3:4b-instruct-2507-q4_K_M", "num_ctx": 4096}
    )
    assert saved["enabled"] is True
    assert path.exists()

    # Файл перекрывает окружение — иначе настройка не пережила бы перезапуск.
    settings = llm.settings_from_env({"LLM_ANALYZER_ENABLED": "0"})
    assert settings.enabled is True
    assert settings.primary_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert settings.num_ctx == 4096

    # Выключение тоже сохраняется.
    settings_store.save_settings({"enabled": False})
    assert llm.settings_from_env({}).enabled is False


def test_settings_reject_broken_file_and_bad_values(tmp_path, monkeypatch):
    """Битый файл настроек и мусор в значениях не ломают приложение."""
    from backend.llm import settings_store

    path = tmp_path / "llm_settings.json"
    monkeypatch.setenv(settings_store.SETTINGS_ENV, str(path))
    path.write_text("{это не json", encoding="utf-8")
    assert settings_store.load_settings() == {}
    assert llm.settings_from_env({}).enabled is False

    saved = settings_store.save_settings(
        {"num_ctx": "не число", "context_replicas": -5, "primary_model": "  ", "enabled": True}
    )
    assert saved["enabled"] is True
    assert "num_ctx" not in saved and "context_replicas" not in saved
    assert "primary_model" not in saved


def test_llm_settings_endpoint_toggles_and_resets_analyzer(monkeypatch, tmp_path):
    """Ручка настроек включает анализ сразу, без перезапуска сервера."""
    import asyncio

    from test_projects_api import _client

    from backend.llm import settings_store

    monkeypatch.setenv(settings_store.SETTINGS_ENV, str(tmp_path / "llm.json"))
    llm.reset_analyzer()
    client = FakeOllamaClient(models=MODELS)
    monkeypatch.setattr(llm, "_analyzer", llm.LinguisticAnalyzer(settings=llm.AnalyzerSettings(enabled=False), client=client))

    async def scenario() -> dict:
        async with _client(monkeypatch) as api:
            before = (await api.get("/api/llm/status")).json()
            response = await api.post(
                "/api/llm/settings",
                json={"enabled": True, "primary_model": "qwen3:4b-instruct-2507-q8_0"},
            )
            assert response.status_code == 200, response.text
            after = (await api.get("/api/llm/status")).json()
            return {"before": before, "after": after, "body": response.json()}

    result = asyncio.run(scenario())
    assert result["before"]["enabled"] is False
    assert result["after"]["enabled"] is True
    assert result["after"]["settings"]["primary_model"] == "qwen3:4b-instruct-2507-q8_0"
    assert result["body"]["saved"]["enabled"] is True


# --- окно сцены (§6, §7): один запрос — разборы нескольких реплик --------------
SCENE = [
    {"replica_id": 7, "speaker": "Аня", "target_text": "Где вы были вчера?"},
    {"replica_id": 8, "speaker": "Борис", "target_text": CASTLE},
]


def _window_response(entries: list[dict]) -> str:
    """Ответ модели на окно: конверт с разбором на каждую реплику (§7)."""
    return json.dumps(
        {"schema_version": s.SCHEMA_VERSION, "replicas": entries}, ensure_ascii=False
    )


def _entry(replica_id: int, items: list[dict] | None = None) -> dict:
    return {
        "replica_id": replica_id,
        "items": items or [],
        "utterance": {"class": "NORMAL", "context_dependency": "LOW"},
    }


def test_window_schema_is_an_envelope_of_replica_analyses():
    """Схема окна — конверт: элемент тот же разбор, обязателен только `replica_id`."""
    schema = s.window_json_schema()
    assert schema["required"] == ["schema_version", "replicas"]
    entry = schema["properties"]["replicas"]["items"]
    assert entry["required"] == ["replica_id"]
    assert "utterance" in entry["properties"], "разбор реплики в окне — тот же объект"


def test_window_analysis_is_keyed_by_replica_id():
    """Ответ окна раскладывается по `replica_id`, а не по порядку в массиве."""
    raw = _window_response(
        [_entry(8, [{"span_start": 25, "span_end": 30, "source": "замка",
                     "type": "homograph", "confidence": 0.9, "needs_review": False}]),
         _entry(7)]
    )
    window, errors = s.parse_window_analysis(raw, replica_ids=[7, 8])
    assert errors == []
    by_id = window.by_replica()
    assert set(by_id) == {7, 8}
    assert by_id[8].items[0].source == "замка"
    assert by_id[7].items == ()


def test_window_rejects_unknown_replica_id_and_keeps_the_rest():
    """Чужой id не привязывается «по порядку»: он отброшен, остальные разборы целы."""
    raw = _window_response([_entry(99), _entry(7)])
    window, errors = s.parse_window_analysis(raw, replica_ids=[7, 8])
    assert errors == []
    assert window.unknown_ids == (99,)
    assert set(window.by_replica()) == {7}


def test_window_rejects_duplicate_replica_id():
    """Два разбора одной реплики — конфликт: первый принят, второй отброшен с причиной."""
    raw = _window_response([_entry(7), _entry(7)])
    window, _ = s.parse_window_analysis(raw, replica_ids=[7, 8])
    assert set(window.by_replica()) == {7}
    assert window.rejected_ids() == (7,)
    assert s.ERROR_DUPLICATE_REPLICA in window.rejected[0]["errors"]


def test_window_envelope_error_rejects_the_whole_response():
    """Ошибка конверта — это «ответ не про сцену»: он не применяется целиком."""
    wrong_version = json.dumps(
        {"schema_version": "0", "replicas": [_entry(7)]}, ensure_ascii=False
    )
    window, errors = s.parse_window_analysis(wrong_version, replica_ids=[7])
    assert window is None
    assert s.ERROR_SCHEMA_VERSION in errors

    extra = json.dumps(
        {"schema_version": s.SCHEMA_VERSION, "replicas": [_entry(7)], "text": "чужое"},
        ensure_ascii=False,
    )
    window, errors = s.parse_window_analysis(extra, replica_ids=[7])
    assert window is None
    assert s.ERROR_UNKNOWN_FIELD in errors


def test_window_accepts_single_object_from_small_model():
    """Маленькая модель отвечает одним разбором вместо конверта — он не теряется."""
    raw = _response([], replica_id=7)
    window, errors = s.parse_window_analysis(raw, replica_ids=[7, 8])
    assert errors == []
    assert set(window.by_replica()) == {7}


def test_analyze_window_makes_one_request_for_the_whole_scene():
    """Сцена — один запрос: в нём все реплики с говорящими, ответ — по каждой (§7)."""
    captured: dict = {}

    def responder(case):
        captured.update(case)
        return _window_response([_entry(item["replica_id"]) for item in case["replicas"]])

    analyzer, client = _analyzer(responder)
    results = analyzer.analyze_window(scene=SCENE)

    assert len(client.calls) == 1, "окно — один запрос, а не запрос на реплику"
    assert set(results) == {7, 8}
    assert all(item.status == llm.STATUS_READY for item in results.values())
    assert [item["replica_id"] for item in captured["replicas"]] == [7, 8]
    assert [item["speaker"] for item in captured["replicas"]] == ["Аня", "Борис"]
    assert captured["replicas"][1]["target_text"] == CASTLE
    # Разбор окна принадлежит оконному prompt'у: иначе кеш смешал бы его с одиночным.
    assert results[7].prompt_version == analyzer.window_prompt.version
    assert analyzer.window_prompt.version != analyzer.prompt.version
    assert results[7].context_hash == results[8].context_hash, "контекст у окна общий"


def test_analyze_window_replica_failure_does_not_cancel_other_replicas():
    """Отказ одной реплики не отменяет окно: §47 — «reject field, не падать всем проектом»."""
    def responder(case):
        entries = []
        for item in case["replicas"]:
            items = []
            if item["replica_id"] == 7:
                items = [{"span_start": 0, "span_end": 1, "source": "Г",
                          "type": "выдуманный", "confidence": 0.5, "needs_review": False}]
            entries.append(_entry(item["replica_id"], items))
        return _window_response(entries)

    analyzer, _ = _analyzer(responder)
    results = analyzer.analyze_window(scene=SCENE)

    assert results[7].status == llm.STATUS_FAILED
    assert s.ERROR_UNKNOWN_TYPE in results[7].error
    assert results[8].status == llm.STATUS_READY


def test_analyze_window_missing_replica_is_visible_failure():
    """Реплика, о которой модель не ответила, получает отказ с причиной, а не тишину."""
    def responder(case):
        return _window_response([_entry(8)])

    analyzer, _ = _analyzer(responder)
    results = analyzer.analyze_window(scene=SCENE)

    assert results[7].status == llm.STATUS_FAILED
    assert "не разобрана" in results[7].error
    assert results[8].status == llm.STATUS_READY


def test_analyze_window_is_disabled_without_calling_the_model():
    """Выключенный Analyzer не ходит в модель, но состояние видно по каждой реплике."""
    captured: list = []

    def responder(case):
        captured.append(case)
        return _window_response([])

    analyzer, client = _analyzer(responder, settings=_settings(enabled=False))
    results = analyzer.analyze_window(scene=SCENE)

    assert client.calls == []
    assert captured == []
    assert all(item.status == llm.STATUS_DISABLED for item in results.values())

