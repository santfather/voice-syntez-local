"""Фаза 1 Task 1: клиент Ollama, схемы ответа, версии, prompt и fake client.

Модели и сеть не нужны: HTTP-слой подменяется открывалкой, отвечающей заранее
заготовленными телами (`opener=`), а ошибки соединения проверяются исключениями
`urllib`. Так обычный `pytest` не поднимает LLM (требование §13 Task 1).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse

import pytest

from backend.llm import fake_client, ollama_client, prompt, versioning
from backend.llm import schemas as s


class _FakeResponse:
    """Ответ HTTP-слоя: байты для `.read()` и строки для потокового чтения."""

    def __init__(self, payload) -> None:
        self._payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def __iter__(self):
        return iter(self._payload.splitlines(keepends=True))

    def close(self) -> None:
        self.closed = True


def _opener(routes: dict[str, object]):
    """Открывалка по путям: значение — тело ответа или исключение."""

    def open_(request, timeout=None):
        path = urllib.parse.urlparse(request.full_url).path
        if path not in routes:
            raise AssertionError(f"неожиданный запрос: {path}")
        payload = routes[path]
        if isinstance(payload, Exception):
            raise payload
        if callable(payload):
            return _FakeResponse(payload(json.loads(request.data or b"{}")))
        return _FakeResponse(payload)

    return open_


TAGS = {
    "models": [
        {
            "name": "qwen3:8b",
            "digest": "abc123",
            "size": 5 * 1024**3,
            "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
        },
        {"name": "gemma3:4b", "digest": "def456", "size": 3 * 1024**3, "details": {}},
    ]
}


def _ndjson(*chunks: dict, done: dict | None = None) -> bytes:
    lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    lines.append(json.dumps(done or {"done": True, "eval_count": 42, "eval_duration": 2_000_000_000}))
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- клиент ---------------------------------------------------------------------
def test_ollama_client_health_when_unavailable():
    """Недоступная Ollama — состояние, а не исключение: UI обязан отвечать."""
    client = ollama_client.OllamaClient(
        opener=_opener({"/api/version": urllib.error.URLError("connection refused")})
    )
    status = client.health()
    assert status.available is False
    assert "недоступен" in status.error
    assert status.models == ()


def test_ollama_client_models_parsed_with_digest_and_size():
    client = ollama_client.OllamaClient(
        opener=_opener({"/api/version": {"version": "0.34.1"}, "/api/tags": TAGS})
    )
    status = client.health()
    assert status.available is True
    assert status.version == "0.34.1"
    assert [model.tag for model in status.models] == ["qwen3:8b", "gemma3:4b"]
    first = status.models[0]
    assert first.digest == "abc123"
    assert first.size_gb == pytest.approx(5.0)
    assert first.quantization == "Q4_K_M"


def test_ollama_client_requires_model_with_pull_hint():
    client = ollama_client.OllamaClient(opener=_opener({"/api/tags": TAGS}))
    with pytest.raises(ollama_client.OllamaModelMissingError, match="ollama pull qwen3:14b"):
        client.require_model("qwen3:14b")
    # Тег с :latest и без — одна и та же модель.
    assert client.has_model("gemma3:4b:latest") is True


def test_ollama_client_chat_streams_and_counts_tokens():
    captured: dict = {}

    def chat_route(payload):
        captured.update(payload)
        return _ndjson(
            {"message": {"content": '{"items":'}, "done": False},
            {"message": {"content": " []}"}, "done": False},
            done={"done": True, "eval_count": 100, "eval_duration": 2_000_000_000},
        )

    client = ollama_client.OllamaClient(
        opener=_opener({"/api/chat": chat_route})
    )
    result = client.chat(
        "qwen3:8b",
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        schema={"type": "object"},
        options={"temperature": 0, "num_ctx": 8192},
        keep_alive="5m",
    )
    assert result.text == '{"items": []}'
    assert result.eval_count == 100
    assert result.tokens_per_second == pytest.approx(50.0)
    assert captured["stream"] is True
    assert captured["format"] == {"type": "object"}
    assert captured["options"]["num_ctx"] == 8192
    assert captured["keep_alive"] == "5m"


def test_ollama_client_marks_cancel_and_timeout_separately():
    """Отмена, таймаут и «демон недоступен» — разные случаи с разными советами."""
    client = ollama_client.OllamaClient(
        opener=_opener({"/api/chat": _ndjson({"message": {"content": "x"}, "done": False})})
    )
    with pytest.raises(ollama_client.OllamaError, match="отменён"):
        client.chat("qwen3:8b", [{"role": "user", "content": "u"}], cancel=lambda: True)

    timeout_client = ollama_client.OllamaClient(
        opener=_opener({"/api/chat": TimeoutError("timed out")})
    )
    with pytest.raises(ollama_client.OllamaTimeoutError):
        timeout_client.chat("qwen3:8b", [{"role": "user", "content": "u"}])

    down_client = ollama_client.OllamaClient(
        opener=_opener({"/api/chat": urllib.error.URLError("connection refused")})
    )
    with pytest.raises(ollama_client.OllamaUnavailableError):
        down_client.chat("qwen3:8b", [{"role": "user", "content": "u"}])


def test_ollama_client_refuses_non_http_url(monkeypatch):
    """`file://` — не адрес демона: клиент не читает локальные файлы и не молчит."""
    monkeypatch.setenv(ollama_client.OLLAMA_URL_ENV, "file:///etc/passwd")
    # Пустая карта маршрутов: любой запрос — неожиданный, то есть до сети дело не дошло.
    client = ollama_client.OllamaClient(opener=_opener({}))
    with pytest.raises(ollama_client.OllamaUnavailableError, match="http"):
        client.version()
    # И это состояние, а не исключение для UI: экран статуса обязан ответить.
    status = client.health()
    assert status.available is False
    assert "http" in status.error


def test_ollama_client_unload_confirms_actual_release():
    """Выгрузка подтверждается по /api/ps, а не по факту отправки запроса."""
    state = {"loaded": True}

    def ps_route(_payload=None):
        return {"models": [{"name": "qwen3:8b"}] if state["loaded"] else []}

    def chat_route(payload):
        if payload.get("keep_alive") == 0:
            state["loaded"] = False
            return {"done": True}
        return _ndjson({"message": {"content": "ok"}, "done": False})

    client = ollama_client.OllamaClient(
        opener=_opener({"/api/chat": chat_route, "/api/ps": ps_route})
    )
    assert client.loaded_models() == ["qwen3:8b"]
    assert client.unload("qwen3:8b") is True
    assert client.loaded_models() == []


def test_ollama_client_unload_reports_when_model_stays():
    """Модель осталась загруженной — честный False, а не «выгружено»."""
    client = ollama_client.OllamaClient(
        opener=_opener(
            {
                "/api/chat": {"done": True},
                "/api/ps": {"models": [{"name": "qwen3:8b"}]},
            }
        )
    )
    assert client.unload("qwen3:8b") is False


# --- схема ответа ---------------------------------------------------------------
def _valid_response(**overrides) -> str:
    payload = {
        "schema_version": s.SCHEMA_VERSION,
        "replica_id": 17,
        "items": [
            {
                "span_start": 11,
                "span_end": 16,
                "source": "замок",
                "type": "homograph",
                "meaning": "lock",
                "suggested_form": "замок",
                "confidence": 0.98,
                "needs_review": False,
                "reason_code": "CONTEXT_DISAMBIGUATION",
            }
        ],
        "utterance": {"class": "NORMAL", "context_dependency": "LOW"},
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


TARGET = "Он поменял замок на входной двери."


def test_llm_response_schema_validation():
    analysis, errors = s.parse_analysis(
        _valid_response(), expected_replica_id=17, target_text=TARGET
    )
    assert errors == []
    assert analysis is not None
    assert analysis.items[0].source == "замок"
    assert analysis.items[0].type == s.TYPE_HOMOGRAPH
    assert analysis.utterance.cls == "NORMAL"
    assert analysis.to_dict()["schema_version"] == s.SCHEMA_VERSION


def test_llm_response_rejects_changed_source_span():
    """`source` не совпал с текстом в этих границах — ответ не применяется."""
    broken = json.loads(_valid_response())
    broken["items"][0]["source"] = "замок на"
    analysis, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert analysis is None
    assert errors == [s.ERROR_SOURCE_MISMATCH]


def test_llm_response_rejects_out_of_bounds_span():
    broken = json.loads(_valid_response())
    broken["items"][0]["span_end"] = len(TARGET) + 5
    analysis, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert analysis is None
    assert s.ERROR_SPAN_BOUNDS in errors


def test_llm_response_rejects_unknown_replica():
    analysis, errors = s.parse_analysis(
        _valid_response(), expected_replica_id=99, target_text=TARGET
    )
    assert analysis is None
    assert errors == [s.ERROR_UNKNOWN_REPLICA]


def test_llm_response_rejects_unknown_type_and_reason():
    broken = json.loads(_valid_response())
    broken["items"][0]["type"] = "магия"
    broken["items"][0]["reason_code"] = "НЕИЗВЕСТНО"
    _, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert s.ERROR_UNKNOWN_TYPE in errors
    assert s.ERROR_UNKNOWN_REASON in errors


def test_llm_response_rejects_bad_confidence():
    broken = json.loads(_valid_response())
    broken["items"][0]["confidence"] = 1.5
    _, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert errors == [s.ERROR_CONFIDENCE_RANGE]


def test_llm_response_accepts_schema_version_noise():
    """`"1.0"`, `"1"` и `"v1"` — одна и та же версия схемы, а `"2"` — другая.

    Модели почти всегда пишут `"1.0"`: считать это ошибкой значило бы мерить
    педантичность формата, а не русский язык. Настоящую смену версии разбор обязан
    ловить.
    """
    # Пустая версия приравнивается к текущей: отсутствие поля в контракте
    # допускалось и раньше, менять это здесь незачем.
    current = s.SCHEMA_VERSION
    for value in (current, f"{current}.0", f"v{current}", f" {current}.0 ", ""):
        payload = json.loads(_valid_response())
        payload["schema_version"] = value
        analysis, errors = s.parse_analysis(
            json.dumps(payload, ensure_ascii=False),
            expected_replica_id=17,
            target_text=TARGET,
        )
        assert errors == [], value
        assert analysis is not None
        assert analysis.schema_version == current

    # Прошлая версия схемы и «почти текущая» — разные контракты: молча принять их
    # значило бы разобрать ответ по полям, которых в нём может не быть.
    for value in (str(int(current) - 1), f"{current}.1"):
        payload = json.loads(_valid_response())
        payload["schema_version"] = value
        analysis, errors = s.parse_analysis(
            json.dumps(payload, ensure_ascii=False),
            expected_replica_id=17,
            target_text=TARGET,
        )
        assert analysis is None, value
        assert s.ERROR_SCHEMA_VERSION in errors, value


def test_llm_response_rejects_rewritten_text_field():
    """Попытка вернуть переписанный текст не проходит валидацию.

    Схема ответа поля для текста не содержит, но модель может добавить его сама.
    Такой ответ обязан быть отвергнут целиком — иначе однажды чужой код возьмёт
    текст из ответа модели, и «свободный rewrite» вернётся через чёрный ход.
    """
    for field in ("text", "final_text", "normalized"):
        broken = json.loads(_valid_response())
        broken[field] = "Он сменил замок на входной двери."
        analysis, errors = s.parse_analysis(
            json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
        )
        assert analysis is None, field
        assert errors == [s.ERROR_UNKNOWN_FIELD], field


def test_llm_response_rejects_unknown_item_and_utterance_fields():
    broken = json.loads(_valid_response())
    broken["items"][0]["replacement"] = "замо́к"
    broken["utterance"]["tone"] = "calm"
    _, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert errors == [s.ERROR_UNKNOWN_FIELD]


def test_analysis_rejects_conflicting_overlapping_patches():
    """Две аннотации на один и тот же участок — конфликт, а не «две правки»."""
    broken = json.loads(_valid_response())
    broken["items"].append(dict(broken["items"][0], meaning="castle"))
    _, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert s.ERROR_OVERLAP in errors


def test_malformed_json_is_not_applied():
    for raw in ("", "не json", "[1, 2]", "{сломано"):
        analysis, errors = s.parse_analysis(raw, expected_replica_id=17, target_text=TARGET)
        assert analysis is None
        assert errors and errors[0] in (s.ERROR_NOT_JSON, s.ERROR_NOT_OBJECT)


def test_schema_version_mismatch_is_rejected():
    _, errors = s.parse_analysis(
        _valid_response(schema_version="99"), expected_replica_id=17, target_text=TARGET
    )
    assert errors == [s.ERROR_SCHEMA_VERSION]


def test_analysis_json_schema_has_no_rewrite_field():
    """В схеме ответа нет поля для нового текста — rewrite невозможен по протоколу."""
    schema = s.analysis_json_schema()
    properties = set(schema["properties"])
    assert properties == {"schema_version", "replica_id", "items", "utterance"}
    item_properties = set(schema["properties"]["items"]["items"]["properties"])
    assert "text" not in item_properties
    assert "rewritten_text" not in item_properties
    assert item_properties >= {"span_start", "span_end", "source", "type", "needs_review"}


def test_json_in_markdown_fence_is_accepted():
    """Забор вокруг JSON — не повод терять корректный ответ."""
    raw = f"```json\n{_valid_response()}\n```"
    analysis, errors = s.parse_analysis(raw, expected_replica_id=17, target_text=TARGET)
    assert errors == []
    assert analysis is not None


def test_llm_response_rejects_wall_of_text():
    """Гигантский ответ отбрасывается, а не разворачивается в разбор целиком."""
    huge = "x" * (s.MAX_RESPONSE_CHARS + 1)
    analysis, errors = s.parse_analysis(huge, expected_replica_id=17, target_text=TARGET)
    assert analysis is None
    assert errors == [s.ERROR_TOO_LARGE]
    # Тот же порог и на окне: ответ не должен успеть стать объектом ни на одном пути.
    window, window_errors = s.parse_window_analysis(
        huge, replica_ids=[17], target_texts={17: TARGET}
    )
    assert window is None
    assert window_errors == [s.ERROR_TOO_LARGE]


def test_llm_response_rejects_hundreds_of_annotations():
    """Сотни аннотаций к одной реплике — мусорный ответ, он не применяется."""
    broken = json.loads(_valid_response())
    broken["items"] = broken["items"] * (s.MAX_ANNOTATIONS + 1)
    analysis, errors = s.parse_analysis(
        json.dumps(broken, ensure_ascii=False), expected_replica_id=17, target_text=TARGET
    )
    assert analysis is None
    assert errors == [s.ERROR_TOO_MANY_ITEMS]


# --- prompt и версии ------------------------------------------------------------
def test_prompt_loads_with_version_and_placeholder():
    template = prompt.load_prompt()
    assert template.version == versioning.PROMPT_VERSION
    assert prompt.PLACEHOLDER in template.user_template
    assert prompt.SCHEMA_PLACEHOLDER in template.system
    schema_text = json.dumps(s.analysis_json_schema(), ensure_ascii=False, indent=2)
    messages = template.messages(
        {"replica_id": 1, "target_text": "Да."}, schema_json=schema_text
    )
    assert messages[0]["role"] == "system"
    assert "ТОЛЬКО JSON" in messages[0]["content"]
    assert '"target_text": "Да."' in messages[1]["content"]
    # Схема доходит до модели целиком и ровно та же, что проверяет backend: иначе
    # модель учили бы одной схеме, а принимали другую.
    assert prompt.SCHEMA_PLACEHOLDER not in messages[0]["content"]
    assert '"span_start"' in messages[0]["content"]
    assert "homograph" in messages[0]["content"]
    assert "CONTEXT_DISAMBIGUATION" in messages[0]["content"]


def test_prompt_requires_case_and_schema_placeholders():
    with pytest.raises(ValueError, match="версия"):
        prompt.parse_prompt("[system]\ns\n[user]\n{{case}}", name="x")
    with pytest.raises(ValueError, match="schema"):
        prompt.parse_prompt(
            "# version: 1\n[system]\nбез плейсхолдера\n[user]\n{{case}}", name="x"
        )
    with pytest.raises(ValueError, match="case"):
        prompt.parse_prompt(
            "# version: 1\n[system]\n{{schema}}\n[user]\nбез плейсхолдера", name="x"
        )


def test_prompt_without_placeholder_is_rejected():
    with pytest.raises(ValueError, match="placeholder|\\{\\{case\\}\\}"):
        prompt.parse_prompt("# version: 1\n[system]\ns\n[user]\nбез подстановки", name="x")


def test_model_metadata_is_saved():
    """§4: в паспорте прогона есть тег, digest, размер, контекст и параметры."""
    metadata = versioning.build_run_metadata(
        model_tag="qwen3:8b",
        ollama_version="0.34.1",
        model_digest="abc123",
        model_size_gb=5.2,
        context=8192,
        temperature=0.0,
        seed=0,
        options={"num_predict": 512},
        include_machine=True,
    )
    payload = metadata.to_dict()
    assert payload["model_tag"] == "qwen3:8b"
    assert payload["model_digest"] == "abc123"
    assert payload["model_size_gb"] == 5.2
    assert payload["context"] == 8192
    assert payload["schema_version"] == s.SCHEMA_VERSION
    assert payload["prompt_version"] == versioning.PROMPT_VERSION
    assert payload["total_ram_gb"] > 0
    assert payload["chip"]


def test_prompt_version_is_saved():
    metadata = versioning.build_run_metadata(model_tag="m", include_machine=False)
    payload = metadata.to_dict()
    assert payload["prompt_version"] == versioning.PROMPT_VERSION
    assert payload["dataset_version"] == "1"
    assert payload["benchmark_version"] == versioning.BENCHMARK_VERSION
    assert payload["timestamp"]


def test_run_metadata_records_git_commit_when_available():
    commit = versioning.git_commit()
    assert commit == "" or len(commit) >= 7


# --- fake client ----------------------------------------------------------------
def test_fake_client_answers_with_gold_annotations():
    case = {
        "replica_id": 17,
        "target_text": TARGET,
        "expected": json.loads(_valid_response())["items"],
    }
    client = fake_client.FakeOllamaClient()
    result = client.chat("fake:perfect", [{"role": "user", "content": json.dumps(case, ensure_ascii=False)}])
    analysis, errors = s.parse_analysis(
        result.text, expected_replica_id=17, target_text=TARGET
    )
    assert errors == []
    assert analysis is not None
    assert analysis.items[0].source == "замок"


def test_fake_client_can_simulate_unavailable_and_setup():
    down = fake_client.FakeOllamaClient(available=False, error="нет демона")
    assert down.health().available is False
    with pytest.raises(ollama_client.OllamaUnavailableError):
        down.chat("fake:perfect", [{"role": "user", "content": "{}"}])

    up = fake_client.FakeOllamaClient(responses_by_model={"qwen3:8b": '{"items": []}'})
    assert up.has_model("qwen3:8b") is True
    assert up.chat("qwen3:8b", [{"role": "user", "content": "{}"}]).text == '{"items": []}'
    assert up.loaded_models() == ["qwen3:8b"]
    assert up.unload("qwen3:8b") is True
    assert up.loaded_models() == []


def test_ollama_client_reads_model_capabilities():
    """Паспорт модели (`/api/show`) нужен, чтобы понять, умеет ли модель thinking.

    Ошибка здесь тихая: если capabilities не читаются, runner не отключает скрытое
    рассуждение, и модель тратит сотни токенов на служебные размышления (у
    qwen3:8b — 2186 токенов и 102 c на кейс вместо 149 токенов и 6 c).
    """
    client = ollama_client.OllamaClient(
        opener=_opener(
            {
                "/api/show": lambda payload: {
                    "capabilities": ["completion", "thinking"]
                    if payload.get("model") == "qwen3:8b"
                    else ["completion", "vision"]
                }
            }
        )
    )
    assert client.capabilities("qwen3:8b") == ("completion", "thinking")
    assert client.capabilities("gemma3:4b") == ("completion", "vision")
    # Демон ответил не тем, что ожидалось — пустой список, а не падение.
    broken = ollama_client.OllamaClient(
        opener=_opener({"/api/show": urllib.error.URLError("нет соединения")})
    )
    assert broken.capabilities("qwen3:8b") == ()


def test_ollama_client_sends_think_only_when_asked():
    """`think` уходит в запрос только при явном значении: старые модели его не знают."""
    chat_route = _ndjson(
        {"message": {"content": "{}"}, "done": False},
        done={"done": True, "eval_count": 1, "eval_duration": 1_000_000_000},
    )
    sent: list[dict] = []

    def capture(request, timeout=None):
        sent.append(json.loads(request.data or b"{}"))
        return _FakeResponse(chat_route)

    client = ollama_client.OllamaClient(opener=capture)
    client.chat("qwen3:8b", [{"role": "user", "content": "x"}], think=False)
    client.chat("gemma3:4b", [{"role": "user", "content": "x"}])
    assert sent[0]["think"] is False
    assert "think" not in sent[1]
