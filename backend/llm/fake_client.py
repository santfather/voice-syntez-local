"""Подставной клиент Ollama: тесты и `--dry-run` без сети, моделей и памяти.

Два применения, и оба важны:

* **pytest** — обычный прогон не должен поднимать LLM (требование §13 Task 1):
  тесты работают с этим клиентом и проверяют схемы, метрики, версии, память и
  последовательность запуска;
* **benchmark `--dry-run`** — прогон полного пути (prompt → клиент → разбор →
  метрики → отчёт) на «идеальной» модели: она отвечает gold-аннотациями кейса.
  Это даёт верхнюю границу метрик и проверяет, что runner и метрики считают то,
  что должны, ещё до скачивания моделей.

Клиент повторяет интерфейс настоящего (`backend.llm.ollama_client.OllamaClient`),
поэтому подмена не требует ветвлений в коде runner'а.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable

from .ollama_client import (
    ChatResult,
    OllamaModel,
    OllamaStatus,
    OllamaUnavailableError,
)
from .schemas import SCHEMA_VERSION

DEFAULT_MODELS = (
    OllamaModel(tag="fake:perfect", digest="fake0001", size_bytes=1024**3,
                parameter_size="4B", quantization="Q4_K_M"),
)


class FakeOllamaClient:
    """Ollama без Ollama: ответы задаются заранее или берутся из кейса.

    `responder(case)` получает payload кейса и возвращает текст ответа. По
    умолчанию отвечает «идеально»: gold-аннотациями из `gold_by_replica` (по
    числовому id реплики) или из самого payload кейса, если он их содержит. Так
    `--dry-run` получает верхнюю границу метрик, а gold в prompt настоящего
    прогона не попадает: runner кладёт его только в подставной клиент.
    """

    def __init__(
        self,
        *,
        models: Iterable[OllamaModel] | None = None,
        version: str = "0.34.0-fake",
        responder: Callable[[dict], str] | None = None,
        responses_by_model: dict[str, str] | None = None,
        gold_by_replica: dict[int, list[dict]] | None = None,
        available: bool = True,
        error: str = "",
        latency_sec: float = 0.0,
        fail_on_call: int | None = None,
        unload_works: bool = True,
    ) -> None:
        self._responses = dict(responses_by_model or {})
        self._gold = dict(gold_by_replica or {})
        if models is not None:
            self._models = list(models)
        elif self._responses:
            # Заготовленные ответы объявляют и список моделей: тест, который
            # задаёт «так отвечает qwen3:8b», не должен ещё и перечислять модели.
            self._models = [OllamaModel(tag=tag) for tag in self._responses]
        else:
            self._models = list(DEFAULT_MODELS)
        self._version = version
        self._responder = responder
        self._available = available
        self._error = error
        self._latency = latency_sec
        self._fail_on_call = fail_on_call
        self._unload_works = unload_works
        self.calls: list[dict] = []
        self.unloaded: list[str] = []
        self._loaded: list[str] = []

    # -- совместимость с настоящим клиентом -------------------------------------
    def version(self) -> str:
        if not self._available:
            raise OllamaUnavailableError(self._error or "fake: демон недоступен")
        return self._version

    def models(self) -> list[OllamaModel]:
        if not self._available:
            raise OllamaUnavailableError(self._error or "fake: демон недоступен")
        return list(self._models)

    def health(self) -> OllamaStatus:
        if not self._available:
            return OllamaStatus(available=False, error=self._error or "fake: демон недоступен")
        return OllamaStatus(available=True, version=self._version, models=tuple(self._models))

    def has_model(self, tag: str) -> bool:
        return any(model.tag == tag for model in self._models)

    def model_info(self, tag: str) -> OllamaModel | None:
        for model in self._models:
            if model.tag == tag:
                return model
        return None

    def require_model(self, tag: str) -> OllamaModel:
        model = self.model_info(tag)
        if model is None:
            from .ollama_client import OllamaModelMissingError

            raise OllamaModelMissingError(f"Модель «{tag}» не найдена локально.")
        return model

    def running_models(self) -> list[dict]:
        return [{"name": tag, "size": 1024**3} for tag in self._loaded]

    def loaded_models(self, *, exclude: str | None = None) -> list[str]:
        return [tag for tag in self._loaded if tag != exclude]

    def unload(self, tag: str) -> bool:
        self.unloaded.append(tag)
        if not self._unload_works:
            return False
        self._loaded = [item for item in self._loaded if item != tag]
        return True

    def unload_others(self, keep: str | None = None) -> list[str]:
        unloaded = [tag for tag in self.loaded_models(exclude=keep) if self.unload(tag)]
        return unloaded

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        schema: dict | None = None,
        options: dict | None = None,
        keep_alive=None,
        timeout: float | None = None,
        cancel: Callable[[], bool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        self.calls.append(
            {"model": model, "messages": messages, "schema": schema, "options": dict(options or {})}
        )
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise OllamaUnavailableError("fake: сбой на вызове")
        if not self._available:
            raise OllamaUnavailableError(self._error or "fake: демон недоступен")
        if cancel is not None and cancel():
            raise OllamaUnavailableError("fake: запрос отменён")
        if self._latency:
            time.sleep(self._latency)
        if model not in self._loaded:
            # Настоящая Ollama держит модель загруженной после запроса — здесь это
            # тоже видно, чтобы memory policy проверялась на честном состоянии.
            self._loaded.append(model)

        case = _case_from_messages(messages, self._responses)
        text = self._respond_for(model, case)
        if on_token is not None:
            on_token(text)
        return ChatResult(
            text=text,
            model=model,
            duration_sec=self._latency,
            eval_count=max(len(text) // 4, 1),
            prompt_eval_count=sum(len(str(m.get("content", ""))) for m in messages) // 4,
            raw={"eval_duration": int(max(self._latency, 0.001) * 1e9), "done": True},
        )

    # -- ответы -----------------------------------------------------------------
    def _respond_for(self, model: str, case: dict) -> str:
        if self._responder is not None:
            return self._responder(case)
        preset = self._responses.get(model)
        if preset is not None:
            return _shape_response(preset, case)
        gold = case.get("expected") or self._gold.get(int(case.get("replica_id") or -1), [])
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "replica_id": case.get("replica_id", 0),
                "items": gold,
                "utterance": {"class": "NORMAL", "context_dependency": "LOW"},
            },
            ensure_ascii=False,
        )


def _shape_response(preset: str | dict, case: dict) -> str:
    """Подставляет в заготовку ответа `replica_id` кейса.

    Заготовки в тестах пишутся как «ответ модели» и не должны знать, какой id
    придёт: id подставляет клиент — так же, как это делает настоящая модель по
    инструкции prompt'а.
    """
    if isinstance(preset, dict):
        payload = dict(preset)
        payload.setdefault("replica_id", case.get("replica_id", 0))
        payload.setdefault("schema_version", SCHEMA_VERSION)
        return json.dumps(payload, ensure_ascii=False)
    return preset


def _case_from_messages(messages: list[dict], responses: dict[str, list[str]]) -> dict:
    """Достаёт JSON кейса из пользовательского сообщения.

    Runner передаёт кейс в prompt как JSON — подставной клиент читает его обратно,
    чтобы ответить gold-аннотациями. Это ровно тот путь, которым пойдёт настоящая
    модель: никакого «второго канала» с кейсом у клиента нет.
    """
    for message in reversed(messages):
        content = str(message.get("content") or "")
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
            except ValueError:
                continue
            if isinstance(payload, dict) and "target_text" in payload:
                return payload
    return {}
