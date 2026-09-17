"""Клиент локального Ollama: единственное место, где проект говорит с LLM.

Зачем отдельный слой. HTTP-вызовы, размазанные по роутам, инструментам и UI,
ломаются по-разному в каждом месте: где-то нет таймаута, где-то не выгружается
модель, где-то ошибка соединения выглядит как пустой ответ. Здесь собрано всё
общение с Ollama: health, список моделей, чат со структурированным выходом,
выгрузка и диагностика.

Почему HTTP API, а не CLI. Приложение не должно зависеть от программы в `PATH` и
от её вывода: Ollama и так поднимает локальный демон, а CLI — лишь его клиент.
Плюс HTTP даёт то, чего у CLI нет: таймауты, потоковый ответ и явную выгрузку
модели (`keep_alive: 0`).

Почему стандартная библиотека. Ради одного POST'а новая зависимость не нужна:
`urllib.request` умеет и таймаут, и потоковое чтение NDJSON, а требования проекта
остаются прежними. Ollama — **опциональная** локальная зависимость: отсутствие
демона не должно ломать TTS, поэтому все ошибки соединения здесь превращаются в
`OllamaUnavailableError`, а вызывающий решает, что делать (в Analyzer это откат на
детерминированный путь, в benchmark — остановка прогона).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("tts.llm.ollama")

OLLAMA_URL_ENV = "TTS_OLLAMA_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
# Таймаут одного запроса. Модель на 8B при 18 ГБ отвечает секунды, но первый
# прогон включает загрузку весов в память — запас нужен.
DEFAULT_TIMEOUT_SEC = float(os.environ.get("TTS_OLLAMA_TIMEOUT_SEC", "180"))
# Health-check обязан быть быстрым: его дёргают и `/api/llm/status`, и UI, и
# benchmark перед каждым прогоном.
HEALTH_TIMEOUT_SEC = float(os.environ.get("TTS_OLLAMA_HEALTH_TIMEOUT_SEC", "3"))


class OllamaError(RuntimeError):
    """Общая база ошибок клиента: у каждой есть машинный `kind`."""

    kind = "OLLAMA_ERROR"


class OllamaUnavailableError(OllamaError):
    """Демон Ollama недоступен: не запущен, другой порт, нет сети до него."""

    kind = "OLLAMA_UNAVAILABLE"


class OllamaTimeoutError(OllamaError):
    """Ответа не дождались: запрос оборван по таймауту."""

    kind = "OLLAMA_TIMEOUT"


class OllamaModelMissingError(OllamaError):
    """Запрошенной модели нет локально: её нужно скачать (`ollama pull`)."""

    kind = "OLLAMA_MODEL_MISSING"


class OllamaResponseError(OllamaError):
    """Демон ответил, но ответ не разобран или содержит ошибку."""

    kind = "OLLAMA_RESPONSE"


@dataclass(frozen=True)
class OllamaModel:
    """Локально доступная модель: тег, digest и размер — для отчёта и версий."""

    tag: str
    digest: str = ""
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 2)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class OllamaStatus:
    """Итог health-check: доступен ли демон, его версия и что лежит локально."""

    available: bool
    version: str = ""
    error: str = ""
    models: tuple[OllamaModel, ...] = ()

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "version": self.version,
            "error": self.error,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True)
class ChatResult:
    """Ответ модели: текст, счётчики и тайминги — всё, что нужно метрикам."""

    text: str
    model: str
    duration_sec: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Скорость генерации именно ответа: время загрузки в неё не входит."""
        # `eval_duration` — время генерации в наносекундах; если демон его не
        # прислал, честнее вернуть 0, чем считать по общему времени запроса.
        eval_ns = int(self.raw.get("eval_duration") or 0)
        if eval_ns <= 0 or not self.eval_count:
            return 0.0
        return round(self.eval_count / (eval_ns / 1e9), 2)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "duration_sec": round(self.duration_sec, 3),
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "tokens_per_second": self.tokens_per_second,
        }


class OllamaClient:
    """Синхронный клиент Ollama. Один экземпляр на процесс — и один поток."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(OLLAMA_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = DEFAULT_TIMEOUT_SEC if timeout is None else float(timeout)
        # Открывалка подменяется в тестах: HTTP-мок без сети и без сокетов.
        self._open = opener or urllib.request.urlopen

    # -- низкий уровень --------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        path: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
        stream: bool = False,
    ):
        """Один запрос к демону. Ошибки соединения — `OllamaUnavailableError`.

        Таймаут отделён от «демон недоступен»: это разные советы пользователю
        (подождать/уменьшить модель против запустить `ollama serve`).
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            response = self._open(request, timeout=self.timeout if timeout is None else timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001 — тело ошибки может быть недоступно
                detail = ""
            raise OllamaResponseError(f"Ollama ответил {exc.code}: {detail or exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaTimeoutError(
                f"Ollama не ответил за {self.timeout if timeout is None else timeout:.0f} с"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"Ollama недоступен по {self.base_url}: {exc.reason}"
            ) from exc
        except OSError as exc:  # соединение оборвано на уровне сокета
            raise OllamaUnavailableError(f"Ollama недоступен по {self.base_url}: {exc}") from exc
        return response

    @staticmethod
    def _read_json(response) -> dict:
        raw = response.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise OllamaResponseError(f"Ollama вернул не JSON: {raw[:200]!r}") from exc

    # -- health и модели -------------------------------------------------------
    def version(self) -> str:
        payload = self._read_json(self._request("/api/version", timeout=HEALTH_TIMEOUT_SEC))
        return str(payload.get("version") or "")

    def models(self) -> list[OllamaModel]:
        """/api/tags: что лежит локально, с digest и размером."""
        payload = self._read_json(self._request("/api/tags", timeout=HEALTH_TIMEOUT_SEC))
        result: list[OllamaModel] = []
        for item in payload.get("models") or []:
            details = item.get("details") or {}
            tag = str(item.get("name") or item.get("model") or "")
            if not tag:
                continue
            result.append(
                OllamaModel(
                    tag=tag,
                    digest=str(item.get("digest") or ""),
                    size_bytes=int(item.get("size") or 0),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                )
            )
        return result

    def health(self) -> OllamaStatus:
        """Доступность демона и список моделей — без исключений.

        Роут статуса и UI обязаны отвечать всегда: недоступная Ollama — это
        состояние, а не ошибка приложения.
        """
        try:
            version = self.version()
        except OllamaError as exc:
            return OllamaStatus(available=False, error=str(exc))
        try:
            models = tuple(self.models())
        except OllamaError as exc:
            return OllamaStatus(available=True, version=version, error=str(exc))
        return OllamaStatus(available=True, version=version, models=models)

    def has_model(self, tag: str) -> bool:
        wanted = _normalize_tag(tag)
        return any(_normalize_tag(model.tag) == wanted for model in self.models())

    def model_info(self, tag: str) -> OllamaModel | None:
        wanted = _normalize_tag(tag)
        for model in self.models():
            if _normalize_tag(model.tag) == wanted:
                return model
        return None

    def require_model(self, tag: str) -> OllamaModel:
        """Модель обязана быть локально: иначе понятная ошибка, а не пустой ответ."""
        model = self.model_info(tag)
        if model is None:
            raise OllamaModelMissingError(
                f"Модель «{tag}» не найдена локально. Скачайте её: ollama pull {tag}"
            )
        return model

    def running_models(self) -> list[dict]:
        """/api/ps: что сейчас загружено и сколько занимает — для memory policy."""
        payload = self._read_json(self._request("/api/ps", timeout=HEALTH_TIMEOUT_SEC))
        return list(payload.get("models") or [])

    def loaded_models(self, *, exclude: str | None = None) -> list[str]:
        """Теги загруженных сейчас моделей (кроме указанной) — их надо выгрузить.

        Правило «одна тяжёлая модель за раз» опирается на это: перед прогоном
        следующей модели убеждаемся, что предыдущая действительно ушла.
        """
        skip = _normalize_tag(exclude) if exclude else None
        tags: list[str] = []
        for item in self.running_models():
            tag = str(item.get("name") or item.get("model") or "")
            if tag and _normalize_tag(tag) != skip:
                tags.append(tag)
        return tags

    def unload(self, tag: str) -> bool:
        """Выгружает модель из памяти (`keep_alive: 0`) и подтверждает результат.

        Выгрузка обязательна между прогонами: без неё две модели могут оказаться
        в unified memory одновременно, а на 18 ГБ это прямой путь к memory
        pressure рядом с TTS.
        """
        payload = {"model": tag, "keep_alive": 0, "messages": []}
        try:
            self._read_json(self._request("/api/chat", payload, timeout=HEALTH_TIMEOUT_SEC))
        except OllamaError as exc:
            logger.warning("Не удалось выгрузить %s: %s", tag, exc)
        else:
            logger.info("Модель %s выгружена", tag)
        # Проверяем факт, а не намерение: демон мог оставить модель из-за
        # параллельного запроса.
        for _ in range(10):
            try:
                if _normalize_tag(tag) not in {_normalize_tag(item) for item in self.loaded_models()}:
                    return True
            except OllamaError:
                return False
            time.sleep(0.5)
        logger.warning("Модель %s всё ещё загружена после выгрузки", tag)
        return False

    def unload_others(self, keep: str | None = None) -> list[str]:
        """Выгружает всё лишнее перед прогоном и возвращает список выгруженных."""
        unloaded: list[str] = []
        for tag in self.loaded_models(exclude=keep):
            if self.unload(tag):
                unloaded.append(tag)
        return unloaded

    # -- чат -------------------------------------------------------------------
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        schema: dict | None = None,
        options: dict | None = None,
        keep_alive: str | int | None = None,
        timeout: float | None = None,
        cancel: Callable[[], bool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """Один запрос к модели со структурированным выходом.

        `schema` передаётся в `format`: Ollama ограничивает генерацию этой схемой,
        и это единственный надёжный способ получить JSON, а не «JSON в прозе».
        Потоковое чтение нужно не ради красоты: так работает отмена (проверяем
        флаг между строками) и виден прогресс на длинных ответах.
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "format": schema if schema is not None else "json",
            "options": dict(options or {}),
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        started = time.monotonic()
        response = self._request("/api/chat", payload, timeout=timeout, stream=True)
        parts: list[str] = []
        final: dict = {}
        try:
            for line in response:
                if cancel is not None and cancel():
                    raise OllamaError("запрос отменён вызывающим")
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    chunk = json.loads(text)
                except ValueError as exc:
                    raise OllamaResponseError(f"Ollama вернул не NDJSON: {text[:200]!r}") from exc
                if chunk.get("error"):
                    raise OllamaResponseError(str(chunk["error"]))
                piece = str((chunk.get("message") or {}).get("content") or "")
                if piece:
                    parts.append(piece)
                    if on_token is not None:
                        on_token(piece)
                if chunk.get("done"):
                    final = chunk
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        return ChatResult(
            text="".join(parts),
            model=model,
            duration_sec=time.monotonic() - started,
            eval_count=int(final.get("eval_count") or 0),
            prompt_eval_count=int(final.get("prompt_eval_count") or 0),
            raw=final,
        )


def _normalize_tag(tag: str) -> str:
    """Сравнение тегов без `:latest`: `qwen3:8b` и `qwen3:8b` — одно и то же.

    Ollama считает `name` и `name:latest` одной моделью, и требовать от
    пользователя писать суффикс было бы лишней строгостью.
    """
    return (tag or "").strip().removesuffix(":latest")


_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    """Единственный клиент на процесс: адрес и таймаут берутся из окружения."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def reset_client() -> None:
    """Сбрасывает singleton — нужно тестам и смене адреса в рантайме."""
    global _client
    _client = None
