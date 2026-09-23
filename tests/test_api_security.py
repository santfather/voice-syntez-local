"""Безопасность и гигиена API: лимиты тела, request-id, `/healthz`, общий 500.

Тесты идут двумя путями. Через приложение целиком (ASGI-транспорт) — там, где
проверяется ответ клиенту. Напрямую через middleware — там, где нужен ответ,
которого HTTP-клиент не отдаёт: например, тело без `Content-Length` в httpx
не собрать, а именно этот случай и закрывает счётчик прочитанных байт.
"""

import asyncio
import contextlib
import io
import logging
import zipfile

import httpx
import pytest

from backend import config, main, project_export

TAKE_DIR = project_export.TAKE_DIR


@contextlib.asynccontextmanager
async def _client(**transport_options):
    """ASGI-клиент без очереди задач: проверяются только middleware и роуты."""
    transport = httpx.ASGITransport(app=main.app, **transport_options)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


class _ReadBodyApp:
    """Заглушка вместо FastAPI: вычитывает тело и отвечает 200.

    Нужна, чтобы проверять лимит без httpx: клиент вычитывает тело сам и до
    приложения, поэтому middleware на нём не проверить.
    """

    async def __call__(self, scope, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _ReplyApp:
    """Заглушка вместо FastAPI: сразу отвечает 200, тело не читает."""

    async def __call__(self, scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _scope(path: str = "/api/parse", headers: list | None = None) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers or []),
    }


async def _through(scope: dict, middleware, messages: list[dict]) -> list[dict]:
    """Прогоняет один запрос через middleware и возвращает отправленные сообщения."""
    sent: list[dict] = []
    queue = list(messages)

    async def receive():
        return queue.pop(0) if queue else {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


# --- лимит размера тела запроса ------------------------------------------------
def test_body_limit_rejects_declared_length(monkeypatch):
    """Объявленный `Content-Length` больше предела — 413 без чтения тела."""
    monkeypatch.setattr(config, "MAX_JSON_BODY_BYTES", 10)
    middleware = main.RequestSizeLimitMiddleware(_ReadBodyApp())
    sent = asyncio.run(
        _through(_scope(headers=[(b"content-length", b"1000")]), middleware, [])
    )
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


def test_body_limit_counts_streamed_chunks(monkeypatch):
    """Тело без `Content-Length` тоже ограничено: байты считаются по мере чтения.

    Иначе `Transfer-Encoding: chunked` обходил бы предел целиком — заголовка, по
    которому его можно отсечь заранее, в запросе просто нет.
    """
    monkeypatch.setattr(config, "MAX_JSON_BODY_BYTES", 10)
    middleware = main.RequestSizeLimitMiddleware(_ReadBodyApp())
    messages = [
        {"type": "http.request", "body": b"x" * 8, "more_body": True},
        {"type": "http.request", "body": b"x" * 8, "more_body": False},
    ]
    sent = asyncio.run(_through(_scope(), middleware, messages))
    assert sent[0]["status"] == 413


def test_body_limit_allows_request_within_limit(monkeypatch):
    """Счётчик не срабатывает на корректном теле — иначе лимит ломал бы работу."""
    monkeypatch.setattr(config, "MAX_JSON_BODY_BYTES", 1024)
    middleware = main.RequestSizeLimitMiddleware(_ReadBodyApp())
    messages = [{"type": "http.request", "body": b"x" * 16, "more_body": False}]
    sent = asyncio.run(_through(_scope(), middleware, messages))
    assert sent[0]["status"] == 200


def test_import_path_has_its_own_higher_limit(monkeypatch):
    """У импорта архива предел выше текстового: он переносит файл, а не форму."""
    monkeypatch.setattr(config, "MAX_JSON_BODY_BYTES", 10)
    monkeypatch.setattr(config, "MAX_ARCHIVE_BYTES", 4 * 1024 * 1024)
    middleware = main.RequestSizeLimitMiddleware(_ReadBodyApp())
    scope = _scope(
        path=main._IMPORT_PATH,
        headers=[(b"content-length", str(1024 * 1024).encode())],
    )
    sent = asyncio.run(_through(scope, middleware, []))
    assert sent[0]["status"] == 200


def test_json_body_over_limit_via_api(monkeypatch):
    """Через реальный запрос: большое JSON-тело не доходит до обработчика."""
    monkeypatch.setattr(config, "MAX_JSON_BODY_BYTES", 64)

    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/parse", json={"dialogue_text": "ИВАН: " + "я" * 500}
            )
            assert response.status_code == 413, response.text
            assert "МБ" in response.json()["detail"]

    _run(scenario)


# --- лимиты архива (zip-bomb) --------------------------------------------------
def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_archive_rejects_too_many_entries(monkeypatch):
    monkeypatch.setattr(config, "ARCHIVE_MAX_ENTRIES", 2)
    data = _zip_bytes(
        {
            project_export.MANIFEST_NAME: b"{}",
            f"{TAKE_DIR}/a.wav": b"a",
            f"{TAKE_DIR}/b.wav": b"b",
        }
    )
    with pytest.raises(project_export.ProjectExportError, match="слишком много записей"):
        project_export.import_archive(data)


def test_archive_rejects_huge_uncompressed_before_extracting(monkeypatch):
    """«Архив-бомба» отклоняется по заголовкам ZIP, до распаковки.

    Сжатые нули весят килобайты, а разворачиваются в мегабайты: предел на сжатый
    файл от такого не спасает, а проверка обязана сработать до записи на диск.
    """
    monkeypatch.setattr(config, "ARCHIVE_MAX_UNCOMPRESSED_BYTES", 1024)
    extracted: list = []
    monkeypatch.setattr(project_export, "_extract", lambda *args, **kwargs: extracted.append(args))
    data = _zip_bytes(
        {
            project_export.MANIFEST_NAME: b"{}",
            f"{TAKE_DIR}/big.wav": b"\0" * 64_000,
        }
    )
    with pytest.raises(project_export.ProjectExportError, match="Распакованный архив"):
        project_export.import_archive(data)
    assert extracted == [], "распаковка не должна начинаться до проверки размера"


def test_archive_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(config, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(project_export.ProjectExportError, match="Архив больше"):
        project_export.import_archive(b"x" * 500)


def test_oversized_archive_rejected_through_api(monkeypatch):
    """Через API: слишком большой архив — 400 с понятным текстом, не 500."""
    monkeypatch.setattr(config, "MAX_ARCHIVE_BYTES", 100)

    async def scenario():
        async with _client() as client:
            response = await client.post(
                "/api/projects/import",
                files={"file": ("p.ttsproject", b"x" * 500, "application/zip")},
            )
            assert response.status_code == 400, response.text
            assert "Архив больше" in response.json()["detail"]

    _run(scenario)


# --- request-id ----------------------------------------------------------------
def test_request_id_added_to_response(monkeypatch):
    """Ответ несёт request-id — по нему запрос находят в журнале."""
    middleware = main.RequestIdMiddleware(_ReplyApp())
    sent = asyncio.run(_through(_scope(), middleware, []))
    headers = dict(sent[0]["headers"])
    assert headers[b"x-request-id"]


def test_request_id_from_client_is_kept():
    middleware = main.RequestIdMiddleware(_ReplyApp())
    scope = _scope(headers=[(b"x-request-id", b"abc-123")])
    sent = asyncio.run(_through(scope, middleware, []))
    assert dict(sent[0]["headers"])[b"x-request-id"] == b"abc-123"


def test_request_id_with_bad_characters_is_replaced():
    """Строка не по формату не попадает в ответ: перевод строки — это подмена ответа."""
    middleware = main.RequestIdMiddleware(_ReplyApp())
    scope = _scope(headers=[(b"x-request-id", b"bad value!")])
    sent = asyncio.run(_through(scope, middleware, []))
    assert dict(sent[0]["headers"])[b"x-request-id"] != b"bad value!"


def test_request_id_visible_in_response_header():
    async def scenario():
        async with _client() as client:
            response = await client.get("/healthz")
            assert response.headers["x-request-id"]

    _run(scenario)


# --- общий обработчик исключений -----------------------------------------------
def test_unhandled_exception_is_logged_without_leaking_text(monkeypatch, caplog):
    """Непредвиденная ошибка: трейсбек в журнал, клиенту — общий текст (F-A4).

    `raise_app_exceptions=False` нужен потому, что Starlette после ответа
    пробрасывает исключение дальше — так uvicorn пишет свой трейсбек. В тесте
    ответ уже отправлен, и именно его мы проверяем.
    """
    secret = "внутренняя деталь: /Users/секретный/путь"

    def boom(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(main, "parse_dialogue", boom)

    async def scenario():
        async with _client(raise_app_exceptions=False) as client:
            response = await client.post("/api/parse", json={"dialogue_text": "ИВАН: привет"})
            assert response.status_code == 500, response.text
            assert secret not in response.text
            assert response.json()["detail"] == (
                "Внутренняя ошибка сервера. Подробности — в журнале приложения."
            )
            assert response.json()["request_id"] == response.headers["x-request-id"]

    with caplog.at_level(logging.ERROR):
        _run(scenario)
    assert secret in caplog.text, "причина обязана остаться в журнале"
    assert "Traceback" in caplog.text


# --- /healthz ------------------------------------------------------------------
def test_healthz_reports_healthy_service(monkeypatch):
    """/healthz отвечает 200: база пуста, но цела, упавших движков нет."""
    async def scenario():
        async with _client() as client:
            response = await client.get("/healthz")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["status"] == "ok"
            assert payload["database"] == "ok"

    _run(scenario)


def test_healthz_reports_failed_engine(monkeypatch):
    """Упавший движок — 503: сторож обязан видеть разницу с «всё хорошо»."""
    from backend.engines.base import STATE_FAILED

    class _FailedEngine:
        state = STATE_FAILED

    monkeypatch.setattr(main, "created_engines", lambda: {"f5": _FailedEngine()})

    async def scenario():
        async with _client() as client:
            response = await client.get("/healthz")
            assert response.status_code == 503, response.text
            assert response.json()["status"] == "degraded"
            assert response.json()["engines"] == {"f5": STATE_FAILED}

    _run(scenario)
