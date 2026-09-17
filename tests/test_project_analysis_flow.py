"""Обязательная подготовка диалога: анализ, состояния, guard и контракт рендера.

Проверяется то, ради чего шаг и делался: синтез невозможен по сырому тексту, анализ
считается без поднятия моделей, инвалидация точечная, а рендер берёт **сохранённый**
текст, а не пересчитывает его заново. Модели не поднимаются: движки подменяются
заглушками, а ударения — детерминированной подменой акцентуатора.
"""

from __future__ import annotations

import asyncio
import json

from conftest import (  # noqa: F401 — sine нужен фикстуре voices
    StubEngine,
    analyze_project,
    sine,
)
from test_projects_api import _client, _create_project, _wait_job
from test_text_preview_api import F5_VOICE, XTTS_VOICE

from backend import audio_pipeline, config

DIALOGUE = "ИВАН: Сегодня хорошая погода.\nМАРГО: И я рад тебя видеть."
# Неоднозначное «все»: анализ обязан остановиться на `needs_review`, пока слово не
# подтверждено или не пропущено — молча решать за пользователя нельзя.
AMBIGUOUS = "ИВАН: Мы все успеем."


def _run(scenario) -> None:
    asyncio.run(scenario())


def _unstressed(text: str) -> str:
    """Текст без знаков ударения: сравнивать чтение, а не разметку.

    Ударения ставит RUAccent (в этом окружении он настоящий), и «пагада» вполне
    законно становится «паг+ада». Проверять надо слово, а не то, где именно
    оказался «+».
    """
    return (text or "").replace("+", "")


async def _analyzed(client, project: dict, **body) -> dict:
    """Один прогон анализа — сырое состояние, без прохода review."""
    response = await client.post(f"/api/projects/{project['id']}/analyze", json=body or {})
    assert response.status_code == 200, response.text
    return response.json()


async def _project_with_voice(client, voice_id: str, text: str = DIALOGUE) -> dict:
    """Проект с разобранным текстом и назначенными голосами (без анализа)."""
    project = await _create_project(client, text=text)
    speakers = {
        replica["speaker"]: voice_id
        for replica in (
            await client.post(f"/api/projects/{project['id']}/parse", json={})
        ).json()["replicas"]
    }
    patched = await client.patch(
        f"/api/projects/{project['id']}",
        json={"speakers": {key: {"voice_id": value} for key, value in speakers.items()}},
    )
    assert patched.status_code == 200, patched.text
    return patched.json()


def _state(client, project_id: str):
    async def fetch():
        response = await client.get(f"/api/projects/{project_id}/analysis")
        assert response.status_code == 200, response.text
        return response.json()

    return fetch


# --- 1. Рендер запрещён до анализа ----------------------------------------------
def test_dialogue_cannot_render_before_analysis(voices, stub, monkeypatch):
    """Прямой вызов API тоже получает отказ: guard стоит на сервере, не в кнопке."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)

            refused = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert refused.status_code == 409, refused.text
            detail = refused.json()["detail"]
            assert "не подготовлен" in detail
            assert "анализ" in detail.lower()

            await analyze_project(client, project["id"])
            accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])

    _run(scenario)


def test_analysis_state_is_visible_without_recalculation(voices, stub, monkeypatch):
    """`GET /analysis` показывает состояние и причину, ничего не пересчитывая."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            empty = (await client.get(f"/api/projects/{project['id']}/analysis")).json()
            assert empty["status"] == config.PROJECT_ANALYSIS_RAW
            assert empty["replicas_pending"] == empty["replicas_total"] == 2

            await analyze_project(client, project["id"])
            ready = (await client.get(f"/api/projects/{project['id']}/analysis")).json()
            assert ready["status"] == config.PROJECT_ANALYSIS_READY
            assert ready["replicas_done"] == 2
            assert ready["replicas_pending"] == 0
            assert ready["analysis_version"] >= 1
            assert ready["candidates_total"] == 0

    _run(scenario)


# --- 2. Анализ разбирает текст сам ---------------------------------------------
def test_dialogue_analysis_creates_replicas(voices, stub, monkeypatch):
    """`/analyze` сам разбирает текст, если реплик ещё нет."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _create_project(client, text=DIALOGUE)
            assert project["replicas_count"] == 0
            # Первый прогон разбирает текст сам. Голоса ещё не выбраны, и это
            # честно видно в отчёте: реплики есть, подготовить их нечем.
            first = await _analyzed(client, project)
            assert first["replicas_total"] == 2
            assert [speaker["key"] for speaker in first["speakers"]] == ["ИВАН", "МАРГО"]
            assert first["status"] == config.PROJECT_ANALYSIS_ERROR
            assert len(first["errors"]) == 2
            assert "не назначен голос" in first["errors"][0]["text"]
            saved = (await client.get(f"/api/projects/{project['id']}")).json()
            assert len(saved["replicas"]) == 2

            # После выбора голосов тот же шаг доводит проект до готовности.
            await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": F5_VOICE}, "МАРГО": {"voice_id": F5_VOICE}}},
            )
            report = await analyze_project(client, project["id"])
            assert report["replicas_analyzed"] == 2
            saved = (await client.get(f"/api/projects/{project['id']}")).json()
            assert all(item["final_text"] for item in saved["replicas"])

    _run(scenario)


# --- 3. Инвалидация -------------------------------------------------------------
def test_edit_source_invalidates_analysis(voices, stub, monkeypatch):
    """Правка исходного текста возвращает проект в `raw` и снова запрещает рендер."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])

            patched = await client.patch(
                f"/api/projects/{project['id']}",
                json={"source_text": DIALOGUE + "\nИВАН: Ещё одна реплика."},
            )
            assert patched.status_code == 200, patched.text
            assert patched.json()["analysis_status"] == config.PROJECT_ANALYSIS_RAW
            assert "изменён" in (patched.json()["analysis_error"] or "")

            refused = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert refused.status_code == 409, refused.text
            assert "исходный текст изменён" in refused.json()["detail"]

    _run(scenario)


def test_edit_replica_invalidates_only_required_analysis(voices, stub, monkeypatch):
    """Правка текста одной реплики устаревает только её подготовку."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])

            patched = await client.patch(
                f"/api/projects/{project['id']}/replicas/0",
                json={"text": "Совсем другой текст реплики."},
            )
            assert patched.status_code == 200, patched.text
            state = (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()
            assert state["status"] == config.PROJECT_ANALYSIS_RAW
            assert state["replicas_pending"] == 1
            assert state["replicas_done"] == 1

            # Точечный анализ возвращает проект в строй, не трогая вторую реплику.
            report = await _analyzed(client, project, indexes=[0])
            assert report["updated_replicas"] == [0]
            assert (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()["status"] == config.PROJECT_ANALYSIS_READY

    _run(scenario)


def test_voice_engine_change_invalidates_engine_dependent_analysis(voices, stub, monkeypatch):
    """Смена голоса спикера устаревает подготовку его реплик (ударения зависят от движка)."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])

            before = [
                item["final_text"]
                for item in (await client.get(f"/api/projects/{project['id']}")).json()[
                    "replicas"
                ]
            ]
            assert before

            patched = await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"ИВАН": {"voice_id": XTTS_VOICE}}},
            )
            assert patched.status_code == 200, patched.text
            state = (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()
            assert state["status"] == config.PROJECT_ANALYSIS_RAW
            # Устарели только реплики ИВАН (одна из двух), МАРГО осталась готовой.
            assert state["replicas_pending"] == 1
            assert state["replicas_done"] == 1
            assert "голос" in (state["analysis_error"] or "")

    _run(scenario)


def test_overrides_do_not_invalidate_analysis(voices, stub, monkeypatch):
    """Ползунки синтеза не отменяют подготовку текста: они его не меняют."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])

            patched = await client.patch(
                f"/api/projects/{project['id']}/replicas/0",
                json={"overrides": {"speed": 1.15}},
            )
            assert patched.status_code == 200, patched.text
            state = (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()
            assert state["status"] == config.PROJECT_ANALYSIS_READY

    _run(scenario)


def test_reanalyze_only_requested_replicas(voices, stub, monkeypatch):
    """`indexes` пересчитывает ровно указанные реплики, остальные не трогает."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            version = (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()["analysis_version"]

            report = await _analyzed(client, project, indexes=[1])
            assert report["updated_replicas"] == [1]
            assert report["replicas_analyzed"] == 1
            state = (
                await client.get(f"/api/projects/{project['id']}/analysis")
            ).json()
            assert state["analysis_version"] > version
            assert state["status"] == config.PROJECT_ANALYSIS_READY

    _run(scenario)


def test_unknown_replica_index_is_rejected(voices, stub, monkeypatch):
    """Опечатка в номере реплики — ошибка запроса, а не молчаливый пропуск."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            bad = await client.post(
                f"/api/projects/{project['id']}/analyze", json={"indexes": [7]}
            )
            assert bad.status_code == 400, bad.text
            assert "Нет таких реплик" in bad.json()["detail"]

    _run(scenario)


# --- 4. Анализ дешёвый: без моделей ---------------------------------------------
def test_analysis_does_not_load_engines(voices, stub, monkeypatch):
    """Анализ не поднимает веса TTS: иначе подготовка стоила бы как синтез."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)

            def explode(*_args, **_kwargs):
                raise AssertionError("анализ не должен создавать TTS-движок")

            from backend.engines import registry

            monkeypatch.setattr(registry, "get_engine", explode)
            monkeypatch.setattr(registry, "_create", explode)
            await analyze_project(client, project["id"])

            assert registry.created_engines() == {}
            assert stub.loads == 0  # заглушка тоже не поднималась

    _run(scenario)


# --- 5. Контракт рендера: сохранённый текст -------------------------------------
def test_render_uses_saved_final_text(voices, stub, monkeypatch):
    """Рендер берёт сохранённый текст: изменения словаря после анализа его не трогают."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            saved = [
                item["final_text"]
                for item in (await client.get(f"/api/projects/{project['id']}")).json()[
                    "replicas"
                ]
            ]

            # Акцентуатор меняется ПОСЛЕ анализа: если бы рендер считал текст
            # заново, в модель ушёл бы другой текст. Он обязан взять сохранённый —
            # тот, который пользователь видел и подтверждал.
            monkeypatch.setattr(
                audio_pipeline, "accentuate", lambda text: f"[{text}]"
            )

            stub.calls.clear()
            accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])

            assert [call["text"] for call in stub.calls] == saved
            assert all(not call["text"].startswith("[") for call in stub.calls)

    _run(scenario)


def test_preview_final_equals_render_input(voices, stub, monkeypatch):
    """То, что показывает панель «Что услышит модель», и есть вход движка."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]

            preview = await client.post(
                "/api/text/preview",
                json={
                    "text": replica["source_text"],
                    "voice_id": F5_VOICE,
                    "auto_accent": True,
                },
            )
            assert preview.status_code == 200, preview.text
            assert preview.json()["final"] == replica["final_text"]

            stub.calls.clear()
            accepted = await client.post(f"/api/projects/{project['id']}/render", json={})
            await _wait_job(client, accepted.json()["job_id"])
            assert stub.calls[0]["text"] == preview.json()["final"]

    _run(scenario)


def test_regenerate_uses_saved_final_text(voices, stub, monkeypatch):
    """Пересинтез реплики идёт по тому же сохранённому тексту."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]

            stub.calls.clear()
            accepted = await client.post(
                f"/api/projects/{project['id']}/replicas/0/regenerate"
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])
            assert [call["text"] for call in stub.calls] == [replica["final_text"]]

    _run(scenario)


def test_regenerate_requires_prepared_replica(voices, stub, monkeypatch):
    """Неподготовленную реплику пересинтезировать нельзя: текста для неё ещё нет."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            refused = await client.post(
                f"/api/projects/{project['id']}/replicas/0/regenerate"
            )
            assert refused.status_code == 409, refused.text
            assert "не подготовлена" in refused.json()["detail"]

    _run(scenario)


# --- 6. Отчёт анализа -----------------------------------------------------------
def test_analysis_report_shape(voices, stub, monkeypatch):
    """Ответ анализа содержит поля §12, и счётчики в нём согласованы."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            report = await analyze_project(client, project["id"])

            for key in (
                "status",
                "replicas_total",
                "replicas_analyzed",
                "speakers",
                "candidates",
                "warnings",
                "errors",
                "analysis_version",
                "updated_replicas",
            ):
                assert key in report, key
            assert report["status"] == "ready"
            assert report["replicas_analyzed"] == report["replicas_total"] == 2
            assert report["errors"] == []
            assert sorted(report["updated_replicas"]) == [0, 1]
            json.dumps(report)  # ответ обязан быть сериализуемым как есть

    _run(scenario)


# --- 7. Review: неоднозначное слово блокирует ready -----------------------------
def test_pronunciation_candidates_require_review(voices, stub, monkeypatch):
    """Неоднозначное слово останавливает подготовку и объясняет, что делать."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE, text=AMBIGUOUS)
            report = await _analyzed(client, project)

            assert report["status"] == config.PROJECT_ANALYSIS_NEEDS_REVIEW
            assert [item["word"] for item in report["candidates"]] == ["все"]

            refused = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert refused.status_code == 409, refused.text
            assert "подтвержд" in refused.json()["detail"]

    _run(scenario)


def test_pronunciation_skip_allows_ready(voices, stub, monkeypatch, fake_accent):
    """Пропуск предложения снимает блокировку и не меняет произношение."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE, text=AMBIGUOUS)
            report = await _analyzed(client, project)
            candidate = report["candidates"][0]

            skipped = await client.post(
                "/api/pronunciation",
                json={
                    "source": candidate["word"],
                    "target": candidate["target"],
                    "enabled": False,
                    "note": "проверка в тесте",
                },
            )
            assert skipped.status_code in (200, 201), skipped.text

            after = await _analyzed(
                client, project, indexes=[candidate["replica_index"]]
            )
            assert after["status"] == config.PROJECT_ANALYSIS_READY
            assert after["candidates"] == []
            # Пропуск — «оставить как есть»: в тексте по-прежнему «все», не «всё».
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "все" in replica["final_text"]

    _run(scenario)


# --- 8. Разметка слотов не доезжает до модели -----------------------------------
def test_speaker_markers_removed_before_preprocessing(voices, stub, monkeypatch):
    """Служебные маркеры разбираются до подготовки и в текст модели не попадают."""

    async def scenario():
        async with _client(monkeypatch) as client:
            text = "АРТЕМ(1): Привет! (1 speed=1.2 cfg=2.5) Как дела?"
            project = await _create_project(client, text=text)
            await client.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {"#1": {"voice_id": F5_VOICE}}},
            )
            await analyze_project(client, project["id"])
            replicas = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"]

            for replica in replicas:
                assert "speed=" not in replica["source_text"]
                assert "(1)" not in replica["source_text"]
                assert "speed=" not in replica["final_text"]
                assert "АРТЕМ" not in replica["final_text"]

    _run(scenario)


# --- 9. Ударения: F5 получает, XTTS — нет ---------------------------------------
def test_f5_final_contains_expected_accent_markup(voices, stub, monkeypatch, fake_accent):
    """Для движка с поддержкой ударений сохранённый текст размечен."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            replicas = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"]
            assert all(item["supports_accents"] for item in replicas)
            assert all(item["final_text"].startswith("[") for item in replicas)

    _run(scenario)


def test_xtts_final_does_not_receive_f5_accent_markup(voices, stub, monkeypatch, fake_accent):
    """Для движка без ударений разметки нет — и «+» из словаря тоже не уходит."""

    async def scenario():
        async with _client(monkeypatch) as client:
            created = await client.post(
                "/api/pronunciation", json={"source": "погода", "target": "пог+ода"}
            )
            assert created.status_code in (200, 201), created.text
            project = await _project_with_voice(client, XTTS_VOICE)
            await analyze_project(client, project["id"])
            replicas = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"]

            assert all(not item["supports_accents"] for item in replicas)
            for item in replicas:
                assert "+" not in item["final_text"]
                assert not item["final_text"].startswith("[")

    _run(scenario)


def test_disabled_auto_accent_does_not_block_ready(voices, stub, monkeypatch):
    """Выключенные пользователем ударения — не ошибка: проект всё равно готов."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            report = await analyze_project(client, project["id"], auto_accent=False)
            assert report["status"] == config.PROJECT_ANALYSIS_READY
            replicas = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"]
            assert all(item["auto_accent"] is False for item in replicas)

    _run(scenario)


def test_accentizer_failure_is_visible(voices, stub, monkeypatch):
    """Падение акцентуатора видно как ошибка подготовки, а не как «готово без ударений»."""

    def broken(_text: str) -> str:
        raise RuntimeError("RUAccent недоступен")

    monkeypatch.setattr(audio_pipeline, "accentuate", broken)

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE)
            report = await _analyzed(client, project)

            assert report["status"] == config.PROJECT_ANALYSIS_ERROR
            assert report["errors"], report
            assert "RUAccent недоступен" in report["errors"][0]["text"]
            refused = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert refused.status_code == 409, refused.text
            assert "анализ не удался" in refused.json()["detail"].lower()

    _run(scenario)


# --- 10. Legacy-вход остаётся рабочим -------------------------------------------
def test_legacy_generate_still_renders_without_analysis(voices, stub, monkeypatch):
    """`/api/generate` — прежний вход без сохранённого анализа; он продолжает работать.

    Проектный рендер без подготовки запрещён, а разовая задача по тексту готовит
    текст на месте тем же каноническим путём: правило «нельзя синтезировать
    неподготовленный проект» к ней не относится, и это должно быть явно.
    """

    async def scenario():
        async with _client(monkeypatch) as client:
            stub.calls.clear()
            accepted = await client.post(
                "/api/generate",
                json={
                    "dialogue_text": "ИВАН: Сегодня хорошая погода.",
                    "speakers": {"ИВАН": {"voice_id": F5_VOICE}},
                    "output_format": "wav",
                },
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(client, accepted.json()["job_id"])
            assert [call["text"] for call in stub.calls], "разовый рендер не дошёл до движка"

    _run(scenario)


# --- 11. Область словаря: проект важнее глобального ------------------------------
def test_project_dictionary_overrides_global(voices, stub, monkeypatch):
    """Правило проекта побеждает глобальное: одно слово может читаться по-разному."""

    async def scenario():
        async with _client(monkeypatch) as client:
            global_rule = await client.post(
                "/api/pronunciation", json={"source": "погода", "target": "пагада"}
            )
            assert global_rule.status_code in (200, 201), global_rule.text
            project = await _project_with_voice(client, F5_VOICE)
            local_rule = await client.post(
                "/api/pronunciation",
                # Замена проекта не должна содержать исходное слово: правила
                # применяются по очереди, и глобальное правило сработало бы уже
                # на результате проектного — это порядок, а не приоритет.
                json={"source": "погода", "target": "погодушка", "project_id": project["id"]},
            )
            assert local_rule.status_code in (200, 201), local_rule.text

            await analyze_project(client, project["id"])
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "погодушка" in _unstressed(replica["final_text"])
            assert "пагада" not in _unstressed(replica["final_text"])

            # Глобальный словарь при этом не изменился — правило проекта живёт отдельно.
            listed = await client.get("/api/pronunciation")
            assert [item["target"] for item in listed.json()["entries"]] == ["пагада"]

    _run(scenario)


def test_global_dictionary_used_when_no_project_override(voices, stub, monkeypatch):
    """Без правила проекта работает глобальное — приоритет не отменяет общий словарь."""

    async def scenario():
        async with _client(monkeypatch) as client:
            created = await client.post(
                "/api/pronunciation", json={"source": "погода", "target": "пагада"}
            )
            assert created.status_code in (200, 201), created.text
            project = await _project_with_voice(client, F5_VOICE)
            await analyze_project(client, project["id"])
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "пагада" in _unstressed(replica["final_text"])

    _run(scenario)


def test_dictionary_change_reanalyzes_affected_replicas(voices, stub, monkeypatch):
    """Правка словаря устаревает ровно те реплики, где встретилось слово."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(
                client, F5_VOICE, text="ИВАН: Сегодня хорошая погода.\nМАРГО: И я рад тебя видеть."
            )
            await analyze_project(client, project["id"])

            created = await client.post(
                "/api/pronunciation", json={"source": "погода", "target": "пагада"}
            )
            assert created.status_code in (200, 201), created.text
            assert created.json()["affected"], "правило обязано устаревать затронутые реплики"

            state = (await client.get(f"/api/projects/{project['id']}/analysis")).json()
            assert state["status"] == config.PROJECT_ANALYSIS_RAW
            # Слово есть только в первой реплике — вторая остаётся подготовленной.
            assert state["replicas_pending"] == 1
            assert state["replicas_done"] == 1
            assert "словарь" in (state["analysis_error"] or "")

            refused = await client.post(f"/api/projects/{project['id']}/render", json={})
            assert refused.status_code == 409, refused.text

            # Повторный анализ подхватывает правило, и проект снова готов.
            await analyze_project(client, project["id"])
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "пагада" in _unstressed(replica["final_text"])

    _run(scenario)


def test_review_accepts_into_project_scope(voices, stub, monkeypatch):
    """Review умеет принять слово только для проекта — и не трогает общий словарь."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE, text=AMBIGUOUS)
            report = await _analyzed(client, project)
            candidate = report["candidates"][0]
            assert candidate["word"] == "все"

            reviewed = await client.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={
                    "source": candidate["word"],
                    "target": candidate["target"],
                    "scope": "project",
                    "replica_index": candidate["replica_index"],
                },
            )
            assert reviewed.status_code == 200, reviewed.text
            assert reviewed.json()["analysis"]["status"] == config.PROJECT_ANALYSIS_READY
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "всё" in _unstressed(replica["final_text"])

            # Глобальный словарь пуст: решение принято для проекта.
            assert (await client.get("/api/pronunciation")).json()["entries"] == []
            # А в проектном словаре правило есть.
            local = await client.get(f"/api/pronunciation?project_id={project['id']}")
            assert [item["source"] for item in local.json()["entries"]] == ["все"]

    _run(scenario)


def test_review_skip_does_not_change_pronunciation(voices, stub, monkeypatch, fake_accent):
    """Пропуск в review — выключенное правило: слово не меняется и не предлагается снова."""

    async def scenario():
        async with _client(monkeypatch) as client:
            project = await _project_with_voice(client, F5_VOICE, text=AMBIGUOUS)
            report = await _analyzed(client, project)
            candidate = report["candidates"][0]

            skipped = await client.post(
                f"/api/projects/{project['id']}/pronunciation/review",
                json={
                    "source": candidate["word"],
                    "target": candidate["target"],
                    "scope": "project",
                    "enabled": False,
                    "replica_index": candidate["replica_index"],
                },
            )
            assert skipped.status_code == 200, skipped.text
            state = skipped.json()["analysis"]
            assert state["status"] == config.PROJECT_ANALYSIS_READY
            assert state["candidates"] == []
            replica = (await client.get(f"/api/projects/{project['id']}")).json()["replicas"][0]
            assert "все" in _unstressed(replica["final_text"])
            assert "всё" not in _unstressed(replica["final_text"])

    _run(scenario)
