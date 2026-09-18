"""Регрессионный диалог UPDATE 2 (§2, §43, §44) — обязательный fixture.

Тот самый диалог с опечаткой «быает» и с именем, записанным двумя способами
(«АРТЁМ» и «АРТЕМ»). Проверяется не звук, а контракты, которые ломались или могли
сломаться: реплики не теряются и не слипаются, разные спикеры не попадают в один
вызов синтеза, опечатка исправляется только через подтверждение, эмоция не
проникает в текст, а у каждой готовой реплики есть звук.

Настоящие модели здесь не нужны: движок — заглушка conftest, она пишет в журнал
ровно то, что ушло в синтез.
"""

import asyncio

from conftest import analyze_project
from test_projects_api import _client

from backend import config, emotions
from backend.dialogue_parser import parse_dialogue

FIXTURE = "tests/fixtures/short_dialogue_ru.txt"
MARGARITA = "Дайте пройти. Пожалуйста."
TWO_SENTENCES = "Всякое быает. Оставь номер."
CORRECTED = "Всякое бывает. Оставь номер."


def _text() -> str:
    with open(FIXTURE, encoding="utf-8") as handle:
        return handle.read()


def test_fixture_parses_into_one_speaker_per_character():
    """«АРТЁМ» и «АРТЕМ» — один персонаж, а не два голоса (§2, §44)."""
    parsed = parse_dialogue(_text(), config.chunk_chars(config.CHUNK_STRATEGY_DEFAULT))
    assert len(parsed.replicas) == 10
    assert [item.key for item in parsed.voices] == ["АРТЁМ", "МАРГАРИТА"]
    assert [replica.voice for replica in parsed.replicas] == [
        "АРТЁМ",
        "МАРГАРИТА",
    ] * 5
    # Реплики не переставлены и текст не переписан — включая опечатку.
    assert parsed.replicas[1].text == MARGARITA
    assert parsed.replicas[6].text == TWO_SENTENCES
    assert "быает" in parsed.replicas[6].text


def test_short_replicas_with_two_sentences_stay_single(stub, fake_store):
    """Реплики с двумя предложениями внутри остаются одним синтезом (§16, §44)."""
    parsed = parse_dialogue(_text(), config.chunk_chars(config.CHUNK_STRATEGY_DEFAULT))
    assert len(parsed.replicas) == 10
    texts = [replica.text for replica in parsed.replicas]
    assert MARGARITA in texts and TWO_SENTENCES in texts
    assert all("\n" not in text for text in texts)


def test_regression_dialogue_renders_without_losing_replicas(stub, fake_store, monkeypatch):
    """Полный проход: разбор → анализ → рендер → варианты, все реплики на месте."""

    async def scenario():
        async with _client(monkeypatch) as http:
            created = await http.post(
                "/api/projects",
                json={"name": "Регрессия §2", "source_text": _text(), "mode": "dialogue"},
            )
            assert created.status_code == 201, created.text
            project = created.json()
            parsed = await http.post(f"/api/projects/{project['id']}/parse", json={})
            assert parsed.status_code == 200, parsed.text
            replicas = parsed.json()["replicas"]
            # Спикеров двое, а не трое: имена с «ё» и без — один персонаж.
            speakers = {item["speaker"] for item in replicas}
            assert speakers == {"АРТЁМ", "МАРГАРИТА"}
            voice_id = fake_store.id
            await http.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {key: {"voice_id": voice_id} for key in speakers}},
            )
            await analyze_project(http, project["id"])

            accepted = await http.post(
                f"/api/projects/{project['id']}/render",
                json={"output_format": "wav", "output_name": "регрессия-2"},
            )
            assert accepted.status_code == 202, accepted.text
            await _wait_job(http, accepted.json()["job_id"])
            state = (await http.get(f"/api/projects/{project['id']}")).json()
            return state

    state = asyncio.run(scenario())
    assert len(state["replicas"]) == 10

    # Ни одна реплика не потеряна и не слиплась: порядок и тексты сохранены.
    assert [row["text"] for row in state["replicas"]][6] == TWO_SENTENCES
    # Синтез шёл по одной реплике: заглушка вызвана по разу на реплику, и в каждом
    # вызове — только слова одного спикера (§27, §44).
    assert len(stub.calls) == 10
    for call in stub.calls:
        assert "\n" not in call["text"]

    # У каждой реплики есть звук и вариант; длительность положительная.
    for row in state["replicas"]:
        assert row["takes"], f"реплика {row['index']} без варианта"
        take = row["takes"][-1]
        assert take["duration_sec"] > 0
    # Эмоция не попала в произносимый текст, а вопрос распознан (§4, §6).
    assert "emotion" not in " ".join(row["final_text"] or "" for row in state["replicas"]).lower()
    question = next(row for row in state["replicas"] if row["text"] == "Это проблема?")
    assert question["emotion_detected"] == emotions.EMOTION_QUESTION
    assert question["emotion_effective"] == emotions.EMOTION_QUESTION


def test_typo_is_corrected_only_through_confirmed_review(stub, fake_store, monkeypatch):
    """Опечатка «быает» исправляется подтверждением правила, а не молча (§2, §44)."""

    async def scenario():
        async with _client(monkeypatch) as http:
            created = await http.post(
                "/api/projects",
                json={"name": "Регрессия опечатки", "source_text": _text(), "mode": "dialogue"},
            )
            project = created.json()
            await http.post(f"/api/projects/{project['id']}/parse", json={})
            parsed = await http.get(f"/api/projects/{project['id']}")
            speakers = {item["speaker"] for item in parsed.json()["replicas"]}
            voice_id = fake_store.id
            await http.patch(
                f"/api/projects/{project['id']}",
                json={"speakers": {key: {"voice_id": voice_id} for key in speakers}},
            )
            state = await analyze_project(http, project["id"])
            # До подтверждения текст остаётся таким, каким его написал пользователь.
            before = (await http.get(f"/api/projects/{project['id']}")).json()
            assert any("быает" in (row["final_text"] or "") for row in before["replicas"])

            # Решение принимает человек: правило создаётся явным запросом.
            rule = await http.post(
                "/api/pronunciation",
                json={
                    "source": "быает",
                    "target": "бывает",
                    "enabled": True,
                    "note": "подтверждено пользователем",
                },
            )
            assert rule.status_code in (200, 201), rule.text
            index = next(
                row["index"] for row in before["replicas"] if "быает" in (row["final_text"] or "")
            )
            await http.post(
                f"/api/projects/{project['id']}/analyze", json={"indexes": [index]}
            )
            after = (await http.get(f"/api/projects/{project['id']}")).json()
            corrected = next(row for row in after["replicas"] if row["index"] == index)
            assert corrected["final_text"] == CORRECTED
            # Исходный текст не переписан: правка живёт в правилах, а не в source.
            assert corrected["source_text"] == TWO_SENTENCES
            assert state["status"] in (
                config.PROJECT_ANALYSIS_READY,
                config.PROJECT_ANALYSIS_NEEDS_REVIEW,
            )
            return corrected

    corrected = asyncio.run(scenario())
    assert corrected["final_text"] == CORRECTED


async def _wait_job(client, job_id: str, timeout: float = 60.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] == "error":
            raise AssertionError(data["error"])
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"задача {job_id} не завершилась")
