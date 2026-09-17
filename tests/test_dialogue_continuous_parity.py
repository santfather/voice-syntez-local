"""Parity трёх путей до модели: «Сплошной текст», диалог и проект.

Причина, по которой режимы звучат по-разному, измеряется не «на слух»: F5-TTS
недетерминирован, и два рендера одного текста без закреплённого сида — это два
разных случайных розыгрыша. Поэтому проверяем то, что действительно обязано
совпадать, — фактические аргументы на границе ``engine.synthesize()``.

Все три пути сходятся в ``audio_pipeline.render_dialogue`` и проходят один
preprocessing (нормализация → «ё» → словарь → ударения). Заглушка-шпион
``StubEngine`` записывает текст, скорость, ручки движка, сид и референс, а тесты
сравнивают именно эти записи, а не тела ответов API.

Различаются режимы только паузой (перед репликой спикера против пауз между
кусками нарезки) и самой нарезкой — на вход движка это не влияет, поэтому
сравнение идёт по одному и тому же произносимому тексту.
"""

from __future__ import annotations

import asyncio
import random

from test_api import _client, _generate, _wait

from backend import audio_pipeline, config
from backend.engines.base import ENGINE_INFOS, EngineInfo
from backend.text_normalization.pronunciation import PronunciationRule

# Произносимая фраза. В диалоге и проекте она приходит с маркером спикера
# «ИВАН:», который парсер снимает до preprocessing, — на границе движка текст
# обязан совпасть со сплошным текстом.
PHRASE = "Сегодня хорошая погода."
# Текст с числом: нормализация обязана сработать одинаково во всех режимах.
NUMBER_PHRASE = "В 2026 году мы встретились."
NUMBER_FINAL = "В две тысячи двадцать шестом году мы встретились."
# «Ё» восстанавливается отдельной стадией — итог отличается от входа.
YO_PHRASE = "Еще в 2026 году все было хорошо."
YO_FINAL = "Ещё в две тысячи двадцать шестом году все было хорошо."
# Словарь произношения: правило подменяется детерминированно, без базы.
DICT_PHRASE = "Мы любим SQL."
DICT_FINAL = "Мы любим эскьюэль."

# Одно и то же состояние RNG до постановки задачи — чтобы сид куска совпал.
SEED = 1234


def _advance_seed(value: int = SEED) -> None:
    """Ставит RNG в известное состояние до постановки задачи.

    Сид выбирается ``random.randrange`` внутри пайплайна (``_synthesize_chunk``),
    поэтому без этого шага два рендера одного текста получают разные сиды, и
    parity-тест сравнивал бы случайность, а не пути.
    """
    random.seed(value)


def _calls_since(stub, mark: int) -> list[dict]:
    """Записанные заглушкой вызовы после отметки — вход конкретного рендера."""
    return [dict(call) for call in stub.calls[mark:]]


def _is_done(job: dict) -> bool:
    return job["status"] == "done"


async def _run_continuous(client, stub, text: str, *, seed: int = SEED, **settings):
    """Рендер сплошного текста и фактические вызовы движка."""
    payload = {"text": text, "voice_id": "voice1", "output_format": "wav", **settings}
    mark = len(stub.calls)
    _advance_seed(seed)
    response = await client.post("/api/render-text", json=payload)
    assert response.status_code == 202, response.text
    await _wait(client, response.json()["job_id"], _is_done)
    return _calls_since(stub, mark)


async def _run_dialogue(client, stub, text: str, *, seed: int = SEED, speaker=None):
    """Рендер диалога и фактические вызовы движка."""
    speaker = {"voice_id": "voice1"} if speaker is None else speaker
    payload = {
        "dialogue_text": text,
        "speakers": {"ИВАН": speaker},
        "output_format": "wav",
    }
    mark = len(stub.calls)
    _advance_seed(seed)
    response = await client.post("/api/generate", json=payload)
    assert response.status_code == 202, response.text
    await _wait(client, response.json()["job_id"], _is_done)
    return _calls_since(stub, mark)


async def _run_project(client, stub, source_text: str, *, seed: int = SEED, speaker=None):
    """Проектный путь: PATCH голоса → parse → render. Возвращает вызовы и id."""
    speaker = {"voice_id": "voice1"} if speaker is None else speaker
    created = await client.post(
        "/api/projects",
        json={"name": "Parity", "source_text": source_text, "mode": "dialogue"},
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    assigned = await client.patch(
        f"/api/projects/{project_id}", json={"speakers": {"ИВАН": speaker}}
    )
    assert assigned.status_code == 200, assigned.text
    parsed = await client.post(
        f"/api/projects/{project_id}/parse", json={"chunk_strategy": "paragraph"}
    )
    assert parsed.status_code == 200, parsed.text

    mark = len(stub.calls)
    _advance_seed(seed)
    accepted = await client.post(
        f"/api/projects/{project_id}/render", json={"output_format": "wav"}
    )
    assert accepted.status_code == 202, accepted.text
    await _wait(client, accepted.json()["job_id"], _is_done)
    return _calls_since(stub, mark), project_id


def _single(calls: list[dict], name: str) -> dict:
    """Один кусок на рендер — иначе сравнение текста было бы сравнением нарезки."""
    assert len(calls) == 1, (name, calls)
    return calls[0]


def test_same_text_continuous_vs_dialogue_normalized_text(stub, fake_store, monkeypatch):
    """Нормализация текста совпадает: сплошной текст, диалог и проект."""

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, NUMBER_PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {NUMBER_PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {NUMBER_PHRASE}")

            assert _single(continuous, "continuous")["text"] == NUMBER_FINAL
            assert _single(dialogue, "dialogue")["text"] == NUMBER_FINAL
            assert _single(project, "project")["text"] == NUMBER_FINAL
            # Без изменения текста тест был бы пустым: число обязано развернуться.
            assert NUMBER_FINAL != NUMBER_PHRASE
            assert continuous == dialogue == project

    asyncio.run(scenario())


def test_same_text_continuous_vs_dialogue_dictionary_text(stub, fake_store, monkeypatch):
    """Словарь произношения применяется одинаково во всех трёх режимах."""
    monkeypatch.setattr(
        audio_pipeline,
        "active_rules",
        lambda: [PronunciationRule("SQL", "эскьюэль")],
    )

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, DICT_PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {DICT_PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {DICT_PHRASE}")

            assert _single(continuous, "continuous")["text"] == DICT_FINAL
            assert _single(dialogue, "dialogue")["text"] == DICT_FINAL
            assert _single(project, "project")["text"] == DICT_FINAL
            assert continuous == dialogue == project

    asyncio.run(scenario())


def test_same_text_continuous_vs_dialogue_final_text(stub, fake_store, monkeypatch):
    """Итог preprocessing (включая «ё» и ударения) совпадает во всех режимах."""
    # Движок, который «понимает» ударения: стадия `accentuate` должна получить
    # один и тот же текст во всех режимах. Сама разметка подменяется детерминированно
    # — реальный RUAccent в тестах не поднимается.
    stub.info = EngineInfo(
        id="stub",
        label="Заглушка",
        description="Тестовый движок вместо F5-TTS и XTTS — модели не поднимаются.",
        supports_accents=True,
    )
    monkeypatch.setattr(
        audio_pipeline,
        "accentuate",
        lambda text: text.replace("году", "го+ду"),
    )
    expected = YO_FINAL.replace("году", "го+ду")

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, YO_PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {YO_PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {YO_PHRASE}")

            assert _single(continuous, "continuous")["text"] == expected
            assert _single(dialogue, "dialogue")["text"] == expected
            assert _single(project, "project")["text"] == expected
            # Стадии «ё» и ударений действительно отработали, а не совпали вход и выход.
            assert expected != YO_PHRASE
            assert continuous == dialogue == project

    asyncio.run(scenario())


def test_same_voice_continuous_vs_dialogue_effective_settings(stub, fake_store, monkeypatch):
    """Выбранные ручки доезжают до движка одинаково во всех трёх режимах."""
    settings = {"speed": 1.15, "cfg_strength": 2.5, "nfe_step": 16}
    project_speaker = {"voice_id": "voice1", "overrides": dict(settings)}

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, PHRASE, **settings)
            dialogue = await _run_dialogue(
                client, stub, f"ИВАН: {PHRASE}", speaker={"voice_id": "voice1", **settings}
            )
            project, _project_id = await _run_project(
                client, stub, f"ИВАН: {PHRASE}", speaker=project_speaker
            )

            for name, calls in (
                ("continuous", continuous),
                ("dialogue", dialogue),
                ("project", project),
            ):
                call = _single(calls, name)
                assert call["speed"] == settings["speed"], name
                assert call["params"]["cfg_strength"] == settings["cfg_strength"], name
                assert call["params"]["nfe_step"] == settings["nfe_step"], name
                assert call["params"]["target_rms"] == config.DEFAULT_TARGET_RMS, name
                assert (
                    call["params"]["cross_fade_duration"] == config.DEFAULT_CROSS_FADE_DURATION
                ), name
            assert continuous == dialogue == project

    asyncio.run(scenario())


def test_same_voice_continuous_vs_dialogue_reference_input(stub, fake_store, monkeypatch):
    """Референс-аудио и референс-текст одинаковы: путь к голосу не различается."""

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {PHRASE}")

            for name, calls in (
                ("continuous", continuous),
                ("dialogue", dialogue),
                ("project", project),
            ):
                call = _single(calls, name)
                assert call["ref_audio_path"] == str(fake_store.audio_path), name
                assert call["ref_text"] == fake_store.ref_text, name
            assert continuous == dialogue == project

    asyncio.run(scenario())


def test_same_voice_continuous_vs_dialogue_engine_input(stub, fake_store, monkeypatch):
    """Полный вход на границе ``engine.synthesize()`` совпадает у трёх путей."""

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            continuous = await _run_continuous(client, stub, PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {PHRASE}")

            assert continuous == dialogue == project
            call = _single(continuous, "continuous")
            assert set(call) == {"text", "ref_audio_path", "ref_text", "speed", "params"}
            assert isinstance(call["params"]["seed"], int)

    asyncio.run(scenario())


def test_preview_final_equals_render_input(stub, fake_store, monkeypatch):
    """Итог «Что услышит модель» — ровно то, что уходит в движок при рендере."""
    # Движок без ударений, как у заглушки: preview с `engine=xtts` отдаёт тот же
    # путь preprocessing, что и синтез (`supports_accents=False`).
    assert ENGINE_INFOS["xtts"].supports_accents is False

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            preview = await client.post(
                "/api/text/preview",
                json={"text": YO_PHRASE, "engine": "xtts", "auto_accent": True},
            )
            assert preview.status_code == 200, preview.text
            stages = preview.json()
            assert stages["final"] == YO_FINAL
            assert stages["accents_applied"] is False

            continuous = await _run_continuous(client, stub, YO_PHRASE)
            dialogue = await _run_dialogue(client, stub, f"ИВАН: {YO_PHRASE}")
            project, _project_id = await _run_project(client, stub, f"ИВАН: {YO_PHRASE}")
            for name, calls in (
                ("continuous", continuous),
                ("dialogue", dialogue),
                ("project", project),
            ):
                assert _single(calls, name)["text"] == stages["final"], name

    asyncio.run(scenario())


def test_same_seed_gives_same_engine_input(stub, fake_store, monkeypatch):
    """Сид определяет вход движка: то же состояние RNG — тот же вызов, другое — нет."""
    assert stub.supports_seed is True

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            first = await _run_continuous(client, stub, PHRASE, seed=111)
            second = await _run_continuous(client, stub, PHRASE, seed=111)
            other = await _run_continuous(client, stub, PHRASE, seed=222)

            assert first == second
            assert first[0]["params"]["seed"] == second[0]["params"]["seed"]
            assert first[0]["params"]["seed"] != other[0]["params"]["seed"]
            assert first != other

    asyncio.run(scenario())


def test_render_records_seed_for_every_chunk(stub, fake_store, monkeypatch):
    """Сид каждого куска сохранён и в задаче, и в варианте проекта — иначе не повторить."""

    async def scenario():
        async with _client(monkeypatch) as (client, _queue):
            job_id = await _generate(client)
            data = await _wait(client, job_id, _is_done)
            seeds = [call["params"]["seed"] for call in stub.calls]
            assert len(seeds) == len(data["replicas"]) == 2
            assert all(isinstance(seed, int) for seed in seeds)
            assert [replica["seed"] for replica in data["replicas"]] == seeds

            project, project_id = await _run_project(client, stub, f"ИВАН: {PHRASE}")
            project_seeds = [call["params"]["seed"] for call in project]
            reopened = (await client.get(f"/api/projects/{project_id}")).json()
            takes = [replica["takes"][0] for replica in reopened["replicas"]]
            assert [take["seed"] for take in takes] == project_seeds
            assert all(isinstance(take["seed"], int) for take in takes)

    asyncio.run(scenario())
