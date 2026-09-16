"""Словарь произношения: правила, применение, доезд до синтеза и REST API.

Тесты повторяют десять пунктов плана фазы 6 и проверяют границы, на которых
словарь чаще всего ломается: подстрока внутри слова, регистр, порядок длинных
правил, приоритет над RUAccent и знак «+» у движка без ударений. Модели не
поднимаются: движок — заглушка, база уведена в tmp_path фикстурой `workspace`.
"""

import asyncio
import contextlib

import httpx
import pytest
from conftest import StubEngine

from backend import audio_pipeline, main
from backend.accentizer import Accentizer
from backend.audio_pipeline import RenderSettings, SpeakerSettings, _text_for_engine
from backend.dialogue_parser import Replica
from backend.engines.base import STATE_READY, EngineInfo
from backend.pronunciation import get_store
from backend.text_normalization import normalize, normalize_report
from backend.text_normalization.pronunciation import (
    PronunciationRule,
    apply_pronunciation,
)


class AccentStubEngine(StubEngine):
    """StubEngine, который говорит, что понимает «+»-ударения (как F5-TTS)."""

    info = EngineInfo(
        id="stub-accents",
        label="Заглушка с ударениями",
        description="Тестовый движок вместо F5-TTS — модель не поднимается.",
        supports_accents=True,
    )


# --- 1-2. точное слово и границы -------------------------------------------------
def test_exact_word_replacement():
    rules = [PronunciationRule("SQL", "эскьюэль")]
    assert apply_pronunciation("Мы любим SQL.", rules) == "Мы любим эскьюэль."


def test_whole_word_does_not_match_inside_longer_word():
    rules = [PronunciationRule("SQL", "эскьюэль")]
    text = "PostgreSQL и MySQL — это SQL."
    assert apply_pronunciation(text, rules) == "PostgreSQL и MySQL — это эскьюэль."


def test_whole_word_boundaries_work_for_cyrillic():
    rules = [PronunciationRule("кот", "кат")]
    # «который» содержит «кот», но это другое слово: с \b кириллица ведёт себя
    # неочевидно, поэтому границы заданы явным классом символов.
    assert apply_pronunciation("Кот спит, который час.", rules) == "кат спит, который час."


def test_loose_rule_matches_inside_words():
    rules = [PronunciationRule("SQL", "эскьюэль", whole_word=False)]
    assert apply_pronunciation("PostgreSQL", rules) == "Postgreэскьюэль"


def test_special_characters_are_escaped_literally():
    assert apply_pronunciation("Я знаю C++ хорошо", [PronunciationRule("C++", "си плюс плюс")]) == (
        "Я знаю си плюс плюс хорошо"
    )
    # Точка — именно точка, а не «любой символ»: «axb» замене не подлежит.
    assert apply_pronunciation("a.b и axb", [PronunciationRule("a.b", "эй-би")]) == "эй-би и axb"


# --- 3. регистр -------------------------------------------------------------------
def test_case_sensitive_switch():
    insensitive = [PronunciationRule("sql", "эскьюэль")]
    assert apply_pronunciation("SQL и sql", insensitive) == "эскьюэль и эскьюэль"
    sensitive = [PronunciationRule("SQL", "эскьюэль", case_sensitive=True)]
    assert apply_pronunciation("SQL и sql", sensitive) == "эскьюэль и sql"


# --- 4. несколько терминов и приоритет длинного правила ---------------------------
def test_longer_source_is_applied_before_shorter():
    rules = [
        PronunciationRule("OpenAI", "оупен эй-ай"),
        PronunciationRule("OpenAI API", "оупен эй-ай апи"),
    ]
    assert apply_pronunciation("OpenAI API и OpenAI", rules) == "оупен эй-ай апи и оупен эй-ай"


def test_rules_apply_to_result_of_previous_rule():
    # «калькулятор» заменяется на «SQL», а следующее правило читает уже его:
    # каждое правило видит результат предыдущего.
    rules = [
        PronunciationRule("калькулятор", "SQL"),
        PronunciationRule("SQL", "эскьюэль"),
    ]
    assert apply_pronunciation("Это калькулятор.", rules) == "Это эскьюэль."


def test_several_terms_in_one_pass():
    rules = [
        PronunciationRule("PostgreSQL", "постгрес"),
        PronunciationRule("SQL", "эскьюэль"),
        PronunciationRule("OpenAI", "оупен эй-ай"),
    ]
    text = "PostgreSQL, SQL и OpenAI"
    assert apply_pronunciation(text, rules) == "постгрес, эскьюэль и оупен эй-ай"


# --- 5-6. разметка ударений F5 и движок без ударений ------------------------------
def test_stress_markup_is_kept_for_engine_with_accents():
    rules = [PronunciationRule("звонит", "звон+ит")]
    assert apply_pronunciation("Он звонит", rules, supports_accents=True) == "Он звон+ит"


def test_stress_markup_is_stripped_for_engine_without_accents():
    rules = [PronunciationRule("звонит", "звон+ит")]
    # XTTS прочитала бы «+» вслух — для неё правило становится «совместимой» заменой.
    assert apply_pronunciation("Он звонит", rules, supports_accents=False) == "Он звонит"


def test_plus_that_is_not_stress_markup_is_kept():
    rules = [PronunciationRule("язык", "C++ язык")]
    assert apply_pronunciation("язык", rules, supports_accents=False) == "C++ язык"


# --- 7. словарь применяется после нормализации ------------------------------------
def test_dictionary_sees_word_appeared_after_normalization():
    # «12» → «двенадцать» до словаря, иначе правило не нашло бы слово.
    rules = [PronunciationRule("двенадцать", "двена+дцать")]
    assert normalize("12 попыток", rules) == "двена+дцать попыток"
    # «25 руб.» → «двадцать пять рублей» — правило ловит уже развёрнутое слово.
    money = [PronunciationRule("рублей", "рубле+й")]
    assert normalize("25 руб.", money) == "двадцать пять рубле+й"


def test_dictionary_can_override_latin_acronym():
    # Шаг латиницы иначе прочитал бы SQL по буквам: правило обязано успеть раньше.
    rules = [PronunciationRule("SQL", "эскьюэль")]
    assert normalize("База SQL готова", rules) == "База эскьюэль готова"


# --- 9. пустой словарь ничего не меняет -------------------------------------------
def test_empty_dictionary_changes_nothing():
    text = "SQL и PostgreSQL, 12 попыток."
    assert apply_pronunciation(text) == text
    assert apply_pronunciation(text, []) == text
    assert normalize(text, []) == normalize(text)
    assert normalize(text, None) == normalize(text)


def test_disabled_rule_is_skipped():
    rules = [PronunciationRule("SQL", "эскьюэль", enabled=False)]
    assert apply_pronunciation("SQL", rules) == "SQL"


# --- 10. некорректные записи не ломают синтез -------------------------------------
def test_malformed_entries_are_skipped_without_failing():
    entries = [
        PronunciationRule("", "эскьюэль"),        # пустой источник
        PronunciationRule("SQL", ""),            # пустая замена
        PronunciationRule("   ", "ничего"),      # источник из пробелов
        ("PostgreSQL", None),                    # замена не строка
        "совсем не правило",                     # непонятный формат
        {"source": "SQL", "target": "эскьюэль"},  # корректная запись
    ]
    assert apply_pronunciation("SQL", entries) == "эскьюэль"


# --- приоритет словаря над RUAccent ----------------------------------------------
def _accentizer_with_fake_accent():
    """Accentizer с подставной моделью: настоящий RUAccent не поднимается."""
    accentizer = Accentizer()
    calls: list[str] = []

    class FakeAccent:
        def process_all(self, text: str) -> str:
            calls.append(text)
            return f"<{text}>"

    accentizer._accent = FakeAccent()
    accentizer._state = STATE_READY
    return accentizer, calls


def test_manual_stress_survives_and_rest_is_accentized():
    accentizer, calls = _accentizer_with_fake_accent()
    result = accentizer.accentuate("Он звон+ит другу, и другу весело.")
    # Ручное ударение осталось дословно, остальной текст всё равно прошёл модель.
    assert result == "<Он >звон+ит< другу, и другу весело.>"
    # Акцентируем промежутки между защищёнными словами, а не каждое слово.
    assert calls == ["Он ", " другу, и другу весело."]


def test_accentizer_without_manual_stress_keeps_old_behaviour():
    accentizer, calls = _accentizer_with_fake_accent()
    assert accentizer.accentuate("Он звонит другу.") == "<Он звонит другу.>"
    assert calls == ["Он звонит другу."]


def test_fully_manual_text_does_not_load_accent_model():
    accentizer, calls = _accentizer_with_fake_accent()
    assert accentizer.accentuate("звон+ит") == "звон+ит"
    assert calls == []


def test_dictionary_has_priority_over_ruaccent(monkeypatch):
    monkeypatch.setattr(
        audio_pipeline, "active_rules", lambda: [PronunciationRule("звонит", "звон+ит")]
    )
    # Заглушка «RUAccent» подчёркивает, что словарь применился до акцентуации:
    # ручной «+» в неё уже пришёл, а нетронутый текст она разметила.
    monkeypatch.setattr(
        audio_pipeline, "accentuate", lambda text: text.replace("другу", "дру+гу")
    )
    assert _text_for_engine("Он звонит другу", AccentStubEngine(), True) == "Он звон+ит дру+гу"


# --- интеграция с пайплайном ------------------------------------------------------
def test_text_for_engine_applies_dictionary_for_both_engines():
    store = get_store()
    store.create(source="SQL", target="эскьюэль")
    store.create(source="звонит", target="звон+ит")

    plain = StubEngine()  # supports_accents=False
    assert _text_for_engine("SQL и звонит", plain, True) == "эскьюэль и звонит"

    accents = AccentStubEngine()  # supports_accents=True
    # auto_accent=False: ударения ставит RUAccent, а не словарь, — проверяем только
    # что разметка из замены дошла до движка, который её понимает.
    assert _text_for_engine("SQL и звонит", accents, False) == "эскьюэль и звон+ит"


def test_dictionary_reaches_stub_engine_during_render(stub, fake_store):
    get_store().create(source="SQL", target="эскьюэль")
    asyncio.run(
        audio_pipeline.render_dialogue(
            job_id="job-dictionary",
            replicas=[Replica(voice="#1", text="SQL и PostgreSQL", line_number=1)],
            speakers={"#1": SpeakerSettings(voice_id="voice1")},
            settings=RenderSettings(pause_ms=0, output_format="wav"),
        )
    )
    assert stub.calls[0]["text"] == "эскьюэль и PostgreSQL"


def test_active_rules_cache_is_invalidated_by_writes():
    store = get_store()
    assert store.active_rules() == []

    entry = store.create(source="SQL", target="эскьюэль")
    assert [rule.source for rule in store.active_rules()] == ["SQL"]

    store.update(entry["id"], target="сиквел")
    assert store.active_rules()[0].target == "сиквел"

    store.update(entry["id"], enabled=False)
    assert store.active_rules() == []

    store.delete(entry["id"])
    assert store.active_rules() == []


def test_repeated_create_updates_instead_of_duplicating():
    store = get_store()
    first = store.create(source="SQL", target="эскьюэль")
    second = store.create(source="SQL", target="сиквел")
    assert first["id"] == second["id"]
    assert len(store.list_entries()) == 1
    assert store.list_entries()[0]["target"] == "сиквел"


def test_migration_3_upgrades_existing_database(tmp_path):
    """База, созданная на фазе 5 (версия 2), доводится до словаря без пересоздания."""
    import sqlite3

    from backend.db import migrations

    connection = sqlite3.connect(tmp_path / "old.db")
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for version, script in migrations.MIGRATIONS:
        if version <= 2:
            connection.executescript(script)
            connection.execute("DELETE FROM schema_version")
            connection.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 2

    assert migrations.apply_migrations(connection) == 3
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'pronunciation_entries'"
    ).fetchone() is not None
    # Повторный запуск миграций идемпотентен, а уникальность правила на месте.
    assert migrations.apply_migrations(connection) == 3
    connection.execute(
        "INSERT INTO pronunciation_entries"
        " (source, target, case_sensitive, whole_word, enabled, note, created_at, updated_at)"
        " VALUES ('SQL', 'эскьюэль', 0, 1, 1, '', 'now', 'now')"
    )
    # UNIQUE(source, case_sensitive) не даёт двум одинаковым правилам сосуществовать.
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO pronunciation_entries"
            " (source, target, case_sensitive, whole_word, enabled, note, created_at, updated_at)"
            " VALUES ('SQL', 'сиквел', 0, 1, 1, '', 'now', 'now')"
        )
    count = connection.execute("SELECT COUNT(*) FROM pronunciation_entries").fetchone()[0]
    connection.close()
    assert count == 1


# --- REST API ---------------------------------------------------------------------
@contextlib.asynccontextmanager
async def _client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _run(scenario) -> None:
    asyncio.run(scenario())


def test_pronunciation_crud_api():
    async def scenario():
        async with _client() as client:
            created = await client.post(
                "/api/pronunciation", json={"source": "OpenAI", "target": "оупен эй-ай"}
            )
            assert created.status_code == 201, created.text
            entry = created.json()
            assert entry["source"] == "OpenAI"
            assert entry["target"] == "оупен эй-ай"
            assert entry["whole_word"] is True
            assert entry["case_sensitive"] is False
            assert entry["enabled"] is True

            listing = await client.get("/api/pronunciation")
            assert listing.status_code == 200
            assert listing.json()["count"] == 1

            patched = await client.patch(
                f"/api/pronunciation/{entry['id']}",
                json={"enabled": False, "target": "оупен", "case_sensitive": True},
            )
            assert patched.status_code == 200, patched.text
            assert patched.json()["enabled"] is False
            assert patched.json()["target"] == "оупен"
            assert patched.json()["case_sensitive"] is True

            missing = await client.patch("/api/pronunciation/9999", json={"enabled": True})
            assert missing.status_code == 404
            assert "не найдено" in missing.json()["detail"]

            deleted = await client.delete(f"/api/pronunciation/{entry['id']}")
            assert deleted.status_code == 200
            assert (await client.get("/api/pronunciation")).json()["count"] == 0
            assert (await client.delete("/api/pronunciation/9999")).status_code == 404

    _run(scenario)


def test_pronunciation_api_validation_and_upsert():
    async def scenario():
        async with _client() as client:
            empty = await client.post("/api/pronunciation", json={"source": "   ", "target": "x"})
            assert empty.status_code == 400
            assert "пуст" in empty.json()["detail"]

            empty_target = await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": " "}
            )
            assert empty_target.status_code == 400

            first = await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": "эскьюэль"}
            )
            second = await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": "сиквел"}
            )
            assert second.status_code == 201
            assert second.json()["id"] == first.json()["id"]
            listing = await client.get("/api/pronunciation")
            assert listing.json()["count"] == 1
            assert listing.json()["entries"][0]["target"] == "сиквел"

    _run(scenario)


def test_patch_to_existing_source_returns_409():
    async def scenario():
        async with _client() as client:
            first = await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": "эскьюэль"}
            )
            second = await client.post(
                "/api/pronunciation", json={"source": "PostgreSQL", "target": "постгрес"}
            )
            conflict = await client.patch(
                f"/api/pronunciation/{second.json()['id']}", json={"source": "SQL"}
            )
            assert conflict.status_code == 409
            assert "уже есть" in conflict.json()["detail"]
            assert first.status_code == 201

    _run(scenario)


def test_preview_shows_stages_and_does_not_write():
    async def scenario():
        async with _client() as client:
            await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": "эскьюэль"}
            )
            before = (await client.get("/api/pronunciation")).json()["count"]

            preview = await client.post(
                "/api/pronunciation/preview", json={"text": "12 запросов и SQL."}
            )
            assert preview.status_code == 200, preview.text
            data = preview.json()
            assert data["original"] == "12 запросов и SQL."
            # Без словаря шаг латиницы прочитал бы SQL по буквам — это и показывает
            # стадия «после нормализации»; словарь перекрывает её на следующем шаге.
            assert data["normalized"] == "двенадцать запросов и эс-кью-эл."
            assert data["result"] == "двенадцать запросов и эскьюэль."
            assert data["matches"] == [{"source": "SQL", "target": "эскьюэль", "count": 1}]

            # Preview ничего не пишет в базу.
            after = (await client.get("/api/pronunciation")).json()["count"]
            assert after == before

    _run(scenario)


def test_preview_respects_engine_and_rejects_unknown():
    async def scenario():
        async with _client() as client:
            await client.post(
                "/api/pronunciation", json={"source": "звонит", "target": "звон+ит"}
            )

            f5 = await client.post(
                "/api/pronunciation/preview", json={"text": "Он звонит", "engine": "f5"}
            )
            assert f5.status_code == 200
            assert f5.json()["result"] == "Он звон+ит"
            assert f5.json()["matches"][0]["target"] == "звон+ит"

            xtts = await client.post(
                "/api/pronunciation/preview", json={"text": "Он звонит", "engine": "xtts"}
            )
            assert xtts.status_code == 200
            assert xtts.json()["result"] == "Он звонит"
            assert xtts.json()["matches"][0]["target"] == "звонит"

            unknown = await client.post(
                "/api/pronunciation/preview", json={"text": "Он звонит", "engine": "nope"}
            )
            assert unknown.status_code == 400
            assert "Неизвестный движок" in unknown.json()["detail"]

            empty = await client.post("/api/pronunciation/preview", json={"text": "   "})
            assert empty.status_code == 400

    _run(scenario)


def test_preview_matches_what_normalize_report_returns():
    # Preview обязан показывать ровно тот результат, что уйдёт в модель, — без
    # второй реализации порядка шагов в обработчике запроса.
    async def scenario():
        async with _client() as client:
            await client.post(
                "/api/pronunciation", json={"source": "SQL", "target": "эскьюэль"}
            )
            preview = await client.post(
                "/api/pronunciation/preview", json={"text": "SQL и 3 попытки.", "engine": "xtts"}
            )
            result, matches = normalize_report(
                "SQL и 3 попытки.",
                get_store().active_rules(),
                supports_accents=False,
            )
            assert preview.json()["result"] == result
            assert preview.json()["matches"] == matches

    _run(scenario)


# --- словарь и проверка качества (QA) -------------------------------------------
def test_wer_counts_dictionary_pronunciation_as_correct(stub, fake_store, monkeypatch):
    """Словарь не должен выглядеть ошибкой в QA.

    Модель произносит замену, Whisper пишет услышанное («эскьюэль»), а reference
    раньше нормализовался без словаря («эс-кью-эл») — WER 0.75 на ровном месте,
    лишние попытки и лишний Whisper. Словарь обязан применяться к обеим сторонам.
    """
    from test_audio_pipeline import _answers, _qa_render, _render, _replica, _speaker

    get_store().create(source="SQL", target="эскьюэль")
    _answers(monkeypatch, "Готов эскьюэль")
    result = _render([_replica("Готов SQL.")], {"#1": _speaker()}, _qa_render(max_attempts=2))

    assert result.qa[0].wer == 0.0
    assert result.qa[0].attempts == 1
    assert len(stub.calls) == 1  # порог взят с первой попытки — повторять нечего


def test_wer_accepts_latin_spelling_from_whisper(stub, fake_store, monkeypatch):
    """Обратный случай: Whisper записал услышанное как «SQL» — тоже не ошибка."""
    from test_audio_pipeline import _answers, _qa_render, _render, _replica, _speaker

    get_store().create(source="OpenAI", target="оупен эй-ай")
    _answers(monkeypatch, "Открой OpenAI")
    result = _render(
        [_replica("Открой OpenAI.")], {"#1": _speaker()}, _qa_render(max_attempts=2)
    )

    assert result.qa[0].wer == 0.0
    assert len(stub.calls) == 1
