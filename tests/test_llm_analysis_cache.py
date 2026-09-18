"""Кеш и инвалидация разборов LLM (Task 2, фаза 3).

Проверяется то, ради чего кеш вообще заведён и чем он опасен: одинаковый вход не
должен гоняться через модель дважды, но ни одно изменение входа — текст, контекст,
словарь, digest модели, версия prompt'а или схемы — не должно оставлять старый разбор
«действительным». Ошибка здесь тихая: пользователь увидит предложение, сделанное по
другому тексту, и не поймёт, почему оно не подходит.

Модель не поднимается: разборы строятся как объекты, а база — временная (фикстура
`workspace` из conftest уводит `DB_PATH` в tmp_path).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.db.connection import connect, init_db
from backend.db.migrations import MIGRATIONS, apply_migrations
from backend.db.repositories.llm_analyses import LlmAnalysesRepository
from backend.db.store import get_projects_store
from backend.llm import analysis_cache as cache
from backend.llm import versioning
from backend.llm import schemas as s
from backend.llm.analyzer import (
    STATUS_NEEDS_REVIEW,
    STATUS_READY,
    STATUS_STALE,
    ReplicaAnalysis,
)

TEXT = "Мы гуляли вокруг старого замка на холме."


def _analysis(**overrides) -> ReplicaAnalysis:
    base = {
        "replica_id": 1,
        "status": STATUS_READY,
        "items": (
            s.Annotation(
                span_start=25,
                span_end=30,
                source="замка",
                type=s.TYPE_HOMOGRAPH,
                meaning="строение, крепость",
                confidence=0.9,
                needs_review=False,
            ),
        ),
        "model_tag": "qwen3:8b",
        "model_digest": "digest-primary",
        "prompt_version": versioning.PROMPT_VERSION,
        "schema_version": s.SCHEMA_VERSION,
        "source_hash": cache.text_hash(TEXT),
        "context_hash": cache.context_hash({"target_text": TEXT}),
        "dictionary_hash": cache.dictionary_hash([]),
    }
    return ReplicaAnalysis(**{**base, **overrides})


def _project_id() -> str:
    return get_projects_store().create_project("Кеш", source_text="")["id"]


def test_analysis_is_cached_for_unchanged_replica():
    """Тот же вход — разбор берётся из кеша, модель не нужна."""
    store = get_projects_store()
    project_id = _project_id()
    analysis = _analysis()

    with connect() as connection, connection:
        repo = LlmAnalysesRepository(connection)
        analysis_cache = cache.AnalysisCache(repo)
        analysis_cache.save(project_id, 0, analysis)
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(analysis))

    assert hit.hit is True
    assert hit.analysis is not None
    assert hit.analysis.items[0].source == "замка"
    assert hit.analysis.status == STATUS_READY
    assert store.llm_analyses(project_id)[0]["model_tag"] == "qwen3:8b"


def test_source_change_invalidates_analysis():
    """Правка текста реплики делает прежний разбор недействительным."""
    project_id = _project_id()
    analysis = _analysis()
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)

        changed = _analysis(source_hash=cache.text_hash(TEXT + " И ещё."))
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(changed))
    assert hit.hit is False
    assert "source_text_hash" in hit.reason


def test_context_change_invalidates_dependent_analysis():
    """Изменился сосед — разбор соседней реплики тоже перестаёт быть верным."""
    project_id = _project_id()
    analysis = _analysis(
        context_hash=cache.context_hash(
            {"target_text": TEXT, "context_before": ["Ты уже пришёл?"]}
        )
    )
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)

        new_context = cache.context_hash(
            {"target_text": TEXT, "context_before": ["Ты уже пришёл домой?"]}
        )
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(_analysis(context_hash=new_context)))
    assert hit.hit is False
    assert "context_hash" in hit.reason


def test_model_and_prompt_and_schema_changes_invalidate_analysis():
    """Смена модели, prompt'а или схемы — другой эксперимент, прежний разбор не годится."""
    project_id = _project_id()
    analysis = _analysis()
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)

        for override, field in (
            ({"model_digest": "другой-digest"}, "model_digest"),
            ({"prompt_version": versioning.PROMPT_VERSION + "-другой"}, "prompt_version"),
            ({"schema_version": str(int(s.SCHEMA_VERSION) + 1)}, "schema_version"),
        ):
            hit = analysis_cache.lookup(project_id, 0, cache.key_for(_analysis(**override)))
            assert hit.hit is False, override
            assert field in hit.reason


def test_dictionary_change_invalidates_affected_analysis():
    """Правило словаря меняет чтение — разбор, сделанный до правки, недействителен."""
    project_id = _project_id()
    before = [{"source": "замок", "target": "замо́к", "project_id": project_id}]
    after = [{"source": "замок", "target": "за́мок", "project_id": project_id}]
    analysis = _analysis(dictionary_hash=cache.dictionary_hash(before))

    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)
        hit = analysis_cache.lookup(
            project_id, 0, cache.key_for(_analysis(dictionary_hash=cache.dictionary_hash(after)))
        )
    assert cache.dictionary_hash(before) != cache.dictionary_hash(after)
    assert hit.hit is False
    assert "dictionary_hash" in hit.reason


def test_dictionary_hash_ignores_cosmetic_fields():
    """Заметка и время правки правила на разбор не влияют — и не инвалидируют его."""
    base = [{"source": "замок", "target": "замо́к", "note": "старое"}]
    cosmetic = [
        {
            "source": "замок",
            "target": "замо́к",
            "note": "новое",
            "created_at": "2026-09-18T00:00:00+00:00",
            "enabled": True,
        }
    ]
    assert cache.dictionary_hash(base) == cache.dictionary_hash(cosmetic)


def test_stale_analysis_is_not_returned_as_fresh():
    """Помеченный устаревшим разбор не отдаётся как актуальный, но и не удаляется."""
    store = get_projects_store()
    project_id = _project_id()
    analysis = _analysis()
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)

    store.mark_llm_analyses_stale(project_id, [0])

    with connect() as connection:
        repo = LlmAnalysesRepository(connection)
        hit = cache.AnalysisCache(repo).lookup(project_id, 0, cache.key_for(analysis))
        stored = repo.get(project_id, 0)

    assert hit.hit is False
    assert "устаревшим" in hit.reason
    assert stored is not None and stored["status"] == STATUS_STALE, "история сохраняется"


def test_explicit_invalidation_covers_context_neighbourhood():
    """Явная инвалидация помечает реплику и её контекстный район, а не весь проект."""
    project_id = _project_id()
    with connect() as connection, connection:
        repo = LlmAnalysesRepository(connection)
        analysis_cache = cache.AnalysisCache(repo)
        for index in range(6):
            analysis_cache.save(project_id, index, _analysis(replica_id=index + 1))
        affected = analysis_cache.mark_stale(project_id, [3])
        statuses = {int(row["replica_index"]): row["status"] for row in repo.list_for_project(project_id)}

    assert affected == 5, "реплика 3 и по два соседа с каждой стороны"
    assert statuses[1] == STATUS_STALE
    assert statuses[5] == STATUS_STALE
    assert statuses[0] == STATUS_READY, "дальние реплики не тронуты"


def test_invalidate_analysis_marks_llm_analyses_stale():
    """Существующая инвалидация подготовки обязана устаревать и разборы LLM."""
    store = get_projects_store()
    project = store.create_project(
        "Диалог", source_text="ИВАН: Мы гуляли вокруг старого замка на холме."
    )
    project = store.parse_project(project["id"])
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project["id"], 0, _analysis())

    store.invalidate_analysis(project["id"], indexes=[0], reason="текст изменён")
    stored = store.llm_analyses(project["id"])[0]
    assert stored["status"] == STATUS_STALE


def test_needs_review_analysis_is_valid_for_cache():
    """Разбор с `needs_review` считается валидным: пересчитывать его незачем."""
    project_id = _project_id()
    analysis = _analysis(status=STATUS_NEEDS_REVIEW)
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(analysis))
    assert hit.hit is True
    assert hit.analysis is not None and hit.analysis.status == STATUS_NEEDS_REVIEW


def test_failed_analysis_is_not_served_from_cache():
    """Неудачный разбор не кешируется как актуальный: его нужно повторить."""
    project_id = _project_id()
    analysis = _analysis(status="FAILED", error="OllamaUnavailableError")
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 0, analysis)
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(analysis))
    assert hit.hit is False
    assert "FAILED" in hit.reason


def test_corrupted_saved_analysis_is_not_served():
    """Повреждённая запись в базе не должна притворяться валидным разбором."""
    project_id = _project_id()
    with connect() as connection, connection:
        repo = LlmAnalysesRepository(connection)
        analysis_cache = cache.AnalysisCache(repo)
        analysis_cache.save(project_id, 0, _analysis())
        connection.execute(
            "UPDATE llm_analyses SET analysis_json = '{битый json' WHERE project_id = ?",
            (project_id,),
        )
        hit = analysis_cache.lookup(project_id, 0, cache.key_for(_analysis()))
    assert hit.hit is False
    assert "повреждён" in hit.reason


def test_migration_7_upgrades_existing_database_without_data_loss(tmp_path):
    """Миграция разборов добавляется к существующей базе и не трогает данные."""
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    # База «прошлой версии»: схема до разборов LLM.
    for version, script in MIGRATIONS:
        if version >= 7:
            break
        connection.executescript(script)
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    connection.execute("DELETE FROM schema_version")
    connection.execute("INSERT INTO schema_version (version) VALUES (6)")
    connection.execute(
        "INSERT INTO projects (id, name, source_text, mode, render_settings, status, created_at, updated_at)"
        " VALUES ('p1', 'Старый проект', 'текст', 'dialogue', '{}', 'raw', '2026-01-01', '2026-01-01')"
    )
    connection.commit()
    assert apply_migrations(connection) == MIGRATIONS[-1][0]
    # Идемпотентность: повторный вызов ничего не меняет.
    assert apply_migrations(connection) == MIGRATIONS[-1][0]
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "llm_analyses" in tables
    assert connection.execute("SELECT name FROM projects WHERE id = 'p1'").fetchone()[0] == (
        "Старый проект"
    )
    # Новая таблица работает на обновлённой базе.
    repo = LlmAnalysesRepository(connection)
    repo.upsert(
        analysis_id="a1",
        project_id="p1",
        replica_id=1,
        replica_index=0,
        analysis_json="{}",
        status="READY",
    )
    assert repo.get("p1", 0)["status"] == "READY"
    connection.close()


def test_init_db_creates_analysis_table(workspace):
    """Обычная инициализация базы создаёт таблицу разборов и её индексы."""
    init_db()
    with connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        indexes = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    assert "llm_analyses" in tables
    assert {"idx_llm_analyses_project", "idx_llm_analyses_status"} <= indexes


@pytest.mark.parametrize("status", [STATUS_READY, STATUS_NEEDS_REVIEW])
def test_saved_analysis_round_trips_annotations(status):
    """Разбор переживает запись и чтение: аннотации, подсказка и служебные поля."""
    project_id = _project_id()
    analysis = _analysis(status=status)
    with connect() as connection, connection:
        analysis_cache = cache.AnalysisCache(LlmAnalysesRepository(connection))
        analysis_cache.save(project_id, 2, analysis)
        hit = analysis_cache.lookup(project_id, 2, cache.key_for(analysis))
    restored = hit.analysis
    assert restored is not None
    assert restored.replica_id == analysis.replica_id
    assert restored.status == status
    assert restored.items[0].meaning == "строение, крепость"
    assert restored.items[0].needs_review is False
    assert restored.model_tag == "qwen3:8b"
    assert restored.prompt_version == versioning.PROMPT_VERSION


def test_llm_model_is_unloaded_after_analysis_pass():
    """После прохода анализа модель выгружается: память нужна рендеру (§14.3).

    Иначе резидентная LLM (2--6 GB) останется висеть рядом с F5, и синтез упрётся в
    ту же память, которую анализатор только что занимал.
    """
    from backend.llm import analyzer as llm
    from backend.llm import integration as integration_module
    from backend.llm import memory_policy
    from backend.llm import scheduler as scheduler_module
    from backend.llm.fake_client import FakeOllamaClient
    from backend.llm.ollama_client import OllamaModel

    client = FakeOllamaClient(
        models=[OllamaModel(tag="qwen3:8b", digest="d1", size_bytes=5 * 1024**3)],
        responder=lambda case: json.dumps(
            {
                "schema_version": s.SCHEMA_VERSION,
                "replica_id": case["replica_id"],
                "items": [],
                "utterance": {"class": "NORMAL"},
            },
            ensure_ascii=False,
        ),
    )
    analyzer = llm.LinguisticAnalyzer(
        settings=llm.AnalyzerSettings(enabled=True), client=client
    )
    runner = integration_module.ProjectLlmAnalyzer(
        analyzer=analyzer,
        scheduler=scheduler_module.HeavyScheduler(
            gate=memory_policy.HeavyGate(),
            sensor=lambda: {"system_percent": 40.0, "available_gb": 12.0, "pressure": "green"},
        ),
        lookup=lambda index, key: None,
        save=lambda index, analysis: None,
    )
    outcome = runner.run(project_id="p1", replicas=[{"index": 0, "text": "Да."}])
    assert outcome.calls == 1
    assert outcome.unloaded is True
    assert client.unloaded == ["qwen3:8b"], "модель обязана быть выгружена после прохода"
    assert client.loaded_models() == []

    # Проход целиком из кеша модель не грузит и никого не выгружает.
    cached = integration_module.ProjectLlmAnalyzer(
        analyzer=analyzer,
        scheduler=runner.scheduler,
        lookup=lambda index, key: integration_module.ReplicaAnalysis(
            replica_id=1, status=llm.STATUS_READY
        ),
        save=lambda index, analysis: None,
    )
    outcome = cached.run(project_id="p1", replicas=[{"index": 0, "text": "Да."}])
    assert outcome.calls == 0 and outcome.unloaded is False
