"""Хранилище проектов: SQLite-слой без HTTP и без моделей синтеза.

Проверяется именно то, ради чего проект и заводится: данные переживают закрытие
соединения, связанные записи уходят каскадом, а два проекта не смешиваются.
"""

import sqlite3

from backend import config
from backend.db import connection
from backend.db.repositories.takes import TakesRepository
from backend.db.store import ProjectsStore

DIALOGUE = "ИВАН: Первая реплика.\nМАРГО: Вторая реплика."


def _rows(table: str) -> int:
    with connection.transaction() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_project_survives_reopening_connection(workspace):
    store = ProjectsStore()
    created = store.create_project("Первый", source_text=DIALOGUE)
    assert created["name"] == "Первый"
    assert created["mode"] == config.PROJECT_MODE_DIALOGUE
    assert created["status"] == config.PROJECT_STATUS_DRAFT

    # Новый экземпляр хранилища — это новые соединения: так же ведёт себя
    # перезапуск backend, после которого проект обязан открываться.
    reopened = ProjectsStore().get_project(created["id"])
    assert reopened is not None
    assert reopened["source_text"] == DIALOGUE
    assert reopened["created_at"] == created["created_at"]


def test_update_source_text_and_settings(workspace):
    store = ProjectsStore()
    project = store.create_project("Правка", source_text="старый текст")
    updated = store.update_project(
        project["id"], source_text="новый текст", render_settings={"pause_ms": 250}
    )
    assert updated["source_text"] == "новый текст"
    assert updated["render_settings"] == {"pause_ms": 250}
    assert updated["updated_at"] >= project["updated_at"]


def test_parse_saves_speakers_and_replicas(workspace):
    store = ProjectsStore()
    project = store.create_project("Разбор", source_text=DIALOGUE)
    parsed = store.parse_project(project["id"])

    assert [item["key"] for item in parsed["speakers"]] == ["ИВАН", "МАРГО"]
    assert [item["text"] for item in parsed["replicas"]] == [
        "Первая реплика.",
        "Вторая реплика.",
    ]
    # Порядок реплик стабилен и совпадает с текстом — по нему их адресует UI.
    assert [item["index"] for item in parsed["replicas"]] == [0, 1]
    assert [item["speaker"] for item in parsed["replicas"]] == ["ИВАН", "МАРГО"]


def test_parse_keeps_assigned_voices(workspace):
    store = ProjectsStore()
    project = store.create_project("Голоса", source_text=DIALOGUE)
    store.parse_project(project["id"])
    store.update_project(
        project["id"],
        speakers={
            "ИВАН": {"voice_id": "voice-male"},
            "МАРГО": {"voice_id": "voice-female"},
        },
    )
    # Правка текста не должна означать «выбери голоса заново».
    store.update_project(project["id"], source_text=DIALOGUE + "\nИВАН: Третья.")
    reparsed = store.parse_project(project["id"])
    assigned = {item["key"]: item["voice_id"] for item in reparsed["speakers"]}
    assert assigned == {"ИВАН": "voice-male", "МАРГО": "voice-female"}
    assert [item["voice_id"] for item in reparsed["replicas"]] == [
        "voice-male",
        "voice-female",
        "voice-male",
    ]


def test_takes_are_saved_and_selected(workspace):
    store = ProjectsStore()
    project = store.create_project("Варианты", source_text=DIALOGUE)
    parsed = store.parse_project(project["id"])
    replica_id = parsed["replicas"][0]["id"]

    with connection.transaction() as conn:
        first = TakesRepository(conn).add(
            replica_id, audio_path="a.wav", label="рендер 1", seed=11, engine="f5"
        )
        second = TakesRepository(conn).add(
            replica_id, audio_path="b.wav", label="рендер 2", seed=22, engine="xtts"
        )

    store.select_take(project["id"], 0, int(second["id"]))
    project_after = store.get_project(project["id"])
    replica = project_after["replicas"][0]
    assert replica["selected_take_id"] == second["id"]
    assert replica["status"] == config.REPLICA_STATUS_RENDERED
    assert [item["id"] for item in replica["takes"]] == [first["id"], second["id"]]

    # Выбор варианта не требует пересинтеза: он уже лежит готовым файлом.
    store.select_take(project["id"], 0, int(first["id"]))
    assert store.get_project(project["id"])["replicas"][0]["selected_take_id"] == first["id"]


def test_save_render_takes_marks_replicas_and_project(workspace):
    store = ProjectsStore()
    project = store.create_project("Рендер", source_text=DIALOGUE)
    store.parse_project(project["id"])
    store.save_render_takes(
        project["id"],
        "job123",
        [
            {
                "index": index,
                "audio_path": str(workspace / f"r{index}.wav"),
                "label": "рендер job123",
                "seed": 100 + index,
                "engine": "f5",
                "parameters": {"speed": 1.0},
                "duration_sec": 1.5,
                "qa": {"status": "passed", "wer": 0.02, "attempts": 1},
            }
            for index in range(2)
        ],
    )
    project_after = store.get_project(project["id"])
    assert project_after["status"] == config.PROJECT_STATUS_RENDERED
    assert project_after["job_id"] == "job123"
    assert [item["status"] for item in project_after["replicas"]] == [
        config.REPLICA_STATUS_RENDERED,
        config.REPLICA_STATUS_RENDERED,
    ]
    take = project_after["replicas"][1]["takes"][0]
    assert take["engine"] == "f5"
    assert take["seed"] == 101
    assert take["qa"]["wer"] == 0.02
    assert project_after["replicas"][1]["selected_take_id"] == take["id"]


def test_delete_project_cascades(workspace):
    store = ProjectsStore()
    project = store.create_project("Удаляемый", source_text=DIALOGUE)
    store.parse_project(project["id"])
    store.save_render_takes(
        project["id"],
        "job",
        [{"index": 0, "audio_path": str(workspace / "r0.wav"), "duration_sec": 1.0}],
    )
    assert _rows("takes") == 1

    assert store.delete_project(project["id"]) is True
    assert store.get_project(project["id"]) is None
    # Спикеры, реплики и варианты уходят вместе с проектом: иначе в базе остались
    # бы записи, которые не к чему отнести.
    assert (_rows("projects"), _rows("speakers"), _rows("replicas"), _rows("takes")) == (0, 0, 0, 0)
    assert store.delete_project(project["id"]) is False


def test_projects_do_not_mix(workspace):
    store = ProjectsStore()
    first = store.create_project("Первый", source_text=DIALOGUE)
    second = store.create_project("Второй", source_text="ПЕТЯ: Совсем другой текст.")
    store.parse_project(first["id"])
    store.parse_project(second["id"])
    store.update_project(first["id"], speakers={"ИВАН": {"voice_id": "voice-a"}})

    left = store.get_project(first["id"])
    right = store.get_project(second["id"])
    assert [item["speaker"] for item in left["replicas"]] == ["ИВАН", "МАРГО"]
    assert [item["text"] for item in right["replicas"]] == ["Совсем другой текст."]
    assert [item["key"] for item in right["speakers"]] == ["ПЕТЯ"]
    # Назначение голоса в одном проекте не задевает другой.
    assert {item["voice_id"] for item in right["speakers"]} == {""}
    assert len(store.list_projects()) == 2


def test_unknown_project_is_none_or_key_error(workspace):
    store = ProjectsStore()
    assert store.get_project("нет-такого") is None
    assert store.delete_project("нет-такого") is False
    try:
        store.parse_project("нет-такого")
    except KeyError:
        pass
    else:  # pragma: no cover — отсутствие ошибки означает тихую пустую ветку
        raise AssertionError("разбор несуществующего проекта должен быть ошибкой")


def test_projects_do_not_touch_voices_json(workspace):
    """Проекты живут в SQLite и не переписывают голоса.

    Голоса и проекты — разные сущности: падение на записи проекта не должно
    оставлять повреждённый `voices.json`, из-за которого пропадут все голоса.
    """
    store = ProjectsStore()
    project = store.create_project("Не трогает голоса", source_text=DIALOGUE)
    store.parse_project(project["id"])
    store.update_project(project["id"], speakers={"ИВАН": {"voice_id": "x"}})
    assert not (workspace / "voices" / "voices.json").exists()


def test_invalid_mode_is_rejected(workspace):
    store = ProjectsStore()
    try:
        store.create_project("Кривой", mode="что-то")
    except ValueError as exc:
        assert "режим" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("неизвестный режим должен быть ошибкой")


def test_schema_created_automatically(workspace):
    """База создаётся при первом обращении, без ручного SQL."""
    assert not config.DB_PATH.exists()
    ProjectsStore().list_projects()
    assert config.DB_PATH.exists()
    with sqlite3.connect(config.DB_PATH) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"projects", "speakers", "replicas", "takes", "schema_version"} <= tables
