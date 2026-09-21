"""Хранилище проектов записи: диск, дубли, роли и готовность (§35).

Здесь проверяется именно **персистентность и правила жизненного цикла**, а не API:
каждый тест читает состояние новым объектом `RecordingStore`, поэтому «сохранилось
в памяти процесса» не может выдать себя за «сохранилось на диск». Сырые дубли
пишутся синтетическими байтами: хранилищу всё равно, что внутри файла, — оно
отвечает за имя, место и неразрушимость прошлых дублей, а валидность звука
проверяет API (см. test_recording_api.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend import config
from backend.recording_store import RecordingStore, safe_component

DIALOGUE_REPLICAS = [
    {"index": 0, "speaker": "Анна", "text": "Первая реплика."},
    {"index": 1, "speaker": "Игорь", "text": "Вторая реплика."},
]


def _store() -> RecordingStore:
    """Новый объект хранилища поверх каталога записей из `workspace`.

    Новый объект — принципиально: синглтон в памяти пережил бы удаление каталога
    и скрыл бы ошибку сериализации, а состояние проекта обязано переживать
    перезапуск backend.
    """
    return RecordingStore(config.RECORDINGS_DIR)


def _project(store: RecordingStore, replicas: list[dict] | None = None):
    """Проект с разобранным диалогом по умолчанию — заготовка для тестов дублей."""
    project = store.create_project("Запись")
    store.set_replicas(project.id, replicas if replicas is not None else DIALOGUE_REPLICAS)
    return store.require_project(project.id)


def _add_take(store: RecordingStore, project_id: str, index: int, payload: bytes):
    return store.add_take(
        project_id,
        index,
        voice_profile_id="",
        raw_bytes=payload,
        suffix=".wav",
        duration_sec=0.5,
    )


# --- 1. Persistence ----------------------------------------------------------
def test_project_survives_new_store_object(workspace):
    """Состояние проекта живёт на диске, а не в памяти создавшего его объекта."""
    first = _store()
    project = first.create_project("Диктофон", "Анна: Привет.")

    second = _store()
    loaded = second.get_project(project.id)
    assert loaded is not None, "новый объект не увидел проект — состояние не на диске"
    assert (loaded.id, loaded.name, loaded.dialogue_text) == (
        project.id,
        "Диктофон",
        "Анна: Привет.",
    )
    # Индекс — второй файл состояния: без него список проектов пуст после перезапуска.
    assert [item["id"] for item in second.list_projects()] == [project.id]


# --- 2. Разбор диалога -------------------------------------------------------
def test_set_replicas_keeps_indexes_and_order_and_drops_vanished_takes(workspace):
    """Порядок и индексы реплик — из парсера; дубли исчезнувших реплик не остаются.

    Порядок задаёт монтаж (§4), поэтому пересортировка «по индексу» здесь была бы
    ошибкой: тест подаёт индексы вразнобой и требует ровно этот порядок.
    """
    store = _store()
    project = store.create_project("Порядок")
    store.set_replicas(
        project.id,
        [
            {"index": 5, "speaker": "Анна", "text": "Пятая"},
            {"index": 2, "speaker": "Игорь", "text": "Вторая"},
            {"index": 9, "speaker": "Анна", "text": "Девятая"},
        ],
    )
    loaded = store.require_project(project.id)
    assert [item["index"] for item in loaded.replicas] == [5, 2, 9]
    assert [item["text"] for item in loaded.replicas] == ["Пятая", "Вторая", "Девятая"]

    profile = store.add_voice_profile(project.id, "Голос")
    store.assign_role(project.id, "Анна", profile.id)
    store.assign_role(project.id, "Игорь", profile.id)
    _add_take(store, project.id, 5, b"take-5")
    _add_take(store, project.id, 2, b"take-2")

    # Новый текст: реплик 5, 2 и 9 больше нет — их дубли и назначения ролей
    # не должны «висеть» в проекте и портить готовность.
    refreshed = store.set_replicas(
        project.id,
        [
            {"index": 0, "speaker": "Пётр", "text": "Новая"},
            {"index": 1, "speaker": "Пётр", "text": "Ещё"},
        ],
    )
    assert [item["index"] for item in refreshed.replicas] == [0, 1]
    assert refreshed.takes == []
    assert refreshed.active_takes == {}
    assert refreshed.role_voices == {}


# --- 3. Добавление дубля -----------------------------------------------------
def test_add_take_writes_raw_file_and_never_overwrites_previous(workspace):
    """Прошлый дубль остаётся и на диске, и в списке: сравнивать всегда есть с чем.

    Перезапись «тем же именем» уничтожила бы предыдущую попытку, поэтому тест
    сверяет и содержимое обоих файлов, и оба элемента в `takes`.
    """
    store = _store()
    project = _project(store)

    first = _add_take(store, project.id, 0, b"first-attempt")
    second = _add_take(store, project.id, 0, b"second-attempt")

    assert first.id != second.id
    assert first.raw_file != second.raw_file, "второй дубль занял имя первого"
    raw = store.raw_dir(project.id)
    assert (raw / first.raw_file).read_bytes() == b"first-attempt"
    assert (raw / second.raw_file).read_bytes() == b"second-attempt"

    loaded = store.require_project(project.id)
    assert [take.id for take in loaded.takes_for(0)] == [first.id, second.id]
    # Активным становится именно новый дубль — пользователь только что записал его.
    assert loaded.active_take(0).id == second.id
    assert len(list(raw.iterdir())) == 2


# --- 4. Выбор дубля ----------------------------------------------------------
def test_select_take_switches_active_and_rejects_take_of_another_replica(workspace):
    """Активный дубль — решение пользователя; чужой дубль этой репликой не станет."""
    store = _store()
    project = _project(store)
    first = _add_take(store, project.id, 0, b"first")
    _add_take(store, project.id, 0, b"second")
    foreign = _add_take(store, project.id, 1, b"other-replica")

    selected = store.select_take(project.id, 0, first.id)
    assert selected.active_take(0).id == first.id

    # Дубль существует, но принадлежит другой реплике: подставить его нельзя,
    # иначе монтаж собрал бы чужой текст.
    with pytest.raises(KeyError):
        store.select_take(project.id, 0, foreign.id)

    # Дубль из другого проекта — тоже чужой: id уникален, но ищется только среди
    # дублей этого проекта, поэтому подставить запись другого диалога невозможно.
    other = _project(store)
    alien = _add_take(store, other.id, 0, b"another-project")
    with pytest.raises(KeyError):
        store.select_take(project.id, 0, alien.id)
    assert store.require_project(project.id).active_take(0).id == first.id


# --- 5. Удаление дубля -------------------------------------------------------
def test_delete_active_take_falls_back_and_last_delete_clears_replica(workspace):
    """Удалили активный — активным становится оставшийся; удалили последний — «не записано».

    Файл обязан уйти вместе с записью о дубле, а реплика без дублей — попасть в
    `missing_replicas`, иначе рендер собрал бы диалог без неё.
    """
    store = _store()
    project = _project(store)
    first = _add_take(store, project.id, 0, b"first")
    second = _add_take(store, project.id, 0, b"second")
    raw = store.raw_dir(project.id)

    # Активен второй (его записали последним) — удаляем именно активный.
    after_first_delete = store.delete_take(project.id, 0, second.id)
    assert after_first_delete.active_take(0).id == first.id
    assert not (raw / second.raw_file).exists(), "файл удалённого дубля остался на диске"

    after_last_delete = store.delete_take(project.id, 0, first.id)
    assert after_last_delete.active_take(0) is None
    assert after_last_delete.takes_for(0) == []
    assert 0 in after_last_delete.missing_replicas()
    assert not (raw / first.raw_file).exists()


# --- 6. Голоса и роли --------------------------------------------------------
def test_voice_profiles_and_role_assignment_persist(workspace):
    """Настройки обработки и назначение роли переживают перезапуск и не берут чужих id."""
    store = _store()
    project = _project(store)
    profile = store.add_voice_profile(
        project.id, "Низкий", speed=0.9, pitch_semitones=-2.0, denoise=True
    )
    store.assign_role(project.id, "Анна", profile.id)
    store.update_voice_profile(
        project.id, profile.id, name="Высокий", speed=1.2, pitch_semitones=3.0, denoise=False
    )

    loaded = _store().require_project(project.id)
    saved = loaded.profile(profile.id)
    assert saved is not None
    assert (saved.name, saved.speed, saved.pitch_semitones, saved.denoise) == (
        "Высокий",
        1.2,
        3.0,
        False,
    )
    assert loaded.role_voices == {"Анна": profile.id}

    # Несуществующий голос нельзя ни назначить, ни отредактировать: иначе роль
    # осталась бы «назначенной» в никуда, а UI показывал бы пустое имя.
    with pytest.raises(KeyError):
        _store().assign_role(project.id, "Анна", "нет-такого-голоса")
    with pytest.raises(KeyError):
        _store().update_voice_profile(project.id, "нет-такого-голоса", speed=1.1)


# --- 7. Безопасность путей ---------------------------------------------------
def test_safe_component_and_project_dir_never_escape_recordings_dir(workspace):
    """Имя проекта не должно уметь вывести каталог за пределы `recordings/`.

    Проект специально называется `../../etc/passwd`: если бы имя попадало в путь
    как есть, запись создала бы каталоги в чужом месте, а `..` в URL позволил бы
    читать файлы вне хранилища.
    """
    cleaned = safe_component("../../etc/passwd")
    assert "/" not in cleaned and "\\" not in cleaned
    assert ".." not in cleaned
    # Результат — ровно один компонент пути, а не относительный маршрут.
    assert Path(cleaned).parts == (cleaned,)
    # Обратные слэши Windows — тот же разделитель: они тоже не должны пройти.
    windows = safe_component("..\\..\\windows\\system32")
    assert "/" not in windows and "\\" not in windows and ".." not in windows
    assert Path(windows).parts == (windows,)

    store = _store()
    project = store.create_project("../../etc/passwd")
    root = Path(config.RECORDINGS_DIR).resolve()
    for directory in (
        store.project_dir(project.id),
        store.raw_dir(project.id),
        store.processed_dir(project.id),
        store.preview_dir(project.id),
        store.output_dir(project.id),
        # Идентификатор из URL проверяется тем же помощником.
        store.project_dir("../../etc/passwd"),
    ):
        assert directory.resolve().is_relative_to(root), f"{directory} вне каталога записей"
    assert store.raw_dir(project.id).is_dir()


# --- 8. Готовность -----------------------------------------------------------
def test_readiness_counts_replicas_roles_and_missing(workspace):
    """Готовность считается по активным дублям, а роль готова только целиком.

    Диалог: Анна (0, 2) и Игорь (1, 3). После записи реплик 0 и 2 роль Анны
    закрыта, а Игорь — нет: рендер соберёт диалог только когда записаны все.
    """
    store = _store()
    project = _project(
        store,
        [
            {"index": 0, "speaker": "Анна", "text": "Раз"},
            {"index": 1, "speaker": "Игорь", "text": "Два"},
            {"index": 2, "speaker": "Анна", "text": "Три"},
            {"index": 3, "speaker": "Игорь", "text": "Четыре"},
        ],
    )
    _add_take(store, project.id, 0, b"anna-1")
    _add_take(store, project.id, 2, b"anna-2")
    readiness = store.require_project(project.id).readiness()
    assert readiness["replicas_total"] == 4
    assert readiness["replicas_recorded"] == 2
    assert readiness["roles_total"] == 2
    assert readiness["roles_ready"] == 1
    assert readiness["missing"] == [1, 3]
    assert readiness["progress"] == 0.5
    assert readiness["ready_to_render"] is False

    _add_take(store, project.id, 1, b"igor-1")
    partly = store.require_project(project.id).readiness()
    assert partly["missing"] == [3]
    assert partly["roles_ready"] == 1, "роль без последней реплики не может считаться готовой"

    _add_take(store, project.id, 3, b"igor-2")
    full = store.require_project(project.id).readiness()
    assert full["missing"] == []
    assert full["roles_ready"] == 2
    assert full["progress"] == 1.0
    assert full["ready_to_render"] is True
