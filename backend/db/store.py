"""Фасад над таблицами: транзакции, каскады и файлы вариантов в одном месте.

API работает только через этот класс: держать границы транзакций в обработчиках
запросов означало бы, что «назначить голос» и «обновить реплики этого спикера» —
две независимые записи, и половина из них может не примениться.
"""

import logging
import shutil
import uuid
from pathlib import Path

from .. import config
from ..dialogue_parser import parse_dialogue, voice_label
from .connection import transaction
from .repositories.projects import ProjectsRepository
from .repositories.replicas import ReplicasRepository
from .repositories.takes import TakesRepository

logger = logging.getLogger(__name__)

# Значение не передано вовсе — в отличие от `None`, которое означает «вернуть
# наследуемое»: у правки реплики это разные вещи, и различать их должен вызывающий.
UNSET = object()

# Поля, которые можно править у одной реплики поверх карточки спикера. Список
# закрытый: всё остальное в правках — либо чужая ручка, либо опечатка, и молча
# принимать такое значит копить в базе значения, которые ни на что не влияют.
REPLICA_OVERRIDE_FIELDS = (
    "speed",
    "cfg_strength",
    "nfe_step",
    "target_rms",
    "gain_db",
    "pitch_semitones",
    "pause_override_ms",
)


def merge_overrides(current: dict, patch: dict) -> dict:
    """Правки реплики поверх текущих: `None` возвращает параметр к наследуемому.

    Склейка, а не замена: интерфейс присылает по одному изменённому полю за раз,
    и полная замена стирала бы остальные правки карточки. Ручки движка вложены, и
    внутри `engine_params` сброс работает по ключам.
    """
    result = dict(current)
    for name, value in patch.items():
        if name == "engine_params" and isinstance(value, dict):
            params = dict(result.get("engine_params") or {})
            for param, param_value in value.items():
                if param_value is None:
                    params.pop(param, None)
                else:
                    params[param] = param_value
            if params:
                result["engine_params"] = params
            else:
                result.pop("engine_params", None)
            continue
        if value is None:
            result.pop(name, None)
        elif name in REPLICA_OVERRIDE_FIELDS:
            result[name] = value
    return result


class ProjectsStore:
    # -- проекты ---------------------------------------------------------------
    def create_project(
        self,
        name: str,
        source_text: str = "",
        mode: str = config.PROJECT_MODE_DIALOGUE,
        render_settings: dict | None = None,
    ) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Не задано имя проекта")
        if mode not in config.PROJECT_MODES:
            raise ValueError(f"Неизвестный режим проекта: {mode}")
        with transaction() as connection:
            project = ProjectsRepository(connection).create(
                name=name, source_text=source_text, mode=mode, render_settings=render_settings
            )
        logger.info("Проект %s создан: «%s»", project["id"], project["name"])
        return self.get_project(project["id"])  # type: ignore[return-value]

    def list_projects(self) -> list[dict]:
        with transaction() as connection:
            return ProjectsRepository(connection).list_projects()

    def get_project(self, project_id: str) -> dict | None:
        """Проект целиком: спикеры, реплики и варианты каждой реплики."""
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            replicas_repo = ReplicasRepository(connection)
            takes_repo = TakesRepository(connection)
            project = projects.get(project_id)
            if project is None:
                return None
            replicas = replicas_repo.list_for_project(project_id)
            takes: dict[int, list[dict]] = {}
            for take in takes_repo.list_for_project(project_id):
                takes.setdefault(int(take["replica_id"]), []).append(take)
            for replica in replicas:
                replica["takes"] = takes.get(int(replica["id"]), [])
                replica["label"] = voice_label(replica["speaker"])
            project["speakers"] = projects.speakers(project_id)
            project["replicas"] = replicas
            return project

    def update_project(self, project_id: str, **fields) -> dict:
        """Меняет проект. Спикеры обновляются тем же вызовом, что и настройки.

        Назначение голоса идёт через один PATCH: раздельные ручки означали бы, что
        интерфейс должен помнить, каким запросом что меняется, а падение между
        двумя запросами оставило бы проект в половинчатом состоянии.
        """
        speakers = fields.pop("speakers", None)
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            if projects.get(project_id) is None:
                raise KeyError(project_id)
            if speakers is not None:
                projects.replace_speakers(project_id, speakers)
                ReplicasRepository(connection).sync_voices(project_id, speakers)
            if fields:
                projects.update(project_id, **fields)
            else:
                projects.update(project_id)
        return self.get_project(project_id)  # type: ignore[return-value]

    def delete_project(self, project_id: str) -> bool:
        """Удаляет проект и файлы его вариантов.

        Файлы — до записи в базу: если удаление файла не удалось, проект лучше
        оставить целым, чем получить записи, ссылающиеся в никуда.
        """
        with transaction() as connection:
            takes_repo = TakesRepository(connection)
            paths = takes_repo.paths(project_id)
            deleted = ProjectsRepository(connection).delete(project_id)
        if not deleted:
            return False
        for raw in paths:
            Path(raw).unlink(missing_ok=True)
        shutil.rmtree(self.project_dir(project_id), ignore_errors=True)
        logger.info("Проект %s удалён", project_id)
        return True

    # -- разбор текста ---------------------------------------------------------
    def parse_project(self, project_id: str, chunk_strategy: str | None = None) -> dict:
        """Разбирает исходный текст проекта в реплики и сохраняет их.

        Голоса, уже назначенные спикерам, сохраняются: правка текста не должна
        означать «выбери голоса заново».
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            project = projects.get(project_id)
            if project is None:
                raise KeyError(project_id)
            strategy = chunk_strategy or project["render_settings"].get(
                "chunk_strategy", config.CHUNK_STRATEGY_DEFAULT
            )
            try:
                parsed = parse_dialogue(project["source_text"], config.chunk_chars(strategy))
            except ValueError as exc:
                raise ValueError(str(exc)) from exc

            speakers = {
                voice.key: {"label": voice.label, "voice_id": ""} for voice in parsed.voices
            }
            projects.replace_speakers(project_id, speakers)

            assigned = {
                item["key"]: item["voice_id"] for item in projects.speakers(project_id)
            }
            ReplicasRepository(connection).replace(
                project_id,
                [
                    {
                        "index": index,
                        "text": replica.text,
                        "speaker": replica.voice,
                        "voice_id": assigned.get(replica.voice, ""),
                        "overrides": replica.overrides,
                    }
                    for index, replica in enumerate(parsed.replicas)
                ],
            )
            render_settings = dict(project["render_settings"])
            render_settings["chunk_strategy"] = strategy
            projects.update(project_id, render_settings=render_settings)
        logger.info("Проект %s: разобрано %s реплик", project_id, len(parsed.replicas))
        return self.get_project(project_id)  # type: ignore[return-value]

    # -- реплики ---------------------------------------------------------------
    def patch_replica(
        self,
        project_id: str,
        index: int,
        voice_id: object = UNSET,
        overrides: dict | None = None,
        reset_overrides: bool = False,
    ) -> dict:
        """Правит одну реплику: голос, отдельные параметры или всё сразу.

        Один вызов на все правки: карточка меняет то голос, то ползунок, и
        отдельные ручки означали бы, что интерфейс помнит, чем что менять.
        """
        with transaction() as connection:
            replicas = ReplicasRepository(connection)
            replica = replicas.by_index(project_id, index)
            if replica is None:
                raise KeyError(index)
            replica_id = int(replica["id"])
            if voice_id is not UNSET:
                replicas.set_voice(replica_id, voice_id or None)  # type: ignore[arg-type]
            if reset_overrides:
                replicas.set_overrides(replica_id, {})
            elif overrides:
                replicas.set_overrides(
                    replica_id, merge_overrides(replica["overrides"], overrides)
                )
        return self.get_project(project_id)  # type: ignore[return-value]

    def set_replica_voice(self, project_id: str, index: int, voice_id: str | None) -> dict:
        """Меняет голос одной реплики — остальные остаются как были."""
        return self.patch_replica(project_id, index, voice_id=voice_id)

    def set_replica_overrides(self, project_id: str, index: int, overrides: dict) -> dict:
        with transaction() as connection:
            replicas = ReplicasRepository(connection)
            replica = replicas.by_index(project_id, index)
            if replica is None:
                raise KeyError(index)
            replicas.set_overrides(int(replica["id"]), overrides)
        return self.get_project(project_id)  # type: ignore[return-value]

    def get_take(self, project_id: str, index: int, take_id: int) -> dict | None:
        """Вариант реплики, если он действительно ей принадлежит."""
        with transaction() as connection:
            replica = ReplicasRepository(connection).by_index(project_id, index)
            if replica is None:
                return None
            take = TakesRepository(connection).get(take_id)
            if take is None or int(take["replica_id"]) != int(replica["id"]):
                return None
            return take

    def save_replica_take(self, project_id: str, index: int, take: dict) -> None:
        """Добавляет реплике новый вариант и делает его активным.

        Файл копируется в каталог проекта, а не берётся по ссылке: вариант задачи
        вытесняется лимитом и чистится вместе с `output/`, а вариант проекта должен
        переживать и то, и другое.
        """
        source = Path(str(take["audio_path"]))
        target = self.project_dir(project_id) / f"r{index + 1}-{uuid.uuid4().hex[:8]}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        try:
            with transaction() as connection:
                replicas = ReplicasRepository(connection)
                replica = replicas.by_index(project_id, index)
                if replica is None:
                    raise KeyError(index)
                saved = TakesRepository(connection).add(
                    replica_id=int(replica["id"]),
                    audio_path=str(target),
                    label=str(take.get("label", "")),
                    seed=take.get("seed"),
                    engine=str(take.get("engine", "")),
                    parameters=take.get("parameters") or {},
                    duration_sec=float(take.get("duration_sec") or 0.0),
                    qa=take.get("qa"),
                    quality=take.get("quality"),
                )
                replicas.select_take(int(replica["id"]), int(saved["id"]))
                ProjectsRepository(connection).update(project_id)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def select_take(self, project_id: str, index: int, take_id: int) -> dict:
        """Ставит выбранный вариант активным для реплики."""
        with transaction() as connection:
            replicas = ReplicasRepository(connection)
            takes = TakesRepository(connection)
            replica = replicas.by_index(project_id, index)
            if replica is None:
                raise KeyError(index)
            take = takes.get(take_id)
            if take is None or int(take["replica_id"]) != int(replica["id"]):
                raise KeyError(take_id)
            replicas.select_take(int(replica["id"]), take_id)
        return self.get_project(project_id)  # type: ignore[return-value]

    # -- результат рендера -----------------------------------------------------
    def save_render_takes(self, project_id: str, job_id: str, takes: list[dict]) -> None:
        """Сохраняет куски готового рендера вариантами реплик.

        Каждый вызов добавляет новый набор вариантов, а не перезаписывает прошлый:
        прошлый рендер остаётся доступен для прослушивания и сравнения.
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            replicas = ReplicasRepository(connection)
            takes_repo = TakesRepository(connection)
            by_index = {int(item["index"]): item for item in replicas.list_for_project(project_id)}
            for item in takes:
                replica = by_index.get(int(item["index"]))
                if replica is None:
                    continue
                take = takes_repo.add(
                    replica_id=int(replica["id"]),
                    audio_path=str(item["audio_path"]),
                    label=str(item.get("label", "")),
                    seed=item.get("seed"),
                    engine=str(item.get("engine", "")),
                    parameters=item.get("parameters") or {},
                    duration_sec=float(item.get("duration_sec") or 0.0),
                    qa=item.get("qa"),
                    quality=item.get("quality"),
                )
                replicas.select_take(int(replica["id"]), int(take["id"]))
            projects.update(
                project_id,
                status=config.PROJECT_STATUS_RENDERED,
                job_id=job_id,
                last_error=None,
            )

    def set_project_status(
        self, project_id: str, status: str, job_id: str | None = None, error: str | None = None
    ) -> None:
        with transaction() as connection:
            ProjectsRepository(connection).update(
                project_id, status=status, job_id=job_id, last_error=error
            )

    def project_dir(self, project_id: str) -> Path:
        """Каталог кусков проекта: удаляется вместе с проектом."""
        return config.PROJECTS_OUTPUT_DIR / project_id


_store = ProjectsStore()


def get_projects_store() -> ProjectsStore:
    return _store
