"""Фасад над таблицами: транзакции, каскады и файлы вариантов в одном месте.

API работает только через этот класс: держать границы транзакций в обработчиках
запросов означало бы, что «назначить голос» и «обновить реплики этого спикера» —
две независимые записи, и половина из них может не примениться.
"""

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..dialogue_parser import parse_dialogue, voice_label
from .connection import transaction
from .repositories.crashes import WorkerCrashesRepository
from .repositories.projects import ProjectsRepository
from .repositories.replicas import ReplicasRepository
from .repositories.takes import TakesRepository


def _analysis_status_from_rows(rows: list[dict]) -> str:
    """Состояние проекта по фактическому состоянию его реплик.

    Считается по строкам, а не берётся из итога последнего прогона: анализ бывает
    точечным (пересчитали одну реплику), и «ready» от одного удачного куска при
    десятке неподготовленных было бы неправдой. Кандидаты важнее: если у реплики
    есть неразрешённые слова, состояние `needs_review` — даже когда сама она уже
    подготовлена.
    """
    if not rows:
        return config.PROJECT_ANALYSIS_RAW
    if any(row["pronunciation_candidates"] for row in rows):
        return config.PROJECT_ANALYSIS_NEEDS_REVIEW
    if all(row["analysis_status"] == config.REPLICA_ANALYSIS_ERROR for row in rows):
        return config.PROJECT_ANALYSIS_ERROR
    if all(row["analysis_status"] == config.REPLICA_ANALYSIS_DONE for row in rows):
        return config.PROJECT_ANALYSIS_READY
    return config.PROJECT_ANALYSIS_RAW


def _now_iso() -> str:
    """Метка времени анализа в том же формате, что у остальных записей базы."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

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
            current = projects.get(project_id)
            if current is None:
                raise KeyError(project_id)
            replicas = ReplicasRepository(connection)
            # Причина, по которой подготовка перестала быть актуальной. Список
            # реплик (`None` — все) и текст причины идут вместе: интерфейс обязан
            # показать не только «кнопка недоступна», но и почему.
            stale: list[int] | None = None
            reason: str | None = None
            if speakers is not None:
                projects.replace_speakers(project_id, speakers)
                changed = replicas.sync_voices(project_id, speakers)
                if changed:
                    stale = changed
                    reason = "изменился голос спикера: подготовка зависит от движка"
            if "source_text" in fields and fields["source_text"] != current["source_text"]:
                stale = None  # текст переписан целиком — устарели все реплики
                reason = "исходный текст изменён"
            if "mode" in fields and fields["mode"] != current["mode"]:
                stale = None
                reason = "изменён режим проекта"
            if fields:
                projects.update(project_id, **fields)
            else:
                projects.update(project_id)
            if reason is not None:
                replicas.mark_analysis_pending(project_id, stale)
                projects.update(
                    project_id,
                    analysis_status=config.PROJECT_ANALYSIS_RAW,
                    analysis_error=reason,
                )
                logger.info("Проект %s: подготовка устарела (%s)", project_id, reason)
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
            # Новый разбор — новая подготовка: старые стадии относились к прежним
            # репликам, и «готово» после правки одной строки было бы неправдой.
            replicas = ReplicasRepository(connection)
            replicas.mark_analysis_pending(project_id)
            projects.update(
                project_id,
                analysis_status=config.PROJECT_ANALYSIS_RAW,
                analysis_error="текст разобран заново — нужен анализ",
            )
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
        text: object = UNSET,
    ) -> dict:
        """Правит одну реплику: текст, голос, отдельные параметры или всё сразу.

        Один вызов на все правки: карточка меняет то текст, то голос, то ползунок,
        и отдельные ручки означали бы, что интерфейс помнит, чем что менять.

        Инвалидация подготовки — точечная и по делу: правка текста и смена голоса
        меняют то, что уйдёт в модель, а ползунок громкости — нет. Поэтому
        `overrides` не сбрасывают анализ: иначе каждое движение слайдера требовало
        бы прогонять подготовку заново.
        """
        with transaction() as connection:
            replicas = ReplicasRepository(connection)
            replica = replicas.by_index(project_id, index)
            if replica is None:
                raise KeyError(index)
            replica_id = int(replica["id"])
            reason: str | None = None
            if text is not UNSET and str(text) != replica["text"]:
                replicas.set_text(replica_id, str(text))
                reason = "изменён текст реплики"
            if voice_id is not UNSET:
                replicas.set_voice(replica_id, voice_id or None)  # type: ignore[arg-type]
                reason = reason or "изменён голос реплики"
            if reset_overrides:
                replicas.set_overrides(replica_id, {})
            elif overrides:
                replicas.set_overrides(
                    replica_id, merge_overrides(replica["overrides"], overrides)
                )
            if reason is not None:
                replicas.mark_analysis_pending(project_id, [index])
                ProjectsRepository(connection).update(
                    project_id,
                    analysis_status=config.PROJECT_ANALYSIS_RAW,
                    analysis_error=f"реплика {index + 1}: {reason}",
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

    def set_replica_status(self, project_id: str, index: int, status: str) -> bool:
        """Статус одной реплики: `rendering` перед синтезом, `interrupted` при срыве.

        Без обновления проекта: у рендера и у анализа свои статусы, и трогать
        проект из очереди по поводу одной реплики значило бы гасить более общий
        сигнал («идёт рендер») ради частного. `False` — реплики с таким индексом
        нет (разовая задача без проекта).
        """
        if status not in config.REPLICA_STATUSES:
            raise ValueError(f"Неизвестный статус реплики: {status}")
        with transaction() as connection:
            replica = ReplicasRepository(connection).by_index(project_id, index)
            if replica is None:
                return False
            return ReplicasRepository(connection).set_status(int(replica["id"]), status)

    # -- диагностика падений воркера (creash_report) ---------------------------
    def record_worker_crash(
        self,
        *,
        job_id: str = "",
        project_id: str | None = None,
        replica_index: int | None = None,
        **fields,
    ) -> dict:
        """Записывает падение процесса синтеза. Возвращает сохранённую запись.

        `replica_id` разрешается здесь, а не вызывающим: очередь знает индекс
        реплики, а идентификатор строки — деталь схемы, и тащить её в очередь
        незачем. Ошибка записи не должна губить задачу: диагностика вторична по
        отношению к уже готовому аудио (см. `_mark_project` в очереди).
        """
        with transaction() as connection:
            replica_id = None
            if project_id is not None and replica_index is not None:
                replica = ReplicasRepository(connection).by_index(project_id, int(replica_index))
                if replica is not None:
                    replica_id = int(replica["id"])
            return WorkerCrashesRepository(connection).add(
                job_id=job_id,
                project_id=project_id,
                replica_id=replica_id,
                replica_index=None if replica_index is None else int(replica_index),
                **fields,
            )

    def list_worker_crashes(self, limit: int = 20, project_id: str | None = None) -> list[dict]:
        """Последние падения — от свежих к старым (для диагностики и интерфейса)."""
        with transaction() as connection:
            return WorkerCrashesRepository(connection).list_recent(limit, project_id)

    # -- подготовка текста (анализ) --------------------------------------------
    def save_analysis(self, project_id: str, summary) -> dict:
        """Сохраняет результаты анализа проекта одной транзакцией.

        Одной — потому что состояние проекта и стадии реплик обязаны совпадать:
        половина записанного анализа означала бы либо «ready» без подготовленного
        текста, либо подготовленный текст, о котором проект ещё не знает.
        Версия анализа увеличивается здесь же, поэтому стадии всегда помечены той
        версией, в которой они получены.
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            project = projects.get(project_id)
            if project is None:
                raise KeyError(project_id)
            version = int(project["analysis_version"]) + 1
            replicas = ReplicasRepository(connection)
            rows = {int(item["index"]): item for item in replicas.list_for_project(project_id)}
            saved = 0
            for preparation in summary.replicas:
                row = rows.get(preparation.index)
                if row is None:  # реплику убрали, пока шёл анализ
                    continue
                replicas.save_analysis(int(row["id"]), {**preparation.to_dict(), "analysis_version": version})
                saved += 1
            # Состояние — по факту записанного, а не по итогу этого прогона:
            # точечная подготовка одной реплики не делает проект готовым.
            status = _analysis_status_from_rows(replicas.list_for_project(project_id))
            projects.update(
                project_id,
                analysis_status=status,
                analysis_version=version,
                analysis_error=summary.error,
                analysis_finished_at=_now_iso(),
            )
        logger.info(
            "Проект %s: анализ версии %s — %s из %s реплик, состояние %s",
            project_id, version, saved, len(summary.replicas), status,
        )
        return self.get_project(project_id)  # type: ignore[return-value]

    def mark_analysis_started(self, project_id: str) -> dict:
        """Отмечает начало анализа: интерфейс видит «анализируется», а не «сырой»."""
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            if projects.get(project_id) is None:
                raise KeyError(project_id)
            projects.update(
                project_id,
                analysis_status=config.PROJECT_ANALYSIS_ANALYZING,
                analysis_started_at=_now_iso(),
                analysis_error=None,
            )
        return self.get_project(project_id)  # type: ignore[return-value]

    def invalidate_analysis(
        self, project_id: str, *, indexes: list[int] | None = None, reason: str
    ) -> dict:
        """Помечает подготовку устаревшей, не стирая её.

        Стадии остаются: пользователю полезно видеть, что было подготовлено в
        прошлый раз, а `analysis_status = pending` уже говорит, что брать этот
        `final_text` нельзя. `indexes=None` — устарели все реплики (например,
        изменился словарь произношения).
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            if projects.get(project_id) is None:
                raise KeyError(project_id)
            replicas = ReplicasRepository(connection)
            replicas.mark_analysis_pending(project_id, indexes)
            projects.update(
                project_id,
                analysis_status=config.PROJECT_ANALYSIS_RAW,
                analysis_error=reason,
            )
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

    # -- импорт проекта --------------------------------------------------------
    def import_project_content(
        self,
        project_id: str,
        speakers: dict[str, dict],
        replicas: list[dict],
        takes: dict[int, list[dict]],
        status: str | None = None,
    ) -> dict:
        """Наполняет пустой проект содержимым архива за одну транзакцию.

        Импорт — это не «разобрать текст»: у реплик уже есть тексты, голоса,
        статусы и готовые take'ы, а порядок take'ов и активный из них несут смысл.
        Разбор (`parse_project`) пересоздал бы их из исходного текста и потерял бы
        всё это, поэтому строки пишутся напрямую, а выбор take'а восстанавливается
        после вставки — по id, которого до неё ещё нет.

        Одна транзакция на весь проект: частично восстановленный диалог
        (спикеры есть, take'ов нет) выглядел бы рабочим, но звучал бы иначе.
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            if projects.get(project_id) is None:
                raise KeyError(project_id)
            projects.replace_speakers(project_id, speakers)
            replicas_repo = ReplicasRepository(connection)
            replicas_repo.replace(
                project_id,
                [
                    {
                        "index": int(item["index"]),
                        "text": str(item.get("text") or ""),
                        "speaker": str(item.get("speaker") or ""),
                        "voice_id": str(item.get("voice_id") or ""),
                        "overrides": item.get("overrides") or {},
                    }
                    for item in replicas
                ],
            )
            by_index = {
                int(item["index"]): item for item in replicas_repo.list_for_project(project_id)
            }
            takes_repo = TakesRepository(connection)
            for item in replicas:
                index = int(item["index"])
                replica = by_index.get(index)
                if replica is None:
                    continue
                replica_id = int(replica["id"])
                override = item.get("voice_override")
                if override:
                    replicas_repo.set_voice(replica_id, str(override))
                selected = False
                for take in takes.get(index, []):
                    saved = takes_repo.add(
                        replica_id=replica_id,
                        audio_path=str(take["audio_path"]),
                        label=str(take.get("label") or ""),
                        seed=take.get("seed"),
                        engine=str(take.get("engine") or ""),
                        parameters=take.get("parameters") or {},
                        duration_sec=float(take.get("duration_sec") or 0.0),
                        qa=take.get("qa"),
                        quality=take.get("quality"),
                    )
                    if take.get("selected") and not selected:
                        replicas_repo.select_take(replica_id, int(saved["id"]))
                        selected = True
                # Статус восстанавливается только у реплики без активного take:
                # выбор take'а сам ставит `rendered`, и переписывать его нечем.
                if not selected:
                    replicas_repo.set_status(
                        replica_id, str(item.get("status") or config.REPLICA_STATUS_PENDING)
                    )
            if status is None:
                projects.update(project_id)
            else:
                projects.update(project_id, status=status)
        logger.info(
            "Проект %s: импортировано %s спикеров и %s реплик",
            project_id, len(speakers), len(replicas),
        )
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

    def save_partial_takes(self, project_id: str, job_id: str, takes: list[dict]) -> None:
        """Сохраняет куски **прерванного** рендера вариантами реплик.

        Отличие от `save_render_takes` ровно одно и принципиальное: проект не
        помечается `rendered`. Рендер не завершился, и «готово» здесь было бы
        ложью, из-за которой следующий запуск не отличил бы целый проект от
        наполовину собранного. Готовые реплики при этом становятся текущими
        вариантами: работа не пропадает, её можно послушать и продолжить.
        """
        with transaction() as connection:
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

    def set_project_status(
        self, project_id: str, status: str, job_id: str | None = None, error: str | None = None
    ) -> None:
        with transaction() as connection:
            ProjectsRepository(connection).update(
                project_id, status=status, job_id=job_id, last_error=error
            )

    # -- восстановление после перезапуска (creash_report) ----------------------
    def interrupt_stale_state(self) -> dict:
        """Переводит состояния «идёт» в прерванные: очередь в памяти, её больше нет.

        После перезапуска приложения задача не продолжается, а её статусы в базе
        остались «rendering»/«analyzing». Оставлять их как есть — значит
        показывать пользователю вечный синтез, запрещать повторный рендер
        (проект выглядит занятым) и прятать, что реплика не готова. Поэтому:

        * проект со статусом `rendering` → `draft` с причиной в `last_error`;
        * реплика со статусом `rendering` → `interrupted`;
        * анализ со статусом `analyzing` → `raw` с причиной в `analysis_error`.

        Возвращает то, что нужно для диагностики и лога: счётчики и список
        восстановленных проектов (id, имя, последняя задача).
        """
        with transaction() as connection:
            projects = ProjectsRepository(connection)
            replicas = ReplicasRepository(connection)
            stale_projects = projects.list_by_status(config.PROJECT_STATUS_RENDERING)
            stale_analyses = projects.list_by_status(
                config.PROJECT_ANALYSIS_ANALYZING, column="analysis_status"
            )
            stale_replicas = replicas.list_by_status(config.REPLICA_STATUS_RENDERING)
            for replica in stale_replicas:
                replicas.set_status(int(replica["id"]), config.REPLICA_STATUS_INTERRUPTED)
            for project in stale_projects:
                projects.update(
                    project["id"],
                    status=config.PROJECT_STATUS_DRAFT,
                    last_error=config.RECOVERY_RENDER_MESSAGE,
                )
            for project in stale_analyses:
                projects.update(
                    project["id"],
                    analysis_status=config.PROJECT_ANALYSIS_RAW,
                    analysis_error=config.RECOVERY_ANALYSIS_MESSAGE,
                )
            return {
                "projects": [
                    {"id": item["id"], "name": item["name"], "job_id": item["job_id"]}
                    for item in stale_projects
                ],
                "replicas": len(stale_replicas),
                "replica_indexes": [
                    {"project_id": item["project_id"], "index": item["index"]}
                    for item in stale_replicas
                ],
                "analyses": len(stale_analyses),
            }

    def project_dir(self, project_id: str) -> Path:
        """Каталог кусков проекта: удаляется вместе с проектом."""
        return config.PROJECTS_OUTPUT_DIR / project_id


_store = ProjectsStore()


def get_projects_store() -> ProjectsStore:
    return _store
