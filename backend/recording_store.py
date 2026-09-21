"""Хранилище проектов записи («Сам себе звукорежиссер»).

Отдельная сущность, а не TTS-проект: у записи другой жизненный цикл (сырые дубли,
неразрушающая обработка, монтаж) и другой источник звука — микрофон, а не модель.
Смешивать её с `voices/` нельзя: там референсы TTS-голосов, и сведение двух
разных по смыслу наборов файлов в один каталог однажды привело бы к тому, что
сырая запись попала бы в модель как эталон голоса.

Раскладка (§20 пакета):

```text
recordings/
  projects.json                 индекс: id → имя, статусы, время
  {project_id}/
    project.json                полное состояние проекта
    raw/                        сырые дубли (не изменяются никогда)
    processed/                  кэш обработанных версий
    previews/                   временные превью обработки
    output/                     готовый мастер и MP3
```

Запись атомарна (`tmp` → `os.replace`): состояние переживает перезапуск, и
половина файла не должна выглядеть как целый проект. Все имена файлов
генерируются здесь: пользовательское имя роли в путь не попадает никогда (§23).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

INDEX_NAME = "projects.json"
PROJECT_NAME = "project.json"
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
PREVIEW_DIR = "previews"
OUTPUT_DIR = "output"

PROJECT_STATUS_DRAFT = "draft"
PROJECT_STATUS_RENDERED = "rendered"
PROJECT_STATUS_ERROR = "error"

# Роли и реплики приходят из общего разбора диалога; здесь только их хранение.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: dict) -> None:
    """Пишет JSON атомарно: временный файл рядом, затем `os.replace`.

    Частично записанный `project.json` выглядит как валидный проект с половиной
    дублей — и «продолжить с того же места» стало бы невозможным. `os.replace`
    в пределах одного каталога атомарен, поэтому читатель всегда видит либо
    прежнее состояние, либо новое.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def safe_component(value: str) -> str:
    """Безопасное имя файла/идентификатора: без разделителей пути и спецсимволов.

    Пользовательские имена (роль, профиль голоса) в пути не участвуют — этот
    помощник нужен для идентификаторов, но даже их проверяем: идентификатор
    приходит из URL.
    """
    cleaned = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in str(value or "")
    ).strip("_")
    return cleaned[:64]


@dataclass
class RecordedVoiceProfile:
    """Настройки обработки одного записываемого голоса (§6, §11).

    Отдельная сущность от TTS-голоса: здесь нет движка, референса и `ref_text` —
    только имя и неразрушающие настройки обработки записанного звука.
    """

    id: str
    name: str
    speed: float = 1.0
    pitch_semitones: float = 0.0
    denoise: bool = False
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> RecordedVoiceProfile:
        return cls(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            name=str(raw.get("name") or "Голос"),
            speed=float(raw.get("speed", 1.0) or 1.0),
            pitch_semitones=float(raw.get("pitch_semitones", 0.0) or 0.0),
            denoise=bool(raw.get("denoise", False)),
            created_at=str(raw.get("created_at") or _now()),
        )


@dataclass
class RecordedTake:
    """Один записанный дубль реплики (§10).

    `raw_audio_path` — только имя файла внутри `raw/`: абсолютный путь в
    состоянии проекта сделал бы его непереносимым (и утёк бы в UI/экспорт).
    Обработанная версия здесь не хранится: она производная от raw и настроек
    голоса и пересчитывается заново (§16).
    """

    id: str
    replica_index: int
    voice_profile_id: str
    raw_file: str
    duration_sec: float = 0.0
    created_at: str = field(default_factory=_now)
    # Отметки лёгкого анализа записи (warning, не blocker): тихо/перегруз.
    level_warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> RecordedTake:
        return cls(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            replica_index=int(raw.get("replica_index") or 0),
            voice_profile_id=str(raw.get("voice_profile_id") or ""),
            raw_file=str(raw.get("raw_file") or ""),
            duration_sec=float(raw.get("duration_sec") or 0.0),
            created_at=str(raw.get("created_at") or _now()),
            level_warning=str(raw.get("level_warning") or ""),
        )


@dataclass
class RecordingProject:
    """Проект записи: диалог, роли, дубли и настройки (§19).

    `replicas` — результат **существующего** разбора диалога: индекс, спикер,
    текст. Собственного парсера у режима нет и быть не должно.
    """

    id: str
    name: str
    dialogue_text: str = ""
    status: str = PROJECT_STATUS_DRAFT
    # Список реплик в исходном порядке: `{"index", "speaker", "text"}`.
    replicas: list = field(default_factory=list)
    # Назначение роли на записываемый голос: `{"speaker": profile_id}`.
    role_voices: dict = field(default_factory=dict)
    voice_profiles: list = field(default_factory=list)
    takes: list = field(default_factory=list)
    # Активный дубль реплики: индекс → take id. Отдельно от `takes`, потому что
    # это решение пользователя, а не свойство файла.
    active_takes: dict = field(default_factory=dict)
    # Порядок отображения при записи: `dialogue` или `roles`. На порядок монтажа
    # не влияет никогда (§4).
    recording_order: str = "dialogue"
    render_settings: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_error: str = ""

    # -- производные -----------------------------------------------------------
    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for replica in self.replicas:
            speaker = str(replica.get("speaker") or "")
            if speaker and speaker not in seen:
                seen.append(speaker)
        return seen

    def profile(self, profile_id: str) -> RecordedVoiceProfile | None:
        for item in self.voice_profiles:
            if item.id == profile_id:
                return item
        return None

    def takes_for(self, index: int) -> list[RecordedTake]:
        return [take for take in self.takes if take.replica_index == index]

    def active_take(self, index: int) -> RecordedTake | None:
        take_id = str(self.active_takes.get(str(index)) or "")
        if not take_id:
            return None
        for take in self.takes:
            if take.id == take_id and take.replica_index == index:
                return take
        return None

    def missing_replicas(self) -> list[int]:
        """Индексы реплик без активного дубля — рендер из-за них невозможен."""
        return [
            int(replica["index"])
            for replica in self.replicas
            if self.active_take(int(replica["index"])) is None
        ]

    def readiness(self) -> dict:
        total = len(self.replicas)
        done = total - len(self.missing_replicas())
        roles_ready = 0
        for speaker in self.speakers:
            indexes = [
                int(replica["index"])
                for replica in self.replicas
                if str(replica.get("speaker")) == speaker
            ]
            if indexes and all(self.active_take(index) is not None for index in indexes):
                roles_ready += 1
        return {
            "replicas_total": total,
            "replicas_recorded": done,
            "roles_total": len(self.speakers),
            "roles_ready": roles_ready,
            "missing": self.missing_replicas(),
            "progress": round(done / total, 4) if total else 0.0,
            "ready_to_render": bool(total) and not self.missing_replicas(),
        }

    # -- сериализация ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "dialogue_text": self.dialogue_text,
            "status": self.status,
            "replicas": [dict(item) for item in self.replicas],
            "role_voices": dict(self.role_voices),
            "voice_profiles": [item.to_dict() for item in self.voice_profiles],
            "takes": [item.to_dict() for item in self.takes],
            "active_takes": dict(self.active_takes),
            "recording_order": self.recording_order,
            "render_settings": dict(self.render_settings),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> RecordingProject:
        return cls(
            id=str(raw.get("id") or uuid.uuid4().hex[:12]),
            name=str(raw.get("name") or "Запись"),
            dialogue_text=str(raw.get("dialogue_text") or ""),
            status=str(raw.get("status") or PROJECT_STATUS_DRAFT),
            replicas=[dict(item) for item in (raw.get("replicas") or [])],
            role_voices={str(k): str(v) for k, v in (raw.get("role_voices") or {}).items()},
            voice_profiles=[
                RecordedVoiceProfile.from_dict(item) for item in (raw.get("voice_profiles") or [])
            ],
            takes=[RecordedTake.from_dict(item) for item in (raw.get("takes") or [])],
            active_takes={str(k): str(v) for k, v in (raw.get("active_takes") or {}).items()},
            recording_order=str(raw.get("recording_order") or "dialogue"),
            render_settings=dict(raw.get("render_settings") or {}),
            created_at=str(raw.get("created_at") or _now()),
            updated_at=str(raw.get("updated_at") or _now()),
            last_error=str(raw.get("last_error") or ""),
        )


class RecordingStore:
    """CRUD проектов записи, дублей и настроек обработки.

    Все операции под одним замком и с атомарной записью: состояние проекта
    обновляется после каждого события (добавили дубль, выбрали активный, сменили
    обработку), и параллельный запрос не должен затирать чужую правку.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else config.RECORDINGS_DIR
        self._lock = threading.RLock()

    # -- пути ------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    def project_dir(self, project_id: str) -> Path:
        return self._root / safe_component(project_id)

    def raw_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / RAW_DIR

    def processed_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PROCESSED_DIR

    def preview_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PREVIEW_DIR

    def output_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / OUTPUT_DIR

    def raw_path(self, project_id: str, take: RecordedTake) -> Path:
        """Путь сырого дубля. Проверяется, что имя ведёт внутрь `raw/`."""
        name = os.path.basename(take.raw_file or "")
        if not name:
            raise KeyError(take.id)
        return self.raw_dir(project_id) / name

    # -- индекс ----------------------------------------------------------------
    def _read_index(self) -> list[dict]:
        path = self._root / INDEX_NAME
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.error("Индекс проектов записи повреждён: %s", exc)
            return []
        items = payload.get("projects") if isinstance(payload, dict) else payload
        return [dict(item) for item in (items or []) if isinstance(item, dict)]

    def _write_index(self, items: list[dict]) -> None:
        _atomic_write(self._root / INDEX_NAME, {"projects": items})

    def _touch_index(self, project: RecordingProject) -> None:
        items = [item for item in self._read_index() if str(item.get("id")) != project.id]
        items.append(
            {
                "id": project.id,
                "name": project.name,
                "updated_at": project.updated_at,
                "status": project.status,
                "replicas": len(project.replicas),
                "recorded": len(project.replicas) - len(project.missing_replicas()),
            }
        )
        items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        self._write_index(items)

    # -- проекты ---------------------------------------------------------------
    def list_projects(self) -> list[dict]:
        with self._lock:
            return self._read_index()

    def create_project(self, name: str, dialogue_text: str = "") -> RecordingProject:
        name = name.strip()
        if not name:
            raise ValueError("Не задано имя проекта записи")
        project = RecordingProject(
            id=uuid.uuid4().hex[:12], name=name, dialogue_text=dialogue_text
        )
        with self._lock:
            for directory in (
                self.raw_dir(project.id),
                self.processed_dir(project.id),
                self.preview_dir(project.id),
                self.output_dir(project.id),
            ):
                directory.mkdir(parents=True, exist_ok=True)
            self._save(project)
        logger.info("Проект записи %s создан: «%s»", project.id, project.name)
        return project

    def save(self, project: RecordingProject) -> RecordingProject:
        with self._lock:
            return self._save(project)

    def _save(self, project: RecordingProject) -> RecordingProject:
        project.updated_at = _now()
        _atomic_write(self.project_dir(project.id) / PROJECT_NAME, project.to_dict())
        self._touch_index(project)
        return project

    def get_project(self, project_id: str) -> RecordingProject | None:
        path = self.project_dir(project_id) / PROJECT_NAME
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.error("Проект записи %s не прочитан: %s", project_id, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return RecordingProject.from_dict(payload)

    def require_project(self, project_id: str) -> RecordingProject:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        return project

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        dialogue_text: str | None = None,
        recording_order: str | None = None,
        render_settings: dict | None = None,
        status: str | None = None,
        last_error: str | None = None,
    ) -> RecordingProject:
        with self._lock:
            project = self.require_project(project_id)
            if name is not None:
                project.name = name.strip() or project.name
            if dialogue_text is not None:
                project.dialogue_text = dialogue_text
            if recording_order is not None:
                if recording_order not in ("dialogue", "roles"):
                    raise ValueError("Порядок записи бывает «dialogue» или «roles»")
                project.recording_order = recording_order
            if render_settings is not None:
                project.render_settings = dict(render_settings)
            if status is not None:
                project.status = status
            if last_error is not None:
                project.last_error = last_error
            return self._save(project)

    def delete_project(self, project_id: str) -> bool:
        with self._lock:
            directory = self.project_dir(project_id)
            if not directory.exists():
                return False
            shutil.rmtree(directory, ignore_errors=True)
            self._write_index(
                [item for item in self._read_index() if str(item.get("id")) != project_id]
            )
        logger.info("Проект записи %s удалён", project_id)
        return True

    # -- разбор ----------------------------------------------------------------
    def set_replicas(self, project_id: str, replicas: list[dict]) -> RecordingProject:
        """Сохраняет разбор диалога, **не меняя** порядок реплик.

        Индексы приходят из общего парсера и остаются исходными: порядок монтажа
        определяется ими, а не порядком записи (§4, Test C).
        """
        with self._lock:
            project = self.require_project(project_id)
            project.replicas = [
                {
                    "index": int(item["index"]),
                    "speaker": str(item.get("speaker") or ""),
                    "text": str(item.get("text") or ""),
                }
                for item in replicas
            ]
            valid = {int(item["index"]) for item in project.replicas}
            project.takes = [take for take in project.takes if take.replica_index in valid]
            project.active_takes = {
                key: value
                for key, value in project.active_takes.items()
                if int(key) in valid
            }
            # Роли, которых больше нет в тексте, теряют назначение голоса.
            speakers = set(project.speakers)
            project.role_voices = {
                key: value for key, value in project.role_voices.items() if key in speakers
            }
            return self._save(project)

    # -- записываемые голоса ---------------------------------------------------
    def add_voice_profile(
        self,
        project_id: str,
        name: str,
        *,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        denoise: bool = False,
    ) -> RecordedVoiceProfile:
        with self._lock:
            project = self.require_project(project_id)
            profile = RecordedVoiceProfile(
                id=uuid.uuid4().hex[:12],
                name=name.strip() or f"Голос {len(project.voice_profiles) + 1}",
                speed=speed,
                pitch_semitones=pitch_semitones,
                denoise=denoise,
            )
            project.voice_profiles.append(profile)
            self._save(project)
            return profile

    def update_voice_profile(
        self,
        project_id: str,
        profile_id: str,
        *,
        name: str | None = None,
        speed: float | None = None,
        pitch_semitones: float | None = None,
        denoise: bool | None = None,
    ) -> RecordedVoiceProfile:
        with self._lock:
            project = self.require_project(project_id)
            profile = project.profile(profile_id)
            if profile is None:
                raise KeyError(profile_id)
            if name is not None:
                profile.name = name.strip() or profile.name
            if speed is not None:
                profile.speed = float(speed)
            if pitch_semitones is not None:
                profile.pitch_semitones = float(pitch_semitones)
            if denoise is not None:
                profile.denoise = bool(denoise)
            self._save(project)
            return profile

    def assign_role(self, project_id: str, speaker: str, profile_id: str | None) -> RecordingProject:
        """Назначает роли записываемый голос; `None` — снять назначение."""
        with self._lock:
            project = self.require_project(project_id)
            if speaker not in project.speakers:
                raise KeyError(speaker)
            if profile_id and project.profile(profile_id) is None:
                raise KeyError(profile_id)
            if profile_id:
                project.role_voices[speaker] = profile_id
            else:
                project.role_voices.pop(speaker, None)
            return self._save(project)

    # -- дубли -----------------------------------------------------------------
    def add_take(
        self,
        project_id: str,
        index: int,
        *,
        voice_profile_id: str,
        raw_bytes: bytes,
        suffix: str,
        duration_sec: float,
        level_warning: str = "",
    ) -> RecordedTake:
        """Сохраняет сырой дубль и делает его активным.

        Файл имени формируется здесь (`r{index:04d}_take{номер}{suffix}`) и только
        внутри `raw/`: ни имя роли, ни исходное имя файла из браузера в путь не
        попадают (§23). Прошлый дубль не перезаписывается — он остаётся в списке.
        """
        with self._lock:
            project = self.require_project(project_id)
            indexes = {int(item["index"]) for item in project.replicas}
            if int(index) not in indexes:
                raise KeyError(index)
            if voice_profile_id and project.profile(voice_profile_id) is None:
                raise KeyError(voice_profile_id)
            take_id = uuid.uuid4().hex[:12]
            number = len(project.takes_for(int(index))) + 1
            name = f"r{int(index):04d}_take{number:02d}{(suffix or '.wav').lower()}"
            target = self.raw_dir(project_id) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw_bytes)
            take = RecordedTake(
                id=take_id,
                replica_index=int(index),
                voice_profile_id=str(voice_profile_id or ""),
                raw_file=name,
                duration_sec=float(duration_sec or 0.0),
                level_warning=str(level_warning or ""),
            )
            project.takes.append(take)
            project.active_takes[str(int(index))] = take.id
            self._save(project)
        logger.info(
            "Проект записи %s: дубль реплики %s сохранён (%s, %.2f с)",
            project_id, index, name, duration_sec,
        )
        return take

    def list_takes(self, project_id: str, index: int | None = None) -> list[RecordedTake]:
        project = self.require_project(project_id)
        if index is None:
            return list(project.takes)
        return project.takes_for(int(index))

    def select_take(self, project_id: str, index: int, take_id: str) -> RecordingProject:
        with self._lock:
            project = self.require_project(project_id)
            take = next(
                (
                    item
                    for item in project.takes
                    if item.id == take_id and item.replica_index == int(index)
                ),
                None,
            )
            if take is None:
                raise KeyError(take_id)
            project.active_takes[str(int(index))] = take.id
            return self._save(project)

    def delete_take(self, project_id: str, index: int, take_id: str) -> RecordingProject:
        """Удаляет дубль; если он был активным, активным становится последний.

        Отсутствие дублей — не ошибка: реплика просто снова «не записана», и
        рендер её не пропустит (§10, Test F).
        """
        with self._lock:
            project = self.require_project(project_id)
            take = next(
                (
                    item
                    for item in project.takes
                    if item.id == take_id and item.replica_index == int(index)
                ),
                None,
            )
            if take is None:
                raise KeyError(take_id)
            project.takes = [item for item in project.takes if item.id != take_id]
            if str(project.active_takes.get(str(int(index)))) == take_id:
                remaining = project.takes_for(int(index))
                if remaining:
                    project.active_takes[str(int(index))] = remaining[-1].id
                else:
                    project.active_takes.pop(str(int(index)), None)
            self._save(project)
            path = self.raw_dir(project_id) / os.path.basename(take.raw_file)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Сырой дубль %s не удалён: %s", path, exc)
        return project

    # -- кэш обработки ---------------------------------------------------------
    def processed_path(self, project_id: str, take: RecordedTake, settings_key: str) -> Path:
        """Путь обработанной версии дубля с ключом настроек в имени.

        Ключ в имени — то, что делает кэш неразрушающим (§16): другая скорость или
        высота даёт другой файл, а «поверх уже обработанного» не выполняется
        никогда. Файлы с прежними настройками можно удалять без сожаления.
        """
        directory = self.processed_dir(project_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe_component(take.id)}-{safe_component(settings_key)}.wav"

    def preview_path(self, project_id: str, name: str) -> Path:
        directory = self.preview_dir(project_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / safe_component(name)

    def clear_processed(self, project_id: str) -> None:
        shutil.rmtree(self.processed_dir(project_id), ignore_errors=True)


_store: RecordingStore | None = None
_store_lock = threading.Lock()


def get_store() -> RecordingStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RecordingStore()
    return _store


def reset_store() -> None:
    """Сброс singleton'а — для тестов и после смены каталога записей."""
    global _store
    with _store_lock:
        _store = None


__all__ = [
    "PROJECT_STATUS_DRAFT",
    "PROJECT_STATUS_ERROR",
    "PROJECT_STATUS_RENDERED",
    "RecordedTake",
    "RecordedVoiceProfile",
    "RecordingProject",
    "RecordingStore",
    "get_store",
    "reset_store",
    "safe_component",
]
