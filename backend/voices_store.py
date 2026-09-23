"""CRUD над voices.json и файлами референс-аудио."""

import json
import logging
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import audio_analysis, config, emotions, transcribe
from .denoise import clean_bytes as clean_reference_bytes
from .engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    builtin_voices,
    default_engine_for_gender,
    normalize_engine_params,
    requires_reference,
)
from .settings_resolution import COMMON_FIELDS, INT_FIELDS

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif", ".webm"}
MAX_AUDIO_BYTES = 50 * 1024 * 1024
DEMO_PREFIX = "[demo]"

# Имя файла референса: `<voice_id><suffix>` у основного и
# `<voice_id>-<profile_id><suffix>` у интонационного профиля — оба id это
# `uuid.uuid4().hex[:12]`. По этому шаблону восстанавливается привязка
# «файл → голос», когда `voices.json` пропал (F-F2).
_REFERENCE_FILE_RE = re.compile(r"^(?P<voice>[0-9a-f]{12})(?:-(?P<profile>[0-9a-f]{12}))?$")


def _with_demo_prefix(name: str) -> str:
    """Демо-клип F5-TTS помечаем в названии, чтобы его не приняли за реальную запись."""
    return name if name.lower().startswith(DEMO_PREFIX) else f"{DEMO_PREFIX} {name}"


def _resolve_engine(engine: str | None, gender: str) -> str:
    """Движок голоса: явный выбор пользователя, иначе подсказка по полу.

    Терпимо относится к неизвестному id: `voices.json` могли дописать руками или
    откатить, и терять из-за этого весь список голосов незачем. Явный выбор
    через API проверяется отдельно (`check_engine`) — там ошибку видно сразу.
    """
    if engine in ENGINE_INFOS:
        return engine
    if engine:
        logger.warning("Неизвестный движок «%s» у голоса — беру подсказку по полу", engine)
    return default_engine_for_gender(gender)


def check_engine(engine: str) -> str:
    """Проверяет движок, пришедший от пользователя: опечатка должна быть ошибкой."""
    if engine not in ENGINE_INFOS:
        known = ", ".join(sorted(ENGINE_INFOS))
        raise ValueError(f"Неизвестный движок синтеза: {engine or '(пусто)'}. Доступны: {known}")
    return engine


def _normalize_preset(raw: dict | None) -> dict:
    """Приводит пресет голоса к известным ручкам синтеза и к числам.

    Пресет — это то, что пользователь подобрал в «Прослушать»: скорость, CFG, NFE
    и прочие общие ручки (см. `settings_resolution`). Чужой ключ или строка вместо
    числа означали бы значение, которое пайплайн либо не прочитает, либо прочитает
    неверно, поэтому `voices.json` при чтении чистится — файл правят и руками.
    """
    if not isinstance(raw, dict):
        return {}
    preset: dict[str, float | int] = {}
    for name, value in raw.items():
        if name not in COMMON_FIELDS or value is None:
            continue
        try:
            preset[name] = int(value) if name in INT_FIELDS else float(value)
        except (TypeError, ValueError):
            logger.warning("Пресет голоса: поле %s = %r не число — пропускаю", name, value)
    return preset


def _inspect_reference(audio_bytes: bytes, suffix: str, name: str, gender: str) -> dict:
    """Проверяет референс перед сохранением: F0 против заявленного пола, полоса частот и демо-файлы F5-TTS.

    Ничего не блокирует — только собирает метки для карточки голоса и лога.
    """
    estimate = audio_analysis.analyze(audio_bytes, suffix)
    demo_source = audio_analysis.find_demo_source(audio_bytes)
    if demo_source:
        logger.warning("Загруженная запись — копия демо-файла F5-TTS %s", demo_source)
    return {
        "f0_hz": estimate.f0_hz,
        "gender_warning": audio_analysis.check_gender_mismatch(estimate.f0_hz, gender),
        "band_warning": estimate.band_warning,
        "is_demo": bool(demo_source),
        "demo_source": demo_source,
        "name": _with_demo_prefix(name) if demo_source else name,
    }


def _duration_sec(audio_bytes: bytes, suffix: str) -> float | None:
    """Длительность записи для карточки профиля (§18).

    Неудачное декодирование не отменяет сохранение: профиль уже записан, и
    отсутствие длительности — не повод терять пользовательскую запись.
    """
    try:
        samples, rate = audio_analysis.load_mono(audio_bytes, suffix)
    except Exception as exc:  # noqa: BLE001 — метрика, а не проверка качества
        logger.warning("Не удалось измерить длительность референса: %s", exc)
        return None
    return round(len(samples) / rate, 3) if rate else None


# Статусы проверки профиля (§22). `unknown` — запись, которую не сверяли с
# расшифровкой (старая или загруженная руками): предпочитать её проверенной
# нельзя, но и терять не за что.
QUALITY_OK = "ok"
QUALITY_WARNING = "warning"
QUALITY_UNKNOWN = "unknown"

# Статус расшифровки референса (§18): что именно известно про `ref_text`.
# Отдельно от `quality_status` — там качество записи целиком, здесь только текст.
TRANSCRIPTION_OK = "ok"  # распознавание прошло, текст совпал
TRANSCRIPTION_WARNING = "warning"  # распознавание прошло, текст расходится
TRANSCRIPTION_FROM_FORM = "from_form"  # текст взят из формы, без распознавания
TRANSCRIPTION_UNKNOWN = "unknown"  # не проверяли (старые профили)


def _as_bool(value: object, *, default: bool) -> bool:
    """Флаг из JSON: `voices.json` правят руками, и `"false"` строкой — не редкость."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "да"}
    return bool(value)


@dataclass
class ReferenceProfile:
    """Один референс голоса: нейтральный или интонационный профиль (UPDATE 2 §8,
    UPDATE 3 §18).

    Профиль принадлежит **голосу**: он лежит внутри записи голоса, и это делает
    «взять эмоциональный референс чужого голоса» структурно невозможным, а не
    запрещённым проверкой, которую однажды забудут.

    `quality_status` — итог уже существующей проверки «референс ↔ расшифровка»:
    `ok` (совпало), `warning` (расшифровка расходится), `unknown` (не проверяли).
    Резолвер не предпочитает профиль с `warning`, если есть нейтральный (§38).

    Как читать поля, которые легко перепутать:

    * `emotion` — ключ профиля: какой интонации принадлежит запись. Хранится под
      этим именем, потому что так он уже лежит в `voices.json` и в API; наружу
      отдаётся ещё и как `profile_key` (§18) — второе имя того же значения.
    * `source_record_phrase_id` — какая фраза `RECORD_PHRASES` записана (§16).
    * `is_default` — профиль по умолчанию для своей интонации: при нескольких
      записях одной эмоции резолвер берёт сначала его (§27).
    * `enabled` — выключенный профиль виден в интерфейсе, но автоматика его не
      выбирает (§35): неудачную запись не обязательно удалять, чтобы она перестала
      влиять на синтез.
    * `enabled_for_auto` — разрешён ли профиль **автоматическому** выбору референса
      (§35). По умолчанию `False`: пока профиль не подтверждён benchmark'ом и
      прослушиванием (§34), автоматика его не берёт — даже если интонация совпала.
      Ручной выбор реплики (`reference_profile_id`) этот флаг не ограничивает:
      экспериментальный профиль остаётся доступным человеку, но не становится
      производственной маршрутизацией сам по себе.
    """

    id: str
    # Голос-владелец (§18, §19). Поле производное: единственный источник истины —
    # вложенность профиля в запись голоса, и `VoicesStore._read` восстанавливает
    # значение оттуда. Хранится ради самодостаточности профиля в API и метаданных.
    voice_id: str = ""
    emotion: str = emotions.EMOTION_NEUTRAL
    label: str = ""
    audio_file: str = ""  # имя файла внутри voices/
    ref_text: str = ""
    # Кем записан профиль: `legacy` — основной референс голоса, `record` — запись
    # из интерфейса. Нужен, чтобы миграция не выдавала старое за новое.
    source: str = "legacy"
    quality_status: str = QUALITY_UNKNOWN
    quality_note: str = ""
    transcription_status: str = TRANSCRIPTION_UNKNOWN
    duration_sec: float | None = None  # длительность записи; None — измерить не удалось
    # Движки, для которых профиль проверен. Пустой список — «любой движок голоса».
    engine_compatibility: list = field(default_factory=list)
    source_record_phrase_id: str = ""
    is_default: bool = False
    enabled: bool = True
    # Подтверждён ли профиль benchmark'ом (§35). Значение по умолчанию —
    # консервативное: новая запись не становится автоматической сама по себе.
    enabled_for_auto: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    # Меняется при правке профиля; пустая строка — «не правили с момента записи».
    updated_at: str = ""

    @property
    def profile_key(self) -> str:
        """Ключ профиля (§18) — имя поля в терминах UPDATE 3.

        Хранится как `emotion`: переименование в `voices.json` заставило бы
        мигрировать данные ради смены ярлыка, а старое имя уже ушло в API.
        """
        return self.emotion

    @property
    def audio_path(self) -> Path:
        return config.VOICES_DIR / self.audio_file

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_audio"] = bool(self.audio_file) and self.audio_path.exists()
        data["emotion_title"] = emotions.EMOTION_TITLES.get(self.emotion, self.emotion)
        # Имя поля из §18: те же данные, второй сущности для них не заводим.
        data["profile_key"] = self.emotion
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "ReferenceProfile | None":
        """Профиль из JSON; `None` — запись без обязательных полей.

        Терпимость к старым и чужим файлам та же, что у голоса: неизвестные ключи
        отбрасываются, а запись без идентификатора или файла пропускается, чтобы
        один битый профиль не уносил с собой весь голос.
        """
        if not isinstance(raw, dict):
            return None
        known = {key: value for key, value in raw.items() if key in _PROFILE_FIELDS}
        if not known.get("id") or not known.get("audio_file"):
            return None
        # `profile_key` — имя поля из §18, `emotion` — то, под которым профиль
        # лежит в `voices.json` и в API. Принимаем оба: запись нового формата не
        # должна терять интонацию при чтении.
        if raw.get("emotion") is None and raw.get("profile_key") is not None:
            known["emotion"] = raw["profile_key"]
        known["emotion"] = emotions.normalize_emotion(known.get("emotion"))
        compatibility = known.get("engine_compatibility") or []
        known["engine_compatibility"] = (
            [str(item) for item in compatibility] if isinstance(compatibility, list) else []
        )
        known["voice_id"] = str(known.get("voice_id") or "")
        known["updated_at"] = str(known.get("updated_at") or "")
        known["source_record_phrase_id"] = str(known.get("source_record_phrase_id") or "")
        known["transcription_status"] = str(
            known.get("transcription_status") or TRANSCRIPTION_UNKNOWN
        )
        known["enabled"] = _as_bool(known.get("enabled"), default=True)
        known["is_default"] = _as_bool(known.get("is_default"), default=False)
        # Флаг подтверждения (§35) по умолчанию снят: старый профиль, у которого
        # поля нет вовсе, не должен задним числом получить право на автоматику.
        known["enabled_for_auto"] = _as_bool(known.get("enabled_for_auto"), default=False)
        try:
            known["duration_sec"] = (
                float(known["duration_sec"]) if known.get("duration_sec") is not None else None
            )
        except (TypeError, ValueError):
            known["duration_sec"] = None
        try:
            return cls(**known)
        except TypeError:
            return None


@dataclass
class Voice:
    id: str
    name: str
    gender: str
    ref_text: str
    audio_file: str  # имя файла внутри voices/
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    f0_hz: float | None = None  # измеренная высота тона референса, Гц
    gender_warning: str | None = None  # F0 противоречит заявленному полу
    band_warning: str | None = None  # запись узкополосная: верх срезан кодеком/телефоном
    is_demo: bool = False  # файл — копия демо-клипа из пакета f5_tts
    demo_source: str | None = None  # имя этого демо-файла
    ref_text_warning: str | None = None  # расшифровка не совпала с записью
    # Движок синтеза выбирается отдельно для каждого голоса (см. engines/base):
    # один и тот же текст может озвучиваться разными моделями в одном диалоге.
    engine: str = ENGINE_F5
    # Значения ручек выбранного движка (температура и т.п.) — дефолты для этого
    # голоса; карточка слота может переопределить их на одну генерацию.
    engine_params: dict = field(default_factory=dict)
    # Пресет: общие ручки синтеза (скорость, CFG, NFE, громкость), подобранные
    # пользователем в «Прослушать». Это слой между паспортом движка и слотом
    # проекта: новый диалог берёт эти значения сам, без повторной настройки.
    preset: dict = field(default_factory=dict)
    # Референс-профили: нейтральный плюс эмоциональные (UPDATE 2 §8). Пустой
    # список — старый голос: его основной референс читается как NEUTRAL
    # (`reference_profiles`), поэтому миграция данных не нужна.
    profiles: list = field(default_factory=list)

    @property
    def audio_path(self) -> Path:
        return config.VOICES_DIR / self.audio_file

    def reference_profiles(self) -> list["ReferenceProfile"]:
        """Профили голоса, включая проекцию старого референса в NEUTRAL (§9).

        Голос, сохранённый до появления эмоций, имеет один `audio_file` и один
        `ref_text`. Он обязан продолжать работать и всегда доступен как NEUTRAL —
        даже когда рядом уже записаны эмоциональные профили: «нет референса для
        восторга» не должно означать «нет референса вообще».

        Проекция вычисляется на чтение: переписывать `voices.json` при каждом
        чтении нельзя, а разойтись двум представлениям одного референса негде —
        источник один.
        """
        profiles = [item for item in self.profiles if isinstance(item, ReferenceProfile)]
        neutral = self.neutral_profile()
        if not neutral.audio_file:
            return profiles
        if any(item.audio_file == neutral.audio_file for item in profiles):
            return profiles
        return [neutral, *profiles]

    def neutral_profile(self) -> "ReferenceProfile":
        """Основной референс голоса как NEUTRAL-профиль.

        Качество берётся из уже посчитанных предупреждений голоса: расхождение
        расшифровки (`ref_text_warning`) — ровно тот случай, когда эмоциональный
        профиль нельзя предпочитать нейтральному.
        """
        return ReferenceProfile(
            id=f"{self.id}-neutral",
            voice_id=self.id,
            emotion=emotions.EMOTION_NEUTRAL,
            label="Основной",
            audio_file=self.audio_file,
            ref_text=self.ref_text,
            source="legacy",
            quality_status=QUALITY_WARNING if self.ref_text_warning else QUALITY_OK,
            quality_note=self.ref_text_warning or "",
            # Старый голос расшифровку не сверял: у него нет и признака, что текст
            # взят из формы, поэтому «неизвестно» — честнее, чем «совпало».
            transcription_status=(
                TRANSCRIPTION_WARNING if self.ref_text_warning else TRANSCRIPTION_UNKNOWN
            ),
            engine_compatibility=[self.engine] if self.engine else [],
            created_at=self.created_at,
            updated_at=self.created_at,
        )

    def profile(self, profile_id: str) -> "ReferenceProfile | None":
        for item in self.reference_profiles():
            if item.id == profile_id:
                return item
        return None

    def to_dict(self) -> dict:
        data = asdict(self)
        # Наличие референса — это имя файла, а не существование пути: у голоса
        # пресетного движка `audio_file` пуст, и `VOICES_DIR / ""` — сам каталог
        # `voices/`, который существует всегда. Без проверки имени интерфейс
        # показывал бы «референс есть» у голоса, у которого его нет.
        data["has_audio"] = bool(self.audio_file) and self.audio_path.exists()
        # Профили отдаются уже с проекцией старого референса: интерфейсу не нужно
        # знать, был голос записан до появления эмоций или после.
        data["reference_profiles"] = [item.to_dict() for item in self.reference_profiles()]
        return data


_VOICE_FIELDS = {f.name for f in fields(Voice)}
_PROFILE_FIELDS = {f.name for f in fields(ReferenceProfile)}


class VoicesStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    # -- внутреннее ------------------------------------------------------------
    def _payload(self) -> dict:
        """Содержимое voices.json как словарь; отсутствующий или битый файл — пустой.

        Отдельно от `_read`, потому что в файле лежит не только список голосов:
        рядом живёт отметка о заведённых встроенных голосах движков, и она не
        должна теряться при каждой записи списка.
        """
        if not config.VOICES_JSON.exists():
            return {}
        try:
            payload = json.loads(config.VOICES_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("voices.json повреждён, читаю как пустой: %s", exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _read_seeded(self) -> list[str]:
        """Движки, встроенные голоса которых уже заведены (см. `ensure_builtin_voices`)."""
        seeded = self._payload().get("builtin_seeded")
        if not isinstance(seeded, list):
            return []
        return [item for item in seeded if isinstance(item, str)]

    def _read(self) -> list[Voice]:
        payload = self._payload()
        voices: list[Voice] = []
        for item in payload.get("voices", []):
            if not isinstance(item, dict):
                continue
            # Лишние/незнакомые поля (например, дописанные вручную или новой
            # версией) не должны ронять весь список голосов.
            known = {key: value for key, value in item.items() if key in _VOICE_FIELDS}
            try:
                voice = Voice(**known)
                # Движок и его ручки приводим к известным значениям: файл могли
                # править руками, а дальше эти данные уходят прямо в модель.
                voice.engine = _resolve_engine(voice.engine, voice.gender)
                voice.engine_params = normalize_engine_params(
                    voice.engine, voice.engine_params if isinstance(voice.engine_params, dict) else {}
                )
                voice.preset = _normalize_preset(voice.preset)
                # Профили: битая запись пропускается поштучно, остальные остаются.
                voice.profiles = [
                    profile
                    for profile in (
                        ReferenceProfile.from_dict(raw) for raw in (voice.profiles or [])
                    )
                    if profile is not None
                ]
                # Владелец профиля — сама запись голоса: поле восстанавливается из
                # неё, чтобы `voice_id` не мог разойтись с реальностью при правке
                # `voices.json` руками.
                for profile in voice.profiles:
                    profile.voice_id = voice.id
                voices.append(voice)
            except TypeError as exc:  # запись без обязательного поля — пропускаем
                logger.warning("Пропускаю некорректную запись в voices.json: %s", exc)
        return voices

    def _write(self, voices: list[Voice], seeded: list[str] | None = None) -> None:
        """Записывает список голосов, сохраняя отметку о встроенных голосах движков.

        `seeded=None` — «не трогать отметку»: её меняет только
        `ensure_builtin_voices`, а остальные записи (создание голоса, правка
        движка, профили) должны её сохранить, иначе встроенные голоса вернулись бы
        после первого же удаления.
        """
        data: dict = {"voices": [asdict(v) for v in voices]}
        marker = sorted({*(self._read_seeded() if seeded is None else seeded)})
        if marker:
            data["builtin_seeded"] = marker
        config.VOICES_JSON.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_ref_text(self, audio_path: Path, ref_text: str, verify: bool) -> tuple[str, str | None]:
        """Готовит дословную расшифровку референса и сверяет её с введённой вручную.

        Пустое поле заполняем распознанным текстом: без него F5-TTS поднимает
        собственный ASR уже во время синтеза, в основном процессе.

        `verify=False` — запись сделана в браузере по показанной на экране фразе:
        текст известен точно, и распознавание (десятки секунд) превратилось бы из
        проверки в лишний шаг. Отключает его пользователь осознанно, галочкой.
        """
        declared = ref_text.strip()
        if not verify:
            if not declared:
                raise ValueError(
                    "Нет расшифровки записи — впишите текст вручную или включите сверку с записью"
                )
            logger.info("Расшифровка референса %s взята из формы без распознавания", audio_path.name)
            return declared, None

        try:
            result = transcribe.transcribe_file(audio_path)
        except RuntimeError as exc:
            logger.warning("Не удалось распознать референс %s: %s", audio_path.name, exc)
            if declared:
                return declared, None
            raise ValueError(
                "Не удалось распознать речь в записи — впишите расшифровку вручную"
            ) from exc

        if not result.ref_text:
            if declared:
                return declared, None
            raise ValueError("В записи не распознана речь — впишите расшифровку вручную")
        if not declared:
            logger.info("Расшифровка референса %s заполнена автоматически", audio_path.name)
            return result.ref_text, None
        return declared, transcribe.check_ref_text_match(declared, result.ref_text)

    def _create_builtin_voice(self, name: str, gender: str, engine: str) -> Voice:
        """Карточка голоса для движка, который говорит своими встроенными голосами.

        Референса у неё нет и быть не должно: движок выбирает встроенный голос по
        полу (`EngineInfo.builtin_voices`), а запись он игнорирует. Пустой
        `audio_file` и есть признак «запись не нужна» — по нему интерфейс не
        показывает панель референсов, а пайплайн не ищет аудио. Имя и пол остаются
        пользовательскими: карточка — то, как он называет голос движка.
        """
        with self._lock:
            voices = self._read()
            voice = Voice(
                id=uuid.uuid4().hex[:12],
                name=name,
                gender=gender,
                ref_text="",
                audio_file="",
                engine=engine,
            )
            voices.append(voice)
            self._write(voices)
        logger.info(
            "Добавлен голос без референса «%s» (%s), движок %s", voice.name, voice.id, engine
        )
        return voice

    # -- публичное API ---------------------------------------------------------
    def list(self) -> list[Voice]:
        with self._lock:
            return self._read()

    def get(self, voice_id: str) -> Voice | None:
        return next((v for v in self.list() if v.id == voice_id), None)

    def recover_from_files(self) -> int:
        """Пересобирает `voices.json` сканированием каталога `voices/` (F-F2).

        Файл может пропасть — удалён руками или не пережил сбой, — а записи в
        каталоге останутся. Пустой список голосов выглядел бы как «ничего не
        записано», и пользователь заново писал бы то, что уже лежит на диске,
        поэтому библиотека собирается обратно по именам файлов.

        Восстанавливается только закодированное в имени: какой файл какому голосу
        принадлежит и какие профили у него есть. Имя, пол, расшифровку и движок по
        файлу не прочитать — берутся значения по умолчанию, а метки (F0, полоса)
        досчитает `inspect_existing` при следующем запуске.

        Существующий файл не трогается: это спасение потерянного, а не миграция.
        Возвращает число восстановленных голосов.
        """
        if config.VOICES_JSON.exists() or not config.VOICES_DIR.is_dir():
            return 0
        main: dict[str, str] = {}
        profiles: dict[str, list[tuple[str, str]]] = {}
        for path in sorted(config.VOICES_DIR.iterdir()):
            if path.is_dir() or path.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
                continue
            match = _REFERENCE_FILE_RE.match(path.stem)
            if match is None:
                continue
            voice_id = match.group("voice")
            profile_id = match.group("profile")
            if profile_id is None:
                main.setdefault(voice_id, path.name)
            else:
                profiles.setdefault(voice_id, []).append((profile_id, path.name))

        voices: list[Voice] = []
        for voice_id in sorted({*main, *profiles}):
            voices.append(
                Voice(
                    id=voice_id,
                    name=f"Восстановленный голос {voice_id}",
                    gender="other",
                    ref_text="",
                    audio_file=main.get(voice_id, ""),
                    profiles=[
                        ReferenceProfile(
                            id=profile_id,
                            voice_id=voice_id,
                            emotion=emotions.EMOTION_NEUTRAL,
                            label="Восстановлен из файла",
                            audio_file=file_name,
                        )
                        for profile_id, file_name in profiles.get(voice_id, [])
                    ],
                )
            )
        if not voices:
            return 0
        with self._lock:
            self._write(voices)
        logger.warning(
            "voices.json отсутствовал — восстановлено голосов по файлам: %s", len(voices)
        )
        return len(voices)

    # Аннотация `list` здесь строкой: в теле класса это имя уже занято методом
    # списка голосов, и по значению она взяла бы его.
    def ensure_builtin_voices(self, engine_ids: "list[str]") -> int:
        """Заводит карточки встроенных голосов для движков, у которых есть веса.

        Вызывается при старте и после скачивания модели: пока файлов движка нет,
        голосам не с чем работать, а как только они появились — карточки должны
        быть видны среди остальных, без записи референса и без подкладывания аудио
        (см. `EngineInfo.builtin_voices`).

        Идемпотентно **по движку**: отметка `builtin_seeded` в `voices.json`
        помнит, для каких движков карточки уже заводились, поэтому удалённый
        пользователем встроенный голос не возвращается при следующем запуске —
        иначе удалить его было бы нельзя вовсе. Повторный запуск с тем же списком
        движков не создаёт ничего.

        Список движков передаёт вызывающий: «какие веса лежат на диске» — знание
        `model_manager`, а не хранилища голосов.
        """
        created = 0
        with self._lock:
            voices = self._read()
            seeded = self._read_seeded()
            for engine_id in engine_ids:
                presets = builtin_voices(engine_id)
                if not presets or engine_id in seeded:
                    continue
                for preset in presets:
                    voices.append(
                        Voice(
                            id=uuid.uuid4().hex[:12],
                            name=preset.label,
                            gender=preset.gender,
                            ref_text="",
                            audio_file="",
                            engine=engine_id,
                        )
                    )
                    created += 1
                seeded.append(engine_id)
            if created:
                self._write(voices, seeded)
        if created:
            logger.info("Завёл встроенные голоса движков: %s", created)
        return created

    def create(
        self,
        name: str,
        gender: str,
        ref_text: str,
        audio_filename: str,
        audio_bytes: bytes,
        verify_ref_text: bool = True,
        engine: str = "",
        denoise: bool = False,
    ) -> Voice:
        """Заводит голос: движку без клонирования — без записи, иначе с референсом."""
        if not name.strip():
            raise ValueError("Не задано имя голоса")

        gender = gender if gender in ("male", "female", "other") else "other"
        engine = check_engine(engine) if engine else default_engine_for_gender(gender)
        # Движку без клонирования запись не нужна: он говорит своим встроенным
        # голосом, и карточка отличается от заведённой приложением только именем и
        # полом. Проверять здесь аудио значило бы требовать то, что движок
        # игнорирует (см. EngineInfo.builtin_voices).
        if not requires_reference(engine):
            return self._create_builtin_voice(name.strip(), gender, engine)

        if not audio_filename:
            # Файла нет вовсе. Движкам со встроенными голосами запись не нужна, и
            # они ушли веткой выше (см. `requires_reference`), поэтому здесь её
            # отсутствие — ошибка запроса, а не повод завести голос без референса.
            raise ValueError("Нужна запись голоса: этот движок синтезирует по референсу")
        suffix = Path(audio_filename).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"Неподдерживаемый формат аудио: {suffix or '(нет расширения)'}")
        if not audio_bytes:
            raise ValueError("Пустой файл аудио")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(f"Файл больше {MAX_AUDIO_BYTES // (1024 * 1024)} МБ")

        if denoise:
            # Чистим до всех проверок и до записи файла: и F0, и полоса, и расшифровка
            # должны считаться по той записи, которая реально уйдёт в модель.
            audio_bytes = clean_reference_bytes(audio_bytes, suffix)
            suffix = ".wav"  # денойзер всегда отдаёт wav, чем бы ни был исходник
        checks = _inspect_reference(audio_bytes, suffix, name.strip(), gender)

        voice_id = uuid.uuid4().hex[:12]
        filename = f"{voice_id}{suffix}"
        audio_path = config.VOICES_DIR / filename
        audio_path.write_bytes(audio_bytes)
        try:
            # Распознавание — до захвата блокировки: это десятки секунд, и держать
            # на них список голосов (GET /api/voices) незачем.
            final_text, text_warning = self._resolve_ref_text(audio_path, ref_text, verify_ref_text)
            with self._lock:
                voices = self._read()
                voice = Voice(
                    id=voice_id,
                    name=checks["name"],
                    gender=gender,
                    ref_text=final_text,
                    audio_file=filename,
                    f0_hz=checks["f0_hz"],
                    gender_warning=checks["gender_warning"],
                    band_warning=checks["band_warning"],
                    is_demo=checks["is_demo"],
                    demo_source=checks["demo_source"],
                    ref_text_warning=text_warning,
                    engine=engine,
                )
                voices.append(voice)
                self._write(voices)
        except Exception:
            # Запись не сохранилась — не оставляем осиротевший файл в voices/.
            audio_path.unlink(missing_ok=True)
            raise
        logger.info(
            "Добавлен голос %s (%s), движок %s, F0=%s Гц", voice.name, voice.id, voice.engine, voice.f0_hz
        )
        if voice.is_demo:
            logger.warning(
                "Голос «%s» — это демо-клип F5-TTS (%s), а не пользовательская запись",
                voice.name, voice.demo_source,
            )
        if voice.gender_warning:
            logger.warning("Голос «%s»: %s", voice.name, voice.gender_warning)
        if voice.band_warning:
            logger.warning("Голос «%s»: %s", voice.name, voice.band_warning)
        if voice.ref_text_warning:
            logger.warning("Голос «%s»: %s", voice.name, voice.ref_text_warning)
        return voice

    def update(
        self,
        voice_id: str,
        engine: str | None = None,
        engine_params: dict | None = None,
        preset: dict | None = None,
    ) -> Voice:
        """Меняет движок, его ручки и пресет у существующего голоса.

        Записи голоса не трогаются: движок — это свойство голоса, а не проекта,
        и его смена не требует перезаписи референса или его расшифровки.
        `preset=None` — «не трогать», пустой словарь — «сбросить подобранные
        настройки»: так кнопка сброса возвращает голос к паспорту движка.
        """
        engine = check_engine(engine) if engine else None
        with self._lock:
            voices = self._read()
            voice = next((v for v in voices if v.id == voice_id), None)
            if voice is None:
                raise KeyError(voice_id)
            if engine is not None:
                voice.engine = engine
            if engine_params is not None:
                voice.engine_params = normalize_engine_params(voice.engine, engine_params)
            # Ручки прошлого движка к новому не относятся (у XTTS нет nfe_step),
            # поэтому при смене движка набор значений пересобирается заново.
            elif engine is not None:
                voice.engine_params = {}
            if preset is not None:
                voice.preset = _normalize_preset(preset)
            self._write(voices)
        logger.info(
            "Голос %s: движок %s, пресет %s", voice_id, voice.engine, voice.preset or "пуст"
        )
        return voice

    # -- референс-профили (UPDATE 2 §8, §37, §38) -------------------------------
    def add_reference(
        self,
        voice_id: str,
        *,
        emotion: str,
        audio_filename: str,
        audio_bytes: bytes,
        ref_text: str = "",
        label: str = "",
        verify_ref_text: bool = True,
        source_record_phrase_id: str = "",
        is_default: bool = False,
        enabled: bool = True,
        enabled_for_auto: bool = False,
    ) -> ReferenceProfile:
        """Добавляет интонационный референс существующему голосу.

        Тот же путь, что и у основного референса: файл пишется в `voices/`, а
        расшифровка либо сверяется с записью, либо берётся из формы. Качество
        проставляется здесь же — резолвер не должен выбирать профиль, про который
        известно, что он не совпадает с записью (§38).

        `verify_ref_text=False` — запись сделана по показанной фразе (мастер из
        §16): текст известен точно, и это отдельный статус расшифровки, а не
        «совпало» — он честно говорит, что распознавания не было.

        `source_record_phrase_id` делает запись повторяемой: одна фраза
        `RECORD_PHRASES` — один профиль (§16), поэтому повторная запись той же фразы
        **заменяет** прежний профиль, а не добавляет второй. Иначе в голосе копились
        бы дубликаты одной интонации, а резолвер выбирал бы из них самый ранний — и
        «записал заново» не давало бы никакого эффекта.

        `enabled_for_auto` по умолчанию `False` (§35): свежая запись ещё не
        подтверждена ни benchmark'ом, ни прослушиванием, поэтому в автоматическую
        маршрутизацию не попадает. Включить флаг можно вручную — тогда профиль
        станет кандидатом автоматического выбора референса.
        """
        if str(emotion or "").strip().upper() == emotions.EMOTION_AUTO:
            raise ValueError("AUTO — не эмоция референса: выберите конкретную")
        # Ключом профиля может быть только значение набора §10: записать «испуг»
        # отдельным профилем нельзя — под него нет фразы RECORD_PHRASES. Отказ, а не
        # подстановка NEUTRAL: молчаливое переименование «испуга» в нейтральный
        # профиль поставило бы под видом основного референса чужую интонацию (§17).
        if not emotions.is_profile_key(emotion):
            raise ValueError(
                f"Интонация «{str(emotion or '').strip()}» не записывается профилем: "
                f"доступны {', '.join(emotions.PROFILE_KEYS)}"
            )
        emotion = emotions.normalize_profile_key(emotion)
        suffix = Path(audio_filename or "").suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"Формат {suffix or 'без расширения'} не поддерживается")
        if not audio_bytes:
            raise ValueError("Пустой файл записи")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError("Файл слишком большой")
        # Замер длительности — до захвата блокировки: это декодирование, и держать
        # на нём список голосов незачем.
        duration = _duration_sec(audio_bytes, suffix)
        with self._lock:
            voices = self._read()
            voice = next((v for v in voices if v.id == voice_id), None)
            if voice is None:
                raise KeyError(voice_id)
            profile_id = uuid.uuid4().hex[:12]
            filename = f"{voice_id}-{profile_id}{suffix}"
            path = config.VOICES_DIR / filename
            path.write_bytes(audio_bytes)
            try:
                resolved_text, warning = self._resolve_ref_text(path, ref_text, verify_ref_text)
            except ValueError:
                path.unlink(missing_ok=True)
                raise
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            profile = ReferenceProfile(
                id=profile_id,
                voice_id=voice_id,
                emotion=emotion,
                label=label.strip() or emotions.EMOTION_TITLES.get(emotion, emotion),
                audio_file=filename,
                ref_text=resolved_text,
                source="record",
                quality_status=QUALITY_WARNING if warning else QUALITY_OK,
                quality_note=warning or "",
                transcription_status=(
                    TRANSCRIPTION_FROM_FORM
                    if not verify_ref_text
                    else TRANSCRIPTION_WARNING if warning else TRANSCRIPTION_OK
                ),
                duration_sec=duration,
                engine_compatibility=[voice.engine],
                source_record_phrase_id=str(source_record_phrase_id or ""),
                is_default=bool(is_default),
                enabled=bool(enabled),
                enabled_for_auto=bool(enabled_for_auto),
                created_at=now,
                updated_at=now,
            )
            # Профили хранятся отдельным списком; старый одиночный референс
            # остаётся нетронутым и читается как NEUTRAL.
            replaced: ReferenceProfile | None = None
            phrase_id = str(source_record_phrase_id or "")
            if phrase_id:
                replaced = next(
                    (
                        item
                        for item in voice.profiles
                        if str(item.source_record_phrase_id or "") == phrase_id
                    ),
                    None,
                )
                if replaced is not None:
                    voice.profiles = [
                        item for item in voice.profiles if item.id != replaced.id
                    ]
            voice.profiles = [*voice.profiles, profile]
            self._write(voices)
        if replaced is not None:
            # Файл прежней записи удаляем после записи списка: если удаление не
            # удастся, в `voices.json` уже не будет профиля, который на него ссылается.
            try:
                replaced.audio_path.unlink(missing_ok=True)
            except OSError as exc:  # файл мог быть удалён руками
                logger.warning("Прежний файл записи %s не удалён: %s", replaced.audio_file, exc)
            logger.info(
                "Голос %s: референс %s заменён записью фразы %s",
                voice_id, replaced.id, phrase_id,
            )
        logger.info("Голос %s: добавлен референс %s (%s)", voice_id, profile.id, emotion)
        return profile

    def update_reference(
        self,
        voice_id: str,
        profile_id: str,
        *,
        emotion: str | None = None,
        label: str | None = None,
        is_default: bool | None = None,
        enabled: bool | None = None,
        enabled_for_auto: bool | None = None,
    ) -> ReferenceProfile:
        """Правит профиль, не перезаписывая запись.

        `enabled` — это выключатель, а не удаление (§35): запись остаётся в голосе
        и в интерфейсе, но автоматика её больше не выбирает.

        `enabled_for_auto` — право на **автоматический** выбор по интонации (§35).
        Ставится вручную, когда профиль подтверждён benchmark'ом (§33) и
        прослушиванием (§34). На ручной выбор реплики флаг не влияет.
        """
        with self._lock:
            voices = self._read()
            voice = next((v for v in voices if v.id == voice_id), None)
            if voice is None:
                raise KeyError(voice_id)
            profile = next((item for item in voice.profiles if item.id == profile_id), None)
            if profile is None:
                raise KeyError(profile_id)
            changed = False
            if emotion is not None:
                if str(emotion).strip().upper() == emotions.EMOTION_AUTO:
                    raise ValueError("AUTO — не эмоция референса: выберите конкретную")
                chosen = emotions.normalize_profile_key(emotion)
                changed = chosen != profile.emotion
                profile.emotion = chosen
            if label is not None:
                new_label = label.strip() or profile.label
                changed = changed or new_label != profile.label
                profile.label = new_label
            if enabled is not None and bool(enabled) != profile.enabled:
                profile.enabled = bool(enabled)
                changed = True
            if is_default is not None and bool(is_default) != profile.is_default:
                profile.is_default = bool(is_default)
                changed = True
            if enabled_for_auto is not None and bool(enabled_for_auto) != profile.enabled_for_auto:
                profile.enabled_for_auto = bool(enabled_for_auto)
                changed = True
            if changed:
                profile.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write(voices)
        logger.info("Голос %s: референс %s обновлён (%s)", voice_id, profile_id, profile.emotion)
        return profile

    def delete_reference(self, voice_id: str, profile_id: str) -> bool:
        """Удаляет профиль вместе с его файлом. Основной референс не удаляется."""
        with self._lock:
            voices = self._read()
            voice = next((v for v in voices if v.id == voice_id), None)
            if voice is None:
                raise KeyError(voice_id)
            profile = next((item for item in voice.profiles if item.id == profile_id), None)
            if profile is None:
                return False
            voice.profiles = [item for item in voice.profiles if item.id != profile_id]
            self._write(voices)
            try:
                profile.audio_path.unlink(missing_ok=True)
            except OSError as exc:  # файл мог быть удалён руками
                logger.warning("Файл референса %s не удалён: %s", profile.audio_file, exc)
        logger.info("Голос %s: референс %s удалён", voice_id, profile_id)
        return True

    def inspect_existing(self) -> int:
        """Считает F0 и метки для голосов, загруженных до появления проверки."""
        with self._lock:
            voices = self._read()
            changed = 0
            for voice in voices:
                # У голоса без референса проверять нечего: `audio_path` у него —
                # сам каталог `voices/`, и чтение его байтов каждый запуск
                # заканчивалось бы предупреждением в логе.
                if voice.f0_hz is not None or not voice.audio_file or not voice.audio_path.exists():
                    continue
                try:
                    audio_bytes = voice.audio_path.read_bytes()
                except OSError as exc:
                    logger.warning("Не удалось прочитать %s: %s", voice.audio_file, exc)
                    continue
                checks = _inspect_reference(audio_bytes, voice.audio_path.suffix, voice.name, voice.gender)
                voice.name = checks["name"]
                voice.f0_hz = checks["f0_hz"]
                voice.gender_warning = checks["gender_warning"]
                voice.is_demo = checks["is_demo"]
                voice.demo_source = checks["demo_source"]
                changed += 1
            if changed:
                self._write(voices)
        if changed:
            logger.info("Проверил %s голосов: F0 и метки обновлены", changed)
        return changed

    def delete(self, voice_id: str) -> bool:
        """Удаляет голос вместе с его файлом; неизвестный id — `False`."""
        with self._lock:
            voices = self._read()
            keep = [v for v in voices if v.id != voice_id]
            if len(keep) == len(voices):
                return False
            removed = next(v for v in voices if v.id == voice_id)
            # У голоса без референса имя файла пусто, и `VOICES_DIR / ""` — сам
            # каталог `voices/`: удалять там нечего.
            if removed.audio_file:
                try:
                    removed.audio_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Не удалось удалить файл голоса %s: %s", removed.audio_file, exc)
            self._write(keep)
        logger.info("Удалён голос %s", voice_id)
        return True


_store = VoicesStore()


def get_store() -> VoicesStore:
    return _store
