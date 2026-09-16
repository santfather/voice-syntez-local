"""CRUD над voices.json и файлами референс-аудио."""

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import audio_analysis, config, transcribe
from .denoise import clean_bytes as clean_reference_bytes
from .engines.base import (
    ENGINE_F5,
    ENGINE_INFOS,
    default_engine_for_gender,
    normalize_engine_params,
)
from .settings_resolution import COMMON_FIELDS, INT_FIELDS

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif", ".webm"}
MAX_AUDIO_BYTES = 50 * 1024 * 1024
DEMO_PREFIX = "[demo]"


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

    @property
    def audio_path(self) -> Path:
        return config.VOICES_DIR / self.audio_file

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_audio"] = self.audio_path.exists()
        return data


_VOICE_FIELDS = {f.name for f in fields(Voice)}


class VoicesStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    # -- внутреннее ------------------------------------------------------------
    def _read(self) -> list[Voice]:
        if not config.VOICES_JSON.exists():
            return []
        try:
            payload = json.loads(config.VOICES_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("voices.json повреждён, читаю как пустой: %s", exc)
            return []
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
                voices.append(voice)
            except TypeError as exc:  # запись без обязательного поля — пропускаем
                logger.warning("Пропускаю некорректную запись в voices.json: %s", exc)
        return voices

    def _write(self, voices: list[Voice]) -> None:
        config.VOICES_JSON.write_text(
            json.dumps({"voices": [asdict(v) for v in voices]}, ensure_ascii=False, indent=2),
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

    # -- публичное API ---------------------------------------------------------
    def list(self) -> list[Voice]:
        with self._lock:
            return self._read()

    def get(self, voice_id: str) -> Voice | None:
        return next((v for v in self.list() if v.id == voice_id), None)

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
        suffix = Path(audio_filename).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"Неподдерживаемый формат аудио: {suffix or '(нет расширения)'}")
        if not audio_bytes:
            raise ValueError("Пустой файл аудио")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(f"Файл больше {MAX_AUDIO_BYTES // (1024 * 1024)} МБ")
        if not name.strip():
            raise ValueError("Не задано имя голоса")

        gender = gender if gender in ("male", "female", "other") else "other"
        engine = check_engine(engine) if engine else default_engine_for_gender(gender)
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

    def inspect_existing(self) -> int:
        """Считает F0 и метки для голосов, загруженных до появления проверки."""
        with self._lock:
            voices = self._read()
            changed = 0
            for voice in voices:
                if voice.f0_hz is not None or not voice.audio_path.exists():
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
        with self._lock:
            voices = self._read()
            keep = [v for v in voices if v.id != voice_id]
            if len(keep) == len(voices):
                return False
            removed = next(v for v in voices if v.id == voice_id)
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
