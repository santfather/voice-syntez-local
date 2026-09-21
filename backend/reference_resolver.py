"""Выбор референса голоса под интонацию реплики (UPDATE 2 §10, §38; UPDATE 3 §23–§28).

Единственное место, которое отвечает на вопрос «какой файл и какая расшифровка
уйдут в движок». Раньше это решал каждый вызов по-своему (`voice.audio_path` и
`voice.ref_text`), и появление интонационных референсов превратило бы такой
прямой доступ в разбросанный по коду выбор файла. Здесь правило одно:

1. явный выбор пользователя (`profile_id`) — если профиль принадлежит **этому**
   голосу, прошёл проверку качества и совместим с движком;
2. запрошенный профиль по цепочке отката (см. `FALLBACK_CHAINS`);
3. NEUTRAL того же голоса — с пометкой `fallback_used`, но без отказа:
   отсутствие интонационного референса не блокирует синтез (§9);
4. иначе — ошибка, потому что синтезировать без референса нельзя.

Шаги 2–3 — автоматика, и она берёт только подтверждённые профили: у профиля
подтверждение выражается флагом `enabled_for_auto` (§35). Явный выбор (шаг 1)
этим флагом не ограничен.

Гарантия «референс только того же голоса» обеспечивается структурно: профили
хранятся внутри записи голоса, и резолвер физически не может обратиться к чужой
записи. Проверка `voice_id` в результате — страховка от ошибки в вызывающем коде,
а не основной механизм.

LLM никогда не выбирает файл: она называет только намерение (`prosody_effective`),
а путь к аудио получается здесь — детерминированно и без модели (§23).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config, emotions
from .voices_store import ReferenceProfile, Voice

logger = logging.getLogger(__name__)

# Режимы отката (§26). `neutral_only` — консервативный производственный режим:
# §26 разрешает откат «на соседний профиль» только после Reference Prosody Transfer
# Benchmark, и по умолчанию его нет. `chain` включает таблицу цепочек целиком —
# он и проверяется benchmark'ом (Phase 11), и включается по факту подтверждения.
FALLBACK_MODE_NEUTRAL_ONLY = "neutral_only"
FALLBACK_MODE_CHAIN = "chain"
FALLBACK_MODES: tuple[str, ...] = (FALLBACK_MODE_NEUTRAL_ONLY, FALLBACK_MODE_CHAIN)

# Таблица отката: централизованная, документированная и покрытая тестами (§26).
# Первый элемент — сам профиль, последний обязан быть NEUTRAL: синтез без
# референса невозможен, а нейтральная запись есть у каждого голоса.
#
# Цепочки «дальше NEUTRAL» — это предложения, а не факты: §35 требует подтвердить
# каждую измеримым переносом просодии, поэтому по умолчанию они выключены
# (`config.PROSODY_FALLBACK_MODE`), а `SURPRISE`/`FEAR` не притворяются
# записанными профилями — под них записи нет вовсе (§17).
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    emotions.EMOTION_NEUTRAL: (emotions.EMOTION_NEUTRAL,),
    emotions.EMOTION_CALM: (emotions.EMOTION_CALM, emotions.EMOTION_NEUTRAL),
    emotions.EMOTION_QUESTION: (
        emotions.EMOTION_QUESTION,
        emotions.EMOTION_NEUTRAL_QUESTION,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_NEUTRAL_QUESTION: (
        emotions.EMOTION_NEUTRAL_QUESTION,
        emotions.EMOTION_QUESTION,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_EXCLAMATION: (
        emotions.EMOTION_EXCLAMATION,
        emotions.EMOTION_EXCITED,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_DELIGHT: (
        emotions.EMOTION_DELIGHT,
        emotions.EMOTION_EXCLAMATION,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_SAD_SYMPATHETIC: (
        emotions.EMOTION_SAD_SYMPATHETIC,
        emotions.EMOTION_CALM,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_IRONIC: (emotions.EMOTION_IRONIC, emotions.EMOTION_NEUTRAL),
    emotions.EMOTION_STRICT: (emotions.EMOTION_STRICT, emotions.EMOTION_NEUTRAL),
    emotions.EMOTION_ENUMERATION: (
        emotions.EMOTION_ENUMERATION,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_EXCITED: (
        emotions.EMOTION_EXCITED,
        emotions.EMOTION_EXCLAMATION,
        emotions.EMOTION_NEUTRAL,
    ),
    # Семантические эмоции без собственной записи (§17): начинать цепочку с себя
    # нечем, поэтому первым идёт ближайший записываемый профиль — и это видно как
    # откат, а не как «нашли испуг».
    emotions.EMOTION_SURPRISE: (
        emotions.EMOTION_EXCLAMATION,
        emotions.EMOTION_EXCITED,
        emotions.EMOTION_NEUTRAL,
    ),
    emotions.EMOTION_FEAR: (emotions.EMOTION_STRICT, emotions.EMOTION_NEUTRAL),
}


class ReferenceUnavailableError(ValueError):
    """Референс не найден: синтез этой реплики невозможен, и причина называется."""


def fallback_mode(value: str | None = None) -> str:
    """Действующий режим отката: аргумент → конфигурация → консервативный.

    Неизвестное значение не подменяется молча: оно логируется и трактуется как
    консервативное, потому что «непонятная настройка» не повод расширять выбор
    референсов за пределы подтверждённого benchmark'ом.
    """
    chosen = str(value if value is not None else config.PROSODY_FALLBACK_MODE).strip()
    if chosen in FALLBACK_MODES:
        return chosen
    logger.warning(
        "Неизвестный режим отката просодии «%s» — беру %s", chosen, FALLBACK_MODE_NEUTRAL_ONLY
    )
    return FALLBACK_MODE_NEUTRAL_ONLY


def fallback_chain(
    emotion: str, *, mode: str | None = None
) -> tuple[str, ...]:
    """Цепочка профилей для запрошенной эмоции — по порядку предпочтения (§26).

    Порядок детерминирован: ни времени, ни содержимого `voices.json` в нём нет,
    поэтому один и тот же голос и запрос всегда дают один и тот же ответ.
    """
    requested = emotions.normalize_emotion(emotion)
    if fallback_mode(mode) == FALLBACK_MODE_NEUTRAL_ONLY:
        chain = (requested, emotions.EMOTION_NEUTRAL)
    else:
        chain = FALLBACK_CHAINS.get(requested, (requested, emotions.EMOTION_NEUTRAL))
    # NEUTRAL обязан быть последним: без него цепочка не гарантирует синтез.
    result = [item for item in dict.fromkeys(chain) if item != emotions.EMOTION_NEUTRAL]
    result.append(emotions.EMOTION_NEUTRAL)
    return tuple(result)


@dataclass(frozen=True)
class ResolvedReference:
    """Что уйдёт в движок: файл, расшифровка и как это было выбрано.

    Это и есть `ResolvedProsody` из §24: отдельная сущность с тем же набором полей
    означала бы два ответа на один вопрос, и однажды они разошлись бы. Имена полей
    оставлены прежними (`requested_emotion`/`resolved_emotion`), а `to_dict()`
    отдаёт их же под именами §24 (`requested_profile`/`resolved_profile`).

    `dialogue_act` и `intensity` — диагностика, а не вход резолвера: LLM называет
    намерение, но выбирать файл по числу она не может (§23). Они попадают в
    результат, чтобы `diagnostics` и take-метаданные не собирали их второй раз.
    """

    voice_id: str
    requested_emotion: str
    resolved_emotion: str
    profile_id: str
    audio_path: Path
    ref_text: str
    fallback_used: bool = False
    reason: str = ""
    quality_status: str = "unknown"
    engine_id: str = ""
    fallback_reason: str = ""
    dialogue_act: str = ""
    intensity: float | None = None
    # Сколько профилей просмотрено до успеха: видно, что откат был, а не «повезло».
    candidates_tried: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "requested_emotion": self.requested_emotion,
            "resolved_emotion": self.resolved_emotion,
            # Имена UPDATE 3 §24: те же значения, второй сущности не заводим.
            "requested_profile": self.requested_emotion,
            "resolved_profile": self.resolved_emotion,
            "reference_profile_id": self.profile_id,
            "reference_emotion": self.resolved_emotion,
            "reference_fallback_used": self.fallback_used,
            "reference_fallback_reason": self.fallback_reason,
            "reference_quality": self.quality_status,
            "reference_reason": self.reason,
            "reference_audio": self.audio_path.name,
            "reference_text": self.ref_text,
            "engine_id": self.engine_id,
            "dialogue_act": self.dialogue_act,
            "intensity": self.intensity,
            "candidates_tried": list(self.candidates_tried),
        }


def _usable(
    profile: ReferenceProfile,
    engine_id: str,
    *,
    check_engine: bool = True,
    check_auto: bool = True,
) -> tuple[bool, str]:
    """Годится ли профиль для этого движка прямо сейчас.

    Причины отказа называются словами: пользователь должен видеть в интерфейсе,
    почему его интонационный референс не взяли, а не догадываться.

    `check_engine=False` — для нейтрального референса голоса: он не «выбран под
    движок», он и есть голос, и отказ из-за несовпадения идентификатора движка
    запретил бы синтез там, где раньше всё работало (сравнение движков одним
    голосом, смена движка голоса). Интонационный профиль, наоборот, проверяется:
    его просодия измерялась на конкретном движке (§28).

    `check_auto=False` — для явного выбора пользователя (`profile_id`): §35
    запрещает **автоматике** брать неподтверждённый профиль, но человек вправе
    выбрать его сам. Иначе экспериментальный профиль нельзя было бы даже
    прослушать, и «подтвердить benchmark'ом» стало бы нечем.
    """
    if not profile.enabled:
        return False, "профиль выключен"
    if not profile.audio_file:
        return False, "у профиля нет файла"
    if not profile.audio_path.exists():
        return False, "файл референса отсутствует"
    if profile.quality_status == "warning":
        # Расшифровка расходится с записью: такой референс нельзя предпочитать
        # нейтральному только из-за подходящей интонации (§22, §35).
        return False, profile.quality_note or "референс не прошёл проверку"
    if check_auto and not profile.enabled_for_auto:
        # §35: пользу профиля доказывает benchmark (§33) и прослушивание (§34).
        # Пока доказательства нет, автоматическая маршрутизация его не берёт —
        # но профиль остаётся доступным вручную (`check_auto=False`).
        return False, "профиль не подтверждён для автоматического выбора"
    if not check_engine:
        return True, ""
    if not engine_compatible(profile, engine_id):
        return False, f"профиль не проверен для движка {engine_id}"
    return True, ""


def engine_compatible(profile: ReferenceProfile, engine_id: str) -> bool:
    """Совместим ли профиль с движком (§28).

    Пустой список совместимости — «любой движок голоса»: старые и загруженные
    руками профили не должны становиться неиспользуемыми из-за отсутствия поля.
    Неизвестный движок (`engine_id` пуст) тоже не блокирует: отказ в синтезе
    из-за неопределённости был бы хуже, чем попытка тем движком, что выбрал голос.
    """
    compatibility = [str(item) for item in (profile.engine_compatibility or [])]
    if not compatibility or not engine_id:
        return True
    return engine_id in compatibility


def _candidates_for(profiles: list[ReferenceProfile], emotion: str) -> list[ReferenceProfile]:
    """Профили этой эмоции в порядке предпочтения.

    Сначала включённые и проверенные (`quality_status == "ok"`), затем профиль по
    умолчанию голоса, затем более ранний: «unknown» — это запись без сверки, и
    предпочитать её проверенной нельзя, но и терять её тоже.
    """
    found = [item for item in profiles if item.emotion == emotion]
    return sorted(
        found,
        key=lambda item: (
            item.quality_status != "ok",
            not item.is_default,
            item.created_at,
        ),
    )


def resolve_reference(
    voice: Voice,
    engine_id: str,
    emotion: str = emotions.EMOTION_NEUTRAL,
    *,
    profile_id: str = "",
    fallback: str | None = None,
) -> ResolvedReference:
    """Референс для реплики: подходящий профиль, иначе — по цепочке отката.

    `profile_id` — явный выбор пользователя (например, из карточки реплики): он
    сильнее и интонации, и автоматики, но по-прежнему обязан принадлежать этому
    голосу. Неизвестный или непригодный профиль не подменяется молча — о нём
    пишется в лог, а решение принимается обычным порядком.

    Автоматический выбор (шаги 2–3) берёт только подтверждённые профили (§35):
    у неподтверждённого `enabled_for_auto == False`, и он выпадает из отбора —
    ровно так профиль остаётся доступным вручную, но не становится production'ом
    сам по себе. NEUTRAL — сам голос, а не интонационный профиль, поэтому гейт
    на него не распространяется: иначе подтверждать пришлось бы и базовый
    референс, а синтез остался бы без референса вообще.

    `fallback` — режим отката (§26); `None` берёт `config.PROSODY_FALLBACK_MODE`.
    """
    requested = emotions.normalize_emotion(emotion)
    mode = fallback_mode(fallback)
    profiles = voice.reference_profiles()
    if not profiles:
        raise ReferenceUnavailableError(
            f"У голоса «{voice.name}» нет ни одного референса — синтез невозможен"
        )

    tried: list[str] = []
    if profile_id:
        explicit = voice.profile(profile_id)
        if explicit is None:
            logger.warning(
                "Голос %s: запрошенный профиль %s не найден — выбираю по интонации",
                voice.id, profile_id,
            )
        else:
            ok, why = _usable(explicit, engine_id, check_auto=False)
            if ok:
                return ResolvedReference(
                    voice_id=voice.id,
                    requested_emotion=requested,
                    resolved_emotion=explicit.emotion,
                    profile_id=explicit.id,
                    audio_path=explicit.audio_path,
                    ref_text=explicit.ref_text,
                    fallback_used=explicit.emotion != requested,
                    reason="выбран явно",
                    quality_status=explicit.quality_status,
                    engine_id=engine_id,
                    candidates_tried=(explicit.id,),
                )
            tried.append(explicit.id)
            logger.warning(
                "Голос %s: профиль %s не годится (%s) — выбираю по интонации",
                voice.id, explicit.id, why,
            )

    rejection = ""
    for position, candidate_emotion in enumerate(fallback_chain(requested, mode=mode)):
        # NEUTRAL — сам голос: он всегда доступен и движком не ограничен.
        is_neutral = candidate_emotion == emotions.EMOTION_NEUTRAL
        for candidate in _candidates_for(profiles, candidate_emotion):
            tried.append(candidate.id)
            ok, why = _usable(
                candidate, engine_id, check_engine=not is_neutral, check_auto=not is_neutral
            )
            if not ok:
                rejection = rejection or why
                continue
            fallback_used = position > 0 or candidate.emotion != requested
            reason = _reason(requested, candidate.emotion, position, mode, rejection)
            return ResolvedReference(
                voice_id=voice.id,
                requested_emotion=requested,
                resolved_emotion=candidate.emotion,
                profile_id=candidate.id,
                audio_path=candidate.audio_path,
                ref_text=candidate.ref_text,
                fallback_used=fallback_used,
                reason=reason,
                quality_status=candidate.quality_status,
                engine_id=engine_id,
                fallback_reason=reason if fallback_used else "",
                candidates_tried=tuple(tried),
            )

    raise ReferenceUnavailableError(
        f"Голос «{voice.name}»: "
        + (rejection or f"нет пригодного референса для {requested} и нейтрального")
    )


def _reason(
    requested: str, resolved: str, position: int, mode: str, rejection: str = ""
) -> str:
    """Почему выбран именно этот профиль — словами, для интерфейса и логов.

    `rejection` — причина, по которой более подходящий профиль не подошёл. Без неё
    откат выглядел бы как «профиля нет», хотя он есть и забракован: например, его
    расшифровка расходится с записью (§22, §35).
    """
    if position == 0 and resolved == requested:
        return "интонационный профиль"
    if resolved == emotions.EMOTION_NEUTRAL:
        base = f"референса для интонации {requested} нет — взят нейтральный"
    elif mode == FALLBACK_MODE_NEUTRAL_ONLY:
        base = "нейтральный референс"
    else:
        base = f"профиль {requested} отсутствует — использован ближайший {resolved}"
    return f"{base} ({rejection})" if rejection else base


def resolve_prosody(
    voice: Voice,
    engine_id: str,
    prosody_effective: str,
    dialogue_act: str = "",
    intensity: float | None = None,
    *,
    profile_id: str = "",
    fallback: str | None = None,
) -> ResolvedReference:
    """Точка входа просодической маршрутизации (UPDATE 3 §23, §24).

    Отдельная функция, а не второй резолвер: она добавляет к `resolve_reference`
    только диагностический контекст (`dialogue_act`, `intensity`) и повторяет тот
    же детерминированный порядок решений. LLM не участвует: она называет
    `prosody_effective`, а файл выбирает backend.
    """
    resolved = resolve_reference(
        voice, engine_id, prosody_effective, profile_id=profile_id, fallback=fallback
    )
    if not dialogue_act and intensity is None:
        return resolved
    return ResolvedReference(
        voice_id=resolved.voice_id,
        requested_emotion=resolved.requested_emotion,
        resolved_emotion=resolved.resolved_emotion,
        profile_id=resolved.profile_id,
        audio_path=resolved.audio_path,
        ref_text=resolved.ref_text,
        fallback_used=resolved.fallback_used,
        reason=resolved.reason,
        quality_status=resolved.quality_status,
        engine_id=resolved.engine_id,
        fallback_reason=resolved.fallback_reason,
        dialogue_act=str(dialogue_act or ""),
        intensity=intensity,
        candidates_tried=resolved.candidates_tried,
    )


def reference_status(voice: Voice, engine_id: str, emotion: str) -> dict:
    """Что произойдёт при выборе этой интонации — для интерфейса и диагностики.

    Ошибка здесь не поднимается: карточка голоса обязана показать «нет референса»
    вместо того, чтобы ломаться целиком из-за одной эмоции.
    """
    try:
        return {**resolve_reference(voice, engine_id, emotion).to_dict(), "available": True}
    except ReferenceUnavailableError as exc:
        return {
            "voice_id": voice.id,
            "requested_emotion": emotions.normalize_emotion(emotion),
            "resolved_emotion": "",
            "reference_profile_id": "",
            "reference_fallback_used": False,
            "available": False,
            "error": str(exc),
        }


__all__ = [
    "FALLBACK_CHAINS",
    "FALLBACK_MODES",
    "FALLBACK_MODE_CHAIN",
    "FALLBACK_MODE_NEUTRAL_ONLY",
    "ReferenceUnavailableError",
    "ResolvedReference",
    "engine_compatible",
    "fallback_chain",
    "fallback_mode",
    "reference_status",
    "resolve_prosody",
    "resolve_reference",
]
