"""Выбор референса голоса под эмоцию реплики (UPDATE 2 §10, §38).

Единственное место, которое отвечает на вопрос «какой файл и какая расшифровка
уйдут в движок». Раньше это решал каждый вызов по-своему (`voice.audio_path` и
`voice.ref_text`), и появление эмоциональных референсов превратило бы такой
прямой доступ в разбросанный по коду выбор файла. Здесь правило одно:

1. запрошенная эмоция — если у **этого же** голоса есть подходящий профиль,
   он прошёл проверку качества и совместим с движком;
2. иначе NEUTRAL того же голоса — с пометкой `fallback_used`, но без отказа:
   отсутствие эмоционального референса не блокирует синтез (§9);
3. иначе — ошибка, потому что синтезировать без референса нельзя.

Гарантия «референс только того же голоса» обеспечивается структурно: профили
хранятся внутри записи голоса, и резолвер физически не может обратиться к чужой
записи. Проверка `voice_id` в результате — страховка от ошибки в вызывающем коде,
а не основной механизм.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import emotions
from .voices_store import ReferenceProfile, Voice

logger = logging.getLogger(__name__)


class ReferenceUnavailableError(ValueError):
    """Референс не найден: синтез этой реплики невозможен, и причина называется."""


@dataclass(frozen=True)
class ResolvedReference:
    """Что уйдёт в движок: файл, расшифровка и как это было выбрано."""

    voice_id: str
    requested_emotion: str
    resolved_emotion: str
    profile_id: str
    audio_path: Path
    ref_text: str
    fallback_used: bool = False
    reason: str = ""
    quality_status: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "requested_emotion": self.requested_emotion,
            "resolved_emotion": self.resolved_emotion,
            "reference_profile_id": self.profile_id,
            "reference_emotion": self.resolved_emotion,
            "reference_fallback_used": self.fallback_used,
            "reference_quality": self.quality_status,
            "reference_reason": self.reason,
        }


def _usable(
    profile: ReferenceProfile, engine_id: str, *, check_engine: bool = True
) -> tuple[bool, str]:
    """Годится ли профиль для этого движка прямо сейчас.

    Причины отказа называются словами: пользователь должен видеть в интерфейсе,
    почему его эмоциональный референс не взяли, а не догадываться.

    `check_engine=False` — для нейтрального референса голоса: он не «выбран под
    движок», он и есть голос, и отказ из-за несовпадения идентификатора движка
    запретил бы синтез там, где раньше всё работало (сравнение движков одним
    голосом, смена движка голоса). Эмоциональный профиль, наоборот, проверяется:
    его просодия измерялась на конкретном движке.
    """
    if not profile.audio_file:
        return False, "у профиля нет файла"
    if not profile.audio_path.exists():
        return False, "файл референса отсутствует"
    if profile.quality_status == "warning":
        # Расшифровка расходится с записью: такой референс нельзя предпочитать
        # нейтральному только из-за подходящей эмоции (§38).
        return False, profile.quality_note or "референс не прошёл проверку"
    if not check_engine:
        return True, ""
    compatibility = [str(item) for item in (profile.engine_compatibility or [])]
    if compatibility and engine_id and engine_id not in compatibility:
        return False, f"профиль не проверен для движка {engine_id}"
    return True, ""


def resolve_reference(
    voice: Voice,
    engine_id: str,
    emotion: str = emotions.EMOTION_NEUTRAL,
    *,
    profile_id: str = "",
) -> ResolvedReference:
    """Референс для реплики: эмоциональный, если он есть, иначе нейтральный.

    `profile_id` — явный выбор пользователя (например, из карточки реплики): он
    сильнее и эмоции, и автоматики, но по-прежнему обязан принадлежать этому
    голосу. Неизвестный или непригодный профиль не подменяется молча — о нём
    пишется в лог, а решение принимается обычным порядком.
    """
    requested = emotions.normalize_emotion(emotion)
    profiles = voice.reference_profiles()
    if not profiles:
        raise ReferenceUnavailableError(
            f"У голоса «{voice.name}» нет ни одного референса — синтез невозможен"
        )

    if profile_id:
        explicit = voice.profile(profile_id)
        if explicit is None:
            logger.warning(
                "Голос %s: запрошенный профиль %s не найден — выбираю по эмоции",
                voice.id, profile_id,
            )
        else:
            ok, why = _usable(explicit, engine_id)
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
                )
            logger.warning(
                "Голос %s: профиль %s не годится (%s) — выбираю по эмоции",
                voice.id, explicit.id, why,
            )

    reason = ""
    if requested != emotions.EMOTION_NEUTRAL:
        candidates = [item for item in profiles if item.emotion == requested]
        if not candidates:
            reason = f"референса для эмоции {requested} нет"
        else:
            # Сначала проверенные профили, потом непроверенные: «unknown» — это
            # запись без сверки, и предпочитать её проверенной нельзя.
            candidates.sort(key=lambda item: (item.quality_status != "ok", item.created_at))
            for candidate in candidates:
                ok, why = _usable(candidate, engine_id)
                if ok:
                    return ResolvedReference(
                        voice_id=voice.id,
                        requested_emotion=requested,
                        resolved_emotion=candidate.emotion,
                        profile_id=candidate.id,
                        audio_path=candidate.audio_path,
                        ref_text=candidate.ref_text,
                        reason="эмоциональный профиль",
                        quality_status=candidate.quality_status,
                    )
                reason = reason or why

    neutral = next(
        (
            item
            for item in sorted(profiles, key=lambda item: (item.quality_status != "ok", item.created_at))
            if item.emotion == emotions.EMOTION_NEUTRAL
        ),
        None,
    )
    if neutral is not None:
        # Нейтральный референс — сам голос: совместимость с движком у него не
        # проверяется (см. `_usable`).
        ok, why = _usable(neutral, engine_id, check_engine=False)
        if ok:
            return ResolvedReference(
                voice_id=voice.id,
                requested_emotion=requested,
                resolved_emotion=emotions.EMOTION_NEUTRAL,
                profile_id=neutral.id,
                audio_path=neutral.audio_path,
                ref_text=neutral.ref_text,
                fallback_used=requested != emotions.EMOTION_NEUTRAL,
                reason=reason or "нейтральный референс",
                quality_status=neutral.quality_status,
            )
        reason = reason or why

    raise ReferenceUnavailableError(
        f"Голос «{voice.name}»: {reason or 'нет пригодного референса'}"
    )


def reference_status(voice: Voice, engine_id: str, emotion: str) -> dict:
    """Что произойдёт при выборе этой эмоции — для интерфейса и диагностики.

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
    "ReferenceUnavailableError",
    "ResolvedReference",
    "reference_status",
    "resolve_reference",
]
