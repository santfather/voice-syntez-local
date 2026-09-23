"""Обязательная подготовка диалога: стадии текста по репликам и состояние проекта.

Зачем отдельный модуль. До этого текст готовился к синтезу только в момент самого
синтеза: пользователь нажимал «Сгенерировать аудио», и уже внутри задачи выяснялось,
что в реплике есть неоднозначное «е/ё» или слово с плавающим ударением. Увидеть это
заранее было нечем, а сравнить два прогона — не с чем: подготовленный текст нигде не
сохранялся. Здесь подготовка отделена от синтеза и **фиксируется**: у каждой реплики
остаются все стадии и итоговый `final_text`, по которому потом и идёт рендер.

Почему не второй pipeline. Стадии считает ровно та же функция, что показывает панель
«Что услышит модель» (`audio_pipeline.preview_text`), а кандидатов в словарь ищет та же
`pronunciation_suggest.build_suggestions`, только ей передаются уже посчитанные стадии,
чтобы акцентуация не выполнялась дважды на одну реплику. Своей реализации нормализации,
«ё», словаря или ударений здесь нет и быть не должно: разошедшись один раз, она начнёт
показывать одно, а отправлять в модель другое.

Кандидаты в словарь — единственная причина, по которой проект не становится `ready`:
неоднозначное «е/ё» и омограф ударения требуют решения человека, а решать их молча
за него нельзя. Всё остальное (нормализация, бесспорная «ё», словарь, ударения)
происходит автоматически.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import config
from .audio_pipeline import (
    SpeakerSettings,
    preview_text,
    tuning_for,
    tuning_parameters,
)
from .dialogue_parser import Replica
from .engines.base import ENGINE_INFOS
from .pronunciation_suggest import build_suggestions
from .voices_store import Voice

logger = logging.getLogger(__name__)

# Словарь состояний живёт в `config` рядом с остальными статусами проекта и
# реплики: он же нужен репозиториям и API, а второй набор констант разошёлся бы с
# первым на первой же правке.
STATUS_RAW = config.PROJECT_ANALYSIS_RAW
STATUS_ANALYZING = config.PROJECT_ANALYSIS_ANALYZING
STATUS_NEEDS_REVIEW = config.PROJECT_ANALYSIS_NEEDS_REVIEW
STATUS_READY = config.PROJECT_ANALYSIS_READY
STATUS_ERROR = config.PROJECT_ANALYSIS_ERROR
PROJECT_STATUSES = config.PROJECT_ANALYSIS_STATUSES

REPLICA_PENDING = config.REPLICA_ANALYSIS_PENDING
REPLICA_DONE = config.REPLICA_ANALYSIS_DONE
REPLICA_ERROR = config.REPLICA_ANALYSIS_ERROR

# Сколько кандидатов показывать на реплику. Меньше, чем в общем списке: реплика
# диалога короткая, и два десятка предложений на неё — уже шум, а не подсказка.
CANDIDATES_PER_REPLICA = 10


@dataclass(frozen=True)
class ReplicaPreparation:
    """Результат подготовки одной реплики: стадии, находки и контекст движка."""

    index: int
    source_text: str
    normalized_text: str
    yo_text: str
    dictionary_text: str
    accentized_text: str
    final_text: str
    dictionary_matches: list[dict]
    pronunciation_candidates: list[dict]
    supports_accents: bool
    auto_accent: bool
    effective_engine_params: dict
    engine: str
    voice_id: str
    seconds: float
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.pronunciation_candidates)

    def to_dict(self) -> dict:
        """Плоское представление подготовки реплики для API.

        `analysis_status` несёт итог: без него провалившаяся реплика записалась
        бы как `done` и проект считался бы готовым к синтезу.
        """
        return {
            "index": self.index,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "yo_text": self.yo_text,
            "dictionary_text": self.dictionary_text,
            "accentized_text": self.accentized_text,
            "final_text": self.final_text,
            "dictionary_matches": list(self.dictionary_matches),
            "pronunciation_candidates": list(self.pronunciation_candidates),
            "supports_accents": self.supports_accents,
            "auto_accent": self.auto_accent,
            "effective_engine_params": dict(self.effective_engine_params),
            "engine": self.engine,
            "voice_id": self.voice_id,
            "needs_review": self.needs_review,
            "error": self.error,
            # Состояние подготовки реплики едет вместе со стадиями: без него
            # провалившаяся реплика записалась бы как `done`, и проект с
            # неудавшимся анализом считался бы готовым к синтезу.
            "analysis_status": REPLICA_ERROR if self.error else REPLICA_DONE,
        }


def supports_accents_for(engine_id: str) -> bool:
    """Понимает ли движок «+»-ударения — по паспорту, без создания движка.

    Анализ обязан быть дешёвым (§14): поднимать веса ради ответа на этот вопрос
    нельзя, а паспорт лежит в `ENGINE_INFOS` и доступен всегда.
    """
    info = ENGINE_INFOS.get(engine_id)
    return bool(info.supports_accents) if info else False


def _failed_preparation(
    *, index: int, source_text: str, engine: str, voice_id: str, error: str
) -> ReplicaPreparation:
    """Реплика, которую подготовить не удалось: пустые стадии и причина.

    Отдельная функция, потому что «не удалось» — это обычный исход подготовки
    (нет голоса, упала акцентуация), и он не должен выражаться исключением:
    одна такая реплика не отменяет анализ остальных.
    """
    return ReplicaPreparation(
        index=index,
        source_text=source_text,
        normalized_text="",
        yo_text="",
        dictionary_text="",
        accentized_text="",
        final_text="",
        dictionary_matches=[],
        pronunciation_candidates=[],
        supports_accents=supports_accents_for(engine),
        auto_accent=False,
        effective_engine_params={},
        engine=engine,
        voice_id=voice_id,
        seconds=0.0,
        error=error,
    )


def prepare_replica(
    replica: Replica,
    speaker: SpeakerSettings,
    voice: Voice | None,
    *,
    index: int = 0,
    rules=None,
    auto_accent: bool = True,
    candidates_limit: int = CANDIDATES_PER_REPLICA,
) -> ReplicaPreparation:
    """Готовит одну реплику: нормализация → «ё» → словарь → ударения → `final_text`.

    Параметры движка и спикера считаются теми же функциями, что при синтезе
    (`tuning_for` и `tuning_parameters`), поэтому сохранённые `final_text` и
    `effective_engine_params` описывают ровно тот кусок, который потом уйдёт в
    модель. Ошибка подготовки не выбрасывается наружу: у реплики есть своё поле
    `error`, и одна неудачная реплика не должна отменять анализ остальных.
    """
    if voice is None:
        return _failed_preparation(
            index=index,
            source_text=replica.text,
            engine="",
            voice_id="",
            error="не назначен голос: выберите голос спикеру или этой реплике",
        )
    started = time.monotonic()
    settings = tuning_for(voice, speaker, replica.overrides)
    engine = voice.engine
    accents = supports_accents_for(engine)
    try:
        stages = preview_text(
            replica.text,
            supports_accents=accents,
            auto_accent=auto_accent,
            rules=rules,
        )
        # Стадии передаются в поиск кандидатов: иначе акцентуация считалась бы
        # второй раз на ту же реплику (см. `build_suggestions`).
        suggestions = build_suggestions(
            replica.text,
            rules=rules,
            stages=stages,
            supports_accents=accents,
            auto_accent=auto_accent,
            limit=candidates_limit,
        )
        error = None
    except Exception as exc:  # noqa: BLE001 — реплика сообщает об ошибке сама
        logger.warning("Реплика %s: подготовка не удалась (%s)", index + 1, exc)
        stages = None
        suggestions = None
        error = f"{type(exc).__name__}: {exc}"

    empty = error is not None
    return ReplicaPreparation(
        index=index,
        source_text=replica.text,
        normalized_text="" if empty else stages.normalized,
        yo_text="" if empty else stages.yo,
        dictionary_text="" if empty else stages.dictionary,
        accentized_text="" if empty else stages.accentized,
        final_text="" if empty else stages.final,
        dictionary_matches=[] if empty else list(stages.matches),
        pronunciation_candidates=[] if empty else [c.to_dict() for c in suggestions.candidates],
        supports_accents=accents,
        auto_accent=bool(auto_accent),
        effective_engine_params=tuning_parameters(settings),
        engine=engine,
        voice_id=voice.id,
        seconds=round(time.monotonic() - started, 4),
        error=error,
    )


@dataclass
class AnalysisSummary:
    """Итог анализа: состояние проекта, реплики и общие счётчики для интерфейса."""

    status: str
    replicas: list[ReplicaPreparation] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0

    @property
    def replicas_total(self) -> int:
        return len(self.replicas)

    @property
    def replicas_analyzed(self) -> int:
        return sum(1 for item in self.replicas if item.error is None)

    @property
    def candidates(self) -> list[dict]:
        """Кандидаты с номером реплики — интерфейсу нужно, к чему они относятся."""
        return [
            {**candidate, "replica_index": item.index}
            for item in self.replicas
            for candidate in item.pronunciation_candidates
        ]

    @property
    def warnings(self) -> list[dict]:
        """Реплики, которые подготовились, но требуют внимания человека.

        Ошибки сюда не попадают: они в `errors`. Предупреждение — это «подготовлено,
        но проверьте»: например, у движка без поддержки ударений текст уйдёт без них,
        хотя тумблер включён.
        """
        result: list[dict] = []
        for item in self.replicas:
            if item.error is None and item.auto_accent and not item.supports_accents:
                result.append(
                    {
                        "replica_index": item.index,
                        "kind": "accents_unsupported",
                        "text": (
                            f"Реплика {item.index + 1}: движок «{item.engine}» не поддерживает "
                            "ударения — текст уйдёт без разметки"
                        ),
                    }
                )
        return result

    @property
    def errors(self) -> list[dict]:
        return [
            {"replica_index": item.index, "text": item.error}
            for item in self.replicas
            if item.error is not None
        ]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "replicas_total": self.replicas_total,
            "replicas_analyzed": self.replicas_analyzed,
            "seconds": round(self.seconds, 3),
            "candidates": self.candidates,
            "warnings": self.warnings,
            "errors": self.errors,
            "error": self.error,
        }


def summarize(preparations: list[ReplicaPreparation], *, seconds: float = 0.0) -> AnalysisSummary:
    """Определяет состояние проекта по итогам подготовки реплик.

    `needs_review` — есть неоднозначные слова, которые человек ещё не разрешил;
    `error` — ни одной реплики подготовить не удалось (это уже не «проверьте», а
    «почините»); иначе `ready`. Состояние считается по факту, а не выставляется
    вызывающим: иначе появился бы путь «поставить ready руками» в обход проверки.
    """
    if preparations and all(item.error is not None for item in preparations):
        status = STATUS_ERROR
    elif any(item.needs_review for item in preparations):
        status = STATUS_NEEDS_REVIEW
    else:
        status = STATUS_READY
    return AnalysisSummary(status=status, replicas=list(preparations), seconds=seconds)
