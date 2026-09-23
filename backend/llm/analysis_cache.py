"""Кеш разборов LLM: когда разбор ещё про этот вход, а когда уже устарел.

Разбор дорогой (секунды на реплику и гигабайты памяти под модель), поэтому
одинаковый неизменившийся текст не должен гоняться через LLM повторно. Но кеш здесь
опаснее обычного: если принять старый разбор за новый, пользователь увидит
предложение, сделанное по другому тексту, другой модели или другому prompt'у.

Поэтому действительность разбора — это **сравнение входа целиком**, а не время
создания:

```text
source_text_hash  — изменился текст реплики
context_hash      — изменились соседи, по которым модель читала контекст
dictionary_hash   — изменился словарь, влияющий на реплику
model_digest      — та же модель пересобрана или выбрана другая
prompt_version    — изменился prompt (условия задачи)
schema_version    — изменился контракт ответа
```

Совпало всё — кеш валиден. Расхождение — разбор помечается `STALE` и не отдаётся
как актуальный. Хеши считаются по фактическому входу, поэтому «изменилась одна
реплика» автоматически инвалидирует и её соседей: у соседа в `context_hash` лежит
изменившийся текст.

Модуль не знает про FastAPI и про то, когда именно звать модель: он отвечает только
на вопрос «есть ли валидный разбор для такого входа».
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import schemas as s
from .analyzer import STATUS_STALE, ReplicaAnalysis, context_hash, text_hash

logger = logging.getLogger("tts.llm.cache")


def rule_fields(rule: object) -> dict:
    """Канонические поля правила словаря: правило приходит объектом или словарём.

    Возвращаются только поля, влияющие на чтение. Заметка и время правки остаются
    за бортом: они меняют запись, но не произношение, и не должны инвалидировать
    готовый разбор (§18).
    """
    if isinstance(rule, Mapping):
        get = rule.get
    else:

        def get(name: str, default: object = None) -> object:
            return getattr(rule, name, default)

    return {
        "source": str(get("source") or ""),
        "target": str(get("target") or ""),
        "case_sensitive": bool(get("case_sensitive", False)),
        "whole_word": bool(get("whole_word", True)),
        "enabled": bool(get("enabled", True)),
        "project_id": get("project_id") or "",
    }


def dictionary_hash(rules: Iterable[object]) -> str:
    """Хеш словаря, влияющего на реплику: правила проекта и глобальные.

    Учитываются только поля, которые меняют чтение (`source`, `target`,
    `case_sensitive`, `whole_word`, `enabled`): заметка или время правки на разбор
    не влияют и не должны его инвалидировать.
    """
    payload = [rule_fields(rule) for rule in rules]
    payload.sort(
        key=lambda item: (
            str(item.get("project_id") or ""),
            str(item.get("source") or ""),
            str(item.get("target") or ""),
        )
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AnalysisKey:
    """Вход, от которого зависит разбор. Совпал целиком — кеш действителен."""

    source_text_hash: str
    context_hash: str
    dictionary_hash: str
    model_digest: str
    prompt_version: str
    schema_version: str

    def to_dict(self) -> dict:
        return {
            "source_text_hash": self.source_text_hash,
            "context_hash": self.context_hash,
            "dictionary_hash": self.dictionary_hash,
            "model_digest": self.model_digest,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass
class CacheLookup:
    """Результат поиска в кеше: сам разбор и почему он не подошёл."""

    analysis: ReplicaAnalysis | None = None
    reason: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.analysis is not None


class AnalysisCache:
    """Кеш разборов поверх репозитория `llm_analyses`.

    Репозиторий передаётся снаружи вместе с соединением: класс не открывает
    транзакции сам, потому что вызывающий код (анализ проекта) уже работает в своей
    транзакции и не должен получать вторую.
    """

    def __init__(self, repository) -> None:
        self._repo = repository

    # -- поиск ------------------------------------------------------------------
    def lookup(
        self, project_id: str, replica_index: int, key: AnalysisKey
    ) -> CacheLookup:
        """Отдаёт кешированный разбор, только если ключ совпал целиком.

        Любое расхождение или негодный статус — промах с причиной в `reason`, а
        не частично применённый разбор.
        """
        row = self._repo.get(project_id, replica_index)
        if row is None:
            return CacheLookup(reason="нет сохранённого разбора")
        if row["status"] == STATUS_STALE:
            return CacheLookup(reason="разбор помечен устаревшим")
        if row["status"] not in {"READY", "NEEDS_REVIEW"}:
            return CacheLookup(reason=f"разбор в состоянии {row['status']}")
        mismatches = [
            name
            for name in key.to_dict()
            if str(row.get(name) or "") != str(key.to_dict()[name] or "")
        ]
        if mismatches:
            return CacheLookup(reason=", ".join(mismatches))
        try:
            payload = json.loads(row["analysis_json"] or "{}")
        except ValueError:
            return CacheLookup(reason="сохранённый разбор повреждён")
        return CacheLookup(analysis=analysis_from_dict(payload))

    # -- запись -----------------------------------------------------------------
    def save(self, project_id: str, replica_index: int, analysis: ReplicaAnalysis) -> dict:
        """Сохраняет разбор вместе с ключом, по которому его потом найдут."""
        key = key_for(analysis)
        return self._repo.upsert(
            analysis_id=uuid.uuid4().hex,
            project_id=project_id,
            replica_id=analysis.replica_id,
            replica_index=replica_index,
            analysis_json=analysis.to_json(),
            status=analysis.status,
            error=analysis.error,
            model_tag=analysis.model_tag,
            **key.to_dict(),
        )

    def mark_stale(
        self, project_id: str, replica_indexes: Sequence[int] | None = None
    ) -> int:
        """Помечает устаревшими реплики и их контекстный район.

        Соседи попадают сюда, потому что их разбор мог опираться на изменившуюся
        реплику; хеш контекста поймал бы это и при поиске, но явная пометка делает
        состояние видимым в интерфейсе до следующего анализа.
        """
        if replica_indexes is None:
            return self._repo.mark_stale(project_id)
        indexes: set[int] = set()
        for index in replica_indexes:
            for offset in range(-CONTEXT_NEIGHBOURHOOD, CONTEXT_NEIGHBOURHOOD + 1):
                if index + offset >= 0:
                    indexes.add(index + offset)
        return self._repo.mark_stale(project_id, sorted(indexes))


# Насколько широко помечать соседей при явной инвалидации: два влево и два вправо —
# ровно та политика контекста, по которой строится запрос к модели.
CONTEXT_NEIGHBOURHOOD = 2


def key_for(analysis: ReplicaAnalysis) -> AnalysisKey:
    return AnalysisKey(
        source_text_hash=analysis.source_hash,
        context_hash=analysis.context_hash,
        dictionary_hash=analysis.dictionary_hash,
        model_digest=analysis.model_digest,
        prompt_version=analysis.prompt_version,
        schema_version=analysis.schema_version,
    )


def analysis_from_dict(payload: dict[str, Any]) -> ReplicaAnalysis:
    """Восстанавливает разбор из сохранённого JSON.

    Аннотации проходят через ту же схему, что и ответ модели: доверять содержимому
    базы так же нельзя, как ответу LLM (запись могла остаться от старой версии кода).
    """
    items: list[s.Annotation] = []
    for raw in payload.get("items") or []:
        if isinstance(raw, dict):
            items.append(s.Annotation.from_dict(raw))
    utterance_raw = payload.get("utterance") or {}
    return ReplicaAnalysis(
        replica_id=int(payload.get("replica_id") or 0),
        status=str(payload.get("status") or "READY"),
        items=tuple(items),
        dropped=tuple(payload.get("dropped") or ()),
        utterance=s.UtteranceHint.from_dict(utterance_raw if isinstance(utterance_raw, dict) else {}),
        model_tag=str(payload.get("model_tag") or ""),
        model_digest=str(payload.get("model_digest") or ""),
        prompt_version=str(payload.get("prompt_version") or ""),
        schema_version=str(payload.get("schema_version") or s.SCHEMA_VERSION),
        source_hash=str(payload.get("source_hash") or ""),
        context_hash=str(payload.get("context_hash") or ""),
        dictionary_hash=str(payload.get("dictionary_hash") or ""),
        seconds=float(payload.get("seconds") or 0.0),
        error=str(payload.get("error") or ""),
        repaired=bool(payload.get("repaired", False)),
        markup_stripped=int(payload.get("markup_stripped") or 0),
    )


__all__ = [
    "CONTEXT_NEIGHBOURHOOD",
    "AnalysisCache",
    "AnalysisKey",
    "CacheLookup",
    "analysis_from_dict",
    "context_hash",
    "dictionary_hash",
    "key_for",
    "text_hash",
]
