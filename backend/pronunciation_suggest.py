"""Кандидаты в словарь произношения: что стоит уточнить в тексте.

Словарь пополняется вручную, и это правильно: правило должно быть осознанным
решением. Плохо другое — искать неоднозначные слова каждый раз заново. Модуль
находит кандидатов в тексте, который реально уйдёт в модель, и отдаёт их
пользователю на подтверждение. Подтверждение и отклонение — обычные записи
`pronunciation_entries` через существующий API: отдельного хранилища нет.

Источники:

* ``yo_homograph`` — слова из ``yo_ambiguous.tsv.gz`` (все/всё, небо/нёбо,
  пчелы/пчёлы). Автоматически такие пары не меняются никогда, поэтому и
  предлагается выбор: вариант с «ё» или «оставить как есть».
* ``stress_homograph`` — омографы ударения из словаря ruaccent (за+мок/замо+к).
  Предлагается тот вариант, который реально выбрал RUAccent (виден в
  `accentized`), альтернативы — остальные. Сам словарь ruaccent **только
  читается** и в репозиторий не копируется: лицензия CC BY-NC-ND этого не
  позволяет.
* ``yo_low_confidence`` — предсказания модели ё-омографов там, где она сама
  отдаёт `score` и не уверена. Нет модели или нет score — источник пропускается.
* ``rare`` — консервативный fallback: длинные слова, которых нет ни в одном
  списке. Идёт последним и с пустой заменой: угадывать ударение за пользователя
  нельзя, он впишет его сам.

Слово, уже покрытое правилом словаря, не предлагается. Проверяются **и
выключенные** правила: отклонённое предложение остаётся выключенной записью на
вкладке 04 и работает как память — иначе одно и то же слово возвращалось бы в
панель на каждом новом тексте.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .accentizer import Accentizer
from .audio_pipeline import preview_text
from .text_normalization import yo_restoration
from .text_normalization.pronunciation import coverage_predicate

logger = logging.getLogger(__name__)

# Те же границы слова, что в `pronunciation.py` и `yo_restoration.py`; «+» входит в
# класс, поэтому слово с ручной разметкой ударения не считается кандидатом.
_BOUNDARY_CHARS = "0-9A-Za-z\u0400-\u04FF+"
_WORD_RE = re.compile(rf"(?<![{_BOUNDARY_CHARS}])([А-Яа-яЁё]+)(?![{_BOUNDARY_CHARS}])")
_RUSSIAN_RE = re.compile(r"[А-Яа-яЁё]+")
_STRESSED_TOKEN_RE = re.compile(r"[А-Яа-яЁё+]+")

# Порядок групп в списке предложений: ё-омографы, затем ударение, затем редкие.
_KIND_RANK = {
    "yo_homograph": 0,
    "yo_low_confidence": 1,
    "stress_homograph": 2,
    "rare": 3,
}

# Ниже этого score модель ё-фикации считается неуверенной. Порог намеренно выше
# 0.5: предложение должно появляться там, где модель колеблется, а не всегда.
YO_CONFIDENCE_THRESHOLD = 0.75

# Длинное слово вне всех списков — единственная эвристика для «редкое». Порог
# высокий, чтобы таких кандидатов было мало и они не вытесняли настоящие омографы.
RARE_MIN_LENGTH = 12

DEFAULT_LIMIT = 20

_omograph_lock = threading.Lock()
_omographs: dict[str, list[str]] | None = None


@dataclass(frozen=True)
class Suggestion:
    """Один кандидат в словарь.

    `confidence` заполняется только настоящим score модели; для источников, где
    уверенности никто не считал, остаётся None — выдумывать её нельзя.
    """

    word: str
    target: str
    kind: str
    reason: str
    alternatives: list[str] = field(default_factory=list)
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "target": self.target,
            "kind": self.kind,
            "reason": self.reason,
            "alternatives": list(self.alternatives),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Suggestions:
    """Ответ на «что уточнить»: кандидаты и сколько слов вообще просмотрено."""

    candidates: list[Suggestion]
    considered: int

    def to_dict(self) -> dict:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "considered": self.considered,
        }


def omographs_path() -> Path | None:
    """Путь к списку омографов ударения внутри установленного ruaccent.

    Данные читаются на месте: копировать их в репозиторий запрещено лицензией
    (CC BY-NC-ND). Нет библиотеки или файла — источник просто пропускается.
    """
    try:
        import ruaccent
    except Exception:  # noqa: BLE001 — библиотека необязательна
        return None
    path = Path(ruaccent.__file__).resolve().parent / "dictionary" / "omographs.json.gz"
    return path if path.exists() else None


def load_omographs() -> dict[str, list[str]]:
    """Ленивая загрузка списка омографов ударения (только чтение, без копирования)."""
    global _omographs
    with _omograph_lock:
        if _omographs is not None:
            return _omographs
        result: dict[str, list[str]] = {}
        path = omographs_path()
        if path is not None:
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    for key, value in raw.items():
                        if isinstance(value, list) and value:
                            result[str(key).lower()] = [str(item) for item in value]
            except (OSError, EOFError, ValueError) as exc:
                logger.warning("Список омографов ударения недоступен (%s) — источник пропущен", exc)
        _omographs = result
        return _omographs


def reset_cache() -> None:
    """Сбрасывает ленивый список омографов — для тестов и смены установки ruaccent."""
    global _omographs
    with _omograph_lock:
        _omographs = None


def _stress_choices(accented: str | None) -> dict[str, list[str]]:
    """Варианты ударения, реально выбранные RUAccent: «без +» → [«с +», ...].

    Акцентированный текст — единственный источник того, что выбрала модель: если
    RUAccent недоступен или ударения выключены, «+» в тексте нет и словарь вернёт
    пустой результат.
    """
    choices: dict[str, list[str]] = {}
    for token in _STRESSED_TOKEN_RE.findall(accented or ""):
        if "+" not in token:
            continue
        choices.setdefault(token.replace("+", "").lower(), []).append(token)
    return choices


def _is_candidate(word: str, covered) -> bool:
    """Слово вообще может быть предложено: русское, без «+» и не покрыто правилом."""
    if not word or "+" in word:
        return False
    if _RUSSIAN_RE.fullmatch(word) is None:
        return False
    return not covered(word)


def _pick_stress(word: str, variants: list[str], queues: Mapping[str, list[str]]) -> str:
    """Вариант, выбранный RUAccent, иначе первый из словаря омографов."""
    queue = queues.get(word.lower())
    if queue:
        return queue[0]
    return variants[0]


def build_suggestions(
    text: str,
    *,
    rules=None,
    accentized: str | None = None,
    supports_accents: bool = True,
    auto_accent: bool = True,
    limit: int = DEFAULT_LIMIT,
    accentizer: Accentizer | None = None,
) -> Suggestions:
    """Считает кандидатов по фактическому пути текста до модели.

    Текст проходит те же стадии, что и preview (нормализация → «ё» → словарь →
    акцентуация), поэтому кандидаты ищутся по тому, что уйдёт в модель, а не по
    сырому исходнику. Ничего не пишется в базу и не синтезируется.
    """
    clean = (text or "").strip()
    if not clean:
        return Suggestions(candidates=[], considered=0)

    stages = preview_text(
        clean, supports_accents=supports_accents, auto_accent=auto_accent, rules=rules
    )
    working = stages.dictionary
    accented = stages.accentized if accentized is None else accentized

    # Покрытие — вместе с выключенными правилами: это память об отклонённом.
    covered = coverage_predicate(rules, include_disabled=True)
    words = [(match.group(1), match.start()) for match in _WORD_RE.finditer(working)]
    considered = len({word.lower() for word, _ in words})

    ambiguous = yo_restoration.ambiguous_dictionary()
    safe = yo_restoration.safe_dictionary()
    omographs = load_omographs()
    queues = _stress_choices(accented)

    found: dict[str, tuple[int, int, Suggestion]] = {}

    def add(rank: int, position: int, suggestion: Suggestion) -> None:
        """Добавляет кандидата; дубль обогащается, а не дублируется.

        Одно и то же слово не должно стоять в списке дважды. Если второй источник
        знает про него больше (например, у модели есть score), первый кандидат
        дополняется уверенностью и пояснением, а порядок и позиция сохраняются.
        """
        key = suggestion.word.lower()
        existing = found.get(key)
        if existing is None:
            found[key] = (rank, position, suggestion)
            return
        if existing[2].confidence is None and suggestion.confidence is not None:
            merged = replace(
                existing[2],
                confidence=suggestion.confidence,
                reason=f"{existing[2].reason} {suggestion.reason}",
            )
            found[key] = (existing[0], existing[1], merged)

    # 1. Неоднозначные «е/ё»: автоматически они не меняются никогда.
    if ambiguous:
        for word, position in words:
            target = ambiguous.get(word.lower())
            if target is None or "ё" in word.lower():
                continue
            if not _is_candidate(word, covered):
                continue
            yo_word = yo_restoration.match_case(word, target)
            add(
                _KIND_RANK["yo_homograph"],
                position,
                Suggestion(
                    word=word,
                    target=yo_word,
                    kind="yo_homograph",
                    reason=(
                        f"«{word}» — неоднозначная «е/ё»: автоматически не меняется. "
                        f"Вариант с «ё» — «{yo_word}»; если слово читается с «е», "
                        "оставьте как есть и отклоните предложение."
                    ),
                    alternatives=[word],
                ),
            )

    # 2. Омографы ударения: предлагается вариант, выбранный RUAccent.
    if omographs:
        for word, position in words:
            variants = omographs.get(word.lower())
            if not variants:
                continue
            if not _is_candidate(word, covered):
                continue
            target = _pick_stress(word, variants, queues)
            alternatives = [variant for variant in variants if variant != target]
            chosen_by_model = bool(queues.get(word.lower()))
            add(
                _KIND_RANK["stress_homograph"],
                position,
                Suggestion(
                    word=word,
                    target=target,
                    kind="stress_homograph",
                    reason=(
                        f"«{word}» — омограф ударения. "
                        + (
                            f"RUAccent выбрал «{target}»."
                            if chosen_by_model
                            else f"Вариантов несколько, первый — «{target}»."
                        )
                        + (
                            " Другие варианты: "
                            + ", ".join(f"«{item}»" for item in alternatives)
                            + "."
                            if alternatives
                            else ""
                        )
                    ),
                    alternatives=alternatives,
                ),
            )

    # 3. Неуверенность модели ё-фикации — только если модель уже поднята и даёт score.
    active = accentizer if accentizer is not None else Accentizer.instance()
    if getattr(active, "is_loaded", False):
        predictions = active.yo_homograph_scores(working)
        if predictions:
            scores: dict[str, float] = {}
            for item in predictions:
                word = str(item.get("word", "")).lower()
                score = item.get("score")
                if not word or not isinstance(score, (int, float)):
                    continue
                # Самое низкое значение — самая сильная неуверенность по слову.
                scores[word] = min(score, scores.get(word, 1.0))
            for word, position in words:
                score = scores.get(word.lower())
                if score is None or score >= YO_CONFIDENCE_THRESHOLD:
                    continue
                if "ё" in word.lower():
                    continue
                target = ambiguous.get(word.lower()) if ambiguous else None
                if target is None:
                    target = active.yo_form(word)
                if target is None:
                    continue
                if not _is_candidate(word, covered):
                    continue
                yo_word = yo_restoration.match_case(word, target)
                add(
                    _KIND_RANK["yo_low_confidence"],
                    position,
                    Suggestion(
                        word=word,
                        target=yo_word,
                        kind="yo_low_confidence",
                        reason=(
                            f"«{word}» — модель ё-фикации не уверена (score {score:.2f}); "
                            f"возможен вариант «{yo_word}»."
                        ),
                        alternatives=[word],
                        confidence=score,
                    ),
                )

    # 4. Редкие слова — консервативный fallback, последним и без выдуманной замены.
    for word, position in words:
        if len(word) < RARE_MIN_LENGTH:
            continue
        lower = word.lower()
        if lower in safe or lower in ambiguous or lower in omographs:
            continue
        if not _is_candidate(word, covered):
            continue
        queue = queues.get(lower)
        target = queue[0] if queue else ""
        add(
            _KIND_RANK["rare"],
            position,
            Suggestion(
                word=word,
                target=target,
                kind="rare",
                reason=(
                    f"«{word}» ({len(word)} знаков) — редкое слово: проверьте ударение "
                    "и впишите замену вручную, например «звон+ит»."
                ),
                alternatives=[],
            ),
        )

    ordered = sorted(found.values(), key=lambda item: (item[0], item[1]))
    return Suggestions(
        candidates=[item[2] for item in ordered[: max(limit, 0)]],
        considered=considered,
    )


def suggest(
    text: str,
    *,
    rules=None,
    accentized: str | None = None,
    limit: int = DEFAULT_LIMIT,
    supports_accents: bool = True,
    auto_accent: bool = True,
    accentizer: Accentizer | None = None,
) -> list[Suggestion]:
    """Кандидаты в словарь произношения для текста — см. `build_suggestions`."""
    return build_suggestions(
        text,
        rules=rules,
        accentized=accentized,
        supports_accents=supports_accents,
        auto_accent=auto_accent,
        limit=limit,
        accentizer=accentizer,
    ).candidates
