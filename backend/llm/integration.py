"""Встраивание LLM-анализа в подготовку проекта: кандидаты, сверка, подстатус.

Три задачи этого модуля, и все они про одно: LLM предлагает, backend решает.

1. **Кандидаты.** Разбор превращается в те же кандидаты в словарь, что уже умеет
   интерфейс (`pronunciation_candidates`), с пометкой источника `llm`. Пользователь
   видит предложение рядом с предложениями детерминированного слоя и решает сам.
2. **Сверка с детерминированным слоем (§10).** Если словарь или «ё»-стадия уже
   дали тот же ответ — это согласие; если дали другой — конфликт, и он уходит в
   review. Молча предпочесть одну сторону нельзя: обе стороны ошибаются по-своему,
   а цена ошибки — неверное чтение в озвучке.
3. **Подстатус проекта (§12).** `DISABLED/PENDING/RUNNING/READY/NEEDS_REVIEW/FAILED`
   — отдельная ось от готовности текста. Ошибка LLM не ломает проект: детерминированная
   подготовка остаётся, но пользователь видит, что анализа не было.

Чего здесь нет и не должно быть: записи в словарь. LLM создаёт только кандидата;
принять его в проектный или глобальный словарь может лишь явное действие человека
(фаза 6).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from . import memory_policy as memory
from . import scheduler as scheduler_module
from . import schemas as s
from .analysis_cache import AnalysisKey, dictionary_hash, rule_fields, text_hash
from .analyzer import (
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_NEEDS_REVIEW,
    STATUS_READY,
    LinguisticAnalyzer,
    ReplicaAnalysis,
    build_context,
    context_hash,
)

logger = logging.getLogger("tts.llm.integration")

LLM_SOURCE = "llm"
# Виды согласия с детерминированным слоем (§10): согласие, конфликт, «своего мнения
# нет». Третье состояние обязательно: чаще всего словарь просто не знает слова, и
# называть это согласием значило бы врать в интерфейсе.
AGREEMENT_AGREE = "agree"
AGREEMENT_CONFLICT = "conflict"
AGREEMENT_NONE = "none"

# Порядок подстатусов при сведении по репликам: худшее побеждает.
_STATUS_SEVERITY = {
    STATUS_DISABLED: 0,
    STATUS_READY: 1,
    STATUS_NEEDS_REVIEW: 2,
    STATUS_FAILED: 3,
}


def llm_candidates(
    analysis: ReplicaAnalysis,
    *,
    replica_index: int,
    preparation=None,
    rules: Sequence[object] = (),
) -> list[dict]:
    """Превращает разбор в кандидаты словаря, не меняя ни текст, ни правила.

    `target` остаётся подсказкой: у омографа это словарная форма чтения, у «ё» —
    форма с «ё», у имени может быть пустым (тогда пользователь впишет чтение сам).
    Пустой `target` — не ошибка: кандидат означает «проверьте это слово».

    `rules` — **все** правила проекта и глобальные, включая выключенные: выключенное
    правило это память об отклонённом предложении, и повторно предлагать то же слово
    значит спрашивать одно и то же по кругу. Кандидат при этом не исчезает из
    разбора: он остаётся с `resolved_by`, чтобы решение было видно в интерфейсе.
    """
    candidates: list[dict] = []
    for item in analysis.items:
        opinion, detail, resolved_by = cross_check(item, preparation, rules)
        reason = item.meaning or item.reason_code or item.type
        if resolved_by == "agreement":
            reason = f"{reason} — согласие с детерминированным слоем"
        elif resolved_by == "rejected":
            reason = f"{reason} — уже отклонено при подготовке"
        candidates.append(
            {
                "word": item.source,
                "target": item.suggested_form,
                "kind": f"{LLM_SOURCE}_{item.type}",
                "reason": reason,
                "alternatives": [],
                "confidence": item.confidence,
                # Поля ниже нужны интерфейсу и review, а детерминированным
                # кандидатам они не нужны — там нет модели и её уверенности.
                "source": LLM_SOURCE,
                "type": item.type,
                "meaning": item.meaning,
                # Решённое детерминированным слоем не требует решения человека:
                # согласие — потому что стороны совпали, отклонение — потому что
                # человек уже отказался от этого слова.
                "needs_review": bool(item.needs_review) and not resolved_by,
                "resolved_by": resolved_by,
                "reason_code": item.reason_code,
                "span_start": item.span_start,
                "span_end": item.span_end,
                "replica_index": replica_index,
                "agreement": opinion,
                "agreement_detail": detail,
                "model_tag": analysis.model_tag,
                "model_digest": analysis.model_digest,
            }
        )
    for dropped in analysis.dropped:
        # Отброшенные аннотации тоже показываем: «модель предложила слово, которого
        # нет в тексте» — это диагностика, а не тишина.
        candidates.append(
            {
                "word": str(dropped.get("source") or ""),
                "target": "",
                "kind": f"{LLM_SOURCE}_dropped",
                "reason": f"предложение отброшено: {dropped.get('reason', '')}",
                "alternatives": [],
                "confidence": None,
                "source": LLM_SOURCE,
                "type": str(dropped.get("type") or ""),
                "meaning": "",
                "needs_review": True,
                "reason_code": str(dropped.get("reason") or ""),
                "span_start": None,
                "span_end": None,
                "replica_index": replica_index,
                "agreement": AGREEMENT_NONE,
                "agreement_detail": "",
                "resolved_by": "",
                "model_tag": analysis.model_tag,
                "model_digest": analysis.model_digest,
            }
        )
    return candidates


def cross_check(
    item: s.Annotation,
    preparation,
    rules: Sequence[object] = (),
) -> tuple[str, str, str]:
    """Сверяет предложение модели со словарём и «ё»-стадией (§10).

    Возвращает `(согласие, пояснение, resolved_by)`, где `resolved_by` — почему
    предложение не требует решения человека:
    `agreement` — словарь или «ё»-стадия дают то же чтение;
    `rejected` — правило по этому слову уже есть, но выключено (человек отказался);
    пустая строка — решение за человеком.
    """
    words = _rule_targets(item.source, rules)
    if words:
        for target, enabled in words:
            if _same_reading(item.suggested_form, target):
                if enabled:
                    return AGREEMENT_AGREE, f"словарь: «{item.source}» → «{target}»", "agreement"
                return AGREEMENT_NONE, "слово уже отклонено при подготовке", "rejected"
        target, enabled = words[0]
        if enabled:
            return (
                AGREEMENT_CONFLICT,
                f"словарь даёт «{target}», модель — «{item.suggested_form}»",
                "",
            )
        return AGREEMENT_NONE, "слово уже отклонено при подготовке", "rejected"
    opinion, detail = deterministic_opinion(item, preparation)
    if opinion == AGREEMENT_AGREE:
        return opinion, detail, "agreement"
    return opinion, detail, ""


def _rule_targets(source: str, rules: Sequence[object]) -> list[tuple[str, bool]]:
    """Правила словаря, покрывающие это слово: (target, включено).

    Проверяются и выключенные правила: они хранят память об отклонённом предложении,
    иначе одно и то же слово предлагалось бы после каждого анализа.
    """
    if not source:
        return []
    from ..text_normalization import pronunciation as pronunciation_rules

    found: list[tuple[str, bool]] = []
    for rule in rules:
        fields = rule_fields(rule)
        compiled = pronunciation_rules.compile_rule(
            pronunciation_rules.PronunciationRule(
                source=str(fields.get("source") or ""),
                target=str(fields.get("target") or ""),
                case_sensitive=bool(fields.get("case_sensitive", False)),
                whole_word=bool(fields.get("whole_word", True)),
                enabled=bool(fields.get("enabled", True)),
            )
        )
        if compiled is None:
            continue
        if compiled.pattern.search(source):
            found.append(
                (str(fields.get("target") or ""), bool(fields.get("enabled", True)))
            )
    return found


def deterministic_opinion(item: s.Annotation, preparation) -> tuple[str, str]:
    """Что об этом слове думает детерминированный слой (§10).

    Сравниваются только те решения, которые уже приняты в подготовке: правило
    словаря (`dictionary_matches`) и восстановленная «ё» (стадия `yo_text`). Ничего
    не пересчитывается: второй расчёт мог бы разойтись с тем, что уйдёт в модель.
    """
    if preparation is None:
        return AGREEMENT_NONE, ""
    word = item.source
    for match in getattr(preparation, "dictionary_matches", []) or []:
        source = str(match.get("source") or "")
        if not source or source.lower() != word.lower():
            continue
        target = str(match.get("target") or "")
        if item.suggested_form and target and _same_reading(item.suggested_form, target):
            return AGREEMENT_AGREE, f"словарь проекта/глобальный: «{source}» → «{target}»"
        if target:
            return AGREEMENT_CONFLICT, f"словарь даёт «{target}», модель — «{item.suggested_form}»"
    if item.type == s.TYPE_YO:
        yo_text = getattr(preparation, "yo_text", "") or ""
        source_text = getattr(preparation, "source_text", "") or ""
        if yo_text and source_text and item.suggested_form:
            # «Ё»-стадия уже восстановила эту форму — значит модель подтверждает
            # детерминированный результат, а не спорит с ним.
            if item.suggested_form.lower() in yo_text.lower() and item.source.lower() not in yo_text.lower():
                return AGREEMENT_AGREE, "«ё» восстановлена детерминированной стадией"
            if item.source.lower() in yo_text.lower():
                return AGREEMENT_CONFLICT, "«ё»-стадия оставила «е» как есть"
    return AGREEMENT_NONE, ""


def _same_reading(left: str, right: str) -> bool:
    """Совпадают ли чтения: слово и место ударения.

    Сравнивать «просто слова» нельзя: в проекте ударение кодируется знаком `+` перед
    ударной гласной, и `зам+ок` (запор) отличается от `з+амок` (строение) только
    позицией знака. Поэтому оба написания приводятся к паре «слово + индекс ударной
    гласной»: `+` перед гласной и комбинирующий акут после неё (так пишет RUAccent)
    дают один и тот же индекс. Если ударение не указано ни с одной стороны, слова
    считаются совпадающими: у словаря просто нет мнения о месте ударения.
    """
    left_word, left_stress = _reading_key(left)
    right_word, right_stress = _reading_key(right)
    if left_word != right_word:
        return False
    if left_stress is None or right_stress is None:
        return True
    return left_stress == right_stress


def _reading_key(text: str) -> tuple[str, int | None]:
    """Слово и индекс ударной гласной (None — ударение не указано)."""
    stress: int | None = None
    plus = text.find("+")
    word = text.replace("+", "")
    if plus >= 0:
        stress = plus
    acute = word.find("\u0301")
    if acute >= 0:
        stress = acute - 1
        word = word.replace("\u0301", "")
    return word.strip().lower(), stress


def merge_preparation_candidates(preparation, extra: Sequence[dict]):
    """Добавляет LLM-кандидатов к детерминированным, не трогая стадии текста.

    Детерминированные кандидаты идут первыми: они воспроизводимы и не зависят от
    модели, поэтому в интерфейсе их видно раньше предложений LLM.
    """
    from dataclasses import replace as _replace

    existing = list(getattr(preparation, "pronunciation_candidates", []) or [])
    return _replace(preparation, pronunciation_candidates=existing + list(extra))


def worst_status(statuses: Iterable[str], *, enabled: bool) -> str:
    """Сводит подстатусы реплик в один статус проекта: худшее побеждает."""
    if not enabled:
        return STATUS_DISABLED
    worst = STATUS_READY
    for status in statuses:
        if _STATUS_SEVERITY.get(status, 0) > _STATUS_SEVERITY.get(worst, 0):
            worst = status
    return worst


def utterance_hints(rows: Iterable[Mapping[str, object]]) -> dict[int, dict]:
    """Подсказки коротких реплик из сохранённых разборов: индекс → класс и соседи.

    Наружу отдаётся только то, что разрешено схемой: класс реплики, зависимость от
    контекста и релевантные id. Ни текст, ни уверенность сюда не попадают — слой
    коротких реплик не должен получать от модели ничего, кроме подсказки.
    """
    hints: dict[int, dict] = {}
    for row in rows:
        if str(row.get("status") or "") not in (STATUS_READY, STATUS_NEEDS_REVIEW):
            continue
        try:
            payload = json.loads(str(row.get("analysis_json") or "{}"))
        except ValueError:
            continue
        utterance = payload.get("utterance") or {}
        if not isinstance(utterance, dict):
            continue
        hints[int(row.get("replica_index") or 0)] = {
            "class": str(utterance.get("class") or ""),
            "context_dependency": str(utterance.get("context_dependency") or ""),
            "relevant_replica_ids": [
                int(item)
                for item in (utterance.get("relevant_replica_ids") or [])
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            ],
            # Эмоция и акт реплики (UPDATE 2 §32). Короткий слой их не использует:
            # они нужны, чтобы после анализа записать эмоцию реплики в проект и
            # показать её в интерфейсе — без второго чтения разборов.
            "emotion": str(utterance.get("emotion") or ""),
            "emotion_confidence": float(utterance.get("emotion_confidence") or 0.0),
            "dialogue_act": str(utterance.get("dialogue_act") or ""),
        }
    return hints


def replica_emotions(rows: Iterable[Mapping[str, object]]) -> dict[int, dict]:
    """Эмоция по репликам из сохранённых разборов: индекс → эмоция и уверенность.

    Отдельно от `utterance_hints`, потому что читатели разные: подсказки короткого
    слоя — про класс и соседей, а эмоция уходит в данные проекта и в интерфейс.
    Смешивать их значило бы тянуть в короткий слой лишнее.
    """
    result: dict[int, dict] = {}
    for row in rows:
        if str(row.get("status") or "") not in (STATUS_READY, STATUS_NEEDS_REVIEW):
            continue
        try:
            payload = json.loads(str(row.get("analysis_json") or "{}"))
        except ValueError:
            continue
        utterance = payload.get("utterance") or {}
        if not isinstance(utterance, dict) or not utterance.get("emotion"):
            continue
        result[int(row.get("replica_index") or 0)] = {
            "emotion": str(utterance.get("emotion") or ""),
            "confidence": float(utterance.get("emotion_confidence") or 0.0),
            "dialogue_act": str(utterance.get("dialogue_act") or ""),
            "context_dependency": str(utterance.get("context_dependency") or ""),
        }
    return result


@dataclass
class ProjectAnalysisOutcome:
    """Итог LLM-прохода по проекту: разборы, подстатус и что было с кешем."""

    status: str
    model_tag: str = ""
    model_digest: str = ""
    analyses: dict[int, ReplicaAnalysis] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
    from_cache: int = 0
    calls: int = 0
    blocked_reason: str = ""
    unloaded: bool = False

    @property
    def failed(self) -> list[dict]:
        """Реплики, которые модель не разобрала, с причиной каждой."""
        return [
            {"replica_index": index, "error": item.error or "разбор не удался"}
            for index, item in sorted(self.analyses.items())
            if item.status == STATUS_FAILED
        ]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "replicas": len(self.analyses),
            "from_cache": self.from_cache,
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "blocked_reason": self.blocked_reason,
            "unloaded": self.unloaded,
            "replicas_failed": len(self.failed),
            "failed": self.failed[:20],
        }


class ProjectLlmAnalyzer:
    """LLM-проход по репликам проекта: кеш, слот, анализ, подстатус.

    Класс не знает про SQLite и FastAPI: кеш приходит двумя функциями
    (`lookup`/`save`), слот — планировщиком, модель — анализатором. Так его можно
    проверить без базы и без модели, а политику хранения менять отдельно.
    """

    def __init__(
        self,
        *,
        analyzer: LinguisticAnalyzer | None = None,
        scheduler: scheduler_module.HeavyScheduler | None = None,
        lookup: Callable[[int, AnalysisKey], ReplicaAnalysis | None] | None = None,
        save: Callable[[int, ReplicaAnalysis], None] | None = None,
        model_size_gb: float | None = None,
        unload_after: bool = True,
        clock=time.monotonic,
    ) -> None:
        self.analyzer = analyzer or LinguisticAnalyzer()
        self.scheduler = scheduler or scheduler_module.get_scheduler()
        self._lookup = lookup
        self._save = save
        self.model_size_gb = model_size_gb
        # Выгружать модель после прохода (§14.3): дальше идёт рендер, которому нужна
        # память под F5/XTTS. Держать резидентную LLM «на всякий случай» значит
        # отдать под неё 2--6 GB, которые понадобятся синтезу.
        self.unload_after = unload_after
        self.clock = clock

    def run(
        self,
        *,
        project_id: str,
        replicas: Sequence[Mapping[str, object]],
        rules: Sequence[Mapping[str, object]] = (),
        on_progress: Callable[[str], None] | None = None,
    ) -> ProjectAnalysisOutcome:
        """Анализирует реплики проекта: кеш → слот → модель → сохранение.

        Если Analyzer выключен, ничего не делается: `DISABLED` — честный статус, а
        не «тихо пропустили». Если память или слот не дают начать, разбор не
        подменяется детерминированным результатом: статус `FAILED` с причиной.
        """
        settings = self.analyzer.settings
        if not settings.enabled:
            return ProjectAnalysisOutcome(status=STATUS_DISABLED)
        report = on_progress or (lambda message: logger.info("%s", message))
        started = self.clock()
        rules_digest = dictionary_hash(rules)
        texts = [str(row.get("text") or "") for row in replicas]
        analyses: dict[int, ReplicaAnalysis] = {}
        from_cache = 0
        calls = 0
        model_tag = ""
        digest = ""

        # Слот берётся на весь проход: держать модель загруженной между репликами
        # дешевле, чем грузить её заново на каждую, а параллельно с синтезом она
        # всё равно работать не должна (§14.2).
        try:
            with self.scheduler.hold(
                memory.HEAVY_LLM,
                model_size_gb=self.model_size_gb,
                owner=f"analysis:{project_id}",
            ):
                for position, row in enumerate(replicas):
                    # Проход длинный (десятки реплик), и память может уйти в HARD
                    # STOP уже во время него: модель загружена, KV-кэш растёт.
                    # Проверяем перед каждой репликой и останавливаемся с причиной —
                    # частичный разбор лучше, чем swap на всей машине (§14).
                    abort = self.scheduler.check(
                        memory.HEAVY_LLM, model_size_gb=self.model_size_gb
                    )
                    if abort.memory_level == memory.LEVEL_HARD_STOP:
                        # Модель могла быть загружена предыдущими репликами: при
                        # остановке её обязательно возвращаем системе.
                        self._unload_model_safely(model_tag or self.analyzer.selected_model())
                        return ProjectAnalysisOutcome(
                            status=STATUS_FAILED,
                            model_tag=model_tag,
                            model_digest=digest,
                            analyses=analyses,
                            seconds=self.clock() - started,
                            error=f"HARD STOP: {abort.memory_reason}",
                            blocked_reason=abort.memory_reason,
                            from_cache=from_cache,
                            calls=calls,
                        )
                    index = int(row.get("index") or position)
                    target = texts[position]
                    context = build_context(
                        target,
                        texts[max(0, position - settings.context_replicas) : position],
                        texts[position + 1 : position + 1 + settings.context_replicas],
                        limit=settings.context_replicas,
                        budget_chars=settings.context_chars,
                    )
                    key = AnalysisKey(
                        source_text_hash=text_hash(target),
                        context_hash=context_hash(context),
                        dictionary_hash=rules_digest,
                        # Digest модели ещё не известен до первого ответа: кеш по
                        # нему проверяется на втором проходе, а здесь он берётся из
                        # уже сохранённого разбора при поиске.
                        model_digest=digest,
                        prompt_version=self.analyzer.prompt.version,
                        schema_version=s.SCHEMA_VERSION,
                    )
                    cached = self._lookup_analysis(index, key) if self._lookup else None
                    if cached is not None:
                        analyses[index] = cached
                        from_cache += 1
                        model_tag = model_tag or cached.model_tag
                        digest = digest or cached.model_digest
                        continue
                    analysis = self.analyzer.analyze_replica(
                        replica_id=int(row.get("replica_id") or 0) or index + 1,
                        target_text=target,
                        before=context["context_before"],
                        after=context["context_after"],
                        dictionary_digest=rules_digest,
                    )
                    calls += 1
                    analyses[index] = analysis
                    model_tag = analysis.model_tag or model_tag
                    digest = analysis.model_digest or digest
                    if self._save is not None:
                        # Сохраняем и FAILED: причина отказа реплики должна быть
                        # видна пользователю, а не растворяться в общем «ошибка».
                        # Кеш такие записи не отдаёт (см. AnalysisCache.lookup).
                        self._save(index, analysis)
        except scheduler_module.HeavyBlockedError as exc:
            # Не «продолжим без LLM»: пользователь обязан увидеть, что анализа не
            # было, и повторить его позже.
            report(f"LLM-анализ не запущен: {exc}")
            return ProjectAnalysisOutcome(
                status=STATUS_FAILED,
                analyses=analyses,
                seconds=self.clock() - started,
                error=str(exc),
                blocked_reason=exc.decision.memory_reason or str(exc),
                from_cache=from_cache,
                calls=calls,
            )
        except Exception as exc:  # noqa: BLE001 — LLM не должна ронять подготовку
            logger.warning("LLM-анализ проекта %s не удался: %s", project_id, exc)
            return ProjectAnalysisOutcome(
                status=STATUS_FAILED,
                analyses=analyses,
                seconds=self.clock() - started,
                error=f"{type(exc).__name__}: {exc}",
                from_cache=from_cache,
                calls=calls,
            )

        status = worst_status((item.status for item in analyses.values()), enabled=True)
        if not analyses:
            status = STATUS_READY
        unloaded = self._release_model(model_tag or self.analyzer.selected_model(), calls)
        return ProjectAnalysisOutcome(
            status=status,
            model_tag=model_tag,
            model_digest=digest,
            analyses=analyses,
            seconds=self.clock() - started,
            from_cache=from_cache,
            calls=calls,
            unloaded=unloaded,
        )

    def _unload_model_safely(self, model: str) -> bool:
        """Выгружает модель, не поднимая исключение: остановка важнее деталей."""
        if not model:
            return False
        try:
            return bool(self.analyzer.client.unload(model))
        except Exception as exc:  # noqa: BLE001 — выгрузка не повод падать
            logger.warning("Не удалось выгрузить модель %s: %s", model, exc)
            return False

    def _release_model(self, model: str, calls: int) -> bool:
        """Освобождает память модели после прохода.

        Выгружаем только если модель реально загружалась в этом проходе: при полном
        попадании в кеш выгружать нечего, и трогать чужую загруженную модель нельзя.
        Ошибка выгрузки не отменяет результат анализа — она логируется.
        """
        if not self.unload_after or not model or calls == 0:
            return False
        try:
            released = bool(self.analyzer.client.unload(model))
        except Exception as exc:  # noqa: BLE001 — выгрузка не повод терять разбор
            logger.warning("Не удалось выгрузить модель %s: %s", model, exc)
            return False
        if released:
            logger.info("Модель %s выгружена после анализа: память возвращена синтезу", model)
        return released

    def _lookup_analysis(self, index: int, key: AnalysisKey) -> ReplicaAnalysis | None:
        """Поиск в кеше с учётом digest модели: он известен из прошлого разбора.

        Ключ поиска требует digest, но до первого ответа его взять неоткуда. Поэтому
        поиск идёт в два шага: сначала по входу без digest (модель та же, что была),
        и результат принимается, только если сохранённый digest совпал с текущей
        выбранной моделью.
        """
        if self._lookup is None:
            return None
        current = self._analyzer_digest()
        candidate_key = AnalysisKey(
            source_text_hash=key.source_text_hash,
            context_hash=key.context_hash,
            dictionary_hash=key.dictionary_hash,
            model_digest=current,
            prompt_version=key.prompt_version,
            schema_version=key.schema_version,
        )
        return self._lookup(index, candidate_key)

    def _analyzer_digest(self) -> str:
        try:
            model = self.analyzer.selected_model()
            info = self.analyzer.client.model_info(model)
        except Exception as exc:  # noqa: BLE001 — паспорт модели не критичен
            logger.debug("Не удалось прочитать digest модели: %s", exc)
            return ""
        return info.digest if info is not None else ""
