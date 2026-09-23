"""LinguisticAnalyzer: контекстный русский анализ реплики локальной LLM (Task 2).

Задача Analyzer'а — добавить контекстное понимание русского языка перед синтезом,
не забирая управление текстом у детерминированного pipeline'а. Отсюда все решения
этого модуля:

* **Аннотации, а не текст.** Результат — список участков исходной строки с типом,
  чтением и уверенностью. `final_text` формируют существующие стадии
  (`project_analysis` → engine adapter + RUAccent), и LLM не может его переписать:
  в контракте ответа нет поля для нового текста, а лишнее поле отвергает ответ.
* **Backend — источник истины по границам.** Benchmark показал, что модели находят
  нужное слово, но почти не умеют считать символы (18--24 точных границы из 339
  ответов). Поэтому `source` — главное, что мы берём у модели, а границы backend
  ищет сам (`locate_annotations`); аннотацию с несуществующим `source` отбрасываем
  с причиной, а не применяем и не «додумываем».
* **Отказ виден, а не спрятан.** Недоступная Ollama, таймаут или невалидный ответ
  дают статус `failed` с причиной. Никакого тихого перехода в «просто
  детерминированный режим» под видом успешного анализа (§12, §17 Task 2).

Модуль не знает про FastAPI, БД и очередь: он умеет ровно одно — собрать
ограниченный контекст, вызвать модель через `OllamaClient`, проверить ответ и
вернуть разбор. Хранение, состояния проекта и review живут выше (фазы 3--6).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from . import schemas as s
from .ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModel,
    get_client,
)
from .prompt import ANALYZER_WINDOW_PROMPT, PromptTemplate, load_prompt
from .settings_store import load_settings

logger = logging.getLogger("tts.llm.analyzer")

# --- состояния анализа реплики (§12) ------------------------------------------
STATUS_DISABLED = "DISABLED"
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_READY = "READY"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"
STATUS_FAILED = "FAILED"
STATUS_STALE = "STALE"
ANALYSIS_STATUSES: tuple[str, ...] = (
    STATUS_DISABLED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_READY,
    STATUS_NEEDS_REVIEW,
    STATUS_FAILED,
    STATUS_STALE,
)

# Причины, по которым аннотация не принята в разбор (видны пользователю в review).
DROP_SOURCE_NOT_FOUND = "source_not_found"
DROP_UNKNOWN_TYPE = "unknown_type"
DROP_OVERLAP = "overlap"
# Разметка ударения F5 (`+`) — ответственность engine adapter'а и RUAccent, не LLM
# (§10, §27 Task 2). Если модель всё же прислала `+`, знак снимается, а разбор
# уходит в review: молча «почищенная» подсказка выглядела бы как ответ модели.
MARKUP_STRIPPED = "stress_markup_stripped"

DEFAULT_PRIMARY_MODEL = "qwen3:8b"
DEFAULT_FALLBACK_MODEL = "qwen3:4b-instruct-2507-q8_0"
DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT_SEC = 90.0
DEFAULT_CONTEXT_REPLICAS = 2
DEFAULT_CONTEXT_CHARS = 4000
# Сколько реплик сцены уходит в один запрос (§7). Шесть — компромисс: модель
# видит реплику, соседей с обеих сторон и говорящих, а `num_ctx` остаётся в
# пределах разумного. Реплика на запрос — это ровно то, что запрещено.
DEFAULT_SCENE_REPLICAS = 6


@dataclass(frozen=True)
class AnalyzerSettings:
    """Настройки Analyzer'а: из окружения проекта и из результатов benchmark'а.

    Имена переменных — как в постановке Task 2 (`LLM_PRIMARY_MODEL`,
    `LLM_FALLBACK_MODEL`); проектная приставка `TTS_` тоже принимается, чтобы не
    заводить второй стиль конфигурации.
    """

    enabled: bool = False
    primary_model: str = DEFAULT_PRIMARY_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = 0.0
    seed: int = 0
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    max_repairs: int = 1
    context_replicas: int = DEFAULT_CONTEXT_REPLICAS
    context_chars: int = DEFAULT_CONTEXT_CHARS
    # Размер окна сцены (§6, §7): сколько реплик диалога уходит в один запрос.
    scene_replicas: int = DEFAULT_SCENE_REPLICAS
    response_format: str = "json"
    think: str = "auto"
    # Режим проекта по умолчанию: «LLM обязателен перед рендером» или опционален.
    required_for_render: bool = False

    def to_dict(self) -> dict:
        """Плоское представление настроек анализатора для экрана настроек."""
        return {
            "enabled": self.enabled,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout_sec": self.timeout_sec,
            "max_repairs": self.max_repairs,
            "context_replicas": self.context_replicas,
            "context_chars": self.context_chars,
            "scene_replicas": self.scene_replicas,
            "response_format": self.response_format,
            "think": self.think,
            "required_for_render": self.required_for_render,
        }


def _env(env: Mapping[str, str], name: str) -> str | None:
    """Значение переменной: сначала имя из постановки, потом проектное `TTS_`."""
    for key in (name, f"TTS_{name}"):
        raw = env.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "да"}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Переменная %s = %r не число — беру %s", name, raw, default)
        return default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Переменная %s = %r не число — беру %s", name, raw, default)
        return default


def settings_from_env(env: Mapping[str, str] | None = None) -> AnalyzerSettings:
    """Настройки Analyzer'а: окружение как база + то, что включили в интерфейсе.

    Окружение (`LLM_*`) описывает развёртывание, файл `data/llm_settings.json` —
    выбор пользователя; файл важнее, иначе галочка в приложении не пережила бы
    перезапуск. По умолчанию (без файла и без переменных) Analyzer **выключен**:
    включение — осознанное действие, потому что это отдельная модель рядом с TTS.
    """
    source = os.environ if env is None else env
    base = _settings_from_env(source)
    return apply_overrides(base, load_settings())


def apply_overrides(settings: AnalyzerSettings, overrides: Mapping[str, object]) -> AnalyzerSettings:
    """Накладывает сохранённые настройки на значения окружения."""
    if not overrides:
        return settings
    values = settings.to_dict()
    for key in ("enabled", "required_for_render"):
        if key in overrides:
            values[key] = bool(overrides[key])
    for key in ("num_ctx", "context_replicas", "scene_replicas"):
        if key in overrides:
            try:
                number = int(overrides[key])
            except (TypeError, ValueError):
                continue
            if number > 0:
                values[key] = number
    for key in ("primary_model", "fallback_model"):
        text = str(overrides.get(key) or "").strip()
        if text:
            values[key] = text
    return replace(settings, **{
        key: values[key]
        for key in (
            "enabled",
            "primary_model",
            "fallback_model",
            "required_for_render",
            "num_ctx",
            "context_replicas",
            "scene_replicas",
        )
    })


def _settings_from_env(source: Mapping[str, str]) -> AnalyzerSettings:
    """Значения по умолчанию из окружения: база для пользовательских настроек."""
    return AnalyzerSettings(
        enabled=_env_flag(source, "LLM_ANALYZER_ENABLED", False),
        primary_model=_env(source, "LLM_PRIMARY_MODEL") or DEFAULT_PRIMARY_MODEL,
        fallback_model=_env(source, "LLM_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL,
        num_ctx=_env_int(source, "LLM_ANALYZER_NUM_CTX", DEFAULT_NUM_CTX),
        temperature=_env_float(source, "LLM_ANALYZER_TEMPERATURE", 0.0),
        seed=_env_int(source, "LLM_ANALYZER_SEED", 0),
        timeout_sec=_env_float(source, "LLM_ANALYZER_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC),
        max_repairs=_env_int(source, "LLM_ANALYZER_MAX_REPAIRS", 1),
        context_replicas=_env_int(
            source, "LLM_ANALYZER_CONTEXT_REPLICAS", DEFAULT_CONTEXT_REPLICAS
        ),
        context_chars=_env_int(source, "LLM_ANALYZER_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS),
        scene_replicas=_env_int(source, "LLM_ANALYZER_SCENE_REPLICAS", DEFAULT_SCENE_REPLICAS),
        response_format=_env(source, "LLM_ANALYZER_RESPONSE_FORMAT") or "json",
        think=_env(source, "LLM_ANALYZER_THINK") or "auto",
        required_for_render=_env_flag(source, "LLM_ANALYZER_REQUIRED_FOR_RENDER", False),
    )


def text_hash(text: str) -> str:
    """Короткий хеш текста: по нему analysis считается устаревшим (§11, §18)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_context(
    target_text: str,
    before: Sequence[str] = (),
    after: Sequence[str] = (),
    *,
    limit: int = DEFAULT_CONTEXT_REPLICAS,
    budget_chars: int = DEFAULT_CONTEXT_CHARS,
) -> dict:
    """Ограниченный контекст реплики: цель + соседи в пределах бюджета (§6).

    Бюджет обрезается от дальних соседей к ближним: ближайшая реплика важнее для
    смысла, а «весь проект целиком» раздувает KV-кэш и latency. Целевая реплика не
    обрезается никогда — анализ без неё бессмыслен.
    """
    selected_before: list[str] = []
    selected_after: list[str] = []
    spent = len(target_text)
    # Бюджет жёсткий: сосед добавляется, только если он в него влезает. Целевая
    # реплика не обрезается никогда — анализ без неё бессмыслен.
    for text in reversed(list(before)[-limit:]):
        if spent + len(text) > budget_chars:
            break
        selected_before.insert(0, text)
        spent += len(text)
    for text in list(after)[:limit]:
        if spent + len(text) > budget_chars:
            break
        selected_after.append(text)
        spent += len(text)
    return {
        "target_text": target_text,
        "context_before": selected_before,
        "context_after": selected_after,
        "budget_chars": budget_chars,
    }


def context_hash(context: Mapping[str, Any]) -> str:
    """Хеш контекста: изменение соседей тоже инвалидирует анализ (§11)."""
    payload = json.dumps(
        {
            "target_text": context.get("target_text", ""),
            "context_before": list(context.get("context_before") or []),
            "context_after": list(context.get("context_after") or []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return text_hash(payload)


def inference_hash(settings: AnalyzerSettings) -> str:
    """Хеш параметров запроса к модели: они тоже вход разбора, а не оформление (§46).

    При том же тексте, модели и prompt'е ответ зависит от `num_ctx`, `temperature` и
    `seed`. Без их хеша в ключе кеша смена параметра оставляла бы прежний разбор
    действительным, и пользователь видел бы результат, полученный не при тех
    настройках, что стоят сейчас.
    """
    payload = json.dumps(
        {
            "num_ctx": int(settings.num_ctx),
            "temperature": float(settings.temperature),
            "seed": int(settings.seed),
        },
        sort_keys=True,
    )
    return text_hash(payload)


def plan_windows(
    count: int, *, size: int = DEFAULT_SCENE_REPLICAS, overlap: int = DEFAULT_CONTEXT_REPLICAS
) -> tuple[tuple[int, int], ...]:
    """Разбивает реплики на окна анализа: короткая сцена — одно окно (§6, §7).

    Окна идут с перекрытием: реплика на границе окна должна видеть соседей с
    другой стороны, иначе смысл «Это проблема?» снова решался бы по одной строке.
    Ответ на перекрывающуюся реплику берётся из первого окна — так у каждой
    реплики ровно один разбор, и результат не зависит от порядка обхода.

    Возвращает полуинтервалы `(начало, конец)` по позициям реплик.
    """
    if count <= 0 or size <= 0:
        return ()
    step = max(size - max(overlap, 0), 1)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + size, count)
        windows.append((start, end))
        if end >= count:
            break
        start += step
    return tuple(windows)


def scene_hash(scene: Sequence[Mapping[str, object]]) -> str:
    """Хеш окна: id, говорящий и текст всех реплик, увиденных моделью (§11).

    Ключ кеша должен описывать **то, что реально ушло в запрос**. Для окна это вся
    сцена, а не только ±2 соседа: иначе разбор, полученный по шести репликам,
    вернулся бы из кеша для той же реплики в другой сцене — и контекстная разница,
    ради которой всё делалось, исчезла бы.
    """
    payload = json.dumps(
        [
            {
                "replica_id": int(item.get("replica_id") or 0),
                "speaker": str(item.get("speaker") or ""),
                "target_text": str(item.get("target_text") or ""),
            }
            for item in scene
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return text_hash(payload)


def locate_annotations(
    items: Iterable[s.Annotation], target_text: str
) -> tuple[tuple[s.Annotation, ...], tuple[dict, ...]]:
    """Приводит аннотации к тексту: границы ищет backend, а не модель.

    Правило benchmark'а в коде: если `source` есть в тексте, берём **его**
    вхождение (учитывая заявленную позицию как подсказку), а не доверяем
    `span_start/span_end`. Если `source` в тексте нет — аннотация отбрасывается с
    причиной: выдуманного слова в пользовательском тексте быть не может. Пересечения
    разрешаются в пользу первой (более длинной) аннотации.

    Возвращает `(принятые аннотации, отброшенные с причиной)`.
    """
    accepted: list[s.Annotation] = []
    dropped: list[dict] = []
    for item in items:
        if item.type not in s.ANNOTATION_TYPES:
            dropped.append({"source": item.source, "type": item.type, "reason": DROP_UNKNOWN_TYPE})
            continue
        start, actual = _find_source(target_text, item.source, item.span_start)
        if start is None:
            dropped.append(
                {"source": item.source, "type": item.type, "reason": DROP_SOURCE_NOT_FOUND}
            )
            continue
        if any(
            max(existing.span_start, start) < min(existing.span_end, start + len(actual))
            for existing in accepted
        ):
            dropped.append({"source": item.source, "type": item.type, "reason": DROP_OVERLAP})
            continue
        cleaned, _ = strip_stress_markup(item)
        accepted.append(
            replace(cleaned, span_start=start, span_end=start + len(actual), source=actual)
        )
    accepted.sort(key=lambda item: item.span_start)
    return tuple(accepted), tuple(dropped)


def strip_stress_markup(item: s.Annotation) -> tuple[s.Annotation, bool]:
    """Убирает знаки ударения F5 из подсказки модели.

    LLM не должна формировать engine-specific разметку: у XTTS нет `+`, и знак,
    пришедший из ответа модели, попал бы в чужой движок. Поэтому `+` снимается
    здесь, а сам факт виден в разборе (`markup_stripped`).
    """
    form = item.suggested_form
    meaning = item.meaning
    if "+" not in form and "+" not in meaning:
        return item, False
    return (
        replace(
            item,
            suggested_form=form.replace("+", ""),
            meaning=meaning.replace("+", ""),
        ),
        True,
    )


def _find_source(target_text: str, source: str, hint: int) -> tuple[int | None, str]:
    """Ищет `source` в тексте; ближайшее к подсказке вхождение побеждает.

    Сравнение без учёта регистра: модель может изменить регистр слова, а текст
    менять нельзя — поэтому возвращается фактическая подстрока из текста.
    """
    needle = (source or "").strip()
    if not needle:
        return None, ""
    lowered = target_text.lower()
    lowered_needle = needle.lower()
    positions = []
    start = lowered.find(lowered_needle)
    while start >= 0:
        positions.append(start)
        start = lowered.find(lowered_needle, start + 1)
    if not positions:
        return None, ""
    best = min(positions, key=lambda position: abs(position - max(hint, 0)))
    return best, target_text[best : best + len(needle)]


@dataclass(frozen=True)
class ReplicaAnalysis:
    """Разбор одной реплики: аннотации, подсказка о реплике и служебные факты."""

    replica_id: int
    status: str
    items: tuple[s.Annotation, ...] = ()
    dropped: tuple[dict, ...] = ()
    utterance: s.UtteranceHint = field(default_factory=s.UtteranceHint)
    model_tag: str = ""
    model_digest: str = ""
    prompt_version: str = ""
    schema_version: str = s.SCHEMA_VERSION
    # Хеш параметров запроса (`num_ctx`, `temperature`, `seed`): разбор делался при
    # этих настройках, и смена любой из них делает его недействительным (§46).
    inference_hash: str = ""
    source_hash: str = ""
    context_hash: str = ""
    # Хеш словаря, влияющего на реплику: разбор перестаёт быть действительным при
    # изменении правила, а не только текста (§11, §18 Task 2).
    dictionary_hash: str = ""
    seconds: float = 0.0
    error: str = ""
    # Ответ пришлось запрашивать повторно (repair): это диагностика, а не ошибка.
    repaired: bool = False
    # Сколько подсказок пришло с разметкой ударения F5 (`+`): знак снят, разбор в review.
    markup_stripped: int = 0

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    def to_dict(self) -> dict:
        """Плоское представление разбора реплики: пункты, отброшенное и провенанс."""
        return {
            "replica_id": self.replica_id,
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "dropped": [dict(item) for item in self.dropped],
            "utterance": self.utterance.to_dict(),
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "inference_hash": self.inference_hash,
            "source_hash": self.source_hash,
            "context_hash": self.context_hash,
            "dictionary_hash": self.dictionary_hash,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "repaired": self.repaired,
            "markup_stripped": self.markup_stripped,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class LinguisticAnalyzer:
    """Анализ реплик выбранной локальной моделью через Ollama.

    Класс не хранит состояние проекта: он получает текст и контекст, возвращает
    разбор. Решения о том, когда звать (память, очередь, состояния, review),
    принимают вызывающие слои.
    """

    def __init__(
        self,
        *,
        settings: AnalyzerSettings | None = None,
        client: OllamaClient | None = None,
        prompt: PromptTemplate | None = None,
        window_prompt: PromptTemplate | None = None,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings or settings_from_env()
        self.client = client or get_client()
        self.prompt = prompt or load_prompt()
        # Оконный prompt отдельный: у окна другой контракт (конверт `replicas[]`),
        # и версия именно этого файла уходит в `prompt_version` оконных разборов —
        # иначе вчерашний разбор сцены нашёлся бы в кеше по одиночному prompt'у.
        self.window_prompt = window_prompt or load_prompt(ANALYZER_WINDOW_PROMPT)
        self.clock = clock
        self.last_model = ""

    # -- статус -----------------------------------------------------------------
    def selected_model(self) -> str:
        """Модель для работы: primary, если скачана, иначе fallback.

        Подмена primary на fallback — это **состояние**, которое видно в статусе и
        в результате анализа (`model_tag`), а не тихий откат: иначе результат
        одной модели однажды приняли бы за результат другой.
        """
        try:
            if self.client.has_model(self.settings.primary_model):
                return self.settings.primary_model
            if self.client.has_model(self.settings.fallback_model):
                return self.settings.fallback_model
        except OllamaError as exc:
            logger.debug("Не удалось проверить модели: %s", exc)
        return self.settings.primary_model

    def status(self) -> dict:
        """Готовность Analyzer'а: для `/api/llm/status`, без запуска анализа."""
        payload: dict = {
            "enabled": self.settings.enabled,
            "state": STATUS_DISABLED if not self.settings.enabled else STATUS_PENDING,
            "settings": self.settings.to_dict(),
            "prompt_version": self.prompt.version,
            "schema_version": s.SCHEMA_VERSION,
            "reason": "" if self.settings.enabled else "Analyzer выключен (LLM_ANALYZER_ENABLED=0)",
        }
        try:
            health = self.client.health()
        except OllamaError as exc:
            health = None
            payload.update(
                {"ollama": {"available": False, "error": str(exc)}, "model": "", "reason": str(exc)}
            )
        if health is not None:
            payload["ollama"] = {
                "available": health.available,
                "version": health.version,
                "error": health.error,
            }
            if health.available:
                payload["model"] = self.selected_model()
                payload["model_available"] = self.client.has_model(payload["model"])
                if not payload["reason"]:
                    payload["reason"] = (
                        "" if payload["model_available"] else f"Модель «{payload['model']}» не скачана"
                    )
        payload["ollama"]["models"] = [info.tag for info in self._models_safe()]
        return payload

    def models(self) -> list[dict]:
        """Скачанные модели с признаком «primary/fallback» для `/api/llm/models`."""
        result = []
        for info in self._models_safe():
            result.append(
                {
                    **info.to_dict(),
                    "role": (
                        "primary"
                        if info.tag == self.settings.primary_model
                        else ("fallback" if info.tag == self.settings.fallback_model else "")
                    ),
                }
            )
        return result

    def _model_exists(self, model: str) -> bool:
        try:
            return bool(self.client.has_model(model))
        except OllamaError as exc:
            logger.debug("Не удалось проверить модель %s: %s", model, exc)
            return False

    def _models_safe(self) -> list[OllamaModel]:
        try:
            return list(self.client.models())
        except OllamaError as exc:
            logger.info("Ollama недоступна: %s", exc)
            return []

    # -- анализ -----------------------------------------------------------------
    def analyze_replica(
        self,
        *,
        replica_id: int,
        target_text: str,
        before: Sequence[str] = (),
        after: Sequence[str] = (),
        dictionary_digest: str = "",
        cancel=None,
    ) -> ReplicaAnalysis:
        """Анализирует одну реплику и возвращает разбор (никогда не бросает).

        Ошибка Ollama, таймаут или невалидный ответ — это `status=failed` с
        причиной: backend обязан переживать недоступность LLM (§13), а
        пользователь — видеть, что анализа не было, а не «тихий» успех.
        """
        context = build_context(
            target_text,
            before,
            after,
            limit=self.settings.context_replicas,
            budget_chars=self.settings.context_chars,
        )
        source_digest = text_hash(target_text)
        context_digest = context_hash(context)
        if not self.settings.enabled:
            return self._disabled(replica_id, source_digest, context_digest, dictionary_digest)

        started = self.clock()
        model = self.selected_model()
        self.last_model = model
        if not self._model_exists(model):
            # Явная причина вместо HTTP-ошибки Ollama: пользователю важно понять,
            # что модель не скачана, а не «что-то пошло не так».
            return self._failure(
                replica_id,
                model,
                source_digest,
                context_digest,
                started,
                f"Модель «{model}» не скачана локально",
                dictionary_digest=dictionary_digest,
            )
        payload = {
            "replica_id": replica_id,
            "target_text": context["target_text"],
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "language": "ru",
        }
        messages = self.prompt.messages(payload, schema_json=self._schema_text())
        try:
            analysis, errors, repaired = self._request_analysis(model, messages, payload, cancel)
        except OllamaError as exc:
            return self._failure(
                replica_id,
                model,
                source_digest,
                context_digest,
                started,
                f"{type(exc).__name__}: {exc}",
                dictionary_digest=dictionary_digest,
            )

        if analysis is None:
            return self._failure(
                replica_id,
                model,
                source_digest,
                context_digest,
                started,
                "Ответ не прошёл проверку: " + ", ".join(errors),
                dictionary_digest=dictionary_digest,
            )

        return self._build_analysis(
            analysis,
            replica_id=replica_id,
            target_text=target_text,
            model=model,
            source_digest=source_digest,
            context_digest=context_digest,
            dictionary_digest=dictionary_digest,
            started=started,
            repaired=repaired,
        )

    def analyze_window(
        self,
        *,
        scene: Sequence[Mapping[str, object]],
        dictionary_digest: str = "",
        cancel=None,
    ) -> dict[int, ReplicaAnalysis]:
        """Анализирует окно реплик одним запросом (§6, §7). Никогда не бросает.

        `scene` — реплики так, как их видит модель: `replica_id`, `speaker`,
        `target_text`. Возвращаются разборы по `replica_id`; реплика, о которой
        модель не ответила или чей разбор не прошёл проверку, получает `FAILED` с
        причиной. Это не «шумный отказ»: пользователь обязан видеть, какие реплики
        остались без анализа, — иначе пропуск выглядел бы как «ошибок нет».

        Контекст здесь — всё окно, а не ±2 соседа: смысл реплики в диалоге
        («Правда?» после хорошей и после плохой новости) определяется сценой, и
        обрезать её до двух соседей значило бы вернуть ту же слепоту, только
        дороже — один запрос на несколько реплик.
        """
        texts = {
            int(item.get("replica_id") or 0): str(item.get("target_text") or "")
            for item in scene
        }
        ids = list(texts)
        scene_digest = scene_hash(scene)
        if not self.settings.enabled:
            return {
                replica_id: self._disabled(
                    replica_id,
                    text_hash(texts[replica_id]),
                    scene_digest,
                    dictionary_digest,
                    prompt_version=self.window_prompt.version,
                )
                for replica_id in ids
            }

        started = self.clock()
        model = self.selected_model()
        self.last_model = model
        model_digest = self._model_digest(model)

        def failed(replica_id: int, error: str) -> ReplicaAnalysis:
            """Разбор-ошибка для реплики окна с уже посчитанным digest модели."""
            # digest читается один раз на окно, а не на каждую реплику: паспорт
            # модели — это её свойство, а не свойство разбора.
            return self._failure(
                replica_id,
                model,
                text_hash(texts[replica_id]),
                scene_digest,
                started,
                error,
                dictionary_digest=dictionary_digest,
                model_digest=model_digest,
                prompt_version=self.window_prompt.version,
            )

        if not self._model_exists(model):
            # Явная причина вместо HTTP-ошибки Ollama: пользователю важно понять,
            # что модель не скачана, а не «что-то пошло не так».
            return {
                replica_id: failed(replica_id, f"Модель «{model}» не скачана локально")
                for replica_id in ids
            }

        payload = {
            "replicas": [
                {
                    "replica_id": int(item.get("replica_id") or 0),
                    "speaker": str(item.get("speaker") or ""),
                    "target_text": str(item.get("target_text") or ""),
                }
                for item in scene
            ],
            "language": "ru",
        }
        messages = self.window_prompt.messages(payload, schema_json=self._window_schema_text())
        try:
            window, errors, repaired = self._request_window(
                model, messages, ids, texts, cancel
            )
        except OllamaError as exc:
            error = f"{type(exc).__name__}: {exc}"
            return {replica_id: failed(replica_id, error) for replica_id in ids}

        if window is None:
            error = "Ответ не прошёл проверку: " + ", ".join(errors)
            return {replica_id: failed(replica_id, error) for replica_id in ids}

        rejected = {
            int(item.get("replica_id") or 0): ", ".join(item.get("errors") or [])
            for item in window.rejected
        }
        parsed = window.by_replica()
        results: dict[int, ReplicaAnalysis] = {}
        for replica_id in ids:
            analysis = parsed.get(replica_id)
            if analysis is None:
                results[replica_id] = failed(
                    replica_id,
                    "Реплика не разобрана в окне"
                    + (f": {rejected[replica_id]}" if rejected.get(replica_id) else ""),
                )
                continue
            results[replica_id] = self._build_analysis(
                analysis,
                replica_id=replica_id,
                target_text=texts[replica_id],
                model=model,
                model_digest=model_digest,
                source_digest=text_hash(texts[replica_id]),
                context_digest=scene_digest,
                dictionary_digest=dictionary_digest,
                started=started,
                repaired=repaired,
                prompt_version=self.window_prompt.version,
            )
        return results

    def _build_analysis(
        self,
        analysis: s.LinguisticAnalysis,
        *,
        replica_id: int,
        target_text: str,
        model: str,
        source_digest: str,
        context_digest: str,
        dictionary_digest: str,
        started: float,
        model_digest: str | None = None,
        repaired: bool = False,
        prompt_version: str | None = None,
    ) -> ReplicaAnalysis:
        """Собирает разбор реплики из ответа модели: границы, статус, служебные поля.

        Общий путь одиночного анализа и окна: разница только в том, откуда взялся
        ответ, а решение «что принять» обязано быть одинаковым — иначе одна и та же
        реплика получала бы разный разбор в зависимости от размера окна.

        `prompt_version` — версия prompt'а, чьим ответом получен разбор: у окна она
        своя, и от неё зависит попадание в кеш (§46).
        """
        items, dropped = locate_annotations(analysis.items, target_text)
        markup_stripped = sum(
            1 for item in analysis.items if strip_stress_markup(item)[1]
        )
        # NEEDS_REVIEW — не только «модель сомневается», но и «часть аннотаций
        # отброшена или подсказка пришла с чужой разметкой»: пользователь должен
        # видеть, что разбор неполный или изменённый.
        status = STATUS_READY
        if dropped or markup_stripped or any(item.needs_review for item in items):
            status = STATUS_NEEDS_REVIEW
        return ReplicaAnalysis(
            replica_id=replica_id,
            status=status,
            items=items,
            dropped=dropped,
            utterance=analysis.utterance,
            model_tag=model,
            model_digest=self._model_digest(model) if model_digest is None else model_digest,
            prompt_version=prompt_version or self.prompt.version,
            inference_hash=inference_hash(self.settings),
            source_hash=source_digest,
            context_hash=context_digest,
            dictionary_hash=dictionary_digest,
            seconds=self.clock() - started,
            repaired=repaired,
            markup_stripped=markup_stripped,
        )

    def _disabled(
        self,
        replica_id: int,
        source_digest: str,
        context_digest: str,
        dictionary_digest: str,
        prompt_version: str | None = None,
    ) -> ReplicaAnalysis:
        """Разбор выключенного Analyzer'а: не «пусто», а состояние с причиной."""
        return ReplicaAnalysis(
            replica_id=replica_id,
            status=STATUS_DISABLED,
            model_tag=self.settings.primary_model,
            prompt_version=prompt_version or self.prompt.version,
            inference_hash=inference_hash(self.settings),
            source_hash=source_digest,
            context_hash=context_digest,
            dictionary_hash=dictionary_digest,
            error="Analyzer выключен",
        )

    def _request_analysis(
        self, model: str, messages: list[dict], payload: dict, cancel
    ) -> tuple[s.LinguisticAnalysis | None, list[str], bool]:
        """Один запрос и, при невалидном ответе, один ограниченный повтор (§13)."""
        schema = s.analysis_json_schema() if self.settings.response_format == "schema" else None
        chat = self.client.chat(
            model,
            messages,
            schema=schema,
            options=self._options(),
            keep_alive="5m",
            timeout=self.settings.timeout_sec,
            cancel=cancel,
            think=self._think_for(model),
        )
        analysis, errors = s.parse_analysis(
            chat.text,
            expected_replica_id=payload["replica_id"],
            target_text=payload["target_text"],
            check_spans=False,
            # Код причины — пояснение для интерфейса, а не решение о тексте:
            # неизвестный код не должен стоить всей реплики.
            check_reasons=False,
        )
        repairs = 0
        repair_messages = messages
        raw_text = chat.text
        while analysis is None and repairs < max(int(self.settings.max_repairs), 0):
            repairs += 1
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw_text[:2000]},
                {
                    "role": "user",
                    "content": (
                        "Предыдущий ответ не прошёл проверку схемы. Верни ТОЛЬКО JSON по "
                        "схеме, не убирая найденные аннотации. Ошибки: "
                        + ", ".join(dict.fromkeys(errors))
                        + "."
                    ),
                },
            ]
            chat = self.client.chat(
                model,
                repair_messages,
                schema=schema,
                options=self._options(),
                keep_alive="5m",
                timeout=self.settings.timeout_sec,
                cancel=cancel,
                think=self._think_for(model),
            )
            raw_text = chat.text
            analysis, errors = s.parse_analysis(
                raw_text,
                expected_replica_id=payload["replica_id"],
                target_text=payload["target_text"],
                check_spans=False,
                check_reasons=False,
            )
        return analysis, errors, repairs > 0

    def _request_window(
        self,
        model: str,
        messages: list[dict],
        replica_ids: Sequence[int],
        texts: Mapping[int, str],
        cancel,
    ) -> tuple[s.WindowAnalysis | None, list[str], bool]:
        """Один запрос на окно сцены и один ограниченный повтор при отказе (§7, §13).

        Ответ проверяется целиком (`parse_window_analysis`), но цена ошибки другая:
        не прошедшая проверку реплика теряет только свой разбор, а не весь запрос.
        """
        schema = s.window_json_schema() if self.settings.response_format == "schema" else None
        chat = self.client.chat(
            model,
            messages,
            schema=schema,
            options=self._options(),
            keep_alive="5m",
            timeout=self.settings.timeout_sec,
            cancel=cancel,
            think=self._think_for(model),
        )
        window, errors = s.parse_window_analysis(
            chat.text,
            replica_ids=replica_ids,
            target_texts=texts,
            check_spans=False,
            check_reasons=False,
        )
        repairs = 0
        raw_text = chat.text
        while window is None and repairs < max(int(self.settings.max_repairs), 0):
            repairs += 1
            chat = self.client.chat(
                model,
                [
                    *messages,
                    {"role": "assistant", "content": raw_text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "Предыдущий ответ не прошёл проверку схемы. Верни ТОЛЬКО JSON "
                            "по схеме — объект с полем replicas, по разбору на КАЖДЫЙ "
                            "replica_id из входа. Ошибки: "
                            + ", ".join(dict.fromkeys(errors))
                            + "."
                        ),
                    },
                ],
                schema=schema,
                options=self._options(),
                keep_alive="5m",
                timeout=self.settings.timeout_sec,
                cancel=cancel,
                think=self._think_for(model),
            )
            raw_text = chat.text
            window, errors = s.parse_window_analysis(
                raw_text,
                replica_ids=replica_ids,
                target_texts=texts,
                check_spans=False,
                check_reasons=False,
            )
        return window, errors, repairs > 0

    def _options(self) -> dict:
        """Параметры запроса: одни и те же для реплики, окна и повторов."""
        return {
            "num_ctx": self.settings.num_ctx,
            "temperature": self.settings.temperature,
            "seed": self.settings.seed,
        }

    def _think_for(self, model: str) -> bool | None:
        """`auto`: выключить скрытое рассуждение у моделей, которые это умеют."""
        if self.settings.think == "off":
            return False
        if self.settings.think == "on":
            return True
        try:
            capabilities = self.client.capabilities(model)
        except Exception as exc:  # noqa: BLE001 — паспорт модели не критичен
            logger.debug("Не удалось прочитать capabilities %s: %s", model, exc)
            return None
        return False if "thinking" in capabilities else None

    def _schema_text(self) -> str:
        return json.dumps(s.analysis_json_schema(), ensure_ascii=False, indent=2)

    def _window_schema_text(self) -> str:
        return json.dumps(s.window_json_schema(), ensure_ascii=False, indent=2)

    def _model_digest(self, model: str) -> str:
        try:
            info = self.client.model_info(model)
        except OllamaError as exc:
            logger.debug("Не удалось прочитать digest %s: %s", model, exc)
            return ""
        return info.digest if info is not None else ""

    def _failure(
        self,
        replica_id: int,
        model: str,
        source_digest: str,
        context_digest: str,
        started: float,
        error: str,
        dictionary_digest: str = "",
        model_digest: str | None = None,
        prompt_version: str | None = None,
    ) -> ReplicaAnalysis:
        logger.info("Анализ реплики %s не выполнен: %s", replica_id, error)
        return ReplicaAnalysis(
            replica_id=replica_id,
            status=STATUS_FAILED,
            model_tag=model,
            model_digest=self._model_digest(model) if model_digest is None else model_digest,
            prompt_version=prompt_version or self.prompt.version,
            inference_hash=inference_hash(self.settings),
            source_hash=source_digest,
            context_hash=context_digest,
            dictionary_hash=dictionary_digest,
            seconds=self.clock() - started,
            error=error,
        )


_analyzer: LinguisticAnalyzer | None = None


def get_analyzer() -> LinguisticAnalyzer:
    """Singleton Analyzer'а: настройки читаются один раз на процесс."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LinguisticAnalyzer()
    return _analyzer


def reset_analyzer() -> None:
    """Сброс singleton'а — нужен тестам и смене настроек в рантайме."""
    global _analyzer
    _analyzer = None
