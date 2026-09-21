"""Скрытый речевой прогрев коротких реплик (warm-up prefix).

Зачем. Каждая реплика синтезируется независимо, и у модели нет «предыдущей речи»:
на односложной фразе это слышно как неестественный старт и съеденная атака первого
слова. Приём: перед короткой репликой в TTS уходит небольшой естественный текст
(прогрев), а в готовый файл попадает **только цель** — звук прогрева срезается по
границе, найденной распознаванием.

Границы ответственности:

* прогрев — **внутренний технический контекст**, а не часть текста пользователя:
  ни `dialogue_text`, ни `final_text`, ни метаданные реплики его не содержат;
* LLM **не переписывает** целевую фразу — она только порождает префикс;
* QA сравнивает готовое аудио с **целью**, а не с «префикс + цель»;
* любая ошибка (LLM недоступна, таймаут, пустой ответ, не найденная граница,
  нехватка памяти) означает обычный синтез цели без прогрева — задача из-за
  прогрева не падает никогда.

Модуль не знает про `audio_pipeline` и движки: он отвечает на вопросы «нужен ли
прогрев», «какой текст префикса» и «где начинается цель в готовом аудио». Вызов
движка и постобработка остаются в пайплайне.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import config
from .transcribe import WordStamp

logger = logging.getLogger(__name__)

# Версия prompt'а: меняет и текст запроса, и ключ кэша (§17). Растёт вместе с
# формулировкой правил — иначе после правки prompt'а кэш отдавал бы старые префиксы.
PROMPT_VERSION = "1"

# Ключ ответа модели. Схема из одного поля — не украшение: клиент Ollama просит
# структурированный ответ, и «просто текст» он бы обернул в JSON сам, оставив разбор
# на догадки. Одно поле `prefix` делает ответ однозначным.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {"prefix": {"type": "string"}},
    "required": ["prefix"],
}

SYSTEM_PROMPT = """Ты создаёшь скрытый речевой контекст для TTS.

Тебе дана короткая целевая реплика диалога и соседний контекст.
Сгенерируй 1–2 естественных предложения, которые могли бы непосредственно
предшествовать целевой реплике и помогли бы речевой модели начать говорить
с естественной интонацией.

Правила:
- не изменяй и не повторяй TARGET;
- верни только PREFIX в поле prefix;
- используй язык TARGET;
- не добавляй имя спикера;
- не используй markdown, кавычки, XML или SSML;
- PREFIX не должен быть обязательной частью смысла диалога;
- максимум {max_prefix_chars} символов."""

# Причины отказа — коды: их читает лог и метаданные варианта, а не человек.
REASON_DISABLED = "disabled"
REASON_ENGINE = "engine_not_allowed"
REASON_TOO_LONG = "target_too_long"
REASON_EMPTY = "target_empty"
REASON_MEMORY = "memory_pressure"
REASON_LLM_ERROR = "llm_error"
REASON_LLM_TIMEOUT = "llm_timeout"
REASON_LLM_UNAVAILABLE = "llm_unavailable"
REASON_EMPTY_RESPONSE = "empty_response"
REASON_TOO_LONG_PREFIX = "prefix_too_long"
REASON_REPEATS_TARGET = "prefix_repeats_target"
REASON_MARKUP = "prefix_has_markup"
REASON_ALIGNMENT = "alignment_failed"
REASON_OK = "ok"

# Разметка, которой в произносимом префиксе быть не должно: markdown, XML/SSML,
# кавычки-обёртки и технические скобки. Проверяется до синтеза, а не после.
_FORBIDDEN_MARKUP = re.compile(r"[<>\[\]{}*_#`|]|```")
_QUOTES = "\"'«»“”„"
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё+-]+")
_WS_RE = re.compile(r"\s+")


@dataclass
class WarmupContext:
    """Решение о прогреве: цель, префикс и чем всё закончилось.

    `enabled` — «прогрев запрошен и применён», а не «функция включена»: если
    префикс не получен, структура остаётся с `prefix_text = None`, и пайплайн
    синтезирует цель обычным путём.
    """

    target_text: str
    prefix_text: str | None = None
    enabled: bool = False
    reason: str = ""
    cache_hit: bool = False
    seconds: float = 0.0
    model: str = ""
    # Заполняется при синтезе: где в готовом аудио начинается цель и был ли откат.
    boundary_sec: float | None = None
    fallback: str = ""

    @property
    def synthesis_text(self) -> str:
        if not self.prefix_text:
            return self.target_text
        return f"{self.prefix_text.rstrip()} {self.target_text.lstrip()}"

    def to_dict(self, *, include_prefix: bool = True) -> dict:
        payload = {
            "warmup_used": bool(self.prefix_text),
            "warmup_boundary_sec": self.boundary_sec,
            "warmup_fallback": bool(self.fallback),
            "warmup_reason": self.reason,
            "warmup_cache_hit": self.cache_hit,
            "warmup_prefix_chars": len(self.prefix_text or ""),
        }
        if include_prefix and self.prefix_text:
            # Префикс — технический текст, а не реплика пользователя (§15): в
            # метаданных варианта он нужен для диагностики, в обычном UI — нет.
            payload["warmup_prefix"] = self.prefix_text
        return payload


@dataclass(frozen=True)
class TargetBoundary:
    """Начало целевой реплики в аудио, синтезированном как «префикс + цель»."""

    start_sec: float
    confidence: float | None = None
    method: str = "alignment"
    matched_words: int = 0
    total_words: int = 0

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 4),
            "confidence": None if self.confidence is None else round(self.confidence, 3),
            "method": self.method,
            "matched_words": self.matched_words,
            "total_words": self.total_words,
        }


# --- eligibility --------------------------------------------------------------
def word_count(text: str) -> int:
    """Слов в тексте: та же мера, что у слоя коротких реплик (знаки не считаются)."""
    return len([word for word in _WORD_RE.findall(_without_accents(text or "")) if word.strip("+-")])


def _without_accents(text: str) -> str:
    return (text or "").replace("+", "")


def enabled_for_engine(engine_id: str) -> bool:
    """Разрешён ли прогрев для движка (`TTS_WARMUP_ENGINES`).

    Отключение прогрева для конкретного движка — настройка, а не правка кода: если
    A/B покажет ухудшение на XTTS, он исключается из списка без изменения
    пайплайна (§13).
    """
    allowed = {item.strip() for item in config.WARMUP_ENGINES if item.strip()}
    return str(engine_id or "") in allowed


def eligibility_reason(
    target_text: str, engine_id: str, *, enabled: bool | None = None
) -> str:
    """Почему прогрев применяется или нет. Пустая строка — применяется.

    Решение принимается по **подготовленному** тексту: реплика, которая после
    нормализации перестала быть короткой, прогрева не получает (§3).
    """
    if enabled is None:
        enabled = config.WARMUP_ENABLED
    if not enabled:
        return REASON_DISABLED
    if not enabled_for_engine(engine_id):
        return REASON_ENGINE
    clean = (target_text or "").strip()
    if not clean:
        return REASON_EMPTY
    if len(clean) > config.WARMUP_MAX_CHARS or word_count(clean) > config.WARMUP_MAX_WORDS:
        return REASON_TOO_LONG
    return ""


# --- кэш ----------------------------------------------------------------------
def cache_key(
    *,
    engine_id: str,
    previous_text: str | None,
    target_text: str,
    next_text: str | None,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Ключ кэша префикса (§17): движок, соседи, цель и версия prompt'а.

    Версия в ключе обязательна: правка prompt'а меняет смысл ответа, и старые
    префиксы должны перестать использоваться сами, без ручной чистки кэша.
    """
    payload = json.dumps(
        {
            "engine": str(engine_id or ""),
            "previous": str(previous_text or ""),
            "target": str(target_text or ""),
            "next": str(next_text or ""),
            "prompt_version": str(prompt_version),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class WarmupCache:
    """Небольшой LRU в памяти: префикс для той же тройки текстов не спрашивают снова.

    Память, а не база: прогрев — вспомогательный слой со сроком жизни сессии, и
    переживать перезапуск ему незачем (в отличие от разборов LLM, которые стоят
    дорого и участвуют в подготовке текста).
    """

    def __init__(self, size: int | None = None) -> None:
        self._size = max(int(size if size is not None else config.WARMUP_CACHE_SIZE), 0)
        self._items: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: str) -> None:
        if self._size <= 0 or not value:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


_cache: WarmupCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> WarmupCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = WarmupCache()
    return _cache


def reset_cache() -> None:
    """Сброс кэша — для тестов и для смены настроек."""
    global _cache
    with _cache_lock:
        _cache = None


# --- валидация ответа ---------------------------------------------------------
def validate_prefix(prefix: str, target_text: str) -> tuple[str, str]:
    """Приводит ответ модели к пригодному префиксу; пустая строка — не годится.

    Возвращается `(prefix, reason)`: причина нужна логу и метаданным, чтобы
    «почему прогрева нет» не приходилось угадывать. Обрезка по лимиту — не ошибка:
    длинный, но осмысленный префикс обрезается по границе предложения, а не
    выбрасывается целиком.
    """
    text = _WS_RE.sub(" ", str(prefix or "").strip())
    text = text.strip(_QUOTES + " \t\n")
    if not text:
        return "", REASON_EMPTY_RESPONSE
    if _FORBIDDEN_MARKUP.search(text):
        return "", REASON_MARKUP
    # Имя спикера и служебные пометки вида «PREFIX:» в речь не идут.
    if re.match(r"^(prefix|префикс|target|цель|ответ)\s*:", text, re.IGNORECASE):
        text = re.sub(r"^[^:]{1,12}:\s*", "", text).strip()
    if not text:
        return "", REASON_EMPTY_RESPONSE
    target_words = [word.lower() for word in _WORD_RE.findall(_without_accents(target_text))]
    prefix_words = [word.lower() for word in _WORD_RE.findall(_without_accents(text))]
    if target_words and _contains_run(prefix_words, target_words):
        # Повтор цели внутри префикса превратил бы синтез в «цель … цель», и
        # обрезка по первому вхождению съела бы настоящую реплику.
        return "", REASON_REPEATS_TARGET
    if len(text) > config.WARMUP_MAX_PREFIX_CHARS:
        text = _cut_by_sentence(text, config.WARMUP_MAX_PREFIX_CHARS)
        if not text:
            return "", REASON_TOO_LONG_PREFIX
    if not text[-1] in ".!?…":
        text = f"{text}."
    return text, REASON_OK


def _cut_by_sentence(text: str, limit: int) -> str:
    """Обрезает префикс по границе предложения, а не по символу.

    Обрыв посреди слова звучал бы как дефект синтеза, а не как длинный префикс.
    Если границы предложения в пределах лимита нет — берётся последний пробел.
    """
    window = text[:limit]
    end = max(window.rfind("."), window.rfind("!"), window.rfind("?"), window.rfind("…"))
    if end >= max(20, limit // 3):
        return window[: end + 1].strip()
    space = window.rfind(" ")
    return (window[:space] if space >= max(20, limit // 3) else "").strip()


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    """Есть ли в префиксе целевая фраза целиком (подряд, без учёта регистра)."""
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == needle:
            return True
    return False


# --- выбор границы цели -------------------------------------------------------
def locate_target_start(
    audio: Any,
    sample_rate: int,
    full_text: str,
    target_text: str,
    *,
    asr: Callable[[Any], list[WordStamp]] | None = None,
    min_confidence: float | None = None,
) -> TargetBoundary | None:
    """Где в синтезированном «префикс + цель» начинается цель.

    Механизм — **таймстемпы слов того же Whisper**, что уже используется проектом
    (`transcribe.transcribe_words`): отдельная alignment-модель не добавляется, а
    слова цели ищутся как подпоследовательность распознанного текста. Способ
    «по доле символов» или «по расчётной длительности» запрещён: скорость речи
    неравномерна, и граница попадёт внутрь первого слова (§9).

    Возвращается `None`, если уверенность ниже порога: вызывающий обязан
    синтезировать цель заново, а не резать сомнительный результат.
    """
    from . import short_utterance as su
    from . import short_utterance_boundary as boundary_module

    threshold = (
        config.WARMUP_ALIGNMENT_MIN_CONFIDENCE
        if min_confidence is None
        else float(min_confidence)
    )
    target_words = [
        word.replace("ё", "е") for word in su._words(target_text or "")
    ]
    if not target_words:
        return None
    # Цель — суффикс синтез-текста, поэтому ищем её с конца: так «Да.» не совпадёт
    # с таким же словом внутри префикса.
    boundary = boundary_module.find_boundary_asr(
        audio,
        target_text,
        side=su.SIDE_SUFFIX,
        asr=asr,
    )
    if boundary is None:
        return None
    if boundary.confidence < threshold:
        logger.info(
            "warmup: граница цели ненадёжна (совпало %s из %s слов, порог %.2f)",
            boundary.matched_words, boundary.total_words, threshold,
        )
        return None
    return TargetBoundary(
        start_sec=max(boundary.start / float(sample_rate or 1), 0.0),
        confidence=boundary.confidence,
        method=boundary.method,
        matched_words=boundary.matched_words,
        total_words=boundary.total_words,
    )


# --- сервис -------------------------------------------------------------------
class WarmupService:
    """Сборка префикса: решение, кэш, LLM, валидация и безопасный откат.

    Сервис не бросает исключений наружу: любой сбой возвращает структуру без
    префикса и причину. Прогрев — улучшение качества, а не зависимость (§16).
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        analyzer: Any | None = None,
        cache: WarmupCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._analyzer = analyzer
        self._cache = cache if cache is not None else get_cache()
        self._clock = clock

    # -- зависимости ----------------------------------------------------------
    def _resolve_analyzer(self) -> Any | None:
        if self._analyzer is not None:
            return self._analyzer
        from .llm import analyzer as llm_analyzer

        return llm_analyzer.get_analyzer()

    def _resolve_client(self, analyzer: Any) -> Any | None:
        if self._client is not None:
            return self._client
        return getattr(analyzer, "client", None)

    # -- основной вход --------------------------------------------------------
    def build(
        self,
        *,
        target_text: str,
        engine_id: str,
        previous_text: str | None = None,
        next_text: str | None = None,
        speaker: str = "",
        enabled: bool | None = None,
    ) -> WarmupContext:
        """Префикс для реплики или структура с причиной, почему его нет."""
        started = self._clock()
        context = WarmupContext(target_text=target_text)
        reason = eligibility_reason(target_text, engine_id, enabled=enabled)
        if reason:
            context.reason = reason
            return context

        analyzer = self._resolve_analyzer()
        # Память проверяется до вызова: LLM и TTS одновременно не нужны, а
        # нехватка памяти — штатная причина обойтись обычным синтезом (§16, §18).
        if self._memory_tight():
            context.reason = REASON_MEMORY
            logger.info("warmup skipped engine=%s reason=%s", engine_id, REASON_MEMORY)
            return context
        client = self._resolve_client(analyzer) if analyzer is not None else None
        if client is None:
            context.reason = REASON_LLM_UNAVAILABLE
            logger.info("warmup skipped engine=%s reason=%s", engine_id, REASON_LLM_UNAVAILABLE)
            return context

        key = cache_key(
            engine_id=engine_id,
            previous_text=previous_text,
            target_text=target_text,
            next_text=next_text,
        )
        cached = self._cache.get(key)
        if cached:
            prefix, validate_reason = validate_prefix(cached, target_text)
            if prefix and validate_reason == REASON_OK:
                context.prefix_text = prefix
                context.enabled = True
                context.cache_hit = True
                context.reason = REASON_OK
                context.seconds = self._clock() - started
                logger.info(
                    "warmup generated engine=%s prefix_chars=%s cache_hit=true",
                    engine_id, len(prefix),
                )
                return context

        model = ""
        try:
            model = str(analyzer.selected_model() or "")
        except Exception as exc:  # noqa: BLE001 — выбор модели не повод падать
            logger.debug("warmup: модель не выбрана (%s)", exc)
        if not model:
            context.reason = REASON_LLM_UNAVAILABLE
            logger.info("warmup skipped engine=%s reason=%s", engine_id, REASON_LLM_UNAVAILABLE)
            return context

        try:
            answer = self._ask(client, model, target_text, previous_text, next_text, speaker)
        except TimeoutError:
            context.reason = REASON_LLM_TIMEOUT
            context.seconds = self._clock() - started
            logger.info("warmup fallback reason=%s engine=%s", REASON_LLM_TIMEOUT, engine_id)
            return context
        except Exception as exc:  # noqa: BLE001 — любая ошибка LLM = обычный синтез
            code = getattr(exc, "code", "") or type(exc).__name__
            context.reason = (
                REASON_LLM_TIMEOUT if "timeout" in str(code).lower() else REASON_LLM_ERROR
            )
            context.seconds = self._clock() - started
            logger.info("warmup fallback reason=%s engine=%s (%s)", context.reason, engine_id, exc)
            return context

        prefix, validate_reason = validate_prefix(answer, target_text)
        context.seconds = self._clock() - started
        context.model = model
        if not prefix:
            context.reason = validate_reason
            logger.info("warmup fallback reason=%s engine=%s", validate_reason, engine_id)
            return context
        self._cache.put(key, prefix)
        context.prefix_text = prefix
        context.enabled = True
        context.reason = REASON_OK
        logger.info(
            "warmup generated engine=%s prefix_chars=%s cache_hit=false seconds=%.2f",
            engine_id, len(prefix), context.seconds,
        )
        return context

    # -- LLM ------------------------------------------------------------------
    def _ask(
        self,
        client: Any,
        model: str,
        target_text: str,
        previous_text: str | None,
        next_text: str | None,
        speaker: str,
    ) -> str:
        """Один короткий запрос: строгие параметры, мало токенов, свой таймаут."""
        system = SYSTEM_PROMPT.format(max_prefix_chars=config.WARMUP_MAX_PREFIX_CHARS)
        user = (
            f"PREVIOUS:\n{previous_text or 'нет'}\n\n"
            f"TARGET:\n{target_text}\n\n"
            f"NEXT:\n{next_text or 'нет'}"
        )
        options = {
            "num_ctx": config.WARMUP_NUM_CTX,
            "temperature": config.WARMUP_TEMPERATURE,
            "seed": config.WARMUP_SEED,
            "num_predict": config.WARMUP_MAX_TOKENS,
        }
        result = client.chat(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=RESPONSE_SCHEMA,
            options=options,
            keep_alive=config.WARMUP_KEEP_ALIVE,
            timeout=config.WARMUP_LLM_TIMEOUT_SEC,
            think=False,
        )
        text = str(getattr(result, "text", "") or "").strip()
        try:
            payload = json.loads(text)
        except ValueError:
            # Модель ответила не-JSON: принимаем как есть — валидация ниже всё
            # равно проверит разметку, повторы и длину.
            return text
        if isinstance(payload, dict):
            return str(payload.get("prefix") or "")
        return ""

    def _memory_tight(self) -> bool:
        """Опасно ли поднимать модель сейчас.

        Прогрев не стоит того, чтобы вытеснять синтез: если система на грани или
        политика памяти LLM говорит HARD_STOP, прогрев пропускается целиком.
        """
        try:
            from . import memory_monitor, resource_guard

            # Смотрится состояние памяти, а не решение тяжёлого планировщика:
            # во время рендера слот занят самим синтезом (`HEAVY_TTS`), поэтому
            # «не allowed» означало бы «прогрев не работает при рендере никогда».
            # Уровень памяти — тот же сигнал, что использует планировщик
            # (`memory_monitor`), но без учёта занятости слота.
            return resource_guard.memory_state().get("state") == memory_monitor.STATE_CRITICAL
        except Exception as exc:  # noqa: BLE001 — проверка памяти не должна мешать
            logger.debug("warmup: состояние памяти не прочитано (%s)", exc)
            return False


_service: WarmupService | None = None
_service_lock = threading.Lock()


def get_service() -> WarmupService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = WarmupService()
    return _service


def reset_service() -> None:
    """Сброс singleton'а — для тестов и после смены настроек LLM."""
    global _service
    with _service_lock:
        _service = None


__all__ = [
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA",
    "TargetBoundary",
    "WarmupCache",
    "WarmupContext",
    "WarmupService",
    "cache_key",
    "eligibility_reason",
    "enabled_for_engine",
    "get_cache",
    "get_service",
    "locate_target_start",
    "reset_cache",
    "reset_service",
    "validate_prefix",
    "word_count",
]
