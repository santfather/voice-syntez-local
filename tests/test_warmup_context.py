"""Прогрев коротких реплик: решение, префикс, кэш и граница цели.

Тесты держат главный инвариант слоя: прогрев — улучшение, а не зависимость.
Любой сбой (LLM, память, разметка, alignment) обязан закончиться обычным
синтезом цели, а не ошибкой задачи и не чужим текстом в готовом файле.
"""

from __future__ import annotations

import json
import types

import pytest
from conftest import sine

from backend import config, resource_guard, warmup_context
from backend.engines.base import SAMPLE_RATE
from backend.short_utterance_boundary import PAD_BEFORE_SEC
from backend.transcribe import WordStamp

ENGINE = "f5"
TARGET = "Да."
PREFIX = "Тише, сейчас начнём."


class FakeClient:
    """Подставной LLM-клиент: модель не поднимается, вызовы считаются.

    Подпись `chat` повторяет рабочую: если сервис однажды перестанет передавать
    схему, таймаут или `think`, это должно быть видно здесь, а не на живой модели.
    """

    def __init__(self, answer: object = None, error: Exception | None = None) -> None:
        self.answer = json.dumps({"prefix": PREFIX}, ensure_ascii=False) if answer is None else answer
        self.error = error
        self.calls: list[dict] = []

    def chat(
        self,
        model,
        messages,
        *,
        schema=None,
        options=None,
        keep_alive=None,
        timeout=None,
        cancel=None,
        on_token=None,
        think=None,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "schema": schema,
                "options": dict(options or {}),
                "keep_alive": keep_alive,
                "timeout": timeout,
                "think": think,
            }
        )
        if self.error is not None:
            raise self.error
        text = (
            self.answer
            if isinstance(self.answer, str)
            else json.dumps(self.answer, ensure_ascii=False)
        )
        return types.SimpleNamespace(text=text)


def _service(client: FakeClient, *, cache=None) -> warmup_context.WarmupService:
    """Сервис с подставным клиентом и моделью: реальный анализатор не нужен."""
    analyzer = types.SimpleNamespace(client=client, selected_model=lambda: "fake:perfect")
    return warmup_context.WarmupService(
        client=client,
        analyzer=analyzer,
        cache=cache if cache is not None else warmup_context.WarmupCache(size=4),
    )


@pytest.fixture(autouse=True)
def _warmup_environment(monkeypatch):
    """Прогрев включён, движок разрешён, память «в норме» — фон для всех тестов.

    Память подменяется модулем `resource_guard`, а не копией внутри сервиса: сам
    сервис импортирует наблюдатель внутри `_memory_tight`, и подмена модуля — это
    ровно та точка, куда он смотрит.
    """
    monkeypatch.setattr(config, "WARMUP_ENABLED", True)
    monkeypatch.setattr(config, "WARMUP_ENGINES", (ENGINE,))
    monkeypatch.setattr(resource_guard, "memory_state", lambda: {"state": "ok"})


def _word_stamps() -> list[WordStamp]:
    """«Префикс» и «цель» так, как их вернул бы Whisper."""
    return [WordStamp("префикс", 0.0, 0.5), WordStamp("цель", 0.6, 1.0)]


# --- элигибельность -----------------------------------------------------------
def test_short_replica_asks_llm_and_returns_prefix():
    """Короткая реплика — единственный случай, когда прогрев вообще спрашивают."""
    client = FakeClient()
    context = _service(client).build(target_text=TARGET, engine_id=ENGINE)

    assert warmup_context.eligibility_reason(TARGET, ENGINE) == ""
    assert context.prefix_text == PREFIX
    assert context.enabled is True
    assert context.reason == warmup_context.REASON_OK
    assert len(client.calls) == 1
    # Запрос уходит с целью, схемой ответа и жёстким лимитом времени: без схемы
    # модель ответила бы свободным текстом, и разбор стал бы догадкой.
    call = client.calls[0]
    assert TARGET in call["messages"][-1]["content"]
    assert call["schema"] == warmup_context.RESPONSE_SCHEMA
    assert call["timeout"] == config.WARMUP_LLM_TIMEOUT_SEC
    assert call["think"] is False


def test_long_replica_never_calls_llm():
    """Длинная реплика идёт обычным путём: тратить на неё LLM нечего."""
    client = FakeClient()
    long_text = "Это очень длинная реплика, которая заведомо превышает лимит прогрева, " * 2

    assert warmup_context.eligibility_reason(long_text, ENGINE) == warmup_context.REASON_TOO_LONG
    context = _service(client).build(target_text=long_text, engine_id=ENGINE)

    assert context.prefix_text is None
    assert context.reason == warmup_context.REASON_TOO_LONG
    assert client.calls == []


def test_disabled_flag_turns_warmup_off(monkeypatch):
    """Выключенный прогрев не должен стоить ни одного вызова модели."""
    client = FakeClient()

    monkeypatch.setattr(config, "WARMUP_ENABLED", False)
    assert (
        warmup_context.eligibility_reason(TARGET, ENGINE) == warmup_context.REASON_DISABLED
    )
    context = _service(client).build(target_text=TARGET, engine_id=ENGINE)
    assert context.prefix_text is None
    assert context.reason == warmup_context.REASON_DISABLED

    # Явный `enabled=False` сильнее включённой политики приложения: задача вправе
    # отказаться от прогрева, даже когда он разрешён глобально.
    monkeypatch.setattr(config, "WARMUP_ENABLED", True)
    context = _service(client).build(target_text=TARGET, engine_id=ENGINE, enabled=False)
    assert context.reason == warmup_context.REASON_DISABLED
    assert client.calls == []


def test_engine_outside_allowlist_disables_warmup(monkeypatch):
    """Список движков — настройка: XTTS отключается без правки пайплайна."""
    assert warmup_context.enabled_for_engine(ENGINE) is True
    assert warmup_context.enabled_for_engine("xtts") is False
    assert (
        warmup_context.eligibility_reason(TARGET, "xtts") == warmup_context.REASON_ENGINE
    )

    monkeypatch.setattr(config, "WARMUP_ENGINES", ("xtts",))
    assert warmup_context.eligibility_reason(TARGET, "xtts") == ""
    context = _service(FakeClient()).build(target_text=TARGET, engine_id="xtts")
    assert context.prefix_text == PREFIX


def test_llm_error_falls_back_to_plain_synthesis():
    """Упавшая LLM не должна ронять задачу: наружу летит только контекст без префикса."""
    client = FakeClient(error=RuntimeError("модель недоступна"))
    context = _service(client).build(target_text=TARGET, engine_id=ENGINE)

    assert context.prefix_text is None
    assert context.enabled is False
    assert context.reason == warmup_context.REASON_LLM_ERROR
    assert context.synthesis_text == TARGET


def test_llm_timeout_falls_back_to_plain_synthesis():
    """Таймаут LLM — обычный синтез с отдельной причиной, а не общая ошибка."""
    context = _service(FakeClient(error=TimeoutError("долго"))).build(
        target_text=TARGET, engine_id=ENGINE
    )

    assert context.prefix_text is None
    assert context.reason == warmup_context.REASON_LLM_TIMEOUT


def test_empty_llm_response_is_not_a_prefix():
    """Пустой ответ — это отсутствие прогрева, а не пустая строка в синтез-тексте."""
    for answer in (json.dumps({"prefix": ""}), ""):
        context = _service(FakeClient(answer=answer)).build(
            target_text=TARGET, engine_id=ENGINE
        )
        assert context.prefix_text is None
        assert context.reason == warmup_context.REASON_EMPTY_RESPONSE


# --- валидация префикса -------------------------------------------------------
def test_validate_prefix_rejects_markup():
    """Markdown и XML в речь не идут: TTS прочитал бы теги вслух."""
    for bad in ("<emotion>ура</emotion>", "**жирный**"):
        assert warmup_context.validate_prefix(bad, TARGET) == (
            "",
            warmup_context.REASON_MARKUP,
        )


def test_validate_prefix_rejects_repeats_of_target():
    """Префикс, целиком равный цели, дал бы «цель … цель» и обрезку не по тому месту."""
    assert warmup_context.validate_prefix(TARGET, TARGET) == (
        "",
        warmup_context.REASON_REPEATS_TARGET,
    )
    assert warmup_context.validate_prefix("да", TARGET) == (
        "",
        warmup_context.REASON_REPEATS_TARGET,
    )


def test_validate_prefix_strips_quotes_and_service_label():
    """Кавычки и приставка «PREFIX:» — обёртка модели, а не произносимый текст."""
    stripped, reason = warmup_context.validate_prefix("«Тише, сейчас начнём»", TARGET)
    assert (stripped, reason) == ("Тише, сейчас начнём.", warmup_context.REASON_OK)

    labelled, reason = warmup_context.validate_prefix("PREFIX: Тише, сейчас начнём", TARGET)
    assert (labelled, reason) == ("Тише, сейчас начнём.", warmup_context.REASON_OK)


def test_too_long_prefix_is_cut_by_sentence(monkeypatch):
    """Длинный осмысленный префикс обрезается по точке, а не по символу."""
    monkeypatch.setattr(config, "WARMUP_MAX_PREFIX_CHARS", 40)
    text = "Первое предложение тут. И дальше много лишних слов без конца"

    prefix, reason = warmup_context.validate_prefix(text, TARGET)

    assert reason == warmup_context.REASON_OK
    assert prefix == "Первое предложение тут."
    assert len(prefix) <= 40


def test_hopelessly_long_prefix_is_rejected(monkeypatch):
    """Одно слово длиннее лимита резать негде: обрывок слова звучал бы дефектом."""
    monkeypatch.setattr(config, "WARMUP_MAX_PREFIX_CHARS", 20)

    prefix, reason = warmup_context.validate_prefix("о" * 30, TARGET)

    assert prefix == ""
    assert reason == warmup_context.REASON_TOO_LONG_PREFIX


# --- инварианты цели и метаданных ---------------------------------------------
def test_target_is_never_replaced_or_returned_as_prefix():
    """Цель неприкосновенна: LLM её не переписывает и она не подменяет префикс."""
    context = warmup_context.WarmupContext(target_text=TARGET)
    assert context.synthesis_text == TARGET

    context.prefix_text = PREFIX
    assert context.synthesis_text == f"{PREFIX} {TARGET}"

    assert warmup_context.validate_prefix(TARGET, TARGET)[0] == ""


def test_prefix_is_not_replica_text_in_metadata():
    """В обычных метаданных префикса нет: пользователь видит только свою реплику."""
    context = warmup_context.WarmupContext(target_text=TARGET, prefix_text=PREFIX)

    public = context.to_dict(include_prefix=False)
    assert "warmup_prefix" not in public
    assert public["warmup_prefix_chars"] == len(PREFIX)
    assert public["warmup_used"] is True
    # Диагностике префикс нужен целиком — иначе непонятно, что ушло в модель.
    assert context.to_dict()["warmup_prefix"] == PREFIX


# --- кэш ----------------------------------------------------------------------
def test_second_build_hits_cache_and_skips_llm():
    """Одинаковая тройка текстов не должна стоить второго вызова модели."""
    cache = warmup_context.WarmupCache(size=2)
    client = FakeClient()
    service = _service(client, cache=cache)

    first = service.build(target_text=TARGET, engine_id=ENGINE)
    second = service.build(target_text=TARGET, engine_id=ENGINE)

    assert len(client.calls) == 1
    assert second.cache_hit is True
    assert second.prefix_text == first.prefix_text
    assert second.prefix_text == PREFIX


def test_cache_key_depends_on_prompt_version():
    """Правка prompt'а обязана обесценить старые префиксы без ручной чистки кэша."""
    first = warmup_context.cache_key(
        engine_id=ENGINE, previous_text=None, target_text=TARGET, next_text=None,
        prompt_version="1",
    )
    second = warmup_context.cache_key(
        engine_id=ENGINE, previous_text=None, target_text=TARGET, next_text=None,
        prompt_version="2",
    )
    again = warmup_context.cache_key(
        engine_id=ENGINE, previous_text=None, target_text=TARGET, next_text=None,
        prompt_version="1",
    )

    assert first != second
    assert first == again


def test_cache_evicts_oldest_and_respects_recency():
    """LRU на два места: переполнение не должно съедать только что спрошенный префикс."""
    cache = warmup_context.WarmupCache(size=2)
    cache.put("a", "раз")
    cache.put("b", "два")
    cache.put("c", "три")

    assert cache.get("a") is None
    assert (cache.get("b"), cache.get("c")) == ("два", "три")
    assert len(cache) == 2

    cache.get("b")  # «b» снова свежий — вытесняться должен «c»
    cache.put("d", "четыре")
    assert cache.get("c") is None
    assert cache.get("b") == "два"


def test_singletons_reset():
    """Сброс кэша и сервиса обязателен после смены настроек LLM, иначе течёт старое."""
    warmup_context.reset_cache()
    first_cache = warmup_context.get_cache()
    assert warmup_context.get_cache() is first_cache
    warmup_context.reset_cache()
    assert warmup_context.get_cache() is not first_cache

    warmup_context.reset_service()
    first_service = warmup_context.get_service()
    assert warmup_context.get_service() is first_service
    warmup_context.reset_service()
    assert warmup_context.get_service() is not first_service
    warmup_context.reset_service()


# --- память -------------------------------------------------------------------
def test_memory_pressure_skips_llm(monkeypatch):
    """Прогрев не должен вытеснять синтез: при критической памяти LLM не зовут."""
    client = FakeClient()
    service = _service(client)
    monkeypatch.setattr(resource_guard, "memory_state", lambda: {"state": "critical"})

    context = service.build(target_text=TARGET, engine_id=ENGINE)

    assert context.prefix_text is None
    assert context.reason == warmup_context.REASON_MEMORY
    assert client.calls == []


# --- граница цели в готовом аудио ---------------------------------------------
def test_locate_target_start_returns_first_target_word():
    """Граница берётся по таймстемпу первого слова цели, а не по доле длительности."""
    audio = sine(3.0, 220.0)

    boundary = warmup_context.locate_target_start(
        audio, SAMPLE_RATE, f"префикс {TARGET}", "цель", asr=lambda _audio: _word_stamps()
    )

    assert boundary is not None
    assert boundary.start_sec == pytest.approx(0.6 - PAD_BEFORE_SEC, abs=0.02)
    assert boundary.method == "asr"
    assert boundary.confidence == pytest.approx(1.0)
    assert boundary.matched_words == 1
    assert boundary.total_words == 1


def test_locate_target_start_rejects_low_confidence():
    """Не все слова цели подтверждены — резать нельзя, честнее синтезировать заново."""
    audio = sine(3.0, 220.0)

    partial = warmup_context.locate_target_start(
        audio, SAMPLE_RATE, "цель важная", "цель важная", asr=lambda _audio: _word_stamps()
    )
    strict = warmup_context.locate_target_start(
        audio, SAMPLE_RATE, f"префикс {TARGET}", "цель", asr=lambda _audio: _word_stamps(),
        min_confidence=1.5,
    )

    assert partial is None
    assert strict is None


def test_locate_target_start_survives_broken_asr():
    """Недоступное распознавание — отсутствие границы, а не падение задачи."""

    def broken(_audio):
        raise RuntimeError("Whisper недоступен")

    audio = sine(3.0, 220.0)

    assert warmup_context.locate_target_start(
        audio, SAMPLE_RATE, f"префикс {TARGET}", "цель", asr=broken
    ) is None
