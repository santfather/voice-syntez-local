"""Слой коротких реплик: класс реплики, контекст и план синтеза.

Зачем отдельный слой. Короткие реплики («Да.», «Привет!», «Как дела?») звучат
заметно хуже длинных: модели не хватает языкового и просодического контекста —
интонации, темпа, атаки начала фразы. Дать ей этот контекст можно только на слое
синтеза: менять сам текст реплики нельзя, пользователь должен услышать ровно то,
что подготовил.

Три правила, на которых держится модуль:

1. **Подготовленный текст неприкосновенен.** `final_text` (нормализация, «ё»,
   словарь, ударения) — единственный источник истины. Слой лишь строит из него
   `tts_synthesis_text` и никогда не пишет обратно ни в реплику, ни в её стадии.
2. **NORMAL не трогается.** Классификатор отвечает на вопрос «короткая ли
   реплика», и всё, что не `short`/`very_short`, идёт прежним путём без единого
   лишнего шага — иначе улучшение коротких фраз ухудшило бы длинные.
3. **Контекст не попадает в готовый WAV.** Если контекст добавлен, синтез обязан
   обрезаться до целевой реплики по надёжной границе; нет надёжной границы —
   стратегия откатывается на DIRECT, а не «примерно по времени» (см.
   `short_utterance_boundary.py`).

Модуль намеренно ничего не знает ни про torch, ни про движки: здесь только текст,
классы и планы. Благодаря этому всё поведение проверяется тестами без моделей, а
движок получает уже готовую строку.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field, replace

import numpy as np

from . import config

logger = logging.getLogger(__name__)

# --- классы реплик ------------------------------------------------------------
CLASS_VERY_SHORT = "very_short"
CLASS_SHORT = "short"
CLASS_NORMAL = "normal"
CLASSES = (CLASS_VERY_SHORT, CLASS_SHORT, CLASS_NORMAL)

# Слово — последовательность букв или цифр. Дефис разделяет слова («что-то» — два
# слова): для оценки длины реплики это вернее, чем считать «что-то» одним словом,
# потому что модель произносит его как два.
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
# Знак ударения RUAccent — разметка, а не часть слова и не разделитель:
# «Прив+ет!» — одно слово, а не два. Без этой замены подготовленный для F5 текст
# классифицировался бы как вдвое более длинный, и короткая реплика не попала бы в
# короткую обработку ровно там, где она нужна.
_ACCENT_MARK = "+"


def _without_accents(text: str) -> str:
    return (text or "").replace(_ACCENT_MARK, "")


def count_words(text: str) -> int:
    """Сколько слов в тексте — по буквенно-цифровым последовательностям."""
    return len(_WORD_RE.findall(_without_accents(text)))


def count_chars(text: str) -> int:
    """Длина текста без крайних пробелов и без знаков ударения: считаем произносимое."""
    return len(_without_accents(text).strip())


@dataclass(frozen=True)
class ShortThresholds:
    """Пороги короткой реплики. Конфигурируемы: §2 требует настраиваемости.

    Два признака, а не один: слова — основная ось, знаки уточняют её на шаг
    вверх для текстов, длинных при своём числе слов («Международный
    авиационно-космический» — два слова, но уже не «Привет!»).
    """

    very_short_words: int = config.SHORT_UTTERANCE_VERY_SHORT_WORDS
    short_words: int = config.SHORT_UTTERANCE_SHORT_WORDS
    very_short_chars: int = config.SHORT_UTTERANCE_VERY_SHORT_CHARS
    short_chars: int = config.SHORT_UTTERANCE_SHORT_CHARS

    def to_dict(self) -> dict:
        return {
            "very_short_words": self.very_short_words,
            "short_words": self.short_words,
            "very_short_chars": self.very_short_chars,
            "short_chars": self.short_chars,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> ShortThresholds:
        """Пороги из запроса/настроек: нечисло и отрицательное отбрасываются.

        Опечатка в интерфейсе не должна превращать все реплики в «короткие»:
        неизвестное значение оставляет порог по умолчанию.
        """
        raw = raw or {}
        values: dict[str, int] = {}
        for name in ("very_short_words", "short_words", "very_short_chars", "short_chars"):
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = int(value)
            if number > 0:
                values[name] = number
        return replace(cls(), **values)


@dataclass(frozen=True)
class UtteranceClass:
    """Класс реплики: короткая она или обычная, и чем это решено."""

    kind: str
    word_count: int
    char_count: int

    @property
    def is_short(self) -> bool:
        return self.kind in (CLASS_SHORT, CLASS_VERY_SHORT)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "word_count": self.word_count, "char_count": self.char_count}


def classify_utterance(text: str, thresholds: ShortThresholds | None = None) -> UtteranceClass:
    """Классифицирует реплику по числу слов и знаков.

    Слова — основная ось (реплика из двух слов коротка независимо от длины слов),
    знаки — уточнение: текст, длинный для своего числа слов, переводится на шаг
    выше, потому что звучит он уже не как короткая реплика. Обратного хода нет:
    длинная по словам фраза не станет короткой из-за малого числа знаков.

    Ошибка в сторону «короче» дешевле: короткая обработка всё равно откатывается
    на DIRECT, если не может дать контекст, — а обратная ошибка оставила бы
    проблемную реплику без обработки.
    """
    limits = thresholds or ShortThresholds()
    words = count_words(text)
    chars = count_chars(text)
    if words <= limits.very_short_words:
        kind = CLASS_VERY_SHORT
    elif words <= limits.short_words:
        kind = CLASS_SHORT
    else:
        kind = CLASS_NORMAL
    if kind == CLASS_VERY_SHORT and chars > limits.very_short_chars:
        kind = CLASS_SHORT
    elif kind == CLASS_SHORT and chars > limits.short_chars:
        kind = CLASS_NORMAL
    return UtteranceClass(kind=kind, word_count=words, char_count=chars)


# --- стратегии ----------------------------------------------------------------
STRATEGY_DIRECT = "direct"
STRATEGY_PUNCTUATION = "punctuation"
STRATEGY_SAME_SPEAKER_CONTEXT = "same_speaker_context"
STRATEGY_SYNTHETIC_CONTEXT = "synthetic_context"
STRATEGY_BATCH_AND_CROP = "batch_and_crop"
# `auto` — выбор по движку из измеренной политики (см. `default_strategy`).
STRATEGY_AUTO = "auto"

# Порядок — от самой безопасной к самой рискованной: он же порядок отчёта
# benchmark'а и подсказка в интерфейсе.
STRATEGIES = (
    STRATEGY_DIRECT,
    STRATEGY_PUNCTUATION,
    STRATEGY_SAME_SPEAKER_CONTEXT,
    STRATEGY_SYNTHETIC_CONTEXT,
    STRATEGY_BATCH_AND_CROP,
)
SELECTABLE_STRATEGIES = (STRATEGY_AUTO, *STRATEGIES)
# Стратегии, добавляющие контекст: их результат обязательно обрезается до цели.
CONTEXT_STRATEGIES = (
    STRATEGY_SAME_SPEAKER_CONTEXT,
    STRATEGY_SYNTHETIC_CONTEXT,
    STRATEGY_BATCH_AND_CROP,
)

# --- variant'ы для benchmark (§6, §30) ----------------------------------------
# Пунктуация несёт интонацию, поэтому «заменить ! на .» — эксперимент, а не
# решение: варианты живут здесь, чтобы benchmark мог их сравнить, а производство
# по умолчанию пользовалось исходной пунктуацией.
PUNCTUATION_AS_IS = "as_is"
PUNCTUATION_PERIOD = "period"
PUNCTUATION_ELLIPSIS = "ellipsis"
PUNCTUATION_VARIANTS = (PUNCTUATION_AS_IS, PUNCTUATION_PERIOD, PUNCTUATION_ELLIPSIS)

# Как приставлять контекст к цели (§30: A — только цель, B — префикс, C — суффикс,
# D — оба). Проверяется benchmark'ом; по умолчанию префикс: он даёт модели
# «разбег» перед короткой фразой, на который и жалуется наблюдаемое качество.
SIDE_PREFIX = "prefix"
SIDE_SUFFIX = "suffix"
SIDE_BOTH = "both"
CONTEXT_SIDES = (SIDE_PREFIX, SIDE_SUFFIX, SIDE_BOTH)

# Откуда взят контекст — для метаданных куска (§24).
SOURCE_SAME_SPEAKER_PREVIOUS = "same_speaker_previous"
SOURCE_SAME_SPEAKER_NEXT = "same_speaker_next"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_GROUP = "group"


def strip_accents(text: str) -> str:
    """Убирает разметку ударений RUAccent.

    Нужно для движков, которые «+» не понимают (XTTS прочитала бы его вслух):
    контекст может прийти из реплики, подготовленной для F5, и попасть в синтез
    другого движка. Тот же смысл у `preview_text(supports_accents=False)`, но
    здесь текст уже подготовлен, и повторять весь preprocessing нельзя.
    """
    return (text or "").replace("+", "")


# --- контекст -----------------------------------------------------------------
@dataclass(frozen=True)
class ShortUtteranceContext:
    """Соседи короткой реплики — источник контекста для синтеза.

    `previous_text`/`next_text` — соседи вообще, `same_speaker_*` — только того же
    спикера. Различие принципиальное: реплику другого голоса нельзя прочитать
    текущим голосом (это была бы речь не того человека), поэтому кросс-спикерный
    контекст остаётся текстовым ориентиром для benchmark'а и не участвует в
    производственных стратегиях без отдельной проверки (§8).

    Все тексты — **подготовленные** (`final_text`), а не исходные: контекст обязан
    пройти тот же путь, что и цель, иначе ударения и словарь применялись бы только
    к половине того, что услышит модель.
    """

    target_index: int
    target_text: str
    speaker: str = ""
    voice: str = ""
    previous_text: str | None = None
    next_text: str | None = None
    same_speaker_previous: str | None = None
    same_speaker_next: str | None = None
    utterance: UtteranceClass | None = None

    def to_dict(self) -> dict:
        return {
            "target_index": self.target_index,
            "target_text": self.target_text,
            "speaker": self.speaker,
            "voice": self.voice,
            "previous_text": self.previous_text,
            "next_text": self.next_text,
            "same_speaker_previous": self.same_speaker_previous,
            "same_speaker_next": self.same_speaker_next,
            "utterance": self.utterance.to_dict() if self.utterance else None,
        }


def _prepared_text(replica) -> str:
    """Подготовленный текст реплики: `final_text`, иначе исходный.

    `final_text` — то, что пользователь подтвердил; если его нет (разовая задача
    без анализа), берётся исходный текст: слой не готовит текст сам и не должен —
    это дело пайплайна.
    """
    prepared = (getattr(replica, "final_text", None) or "").strip()
    return prepared or (getattr(replica, "text", "") or "").strip()


def _speaker_of(replica) -> str:
    """Ключ спикера реплики: им решается, чей это контекст."""
    return str(getattr(replica, "voice", "") or "")


def _nearest_same_speaker(
    prepared: list[str], speakers: list[str], index: int, step: int, window: int
) -> str | None:
    """Ближайшая реплика того же спикера в пределах окна — до или после цели.

    Ищем не только вплотную: в диалоге соседняя реплика почти всегда принадлежит
    другому персонажу («— Ты уже пришёл? — Да.»), и контекст того же голоса
    оказывается на расстоянии двух-трёх реплик. Именно он и нужен: это речь того
    же человека, её и может прочитать текущий голос.
    """
    position = index + step
    while 0 <= position < len(prepared) and (position - index) * step <= window:
        if not prepared[position]:
            position += step
            continue
        if speakers[position] == speakers[index]:
            return prepared[position]
        position += step
    return None


def build_contexts(
    replicas,
    *,
    thresholds: ShortThresholds | None = None,
    window: int | None = None,
) -> list[ShortUtteranceContext]:
    """Собирает контекст каждой реплики, не меняя ни одну из них.

    Контекст ищется только среди **подготовленного** текста соседей: то, что не
    пройдёт в модель как цель, не должно попадать в неё и как контекст.
    Кросс-спикерные соседи сохраняются отдельными полями — их использует
    benchmark, чтобы проверить гипотезу §8, — но в производственных стратегиях
    они не участвуют: чужой голос не может прочитать реплику другого персонажа.

    `window` ограничивает, насколько далеко ищется реплика того же спикера: текст
    из начала сцены — уже не контекст для текущей фразы.
    """
    items = list(replicas)
    prepared = [_prepared_text(replica) for replica in items]
    speakers = [_speaker_of(replica) for replica in items]
    limit = config.SHORT_UTTERANCE_CONTEXT_WINDOW if window is None else window
    contexts: list[ShortUtteranceContext] = []
    for index, replica in enumerate(items):
        contexts.append(
            ShortUtteranceContext(
                target_index=index,
                target_text=prepared[index],
                speaker=speakers[index],
                voice=str(getattr(replica, "voice_id", "") or ""),
                previous_text=prepared[index - 1] if index > 0 else None,
                next_text=prepared[index + 1] if index + 1 < len(prepared) else None,
                same_speaker_previous=_nearest_same_speaker(
                    prepared, speakers, index, -1, limit
                ),
                same_speaker_next=_nearest_same_speaker(prepared, speakers, index, 1, limit),
                utterance=classify_utterance(prepared[index], thresholds),
            )
        )
    return contexts


# --- план синтеза -------------------------------------------------------------
@dataclass(frozen=True)
class SynthesisPlan:
    """Что именно уйдёт в движок и откуда взялся контекст.

    `target_text` — подготовленная реплика; она же остаётся в проекте и в
    `final_text`. `synthesis_text` — то, что читает модель: либо цель, либо цель с
    контекстом. Обрезка нужна ровно тогда, когда эти строки различаются.
    """

    target_text: str
    synthesis_text: str
    strategy: str = STRATEGY_DIRECT
    context_text: str = ""
    context_source: str = ""
    utterance_class: str = CLASS_NORMAL
    side: str = SIDE_PREFIX
    # Причина отказа от стратегии (например, «нет надёжной границы»): нужна
    # метаданным и логу, чтобы «почему direct» не приходилось угадывать.
    note: str = ""

    @property
    def needs_crop(self) -> bool:
        return self.synthesis_text != self.target_text

    @property
    def synthesis_hash(self) -> str:
        """Короткий отпечаток синтез-текста: воспроизводимость без хранения текста."""
        return hashlib.sha1(self.synthesis_text.encode("utf-8")).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "tts_target_text": self.target_text,
            "tts_context_text": self.context_text,
            "tts_synthesis_text": self.synthesis_text,
            "short_utterance_strategy": self.strategy,
            "short_utterance_context_source": self.context_source,
            "short_utterance_class": self.utterance_class,
            "short_utterance_side": self.side,
            "short_utterance_note": self.note,
            "synthesis_text_hash": self.synthesis_hash,
        }


def normalize_punctuation(text: str) -> str:
    """Приводит пунктуацию к однозначному виду, не меняя её смысла.

    Схлопывает повторы («!!!» → «!»), убирает пробелы перед знаком и оставляет
    терминальный знак. Это **не** замена «!» на «.»: интонация восклицания несёт
    смысл, и менять её можно только по результатам benchmark (см. варианты).
    """
    cleaned = re.sub(r"\s+([,.!?;:…])", r"\1", text or "")
    cleaned = re.sub(r"([!?…])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\.{2,}", "…", cleaned)
    cleaned = cleaned.strip()
    if cleaned and cleaned[-1] not in ".!?…":
        cleaned += "."
    return cleaned


def apply_punctuation_variant(text: str, variant: str) -> str:
    """Вариант пунктуации для benchmark (§6). В производстве — `as_is`."""
    if variant == PUNCTUATION_PERIOD:
        return re.sub(r"[!?…]+", ".", text or "").strip()
    if variant == PUNCTUATION_ELLIPSIS:
        base = re.sub(r"[!?…]+", "", text or "").strip()
        return (base + "…") if base else base
    return text


def _same_speaker_context(context: ShortUtteranceContext) -> tuple[str, str]:
    """Первый доступный одноимённый контекст: предыдущая реплика важнее следующей."""
    if context.same_speaker_previous:
        return context.same_speaker_previous, SOURCE_SAME_SPEAKER_PREVIOUS
    if context.same_speaker_next:
        return context.same_speaker_next, SOURCE_SAME_SPEAKER_NEXT
    return "", ""


def _join(context_text: str, target: str, side: str) -> str:
    if not context_text:
        return target
    if side == SIDE_SUFFIX:
        return f"{target} {context_text}"
    if side == SIDE_BOTH:
        return f"{context_text} {target} {context_text}"
    return f"{context_text} {target}"


def build_plan(
    context: ShortUtteranceContext,
    *,
    strategy: str = STRATEGY_DIRECT,
    supports_accents: bool = True,
    carrier: str | None = None,
    side: str | None = None,
    punctuation_variant: str = PUNCTUATION_AS_IS,
    allow_context: bool = True,
) -> SynthesisPlan:
    """Строит план синтеза для одной реплики.

    Контекст всегда берётся из подготовленных текстов соседей и, если движок не
    понимает ударения, очищается от «+»: иначе XTTS прочитала бы знак вслух
    (§15). `allow_context=False` — откат на DIRECT: его ставит вызывающий, когда
    не может гарантировать надёжную границу для обрезки (§31).
    """
    target = context.target_text
    utterance = context.utterance or classify_utterance(target)
    # Сторона контекста по умолчанию зависит от стратегии: синтетический carrier
    # обрамляет цель с двух сторон (§7), а реплика того же спикера идёт префиксом —
    # именно начало короткой фразы и страдает от отсутствия «разбега» (§30).
    if side is None:
        side = SIDE_BOTH if strategy == STRATEGY_SYNTHETIC_CONTEXT else SIDE_PREFIX
    base = SynthesisPlan(
        target_text=target,
        synthesis_text=target,
        strategy=strategy,
        utterance_class=utterance.kind,
        side=side,
    )

    if not target or strategy == STRATEGY_DIRECT:
        return base

    if strategy == STRATEGY_PUNCTUATION:
        # Нормализация (схлопнуть «!!!», поставить терминальный знак) — суть
        # стратегии; вариант (заменить «!» на «.», добавить «…») — отдельный
        # эксперимент benchmark'а, потому что пунктуация несёт интонацию (§6).
        normalized = normalize_punctuation(target)
        variant = apply_punctuation_variant(normalized, punctuation_variant)
        return replace(base, synthesis_text=variant, note="пунктуация")

    if strategy in CONTEXT_STRATEGIES and not allow_context:
        return replace(
            base, strategy=STRATEGY_DIRECT, note="нет надёжной границы для обрезки"
        )

    if strategy == STRATEGY_SAME_SPEAKER_CONTEXT:
        raw, source = _same_speaker_context(context)
        if not raw:
            return replace(base, strategy=STRATEGY_DIRECT, note="нет реплики того же спикера рядом")
    elif strategy == STRATEGY_SYNTHETIC_CONTEXT:
        raw, source = (carrier or config.SHORT_UTTERANCE_SYNTHETIC_CARRIER), SOURCE_SYNTHETIC
    elif strategy == STRATEGY_BATCH_AND_CROP:
        # Текст группы подставляет вызывающий (`build_batches`): здесь только
        # фиксируется, что реплика идёт в общем синтезе.
        return replace(base, synthesis_text=target, context_source=SOURCE_GROUP)
    else:
        raise ValueError(f"Неизвестная стратегия короткой реплики: {strategy}")

    context_text = raw if supports_accents else strip_accents(raw)
    synthesis = _join(context_text, target, side)
    return replace(base, synthesis_text=synthesis, context_text=context_text, context_source=source)


@dataclass(frozen=True)
class ShortBatch:
    """Группа подряд идущих коротких реплик одного спикера (§10).

    Синтез одной строкой даёт модели настоящий авторский контекст — это лучше
    синтетического carrier'а. Границы кусков после общего синтеза определяются по
    реальным таймстемпам, а не по расчёту длительности (§11).
    """

    indexes: tuple[int, ...]
    speaker: str
    texts: tuple[str, ...] = field(default=())

    @property
    def text(self) -> str:
        return " ".join(text for text in self.texts if text)

    def to_dict(self) -> dict:
        return {"indexes": list(self.indexes), "speaker": self.speaker, "text": self.text}


def build_batches(
    contexts: list[ShortUtteranceContext],
    *,
    max_replicas: int | None = None,
    allow_cross_speaker: bool = False,
) -> dict[int, ShortBatch]:
    """Группирует подряд идущие короткие реплики одного спикера.

    Возвращает отображение «индекс реплики → её группа», включая группы из одной
    реплики (их обрабатывать не нужно, но вызывающему так проще). Реплики разных
    спикеров не объединяются никогда: это был бы один голос на два персонажа
    (§10), поэтому `allow_cross_speaker` существует только для benchmark'а.
    """
    limit = max_replicas or config.SHORT_UTTERANCE_BATCH_MAX
    groups: list[ShortBatch] = []
    current: list[int] = []
    for context in contexts:
        short = (context.utterance or classify_utterance(context.target_text)).is_short
        index = context.target_index
        if not short:
            if current:
                groups.append(_make_batch(contexts, current))
                current = []
            continue
        if current:
            previous = contexts[current[-1]]
            same = previous.speaker == context.speaker or allow_cross_speaker
            if not same or len(current) >= limit:
                groups.append(_make_batch(contexts, current))
                current = []
        current.append(index)
    if current:
        groups.append(_make_batch(contexts, current))

    mapping: dict[int, ShortBatch] = {}
    for group in groups:
        for index in group.indexes:
            mapping[index] = group
    return mapping


def _make_batch(contexts: list[ShortUtteranceContext], indexes: list[int]) -> ShortBatch:
    by_index = {context.target_index: context for context in contexts}
    return ShortBatch(
        indexes=tuple(indexes),
        speaker=by_index[indexes[0]].speaker if indexes else "",
        texts=tuple(by_index[index].target_text for index in indexes),
    )


def default_strategy(engine_id: str) -> str:
    """Производственная стратегия для движка — из измеренной политики.

    Политика заполняется по результатам benchmark (`config`), а до него отдаёт
    DIRECT: включать недоказанную стратегию по умолчанию нельзя (§29, Phase 4).
    """
    return config.SHORT_UTTERANCE_ENGINE_STRATEGIES.get(engine_id, STRATEGY_DIRECT)


def resolve_strategy(engine_id: str, requested: str) -> str:
    """Превращает `auto` (и пустое значение) в стратегию для конкретного движка."""
    if not requested or requested == STRATEGY_AUTO:
        return default_strategy(engine_id)
    if requested not in STRATEGIES:
        raise ValueError(f"Неизвестная стратегия короткой реплики: {requested}")
    return requested


# --- short QA (§20, §21) ------------------------------------------------------
# Причины повтора — коды, а не фразы: их читает и лог, и интерфейс (переводит в
# подписи), и тесты (сравнивают точно).
REASON_EMPTY = "empty"
REASON_SILENCE = "silence"
REASON_DURATION_SHORT = "duration_short"
REASON_DURATION_LONG = "duration_long"
REASON_REPETITION = "repetition"
REASON_MISSING_WORDS = "missing_words"
REASON_EXTRA_WORDS = "extra_words"

# Русские подписи — для лога и сообщения в интерфейсе.
REASON_TITLES = {
    REASON_EMPTY: "движок вернул пустое аудио",
    REASON_SILENCE: "почти вся реплика — тишина",
    REASON_DURATION_SHORT: "слишком короткое аудио для этого текста",
    REASON_DURATION_LONG: "слишком длинное аудио: речь растянута или повторяется",
    REASON_REPETITION: "слово повторяется в озвучке",
    REASON_MISSING_WORDS: "в озвучке не слышно части слов",
    REASON_EXTRA_WORDS: "в озвучке слышны лишние слова",
    # Причины отбора `qa_screening` приходят как есть: они уже названы.
    "too_short": "аудио короче нижней границы",
    "level": "уровень вне допустимого диапазона",
    "clipping": "перегруз: пик упирается в потолок",
}


def describe_short_reasons(reasons: list[str]) -> str:
    """Причины одной строкой по-русски."""
    return ", ".join(REASON_TITLES.get(reason, reason) for reason in reasons)


def expected_duration(chars: int, *, chars_per_sec: float | None = None) -> float:
    """Ожидаемая длительность реплики по длине текста.

    Одно число на все короткие реплики неверно: «Да.» и «До встречи!» различаются
    втрое. Скорость берётся из того же ориентира, что у отбора кусков
    (`QA_SCREEN_CHARS_PER_SEC`), — второй ориентир однажды разошёлся бы с первым.
    """
    rate = chars_per_sec or config.SHORT_UTTERANCE_CHARS_PER_SEC
    return max(chars, 1) / max(rate, 1.0)


@dataclass(frozen=True)
class ShortVerdict:
    """Вердикт короткой реплики: годна ли она и почему нет."""

    ok: bool
    reasons: tuple[str, ...] = ()
    duration_sec: float = 0.0
    expected_sec: float = 0.0
    wer: float | None = None
    transcription: str = ""

    @property
    def score(self) -> tuple[int, float]:
        """Ключ выбора лучшей попытки: меньше причин, затем ниже WER.

        Знаки у обоих слагаемых отрицательные, потому что выбирается максимум:
        «−причины» и «−WER» — так больше значит лучше.
        """
        return (-len(self.reasons), -(self.wer if self.wer is not None else 1.0))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "duration_sec": round(self.duration_sec, 3),
            "expected_sec": round(self.expected_sec, 3),
            "wer": self.wer,
        }


def _words(text: str) -> list[str]:
    """Слова для сравнения «что просили» и «что услышали»."""
    return [word.lower() for word in _WORD_RE.findall(_without_accents(text))]


def check_chunk(
    chunk: np.ndarray,
    expected_text: str,
    *,
    transcription: str | None = None,
    wer: float | None = None,
    min_duration_sec: float | None = None,
) -> ShortVerdict:
    """Короткая реплика: дешёвый отбор + проверки, специфичные для короткого текста.

    Почему не хватает обычного QA. На «Да.» одна ошибка распознавания даёт WER 1.0,
    и порог по WER либо валит нормальную реплику, либо пропускает «Да, да, да…»;
    поэтому причины проверяются отдельно и по-человечески: пустое аудио, тишина,
    аномальная длительность (в любую сторону), повтор слова, потерянные и лишние
    слова. Расшифровка не обязательна: без неё проверяются waveform и длительность,
    а с ней — ещё и слова. Это позволяет обойтись без лишнего запуска Whisper там,
    где он не нужен (см. `_synthesize_checked`).

    Порог длительности считается от текста, а не задаётся одним числом для всех
    реплик: «Да.» и «До встречи!» должны иметь разные границы.
    """
    from . import qa_screening

    y = np.asarray(chunk, dtype=np.float32).reshape(-1)
    duration = y.size / 24000.0
    expected = expected_duration(count_chars(expected_text))
    reasons: list[str] = []

    screening = qa_screening.screen_chunk(y, expected_text)
    for reason in screening.reasons:
        if reason == qa_screening.REASON_REPEAT:
            reasons.append(REASON_REPETITION)
        else:
            reasons.append(reason)

    floor = config.SHORT_UTTERANCE_MIN_DURATION_SEC if min_duration_sec is None else min_duration_sec
    lower = max(floor, expected * config.SHORT_UTTERANCE_MIN_DURATION_RATIO)
    if duration < lower and REASON_DURATION_SHORT not in reasons and REASON_EMPTY not in reasons:
        reasons.append(REASON_DURATION_SHORT)
    if duration > expected * config.QA_SCREEN_LONG_RATIO and REASON_DURATION_LONG not in reasons:
        reasons.append(REASON_DURATION_LONG)

    if transcription is not None and transcription.strip():
        target_words = _words(expected_text)
        heard_words = _words(transcription)
        heard_set = set(heard_words)
        target_set = set(target_words)
        if any(word not in heard_set for word in target_words):
            reasons.append(REASON_MISSING_WORDS)
        if any(word not in target_set for word in heard_words):
            reasons.append(REASON_EXTRA_WORDS)
        # «Да, да, да…»: слово, которое в тексте встречается один раз, а в озвучке
        # повторяется — самый частый дефект коротких реплик.
        for word in target_words:
            if heard_words.count(word) >= 3 and target_words.count(word) <= 1:
                reasons.append(REASON_REPETITION)
                break

    # Порядок причин стабилен: один и тот же дефект даёт одну и ту же строку.
    unique = tuple(dict.fromkeys(reasons))
    return ShortVerdict(
        ok=not unique,
        reasons=unique,
        duration_sec=duration,
        expected_sec=expected,
        wer=wer,
        transcription=transcription or "",
    )


def should_retry_short(*, attempt: int, max_attempts: int, verdict: ShortVerdict | None) -> bool:
    """Нужен ли повтор короткой реплики (§19).

    Повтор даётся только на проваленном вердикте и только в пределах лимита:
    генерация «десятков вариантов» запрещена — это не решение проблемы, а её
    маскировка (`§18`).
    """
    if verdict is None or verdict.ok:
        return False
    # Попытки нумеруются с единицы: «повтор после нулевой» смысла не имеет, а
    # нулевой лимит не должен превращаться в бесконечный цикл.
    return 1 <= attempt < max(1, max_attempts)


def best_verdict(candidates: list[ShortVerdict]) -> int:
    """Индекс лучшей попытки: меньше причин, затем ниже WER."""
    if not candidates:
        raise ValueError("Нет ни одной попытки")
    return max(range(len(candidates)), key=lambda index: candidates[index].score)
