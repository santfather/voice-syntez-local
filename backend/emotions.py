"""Эмоция реплики: словарь значений и правила выбора действующего (UPDATE 2 §3–§6).

Эмоция — это **метаданные**, а не слово в тексте. Ни один служебный маркер не
попадает ни в `source_text`, ни в `final_text`: синтезу уходит ровно тот текст,
который пользователь написал и подтвердил, а эмоция едет рядом отдельным полем.
Это не стилистическое требование, а защита от класса дефектов: тег, попавший в
текст, читается моделью вслух, портит словарь и расходует контекст.

Слой эмоции отвечает только за **выбор референса** (просодический conditioning) и
ничего не исправляет в аудио: если слово обрезано, это лечит не эмоция (§54).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Минимальный production-набор (§3.1). Порядок — он же порядок в интерфейсе.
EMOTION_AUTO = "AUTO"
EMOTION_NEUTRAL = "NEUTRAL"
EMOTION_QUESTION = "QUESTION"
EMOTION_DELIGHT = "DELIGHT"
EMOTION_SURPRISE = "SURPRISE"
EMOTION_FEAR = "FEAR"

# Значения, которые пользователь может выбрать руками (AUTO — «пусть решает LLM»).
SELECTABLE_EMOTIONS: tuple[str, ...] = (
    EMOTION_AUTO,
    EMOTION_NEUTRAL,
    EMOTION_QUESTION,
    EMOTION_DELIGHT,
    EMOTION_SURPRISE,
    EMOTION_FEAR,
)
# Значения, которые имеют смысл как конкретный референс: `AUTO` сюда не входит —
# это не эмоция, а способ её выбрать.
EMOTIONS: tuple[str, ...] = SELECTABLE_EMOTIONS[1:]
# Эмоции, для которых отдельного референса может не быть: отсутствие не блокирует
# синтез, а даёт NEUTRAL с пометкой об откате (§9).
OPTIONAL_EMOTIONS: tuple[str, ...] = (
    EMOTION_QUESTION,
    EMOTION_DELIGHT,
    EMOTION_SURPRISE,
    EMOTION_FEAR,
)

EMOTION_TITLES = {
    EMOTION_AUTO: "Авто",
    EMOTION_NEUTRAL: "Нейтрально",
    EMOTION_QUESTION: "Вопрос",
    EMOTION_DELIGHT: "Восторг",
    EMOTION_SURPRISE: "Удивление",
    EMOTION_FEAR: "Испуг",
}


def normalize_emotion(value: object, *, default: str = EMOTION_NEUTRAL) -> str:
    """Приводит значение к известной эмоции; неизвестное — `default`.

    Свободная строка из LLM или из старого запроса не должна становиться новым
    значением словаря: иначе «emotion» из ответа модели попадала бы в UI как есть.
    """
    text = str(value or "").strip().upper()
    if text in SELECTABLE_EMOTIONS:
        return text
    return default


def is_emotion(value: object) -> bool:
    return str(value or "").strip().upper() in EMOTIONS


def emotion_effective(detected: object, override: object) -> str:
    """Действующая эмоция: ручной выбор важнее автоматического (§6).

    Порядок именно такой: `override` → `detected` → `NEUTRAL`. Выключенный LLM не
    оставляет реплику без эмоции — она получает NEUTRAL, и приложение работает
    дальше без модели (§6).
    """
    chosen = normalize_emotion(override, default="") if override else ""
    if chosen and chosen != EMOTION_AUTO:
        return chosen
    detected_value = normalize_emotion(detected, default="") if detected else ""
    if detected_value and detected_value != EMOTION_AUTO:
        return detected_value
    return EMOTION_NEUTRAL


# --- детерминированная подсказка ---------------------------------------------
# Слова, по которым вопрос или восклицание видно без модели. Это **не** замена
# LLM: правило срабатывает только на очевидном и только когда модель недоступна
# (§6). Ошибиться в сторону NEUTRAL безопаснее, чем угадать эмоцию неверно.
_QUESTION_WORDS = (
    "разве",
    "неужели",
    "правда",
    "почему",
    "зачем",
    "кто",
    "что",
    "где",
    "когда",
    "как",
    "какой",
    "какая",
    "какие",
    "сколько",
    "можно",
    "могу",
    "хочешь",
    "уверен",
)
_DELIGHT_WORDS = ("ура", "победа", "получилось", "супер", "здорово", "отлично", "класс")
_SURPRISE_WORDS = ("невероятно", "серьёзно", "неужели", "ого", "ого-го", "вот это")
_FEAR_WORDS = ("страшно", "боюсь", "испуг", "опасно", "тише", "помогите")

_WORD_RE = re.compile(r"[а-яёa-z0-9-]+", re.IGNORECASE)


@dataclass(frozen=True)
class HeuristicEmotion:
    """Результат детерминированной подсказки: эмоция и уверенность."""

    emotion: str
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "primary": self.emotion,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


def heuristic_emotion(text: str) -> HeuristicEmotion:
    """Эмоция по самому тексту, без LLM (§6, §33).

    Уверенность намеренно низкая: это подсказка на случай недоступной модели, и
    выдавать её за результат анализа нельзя. Вопрос определяется знаком вопроса,
    восклицание — знаком и словом, потому что «!» сам по себе не отличает восторг
    от испуга.
    """
    clean = str(text or "").strip()
    if not clean:
        return HeuristicEmotion(EMOTION_NEUTRAL, 0.2, "пустой текст")
    words = [word.lower() for word in _WORD_RE.findall(clean.replace("+", ""))]
    word_set = set(words)
    if "?" in clean or word_set & set(_QUESTION_WORDS):
        return HeuristicEmotion(EMOTION_QUESTION, 0.5, "вопрос по знаку или слову")
    if "!" in clean:
        if word_set & set(_FEAR_WORDS):
            return HeuristicEmotion(EMOTION_FEAR, 0.4, "восклицание со словом испуга")
        if word_set & set(_DELIGHT_WORDS):
            return HeuristicEmotion(EMOTION_DELIGHT, 0.45, "восклицание со словом радости")
        if word_set & set(_SURPRISE_WORDS):
            return HeuristicEmotion(EMOTION_SURPRISE, 0.4, "восклицание со словом удивления")
        return HeuristicEmotion(EMOTION_SURPRISE, 0.25, "восклицание без уточняющих слов")
    if word_set & set(_FEAR_WORDS):
        return HeuristicEmotion(EMOTION_FEAR, 0.35, "слово испуга")
    return HeuristicEmotion(EMOTION_NEUTRAL, 0.3, "явных признаков нет")


__all__ = [
    "EMOTIONS",
    "EMOTION_AUTO",
    "EMOTION_DELIGHT",
    "EMOTION_FEAR",
    "EMOTION_NEUTRAL",
    "EMOTION_QUESTION",
    "EMOTION_SURPRISE",
    "EMOTION_TITLES",
    "OPTIONAL_EMOTIONS",
    "SELECTABLE_EMOTIONS",
    "HeuristicEmotion",
    "emotion_effective",
    "heuristic_emotion",
    "is_emotion",
    "normalize_emotion",
]
