"""Эмоция реплики: словарь значений и правила выбора действующего (UPDATE 2 §3–§6,
UPDATE 3 §10–§13).

Эмоция — это **метаданные**, а не слово в тексте. Ни один служебный маркер не
попадает ни в `source_text`, ни в `final_text`: синтезу уходит ровно тот текст,
который пользователь написал и подтвердил, а эмоция едет рядом отдельным полем.
Это не стилистическое требование, а защита от класса дефектов: тег, попавший в
текст, читается моделью вслух, портит словарь и расходует контекст.

Слой эмоции отвечает только за **выбор референса** (просодический conditioning) и
ничего не исправляет в аудио: если слово обрезано, это лечит не эмоция (§54).

Словарей два, и они намеренно разные:

* `PROFILE_KEYS` — то, что может быть **записано** как референс-профиль голоса
  (UPDATE 3 §10): 11 значений, ровно под 11 фраз `RECORD_PHRASES`.
* `EMOTIONS` — то, что модель может **определить** (семантика реплики). Оно шире:
  `SURPRISE` и `FEAR` из UPDATE 2 остаются различимыми, хотя отдельной записи под
  них нет — резолвер уводит их в ближайший разрешённый профиль с явным откатом
  (UPDATE 3 §17). Свести этот словарь к списку профилей значило бы потерять
  разницу между «модель поняла, что это испуг» и «модель не поняла ничего».

Обратная совместимость: все значения UPDATE 2 входят в `EMOTIONS`, поэтому старые
проекты и голоса читаются без миграции данных.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# `AUTO` — не эмоция, а способ её выбрать («пусть решает LLM»).
EMOTION_AUTO = "AUTO"

EMOTION_NEUTRAL = "NEUTRAL"
EMOTION_CALM = "CALM"
EMOTION_QUESTION = "QUESTION"
EMOTION_NEUTRAL_QUESTION = "NEUTRAL_QUESTION"
EMOTION_EXCLAMATION = "EXCLAMATION"
EMOTION_DELIGHT = "DELIGHT"
EMOTION_SAD_SYMPATHETIC = "SAD_SYMPATHETIC"
EMOTION_IRONIC = "IRONIC"
EMOTION_STRICT = "STRICT"
EMOTION_ENUMERATION = "ENUMERATION"
EMOTION_EXCITED = "EXCITED"
# Значения UPDATE 2, оставленные различимыми: под них нет отдельной записи, и
# резолвер подбирает ближайший профиль с явным `fallback_used` (§17).
EMOTION_SURPRISE = "SURPRISE"
EMOTION_FEAR = "FEAR"

# Production-набор профилей (UPDATE 3 §10): порядок — он же порядок в интерфейсе.
# Только эти значения могут быть ключом записанного референса.
PROFILE_KEYS: tuple[str, ...] = (
    EMOTION_NEUTRAL,
    EMOTION_CALM,
    EMOTION_QUESTION,
    EMOTION_NEUTRAL_QUESTION,
    EMOTION_EXCLAMATION,
    EMOTION_DELIGHT,
    EMOTION_SAD_SYMPATHETIC,
    EMOTION_IRONIC,
    EMOTION_STRICT,
    EMOTION_ENUMERATION,
    EMOTION_EXCITED,
)

# Семантические эмоции: `AUTO` сюда не входит — это не значение анализа.
EMOTIONS: tuple[str, ...] = (*PROFILE_KEYS, EMOTION_SURPRISE, EMOTION_FEAR)

# Значения, которые пользователь может выбрать руками. Шире словаря профилей:
# ручной выбор «Удивление» остаётся возможным для старых проектов и честно
# показывается как откат, если записи под него нет (§17).
SELECTABLE_EMOTIONS: tuple[str, ...] = (EMOTION_AUTO, *EMOTIONS)

# Эмоции, для которых отдельного референса может не быть: отсутствие не блокирует
# синтез, а даёт безопасный откат с пометкой (§9, §26).
OPTIONAL_EMOTIONS: tuple[str, ...] = tuple(
    value for value in EMOTIONS if value != EMOTION_NEUTRAL
)

EMOTION_TITLES = {
    EMOTION_AUTO: "Авто",
    EMOTION_NEUTRAL: "Нейтрально",
    EMOTION_CALM: "Спокойно",
    EMOTION_QUESTION: "Вопрос",
    EMOTION_NEUTRAL_QUESTION: "Вопрос + нейтрально",
    EMOTION_EXCLAMATION: "Восклицание",
    EMOTION_DELIGHT: "Радость / восторг",
    EMOTION_SAD_SYMPATHETIC: "Огорчение / сочувствие",
    EMOTION_IRONIC: "Ирония",
    EMOTION_STRICT: "Строго",
    EMOTION_ENUMERATION: "Перечисление",
    EMOTION_EXCITED: "Взволнованно",
    EMOTION_SURPRISE: "Удивление",
    EMOTION_FEAR: "Испуг",
}

# Короткие подписи для компактных мест интерфейса (dropdown реплики, карточка take).
EMOTION_SHORT_TITLES = {
    **EMOTION_TITLES,
    EMOTION_SAD_SYMPATHETIC: "Сочувствие",
    EMOTION_NEUTRAL_QUESTION: "Вопрос",
}


def is_profile_key(value: object) -> bool:
    """Может ли это значение быть ключом записанного референс-профиля (§10)."""
    return str(value or "").strip().upper() in PROFILE_KEYS


def normalize_profile_key(value: object, *, default: str = EMOTION_NEUTRAL) -> str:
    """Приводит ключ профиля к production-набору; вне набора — `default`.

    Отдельно от `normalize_emotion`: там словарь шире, и FEAR, попавший в профиль,
    создал бы запись, которой не существует в наборе §10.
    """
    text = str(value or "").strip().upper()
    if text in PROFILE_KEYS:
        return text
    return default


def emotion_title(value: object) -> str:
    """Подпись эмоции для интерфейса; неизвестное значение отдаётся как есть."""
    text = str(value or "").strip().upper()
    if not text or text == EMOTION_AUTO:
        return EMOTION_TITLES[EMOTION_AUTO]
    return EMOTION_TITLES.get(text, text)


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


def prosody_effective(
    detected: object, override: object, recommended: object = ""
) -> str:
    """Действующий профиль просодии: override → рекомендация модели → эмоция (§24).

    Отличается от `emotion_effective` одним шагом: модель называет не только
    эмоцию, но и **записываемый** профиль (`recommended_profile`, §10). Если он
    есть, именно он и есть цель маршрутизации — модель уже приняла решение о
    замене (`SURPRISE → EXCLAMATION`), и резолверу не нужно выводить её самому.

    Ручной выбор пользователя сильнее: `override` проверяется первым и берётся
    даже тогда, когда модель рекомендовала другое (§5, §37). Пустая рекомендация
    не ошибка: тогда работает прежнее правило `override → detected → NEUTRAL`, а
    семантические значения уводит в ближайший профиль таблица отката (§17).
    """
    chosen = normalize_emotion(override, default="") if override else ""
    if chosen and chosen != EMOTION_AUTO:
        return chosen
    suggested = normalize_profile_key(recommended, default="") if recommended else ""
    if suggested:
        return suggested
    return emotion_effective(detected, "")


# --- совместимость просодии ---------------------------------------------------
# Регистр просодии: грубая группа интонаций, внутри которой реплики звучат
# сопоставимо. Нужна слою коротких реплик (UPDATE 3 §52): контекст для короткой
# фразы берётся из реплики того же спикера, но одинаковый голос ещё не значит
# одинаковую интонацию — спокойная фраза и крик склеились бы в одно звучание, и
# модель прочитала бы цель в чужом регистре.
#
# Групп намеренно мало: это политика совместимости, а не классификация эмоций.
# Внутри группы замена контекста безопасна, между группами — нет.
#
# Вопрос стоит вместе со спокойным регистром, а не отдельно. Разделять их было бы
# ошибкой: вопрос и спокойный ответ — одна интонационная линия («Как дела?» —
# «Всё хорошо.»), и запрет контекста между ними ломал бы самый обычный диалог.
# Политика защищает от смены **силы** звучания, а не от смены речевой функции.
PROSODY_REGISTER_CALM = "calm"
PROSODY_REGISTER_HIGH = "high"
PROSODY_REGISTER_MARKED = "marked"

PROSODY_REGISTERS: dict[str, str] = {
    EMOTION_NEUTRAL: PROSODY_REGISTER_CALM,
    EMOTION_CALM: PROSODY_REGISTER_CALM,
    EMOTION_QUESTION: PROSODY_REGISTER_CALM,
    EMOTION_NEUTRAL_QUESTION: PROSODY_REGISTER_CALM,
    EMOTION_ENUMERATION: PROSODY_REGISTER_CALM,
    EMOTION_SAD_SYMPATHETIC: PROSODY_REGISTER_CALM,
    EMOTION_EXCLAMATION: PROSODY_REGISTER_HIGH,
    EMOTION_DELIGHT: PROSODY_REGISTER_HIGH,
    EMOTION_EXCITED: PROSODY_REGISTER_HIGH,
    EMOTION_SURPRISE: PROSODY_REGISTER_HIGH,
    EMOTION_IRONIC: PROSODY_REGISTER_MARKED,
    EMOTION_STRICT: PROSODY_REGISTER_MARKED,
    EMOTION_FEAR: PROSODY_REGISTER_MARKED,
}


def prosody_register(value: object) -> str:
    """Регистр интонации; пустое значение — пустая строка («неизвестно»)."""
    text = str(value or "").strip().upper()
    if not text or text == EMOTION_AUTO:
        return ""
    return PROSODY_REGISTERS.get(text, "")


def prosody_compatible(first: object, second: object) -> bool:
    """Совместимы ли две интонации как цель и контекст (§52).

    Неизвестная интонация не блокирует: без анализа (LLM выключена) слой коротких
    реплик обязан работать как раньше, и запрет на контекст из-за отсутствия
    метаданных был бы отказом функциональности ради политики.
    """
    left = prosody_register(first)
    right = prosody_register(second)
    if not left or not right:
        return True
    return left == right


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
_DELIGHT_WORDS = (
    "ура",
    "победа",
    "победили",
    "получилось",
    "супер",
    "здорово",
    "отлично",
    "класс",
    "сбылось",
)
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
    "EMOTION_CALM",
    "EMOTION_DELIGHT",
    "EMOTION_ENUMERATION",
    "EMOTION_EXCLAMATION",
    "EMOTION_EXCITED",
    "EMOTION_FEAR",
    "EMOTION_IRONIC",
    "EMOTION_NEUTRAL",
    "EMOTION_NEUTRAL_QUESTION",
    "EMOTION_QUESTION",
    "EMOTION_SAD_SYMPATHETIC",
    "EMOTION_SHORT_TITLES",
    "EMOTION_STRICT",
    "EMOTION_SURPRISE",
    "EMOTION_TITLES",
    "OPTIONAL_EMOTIONS",
    "PROFILE_KEYS",
    "PROSODY_REGISTERS",
    "SELECTABLE_EMOTIONS",
    "HeuristicEmotion",
    "emotion_effective",
    "emotion_title",
    "heuristic_emotion",
    "is_emotion",
    "is_profile_key",
    "normalize_emotion",
    "normalize_profile_key",
    "prosody_effective",
]
