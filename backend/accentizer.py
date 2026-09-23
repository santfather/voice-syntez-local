"""Обёртка над RUAccent: автоматическая расстановка ударений перед синтезом."""

import logging
import re
import threading
import time
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

# Уже расставленное пользователем ударение: "+" перед гласной (формат F5-TTS_RUSSIAN).
_MANUAL_STRESS_RE = re.compile(r"\+[аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY]")


def _tokens(text: str) -> list[str]:
    """Текст, разбитый на слова и пробелы: разделитель с группой сохраняет пробелы."""
    return re.split(r"(\s+)", text)


# Граница предложения: знак конца или перевод строки. Модель ё-омографов читает
# одно предложение за вызов — длинный текст она молча обрезала бы по лимиту входа.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def _sentences(text: str) -> list[str]:
    """Куски текста по границам предложений — вход модели ё-омографов."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _is_manual(token: str) -> bool:
    """Слово, ударение в котором уже расставлено вручную."""
    return bool(token) and not token.isspace() and _MANUAL_STRESS_RE.search(token) is not None


def _needs_accentuation(text: str) -> bool:
    """Есть ли в тексте слово без ручного ударения — то, ради чего грузить модель."""
    return any(not _is_manual(token) and token.strip() for token in _tokens(text))


# Если загрузка упала — не долбить сеть/диск на каждой реплике, но и не оставаться
# без ударений до перезапуска сервера: пробуем снова не чаще раза в минуту.
_RETRY_COOLDOWN_SEC = 60.0

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"


def _fix_accent_model_inputs(accent) -> None:
    """Включает token_type_ids в входы nn_accent.

    Модель nn_accent (доставляет ударение в словах вне словаря) объявляет во
    входе token_type_ids, а CharTokenizer его не отдаёт: в новых transformers
    список `model_input_names` у PreTrainedTokenizer сузился до
    ["input_ids", "attention_mask"]. Из-за этого `put_accent` падал с
    «Required inputs (['token_type_ids']) are missing from input feed» на любом
    слове вне словаря («FastAPI», «venv»), исключение вылетало из process_all —
    и весь кусок оставался вообще без ударений.
    """
    tokenizer = getattr(getattr(accent, "accent_model", None), "tokenizer", None)
    names = getattr(tokenizer, "model_input_names", None)
    if tokenizer is not None and names is not None and "token_type_ids" not in names:
        tokenizer.model_input_names = [*names, "token_type_ids"]
        logger.info("RUAccent: nn_accent требует token_type_ids — добавил их в входы токенизатора")


class Accentizer:
    """Singleton: модель ударений грузится один раз при первом использовании."""

    _instance: Optional["Accentizer"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._accent = None
        # Два разных лока, а не один: загрузка модели и её инференс — разные
        # критические секции. Общий лок держал бы загрузку в очереди за чужим
        # инференсом, хотя грузить модель и считать ударения можно параллельно.
        self._lock = threading.Lock()
        # Модель RUAccent — один объект на процесс, и её инференс не потокобезопасен:
        # два одновременных вызова делят внутреннее состояние и портят результат.
        # Параллельные запросы (реплика проекта, preview, предложения для словаря)
        # сериализуются здесь — ровно так же, как `_infer_lock` у движков синтеза.
        self._infer_lock = threading.Lock()
        self._state = STATE_IDLE
        self._last_error: str | None = None
        self._retry_after = 0.0

    @classmethod
    def instance(cls) -> "Accentizer":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def state(self) -> str:
        """idle | loading | ready | failed — чтобы UI отличал «нет ударений» от «модель ошиблась»."""
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_loaded(self) -> bool:
        return self._state == STATE_READY

    def load(self) -> bool:
        """Грузит RUAccent. Возвращает False, если библиотека/модели недоступны."""
        with self._lock:
            if self._accent is not None:
                return True
            if time.monotonic() < self._retry_after:
                return False
            self._state = STATE_LOADING
            try:
                from ruaccent import RUAccent

                accent = RUAccent()
                accent.load(
                    omograph_model_size=config.OMOGRAPH_MODEL_SIZE,
                    use_dictionary=True,
                    tiny_mode=False,
                )
                _fix_accent_model_inputs(accent)
                self._accent = accent
                self._state = STATE_READY
                self._last_error = None
                logger.info("RUAccent загружен (%s)", config.OMOGRAPH_MODEL_SIZE)
                return True
            except Exception as exc:  # noqa: BLE001 — сеть/модели могут быть недоступны
                self._state = STATE_FAILED
                self._last_error = str(exc)
                self._retry_after = time.monotonic() + _RETRY_COOLDOWN_SEC
                logger.error(
                    "RUAccent недоступен — синтез пойдёт БЕЗ ударений (повтор через %.0f с): %s",
                    _RETRY_COOLDOWN_SEC,
                    exc,
                )
                return False

    def accentuate(self, text: str) -> str:
        """Расставляет ударения, сохраняя уже размеченные пользователем слова.

        Без ручных «+» поведение прежнее: весь текст уходит в модель одним куском.
        Если размеченные слова есть, они остаются дословно, а промежутки между ними
        акцентируются сегментами — один вызов модели на промежуток, а не на слово:
        RUAccent читает контекст, и на одиночных словах ошибается чаще. Раньше одна
        такая запись лишала ударений всю реплику — теперь только своё слово.
        """
        if not text.strip():
            return text
        manual = _MANUAL_STRESS_RE.search(text) is not None
        if manual and not _needs_accentuation(text):
            # Размечено каждое слово — модели здесь делать нечего (и грузить её незачем).
            return text
        if not self.load():
            return text
        try:
            # Весь инференс — под локом: `_accent_between_manual` зовёт модель по
            # сегментам, и без общей секции сегменты одной реплики перемешались бы
            # с сегментами соседнего запроса.
            with self._infer_lock:
                if manual:
                    return self._accent_between_manual(text)
                return self._accent.process_all(text)
        except Exception as exc:  # noqa: BLE001 — синтез без ударений лучше отказа
            self._state = STATE_FAILED
            self._last_error = str(exc)
            logger.warning("Не удалось расставить ударения (%s). Синтез без них.", exc)
            return text

    def _accent_between_manual(self, text: str) -> str:
        """Акцентирует промежутки между словами с ручным «+», не трогая сами слова."""
        result: list[str] = []
        pending: list[str] = []

        def flush() -> None:
            if not pending:
                return
            segment = "".join(pending)
            pending.clear()
            result.append(self._accent.process_all(segment) if segment.strip() else segment)

        for token in _tokens(text):
            if _is_manual(token):
                flush()
                result.append(token)
            else:
                pending.append(token)
        flush()
        return "".join(result)

    # -- только чтение: источники предложений в словарь --------------------------
    # Обе обёртки не поднимают модель (работают лишь когда она уже поднята) и молча
    # возвращают None, если библиотека недоступна или сменила формат: предложения —
    # вспомогательная функция, и она не имеет права уронить разбор текста.
    def yo_homograph_scores(self, text: str) -> list[dict] | None:
        """Предсказания модели ё-омографов вместе со `score` — или None.

        `score` берётся только настоящий, из модели: если библиотека его не отдаёт,
        источник предложений пропускается, а не получает выдуманную уверенность.

        Модель читает одно предложение за вызов и обрезает слишком длинный вход,
        поэтому текст режется по границам предложений, а результаты складываются.
        Упавшее предложение пропускается — из-за одного сбоя терять остальные нельзя.
        """
        if not self.is_loaded:
            return None
        model = getattr(self._accent, "yo_homograph_model", None)
        if model is None:
            return None
        result: list[dict] = []
        for sentence in _sentences(text):
            try:
                # Тот же лок, что и в `accentuate`: модель одна на процесс, и
                # параллельный вызов из соседнего запроса испортил бы её состояние.
                with self._infer_lock:
                    entities = model.predict_yo_homographs(sentence)
            except Exception as exc:  # noqa: BLE001 — недоступность модели не ошибка
                logger.warning("Модель ё-омографов не ответила на предложение: %s", exc)
                continue
            if not isinstance(entities, list):
                continue
            for item in entities:
                if not isinstance(item, dict):
                    continue
                word = item.get("word")
                score = item.get("score")
                if not isinstance(word, str) or not word:
                    continue
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    continue
                result.append(
                    {"word": word, "score": float(score), "entity": str(item.get("entity", ""))}
                )
        return result

    def yo_form(self, word: str) -> str | None:
        """Ё-форма слова из словаря ruaccent (только чтение) или None."""
        if not self.is_loaded:
            return None
        mapping = getattr(self._accent, "yo_homographs", None)
        if not isinstance(mapping, dict):
            return None
        value = mapping.get(word.lower())
        return value if isinstance(value, str) and value else None


def accentuate(text: str) -> str:
    return Accentizer.instance().accentuate(text)
