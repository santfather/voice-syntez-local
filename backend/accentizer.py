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
        self._lock = threading.Lock()
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
            except Exception as exc:  # сеть/модели могут быть недоступны
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
        """Расставляет ударения. Текст с ручными "+" возвращается без изменений."""
        if not text.strip() or _MANUAL_STRESS_RE.search(text):
            return text
        if not self.load():
            return text
        try:
            return self._accent.process_all(text)
        except Exception as exc:
            self._state = STATE_FAILED
            self._last_error = str(exc)
            logger.warning("Не удалось расставить ударения (%s). Синтез без них.", exc)
            return text


def accentuate(text: str) -> str:
    return Accentizer.instance().accentuate(text)
