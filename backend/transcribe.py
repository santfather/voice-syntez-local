"""Расшифровка референс-аудио и сверка её с текстом, который ввёл пользователь.

F5-TTS выравнивает синтез по паре «референс-аудио + его дословная расшифровка».
Если текст не совпадает с записью, модель «плывёт»: тянет длительность, добавляет
лишние слова, ломает интонацию — вплоть до неразборчивой каши. Проверка ниже
ловит такой рассинхрон (см. `check_ref_text_match`).

Само распознавание выполняется в отдельном процессе (`transcribe_worker`), потому
что Whisper large-v3-turbo не помещается в бюджет памяти основного процесса.
"""

import difflib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

WORKER_MODULE = "backend.transcribe_worker"
WORKER_TIMEOUT_SEC = float(os.environ.get("TTS_ASR_TIMEOUT_SEC", "900"))

# Ниже этой доли совпавших слов считаем, что расшифровка от другой записи.
MATCH_THRESHOLD = 0.6

_WORD_RE = re.compile(r"[a-zа-яё]+")


@dataclass
class Transcription:
    ref_text: str  # что реально произнесено в записи
    effective_sec: float  # длительность фрагмента, который уйдёт в модель
    full_sec: float  # длительность исходного файла


def transcribe_file(audio_path: Path) -> Transcription:
    """Распознаёт речь в записи; режет её так же, как это делает синтез."""
    command = [sys.executable, "-m", WORKER_MODULE, str(audio_path)]
    try:
        proc = subprocess.run(
            command,
            cwd=config.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"распознавание не уложилось в {WORKER_TIMEOUT_SEC:.0f} с") from exc

    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError(f"распознавание не удалось: {tail or 'код ' + str(proc.returncode)}")

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return Transcription(
            ref_text=str(payload["ref_text"]),
            effective_sec=float(payload["effective_sec"]),
            full_sec=float(payload["full_sec"]),
        )
    except (ValueError, KeyError, IndexError) as exc:
        raise RuntimeError("распознавание вернуло неожиданный ответ") from exc


def _words(text: str) -> list[str]:
    """Слова в нижнем регистре: пунктуация и знаки ударения (`+`) не в счёт."""
    return _WORD_RE.findall(text.lower())


def similarity(declared: str, recognized: str) -> float:
    """Доля слов из `declared`, найденных в `recognized` (0..1)."""
    declared_words = _words(declared)
    if not declared_words or not recognized.strip():
        return 0.0
    matcher = difflib.SequenceMatcher(None, declared_words, _words(recognized))
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(declared_words)


def check_ref_text_match(declared: str, recognized: str) -> str:
    """Возвращает предупреждение, если расшифровка не похожа на запись, иначе ''."""
    score = similarity(declared, recognized)
    if score >= MATCH_THRESHOLD:
        return ""
    return (
        f"Расшифровка совпала с записью лишь на {score * 100:.0f}%. "
        f"Похоже, в поле указан текст от другой записи — синтез этим голосом будет "
        f"невнятным. В записи распознано: «{recognized}»"
    )
