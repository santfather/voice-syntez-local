"""Расшифровка аудио и сверка её с ожидаемым текстом.

Основное назначение — референс-аудио голоса. F5-TTS выравнивает синтез по паре
«референс-аудио + его дословная расшифровка». Если текст не совпадает с записью,
модель «плывёт»: тянет длительность, добавляет лишние слова, ломает интонацию —
вплоть до неразборчивой каши. Проверка ниже ловит такой рассинхрон
(см. `check_ref_text_match`).

Второе назначение — готовый результат синтеза (строгая проверка куска, Фаза 7):
там расшифровка нужна не для модели, а для ответа на вопрос «модель вообще
сказала то, что просили». Порог для этого случая другой и живёт в конфиге
(`config.QA_WER_THRESHOLD`), а не здесь: см. `word_error_rate`.

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

from . import config, memory_guard

logger = logging.getLogger(__name__)

WORKER_MODULE = "backend.transcribe_worker"
WORKER_TIMEOUT_SEC = float(os.environ.get("TTS_ASR_TIMEOUT_SEC", "900"))
# Режим «расшифровать файл целиком»: без него воркер режет запись так же, как
# референс для модели (первые ~12 секунд).
WORKER_FULL_FLAG = "--full"
# Режим «с таймстемпами слов»: нужен границам коротких реплик, где обрезка
# контекстного синтеза обязана опираться на реальные границы слов.
WORKER_WORDS_FLAG = "--words"

# Ниже этой доли совпавших слов считаем, что расшифровка от другой записи.
MATCH_THRESHOLD = 0.6

_WORD_RE = re.compile(r"[a-zа-яё]+")


@dataclass
class Transcription:
    ref_text: str  # что реально произнесено в записи
    effective_sec: float  # длительность фрагмента, который уйдёт в модель
    full_sec: float  # длительность исходного файла


@dataclass(frozen=True)
class WordStamp:
    """Слово и его границы в секундах — то, по чему режется контекстный синтез."""

    word: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


def _run_worker(audio_path: Path, full: bool) -> Transcription:
    """Запускает воркер распознавания и разбирает его ответ."""
    command = [sys.executable, "-m", WORKER_MODULE, str(audio_path)]
    if full:
        command.append(WORKER_FULL_FLAG)
    # Whisper не считается TTS-движком (он транзиентный), но в пике ест ~1.5 ГБ:
    # поверх двух поднятых движков это тот самый случай из E2E. Отказа здесь нет
    # (проверка качества нужна именно на слабых машинах) — только предупреждение.
    memory_guard.check_whisper_load_budget(memory_guard.loaded_tts_engines())
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


def transcribe_file(audio_path: Path) -> Transcription:
    """Распознаёт речь в записи; режет её так же, как это делает синтез."""
    return _run_worker(audio_path, full=False)


def _run_words_worker(audio_path: Path) -> list[WordStamp]:
    """Запускает воркер в режиме таймстемпов слов."""
    command = [sys.executable, "-m", WORKER_MODULE, str(audio_path), WORKER_WORDS_FLAG]
    # См. `_run_worker`: тот же транзиентный Whisper, то же предупреждение.
    memory_guard.check_whisper_load_budget(memory_guard.loaded_tts_engines())
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
        return [
            WordStamp(word=str(item["word"]), start=float(item["start"]), end=float(item["end"]))
            for item in payload.get("words") or []
        ]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("распознавание вернуло неожиданный ответ") from exc


def transcribe_words(audio_path: Path) -> list[WordStamp]:
    """Распознаёт запись целиком и отдаёт границы слов.

    Нужно границам коротких реплик: без реальных границ контекстный синтез не
    режется, а откатывается на DIRECT (см. `short_utterance_boundary`).
    """
    return _run_words_worker(audio_path)


def transcribe_audio(audio_path: Path) -> str:
    """Распознаёт запись целиком — без обрезки под референс.

    Нужно строгой проверке синтеза: проверяемый кусок бывает длиннее 12 секунд, а
    обрезка референса оставила бы от него только начало, и расхождение в хвосте
    осталось бы незамеченным.
    """
    return _run_worker(audio_path, full=True).ref_text


def _words(text: str) -> list[str]:
    """Слова в нижнем регистре: пунктуация и знаки ударения (`+`) не в счёт."""
    return _WORD_RE.findall(text.lower())


def _matched_words(text: str) -> list[str]:
    """Слова для сверки результата синтеза: «ё» и «е» считаются одним словом.

    Whisper пишет «е» там, где в исходном тексте «ё» (и наоборот). На строгом
    пороге проверки результата это давало бы провал на ровном месте, тогда как на
    слух и по смыслу слово то же.
    """
    return [word.replace("ё", "е") for word in _words(text)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Доля ошибочных слов (0..1): замена, пропуск и вставка весят одинаково.

    Своё расстояние Левенштейна по словам, а не `SequenceMatcher`: тот считает
    длину совпавших блоков и на перестановках завышает ошибку почти вдвое («а, б,
    в» против «в, б, а» — это две замены, а не четыре). Для сверки референса, где
    важно «текст вообще от этой записи или от другой», такая разница не значит
    ничего, а для приёмки готового куска решает, повторять синтез или нет.

    Пустая расшифровка даёт 1.0: молчание модели — это провал проверки, а не
    «сверять нечего».
    """
    expected = _matched_words(reference)
    if not expected:
        return 0.0
    actual = _matched_words(hypothesis)
    # Одна строка на память вместо матрицы: кусок — это десятки слов, но цикл
    # вызывается на каждой попытке каждой реплики.
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            cost = 0 if expected_word == actual_word else 1
            current.append(
                min(
                    previous[column] + 1,  # лишнее слово в расшифровке
                    current[column - 1] + 1,  # пропущенное слово
                    previous[column - 1] + cost,  # замена
                )
            )
        previous = current
    return previous[-1] / len(expected)


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
