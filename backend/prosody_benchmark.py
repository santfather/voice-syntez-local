"""Reference Prosody Transfer Benchmark: инфраструктура сравнения профилей (§30–§35).

Задача benchmark'а — ответить на один вопрос: **меняет ли смена reference-профиля
просодию target и не ухудшает ли она при этом текст и голос**. Ответ даётся
числами на фиксированной матрице, а не впечатлением от одной генерации.

Матрица (§32) устроена так, что меняется ровно одна переменная: голос, движок,
сид, скорость, ручки движка и текст зафиксированы, а различается только
`ReferenceProfile`. Один и тот же сид обязателен — иначе «профиль звучит иначе»
неотличимо от «модель сгенерировала иначе» (см. `audio_pipeline.synthesize_take`).

Что считается автоматически (§33): ASR-расшифровка, WER, совпадение первого и
последнего слова, лишние слова, повторы, длительность, время генерации и статус
текстовой проверки. Метаданные (voice/engine/profile/seed/reference_profile_id)
лежат рядом с каждым замером.

Чего здесь намеренно нет: `enabled_for_auto` и любого «лучшего профиля».
Автоматика решает только «текст не сломался»; перенос интонации оценивает человек
по форме прослушивания (§34), а gating профилей — фаза Phase 11. Whisper не
заменяет прослушивание и не должен.

Матрица выполняется последовательно: модель в процессе одна, и параллельные
синтезы были бы и гонкой за память, и недостоверным временем генерации.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import audio_pipeline, config, emotions, reference_resolver
from .audio_pipeline import (
    NoteCallback,
    ProgressCallback,
    RenderSettings,
    SpeakerSettings,
    WaitForMemory,
)
from .transcribe import transcribe_audio, word_error_rate
from .voices_store import Voice

logger = logging.getLogger(__name__)

# Версия корпуса и формата отчёта. Растёт при несовместимом изменении набора
# кейсов или полей метрик: старые отчёты должны оставаться читаемыми как есть.
BENCHMARK_VERSION = "1"

# --- категории корпуса (§31) ---------------------------------------------------
CATEGORY_NEUTRAL = "neutral"
CATEGORY_QUESTION = "question"
CATEGORY_DELIGHT = "delight"
CATEGORY_SAD = "sad_sympathetic"
CATEGORY_IRONY = "irony"
CATEGORY_STRICT = "strict"
CATEGORY_ENUMERATION = "enumeration"
CATEGORY_EXCITED = "excited"
CATEGORY_AMBIGUOUS = "ambiguous"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_NEUTRAL,
    CATEGORY_QUESTION,
    CATEGORY_DELIGHT,
    CATEGORY_SAD,
    CATEGORY_IRONY,
    CATEGORY_STRICT,
    CATEGORY_ENUMERATION,
    CATEGORY_EXCITED,
    CATEGORY_AMBIGUOUS,
)

CATEGORY_TITLES: dict[str, str] = {
    CATEGORY_NEUTRAL: "Нейтрально",
    CATEGORY_QUESTION: "Вопрос",
    CATEGORY_DELIGHT: "Радость",
    CATEGORY_SAD: "Огорчение / сочувствие",
    CATEGORY_IRONY: "Ирония",
    CATEGORY_STRICT: "Строго",
    CATEGORY_ENUMERATION: "Перечисление",
    CATEGORY_EXCITED: "Взволнованно",
    CATEGORY_AMBIGUOUS: "Неоднозначное (зависит от контекста)",
}

# Минимум реплик в корпусе (§31). Меньше — benchmark не покрывает категории.
MIN_CORPUS_CASES = 30

# Профили, которые сравниваются минимум (§32). Порядок — порядок колонок в отчёте.
MATRIX_PROFILES: tuple[str, ...] = (
    emotions.EMOTION_NEUTRAL,
    emotions.EMOTION_QUESTION,
    emotions.EMOTION_DELIGHT,
    emotions.EMOTION_IRONIC,
    emotions.EMOTION_STRICT,
    emotions.EMOTION_EXCITED,
)

# Сид по умолчанию: у матрицы он один на все ячейки (§32).
DEFAULT_SEED = 0

# --- статусы -------------------------------------------------------------------
CELL_DONE = "done"
CELL_ERROR = "error"

# Статус автоматической текстовой проверки: это **не** QA-цикл пайплайна (он
# выключен, чтобы генерация была одиночной и детерминированной), а вывод по
# расшифровке и метрикам. `review` — текст цел, но есть повод послушать.
TEXT_QA_PASSED = "passed"
TEXT_QA_REVIEW = "review"
TEXT_QA_FAILED = "failed"

# Порог лишних слов, после которого текст считается подозрительным: одно-два
# служебных слова Whisper добавляет часто, длинный хвост — уже признак повтора.
MAX_EXTRA_WORDS = 2
# Повтор слова подряд столько раз — дефект («да, да, да»).
MIN_REPEAT_RUN = 3
# Столько повторов n-граммы — дефект («это правда, это правда»).
REPEATED_NGRAM_COUNT = 2

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- корпус (§31) ---------------------------------------------------------------
@dataclass(frozen=True)
class CorpusCase:
    """Одна реплика корпуса: текст, категория и ожидаемое намерение."""

    id: str
    category: str
    text: str
    expected_intent: str
    ambiguous: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "category_title": CATEGORY_TITLES.get(self.category, self.category),
            "text": self.text,
            "expected_intent": self.expected_intent,
            "ambiguous": self.ambiguous,
            "note": self.note,
        }


def load_corpus(path: Path) -> list[CorpusCase]:
    """Читает JSONL-корпус: одна строка — одна реплика.

    Ошибка разбора называется строкой файла: без номера правку в корпусе искать
    руками, а корпус — данные, а не код, и ломается он именно опечаткой.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Корпус не найден: {path}") from exc

    cases: list[CorpusCase] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: строка {number}: не JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}: строка {number}: ожидался JSON-объект")
        missing = [name for name in ("id", "category", "text", "expected_intent") if name not in payload]
        if missing:
            raise ValueError(
                f"{path.name}: строка {number}: нет полей {', '.join(missing)}"
            )
        cases.append(
            CorpusCase(
                id=str(payload["id"]).strip(),
                category=str(payload["category"]).strip().lower(),
                text=str(payload["text"]).strip(),
                expected_intent=str(payload["expected_intent"] or "").strip().upper(),
                ambiguous=bool(payload.get("ambiguous", False)),
                note=str(payload.get("note") or ""),
            )
        )
    if not cases:
        raise ValueError(f"{path.name}: корпус пуст")
    return cases


def corpus_problems(cases: Sequence[CorpusCase]) -> list[str]:
    """Проверяет корпус до прогона: ошибку в данных дешевле найти здесь.

    Проблемы возвращаются списком, а не исключением: `--check` печатает их все
    разом, а тесты проверяют, что их нет.
    """
    problems: list[str] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for case in cases:
        if case.id in seen:
            problems.append(f"повтор id: {case.id}")
        seen.add(case.id)
        if case.category not in CATEGORIES:
            problems.append(f"{case.id}: неизвестная категория «{case.category}»")
        counts[case.category] += 1
        if not case.text:
            problems.append(f"{case.id}: пустой текст")
        elif len(case.text) > config.MAX_REPLICA_CHARS:
            problems.append(
                f"{case.id}: текст длиннее {config.MAX_REPLICA_CHARS} знаков"
            )
        if case.expected_intent and case.expected_intent not in emotions.EMOTIONS:
            problems.append(
                f"{case.id}: ожидаемое намерение «{case.expected_intent}» вне словаря эмоций"
            )
    for category in CATEGORIES:
        if not counts.get(category):
            problems.append(f"категория «{category}» пуста")
    if len(cases) < MIN_CORPUS_CASES:
        problems.append(
            f"реплик {len(cases)}, а нужно минимум {MIN_CORPUS_CASES} (§31)"
        )
    return problems


def category_counts(cases: Iterable[CorpusCase]) -> dict[str, int]:
    counts = Counter(case.category for case in cases)
    return {category: counts.get(category, 0) for category in CATEGORIES}


# --- матрица (§32) --------------------------------------------------------------
@dataclass(frozen=True)
class MatrixCell:
    """Ячейка матрицы: реплика и конкретный разрешённый reference под неё."""

    case: CorpusCase
    profile_emotion: str
    resolved: reference_resolver.ResolvedReference

    @property
    def reference_profile_id(self) -> str:
        return self.resolved.profile_id


def available_resolutions(
    voice: Voice,
    engine_id: str,
    profiles: Sequence[str] = MATRIX_PROFILES,
) -> dict[str, reference_resolver.ResolvedReference]:
    """Какие из профилей матрицы реально доступны у этого голоса на этом движке.

    Разрешение идёт **боевым** `reference_resolver`: benchmark обязан мерить тот
    же путь выбора файла, что и production, иначе он измерит свою копию логики.
    Профиль без файла, выключенный, не прошедший проверку или чужой для движка
    сюда не попадёт — и это видно как отсутствующая колонка, а не как тихая
    подмена нейтральным.

    Профиль запрашивается явно (`profile_id`), а не подбором по интонации: гейт
    §35 закрывает автоматику для неподтверждённых профилей, и подбором benchmark
    не увидел бы ровно те профили, которые и должен подтвердить или забраковать.
    """
    found: dict[str, reference_resolver.ResolvedReference] = {}
    all_profiles = voice.reference_profiles()
    for emotion in profiles:
        for candidate in [item for item in all_profiles if item.emotion == emotion]:
            try:
                resolved = reference_resolver.resolve_reference(
                    voice, engine_id, emotion, profile_id=candidate.id
                )
            except reference_resolver.ReferenceUnavailableError:
                continue
            if resolved.profile_id == candidate.id:
                found[emotion] = resolved
                break
    return found


def plan_matrix(
    cases: Sequence[CorpusCase],
    resolutions: dict[str, reference_resolver.ResolvedReference],
) -> list[MatrixCell]:
    """Полный список ячеек: реплика × доступные профили.

    Порядок — «все профили одной реплики подряд»: так человек слушает их
    сравнением, а не ищет соответствующую пару в разросшемся списке.
    """
    cells: list[MatrixCell] = []
    for case in cases:
        for emotion in MATRIX_PROFILES:
            resolved = resolutions.get(emotion)
            if resolved is None:
                continue
            cells.append(MatrixCell(case=case, profile_emotion=emotion, resolved=resolved))
    return cells


def matrix_profiles(resolutions: dict[str, reference_resolver.ResolvedReference]) -> tuple[str, ...]:
    return tuple(emotion for emotion in MATRIX_PROFILES if emotion in resolutions)


# --- метрики (§33) --------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    """Слова для сверки: регистр, «ё» и пунктуация не считаются."""
    return [token.lower().replace("ё", "е") for token in _TOKEN_RE.findall(text or "")]


def detect_repetition(tokens: Sequence[str]) -> str:
    """Ищет повтор — самый частый дефект синтеза на короткой реплике.

    Возвращает описание найденного повтора или пустую строку. Два независимых
    признака: слово, повторённое подряд, и n-грамма, сказанная дважды.
    """
    run = 1
    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            run += 1
            if run >= MIN_REPEAT_RUN:
                return f"слово «{tokens[index]}» подряд {run} раза"
        else:
            run = 1
    for size in (2, 3):
        grams = Counter(
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        )
        for gram, count in grams.items():
            if count >= REPEATED_NGRAM_COUNT:
                return f"повтор «{' '.join(gram)}» {count} раза"
    return ""


def text_qa_status(
    *,
    transcript: str,
    wer: float | None,
    first_word_ok: bool,
    last_word_ok: bool,
    extra_words: int,
    repetition: str,
) -> str:
    """Автоматическая проверка текста по расшифровке (§33).

    `failed` — модель промолчала или текст разошёлся с заданием по WER; `review` —
    текст в целом тот, но есть дефект (потерянное слово, лишний хвост, повтор) —
    такой take человек обязан послушать. Разделение важно: WER усредняет и
    скрывает, что именно сломалось.
    """
    if not transcript.strip():
        return TEXT_QA_FAILED
    if wer is not None and wer > config.QA_WER_THRESHOLD:
        return TEXT_QA_FAILED
    if not first_word_ok or not last_word_ok or repetition or extra_words > MAX_EXTRA_WORDS:
        return TEXT_QA_REVIEW
    return TEXT_QA_PASSED


@dataclass
class CellMetrics:
    """Метрики одной ячейки (§33): текст, время и автоматический вердикт."""

    transcript: str = ""
    wer: float | None = None
    first_word_ok: bool = False
    last_word_ok: bool = False
    extra_words: int = 0
    repetition: str = ""
    duration_sec: float | None = None
    generation_sec: float | None = None
    text_qa: str = TEXT_QA_FAILED

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "wer": None if self.wer is None else round(self.wer, 4),
            "first_word_ok": self.first_word_ok,
            "last_word_ok": self.last_word_ok,
            "extra_words": self.extra_words,
            "repetition": self.repetition,
            "duration_sec": None if self.duration_sec is None else round(self.duration_sec, 3),
            "generation_sec": None if self.generation_sec is None else round(self.generation_sec, 2),
            "text_qa": self.text_qa,
        }


def measure_metrics(
    text: str,
    transcript: str,
    *,
    duration_sec: float | None = None,
    generation_sec: float | None = None,
) -> CellMetrics:
    """Считает всё, что считается по тексту и расшифровке одной ячейки.

    WER берётся у существующего `transcribe.word_error_rate` — benchmark не
    заводит вторую формулу, иначе его числа нельзя было бы сверять с QA-циклом.
    """
    expected = _tokens(text)
    got = _tokens(transcript)
    wer = word_error_rate(text, transcript) if transcript.strip() else 1.0
    first_word_ok = bool(got) and got[0] == expected[0]
    last_word_ok = bool(got) and got[-1] == expected[-1]
    extra_words = sum((Counter(got) - Counter(expected)).values())
    repetition = detect_repetition(got)
    return CellMetrics(
        transcript=transcript,
        wer=wer,
        first_word_ok=first_word_ok,
        last_word_ok=last_word_ok,
        extra_words=extra_words,
        repetition=repetition,
        duration_sec=duration_sec,
        generation_sec=generation_sec,
        text_qa=text_qa_status(
            transcript=transcript,
            wer=wer,
            first_word_ok=first_word_ok,
            last_word_ok=last_word_ok,
            extra_words=extra_words,
            repetition=repetition,
        ),
    )


# --- результат прогона ----------------------------------------------------------
@dataclass
class CellResult:
    """Итог одной ячейки: метаданные (§33), статус, файл и метрики."""

    case_id: str
    category: str
    text: str
    expected_intent: str
    ambiguous: bool
    profile_emotion: str
    reference_profile_id: str
    reference_audio: str
    voice_id: str
    engine: str
    seed: int | None = None
    status: str = CELL_ERROR
    error: str | None = None
    audio_file: str = ""
    metrics: CellMetrics | None = None

    def to_dict(self) -> dict:
        """Плоское представление ячейки матрицы: вход, результат и метрики."""
        return {
            "case_id": self.case_id,
            "category": self.category,
            "text": self.text,
            "expected_intent": self.expected_intent,
            "ambiguous": self.ambiguous,
            "profile": self.profile_emotion,
            "reference_profile_id": self.reference_profile_id,
            "reference_audio": self.reference_audio,
            "voice_id": self.voice_id,
            "engine": self.engine,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "audio_file": self.audio_file,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass
class ProsodyBenchmarkRun:
    """Один прогон матрицы: что сравнивали и что получилось."""

    id: str
    voice_id: str
    voice_name: str
    engine: str
    profiles: tuple[str, ...]
    cases_total: int
    seed: int = DEFAULT_SEED
    speed: float | None = None
    created_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    error: str | None = None
    cells: list[CellResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Плоское представление прогона матрицы вместе со всеми ячейками."""
        return {
            "benchmark_id": self.id,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "engine": self.engine,
            "profiles": list(self.profiles),
            "cases_total": self.cases_total,
            "seed": self.seed,
            "speed": self.speed,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "cells": [cell.to_dict() for cell in self.cells],
        }


def make_run_id() -> str:
    return uuid.uuid4().hex[:12]


def make_run(
    voice: Voice,
    engine_id: str,
    *,
    cases_total: int,
    profiles: Sequence[str],
    seed: int = DEFAULT_SEED,
    speed: float | None = None,
) -> ProsodyBenchmarkRun:
    """Заводит прогон матрицы по голосу, движку и списку профилей."""
    return ProsodyBenchmarkRun(
        id=make_run_id(),
        voice_id=voice.id,
        voice_name=voice.name,
        engine=engine_id,
        profiles=tuple(profiles),
        cases_total=cases_total,
        seed=seed,
        speed=speed,
    )


def cell_audio_name(case_id: str, profile_emotion: str, engine_id: str, seed: int) -> str:
    """Имя файла ячейки: видно реплику, профиль, движок и сид — без чтения отчёта."""
    return f"{case_id}__{profile_emotion.lower()}__{engine_id}__s{seed}.wav"


# --- выполнение матрицы ---------------------------------------------------------
def tuning_for_engine(
    voice: Voice, engine_id: str, speed: float | None = None
) -> SpeakerSettings:
    """Параметры синтеза для ячейки: пресет голоса плюс зафиксированные ручки.

    Как в сравнении движков (`backend.benchmark`): пресет голоса общий для всей
    матрицы, а `speed` — единственная ручка, которую benchmark может зафиксировать
    сверху. Всё остальное обязано оставаться одинаковым между профилями (§32).
    """
    view = replace(voice, engine=engine_id)
    tuning = audio_pipeline.tuning_for(view, SpeakerSettings(voice_id=voice.id))
    return replace(tuning, speed=speed) if speed is not None else tuning


async def run_matrix(
    run: ProsodyBenchmarkRun,
    voice: Voice,
    engine,
    cells: Sequence[MatrixCell],
    *,
    out_dir: Path,
    transcribe: Callable[[Path], str] | None = transcribe_audio,
    auto_accent: bool = True,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> None:
    """Прогоняет матрицу последовательно: одна ячейка — один синтез.

    QA-цикл выключен намеренно. Он повторяет генерацию до «удачной» попытки, и
    тогда у профилей сравнивались бы разные сиды и разное число попыток — то есть
    benchmark мерил бы везение, а не перенос просодии. Текст проверяется отдельно,
    по расшифровке (см. `measure_metrics`), и статус попадает в отчёт.

    `transcribe` — распознавание для метрик; `None` отключает ASR (быстрый
    прогон без текстовых метрик). Сбой распознавания не рушит ячейку: аудио уже
    сгенерировано, и потерять замер времени из-за Whisper нельзя.
    """
    settings = RenderSettings(
        pause_ms=0,
        cross_fade_duration=config.DEFAULT_CROSS_FADE_DURATION,
        auto_accent=auto_accent,
        output_format="wav",
        qa=None,
    )
    tuning = tuning_for_engine(voice, run.engine, run.speed)
    total = len(cells)
    for index, cell in enumerate(cells, start=1):
        if should_abort and should_abort():
            raise audio_pipeline.JobCancelledError(
                f"Отменено на ячейке {index} из {total}"
            )
        label = f"{cell.case.id}/{cell.profile_emotion}"
        if on_progress:
            on_progress(index, total, label)
        result = CellResult(
            case_id=cell.case.id,
            category=cell.case.category,
            text=cell.case.text,
            expected_intent=cell.case.expected_intent,
            ambiguous=cell.case.ambiguous,
            profile_emotion=cell.profile_emotion,
            reference_profile_id=cell.resolved.profile_id,
            reference_audio=cell.resolved.audio_path.name,
            voice_id=voice.id,
            engine=run.engine,
            seed=run.seed,
        )
        started = time.monotonic()
        try:
            chunk, seed, _qa = await audio_pipeline.synthesize_take(
                engine,
                voice,
                cell.case.text,
                tuning,
                settings,
                label=label,
                wait_for_memory=wait_for_memory,
                on_note=on_note,
                should_abort=should_abort,
                reference=cell.resolved,
                seed=run.seed,
            )
            generation_sec = time.monotonic() - started
            path = out_dir / "audio" / cell_audio_name(
                cell.case.id, cell.profile_emotion, run.engine, run.seed
            )
            duration_sec = await asyncio.to_thread(
                audio_pipeline.write_take, path, chunk, tuning, cell.case.text
            )
        except audio_pipeline.JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — сбой одной ячейки не рушит матрицу
            logger.warning("Просодия %s: ячейка %s не сгенерирована (%s)", run.id, label, exc)
            result.status = CELL_ERROR
            result.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            run.cells.append(result)
            continue

        result.status = CELL_DONE
        result.seed = seed
        result.audio_file = path.name
        transcript = ""
        if transcribe is not None:
            try:
                transcript = await asyncio.to_thread(transcribe, path)
            except Exception as exc:  # noqa: BLE001 — распознавание может быть недоступно
                logger.warning("Просодия %s: расшифровка %s недоступна (%s)", run.id, label, exc)
        result.metrics = measure_metrics(
            cell.case.text,
            transcript,
            duration_sec=duration_sec,
            generation_sec=generation_sec,
        )
        run.cells.append(result)
    run.finished_at = _now_iso()


# --- отчёт ----------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _profile_entry(cells: Sequence[CellResult]) -> dict:
    done = [cell for cell in cells if cell.status == CELL_DONE and cell.metrics is not None]
    metrics = [cell.metrics for cell in done]
    passed = sum(1 for item in metrics if item.text_qa == TEXT_QA_PASSED)
    review = sum(1 for item in metrics if item.text_qa == TEXT_QA_REVIEW)
    failed = sum(1 for item in metrics if item.text_qa == TEXT_QA_FAILED)
    wers = [item.wer for item in metrics if item.wer is not None]
    return {
        "cells": len(cells),
        "done": len(done),
        "errors": sum(1 for cell in cells if cell.status == CELL_ERROR),
        "text_qa_passed": passed,
        "text_qa_review": review,
        "text_qa_failed": failed,
        "text_ok_share": _share(passed, len(done)),
        "wer_mean": _mean(wers),
        "wer_max": round(max(wers), 4) if wers else None,
        "first_word_ok_share": _share(
            sum(1 for item in metrics if item.first_word_ok), len(done)
        ),
        "last_word_ok_share": _share(
            sum(1 for item in metrics if item.last_word_ok), len(done)
        ),
        "extra_words_mean": _mean([float(item.extra_words) for item in metrics]),
        "repetition_share": _share(
            sum(1 for item in metrics if item.repetition), len(done)
        ),
        "duration_mean": _mean(
            [item.duration_sec for item in metrics if item.duration_sec is not None]
        ),
        "generation_mean": _mean(
            [item.generation_sec for item in metrics if item.generation_sec is not None]
        ),
    }


def summarize(run: ProsodyBenchmarkRun) -> dict:
    """Сводка по профилям и категориям с явным сравнением с NEUTRAL.

    Нейтральный профиль — база: он есть у каждого голоса и не несёт интонации.
    Разница «профиль минус нейтральный» и есть ответ на вопрос §35 «какие профили
    реально дают устойчивый перенос»: положительный `wer_delta_vs_neutral` означает
    ухудшение текста, отрицательный `text_ok_delta_vs_neutral` — потерю качества.
    """
    by_profile: dict[str, dict] = {}
    for emotion in run.profiles:
        entry = _profile_entry([cell for cell in run.cells if cell.profile_emotion == emotion])
        by_profile[emotion] = entry
    neutral = by_profile.get(emotions.EMOTION_NEUTRAL)
    for emotion, entry in by_profile.items():
        entry["wer_delta_vs_neutral"] = (
            None
            if neutral is None
            or entry["wer_mean"] is None
            or neutral["wer_mean"] is None
            else round(entry["wer_mean"] - neutral["wer_mean"], 4)
        )
        entry["text_ok_delta_vs_neutral"] = (
            None
            if neutral is None
            or entry["text_ok_share"] is None
            or neutral["text_ok_share"] is None
            else round(entry["text_ok_share"] - neutral["text_ok_share"], 4)
        )

    by_category: dict[str, dict] = {}
    for category in CATEGORIES:
        subset = [cell for cell in run.cells if cell.category == category]
        if not subset:
            continue
        entry = _profile_entry(subset)
        entry["title"] = CATEGORY_TITLES.get(category, category)
        entry["profiles"] = sorted({cell.profile_emotion for cell in subset})
        by_category[category] = entry
    return {"by_profile": by_profile, "by_category": by_category}


def build_report(run: ProsodyBenchmarkRun) -> dict:
    """Собирает отчёт: метаданные прогона, ячейки и сводку.

    Отчёт самодостаточен: в каждой ячейке лежат и текст, и профиль, и сид, поэтому
    цифру можно перепроверить, не поднимая код.
    """
    return {
        "benchmark": "reference_prosody",
        "version": BENCHMARK_VERSION,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "voice_id": run.voice_id,
        "voice_name": run.voice_name,
        "engine": run.engine,
        "seed": run.seed,
        "speed": run.speed,
        "profiles": list(run.profiles),
        "cases_total": run.cases_total,
        "cells_total": len(run.cells),
        "cells": [cell.to_dict() for cell in run.cells],
        "summary": summarize(run),
        "errors": [
            {"case_id": cell.case_id, "profile": cell.profile_emotion, "error": cell.error}
            for cell in run.cells
            if cell.status == CELL_ERROR
        ],
    }


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_report(report: dict) -> str:
    """Markdown-отчёт: сводка по профилям, категории и все ячейки.

    Таблица профилей — главное: в ней рядом стоят WER, совпадение слов, повторы и
    разница с NEUTRAL. Вывод «профиль полезен» здесь не пишется: числа отвечают
    только за текст, а перенос интонации подтверждает человек по `review.md`.
    """
    lines = [
        "# Reference Prosody Transfer Benchmark",
        "",
        f"- Версия формата: {report.get('version')}",
        f"- Голос: {report.get('voice_name')} (`{report.get('voice_id')}`)",
        f"- Движок: `{report.get('engine')}`",
        f"- Сид: {report.get('seed')}",
        f"- Реплик: {report.get('cases_total')}, ячеек: {report.get('cells_total')}",
        f"- Профили: {', '.join(report.get('profiles') or [])}",
        f"- Запуск: {report.get('created_at')} → {report.get('finished_at')}",
        "",
        "## Сводка по профилям",
        "",
        "`wer_delta_vs_neutral` > 0 — текст хуже нейтрального; "
        "`text_ok_delta_vs_neutral` < 0 — доля целых текстов ниже нейтральной.",
        "",
        "| Профиль | Ячеек | Ошибок | WER сред. | ΔWER | Текст ok | ΔТекст | Первое слово | Последнее слово | Лишние | Повторы | Длит., с | Ген., с |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for emotion, entry in (report.get("summary") or {}).get("by_profile", {}).items():
        cells = [
            emotion,
            entry["cells"],
            entry["errors"],
            _format_value(entry["wer_mean"]),
            _format_value(entry["wer_delta_vs_neutral"]),
            _format_value(entry["text_ok_share"]),
            _format_value(entry["text_ok_delta_vs_neutral"]),
            _format_value(entry["first_word_ok_share"]),
            _format_value(entry["last_word_ok_share"]),
            _format_value(entry["extra_words_mean"]),
            _format_value(entry["repetition_share"]),
            _format_value(entry["duration_mean"]),
            _format_value(entry["generation_mean"]),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")

    by_category = (report.get("summary") or {}).get("by_category") or {}
    if by_category:
        lines += [
            "",
            "## По категориям",
            "",
            "| Категория | Ячеек | WER сред. | Текст ok | Повторы | Длит., с |",
            "|---|---|---|---|---|---|",
        ]
        for category, entry in by_category.items():
            lines.append(
                f"| {entry.get('title', category)} | {entry['cells']} | "
                f"{_format_value(entry['wer_mean'])} | {_format_value(entry['text_ok_share'])} | "
                f"{_format_value(entry['repetition_share'])} | "
                f"{_format_value(entry['duration_mean'])} |"
            )

    errors = report.get("errors") or []
    if errors:
        lines += ["", "## Ошибки генерации", "", "| Реплика | Профиль | Ошибка |", "|---|---|---|"]
        for item in errors:
            lines.append(f"| `{item['case_id']}` | {item['profile']} | {item['error']} |")

    lines += [
        "",
        "## Ячейки",
        "",
        "| Реплика | Профиль | Статус | WER | Текст | Расшифровка | Файл |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in report.get("cells") or []:
        metrics = cell.get("metrics") or {}
        lines.append(
            f"| `{cell['case_id']}` | {cell['profile']} | {cell['status']} | "
            f"{_format_value(metrics.get('wer'))} | {_format_value(metrics.get('text_qa'))} | "
            f"{metrics.get('transcript', '')} | `{cell.get('audio_file', '')}` |"
        )

    lines += [
        "",
        "## Что дальше",
        "",
        "Числа говорят только о тексте: перенос интонации оценивает человек по "
        "`review.md` (§34). Production-маршрутизация включается после прослушивания "
        "(§35), а не по этой таблице.",
        "",
    ]
    return "\n".join(lines)


def render_review(report: dict) -> str:
    """Форма listening review (§34): пустые поля, которые заполняет человек.

    Колонки взяты из §34 дословно. `notes` предзаполняется контекстом кейса, чтобы
    спорную реплику не приходилось вспоминать: для неоднозначных реплик именно
    контекст решает, считается интонация попаданием или нет.
    """
    lines = [
        "# Listening review: reference prosody",
        "",
        f"Голос: {report.get('voice_name')} (`{report.get('voice_id')}`), "
        f"движок `{report.get('engine')}`, сид {report.get('seed')}.",
        "",
        "Заполняется вручную: Whisper здесь не судья (§34). Для каждой строки "
        "поставьте `pass`/`fail` в первых четырёх колонках, `yes`/`no` — в двух "
        "следующих, и коротко опишите дефект в `notes`.",
        "",
        "Порядок прослушивания: все профили одной реплики идут подряд — так слышно "
        "именно смену интонации, а не разницу между фразами.",
        "",
        "| target_text | expected_intent | reference_profile | text_complete | clarity | voice_identity | prosody_match | overacting | artifacts | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in report.get("cells") or []:
        if cell.get("status") != CELL_DONE:
            continue
        intent = cell.get("expected_intent") or "—"
        lines.append(
            f"| {cell['text']} | {intent} | {cell['profile']} |  |  |  |  |  |  |  |"
        )
    lines += [
        "",
        "## Как читать колонки",
        "",
        "- `text_complete` — сказано всё, без пропусков и лишнего.",
        "- `clarity` — дикция разборчива, без смазывания.",
        "- `voice_identity` — голос узнаваем как тот же, что и в других профилях.",
        "- `prosody_match` — интонация соответствует `expected_intent` (для "
        "неоднозначных реплик — с учётом контекста из `notes`).",
        "- `overacting` — интонация наиграна, звучит неестественно.",
        "- `artifacts` — щелчки, металл, обрывы, дыхание не на месте.",
        "- `notes` — что именно не так; для спорных реплик здесь лежит подсказка "
        "из корпуса.",
        "",
    ]
    return "\n".join(lines)


def write_report(out_dir: Path, report: dict) -> None:
    """Пишет три файла прогона: JSON (данные), Markdown (вывод), review (форма)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(format_report(report), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(report), encoding="utf-8")


__all__ = [
    "BENCHMARK_VERSION",
    "CATEGORIES",
    "CATEGORY_TITLES",
    "CELL_DONE",
    "CELL_ERROR",
    "DEFAULT_SEED",
    "MATRIX_PROFILES",
    "MIN_CORPUS_CASES",
    "TEXT_QA_FAILED",
    "TEXT_QA_PASSED",
    "TEXT_QA_REVIEW",
    "CellMetrics",
    "CellResult",
    "CorpusCase",
    "MatrixCell",
    "ProsodyBenchmarkRun",
    "available_resolutions",
    "build_report",
    "category_counts",
    "cell_audio_name",
    "corpus_problems",
    "detect_repetition",
    "format_report",
    "load_corpus",
    "make_run",
    "make_run_id",
    "matrix_profiles",
    "measure_metrics",
    "plan_matrix",
    "render_review",
    "run_matrix",
    "summarize",
    "text_qa_status",
    "tuning_for_engine",
    "write_report",
]
