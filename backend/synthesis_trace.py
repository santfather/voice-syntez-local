"""Трейс синтеза: где именно у короткой реплики теряется начало и конец (§13–§15).

Зачем этот модуль. Жалоба «модель проглатывает окончания или не договаривает
фразу» одинаково правдоподобно объясняется тремя разными причинами: модель сама
не произнесла слово, постобработка срезала его краем, или контекстный слой обрезал
цель по неточной границе. Пока эти варианты не разделены измерением, любая правка
пайплайна — угадывание, а «стало лучше» — самообман.

Поэтому трейс делает ровно две вещи:

* пишет **срез аудио на каждой стадии** обработки одной реплики
  (`01_raw_engine` → `02_post_context_extract` → `03_post_edge_trim` →
  `04_final_replica`) — по ним слышно и видно, где пропал звук;
* пишет **числа стадий** (сколько сэмплов снято обрезкой тишины, кроссфейдом,
  обрезкой контекста, какова длительность на каждом шаге) в `trace.jsonl` — по ним
  видно, где пропал звук, даже без прослушивания.

Стадия, которая не применялась, отмечается явно (`applied: false`), а не молчанием:
«обрезано на 0 сэмплов» и «обрезки не было» — разные факты.

Трейс выключен по умолчанию (`TTS_SYNTHESIS_TRACE=1` включает) и не является
частью продакшн-пайплайна: он только наблюдает. Когда он включён, аудио стадий
лежит в `output/trace/<job_id>/r001/…`, а каталоги старых прогонов вытесняются
(`KEEP_RUNS`), чтобы диагностика не превращалась в склад.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .engines.base import SAMPLE_RATE

logger = logging.getLogger(__name__)

TRACE_ENV = "TTS_SYNTHESIS_TRACE"
TRACE_DIR_ENV = "TTS_SYNTHESIS_TRACE_DIR"
# Сколько прогонов трейса хранить. Трейс — инструмент разбора, а не архив: десять
# последних прогонов достаточно, чтобы сравнить «до и после» правки пайплайна.
KEEP_RUNS = 10

# Имена стадий. Префикс — порядок обработки: файлы читаются глазами в каталоге,
# и «02» перед «01» там оказаться не может.
STAGE_RAW = "01_raw_engine"
STAGE_CONTEXT = "02_post_context_extract"
STAGE_TRIM = "03_post_edge_trim"
STAGE_FINAL = "04_final_replica"


def enabled() -> bool:
    """Включён ли трейс. Читается в момент старта задачи, а не на импорте модуля.

    Переменная, прочитанная на импорте, не переключалась бы в тестах и в живом
    сервере: `TTS_SYNTHESIS_TRACE=1` перед запуском и всё. Один `os.environ` на
    задачу дешевле, чем неожиданное поведение.
    """
    return str(os.environ.get(TRACE_ENV, "0")).strip().lower() not in ("0", "false", "no", "off", "")


def trace_root() -> Path:
    """Каталог трейсов: `output/trace` или явный путь из окружения."""
    override = os.environ.get(TRACE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(config.OUTPUT_DIR) / "trace"


@dataclass
class StageInfo:
    """Что произошло с аудио на стадии: длительность и снятые сэмплы."""

    applied: bool = False
    samples: int = 0
    dropped_start: int = 0
    dropped_end: int = 0
    # Уровень снятого и оставшегося края. Это и есть ответ на вопрос «пайплайн
    # срезал слово или тишину»: если снятый хвост звучит на уровне речи, обрезка
    # съела звук; если он ниже порога тишины — резать было нечего.
    dropped_head_rms_dbfs: float | None = None
    dropped_tail_rms_dbfs: float | None = None
    dropped_head_peak_dbfs: float | None = None
    dropped_tail_peak_dbfs: float | None = None
    kept_rms_dbfs: float | None = None
    kept_peak_dbfs: float | None = None
    note: str = ""


@dataclass
class ReplicaTrace:
    """Запись одной реплики: вход синтеза, стадии обработки и итог проверки.

    Поля сгруппированы по вопросам, на которые они отвечают: чем синтезировали
    (движок, голос, референс, параметры), что именно ушло в модель (тексты и план
    короткой реплики), что стало с аудио (стадии) и что услышала проверка (ASR).
    """

    index: int
    # --- паспорт ---------------------------------------------------------------
    job_id: str = ""
    project_id: str = ""
    speaker: str = ""
    voice_id: str = ""
    engine: str = ""
    label: str = ""
    position: str = ""
    # --- текст -----------------------------------------------------------------
    source_text: str = ""
    final_text: str = ""
    prepared_from_final: bool = False
    word_count: int = 0
    char_count: int = 0
    # --- эмоция и референс (заполняется слоем эмоций; до него — пусто) ----------
    emotion_detected: str = ""
    emotion_override: str = ""
    emotion_effective: str = ""
    reference_profile_id: str = ""
    reference_audio: str = ""
    reference_text: str = ""
    reference_emotion: str = ""
    reference_fallback_used: bool = False
    # --- план синтеза ----------------------------------------------------------
    utterance_class: str = ""
    short_strategy: str = ""
    short_strategy_requested: str = ""
    tts_target_text: str = ""
    tts_context_text: str = ""
    tts_synthesis_text: str = ""
    short_fallback: str = ""
    short_attempts: int = 0
    # --- параметры движка ------------------------------------------------------
    speed: float = 1.0
    seed: int | None = None
    engine_params: dict[str, Any] = field(default_factory=dict)
    # --- куски и края ----------------------------------------------------------
    text_chunks: int = 1
    chunk_boundaries: list[int] = field(default_factory=list)
    edge_trim_enabled: bool = True
    edge_silence_db: float = 0.0
    edge_fade_ms: float = 0.0
    pre_guard_ms: float = 0.0
    post_guard_ms: float = 0.0
    # --- стадии аудио ----------------------------------------------------------
    stages: dict[str, StageInfo] = field(default_factory=dict)
    # --- проверка --------------------------------------------------------------
    qa_status: str = ""
    qa_transcript: str = ""
    qa_wer: float | None = None
    first_word_ok: bool | None = None
    last_word_ok: bool | None = None
    repetition_detected: bool = False
    asr_words: list[str] = field(default_factory=list)
    expected_words: list[str] = field(default_factory=list)
    debug_files: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: _now())
    seconds: float = 0.0
    # Ссылка на прогон, которому принадлежит запись: через неё стадия сама
    # сохраняет файл. В `to_dict` не попадает — это не данные, а способ записи.
    _trace: SynthesisTrace | None = field(default=None, repr=False, compare=False)

    # --- стадии ---------------------------------------------------------------
    def stage(self, name: str, audio: np.ndarray | None, *, note: str = "") -> None:
        """Отмечает стадию: сколько в ней сэмплов и (если можно) пишет файл."""
        info = self.stages.setdefault(name, StageInfo())
        if audio is None:
            # Стадия не применялась: это факт, который надо записать явно, иначе
            # «нет файла» неотличимо от «файл потерялся».
            info.applied = False
            info.note = note or "не применялась"
            return
        info.applied = True
        info.samples = int(np.asarray(audio).size)
        if note:
            info.note = note
        if self._trace is not None:
            path = self._trace.save_audio(self, name, audio)
            if path:
                self.debug_files[name] = path

    def trim(self, raw: np.ndarray, start: int, end: int, note: str = "") -> None:
        """Записывает обрезку краёв как отдельную стадию с числами и уровнями.

        Уровни снятого края важнее самих сэмплов: 30 мс тишины и 30 мс тихого
        окончания — одинаковые числа и совершенно разные последствия, и различить
        их можно только по энергии. Поэтому срез измеряется здесь, а не выводится
        читателем отчёта.
        """
        data = np.asarray(raw, dtype=np.float32).reshape(-1)
        info = self.stages.setdefault(STAGE_TRIM, StageInfo())
        info.applied = True
        info.samples = int(end - start)
        info.dropped_start = int(start)
        info.dropped_end = int(max(0, data.size - end))
        info.note = note or "тишина по краям"
        info.dropped_head_rms_dbfs = _rms_dbfs(data[:start])
        info.dropped_head_peak_dbfs = _peak_dbfs(data[:start])
        info.dropped_tail_rms_dbfs = _rms_dbfs(data[end:])
        info.dropped_tail_peak_dbfs = _peak_dbfs(data[end:])
        info.kept_rms_dbfs = _rms_dbfs(data[start:end])
        info.kept_peak_dbfs = _peak_dbfs(data[start:end])

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("_trace", None)
        payload["stages"] = {name: asdict(info) for name, info in self.stages.items()}
        payload["stages"]["durations_sec"] = {
            name: round(info.samples / SAMPLE_RATE, 4)
            for name, info in self.stages.items()
            if info.samples
        }
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rms_dbfs(audio: np.ndarray) -> float | None:
    """RMS куска в dBFS; `None` — считать нечего (пусто)."""
    data = np.asarray(audio, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(data))))
    return None if rms <= 1e-9 else round(20.0 * float(np.log10(rms)), 2)


def _peak_dbfs(audio: np.ndarray) -> float | None:
    """Пик куска в dBFS; `None` — считать нечего (пусто)."""
    data = np.asarray(audio, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return None
    peak = float(np.max(np.abs(data)))
    return None if peak <= 1e-9 else round(20.0 * float(np.log10(peak)), 2)


class SynthesisTrace:
    """Трейс одного прогона: каталог, записи реплик и файлы стадий."""

    def __init__(self, job_id: str, *, root: Path | None = None) -> None:
        self.job_id = str(job_id)
        self.root = (root or trace_root()) / self._safe(self.job_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[ReplicaTrace] = []
        self._prune()

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))[:64]

    def _prune(self) -> None:
        """Оставляет `KEEP_RUNS` последних прогонов трейса."""
        parent = self.root.parent
        try:
            runs = sorted(
                (path for path in parent.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for stale in runs[KEEP_RUNS:]:
            shutil.rmtree(stale, ignore_errors=True)

    # --- записи ---------------------------------------------------------------
    def replica(self, index: int, **fields: Any) -> ReplicaTrace:
        """Начинает запись реплики; поля паспорта передаются сразу."""
        record = ReplicaTrace(index=int(index), job_id=self.job_id, **fields)
        record._trace = self
        self.records.append(record)
        return record

    def save_audio(self, record: ReplicaTrace, stage: str, audio: np.ndarray) -> str:
        """Сохраняет аудио стадии и возвращает относительный путь для отчёта."""
        directory = self.root / f"r{record.index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stage}.wav"
        try:
            _write_wav(path, audio)
        except Exception as exc:  # noqa: BLE001 — трейс не имеет права ломать синтез
            logger.warning("Трейс: стадия %s реплики %s не сохранена (%s)", stage, record.index, exc)
            return ""
        try:
            return str(path.relative_to(self.root))
        except ValueError:  # pragma: no cover — путь всегда внутри root
            return str(path)

    def write(self, record: ReplicaTrace | None = None) -> Path:
        """Пишет `trace.jsonl` целиком. Файл маленький — перезапись дешевле дозаписи."""
        target = self.root / "trace.jsonl"
        payload = [item.to_dict() for item in (self.records if record is None else [record])]
        mode = "w" if record is None else "a"
        with target.open(mode, encoding="utf-8") as stream:
            for item in payload:
                stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return target

    def write_summary(self) -> Path:
        """Пишет `summary.json` — сводку прогона для быстрого чтения."""
        target = self.root / "summary.json"
        data = summary(self.records)
        data["job_id"] = self.job_id
        data["trace_dir"] = str(self.root)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """WAV стадии. Импорт soundfile локальный: трейс не тянет его без надобности."""
    import soundfile as sf

    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(path, data, SAMPLE_RATE, format="WAV", subtype="PCM_16")


def start(job_id: str, *, root: Path | None = None) -> SynthesisTrace | None:
    """Трейс для задачи или `None`, если он выключен.

    Выключенный трейс — это `None`, а не объект-пустышка: пайплайн проверяет
    `if trace is not None` в горячем цикле, и одна проверка дешевле, чем десяток
    вызовов, которые внутри себя решают ничего не делать.
    """
    if not enabled():
        return None
    try:
        trace = SynthesisTrace(job_id, root=root)
    except OSError as exc:
        # Трейс — диагностика: если каталог недоступен, синтез всё равно нужен.
        logger.warning("Трейс синтеза недоступен (%s) — продолжаю без него", exc)
        return None
    logger.info("Трейс синтеза включён: %s", trace.root)
    return trace


def summary(records: list[ReplicaTrace]) -> dict:
    """Сводка по прогону: сколько реплик, где терялись сэмплы, что услышал ASR.

    Считается по записанным данным, а не по ходу синтеза: сводка нужна разбору
    после прогона, и она не должна влиять на сам синтез.
    """
    trim_drops = [
        (item.index, item.stages[STAGE_TRIM].dropped_start, item.stages[STAGE_TRIM].dropped_end)
        for item in records
        if STAGE_TRIM in item.stages and item.stages[STAGE_TRIM].applied
    ]
    # Порог «снятый край звучал как речь»: на 6 дБ выше порога тишины. Число
    # намеренно грубое — это повод послушать, а не приговор.
    speech_level = float(config.EDGE_SILENCE_DB) + 6.0
    suspicious: list[dict] = []
    for item in records:
        info = item.stages.get(STAGE_TRIM)
        if info is None or not info.applied:
            continue
        for side, level in (
            ("start", info.dropped_head_rms_dbfs),
            ("end", info.dropped_tail_rms_dbfs),
        ):
            if level is not None and level > speech_level:
                suspicious.append({"index": item.index, "side": side, "rms_dbfs": level})
    return {
        "replicas": len(records),
        "trimmed": len(trim_drops),
        "trim_dropped_ms": [
            {
                "index": index,
                "start_ms": round(start / SAMPLE_RATE * 1000, 1),
                "end_ms": round(end / SAMPLE_RATE * 1000, 1),
            }
            for index, start, end in trim_drops
        ],
        "suspicious_trims": suspicious,
        "first_word_failed": [item.index for item in records if item.first_word_ok is False],
        "last_word_failed": [item.index for item in records if item.last_word_ok is False],
        "qa_failed": [item.index for item in records if item.qa_status and item.qa_status != "passed"],
    }
