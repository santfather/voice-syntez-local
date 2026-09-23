"""Сравнение одного голоса на нескольких движках.

Пользователь задаёт тестовую фразу, а система читает её тем же reference и тем же
`ref_text`, отдельно каждым выбранным движком. Результат каждого движка — свой
файл и своя строка с временем синтеза, длительностью и (если включена проверка)
WER. Никакого ранжирования и «лучшего движка» здесь нет и быть не должно: WER —
справка для человека, а решение принимает он сам, прослушав takes.

Движки обходятся последовательно: модель в процессе одна, и параллельный прогон
был бы и гонкой за Metal-контекст, и двойным расходом памяти. Запуск идёт через
единственный воркер очереди (`job_queue`), а не из роута: инференс в обход
очереди ломает и серийность, и подсчёт памяти.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import audio_pipeline, config
from .audio_pipeline import (
    NoteCallback,
    ProgressCallback,
    QaOutcome,
    QaSettings,
    RenderSettings,
    SpeakerSettings,
    WaitForMemory,
)
from .engines.base import ENGINE_INFOS
from .voices_store import Voice

logger = logging.getLogger(__name__)

# Сколько запусков сравнения держать в памяти. Как и у задач (`MAX_KEPT_JOBS`),
# это не «история в базе»: сравнение — разовый инструмент, а файлы его takes
# лежат в output/benchmarks/ и переживают вытеснение записи.
MAX_KEPT_BENCHMARKS = 20

# Псевдо-спикер тестовой фразы (в диалоге не встречается).
BENCHMARK_SPEAKER = "benchmark"

# Чем закончился прогон одного движка.
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

# Статус всего запуска. Отдельно от статуса движка: один упавший движок не делает
# сравнение неудачным — run завершается `done`, а причину видно в его строке.
RUN_QUEUED = "queued"
RUN_RUNNING = "running"
RUN_DONE = "done"
RUN_ERROR = "error"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BenchmarkResult:
    """Итог одного движка: файл, время синтеза, длительность, QA и параметры."""

    engine: str
    engine_label: str
    status: str = STATUS_SKIPPED
    error: str | None = None
    audio_path: Path | None = None
    duration_sec: float | None = None
    # Время самого синтеза (без загрузки модели): загрузка — это десятки секунд
    # один раз, и подмешивать её в сравнение движков значило бы сравнивать модели
    # по времени подъёма, а не по скорости генерации.
    render_sec: float | None = None
    qa: QaOutcome | None = None
    params: dict = field(default_factory=dict)
    seed: int | None = None

    def to_dict(self, run_id: str) -> dict:
        """Плоское представление итога движка; `run_id` нужен для ссылки на аудио."""
        return {
            "engine": self.engine,
            "engine_label": self.engine_label,
            "status": self.status,
            "error": self.error,
            "duration_sec": None if self.duration_sec is None else round(self.duration_sec, 2),
            "render_sec": None if self.render_sec is None else round(self.render_sec, 2),
            "qa": None if self.qa is None else self.qa.to_dict(),
            "params": self.params,
            "seed": self.seed,
            "audio_url": (
                f"/api/benchmarks/{run_id}/{self.engine}/audio"
                if self.status == STATUS_DONE
                else None
            ),
        }


@dataclass
class BenchmarkRun:
    """Запуск сравнения целиком: фраза, движки и результат каждого из них."""

    id: str
    voice_id: str
    text: str
    engines: list[str]
    status: str = RUN_QUEUED
    created_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    error: str | None = None
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.id,
            "voice_id": self.voice_id,
            "text": self.text,
            "engines": list(self.engines),
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "results": [result.to_dict(self.id) for result in self.results],
        }


# --- реестр запусков ----------------------------------------------------------
# В памяти процесса, под блокировкой: роут читает запуск, а воркер его дописывает.
_runs: dict[str, BenchmarkRun] = {}
_order: list[str] = []
_lock = threading.Lock()


def register(run: BenchmarkRun) -> BenchmarkRun:
    with _lock:
        _runs[run.id] = run
        _order.append(run.id)
        _prune_locked()
    return run


def get_run(run_id: str) -> BenchmarkRun | None:
    with _lock:
        return _runs.get(run_id)


def list_runs() -> list[BenchmarkRun]:
    with _lock:
        return [_runs[key] for key in _order if key in _runs]


def reset_registry() -> None:
    """Очищает реестр — тестам нужен предсказуемый старт без чужих запусков."""
    with _lock:
        _runs.clear()
        _order.clear()


def _prune_locked() -> None:
    """Держит не больше `MAX_KEPT_BENCHMARKS` записей.

    Идущий запуск не вытесняется: его ещё читает интерфейс, и потерять его
    посреди работы значило бы показать пользователю пустую панель.
    """
    while len(_order) > MAX_KEPT_BENCHMARKS:
        oldest = _order.pop(0)
        run = _runs.get(oldest)
        if run is not None and run.status in (RUN_QUEUED, RUN_RUNNING):
            _order.append(oldest)
            break
        _runs.pop(oldest, None)


def make_run_id() -> str:
    return uuid.uuid4().hex[:12]


# --- ход сравнения ------------------------------------------------------------
def take_path(run_id: str, engine_id: str) -> Path:
    """Файл результата одного движка: `output/benchmarks/{run_id}-{engine}.wav`."""
    return config.BENCHMARKS_DIR / f"{run_id}-{engine_id}.wav"


def check_reference(voice: Voice) -> None:
    """Готов ли reference к сравнению: без файла или текста синтез невозможен.

    Те же проверки, что и в пайплайне (`audio_pipeline._resolve_voice`), но здесь
    они делаются до постановки задачи: понять «у голоса нет расшифровки» из
    строки результата можно, а вот ждать ради этого прогон моделей — незачем.

    Голос встроенного движка сравнить не с чем: движки сопоставляются по одной и
    той же записи, а у такого голоса её нет. Ошибка здесь честнее прогона, в
    котором половина строк была бы получена без референса.
    """
    if not voice.audio_file:
        raise ValueError(
            f"У голоса «{voice.name}» нет записи: сравнение движков идёт по референсу, "
            "а этот голос озвучивается встроенным голосом своего движка"
        )
    if not voice.audio_path.exists():
        raise ValueError(f"Файл референса для голоса «{voice.name}» потерян ({voice.audio_file})")
    if not voice.ref_text.strip():
        raise ValueError(
            f"У голоса «{voice.name}» не заполнен референс-текст — без него синтез невозможен"
        )


def tuning_for_engine(voice: Voice, engine_id: str) -> SpeakerSettings:
    """Параметры одного движка для этого голоса.

    Голос берётся тот же (тот же reference и `ref_text`), меняется только движок,
    для которого считается иерархия. Чужие ручки движка отсекает резолв: в
    `engine_params` остаются только объявленные в паспорте этого движка
    (см. `settings_resolution.resolve_synthesis_settings`), поэтому температура
    XTTS не доедет до F5 и наоборот. Пресет голоса (скорость, CFG, NFE) общий
    осознанно: сравнение должно идти на одних настройках, а не на дефолтах
    каждой модели.
    """
    view = replace(voice, engine=engine_id)
    return audio_pipeline.tuning_for(view, SpeakerSettings(voice_id=voice.id))


async def _run_engine(
    run: BenchmarkRun,
    voice: Voice,
    engine_id: str,
    text: str,
    settings: RenderSettings,
    wait_for_memory: WaitForMemory | None,
    on_note: NoteCallback | None,
    should_abort: Callable[[], bool] | None = None,
) -> BenchmarkResult:
    """Готовит модель и синтезирует фразу одним движком.

    Ошибка любого шага — результат этого движка со статусом `error` и понятным
    русским текстом: недоступная модель не должна рушить сравнение остальных.
    Отмена сюда не относится: она поднимается наружу (`JobCancelledError`), иначе
    прогон продолжился бы со следующим движком, как будто ничего не случилось.
    """
    if should_abort and should_abort():
        raise audio_pipeline.JobCancelledError("Отменено до запуска движка")
    info = ENGINE_INFOS.get(engine_id)
    if info is None:
        return BenchmarkResult(
            engine=engine_id,
            engine_label=engine_id,
            status=STATUS_SKIPPED,
            error=f"Движок «{engine_id}» неизвестен — пропущен",
        )

    result = BenchmarkResult(engine=engine_id, engine_label=info.label)
    if on_note:
        on_note(f"{info.label}: готовлю модель")
    try:
        engine = audio_pipeline.engine_for_id(engine_id)
        await asyncio.to_thread(engine.load)
    except audio_pipeline.JobCancelledError:
        raise  # отмена — не ошибка движка, а остановка всего прогона
    except Exception as exc:  # noqa: BLE001 — модель может не подняться по десятку причин
        logger.warning("Сравнение %s: движок %s не поднялся (%s)", run.id, engine_id, exc)
        result.status = STATUS_ERROR
        result.error = f"{info.label}: модель не поднялась — {exc}"
        return result

    tuning = tuning_for_engine(voice, engine_id)
    result.params = audio_pipeline.tuning_parameters(tuning)
    if on_note:
        on_note(f"{info.label}: синтез")
    started = time.monotonic()
    try:
        chunk, seed, qa = await audio_pipeline.synthesize_take(
            engine,
            voice,
            text,
            tuning,
            settings,
            label=info.label,
            wait_for_memory=wait_for_memory,
            on_note=on_note,
            should_abort=should_abort,
        )
        render_sec = time.monotonic() - started
        path = take_path(run.id, engine_id)
        duration = await asyncio.to_thread(
            audio_pipeline.write_take, path, chunk, tuning, text
        )
    except audio_pipeline.JobCancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — движок может упасть на самом синтезе
        logger.warning("Сравнение %s: движок %s не синтезировал (%s)", run.id, engine_id, exc)
        result.status = STATUS_ERROR
        result.error = f"{info.label}: синтез не удался — {exc}"
        return result

    result.status = STATUS_DONE
    result.audio_path = path
    result.duration_sec = duration
    result.render_sec = render_sec
    result.qa = qa
    result.seed = seed
    logger.info(
        "Сравнение %s: %s — %.2f c синтеза, %.2f c аудио, сид %s",
        run.id, engine_id, render_sec, duration, seed,
    )
    return result


async def run_benchmark(
    run: BenchmarkRun,
    voice: Voice,
    *,
    qa_mode: str,
    auto_accent: bool,
    wait_for_memory: WaitForMemory | None = None,
    on_note: NoteCallback | None = None,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> None:
    """Прогоняет фразу голоса по движкам строго по одному.

    `run.results` пополняется после каждого движка целиком: интерфейс, читающий
    запуск в это время, никогда не видит строку без итога. Ошибка движка
    остаётся в его строке, поэтому весь прогон завершается `done` — сравнение
    состоялось, если состоялся хотя бы один движок.

    Отмена проверяется между движками (и внутри синтеза — между попытками
    проверки): уже снятые take остаются в `run.results`, а прогон поднимает
    `JobCancelledError` наружу, к очереди.
    """
    settings = RenderSettings(
        pause_ms=0,
        cross_fade_duration=config.DEFAULT_CROSS_FADE_DURATION,
        auto_accent=auto_accent,
        output_format="wav",
        qa=QaSettings.for_mode(qa_mode),
    )
    total = len(run.engines)
    for index, engine_id in enumerate(run.engines, start=1):
        if should_abort and should_abort():
            raise audio_pipeline.JobCancelledError(
                f"Отменено на движке {index} из {total}"
            )
        if on_progress:
            label = ENGINE_INFOS[engine_id].label if engine_id in ENGINE_INFOS else engine_id
            on_progress(index, total, label)
        result = await _run_engine(
            run, voice, engine_id, run.text, settings, wait_for_memory, on_note, should_abort
        )
        run.results.append(result)
