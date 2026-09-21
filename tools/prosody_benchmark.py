"""Benchmark reference-профилей: сравнение просодии на фиксированной матрице (§30–§35).

Инструмент запускается **в процессе** и поднимает настоящую модель (F5 или XTTS):
дашборд перед прогоном лучше остановить — два процесса с моделями не влезут в
память Apple Silicon.

Что делает прогон:

1. берёт корпус реплик (`benchmarks/prosody/corpus.v1.jsonl`, §31);
2. строит матрицу «реплика × доступные профили» (§32): голос, движок, сид,
   скорость, ручки движка и текст зафиксированы, меняется только `ReferenceProfile`;
3. синтезирует каждую ячейку и считает метрики текста (§33): WER, первое и
   последнее слово, лишние слова, повторы, длительность, время генерации;
4. пишет WAV для прослушивания и три файла отчёта: `report.json`, `report.md` и
   `review.md` — форму ручного прослушивания (§34).

Границы честности инструмента: он отвечает только на «текст не сломался». Перенос
интонации оценивает человек по `review.md`, а production-маршрутизация профилей
включается после этого прослушивания (§35), а не по таблицам отчёта.

Запуск:

    ./venv/bin/python tools/prosody_benchmark.py --check        # корпус, без моделей
    ./venv/bin/python tools/prosody_benchmark.py --plan --voice <id>   # матрица, без моделей
    ./venv/bin/python tools/prosody_benchmark.py --voice <id>   # прогон (реальная модель)
    ./venv/bin/python tools/prosody_benchmark.py --voice <id> --limit 3 --no-asr
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Инструмент запускается из корня репозитория и как `tools/prosody_benchmark.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, emotions
from backend import prosody_benchmark as pb
from backend.engines.registry import get_engine
from backend.transcribe import transcribe_audio
from backend.voices_store import Voice, get_store

logger = logging.getLogger("prosody_benchmark")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "benchmarks" / "prosody" / "corpus.v1.jsonl"

# Сколько строк плана печатать: полная матрица — сотни ячеек, и глазами её не
# читают, а считают по числу.
PLAN_PREVIEW = 40


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark переноса просодии по reference-профилям",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="JSONL-корпус реплик")
    parser.add_argument("--voice", default="", help="id голоса (обязателен, если голосов больше одного)")
    parser.add_argument("--engine", default="", help="движок; по умолчанию — движок голоса")
    parser.add_argument(
        "--profiles",
        default=",".join(pb.MATRIX_PROFILES),
        help="профили матрицы через запятую (сравнивается минимум §32)",
    )
    parser.add_argument("--seed", type=int, default=pb.DEFAULT_SEED, help="сид, один на всю матрицу")
    parser.add_argument("--speed", type=float, default=None, help="зафиксировать скорость")
    parser.add_argument("--limit", type=int, default=0, help="ограничить число реплик (быстрый прогон)")
    parser.add_argument("--out", default="", help="каталог отчёта (по умолчанию output/benchmarks/prosody/<id>)")
    parser.add_argument("--check", action="store_true", help="только проверка корпуса, без моделей")
    parser.add_argument("--plan", action="store_true", help="показать матрицу, без моделей")
    parser.add_argument("--no-asr", action="store_true", help="без расшифровки (быстрее, без метрик текста)")
    parser.add_argument("--json", action="store_true", help="напечатать путь к JSON-отчёту")
    return parser.parse_args(argv)


def parse_profiles(value: str) -> tuple[str, ...]:
    """Профили матрицы из аргумента: неизвестное имя — ошибка, а не пропуск.

    Молча выбросить опечатку значило бы прогнать матрицу без запрошенной колонки
    и не заметить этого до чтения отчёта.
    """
    names = tuple(dict.fromkeys(item.strip().upper() for item in value.split(",") if item.strip()))
    if not names:
        raise SystemExit("Список профилей пуст")
    unknown = [name for name in names if name not in emotions.EMOTIONS]
    if unknown:
        known = ", ".join(emotions.EMOTIONS)
        raise SystemExit(f"Неизвестные профили: {', '.join(unknown)}. Доступны: {known}")
    return names


def load_cases(args: argparse.Namespace) -> list[pb.CorpusCase]:
    """Читает и проверяет корпус до любого прогона.

    Проверка идёт по **полному** корпусу, а `--limit` применяется уже после неё:
    иначе быстрый прогон на трёх репликах «проходил» бы проверку, которую полный
    корпус не проходит.
    """
    cases = pb.load_corpus(args.corpus)
    problems = pb.corpus_problems(cases)
    if problems:
        for problem in problems:
            print(f"корпус: {problem}", file=sys.stderr)
        raise SystemExit(f"Корпус не готов к прогону: {args.corpus}")
    return cases[: args.limit] if args.limit else cases


def print_corpus(cases: list[pb.CorpusCase], corpus: Path) -> None:
    ambiguous = sum(1 for case in cases if case.ambiguous)
    print(f"Корпус: {corpus} — реплик {len(cases)}, из них неоднозначных {ambiguous}")
    for category, count in pb.category_counts(cases).items():
        title = pb.CATEGORY_TITLES.get(category, category)
        print(f"  {category:16} {count:3}  {title}")


def pick_voice(requested: str, *, store=None) -> Voice:
    """Голос для прогона: явный или единственный имеющийся.

    Автовыбора «первого попавшегося» нет: у голосов разные референсы, и прогон не
    того голоса — это не ошибка инструмента, а незамеченная потеря времени.
    """
    store = store or get_store()
    voices = store.list()
    if not voices:
        raise SystemExit("Нет голосов: создайте хотя бы один в дашборде")
    if requested:
        voice = next((item for item in voices if item.id == requested), None)
        if voice is None:
            raise SystemExit(f"Голос {requested} не найден. Доступны: {_voice_list(voices)}")
        return voice
    if len(voices) == 1:
        return voices[0]
    raise SystemExit(f"Укажите --voice. Доступны: {_voice_list(voices)}")


def _voice_list(voices: list[Voice]) -> str:
    return ", ".join(f"{voice.id} ({voice.name}, {voice.engine})" for voice in voices)


def resolve_profiles(
    voice: Voice, engine_id: str, profiles: tuple[str, ...]
) -> dict[str, object]:
    """Разрешённые профили матрицы: без собственного референса колонки не будет.

    NEUTRAL обязателен: он база сравнения, и без него отчёт не отвечает на вопрос
    «профиль лучше или хуже нейтрального».
    """
    resolutions = pb.available_resolutions(voice, engine_id, profiles)
    if emotions.EMOTION_NEUTRAL not in resolutions:
        raise SystemExit(
            f"У голоса «{voice.name}» нет доступного нейтрального референса — "
            "сравнивать не с чем"
        )
    missing = [name for name in profiles if name not in resolutions]
    if missing:
        print(
            "Профили без собственного референса (колонки не будет): " + ", ".join(missing),
            file=sys.stderr,
        )
    return resolutions


def print_plan(run: pb.ProsodyBenchmarkRun, cells: list[pb.MatrixCell]) -> None:
    print(f"Голос: {run.voice_name} ({run.voice_id}), движок {run.engine}")
    print(f"Профили ({len(run.profiles)}): {', '.join(run.profiles)}")
    print(f"Реплик: {run.cases_total}, ячеек: {len(cells)}, сид: {run.seed}, скорость: {run.speed or 'пресет'}")
    print("Ячейки (все профили одной реплики подряд):")
    for cell in cells[:PLAN_PREVIEW]:
        print(
            f"  {cell.case.id:18} {cell.profile_emotion:12} "
            f"{cell.reference_profile_id:28} {cell.resolved.audio_path.name}"
        )
    if len(cells) > PLAN_PREVIEW:
        print(f"  … ещё {len(cells) - PLAN_PREVIEW}")


def run(
    args: argparse.Namespace,
    *,
    engine=None,
    store=None,
    transcribe=transcribe_audio,
) -> int:
    """Тело CLI, отделённое от разбора аргументов: так его проверяют тесты."""
    try:
        cases = load_cases(args)
    except ValueError as exc:
        # Битый корпус — ошибка данных, а не сбой инструмента: показываем причину
        # (с номером строки) и выходим, не поднимая модель.
        raise SystemExit(str(exc)) from exc
    if args.check:
        print_corpus(cases, args.corpus)
        return 0

    profiles = parse_profiles(args.profiles)
    voice = pick_voice(args.voice, store=store)
    engine_id = args.engine or voice.engine
    resolutions = resolve_profiles(voice, engine_id, profiles)
    matrix_profiles = pb.matrix_profiles(resolutions)
    cells = pb.plan_matrix(cases, resolutions)
    run_state = pb.make_run(
        voice,
        engine_id,
        cases_total=len(cases),
        profiles=matrix_profiles,
        seed=args.seed,
        speed=args.speed,
    )
    if args.plan:
        print_plan(run_state, cells)
        return 0

    out_dir = Path(args.out) if args.out else config.BENCHMARKS_DIR / "prosody" / run_state.id
    print(f"Голос: {voice.name} ({voice.id}), движок {engine_id}")
    print(f"Профили: {', '.join(matrix_profiles)}; реплик {len(cases)}; ячеек {len(cells)}; сид {run_state.seed}")
    print(f"Отчёт: {out_dir}")
    if engine is None:
        print("Поднимаю модель (это может занять минуты)…")
        engine = get_engine(engine_id)
        engine.load()
    asyncio.run(
        pb.run_matrix(
            run_state,
            voice,
            engine,
            cells,
            out_dir=out_dir,
            transcribe=None if args.no_asr else transcribe,
            on_note=lambda note: print(f"    {note}", flush=True),
            on_progress=lambda index, total, label: print(f"[{index}/{total}] {label}", flush=True),
        )
    )
    report = pb.build_report(run_state)
    pb.write_report(out_dir, report)
    errors = len(report["errors"])
    print(f"Готово: ячеек {report['cells_total']}, ошибок {errors}")
    print(f"Markdown: {out_dir / 'report.md'}")
    print(f"Прослушивание: {out_dir / 'review.md'}")
    if args.json:
        print(str(out_dir / "report.json"))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
