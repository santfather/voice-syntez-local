"""Очистка собственного кеша приложения: output/, benchmarks/ и временные файлы.

Граница этой функции проведена осознанно и намеренно узко. Под очистку попадают
ровно три категории, и ни одна из них не может дотянуться до данных, потеря
которых невосстановима:

- ``output`` — транзитные готовые файлы задач в корне ``output/``. Это тот же
  набор, что чистит TTL-очистка (`audio_pipeline.cleanup_output`), только по
  требованию и не дожидаясь срока;
- ``benchmarks`` — результаты сравнения движков в ``output/benchmarks/``. Они не
  попадают под TTL вообще (тот обходит только корень ``output/``), поэтому это
  первый настоящий кандидат на ручную очистку;
- ``temp_files`` — опознаваемые остатки самого приложения: ``*.tmp.wav`` рядом с
  целью в ``output/`` и ``output/projects/*/`` (их оставляет ``_write_audio``,
  когда конвертирует mp3 через промежуточный wav) и наши каталоги/файлы в
  системном temp с известными префиксами.

Чего здесь нет и не будет: ``models/`` и кеш ``huggingface_hub`` (гигабайты,
требующие интернета на повторную загрузку), ``data/voice_syntez.db`` (проекты и
словарь произношения), ``voices/`` (референсы и ``voices.json``) и
``output/projects/`` (take'ы проектов — это данные проекта, а не транзитный кеш).
Удаление весов — отдельный осознанный поток во вкладке `05` «Модели», и смешивать
его с этой формой нельзя.

Безопасность — главное требование, а не украшение. Путь каждой категории жёстко
задан кодом, а перед удалением проверяется, что реальный путь (после `resolve()`)
лежит внутри ожидаемого корня и что это файл (или наш временный каталог). Ошибка
доступа к одной записи не срывает очистку остальных: она логируется, остальное
удаляется. Пустая или отсутствующая категория — не ошибка, а ноль освобождённого.
"""

import logging
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from . import benchmark, config

logger = logging.getLogger(__name__)

# Категории, которые принимает API. Имена совпадают с ключами ответа, чтобы
# интерфейсу не приходилось держать вторую таблицу переводов.
TARGET_OUTPUT = "output"
TARGET_BENCHMARKS = "benchmarks"
TARGET_TEMP_FILES = "temp_files"
TARGETS = (TARGET_OUTPUT, TARGET_BENCHMARKS, TARGET_TEMP_FILES)

# Префиксы «наших» записей в системном temp. Это ровно те имена, которые
# создаёт код проекта: каталог экспорта (`project_export._WORK_PREFIX`), каталог
# импорта (`project_export._IMPORT_PREFIX`), каталог очистки референса
# (`denoise`) и временные файлы расшифровки/анализа (``voice-syntez-``).
TEMP_PREFIXES = ("tts-export-", "tts-import-", "voice-denoise-", "voice-syntez-")

# Порог возраста для записей в системном temp. Активный экспорт живёт секунды,
# а импорт — минуты, поэтому удалять их «на ходу» нельзя; час заведомо больше
# любой живой операции и заведомо меньше «мусора, который копился неделями».
TEMP_MIN_AGE_SEC = 3600.0

# Имена, по которым узнаются промежуточные файлы записи: `_write_audio` пишет
# результат под `<имя>.part` и конвертирует через `<имя>.part.wav` (раньше —
# `<имя>.tmp.wav`, такие остатки тоже узнаются).
TEMP_FILE_SUFFIX = ".tmp.wav"
TEMP_PART_MARKER = ".part"

# Что никогда не удаляется даже в «своей» категории: маркер пустого каталога в
# репозитории и подкаталоги, которые перечислены отдельными категориями.
_KEEP_NAMES = frozenset({".gitkeep"})


# --- инвентарь и очистка ------------------------------------------------------
def inventory() -> dict:
    """Сколько сейчас занимает каждая категория — для интерфейса.

    Считается по тому же набору путей, который удаляет `clear`, поэтому показанное
    число и освобождённое место не расходятся. Отсутствующий каталог даёт ноль, а
    не ошибку.
    """
    files, dirs = _scan(targets=TARGETS)
    result: dict = {}
    for target in TARGETS:
        size, count = _measure(files[target], dirs[target])
        result[f"{target}_mb"] = _to_mb(size)
        if target == TARGET_TEMP_FILES:
            result["temp_files_count"] = count
    return result


def clear(targets: Iterable[str], *, min_age_sec: float = TEMP_MIN_AGE_SEC) -> dict:
    """Освобождает выбранные категории и отчитывается, что именно удалено.

    Неизвестная категория — это ошибка вызывающего (API отвечает на неё 400
    раньше), поэтому здесь она не «проглатывается»: молча вернуть ноль значило бы
    сделать вид, что очистка состоялась. `min_age_sec` существует для тестов:
    порог возраста записей в системном temp, чтобы не удалить живую операцию.
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Неизвестная категория очистки: {', '.join(unknown)}")

    files, dirs = _scan(targets=targets, min_age_sec=min_age_sec)
    freed_mb: dict = {}
    freed_files: dict = {}
    total_bytes = 0
    for target in targets:
        size, count = _remove(target, files[target], dirs[target])
        freed_mb[target] = _to_mb(size)
        freed_files[target] = count
        total_bytes += size
        if count:
            logger.info(
                "Очистка кеша: %s — удалено %s записей, %.1f МБ",
                target, count, _to_mb(size),
            )
    if TARGET_BENCHMARKS in targets:
        _reset_benchmark_registry()
    return {
        "freed_mb": freed_mb,
        "freed_files": freed_files,
        "total_mb": _to_mb(total_bytes),
    }


def _reset_benchmark_registry() -> None:
    """Забывает запуски сравнения после удаления их файлов.

    Без этого интерфейс продолжил бы показывать строки сравнения со ссылками на
    несуществующие wav: реестр живёт в памяти процесса и о чистке диска не знает.
    """
    if benchmark.list_runs():
        logger.info("Очистка кеша: реестр сравнения движков сброшен")
    benchmark.reset_registry()


# --- где что лежит ------------------------------------------------------------
def _scan(targets: Iterable[str], min_age_sec: float = TEMP_MIN_AGE_SEC) -> tuple[dict, dict]:
    """Собирает записи по категориям: `{target: [файлы]}`, `{target: [каталоги]}`.

    Сканирование отделено от удаления, чтобы инвентарь и очистка опирались на один
    и тот же отбор: иначе интерфейс обещал бы одно, а кнопка удаляла другое.
    """
    wanted = set(targets)
    files: dict = {target: [] for target in TARGETS}
    dirs: dict = {target: [] for target in TARGETS}
    if TARGET_OUTPUT in wanted:
        files[TARGET_OUTPUT] = _output_files()
    if TARGET_BENCHMARKS in wanted:
        files[TARGET_BENCHMARKS] = _files_in(config.BENCHMARKS_DIR)
    if TARGET_TEMP_FILES in wanted:
        files[TARGET_TEMP_FILES] = _tmp_wav_files()
        ours = _system_temp_entries(min_age_sec)
        # В системном temp наши остатки бывают и файлами (расшифровка, анализ), и
        # каталогами (экспорт, импорт, очистка записи): удаляются они по-разному.
        files[TARGET_TEMP_FILES].extend(entry for entry in ours if entry.is_file())
        dirs[TARGET_TEMP_FILES] = [entry for entry in ours if entry.is_dir()]
    return files, dirs


def _output_files() -> list[Path]:
    """Только транзитные файлы в корне `output/`.

    Подкаталоги не трогаются: `output/projects/` — take'ы проектов, а
    `output/benchmarks/` — отдельная категория со своим реестром. `.gitkeep`
    остаётся всегда, иначе пустой каталог выпадет из репозитория. Промежуточные
    `.tmp.wav` сюда не входят: это своя категория, и считать их дважды значило бы
    обещать в интерфейсе больше места, чем освободится.
    """
    return [
        entry
        for entry in _iterdir(config.OUTPUT_DIR)
        if entry.is_file()
        and entry.name not in _KEEP_NAMES
        and not _is_temp_file(entry.name)
    ]


def _is_temp_file(name: str) -> bool:
    """Промежуточный ли это файл записи — по явному признаку, а не «похоже на временный»."""
    return name.endswith(TEMP_FILE_SUFFIX) or TEMP_PART_MARKER in name


def _files_in(directory: Path) -> list[Path]:
    """Файлы внутри каталога (без рекурсии, без подкаталогов и без `.gitkeep`)."""
    return [
        entry
        for entry in _iterdir(directory)
        if entry.is_file() and entry.name not in _KEEP_NAMES
    ]


def _tmp_wav_files() -> list[Path]:
    """Наши промежуточные файлы записи в `output/` и в каталогах take'ов проектов.

    Каталоги проектов обходятся по одному уровню: take'ы лежат как
    `output/projects/{id}/*.wav`, и `_write_audio` оставляет `<имя>.part` рядом с
    целью именно там. Отбор строгий — по явным признакам (`.tmp.wav`, `.part`), а
    не «всё, что похоже на временный файл»: чужой файл трогать нельзя.
    """
    found = [
        entry
        for entry in _iterdir(config.OUTPUT_DIR)
        if entry.is_file() and _is_temp_file(entry.name)
    ]
    projects = config.PROJECTS_OUTPUT_DIR
    if _is_dir(projects):
        for project_dir in _iterdir(projects):
            if not project_dir.is_dir():
                continue
            found.extend(
                entry
                for entry in _iterdir(project_dir)
                if entry.is_file() and _is_temp_file(entry.name)
            )
    return found


def _system_temp_entries(min_age_sec: float) -> list[Path]:
    """Наши записи в системном temp старше порога — файлы и каталоги.

    Отбор идёт по **явному префиксу из `TEMP_PREFIXES`**, а не по «похоже на
    временное»: в системном temp лежат чужие файлы, и удалить их было бы худшей
    ошибкой, чем оставить свой мусор. Возраст берётся по времени изменения самой
    записи: активный экспорт/импорт/денойз обновляет каталог, пока работает.
    """
    threshold = time.time() - max(min_age_sec, 0.0)
    roots = {Path(tempfile.gettempdir()), Path("/tmp")}
    found: list[Path] = []
    for root in roots:
        if not _is_dir(root):
            continue
        for entry in _iterdir(root):
            if not entry.name.startswith(TEMP_PREFIXES):
                continue
            try:
                if entry.stat().st_mtime >= threshold:
                    continue
            except OSError as exc:
                logger.warning("Не удалось проверить возраст %s: %s", entry, exc)
                continue
            found.append(entry)
    return found


# --- измерение и удаление -----------------------------------------------------
def _measure(files: Iterable[Path], dirs: Iterable[Path]) -> tuple[int, int]:
    size = 0
    count = 0
    for path in files:
        size += _file_size(path)
        count += 1
    for path in dirs:
        entries, bytes_ = _tree_size(path)
        size += bytes_
        count += entries
    return size, count


def _remove(target: str, files: Iterable[Path], dirs: Iterable[Path]) -> tuple[int, int]:
    """Удаляет записи, считая только действительно освобождённое.

    Перед каждым удалением проверяется, что реальный путь лежит внутри корня этой
    категории: даже ошибка в отборе не должна дотянуться до `data/`, `voices/`
    или `models/`. Ошибка на одной записи не срывает остальные — она логируется,
    а очистка идёт дальше. Так же ведёт себя TTL-очистка в пайплайне.
    """
    size = 0
    count = 0
    roots = _roots(target)
    for path in files:
        bytes_ = _file_size(path)
        # Файлы категории `temp_files` живут в двух разных мирах: промежуточные
        # файлы записи (`.part`, `.tmp.wav`) — рядом с целью в output/, а остатки
        # вроде `voice-syntez-*.wav` — в системном temp. Разрешение и проверка у
        # них разные.
        removed = (
            _unlink_in_temp(path)
            if target == TARGET_TEMP_FILES and not _is_temp_file(path.name)
            else _unlink(path, roots)
        )
        if removed:
            size += bytes_
            count += 1
    for path in dirs:
        entries, bytes_ = _tree_size(path)
        if _rmtree(path):
            size += bytes_
            count += entries
    return size, count


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        logger.warning("Не удалось измерить %s: %s", path, exc)
        return 0


def _tree_size(path: Path) -> tuple[int, int]:
    """Размер каталога и число файлов в нём (символические ссылки не разворачиваются)."""
    size = 0
    count = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        size += _file_size(entry)
        count += 1
    return count, size


def _unlink(path: Path, roots: tuple[Path, ...]) -> bool:
    """Удаляет файл, только если он внутри одного из разрешённых корней и не симлинк."""
    if not _within(path, roots):
        logger.warning("Пропускаю %s: путь вне ожидаемого каталога", path)
        return False
    if path.is_symlink():
        logger.warning("Пропускаю %s: символическая ссылка", path)
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Не удалось удалить %s: %s", path, exc)
        return False
    return True


def _unlink_in_temp(path: Path) -> bool:
    """Удаляет наш файл в системном temp: имя с нашим префиксом и путь внутри temp."""
    if not path.name.startswith(TEMP_PREFIXES):
        logger.warning("Пропускаю %s: имя без нашего префикса", path)
        return False
    if not _in_system_temp(path):
        logger.warning("Пропускаю %s: путь вне системного каталога временных файлов", path)
        return False
    if path.is_symlink():
        logger.warning("Пропускаю %s: символическая ссылка", path)
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Не удалось удалить %s: %s", path, exc)
        return False
    return True


def _rmtree(path: Path) -> bool:
    """Удаляет наш временный каталог: префикс в имени + путь внутри системного temp."""
    if not path.name.startswith(TEMP_PREFIXES):
        logger.warning("Пропускаю %s: имя без нашего префикса", path)
        return False
    if not _in_system_temp(path):
        logger.warning("Пропускаю %s: путь вне системного каталога временных файлов", path)
        return False
    if path.is_symlink():
        logger.warning("Пропускаю %s: символическая ссылка", path)
        return False
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Не удалось удалить %s: %s", path, exc)
        return False
    return True


# --- проверки безопасности ----------------------------------------------------
def _roots(target: str) -> tuple[Path, ...]:
    """Корни, внутри которых обязана лежать каждая запись категории.

    Путь категории жёстко задан кодом: `output` — только корень `output/`,
    `benchmarks` — только `output/benchmarks/`, `temp_files` — корень `output/` и
    каталоги take'ов проектов. Список не выводится из содержимого каталогов, а
    объявлен здесь, поэтому ошибочный вход не может его расширить.
    """
    if target == TARGET_OUTPUT:
        return (config.OUTPUT_DIR,)
    if target == TARGET_BENCHMARKS:
        return (config.BENCHMARKS_DIR,)
    return (config.OUTPUT_DIR, config.PROJECTS_OUTPUT_DIR)


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    """Лежит ли реальный путь внутри одного из ожидаемых корней (после `resolve()`).

    `is_relative_to` — а не сравнение строк: он не путает `/tmp/ab` с `/tmp/abc`
    и учитывает символические ссылки в родительских каталогах. Отсутствующий файл
    всё равно проверяется: `resolve()` работает и с несуществующим путём.
    """
    for root in roots:
        try:
            if path.resolve().is_relative_to(root.resolve()):
                return True
        except OSError as exc:
            logger.warning("Не удалось проверить путь %s: %s", path, exc)
    return False


def _in_system_temp(path: Path) -> bool:
    roots = [Path("/tmp"), Path(tempfile.gettempdir())]
    for root in roots:
        try:
            if path.resolve().is_relative_to(root.resolve()):
                return True
        except OSError:
            continue
    return False


def _iterdir(directory: Path) -> list[Path]:
    """Содержимое каталога; отсутствующий каталог — пустой список, а не исключение."""
    try:
        return list(directory.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Не удалось прочитать %s: %s", directory, exc)
        return []


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _to_mb(size: int) -> float:
    """Мегабайты с двумя знаками: интерфейсу нужна точность, а не много цифр."""
    return round(size / (1024 * 1024), 2)
