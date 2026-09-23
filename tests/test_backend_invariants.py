"""Инварианты `backend/`, на которых держится диагностика падений (P2 §4.2, P3 §5.3).

Два инварианта, которые нарушаются одной строкой и не видны на ревью:

1. **Никаких вызовов, убивающих процесс мимо Python.** `os.abort()` и `os._exit()`
   не оставляют ни исключения, ни строки в журнале: разбирать такое падение
   приходится по форме кадров в `Python-*.ips`. `faulthandler.disable()` выключает
   единственный источник Python-стека при нативном падении (он включён в воркере,
   см. `worker_process._enable_faulthandler`), а `set_start_method("fork")` —
   ровно тот `fork` после инициализации MPS, ради ухода от которого воркер
   стартует через `Popen` (см. `docs/architecture.md`).

2. **Тяжёлые нативные библиотеки не встречаются в одном процессе с torch.**
   В venv лежат две копии `libomp.dylib` — у torch и у sklearn; встретившись в
   одном процессе, они дают настоящий SIGABRT из `__kmp_abort_process`, который
   ничем не отличается от тестового падения из `tests/fake_worker_engine.py`.
   Поэтому DeepFilterNet живёт в отдельном процессе (`backend/denoise_worker.py`),
   а `backend/denoise.py` остаётся обёрткой над `subprocess.run`.
   `KMP_DUPLICATE_LIB_OK=TRUE` проблему не решает, а только маскирует.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

# Атрибуты `os`, которые нельзя звать ни напрямую, ни через псевдоним (значение —
# текст для сообщения теста).
FORBIDDEN_OS_ATTRS = {
    "abort": "нативное падение мимо обработчиков: ни лога, ни исключения",
    "_exit": "выход без finally и без flush — диагностика теряется",
}

# Прочие запрещённые вызовы по полному имени.
FORBIDDEN_CALLS = {
    "faulthandler.disable": "выключает Python-стек нативного падения",
}

# Пакеты, которые нельзя импортировать в процесс, где живёт torch: своя копия
# libomp или свой аллокатор — источник настоящего SIGABRT и конфликта потоков.
HEAVY_NATIVE_PACKAGES = ("df", "libdf", "sklearn", "onnxruntime")


def _backend_modules() -> list[Path]:
    return sorted(BACKEND.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.expr) -> str:
    """`os.path.join` из атрибутов и имён; пустая строка, если это не цепочка имён."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _os_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Имена, которыми в модуле зовётся `os`.

    Возвращает (имена модуля, имена, импортированные из `os` поимённо). Без этого
    правило обходится одной строкой: `import os as o` или `from os import abort`
    дают тот же самый вызов под другим именем.
    """
    modules = {"os"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    modules.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            imported.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in FORBIDDEN_OS_ATTRS
            )
    return modules, imported


def _forbidden_reason(name: str, os_modules: set[str], os_imported: set[str]) -> str | None:
    """Почему вызов с таким именем запрещён, или None.

    Имя `abort` само по себе ничего не значит — в `audio_pipeline.py` так названа
    локальная переменная с методом движка. Поэтому запрет срабатывает не на имя, а
    на его происхождение: `os.abort` (в том числе через псевдоним модуля) или
    `from os import abort`.
    """
    reason = FORBIDDEN_CALLS.get(name)
    if reason is not None:
        return reason
    head, _, attr = name.rpartition(".")
    if head in os_modules and attr in FORBIDDEN_OS_ATTRS:
        return FORBIDDEN_OS_ATTRS[attr]
    if not head and name in os_imported:
        return FORBIDDEN_OS_ATTRS[name]
    return None


def test_backend_has_no_process_killing_calls():
    """В `backend/` нет вызовов, обходящих логирование и обработчики исключений."""
    found: list[str] = []
    for path in _backend_modules():
        tree = _parse(path)
        os_modules, os_imported = _os_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted_name(node.func)
            reason = _forbidden_reason(name, os_modules, os_imported)
            if reason is not None:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}() — {reason}")
            if name.endswith("set_start_method") and any(
                isinstance(arg, ast.Constant) and arg.value == "fork" for arg in node.args
            ):
                found.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} set_start_method('fork') — "
                    "fork после инициализации MPS ломает дочерний процесс на macOS"
                )
    assert not found, "запрещённые вызовы в backend/:\n" + "\n".join(found)


def test_no_module_level_import_of_heavy_native_libs_in_backend():
    """Тяжёлые нативные пакеты не импортируются на уровне модуля в `backend/`.

    Импорт внутри функции (как в `denoise_worker.main`) допустим: он выполняется
    только в том процессе, который для этой библиотеки и создан. Ленивый импорт —
    не стилистика, а гарантия, что `import backend.main` не притащит вторую
    `libomp` в процесс с torch.
    """
    found: list[str] = []
    for path in _backend_modules():
        for node in _parse(path).body:  # только верхний уровень модуля
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in HEAVY_NATIVE_PACKAGES:
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} import {name}")
    assert not found, "импорт тяжёлых пакетов на уровне модуля:\n" + "\n".join(found)


def test_denoise_wrapper_does_not_pull_heavy_libs_into_process():
    """Импорт обёртки шумоподавления не подтягивает `df`/`sklearn`/`onnxruntime`.

    Проверка идёт в отдельном интерпретаторе: в общем прогоне эти пакеты мог
    импортировать другой тест, и вердикт зависел бы от порядка тестов.
    """
    script = textwrap.dedent(
        """
        import json, sys
        import backend.denoise
        print(json.dumps([n for n in ("df", "libdf", "sklearn", "onnxruntime") if n in sys.modules]))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    pulled = json.loads(result.stdout.strip().splitlines()[-1])
    assert pulled == [], f"backend.denoise тянет в процесс: {pulled}"
