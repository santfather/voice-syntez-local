"""Очистка референса от шума и реверба перед сохранением голоса.

Реверб модель клонирует как «характеристику голоса»: в синтезе он звучит как каша, и
исправить это после синтеза нельзя — только на этапе референса. Здесь запись прогоняется
через DeepFilterNet (`denoise_worker`) и заменяет исходный файл голоса.

Зависимость опциональная и ставится отдельно от основного набора:

    ./venv/bin/pip install --no-deps -r requirements-denoise.txt

`--no-deps` обязателен: deepfilternet требует numpy<2.0, а проект проверен на numpy 2.4.6
(сама библиотека на numpy 2.x работает — проверено). Если пакета нет, `clean_bytes` вернёт
понятную ошибку, а интерфейс погасит тумблер (см. `/api/status`).
"""

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

WORKER_MODULE = "backend.denoise_worker"
WORKER_TIMEOUT_SEC = float(os.environ.get("TTS_DENOISE_TIMEOUT_SEC", "600"))

INSTALL_HINT = (
    "DeepFilterNet не установлен, поэтому очистка записи недоступна. Поставьте "
    "опциональные зависимости: ./venv/bin/pip install --no-deps -r requirements-denoise.txt"
)

# Ошибка импорта внутри воркера так выглядит в stderr; по ней отличаем «нет зависимости»
# от настоящего сбоя обработки.
_MISSING_MODULE_RE = re.compile(r"No module named '(df|libdf)'")


def is_available() -> bool:
    """Установлен ли DeepFilterNet (наличие модулей, без импорта: он тянет torch)."""
    return all(importlib.util.find_spec(name) for name in ("df", "libdf"))


def clean_bytes(data: bytes, suffix: str) -> bytes:
    """Прогоняет запись через DeepFilterNet и возвращает WAV с очищенным звуком.

    Возвращаются именно байты: вызывающий код (`VoicesStore.create`) решает сам, под каким
    именем их сохранить — после очистки расширение всегда wav, независимо от исходного.
    """
    with tempfile.TemporaryDirectory(prefix="voice-denoise-") as tmpdir:
        source_path = Path(tmpdir) / f"source{suffix}"
        target_path = Path(tmpdir) / "cleaned.wav"
        source_path.write_bytes(data)
        command = [sys.executable, "-m", WORKER_MODULE, str(source_path), str(target_path)]
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
            raise RuntimeError(
                f"очистка записи не уложилась в {WORKER_TIMEOUT_SEC:.0f} с"
            ) from exc
        if proc.returncode != 0:
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            if _MISSING_MODULE_RE.search(proc.stderr or ""):
                raise RuntimeError(INSTALL_HINT)
            raise RuntimeError(f"очистка записи не удалась: {tail or 'код ' + str(proc.returncode)}")
        if not target_path.exists():
            raise RuntimeError("очистка записи не вернула файл")
        logger.info("Референс очищен: %s", _summary(proc.stdout))
        return target_path.read_bytes()


def _summary(stdout: str) -> str:
    """Читаемая сводка из последней JSON-строки воркера (или весь вывод, если её нет)."""
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return stdout.strip() or "без подробностей"
    return (
        f"{payload.get('target_sec')} с, {payload.get('sample_rate')} Гц, "
        f"пик {payload.get('peak')}"
    )
