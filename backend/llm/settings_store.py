"""Настройки анализатора, которые меняются из интерфейса.

Окружение (`LLM_*`) — значения по умолчанию для развёртывания, а файл
`data/llm_settings.json` — то, что пользователь переключил в приложении. Файл
перекрывает окружение, потому что иначе галочка в интерфейсе не работала бы после
перезапуска: «включил, перезапустил — снова выключено» — это не настройка, а
недоразумение.

Почему не в SQLite: это настройка приложения, а не данные проекта. Она нужна до
первого обращения к базе (анализатор создаётся лениво) и переживает перенос базы.
Файл читается при создании анализатора и пишется одним атомарным `replace`, чтобы
падение в момент записи не оставляло половину JSON.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("tts.llm.settings")

SETTINGS_ENV = "TTS_LLM_SETTINGS_PATH"
# Поля, которые можно менять из интерфейса. Список закрытый: остальное —
# развёртывание (адрес Ollama, таймауты), и менять это галочкой нельзя.
EDITABLE_FIELDS: tuple[str, ...] = (
    "enabled",
    "primary_model",
    "fallback_model",
    "required_for_render",
    "num_ctx",
    "context_replicas",
)


def settings_path() -> Path:
    """Путь к файлу настроек: окружение или `data/llm_settings.json`."""
    explicit = os.environ.get(SETTINGS_ENV, "").strip()
    if explicit:
        return Path(explicit)
    from .. import config

    return Path(config.DATA_DIR) / "llm_settings.json"


def load_settings(path: Path | None = None) -> dict:
    """Читает сохранённые настройки. Нет файла или битый JSON — пустой словарь.

    Битый файл не должен мешать приложению работать: анализатор соберётся на
    значениях окружения, а что именно было не так — в логе.
    """
    target = Path(path) if path is not None else settings_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Не удалось прочитать настройки анализатора (%s): %s", target, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in EDITABLE_FIELDS}


def save_settings(patch: dict, path: Path | None = None) -> dict:
    """Сохраняет изменение настроек и возвращает полный набор.

    Проверяются только типы: модель и пороги зависят от машины, и «модель не
    скачана» — это состояние, которое анализатор покажет в статусе, а не повод
    отказывать в сохранении настройки.
    """
    target = Path(path) if path is not None else settings_path()
    current = load_settings(target)
    for key, value in patch.items():
        if key not in EDITABLE_FIELDS or value is None:
            continue
        if key in ("enabled", "required_for_render"):
            current[key] = bool(value)
        elif key in ("num_ctx", "context_replicas"):
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                current[key] = number
        else:
            text = str(value).strip()
            if text:
                current[key] = text
    target.parent.mkdir(parents=True, exist_ok=True)
    # Атомарная замена: половина JSON в файле настроек хуже, чем отсутствие файла.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(current, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)
    logger.info("Настройки анализатора сохранены: %s", ", ".join(sorted(current)))
    return current
