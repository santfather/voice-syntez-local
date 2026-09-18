"""Версионный prompt: один и тот же текст для benchmark и Analyzer (§8).

Prompt — часть воспроизводимости, а не строчка в коде: его версия сохраняется
рядом с результатами, и смена prompt'а обязана инвалидировать сохранённый анализ
(Task 2 §11). Поэтому он лежит файлом с заголовком версии, а не собирается
f-строками по месту.

Файл состоит из двух секций, `[system]` и `[user]`; в пользовательской секции
плейсхолдер `{{case}}` заменяется JSON-объектом кейса. Такой формат удобно
ревьюить глазами и диффить в git — в отличие от строки, склеенной в коде.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "russian_linguistics" / "prompts"
ANALYZER_PROMPT = "analyzer.v3.txt"
# v2 остаётся на диске: по нему получены сохранённые benchmark-результаты, и
# удалять его значило бы сделать прошлые отчёты невоспроизводимыми.
ANALYZER_PROMPT_V2 = "analyzer.v2.txt"
# v1 остаётся в дереве: по нему уже были прогоны, и он нужен для сравнения
# «до/после» — но по умолчанию используется v2.
ANALYZER_PROMPT_V1 = "analyzer.v1.txt"
PLACEHOLDER = "{{case}}"
# Схему ответа prompt получает плейсхолдером, а не копией в тексте: иначе через
# месяц файл и валидатор разошлись бы, и модель учили бы одной схеме, а проверяли
# другой.
SCHEMA_PLACEHOLDER = "{{schema}}"


@dataclass(frozen=True)
class PromptTemplate:
    """Разобранный prompt: версия, системная часть и шаблон пользовательской."""

    name: str
    version: str
    system: str
    user_template: str
    path: str = ""

    def messages(
        self,
        case_payload: dict,
        *,
        schema_json: str = "",
        system_extra: str = "",
    ) -> list[dict]:
        """Сообщения для `/api/chat`: system + user с подставленным кейсом.

        JSON кейса идёт с `ensure_ascii=False`: модели проще читать русский текст
        как русский, а не как `\\u0442`-последовательности. `schema_json` — та же
        схема, по которой backend проверяет ответ: модель обязана видеть ровно её,
        а не пересказ.
        """
        case_json = json.dumps(case_payload, ensure_ascii=False, indent=2)
        user = self.user_template.replace(PLACEHOLDER, case_json)
        system = self.system.replace(SCHEMA_PLACEHOLDER, schema_json) if schema_json else self.system
        system = system if not system_extra else f"{system}\n\n{system_extra}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def parse_prompt(text: str, *, name: str, path: str = "") -> PromptTemplate:
    """Разбирает файл prompt'а на секции и версию.

    Версия обязательна: без неё нельзя понять, почему вчерашний анализ перестал
    совпадать с сегодняшним.
    """
    version = ""
    for line in text.splitlines():
        match = re.match(r"^#\s*version:\s*(\S+)", line.strip())
        if match:
            version = match.group(1)
            break
    if not version:
        raise ValueError(f"в prompt «{name}» не указана версия (# version: N)")

    sections: dict[str, list[str]] = {"system": [], "user": []}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "[system]":
            current = "system"
            continue
        if stripped == "[user]":
            current = "user"
            continue
        if current:
            sections[current].append(line)

    system = "\n".join(sections["system"]).strip()
    user = "\n".join(sections["user"]).strip()
    if not system or not user:
        raise ValueError(f"prompt «{name}»: нужны непустые секции [system] и [user]")
    if PLACEHOLDER not in user:
        raise ValueError(f"prompt «{name}»: в секции [user] нет {PLACEHOLDER}")
    if SCHEMA_PLACEHOLDER not in system:
        raise ValueError(f"prompt «{name}»: в секции [system] нет {SCHEMA_PLACEHOLDER}")
    return PromptTemplate(name=name, version=version, system=system, user_template=user, path=path)


def load_prompt(name: str = ANALYZER_PROMPT, *, directory: Path | None = None) -> PromptTemplate:
    """Читает prompt из каталога `benchmarks/russian_linguistics/prompts`."""
    base = Path(directory) if directory else PROMPTS_DIR
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"prompt не найден: {path}")
    return parse_prompt(path.read_text(encoding="utf-8"), name=name, path=str(path))
