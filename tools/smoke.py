#!/usr/bin/env python3
"""Живой end-to-end smoke-тест: ручной сценарий §7 плана на реальных моделях.

Инструмент ходит по REST API запущенного дашборда и проходит путь пользователя
целиком: создать проект → назначить два голоса → разобрать диалог со смешанными
спикерами → отрендерить F5 + XTTS → Smart QA → перегенерировать реплику →
вернуть старый take → перезапустить backend → открыть проект заново → выгрузить
WAV и SRT → экспортировать проект → импортировать его → отрендерить снова.

Модели поднимает сам сервер: инструмент их не запускает и ничего не мокает.
Поэтому в pytest он не входит (см. `pytest.ini`) и запускается вручную, когда
дашборд уже работает. Если сервер недоступен, полный прогон завершается понятной
ошибкой до первого синтеза.

    ./venv/bin/python tools/smoke.py --dry-run          # план и проверка доступа к API
    ./venv/bin/python tools/smoke.py                    # полный прогон
    ./venv/bin/python tools/smoke.py --base-url http://127.0.0.1:8001
    ./venv/bin/python tools/smoke.py --no-restart       # без шага ручного перезапуска
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Инструмент запускается из корня репозитория и как `tools/smoke.py`: корень
# нужен, чтобы импортировался помощник лаунчера (только стандартная библиотека).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import launcher

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Диалог со смешанными спикерами: две роли, чтобы F5 и XTTS встретились в одной
# дорожке. Реплики короткие — прогон и так длинный.
SMOKE_DIALOGUE = (
    "АРТЕМ: Привет! Это дымовой тест студии.\n"
    "МАРГО: Отлично, я как раз готова записать первую сцену.\n"
    "АРТЕМ: Тогда начнём с короткой реплики."
)
# План шагов: он же печатается в --dry-run, поэтому текст и код не расходятся.
PLAN = (
    "создать проект",
    "добавить два голоса (F5 и XTTS)",
    "разобрать диалог со смешанными спикерами и назначить голоса",
    "подготовить диалог (анализ и review)",
    "рендер F5 + XTTS",
    "Smart QA (рендер с qa=smart)",
    "перегенерация реплики",
    "выбор старого take",
    "перезапуск backend (вручную)",
    "повторное открытие проекта",
    "экспорт WAV",
    "экспорт SRT",
    "экспорт проекта (.ttsproject)",
    "импорт проекта",
    "повторный рендер импортированного проекта",
)


class SmokeError(RuntimeError):
    """Шаг не прошёл: текст ошибки показывается в отчёте как есть."""


class Report:
    """Итоговая сводка PASS/FAIL/SKIP по шагам."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        self.items.append((name, status, detail))
        suffix = f": {detail}" if detail else ""
        print(f"  [{mark}] {name}{suffix}", flush=True)

    def summary(self) -> tuple[int, int, int]:
        passed = sum(1 for _name, status, _d in self.items if status == "PASS")
        failed = sum(1 for _name, status, _d in self.items if status == "FAIL")
        skipped = sum(1 for _name, status, _d in self.items if status == "SKIP")
        return passed, failed, skipped


class Api:
    """Минимальный клиент REST без сторонних зависимостей."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None):
        """Возвращает `(http_status, body_bytes)`. `0` — сеть недоступна."""
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            return 0, repr(exc).encode()

    def json(self, method: str, path: str, payload: dict | None = None, expect: tuple[int, ...] = (200, 201, 202)):
        status, body = self.call(method, path, payload)
        if status == 0:
            raise SmokeError(f"{method} {path}: сервер недоступен ({body.decode(errors='replace')})")
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise SmokeError(f"{method} {path}: ответ не JSON (HTTP {status})") from exc
        if status not in expect:
            detail = data.get("detail") if isinstance(data, dict) else data
            raise SmokeError(f"{method} {path}: HTTP {status} — {detail}")
        return data

    def download(self, method: str, path: str, payload: dict | None = None) -> bytes:
        status, body = self.call(method, path, payload, timeout=max(self.timeout, 600.0))
        if status != 200:
            try:
                detail = json.loads(body).get("detail")
            except ValueError:
                detail = body.decode("utf-8", errors="replace")[:200]
            raise SmokeError(f"{method} {path}: HTTP {status} — {detail}")
        return body

    def upload(self, path: str, field: str, filename: str, content: bytes) -> dict:
        """multipart/form-data без сторонних библиотек: один файл — одно поле."""
        boundary = f"----smoke{uuid.uuid4().hex}"
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=head + content + tail,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=max(self.timeout, 300.0)) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise SmokeError(f"POST {path}: сервер недоступен ({exc})") from exc
        if status != 201:
            try:
                detail = json.loads(body).get("detail")
            except ValueError:
                detail = body.decode("utf-8", errors="replace")[:200]
            raise SmokeError(f"POST {path}: HTTP {status} — {detail}")
        return json.loads(body)

    def status(self) -> dict | None:
        status, body = self.call("GET", "/api/status", timeout=5.0)
        if status != 200:
            return None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        return data if isinstance(data, dict) and "engines" in data else None


def pick_voices(api: Api, f5_id: str | None, xtts_id: str | None) -> dict[str, str]:
    """Два голоса для смешанного диалога: один на F5, один на XTTS.

    Новые голоса инструмент не создаёт: референс — это запись пользователя, и
    подсовывать её автоматически нельзя. Если подходящих голосов нет, ошибка
    говорит, что сделать в интерфейсе.
    """
    voices = api.json("GET", "/api/voices")["voices"]
    by_id = {voice["id"]: voice for voice in voices}
    if f5_id:
        if f5_id not in by_id:
            raise SmokeError(f"голос --f5-voice {f5_id} не найден в /api/voices")
        f5 = by_id[f5_id]
    else:
        f5 = next((voice for voice in voices if voice.get("engine") == "f5"), None)
    if xtts_id:
        if xtts_id not in by_id:
            raise SmokeError(f"голос --xtts-voice {xtts_id} не найден в /api/voices")
        xtts = by_id[xtts_id]
    else:
        xtts = next(
            (voice for voice in voices if voice.get("engine") in ("xtts", "xtts-banana")), None
        )
    if f5 is None or xtts is None:
        raise SmokeError(
            "нужны два голоса: один на движке f5 и один на xtts (или xtts-banana). "
            "Создайте их во вкладке «Голоса» дашборда или укажите --f5-voice/--xtts-voice."
        )
    if not f5.get("has_audio") or not xtts.get("has_audio"):
        raise SmokeError("у выбранных голосов нет файла референса — пересоздайте их в дашборде")
    return {
        "f5": f5["id"],
        "xtts": xtts["id"],
        "xtts_engine": xtts.get("engine") or "xtts",
        "f5_name": f5["name"],
        "xtts_name": xtts["name"],
    }


def wait_for_job(api: Api, job_id: str, timeout: float, label: str) -> dict:
    """Ждёт завершения задачи, печатая прогресс, и возвращает финальное состояние."""
    started = time.monotonic()
    last_message = ""
    while time.monotonic() - started < timeout:
        data = api.json("GET", f"/api/jobs/{job_id}")
        message = data.get("message") or ""
        if message != last_message:
            print(f"      {label}: {message}", flush=True)
            last_message = message
        if data["status"] in ("done", "error", "cancelled"):
            if data["status"] != "done":
                raise SmokeError(f"{label}: задача {job_id} завершилась со статусом {data['status']}")
            return data
        time.sleep(1.0)
    raise SmokeError(f"{label}: задача {job_id} не завершилась за {timeout:.0f} с")


def wait_for_server(api: Api, timeout: float) -> dict:
    """Ждёт, пока сервер снова отвечает на /api/status."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        data = api.status()
        if data is not None:
            return data
        time.sleep(1.0)
    raise SmokeError(f"сервер не ответил за {timeout:.0f} с после перезапуска")


def prepare_project(api: Api, project_id: str) -> dict:
    """Обязательная подготовка диалога: анализ, затем review найденных слов.

    Без неё рендер запрещён (409) — и это правильно: синтез идёт по сохранённому
    тексту. Кандидаты принимаются в словарь **проекта**: проверка идёт на живом
    сервере, и трогать общий словарь пользователя ради неё нельзя.
    """
    state = api.json("POST", f"/api/projects/{project_id}/analyze", {})
    for _ in range(5):
        candidates = state.get("candidates") or []
        if state.get("status") != "needs_review" or not candidates:
            break
        for candidate in candidates:
            api.json(
                "POST",
                f"/api/projects/{project_id}/pronunciation/review",
                {
                    "source": candidate["word"],
                    "target": candidate["target"] or candidate["word"],
                    "scope": "project",
                    "enabled": bool((candidate.get("target") or "").strip()),
                    "replica_index": candidate["replica_index"],
                },
            )
        state = await_state(api, project_id)
    state = await_state(api, project_id)
    if state.get("status") != "ready":
        raise SmokeError(
            f"подготовка не дошла до ready: {state.get('status')} ({state.get('analysis_error')})"
        )
    return state


def await_state(api: Api, project_id: str) -> dict:
    """Состояние подготовки из `/analysis`: у отчёта анализа поля счётчиков другие.

    Читать счётчики только из одного ответа — верный способ разойтись в подписях:
    `analyze` отдаёт `replicas_analyzed`, а состояние проекта — `replicas_done`.
    """
    return api.json("GET", f"/api/projects/{project_id}/analysis")


def render_and_wait(api: Api, project_id: str, qa: str, timeout: float, label: str) -> dict:
    """Ставит рендер проекта и дожидается результата; возвращает карточку проекта."""
    api.json("POST", f"/api/projects/{project_id}/render", {"qa": qa})
    project = api.json("GET", f"/api/projects/{project_id}")
    job_id = project.get("job_id")
    if job_id:
        wait_for_job(api, job_id, timeout, label)
    return api.json("GET", f"/api/projects/{project_id}")


# --- шаги сценария -------------------------------------------------------------
def run_scenario(api: Api, report: Report, args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    project_id = ""
    imported_id = ""
    old_take_id: int | None = None
    try:
        # 1. Проект
        project = api.json(
            "POST",
            "/api/projects",
            {"name": f"SMOKE {time.strftime('%Y-%m-%d %H:%M')}", "source_text": SMOKE_DIALOGUE},
        )
        project_id = project["id"]
        report.add(PLAN[0], "PASS", f"проект {project_id}")

        # 2. Два голоса: берём уже созданные, ничего не подменяя.
        voices = pick_voices(api, args.f5_voice, args.xtts_voice)
        report.add(PLAN[1], "PASS", f"F5 «{voices['f5_name']}», XTTS «{voices['xtts_name']}»")

        # 3. Разбор текста и назначение голосов спикерам.
        parsed = api.json("POST", f"/api/projects/{project_id}/parse", {"chunk_strategy": "short"})
        speakers = [item["key"] for item in parsed["speakers"]]
        if len(speakers) < 2:
            raise SmokeError(f"в диалоге ожидались два спикера, разобрано: {speakers}")
        assignment = {
            key: {"voice_id": voices["f5"] if index == 0 else voices["xtts"]}
            for index, key in enumerate(speakers)
        }
        project = api.json("PATCH", f"/api/projects/{project_id}", {"speakers": assignment})
        engines = sorted({replica["engine"] for replica in project["replicas"]})
        # Смешанный диалог — это F5 плюс XTTS-семейство: у второго движка два
        # варианта (базовая модель и русский файнтюн), и оба годятся. Требовать
        # ровно «xtts» значило бы падать на рабочей конфигурации пользователя.
        second = [item for item in engines if item in ("xtts", "xtts-banana")]
        if "f5" not in engines or len(second) != 1 or len(engines) != 2:
            raise SmokeError(
                f"в проекте ожидались два движка: f5 и один из xtts/xtts-banana, получено {engines}"
            )
        report.add(PLAN[2], "PASS", f"спикеров {len(speakers)}, реплик {len(project['replicas'])}, движки {engines}")

        # 3b. Обязательная подготовка: анализ и review найденных слов. Рендер без
        # неё сервер не примет, поэтому шаг стоит до первого синтеза.
        state = prepare_project(api, project_id)
        report.add(
            PLAN[3],
            "PASS",
            f"состояние {state['status']}, подготовлено {state['replicas_done']}/{state['replicas_total']}",
        )

        # 4. Смешанный рендер F5 + XTTS.
        project = render_and_wait(api, project_id, "off", args.timeout, "рендер")
        _require_takes(project, "рендер")
        report.add(PLAN[4], "PASS", f"{len(project['replicas'])} реплик с take'ами")

        # 5. Smart QA тем же текстом.
        project = render_and_wait(api, project_id, "smart", args.timeout, "smart qa")
        _require_takes(project, "smart qa")
        modes = sorted({(replica.get("qa") or {}).get("mode", "—") for replica in project["replicas"]})
        report.add(PLAN[5], "PASS", f"qa режимы: {modes}")

        # 6. Перегенерация первой реплики: появляется второй take. Задача у
        # перегенерации своя — проект её статусом не отмечает, поэтому ждём
        # именно job_id из ответа, а не прошлый рендер.
        before = api.json("GET", f"/api/projects/{project_id}")
        takes_before = before["replicas"][0]["takes"]
        regen = api.json("POST", f"/api/projects/{project_id}/replicas/0/regenerate")
        wait_for_job(api, regen["job_id"], args.timeout, "перегенерация")
        project = api.json("GET", f"/api/projects/{project_id}")
        replica = project["replicas"][0]
        if len(replica["takes"]) <= len(takes_before):
            raise SmokeError(
                f"после перегенерации ожидался новый take: было {len(takes_before)}, "
                f"стало {len(replica['takes'])}"
            )
        report.add(PLAN[6], "PASS", f"take'ов у реплики 0: {len(takes_before)} → {len(replica['takes'])}")

        # 7. Возврат старого take — без синтеза.
        current_id = replica["selected_take_id"]
        old_take_id = next(take["id"] for take in replica["takes"] if take["id"] != current_id)
        selected = api.json(
            "POST", f"/api/projects/{project_id}/replicas/0/takes/{old_take_id}"
        )
        if selected["replica"]["selected_take_id"] != old_take_id:
            raise SmokeError("выбор старого take не применился")
        report.add(PLAN[7], "PASS", f"активный take снова {old_take_id}")

        # 8. Перезапуск backend: его делает человек, инструмент только ждёт.
        if args.no_restart:
            report.add(PLAN[8], "SKIP", "пропущено по --no-restart")
        else:
            print(
                "\n  Перезапустите backend вручную (Ctrl+C и снова ./run.sh или "
                "VOICE_SYNTEZ.command), затем нажмите Enter…",
                flush=True,
            )
            try:
                input()
            except EOFError as exc:
                raise SmokeError(
                    "перезапуск требует интерактивного терминала; используйте --no-restart"
                ) from exc
            wait_for_server(api, args.timeout)
            report.add(PLAN[8], "PASS", "сервер снова отвечает")

        # 9. Проект пережил перезапуск.
        reopened = api.json("GET", f"/api/projects/{project_id}")
        if reopened["id"] != project_id or not reopened["replicas"]:
            raise SmokeError("проект не восстановился после перезапуска")
        _require_takes(reopened, "повторное открытие")
        report.add(PLAN[9], "PASS", f"реплик {len(reopened['replicas'])}, take'ы на месте")

        # 10-12. Экспорт: WAV, SRT и архив проекта.
        wav = api.download("GET", f"/api/projects/{project_id}/export/audio?format=wav")
        (workdir / "smoke.wav").write_bytes(wav)
        if len(wav) < 44:
            raise SmokeError(f"WAV подозрительно мал: {len(wav)} байт")
        report.add(PLAN[10], "PASS", f"{len(wav)} байт → {workdir / 'smoke.wav'}")

        srt = api.download("GET", f"/api/projects/{project_id}/export/subtitles?format=srt")
        (workdir / "smoke.srt").write_bytes(srt)
        text = srt.decode("utf-8", errors="replace")
        if "-->" not in text:
            raise SmokeError("в SRT нет таймстемпов '-->'")
        report.add(PLAN[11], "PASS", f"таймстемпов {text.count('-->')}")

        archive = api.download("POST", f"/api/projects/{project_id}/export")
        archive_path = workdir / "smoke.ttsproject"
        archive_path.write_bytes(archive)
        if not archive.startswith(b"PK"):
            raise SmokeError("архив проекта не похож на zip")
        report.add(PLAN[12], "PASS", f"{len(archive)} байт → {archive_path}")

        # 13. Импорт архива — новый проект, исходный не трогается.
        imported = api.upload("/api/projects/import", "file", "smoke.ttsproject", archive)
        imported_id = imported["id"]
        # Импортированный проект приходит без подготовки — и это правильно: текст
        # готовится под словарь и движки **этой** машины, а архив переносит проект,
        # а не результаты чужой подготовки.
        prepare_project(api, imported_id)
        if imported_id == project_id:
            raise SmokeError("импорт должен создавать новый проект")
        report.add(PLAN[13], "PASS", f"новый проект {imported_id}")

        # 14. Повторный рендер уже импортированного проекта.
        imported = render_and_wait(api, imported_id, "off", args.timeout, "повторный рендер")
        _require_takes(imported, "повторный рендер")
        report.add(PLAN[14], "PASS", f"{len(imported['replicas'])} реплик с take'ами")
    except SmokeError as exc:
        report.add("сценарий", "FAIL", str(exc))
    finally:
        if project_id:
            print(f"\n  Проект прогона: {project_id}")
        if imported_id:
            print(f"  Импортированный проект: {imported_id}")
        print(f"  Файлы экспорта: {workdir}")
        if args.cleanup:
            for target in (project_id, imported_id):
                if target:
                    status, _body = api.call("DELETE", f"/api/projects/{target}")
                    print(f"  Удалён проект {target}: HTTP {status}")


def _require_takes(project: dict, label: str) -> None:
    """У каждой реплики должен быть активный take — иначе рендер не состоялся."""
    missing = [
        replica["index"]
        for replica in project["replicas"]
        if not replica.get("selected_take_id") or not replica.get("takes")
    ]
    if missing:
        raise SmokeError(f"{label}: у реплик {missing} нет take'ов")


def print_plan(base_url: str) -> None:
    print("План ручного end-to-end сценария (§7):")
    for index, title in enumerate(PLAN, start=1):
        print(f"  {index:>2}. {title}")
    print(f"\nСервер: {base_url}")


def resolve_base_url(explicit: str | None) -> str:
    """Явный адрес или порт уже запущенного приложения (чтобы не гадать про 8000)."""
    if explicit:
        return explicit.rstrip("/")
    port = launcher.running_port(8000)
    return f"http://127.0.0.1:{port}" if port else DEFAULT_BASE_URL


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="smoke.py",
        description="Живой end-to-end smoke-тест TTS-дашборда на реальных моделях.",
    )
    parser.add_argument("--base-url", help="адрес дашборда; по умолчанию ищется запущенный на 8000+")
    parser.add_argument("--dry-run", action="store_true", help="напечатать план и проверить доступность API без синтеза")
    parser.add_argument("--no-restart", action="store_true", help="не ждать ручного перезапуска backend")
    parser.add_argument("--cleanup", action="store_true", help="удалить созданные проекты в конце")
    parser.add_argument("--f5-voice", help="id голоса на F5 (по умолчанию первый подходящий)")
    parser.add_argument("--xtts-voice", help="id голоса на XTTS (по умолчанию первый подходящий)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="таймаут задач синтеза, с")
    parser.add_argument(
        "--workdir",
        default="output/smoke",
        help="куда складывать файлы экспорта (по умолчанию output/smoke)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    base_url = resolve_base_url(args.base_url)
    api = Api(base_url)
    print_plan(base_url)

    if args.dry_run:
        status = api.status()
        if status is None:
            print("\nСервер недоступен — проверен только план; синтез не запускался.")
            return 0
        engines = [(engine.get("id"), engine.get("state")) for engine in status.get("engines") or []]
        print(
            f"\nAPI доступен: устройство={status.get('device')}, движки={engines}, "
            f"очередь={status.get('queue_size')}. Синтез в --dry-run не запускается."
        )
        print("Готово: dry-run (без моделей).")
        return 0

    status = api.status()
    if status is None:
        print(
            f"\nСервер {base_url} недоступен. Полный прогон невозможен: инструмент не поднимает "
            "модели сам. Запустите дашборд (bash VOICE_SYNTEZ.command или ./run.sh) и повторите.",
            file=sys.stderr,
        )
        return 2
    print(
        f"\nAPI доступен: устройство={status.get('device')}, очередь={status.get('queue_size')}.\n"
        "Начинаю прогон (это поднимает реальные модели)…\n"
    )

    report = Report()
    run_scenario(api, report, args)
    passed, failed, skipped = report.summary()
    print(f"\n=== ИТОГ: PASS {passed}, FAIL {failed}, SKIP {skipped} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
