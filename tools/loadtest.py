#!/usr/bin/env python3
"""Нагрузочный тест локального TTS-дашборда.

Фазы (можно запускать выборочно):
  light   — последовательная латентность лёгких ручек;
  load    — те же ручки при конкурентности 1/5/20/50;
  heavy   — отзывчивость API во время тяжёлой задачи с qa=strict;
  queue   — 6 задач разом: строгая серийность, ожидание и время работы;
  qa      — один и тот же текст в режимах off/smart/strict: цена проверки;
  engines — XTTS холодный и прогретый плюс смешанный диалог F5 + XTTS.

Ресурсы (RSS сервера, RSS дочерних процессов Whisper, CPU, память системы)
снимаются фоновым сэмплером и печатаются по каждой фазе отдельно.

Примеры:
    ./venv/bin/python tools/loadtest.py               # все фазы
    ./venv/bin/python tools/loadtest.py light queue   # только выбранные
    ./venv/bin/python tools/loadtest.py --base-url http://127.0.0.1:8000 engines
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# Голоса берутся из voices.json дашборда: F5 — финетюн под русский, XTTS — v2.
F5_VOICE = "70ddd8f1acb8"
XTTS_VOICE = "9e13d67577fa"

DIALOGUE_ONE = "ИВАН: Сегодня хорошая погода."
DIALOGUE_MARGO = "МАРГО: Добрый вечер, я как раз собиралась прогуляться."
DIALOGUE_THREE = (
    "ИВАН: Сегодня хорошая погода, и я рад тебя видеть.\n"
    "ИВАН: Давай прогуляемся по набережной после обеда.\n"
    "ИВАН: Только возьму куртку и зонт на всякий случай."
)
# Смешанный диалог: чётные реплики — F5, нечётные — XTTS, чтобы прогрев и цена
# второго движка были видны на фоне уже поднятого первого.
DIALOGUE_MIXED = (
    "ИВАН: Добрый вечер, Маргарита.\n"
    "МАРГО: Добрый вечер. Я как раз собиралась прогуляться.\n"
    "ИВАН: Тогда позвольте составить вам компанию.\n"
    "МАРГО: С удовольствием, только возьму зонт.\n"
    "ИВАН: Непременно, сегодня обещали дождь.\n"
    "МАРГО: Значит, прогулка будет короткой."
)

LIGHT_ROUTES = [
    ("GET", "/api/status", None),
    ("GET", "/api/engines", None),
    ("GET", "/api/voices", None),
    ("POST", "/api/parse", {"dialogue_text": DIALOGUE_THREE, "chunk_strategy": "short"}),
]

PHASES = ("light", "load", "heavy", "queue", "qa", "engines")
REPLICA_MESSAGE = re.compile(r"Реплика (\d+) из (\d+)")

PRINT_LOCK = threading.Lock()


def say(*parts: object) -> None:
    with PRINT_LOCK:
        print(*parts, flush=True)


def to_seconds(value: str | None) -> float | None:
    """ISO-время из статуса задачи → секунды; None остаётся None."""
    try:
        return datetime.fromisoformat(value).timestamp() if value else None
    except ValueError:
        return None


# --- транспорт ----------------------------------------------------------------
def call(method: str, path: str, payload: dict | None = None, timeout: float = 60.0):
    """Возвращает (http_status, elapsed_sec, тело_или_ошибка)."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(BASE_URL + path, data=data, method=method, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, time.perf_counter() - started, body
    except urllib.error.HTTPError as exc:
        return exc.code, time.perf_counter() - started, exc.read()
    except Exception as exc:  # таймаут или обрыв — тоже результат нагрузки
        return 0, time.perf_counter() - started, repr(exc).encode()


def call_json(method: str, path: str, payload: dict | None = None, timeout: float = 60.0):
    status, elapsed, body = call(method, path, payload, timeout)
    try:
        return status, elapsed, json.loads(body)
    except Exception:
        return status, elapsed, None


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def show_latency(title: str, stats: dict) -> None:
    if not stats.get("n"):
        say(f"  {title:<28} нет данных")
        return
    say(
        f"  {title:<28} n={stats['n']:<5} "
        f"mean={stats['mean']*1000:7.1f}мс  p90={stats['p90']*1000:7.1f}мс  "
        f"p99={stats['p99']*1000:7.1f}мс  max={stats['max']*1000:7.1f}мс"
    )


# --- ресурсы ------------------------------------------------------------------
def _run_tool(command: list[str]) -> str:
    """Вывод внешней утилиты или пустая строка, если она недоступна.

    `ps`/`pgrep` могут отсутствовать или быть запрещены политикой песочницы.
    Телеметрия — вспомогательная часть нагрузочного теста: её отказ не должен
    выглядеть как падение теста, поэтому здесь пустой вывод, а не исключение.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout


class Resources:
    """Фоновый сэмплер: RSS/CPU сервера из /api/status плюс RSS его детей через ps."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._warned = False
        self.server_pid = self._find_server()
        self.samples: list[tuple[float, dict]] = []
        self.marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _find_server() -> int | None:
        out = _run_tool(["pgrep", "-f", "uvicorn backend.main"]).split()
        return int(out[0]) if out else None

    def _children(self) -> list[tuple[str, float]]:
        """Дочерние процессы сервера (Whisper-воркер и его помощники) с их RSS."""
        if self.server_pid is None:
            return []
        pids = _run_tool(["pgrep", "-P", str(self.server_pid)]).split()
        found = []
        for pid in pids:
            out = _run_tool(["ps", "-o", "rss=,command=", "-p", pid]).strip()
            if not out:
                continue
            rss, _, command = out.partition(" ")
            found.append((command[:70], float(rss) / 1024))
        return found

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Сэмплер — вспомогательный: недоступный `ps`, запрет на запуск
            # процессов или упавший /api/status не должны убивать поток с
            # трейсбеком посреди прогона. Один раз предупреждаем и продолжаем.
            try:
                _, _, data = call_json("GET", "/api/status", timeout=10)
                children = self._children()
            except Exception as exc:  # noqa: BLE001 — фоновая телеметрия
                if not self._warned:
                    self._warned = True
                    say(f"  (телеметрия недоступна: {type(exc).__name__}: {exc})")
                self._stop.wait(self.interval)
                continue
            if data:
                self.samples.append(
                    (
                        time.monotonic(),
                        {
                            "rss_mb": data.get("rss_mb"),
                            "cpu": data.get("cpu_percent"),
                            "system_mem": data.get("system_mem_percent"),
                            "children_mb": round(sum(rss for _, rss in children), 1),
                            "children_count": len(children),
                            "child_max_mb": round(max((rss for _, rss in children), default=0.0), 1),
                        },
                    )
                )
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def mark(self, label: str) -> None:
        self.marks.append((time.monotonic(), label))

    def report(self) -> None:
        say("\n=== РЕСУРСЫ: пики по фазам ===")
        say(f"  сервер PID {self.server_pid}" if self.server_pid else "  сервер не найден через pgrep")
        bounds = self.marks + [(float("inf"), "конец")]
        for index, (start, label) in enumerate(self.marks):
            end = bounds[index + 1][0]
            window = [sample for moment, sample in self.samples if start <= moment < end]
            if not window:
                say(f"  {label:<26} нет сэмплов (фаза короче интервала опроса)")
                continue
            rss = max(sample["rss_mb"] for sample in window if sample["rss_mb"] is not None)
            cpu = max(sample["cpu"] for sample in window if sample["cpu"] is not None)
            mem = max(sample["system_mem"] for sample in window if sample["system_mem"] is not None)
            children = max(sample["children_mb"] for sample in window)
            children_count = max(sample["children_count"] for sample in window)
            child_max = max(sample["child_max_mb"] for sample in window)
            say(
                f"  {label:<26} RSS сервера ≤ {rss:6.0f} МБ · "
                f"дети ≤ {children:6.0f} МБ в {children_count} проц. "
                f"(один ≤ {child_max:5.0f} МБ) · CPU ≤ {cpu:5.1f}% · "
                f"память системы ≤ {mem:4.1f}%"
            )


# --- постановка задач ---------------------------------------------------------
def start_job(dialogue: str, speakers: dict, qa: str) -> str:
    payload = {
        "dialogue_text": dialogue,
        "speakers": speakers,
        "qa": qa,
        "output_format": "wav",
    }
    status, _, data = call_json("POST", "/api/generate", payload, timeout=30)
    if status != 202 or not data:
        raise RuntimeError(f"не удалось поставить задачу: {status} {data}")
    return data["job_id"]


def job_state(job_id: str) -> dict | None:
    _, _, data = call_json("GET", f"/api/jobs/{job_id}", timeout=30)
    return data


def watch_job(job_id: str, timeout: float = 1800.0, on_sample=None) -> tuple[dict, float, list[tuple[int, float]]]:
    """Ждёт завершения задачи и заодно собирает тайминги по репликам из message.

    Возвращает (статус задачи, время работы, [(номер реплики, секунда от старта)]).
    """
    started = time.monotonic()
    deadline = started + timeout
    replicas: list[tuple[int, float]] = []
    while time.monotonic() < deadline:
        data = job_state(job_id)
        if on_sample is not None:
            on_sample(data)
        if data:
            found = REPLICA_MESSAGE.search(data.get("message") or "")
            if found:
                index = int(found.group(1))
                if not replicas or replicas[-1][0] != index:
                    replicas.append((index, time.monotonic() - started))
        if data and data["status"] in ("done", "error"):
            return data, time.monotonic() - started, replicas
        time.sleep(0.25)
    raise RuntimeError(f"задача {job_id} не завершилась за {timeout:.0f} c")


def show_replicas(timeline: list[tuple[int, float]]) -> None:
    if not timeline:
        return
    previous = 0.0
    parts = []
    for index, moment in timeline:
        parts.append(f"реплика {index}: {moment:5.1f} c (+{moment - previous:4.1f})")
        previous = moment
    say("    " + " · ".join(parts))


# --- фазы ---------------------------------------------------------------------
def phase_light() -> None:
    say("\n=== light. ЛАТЕНТНОСТЬ ЛЁГКИХ РУЧЕК (последовательно, n=20) ===")
    for method, path, payload in LIGHT_ROUTES:
        for _ in range(2):  # прогрев
            call(method, path, payload)
        times, codes = [], set()
        for _ in range(20):
            status, elapsed, _ = call(method, path, payload)
            times.append(elapsed)
            codes.add(status)
        show_latency(f"{method} {path}", percentiles(times))
        if codes != {200}:
            say(f"    коды ответов: {sorted(codes)}")


def phase_load() -> None:
    say("\n=== load. КОНКУРЕНТНОСТЬ: лёгкие ручки ===")
    requests_per_worker = 25
    for workers_count in (1, 5, 20, 50):
        times: list[float] = []
        codes: list[int] = []
        lock = threading.Lock()

        def worker(offset: int) -> None:
            local_times, local_codes = [], []
            for step in range(requests_per_worker):
                method, path, payload = LIGHT_ROUTES[(offset + step) % len(LIGHT_ROUTES)]
                status, elapsed, _ = call(method, path, payload)
                local_times.append(elapsed)
                local_codes.append(status)
            with lock:
                times.extend(local_times)
                codes.extend(local_codes)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers_count)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
        stats = percentiles(times)
        errors = [code for code in codes if code not in (200, 202)]
        say(
            f"  клиентов {workers_count:>3}: всего {len(times):>4} запросов за {wall:5.2f} c → "
            f"{len(times)/wall:6.1f} RPS · p50={stats['p50']*1000:6.1f}мс · "
            f"p99={stats['p99']*1000:6.1f}мс · max={stats['max']*1000:7.1f}мс · ошибок {len(errors)}"
        )


def phase_heavy() -> None:
    say("\n=== heavy. ОТЗЫВЧИВОСТЬ API ВО ВРЕМЯ ТЯЖЁЛОЙ ЗАДАЧИ ===")
    job_id = start_job(DIALOGUE_THREE, {"ИВАН": {"voice_id": F5_VOICE}}, qa="strict")
    say(f"  задача {job_id} (3 реплики, qa=strict), параллельно 10 потоков опроса /api/status")

    times: list[float] = []
    codes: list[int] = []
    stop = threading.Event()
    lock = threading.Lock()

    def poller() -> None:
        local_times, local_codes = [], []
        while not stop.is_set():
            status, elapsed, _ = call("GET", "/api/status", timeout=10)
            local_times.append(elapsed)
            local_codes.append(status)
            time.sleep(0.05)
        with lock:
            times.extend(local_times)
            codes.extend(local_codes)

    threads = [threading.Thread(target=poller, daemon=True) for _ in range(10)]
    for thread in threads:
        thread.start()
    final, wall, timeline = watch_job(job_id)
    stop.set()
    for thread in threads:
        thread.join(timeout=5)

    stats = percentiles(times)
    say(f"  задача заняла {wall:6.1f} c, статус {final['status']} ({final.get('message')})")
    show_replicas(timeline)
    say(f"  за это время /api/status опрошен {len(times)} раз, ошибок {sum(1 for code in codes if code != 200)}")
    show_latency("GET /api/status под нагрузкой", stats)
    if times and stats["p99"] > 1.0:
        say("  ВНИМАНИЕ: p99 выше 1 c — цикл событий подтормаживает во время синтеза")


def phase_queue() -> None:
    say("\n=== queue. ОЧЕРЕДЬ: 6 ОДНОРЕПЛИКОВЫХ ЗАДАЧ РАЗОМ ===")
    speakers = {"ИВАН": {"voice_id": F5_VOICE}}
    jobs = [start_job(DIALOGUE_ONE, speakers, qa="off") for _ in range(6)]
    say(f"  поставлены: {', '.join(jobs)}")

    max_parallel = 0
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        states = [job_state(job_id) for job_id in jobs]
        parallel = sum(1 for state in states if state and state["status"] == "processing")
        max_parallel = max(max_parallel, parallel)
        if all(state and state["status"] in ("done", "error") for state in states):
            break
        time.sleep(1.0)

    states = [job_state(job_id) for job_id in jobs]
    say(f"  одновременно в работе максимум: {max_parallel} (ожидается 1)")

    intervals = []
    for job_id, state in zip(jobs, states):
        if not state:
            continue
        created = to_seconds(state.get("created_at"))
        started = to_seconds(state.get("started_at"))
        finished = to_seconds(state.get("finished_at"))
        wait = started - created if started and created else None
        run = finished - started if finished and started else None
        intervals.append((started, finished))
        say(
            f"  {job_id}: статус={state['status']:<10} "
            f"ожидание={(f'{wait:5.1f} c' if wait is not None else '  —  ')} "
            f"работа={(f'{run:5.1f} c' if run is not None else '  —  ')}"
        )

    known = [pair for pair in intervals if pair[0] and pair[1]]
    overlaps = sum(
        1 for index in range(len(known) - 1) if known[index][1] > known[index + 1][0] + 0.05
    )
    say(f"  пересечений по времени между задачами: {overlaps} (ожидается 0)")


def phase_qa() -> None:
    say("\n=== qa. ЦЕНА ПРОВЕРКИ: off против smart против strict ===")
    speakers = {"ИВАН": {"voice_id": F5_VOICE}}
    for mode in ("off", "smart", "strict"):
        job_id = start_job(DIALOGUE_THREE, speakers, qa=mode)
        final, wall, timeline = watch_job(job_id)
        replicas = final.get("replicas") or []
        marks = [replica.get("qa") for replica in replicas]
        wers = [mark["wer"] for mark in marks if mark]
        attempts = [mark["attempts"] for mark in marks if mark]
        # Сколько кусков реально ушло в Whisper: в Smart это и есть его цена.
        transcribed = sum(1 for mark in marks if mark and mark["wer"] is not None)
        screened = sum(
            1
            for mark in marks
            if mark and mark.get("screening") and mark["screening"]["suspicious"]
        )
        duration = final.get("duration_sec") or 0
        say(
            f"  qa={mode:<6} время {wall:6.1f} c · аудио {duration:5.1f} c "
            f"({wall/duration if duration else 0:4.1f}× реального времени) · "
            f"реплик {len(replicas)} · в Whisper {transcribed}"
            f"{f' (отбор: подозрительных {screened})' if mode == 'smart' else ''} · "
            f"WER {wers} · попытки {attempts}"
        )
        show_replicas(timeline)


def phase_engines() -> None:
    """XTTS: цена холодного старта и смешанный диалог с уже поднятым F5."""
    say("\n=== engines. XTTS ХОЛОДНЫЙ, XTTS ПРОГРЕТЫЙ, СМЕШАННЫЙ ДИАЛОГ ===")
    speakers = {"ИВАН": {"voice_id": F5_VOICE}, "МАРГО": {"voice_id": XTTS_VOICE}}

    for label in ("холодный", "прогретый"):
        job_id = start_job(DIALOGUE_MARGO, {"МАРГО": {"voice_id": XTTS_VOICE}}, qa="off")
        _, _, engines = call_json("GET", "/api/status")
        states = {engine["id"]: engine.get("state") for engine in (engines or {}).get("engines") or []}
        final, wall, timeline = watch_job(job_id)
        say(
            f"  XTTS ({label}): задача {job_id} — {wall:6.1f} c, статус {final['status']}, "
            f"аудио {final.get('duration_sec') or 0:5.1f} c, движки {states}"
        )
        show_replicas(timeline)

    job_id = start_job(DIALOGUE_MIXED, speakers, qa="off")
    final, wall, timeline = watch_job(job_id)
    replicas = final.get("replicas") or []
    say(
        f"  смешанный диалог: задача {job_id} — {wall:6.1f} c, статус {final['status']}, "
        f"аудио {final.get('duration_sec') or 0:5.1f} c, реплик {len(replicas)}"
    )
    show_replicas(timeline)
    for replica in replicas:
        say(
            f"    реплика {replica.get('index')} {replica.get('label')}: "
            f"seed={replica.get('seed')} · qa={replica.get('qa')}"
        )


def phase_overview() -> None:
    status, _, data = call_json("GET", "/api/status")
    if not data:
        say(f"\n=== СОСТОЯНИЕ СЕРВИСА: недоступно (код {status}) ===")
        return
    say(
        f"\n=== СОСТОЯНИЕ СЕРВИСА: устройство={data.get('device')} · "
        f"движки={[(e.get('id'), e.get('state')) for e in data.get('engines') or []]} · "
        f"очередь={data.get('queue_size')} · RSS={data.get('rss_mb')} МБ · "
        f"память системы={data.get('system_mem_percent')}% ==="
    )


def parse_args(argv: list[str]) -> tuple[str, list[str]]:
    base_url = BASE_URL
    rest: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--base-url":
            base_url = argv[index + 1].rstrip("/")
            index += 2
            continue
        rest.append(argv[index])
        index += 1
    unknown = [name for name in rest if name not in PHASES]
    if unknown:
        raise SystemExit(f"неизвестные фазы: {', '.join(unknown)}; доступны: {', '.join(PHASES)}")
    return base_url, rest or list(PHASES)


def main() -> None:
    global BASE_URL
    BASE_URL, phases = parse_args(sys.argv[1:])
    say(f"Нагрузочный тест {BASE_URL}: фазы {', '.join(phases)}")

    resources = Resources()
    resources.start()
    runners = {
        "light": phase_light,
        "load": phase_load,
        "heavy": phase_heavy,
        "queue": phase_queue,
        "qa": phase_qa,
        "engines": phase_engines,
    }
    for name in phases:
        if name not in ("light", "load"):
            resources.mark(name)  # лёгкие фазы короче интервала опроса — сэмплов нет
        runners[name]()
    resources.stop()
    resources.report()
    phase_overview()
    say("\nГотово.")


if __name__ == "__main__":
    main()
