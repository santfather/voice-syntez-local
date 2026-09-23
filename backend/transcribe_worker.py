"""Распознавание речи (Whisper). Запускается отдельным процессом:

    python -m backend.transcribe_worker <путь к аудио> [--full]

Без `--full` запись режется так же, как референс для модели (первые ~12 секунд и
обрезка тишины) — это расшифровка для F5-TTS. С `--full` распознаётся файл
целиком: так проверяется готовый результат синтеза, который длиннее референса.

В stdout печатается одна JSON-строка с результатом, весь служебный вывод
(в том числе болтливые `print` из f5_tts) уходит в stderr.

Отдельный процесс нужен потому, что Whisper large-v3-turbo в fp32 занимает
больше памяти, чем весь остальной бэкенд вместе с F5-TTS. В общем процессе это
упиралось бы в лимит watchdog'а (resource_guard.MAX_RSS_MB), а падение
распознавания уносило бы с собой модель синтеза.

Режим `--words` (`python -m backend.transcribe_worker <путь> --words`) отдаёт
таймстемпы слов. Он нужен границам коротких реплик: контекстный синтез склеивает
несколько фраз в один прогон, и разрезать результат «примерно по времени» нельзя —
только по реальным границам слов (short_utterances §11, §31). Таймстемпы берутся у
того же Whisper, поэтому новая зависимость не появляется.

**Язык задаётся явно** (`ASR_LANGUAGE`, по умолчанию `ru`, переменная
`TTS_ASR_LANGUAGE`). Whisper — многоязычная модель и на коротких репликах
определяет язык по 0.3–0.5 с звука: «Почему?» и «Стой!» распознавались как
`thank you`, «Ты опоздал.» — как `tiap`. Проверка качества считала это провалом
синтеза и отправляла реплику на повтор, хотя модель произносила русский текст
правильно (UPDATE 2 §13: это ложный дефект измерения, а не обрыв). Приложение
русскоязычное целиком (prompt, корпус, интерфейс), поэтому автоопределение здесь
не даёт ничего, кроме ошибок на коротких кусках.
"""

import json
import os
import sys

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
# Язык распознавания. Читается из окружения в момент запуска воркера (см. докстринг):
# автоопределение на коротких репликах ошибается и превращает верный синтез в «провал QA».
ASR_LANGUAGE = os.environ.get("TTS_ASR_LANGUAGE", "ru").strip()
FULL_FLAG = "--full"
WORDS_FLAG = "--words"


def _log(*args) -> None:
    """Служебный вывод — только в stderr, чтобы не ломать JSON в stdout."""
    print(*args, file=sys.stderr, flush=True)


def asr_generate_kwargs(language: str | None = None) -> dict:
    """Аргументы генерации Whisper: задача и **явный** язык.

    Выделено отдельной функцией, чтобы это можно было проверить тестом без
    загрузки модели: именно отсутствие языка давало ложные провалы проверки на
    коротких репликах (см. докстринг модуля).
    """
    chosen = ASR_LANGUAGE if language is None else str(language).strip()
    if not chosen:
        return {"task": "transcribe"}
    return {"task": "transcribe", "language": chosen}


def main() -> int:
    """Точка входа воркера распознавания: печатает JSON-результат в stdout.

    Режет запись ровно так же, как её режет синтез (12 секунд плюс обрезка
    тишины), если не запрошена расшифровка файла целиком: расшифровка обязана
    описывать тот же фрагмент, который уйдёт в модель.
    """
    words = WORDS_FLAG in sys.argv
    full = words or FULL_FLAG in sys.argv
    paths = [arg for arg in sys.argv[1:] if arg not in (FULL_FLAG, WORDS_FLAG)]
    if len(paths) != 1:
        _log(f"usage: python -m backend.transcribe_worker <audio path> [{FULL_FLAG}|{WORDS_FLAG}]")
        return 2

    # Порядок импортов здесь важен: config выставляет лимиты потоков в переменных
    # окружения, и сделать это нужно до первого импорта torch и transformers.
    from . import config  # noqa: I001

    import torch
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    from pydub import AudioSegment
    from transformers import pipeline

    audio_path = paths[0]
    if full:
        effective_path = audio_path
    else:
        # Режем запись ровно так же, как это сделает синтез (12 секунд + обрезка
        # тишины): расшифровка обязана описывать тот же фрагмент, который уйдёт в
        # модель. Непустой ref_text здесь — заглушка: он нужен лишь для того, чтобы
        # библиотека не поднимала внутри свой ASR-пайплайн (мы распознаём сами).
        effective_path, _ = preprocess_ref_audio_text(audio_path, "placeholder", show_info=_log)

    device = config.pick_device()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_ID,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device=device,
    )
    result = pipe(
        effective_path,
        chunk_length_s=30,
        batch_size=8,
        # Язык задаётся явно (см. докстринг): без него короткие реплики
        # распознаются как английский, и проверка качества валит верный синтез.
        generate_kwargs=asr_generate_kwargs(),
        return_timestamps="word" if words else False,
    )
    ref_text = str(result.get("text") or "").strip()

    payload: dict = {
        "ref_text": ref_text,
        "effective_sec": round(AudioSegment.from_file(effective_path).duration_seconds, 2),
        "full_sec": round(AudioSegment.from_file(audio_path).duration_seconds, 2),
    }
    if words:
        # Таймстемп может быть `None` (слово без границы) или `(start, None)`:
        # такие слова пропускаются, а не выдумываются — граница обязана быть
        # настоящей, иначе обрезка вернётся к «примерно по времени».
        payload["words"] = [
            {"word": str(chunk.get("text") or "").strip(), "start": start, "end": end}
            for chunk in (result.get("chunks") or [])
            for start, end in [chunk.get("timestamp") or (None, None)]
            if start is not None and end is not None
        ]

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
