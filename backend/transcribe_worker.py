"""Распознавание речи в референс-аудио (Whisper). Запускается отдельным процессом:

    python -m backend.transcribe_worker <путь к аудио>

В stdout печатается одна JSON-строка с результатом, весь служебный вывод
(в том числе болтливые `print` из f5_tts) уходит в stderr.

Отдельный процесс нужен потому, что Whisper large-v3-turbo в fp32 занимает
больше памяти, чем весь остальной бэкенд вместе с F5-TTS. В общем процессе это
упиралось бы в лимит watchdog'а (resource_guard.MAX_RSS_MB), а падение
распознавания уносило бы с собой модель синтеза.
"""

import json
import sys

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"


def _log(*args) -> None:
    """Служебный вывод — только в stderr, чтобы не ломать JSON в stdout."""
    print(*args, file=sys.stderr, flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        _log("usage: python -m backend.transcribe_worker <audio path>")
        return 2

    # Порядок импортов здесь важен: config выставляет лимиты потоков в переменных
    # окружения, и сделать это нужно до первого импорта torch и transformers.
    from . import config  # noqa: I001

    import torch
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    from pydub import AudioSegment
    from transformers import pipeline

    audio_path = sys.argv[1]

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
    ref_text = pipe(
        effective_path,
        chunk_length_s=30,
        batch_size=8,
        generate_kwargs={"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()

    print(
        json.dumps(
            {
                "ref_text": ref_text,
                "effective_sec": round(
                    AudioSegment.from_file(effective_path).duration_seconds, 2
                ),
                "full_sec": round(AudioSegment.from_file(audio_path).duration_seconds, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
