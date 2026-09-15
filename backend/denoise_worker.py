"""Очистка референса от шума и реверба (DeepFilterNet). Запускается отдельным процессом:

    python -m backend.denoise_worker <входное аудио> <выходной wav>

В stdout печатается одна JSON-строка с длительностями, служебный вывод DFN идёт в stderr.

Отдельный процесс — по той же причине, что и у `transcribe_worker`: модель нужна на один
прогон при сохранении голоса, а её память в основном процессе осталась бы висеть до
перезапуска (watchdog считает RSS всего бэкенда). Процесс умирает — память возвращается.

Реверб и шум модель клонирует как часть тембра: в синтезе это звучит как каша и после
синтеза уже не лечится, поэтому чистим именно референс.
"""

import json
import sys
from pathlib import Path


def _log(*args) -> None:
    """Служебный вывод — только в stderr, чтобы не ломать JSON в stdout."""
    print(*args, file=sys.stderr, flush=True)


def _shim_torchaudio_backend() -> None:
    """Возвращает `torchaudio.backend.common.AudioMetaData`, удалённый в torchaudio 2.9+.

    DeepFilterNet 0.5.6 (последний релиз — август 2023, проект не поддерживается) импортирует
    этот класс на уровне модуля. Нужен он только для аннотаций его собственных
    `load_audio`/`save_audio` — их мы не вызываем, читаем и пишем файлы сами. Держать из-за
    этого torchaudio <= 2.6 нельзя: он связан с torch и f5-tts, а весь набор версий
    проекта проверен на torchaudio 2.11.
    """
    import torchaudio  # noqa: F401  (проверка ниже опирается на наличие атрибута)

    if hasattr(torchaudio, "backend"):
        return
    import types

    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D101 — заглушка вместо удалённого класса
        sample_rate: int
        num_frames: int
        num_channels: int

    common.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
    backend.common = common  # type: ignore[attr-defined]
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend  # type: ignore[attr-defined]


def main() -> int:
    if len(sys.argv) != 3:
        _log("usage: python -m backend.denoise_worker <input audio> <output wav>")
        return 2

    # Порядок импортов важен: config выставляет лимиты потоков в переменных окружения,
    # и сделать это нужно до первого импорта torch.
    from . import config  # noqa: I001

    import numpy as np
    import soundfile as sf
    import torch

    _shim_torchaudio_backend()
    from df.enhance import enhance, init_df

    from .audio_analysis import load_mono

    source_path, target_path = Path(sys.argv[1]), Path(sys.argv[2])
    model, df_state, name = init_df()
    _log(f"DeepFilterNet: модель {name}, частота {df_state.sr()} Гц")

    # DFN работает на своей частоте (48 кГц): запись ресемплится сюда, а результат
    # пишется как есть — лишний обратный ресемпл только испортил бы сигнал.
    audio, _ = load_mono(source_path, target_sr=df_state.sr())
    enhanced = enhance(model, df_state, torch.from_numpy(audio).unsqueeze(0))
    cleaned = enhanced.squeeze(0).numpy().astype(np.float32)

    # Запас перед оцифровкой в int16: перегрузка на выходе денойзера обернулась бы
    # треском, а не клиппингом.
    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak > 0.999:
        cleaned *= 0.999 / peak

    sf.write(target_path, cleaned, df_state.sr(), subtype="PCM_16")
    print(
        json.dumps(
            {
                "source_sec": round(audio.size / df_state.sr(), 2),
                "target_sec": round(cleaned.size / df_state.sr(), 2),
                "sample_rate": df_state.sr(),
                "peak": round(float(np.max(np.abs(cleaned))) if cleaned.size else 0.0, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
