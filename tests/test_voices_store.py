"""Хранилище голосов: восстановление `voices.json` по каталогу `voices/` (F-F2).

Файл `voices.json` — единственное место, откуда известны имена, пол, движок и
расшифровки голосов, но сами записи лежат в `voices/`. Если файл пропал (удалён
руками, не пережил сбой), список голосов не должен выглядеть пустым: по именам
файлов библиотека собирается обратно.

Тесты работают на tmp_path: настоящий `voices/` проекта не читается и не пишется.
"""

from __future__ import annotations

import json

import pytest

from backend import config
from backend.engines.base import ENGINE_KOKORO
from backend.voices_store import VoicesStore

# id в именах файлов — `uuid.uuid4().hex[:12]`, то есть ровно 12 шестнадцатеричных
# знаков. Значения теста любые, но этой форме обязаны соответствовать.
VOICE_ID = "0123456789ab"
PROFILE_ID = "cdef01234567"


def test_voices_json_is_recovered_from_files(workspace):
    """Пропавший `voices.json` пересобирается: голос и его профиль возвращаются."""
    voices_dir = config.VOICES_DIR
    (voices_dir / f"{VOICE_ID}.wav").write_bytes(b"RIFF")
    (voices_dir / f"{VOICE_ID}-{PROFILE_ID}.mp3").write_bytes(b"ID3")
    # Чужое имя в каталоге восстановлению не мешает и голосом не становится.
    (voices_dir / "запись-пользователя.wav").write_bytes(b"RIFF")

    store = VoicesStore()
    assert store.list() == []  # файла ещё нет — библиотека пуста

    assert store.recover_from_files() == 1
    voices = store.list()
    assert [voice.id for voice in voices] == [VOICE_ID]
    assert voices[0].audio_file == f"{VOICE_ID}.wav"
    assert [profile.id for profile in voices[0].profiles] == [PROFILE_ID]
    # Файл профиля — не основной референс: нейтральным он не притворяется.
    assert voices[0].neutral_profile().audio_file == f"{VOICE_ID}.wav"
    assert config.VOICES_JSON.exists()

    # Идемпотентно и безобидно: существующий файл восстановление не трогает.
    assert store.recover_from_files() == 0


def test_recovery_reads_voices_from_files_only(workspace):
    """Голос с одними профилями восстанавливается и без основного референса."""
    (config.VOICES_DIR / f"{VOICE_ID}-{PROFILE_ID}.wav").write_bytes(b"RIFF")

    store = VoicesStore()
    assert store.recover_from_files() == 1

    voice = store.get(VOICE_ID)
    assert voice is not None
    assert voice.audio_file == ""
    assert [profile.id for profile in voice.profiles] == [PROFILE_ID]


def test_recovery_is_silent_when_there_is_nothing_to_restore(workspace):
    """Пустой каталог — не повод писать пустой `voices.json`."""
    store = VoicesStore()
    assert store.recover_from_files() == 0
    assert not config.VOICES_JSON.exists()


def test_write_is_atomic_and_survives_interruption(workspace, monkeypatch):
    """Обрыв записи не оставляет битый `voices.json` и не теряет заведённые голоса.

    Прямая `write_text` усекает файл ещё до начала записи: падение в этот момент
    дало бы обрезанный JSON, `_read` вернул бы пустой список, и следующая запись
    затёрла бы библиотеку насовсем. Поэтому запись идёт во временный файл рядом, а
    подмена — через `os.replace`: обрыв до неё текущее содержимое не трогает.
    """
    store = VoicesStore()
    store.create(
        name="Света", gender="female", ref_text="", audio_filename="",
        audio_bytes=b"", engine=ENGINE_KOKORO,
    )
    before = config.VOICES_JSON.read_text(encoding="utf-8")
    assert [voice.name for voice in store.list()] == ["Света"]

    def broken_replace(*args, **kwargs):
        raise OSError("диск переполнен")

    monkeypatch.setattr("backend.voices_store.os.replace", broken_replace)
    with pytest.raises(OSError):
        store.create(
            name="Дима", gender="male", ref_text="", audio_filename="",
            audio_bytes=b"", engine=ENGINE_KOKORO,
        )

    # Прежний файл цел: обрыва посреди записи не случилось.
    assert config.VOICES_JSON.read_text(encoding="utf-8") == before
    assert [voice.name for voice in store.list()] == ["Света"]


def test_audio_file_outside_voices_dir_is_ignored(workspace):
    """`audio_file` вида `../…` не превращается в путь к чужому файлу.

    `voices.json` правят руками и переносят между машинами. Значение, выводящее
    за `voices/`, читается как «референса нет», поэтому удаление голоса не трогает
    файл за пределами каталога.
    """
    victim = config.VOICES_DIR.parent / "victim.wav"
    victim.write_bytes(b"RIFF")
    outside_profile = "outside-profile"
    config.VOICES_JSON.write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "id": VOICE_ID,
                        "name": "Чужой",
                        "gender": "female",
                        "ref_text": "",
                        "audio_file": "../victim.wav",
                        "profiles": [
                            {
                                "id": outside_profile,
                                "audio_file": "../victim.wav",
                                "emotion": "neutral",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = VoicesStore()
    voice = store.get(VOICE_ID)
    assert voice is not None
    assert voice.audio_file == ""
    assert voice.profiles == []

    assert store.delete(VOICE_ID) is True
    assert victim.read_bytes() == b"RIFF"
