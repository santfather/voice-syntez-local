"""Хранилище голосов: восстановление `voices.json` по каталогу `voices/` (F-F2).

Файл `voices.json` — единственное место, откуда известны имена, пол, движок и
расшифровки голосов, но сами записи лежат в `voices/`. Если файл пропал (удалён
руками, не пережил сбой), список голосов не должен выглядеть пустым: по именам
файлов библиотека собирается обратно.

Тесты работают на tmp_path: настоящий `voices/` проекта не читается и не пишется.
"""

from __future__ import annotations

from backend import config
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
