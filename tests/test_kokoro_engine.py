"""Движок Kokoro-ru (ФАЗА 9): выбор встроенного голоса, режимы, контракт звука.

Настоящие веса здесь не поднимаются: пакет `kokoro` подменяется подставным
модулем, а произносительный словарь — подставным `ru_g2p.py` в каталоге модели.
Поэтому проверяется ровно то, что не видно на реальной модели: какой каталог
выбран, какой чекпоинт и голосовой пакет взяты для этого пола, что уходит в
фонемизатор в каждом режиме, когда чекпоинт поднимается лениво и что происходит
при выгрузке.

Реальные веса — отдельный, gated прогон (`TTS_RUN_REAL_KOKORO=1`,
см. docs/testing.md).
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from backend import config, model_manager
from backend.engines.base import (
    ENGINE_KOKORO,
    ENGINE_MODE_DRAFT,
    ENGINE_MODE_EXPERIMENTAL,
    ENGINE_MODE_QUALITY,
    SAMPLE_RATE,
    STATE_FAILED,
)
from backend.engines.kokoro_engine import (
    KokoroEngine,
    _split_phonemes,
    resolve_model_dir,
)
from backend.engines.registry import create_local_engine

REF_TEXT = "Привет, это референс"

# Подставной словарь: тот же интерфейс, что у `ru_g2p.py` репозитория, но без
# RUAccent и espeak. Пишет вызовы в `calls`, чтобы тест видел, какой путь
# фонемизации выбрал режим.
FAKE_PHONEMES = "ab cd ef gh ij kl mn op qr st uv wx"

FAKE_G2P_SOURCE = f'''PHONEMES = "{FAKE_PHONEMES}"


class RuG2P:
    def __init__(self, espeak_data=None, vocab_path=None, omograph_model_size="turbo3.1", reduction=True):
        self.calls = []
        self.reduction = reduction
        self.omograph_model_size = omograph_model_size
        self.phonemes = PHONEMES
        self.oov = ()

    def phonemize(self, text):
        self.calls.append(("phonemize", text, self.reduction))
        return self.phonemes, self.oov

    def phonemize_accented(self, text):
        self.calls.append(("phonemize_accented", text, self.reduction))
        return self.phonemes, self.oov
'''

# Словарь, которому не хватает `misaki`: именно эту зависимость зовёт `ru_g2p.py`
# репозитория, и отсутствие пакета обязано называть установку, а не «No module».
G2P_WITHOUT_MISAKI = "import misaki_not_installed\n"

# Флаги реального прогона (см. `_real_skip_reason` и docs/testing.md).
REAL_FLAG = "TTS_RUN_REAL_KOKORO"


# --- подставные пакеты --------------------------------------------------------
class _Recorder(dict):
    """Журнал подставного пакета `kokoro`: загрузки, устройства и вызовы модели."""

    def __init__(self, **kwargs):
        super().__init__(loads=[], calls=[], devices=[], samples=2400, **kwargs)


def fake_kokoro(recorder: _Recorder) -> types.ModuleType:
    """Модуль `kokoro` вместо настоящего: тот же интерфейс, никаких весов."""

    class _Output:
        def __init__(self, audio) -> None:
            self.audio = audio

    class KModel:
        def __init__(self, repo_id=None, config=None, model=None, disable_complex=False):
            recorder["loads"].append(
                {
                    "repo_id": repo_id,
                    "config": config,
                    "model": model,
                    "disable_complex": disable_complex,
                }
            )

        def eval(self):
            return self

        def to(self, device):
            recorder["devices"].append(device)
            return self

        def __call__(self, phonemes, ref_s, speed=1.0, return_output=True):
            recorder["calls"].append(
                {"phonemes": phonemes, "style": tuple(ref_s.shape), "speed": speed}
            )
            return _Output(torch.zeros(int(recorder["samples"]), dtype=torch.float32))

    module = types.ModuleType("kokoro")
    module.KModel = KModel
    return module


def write_voice_pack(model_dir: Path, voice: str, rows: int = 4) -> Path:
    """Голосовой пакет `voices/<voice>.pt` — стилевые векторы, как в репозитории."""
    path = model_dir / "voices" / f"{voice}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(rows, 256), path)
    return path


def install_model_into(model_dir: Path, g2p_source: str = FAKE_G2P_SOURCE) -> Path:
    """Раскладывает все обязательные файлы модели по указанному каталогу."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "kokoro-config.json").write_text("{}", encoding="utf-8")
    (model_dir / "ru_g2p.py").write_text(g2p_source, encoding="utf-8")
    (model_dir / "espeak-data").mkdir(exist_ok=True)
    (model_dir / "espeak-data" / "ru_dict").write_bytes(b"dict")
    for weights in (config.KOKORO_BASE_WEIGHTS, config.KOKORO_DIMA_WEIGHTS):
        (model_dir / weights).write_bytes(b"weights")
    for voice in (config.KOKORO_VOICE_FEMALE, config.KOKORO_VOICE_MALE, config.KOKORO_VOICE_OTHER):
        write_voice_pack(model_dir, voice)
    return model_dir


def install_model(tmp_path: Path, g2p_source: str = FAKE_G2P_SOURCE) -> Path:
    """Каталог модели со всеми обязательными файлами (см. KOKORO_REQUIRED_FILES)."""
    return install_model_into(tmp_path / "models" / "kokoro_ru", g2p_source)


@pytest.fixture
def kokoro(monkeypatch, tmp_path):
    """Движок на подставных пакете и словаре плюс журнал вызовов модели."""
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro(recorder))

    model_dir = install_model(tmp_path)
    monkeypatch.setattr(config, "KOKORO_DIR", model_dir)
    monkeypatch.setattr(config, "KOKORO_DEVICE", "cpu")

    engine = KokoroEngine(model_dir)
    return engine, recorder, model_dir


def loaded_voice_packs(engine: KokoroEngine) -> set[str]:
    """Какие голосовые пакеты движок реально прочитал (ключ кеша — путь к файлу)."""
    return {Path(key[0]).name for key in engine._voices}


# --- 1. каталог весов ---------------------------------------------------------
def test_resolve_model_dir_finds_nested_repository(tmp_path):
    """Каталог ищется по «якорю»: в HF-снимке файлы лежат на несколько уровней глубже."""
    nested = tmp_path / "hub" / "models--zaakirio--kokoro-ru" / "snapshots" / "abc"
    install_model_into(nested)

    found = resolve_model_dir(
        tmp_path / "hub", required=config.KOKORO_REQUIRED_FILES, anchor="config.json"
    )

    assert found == nested


def test_resolve_model_dir_without_voice_packs_is_none(tmp_path):
    """Без голосовых пакетов каталог моделью не считается: говорить будет нечем."""
    model_dir = install_model(tmp_path)
    for voice in (config.KOKORO_VOICE_FEMALE, config.KOKORO_VOICE_MALE, config.KOKORO_VOICE_OTHER):
        (model_dir / "voices" / f"{voice}.pt").unlink()

    found = resolve_model_dir(
        model_dir, required=config.KOKORO_REQUIRED_FILES, anchor="config.json"
    )
    missing = resolve_model_dir(tmp_path / "нет-такого", required=("config.json",), anchor="config.json")

    assert found is None
    assert missing is None


# --- 2. загрузка --------------------------------------------------------------
def test_load_reports_missing_weights(tmp_path):
    """Пустой каталог — понятная ошибка про веса, а не про отсутствие пакета.

    Порядок важен и проверяется здесь: сначала файлы, потом импорт. Иначе
    пользователь без скачанной модели получал бы «нет модуля kokoro» и чинил бы
    не то.
    """
    empty = tmp_path / "models" / "kokoro_ru"
    empty.mkdir(parents=True)

    engine = KokoroEngine(empty)
    with pytest.raises(RuntimeError) as exc:
        engine.load()

    assert "не найдены" in str(exc.value)
    assert engine.state == STATE_FAILED


def test_load_reports_missing_misaki(tmp_path, monkeypatch):
    """Словарь репозитория без `misaki` — ошибка называет установку зависимостей."""
    model_dir = install_model(tmp_path, g2p_source=G2P_WITHOUT_MISAKI)
    monkeypatch.setattr(config, "KOKORO_DIR", model_dir)

    engine = KokoroEngine(model_dir)
    with pytest.raises(RuntimeError) as exc:
        engine.load()

    assert "misaki" in str(exc.value)
    assert "requirements-kokoro.txt" in str(exc.value)


def test_load_raises_only_the_phonemizer(kokoro):
    """`load()` поднимает словарь, а не чекпоинт: голос ещё не известен.

    Пайплайн прогревает движки заранее, до первой реплики, и пола голоса в этот
    момент у него нет. Поэтому чекпоинт выбирается первым шагом синтеза, а
    `ready` здесь означает «готов работать».
    """
    engine, recorder, _model_dir = kokoro
    engine.load()

    assert engine.is_loaded
    assert engine._g2p is not None
    assert recorder["loads"] == []


def test_load_passes_device_and_complex_flag(kokoro, monkeypatch):
    engine, recorder, model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="male")

    load = recorder["loads"][0]
    assert Path(load["model"]).name == config.KOKORO_DIMA_WEIGHTS
    assert load["repo_id"] == config.KOKORO_REPO_ID
    assert load["config"] == str(model_dir / "config.json")
    # На CPU комплексные операции поддержаны, и обходить их незачем.
    assert load["disable_complex"] is False
    assert recorder["devices"] == ["cpu"]
    assert engine.device == "cpu"


def test_non_cpu_device_disables_complex(kokoro, monkeypatch):
    """Вне CPU декодер считается вещественно: комплексных операций там нет везде."""
    engine, recorder, _model_dir = kokoro
    monkeypatch.setattr(config, "KOKORO_DEVICE", "mps")

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert recorder["loads"][0]["disable_complex"] is True
    assert recorder["devices"] == ["mps"]


def test_unknown_device_falls_back_to_cpu(kokoro, monkeypatch):
    engine, recorder, _model_dir = kokoro
    monkeypatch.setattr(config, "KOKORO_DEVICE", "tpu")

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert recorder["devices"] == ["cpu"]


def test_missing_package_names_the_install_command(kokoro, monkeypatch):
    """Веса на месте, пакета нет — ошибка называет установку, а не «No module»."""
    engine, _recorder, _model_dir = kokoro
    monkeypatch.setitem(sys.modules, "kokoro", None)

    with pytest.raises(RuntimeError) as exc:
        engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert "kokoro" in str(exc.value)
    assert "requirements-kokoro.txt" in str(exc.value)


# --- 3. встроенный голос по полу ----------------------------------------------
def test_female_voice_uses_the_base_checkpoint(kokoro):
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert Path(recorder["loads"][0]["model"]).name == config.KOKORO_BASE_WEIGHTS
    assert loaded_voice_packs(engine) == {f"{config.KOKORO_VOICE_FEMALE}.pt"}


def test_male_voice_uses_the_dima_checkpoint(kokoro):
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="male")

    assert Path(recorder["loads"][0]["model"]).name == config.KOKORO_DIMA_WEIGHTS
    assert loaded_voice_packs(engine) == {f"{config.KOKORO_VOICE_MALE}.pt"}


def test_empty_gender_uses_the_other_voice(kokoro):
    """Пол не назван — берётся голос «другой», а не мужской по умолчанию."""
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0)

    assert Path(recorder["loads"][0]["model"]).name == config.KOKORO_BASE_WEIGHTS
    assert loaded_voice_packs(engine) == {f"{config.KOKORO_VOICE_OTHER}.pt"}


def test_gender_is_read_case_insensitively(kokoro):
    engine, _recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="FEMALE")

    assert loaded_voice_packs(engine) == {f"{config.KOKORO_VOICE_FEMALE}.pt"}


def test_unknown_gender_uses_the_other_voice(kokoro):
    engine, _recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="робот")

    assert loaded_voice_packs(engine) == {f"{config.KOKORO_VOICE_OTHER}.pt"}


def test_voice_names_lists_all_builtin_voices(kokoro):
    engine, _recorder, _model_dir = kokoro

    assert set(engine.voice_names) == {
        config.KOKORO_VOICE_FEMALE,
        config.KOKORO_VOICE_MALE,
        config.KOKORO_VOICE_OTHER,
    }


# --- 4. режимы и фонемизация --------------------------------------------------
def test_quality_mode_phonemizes_with_reduction(kokoro):
    """Рабочий путь: словарь сам расставляет ударения и редукцию гласных."""
    engine, _recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert engine._g2p.calls == [("phonemize", "Привет", True)]
    # Режим не задан явно — работает объявленный в паспорте дефолт.
    assert engine.resolve_mode({}) == ENGINE_MODE_QUALITY


def test_draft_mode_skips_ruaccent(kokoro):
    """Черновик идёт мимо RUAccent: ударения расставляет espeak, и их больше."""
    engine, _recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female", mode=ENGINE_MODE_DRAFT)

    assert engine._g2p.calls == [("phonemize_accented", "Привет", True)]


def test_experimental_mode_disables_reduction(kokoro):
    """Эксперимент — разметка первой версии файнтюна: редукция гласных выключена."""
    engine, _recorder, _model_dir = kokoro

    engine.synthesize(
        "Привет", "", REF_TEXT, 1.0, gender="female", mode=ENGINE_MODE_EXPERIMENTAL
    )

    assert engine._g2p.calls == [("phonemize", "Привет", False)]


def test_oov_symbols_are_reported(kokoro, caplog):
    """Символ вне словаря модели — предупреждение: слово прозвучит неверно молча."""
    engine, _recorder, _model_dir = kokoro
    engine.load()
    engine._g2p.oov = ("ü",)

    with caplog.at_level("WARNING"):
        engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert "вне словаря" in caplog.text


def test_empty_phonemes_are_an_error(kokoro):
    """Ни одной фонемы — отказ с текстом, а не пустой звук в файле."""
    engine, recorder, _model_dir = kokoro
    engine.load()
    engine._g2p.phonemes = "   "

    with pytest.raises(RuntimeError) as exc:
        engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert "ни одной фонемы" in str(exc.value)
    assert recorder["calls"] == []


def test_long_phonemes_are_split_in_parts(kokoro, monkeypatch):
    """Реплика длиннее контекста `KModel` — несколько проходов, а не падение.

    У модели жёсткий потолок (512 позиций), и строка длиннее падает
    `AssertionError` внутри неё, а не ошибкой синтеза.
    """
    engine, recorder, _model_dir = kokoro
    monkeypatch.setattr(config, "KOKORO_MAX_PHONEMES", 15)

    engine.synthesize("Длинная реплика", "", REF_TEXT, 1.0, gender="female")

    pieces = [call["phonemes"] for call in recorder["calls"]]
    assert len(pieces) == 3
    assert all(len(piece) <= 15 for piece in pieces)
    # Разрез идёт по границам слов, поэтому склейка даёт исходную строку.
    assert " ".join(pieces) == FAKE_PHONEMES


def test_short_phonemes_go_in_one_pass(kokoro):
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert len(recorder["calls"]) == 1


# --- 5. режим резания фонемной строки (единица) -------------------------------
def test_split_phonemes_keeps_word_boundaries():
    assert _split_phonemes("ab cd ef", 100) == ["ab cd ef"]
    assert _split_phonemes("ab cd ef gh", 8) == ["ab cd ef", "gh"]


def test_split_phonemes_cuts_a_word_longer_than_the_limit():
    """Слово длиннее потолка режется по символам: артефакт лучше отказа."""
    assert _split_phonemes("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_split_phonemes_of_empty_string_is_empty():
    assert _split_phonemes("", 100) == []
    assert _split_phonemes("   ", 100) == []


# --- 6. контракт звука --------------------------------------------------------
def test_synthesize_matches_contract(kokoro):
    engine, _recorder, _model_dir = kokoro

    waveform, sample_rate = engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    # Kokoro отдаёт звук без частоты: 24 кГц заданы архитектурой модели.
    assert sample_rate == SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size > 0


def test_speed_goes_to_the_model_as_is(kokoro):
    """Темп считает сама модель: она делит предсказанные длительности на `speed`.

    Поэтому time-stretch готового звука здесь не нужен — в отличие от Qwen3-TTS,
    где скорость приходит ручкой пайплайна уже после инференса.
    """
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 0.75, gender="female")

    assert recorder["calls"][0]["speed"] == pytest.approx(0.75)


def test_styles_are_two_dimensional_for_the_model(kokoro):
    """Пакет публикуется и как `[510, 256]`: модель ждёт двумерный тензор."""
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="female")

    assert recorder["calls"][0]["style"] == (1, 256)


def test_reference_is_ignored(kokoro, tmp_path):
    """Референс и его текст не читаются: движок говорит встроенным голосом.

    Пути заведомо несуществующие — если бы адаптер обратился к ним, тест упал бы
    ошибкой файла, а не сравнением звука.
    """
    engine, recorder, _model_dir = kokoro

    first, _ = engine.synthesize("Привет", str(tmp_path / "нет.wav"), "", 1.0, gender="female")
    second, _ = engine.synthesize("Привет", "другой-путь", "другой текст", 1.0, gender="female")

    assert len(recorder["calls"]) == 2
    assert np.array_equal(first, second)


# --- 7. память и выгрузка -----------------------------------------------------
def test_one_checkpoint_per_female_voice_and_a_second_for_male(kokoro):
    """Голоса женского чекпоинта делят модель, мужской поднимается отдельно."""
    engine, recorder, _model_dir = kokoro

    engine.synthesize("Раз", "", REF_TEXT, 1.0, gender="female")
    engine.synthesize("Два", "", REF_TEXT, 1.0, gender="other")
    engine.synthesize("Три", "", REF_TEXT, 1.0, gender="male")
    # Возврат к женскому голосу второй раз веса не поднимает.
    engine.synthesize("Четыре", "", REF_TEXT, 1.0, gender="female")

    assert [Path(load["model"]).name for load in recorder["loads"]] == [
        config.KOKORO_BASE_WEIGHTS,
        config.KOKORO_DIMA_WEIGHTS,
    ]


def test_voice_pack_cache_is_bounded(kokoro, monkeypatch):
    """Кеш голосовых пакетов ограничен: иначе он рос бы вместе с числом голосов."""
    engine, _recorder, _model_dir = kokoro
    monkeypatch.setattr(config, "KOKORO_VOICE_CACHE_SIZE", 1)

    engine.synthesize("Раз", "", REF_TEXT, 1.0, gender="female")
    engine.synthesize("Два", "", REF_TEXT, 1.0, gender="male")
    engine.synthesize("Три", "", REF_TEXT, 1.0, gender="female")

    assert len(engine._voices) == 1


def test_unload_releases_phonemizer_models_and_packs(kokoro):
    engine, recorder, _model_dir = kokoro
    engine.synthesize("Привет", "", REF_TEXT, 1.0, gender="male")
    assert engine.is_loaded

    engine.unload()

    assert not engine.is_loaded
    assert engine.device is None
    # Словарь, веса и голосовые пакеты — всё, чем занята память движка.
    assert engine._g2p is None
    assert engine._models == {}
    assert engine._voices == {}

    # Выгруженный движок поднимается заново, а не отказывает.
    engine.synthesize("Снова", "", REF_TEXT, 1.0, gender="male")
    assert len(recorder["loads"]) == 2


# --- 8. реестр ----------------------------------------------------------------
def test_registry_builds_kokoro_engine(kokoro):
    engine = create_local_engine("kokoro-ru")
    assert isinstance(engine, KokoroEngine)
    assert engine.id == ENGINE_KOKORO


# --- 9. настоящие веса: отдельный, gated прогон -------------------------------
# Всё выше проверено на подставных пакете и словаре. Здесь — единственная
# проверка, которую подстановка дать не может: что настоящие `kokoro` и `ru_g2p`
# действительно принимают те аргументы, которые им передаёт адаптер, и что
# чекпоинт поднимается на выбранном устройстве. Прогон выключен по умолчанию: он
# требует скачанных весов и минут на первый запуск espeak-ng.
def _real_skip_reason() -> str:
    """Почему реальный прогон сейчас невозможен; пустая строка — возможен.

    Причина формулируется словами, а не через `skipif` с булевым условием: без
    неё непонятно, чего не хватает — флага, весов или чего-то ещё.
    """
    if os.environ.get(REAL_FLAG, "").strip().lower() not in ("1", "true", "yes"):
        return f"реальный прогон не запрошен: {REAL_FLAG}=1"
    if not model_manager.engine_available(ENGINE_KOKORO):
        return "веса Kokoro-ru не установлены (см. requirements-kokoro.txt и docs/install.md)"
    return ""


def test_real_model_synthesizes_a_chunk():
    """Настоящая модель: загрузка, встроенный голос и контракт звука.

    Проверяется только контракт (частота, тип, непустой звук), а не качество
    произношения: его оценивают слухом, и автотест, «проверяющий» его числом,
    создавал бы ложную уверенность. Референс не передаётся намеренно — движок
    пресетный, и пустые `ref_audio_path`/`ref_text` здесь и есть проверка того,
    что путь клонирования не задействован.
    """
    reason = _real_skip_reason()
    if reason:
        pytest.skip(reason)

    engine = KokoroEngine()
    try:
        engine.load()
        assert engine.is_loaded, engine.last_error
        waveform, sample_rate = engine.synthesize(
            "Проверка связи: один, два, три.", "", "", 1.0, gender="other"
        )
    finally:
        engine.unload()

    assert sample_rate == SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    # «Слово», а не щелчок: слишком короткий выход означает, что фонемизация
    # развалилась и модель не начала говорить.
    assert waveform.size > SAMPLE_RATE * 0.2
    assert float(np.max(np.abs(waveform))) > 0.0
