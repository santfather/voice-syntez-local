"""Фразы для записи голоса в `frontend/voice-record.js`: состав, «ё», mapping.

JS-раннера в проекте нет, поэтому тест читает исходники как текст — тем же
приёмом, что `test_launcher.py` читает `VOICE_SYNTEZ.command`. Разбор — регулярные
выражения плюс крошечный сканер строк и комментариев: он переживает
переформатирование массива, экранированные кавычки и комментарии между элементами,
но падает с понятной ошибкой, если структура не нашлась.

Файлов три, потому что код разнесён по смыслу, а не по удобству теста:
список фраз и мастер записи живут в модуле записи (`voice-record.js`), тестовая
фраза с плотной «ё» и её кнопка-пример — в модуле панели preview
(`text-preview.js`), а обработчики событий — в точке входа (`app.js`).

Проверяется не «текст совпадает с документом побайтово», а смысл: первые пять фраз
не тронуты, новых ровно шесть и в порядке документа, в новых есть «ё» и новые
интонационные регистры, каждая фраза короткая, у каждой есть записываемый профиль
(`RECORD_PHRASE_PROFILES`, §16), а тестовая фраза с плотной «ё» остаётся тестовыми
данными и не попадает ни в список записи, ни в UI записи.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NoReturn

from backend import emotions, qa_screening
from backend.text_normalization import normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_JS = PROJECT_ROOT / "frontend" / "app.js"
VOICE_RECORD_JS = PROJECT_ROOT / "frontend" / "voice-record.js"
TEXT_PREVIEW_JS = PROJECT_ROOT / "frontend" / "text-preview.js"
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"

# Первые пять фраз — защита от случайной правки работающего списка: они должны
# совпадать буквально (и подпись, и текст), а не «примерно по смыслу».
ORIGINAL_PHRASES: tuple[tuple[str, str], ...] = (
    (
        "Вопрос и утверждение",
        (
            "Добрый вечер. Мы договаривались встретиться у метро, но я вас так и не увидел. "
            "Вы точно получили моё сообщение?"
        ),
    ),
    (
        "Восклицания, много шипящих",
        (
            "Осторожно, здесь очень скользко! Я чуть не упал, когда выбегал из подъезда. "
            "И это уже третий раз за неделю."
        ),
    ),
    (
        "Спокойная просьба (короче)",
        "Дайте пройти, пожалуйста. Я вас совсем не знаю, и мне нечего вам сказать.",
    ),
    (
        "Только вопросы",
        (
            "Ты уверен, что мы правильно свернули? Кажется, этот поворот был раньше. "
            "Может, спросим дорогу у кого-нибудь?"
        ),
    ),
    (
        "Сложные сочетания согласных",
        (
            "Съешь ещё этих мягких французских булочек, а потом расскажешь, как прошла "
            "твоя поездка в Ярославль."
        ),
    ),
)

# Новые шесть — в порядке документа. Подписи сравниваются буквально, тексты — по
# ключевым словам «ё» (ниже), чтобы тест не дублировал документ целиком.
NEW_PHRASES: tuple[tuple[str, str], ...] = (
    (
        "Радость, восторг",
        (
            "Ты представляешь, у нас всё получилось! Ребёнок сам застегнул все пуговицы, "
            "а потом ещё и спел мне целую песню про самолёт."
        ),
    ),
    (
        "Огорчение, сочувствие",
        (
            "Мне так жаль, что всё так обернулось. Она ждала этот день, а теперь идёт "
            "домой одна и не знает, что делать дальше."
        ),
    ),
    (
        "Лёгкая ирония",
        (
            "Ну конечно, именно сегодня лифт снова сломан. Как будто он специально ждёт, "
            "пока актёр из соседней квартиры опять устроит репетицию."
        ),
    ),
    (
        "Строгий тон, короткий приказ",
        (
            "Немедленно отдайте мне ключи от квартиры. Я всё сказал предельно ясно, "
            "и вы прекрасно понимаете, о чём идёт речь."
        ),
    ),
    (
        "Перечисление, ровный ритм",
        (
            "На столе лежали ключи, кошелёк, блокнот, ручка и ещё какие-то бумаги, "
            "которые я так и не успел разобрать."
        ),
    ),
    (
        "Быстрая, взволнованная речь",
        (
            "Скорее, мы опаздываем! Автобус уже подъезжает, а нам ещё нужно забрать вещи, "
            "запереть дверь и найти, куда делись ключи."
        ),
    ),
)

# Интонационные регистры, которых не было в исходных пяти (по подписям).
NEW_REGISTER_MARKERS: tuple[tuple[str, str], ...] = (
    ("радость", "радость"),
    ("огорчение/сочувствие", "огорчение"),
    ("ирония", "ирония"),
)

# Ориентир длины: F5-TTS берёт из референса первые ~12 с, XTTS — до 30 с, поэтому
# фразы намеренно короткие. Считаем не знаки, а **длительность** — тем же грубым
# оценщиком, которым пользуется Smart QA (`qa_screening._expected_sec`): он
# учитывает длину слов и пунктуацию, а не только их количество. Знаки как критерий
# обманывали бы: 132 знака длинными словами и 132 знака короткими — разное время
# начитки.
REFERENCE_BUDGET_SEC = 12.0


def _reading_seconds(text: str) -> float:
    """Оценка длительности начитки в секундах (оценщик проекта, не свой)."""
    return float(qa_screening._expected_sec(text))

# Слова тестовой фразы, где «ё» гарантирована словарём нормализации. В исходнике
# они записаны через «ё», а в проверке ниже — ещё и через «е».
YO_STRESS_WORDS: tuple[str, ...] = (
    "Ёж",
    "ёлка",
    "идёт",
    "актёр",
    "жёлтом",
    "несёт",
    "мёд",
    "лёд",
    "поёт",
    "мёрзнет",
)

# JS-строковый литерал: одинарные или двойные кавычки с учётом экранирования.
_JS_STRING = r"""(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")"""

_ENTRY = re.compile(
    r"\{\s*label\s*:\s*(?P<label>" + _JS_STRING + r")\s*,\s*"
    r"text\s*:\s*(?P<text>" + _JS_STRING + r")\s*,?\s*\}",
    re.DOTALL,
)
_ARRAY_DECL = re.compile(r"\bconst\s+RECORD_PHRASES\s*=\s*\[")
_YO_DECL = re.compile(
    r"\bconst\s+YO_STRESS_TEST_PHRASE\s*=\s*(?P<value>" + _JS_STRING + r")\s*;"
)

_OPEN_TO_CLOSE = {"[": "]", "{": "}", "(": ")"}
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "/": "/",
    "`": "`",
    "0": "\0",
}


def _fail(message: str, path: Path = VOICE_RECORD_JS) -> NoReturn:
    """Ошибка с именем файла, в котором разбор не нашёл ожидаемую структуру."""
    raise ValueError(f"{path.relative_to(PROJECT_ROOT)}: {message}")


def _strip_comments(source: str) -> str:
    """Убирает `//` и `/* */` комментарии, не трогая строковые литералы.

    Комментарии между элементами массива — обычное дело, а `//` внутри строки
    (URL, например) комментарием не является. Поэтому идём посимвольно и знаем,
    находимся ли внутри строки.
    """
    out: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < len(source):
                out.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"`":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and source[index + 1 : index + 2] == "/":
            while index < len(source) and source[index] != "\n":
                index += 1
            continue
        if char == "/" and source[index + 1 : index + 2] == "*":
            index += 2
            while index + 1 < len(source) and source[index : index + 2] != "*/":
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _matching_bracket(source: str, start: int) -> int:
    """Индекс закрывающей скобки для `source[start]`, с учётом строк."""
    stack: list[str] = []
    quote: str | None = None
    index = start
    while index < len(source):
        char = source[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char in _OPEN_TO_CLOSE:
            stack.append(char)
            index += 1
            continue
        if char in ")]}":
            if not stack or _OPEN_TO_CLOSE[stack[-1]] != char:
                _fail(f"несбалансированная скобка {char!r} в позиции {index}")
            stack.pop()
            if not stack:
                return index
            index += 1
            continue
        index += 1
    _fail("не найдена закрывающая скобка объявления")
    raise AssertionError("недостижимо")  # pragma: no cover - _fail всегда бросает


def _decode_js_string(literal: str) -> str:
    """Снимает кавычки и раскрывает escape-последовательности JS-строки."""
    body = literal[1:-1]
    out: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        nxt = body[index + 1 : index + 2]
        if nxt == "u" and len(body) >= index + 6:
            out.append(chr(int(body[index + 2 : index + 6], 16)))
            index += 6
            continue
        if nxt == "x" and len(body) >= index + 4:
            out.append(chr(int(body[index + 2 : index + 4], 16)))
            index += 4
            continue
        out.append(_ESCAPES.get(nxt, nxt))
        index += 2
    return "".join(out)


def _read_source(path: Path) -> str:
    if not path.is_file():
        _fail(f"нет файла {path}", path)
    return _strip_comments(path.read_text(encoding="utf-8"))


def record_phrases() -> list[tuple[str, str]]:
    """Пары (label, text) из `RECORD_PHRASES`; понятная ошибка при смене структуры."""
    source = _read_source(VOICE_RECORD_JS)
    declaration = _ARRAY_DECL.search(source)
    if not declaration:
        _fail("не найдено объявление `const RECORD_PHRASES = [`")
    end = _matching_bracket(source, declaration.end() - 1)
    body = source[declaration.end() : end]
    entries = [
        (_decode_js_string(match.group("label")), _decode_js_string(match.group("text")))
        for match in _ENTRY.finditer(body)
    ]
    if not entries:
        _fail("в RECORD_PHRASES не разобрано ни одной пары label/text")
    leftovers = _ENTRY.sub("", body).replace(",", "").strip()
    if leftovers:
        _fail(f"в RECORD_PHRASES остался неразобранный фрагмент: {leftovers[:80]!r}")
    return entries


_PROFILE_MAP_DECL = re.compile(r"\bconst\s+RECORD_PHRASE_PROFILES\s*=\s*\{")
_PROFILE_ENTRY = re.compile(
    r"(?P<label>" + _JS_STRING + r")\s*:\s*(?P<key>" + _JS_STRING + r")",
    re.DOTALL,
)


def record_phrase_profiles() -> list[tuple[str, str]]:
    """Пары (подпись фразы, ключ профиля) из `RECORD_PHRASE_PROFILES` (§16)."""
    source = _read_source(VOICE_RECORD_JS)
    declaration = _PROFILE_MAP_DECL.search(source)
    if not declaration:
        _fail("не найдено объявление `const RECORD_PHRASE_PROFILES = {`")
    end = _matching_bracket(source, declaration.end() - 1)
    body = source[declaration.end() : end]
    entries = [
        (_decode_js_string(match.group("label")), _decode_js_string(match.group("key")))
        for match in _PROFILE_ENTRY.finditer(body)
    ]
    if not entries:
        _fail("в RECORD_PHRASE_PROFILES не разобрано ни одной пары подпись/profile_key")
    leftovers = _PROFILE_ENTRY.sub("", body).replace(",", "").strip()
    if leftovers:
        _fail(f"в RECORD_PHRASE_PROFILES остался неразобранный фрагмент: {leftovers[:80]!r}")
    return entries


def yo_stress_test_phrase() -> str:
    """Текст `YO_STRESS_TEST_PHRASE`; ошибка, если объявление пропало."""
    # Фраза живёт в панели preview: ею пользуется «Что услышит модель», а не запись.
    source = _read_source(TEXT_PREVIEW_JS)
    match = _YO_DECL.search(source)
    if not match:
        _fail("не найдено объявление `const YO_STRESS_TEST_PHRASE = '...';`", TEXT_PREVIEW_JS)
    return _decode_js_string(match.group("value"))


# --- 1. Состав и неприкосновенность исходных пяти ------------------------------
def test_record_phrases_has_eleven_and_first_five_are_untouched():
    phrases = record_phrases()
    assert len(phrases) == 11, f"в списке должно быть 11 фраз, а разобрано {len(phrases)}"
    assert phrases[:5] == list(ORIGINAL_PHRASES), (
        "первые пять фраз обязаны совпадать буквально и идти в исходном порядке"
    )


# --- 2. Новые подписи и их порядок --------------------------------------------
def test_new_six_labels_present_in_document_order():
    labels = [label for label, _ in record_phrases()]
    assert labels[5:] == [label for label, _ in NEW_PHRASES], (
        "новые шесть подписей должны идти в порядке документа"
    )
    assert len(set(labels)) == len(labels), "подписи фраз не должны повторяться"


# --- 3. «ё» в новых фразах -----------------------------------------------------
def test_at_least_four_new_phrases_contain_yo():
    new = record_phrases()[5:]
    with_yo = [label for label, text in new if "ё" in text.lower()]
    assert len(with_yo) >= 4, (
        f"минимум четыре новые фразы обязаны содержать «ё», а нашли {len(with_yo)}: {with_yo}"
    )


# --- 4. Новые интонационные регистры ------------------------------------------
def test_new_intonation_registers_present():
    labels = [label.lower() for label, _ in record_phrases()[5:]]
    missing = [
        name
        for name, marker in NEW_REGISTER_MARKERS
        if not any(marker in label for label in labels)
    ]
    assert not missing, f"нет подписей с новыми регистрами: {missing}"


# --- 5. Mapping «подпись фразы → ключ профиля» (§16) ---------------------------
# Документ задаёт initial mapping буквально: одна фраза — одна интонация, и все
# одиннадцать различны. Здесь он записан как ожидание, потому что это **данные
# документа**, а не следствие кода: перепутанная пара не сломала бы ни один тест
# выше, но записала бы «иронию» под профилем восторга.
DOCUMENT_PHRASE_PROFILES: tuple[tuple[str, str], ...] = (
    ("Вопрос и утверждение", "NEUTRAL_QUESTION"),
    ("Восклицания, много шипящих", "EXCLAMATION"),
    ("Спокойная просьба (короче)", "CALM"),
    ("Только вопросы", "QUESTION"),
    ("Сложные сочетания согласных", "NEUTRAL"),
    ("Радость, восторг", "DELIGHT"),
    ("Огорчение, сочувствие", "SAD_SYMPATHETIC"),
    ("Лёгкая ирония", "IRONIC"),
    ("Строгий тон, короткий приказ", "STRICT"),
    ("Перечисление, ровный ритм", "ENUMERATION"),
    ("Быстрая, взволнованная речь", "EXCITED"),
)


def test_mapping_matches_document_and_covers_every_phrase():
    mapping = record_phrase_profiles()
    assert mapping == list(DOCUMENT_PHRASE_PROFILES), (
        "mapping §16 обязан совпадать с документом и идти в порядке фраз"
    )
    # Подписи — тот же список и в том же порядке, что и RECORD_PHRASES: иначе
    # мастер записал бы фразу не под ту интонацию, а заметить это было бы нечем.
    assert [label for label, _ in mapping] == [label for label, _ in record_phrases()]
    assert len({label for label, _ in mapping}) == len(mapping), (
        "одна подпись не может встречаться дважды: по ней ищется профиль фразы"
    )


def test_every_phrase_profile_is_recordable_and_unique():
    """Ключи mapping — только записываемые профили §10, и каждый ровно один раз."""
    keys = [key for _, key in record_phrase_profiles()]
    assert set(keys) <= set(emotions.PROFILE_KEYS), (
        "записать профиль под семантическую эмоцию (SURPRISE/FEAR) нельзя — "
        "backend ответит 400"
    )
    # Одиннадцать разных интонаций: две фразы под один профиль означали бы, что
    # вторая запись молча вытеснит первую как дубликат той же интонации.
    assert len(set(keys)) == len(keys) == len(emotions.PROFILE_KEYS)


def test_wizard_saves_recording_as_profile_of_selected_voice():
    """Мастер пишет запись в профиль выбранного голоса, а не в новый голос (§16)."""
    source = VOICE_RECORD_JS.read_text(encoding="utf-8")
    # Запись уходит в профили конкретного голоса — и вместе с фразой, по которой
    # записана: без `source_record_phrase_id` повторная запись не заменила бы
    # прежний профиль, а легла бы рядом вторым.
    assert "/api/voices/${voice.id}/references" in source
    assert "form.append('source_record_phrase_id', recordPhraseId(index))" in source
    # Режим выбирается явно: пока голос не выбран, запись идёт прежним путём —
    # через форму нового голоса.
    assert "RECORD_NEW_VOICE" in source
    assert "recordTargetVoice()" in source


# --- 6. Длина фразы ------------------------------------------------------------
def test_every_phrase_fits_short_reference_budget():
    """Каждая фраза укладывается в первые ~12 с референса, которые берёт F5.

    Новые фразы чуть длиннее прежних (самая длинная — ~8.8 с против ~7.4 с у самой
    длинной исходной), поэтому проверяется не «как у исходных знак в знак», а
    настоящий лимит движка: 12 секунд с запасом. Меряем оценщиком проекта, а не
    количеством знаков.
    """
    too_long = [
        (label, round(_reading_seconds(text), 1))
        for label, text in record_phrases()
        if _reading_seconds(text) > REFERENCE_BUDGET_SEC
    ]
    assert not too_long, (
        f"фразы длиннее {REFERENCE_BUDGET_SEC:.0f} с не влезают в референс F5: {too_long}"
    )


def test_new_phrases_stay_near_original_reading_time():
    """Новые фразы не растянуты под лимит: они того же порядка, что исходные.

    Документ правок требует, чтобы фразы остались короткими, а не были удлинены
    специально под 30-секундный лимит XTTS. Порог в 1.5× от самой длинной исходной
    фразы фиксирует именно это: разброс есть, растягивания нет.
    """
    phrases = record_phrases()
    originals = [_reading_seconds(text) for _, text in phrases[:5]]
    longest_original = max(originals)
    new = [(label, round(_reading_seconds(text), 1)) for label, text in phrases[5:]]
    stretched = [
        (label, seconds)
        for label, seconds in new
        if seconds > longest_original * 1.5
    ]
    assert not stretched, (
        f"новые фразы заметно длиннее исходных (максимум исходных "
        f"{longest_original:.1f} с): {stretched}"
    )


# --- 7. Тестовая фраза: «ё» восстанавливается, исходник не мутирует ------------
def test_yo_stress_phrase_is_test_data_not_a_record_phrase():
    phrase = yo_stress_test_phrase()
    assert phrase.strip(), "тестовая фраза не должна быть пустой"
    record_texts = [text for _, text in record_phrases()]
    assert phrase not in record_texts, "тестовая фраза не должна быть значением RECORD_PHRASES"
    assert not any(phrase in text for text in record_texts), (
        "тестовая фраза не должна входить в текст ни одной фразы записи"
    )


def test_yo_stress_phrase_restores_yo_without_mutating_source():
    phrase = yo_stress_test_phrase()
    original = str(phrase)

    # В исходнике нет гарантированной «ё», записанной через «е»: нормализация
    # ничего не меняет, значит все перечисленные слова уже стоят с «ё».
    normalized = normalize(phrase)
    assert normalized == phrase, "в исходнике остались слова с «е» на месте «ё»"
    for word in YO_STRESS_WORDS:
        assert word in normalized, f"в тестовой фразе нет слова с «ё»: {word}"

    # Обратная проверка: та же фраза, записанная целиком через «е», после
    # нормализации возвращается к исходнику — слова действительно гарантированные.
    deyo = phrase.replace("ё", "е").replace("Ё", "Е")
    assert deyo != phrase
    assert normalize(deyo) == phrase, "слова с «е» не восстановились в «ё»"

    # Нормализация не мутирует исходный текст константы.
    assert phrase == original


# --- 8. Тестовая фраза не предлагается при записи -----------------------------
def test_yo_stress_phrase_is_not_offered_for_recording():
    # Панель записи и панель preview разнесены по разным модулям, их обработчики
    # живут в точке входа. Проверяем все три файла: фраза не должна оказаться
    # рядом с разметкой записи ни в одном из них.
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (VOICE_RECORD_JS, TEXT_PREVIEW_JS, APP_JS)
    }
    assert "YO_STRESS_TEST_PHRASE" in sources[TEXT_PREVIEW_JS.name], (
        "константа должна использоваться, а не быть мёртвой"
    )
    assert "YO_STRESS_TEST_PHRASE" not in sources[VOICE_RECORD_JS.name], (
        "в модуле записи тестовой фразы быть не должно"
    )

    record_lines = [
        (name, line)
        for name, source in sources.items()
        for line in source.splitlines()
        if "record-phrase" in line
    ]
    assert record_lines, "должны быть обращения к элементам записи"
    record_files = {name for name, _ in record_lines}
    assert "voice-record.js" in record_files, "запись живёт в модуле записи"
    for name, line in record_lines:
        assert "YO_STRESS_TEST_PHRASE" not in line, (
            f"тестовая фраза не должна подставляться в UI записи: {name}: {line.strip()}"
        )

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "YO_STRESS_TEST_PHRASE" not in html


def test_yo_stress_phrase_is_wired_into_preview_panel():
    source = TEXT_PREVIEW_JS.read_text(encoding="utf-8")
    assert 'data-role="yo-example"' in source, "в панели preview должна быть кнопка-пример"
    assert "previewEl(mount, 'text').value = YO_STRESS_TEST_PHRASE" in source, (
        "кнопка-пример обязана подставлять тестовую фразу в поле preview"
    )
    # Сама панель подключена к точке входа: модуль должен быть импортирован, иначе
    # кнопка-пример существовала бы только в тексте.
    assert "} from './text-preview.js';" in APP_JS.read_text(encoding="utf-8")
