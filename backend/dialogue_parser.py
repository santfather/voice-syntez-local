"""Парсинг диалога: имена спикеров, слоты голосов и параметры синтеза в тексте.

Голос задаётся тремя способами, которые можно свободно смешивать:

* ``Иван: Привет!``      — имя спикера в начале строки;
* ``ИВАН(1): Привет!``   — имя-метка (в аудио не читается) + слот 1;
* ``(1) Привет!``        — переключение на слот 1 без имени.

Маркер ``(N)`` действует в любом месте текста и переключает голос для всего,
что идёт до следующего маркера. Параметры синтеза можно задать прямо в маркере:
``(1 speed=1.2 cfg=2.5 nfe=16)`` — они перекрывают настройки карточки в UI.

Маркером считается только «(цифры)», внутри — необязательный список ``ключ=число``.
Всё остальное в скобках (``(тихо)``, ``(2024)``, ``12:30``) остаётся обычным текстом.
"""

import logging
import re
from dataclasses import dataclass, field

from . import config

logger = logging.getLogger(__name__)

# Разделители реплики: ":", "—", "–" и "-" только как отдельный токен.
_SEPARATOR_RE = re.compile(r"^\s*(?P<speaker>[^:\n—–]{1,60}?)\s*(?::|—|–|\s-\s)\s*(?P<text>.*)$")
_SPEAKER_STRIP_RE = re.compile(r"^[\s\-–—*•]+|[\s\-–—*•]+$")
# "12:30" или "1: текст" — не спикер, а часть реплики
_NUMERIC_SPEAKER_RE = re.compile(r"^\d{1,3}$")
# Имя спикера: короткая метка из букв/цифр без знаков препинания —
# иначе «Он подошёл. (2) И тут она сказала:» ушло бы в имя и пропало из аудио.
_NAME_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9 _\-()]{0,29}$")

# Маркер голоса: "(1)" или "(1 speed=1.2 cfg=2.5 nfe=16)"
_MARKER_RE = re.compile(r"\(\s*(?P<slot>\d{1,3})(?P<params>[^()]*)\)")
# Тот же маркер в конце имени-метки: "ИВАН(1)"
_TAIL_MARKER_RE = re.compile(r"\(\s*(?P<slot>\d{1,3})(?P<params>[^()]*)\)\s*$")
_PARAM_TOKEN_RE = re.compile(r"^(?P<key>[A-Za-z_]+)\s*=\s*(?P<value>\d+(?:\.\d+)?)$")

# Слово непосредственно перед маркером и разделитель сразу после него:
# "МАРГО(2): И тебе привет" — МАРГО это метка, а не произносимый текст.
_NAME_BEFORE_RE = re.compile(r"(?P<name>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_]*)\s*$")
_SEPARATOR_AFTER_RE = re.compile(r"\s*[:—–]")
# Мусор в начале куска: "- ", "— ", ": "
_CHUNK_HEAD_RE = re.compile(r"^[\s\-–—:;]+")

_PARAM_ALIASES = {
    "speed": "speed",
    "cfg": "cfg_strength",
    "cfg_strength": "cfg_strength",
    "nfe": "nfe_step",
    "nfe_step": "nfe_step",
}


def slot_key(slot: int) -> str:
    """Ключ голоса для слота. Решётка отличает слоты от имён спикеров."""
    return f"#{slot}"


def voice_label(key: str) -> str:
    """Человекочитаемое имя голоса по его ключу."""
    return f"Слот {key[1:]}" if key.startswith("#") else key


@dataclass
class Replica:
    """Кусок текста, который синтезируется одним голосом и одними параметрами."""

    voice: str  # ключ голоса: "#1" (слот 1) или "ИВАН"
    text: str
    line_number: int
    overrides: dict[str, float] = field(default_factory=dict)
    # Голос именно этой реплики поверх голоса спикера; None — наследовать.
    # В тексте не выражается: это правка карточки реплики в интерфейсе.
    voice_id: str | None = None
    # Текст, подготовленный анализом проекта: нормализация → «ё» → словарь →
    # ударения. Если он задан, синтез обязан взять **именно его** и ничего не
    # пересчитывать — это и есть контракт рендера: в модель уходит то, что
    # пользователь видел и подтверждал. `None` означает подготовку на месте; так
    # работают разовые задачи и legacy-входы, у которых сохранённого анализа нет.
    final_text: str | None = None

    @property
    def label(self) -> str:
        return voice_label(self.voice)

    def to_dict(self) -> dict:
        return {
            "voice": self.voice,
            "label": self.label,
            "text": self.text,
            "line_number": self.line_number,
            "overrides": self.overrides,
            "voice_id": self.voice_id,
            "final_text": self.final_text,
        }


@dataclass
class VoiceRef:
    """Один из голосов, встреченных в тексте."""

    key: str
    label: str
    slot: int | None

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "slot": self.slot}


@dataclass
class ParsedDialogue:
    replicas: list[Replica] = field(default_factory=list)

    @property
    def voices(self) -> list[VoiceRef]:
        """Уникальные голоса в порядке появления в тексте."""
        seen: dict[str, VoiceRef] = {}
        for replica in self.replicas:
            seen.setdefault(
                replica.voice,
                VoiceRef(
                    key=replica.voice,
                    label=replica.label,
                    slot=int(replica.voice[1:]) if replica.voice.startswith("#") else None,
                ),
            )
        return list(seen.values())

    @property
    def override_count(self) -> int:
        return sum(1 for replica in self.replicas if replica.overrides)

    def to_dict(self) -> dict:
        return {
            "replicas": [r.to_dict() for r in self.replicas],
            "voices": [v.to_dict() for v in self.voices],
            "override_count": self.override_count,
        }


@dataclass
class _Marker:
    start: int
    end: int
    slot: int
    overrides: dict[str, float]


def _parse_params(raw: str) -> dict[str, float] | None:
    """Разбирает «speed=1.2 cfg=2.5» в словарь. None — если это не параметры."""
    values: dict[str, float] = {}
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        match = _PARAM_TOKEN_RE.match(token)
        if not match:
            return None
        key = _PARAM_ALIASES.get(match.group("key").lower())
        if key:
            values[key] = float(match.group("value"))
    return values


def _clamp_overrides(values: dict[str, float]) -> dict[str, float]:
    """Приводит параметры из текста к допустимым границам."""
    result: dict[str, float] = {}
    if "speed" in values:
        result["speed"] = min(max(values["speed"], config.SPEED_RANGE[0]), config.SPEED_RANGE[1])
    if "cfg_strength" in values:
        result["cfg_strength"] = min(max(values["cfg_strength"], config.CFG_RANGE[0]), config.CFG_RANGE[1])
    if "nfe_step" in values:
        nfe = round(values["nfe_step"])
        if nfe in config.NFE_ALLOWED:  # недопустимое значение — оставляем как в карточке
            result["nfe_step"] = nfe
    return result


def _split_marker(slot_raw: str, params_raw: str) -> tuple[int, dict[str, float]] | None:
    """Проверяет содержимое скобок и возвращает (слот, overrides)."""
    params = _parse_params(params_raw)
    if params is None:
        return None
    slot = int(slot_raw)
    if not 1 <= slot <= config.MAX_SLOT_NUMBER:
        return None
    return slot, _clamp_overrides(params)


def _find_markers(text: str) -> list[_Marker]:
    markers: list[_Marker] = []
    for match in _MARKER_RE.finditer(text):
        parsed = _split_marker(match.group("slot"), match.group("params"))
        if parsed is None:
            continue
        slot, overrides = parsed
        markers.append(_Marker(start=match.start(), end=match.end(), slot=slot, overrides=overrides))
    return markers


def _split_tail_marker(label: str) -> tuple[int | None, dict[str, float], str]:
    """Отделяет «(1 speed=1.2)» в конце имени. Возвращает (слот, overrides, имя)."""
    match = _TAIL_MARKER_RE.search(label)
    if not match:
        return None, {}, label
    parsed = _split_marker(match.group("slot"), match.group("params"))
    if parsed is None:
        return None, {}, label
    slot, overrides = parsed
    return slot, overrides, label[: match.start()].strip()


def _split_line_prefix(line: str) -> tuple[str | None, int | None, dict[str, float], str]:
    """Выделяет «Имя:» / «ИМЯ(1):» в начале строки.

    Возвращает (имя, слот, overrides, текст реплики); если префикса нет —
    (None, None, {}, исходная строка).
    """
    match = _SEPARATOR_RE.match(line)
    if not match:
        return None, None, {}, line

    raw_label = _SPEAKER_STRIP_RE.sub("", match.group("speaker"))
    if not raw_label or _NUMERIC_SPEAKER_RE.match(raw_label):
        return None, None, {}, line

    slot, overrides, name = _split_tail_marker(raw_label)
    if name:
        if not _NAME_RE.match(name):  # «Как дела? (3) — спросила девушка» — это не имя
            return None, None, {}, line
    elif slot is None:
        return None, None, {}, line

    return name or None, slot, overrides, match.group("text").strip()


def _marker_label(text: str, marker: _Marker) -> tuple[str | None, int, int]:
    """Имя-метка перед маркером: «МАРГО(2):» → ("МАРГО", начало, конец с разделителем)."""
    found = _NAME_BEFORE_RE.search(text[: marker.start])
    if not found:
        return None, marker.start, marker.end
    separator = _SEPARATOR_AFTER_RE.match(text[marker.end:])
    if not separator:
        return None, marker.start, marker.end
    return found.group("name"), found.start(), marker.end + separator.end()


def parse_dialogue(raw_text: str, max_replica_chars: int | None = None) -> ParsedDialogue:
    """Разбирает текст в список кусков, каждый — со своим голосом."""
    if not raw_text or not raw_text.strip():
        return ParsedDialogue()

    replicas: list[Replica] = []
    current: Replica | None = None

    def start(voice: str, overrides: dict[str, float], line_number: int) -> Replica:
        nonlocal current
        current = Replica(
            voice=voice, text="", line_number=line_number, overrides=dict(overrides)
        )
        replicas.append(current)
        return current

    def append(chunk: str, line_number: int) -> None:
        chunk = chunk.strip()
        if not chunk:
            return
        # Текст до первого маркера (или без единого маркера) читает голос слота 1.
        segment = current or start(slot_key(config.DEFAULT_SLOT), {}, line_number)
        if segment.text:
            segment.text = f"{segment.text} {chunk}"
            return
        chunk = _CHUNK_HEAD_RE.sub("", chunk)
        if chunk:
            segment.text = chunk

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        name, slot, prefix_overrides, body = _split_line_prefix(line)
        if name is not None or slot is not None:
            start(slot_key(slot) if slot is not None else name, prefix_overrides, line_number)

        position = 0
        for marker in _find_markers(body):
            _label, label_start, label_end = _marker_label(body, marker)
            append(body[position:label_start], line_number)
            start(slot_key(marker.slot), marker.overrides, line_number)
            position = label_end
        append(body[position:], line_number)

    result = [replica for replica in replicas if replica.text]

    if max_replica_chars:
        result = _split_long_replicas(result, max_replica_chars)

    return ParsedDialogue(replicas=result)


def _split_long_replicas(replicas: list[Replica], max_chars: int) -> list[Replica]:
    """Режет слишком длинные реплики по границам предложений.

    Раньше такая реплика отклонялась с ошибкой, и пользователь делил её вручную —
    обычно по счётчику символов, то есть посреди мысли. Разрыв интонации между
    двумя законченными предложениями слышен заметно слабее.
    """
    expanded: list[Replica] = []
    for replica in replicas:
        if len(replica.text) <= max_chars:
            expanded.append(replica)
            continue
        pieces = split_into_chunks(replica.text, max_chars)
        logger.info(
            "Реплика (строка %s) длиннее %s знаков — разрезана на %s кусков",
            replica.line_number, max_chars, len(pieces),
        )
        expanded.extend(
            Replica(
                voice=replica.voice,
                text=piece,
                line_number=replica.line_number,
                overrides=dict(replica.overrides),
                voice_id=replica.voice_id,
            )
            for piece in pieces
        )
    return expanded


# --- нарезка сплошного текста (режим «Сплошной текст») ------------------------
# F5-TTS синтезирует кусок целиком и ничего не помнит о соседних, поэтому резать
# внутри предложения нельзя: сброс интонации придётся на середину мысли. Копим
# предложения, пока они влезают в лимит, и режем только на их границах.
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")


def _paragraphs(text: str) -> list[str]:
    """Абзацы текста: граница — пустая строка.

    Перенос строки внутри абзаца считается мягким: в .txt это обычно жёсткая
    перенос-разметка, а в .md — разметка блока. Резать по каждому переносу нельзя:
    файл на 3700 знаков давал 70+ кусков по 50 знаков, а каждый кусок — отдельный
    прогон модели (~20 с).
    """
    paragraphs: list[str] = []
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line.strip())
        elif lines:
            paragraphs.append(" ".join(lines))
            lines = []
    if lines:
        paragraphs.append(" ".join(lines))
    return paragraphs


def _sentences(paragraph: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.findall(paragraph) if part.strip()]


def _wrap_long(sentence: str, max_chars: int) -> list[str]:
    """Аварийный разрез предложения, которое не влезает в лимит целиком."""
    pieces: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(" "), window.rfind(","), window.rfind(";"), window.rfind(":"))
        if cut < max_chars // 2:  # нормального места для разрыва нет — режем жёстко
            cut = max_chars - 1
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip():
        pieces.append(rest.strip())
    return [piece for piece in pieces if piece]


def split_into_chunks(text: str, max_chars: int | None = None) -> list[str]:
    """Режет сплошной текст на куски по границам предложений и абзацев."""
    limit = max_chars or config.MAX_REPLICA_CHARS
    # Абзац закрываем, только если кусок уже прилично заполнен: иначе короткие
    # абзацы (заголовки, пункты списка) снова дадут сотни крошечных кусков.
    soft_limit = max(limit // 2, 1)
    chunks: list[str] = []
    buffer = ""
    for paragraph in _paragraphs(text):
        for sentence in _sentences(paragraph):
            for piece in _wrap_long(sentence, limit):
                if not buffer:
                    buffer = piece
                elif len(buffer) + 1 + len(piece) <= limit:
                    buffer = f"{buffer} {piece}"
                else:
                    chunks.append(buffer)
                    buffer = piece
        if buffer and len(buffer) >= soft_limit:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks
