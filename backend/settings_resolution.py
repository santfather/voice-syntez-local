"""Единая точка вычисления параметров синтеза: движок → голос → спикер → реплика.

Иерархия собирается здесь, а не по кусочкам в API и на фронте: пайплайн,
проектные роуты и прослушивание голоса должны отвечать на вопрос «чем сейчас
читается эта реплика» одинаково. Иначе в интерфейсе видно одно значение, в модель
уходит другое, а «настроил голос один раз» перестаёт работать у нового диалога.

Слои — это разреженные словари: значение в слое означает «здесь выбрано именно
это», отсутствия ключа или `None` — «наследуется от нижнего слоя». Так сброс
правки не приходится выражать отдельным действием: снятие ключа и есть сброс.
"""

from typing import Any

from . import config
from .engines.base import engine_info

# Слои иерархии; порядок = приоритет, каждое следующее значение перекрывает предыдущее.
SOURCE_ENGINE = "engine"
SOURCE_VOICE = "voice"
SOURCE_SPEAKER = "speaker"
SOURCE_REPLICA = "replica"

# Ручки, которые есть у любого движка: скорость, ручки F5, пост-обработка куска
# и пауза перед репликой. Всё остальное — ручки конкретного движка, объявленные в
# его паспорте (`engines/base.EngineParam`).
COMMON_FIELDS = (
    "speed",
    "cfg_strength",
    "nfe_step",
    "target_rms",
    "gain_db",
    "pitch_semitones",
    "pause_override_ms",
)
# Целые ручки: в JSON они лежат числом, а движку и расчёту паузы нужен int.
INT_FIELDS = ("nfe_step", "pause_override_ms")

# Границы общих ручек. Те же числа, что и в моделях запросов: источник один —
# конфиг, а не копия в каждой проверке.
_RANGES: dict[str, tuple[float, float]] = {
    "speed": config.SPEED_RANGE,
    "cfg_strength": config.CFG_RANGE,
    "target_rms": config.TARGET_RMS_RANGE,
    "gain_db": config.GAIN_DB_RANGE,
    "pitch_semitones": config.PITCH_SEMITONES_RANGE,
    "pause_override_ms": config.PAUSE_MS_RANGE,
}


def engine_defaults(engine_id: str) -> dict:
    """Нижний слой иерархии: то, чем движок читает текст без выбора пользователя.

    Идентификатор движка лежит в самом словаре (`"engine"`) — по нему зажимаются
    ручки этого движка, а вызывающему не приходится передавать его отдельно.
    """
    info = engine_info(engine_id)
    return {
        "engine": engine_id,
        "speed": config.DEFAULT_SPEED,
        "cfg_strength": config.DEFAULT_CFG_STRENGTH,
        "nfe_step": config.DEFAULT_NFE_STEP,
        "target_rms": config.DEFAULT_TARGET_RMS,
        "gain_db": config.DEFAULT_GAIN_DB,
        "pitch_semitones": config.DEFAULT_PITCH_SEMITONES,
        "pause_override_ms": None,
        "engine_params": {param.name: param.default for param in info.params},
    }


def explicit(layer: dict | None) -> dict:
    """Слой без «пустых» значений: `None` — это «не задано», а не «выключено»."""
    if not layer:
        return {}
    return {key: value for key, value in layer.items() if value is not None}


def clamp_value(engine_id: str, field: str, value: Any) -> Any:
    """Приводит значение к границам поля.

    Зажим идёт в конце, после выбора слоя: иначе значение, попавшее внутрь границ
    только на время, выглядело бы выбранным и перекрывало нижний слой.
    """
    if value is None:
        # «Не задано» — это не число: у паузы пустое значение значит «общая для
        # диалога», и превращать его в границу диапазона было бы подменой смысла.
        return None
    if field == "nfe_step":
        try:
            number = int(value)
        except (TypeError, ValueError):
            return config.DEFAULT_NFE_STEP
        if number in config.NFE_ALLOWED:
            return number
        return min(config.NFE_ALLOWED, key=lambda allowed: abs(allowed - number))
    if field in _RANGES:
        low, high = _RANGES[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            return config.DEFAULT_NFE_STEP if field == "nfe_step" else low
        number = min(max(number, low), high)
        return int(round(number)) if field in INT_FIELDS else round(number, 4)
    for param in engine_info(engine_id).params:
        if param.name == field:
            return param.clamp(value)
    return value  # поле неизвестно движку — оставляем как есть, до модели оно не дойдёт


def _override(resolved: dict[str, dict], field: str, value: Any, source: str, **extra: Any) -> None:
    """Кладёт значение слоя поверх нижнего, сохраняя прежнее как `inherited`.

    Прежнее значение — ровно то, что вернёт сброс этой правки. Интерфейсу оно нужно
    здесь же: иначе подпись «сбросить → 1.00x» пришлось бы считать на клиенте,
    повторив порядок слоёв, — а это второе место, которое потом разойдётся.
    """
    previous = resolved.get(field)
    item = {"value": value, "source": source, **extra}
    if previous is not None:
        item["inherited"] = previous["value"]
    resolved[field] = item


def resolve_synthesis_settings(
    engine_defaults: dict,
    voice_settings: dict | None = None,
    speaker_overrides: dict | None = None,
    replica_overrides: dict | None = None,
) -> dict[str, dict]:
    """Эффективные параметры синтеза с указанием источника каждого значения.

    Возвращает `{поле: {"value": …, "source": "engine" | "voice" | "speaker" | "replica"}}`;
    у перекрытого поля добавляется `inherited` — значение нижнего слоя, то есть то,
    к чему вернёт сброс. Ручки движка помечены флагом `engine_param`. Этого
    достаточно, чтобы показать «наследуется от голоса Анна» и «правка этой
    реплики» одной строкой, без повторного пересчёта иерархии на клиенте.
    """
    engine_id = str(engine_defaults.get("engine") or "")
    declared = dict(engine_defaults.get("engine_params") or {})
    resolved: dict[str, dict] = {}

    for field in COMMON_FIELDS:
        if field in engine_defaults:
            resolved[field] = {"value": engine_defaults[field], "source": SOURCE_ENGINE}
    for name, value in declared.items():
        resolved[name] = {"value": value, "source": SOURCE_ENGINE, "engine_param": True}

    for layer, source in (
        (voice_settings, SOURCE_VOICE),
        (speaker_overrides, SOURCE_SPEAKER),
        (replica_overrides, SOURCE_REPLICA),
    ):
        for field, value in explicit(layer).items():
            if field == "engine_params":
                # Ручки чужого движка (температура у F5, CFG у XTTS) не применяются:
                # их просто нет в паспорте того движка, которым читается реплика.
                for name, param_value in explicit(value).items():
                    if name in declared:
                        _override(resolved, name, param_value, source, engine_param=True)
                continue
            if field in COMMON_FIELDS:
                _override(resolved, field, value, source)

    # Зажим — после выбора слоя, и над обоими числами: иначе подпись «сброс →»
    # показывала бы значение, которого реплика после сброса не получит.
    clamped: dict[str, dict] = {}
    for field, item in resolved.items():
        result = {**item, "value": clamp_value(engine_id, field, item["value"])}
        if "inherited" in item:
            result["inherited"] = clamp_value(engine_id, field, item["inherited"])
        clamped[field] = result
    return clamped


def effective_values(resolved: dict[str, dict]) -> dict[str, Any]:
    """Значения без источников — то, что уходит в модель и в сборку куска."""
    return {field: item["value"] for field, item in resolved.items()}


def split_values(values: dict[str, Any]) -> tuple[dict, dict]:
    """Делит значения на общие ручки и ручки движка."""
    common = {field: value for field, value in values.items() if field in COMMON_FIELDS}
    engine = {field: value for field, value in values.items() if field not in COMMON_FIELDS}
    return common, engine
