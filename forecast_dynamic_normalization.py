from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class SemenSexRatio:
    bull_share: float
    heifer_share: float


def to_semen_ratio(x: Any) -> SemenSexRatio:
    if isinstance(x, SemenSexRatio):
        return x
    if hasattr(x, "bull_share") and hasattr(x, "heifer_share"):
        return SemenSexRatio(float(x.bull_share), float(x.heifer_share))
    if isinstance(x, dict):
        b = float(x.get("bull_share", 0.5))
        h = float(x.get("heifer_share", 1.0 - b))
        return SemenSexRatio(b, h)
    return SemenSexRatio(0.5, 0.5)


def norm_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() in {"nan", "<na>", "none", "null"}:
        return ""
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s


def norm_gender(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("F", "Ж", "FEMALE", "0"):
        return "F"
    if s in ("M", "М", "MALE", "1"):
        return "M"
    return None


def norm_sex(x: object) -> str | None:
    if x is None:
        return None

    v = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    if v == "" or v in {"NAN", "NONE", "NULL", "0", "0.0"}:
        return None

    if v in {"F", "Ж", "ЖЕН", "ЖЕНСКИЙ"}:
        return "F"
    if "ТЕЛ" in v or "ТЁЛ" in v or "HEIF" in v or "FEMALE" in v:
        return "F"

    if v in {"M", "М", "МУЖ", "МУЖСКОЙ"}:
        return "M"
    if "БЫЧ" in v or "BULL" in v or "MALE" in v:
        return "M"

    return None


def norm_event_type(x: object) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")
    if v == "" or v == "NAN":
        return ""
    if ("РОЖ" in v) or ("BORN" in v) or ("BIRTH" in v):
        return "РОЖДЕН"
    if ("ОТЕЛ" in v) or ("CALV" in v):
        return "ОТЕЛ"
    return v


def norm_result(x: object) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")
    if v in {"", "NAN", "NONE", "NULL", "0", "0.0"}:
        return ""
    if v in {"P", "П"}:
        return "P"
    if "PREG" in v:
        return "P"
    if "СТЕЛ" in v or v in {"СТ", "СТ.", "СТ+", "СТЕЛЬНАЯ", "СТЕЛЬН"}:
        return "P"
    return v


def is_transfer_disposal_reason(x: object) -> bool:
    if x is None:
        return False
    v = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    if v in {"", "NAN", "NONE", "NULL"}:
        return False
    if "ПЕРЕЕЗД" in v:
        return True
    if "MOVE" in v or "TRANSFER" in v:
        return True
    return False


def classify_semen_from_bull_type(bull_type: object) -> str:
    v = "" if bull_type is None else str(bull_type).strip().upper()
    if v == "S" or "SEX" in v:
        return "sex"
    return "trad"


def classify_semen_from_bull_type_strict(bull_type: object) -> str | None:
    v = "" if bull_type is None else str(bull_type).strip().upper()
    if v in {"", "NAN", "NONE", "NULL"}:
        return None
    if v == "S" or "SEX" in v:
        return "sex"
    if v in {"T", "TRAD", "TRADITIONAL", "CONV", "CONVENTIONAL"}:
        return "trad"
    return None


__all__ = [
    "SemenSexRatio",
    "to_semen_ratio",
    "norm_id",
    "norm_gender",
    "norm_sex",
    "norm_event_type",
    "norm_result",
    "is_transfer_disposal_reason",
    "classify_semen_from_bull_type",
    "classify_semen_from_bull_type_strict",
]
