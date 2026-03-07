from __future__ import annotations

import calendar
import difflib
import re
from datetime import date
from typing import Any, List

import pandas as pd

from db import engine

def month_end(y: int, m: int) -> date:
    last = calendar.monthrange(y, m)[1]
    return date(y, m, last)

def iter_month_ends(y1: int, m1: int, y2: int, m2: int) -> List[date]:
    out: List[date] = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        out.append(month_end(y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out

def ensure_month_col(df: pd.DataFrame, month_labels: list[str] | None = None) -> pd.DataFrame:
    if "Месяц" in df.columns:
        return df

    rename_map = {}
    for cand in ("month", "Month", "MONTH", "Дата", "date", "Date"):
        if cand in df.columns:
            rename_map[cand] = "Месяц"
            break
    if rename_map:
        df = df.rename(columns=rename_map)
        if "Месяц" in df.columns:
            return df

    if month_labels is not None and len(month_labels) >= len(df):
        df = df.copy()
        df.insert(0, "Месяц", month_labels[:len(df)])
        return df

    df = df.copy()
    df.insert(0, "Месяц", [str(i) for i in range(1, len(df) + 1)])
    return df

def get_max_event_date_from_db() -> date:
    q = """
    SELECT
      GREATEST(
        (SELECT MAX(event_date) FROM calvings_births_raw),
        (SELECT MAX(event_date) FROM inseminations_raw),
        (SELECT MAX(event_date) FROM dryoff_raw),
        (SELECT MAX(event_date) FROM disposals_raw)
      ) AS max_date;
    """
    try:
        df = pd.read_sql(q, con=engine)
        v = df.loc[0, "max_date"]
        if pd.isna(v):
            return date.today()
        return pd.to_datetime(v).date()
    except Exception:
        return date.today()

def norm_label(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ").strip()
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    su = s.upper()
    su = su.replace("ТЁЛОЧК", "ТЁЛК")
    su = su.replace("ТЕЛОЧК", "ТЕЛК")
    return su

def vals_get(vals: dict, want_key: str, norm_map: dict | None = None) -> Any:
    if not isinstance(vals, dict) or not vals:
        return None
    if want_key in vals:
        return vals.get(want_key)

    if norm_map is None:
        norm_map = {norm_label(k): v for k, v in vals.items()}

    nk = norm_label(want_key)
    if nk in norm_map:
        return norm_map.get(nk)

    candidates = [k for k in norm_map.keys() if (nk in k) or (k in nk)]
    if len(candidates) == 1:
        return norm_map.get(candidates[0])

    close = difflib.get_close_matches(nk, list(norm_map.keys()), n=1, cutoff=0.92)
    if close:
        return norm_map.get(close[0])

    return None
