from __future__ import annotations

from typing import Any
import pandas as pd

BAD = "background-color: #ff0000; color: #ffffff; font-weight: 700;"

def fmt_cell(x):
    try:
        if x is None:
            return ""
        f = float(x)
        if pd.isna(f):
            return ""
        return str(int(round(f)))
    except Exception:
        return x


def style_positive_red(df_any: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame("", index=df_any.index, columns=df_any.columns)
    num = df_any.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    s[num > 0.0] = BAD
    return s
