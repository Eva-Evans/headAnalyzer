from __future__ import annotations

"""
forecast_dynamic.py

Динамическая (дневная) симуляция поголовья на основе:
- фактических таблиц (inseminations_raw, calvings_births_raw, dryoff_raw, disposals_raw, bulls_raw)
- параметров модели (model_params/defaults.py)

Ключевая идея:
1) На дату "as_of" собираем агрегированное состояние стада (HerdState).
2) Дальше каждый день "прокручиваем" состояние:
   - возраст/ДИМ сдвигаются
   - беременности "отсчитываются" к отёлу
   - часть open животных осеменяется и часть из них становится стельной
   - часть животных выбывает по hazard-формам
   - в конце месяца применяем ограничения по вместимости (реализация)
3) Для каждой целевой даты (конец месяца) считаем срезы по группам.

ВАЖНО ПРО "НУЛИ В ТЁЛКАХ":
Исторически нули в "Тёлки 3–8 мес" и "Тёлки ≥9 мес" возникают, когда initial state
собирался только из "телят с reg" (строки телят), а в данных таких строк нет или мало.
В этом файле initial state всегда «подсекается» по событиям отёла матери:
- если в calvings_births_raw есть строки телят — используем их;
- если строк телят нет — восстанавливаем рождение телёнка из события отёла коровы (по матери и дате).

При этом:
- НЕ удваиваем: если по отёлу есть строки телят — НЕ синтезируем телёнка "вдобавок";
- если у телёнка не указан пол — распределяем по долям bull/heifer для semen типа (trad/sex),
  определённого по последнему P-осеменению перед отёлом.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from db import engine
from model_params import (
    GESTATION_DAYS,
    DRY_DAYS,
    CONCEPTION_PARAMS,
    DISPOSAL_PARAMS,
    ANNUAL_DISPOSAL_RATE,
    SEMEN_USAGE_PROBS,
    SEMEN_SEX_RATIOS,
    INSEMINATION_PARAMS,
    HERD_CAPACITY,
)

# ------------------------------------------------------------
# Types
# ------------------------------------------------------------

@dataclass(frozen=True)
class SemenSexRatio:
    bull_share: float
    heifer_share: float


def _to_semen_ratio(x: Any) -> SemenSexRatio:
    """Приводим SEMEN_SEX_RATIOS из model_params к нашему dataclass."""
    if isinstance(x, SemenSexRatio):
        return x
    if hasattr(x, "bull_share") and hasattr(x, "heifer_share"):
        return SemenSexRatio(float(x.bull_share), float(x.heifer_share))
    if isinstance(x, dict):
        b = float(x.get("bull_share", 0.5))
        h = float(x.get("heifer_share", 1.0 - b))
        return SemenSexRatio(b, h)
    return SemenSexRatio(0.5, 0.5)


# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------

import re
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def norm_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s

def _extract_calf_births(calv: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Возвращает уникальные рождения телят (reg телёнка) с датой рождения и полом.
    Берём ТОЛЬКО event_type="РОЖДЕН", чтобы не сломаться на "последних" строках.
    """
    if calv.empty:
        return pd.DataFrame(columns=["reg_s", "birth_dt", "sex_norm"])

    c = calv.copy()
    c["event_type_n"] = c["event_type"].apply(norm_event_type)
    c["event_date_n"] = pd.to_datetime(c["event_date"], errors="coerce").dt.normalize()
    c["birth_date_n"] = pd.to_datetime(c["birth_date"], errors="coerce").dt.normalize()
    c["reg_s"] = c["reg"].apply(norm_id)
    c["sex_norm"] = c["sex"].apply(norm_sex)

    born = c[
        (c["event_type_n"] == "РОЖДЕН")
        & (c["reg_s"].notna()) & (c["reg_s"] != "")
        & (c["sex_norm"].isin(["F", "M"]))
    ].copy()

    if born.empty:
        return pd.DataFrame(columns=["reg_s", "birth_dt", "sex_norm"])

    # дата рождения: birth_date если есть, иначе event_date
    born["birth_dt"] = born["birth_date_n"]
    m = born["birth_dt"].isna() & born["event_date_n"].notna()
    born.loc[m, "birth_dt"] = born.loc[m, "event_date_n"]

    born = born[born["birth_dt"].notna() & (born["birth_dt"] <= as_of_ts)].copy()

    # уникализируем по телёнку: берём САМУЮ РАННЮЮ дату рождения
    born = (
        born.sort_values(["reg_s", "birth_dt"], kind="mergesort")
            .groupby("reg_s", sort=False, as_index=False)
            .first()[["reg_s", "birth_dt", "sex_norm"]]
    )
    return born

def norm_sex(x: object) -> str | None:
    """
    Нормализуем пол телёнка.

    В сырых данных часто встречаются:
      - F / Ж / Т / ТЁЛКА / HEIFER / FEMALE
      - M / М / Б / БЫЧОК / BULL / MALE
      - иногда целые слова, иногда одна буква, иногда с пробелами/точками
    """
    if x is None:
        return None

    v = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    if v == "" or v in {"NAN", "NONE", "NULL", "0", "0.0"}:
        return None

    # женский
    if v in {"F", "Ж", "ЖЕН", "ЖЕНСКИЙ"}:
        return "F"
    if "ТЕЛ" in v or "ТЁЛ" in v or "HEIF" in v or "FEMALE" in v:
        return "F"

    # мужской
    if v in {"M", "М", "МУЖ", "МУЖСКОЙ"}:
        return "M"
    if "БЫЧ" in v or "BULL" in v or "MALE" in v:
        return "M"

    return None



def norm_event_type(x: object) -> str:
    """
    Приводим event_type в 2 основных вида:
      - "РОЖДЕН" (строка телёнка)
      - "ОТЕЛ"   (строка отёла/отёл коровы, если такие есть)
    """
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


def classify_semen_from_bull_type(bull_type: object) -> str:
    """Возвращаем 'sex' или 'trad'."""
    v = "" if bull_type is None else str(bull_type).strip().upper()
    if v == "S" or "SEX" in v:
        return "sex"
    return "trad"


# ------------------------------------------------------------
# Small date helpers
# ------------------------------------------------------------

MAX_DIM = 500
MAX_AGE_DAYS = 730     # тёлки до ~24 мес (под UI 9–24)
BULL_AGE_MAX = 90      # бычки держим только 0–3 мес (по сути)
OVERDUE_CLAMP_DAYS = 14


def age_months(d: int) -> int:
    return int(d // 30)


def end_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def shift_right(a: np.ndarray) -> np.ndarray:
    """Возраст +1 день: index i -> i+1. То, что было на хвосте, «вылетает»."""
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out


def shift_left(a: np.ndarray) -> np.ndarray:
    """Countdown -1 день: index i -> i-1. То, что было в 0, «вылетает» (событие наступило)."""
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out

def _effective_ai_interval_days(interval_raw: float, mean_target: float, first: float, spc: float) -> float:
    """
    Стабилизация интервала между осеменениями.

    interval_raw из данных часто шумный/завышенный.
    Мы хотим, чтобы средняя "точка зачатия" не уезжала:
        first + (spc-1)*interval ≈ mean_target

    Поэтому берём derived-интервал из этой формулы и миксуем с raw.
    """
    interval_raw = _clamp(float(interval_raw), 14.0, 90.0)
    spc = float(spc)

    if spc <= 1.01:
        return interval_raw

    derived = (float(mean_target) - float(first)) / max(1e-9, (spc - 1.0))
    derived = _clamp(derived, 14.0, 60.0)

    # raw обычно хуже -> меньший вес
    return 0.30 * interval_raw + 0.70 * derived

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ------------------------------------------------------------
# merge_asof safe (pandas sometimes complains about sorting)
# ------------------------------------------------------------

def _merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str | None = None,
    direction: str = "backward",
    allow_exact_matches: bool = True,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    if direction != "backward":
        raise ValueError("Only direction='backward' implemented in fallback version")

    l = left.copy()
    r = right.copy()
    l["_row_id"] = np.arange(len(l), dtype=np.int64)

    l[left_on] = pd.to_datetime(l[left_on], errors="coerce")
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce")

    if by is None:
        l = l[l[left_on].notna()].copy()
        r = r[r[right_on].notna()].copy()
        l = l.sort_values([left_on], kind="mergesort").reset_index(drop=True)
        r = r.sort_values([right_on], kind="mergesort").reset_index(drop=True)

        out = pd.merge_asof(
            l,
            r,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    l[by] = l[by].astype("string").fillna("").str.strip()
    r[by] = r[by].astype("string").fillna("").str.strip()

    l = l[(l[by] != "") & l[left_on].notna()].copy()
    r = r[(r[by] != "") & r[right_on].notna()].copy()

    # Important: sort by time first, then by key (works for most pandas builds)
    l = l.sort_values([left_on, by], kind="mergesort").reset_index(drop=True)
    r = r.sort_values([right_on, by], kind="mergesort").reset_index(drop=True)

    try:
        out = pd.merge_asof(
            l,
            r,
            by=by,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    except ValueError as e:
        if "keys must be sorted" not in str(e).lower():
            raise

        # fast fallback: numpy searchsorted per group
        out = l.copy()
        for c in r.columns:
            if c not in out.columns:
                out[c] = pd.NA

        r_groups: dict[str, pd.DataFrame] = {}
        for key, grp in r.groupby(by, sort=False):
            r_groups[str(key)] = grp.sort_values(right_on, kind="mergesort")

        for key, lg in out.groupby(by, sort=False):
            rg = r_groups.get(str(key))
            if rg is None or rg.empty:
                continue
            lt = lg[left_on].values.astype("datetime64[ns]").astype("int64")
            rt = rg[right_on].values.astype("datetime64[ns]").astype("int64")
            pos = np.searchsorted(rt, lt, side="right") - 1
            ok = pos >= 0
            if not np.any(ok):
                continue
            out_idx = lg.index.values[ok]
            take = rg.iloc[pos[ok]]
            for c in rg.columns:
                if c == by:
                    continue
                if c in out.columns and c == left_on:
                    continue
                out.loc[out_idx, c] = take[c].values

        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)


# ------------------------------------------------------------
# Runtime params resolve
# ------------------------------------------------------------

def _resolve_runtime_params(overrides: dict | None) -> dict:
    ov = overrides or {}

    gest_default = float(GESTATION_DAYS)
    dry_default = int(DRY_DAYS)

    cp = ov.get("CONCEPTION_PARAMS") or {
        "avg_cow_dim_by_lact": dict(CONCEPTION_PARAMS.avg_cow_dim_by_lact),
        "avg_cow_dim_global": float(CONCEPTION_PARAMS.avg_cow_dim_global),
        "avg_heifer_age_days": float(CONCEPTION_PARAMS.avg_heifer_age_days),
    }

    disp = ov.get("DISPOSAL_PARAMS") or deepcopy(DISPOSAL_PARAMS)
    annual_disp = float(ov.get("ANNUAL_DISPOSAL_RATE", ANNUAL_DISPOSAL_RATE))

    ins = ov.get("INSEMINATION_PARAMS") or {
        "cow_services_per_conception": float(INSEMINATION_PARAMS.cow_services_per_conception),
        "cow_ai_interval_days": float(INSEMINATION_PARAMS.cow_ai_interval_days),
        "cow_first_ai_dim_by_lact": dict(INSEMINATION_PARAMS.cow_first_ai_dim_by_lact),
        "heifer_services_per_conception": float(INSEMINATION_PARAMS.heifer_services_per_conception),
        "heifer_ai_interval_days": float(INSEMINATION_PARAMS.heifer_ai_interval_days),
        "heifer_first_ai_age_days": float(INSEMINATION_PARAMS.heifer_first_ai_age_days),
    }

    semen_usage = ov.get("SEMEN_USAGE_SHARES")

    gest_days = int(round(float(ov.get("GESTATION_DAYS", gest_default))))
    gest_days = max(200, min(310, gest_days))

    dry_days = int(round(float(ov.get("DRY_DAYS", dry_default))))
    dry_days = max(20, min(120, dry_days))

    annual_disp = float(max(0.0, min(0.5, annual_disp)))

    return {
        "GESTATION_DAYS": gest_days,
        "DRY_DAYS": dry_days,
        "CONCEPTION_PARAMS": cp,
        "DISPOSAL_PARAMS": disp,
        "ANNUAL_DISPOSAL_RATE": annual_disp,
        "INSEMINATION_PARAMS": ins,
        "SEMEN_USAGE_SHARES": semen_usage,
    }


# ------------------------------------------------------------
# HerdState
# ------------------------------------------------------------

@dataclass
class HerdState:
    # коровы
    open_dim: Dict[int, np.ndarray]                       # lact_cat -> [0..MAX_DIM]
    preg_lact: Dict[Tuple[int, str], np.ndarray]          # (lact_cat, semen) -> [0..gest_days]
    preg_dry:  Dict[Tuple[int, str], np.ndarray]          # (lact_cat, semen) -> [0..gest_days]
    # тёлки/молодняк
    heifer_age: np.ndarray                                # [0..MAX_AGE_DAYS]
    heifer_preg: Dict[str, np.ndarray]                    # semen -> [0..gest_days]
    bull_age: np.ndarray                                  # [0..BULL_AGE_MAX]


def init_empty_state(gest_days: int) -> HerdState:
    open_dim = {l: np.zeros(MAX_DIM + 1, dtype=float) for l in (1, 2, 3, 4)}
    preg_lact = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    preg_dry  = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    heifer_age = np.zeros(MAX_AGE_DAYS + 1, dtype=float)
    heifer_preg = {s: np.zeros(gest_days + 1, dtype=float) for s in ("trad", "sex")}
    bull_age = np.zeros(BULL_AGE_MAX + 1, dtype=float)
    return HerdState(open_dim, preg_lact, preg_dry, heifer_age, heifer_preg, bull_age)


def _copy_state(s: HerdState) -> HerdState:
    return HerdState(
        open_dim={k: v.copy() for k, v in s.open_dim.items()},
        preg_lact={k: v.copy() for k, v in s.preg_lact.items()},
        preg_dry={k: v.copy() for k, v in s.preg_dry.items()},
        heifer_age=s.heifer_age.copy(),
        heifer_preg={k: v.copy() for k, v in s.heifer_preg.items()},
        bull_age=s.bull_age.copy(),
    )


# ------------------------------------------------------------
# DB load
# ------------------------------------------------------------

def load_tables() -> Dict[str, pd.DataFrame]:
    calv = pd.read_sql(
        "SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw",
        con=engine,
    )
    ins  = pd.read_sql(
        "SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw",
        con=engine,
    )
    dry  = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", con=engine)
    disp = pd.read_sql("SELECT reg, event_date, disposal_reason FROM disposals_raw", con=engine)
    bulls = pd.read_sql("SELECT bull_code, bull_type FROM bulls_raw", con=engine)
    return {"calv": calv, "ins": ins, "dry": dry, "disp": disp, "bulls": bulls}


def latest_data_date(tables: Dict[str, pd.DataFrame]) -> date:
    mx = None
    for key in ("calv", "ins", "dry", "disp"):
        df = tables[key]
        if "event_date" in df.columns and not df.empty:
            d = pd.to_datetime(df["event_date"], errors="coerce").max()
            if pd.notna(d):
                dx = d.date()
                mx = dx if mx is None else max(mx, dx)
    return mx or date.today()


# ------------------------------------------------------------
# Semen usage (trad/sex)
# ------------------------------------------------------------

def compute_semen_usage_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "cow_trad": float(SEMEN_USAGE_PROBS.cow_trad),
        "cow_sex": float(SEMEN_USAGE_PROBS.cow_sex),
        "heifer_trad": float(SEMEN_USAGE_PROBS.heifer_trad),
        "heifer_sex": float(SEMEN_USAGE_PROBS.heifer_sex),
    }

    if ins.empty or bulls.empty:
        return fallback

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    p = ins[(ins["event_date"].notna()) & (ins["result_norm"] == "P") & (ins["bull_s"] != "")].copy()
    if p.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["bull_type_s"] = bulls["bull_type"].astype(str).str.strip().str.upper()
    bulls["semen"] = np.where(bulls["bull_type_s"] == "S", "sex", "trad")
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p["semen_known"] = p["semen"].isin(["trad", "sex"])

    max_dt = p["event_date"].max()
    p_365 = p[p["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy() if pd.notna(max_dt) else p.copy()

    def _shares(df: pd.DataFrame) -> Tuple[float | None, float | None, int, float]:
        if df.empty:
            return None, None, 0, 0.0
        known = df[df["semen_known"]]
        if known.empty:
            return None, None, int(len(df)), 0.0
        total = int(len(known))
        sex = int((known["semen"] == "sex").sum())
        trad = total - sex
        known_rate = float(len(known)) / float(len(df))
        return trad / total, sex / total, total, known_rate

    def _mix(is_cow: bool) -> Tuple[float, float]:
        all_df = p[p["lact"].gt(0) if is_cow else p["lact"].le(0)].copy()
        y_df = p_365[p_365["lact"].gt(0) if is_cow else p_365["lact"].le(0)].copy()

        trad_all, sex_all, n_all, known_rate_all = _shares(all_df)
        trad_y, sex_y, n_y, known_rate_y = _shares(y_df)

        if (trad_all is None) or (n_all < 200) or (known_rate_all < 0.3):
            return (fallback["cow_trad"], fallback["cow_sex"]) if is_cow else (fallback["heifer_trad"], fallback["heifer_sex"])

        w = 0.0
        if (trad_y is not None) and (n_y >= 200) and (known_rate_y >= 0.6):
            w = min(0.7, n_y / 2000.0) * min(1.0, known_rate_y / 0.9)

        sex_final = w * float(sex_y) + (1.0 - w) * float(sex_all)
        sex_final = float(max(0.0, min(1.0, sex_final)))
        trad_final = 1.0 - sex_final
        return trad_final, sex_final

    cow_trad, cow_sex = _mix(True)
    hef_trad, hef_sex = _mix(False)

    def _norm2(a: float, b: float) -> Tuple[float, float]:
        s = max(1e-9, a + b)
        return a / s, b / s

    cow_trad, cow_sex = _norm2(cow_trad, cow_sex)
    hef_trad, hef_sex = _norm2(hef_trad, hef_sex)

    return {
        "cow_trad": float(cow_trad),
        "cow_sex": float(cow_sex),
        "heifer_trad": float(hef_trad),
        "heifer_sex": float(hef_sex),
    }


# ------------------------------------------------------------
# Calf sex ratios from DB (trad vs sex)
# ------------------------------------------------------------

def compute_semen_sex_ratios_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, SemenSexRatio]:
    calv = tables["calv"].copy()
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "trad": _to_semen_ratio(SEMEN_SEX_RATIOS["trad"]),
        "sex": _to_semen_ratio(SEMEN_SEX_RATIOS["sex"]),
    }

    if calv.empty or ins.empty or bulls.empty:
        return fallback

    calv["event_type"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce").dt.normalize()
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    calv["sex_norm"] = calv["sex"].apply(norm_sex)

    born = calv[
    (calv["event_type"] == "РОЖДЕН")
    & (calv["event_date"].notna())
    & (calv["mother_reg_s"] != "")
    ][["mother_reg_s", "event_date", "sex_norm"]].copy()

    # оставляем только те строки, где пол распознан
    born = born[born["sex_norm"].isin(["M", "F"])].copy()

    if born.empty:
        return fallback

    born["calving_dt"] = born["event_date"]
    born["male"] = (born["sex_norm"] == "M").astype(int)
    born["female"] = (born["sex_norm"] == "F").astype(int)

    calv_ev = (
        born.groupby(["mother_reg_s", "calving_dt"], sort=False)[["male", "female"]]
        .sum()
        .reset_index()
        .rename(columns={"mother_reg_s": "reg_s"})
    )

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    p = ins[
        (ins["event_date"].notna())
        & (ins["result_norm"] == "P")
        & (ins["reg_s"] != "")
        & (ins["bull_s"] != "")
    ][["reg_s", "event_date", "bull_s"]].copy()
    if p.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["bull_type_s"] = bulls["bull_type"].astype(str).str.strip().str.upper()
    bulls["semen"] = np.where(bulls["bull_type_s"] == "S", "sex", "trad")
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p = p[p["semen"].isin(["trad", "sex"])].copy()
    if p.empty:
        return fallback

    p = p.rename(columns={"event_date": "ins_dt"})

    m = _merge_asof_safe(
        calv_ev.sort_values(["reg_s", "calving_dt"], kind="mergesort"),
        p[["reg_s", "ins_dt", "semen"]].sort_values(["reg_s", "ins_dt"], kind="mergesort"),
        by="reg_s",
        left_on="calving_dt",
        right_on="ins_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    m = m[m["ins_dt"].notna()].copy()
    if m.empty:
        return fallback

    m["gest_days"] = (m["calving_dt"] - m["ins_dt"]).dt.days
    m = m[(m["gest_days"] >= 200) & (m["gest_days"] <= 310)].copy()
    if m.empty:
        return fallback

    out = dict(fallback)
    for semen in ("trad", "sex"):
        sub = m[m["semen"] == semen]
        total = int(sub["male"].sum() + sub["female"].sum())
        if total < 300:
            continue

        bull_share = float(sub["male"].sum()) / float(total)
        bull_share = max(0.0, min(1.0, bull_share))

        # защита: если база даёт почти 0/1 — считаем, что пол в данных распознан плохо,
        # и используем fallback, иначе в модели телки/бычки "вымрут"
        if bull_share < 0.05 or bull_share > 0.95:
            continue

        # clamp чтобы гарантировать ненулевой женский хвост
        bull_share = max(0.10, min(0.90, bull_share))
        out[semen] = SemenSexRatio(bull_share=bull_share, heifer_share=1.0 - bull_share)


    return out


# ------------------------------------------------------------
# Diagnostics helper (kept for UI/debug; does not change model)
# ------------------------------------------------------------

def report_semen_and_calf_sex_params_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Возвращает удобный словарь для UI:
    - доли использования semen (trad/sex)
    - доли пола телят для trad/sex
    - покрытие матчей по быкам (на P-осеменениях)
    """
    semen_shares = compute_semen_usage_from_db(tables)
    semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    return {
        "semen_shares": semen_shares,
        "semen_sex_ratios": semen_sex_ratios,
    }


# ------------------------------------------------------------
# Disposal shapes
# ------------------------------------------------------------

def hazard_from_pdf(pdf: np.ndarray, *, vwp: int = 0) -> np.ndarray:
    p = pdf.copy().astype(float)
    p[:vwp] = 0.0
    s = p.sum()
    if s <= 0:
        return np.zeros_like(p)
    p /= s

    hz = np.zeros_like(p)
    surv = 1.0
    for i in range(len(p)):
        if i < vwp:
            hz[i] = 0.0
            continue
        pi = float(p[i])
        if surv <= 1e-12:
            hz[i] = 0.0
        else:
            hz[i] = pi / surv
            hz[i] = max(0.0, min(1.0, hz[i]))
            surv *= (1.0 - hz[i])
    return hz


def lognormal_hazard_by_dim(dim_max: int, mean: float, median: float) -> np.ndarray:
    mean = float(mean) if mean and mean > 1 else 1.0
    median = float(median) if median and median > 1 else max(1.0, mean * 0.8)
    if mean < median:
        mean = median * 1.05

    sigma2 = 2.0 * np.log(max(1e-9, mean / median))
    sigma = float(np.sqrt(max(1e-9, sigma2)))
    mu = float(np.log(max(1e-9, median)))

    x = np.arange(dim_max + 1, dtype=float)
    pdf = np.zeros_like(x)
    xx = x[1:]
    pdf[1:] = (1.0 / (xx * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((np.log(xx) - mu) ** 2) / (2.0 * sigma2))
    return hazard_from_pdf(pdf, vwp=0)


def build_disposal_shape(disposal_params: dict) -> Dict[int, np.ndarray]:
    shape = {}
    by_lact = disposal_params.get("by_lact", {})
    for lact_cat in (1, 2, 3, 4):
        s = by_lact.get(lact_cat, {})
        m = float(s.get("mean_dim", 150.0) or 150.0)
        md = float(s.get("median_dim", 120.0) or 120.0)
        hz = lognormal_hazard_by_dim(MAX_DIM, m, md)
        nz = hz[hz > 0]
        sh = np.ones(MAX_DIM + 1, dtype=float) if nz.size == 0 else hz / (float(nz.mean()) if float(nz.mean()) > 0 else 1.0)
        shape[lact_cat] = np.clip(sh, 0.1, 5.0)
    return shape


# ------------------------------------------------------------
# Capacity helpers (month-end “реализация”)
# ------------------------------------------------------------

def _cap(name: str) -> float | None:
    v = HERD_CAPACITY.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _take_from_array(arr: np.ndarray, idx_iter: Iterable[int], need: float) -> float:
    taken = 0.0
    for i in idx_iter:
        if need <= 1e-9:
            break
        v = float(arr[i])
        if v <= 0:
            continue
        x = v if v < need else need
        arr[i] = v - x
        taken += x
        need -= x
    return taken


def _sell_cows_from_doy(state: HerdState, need: float, gest_days: int) -> float:
    sold = 0.0
    for l in (4, 3, 2, 1):
        sold += _take_from_array(state.open_dim[l], range(MAX_DIM, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_lact[(l, semen)], range(gest_days, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold
    return sold


def _sell_cows_from_dry(state: HerdState, need: float, gest_days: int, dry_days: int) -> float:
    sold = 0.0
    hi = min(dry_days, gest_days)
    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_dry[(l, semen)], range(hi, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold
    return sold


def _sell_heifers_by_age(state: HerdState, need: float, age_lo: int, age_hi: int) -> float:
    lo = max(0, int(age_lo))
    hi = min(int(age_hi), len(state.heifer_age) - 1)
    return _take_from_array(state.heifer_age, range(hi, lo - 1, -1), need)


def _sell_neteli_4_6_months(state: HerdState, need: float, gest_days: int) -> float:
    sold = 0.0
    pref_lo = max(0, min(gest_days, 100))
    pref_hi = max(0, min(gest_days, 160))

    pref_range = list(range(pref_hi, pref_lo - 1, -1))
    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], pref_range, need - sold)
        if sold >= need - 1e-9:
            return sold

    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(gest_days, pref_hi + 1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(pref_lo - 1, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold
    return sold


def _apply_capacity_month_end(state: HerdState, *, gest_days: int, dry_days: int) -> dict:
    out = {
        "over_doy": 0.0,
        "over_dry": 0.0,
        "over_h0": 0.0,
        "over_h38": 0.0,
        "over_h9": 0.0,
        "over_neteli": 0.0,
        "sell_cows": 0.0,
        "sell_heifers": 0.0,
        "sell_neteli": 0.0,
    }

    cap_doy = _cap("Дойные коровы")
    cap_dry = _cap("Сухостойные коровы")
    cap_h0 = _cap("Тёлки 0–3 мес")          # в UI мапится на 0–2
    cap_h38 = _cap("Тёлки 3–8 мес")
    cap_h9 = _cap("Тёлки 9–24 мес")
    cap_neteli = _cap("Нетели")

    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    h0 = float(state.heifer_age[:90].sum())          # 0–2 мес
    h38 = float(state.heifer_age[90:270].sum())      # 3–8 мес
    h9 = float(state.heifer_age[270:].sum())         # >=9 мес
    neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

    if cap_doy is not None and doy > cap_doy + 1e-9:
        need = doy - cap_doy
        sold = _sell_cows_from_doy(state, need, gest_days)
        out["over_doy"] = sold
        out["sell_cows"] += sold

    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    doy = float(cows_open + cows_preg_lact)

    if cap_dry is not None and dry > cap_dry + 1e-9:
        need = dry - cap_dry
        sold = _sell_cows_from_dry(state, need, gest_days, dry_days)
        out["over_dry"] = sold
        out["sell_cows"] += sold

    if cap_h0 is not None and h0 > cap_h0 + 1e-9:
        need = h0 - cap_h0
        sold = _sell_heifers_by_age(state, need, 0, 89)
        out["over_h0"] = sold
        out["sell_heifers"] += sold

    if cap_h38 is not None and h38 > cap_h38 + 1e-9:
        need = h38 - cap_h38
        sold = _sell_heifers_by_age(state, need, 90, 269)
        out["over_h38"] = sold
        out["sell_heifers"] += sold

    if cap_h9 is not None and h9 > cap_h9 + 1e-9:
        need = h9 - cap_h9
        sold = _sell_heifers_by_age(state, need, 270, MAX_AGE_DAYS)
        out["over_h9"] = sold
        out["sell_heifers"] += sold

    if cap_neteli is not None and neteli > cap_neteli + 1e-9:
        need = neteli - cap_neteli
        sold = _sell_neteli_4_6_months(state, need, gest_days)
        out["over_neteli"] = sold
        out["sell_neteli"] += sold

    return out


# ------------------------------------------------------------
# Utility: cow lact category
# ------------------------------------------------------------

def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4


# ------------------------------------------------------------
# Initial state (NEW, robust youngstock seeding)
# ------------------------------------------------------------

def _build_cow_like_regs(
    *,
    calv: pd.DataFrame,
    ins: pd.DataFrame,
    dry: pd.DataFrame,
    cows_regs: set[str],
) -> set[str]:
    """
    Список регов, которые С БОЛЬШОЙ вероятностью коровы, даже если lact в inseminations пустой/0.
    Это критично, чтобы не записывать коров в "нетели/тёлки" и не раздувать молодняк.
    """
    out = set(str(x) for x in cows_regs if str(x))

    # 1) все, у кого когда-либо lact>0
    if not ins.empty:
        tmp = ins.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["lact_i"] = pd.to_numeric(tmp["lact"], errors="coerce").fillna(0).astype(int)
        out |= set(tmp.loc[tmp["lact_i"] > 0, "reg_s"].astype(str))

    # 2) все, у кого есть dryoff
    if not dry.empty:
        tmp = dry.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        out |= set(tmp["reg_s"].astype(str))

    # 3) все, кто встречается как мать (mother_reg) или как reg в событии "ОТЕЛ"
    if not calv.empty:
        tmp = calv.copy()
        tmp["event_type_n"] = tmp["event_type"].apply(norm_event_type)
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["mother_reg_s"] = tmp["mother_reg"].apply(norm_id)
        out |= set(tmp.loc[tmp["mother_reg_s"] != "", "mother_reg_s"].astype(str))
        out |= set(tmp.loc[(tmp["event_type_n"] == "ОТЕЛ") & (tmp["reg_s"] != ""), "reg_s"].astype(str))

    out.discard("")
    return out


def _infer_semen_for_calvings(
    calv_ev: pd.DataFrame,
    *,
    ins: pd.DataFrame,
    semen_by_bull: dict[str, str],
    gest_days: int,
) -> pd.DataFrame:
    """
    Для каждого (cow_reg_s, calving_dt) находим последнее P-осеменение перед отёлом,
    проверяем окно гестации и получаем semen ('trad'/'sex'). Если не нашли — 'trad'.
    """
    if calv_ev.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    ins2 = ins.copy()
    if ins2.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    ins2["event_date"] = pd.to_datetime(ins2["event_date"], errors="coerce").dt.normalize()
    ins2["result_norm"] = ins2["result"].apply(norm_result)
    ins2["reg_s"] = ins2["reg"].apply(norm_id)
    ins2["bull_s"] = ins2["bull"].apply(norm_id)

    p = ins2[(ins2["event_date"].notna()) & (ins2["result_norm"] == "P") & (ins2["reg_s"] != "")].copy()
    if p.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p.loc[~p["semen"].isin(["trad", "sex"]), "semen"] = "trad"
    p = p.rename(columns={"event_date": "ins_dt"})

    left = calv_ev.sort_values(["cow_reg_s", "calving_dt"], kind="mergesort").copy()
    right = p[["reg_s", "ins_dt", "semen"]].sort_values(["reg_s", "ins_dt"], kind="mergesort")
    left = left.rename(columns={"cow_reg_s": "reg_s"})

    m = _merge_asof_safe(
        left,
        right,
        by="reg_s",
        left_on="calving_dt",
        right_on="ins_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    # окно гестации
    m["gest_d"] = (m["calving_dt"] - m["ins_dt"]).dt.days
    ok = (m["ins_dt"].notna()) & (m["gest_d"] >= 200) & (m["gest_d"] <= 310)
    m.loc[~ok, "semen"] = "trad"

    m = m.drop(columns=["reg_s"])
    m = m.rename(columns={"reg_s_r": "cow_reg_s"}) if "reg_s_r" in m.columns else m
    return m


def _seed_youngstock_from_calvings(
    *,
    state: HerdState,
    calv: pd.DataFrame,
    ins: pd.DataFrame,
    semen_by_bull: Dict[str, str],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    disposed_regs: set[str],
    as_of_ts: pd.Timestamp,
    gest_days: int,
    cow_like_regs: set[str] | None = None,
) -> None:
    cow_like_regs = cow_like_regs or set()

    calv2 = calv.copy()
    calv2["event_type_n"] = calv2["event_type"].apply(norm_event_type)
    calv2["event_date_n"] = pd.to_datetime(calv2["event_date"], errors="coerce").dt.normalize()
    calv2["reg_s"] = calv2["reg"].apply(norm_id)
    calv2["mother_reg_s"] = calv2["mother_reg"].apply(norm_id)
    calv2["gndr_n"] = calv2.get("gndr", "").apply(norm_gender)

    calv2 = calv2[
        (calv2["event_type_n"] == "РОЖДЕН")
        & (calv2["event_date_n"].notna())
        & (calv2["event_date_n"] <= as_of_ts)
        & (calv2["reg_s"] != "")
    ].copy()

    if calv2.empty:
        return

    ins2 = ins.copy()
    ins2["event_date"] = pd.to_datetime(ins2["event_date"], errors="coerce").dt.normalize()
    ins2["reg_s"] = ins2["reg"].apply(norm_id)
    ins2["bull_s"] = ins2["bull"].apply(norm_id)
    ins2 = ins2[(ins2["event_date"].notna()) & (ins2["event_date"] <= as_of_ts) & (ins2["reg_s"] != "")].copy()

    bull_by_mother_date: dict[tuple[str, pd.Timestamp], str] = {}
    if not ins2.empty:
        ins2 = ins2.sort_values(["reg_s", "event_date"], kind="mergesort")
        tail = ins2.groupby(["reg_s", "event_date"], sort=False).tail(1)
        bull_by_mother_date = dict(zip(zip(tail["reg_s"], tail["event_date"]), tail["bull_s"]))

    for rr in calv2.itertuples(index=False):
        calf_reg = str(rr.reg_s)
        if not calf_reg or calf_reg in disposed_regs:
            continue

        # ключевой фикс: если этот reg уже "похож на корову" — НЕ считаем его тёлкой
        if calf_reg in cow_like_regs:
            continue

        born_dt = rr.event_date_n
        if pd.isna(born_dt):
            continue

        age = int((as_of_ts - pd.Timestamp(born_dt)).days)
        if age < 0 or age > MAX_AGE_DAYS:
            continue

        mother = str(rr.mother_reg_s) if hasattr(rr, "mother_reg_s") else ""
        bull = bull_by_mother_date.get((mother, pd.Timestamp(born_dt).normalize()), "") if mother else ""
        semen = semen_by_bull.get(bull, "trad") if bull else "trad"
        ratio = semen_sex_ratios.get(semen, semen_sex_ratios["trad"])

        g = str(rr.gndr_n) if hasattr(rr, "gndr_n") else ""
        if g == "F":
            state.heifer_age[age] += 1.0
        elif g == "M":
            state.bull_age[age] += 1.0
        else:
            state.heifer_age[age] += float(ratio.heifer_share)
            state.bull_age[age] += float(ratio.bull_share)

def build_initial_state(
    tables: Dict[str, pd.DataFrame],
    as_of: date,
    *,
    gest_days: int | None = None,
    dry_days: int | None = None,
    insemination_params: dict | None = None,
    warmstart_from_services: bool = True,
    semen_sex_ratios: Dict[str, SemenSexRatio] | None = None,
) -> HerdState:
    """
    Собираем агрегированное состояние стада на дату as_of.

    Главное исправление "нулей в тёлках" без раздувания:
      1) Молодняк/тёлки до ~18 мес сеем из calvings_births (РОЖДЕН) + fallback по отёлу матери.
      2) Старше ~18 мес НЕ пытаемся восстановить по древним рождениям (иначе раздувает).
      3) Исключаем cow_like_regs и тёлок-нетелей (P) из возрастного посева, чтобы не было double count.
      4) ins-only подсев тёлок включаем ТОЛЬКО если в calvings почти нет строк телят.
    """
    gest_days = int(gest_days if gest_days is not None else int(GESTATION_DAYS))
    dry_days = int(dry_days if dry_days is not None else int(DRY_DAYS))
    as_of_ts = pd.Timestamp(as_of).normalize()

    state = init_empty_state(gest_days)

    calv = tables["calv"].copy()
    ins = tables["ins"].copy()
    dry = tables["dry"].copy()
    disp = tables["disp"].copy()
    bulls = tables["bulls"].copy()

    # --- normalize inseminations ---
    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    # --- normalize dryoff ---
    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce").dt.normalize()
    dry["reg_s"] = dry["reg"].apply(norm_id)

    # --- normalize disposals ---
    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.normalize()
    disp["reg_s"] = disp["reg"].apply(norm_id)

    # --- bulls semen map ---
    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    # --- disposed up to as_of ---
    disposed_regs = set(
        disp.loc[
            disp["event_date"].notna() & (disp["event_date"] <= as_of_ts),
            "reg_s"
        ].astype(str)
    )
    disposed_regs.discard("")

    # --- last dryoff ---
    dry_ok = dry[(dry["event_date"].notna()) & (dry["event_date"] <= as_of_ts) & (dry["reg_s"] != "")]
    dry_last = dry_ok.groupby("reg_s", sort=False)["event_date"].max().to_dict()

    # --- calving stats for cows (from calves rows: mother_reg + event_date) ---
    calv2 = calv.copy()
    calv2["event_type_n"] = calv2["event_type"].apply(norm_event_type)
    calv2["event_date_n"] = pd.to_datetime(calv2["event_date"], errors="coerce").dt.normalize()
    calv2["reg_s"] = calv2["reg"].apply(norm_id)
    calv2["mother_reg_s"] = calv2["mother_reg"].apply(norm_id)
    calv2 = calv2[(calv2["event_date_n"].notna()) & (calv2["event_date_n"] <= as_of_ts)].copy()

    calves = calv2[
        (calv2["event_type_n"] == "РОЖДЕН")
        & (calv2["mother_reg_s"] != "")
        & (calv2["event_date_n"].notna())
    ][["mother_reg_s", "event_date_n"]].drop_duplicates()

    calv_stats = None
    if not calves.empty:
        calv_stats = (
            calves.rename(columns={"mother_reg_s": "reg_s", "event_date_n": "calving_date"})
            .groupby("reg_s", sort=False)
            .agg(
                n_calvings=("calving_date", "count"),
                last_calving=("calving_date", "max"),
            )
            .reset_index()
        )

    # --- восстановление last_calving из inseminations (DIM) для коров lact>0 ---
    ins_cow_dim = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] > 0)
        & (ins["dim_age"].notna())
        & (ins["dim_age"] >= 0)
        & (ins["dim_age"] <= MAX_DIM)
    ].copy()

    est_stats = None
    if not ins_cow_dim.empty:
        ins_cow_dim = ins_cow_dim.sort_values(["reg_s", "event_date"], kind="mergesort")
        last_dim_row = ins_cow_dim.groupby("reg_s", sort=False).tail(1).copy()
        last_dim_row["last_calving_est"] = last_dim_row["event_date"] - pd.to_timedelta(last_dim_row["dim_age"], unit="D")
        last_dim_row["lact_cat_est"] = last_dim_row["lact"].clip(lower=1, upper=4)
        est_stats = last_dim_row[["reg_s", "last_calving_est", "lact_cat_est", "dim_age"]].copy()

    # --- собрать список коров (union) ---
    if calv_stats is None and est_stats is None:
        cows = pd.DataFrame(columns=["reg_s", "last_calving", "n_calvings", "last_calving_est", "lact_cat_est", "dim_age"])
    elif calv_stats is None:
        cows = est_stats.copy()
        cows["n_calvings"] = pd.NA
        cows["last_calving"] = pd.NA
    elif est_stats is None:
        cows = calv_stats.copy()
        cows["last_calving_est"] = pd.NA
        cows["lact_cat_est"] = pd.NA
        cows["dim_age"] = pd.NA
    else:
        cows = calv_stats.merge(est_stats, on="reg_s", how="outer")

    cows = cows[(cows["reg_s"].notna()) & (cows["reg_s"] != "")].copy()
    cows = cows[~cows["reg_s"].isin(disposed_regs)].copy()

    cows["last_calving"] = cows["last_calving"].where(cows["last_calving"].notna(), cows["last_calving_est"])

    def _lcat(row) -> int:
        if pd.notna(row.get("n_calvings")):
            return lact_cat_from_count(int(row["n_calvings"]))
        if pd.notna(row.get("lact_cat_est")):
            return int(row["lact_cat_est"])
        return 1

    cows["lact_cat"] = cows.apply(_lcat, axis=1)
    cows_regs = set(cows["reg_s"].astype(str).tolist())
    cow_like_regs = _build_cow_like_regs(calv=calv2, ins=ins, dry=dry, cows_regs=cows_regs)

    # --- последние P-осеменения ---
    ins_p = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["result_norm"] == "P")
    ].copy()

    last_p: dict[str, pd.Timestamp] = {}
    last_p_bull: dict[str, str] = {}
    if not ins_p.empty:
        ins_p = ins_p.sort_values(["reg_s", "event_date"], kind="mergesort")
        tail = ins_p.groupby("reg_s", sort=False).tail(1)
        last_p = dict(zip(tail["reg_s"], tail["event_date"]))
        last_p_bull = dict(zip(tail["reg_s"], tail["bull_s"]))

    # --- warmstart параметры ---
    ins_params = insemination_params or {}
    cow_spc = float(ins_params.get("cow_services_per_conception", float(INSEMINATION_PARAMS.cow_services_per_conception)))
    heif_spc = float(ins_params.get("heifer_services_per_conception", float(INSEMINATION_PARAMS.heifer_services_per_conception)))

    p_conc_cow = _clamp(1.0 / max(1e-9, cow_spc), 0.05, 0.95)

    # IMPORTANT: тёлок warmstart лучше делать аккуратнее, иначе резко растут "нетели"
    p_conc_heif_raw = 1.0 / max(1e-9, heif_spc)
    p_conc_heif = _clamp(float(ins_params.get("heifer_warmstart_p", p_conc_heif_raw)), 0.03, 0.35)

    SERVICE_RESULTS = {"", "O", "О"}

    # --- последние сервисы (O/пусто) ---
    last_service_cow: dict[str, pd.Timestamp] = {}
    last_service_bull_cow: dict[str, str] = {}
    last_service_heif: dict[str, pd.Timestamp] = {}
    last_service_bull_heif: dict[str, str] = {}

    if warmstart_from_services:
        ins_svc = ins[
            (ins["event_date"].notna())
            & (ins["event_date"] <= as_of_ts)
            & (ins["reg_s"] != "")
            & (ins["result_norm"].isin(SERVICE_RESULTS))
        ].copy()

        ins_svc = ins_svc[~ins_svc["reg_s"].isin(disposed_regs)]

        if not ins_svc.empty:
            ins_svc = ins_svc.sort_values(["reg_s", "event_date"], kind="mergesort")
            last = ins_svc.groupby("reg_s", sort=False).tail(1)

            cow_last = last[(last["lact"] > 0) | (last["reg_s"].isin(cow_like_regs))]
            heif_last = last[(last["lact"] <= 0) & (~last["reg_s"].isin(cow_like_regs))]

            last_service_cow = dict(zip(cow_last["reg_s"], cow_last["event_date"]))
            last_service_bull_cow = dict(zip(cow_last["reg_s"], cow_last["bull_s"]))

            last_service_heif = dict(zip(heif_last["reg_s"], heif_last["event_date"]))
            last_service_bull_heif = dict(zip(heif_last["reg_s"], heif_last["bull_s"]))

    # --- расклад коров в состояние ---
    for r in cows.itertuples(index=False):
        reg = str(r.reg_s)
        lact_cat = int(r.lact_cat)

        last_calv = getattr(r, "last_calving", pd.NaT)
        dim_guess = getattr(r, "dim_age", pd.NA)

        if pd.notna(last_calv):
            dim = int(max(0, min(MAX_DIM, (as_of_ts - pd.Timestamp(last_calv).normalize()).days)))
        else:
            dim = int(max(0, min(MAX_DIM, float(dim_guess))) if pd.notna(dim_guess) else 0)

        dry_last_dt = dry_last.get(reg, pd.NaT)
        is_dry_fact = (
            pd.notna(dry_last_dt)
            and pd.notna(last_calv)
            and (pd.Timestamp(dry_last_dt) > pd.Timestamp(last_calv))
        )

        open_add = 1.0
        placed_preg = False

        # 1) подтверждённая стельность P
        p_date = last_p.get(reg, pd.NaT)
        bull = last_p_bull.get(reg, "") or ""
        semen = semen_by_bull.get(bull, "trad") if bull else "trad"

        if pd.notna(p_date):
            p_date = pd.Timestamp(p_date).normalize()
            if not (pd.notna(last_calv) and p_date <= pd.Timestamp(last_calv).normalize()):
                days_to_calv = int(gest_days - (as_of_ts - p_date).days)
                if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                    days_to_calv = 0
                if 0 <= days_to_calv <= gest_days:
                    is_dry = is_dry_fact or (days_to_calv <= dry_days)
                    if is_dry:
                        state.preg_dry[(lact_cat, semen)][days_to_calv] += 1.0
                    else:
                        state.preg_lact[(lact_cat, semen)][days_to_calv] += 1.0
                    open_add = 0.0
                    placed_preg = True

        # 2) warmstart от "последнего осеменения" (O/пусто)
        if warmstart_from_services and (not placed_preg):
            s_date = last_service_cow.get(reg, pd.NaT)
            if pd.notna(s_date):
                s_date = pd.Timestamp(s_date).normalize()
                if not (pd.notna(last_calv) and s_date <= pd.Timestamp(last_calv).normalize()):
                    bull2 = last_service_bull_cow.get(reg, "") or ""
                    semen2 = semen_by_bull.get(bull2, "trad") if bull2 else "trad"

                    days_to_calv2 = int(gest_days - (as_of_ts - s_date).days)
                    if days_to_calv2 < 0 and days_to_calv2 >= -OVERDUE_CLAMP_DAYS:
                        days_to_calv2 = 0

                    if 0 <= days_to_calv2 <= gest_days:
                        add = float(p_conc_cow)
                        open_add = 1.0 - add
                        is_dry2 = is_dry_fact or (days_to_calv2 <= dry_days)
                        if is_dry2:
                            state.preg_dry[(lact_cat, semen2)][days_to_calv2] += add
                        else:
                            state.preg_lact[(lact_cat, semen2)][days_to_calv2] += add

        state.open_dim[lact_cat][dim] += open_add

    # ============================================================
    # 1) СНАЧАЛА считаем НЕТЕЛЕЙ (heifer_preg) по inseminations,
    #    чтобы потом НЕ посеять их второй раз как "тёлок по возрасту"
    # ============================================================

    heifer_p = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] <= 0)
        & (ins["result_norm"] == "P")
        & (~ins["reg_s"].isin(cow_like_regs))
        & (~ins["reg_s"].isin(disposed_regs))
    ].copy()

    p_regs = set()
    heifer_last = None
    if not heifer_p.empty:
        heifer_p = heifer_p.sort_values(["reg_s", "event_date"], kind="mergesort")
        heifer_last = heifer_p.groupby("reg_s", sort=False).tail(1)
        p_regs = set(heifer_last["reg_s"].astype(str).tolist())

        for rr in heifer_last.itertuples(index=False):
            p_date = rr.event_date
            if pd.isna(p_date):
                continue
            bull = getattr(rr, "bull_s", "") or ""
            semen = semen_by_bull.get(bull, "trad") if bull else "trad"

            p_date = pd.Timestamp(p_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - p_date).days)
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0
            if 0 <= days_to_calv <= gest_days:
                state.heifer_preg[semen][days_to_calv] += 1.0

    # warmstart для тёлок по сервису (если P нет)
    if warmstart_from_services and last_service_heif:
        for reg, s_date in last_service_heif.items():
            if reg in disposed_regs or reg in cow_like_regs or reg in p_regs:
                continue
            if pd.isna(s_date):
                continue
            bull = last_service_bull_heif.get(reg, "") or ""
            semen = semen_by_bull.get(bull, "trad") if bull else "trad"

            s_date = pd.Timestamp(s_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - s_date).days)
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0
            if not (0 <= days_to_calv <= gest_days):
                continue

            add = float(p_conc_heif)
            state.heifer_preg[semen][days_to_calv] += add

    # ============================================================
    # 2) Теперь СЕЕМ молодняк/тёлок по рождениям/отёлам,
    #    но ТОЛЬКО за последние ~18 месяцев (иначе раздувает).
    #    И НЕ сеем тех, кто уже "нетель по P" или cow_like.
    # ============================================================

    if semen_sex_ratios is None:
        semen_sex_ratios = {
            "trad": _to_semen_ratio(SEMEN_SEX_RATIOS["trad"]),
            "sex": _to_semen_ratio(SEMEN_SEX_RATIOS["sex"]),
        }

    seed_days = int(float(ins_params.get("youngstock_seed_days", 540) or 540))
    seed_days = max(120, min(seed_days, int(MAX_AGE_DAYS)))  # разумные границы

    calv_seed = calv.copy()
    calv_seed["event_date_n"] = pd.to_datetime(calv_seed["event_date"], errors="coerce").dt.normalize()
    calv_seed = calv_seed[
        (calv_seed["event_date_n"].notna())
        & (calv_seed["event_date_n"] <= as_of_ts)
        & (calv_seed["event_date_n"] >= (as_of_ts - pd.Timedelta(days=seed_days)))
    ].copy()

    # исключаем рег.номера тёлок, которые уже учтены как нетели (P),
    # и любые cow_like (на всякий)
    calv_seed["reg_s"] = calv_seed["reg"].apply(norm_id)
    calv_seed = calv_seed[~calv_seed["reg_s"].isin(p_regs)]
    calv_seed = calv_seed[~calv_seed["reg_s"].isin(cow_like_regs)]

    _seed_youngstock_from_calvings(
        state=state,
        calv=calv_seed,
        ins=ins,
        semen_by_bull=semen_by_bull,
        semen_sex_ratios=semen_sex_ratios,
        disposed_regs=disposed_regs,
        as_of_ts=as_of_ts,
        gest_days=gest_days,
        cow_like_regs=(cow_like_regs | p_regs),
    )

    # ============================================================
    # 3) ins-only OPEN heifers — только если calvings почти пустой
    #    (иначе это почти всегда double count)
    # ============================================================

    born_known = 0
    if not calv_seed.empty:
        tmp = calv_seed.copy()
        tmp["event_type_n"] = tmp["event_type"].apply(norm_event_type)
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        born_known = int(((tmp["event_type_n"] == "РОЖДЕН") & (tmp["reg_s"] != "")).sum())

    if born_known < 10:
        heif_any = ins[
            (ins["event_date"].notna())
            & (ins["event_date"] <= as_of_ts)
            & (ins["reg_s"] != "")
            & (ins["lact"] <= 0)
            & (~ins["reg_s"].isin(cow_like_regs))
            & (~ins["reg_s"].isin(disposed_regs))
            & (~ins["reg_s"].isin(p_regs))
        ].copy()

        if not heif_any.empty:
            heif_any = heif_any.sort_values(["reg_s", "event_date"], kind="mergesort")
            last = heif_any.groupby("reg_s", sort=False).tail(1)

            for rr in last.itertuples(index=False):
                age_val = getattr(rr, "dim_age", np.nan)
                if pd.isna(age_val):
                    continue
                age_val = float(age_val)

                # фильтр: dim_age как ВОЗРАСТ, а не DIM
                if age_val < 150 or age_val > MAX_AGE_DAYS:
                    continue

                age = int(max(0, min(MAX_AGE_DAYS, int(age_val))))
                state.heifer_age[age] += 1.0

    return state

import pandas as pd
import numpy as np

def build_early_realization_plan(
    df: pd.DataFrame,
    *,
    lead_neteli_months: int = 2,
    lead_heifer9_months: int = 4,
) -> pd.DataFrame:
    cols = list(df.columns)

    def row(name: str) -> pd.Series:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=cols)

    over_doy = row("Переполнение: Дойные коровы")
    calv_from_neteli = row("Ожидаемый отёл, из них нетелей")

    stock_neteli = row("Нетели")
    stock_h9 = row("Тёлки ≥9 мес")

    plan_neteli = pd.Series(0.0, index=cols)
    plan_h9 = pd.Series(0.0, index=cols)
    plan_cows = pd.Series(0.0, index=cols)

    for i, m in enumerate(cols):
        need = float(over_doy[m])
        if need <= 0:
            continue

        # 1) заранее продаём нетелей (за lead_neteli_months)
        j = i - lead_neteli_months
        if j >= 0 and need > 0:
            m_sell = cols[j]

            cap_by_flow = float(calv_from_neteli[m])  # сколько нетелей "войдёт" в коровы в m
            cap_by_stock = max(0.0, float(stock_neteli[m_sell]) - float(plan_neteli[m_sell]))

            add = min(need, cap_by_flow, cap_by_stock)
            if add > 0:
                plan_neteli[m_sell] += add
                need -= add

        # 2) если не хватило — заранее продаём тёлок ≥9 мес (ещё раньше)
        k = i - lead_heifer9_months
        if k >= 0 and need > 0:
            m_sell = cols[k]

            cap_by_stock = max(0.0, float(stock_h9[m_sell]) - float(plan_h9[m_sell]))
            add = min(need, cap_by_stock)
            if add > 0:
                plan_h9[m_sell] += add
                need -= add

        # 3) остаток — продаём коров в том же месяце (как fallback)
        if need > 0:
            plan_cows[m] += need

    out = pd.DataFrame(index=[
        "План реализации (ранний): нетели",
        "План реализации (ранний): тёлки ≥9 мес",
        "План реализации (ранний): коровы",
    ], columns=cols)

    out.loc["План реализации (ранний): нетели"] = plan_neteli.round(1)
    out.loc["План реализации (ранний): тёлки ≥9 мес"] = plan_h9.round(1)
    out.loc["План реализации (ранний): коровы"] = plan_cows.round(1)

    return out

# Legacy placeholder (не удаляем без явного разрешения пользователя)
def build_initial_state_legacy(*args, **kwargs) -> HerdState:
    """Legacy версия была в предыдущих ревизиях. Оставляем заглушку-алиас на новую."""
    return build_initial_state(*args, **kwargs)


# ------------------------------------------------------------
# Simulation to target month
# ------------------------------------------------------------

def simulate_to_target(
    state: HerdState,
    *,
    start: date,
    target: date,
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    params: dict,
) -> Tuple[HerdState, Dict[str, float]]:

    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])
    cp = params["CONCEPTION_PARAMS"]
    disp_params = params["DISPOSAL_PARAMS"]
    annual_disp = float(params["ANNUAL_DISPOSAL_RATE"])
    ins_p = params["INSEMINATION_PARAMS"]

    target_month = (target.year, target.month)
    calv_total = 0.0
    calv_cows = 0.0
    calv_heifers = 0.0
    exp_bulls = 0.0
    exp_heifers = 0.0

    meta: Dict[str, float] = {
        "cow_doses_total": 0.0, "cow_doses_sex": 0.0, "cow_doses_trad": 0.0,
        "heifer_doses_total": 0.0, "heifer_doses_sex": 0.0, "heifer_doses_trad": 0.0,
        "sell_cows": 0.0, "sell_heifers": 0.0, "sell_neteli": 0.0,
        "over_doy": 0.0, "over_dry": 0.0, "over_h0": 0.0, "over_h38": 0.0, "over_h9": 0.0, "over_neteli": 0.0,
    }

    # --- обработать "отёлы уже в bucket=0" на дате start ---
    def _process_bucket0_for_day(curr_day: date) -> None:
        nonlocal calv_total, calv_cows, calv_heifers, exp_bulls, exp_heifers

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                born = float(state.preg_dry[(l, semen)][0])
                if born > 0:
                    state.preg_dry[(l, semen)][0] = 0.0

                    if (curr_day.year, curr_day.month) == target_month:
                        calv_total += born
                        calv_cows += born
                        sr = semen_sex_ratios[semen]
                        exp_bulls += born * float(sr.bull_share)
                        exp_heifers += born * float(sr.heifer_share)

                    l2 = min(4, l + 1)
                    state.open_dim[l2][0] += born

                    sr = semen_sex_ratios[semen]
                    state.heifer_age[0] += born * float(sr.heifer_share)
                    state.bull_age[0] += born * float(sr.bull_share)

        for semen in ("trad", "sex"):
            born = float(state.heifer_preg[semen][0])
            if born > 0:
                state.heifer_preg[semen][0] = 0.0

                if (curr_day.year, curr_day.month) == target_month:
                    calv_total += born
                    calv_heifers += born
                    sr = semen_sex_ratios[semen]
                    exp_bulls += born * float(sr.bull_share)
                    exp_heifers += born * float(sr.heifer_share)

                state.open_dim[1][0] += born
                sr = semen_sex_ratios[semen]
                state.heifer_age[0] += born * float(sr.heifer_share)
                state.bull_age[0] += born * float(sr.bull_share)

    _process_bucket0_for_day(start)

    # disposal
    p_disp_day_base = 1.0 - (1.0 - annual_disp) ** (1.0 / 365.0)
    disp_shape = build_disposal_shape(disp_params)

    by_lact = disp_params.get("by_lact", {})
    total_n = float(disp_params.get("overall", {}).get("n", 1) or 1)
    shares = {l: (float(by_lact.get(l, {}).get("n", 0) or 0) / total_n) for l in (1, 2, 3, 4)}
    avg_share = sum(shares.values()) / 4.0 if sum(shares.values()) > 0 else 1.0
    w = {l: (shares.get(l, avg_share) / avg_share) for l in (1, 2, 3, 4)}

    cow_trad_share = float(semen_shares["cow_trad"])
    cow_sex_share = float(semen_shares["cow_sex"])
    heif_trad_share = float(semen_shares["heifer_trad"])
    heif_sex_share = float(semen_shares["heifer_sex"])

    snapshot: HerdState | None = None
    if target <= start:
        snapshot = _copy_state(state)

    end_sim = end_of_month(target)
    day = start
    idx_dry = min(dry_days, gest_days)

    while day < end_sim:
        day = day + timedelta(days=1)

        # 1) aging
        for l in (1, 2, 3, 4):
            state.open_dim[l] = shift_right(state.open_dim[l])
        state.heifer_age = shift_right(state.heifer_age)
        state.bull_age = shift_right(state.bull_age)

        # 2) countdown pregnancy
        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                state.preg_lact[(l, semen)] = shift_left(state.preg_lact[(l, semen)])
                state.preg_dry[(l, semen)] = shift_left(state.preg_dry[(l, semen)])
        for semen in ("trad", "sex"):
            state.heifer_preg[semen] = shift_left(state.heifer_preg[semen])

        # 3) auto dryoff (перевод в сухостой dry_days до отёла)
        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                move = float(state.preg_lact[(l, semen)][idx_dry])
                if move > 0:
                    state.preg_lact[(l, semen)][idx_dry] = 0.0
                    state.preg_dry[(l, semen)][idx_dry] += move

        # 4) calvings (bucket 0)
        _process_bucket0_for_day(day)

        # 5) services -> conceptions (коровы)
        for l in (1, 2, 3, 4):
            first_ai = float(ins_p["cow_first_ai_dim_by_lact"].get(l, 70.0))
            spc = float(ins_p["cow_services_per_conception"])
            interval_raw = float(ins_p["cow_ai_interval_days"])
            mean_target = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))

            interval = _effective_ai_interval_days(interval_raw, mean_target, first_ai, spc)
            p_service = 1.0 / max(1.0, interval)
            p_conc = _clamp(1.0 / max(1e-9, spc), 0.05, 0.95)

            open_arr = state.open_dim[l]
            first_i = int(_clamp(first_ai, 0.0, float(MAX_DIM)))
            if first_i >= len(open_arr):
                continue

            eligible = open_arr.copy()
            eligible[:first_i] = 0.0

            services_by_dim = eligible * p_service
            services_total = float(services_by_dim.sum())
            if services_total <= 0:
                continue

            conceived_by_dim = services_by_dim * p_conc
            conceived_total = float(conceived_by_dim.sum())
            if conceived_total <= 0:
                continue

            state.open_dim[l] = np.maximum(0.0, open_arr - conceived_by_dim)

            # распределение по semen
            state.preg_lact[(l, "sex")][gest_days] += services_total * cow_sex_share * p_conc
            state.preg_lact[(l, "trad")][gest_days] += services_total * cow_trad_share * p_conc

            if (day.year, day.month) == target_month:
                meta["cow_doses_total"] += services_total
                meta["cow_doses_sex"] += services_total * cow_sex_share
                meta["cow_doses_trad"] += services_total * cow_trad_share

        # 5b) services -> conceptions (тёлки)
        first_ai_age = float(ins_p["heifer_first_ai_age_days"])
        spc_h = float(ins_p["heifer_services_per_conception"])
        interval_raw_h = float(ins_p["heifer_ai_interval_days"])
        mean_target_h = float(cp["avg_heifer_age_days"])

        interval_h = _effective_ai_interval_days(interval_raw_h, mean_target_h, first_ai_age, spc_h)
        p_service_h = 1.0 / max(1.0, interval_h)
        p_conc_h = _clamp(1.0 / max(1e-9, spc_h), 0.05, 0.95)

        first_h = int(_clamp(first_ai_age, 0.0, float(MAX_AGE_DAYS)))
        if first_h < len(state.heifer_age):
            eligible_h = state.heifer_age.copy()
            eligible_h[:first_h] = 0.0

            services_by_age = eligible_h * p_service_h
            services_total_h = float(services_by_age.sum())
            if services_total_h > 0:
                conceived_by_age = services_by_age * p_conc_h
                conceived_total_h = float(conceived_by_age.sum())
                if conceived_total_h > 0:
                    state.heifer_age = np.maximum(0.0, state.heifer_age - conceived_by_age)
                    state.heifer_preg["sex"][gest_days] += services_total_h * heif_sex_share * p_conc_h
                    state.heifer_preg["trad"][gest_days] += services_total_h * heif_trad_share * p_conc_h

                if (day.year, day.month) == target_month:
                    meta["heifer_doses_total"] += services_total_h
                    meta["heifer_doses_sex"] += services_total_h * heif_sex_share
                    meta["heifer_doses_trad"] += services_total_h * heif_trad_share

        # 6) disposal (коровы)
        for l in (1, 2, 3, 4):
            base = float(p_disp_day_base * w[l])
            base = max(0.0, min(0.02, base))

            haz_open = np.clip(base * disp_shape[l], 0.0, 0.05)
            state.open_dim[l] *= (1.0 - haz_open)

            mean_conc = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
            conc0 = int(round(mean_conc))
            idx = np.arange(gest_days + 1, dtype=int)
            gest_age = (gest_days - idx).astype(int)
            est_dim = np.clip(conc0 + gest_age, 0, MAX_DIM)

            haz_preg = np.clip(base * disp_shape[l][est_dim], 0.0, 0.05)
            for semen in ("trad", "sex"):
                state.preg_lact[(l, semen)] *= (1.0 - haz_preg)
                state.preg_dry[(l, semen)] *= (1.0 - haz_preg)

        # 7) month-end capacity -> "реализация"
        if day == end_of_month(day):
            sold = _apply_capacity_month_end(state, gest_days=gest_days, dry_days=dry_days)
            if (day.year, day.month) == target_month:
                meta["sell_cows"] += float(sold["sell_cows"])
                meta["sell_heifers"] += float(sold["sell_heifers"])
                meta["sell_neteli"] += float(sold["sell_neteli"])
                meta["over_doy"] += float(sold["over_doy"])
                meta["over_dry"] += float(sold["over_dry"])
                meta["over_h0"] += float(sold["over_h0"])
                meta["over_h38"] += float(sold["over_h38"])
                meta["over_h9"] += float(sold["over_h9"])
                meta["over_neteli"] += float(sold["over_neteli"])

        if day == target:
            snapshot = _copy_state(state)

    if snapshot is None:
        snapshot = _copy_state(state)

    meta.update({
        "calv_total": float(calv_total),
        "calv_cows": float(calv_cows),
        "calv_heifers": float(calv_heifers),
        "exp_bulls": float(exp_bulls),
        "exp_heifers": float(exp_heifers),
    })
    return snapshot, meta


# ------------------------------------------------------------
# Public API: compute forecast for a target date
# ------------------------------------------------------------

def compute_forecast_dynamic_from_db(target_date: date, overrides: dict | None = None) -> Dict[str, float]:
    tables = load_tables()
    base = latest_data_date(tables)
    start = min(base, target_date)

    ov = dict(overrides or {})

    # aliases from UI/app.py
    if "gestation_days" in ov and "GESTATION_DAYS" not in ov:
        ov["GESTATION_DAYS"] = ov["gestation_days"]
    if "dry_days" in ov and "DRY_DAYS" not in ov:
        ov["DRY_DAYS"] = ov["dry_days"]
    if "annual_disposal_rate" in ov and "ANNUAL_DISPOSAL_RATE" not in ov:
        ov["ANNUAL_DISPOSAL_RATE"] = ov["annual_disposal_rate"]
    if "conception" in ov and "CONCEPTION_PARAMS" not in ov:
        ov["CONCEPTION_PARAMS"] = ov["conception"]
    if "insemination_params" in ov and "INSEMINATION_PARAMS" not in ov:
        ov["INSEMINATION_PARAMS"] = ov["insemination_params"]
    if "semen_usage" in ov and "SEMEN_USAGE_SHARES" not in ov:
        ov["SEMEN_USAGE_SHARES"] = ov["semen_usage"]

    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

    # semen shares: override or DB
    semen_override = params.get("SEMEN_USAGE_SHARES")
    if isinstance(semen_override, dict) and semen_override:
        semen_shares = {
            "cow_trad": float(semen_override.get("cow_trad", 0.0)),
            "cow_sex": float(semen_override.get("cow_sex", 0.0)),
            "heifer_trad": float(semen_override.get("heifer_trad", 0.0)),
            "heifer_sex": float(semen_override.get("heifer_sex", 0.0)),
        }

        def _norm2(a: float, b: float) -> tuple[float, float]:
            s = max(1e-9, a + b)
            return a / s, b / s

        semen_shares["cow_trad"], semen_shares["cow_sex"] = _norm2(semen_shares["cow_trad"], semen_shares["cow_sex"])
        semen_shares["heifer_trad"], semen_shares["heifer_sex"] = _norm2(semen_shares["heifer_trad"], semen_shares["heifer_sex"])
    else:
        semen_shares = compute_semen_usage_from_db(tables)

    # calf sex ratios by semen
    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj: SemenSexRatio) -> SemenSexRatio:
            bull = float(d.get("bull_share", fallback_obj.bull_share))
            bull = max(0.0, min(1.0, bull))
            return SemenSexRatio(bull_share=bull, heifer_share=1.0 - bull)

        semen_sex_ratios = {
            "trad": _mk_ratio(trad, _to_semen_ratio(SEMEN_SEX_RATIOS["trad"])),
            "sex": _mk_ratio(sex, _to_semen_ratio(SEMEN_SEX_RATIOS["sex"])),
        }
    else:
        semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    warmstart_from_services = bool(ov.get("warmstart_from_services", True))

    state0 = build_initial_state(
        tables,
        as_of=start,
        gest_days=gest_days,
        dry_days=dry_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        warmstart_from_services=warmstart_from_services,
        semen_sex_ratios=semen_sex_ratios,   # важно: для распределения пола телят при пустом поле
    )

    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_date,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )

    # --- агрегаты стада на target_date ---
    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)
    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    #h0_2 = float(state_at_target.heifer_age[:90].sum())
    h0_3 = float(state_at_target.heifer_age[:90].sum())
    h3_8 = float(state_at_target.heifer_age[90:270].sum())
    h9p = float(state_at_target.heifer_age[270:].sum())
    b0_2 = float(state_at_target.bull_age[:61].sum())

    # monthly expectations
    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    return {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),

        "Тёлки 0–3 мес": round(h0_3, 1),     # <- UI перестанет показывать 0
        "Тёлки 0–2 мес": round(h0_3, 1),     # <- legacy алиас, можно убрать потом

        "Бычки 0–2 мес": round(b0_2, 1),
        "Тёлки 3–8 мес": round(h3_8, 1),
        "Тёлки ≥9 мес": round(h9p, 1),
        "Нетели": round(neteli, 1),

        "Ожидаемый отёл, всего": round(calv_total_f, 1),
        "Ожидаемый отёл, из них коров": round(calv_cows_f, 1),
        "Ожидаемый отёл, из них нетелей": round(calv_heifers_f, 1),

        "Ожидаемые бычки (условно)": round(exp_bulls_f, 1),
        "Ожидаемые тёлочки (условно)": round(exp_heifers_f, 1),

        "К реализации: коровы": round(float(meta.get("sell_cows", 0.0)), 1),
        "К реализации: тёлки": round(float(meta.get("sell_heifers", 0.0)), 1),
        "К реализации: нетели": round(float(meta.get("sell_neteli", 0.0)), 1),

        "Переполнение: Дойные коровы": round(float(meta.get("over_doy", 0.0)), 1),
        "Переполнение: Сухостойные коровы": round(float(meta.get("over_dry", 0.0)), 1),
        "Переполнение: Тёлки 0–3 мес": round(float(meta.get("over_h0", 0.0)), 1),
        "Переполнение: Тёлки 3–8 мес": round(float(meta.get("over_h38", 0.0)), 1),
        "Переполнение: Тёлки 9–24 мес": round(float(meta.get("over_h9", 0.0)), 1),
        "Переполнение: Нетели": round(float(meta.get("over_neteli", 0.0)), 1),
    }