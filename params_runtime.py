from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from db import engine
from model_params import (
    CONCEPTION_PARAMS as DEFAULT_CONCEPTION_PARAMS,
    GESTATION_DAYS as DEFAULT_GESTATION_DAYS,
    DRY_DAYS as DEFAULT_DRY_DAYS,
    DISPOSAL_PARAMS as DEFAULT_DISPOSAL_PARAMS,
    ANNUAL_DISPOSAL_RATE as DEFAULT_ANNUAL_DISPOSAL_RATE,
    INSEMINATION_PARAMS as DEFAULT_INSEMINATION_PARAMS,
)

import re


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


def norm_result(x: object) -> str:
    if x is None:
        return ""
    return str(x).replace("\u00a0", " ").strip().upper()


def norm_event_type(x: object) -> str:
    if x is None:
        return ""
    return str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")


def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4


def _safe_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.date


def _merge_asof_by_reg(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()

    left[left_on] = pd.to_datetime(left[left_on], errors="coerce", dayfirst=True)
    right[right_on] = pd.to_datetime(right[right_on], errors="coerce", dayfirst=True)

    left = left.dropna(subset=["reg_s", left_on]).copy()
    right = right.dropna(subset=["reg_s", right_on]).copy()

    # For merge_asof pandas requires global sorting by merge key (date/time) first.
    left = left.sort_values([left_on, "reg_s"], kind="mergesort")
    right = right.sort_values([right_on, "reg_s"], kind="mergesort")

    return pd.merge_asof(
        left,
        right,
        by="reg_s",
        left_on=left_on,
        right_on=right_on,
        direction="backward",
        allow_exact_matches=True,
    )


@dataclass(frozen=True)
class RuntimeParams:
    conception_params: object
    gestation_days: float
    dry_days: int
    disposal_params: dict
    annual_disposal_rate: float
    insemination_params: object
    meta: dict


def _compute_conception_params(ins: pd.DataFrame):
    ins = ins.copy()
    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce", dayfirst=True)
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")

    p = ins[(ins["event_date"].notna()) & (ins["result_norm"] == "P") & (ins["dim_age"].notna())].copy()
    if p.empty:
        return DEFAULT_CONCEPTION_PARAMS

    cows = p[p["lact"] > 0].copy()
    cows["lact_cat"] = cows["lact"].clip(lower=1, upper=4)

    avg_by = (
        cows.groupby("lact_cat")["dim_age"]
        .mean()
        .to_dict()
    )
    avg_by = {int(k): float(v) for k, v in avg_by.items()}

    global_mean = float(cows["dim_age"].mean()) if not cows.empty else float(p["dim_age"].mean())

    heifers = p[p["lact"] <= 0].copy()
    heifer_mean = float(heifers["dim_age"].mean()) if not heifers.empty else float(DEFAULT_CONCEPTION_PARAMS.avg_heifer_age_days)

    from model_params.defaults import ConceptionParams
    return ConceptionParams(
        avg_cow_dim_by_lact={
            1: float(avg_by.get(1, global_mean)),
            2: float(avg_by.get(2, global_mean)),
            3: float(avg_by.get(3, global_mean)),
            4: float(avg_by.get(4, global_mean)),
        },
        avg_cow_dim_global=float(global_mean),
        avg_heifer_age_days=float(heifer_mean),
    )


def _compute_gestation_days(calv: pd.DataFrame, ins: pd.DataFrame) -> Tuple[float, dict]:
    calv = calv.copy()
    ins = ins.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce", dayfirst=True)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    if births.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    births = births[["mother_reg_s", "event_date"]].drop_duplicates().rename(
        columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}
    )

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce", dayfirst=True)
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["reg_s"] = ins["reg"].apply(norm_id)

    p = ins[(ins["event_date"].notna()) & (ins["result_norm"] == "P") & (ins["reg_s"] != "")].copy()
    if p.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    p = p[["reg_s", "event_date"]].rename(columns={"event_date": "p_dt"})
    merged = _merge_asof_by_reg(
        births.rename(columns={"calving_dt": "left_dt"}),
        p.rename(columns={"p_dt": "right_dt"}),
        "left_dt",
        "right_dt",
    )

    merged["gest_days"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["gest_days"].between(200, 310, inclusive="both")].copy()

    if merged.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    mean = float(merged["gest_days"].mean())
    meta = {
        "n": int(len(merged)),
        "min": int(merged["gest_days"].min()),
        "median": float(merged["gest_days"].median()),
        "mean": float(mean),
        "max": int(merged["gest_days"].max()),
    }
    return mean, meta


def _compute_dry_days(calv: pd.DataFrame, dry: pd.DataFrame) -> Tuple[int, dict]:
    calv = calv.copy()
    dry = dry.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce", dayfirst=True)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    if births.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    births = births[["mother_reg_s", "event_date"]].drop_duplicates().rename(
        columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}
    )

    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce", dayfirst=True)
    dry["reg_s"] = dry["reg"].apply(norm_id)
    dry = dry[(dry["reg_s"] != "") & (dry["event_date"].notna())].copy()
    if dry.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    dry = dry[["reg_s", "event_date"]].rename(columns={"event_date": "dry_dt"})

    merged = _merge_asof_by_reg(
        births.rename(columns={"calving_dt": "left_dt"}),
        dry.rename(columns={"dry_dt": "right_dt"}),
        "left_dt",
        "right_dt",
    )

    merged["dry_days"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["dry_days"].between(10, 200, inclusive="both")].copy()

    if merged.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    mean = float(merged["dry_days"].mean())
    median = float(merged["dry_days"].median())
    meta = {
        "n": int(len(merged)),
        "min": int(merged["dry_days"].min()),
        "median": median,
        "mean": mean,
        "max": int(merged["dry_days"].max()),
    }
    return int(round(mean)), meta


def _compute_disposal_params(calv: pd.DataFrame, disp: pd.DataFrame) -> Tuple[dict, float, dict]:
    calv = calv.copy()
    disp = disp.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce", dayfirst=True)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    calv_events = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    calv_events = calv_events.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"})[["reg_s", "calving_dt"]]
    if calv_events.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    calv_counts = calv_events.groupby("reg_s")["calving_dt"].count().to_dict()
    calv_events = calv_events.drop_duplicates()

    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce", dayfirst=True)
    disp["reg_s"] = disp["reg"].apply(norm_id)
    disp["reason"] = disp.get("disposal_reason", "").astype(str).str.lower().str.replace("ё", "е")
    disp = disp[(disp["reg_s"] != "") & (disp["event_date"].notna())].copy()

    if disp.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    disp = disp[~disp["reason"].str.contains("переезд", na=False)].copy()

    merged = _merge_asof_by_reg(
        disp.rename(columns={"event_date": "left_dt"})[["reg_s", "left_dt"]],
        calv_events.rename(columns={"calving_dt": "right_dt"})[["reg_s", "right_dt"]],
        "left_dt",
        "right_dt",
    )
    merged["dim"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["dim"].between(0, 500, inclusive="both")].copy()
    if merged.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    merged["lact_cat"] = merged["reg_s"].map(lambda r: lact_cat_from_count(int(calv_counts.get(r, 1))))
    merged["lact_cat"] = merged["lact_cat"].astype(int)

    by_lact = {}
    for l in (1, 2, 3, 4):
        x = merged[merged["lact_cat"] == l]["dim"]
        if x.empty:
            by_lact[l] = {"n": 0, "mean_dim": 0.0, "median_dim": 0.0}
        else:
            by_lact[l] = {
                "n": int(len(x)),
                "mean_dim": float(x.mean()),
                "median_dim": float(x.median()),
            }

    overall = {
        "n": int(len(merged)),
        "mean_dim": float(merged["dim"].mean()),
        "median_dim": float(merged["dim"].median()),
    }

    disposal_params = {"by_lact": by_lact, "overall": overall}

                                        
    dmin = merged["left_dt"].min()
    dmax = merged["left_dt"].max()
    years = (dmax - dmin).days / 365.25 if pd.notna(dmin) and pd.notna(dmax) else 0.0
    herd_proxy = float(len(calv_counts)) if len(calv_counts) > 0 else 0.0
    if years >= 0.25 and herd_proxy > 0:
        annual_rate = float(overall["n"] / (herd_proxy * years))
        annual_rate = float(max(0.0, min(0.5, annual_rate)))
    else:
        annual_rate = float(DEFAULT_ANNUAL_DISPOSAL_RATE)

    meta = {"n": int(len(merged)), "years": float(years), "herd_proxy": float(herd_proxy)}
    return disposal_params, annual_rate, meta


def _compute_insemination_params(ins: pd.DataFrame, calv: pd.DataFrame):
    ins = ins.copy()
    calv = calv.copy()

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce", dayfirst=True)
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)

    def _month_factors(df: pd.DataFrame) -> dict[int, float]:
        work = df[(df["event_date"].notna()) & (df["reg_s"] != "")].copy()
        if work.empty:
            return {m: 1.0 for m in range(1, 13)}
        work["month"] = work["event_date"].dt.month.astype("Int64")
        work["is_p"] = (work["result_norm"] == "P").astype(float)
        overall_rate = float(work["is_p"].mean())
        if not (overall_rate > 1e-9):
            return {m: 1.0 for m in range(1, 13)}

        factors: dict[int, float] = {}
        for month in range(1, 13):
            part = work.loc[work["month"] == month]
            n = int(len(part))
            if n <= 0:
                factors[month] = 1.0
                continue
            month_rate = float(part["is_p"].mean())
            raw = month_rate / overall_rate if overall_rate > 1e-9 else 1.0
            shrink = min(1.0, n / 80.0)
            factors[month] = float(max(0.75, min(1.25, 1.0 + shrink * (raw - 1.0))))
        return factors

               
    def mean_interval(df: pd.DataFrame) -> float:
        df = df[(df["reg_s"] != "") & (df["event_date"].notna())].copy()
        if df.empty:
            return float("nan")
        df = df.sort_values(["reg_s", "event_date"], kind="mergesort")
        d = df.groupby("reg_s")["event_date"].diff().dt.days
        d = d[(d.notna()) & (d > 0) & (d <= 365)]
        return float(d.mean()) if not d.empty else float("nan")

    cow_ai_interval = mean_interval(ins[ins["lact"] > 0])
    heifer_ai_interval = mean_interval(ins[ins["lact"] <= 0])
    cow_conception_month_factors = _month_factors(ins[ins["lact"] > 0])
    heifer_conception_month_factors = _month_factors(ins[ins["lact"] <= 0])

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce", dayfirst=True)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    births = births.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"})[["reg_s", "calving_dt"]]
    births = births.drop_duplicates().sort_values(["reg_s", "calving_dt"], kind="mergesort")

    cow_first_ai_by_lact = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
    cow_spc = float("nan")

    if not births.empty:
        ins_cow = ins[(ins["lact"] > 0) & (ins["reg_s"] != "") & (ins["event_date"].notna())].copy()
        if not ins_cow.empty:
            left = ins_cow[["reg_s", "event_date", "lact", "result_norm", "dim_age"]].rename(columns={"event_date": "left_dt"})
            right = births.rename(columns={"calving_dt": "right_dt"})[["reg_s", "right_dt"]]
            merged = _merge_asof_by_reg(left, right, "left_dt", "right_dt")
            merged = merged[merged["left_dt"] >= merged["right_dt"]].copy()

            merged = merged.sort_values(["reg_s", "right_dt", "left_dt"], kind="mergesort")
            first_ai = merged.groupby(["reg_s", "right_dt"], sort=False).head(1).copy()
            first_ai["lact_cat"] = first_ai["lact"].clip(lower=1, upper=4)
            if first_ai["dim_age"].notna().any():
                agg = first_ai.groupby("lact_cat")["dim_age"].mean().to_dict()
                for k, v in agg.items():
                    cow_first_ai_by_lact[int(k)] = float(v)

            def attempts_until_p(g: pd.DataFrame) -> float | None:
                g = g.sort_values("left_dt", kind="mergesort")
                idx = g.index[g["result_norm"] == "P"]
                if len(idx) == 0:
                    return None
                first_p_pos = g.index.get_loc(idx[0])
                return float(first_p_pos + 1)

            grp = merged.groupby(["reg_s", "right_dt"], sort=False)
            vals = []
            for _, g in grp:
                v = attempts_until_p(g)
                if v is not None:
                    vals.append(v)
            if vals:
                cow_spc = float(np.mean(vals))

    ins_h = ins[(ins["lact"] <= 0) & (ins["reg_s"] != "") & (ins["event_date"].notna())].copy()
    heifer_first_ai_age = float("nan")
    heifer_spc = float("nan")
    if not ins_h.empty:
        ins_h = ins_h.sort_values(["reg_s", "event_date"], kind="mergesort")
        first = ins_h.groupby("reg_s", sort=False).head(1)
        if first["dim_age"].notna().any():
            heifer_first_ai_age = float(first["dim_age"].mean())

        vals = []
        for _, g in ins_h.groupby("reg_s", sort=False):
            g = g.sort_values("event_date", kind="mergesort")
            idx = g.index[g["result_norm"] == "P"]
            if len(idx) == 0:
                continue
            pos = g.index.get_loc(idx[0])
            vals.append(float(pos + 1))
        if vals:
            heifer_spc = float(np.mean(vals))

    from model_params.defaults import InseminationParams

    def _fallback(x: float, fb: float) -> float:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return float(fb)
        return float(x)

    cow_first_ai_by_lact = {
        1: _fallback(cow_first_ai_by_lact.get(1), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(1)),
        2: _fallback(cow_first_ai_by_lact.get(2), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(2)),
        3: _fallback(cow_first_ai_by_lact.get(3), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(3)),
        4: _fallback(cow_first_ai_by_lact.get(4), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(4)),
    }

    return InseminationParams(
        cow_first_ai_dim_by_lact=cow_first_ai_by_lact,
        cow_ai_interval_days=_fallback(cow_ai_interval, DEFAULT_INSEMINATION_PARAMS.cow_ai_interval_days),
        cow_services_per_conception=_fallback(cow_spc, DEFAULT_INSEMINATION_PARAMS.cow_services_per_conception),
        heifer_first_ai_age_days=_fallback(heifer_first_ai_age, DEFAULT_INSEMINATION_PARAMS.heifer_first_ai_age_days),
        heifer_ai_interval_days=_fallback(heifer_ai_interval, DEFAULT_INSEMINATION_PARAMS.heifer_ai_interval_days),
        heifer_services_per_conception=_fallback(heifer_spc, DEFAULT_INSEMINATION_PARAMS.heifer_services_per_conception),
        cow_conception_month_factors=cow_conception_month_factors,
        heifer_conception_month_factors=heifer_conception_month_factors,
    )


def compute_params_from_db() -> RuntimeParams:
    calv = pd.read_sql("SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw", con=engine)
    ins = pd.read_sql("SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw", con=engine)
    dry = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", con=engine)
    disp = pd.read_sql("SELECT reg, event_date, disposal_reason FROM disposals_raw", con=engine)

    conception_params = _compute_conception_params(ins)
    gest, gest_meta = _compute_gestation_days(calv, ins)
    dry_days, dry_meta = _compute_dry_days(calv, dry)
    disposal_params, annual_rate, disp_meta = _compute_disposal_params(calv, disp)
    insemination_params = _compute_insemination_params(ins, calv)

    meta = {
        "gestation": gest_meta,
        "dry": dry_meta,
        "disposal": disp_meta,
    }

    return RuntimeParams(
        conception_params=conception_params,
        gestation_days=float(gest),
        dry_days=int(dry_days),
        disposal_params=disposal_params,
        annual_disposal_rate=float(annual_rate),
        insemination_params=insemination_params,
        meta=meta,
    )

from dataclasses import dataclass
from datetime import timedelta
from collections import Counter
from calendar import monthrange


@dataclass
class PendingCalvings:
    """Ожидаемые отёлы, уже 'заложенные' до даты старта прогноза."""
    cows: Counter
    heifers: Counter
    meta: dict


def _compute_pending_calvings_from_history(
    ins: pd.DataFrame,
    start_date: date,
    gestation_days: int,
) -> PendingCalvings:
    """
    Берём P-осеменения в окне [start_date - gestation_days, start_date],
    дедуп по животному (берём последнее P), считаем due_date = ai_date + gestation_days.
    """
    if ins is None or ins.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    df = ins.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=True).dt.date
    df["result_norm"] = df["result"].apply(norm_result)
    df["reg_s"] = df["reg"].apply(norm_id)
    df["lact_n"] = pd.to_numeric(df.get("lact", 0), errors="coerce").fillna(0).astype(int)

    df = df[(df["reg_s"] != "") & (df["event_date"].notna()) & (df["result_norm"] == "P")].copy()
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    window_start = start_date - timedelta(days=int(gestation_days))
    df = df[(df["event_date"] >= window_start) & (df["event_date"] <= start_date)].copy()
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    df = df.sort_values(["reg_s", "event_date"], kind="mergesort")
    df = df.groupby("reg_s", sort=False).tail(1)

    df["due_date"] = df["event_date"].apply(lambda d: d + timedelta(days=int(gestation_days)))
                                            
    df = df[df["due_date"] >= start_date].copy()
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    cows_due = df[df["lact_n"] > 0]["due_date"].tolist()
    heifers_due = df[df["lact_n"] <= 0]["due_date"].tolist()

    cows = Counter(cows_due)
    heifers = Counter(heifers_due)
    meta = {"n_total": int(len(df)), "n_cows": int(len(cows_due)), "n_heifers": int(len(heifers_due))}
    return PendingCalvings(cows=cows, heifers=heifers, meta=meta)


def compute_pending_calvings_from_db(
    start_date: date,
    gestation_days: int | None = None,
) -> PendingCalvings:
    """
    Публичная функция: посчитать pending calvings из БД на дату старта прогноза.
    """
    g = int(round(float(gestation_days if gestation_days is not None else DEFAULT_GESTATION_DAYS)))
    ins = pd.read_sql("SELECT reg, lact, event_date, result FROM inseminations_raw", con=engine)
    return _compute_pending_calvings_from_history(ins, start_date=start_date, gestation_days=g)
