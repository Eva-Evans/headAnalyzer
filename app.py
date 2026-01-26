from __future__ import annotations

import calendar
import os
from datetime import date
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from db import engine
from forecast import compute_forecast_from_db

from etl.bulls import read_bulls_txt, load_bulls_to_db
from etl.calvings_births import read_calvings_excel, load_calvings_to_db
from etl.disposals import read_disposals_excel, load_disposals_to_db
from etl.dryoff import read_dryoff_excel, load_dryoff_to_db
from etl.inseminations import read_inseminations_excel, clean_inseminations, load_inseminations_to_db

import model_params as mp


INDICATORS = [
    "Дойные коровы",
    "Сухостойные коровы",
    "Тёлки 0–2 мес",
    "Бычки 0–2 мес",
    "Тёлки 3–8 мес",
    "Тёлки ≥9 мес",
    "Нетели",
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки (условно)",
    "Ожидаемые тёлочки (условно)",
]
REALIZATION_COLS = [
    "К реализации: коровы",
    "К реализации: тёлки",
    "К реализации: нетели",
    "Переполнение: Дойные коровы",
    "Переполнение: Сухостойные коровы",
    "Переполнение: Тёлки 0–3 мес",
    "Переполнение: Тёлки 3–8 мес",
    "Переполнение: Тёлки 9–24 мес",
    "Переполнение: Нетели",
]


# -----------------------------
# helpers
# -----------------------------
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


def capacity_name_for_indicator(indicator: str) -> Optional[str]:
    if indicator.startswith("Дойные коровы"):
        return "Дойные коровы"
    if indicator.startswith("Сухостойные коровы"):
        return "Сухостойные коровы"
    if indicator.startswith("Тёлки 0"):
        return "Тёлки 0–3 мес"
    if indicator.startswith("Тёлки 3"):
        return "Тёлки 3–8 мес"
    if indicator.startswith("Тёлки ≥9") or indicator.startswith("Тёлки 9"):
        return "Тёлки 9–24 мес"
    if indicator.startswith("Нетели"):
        return "Нетели"
    return None


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


def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        s2 = s[:-2]
        if s2.isdigit():
            return s2
    return s


def _norm_event_type(x: Any) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")
    if "РОЖ" in v:
        return "РОЖДЕН"
    return v


def _norm_result(x: Any) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")

    if v in {"P", "П"}:
        return "P"
    if "PREG" in v:
        return "P"
    if "СТЕЛ" in v or v in {"СТ", "СТ.", "СТ+", "СТЕЛЬНАЯ", "СТЕЛЬН"}:
        return "P"

    return v


def _norm_sex(x: Any) -> str | None:
    if x is None:
        return None
    v = str(x).strip().upper()
    if v in ("F", "Ж", "FEMALE"):
        return "F"
    if v in ("M", "М", "MALE"):
        return "M"
    return None


def _classify_semen_from_bull_type(bt: Any) -> str:
    v = "" if bt is None else str(bt).strip().upper()
    if v == "S" or "SEX" in v:
        return "sex"
    return "trad"


def _merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: str,
    left_on: str,
    right_on: str,
    direction: str = "backward",
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    l = left.copy()
    r = right.copy()

    l["_row_id"] = range(len(l))

    l[by] = l[by].astype(str).str.strip()
    r[by] = r[by].astype(str).str.strip()

    l[left_on] = pd.to_datetime(l[left_on], errors="coerce")
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce")

    l = l[(l[by] != "") & l[left_on].notna()].copy()
    r = r[(r[by] != "") & r[right_on].notna()].copy()

    l = l.sort_values([left_on, by], kind="mergesort").reset_index(drop=True)
    r = r.sort_values([right_on, by], kind="mergesort").reset_index(drop=True)

    out = pd.merge_asof(
        l,
        r,
        by=by,
        left_on=left_on,
        right_on=right_on,
        direction=direction,
        allow_exact_matches=allow_exact_matches,
    )

    out = out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)
    return out


def _bayes_smooth_share(sex_count: float, total: float, prior_sex: float, prior_weight: float) -> float:
    total = float(total)
    sex_count = float(sex_count)
    prior_sex = float(prior_sex)
    prior_weight = float(prior_weight)
    denom = max(1e-9, total + prior_weight)
    p = (sex_count + prior_weight * prior_sex) / denom
    return float(max(0.0, min(1.0, p)))


def make_excel_bytes(forecast_df: pd.DataFrame, realization_df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Прогноз")
        realization_df.to_excel(writer, sheet_name="Реализация")
    return out.getvalue()


# -----------------------------
# params from DB
# -----------------------------
def compute_params_from_db() -> Dict[str, Any]:
    calv = pd.read_sql(
        "SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw",
        con=engine,
    )
    ins = pd.read_sql(
        "SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw",
        con=engine,
    )
    dry = pd.read_sql(
        "SELECT reg, dim, event_date FROM dryoff_raw",
        con=engine,
    )
    disp = pd.read_sql(
        "SELECT reg, event_date, disposal_reason FROM disposals_raw",
        con=engine,
    )
    bulls = pd.read_sql(
        "SELECT bull_code, bull_type FROM bulls_raw",
        con=engine,
    )

    out: Dict[str, Any] = {}

    # normalize
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce")
    calv["birth_date"] = pd.to_datetime(calv["birth_date"], errors="coerce")
    calv["event_type_norm"] = calv["event_type"].apply(_norm_event_type)
    calv["reg_s"] = calv["reg"].apply(_norm_id)
    calv["mother_reg_s"] = calv["mother_reg"].apply(_norm_id)
    calv["sex_norm"] = calv["sex"].apply(_norm_sex)

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce")
    ins["result_norm"] = ins["result"].apply(_norm_result)
    ins["reg_s"] = ins["reg"].apply(_norm_id)
    ins["bull_s"] = ins["bull"].apply(_norm_id)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")

    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce")
    dry["reg_s"] = dry["reg"].apply(_norm_id)

    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce")
    disp["reg_s"] = disp["reg"].apply(_norm_id)
    disp["reason_norm"] = disp["disposal_reason"].astype(str).str.lower().str.replace("ё", "е")

    bulls["bull_code_s"] = bulls["bull_code"].apply(_norm_id)
    bulls["semen"] = bulls["bull_type"].apply(_classify_semen_from_bull_type)
    semen_by_bull = dict(zip(bulls["bull_code_s"].astype(str), bulls["semen"].astype(str)))

    # ------------------ conception params (P)
    ins_p = ins[(ins["event_date"].notna()) & (ins["result_norm"] == "P") & (ins["reg_s"] != "")].copy()

    avg_by_lact = dict(mp.CONCEPTION_PARAMS.avg_cow_dim_by_lact)
    avg_global = float(mp.CONCEPTION_PARAMS.avg_cow_dim_global)
    avg_heif_age = float(mp.CONCEPTION_PARAMS.avg_heifer_age_days)

    if not ins_p.empty:
        cows_p = ins_p[(ins_p["lact"] > 0) & ins_p["dim_age"].notna()].copy()
        if not cows_p.empty:
            cows_p["lact_cat"] = cows_p["lact"].clip(lower=1, upper=4)
            g = cows_p.groupby("lact_cat", sort=False)["dim_age"].mean().to_dict()
            avg_by_lact = {int(k): float(v) for k, v in g.items() if pd.notna(v)}
            gmean = cows_p["dim_age"].mean()
            if pd.notna(gmean):
                avg_global = float(gmean)

        heif_p = ins_p[(ins_p["lact"] <= 0) & ins_p["dim_age"].notna()].copy()
        if not heif_p.empty:
            hmean = heif_p["dim_age"].mean()
            if pd.notna(hmean):
                avg_heif_age = float(hmean)

    out["conception"] = {
        "avg_cow_dim_by_lact": {
            1: float(avg_by_lact.get(1, avg_global)),
            2: float(avg_by_lact.get(2, avg_global)),
            3: float(avg_by_lact.get(3, avg_global)),
            4: float(avg_by_lact.get(4, avg_global)),
        },
        "avg_cow_dim_global": float(avg_global),
        "avg_heifer_age_days": float(avg_heif_age),
    }

    # ------------------ gestation days (calf->mother + last P)
    out["gestation_days"] = float(getattr(mp, "GESTATION_DAYS", 272.0))
    calves = calv[
        (calv["event_date"].notna())
        & (calv["event_type_norm"] == "РОЖДЕН")
        & (calv["mother_reg_s"] != "")
    ][["mother_reg_s", "event_date"]].drop_duplicates()

    if not calves.empty and not ins_p.empty:
        left = calves.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}).copy()
        right = ins_p.rename(columns={"event_date": "ins_dt"}).copy()
        right = right[["reg_s", "ins_dt"]].copy()

        pairs = _merge_asof_safe(
            left,
            right,
            by="reg_s",
            left_on="calving_dt",
            right_on="ins_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        pairs = pairs[pairs["ins_dt"].notna()].copy()
        if not pairs.empty:
            pairs["gest_days"] = (pairs["calving_dt"] - pairs["ins_dt"]).dt.days
            pairs = pairs[(pairs["gest_days"] >= 200) & (pairs["gest_days"] <= 310)].copy()
            if not pairs.empty:
                out["gestation_days"] = float(pairs["gest_days"].mean())

    # ------------------ dry days (dryoff -> calving)
    out["dry_days"] = int(getattr(mp, "DRY_DAYS", 53))
    if not dry.empty and not calves.empty:
        left = calves.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}).copy()
        right = dry.rename(columns={"event_date": "dry_dt"}).copy()
        right = right[(right["dry_dt"].notna()) & (right["reg_s"] != "")][["reg_s", "dry_dt"]].copy()

        m = _merge_asof_safe(
            left,
            right,
            by="reg_s",
            left_on="calving_dt",
            right_on="dry_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        m = m[m["dry_dt"].notna()].copy()
        if not m.empty:
            m["dry_days"] = (m["calving_dt"] - m["dry_dt"]).dt.days
            m = m[(m["dry_days"] >= 10) & (m["dry_days"] <= 200)].copy()
            if not m.empty:
                out["dry_days"] = int(round(float(m["dry_days"].median())))

    # ------------------ disposal params (DIM at last ins before disposal)
    disposal_params = {
        "by_lact": {
            1: {"n": 0, "mean_dim": 0.0, "median_dim": 0.0},
            2: {"n": 0, "mean_dim": 0.0, "median_dim": 0.0},
            3: {"n": 0, "mean_dim": 0.0, "median_dim": 0.0},
            4: {"n": 0, "mean_dim": 0.0, "median_dim": 0.0},
        },
        "overall": {"n": 0, "mean_dim": 0.0, "median_dim": 0.0},
    }

    disp2 = disp[
        (disp["event_date"].notna())
        & (disp["reg_s"] != "")
        & (~disp["reason_norm"].str.contains("переезд", na=False))
    ].copy()

    ins2 = ins[(ins["event_date"].notna()) & (ins["reg_s"] != "") & ins["dim_age"].notna()].copy()
    ins2 = ins2[(ins2["dim_age"] >= 0) & (ins2["dim_age"] <= 500)].copy()

    if not disp2.empty and not ins2.empty:
        left = disp2.rename(columns={"event_date": "disp_dt"})[["reg_s", "disp_dt"]].copy()
        right = ins2.rename(columns={"event_date": "ins_dt"})[["reg_s", "ins_dt", "lact", "dim_age"]].copy()

        m = _merge_asof_safe(
            left,
            right,
            by="reg_s",
            left_on="disp_dt",
            right_on="ins_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        m = m[m["ins_dt"].notna()].copy()
        if not m.empty:
            m["lact_cat"] = m["lact"].clip(lower=1, upper=4).astype(int)

            disposal_params["overall"]["n"] = int(len(m))
            disposal_params["overall"]["mean_dim"] = float(m["dim_age"].mean())
            disposal_params["overall"]["median_dim"] = float(m["dim_age"].median())

            for lc in (1, 2, 3, 4):
                sub = m[m["lact_cat"] == lc]
                if sub.empty:
                    continue
                disposal_params["by_lact"][lc]["n"] = int(len(sub))
                disposal_params["by_lact"][lc]["mean_dim"] = float(sub["dim_age"].mean())
                disposal_params["by_lact"][lc]["median_dim"] = float(sub["dim_age"].median())

    out["disposal_params"] = disposal_params
    out["annual_disposal_rate"] = float(getattr(mp, "ANNUAL_DISPOSAL_RATE", 0.0957))

    # ------------------ insemination params (services / intervals / first AI)
    ip = {
        "cow_services_per_conception": float(mp.INSEMINATION_PARAMS.cow_services_per_conception),
        "cow_ai_interval_days": float(mp.INSEMINATION_PARAMS.cow_ai_interval_days),
        "cow_first_ai_dim_by_lact": dict(mp.INSEMINATION_PARAMS.cow_first_ai_dim_by_lact),
        "heifer_services_per_conception": float(mp.INSEMINATION_PARAMS.heifer_services_per_conception),
        "heifer_ai_interval_days": float(mp.INSEMINATION_PARAMS.heifer_ai_interval_days),
        "heifer_first_ai_age_days": float(mp.INSEMINATION_PARAMS.heifer_first_ai_age_days),
    }

    ins_all = ins[(ins["event_date"].notna()) & (ins["reg_s"] != "")].copy()
    if not ins_all.empty:
        # cows
        cows = ins_all[ins_all["lact"] > 0].copy()
        cows = cows[(cows["dim_age"].notna()) & (cows["dim_age"] >= 0) & (cows["dim_age"] <= 500)].copy()
        if not cows.empty:
            cows = cows.sort_values(["reg_s", "lact", "event_date"], kind="mergesort")
            grp = cows.groupby(["reg_s", "lact"], sort=False)

            services_counts: List[float] = []
            first_dims_by_lact: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
            intervals: List[float] = []

            for (_reg, lact), g in grp:
                g = g.sort_values("event_date", kind="mergesort")
                mask_p = (g["result_norm"] == "P").to_numpy()
                if mask_p.sum() == 0:
                    continue
                first_p_pos = int(mask_p.argmax())
                services_counts.append(float(first_p_pos + 1))

                first_dim = g.iloc[0]["dim_age"]
                if pd.notna(first_dim):
                    lc = int(min(4, max(1, int(lact))))
                    first_dims_by_lact[lc].append(float(first_dim))

                dts = pd.to_datetime(g["event_date"]).sort_values()
                if len(dts) >= 2:
                    diffs = dts.diff().dt.days.dropna()
                    intervals.extend([float(x) for x in diffs.values if pd.notna(x)])

            if services_counts:
                ip["cow_services_per_conception"] = float(pd.Series(services_counts).mean())
            if intervals:
                ip["cow_ai_interval_days"] = float(pd.Series(intervals).mean())
            for lc in (1, 2, 3, 4):
                if first_dims_by_lact[lc]:
                    ip["cow_first_ai_dim_by_lact"][lc] = float(pd.Series(first_dims_by_lact[lc]).mean())

        # heifers
        heif = ins_all[ins_all["lact"] <= 0].copy()
        heif = heif[(heif["dim_age"].notna()) & (heif["dim_age"] >= 0) & (heif["dim_age"] <= 900)].copy()
        if not heif.empty:
            heif = heif.sort_values(["reg_s", "event_date"], kind="mergesort")
            grp = heif.groupby("reg_s", sort=False)

            services_counts_h: List[float] = []
            intervals_h: List[float] = []
            first_ages: List[float] = []

            for _reg, g in grp:
                g = g.sort_values("event_date", kind="mergesort")
                mask_p = (g["result_norm"] == "P").to_numpy()
                if mask_p.sum() == 0:
                    continue
                first_p_pos = int(mask_p.argmax())
                services_counts_h.append(float(first_p_pos + 1))

                first_age = g.iloc[0]["dim_age"]
                if pd.notna(first_age):
                    first_ages.append(float(first_age))

                dts = pd.to_datetime(g["event_date"]).sort_values()
                if len(dts) >= 2:
                    diffs = dts.diff().dt.days.dropna()
                    intervals_h.extend([float(x) for x in diffs.values if pd.notna(x)])

            if services_counts_h:
                ip["heifer_services_per_conception"] = float(pd.Series(services_counts_h).mean())
            if intervals_h:
                ip["heifer_ai_interval_days"] = float(pd.Series(intervals_h).mean())
            if first_ages:
                ip["heifer_first_ai_age_days"] = float(pd.Series(first_ages).mean())

    out["insemination_params"] = ip

    # ------------------ semen usage (P inseminations + bulls map)
    semen_usage = {
        "cow_trad": float(mp.SEMEN_USAGE_PROBS.cow_trad),
        "cow_sex": float(mp.SEMEN_USAGE_PROBS.cow_sex),
        "heifer_trad": float(mp.SEMEN_USAGE_PROBS.heifer_trad),
        "heifer_sex": float(mp.SEMEN_USAGE_PROBS.heifer_sex),
        "meta": {"window": "fallback", "n_cow": 0, "n_heifer": 0},
    }

    if not ins_p.empty:
        p = ins_p.copy()
        p["bull_s"] = p["bull_s"].astype(str)
        p["semen"] = p["bull_s"].map(semen_by_bull).fillna("trad")

        max_dt = p["event_date"].max()
        p_last = p
        window = "all"
        if pd.notna(max_dt):
            p_last = p[p["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy()
            window = "last_year"

        def _shares(df: pd.DataFrame) -> Tuple[int, int, int]:
            if df.empty:
                return 0, 0, 0
            total = int(len(df))
            sex = int((df["semen"] == "sex").sum())
            trad = total - sex
            return trad, sex, total

        cow_df = p_last[p_last["lact"] > 0]
        heif_df = p_last[p_last["lact"] <= 0]

        cow_trad, cow_sex, n_cow = _shares(cow_df)
        hef_trad, hef_sex, n_hef = _shares(heif_df)

        if n_cow < 200 or n_hef < 200:
            window = "all"
            cow_trad, cow_sex, n_cow = _shares(p[p["lact"] > 0])
            hef_trad, hef_sex, n_hef = _shares(p[p["lact"] <= 0])

        prior_w = 500.0
        cow_sex_p = _bayes_smooth_share(cow_sex, n_cow, float(mp.SEMEN_USAGE_PROBS.cow_sex), prior_w)
        hef_sex_p = _bayes_smooth_share(hef_sex, n_hef, float(mp.SEMEN_USAGE_PROBS.heifer_sex), prior_w)

        semen_usage = {
            "cow_trad": float(1.0 - cow_sex_p),
            "cow_sex": float(cow_sex_p),
            "heifer_trad": float(1.0 - hef_sex_p),
            "heifer_sex": float(hef_sex_p),
            "meta": {"window": window, "n_cow": int(n_cow), "n_heifer": int(n_hef)},
        }

    out["semen_usage"] = semen_usage

    # ------------------ semen sex ratios (calves sex by semen type)
    semen_sex_ratios = {
        "trad": {
            "bull_share": float(mp.SEMEN_SEX_RATIOS["trad"].bull_share),
            "heifer_share": float(mp.SEMEN_SEX_RATIOS["trad"].heifer_share),
        },
        "sex": {
            "bull_share": float(mp.SEMEN_SEX_RATIOS["sex"].bull_share),
            "heifer_share": float(mp.SEMEN_SEX_RATIOS["sex"].heifer_share),
        },
        "meta": {"n_trad": 0, "n_sex": 0, "window": "fallback"},
    }

    calves_sex = calv[
        (calv["event_date"].notna())
        & (calv["event_type_norm"] == "РОЖДЕН")
        & (calv["mother_reg_s"] != "")
        & (calv["sex_norm"].isin(["F", "M"]))
    ][["mother_reg_s", "event_date", "sex_norm"]].copy()

    if not calves_sex.empty and not ins_p.empty:
        left = calves_sex.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}).copy()
        right = ins_p.rename(columns={"event_date": "ins_dt"}).copy()
        right = right[["reg_s", "ins_dt", "bull_s"]].copy()

        pairs = _merge_asof_safe(
            left,
            right,
            by="reg_s",
            left_on="calving_dt",
            right_on="ins_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        pairs = pairs[pairs["ins_dt"].notna()].copy()
        if not pairs.empty:
            pairs["gest_days"] = (pairs["calving_dt"] - pairs["ins_dt"]).dt.days
            pairs = pairs[(pairs["gest_days"] >= 200) & (pairs["gest_days"] <= 310)].copy()

        if not pairs.empty:
            pairs["semen"] = pairs["bull_s"].astype(str).map(semen_by_bull).fillna("trad")

            max_dt = pairs["calving_dt"].max()
            window = "all"
            pairs_w = pairs
            if pd.notna(max_dt):
                pairs_w = pairs[pairs["calving_dt"] >= (max_dt - pd.Timedelta(days=365))].copy()
                window = "last_year"

            def _ratio(df: pd.DataFrame) -> Tuple[float, int]:
                if df.empty:
                    return 0.0, 0
                n = int(len(df))
                bulls_cnt = int((df["sex_norm"] == "M").sum())
                return float(bulls_cnt / max(1, n)), n

            trad_bull_p, n_trad = _ratio(pairs_w[pairs_w["semen"] == "trad"])
            sex_bull_p, n_sex = _ratio(pairs_w[pairs_w["semen"] == "sex"])

            if n_sex < 200:
                window = "all"
                trad_bull_p, n_trad = _ratio(pairs[pairs["semen"] == "trad"])
                sex_bull_p, n_sex = _ratio(pairs[pairs["semen"] == "sex"])

            prior_w = 300.0
            trad_bull_prior = float(mp.SEMEN_SEX_RATIOS["trad"].bull_share)
            sex_bull_prior = float(mp.SEMEN_SEX_RATIOS["sex"].bull_share)

            trad_bull_p = _bayes_smooth_share(trad_bull_p * n_trad, n_trad, trad_bull_prior, prior_w)
            sex_bull_p = _bayes_smooth_share(sex_bull_p * n_sex, n_sex, sex_bull_prior, prior_w)

            semen_sex_ratios = {
                "trad": {"bull_share": float(trad_bull_p), "heifer_share": float(1.0 - trad_bull_p)},
                "sex": {"bull_share": float(sex_bull_p), "heifer_share": float(1.0 - sex_bull_p)},
                "meta": {"n_trad": int(n_trad), "n_sex": int(n_sex), "window": window},
            }

    out["semen_sex_ratios"] = semen_sex_ratios
    return out


def get_param_source() -> Dict[str, Any]:
    if "computed_params" in st.session_state and isinstance(st.session_state.computed_params, dict):
        return st.session_state.computed_params

    return {
        "conception": {
            "avg_cow_dim_by_lact": dict(mp.CONCEPTION_PARAMS.avg_cow_dim_by_lact),
            "avg_cow_dim_global": float(mp.CONCEPTION_PARAMS.avg_cow_dim_global),
            "avg_heifer_age_days": float(mp.CONCEPTION_PARAMS.avg_heifer_age_days),
        },
        "gestation_days": float(mp.GESTATION_DAYS),
        "dry_days": int(getattr(mp, "DRY_DAYS", 53)),
        "disposal_params": dict(mp.DISPOSAL_PARAMS),
        "annual_disposal_rate": float(getattr(mp, "ANNUAL_DISPOSAL_RATE", 0.0957)),
        "insemination_params": {
            "cow_services_per_conception": float(mp.INSEMINATION_PARAMS.cow_services_per_conception),
            "cow_ai_interval_days": float(mp.INSEMINATION_PARAMS.cow_ai_interval_days),
            "cow_first_ai_dim_by_lact": dict(mp.INSEMINATION_PARAMS.cow_first_ai_dim_by_lact),
            "heifer_services_per_conception": float(mp.INSEMINATION_PARAMS.heifer_services_per_conception),
            "heifer_ai_interval_days": float(mp.INSEMINATION_PARAMS.heifer_ai_interval_days),
            "heifer_first_ai_age_days": float(mp.INSEMINATION_PARAMS.heifer_first_ai_age_days),
        },
        "semen_usage": {
            "cow_trad": float(mp.SEMEN_USAGE_PROBS.cow_trad),
            "cow_sex": float(mp.SEMEN_USAGE_PROBS.cow_sex),
            "heifer_trad": float(mp.SEMEN_USAGE_PROBS.heifer_trad),
            "heifer_sex": float(mp.SEMEN_USAGE_PROBS.heifer_sex),
            "meta": {"window": "fallback", "n_cow": 0, "n_heifer": 0},
        },
        "semen_sex_ratios": {
            "trad": {
                "bull_share": float(mp.SEMEN_SEX_RATIOS["trad"].bull_share),
                "heifer_share": float(mp.SEMEN_SEX_RATIOS["trad"].heifer_share),
            },
            "sex": {
                "bull_share": float(mp.SEMEN_SEX_RATIOS["sex"].bull_share),
                "heifer_share": float(mp.SEMEN_SEX_RATIOS["sex"].heifer_share),
            },
            "meta": {"n_trad": 0, "n_sex": 0, "window": "fallback"},
        },
    }


def _apply_admin_overrides(base_params: Dict[str, Any]) -> Dict[str, Any]:
    p = {k: v for k, v in base_params.items()}
    ov = st.session_state.get("runtime_overrides")
    if not isinstance(ov, dict) or not ov:
        return p

    for key in ("gestation_days", "dry_days", "annual_disposal_rate"):
        if key in ov:
            p[key] = ov[key]

    if isinstance(ov.get("conception"), dict):
        p["conception"] = ov["conception"]
    if isinstance(ov.get("insemination_params"), dict):
        p["insemination_params"] = ov["insemination_params"]
    if isinstance(ov.get("semen_usage"), dict):
        p["semen_usage"] = ov["semen_usage"]
    if isinstance(ov.get("semen_sex_ratios"), dict):
        p["semen_sex_ratios"] = ov["semen_sex_ratios"]

    return p


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Прогноз поголовья", layout="wide")
st.title("Прогноз поголовья по подразделению")

# ---------- Источник данных (НОВОЕ)
st.subheader("0) Источник данных")
data_mode = st.radio(
    "Откуда брать данные для расчёта?",
    options=[
        "Использовать данные из БД (по умолчанию)",
        "Обновить БД из файлов (как раньше)",
    ],
    index=0,
    horizontal=True,
)
need_files = data_mode.startswith("Обновить БД")

if not need_files:
    st.info("Рассчёт возьмёт данные из БД. Загрузка файлов ниже нужна только если хочешь обновить данные в БД.")
else:
    st.warning("В этом режиме при расчёте данные в БД будут заменены загруженными файлами (replace).")

# ---------- admin (one place, not sidebar)
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

ADMIN_KEY_TRUE = os.getenv("ADMIN_KEY", "admin")

with st.expander("Админ-режим", expanded=st.session_state.is_admin):
    if not st.session_state.is_admin:
        k = st.text_input("Ключ доступа", type="password", key="admin_key_input")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Войти", key="admin_login_btn", use_container_width=True):
                if k == ADMIN_KEY_TRUE:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Неверный ключ")
        with c2:
            st.caption("Ключ берётся из переменной окружения ADMIN_KEY (docker-compose).")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Выйти", key="admin_logout_btn", use_container_width=True):
                st.session_state.is_admin = False
                st.session_state.pop("runtime_overrides", None)
                st.rerun()
        with c2:
            st.success("Админ-режим включён")

with st.expander("Справка: как считается прогноз", expanded=False):
    st.markdown(
        """
Мы строим **состояние стада** на дату последнего события в данных (максимальная дата среди: отёлы/рождения, осеменения, запуски, выбытия).

Дальше выполняем **симуляцию по дням**:
- животные “стареют” (DIM/возраст +1 в день),
- появляются новые стельности через параметры осеменения (первая попытка, интервал между попытками, доз на 1 стельность),
- стельные автоматически переходят в сухостой за число дней “длительность сухостоя” до отёла,
- происходят отёлы, добавляются телята по полу (по типу семени),
- происходит выбытие (по годовому проценту и форме по DIM/лактациям).

В прогнозе показывается **состояние на конец каждого месяца** и **ожидаемые отёлы в этом месяце**.
        """
    )

# ---------- загрузка файлов (оставили, но теперь опционально)
st.subheader("1) Загрузка файлов (только для обновления БД)")

col1, col2 = st.columns(2)
with col1:
    calvings_file = st.file_uploader("Отёлы + родившиеся", type=["xls", "xlsx"], key="u_calvings")
    disposals_file = st.file_uploader("Выбытие", type=["xls", "xlsx"], key="u_disposals")
with col2:
    dryoff_file = st.file_uploader("Запуски", type=["xls", "xlsx"], key="u_dryoff")
    inseminations_file = st.file_uploader("Осеменения", type=["xls", "xlsx"], key="u_inseminations")

bulls_file = st.file_uploader("Таблица быков (txt)", type=["txt"], key="u_bulls")

# ---------- месяц прогноза
st.subheader("2) Месяц прогноза")

months_ru = [
    "01 — Январь", "02 — Февраль", "03 — Март", "04 — Апрель",
    "05 — Май", "06 — Июнь", "07 — Июль", "08 — Август",
    "09 — Сентябрь", "10 — Октябрь", "11 — Ноябрь", "12 — Декабрь",
]

today = date.today()
sel_col1, sel_col2 = st.columns([1, 1])
with sel_col1:
    year_sel = st.number_input("Год", min_value=2000, max_value=2100, value=today.year, step=1, key="year_sel")
with sel_col2:
    month_sel_label = st.selectbox("Месяц", months_ru, index=today.month - 1, key="month_sel_label")
    month_sel = int(month_sel_label.split("—")[0].strip())

target_month_end = month_end(int(year_sel), int(month_sel))

# ---------- параметры
st.subheader("3) Параметры модели")

# НОВОЕ: кнопка пересчёта параметров из БД (без файлов)
cA, cB, cC = st.columns([1, 1, 2])
with cA:
    if st.button("Пересчитать параметры из БД", use_container_width=True, key="btn_recalc_params_db"):
        try:
            with st.spinner("Пересчитываю параметры из БД..."):
                st.session_state.computed_params = compute_params_from_db()
            st.success("Параметры обновлены из БД.")
        except Exception as e:
            st.error(f"Не удалось пересчитать параметры из БД: {e}")
with cB:
    if st.button("Сбросить кэш", use_container_width=True, key="btn_clear_cache"):
        st.cache_data.clear()
        st.success("Кэш очищен.")

# Автоподхват параметров из БД (если ещё не считали)
if "computed_params" not in st.session_state:
    try:
        with st.spinner("Параметры ещё не считались — пробую посчитать из БД..."):
            st.session_state.computed_params = compute_params_from_db()
    except Exception:
        # если БД пустая/не готова — остаёмся на дефолтах
        pass

base_params = get_param_source()
final_params_for_forecast = _apply_admin_overrides(base_params)

with st.expander("Параметры модели (из данных/дефолтов)", expanded=False):
    st.markdown("### Вместимость (места)")
    st.table(pd.DataFrame([{"Группа": k, "Мест": int(v)} for k, v in mp.HERD_CAPACITY.items()]))

    conc = base_params.get("conception", {}) or {}
    st.markdown("### Стельность (по плодотворным осеменениям)")
    rows = []
    by_l = conc.get("avg_cow_dim_by_lact", {}) or {}
    for lact_cat in (1, 2, 3, 4):
        v = by_l.get(lact_cat)
        rows.append({
            "Лактация": {1: "1-я", 2: "2-я", 3: "3-я", 4: "4+ (и старше)"}[lact_cat],
            "Средний DIM стельности, дни": round(float(v), 1) if v is not None else None
        })
    st.table(pd.DataFrame(rows))
    st.write(f"Средний DIM стельности по коровам (в целом): {float(conc.get('avg_cow_dim_global', 0.0)):.1f} дней")
    st.write(
        f"Средний возраст стельности тёлок: {float(conc.get('avg_heifer_age_days', 0.0)):.1f} дней "
        f"(≈ {float(conc.get('avg_heifer_age_days', 0.0)) / 30:.1f} месяцев)"
    )

    st.markdown("### Длительность стельности и сухостоя")
    st.write(f"Длительность стельности: {float(base_params.get('gestation_days', 0.0)):.1f} дней")
    st.write(f"Длительность сухостоя: {int(base_params.get('dry_days', 0) or 0)} дней")

    st.markdown("### Выбытие коров (форма по DIM и лактациям)")
    st.write(f"Годовой процент выбытия (в модели): {float(base_params.get('annual_disposal_rate', 0.0)) * 100:.2f} %")

    disp_p = base_params.get("disposal_params", {}) or {}
    by_l = (disp_p.get("by_lact", {}) or {})
    overall_n = float((disp_p.get("overall", {}) or {}).get("n", 0) or 0)
    disp_rows = []
    for lact_cat in (1, 2, 3, 4):
        s = by_l.get(lact_cat, {}) or {}
        n = float(s.get("n", 0) or 0)
        share = (n / overall_n * 100.0) if overall_n > 0 else 0.0
        disp_rows.append({
            "Лактация": {1: "1-я", 2: "2-я", 3: "3-я", 4: "4+ (и старше)"}[lact_cat],
            "Доля выбытий среди выбывших, %": round(share, 2),
            "Медианный DIM выбытия, дни": round(float(s.get("median_dim", 0.0) or 0.0), 1),
            "Средний DIM выбытия, дни": round(float(s.get("mean_dim", 0.0) or 0.0), 1),
        })
    st.table(pd.DataFrame(disp_rows))
    st.caption("DIM = число дней после отёла (Days In Milk) на момент события выбытия.")

    st.markdown("### Осеменения (как часто и сколько доз)")
    ins_p = base_params.get("insemination_params", {}) or {}
    st.table(pd.DataFrame([
        {"Показатель": "Коровы: доз на 1 стельность", "Значение": round(float(ins_p.get("cow_services_per_conception", 0.0)), 3)},
        {"Показатель": "Коровы: интервал между осеменениями, дни", "Значение": round(float(ins_p.get("cow_ai_interval_days", 0.0)), 3)},
        {"Показатель": "Тёлки: доз на 1 стельность", "Значение": round(float(ins_p.get("heifer_services_per_conception", 0.0)), 3)},
        {"Показатель": "Тёлки: интервал между осеменениями, дни", "Значение": round(float(ins_p.get("heifer_ai_interval_days", 0.0)), 3)},
        {"Показатель": "Тёлки: возраст первого осеменения, дни", "Значение": round(float(ins_p.get("heifer_first_ai_age_days", 0.0)), 3)},
    ]))

# ---------- Admin parameter editor (оставил как у тебя, без изменений логики)
if st.session_state.is_admin:
    with st.expander("Админ-панель: ручная настройка параметров", expanded=True):
        curr = final_params_for_forecast

        c = curr.get("conception", {}) or {}
        ip = curr.get("insemination_params", {}) or {}
        su = curr.get("semen_usage", {}) or {}
        ssr = curr.get("semen_sex_ratios", {}) or {}

        with st.form("admin_params_form"):
            st.markdown("### Длительности")
            gest = st.number_input("Длительность стельности (дни)", min_value=200.0, max_value=310.0, value=float(curr.get("gestation_days", 272.0)), step=1.0, key="adm_gest")
            dryd = st.number_input("Длительность сухостоя (дни)", min_value=20, max_value=150, value=int(curr.get("dry_days", 53)), step=1, key="adm_dry")

            st.markdown("### Стельность (целевые средние)")
            colA, colB, colC, colD = st.columns(4)
            l1 = float(colA.number_input("Коровы: средний DIM стельности, 1-я лактация", value=float(c.get("avg_cow_dim_by_lact", {}).get(1, 99.0)), step=0.1, key="adm_c_l1"))
            l2 = float(colB.number_input("Коровы: средний DIM стельности, 2-я лактация", value=float(c.get("avg_cow_dim_by_lact", {}).get(2, 107.0)), step=0.1, key="adm_c_l2"))
            l3 = float(colC.number_input("Коровы: средний DIM стельности, 3-я лактация", value=float(c.get("avg_cow_dim_by_lact", {}).get(3, 105.0)), step=0.1, key="adm_c_l3"))
            l4 = float(colD.number_input("Коровы: средний DIM стельности, 4+ лактация", value=float(c.get("avg_cow_dim_by_lact", {}).get(4, 107.0)), step=0.1, key="adm_c_l4"))
            cg = float(st.number_input("Коровы: средний DIM стельности (в целом)", value=float(c.get("avg_cow_dim_global", 104.0)), step=0.1, key="adm_c_g"))
            ha = float(st.number_input("Тёлки: средний возраст стельности (дни)", value=float(c.get("avg_heifer_age_days", 402.0)), step=0.1, key="adm_h_a"))

            st.markdown("### Выбытие")
            adr = float(st.number_input("Годовой процент выбытия (доля в год)", min_value=0.0, max_value=0.5, value=float(curr.get("annual_disposal_rate", 0.0957)), step=0.001, format="%.3f", key="adm_adr"))

            st.markdown("### Осеменения")
            cow_spc = float(st.number_input("Коровы: доз на 1 стельность", min_value=1.0, max_value=6.0, value=float(ip.get("cow_services_per_conception", 2.04)), step=0.01, key="adm_c_spc"))
            cow_int = float(st.number_input("Коровы: интервал между осеменениями (дни)", min_value=10.0, max_value=120.0, value=float(ip.get("cow_ai_interval_days", 46.8)), step=0.5, key="adm_c_int"))
            heif_spc = float(st.number_input("Тёлки: доз на 1 стельность", min_value=1.0, max_value=6.0, value=float(ip.get("heifer_services_per_conception", 1.95)), step=0.01, key="adm_h_spc"))
            heif_int = float(st.number_input("Тёлки: интервал между осеменениями (дни)", min_value=10.0, max_value=120.0, value=float(ip.get("heifer_ai_interval_days", 25.3)), step=0.5, key="adm_h_int"))
            heif_first = float(st.number_input("Тёлки: возраст первого осеменения (дни)", min_value=200.0, max_value=900.0, value=float(ip.get("heifer_first_ai_age_days", 378.5)), step=1.0, key="adm_h_first"))

            st.markdown("DIM первого осеменения по лактациям")
            cf = dict(ip.get("cow_first_ai_dim_by_lact", {1: 71.4, 2: 72.2, 3: 73.3, 4: 72.9}))
            cc1, cc2, cc3, cc4 = st.columns(4)
            cf1 = float(cc1.number_input("1-я лактация", value=float(cf.get(1, 71.4)), step=0.1, key="adm_cf1"))
            cf2 = float(cc2.number_input("2-я лактация", value=float(cf.get(2, 72.2)), step=0.1, key="adm_cf2"))
            cf3 = float(cc3.number_input("3-я лактация", value=float(cf.get(3, 73.3)), step=0.1, key="adm_cf3"))
            cf4 = float(cc4.number_input("4+ лактация", value=float(cf.get(4, 72.9)), step=0.1, key="adm_cf4"))

            st.markdown("### Доля использования семени (если хочешь вручную)")
            su_c1, su_c2, su_c3, su_c4 = st.columns(4)
            cow_trad = float(su_c1.number_input("Коровы: обычное (доля)", min_value=0.0, max_value=1.0, value=float(su.get("cow_trad", 0.7)), step=0.01, key="adm_su_ct"))
            cow_sex = float(su_c2.number_input("Коровы: сексированное (доля)", min_value=0.0, max_value=1.0, value=float(su.get("cow_sex", 0.3)), step=0.01, key="adm_su_cs"))
            heif_trad = float(su_c3.number_input("Тёлки: обычное (доля)", min_value=0.0, max_value=1.0, value=float(su.get("heifer_trad", 0.3)), step=0.01, key="adm_su_ht"))
            heif_sex = float(su_c4.number_input("Тёлки: сексированное (доля)", min_value=0.0, max_value=1.0, value=float(su.get("heifer_sex", 0.7)), step=0.01, key="adm_su_hs"))

            st.markdown("### Пол телят по типу семени (если хочешь вручную)")
            r1, r2 = st.columns(2)
            trad_bull = float(r1.number_input("Обычное: доля бычков", min_value=0.0, max_value=1.0, value=float(ssr.get("trad", {}).get("bull_share", 0.2483)), step=0.01, key="adm_trad_b"))
            sex_bull = float(r2.number_input("Сексированное: доля бычков", min_value=0.0, max_value=1.0, value=float(ssr.get("sex", {}).get("bull_share", 0.0583)), step=0.01, key="adm_sex_b"))

            saved = st.form_submit_button("Сохранить параметры для прогноза", use_container_width=True)
            if saved:
                def _norm2(a: float, b: float) -> Tuple[float, float]:
                    s = max(1e-9, a + b)
                    return a / s, b / s

                cow_trad, cow_sex = _norm2(cow_trad, cow_sex)
                heif_trad, heif_sex = _norm2(heif_trad, heif_sex)

                trad_bull = max(0.0, min(1.0, trad_bull))
                sex_bull = max(0.0, min(1.0, sex_bull))

                st.session_state.runtime_overrides = {
                    "gestation_days": float(gest),
                    "dry_days": int(dryd),
                    "annual_disposal_rate": float(adr),
                    "conception": {
                        "avg_cow_dim_by_lact": {1: l1, 2: l2, 3: l3, 4: l4},
                        "avg_cow_dim_global": cg,
                        "avg_heifer_age_days": ha,
                    },
                    "insemination_params": {
                        "cow_services_per_conception": cow_spc,
                        "cow_ai_interval_days": cow_int,
                        "cow_first_ai_dim_by_lact": {1: cf1, 2: cf2, 3: cf3, 4: cf4},
                        "heifer_services_per_conception": heif_spc,
                        "heifer_ai_interval_days": heif_int,
                        "heifer_first_ai_age_days": heif_first,
                    },
                    "semen_usage": {
                        "cow_trad": cow_trad,
                        "cow_sex": cow_sex,
                        "heifer_trad": heif_trad,
                        "heifer_sex": heif_sex,
                        "meta": {"window": "admin", "n_cow": 0, "n_heifer": 0},
                    },
                    "semen_sex_ratios": {
                        "trad": {"bull_share": trad_bull, "heifer_share": 1.0 - trad_bull},
                        "sex": {"bull_share": sex_bull, "heifer_share": 1.0 - sex_bull},
                        "meta": {"window": "admin", "n_trad": 0, "n_sex": 0},
                    },
                }
                st.success("Сохранено. Нажми «Рассчитать прогноз».")


# ---------- расчёт
st.subheader("4) Расчёт")
calculate = st.button("Рассчитать прогноз", key="btn_calc_forecast", use_container_width=True)

if calculate:
    # если выбрано обновление из файлов — проверяем обязательные файлы и грузим
    if need_files:
        missing = []
        if calvings_file is None:
            missing.append("Отёлы + родившиеся")
        if disposals_file is None:
            missing.append("Выбытие")
        if dryoff_file is None:
            missing.append("Запуски")
        if inseminations_file is None:
            missing.append("Осеменения")

        if missing:
            st.warning("Не все файлы загружены: " + ", ".join(missing))
            st.stop()

        # load into DB
        try:
            calv_df = read_calvings_excel(calvings_file)
            load_calvings_to_db(calv_df, if_exists="replace")
        except Exception as e:
            st.error(f"Ошибка при загрузке 'Отёлы + родившиеся': {e}")
            st.stop()

        try:
            disp_df = read_disposals_excel(disposals_file)
            load_disposals_to_db(disp_df, if_exists="replace")
        except Exception as e:
            st.error(f"Ошибка при загрузке 'Выбытие': {e}")
            st.stop()

        try:
            dry_df = read_dryoff_excel(dryoff_file)
            load_dryoff_to_db(dry_df, if_exists="replace")
        except Exception as e:
            st.error(f"Ошибка при загрузке 'Запуски': {e}")
            st.stop()

        try:
            ins_df = read_inseminations_excel(inseminations_file)
            ins_df = clean_inseminations(ins_df)
            load_inseminations_to_db(ins_df, if_exists="replace")
        except Exception as e:
            st.error(f"Ошибка при загрузке 'Осеменения': {e}")
            st.stop()

        if bulls_file is not None:
            try:
                bulls_df = read_bulls_txt(bulls_file)
                load_bulls_to_db(bulls_df, if_exists="replace")
            except Exception as e:
                st.error(f"Ошибка при разборе 'Таблица быков': {e}")

        # recompute params после обновления
        try:
            with st.spinner("Пересчитываю параметры из загруженных данных..."):
                st.session_state.computed_params = compute_params_from_db()
        except Exception as e:
            st.error(f"Не удалось пересчитать параметры из данных: {e}")
            st.stop()

    # период прогноза берём от последней даты в БД
    base_date = get_max_event_date_from_db()
    base_month_end = month_end(base_date.year, base_date.month)

    if target_month_end < base_month_end:
        month_ends = [target_month_end]
    else:
        month_ends = iter_month_ends(base_date.year, base_date.month, target_month_end.year, target_month_end.month)

    st.markdown(f"**Период прогноза:** {month_ends[0].strftime('%m.%Y')} → {month_ends[-1].strftime('%m.%Y')}")

    base_params = get_param_source()
    final_params_for_forecast = _apply_admin_overrides(base_params)

    rows = []
    real_rows = []
    prog = st.progress(0.0)

    with st.spinner("Считаю прогноз..."):
        for i, d_end in enumerate(month_ends, start=1):
            try:
                # НОВОЕ: передаём Timestamp, чтобы не ловить dtype=datetime64 vs date
                vals = compute_forecast_from_db(pd.Timestamp(d_end), overrides=final_params_for_forecast)
            except Exception as e:
                st.error(f"Ошибка расчёта на {d_end.strftime('%Y-%m')}: {e}")
                vals = {}

            row = {"Месяц": d_end.strftime("%Y-%m")}
            for k in INDICATORS:
                row[k] = vals.get(k)
            rows.append(row)

            real_row = {"Месяц": d_end.strftime("%Y-%m")}
            for k in REALIZATION_COLS:
                real_row[k] = vals.get(k, 0.0)
            real_rows.append(real_row)

            prog.progress(i / max(1, len(month_ends)))

    prog.empty()

    result = pd.DataFrame(rows).set_index("Месяц")
    real_df = pd.DataFrame(real_rows).set_index("Месяц")

    # НОВОЕ: подсветка КРАСНЫМ В ПРОГНОЗЕ, если есть переполнение (или фактически выше вместимости)
    indicator_to_overflow = {
        "Дойные коровы": "Переполнение: Дойные коровы",
        "Сухостойные коровы": "Переполнение: Сухостойные коровы",
        "Тёлки 0–2 мес": "Переполнение: Тёлки 0–3 мес",
        "Тёлки 3–8 мес": "Переполнение: Тёлки 3–8 мес",
        "Тёлки ≥9 мес": "Переполнение: Тёлки 9–24 мес",
        "Нетели": "Переполнение: Нетели",
    }

    def style_forecast(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # 1) если фактически > вместимости — ярко красим
        for col in df.columns:
            cap_key = capacity_name_for_indicator(col)
            cap = mp.HERD_CAPACITY.get(cap_key) if cap_key else None
            if cap is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna() & (vals > float(cap))
            styles.loc[mask, col] = "background-color:#ff0000;color:white"

        # 2) если модель считает "переполнение" (т.е. нужно продавать) — красим в прогнозе (помягче)
        for ind_col, ov_col in indicator_to_overflow.items():
            if ind_col not in df.columns or ov_col not in real_df.columns:
                continue
            ov = pd.to_numeric(real_df[ov_col], errors="coerce").fillna(0.0)
            mask = ov > 0.0
            # не перебиваем ярко-красное (если уже стоит)
            for idx in df.index[mask]:
                if "background-color:#ff0000" not in styles.loc[idx, ind_col]:
                    styles.loc[idx, ind_col] = "background-color:#ffcccc"

        return styles

    st.subheader("Прогноз (красным подсвечены месяцы, где начинается переполнение по местам)")
    st.dataframe(result.style.apply(style_forecast, axis=None), use_container_width=True)

    st.subheader("5) Реализация (рекомендация продать, чтобы влезть по местам)")
    sales_cols = [c for c in ["К реализации: коровы", "К реализации: тёлки", "К реализации: нетели"] if c in real_df.columns]
    sales_df = real_df[sales_cols].copy() if sales_cols else real_df.copy()
    st.dataframe(sales_df, use_container_width=True)

    with st.expander("Подробно: переполнение по группам (для проверки логики)", expanded=False):
        overflow_cols = [c for c in real_df.columns if c.startswith("Переполнение:")]
        if overflow_cols:
            st.dataframe(real_df[overflow_cols], use_container_width=True)
        else:
            st.info("Нет колонок переполнения в данных.")

    st.caption("Подсветка красным теперь только в прогнозе. Таблица реализации без подсветки.")

    # ---------- НОВОЕ: скачать одним Excel (2 листа)
    st.subheader("Скачать результат")
    excel_bytes = make_excel_bytes(result, real_df)
    st.session_state["last_excel_bytes"] = excel_bytes
    st.download_button(
        label="Скачать Excel: прогноз + реализация",
        data=excel_bytes,
        file_name=f"herd_forecast_{month_ends[0].strftime('%Y-%m')}_to_{month_ends[-1].strftime('%Y-%m')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_excel",
    )

else:
    st.info("Выбери источник данных и нажми «Рассчитать прогноз». Если хочешь обновить БД — выбери режим обновления и загрузи файлы.")
    if "last_excel_bytes" in st.session_state:
        st.download_button(
            label="Скачать последний Excel: прогноз + реализация",
            data=st.session_state["last_excel_bytes"],
            file_name="herd_forecast_last.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_excel_last",
        )
