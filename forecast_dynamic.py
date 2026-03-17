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

from datetime import date, timedelta
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from db import engine
from forecast_dynamic_normalization import (
    SemenSexRatio,
    classify_semen_from_bull_type,
    classify_semen_from_bull_type_strict,
    is_transfer_disposal_reason,
    norm_event_type,
    norm_gender,
    norm_id,
    norm_result,
    norm_sex,
    to_semen_ratio as _to_semen_ratio,
)
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

import re
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


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

    born["birth_dt"] = born["birth_date_n"]
    m = born["birth_dt"].isna() & born["event_date_n"].notna()
    born.loc[m, "birth_dt"] = born.loc[m, "event_date_n"]

    born = born[born["birth_dt"].notna() & (born["birth_dt"] <= as_of_ts)].copy()

                                                                
    born = (
        born.sort_values(["reg_s", "birth_dt"], kind="mergesort")
            .groupby("reg_s", sort=False, as_index=False)
            .first()[["reg_s", "birth_dt", "sex_norm"]]
    )
    return born
MAX_DIM = 500
MAX_AGE_DAYS = 730
BULL_AGE_MAX = 90                                             
OVERDUE_CLAMP_DAYS = 14

BIRTH_OUTPUT_KEYS = (
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
)


def age_months(d: int) -> int:
    return int(d // 30)


def end_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def _month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return end_of_month(date(int(ts.year), int(ts.month), 1))


def _months_between_eom(start_eom: date, end_eom: date) -> int:
    return (int(end_eom.year) - int(start_eom.year)) * 12 + (int(end_eom.month) - int(start_eom.month))


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

    return 0.30 * interval_raw + 0.70 * derived

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _normalize_month_factor_map(raw: Any) -> dict[int, float]:
    # Сезонные коэффициенты пока намеренно отключены.
    return {m: 1.0 for m in range(1, 13)}


def _normalize_semen_usage_shares(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None

    cow_sex = raw.get("cow_sex")
    cow_trad = raw.get("cow_trad")
    heifer_sex = raw.get("heifer_sex")
    heifer_trad = raw.get("heifer_trad")

    if cow_sex is None and cow_trad is None and heifer_sex is None and heifer_trad is None:
        return None

    def _pair(a_raw: Any, b_raw: Any, a_fb: float, b_fb: float) -> tuple[float, float]:
        a = None if a_raw is None else _clamp(float(a_raw), 0.0, 1.0)
        b = None if b_raw is None else _clamp(float(b_raw), 0.0, 1.0)
        if a is None and b is None:
            a, b = float(a_fb), float(b_fb)
        elif a is None:
            a = 1.0 - float(b)
        elif b is None:
            b = 1.0 - float(a)
        s = max(1e-9, float(a) + float(b))
        return float(a) / s, float(b) / s

    cow_trad_n, cow_sex_n = _pair(
        cow_trad,
        cow_sex,
        float(SEMEN_USAGE_PROBS.cow_trad),
        float(SEMEN_USAGE_PROBS.cow_sex),
    )
    heifer_trad_n, heifer_sex_n = _pair(
        heifer_trad,
        heifer_sex,
        float(SEMEN_USAGE_PROBS.heifer_trad),
        float(SEMEN_USAGE_PROBS.heifer_sex),
    )
    return {
        "cow_trad": cow_trad_n,
        "cow_sex": cow_sex_n,
        "heifer_trad": heifer_trad_n,
        "heifer_sex": heifer_sex_n,
    }


def _month_factor_value(month_factors: dict[int, float], dt_like: Any) -> float:
    try:
        month = int(pd.Timestamp(dt_like).month)
    except Exception:
        return 1.0
    return float(month_factors.get(month, 1.0))


                                                              
                                                              

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


                                                              
                                                              

def _resolve_runtime_params(overrides: dict | None) -> dict:
    ov = overrides or {}

    def _as_bool(v: object, default: bool) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return default

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
        "cow_conception_month_factors": dict(INSEMINATION_PARAMS.cow_conception_month_factors),
        "heifer_services_per_conception": float(INSEMINATION_PARAMS.heifer_services_per_conception),
        "heifer_ai_interval_days": float(INSEMINATION_PARAMS.heifer_ai_interval_days),
        "heifer_first_ai_age_days": float(INSEMINATION_PARAMS.heifer_first_ai_age_days),
        "heifer_conception_month_factors": dict(INSEMINATION_PARAMS.heifer_conception_month_factors),
    }
    ins["cow_conception_month_factors"] = _normalize_month_factor_map(
        ins.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    ins["heifer_conception_month_factors"] = _normalize_month_factor_map(
        ins.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    semen_usage = _normalize_semen_usage_shares(ov.get("SEMEN_USAGE_SHARES"))
    if semen_usage is None:
        semen_usage = _normalize_semen_usage_shares(ov.get("semen_usage"))

    cap_norm = dict(_CAP_NORM)
    cap_ov = ov.get("HERD_CAPACITY")
    if cap_ov is None:
        cap_ov = ov.get("herd_capacity")
    if isinstance(cap_ov, dict):
        for k, v in cap_ov.items():
            try:
                iv = int(round(float(v)))
            except Exception:
                continue
            cap_norm[_norm_key(str(k))] = max(0, iv)

    gest_days = int(round(float(ov.get("GESTATION_DAYS", gest_default))))
    gest_days = max(200, min(310, gest_days))

    dry_days = int(round(float(ov.get("DRY_DAYS", dry_default))))
    dry_days = max(20, min(120, dry_days))

    annual_disp = float(max(0.0, min(0.5, annual_disp)))
    apply_capacity = _as_bool(ov.get("APPLY_CAPACITY"), True)
    if _as_bool(ov.get("DISABLE_CAPACITY"), False):
        apply_capacity = False

    return {
        "GESTATION_DAYS": gest_days,
        "DRY_DAYS": dry_days,
        "CONCEPTION_PARAMS": cp,
        "DISPOSAL_PARAMS": disp,
        "ANNUAL_DISPOSAL_RATE": annual_disp,
        "INSEMINATION_PARAMS": ins,
        "SEMEN_USAGE_SHARES": semen_usage,
        "HERD_CAPACITY_NORM": cap_norm,
        "APPLY_CAPACITY": apply_capacity,
    }


                                                              
                                                              

@dataclass
class HerdState:
            
    open_dim: Dict[int, np.ndarray]
    preg_lact: Dict[Tuple[int, str], np.ndarray]
    preg_dry:  Dict[Tuple[int, str], np.ndarray]
                    
    heifer_age: np.ndarray
    heifer_preg: Dict[str, np.ndarray]
    bull_age: np.ndarray


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
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    svc = ins[(ins["event_date"].notna()) & (ins["reg_s"] != "") & (ins["bull_s"] != "")].copy()
    if svc.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type_strict)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    svc["semen"] = svc["bull_s"].map(semen_by_bull)
    svc["semen_known"] = svc["semen"].isin(["trad", "sex"])

                                                                     
    cows = svc[svc["lact"] > 0].copy()
    heif = svc[svc["lact"] <= 0].copy()

    if not cows.empty:
        cows = cows.sort_values(["reg_s", "lact", "event_date"], kind="mergesort")
        cows["service_no"] = cows.groupby(["reg_s", "lact"], sort=False).cumcount() + 1
        cows["policy_sex_allowed"] = cows["lact"].isin([1, 2]) & (cows["service_no"] <= 2)

    if not heif.empty:
        heif = heif.sort_values(["reg_s", "event_date"], kind="mergesort")
        heif["service_no"] = heif.groupby(["reg_s"], sort=False).cumcount() + 1
        heif["policy_sex_allowed"] = heif["service_no"] <= 3

    s = pd.concat([cows, heif], axis=0, ignore_index=True)
    if s.empty:
        return fallback

    max_dt = s["event_date"].max()
    s_365 = s[s["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy() if pd.notna(max_dt) else s.copy()

    def _bayes_smooth_share(obs: float, n: int, prior: float, prior_w: float) -> float:
        return float((obs * n + prior * prior_w) / max(1e-9, n + prior_w))

    def _shares(df: pd.DataFrame) -> Tuple[float | None, float, int, int, float]:
        if df.empty:
            return None, 0.0, 0, 0, 0.0
        total_n = int(len(df))
        policy_sex = float(df["policy_sex_allowed"].mean()) if "policy_sex_allowed" in df.columns else 0.0
        known = df[df["semen_known"]]
        if known.empty:
            return None, policy_sex, total_n, 0, 0.0
        known_n = int(len(known))
        sex_obs = float((known["semen"] == "sex").mean())
        known_rate = float(known_n) / float(total_n)
        return sex_obs, policy_sex, total_n, known_n, known_rate

    def _mix_group(df_all: pd.DataFrame, df_365: pd.DataFrame, prior_sex: float) -> Tuple[float, dict]:
                                                                  
        use_recent = len(df_365) >= 120
        src = df_365 if use_recent else df_all
        window = "last_year" if use_recent else "all"

        sex_obs, policy_sex, total_n, known_n, known_rate = _shares(src)
        if total_n == 0:
            return prior_sex, {"window": window, "n_total": 0, "n_known": 0, "known_rate": 0.0, "policy_sex": 0.0, "obs_sex": None}

                                                               
        if sex_obs is None:
            blended = policy_sex
        else:
            blended = known_rate * sex_obs + (1.0 - known_rate) * policy_sex

        sex_final = _bayes_smooth_share(
            obs=float(max(0.0, min(1.0, blended))),
            n=total_n,
            prior=float(max(0.0, min(1.0, prior_sex))),
            prior_w=300.0,
        )
        sex_final = float(max(0.0, min(1.0, sex_final)))
        meta = {
            "window": window,
            "n_total": int(total_n),
            "n_known": int(known_n),
            "known_rate": float(known_rate),
            "policy_sex": float(policy_sex),
            "obs_sex": None if sex_obs is None else float(sex_obs),
        }
        return sex_final, meta

    cows_all = s[s["lact"] > 0].copy()
    cows_365 = s_365[s_365["lact"] > 0].copy()
    heif_all = s[s["lact"] <= 0].copy()
    heif_365 = s_365[s_365["lact"] <= 0].copy()

    cow_sex, meta_cow = _mix_group(cows_all, cows_365, fallback["cow_sex"])
    hef_sex, meta_heif = _mix_group(heif_all, heif_365, fallback["heifer_sex"])
    cow_trad = 1.0 - cow_sex
    hef_trad = 1.0 - hef_sex

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
        "meta": {
            "method": "services_with_policy_blend",
            "cow": meta_cow,
            "heifer": meta_heif,
        },
    }


                                                              
                                                              

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
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type_strict)
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

                                                                                       
        if bull_share < 0.05 or bull_share > 0.95:
            continue

        bull_share = max(0.10, min(0.90, bull_share))
        out[semen] = SemenSexRatio(bull_share=bull_share, heifer_share=1.0 - bull_share)


    return out


                                                              
                                                              

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


                                                              
                                                              

import re

def _norm_key(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s.upper()

_CAP_NORM = {_norm_key(k): int(v) for k, v in HERD_CAPACITY.items()}


def _cap(name: str, cap_norm: dict[str, int] | None = None) -> int | None:
    cap = cap_norm if isinstance(cap_norm, dict) else _CAP_NORM
    return cap.get(_norm_key(name))



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


def _apply_capacity_month_end(
    state: HerdState,
    *,
    gest_days: int,
    dry_days: int,
    cap_norm: dict[str, int] | None = None,
) -> dict:
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

    cap_doy = _cap("Дойные коровы", cap_norm)
    cap_dry = _cap("Сухостойные коровы", cap_norm)
    cap_h0 = _cap("Тёлки 0–3 мес", cap_norm)
    cap_h38 = _cap("Тёлки 3–8 мес", cap_norm)
    cap_h924 = _cap("Тёлки 9–24 мес", cap_norm)                                            
    cap_neteli = _cap("Нетели", cap_norm)                                                                        

                                                   
    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    h0 = float(state.heifer_age[:90].sum())                     
    h38 = float(state.heifer_age[90:270].sum())               
    h9 = float(state.heifer_age[270:].sum())                  
    neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

                             
    if cap_doy is not None and doy > cap_doy + 1e-9:
        need = doy - cap_doy
        sold = _sell_cows_from_doy(state, need, gest_days)
        out["over_doy"] += sold
        out["sell_cows"] += sold

    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    if cap_dry is not None and dry > cap_dry + 1e-9:
        need = dry - cap_dry
        sold = _sell_cows_from_dry(state, need, gest_days, dry_days)
        out["over_dry"] += sold
        out["sell_cows"] += sold

                                 
    if cap_h0 is not None and h0 > cap_h0 + 1e-9:
        need = h0 - cap_h0
        sold = _sell_heifers_by_age(state, need, 0, 89)
        out["over_h0"] += sold
        out["sell_heifers"] += sold

    if cap_h38 is not None and h38 > cap_h38 + 1e-9:
        need = h38 - cap_h38
        sold = _sell_heifers_by_age(state, need, 90, 269)
        out["over_h38"] += sold
        out["sell_heifers"] += sold

    if cap_h924 is not None:
        h9 = float(state.heifer_age[270:].sum())
        neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

        total_9plus = float(h9 + neteli)
        if total_9plus > cap_h924 + 1e-9:
            need = total_9plus - cap_h924

            sold_h9 = 0.0
            sold_n = 0.0

                                                                           
            if h9 > 1e-9 and need > 1e-9:
                take_h9 = min(h9, need)
                sold_h9 = _sell_heifers_by_age(state, take_h9, 270, MAX_AGE_DAYS)
                need = max(0.0, need - sold_h9)

                                      
            if need > 1e-9:
                sold_n = _sell_neteli_4_6_months(state, need, gest_days)

            out["over_h9"] += float(sold_h9)
            out["over_neteli"] += float(sold_n)
            out["sell_heifers"] += float(sold_h9)
            out["sell_neteli"] += float(sold_n)

                                                                                               
    if cap_neteli is not None:
        neteli2 = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())
        if neteli2 > cap_neteli + 1e-9:
            need = neteli2 - cap_neteli
            sold = _sell_neteli_4_6_months(state, need, gest_days)
            out["over_neteli"] += float(sold)
            out["sell_neteli"] += float(sold)

    return out


                                                              
                                                              

def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4


                                                              
                                                              

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

    if not ins.empty:
        tmp = ins.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["lact_i"] = pd.to_numeric(tmp["lact"], errors="coerce").fillna(0).astype(int)
        out |= set(tmp.loc[tmp["lact_i"] > 0, "reg_s"].astype(str))

    if not dry.empty:
        tmp = dry.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        out |= set(tmp["reg_s"].astype(str))

    if not calv.empty:
        tmp = calv.copy()
        tmp["event_type_n"] = tmp["event_type"].apply(norm_event_type)
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["mother_reg_s"] = tmp["mother_reg"].apply(norm_id)
        out |= set(tmp.loc[tmp["mother_reg_s"] != "", "mother_reg_s"].astype(str))
        out |= set(tmp.loc[(tmp["event_type_n"] == "ОТЕЛ") & (tmp["reg_s"] != ""), "reg_s"].astype(str))

    out.discard("")
    return out


def _estimate_active_cow_regs_at_asof(
    *,
    calv: pd.DataFrame,
    ins: pd.DataFrame,
    dry: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    lookback_days: int = 540,
) -> set[str]:
    """
    Оценка "фактически присутствующих" коров на дату старта прогноза.
    Используем события за последние ~18 месяцев:
    - осеменения lact>0,
    - отёлы (ОТЕЛ по reg),
    - матери в строках РОЖДЕН (mother_reg),
    - запуски (dryoff).
    """
    lo = as_of_ts - pd.Timedelta(days=int(max(120, lookback_days)))
    out: set[str] = set()

    if isinstance(ins, pd.DataFrame) and not ins.empty:
        d = ins.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        m = (
            d["event_date_n"].notna()
            & (d["event_date_n"] >= lo)
            & (d["event_date_n"] <= as_of_ts)
            & (d["lact_n"] > 0)
            & (d["reg_s"] != "")
        )
        out |= set(d.loc[m, "reg_s"].astype(str))

    if isinstance(calv, pd.DataFrame) and not calv.empty:
        d = calv.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["event_type_n"] = d.get("event_type", pd.Series(dtype=object)).apply(norm_event_type)
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        d["mother_reg_s"] = d.get("mother_reg", pd.Series(dtype=object)).apply(norm_id)
        base = d["event_date_n"].notna() & (d["event_date_n"] >= lo) & (d["event_date_n"] <= as_of_ts)
        m1 = base & (d["event_type_n"] == "ОТЕЛ") & (d["reg_s"] != "")
        m2 = base & (d["event_type_n"] == "РОЖДЕН") & (d["mother_reg_s"] != "")
        out |= set(d.loc[m1, "reg_s"].astype(str))
        out |= set(d.loc[m2, "mother_reg_s"].astype(str))

    if isinstance(dry, pd.DataFrame) and not dry.empty:
        d = dry.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        m = (
            d["event_date_n"].notna()
            & (d["event_date_n"] >= lo)
            & (d["event_date_n"] <= as_of_ts)
            & (d["reg_s"] != "")
        )
        out |= set(d.loc[m, "reg_s"].astype(str))

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

    if "event_type" in calv2.columns:
        calv2["event_type_n"] = calv2["event_type"].map(norm_event_type)
    else:
        calv2["event_type_n"] = None

    if "event_date" in calv2.columns:
        calv2["event_date_n"] = pd.to_datetime(calv2["event_date"], errors="coerce").dt.normalize()
    else:
        calv2["event_date_n"] = pd.NaT

    if "reg" in calv2.columns:
        calv2["reg_s"] = calv2["reg"].map(norm_id)
    else:
        calv2["reg_s"] = ""

    if "mother_reg" in calv2.columns:
        calv2["mother_reg_s"] = calv2["mother_reg"].map(norm_id)
    else:
        calv2["mother_reg_s"] = ""

    src = "gndr" if "gndr" in calv2.columns else ("sex" if "sex" in calv2.columns else None)
    if src is not None:
        calv2["gndr_n"] = calv2[src].map(norm_gender)
    else:
        calv2["gndr_n"] = None



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
            if 0 <= age < len(state.bull_age):
                state.bull_age[age] += 1.0

        else:
            state.heifer_age[age] += float(ratio.heifer_share)
            state.bull_age[age] += float(ratio.bull_share)


def _warmstart_heifer_preg_from_stock_if_empty(
    *,
    state: HerdState,
    ins_params: dict,
    gest_days: int,
) -> float:
    """
    Если на старте нет ни одной нетели (heifer_preg == 0), но есть зрелые тёлки,
    добавляем мягкий warm-start нетелей из age-структуры.

    Это защищает от искусственных нулей в подразделениях, где в выгрузке
    не хватает явных осеменений тёлок (lact<=0), но молодняк фактически есть.
    """
    cur = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())
    if cur > 1e-9:
        return 0.0

    first_ai_age = float(
        ins_params.get("heifer_first_ai_age_days", float(INSEMINATION_PARAMS.heifer_first_ai_age_days))
    )
    first_h = int(_clamp(first_ai_age, 0.0, float(MAX_AGE_DAYS)))
    if first_h >= len(state.heifer_age):
        return 0.0

    eligible = float(state.heifer_age[first_h:].sum())
    min_eligible = float(ins_params.get("heifer_zero_warmstart_min_eligible", 50.0))
    if eligible < max(1.0, min_eligible):
        return 0.0

    heif_spc = float(
        ins_params.get("heifer_services_per_conception", float(INSEMINATION_PARAMS.heifer_services_per_conception))
    )
                                                  
    base_conc = 1.0 / max(1e-9, heif_spc)
    scale = float(ins_params.get("heifer_zero_warmstart_scale", 0.25))
    preg_frac = _clamp(base_conc * scale, 0.02, 0.35)
    n_preg = eligible * preg_frac
    if n_preg <= 1e-9:
        return 0.0

    lo = int(_clamp(float(ins_params.get("heifer_zero_warmstart_gest_lo_days", 0.0)), 0.0, float(gest_days)))
    hi = int(_clamp(float(ins_params.get("heifer_zero_warmstart_gest_hi_days", float(gest_days))), 0.0, float(gest_days)))
    if hi < lo:
        lo, hi = hi, lo
    bins = max(1, hi - lo + 1)

    sex_share = _clamp(float(ins_params.get("heifer_zero_warmstart_sex_share", float(SEMEN_USAGE_PROBS.heifer_sex))), 0.0, 1.0)
    add_per_day = n_preg / float(bins)

    state.heifer_preg["sex"][lo : hi + 1] += add_per_day * sex_share
    state.heifer_preg["trad"][lo : hi + 1] += add_per_day * (1.0 - sex_share)

    deplete = min(0.95, n_preg / max(1e-9, eligible))
    state.heifer_age[first_h:] *= (1.0 - deplete)
    return float(n_preg)


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

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce").dt.normalize()
    dry["reg_s"] = dry["reg"].apply(norm_id)

    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.normalize()
    disp["reg_s"] = disp["reg"].apply(norm_id)

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    disp["reason_is_transfer"] = disp["disposal_reason"].apply(is_transfer_disposal_reason)
    disposed_regs = set(
        disp.loc[
            disp["event_date"].notna()
            & (disp["event_date"] <= as_of_ts)
            & (~disp["reason_is_transfer"]),
            "reg_s",
        ].astype(str)
    )
    disposed_regs.discard("")

    dry_ok = dry[(dry["event_date"].notna()) & (dry["event_date"] <= as_of_ts) & (dry["reg_s"] != "")]
    dry_last = dry_ok.groupby("reg_s", sort=False)["event_date"].max().to_dict()

    calv2 = calv.copy()
    calv2["event_type_n"] = calv2["event_type"].apply(norm_event_type)
    calv2["event_date_n"] = pd.to_datetime(calv2["event_date"], errors="coerce").dt.normalize()
    calv2["reg_s"] = calv2["reg"].apply(norm_id)
    calv2["mother_reg_s"] = calv2["mother_reg"].apply(norm_id)
    calv2 = calv2[(calv2["event_date_n"].notna()) & (calv2["event_date_n"] <= as_of_ts)].copy()

    calves_born = calv2[
        (calv2["event_type_n"] == "РОЖДЕН")
        & (calv2["mother_reg_s"] != "")
        & (calv2["event_date_n"].notna())
    ][["mother_reg_s", "event_date_n"]].drop_duplicates()

    calves_otel = calv2[
        (calv2["event_type_n"] == "ОТЕЛ")
        & (calv2["reg_s"] != "")
        & (calv2["event_date_n"].notna())
    ][["reg_s", "event_date_n"]].drop_duplicates()

    calving_events_parts: list[pd.DataFrame] = []
    if not calves_born.empty:
        calving_events_parts.append(
            calves_born.rename(columns={"mother_reg_s": "reg_s", "event_date_n": "calving_date"})
        )
    if not calves_otel.empty:
        calving_events_parts.append(
            calves_otel.rename(columns={"event_date_n": "calving_date"})
        )

    calv_stats = None
    if calving_events_parts:
        calving_events = (
            pd.concat(calving_events_parts, ignore_index=True)
            .drop_duplicates(subset=["reg_s", "calving_date"], keep="last")
        )
        calv_stats = (
            calving_events.groupby("reg_s", sort=False)
            .agg(
                n_calvings=("calving_date", "count"),
                last_calving=("calving_date", "max"),
            )
            .reset_index()
        )

    ins_cow_hist = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] > 0)
    ].copy()

    est_stats = None
    if not ins_cow_hist.empty:
        ins_cow_hist = ins_cow_hist.sort_values(["reg_s", "event_date"], kind="mergesort")
        last_dim_row = ins_cow_hist.groupby("reg_s", sort=False).tail(1).copy()
        last_dim_row["dim_age"] = pd.to_numeric(last_dim_row["dim_age"], errors="coerce")
        valid_dim = last_dim_row["dim_age"].notna() & (last_dim_row["dim_age"] >= 0)
        last_dim_row["last_calving_est"] = pd.NaT
        if bool(valid_dim.any()):
            last_dim_row.loc[valid_dim, "last_calving_est"] = (
                last_dim_row.loc[valid_dim, "event_date"]
                - pd.to_timedelta(last_dim_row.loc[valid_dim, "dim_age"], unit="D")
            )
        last_dim_row["lact_cat_est"] = last_dim_row["lact"].clip(lower=1, upper=4)
        est_stats = last_dim_row[["reg_s", "last_calving_est", "lact_cat_est", "dim_age"]].copy()

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

                                                                                
                                                                                        
    active_cow_regs = _estimate_active_cow_regs_at_asof(
        calv=calv2,
        ins=ins,
        dry=dry,
        as_of_ts=as_of_ts,
        lookback_days=540,
    )
    if active_cow_regs:
        cows_active = cows[cows["reg_s"].isin(active_cow_regs)].copy()
        if not cows_active.empty:
            cows = cows_active

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

    ins_params = insemination_params or {}
    cow_spc = float(ins_params.get("cow_services_per_conception", float(INSEMINATION_PARAMS.cow_services_per_conception)))
    heif_spc = float(ins_params.get("heifer_services_per_conception", float(INSEMINATION_PARAMS.heifer_services_per_conception)))
    cow_month_factors = _normalize_month_factor_map(
        ins_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        ins_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    p_conc_cow_base = 1.0 / max(1e-9, cow_spc)

    p_conc_heif_raw = 1.0 / max(1e-9, heif_spc)
    p_conc_heif_base = float(ins_params.get("heifer_warmstart_p", p_conc_heif_raw))

    SERVICE_RESULTS = {"", "O", "О"}

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
                        month_factor = _month_factor_value(cow_month_factors, s_date)
                        add = _clamp(p_conc_cow_base * month_factor, 0.05, 0.95)
                        open_add = 1.0 - add
                        is_dry2 = is_dry_fact or (days_to_calv2 <= dry_days)
                        if is_dry2:
                            state.preg_dry[(lact_cat, semen2)][days_to_calv2] += add
                        else:
                            state.preg_lact[(lact_cat, semen2)][days_to_calv2] += add

        state.open_dim[lact_cat][dim] += open_add

                                                                  
                                                                     
                                                                  

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

            month_factor = _month_factor_value(heifer_month_factors, s_date)
            add = _clamp(p_conc_heif_base * month_factor, 0.03, 0.35)
            state.heifer_preg[semen][days_to_calv] += add

                                                                  
                                                        
                                                              
                                                                  

    if semen_sex_ratios is None:
        semen_sex_ratios = {
            "trad": _to_semen_ratio(SEMEN_SEX_RATIOS["trad"]),
            "sex": _to_semen_ratio(SEMEN_SEX_RATIOS["sex"]),
        }

    seed_days = int(float(ins_params.get("youngstock_seed_days", 540) or 540))
    seed_days = max(120, min(seed_days, int(MAX_AGE_DAYS)))                    

    calv_seed = calv.copy()
    calv_seed["event_date_n"] = pd.to_datetime(calv_seed["event_date"], errors="coerce").dt.normalize()
    calv_seed = calv_seed[
        (calv_seed["event_date_n"].notna())
        & (calv_seed["event_date_n"] <= as_of_ts)
        & (calv_seed["event_date_n"] >= (as_of_ts - pd.Timedelta(days=seed_days)))
    ].copy()

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

                if age_val < 150 or age_val > MAX_AGE_DAYS:
                    continue

                age = int(max(0, min(MAX_AGE_DAYS, int(age_val))))
                state.heifer_age[age] += 1.0

    _warmstart_heifer_preg_from_stock_if_empty(
        state=state,
        ins_params=ins_params,
        gest_days=gest_days,
    )

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

        j = i - lead_neteli_months
        if j >= 0 and need > 0:
            m_sell = cols[j]

            cap_by_flow = float(calv_from_neteli[m])
            cap_by_stock = max(0.0, float(stock_neteli[m_sell]) - float(plan_neteli[m_sell]))

            add = min(need, cap_by_flow, cap_by_stock)
            if add > 0:
                plan_neteli[m_sell] += add
                need -= add

                                                                        
        k = i - lead_heifer9_months
        if k >= 0 and need > 0:
            m_sell = cols[k]

            cap_by_stock = max(0.0, float(stock_h9[m_sell]) - float(plan_h9[m_sell]))
            add = min(need, cap_by_stock)
            if add > 0:
                plan_h9[m_sell] += add
                need -= add

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

                                                              
                                                              

def simulate_to_target(
    state: HerdState,
    *,
    start: date,
    target: date,
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    params: dict,
) -> Tuple[HerdState, Dict[str, float]]:

    start_ts = pd.Timestamp(start).normalize()
    target_ts = pd.Timestamp(target).normalize()

    end_sim_ts = pd.Timestamp(end_of_month(target_ts.date())).normalize()

    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])
    cp = params["CONCEPTION_PARAMS"]
    disp_params = params["DISPOSAL_PARAMS"]
    annual_disp = float(params["ANNUAL_DISPOSAL_RATE"])
    ins_p = params["INSEMINATION_PARAMS"]

    target_month = (int(target_ts.year), int(target_ts.month))
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

    def _process_bucket0_for_day(curr_day_ts: pd.Timestamp) -> None:
        nonlocal calv_total, calv_cows, calv_heifers, exp_bulls, exp_heifers

        curr_month = (int(curr_day_ts.year), int(curr_day_ts.month))

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                born = float(state.preg_dry[(l, semen)][0])
                if born > 0:
                    state.preg_dry[(l, semen)][0] = 0.0

                    if curr_month == target_month:
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

                if curr_month == target_month:
                    calv_total += born
                    calv_heifers += born
                    sr = semen_sex_ratios[semen]
                    exp_bulls += born * float(sr.bull_share)
                    exp_heifers += born * float(sr.heifer_share)

                state.open_dim[1][0] += born
                sr = semen_sex_ratios[semen]
                state.heifer_age[0] += born * float(sr.heifer_share)
                state.bull_age[0] += born * float(sr.bull_share)

    _process_bucket0_for_day(start_ts)
                                                                     
    if params.get("APPLY_CAPACITY", True) and pd.Timestamp(start).normalize() == pd.Timestamp(end_of_month(start)).normalize():
        sold0 = _apply_capacity_month_end(
            state,
            gest_days=gest_days,
            dry_days=dry_days,
            cap_norm=params.get("HERD_CAPACITY_NORM"),
        )
        if (start.year, start.month) == target_month:
            meta["sell_cows"] += float(sold0["sell_cows"])
            meta["sell_heifers"] += float(sold0["sell_heifers"])
            meta["sell_neteli"] += float(sold0["sell_neteli"])
            meta["over_doy"] += float(sold0["over_doy"])
            meta["over_dry"] += float(sold0["over_dry"])
            meta["over_h0"] += float(sold0["over_h0"])
            meta["over_h38"] += float(sold0["over_h38"])
            meta["over_h9"] += float(sold0["over_h9"])
            meta["over_neteli"] += float(sold0["over_neteli"])

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
    cow_month_factors = _normalize_month_factor_map(
        ins_p.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        ins_p.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    snapshot: HerdState | None = None
    if target_ts <= start_ts:
        snapshot = _copy_state(state)

    idx_dry = min(dry_days, gest_days)

    day = start_ts

    while day < end_sim_ts:
        day = (day + pd.Timedelta(days=1)).normalize()

        for l in (1, 2, 3, 4):
            state.open_dim[l] = shift_right(state.open_dim[l])
        state.heifer_age = shift_right(state.heifer_age)
        state.bull_age = shift_right(state.bull_age)

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                state.preg_lact[(l, semen)] = shift_left(state.preg_lact[(l, semen)])
                state.preg_dry[(l, semen)] = shift_left(state.preg_dry[(l, semen)])
        for semen in ("trad", "sex"):
            state.heifer_preg[semen] = shift_left(state.heifer_preg[semen])

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                move = float(state.preg_lact[(l, semen)][idx_dry])
                if move > 0:
                    state.preg_lact[(l, semen)][idx_dry] = 0.0
                    state.preg_dry[(l, semen)][idx_dry] += move

        _process_bucket0_for_day(day)

        for l in (1, 2, 3, 4):
            first_ai = float(ins_p["cow_first_ai_dim_by_lact"].get(l, 70.0))
            spc = float(ins_p["cow_services_per_conception"])
            interval_raw = float(ins_p["cow_ai_interval_days"])
            mean_target = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
            month_factor = _month_factor_value(cow_month_factors, day)

            interval = _effective_ai_interval_days(interval_raw, mean_target, first_ai, spc)
            p_service = 1.0 / max(1.0, interval)
            p_conc = _clamp((1.0 / max(1e-9, spc)) * month_factor, 0.05, 0.95)

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

            state.preg_lact[(l, "sex")][gest_days] += services_total * cow_sex_share * p_conc
            state.preg_lact[(l, "trad")][gest_days] += services_total * cow_trad_share * p_conc

            if (int(day.year), int(day.month)) == target_month:
                meta["cow_doses_total"] += services_total
                meta["cow_doses_sex"] += services_total * cow_sex_share
                meta["cow_doses_trad"] += services_total * cow_trad_share

        first_ai_age = float(ins_p["heifer_first_ai_age_days"])
        spc_h = float(ins_p["heifer_services_per_conception"])
        interval_raw_h = float(ins_p["heifer_ai_interval_days"])
        mean_target_h = float(cp["avg_heifer_age_days"])
        month_factor_h = _month_factor_value(heifer_month_factors, day)

        interval_h = _effective_ai_interval_days(interval_raw_h, mean_target_h, first_ai_age, spc_h)
        p_service_h = 1.0 / max(1.0, interval_h)
        p_conc_h = _clamp((1.0 / max(1e-9, spc_h)) * month_factor_h, 0.05, 0.95)

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

                if (int(day.year), int(day.month)) == target_month:
                    meta["heifer_doses_total"] += services_total_h
                    meta["heifer_doses_sex"] += services_total_h * heif_sex_share
                    meta["heifer_doses_trad"] += services_total_h * heif_trad_share

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

        day_eom = pd.Timestamp(end_of_month(day.date())).normalize()
        if params.get("APPLY_CAPACITY", True) and day == day_eom:
            sold = _apply_capacity_month_end(
                state,
                gest_days=gest_days,
                dry_days=dry_days,
                cap_norm=params.get("HERD_CAPACITY_NORM"),
            )
            if (int(day.year), int(day.month)) == target_month:
                meta["sell_cows"] += float(sold["sell_cows"])
                meta["sell_heifers"] += float(sold["sell_heifers"])
                meta["sell_neteli"] += float(sold["sell_neteli"])
                meta["over_doy"] += float(sold["over_doy"])
                meta["over_dry"] += float(sold["over_dry"])
                meta["over_h0"] += float(sold["over_h0"])
                meta["over_h38"] += float(sold["over_h38"])
                meta["over_h9"] += float(sold["over_h9"])
                meta["over_neteli"] += float(sold["over_neteli"])

        if day == target_ts:
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


                                                              
                                                              

def compute_forecast_dynamic_from_db(
    target_date: date,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> Dict[str, float]:
    import pandas as pd
    from datetime import date, datetime

    def _as_ts(x):
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.normalize()
        if isinstance(x, datetime):
            return pd.Timestamp(x).normalize()
        if isinstance(x, date):
            return pd.Timestamp(x)
        return pd.Timestamp(x).normalize()

    tables = load_tables()
    base = _as_ts(latest_data_date(tables))
    target_date = _as_ts(target_date)

    if as_of_date is None:
        start = min(base, target_date)
    else:
        as_of_ts = _as_ts(as_of_date)
        if as_of_ts is None or pd.isna(as_of_ts):
            raise ValueError(f"as_of_date is invalid: {as_of_date!r}")
        start = min(min(as_of_ts, base), target_date)

    ov = dict(overrides or {})

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
    if "SEMEN_SEX_RATIOS" in ov and "semen_sex_ratios" not in ov:
        ov["semen_sex_ratios"] = ov["SEMEN_SEX_RATIOS"]
    if "herd_capacity" in ov and "HERD_CAPACITY" not in ov:
        ov["HERD_CAPACITY"] = ov["herd_capacity"]

    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

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

    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj: SemenSexRatio) -> SemenSexRatio:
            bull_raw = d.get("bull_share")
            heif_raw = d.get("heifer_share")

                                                 
            if bull_raw is None and heif_raw is None:
                bull = float(fallback_obj.bull_share)
                heif = float(fallback_obj.heifer_share)
            elif bull_raw is None:
                heif = float(heif_raw)
                bull = 1.0 - heif
            elif heif_raw is None:
                bull = float(bull_raw)
                heif = 1.0 - bull
            else:
                bull = float(bull_raw)
                heif = float(heif_raw)

            bull = max(0.0, min(1.0, bull))
            heif = max(0.0, min(1.0, heif))
            s = max(1e-9, bull + heif)
            bull /= s
            heif /= s
            return SemenSexRatio(bull_share=bull, heifer_share=heif)

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
        semen_sex_ratios=semen_sex_ratios,                                                        
    )

    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_date,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )

    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)
    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    h0_3 = float(state_at_target.heifer_age[:90].sum())
    h3_8 = float(state_at_target.heifer_age[90:270].sum())
    h9p = float(state_at_target.heifer_age[270:].sum())
    b0_2 = float(state_at_target.bull_age[:61].sum())

    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    out = {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),

        "Тёлки 0–3 мес": round(h0_3, 1),
        "Тёлки 0–2 мес": round(h0_3, 1),

        "Бычки 0–2 мес": round(b0_2, 1),
        "Тёлки 3–8 мес": round(h3_8, 1),
        "Тёлки ≥9 мес": round(h9p, 1),
        "Нетели": round(neteli, 1),

        "Ожидаемый отёл, всего": round(calv_total_f, 1),
        "Ожидаемый отёл, из них коров": round(calv_cows_f, 1),
        "Ожидаемый отёл, из них нетелей": round(calv_heifers_f, 1),

        "Ожидаемые бычки": round(exp_bulls_f, 1),
        "Ожидаемые тёлочки": round(exp_heifers_f, 1),

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

    _apply_expected_calving_prob_fallback_from_tables(
        out,
        tables,
        target_date,
        gest_days=gest_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        as_of_date=as_of_date,
    )
    _apply_current_month_observed_births_overlay(
        out,
        tables,
        target_date,
        start_ts=start,
        as_of_date=as_of_date,
    )
    _scale_birth_output(
        out,
        _recent_birth_bias_factor_from_tables(
            tables,
            target_date,
            as_of_date=as_of_date,
            overrides=ov,
            gest_days=gest_days,
            insemination_params=params["INSEMINATION_PARAMS"],
            semen_shares=semen_shares,
            semen_sex_ratios=semen_sex_ratios,
        ),
    )
    return out


def _normalize_input_tables(tables: Dict[str, pd.DataFrame] | None) -> Dict[str, pd.DataFrame]:
    src = tables or {}
    out: Dict[str, pd.DataFrame] = {}

    required_cols = {
        "calv": ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"],
        "ins": ["reg", "lact", "dim_age", "event_date", "bull", "result"],
        "dry": ["reg", "dim", "event_date"],
        "disp": ["reg", "event_date", "disposal_reason"],
        "bulls": ["bull_code", "bull_type"],
    }

    for key, cols in required_cols.items():
        df = src.get(key)
        if not isinstance(df, pd.DataFrame):
            out[key] = pd.DataFrame(columns=cols)
            continue
        dfx = df.copy()
        for c in cols:
            if c not in dfx.columns:
                dfx[c] = pd.NA
        out[key] = dfx[cols].copy()

    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _smape_percent(pred_val: float, fact_val: float) -> float | None:
    scale = abs(float(pred_val)) + abs(float(fact_val))
    if scale < 20.0:
        return None
    return 200.0 * abs(float(pred_val) - float(fact_val)) / scale


def _scale_birth_output(out: Dict[str, float], factor: float) -> None:
    factor = float(factor)
    if abs(factor - 1.0) <= 1e-9:
        return
    for key in BIRTH_OUTPUT_KEYS:
        if key in out:
            out[key] = round(_safe_float(out.get(key), 0.0) * factor, 1)


def _expected_calving_proxy_breakdown_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    as_of_date: date | None = None,
) -> Dict[str, float]:
    out = {key: 0.0 for key in BIRTH_OUTPUT_KEYS}
    ins = tables.get("ins")
    if not isinstance(ins, pd.DataFrame) or ins.empty:
        return out

    d = ins.copy()
    d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
    d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")

    if as_of_date is not None:
        as_of_ts = pd.Timestamp(as_of_date).normalize()
        d = d[d["event_date_n"].notna() & (d["event_date_n"] <= as_of_ts)]
    else:
        d = d[d["event_date_n"].notna()]

    if d.empty:
        return out

    m_start = pd.Timestamp(target_date).normalize().to_period("M").to_timestamp().normalize()
    m_next = (m_start + pd.offsets.MonthBegin(1)).normalize()

    d["due_dt"] = d["event_date_n"] + pd.to_timedelta(int(gest_days), unit="D")
    due = d[(d["due_dt"] >= m_start) & (d["due_dt"] < m_next)].copy()
    if due.empty:
        return out

    cow_spc = _safe_float(
        insemination_params.get("cow_services_per_conception"),
        float(INSEMINATION_PARAMS.cow_services_per_conception),
    )
    heif_spc = _safe_float(
        insemination_params.get("heifer_services_per_conception"),
        float(INSEMINATION_PARAMS.heifer_services_per_conception),
    )

    cow_month_factors = _normalize_month_factor_map(
        insemination_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        insemination_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    due["base_p"] = np.where(
        due["lact_n"] > 0,
        1.0 / max(1e-9, cow_spc),
        1.0 / max(1e-9, heif_spc),
    )
    due["month_factor"] = np.where(
        due["lact_n"] > 0,
        due["event_date_n"].dt.month.map(lambda m: float(cow_month_factors.get(int(m), 1.0))),
        due["event_date_n"].dt.month.map(lambda m: float(heifer_month_factors.get(int(m), 1.0))),
    )
    due["month_factor"] = pd.to_numeric(due["month_factor"], errors="coerce").fillna(1.0)
    due["exp_weight"] = np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95)

    exp_cow = float(due.loc[due["lact_n"] > 0, "exp_weight"].sum())
    exp_heif = float(due.loc[due["lact_n"] <= 0, "exp_weight"].sum())
    exp_unk = float(due.loc[due["lact_n"].isna(), "exp_weight"].sum())
    exp_total = exp_cow + exp_heif + exp_unk

    trad_bull = float(semen_sex_ratios["trad"].bull_share)
    sex_bull = float(semen_sex_ratios["sex"].bull_share)
    bull_share_cow = float(semen_shares.get("cow_trad", 0.0)) * trad_bull + float(semen_shares.get("cow_sex", 0.0)) * sex_bull
    bull_share_heif = float(semen_shares.get("heifer_trad", 0.0)) * trad_bull + float(semen_shares.get("heifer_sex", 0.0)) * sex_bull
    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемый отёл, всего"] = float(exp_total)
    out["Ожидаемый отёл, из них коров"] = float(exp_cow + exp_unk)
    out["Ожидаемый отёл, из них нетелей"] = float(exp_heif)
    out["Ожидаемые бычки"] = float(exp_bulls)
    out["Ожидаемые тёлочки"] = float(exp_heifers)
    return out


def _apply_current_month_observed_births_overlay(
    out: Dict[str, float],
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    start_ts: pd.Timestamp,
    as_of_date: date | None = None,
) -> None:
    if as_of_date is not None:
        return
    if start_ts is None or pd.isna(start_ts):
        return

    target_ts = pd.Timestamp(target_date).normalize()
    start_ts = pd.Timestamp(start_ts).normalize()
    if target_ts.to_period("M") != start_ts.to_period("M"):
        return

    observed_cutoff = (start_ts - pd.Timedelta(days=1)).normalize()
    month_start = target_ts.to_period("M").to_timestamp().normalize()
    if observed_cutoff < month_start:
        return

    from core.calving_facts import actual_birth_stats_from_tables

    observed = actual_birth_stats_from_tables(
        tables.get("calv", pd.DataFrame()),
        tables.get("ins", pd.DataFrame()),
        target_ts.date(),
        as_of_date=observed_cutoff.date(),
    )
    if _safe_float(observed.get("Ожидаемый отёл, всего"), 0.0) <= 0:
        return

    for key in BIRTH_OUTPUT_KEYS:
        out[key] = round(_safe_float(out.get(key), 0.0) + _safe_float(observed.get(key), 0.0), 1)


def _recent_birth_bias_factor_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    as_of_date: date | None,
    overrides: dict | None,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
) -> float:
    if as_of_date is None:
        return 1.0

    ov = dict(overrides or {})
    if bool(ov.get("_DISABLE_DYNAMIC_BIRTH_ADJUST", False)):
        return 1.0
    if not bool(ov.get("auto_birth_bias_correction", True)):
        return 1.0

    from core.calving_facts import actual_birth_stats_from_tables, is_calving_month_complete_from_tables

    target_eom = pd.Timestamp(target_date).to_period("M").to_timestamp("M").date()
    asof_eom = pd.Timestamp(as_of_date).to_period("M").to_timestamp("M").date()
    horizon_m = max(0, _months_between_eom(asof_eom, target_eom))

    window = int(ov.get("birth_bias_window_months", 4) or 4)
    window = max(2, min(6, window))
    min_hist = int(ov.get("birth_bias_min_hist_months", 2) or 2)
    min_hist = max(2, min(window, min_hist))
    factor_min = _clamp(_safe_float(ov.get("birth_bias_factor_min"), 0.9), 0.7, 1.1)
    factor_max = _clamp(_safe_float(ov.get("birth_bias_factor_max"), 1.3), 1.0, 1.6)
    smape_threshold = _clamp(_safe_float(ov.get("birth_bias_smape_threshold"), 10.0), 5.0, 40.0)

    hist: list[tuple[float, float, float]] = []
    nested_ov = dict(ov)
    nested_ov["_DISABLE_DYNAMIC_BIRTH_ADJUST"] = True

    for i in range(1, window + 1):
        past_target = _month_end_shift(target_eom, -i)
        past_asof = _month_end_shift(past_target, -horizon_m)
        if past_asof > past_target:
            continue
        if not is_calving_month_complete_from_tables(tables.get("calv", pd.DataFrame()), past_target):
            continue

        past_pred_vals = compute_forecast_dynamic_from_tables(
            tables,
            past_target,
            overrides=nested_ov,
            as_of_date=past_asof,
        ) or {}
        pred_total = _safe_float(past_pred_vals.get("Ожидаемый отёл, всего"), 0.0)
        if pred_total <= 1e-9:
            continue

        fact_total = _safe_float(
            actual_birth_stats_from_tables(
                tables.get("calv", pd.DataFrame()),
                tables.get("ins", pd.DataFrame()),
                past_target,
                as_of_date=None,
            ).get("Ожидаемый отёл, всего"),
            0.0,
        )
        proxy_total = _safe_float(
            _expected_calving_proxy_breakdown_from_tables(
                tables,
                pd.Timestamp(past_target),
                gest_days=gest_days,
                insemination_params=insemination_params,
                semen_shares=semen_shares,
                semen_sex_ratios=semen_sex_ratios,
                as_of_date=past_asof,
            ).get("Ожидаемый отёл, всего"),
            0.0,
        )
        if proxy_total <= 1e-9:
            continue
        if (abs(pred_total) + abs(fact_total)) < 20.0:
            continue
        hist.append((pred_total, proxy_total, fact_total))

    if len(hist) < min_hist:
        return 1.0

    model_err = pd.Series([fact - pred for pred, _proxy, fact in hist], dtype=float)
    proxy_err = pd.Series([fact - proxy for _pred, proxy, fact in hist], dtype=float)

    same_model = max(float((model_err > 0).mean()), float((model_err < 0).mean()))
    same_proxy = max(float((proxy_err > 0).mean()), float((proxy_err < 0).mean()))
    model_smape_values = [_smape_percent(pred, fact) for pred, _proxy, fact in hist]
    proxy_smape_values = [_smape_percent(proxy, fact) for _pred, proxy, fact in hist]
    model_smape = float(pd.Series([x for x in model_smape_values if x is not None], dtype=float).mean() or 0.0)
    proxy_smape = float(pd.Series([x for x in proxy_smape_values if x is not None], dtype=float).mean() or 0.0)

    if not (
        same_model >= 0.75
        and same_proxy >= 0.75
        and float(model_err.mean()) > 0.0
        and float(proxy_err.mean()) > 0.0
        and model_smape >= smape_threshold
        and proxy_smape >= smape_threshold
    ):
        return 1.0

    sum_pred = float(sum(pred for pred, _proxy, _fact in hist))
    sum_fact = float(sum(fact for _pred, _proxy, fact in hist))
    if sum_pred <= 1e-9:
        return 1.0
    return _clamp(sum_fact / sum_pred, float(factor_min), float(factor_max))


def _apply_expected_calving_prob_fallback_from_tables(
    out: Dict[str, float],
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    as_of_date: date | None = None,
) -> None:
    """
    Фолбэк для ожидаемого отёла в режиме расчёта из DataFrame-таблиц.
    Если основной расчёт дал 0 по "Ожидаемый отёл, всего", считаем прокси:
      expected = count(inseminations_due_in_month) * (1 / services_per_conception)
    """
    existing_total = _safe_float(out.get("Ожидаемый отёл, всего"), 0.0)
    if existing_total > 0:
        return

    ins = tables.get("ins")
    if not isinstance(ins, pd.DataFrame) or ins.empty:
        return

    d = ins.copy()
    d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
    d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")

    if as_of_date is not None:
        as_of_ts = pd.Timestamp(as_of_date).normalize()
        d = d[d["event_date_n"].notna() & (d["event_date_n"] <= as_of_ts)]
    else:
        d = d[d["event_date_n"].notna()]

    if d.empty:
        return

    m_start = pd.Timestamp(target_date).normalize().to_period("M").to_timestamp().normalize()
    m_next = (m_start + pd.offsets.MonthBegin(1)).normalize()

    d["due_dt"] = d["event_date_n"] + pd.to_timedelta(int(gest_days), unit="D")
    due = d[(d["due_dt"] >= m_start) & (d["due_dt"] < m_next)].copy()
    if due.empty:
        return

    n_cow = int((due["lact_n"] > 0).sum())
    n_heif = int((due["lact_n"] <= 0).sum())
    n_unk = int(due["lact_n"].isna().sum())

    if (n_cow + n_heif + n_unk) == 0:
        return

    cow_spc = _safe_float(
        insemination_params.get("cow_services_per_conception"),
        float(INSEMINATION_PARAMS.cow_services_per_conception),
    )
    heif_spc = _safe_float(
        insemination_params.get("heifer_services_per_conception"),
        float(INSEMINATION_PARAMS.heifer_services_per_conception),
    )

    cow_month_factors = _normalize_month_factor_map(
        insemination_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        insemination_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    due["base_p"] = np.where(
        due["lact_n"] > 0,
        1.0 / max(1e-9, cow_spc),
        1.0 / max(1e-9, heif_spc),
    )
    due["month_factor"] = np.where(
        due["lact_n"] > 0,
        due["event_date_n"].dt.month.map(lambda m: float(cow_month_factors.get(int(m), 1.0))),
        due["event_date_n"].dt.month.map(lambda m: float(heifer_month_factors.get(int(m), 1.0))),
    )
    due["month_factor"] = pd.to_numeric(due["month_factor"], errors="coerce").fillna(1.0)
    due["exp_weight"] = np.where(
        due["lact_n"] > 0,
        np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95),
        np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95),
    )

    exp_cow = float(due.loc[due["lact_n"] > 0, "exp_weight"].sum())
    exp_heif = float(due.loc[due["lact_n"] <= 0, "exp_weight"].sum())
    exp_unk = float(due.loc[due["lact_n"].isna(), "exp_weight"].sum())
    exp_total = exp_cow + exp_heif + exp_unk

    out["Ожидаемый отёл, всего"] = round(float(exp_total), 1)
    out["Ожидаемый отёл, из них коров"] = round(float(exp_cow + exp_unk), 1)
    out["Ожидаемый отёл, из них нетелей"] = round(float(exp_heif), 1)

    trad_bull = float(semen_sex_ratios["trad"].bull_share)
    sex_bull = float(semen_sex_ratios["sex"].bull_share)

    bull_share_cow = float(semen_shares.get("cow_trad", 0.0)) * trad_bull + float(semen_shares.get("cow_sex", 0.0)) * sex_bull
    bull_share_heif = float(semen_shares.get("heifer_trad", 0.0)) * trad_bull + float(semen_shares.get("heifer_sex", 0.0)) * sex_bull

    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемые бычки"] = round(float(exp_bulls), 1)
    out["Ожидаемые тёлочки"] = round(float(exp_heifers), 1)


def compute_forecast_dynamic_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: date,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> Dict[str, float]:
    import pandas as pd
    from datetime import date, datetime

    def _as_ts(x):
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.normalize()
        if isinstance(x, datetime):
            return pd.Timestamp(x).normalize()
        if isinstance(x, date):
            return pd.Timestamp(x)
        return pd.Timestamp(x).normalize()

    tables = _normalize_input_tables(tables)
    base = _as_ts(latest_data_date(tables))
    target_date = _as_ts(target_date)

    if as_of_date is None:
        start = min(base, target_date)
    else:
        as_of_ts = _as_ts(as_of_date)
        if as_of_ts is None or pd.isna(as_of_ts):
            raise ValueError(f"as_of_date is invalid: {as_of_date!r}")
        start = min(min(as_of_ts, base), target_date)

    ov = dict(overrides or {})

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
    if "SEMEN_SEX_RATIOS" in ov and "semen_sex_ratios" not in ov:
        ov["semen_sex_ratios"] = ov["SEMEN_SEX_RATIOS"]
    if "herd_capacity" in ov and "HERD_CAPACITY" not in ov:
        ov["HERD_CAPACITY"] = ov["herd_capacity"]

    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

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

    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj: SemenSexRatio) -> SemenSexRatio:
            bull_raw = d.get("bull_share")
            heif_raw = d.get("heifer_share")

            if bull_raw is None and heif_raw is None:
                bull = float(fallback_obj.bull_share)
                heif = float(fallback_obj.heifer_share)
            elif bull_raw is None:
                heif = float(heif_raw)
                bull = 1.0 - heif
            elif heif_raw is None:
                bull = float(bull_raw)
                heif = 1.0 - bull
            else:
                bull = float(bull_raw)
                heif = float(heif_raw)

            bull = max(0.0, min(1.0, bull))
            heif = max(0.0, min(1.0, heif))
            s = max(1e-9, bull + heif)
            bull /= s
            heif /= s
            return SemenSexRatio(bull_share=bull, heifer_share=heif)

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
        semen_sex_ratios=semen_sex_ratios,
    )

    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_date,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )

    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)
    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    h0_3 = float(state_at_target.heifer_age[:90].sum())
    h3_8 = float(state_at_target.heifer_age[90:270].sum())
    h9p = float(state_at_target.heifer_age[270:].sum())
    b0_2 = float(state_at_target.bull_age[:61].sum())

    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    out = {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),
        "Тёлки 0–3 мес": round(h0_3, 1),
        "Тёлки 0–2 мес": round(h0_3, 1),
        "Бычки 0–2 мес": round(b0_2, 1),
        "Тёлки 3–8 мес": round(h3_8, 1),
        "Тёлки ≥9 мес": round(h9p, 1),
        "Нетели": round(neteli, 1),
        "Ожидаемый отёл, всего": round(calv_total_f, 1),
        "Ожидаемый отёл, из них коров": round(calv_cows_f, 1),
        "Ожидаемый отёл, из них нетелей": round(calv_heifers_f, 1),
        "Ожидаемые бычки": round(exp_bulls_f, 1),
        "Ожидаемые тёлочки": round(exp_heifers_f, 1),
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

    _apply_expected_calving_prob_fallback_from_tables(
        out,
        tables,
        target_date,
        gest_days=gest_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        as_of_date=as_of_date,
    )
    _apply_current_month_observed_births_overlay(
        out,
        tables,
        target_date,
        start_ts=start,
        as_of_date=as_of_date,
    )
    _scale_birth_output(
        out,
        _recent_birth_bias_factor_from_tables(
            tables,
            target_date,
            as_of_date=as_of_date,
            overrides=ov,
            gest_days=gest_days,
            insemination_params=params["INSEMINATION_PARAMS"],
            semen_shares=semen_shares,
            semen_sex_ratios=semen_sex_ratios,
        ),
    )
    return out
