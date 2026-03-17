from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import model_params as mp
import pandas as pd
from db import engine
from forecast_dynamic import compute_forecast_dynamic_from_db
from sqlalchemy import text

def _month_bounds(d: date) -> tuple[date, date]:
    ms = date(d.year, d.month, 1)
    me = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return ms, me

def _pop_due(counter, month_start: date, month_end: date) -> float:
    if not counter:
        return 0.0
    due_dates = [dt for dt in counter.keys() if month_start <= dt <= month_end]
    if not due_dates:
        return 0.0
    n = float(sum(counter[dt] for dt in due_dates))
    for dt in due_dates:
        del counter[dt]
    return n

def _to_ts(x) -> pd.Timestamp:
    """
    Приводит любые date/datetime/Timestamp/строку к pd.Timestamp (normalized to 00:00:00).
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    if isinstance(x, datetime):
        return pd.Timestamp(x).normalize()
    if isinstance(x, date):
        return pd.Timestamp(x).normalize()
    return pd.Timestamp(x).normalize()


def _month_start(d_end: date) -> date:
    return date(d_end.year, d_end.month, 1)


def _next_month_start(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return date(int(ts.year), int(ts.month), monthrange(int(ts.year), int(ts.month))[1])


def _months_between_eom(start_eom: date, end_eom: date) -> int:
    return (int(end_eom.year) - int(start_eom.year)) * 12 + (int(end_eom.month) - int(start_eom.month))


def _actual_calvings_total_month(month_end_date: date, as_of_date: date | None = None) -> float:
    m_start = _month_start(month_end_date)
    m_next = _next_month_start(m_start)
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()

    sql = """
    WITH src AS (
      SELECT
        CASE
          WHEN (
            UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%ОТЕЛ%'
            OR UPPER(COALESCE(event_type, '')) LIKE '%CALV%'
          )
          THEN COALESCE(
            NULLIF(TRIM(COALESCE(reg, '')), ''),
            NULLIF(TRIM(COALESCE(animal_id, '')), '')
          )
          ELSE COALESCE(
            NULLIF(TRIM(COALESCE(mother_reg, '')), ''),
            NULLIF(TRIM(COALESCE(reg, '')), ''),
            NULLIF(TRIM(COALESCE(animal_id, '')), '')
          )
        END AS dam_key,
        COALESCE(birth_date::date, event_date::date) AS event_dt
      FROM calvings_births_raw
      WHERE event_date IS NOT NULL
        AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
        AND (
          UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%ОТЕЛ%'
          OR UPPER(COALESCE(event_type, '')) LIKE '%CALV%'
          OR
          UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%РОЖ%'
          OR UPPER(COALESCE(event_type, '')) LIKE '%BORN%'
          OR UPPER(COALESCE(event_type, '')) LIKE '%BIRTH%'
        )
    ),
    scoped AS (
      SELECT
        COALESCE(dam_key, CONCAT('__UNK__', ROW_NUMBER() OVER (ORDER BY event_dt))) AS dam_key,
        event_dt
      FROM src
      WHERE event_dt >= :m_start
        AND event_dt < :m_next
    )
    SELECT COUNT(*) AS n
    FROM (
      SELECT dam_key, event_dt
      FROM scoped
      GROUP BY dam_key, event_dt
    ) z;
    """
    df = pd.read_sql(text(sql), con=engine, params={"m_start": m_start, "m_next": m_next, "as_of_date": as_of_d})
    return float(df.loc[0, "n"]) if not df.empty else 0.0


def _actual_birth_rows_month(month_end_date: date, as_of_date: date | None = None) -> float:
    m_start = _month_start(month_end_date)
    m_next = _next_month_start(m_start)
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()

    sql = """
    SELECT COUNT(*) AS n
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
      AND (
        UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%РОЖ%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BORN%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BIRTH%'
      );
    """
    df = pd.read_sql(text(sql), con=engine, params={"m_start": m_start, "m_next": m_next, "as_of_date": as_of_d})
    birth_rows = float(df.loc[0, "n"]) if not df.empty else 0.0
    if birth_rows > 0:
        return birth_rows
    return _actual_calvings_total_month(month_end_date, as_of_date=as_of_date)


def _proxy_expected_calvings_month(
    month_end_date: date,
    *,
    gest_days: int,
    cow_spc: float,
    heif_spc: float,
    as_of_date: date | None = None,
) -> float:
    m_start = _month_start(month_end_date)
    m_next = _next_month_start(m_start)
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()

    sql = """
    WITH x AS (
      SELECT
        (event_date::date + make_interval(days => :gest_days))::date AS due_dt,
        lact
      FROM inseminations_raw
      WHERE event_date IS NOT NULL
        AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
    )
    SELECT
      COALESCE(count(*) FILTER (WHERE lact > 0), 0)  AS n_cow,
      COALESCE(count(*) FILTER (WHERE lact <= 0), 0) AS n_heifer,
      COALESCE(count(*) FILTER (WHERE lact IS NULL), 0) AS n_unknown
    FROM x
    WHERE due_dt >= :m_start
      AND due_dt <  :m_next;
    """
    df = pd.read_sql(
        text(sql),
        con=engine,
        params={"gest_days": int(gest_days), "m_start": m_start, "m_next": m_next, "as_of_date": as_of_d},
    )
    if df.empty:
        return 0.0
    n_cow = float(df.loc[0, "n_cow"] or 0.0)
    n_heif = float(df.loc[0, "n_heifer"] or 0.0)
    n_unk = float(df.loc[0, "n_unknown"] or 0.0)
    return n_cow / max(1e-9, float(cow_spc)) + n_heif / max(1e-9, float(heif_spc)) + n_unk / max(1e-9, float(cow_spc))


def _calving_correction_factor(target_date: date, overrides: dict | None, as_of_date: date | None) -> float:
    ov = overrides or {}
    enabled = bool(ov.get("auto_calving_calibration", True))
    if not enabled:
        return 1.0

    window_months = int(ov.get("calving_calibration_window_months", 8) or 8)
    window_months = max(3, min(18, window_months))

    gest_days = int(round(_safe_float(ov.get("gestation_days", ov.get("GESTATION_DAYS")), mp.GESTATION_DAYS)))
    ip = ov.get("insemination_params") or ov.get("INSEMINATION_PARAMS") or {}
    cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
    heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)

    ref = _to_ts(as_of_date if as_of_date is not None else target_date)
    ref_eom = date(ref.year, ref.month, monthrange(ref.year, ref.month)[1])

    act_sum = 0.0
    prx_sum = 0.0
    for i in range(window_months):
        d_end = _month_end_shift(ref_eom, -i)
        act_sum += _actual_calvings_total_month(d_end, as_of_date=as_of_date)
        prx_sum += _proxy_expected_calvings_month(
            d_end,
            gest_days=gest_days,
            cow_spc=cow_spc,
            heif_spc=heif_spc,
            as_of_date=as_of_date,
        )

    if prx_sum <= 1e-9:
        return 1.0
    raw = act_sum / prx_sum
    return float(max(0.7, min(1.6, raw)))


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _norm_sex_value(x) -> str | None:
    if x is None:
        return None
    v = str(x).strip().upper().replace("Ё", "Е")
    if v in {"", "NAN", "NONE", "NULL"}:
        return None
    if v in {"M", "М", "MALE", "1", "БЫК", "БЫЧ", "БЫЧОК"}:
        return "M"
    if v in {"F", "Ж", "FEMALE", "2", "ТЕЛКА", "ТЕЛОЧКА"}:
        return "F"
    return None


def _resolve_semen_usage_from_params(overrides: dict | None) -> dict[str, float]:
    ov = overrides or {}
    su = (ov.get("semen_usage") or ov.get("SEMEN_USAGE_SHARES") or {})

    def _pair(a_raw, b_raw, a_fb: float, b_fb: float) -> tuple[float, float]:
        a = None if a_raw is None else _clamp(_safe_float(a_raw, a_fb), 0.0, 1.0)
        b = None if b_raw is None else _clamp(_safe_float(b_raw, b_fb), 0.0, 1.0)
        if a is None and b is None:
            a, b = float(a_fb), float(b_fb)
        elif a is None:
            a = 1.0 - float(b)
        elif b is None:
            b = 1.0 - float(a)
        s = max(1e-9, float(a) + float(b))
        return float(a) / s, float(b) / s

    cow_trad, cow_sex = _pair(
        su.get("cow_trad"),
        su.get("cow_sex"),
        float(mp.SEMEN_USAGE_PROBS.cow_trad),
        float(mp.SEMEN_USAGE_PROBS.cow_sex),
    )
    heif_trad, heif_sex = _pair(
        su.get("heifer_trad"),
        su.get("heifer_sex"),
        float(mp.SEMEN_USAGE_PROBS.heifer_trad),
        float(mp.SEMEN_USAGE_PROBS.heifer_sex),
    )
    return {
        "cow_trad": cow_trad,
        "cow_sex": cow_sex,
        "heifer_trad": heif_trad,
        "heifer_sex": heif_sex,
    }


def _resolve_semen_sex_ratios_from_params(overrides: dict | None) -> dict[str, dict[str, float]]:
    ov = overrides or {}
    ssr = (ov.get("semen_sex_ratios") or ov.get("SEMEN_SEX_RATIOS") or {})

    def _ratio(d: dict | None, bull_fb: float, heif_fb: float) -> dict[str, float]:
        dd = d or {}
        bull_raw = dd.get("bull_share")
        heif_raw = dd.get("heifer_share")
        bull = None if bull_raw is None else _clamp(_safe_float(bull_raw, bull_fb), 0.0, 1.0)
        heif = None if heif_raw is None else _clamp(_safe_float(heif_raw, heif_fb), 0.0, 1.0)
        if bull is None and heif is None:
            bull, heif = float(bull_fb), float(heif_fb)
        elif bull is None:
            bull = 1.0 - float(heif)
        elif heif is None:
            heif = 1.0 - float(bull)
        s = max(1e-9, float(bull) + float(heif))
        return {"bull_share": float(bull) / s, "heifer_share": float(heif) / s}

    return {
        "trad": _ratio(
            ssr.get("trad") if isinstance(ssr, dict) else None,
            float(mp.SEMEN_SEX_RATIOS["trad"].bull_share),
            float(mp.SEMEN_SEX_RATIOS["trad"].heifer_share),
        ),
        "sex": _ratio(
            ssr.get("sex") if isinstance(ssr, dict) else None,
            float(mp.SEMEN_SEX_RATIOS["sex"].bull_share),
            float(mp.SEMEN_SEX_RATIOS["sex"].heifer_share),
        ),
    }


def _prior_bull_share_from_params(overrides: dict | None) -> float:
    ov = overrides or {}
    su = _resolve_semen_usage_from_params(ov)
    ssr = _resolve_semen_sex_ratios_from_params(ov)

    cow_trad = float(su["cow_trad"])
    cow_sex = float(su["cow_sex"])
    heif_trad = float(su["heifer_trad"])
    heif_sex = float(su["heifer_sex"])

    trad_bull = float(ssr["trad"]["bull_share"])
    sex_bull = float(ssr["sex"]["bull_share"])

    cow_bull = cow_trad * trad_bull + cow_sex * sex_bull
    heif_bull = heif_trad * trad_bull + heif_sex * sex_bull
    w_cow = _clamp(_safe_float(ov.get("calf_sex_prior_cow_weight"), 0.75), 0.0, 1.0)
    return _clamp(w_cow * cow_bull + (1.0 - w_cow) * heif_bull, 0.2, 0.8)


def _historical_calf_sex_shares(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> Tuple[float, float] | None:
    ov = overrides or {}
    if not bool(ov.get("auto_calf_sex_calibration", True)):
        return None

    window_months = int(ov.get("calf_sex_calibration_window_months", 24) or 24)
    window_months = max(3, min(24, window_months))

    ref = _to_ts(as_of_date if as_of_date is not None else target_date)
    ref_eom = date(ref.year, ref.month, monthrange(ref.year, ref.month)[1])
    start_eom = _month_end_shift(ref_eom, -(window_months - 1))
    m_start = date(start_eom.year, start_eom.month, 1)
    m_next = _next_month_start(date(ref_eom.year, ref_eom.month, 1))
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()

    sql = """
    SELECT sex
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
      AND (
        UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%РОЖ%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BORN%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BIRTH%'
      );
    """
    df = pd.read_sql(text(sql), con=engine, params={"m_start": m_start, "m_next": m_next, "as_of_date": as_of_d})
    if df.empty:
        return None

    sx = df["sex"].map(_norm_sex_value)
    bulls_known = float((sx == "M").sum())
    heif_known = float((sx == "F").sum())
    total = float(len(sx))
    known = bulls_known + heif_known
    unknown = max(0.0, total - known)
    if total <= 0:
        return None

    prior_bull = _prior_bull_share_from_params(ov)
    bull_known_share = (bulls_known / known) if known > 0 else prior_bull

    bulls_adj = bulls_known + unknown * bull_known_share
    raw_bull = _clamp(bulls_adj / max(1.0, total), 0.1, 0.9)

    smooth_n = int(ov.get("calf_sex_calibration_smooth_n", 150) or 150)
    smooth_n = max(20, min(1000, smooth_n))
    w = known / (known + float(smooth_n))
    bull_final = _clamp(w * raw_bull + (1.0 - w) * prior_bull, 0.1, 0.9)
    return bull_final, 1.0 - bull_final


def _calf_count_factor(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> float:
    ov = overrides or {}
    if not bool(ov.get("auto_calf_count_calibration", True)):
        return 1.0

    window_months = int(ov.get("calf_count_calibration_window_months", 6) or 6)
    window_months = max(3, min(24, window_months))

    ref = _to_ts(as_of_date if as_of_date is not None else target_date)
    ref_eom = date(ref.year, ref.month, monthrange(ref.year, ref.month)[1])

    sum_calvings = 0.0
    sum_birth_rows = 0.0
    for i in range(window_months):
        d_end = _month_end_shift(ref_eom, -i)
        sum_calvings += _actual_calvings_total_month(d_end, as_of_date=as_of_date)
        sum_birth_rows += _actual_birth_rows_month(d_end, as_of_date=as_of_date)

    if sum_calvings <= 1e-9:
        return 1.0

    raw = sum_birth_rows / sum_calvings
    f_min = _safe_float(ov.get("calf_count_factor_min"), 0.8)
    f_max = _safe_float(ov.get("calf_count_factor_max"), 2.5)
    if f_min > f_max:
        f_min, f_max = f_max, f_min
    return _clamp(raw, float(f_min), float(f_max))


def _lead_time_total_factor(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> float:
    ov = overrides or {}
    if not bool(ov.get("auto_lead_time_calibration", True)):
        return 1.0
    if as_of_date is None:
        return 1.0

    t_eom = _to_ts(target_date).date()
    a_eom = _to_ts(as_of_date).date()
    horizon_m = _months_between_eom(a_eom, t_eom)
    if horizon_m <= 0:
        return 1.0

    window = int(ov.get("lead_time_calibration_window_months", 8) or 8)
    window = max(3, min(24, window))

    ip = ov.get("insemination_params") or ov.get("INSEMINATION_PARAMS") or {}
    gest_days = int(round(_safe_float(ov.get("gestation_days", ov.get("GESTATION_DAYS")), mp.GESTATION_DAYS)))
    cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
    heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)

    ratio_points: list[tuple[date, float]] = []
    for i in range(1, window + 1):
        past_target = _month_end_shift(t_eom, -i)
        past_asof = _month_end_shift(past_target, -horizon_m)
        if past_target > a_eom:
            continue
        proxy = _proxy_expected_calvings_month(
            past_target,
            gest_days=gest_days,
            cow_spc=cow_spc,
            heif_spc=heif_spc,
            as_of_date=past_asof,
        )
        if proxy <= 1e-9:
            continue
        actual = _actual_calvings_total_month(past_target, as_of_date=as_of_date)
        ratio_points.append((past_target, actual / proxy))

    if not ratio_points:
        return 1.0

    ratio_points = sorted(ratio_points, key=lambda z: z[0])
    s = pd.Series([r for _, r in ratio_points], dtype=float).clip(lower=0.6, upper=1.8)
    med = float(s.median())

    recent_n = int(ov.get("lead_time_recent_n", 1) or 1)
    recent_n = max(1, min(6, recent_n))
    tail = s.tail(recent_n).to_numpy(dtype=float)
    if len(tail) > 0:
        weights = pd.Series(range(1, len(tail) + 1), dtype=float).to_numpy()
        recent = float((tail * weights).sum() / max(1e-9, weights.sum()))
    else:
        recent = med

    w_recent = _clamp(_safe_float(ov.get("lead_time_recent_weight"), 1.0), 0.0, 1.0)
    raw = (1.0 - w_recent) * med + w_recent * recent

    smooth_k = int(ov.get("lead_time_calibration_smooth_k", 1) or 1)
    smooth_k = max(1, min(24, smooth_k))
    w = len(s) / float(len(s) + smooth_k)
    factor = (1.0 - w) * 1.0 + w * raw

    f_min = _safe_float(ov.get("lead_time_factor_min"), 0.75)
    f_max = _safe_float(ov.get("lead_time_factor_max"), 1.5)
    if f_min > f_max:
        f_min, f_max = f_max, f_min
    return _clamp(factor, float(f_min), float(f_max))


def _is_complete_fact_month(month_end_date: date, as_of_date: date | None = None) -> bool:
    m_start = _month_start(month_end_date)
    m_next = _next_month_start(m_start)
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()

    sql = """
    SELECT MAX(event_date::date) AS max_dt
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
      AND (
        UPPER(REPLACE(COALESCE(event_type, ''), 'Ё', 'Е')) LIKE '%РОЖ%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BORN%'
        OR UPPER(COALESCE(event_type, '')) LIKE '%BIRTH%'
      );
    """
    df = pd.read_sql(text(sql), con=engine, params={"m_start": m_start, "m_next": m_next, "as_of_date": as_of_d})
    if df.empty:
        return False
    mx = pd.to_datetime(df.loc[0, "max_dt"], errors="coerce")
    if pd.isna(mx):
        return False
    return bool(mx.date() >= month_end_date)


def _recent_additive_total_adjustment(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> float:
    ov = overrides or {}
    if not bool(ov.get("auto_additive_heads_calibration", True)):
        return 0.0

    t_eom = _to_ts(target_date).date()
    a_eom = None if as_of_date is None else _to_ts(as_of_date).date()
    horizon_m = 0 if a_eom is None else max(0, _months_between_eom(a_eom, t_eom))

    win = int(ov.get("additive_heads_window_months", 2) or 2)
    win = max(1, min(6, win))
    k = _clamp(_safe_float(ov.get("additive_heads_k"), 0.5), 0.0, 2.0)
    cap = max(10.0, _safe_float(ov.get("additive_heads_cap"), 140.0))

    ip = ov.get("insemination_params") or ov.get("INSEMINATION_PARAMS") or {}
    gest_days = int(round(_safe_float(ov.get("gestation_days", ov.get("GESTATION_DAYS")), mp.GESTATION_DAYS)))
    cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
    heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)

    residuals: list[float] = []
    for i in range(1, win + 1):
        past_target = _month_end_shift(t_eom, -i)
        if a_eom is not None and past_target > a_eom:
            continue
        past_asof = _month_end_shift(past_target, -horizon_m) if a_eom is not None else None

        if not _is_complete_fact_month(past_target, as_of_date=as_of_date):
            continue

        proxy = _proxy_expected_calvings_month(
            past_target,
            gest_days=gest_days,
            cow_spc=cow_spc,
            heif_spc=heif_spc,
            as_of_date=past_asof,
        )
        if proxy <= 1e-9:
            continue
        actual = _actual_calvings_total_month(past_target, as_of_date=as_of_date)
        residuals.append(actual - proxy)

    if not residuals:
        return 0.0

    w = pd.Series(range(1, len(residuals) + 1), dtype=float).to_numpy()
    r = pd.Series(residuals, dtype=float).to_numpy()
    avg = float((r * w).sum() / max(1e-9, w.sum()))
    add = k * avg
    return _clamp(add, -cap, cap)


def _yoy_same_month_total_factor(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> float:
    ov = overrides or {}
    if not bool(ov.get("auto_yoy_same_month_calibration", True)):
        return 1.0

    alpha = _clamp(_safe_float(ov.get("yoy_same_month_alpha"), 0.14), 0.0, 1.0)
    if alpha <= 1e-9:
        return 1.0

    t_eom = _to_ts(target_date).date()
    a_eom = None if as_of_date is None else _to_ts(as_of_date).date()
    horizon_m = 0 if a_eom is None else max(0, _months_between_eom(a_eom, t_eom))
    prev_year_same_month = _month_end_shift(t_eom, -12)

    if a_eom is not None and prev_year_same_month > a_eom:
        return 1.0
    if not _is_complete_fact_month(prev_year_same_month, as_of_date=as_of_date):
        return 1.0

    past_asof = _month_end_shift(prev_year_same_month, -horizon_m) if a_eom is not None else None
    if past_asof is not None and past_asof > prev_year_same_month:
        return 1.0

    nested_ov = dict(ov)
    nested_ov["auto_yoy_same_month_calibration"] = False
    nested_ov["auto_last_known_residual_calibration"] = False

    pred_prev = _safe_float(
        (compute_forecast_from_db(prev_year_same_month, overrides=nested_ov, as_of_date=past_asof) or {}).get(
            "Ожидаемый отёл, всего"
        ),
        0.0,
    )
    if pred_prev <= 1e-9:
        return 1.0

    actual = _actual_calvings_total_month(prev_year_same_month, as_of_date=as_of_date)
    ratio = _clamp(actual / pred_prev, 0.6, 1.8)
    raw_factor = (1.0 - alpha) * 1.0 + alpha * ratio

    f_min = _safe_float(ov.get("yoy_same_month_factor_min"), 0.85)
    f_max = _safe_float(ov.get("yoy_same_month_factor_max"), 1.15)
    if f_min > f_max:
        f_min, f_max = f_max, f_min
    return _clamp(raw_factor, float(f_min), float(f_max))


def _last_known_month_residual_adjustment(
    target_date: date,
    overrides: dict | None,
    as_of_date: date | None,
) -> float:
    ov = overrides or {}
    if not bool(ov.get("auto_last_known_residual_calibration", True)):
        return 0.0

    k = _clamp(_safe_float(ov.get("last_known_residual_k"), 0.01), 0.0, 1.0)
    if k <= 1e-9:
        return 0.0
    cap = max(10.0, _safe_float(ov.get("last_known_residual_cap"), 120.0))

    t_eom = _to_ts(target_date).date()
    a_eom = None if as_of_date is None else _to_ts(as_of_date).date()
    horizon_m = 0 if a_eom is None else max(0, _months_between_eom(a_eom, t_eom))

    if a_eom is None:
        ref_month = _month_end_shift(t_eom, -1)
        ref_asof = None
    else:
        ref_month = a_eom
        ref_asof = _month_end_shift(ref_month, -horizon_m)
        if ref_asof > ref_month:
            return 0.0

    if not _is_complete_fact_month(ref_month, as_of_date=as_of_date):
        return 0.0

    nested_ov = dict(ov)
    nested_ov["auto_yoy_same_month_calibration"] = False
    nested_ov["auto_last_known_residual_calibration"] = False

    pred_ref = _safe_float(
        (compute_forecast_from_db(ref_month, overrides=nested_ov, as_of_date=ref_asof) or {}).get("Ожидаемый отёл, всего"),
        0.0,
    )
    if pred_ref <= 1e-9:
        return 0.0

    actual = _actual_calvings_total_month(ref_month, as_of_date=as_of_date)
    add = k * (actual - pred_ref)
    return _clamp(add, -cap, cap)


def _apply_expected_calving_prob_fallback(
    out: dict,
    d_end: date,
    overrides: dict | None,
    as_of_date: date | None = None,
) -> None:
    """
    Фолбэк для 'Ожидаемый отёл...' если по логике "P-стельностей" в конкретном месяце получилось 0.
    Считаем по ВСЕМ осеменениям, которые попадают по сроку отёла в этот месяц:
      expected = count_insems_due_in_month * (1 / services_per_conception)

    Это как раз чинит кейс "в августе нули, а дальше всё есть" (обычно потому что в нужном месяце нет P).
    """
    overrides = overrides or {}

    existing_total = _safe_float(out.get("Ожидаемый отёл, всего"), 0.0)
    if existing_total > 0:
        return

    gest_days = int(
        round(
            _safe_float(
                overrides.get("gestation_days", overrides.get("GESTATION_DAYS")),
                mp.GESTATION_DAYS,
            )
        )
    )
    m_start = _month_start(d_end)
    m_next = _next_month_start(m_start)

    sql = """
    WITH x AS (
      SELECT
        (event_date::date + make_interval(days => :gest_days))::date AS due_dt,
        lact
      FROM inseminations_raw
      WHERE event_date IS NOT NULL
        AND (CAST(:as_of_date AS date) IS NULL OR event_date::date <= CAST(:as_of_date AS date))
    )
    SELECT
      COALESCE(count(*) FILTER (WHERE lact > 0), 0)  AS n_cow,
      COALESCE(count(*) FILTER (WHERE lact <= 0), 0) AS n_heifer,
      COALESCE(count(*) FILTER (WHERE lact IS NULL), 0) AS n_unknown
    FROM x
    WHERE due_dt >= :m_start
      AND due_dt <  :m_next;
    """
    as_of_d = None if as_of_date is None else _to_ts(as_of_date).date()
    df = pd.read_sql(
        text(sql),
        con=engine,
        params={"gest_days": gest_days, "m_start": m_start, "m_next": m_next, "as_of_date": as_of_d},
    )
    n_cow = int(df.loc[0, "n_cow"])
    n_heif = int(df.loc[0, "n_heifer"])
    n_unk = int(df.loc[0, "n_unknown"])

    if (n_cow + n_heif + n_unk) == 0:
        return

    ip = (
        overrides.get("insemination_params")
        or overrides.get("INSEMINATION_PARAMS")
        or {}
    )
    cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
    heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)

    p_cow = 1.0 / max(1e-9, cow_spc)
    p_heif = 1.0 / max(1e-9, heif_spc)

    exp_cow = n_cow * p_cow
    exp_heif = n_heif * p_heif
    exp_unk = n_unk * p_cow                                           

    exp_total = exp_cow + exp_heif + exp_unk

    out["Ожидаемый отёл, всего"] = float(exp_total)
    out["Ожидаемый отёл, из них коров"] = float(exp_cow + exp_unk)
    out["Ожидаемый отёл, из них нетелей"] = float(exp_heif)

    su = _resolve_semen_usage_from_params(overrides)
    ssr = _resolve_semen_sex_ratios_from_params(overrides)

    cow_trad = float(su["cow_trad"])
    cow_sex = float(su["cow_sex"])
    heif_trad = float(su["heifer_trad"])
    heif_sex = float(su["heifer_sex"])

    trad_bull = float(ssr["trad"]["bull_share"])
    sex_bull = float(ssr["sex"]["bull_share"])

    bull_share_cow = cow_trad * trad_bull + cow_sex * sex_bull
    bull_share_heif = heif_trad * trad_bull + heif_sex * sex_bull

    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемые бычки"] = float(exp_bulls)
    out["Ожидаемые тёлочки"] = float(exp_heifers)

def compute_forecast_from_db(
    target_date: date,
    overrides: Optional[dict] = None,
    as_of_date: Optional[date] = None,
) -> Dict[str, float]:
    """
    Обёртка над динамическим прогнозом + фолбэк для "ожидаемого отёла".
    Ключевое: приводим входную дату к pd.Timestamp, чтобы не ловить сравнения Timestamp vs date.
    """
    d_end_ts = _to_ts(target_date)
    if pd.isna(d_end_ts):
        raise ValueError(f"target_date is invalid: {target_date!r}")

    try:
        out = compute_forecast_dynamic_from_db(d_end_ts, overrides=overrides, as_of_date=as_of_date)
    except (TypeError, ValueError):
        out = compute_forecast_dynamic_from_db(d_end_ts.date(), overrides=overrides, as_of_date=as_of_date)

    try:
        _apply_expected_calving_prob_fallback(out, d_end_ts, overrides, as_of_date=as_of_date)
    except (TypeError, ValueError):
        _apply_expected_calving_prob_fallback(out, d_end_ts.date(), overrides, as_of_date=as_of_date)

                                                                                            
    ov = overrides or {}
    if bool(ov.get("blend_with_proxy_expected_calvings", True)):
        try:
            model_total = _safe_float(out.get("Ожидаемый отёл, всего"), 0.0)
            if model_total > 0:
                ip = ov.get("insemination_params") or ov.get("INSEMINATION_PARAMS") or {}
                gest_days = int(round(_safe_float(ov.get("gestation_days", ov.get("GESTATION_DAYS")), mp.GESTATION_DAYS)))
                cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
                heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)
                proxy_total = _proxy_expected_calvings_month(
                    d_end_ts.date(),
                    gest_days=gest_days,
                    cow_spc=cow_spc,
                    heif_spc=heif_spc,
                    as_of_date=as_of_date,
                )
                if proxy_total > 0:
                    w = _clamp(_safe_float(ov.get("proxy_blend_weight"), 1.0), 0.0, 1.0)
                    blended_total = (1.0 - w) * model_total + w * float(proxy_total)
                    scale = blended_total / max(1e-9, model_total)
                    for k in (
                        "Ожидаемый отёл, всего",
                        "Ожидаемый отёл, из них коров",
                        "Ожидаемый отёл, из них нетелей",
                        "Ожидаемые бычки",
                        "Ожидаемые тёлочки",
                    ):
                        if k in out:
                            out[k] = float(_safe_float(out.get(k), 0.0) * scale)
        except Exception:
            pass

                                                                                       
    factor = _calving_correction_factor(d_end_ts.date(), ov, as_of_date=as_of_date)
    keys = [
        "Ожидаемый отёл, всего",
        "Ожидаемый отёл, из них коров",
        "Ожидаемый отёл, из них нетелей",
        "Ожидаемые бычки",
        "Ожидаемые тёлочки",
    ]
    for k in keys:
        if k in out:
            try:
                out[k] = round(float(out[k]) * factor, 1)
            except Exception:
                pass

                                                                                     
    try:
        total_calv = _safe_float(out.get("Ожидаемый отёл, всего"), 0.0)
        if total_calv > 0:
            sex_shares = _historical_calf_sex_shares(d_end_ts.date(), overrides or {}, as_of_date)
            if sex_shares is not None:
                bull_share, heif_share = sex_shares
                out["Ожидаемые бычки"] = round(total_calv * bull_share, 1)
                out["Ожидаемые тёлочки"] = round(total_calv * heif_share, 1)
    except Exception:
        pass

                                                                                   
    try:
        calf_factor = _calf_count_factor(d_end_ts.date(), overrides or {}, as_of_date)
        for k in ("Ожидаемые бычки", "Ожидаемые тёлочки"):
            if k in out:
                out[k] = round(_safe_float(out.get(k), 0.0) * calf_factor, 1)
    except Exception:
        pass

    try:
        lead_factor = _lead_time_total_factor(d_end_ts.date(), ov, as_of_date)
        keys = [
            "Ожидаемый отёл, всего",
            "Ожидаемый отёл, из них коров",
            "Ожидаемый отёл, из них нетелей",
            "Ожидаемые бычки",
            "Ожидаемые тёлочки",
        ]
        for k in keys:
            if k in out:
                out[k] = round(_safe_float(out.get(k), 0.0) * lead_factor, 1)
    except Exception:
        pass

    try:
        total_key = "Ожидаемый отёл, всего"
        total_before = _safe_float(out.get(total_key), 0.0)
        if total_before > 0:
            add_heads = _recent_additive_total_adjustment(d_end_ts.date(), ov, as_of_date)
            total_after = max(0.0, total_before + add_heads)
            scale = total_after / max(1e-9, total_before)
            for k in (
                "Ожидаемый отёл, всего",
                "Ожидаемый отёл, из них коров",
                "Ожидаемый отёл, из них нетелей",
                "Ожидаемые бычки",
                "Ожидаемые тёлочки",
            ):
                if k in out:
                    out[k] = round(_safe_float(out.get(k), 0.0) * scale, 1)
    except Exception:
        pass

                                                                                              
    try:
        yoy_factor = _yoy_same_month_total_factor(d_end_ts.date(), ov, as_of_date)
        keys = [
            "Ожидаемый отёл, всего",
            "Ожидаемый отёл, из них коров",
            "Ожидаемый отёл, из них нетелей",
            "Ожидаемые бычки",
            "Ожидаемые тёлочки",
        ]
        for k in keys:
            if k in out:
                out[k] = round(_safe_float(out.get(k), 0.0) * yoy_factor, 1)
    except Exception:
        pass

                                                                     
    try:
        total_key = "Ожидаемый отёл, всего"
        total_before = _safe_float(out.get(total_key), 0.0)
        if total_before > 0:
            add_heads = _last_known_month_residual_adjustment(d_end_ts.date(), ov, as_of_date)
            total_after = max(0.0, total_before + add_heads)
            scale = total_after / max(1e-9, total_before)
            for k in (
                "Ожидаемый отёл, всего",
                "Ожидаемый отёл, из них коров",
                "Ожидаемый отёл, из них нетелей",
                "Ожидаемые бычки",
                "Ожидаемые тёлочки",
            ):
                if k in out:
                    out[k] = round(_safe_float(out.get(k), 0.0) * scale, 1)
    except Exception:
        pass

    return out
