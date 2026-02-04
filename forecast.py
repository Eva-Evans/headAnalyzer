from __future__ import annotations

from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd

from db import engine
import model_params as mp
from calendar import monthrange
from datetime import date

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

# важно: этот импорт должен начать работать после шага (2)
from forecast_dynamic import compute_forecast_dynamic_from_db
import pandas as pd
from datetime import date, datetime

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


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _apply_expected_calving_prob_fallback(out: dict, d_end: date, overrides: dict | None) -> None:
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

    gest_days = int(round(_safe_float(overrides.get("gestation_days"), mp.GESTATION_DAYS)))
    m_start = _month_start(d_end)
    m_next = _next_month_start(m_start)

    sql = """
    WITH x AS (
      SELECT
        (event_date::date + make_interval(days => %(gest_days)s))::date AS due_dt,
        lact
      FROM inseminations_raw
      WHERE event_date IS NOT NULL
    )
    SELECT
      COALESCE(count(*) FILTER (WHERE lact > 0), 0)  AS n_cow,
      COALESCE(count(*) FILTER (WHERE lact <= 0), 0) AS n_heifer,
      COALESCE(count(*) FILTER (WHERE lact IS NULL), 0) AS n_unknown
    FROM x
    WHERE due_dt >= %(m_start)s
      AND due_dt <  %(m_next)s;
    """
    df = pd.read_sql(sql, con=engine, params={"gest_days": gest_days, "m_start": m_start, "m_next": m_next})
    n_cow = int(df.loc[0, "n_cow"])
    n_heif = int(df.loc[0, "n_heifer"])
    n_unk = int(df.loc[0, "n_unknown"])

    if (n_cow + n_heif + n_unk) == 0:
        return

    ip = overrides.get("insemination_params", {}) or {}
    cow_spc = _safe_float(ip.get("cow_services_per_conception"), mp.INSEMINATION_PARAMS.cow_services_per_conception)
    heif_spc = _safe_float(ip.get("heifer_services_per_conception"), mp.INSEMINATION_PARAMS.heifer_services_per_conception)

    p_cow = 1.0 / max(1e-9, cow_spc)
    p_heif = 1.0 / max(1e-9, heif_spc)

    exp_cow = n_cow * p_cow
    exp_heif = n_heif * p_heif
    exp_unk = n_unk * p_cow  # неизвестные лактации считаем как коровы

    exp_total = exp_cow + exp_heif + exp_unk

    out["Ожидаемый отёл, всего"] = float(exp_total)
    out["Ожидаемый отёл, из них коров"] = float(exp_cow + exp_unk)
    out["Ожидаемый отёл, из них нетелей"] = float(exp_heif)

    su = overrides.get("semen_usage", {}) or {}
    ssr = overrides.get("semen_sex_ratios", {}) or {}

    cow_trad = _safe_float(su.get("cow_trad"), mp.SEMEN_USAGE_PROBS.cow_trad)
    cow_sex = _safe_float(su.get("cow_sex"), mp.SEMEN_USAGE_PROBS.cow_sex)
    heif_trad = _safe_float(su.get("heifer_trad"), mp.SEMEN_USAGE_PROBS.heifer_trad)
    heif_sex = _safe_float(su.get("heifer_sex"), mp.SEMEN_USAGE_PROBS.heifer_sex)

    trad_bull = _safe_float((ssr.get("trad") or {}).get("bull_share"), mp.SEMEN_SEX_RATIOS["trad"].bull_share)
    sex_bull = _safe_float((ssr.get("sex") or {}).get("bull_share"), mp.SEMEN_SEX_RATIOS["sex"].bull_share)

    bull_share_cow = cow_trad * trad_bull + cow_sex * sex_bull
    bull_share_heif = heif_trad * trad_bull + heif_sex * sex_bull

    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемые бычки (условно)"] = float(exp_bulls)
    out["Ожидаемые тёлочки (условно)"] = float(exp_heifers)


from datetime import date
from typing import Dict, Optional

import pandas as pd


def compute_forecast_from_db(target_date: date, overrides: Optional[dict] = None) -> Dict[str, float]:
    """
    Обёртка над динамическим прогнозом + фолбэк для "ожидаемого отёла".
    Ключевое: приводим входную дату к pd.Timestamp, чтобы не ловить сравнения Timestamp vs date.
    """
    d_end_ts = _to_ts(target_date)
    if pd.isna(d_end_ts):
        raise ValueError(f"target_date is invalid: {target_date!r}")

    # 1) основной расчёт (предпочтительно на Timestamp)
    try:
        out = compute_forecast_dynamic_from_db(d_end_ts, overrides=overrides)
    except (TypeError, ValueError):
        # если внутри compute_forecast_dynamic_from_db ожидают именно datetime.date
        out = compute_forecast_dynamic_from_db(d_end_ts.date(), overrides=overrides)

    # 2) добивка "ожидаемого отёла" (тоже пытаемся сначала с Timestamp)
    try:
        _apply_expected_calving_prob_fallback(out, d_end_ts, overrides)
    except (TypeError, ValueError):
        _apply_expected_calving_prob_fallback(out, d_end_ts.date(), overrides)

    return out
