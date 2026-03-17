from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import date
import math
from typing import Any, Callable, Optional

import pandas as pd

from core.calving_facts import actual_birth_stats_from_tables, is_calving_month_complete_from_tables
from core.constants import INDICATORS, OVERFLOW_COLS
from core.helpers import iter_month_ends, month_end, norm_label, vals_get
from forecast_dynamic import compute_forecast_dynamic_from_tables, latest_data_date

from .common import FARM_BACKTEST_BIRTH_TARGETS, FARM_BACKTEST_TARGETS, FARM_PERCENT_TARGETS, _norm_event_type, _norm_id
from .storage import _load_farm_tables_from_db, _norm_reg_value, _subdivisions_for_farm

_FARM_SANITY_KEYS = [
    "Дойные коровы",
    "Сухостойные коровы",
    "Тёлки 0–3 мес",
    "Тёлки 3–8 мес",
    "Тёлки ≥9 мес",
    "Нетели",
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
]

_TRANSFER_GROUPS_ALL = [
    "Дойные коровы",
    "Сухостойные коровы",
    "Тёлки 0–3 мес",
    "Тёлки 3–8 мес",
    "Тёлки ≥9 мес",
    "Нетели",
    "Бычки 0–2 мес",
]


def backtest_percent_error(pred_val: float, fact_val: float, *, is_pct: bool) -> float | None:
    err_abs = abs(float(pred_val) - float(fact_val))
    if is_pct:
        return (err_abs / abs(float(fact_val)) * 100.0) if abs(float(fact_val)) > 1e-9 else None
    scale = abs(float(pred_val)) + abs(float(fact_val))
    if scale < 20.0:
        return None
    return 200.0 * err_abs / scale


def _month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def _month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return month_end(int(ts.year), int(ts.month))

def _norm_sex_marker_backtest(x: Any) -> str | None:
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

def _actual_birth_stats_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=as_of_date)


def _actual_nonbirth_snapshot_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    dry_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    as_of_date: date,
) -> dict[str, float]:
    out = {
        "Дойные коровы": 0.0,
        "Сухостойные коровы": 0.0,
        "Тёлки 0–3 мес": 0.0,
        "Бычки 0–2 мес": 0.0,
        "Тёлки 3–8 мес": 0.0,
        "Тёлки ≥9 мес": 0.0,
        "Нетели": 0.0,
    }
    as_of_ts = pd.Timestamp(as_of_date).normalize()

    disp = disp_df.copy() if isinstance(disp_df, pd.DataFrame) else pd.DataFrame()
    if not disp.empty:
        disp["event_date_n"] = pd.to_datetime(disp.get("event_date"), errors="coerce").dt.normalize()
        disp["reg_s"] = disp.get("reg", pd.Series(dtype=object)).map(_norm_id)
        disp = disp[(disp["event_date_n"].notna()) & (disp["event_date_n"] <= as_of_ts) & (disp["reg_s"] != "")]
    disposed: set[str] = set(disp["reg_s"].astype(str).tolist()) if not disp.empty else set()

    ins = ins_df.copy() if isinstance(ins_df, pd.DataFrame) else pd.DataFrame()
    if not ins.empty:
        ins["event_date_n"] = pd.to_datetime(ins.get("event_date"), errors="coerce").dt.normalize()
        ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(_norm_id)
        ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
        ins = ins[(ins["event_date_n"].notna()) & (ins["event_date_n"] <= as_of_ts) & (ins["reg_s"] != "")]
    cows_from_ins = set(ins.loc[ins["lact_n"] > 0, "reg_s"].astype(str).tolist()) if not ins.empty else set()
    neteli_from_ins = set(ins.loc[ins["lact_n"] <= 0, "reg_s"].astype(str).tolist()) if not ins.empty else set()

    dry = dry_df.copy() if isinstance(dry_df, pd.DataFrame) else pd.DataFrame()
    if not dry.empty:
        dry["event_date_n"] = pd.to_datetime(dry.get("event_date"), errors="coerce").dt.normalize()
        dry["reg_s"] = dry.get("reg", pd.Series(dtype=object)).map(_norm_id)
        dry = dry[(dry["event_date_n"].notna()) & (dry["event_date_n"] <= as_of_ts) & (dry["reg_s"] != "")]
    if not dry.empty:
        last_dry = (
            dry.sort_values(["reg_s", "event_date_n"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
            .set_index("reg_s")["event_date_n"]
            .to_dict()
        )
    else:
        last_dry = {}

    calv = calv_df.copy() if isinstance(calv_df, pd.DataFrame) else pd.DataFrame()
    if not calv.empty:
        calv["event_date_n"] = pd.to_datetime(calv.get("event_date"), errors="coerce").dt.normalize()
        calv["birth_date_n"] = pd.to_datetime(calv.get("birth_date"), errors="coerce").dt.normalize()
        calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).map(_norm_event_type)
        calv["reg_s"] = calv.get("reg", pd.Series(dtype=object)).map(_norm_id)
        calv["mother_reg_s"] = calv.get("mother_reg", pd.Series(dtype=object)).map(_norm_id)
        calv["sex_norm"] = calv.get("sex", pd.Series(dtype=object)).map(_norm_sex_marker_backtest)
        calv = calv[(calv["event_date_n"].notna()) & (calv["event_date_n"] <= as_of_ts)]
        born = calv.loc[calv["event_type_n"] == "РОЖДЕН"].copy()
    else:
        born = pd.DataFrame()

    if not born.empty:
        born["birth_dt_n"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
    else:
        born["birth_dt_n"] = pd.NaT

    mother_with_calv = set(born.loc[born["mother_reg_s"] != "", "mother_reg_s"].astype(str).tolist()) if not born.empty else set()
    if not born.empty:
        last_calv_by_mother = (
            born.loc[born["mother_reg_s"] != "", ["mother_reg_s", "event_date_n"]]
            .sort_values(["mother_reg_s", "event_date_n"], kind="mergesort")
            .drop_duplicates(subset=["mother_reg_s"], keep="last")
            .set_index("mother_reg_s")["event_date_n"]
            .to_dict()
        )
    else:
        last_calv_by_mother = {}

    cow_candidates = set()
    cow_candidates |= cows_from_ins
    cow_candidates |= set(last_dry.keys())
    cow_candidates |= mother_with_calv
    cows_alive = {reg for reg in cow_candidates if reg and reg not in disposed}

    dry_count = 0
    for reg in cows_alive:
        dry_dt = last_dry.get(reg)
        if dry_dt is None or pd.isna(dry_dt):
            continue
        calv_dt = last_calv_by_mother.get(reg)
        if calv_dt is None or pd.isna(calv_dt):
            dry_count += 1
        elif pd.Timestamp(dry_dt) > pd.Timestamp(calv_dt):
            dry_count += 1
    doy_count = max(0, len(cows_alive) - dry_count)

    neteli_alive = {
        reg for reg in neteli_from_ins
        if reg and reg not in disposed and reg not in cows_alive and reg not in mother_with_calv
    }

    calf_excluded = set(cows_alive) | set(neteli_alive)
    if not born.empty:
        calves_f = born.loc[(born["sex_norm"] == "F") & (born["reg_s"] != ""), ["reg_s", "birth_dt_n"]].copy()
        calves_m = born.loc[(born["sex_norm"] == "M") & (born["reg_s"] != ""), ["reg_s", "birth_dt_n"]].copy()
    else:
        calves_f = pd.DataFrame(columns=["reg_s", "birth_dt_n"])
        calves_m = pd.DataFrame(columns=["reg_s", "birth_dt_n"])

    def _count_by_age(df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype=float)
        work = df.copy()
        work = work[work["birth_dt_n"].notna()].copy()
        if work.empty:
            return pd.Series(dtype=float)
        work = work[~work["reg_s"].astype(str).isin(disposed)]
        work = work[~work["reg_s"].astype(str).isin(calf_excluded)]
        if work.empty:
            return pd.Series(dtype=float)
        age_days = (as_of_ts - pd.to_datetime(work["birth_dt_n"], errors="coerce")).dt.days
        return age_days

    age_f = _count_by_age(calves_f)
    age_m = _count_by_age(calves_m)

    h0_3 = float(((age_f >= 0) & (age_f < 90)).sum()) if not age_f.empty else 0.0
    h3_8 = float(((age_f >= 90) & (age_f < 270)).sum()) if not age_f.empty else 0.0
    h9p = float((age_f >= 270).sum()) if not age_f.empty else 0.0
    b0_2 = float(((age_m >= 0) & (age_m < 61)).sum()) if not age_m.empty else 0.0

    out["Дойные коровы"] = float(doy_count)
    out["Сухостойные коровы"] = float(dry_count)
    out["Нетели"] = float(len(neteli_alive))
    out["Тёлки 0–3 мес"] = h0_3
    out["Бычки 0–2 мес"] = b0_2
    out["Тёлки 3–8 мес"] = h3_8
    out["Тёлки ≥9 мес"] = h9p
    return out

def _is_fact_month_complete_for_subdivision(calv_df: pd.DataFrame, month_end_date: date) -> bool:
    return is_calving_month_complete_from_tables(calv_df, month_end_date)

def _pred_metric_value_for_backtest(pred_vals: dict, metric_name: str, nmap: dict[str, float]) -> float:
    if metric_name in FARM_PERCENT_TARGETS:
        pred_bull = float(vals_get(pred_vals, "Ожидаемые бычки", nmap) or 0.0)
        pred_heif = float(vals_get(pred_vals, "Ожидаемые тёлочки", nmap) or 0.0)
        den = pred_bull + pred_heif
        if den <= 0:
            return 0.0
        if metric_name == "Доля бычков среди рождений, %":
            return pred_bull / den * 100.0
        return pred_heif / den * 100.0
    return float(vals_get(pred_vals, metric_name, nmap) or 0.0)

def _run_farm_backtesting(
    farm_name: str,
    metric_name: str,
    bt_months: int,
    bt_horizon: int,
    complete_only: bool,
    params: dict,
    params_for_subdivision: Optional[Callable[[str, dict[str, pd.DataFrame]], dict]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subdivisions = _subdivisions_for_farm(farm_name, ready_only=True)
    if not subdivisions:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "no_ready_subdivisions"}

    tables_by_sub: dict[str, dict[str, pd.DataFrame]] = {}
    latest_by_sub: dict[str, date] = {}
    base_dates: list[date] = []
    for sub in subdivisions:
        tables = _load_farm_tables_from_db(sub)
        tables_by_sub[sub] = tables
        latest_dt = latest_data_date(tables)
        latest_by_sub[sub] = latest_dt
        base_dates.append(latest_dt)

    if not base_dates:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "no_data"}

    base_date_bt = max(base_dates)
    last_me_bt = month_end(base_date_bt.year, base_date_bt.month)
    target_months = [_month_end_shift(last_me_bt, -i) for i in range(bt_months - 1, -1, -1)]

    farm_rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []

    steps_total = max(1, len(target_months) * len(subdivisions))
    step_idx = 0
    skipped_months = 0
    skipped_sub_months = 0

    for target_me in target_months:
        as_of_me = _month_end_shift(target_me, -int(bt_horizon))

        month_pred = 0.0
        month_fact = 0.0
        month_pred_bulls = 0.0
        month_pred_heif = 0.0
        month_fact_bulls = 0.0
        month_fact_heif = 0.0
        used_subs = 0

        for sub in subdivisions:
            step_idx += 1
            if progress_cb is not None:
                progress_cb(step_idx, steps_total, f"{farm_name} / {sub} / {_month_label(target_me)}")

            tables = tables_by_sub[sub]
            calv_df = tables.get("calv", pd.DataFrame())
            ins_df = tables.get("ins", pd.DataFrame())
            dry_df = tables.get("dry", pd.DataFrame())
            disp_df = tables.get("disp", pd.DataFrame())
            if metric_name in FARM_BACKTEST_BIRTH_TARGETS or metric_name in FARM_PERCENT_TARGETS:
                is_complete = _is_fact_month_complete_for_subdivision(calv_df, target_me)
            else:
                is_complete = bool(latest_by_sub.get(sub, date.min) >= target_me)
            if complete_only and not is_complete:
                skipped_sub_months += 1
                continue

            sub_params = params
            if params_for_subdivision is not None:
                cand = params_for_subdivision(sub, tables)
                if not isinstance(cand, dict) or not cand:
                    raise ValueError(f"{sub}: не получены параметры подразделения.")
                sub_params = cand

            pred_vals = compute_forecast_dynamic_from_tables(
                tables,
                target_me,
                overrides=sub_params,
                as_of_date=as_of_me,
            ) or {}
            nmap = {norm_label(k): v for k, v in pred_vals.items()}
            pred_val = float(_pred_metric_value_for_backtest(pred_vals, metric_name, nmap))
            fact_stats = _actual_birth_stats_month_from_tables(calv_df, ins_df, target_me, as_of_date=None)
            fact_nonbirth = _actual_nonbirth_snapshot_from_tables(
                calv_df=calv_df,
                ins_df=ins_df,
                dry_df=dry_df,
                disp_df=disp_df,
                as_of_date=target_me,
            )
            if metric_name in FARM_BACKTEST_BIRTH_TARGETS or metric_name in FARM_PERCENT_TARGETS:
                fact_val = float(fact_stats.get(metric_name, 0.0))
            else:
                fact_val = float(fact_nonbirth.get(metric_name, 0.0))

            pred_bulls = float(vals_get(pred_vals, "Ожидаемые бычки", nmap) or 0.0)
            pred_heifers = float(vals_get(pred_vals, "Ожидаемые тёлочки", nmap) or 0.0)
            fact_bulls = float(fact_stats.get("Ожидаемые бычки", 0.0))
            fact_heifers = float(fact_stats.get("Ожидаемые тёлочки", 0.0))

            if metric_name in FARM_PERCENT_TARGETS:
                month_pred_bulls += pred_bulls
                month_pred_heif += pred_heifers
                month_fact_bulls += fact_bulls
                month_fact_heif += fact_heifers
                fact_weight = fact_bulls + fact_heifers
            else:
                month_pred += pred_val
                month_fact += fact_val
                fact_weight = abs(fact_val)

            err_sub = pred_val - fact_val
            ape_sub = backtest_percent_error(pred_val, fact_val, is_pct=(metric_name in FARM_PERCENT_TARGETS))
            sub_rows.append(
                {
                    "Месяц факта": target_me.strftime("%Y-%m"),
                    "as-of (на дату)": as_of_me.strftime("%Y-%m"),
                    "Подразделение": sub,
                    "Показатель": metric_name,
                    "Прогноз": round(pred_val, 1),
                    "Факт": round(fact_val, 1),
                    "Ошибка": round(err_sub, 1),
                    "APE, %": None if ape_sub is None else round(float(ape_sub), 1),
                    "Полный месяц факта": bool(is_complete),
                    "Вес по факту": float(fact_weight),
                }
            )
            used_subs += 1

        if used_subs == 0:
            skipped_months += 1
            continue

        if metric_name in FARM_PERCENT_TARGETS:
            den_pred = month_pred_bulls + month_pred_heif
            den_fact = month_fact_bulls + month_fact_heif
            if metric_name == "Доля бычков среди рождений, %":
                month_pred = (month_pred_bulls / den_pred * 100.0) if den_pred > 0 else 0.0
                month_fact = (month_fact_bulls / den_fact * 100.0) if den_fact > 0 else 0.0
            else:
                month_pred = (month_pred_heif / den_pred * 100.0) if den_pred > 0 else 0.0
                month_fact = (month_fact_heif / den_fact * 100.0) if den_fact > 0 else 0.0

        err = month_pred - month_fact
        ape = backtest_percent_error(month_pred, month_fact, is_pct=(metric_name in FARM_PERCENT_TARGETS))
        farm_rows.append(
            {
                "Месяц факта": target_me.strftime("%Y-%m"),
                "as-of (на дату)": as_of_me.strftime("%Y-%m"),
                "Показатель": metric_name,
                "Прогноз": round(month_pred, 1),
                "Факт": round(month_fact, 1),
                "Ошибка": round(err, 1),
                "APE, %": None if ape is None else round(float(ape), 1),
                "Подразделений в расчёте": int(used_subs),
            }
        )

    bt_df = pd.DataFrame(farm_rows)
    sub_df = pd.DataFrame(sub_rows)
    if sub_df.empty:
        summary = {
            "farm": farm_name,
            "metric": metric_name,
            "months": int(bt_months),
            "horizon": int(bt_horizon),
            "complete_only": bool(complete_only),
            "skipped_months": int(skipped_months),
            "skipped_sub_months": int(skipped_sub_months),
            "subdivisions_n": len(subdivisions),
        }
        return bt_df, pd.DataFrame(), summary

    metric_cols = sub_df.copy()
    metric_cols["Ошибка"] = pd.to_numeric(metric_cols["Ошибка"], errors="coerce")
    metric_cols["APE, %"] = pd.to_numeric(metric_cols["APE, %"], errors="coerce")
    metric_cols["Факт"] = pd.to_numeric(metric_cols["Факт"], errors="coerce")
    metric_cols["Прогноз"] = pd.to_numeric(metric_cols["Прогноз"], errors="coerce")
    metric_cols["Вес по факту"] = pd.to_numeric(metric_cols["Вес по факту"], errors="coerce").fillna(0.0)

    sub_summary = (
        metric_cols.groupby("Подразделение", as_index=False)
        .agg(
            n_months=("Месяц факта", "count"),
            mae=("Ошибка", lambda x: float(pd.to_numeric(x, errors="coerce").abs().mean())),
            bias=("Ошибка", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
            weight_raw=("Вес по факту", "sum"),
        )
    )

    if metric_name in FARM_PERCENT_TARGETS:
        sub_perc = (
            metric_cols.groupby("Подразделение")["APE, %"]
            .apply(lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if not pd.to_numeric(x, errors="coerce").dropna().empty else float("nan"))
            .rename("perc_err")
            .reset_index()
        )
        perc_col = "Средняя процентная погрешность, %"
    else:
        stable_rows = metric_cols.loc[
            (metric_cols["Прогноз"].abs() + metric_cols["Факт"].abs()) >= 20.0,
            ["Подразделение", "Прогноз", "Факт", "Ошибка"],
        ].copy()
        if stable_rows.empty:
            sub_perc = pd.DataFrame({"Подразделение": sub_summary["Подразделение"], "perc_err": float("nan")})
        else:
            stable_rows["scale"] = stable_rows["Прогноз"].abs() + stable_rows["Факт"].abs()
            stable_rows["err_abs"] = stable_rows["Ошибка"].abs()
            sub_perc = (
                stable_rows.groupby("Подразделение", as_index=False)
                .agg(den=("scale", "sum"), num=("err_abs", "sum"))
            )
            sub_perc["perc_err"] = sub_perc.apply(
                lambda row: float(200.0 * row["num"] / row["den"]) if float(row["den"]) > 1e-9 else float("nan"),
                axis=1,
            )
            sub_perc = sub_perc[["Подразделение", "perc_err"]]
        perc_col = "Симметричная процентная погрешность, %"

    sub_summary = sub_summary.merge(sub_perc, on="Подразделение", how="left")

    w_sum = float(pd.to_numeric(sub_summary["weight_raw"], errors="coerce").fillna(0.0).sum())
    if w_sum <= 1e-9:
        sub_summary["Вес, %"] = 100.0 / max(1, len(sub_summary))
    else:
        sub_summary["Вес, %"] = pd.to_numeric(sub_summary["weight_raw"], errors="coerce").fillna(0.0) / w_sum * 100.0

    mae_col = "Средняя погрешность, п.п." if metric_name in FARM_PERCENT_TARGETS else "Средняя погрешность, гол."
    bias_col = "Смещение (средняя ошибка), п.п." if metric_name in FARM_PERCENT_TARGETS else "Смещение (средняя ошибка), гол."

    sub_summary[mae_col] = pd.to_numeric(sub_summary["mae"], errors="coerce")
    sub_summary[bias_col] = pd.to_numeric(sub_summary["bias"], errors="coerce")
    sub_summary[perc_col] = pd.to_numeric(sub_summary["perc_err"], errors="coerce")
    sub_summary = sub_summary[["Подразделение", "n_months", "Вес, %", mae_col, perc_col, bias_col]].sort_values(
        ["Вес, %", "Подразделение"],
        ascending=[False, True],
        kind="mergesort",
    )

    weights = pd.to_numeric(sub_summary["Вес, %"], errors="coerce").fillna(0.0) / 100.0
    weighted_mae = float((weights * pd.to_numeric(sub_summary[mae_col], errors="coerce").fillna(0.0)).sum())
    weighted_bias = float((weights * pd.to_numeric(sub_summary[bias_col], errors="coerce").fillna(0.0)).sum())

    if metric_name in FARM_PERCENT_TARGETS:
        weighted_mape = None
    else:
        bt_tmp = bt_df.copy()
        bt_tmp["Факт"] = pd.to_numeric(bt_tmp.get("Факт"), errors="coerce").fillna(0.0)
        bt_tmp["Прогноз"] = pd.to_numeric(bt_tmp.get("Прогноз"), errors="coerce").fillna(0.0)
        bt_tmp["Ошибка"] = pd.to_numeric(bt_tmp.get("Ошибка"), errors="coerce").fillna(0.0)
        scale = bt_tmp["Прогноз"].abs() + bt_tmp["Факт"].abs()
        stable_mask = scale >= 20.0
        den = float(scale.loc[stable_mask].sum())
        num = float(bt_tmp.loc[stable_mask, "Ошибка"].abs().sum())
        weighted_mape = (200.0 * num / den) if den > 1e-9 else None

    summary = {
        "farm": farm_name,
        "metric": metric_name,
        "months": int(bt_months),
        "horizon": int(bt_horizon),
        "complete_only": bool(complete_only),
        "skipped_months": int(skipped_months),
        "skipped_sub_months": int(skipped_sub_months),
        "subdivisions_n": len(subdivisions),
        "weighted_mae": weighted_mae,
        "weighted_mape": weighted_mape,
        "weighted_bias": weighted_bias,
    }
    return bt_df, sub_summary.reset_index(drop=True), summary

def _compute_farm_forecast(
    farm_name: str,
    tables: dict[str, pd.DataFrame],
    target_month_end: date,
    params: dict,
    start_month_end: Optional[date] = None,
    progress_cb: Optional[Callable[[int, int, date], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_date = latest_data_date(tables)
    base_month_end = month_end(base_date.year, base_date.month)
    start_me = month_end(start_month_end.year, start_month_end.month) if isinstance(start_month_end, date) else base_month_end

    if target_month_end < start_me:
        month_ends = [target_month_end]
    else:
        month_ends = iter_month_ends(start_me.year, start_me.month, target_month_end.year, target_month_end.month)

    rows: list[dict[str, Any]] = []
    total_steps = len(month_ends)
    for step_i, d_end in enumerate(month_ends, start=1):
        if progress_cb is not None:
            try:
                progress_cb(step_i, total_steps, d_end)
            except Exception:
                pass
        vals = compute_forecast_dynamic_from_tables(tables, d_end, overrides=params) or {}
        nmap = {norm_label(k): v for k, v in vals.items()}

        row = {"Месяц": _month_label(d_end), "Хозяйство": farm_name}
        for k in [*INDICATORS, *OVERFLOW_COLS]:
            row[k] = float(vals_get(vals, k, nmap) or 0.0)
        rows.append(row)

    info = {
        "farm": farm_name,
        "base_date": base_date,
        "base_month_end": base_month_end,
        "start_month_end": start_me,
        "months_n": len(month_ends),
        "rows_calv": int(len(tables["calv"])),
        "rows_ins": int(len(tables["ins"])),
        "rows_dry": int(len(tables["dry"])),
        "rows_disp": int(len(tables["disp"])),
        "rows_bulls": int(len(tables["bulls"])),
    }
    return pd.DataFrame(rows), info

def _target_row_map(monthly_df: pd.DataFrame, target_month_end: date) -> dict[str, float]:
    if not isinstance(monthly_df, pd.DataFrame) or monthly_df.empty:
        return {}
    month_label = _month_label(target_month_end)
    part = monthly_df.loc[monthly_df["Месяц"].astype(str) == month_label]
    if part.empty:
        return {}
    row = part.iloc[-1].to_dict()
    out: dict[str, float] = {}
    for k in [*INDICATORS, *OVERFLOW_COLS]:
        try:
            out[k] = float(pd.to_numeric(row.get(k), errors="coerce") or 0.0)
        except Exception:
            out[k] = 0.0
    return out

def _farm_sanity_violations_against_subdivisions(
    farm_monthly_df: pd.DataFrame,
    subdivisions: list[str],
    target_month_end: date,
    params: dict,
) -> list[str]:
    if not subdivisions:
        return []
    farm_row = _target_row_map(farm_monthly_df, target_month_end)
    if not farm_row:
        return []

    violations: list[str] = []
    for sub in subdivisions:
        sub_tables = _load_farm_tables_from_db(sub)
        sub_vals = compute_forecast_dynamic_from_tables(sub_tables, target_month_end, overrides=params) or {}
        nmap = {norm_label(k): v for k, v in sub_vals.items()}
        for k in _FARM_SANITY_KEYS:
            farm_v = float(farm_row.get(k, 0.0) or 0.0)
            sub_v = float(vals_get(sub_vals, k, nmap) or 0.0)
            if farm_v + 1e-6 < sub_v:
                violations.append(f"{k}: хозяйство={farm_v:.1f} < {sub}={sub_v:.1f}")
    return violations

def _compute_farm_forecast_sum_of_subdivisions(
    farm_name: str,
    subdivisions: list[str],
    target_month_end: date,
    params: dict,
    params_for_subdivision: Optional[Callable[[str, dict[str, pd.DataFrame]], dict]] = None,
    progress_cb: Optional[Callable[[str, int, int, date], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sub_payload: list[tuple[str, dict[str, pd.DataFrame], date]] = []
    base_months: list[date] = []
    for sub in subdivisions:
        tables = _load_farm_tables_from_db(sub)
        base_date = latest_data_date(tables)
        base_me = month_end(base_date.year, base_date.month)
        sub_payload.append((sub, tables, base_me))
        base_months.append(base_me)

    common_start_month_end = max(base_months) if base_months else target_month_end

    sub_frames: list[pd.DataFrame] = []
    rows_meta: list[dict[str, Any]] = []

    for sub, tables, base_me in sub_payload:
        sub_params = params
        if params_for_subdivision is not None:
            cand = params_for_subdivision(sub, tables)
            if not isinstance(cand, dict) or not cand:
                raise ValueError(f"{sub}: не получены параметры подразделения.")
            sub_params = cand

        def _sub_cb(step_i: int, total_steps: int, d_end: date) -> None:
            if progress_cb is not None:
                progress_cb(sub, step_i, total_steps, d_end)

        monthly_sub, info_sub = _compute_farm_forecast(
            sub,
            tables,
            target_month_end,
            sub_params,
            start_month_end=common_start_month_end,
            progress_cb=_sub_cb,
        )
        if monthly_sub.empty:
            continue
        work = monthly_sub.copy()
        work["Хозяйство"] = farm_name
        sub_frames.append(work)
        rows_meta.append(
            {
                "subdivision": sub,
                "base_month_end": _month_label(base_me),
                "rows_calv": int(info_sub.get("rows_calv", 0)),
                "rows_ins": int(info_sub.get("rows_ins", 0)),
                "rows_dry": int(info_sub.get("rows_dry", 0)),
                "rows_disp": int(info_sub.get("rows_disp", 0)),
                "rows_bulls": int(info_sub.get("rows_bulls", 0)),
            }
        )

    if not sub_frames:
        return pd.DataFrame(), {"farm": farm_name, "calc_mode": "sum_subdivisions", "subdivisions_n": 0}

    all_sub = pd.concat(sub_frames, ignore_index=True)
    num_cols = [c for c in [*INDICATORS, *OVERFLOW_COLS] if c in all_sub.columns]
    agg = all_sub.groupby("Месяц", as_index=False)[num_cols].sum()
    agg.insert(1, "Хозяйство", farm_name)
    agg = agg.sort_values("Месяц", kind="mergesort").reset_index(drop=True)

    info = {
        "farm": farm_name,
        "calc_mode": "sum_subdivisions",
        "subdivisions_n": len(subdivisions),
        "common_start_month_end": common_start_month_end,
        "subdivisions_meta": rows_meta,
    }
    return agg, info

def _is_move_reason(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    if not s:
        return False
    return ("ПЕРЕЕЗД" in s) or ("ПЕРЕВОД" in s) or ("ПЕРЕМЕЩ" in s)

def _subdivision_cow_balance_snapshot(
    farm_name: str,
    target_month_end: date,
    params: dict,
) -> pd.DataFrame:
    subs = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subs:
        return pd.DataFrame()

    lookback_start = pd.Timestamp(target_month_end) - pd.Timedelta(days=365)
    rows: list[dict[str, Any]] = []
    hist_weights: dict[str, float] = {}

    for sub in subs:
        tables = _load_farm_tables_from_db(sub)
        vals = compute_forecast_dynamic_from_tables(tables, target_month_end, overrides=params) or {}
        nmap = {norm_label(k): v for k, v in vals.items()}
        doy = float(vals_get(vals, "Дойные коровы", nmap) or 0.0)
        dry = float(vals_get(vals, "Сухостойные коровы", nmap) or 0.0)
        cows = doy + dry

        ins = tables.get("ins", pd.DataFrame()).copy()
        if isinstance(ins, pd.DataFrame) and not ins.empty:
            ins["event_date"] = pd.to_datetime(ins.get("event_date"), errors="coerce")
            ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
            ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(_norm_reg_value)
            hist_regs = ins.loc[
                (ins["event_date"].notna())
                & (ins["event_date"] >= lookback_start)
                & (ins["event_date"] <= pd.Timestamp(target_month_end))
                & (ins["lact_n"] > 0)
                & (ins["reg_s"] != ""),
                "reg_s",
            ].nunique()
            hist_weights[sub] = float(hist_regs)
        else:
            hist_weights[sub] = 0.0

        rows.append(
            {
                "Подразделение": sub,
                "Дойные коровы": doy,
                "Сухостойные коровы": dry,
                "Коровы всего": cows,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_cows = float(df["Коровы всего"].sum())
    w_sum = float(sum(hist_weights.values()))
    if w_sum <= 1e-9:
        equal = 1.0 / max(1, len(df))
        df["Историческая доля"] = equal
    else:
        df["Историческая доля"] = df["Подразделение"].map(lambda x: float(hist_weights.get(str(x), 0.0)) / w_sum)

    df["Оценка мест (коровы)"] = df["Историческая доля"] * total_cows
    df["Переполнение (оценка)"] = (df["Коровы всего"] - df["Оценка мест (коровы)"]).clip(lower=0.0)
    df["Свободно мест (оценка)"] = (df["Оценка мест (коровы)"] - df["Коровы всего"]).clip(lower=0.0)
    return df.sort_values("Подразделение").reset_index(drop=True)

def _extract_move_flows_from_dry(
    farm_name: str,
    max_days_to_find_destination: int = 120,
) -> pd.DataFrame:
    subs = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subs:
        return pd.DataFrame(columns=["Источник", "Приёмник", "Переездов"])

    events_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []

    def _collect_events(df: pd.DataFrame, sub: str) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        if "reg" not in df.columns or "event_date" not in df.columns:
            return
        d = df[["reg", "event_date"]].copy()
        d["reg_s"] = d["reg"].map(_norm_reg_value)
        d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")
        d = d[(d["reg_s"] != "") & d["event_date"].notna()].copy()
        if d.empty:
            return
        d["subdivision"] = sub
        events_rows.extend(d[["reg_s", "event_date", "subdivision"]].to_dict(orient="records"))

    for sub in subs:
        tables = _load_farm_tables_from_db(sub)
        _collect_events(tables.get("calv", pd.DataFrame()), sub)
        _collect_events(tables.get("ins", pd.DataFrame()), sub)
        _collect_events(tables.get("dry", pd.DataFrame()), sub)
        _collect_events(tables.get("disp", pd.DataFrame()), sub)

        found_moves = False
        dry = tables.get("dry", pd.DataFrame())
        if isinstance(dry, pd.DataFrame) and not dry.empty and "move_reason" in dry.columns:
            d = dry[["reg", "event_date", "move_reason"]].copy()
            d["reg_s"] = d["reg"].map(_norm_reg_value)
            d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")
            d = d[
                (d["reg_s"] != "")
                & d["event_date"].notna()
                & d["move_reason"].map(_is_move_reason)
            ].copy()
            if not d.empty:
                d["source"] = sub
                move_rows.extend(d[["reg_s", "event_date", "source"]].to_dict(orient="records"))
                found_moves = True

        if not found_moves:
            disp = tables.get("disp", pd.DataFrame())
            if isinstance(disp, pd.DataFrame) and not disp.empty and "disposal_reason" in disp.columns:
                d2 = disp[["reg", "event_date", "disposal_reason"]].copy()
                d2["reg_s"] = d2["reg"].map(_norm_reg_value)
                d2["event_date"] = pd.to_datetime(d2["event_date"], errors="coerce")
                d2 = d2[
                    (d2["reg_s"] != "")
                    & d2["event_date"].notna()
                    & d2["disposal_reason"].map(_is_move_reason)
                ].copy()
                if not d2.empty:
                    d2["source"] = sub
                    move_rows.extend(d2[["reg_s", "event_date", "source"]].to_dict(orient="records"))

    if not move_rows or not events_rows:
        return pd.DataFrame(columns=["Источник", "Приёмник", "Переездов"])

    events = (
        pd.DataFrame(events_rows)
        .drop_duplicates(subset=["reg_s", "event_date", "subdivision"], keep="first")
        .sort_values(["reg_s", "event_date", "subdivision"], kind="mergesort")
    )
    moves = pd.DataFrame(move_rows).sort_values(["reg_s", "event_date"], kind="mergesort")

    by_reg: dict[str, tuple[list[int], list[str]]] = {}
    for reg, grp in events.groupby("reg_s", sort=False):
        ords = grp["event_date"].astype("int64").tolist()
        subs_list = grp["subdivision"].astype(str).tolist()
        by_reg[str(reg)] = (ords, subs_list)

    horizon_ns = int(pd.Timedelta(days=max_days_to_find_destination).value)
    flow_counts: dict[tuple[str, str], int] = defaultdict(int)

    for r in moves.itertuples(index=False):
        reg = str(r.reg_s)
        src = str(r.source)
        move_ord = int(pd.Timestamp(r.event_date).value)
        pack = by_reg.get(reg)
        if pack is None:
            continue
        ords, subs_list = pack
        idx = bisect_left(ords, move_ord)
        hi = move_ord + horizon_ns
        dst = None
        while idx < len(ords) and ords[idx] <= hi:
            cand = str(subs_list[idx])
            if cand and cand != src:
                dst = cand
                break
            idx += 1
        if dst:
            flow_counts[(src, dst)] += 1

    if not flow_counts:
        return pd.DataFrame(columns=["Источник", "Приёмник", "Переездов"])

    out = pd.DataFrame(
        [{"Источник": s, "Приёмник": d, "Переездов": int(n)} for (s, d), n in flow_counts.items()]
    )
    return out.sort_values(["Переездов", "Источник", "Приёмник"], ascending=[False, True, True]).reset_index(drop=True)

def _historical_subdivision_shares(farm_name: str, target_month_end: date) -> dict[str, float]:
    subs = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subs:
        return {}

    lookback_start = pd.Timestamp(target_month_end) - pd.Timedelta(days=365)
    weights: dict[str, float] = {}
    for sub in subs:
        tables = _load_farm_tables_from_db(sub)
        ins = tables.get("ins", pd.DataFrame()).copy()
        if not isinstance(ins, pd.DataFrame) or ins.empty:
            weights[sub] = 0.0
            continue
        ins["event_date"] = pd.to_datetime(ins.get("event_date"), errors="coerce")
        ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
        ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(_norm_reg_value)
        hist_regs = ins.loc[
            (ins["event_date"].notna())
            & (ins["event_date"] >= lookback_start)
            & (ins["event_date"] <= pd.Timestamp(target_month_end))
            & (ins["lact_n"] > 0)
            & (ins["reg_s"] != ""),
            "reg_s",
        ].nunique()
        weights[sub] = float(hist_regs)

    w_sum = float(sum(weights.values()))
    if w_sum <= 1e-9:
        eq = 1.0 / max(1, len(subs))
        return {sub: eq for sub in subs}
    return {sub: float(weights.get(sub, 0.0)) / w_sum for sub in subs}

def _subdivision_monthly_cows(
    farm_name: str,
    target_month_end: date,
    params: dict,
    params_for_subdivision: Optional[Callable[[str, dict[str, pd.DataFrame]], dict]] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> pd.DataFrame:
    subs = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subs:
        return pd.DataFrame(columns=["Месяц", "Подразделение", "Дойные коровы", "Сухостойные коровы", "Коровы всего"])

    sub_payload: list[tuple[str, dict[str, pd.DataFrame], date]] = []
    base_months: list[date] = []
    for sub in subs:
        tables = _load_farm_tables_from_db(sub)
        base_date = latest_data_date(tables)
        base_me = month_end(base_date.year, base_date.month)
        sub_payload.append((sub, tables, base_me))
        base_months.append(base_me)
    common_start_month_end = max(base_months) if base_months else target_month_end

    rows: list[pd.DataFrame] = []
    total_subs = len(subs)
    for idx, (sub, tables, _base_me) in enumerate(sub_payload, start=1):
        if progress_cb is not None:
            try:
                progress_cb(sub, idx, total_subs)
            except Exception:
                pass
        sub_params = params
        if params_for_subdivision is not None:
            cand = params_for_subdivision(sub, tables)
            if not isinstance(cand, dict) or not cand:
                raise ValueError(f"{sub}: не получены параметры подразделения.")
            sub_params = cand
        monthly_sub, _ = _compute_farm_forecast(
            sub,
            tables,
            target_month_end,
            sub_params,
            start_month_end=common_start_month_end,
            progress_cb=None,
        )
        if not isinstance(monthly_sub, pd.DataFrame) or monthly_sub.empty:
            continue
        work = monthly_sub.copy()
        work["Подразделение"] = sub
        work["Дойные коровы"] = pd.to_numeric(work.get("Дойные коровы"), errors="coerce").fillna(0.0)
        work["Сухостойные коровы"] = pd.to_numeric(work.get("Сухостойные коровы"), errors="coerce").fillna(0.0)
        work["Коровы всего"] = work["Дойные коровы"] + work["Сухостойные коровы"]
        rows.append(work[["Месяц", "Подразделение", "Дойные коровы", "Сухостойные коровы", "Коровы всего"]].copy())

    if not rows:
        return pd.DataFrame(columns=["Месяц", "Подразделение", "Дойные коровы", "Сухостойные коровы", "Коровы всего"])
    out = pd.concat(rows, ignore_index=True)
    out["Месяц"] = out["Месяц"].astype(str)
    return out.sort_values(["Месяц", "Подразделение"], kind="mergesort").reset_index(drop=True)


def _split_transfer_by_cow_groups(move_total: float, src_groups: dict[str, float]) -> list[dict[str, Any]]:
    total_raw = max(0.0, float(move_total or 0.0))
    total = int(round(total_raw))
    if total <= 0:
        return []

    keys = ("Дойные коровы", "Сухостойные коровы")
    avail = {k: max(0.0, float(src_groups.get(k, 0.0) or 0.0)) for k in keys}
    avail_total = float(sum(avail.values()))
    if avail_total <= 1e-9:
        out0 = {g: 0 for g in _TRANSFER_GROUPS_ALL}
        out0["Дойные коровы"] = int(total)
        return [{"Группа": g, "Рекомендовано перевести, голов": int(out0[g])} for g in _TRANSFER_GROUPS_ALL]

    avail_int = {k: max(0, int(round(avail[k]))) for k in keys}
    base = {
        k: max(0, min(avail_int[k], int(math.floor(total * (avail[k] / avail_total)))))
        for k in keys
    }
    rem = int(total - sum(base.values()))
    if rem > 0:
        frac_order = sorted(
            keys,
            key=lambda k: (total * (avail[k] / avail_total) - math.floor(total * (avail[k] / avail_total))),
            reverse=True,
        )
        for k in frac_order:
            if rem <= 0:
                break
            free = max(0, avail_int[k] - base[k])
            if free <= 0:
                continue
            add = min(free, rem)
            base[k] += int(add)
            rem -= add
    out_all = {g: 0 for g in _TRANSFER_GROUPS_ALL}
    out_all["Дойные коровы"] = int(base.get("Дойные коровы", 0))
    out_all["Сухостойные коровы"] = int(base.get("Сухостойные коровы", 0))
    rows = [{"Группа": g, "Рекомендовано перевести, голов": int(out_all[g])} for g in _TRANSFER_GROUPS_ALL]
    if rem > 0:
        rows[0]["Рекомендовано перевести, голов"] = int(rows[0]["Рекомендовано перевести, голов"] + int(rem))
    return rows


def _demo_capacity_by_subdivision(subs: list[str], total_cows: float, shares: dict[str, float]) -> dict[str, float]:
    if not subs:
        return {}
    ordered = sorted(str(x) for x in subs)
    eq_share = 1.0 / max(1, len(ordered))
    center = (len(ordered) - 1) / 2.0
    raw: dict[str, float] = {}
    for idx, sub in enumerate(ordered):
        base_share = 0.65 * float(shares.get(sub, eq_share) or eq_share) + 0.35 * eq_share
        offset = 0.0 if center <= 0 else (float(idx) - center) / center
        mult = 1.0 + 0.12 * offset
        raw[sub] = max(1e-6, base_share * mult)
    scale = float(total_cows) / max(1e-6, sum(raw.values()))
    return {sub: float(raw[sub] * scale) for sub in ordered}

def _build_transfer_recommendations(
    farm_name: str,
    target_month_end: date,
    params: dict,
    params_for_subdivision: Optional[Callable[[str, dict[str, pd.DataFrame]], dict]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly_base = _subdivision_monthly_cows(
        farm_name,
        target_month_end,
        params,
        params_for_subdivision=params_for_subdivision,
    )
    flows = _extract_move_flows_from_dry(farm_name)
    if monthly_base.empty:
        return pd.DataFrame(), flows, pd.DataFrame(), pd.DataFrame(), {"reason": "no_subdivisions"}

    shares = _historical_subdivision_shares(farm_name, target_month_end)
    if not shares:
        return pd.DataFrame(), flows, pd.DataFrame(), pd.DataFrame(), {"reason": "no_subdivisions"}

    subs = sorted(set(monthly_base["Подразделение"].astype(str).tolist()))
    months = sorted(set(monthly_base["Месяц"].astype(str).tolist()))
    if not subs or not months:
        return pd.DataFrame(), flows, pd.DataFrame(), pd.DataFrame(), {"reason": "no_subdivisions"}

    base_pivot = monthly_base.pivot_table(
        index="Месяц",
        columns="Подразделение",
        values="Коровы всего",
        aggfunc="sum",
        fill_value=0.0,
    )
                                                                                                    
    for sub in subs:
        if sub not in base_pivot.columns:
            base_pivot[sub] = 0.0
    base_pivot = base_pivot.reindex(columns=subs, fill_value=0.0).sort_index()

    do_pivot = monthly_base.pivot_table(
        index="Месяц",
        columns="Подразделение",
        values="Дойные коровы",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=base_pivot.index, columns=subs, fill_value=0.0)
    dry_pivot = monthly_base.pivot_table(
        index="Месяц",
        columns="Подразделение",
        values="Сухостойные коровы",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=base_pivot.index, columns=subs, fill_value=0.0)

    flow_map: dict[tuple[str, str], int] = {}
    if not flows.empty:
        for rr in flows.itertuples(index=False):
            flow_map[(str(rr.Источник), str(rr.Приёмник))] = int(rr.Переездов)

    rec_rows: list[dict[str, Any]] = []
    snap_rows: list[dict[str, Any]] = []

    offset_by_sub = {sub: 0.0 for sub in subs}
    for month in base_pivot.index.astype(str).tolist():
        base_by_sub = {sub: float(base_pivot.at[month, sub]) for sub in subs}
        cows_before = {sub: max(0.0, base_by_sub[sub] + float(offset_by_sub.get(sub, 0.0))) for sub in subs}
        cow_groups_before: dict[str, dict[str, float]] = {}
        for sub in subs:
            do_now = float(do_pivot.at[month, sub]) if sub in do_pivot.columns else 0.0
            dry_now = float(dry_pivot.at[month, sub]) if sub in dry_pivot.columns else 0.0
            base_total = max(0.0, do_now + dry_now)
            target_total = max(0.0, float(cows_before.get(sub, 0.0)))
            if base_total > 1e-9:
                k = target_total / base_total
                do_now *= k
                dry_now *= k
            else:
                do_now = target_total
                dry_now = 0.0
            cow_groups_before[sub] = {
                "Дойные коровы": max(0.0, do_now),
                "Сухостойные коровы": max(0.0, dry_now),
            }

        total_cows = float(sum(cows_before.values()))
        cap_est_by_sub = {sub: float(shares.get(sub, 0.0)) * total_cows for sub in subs}
        cap_by_sub = _demo_capacity_by_subdivision(subs, total_cows, shares)
        overflow_before = {sub: max(0.0, cows_before[sub] - cap_by_sub[sub]) for sub in subs}
        free_before = {sub: max(0.0, cap_by_sub[sub] - cows_before[sub]) for sub in subs}
        free_before_est = dict(free_before)

        free_left = dict(free_before)
        cows_after = dict(cows_before)
        moved_in = {sub: 0.0 for sub in subs}
        moved_out = {sub: 0.0 for sub in subs}

        sources = [sub for sub, ov in overflow_before.items() if ov > 1e-6]
        sources.sort(key=lambda x: overflow_before[x], reverse=True)
        for src in sources:
            need = float(overflow_before[src])
            if need <= 1e-6:
                continue

            candidates: list[tuple[str, float, int]] = []
            for dst in subs:
                if dst == src:
                    continue
                free = float(free_left.get(dst, 0.0))
                if free <= 1e-6:
                    continue
                hist = int(flow_map.get((src, dst), 0))
                candidates.append((dst, free, hist))

            candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
            for dst, _, _hist in candidates:
                if need <= 1e-6:
                    break
                can_take = float(free_left.get(dst, 0.0))
                if can_take <= 1e-6:
                    continue
                move_n = min(need, can_take)
                if move_n <= 1e-6:
                    continue
                free_left[dst] = max(0.0, can_take - move_n)
                need = max(0.0, need - move_n)
                cows_after[src] = max(0.0, float(cows_after.get(src, 0.0)) - move_n)
                cows_after[dst] = max(0.0, float(cows_after.get(dst, 0.0)) + move_n)
                moved_out[src] += move_n
                moved_in[dst] += move_n
                move_groups = _split_transfer_by_cow_groups(move_n, cow_groups_before.get(src, {}))
                if move_groups:
                    src_groups = cow_groups_before.setdefault(
                        src,
                        {"Дойные коровы": 0.0, "Сухостойные коровы": 0.0},
                    )
                    dst_groups = cow_groups_before.setdefault(
                        dst,
                        {"Дойные коровы": 0.0, "Сухостойные коровы": 0.0},
                    )
                    for g in move_groups:
                        g_name = str(g.get("Группа") or "")
                        g_n = int(pd.to_numeric(g.get("Рекомендовано перевести, голов"), errors="coerce") or 0)
                        if not g_name or g_n <= 0:
                            continue
                        src_groups[g_name] = max(0.0, float(src_groups.get(g_name, 0.0)) - g_n)
                        dst_groups[g_name] = max(0.0, float(dst_groups.get(g_name, 0.0)) + g_n)
                groups_text = "; ".join(
                    f"{str(g.get('Группа') or '')}: {int(pd.to_numeric(g.get('Рекомендовано перевести, голов'), errors='coerce') or 0)}"
                    for g in move_groups
                )

                rec_rows.append(
                    {
                        "Месяц": str(month),
                        "Источник (переполнен)": src,
                        "Куда перевести": dst,
                        "Рекомендовано перевести, голов": float(move_n),
                        "Свободно в приёмнике, мест (оценка)": float(can_take),
                        "Свободно в приёмнике, мест": float(can_take),
                        "По группам (гол.)": groups_text,
                        "Детализация по группам": move_groups,
                    }
                )
                rec_rows[-1]["Свободно в приёмнике, мест (оценка)"] = float(free_before_est.get(dst, 0.0))

        overflow_after = {sub: max(0.0, cows_after[sub] - cap_by_sub[sub]) for sub in subs}
        free_after = {sub: max(0.0, cap_by_sub[sub] - cows_after[sub]) for sub in subs}

        for sub in subs:
            snap_rows.append(
                {
                    "Месяц": str(month),
                    "Подразделение": str(sub),
                    "Историческая доля": float(shares.get(sub, 0.0)),
                    "Дойные коровы": float(do_pivot.at[month, sub]) if sub in do_pivot.columns else 0.0,
                    "Сухостойные коровы": float(dry_pivot.at[month, sub]) if sub in dry_pivot.columns else 0.0,
                    "Коровы всего (прогноз)": float(base_by_sub[sub]),
                    "Коровы до переводов": float(cows_before[sub]),
                    "Мест (коровы)": float(cap_by_sub[sub]),
                    "Оценка мест (коровы)": float(cap_est_by_sub[sub]),
                    "Переполнение до перевода": float(overflow_before[sub]),
                    "Свободно мест до перевода": float(free_before[sub]),
                    "Переведено из подразделения": float(moved_out[sub]),
                    "Переведено в подразделение": float(moved_in[sub]),
                    "Коровы после переводов": float(cows_after[sub]),
                    "Переполнение после перевода": float(overflow_after[sub]),
                    "Свободно мест после перевода": float(free_after[sub]),
                    "Корректировка переводами, накопленная": float(cows_after[sub] - base_by_sub[sub]),
                    "Источник мест": "расчёт",
                }
            )

                                                                                             
        for sub in subs:
            offset_by_sub[sub] = float(cows_after[sub] - base_by_sub[sub])

    rec_df = pd.DataFrame(rec_rows)
    snap_monthly_df = pd.DataFrame(snap_rows)

    target_label = _month_label(target_month_end)
    if not snap_monthly_df.empty and target_label in set(snap_monthly_df["Месяц"].astype(str).tolist()):
        snap_final = snap_monthly_df.loc[snap_monthly_df["Месяц"].astype(str) == target_label].copy()
    elif not snap_monthly_df.empty:
        last_m = sorted(snap_monthly_df["Месяц"].astype(str).unique().tolist())[-1]
        snap_final = snap_monthly_df.loc[snap_monthly_df["Месяц"].astype(str) == last_m].copy()
    else:
        snap_final = pd.DataFrame()

    if not snap_final.empty:
        snap_final = snap_final.rename(
            columns={
                "Коровы после переводов": "Коровы всего",
                "Переполнение после перевода": "Переполнение (оценка)",
                "Свободно мест после перевода": "Свободно мест (оценка)",
            }
        )

    total_moved = float(pd.to_numeric(rec_df.get("Рекомендовано перевести, голов"), errors="coerce").fillna(0.0).sum()) if not rec_df.empty else 0.0
    src_monthly_n = int(len(snap_monthly_df.loc[pd.to_numeric(snap_monthly_df.get("Переполнение до перевода"), errors="coerce").fillna(0.0) > 1e-6, ["Месяц", "Подразделение"]].drop_duplicates())) if not snap_monthly_df.empty else 0
    dst_monthly_n = int(len(snap_monthly_df.loc[pd.to_numeric(snap_monthly_df.get("Свободно мест до перевода"), errors="coerce").fillna(0.0) > 1e-6, ["Месяц", "Подразделение"]].drop_duplicates())) if not snap_monthly_df.empty else 0
    meta = {
        "method": "calculated_capacity_plus_carx_flows",
        "capacity_mode": "calculated_capacity",
        "month_from": str(months[0]) if months else None,
        "month_to": str(months[-1]) if months else None,
        "months_n": int(len(months)),
        "sources_n": src_monthly_n,
        "destinations_n": dst_monthly_n,
        "recommendations_n": int(len(rec_df)),
        "total_moved": total_moved,
    }
    return rec_df, flows, snap_final, snap_monthly_df, meta


__all__ = [name for name in globals() if not name.startswith("__")]
