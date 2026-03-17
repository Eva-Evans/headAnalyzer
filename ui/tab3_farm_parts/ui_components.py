from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from core.constants import INDICATORS, INDICATOR_TO_OVERFLOW, OVERFLOW_COLS, OVERFLOW_GROUP_COLS
from core.excel_export import make_excel_bytes_highlight_months_columns
from core.helpers import month_end, norm_label, vals_get
from core.params import apply_admin_overrides, get_or_compute_subdivision_params, get_param_source, inject_live_semen_params
from core.realization import build_early_realization_plan
from ui.styles import BAD, fmt_cell, style_positive_red

from .common import *
from .storage import *
from .compute import *


def _tab3_percent_metric_from_backtest_rows(bt_df: pd.DataFrame, *, is_pct: bool) -> float | None:
    if not isinstance(bt_df, pd.DataFrame) or bt_df.empty:
        return None
    if is_pct:
        series = pd.to_numeric(bt_df.get("APE, %"), errors="coerce").dropna()
        return float(series.mean()) if not series.empty else None

    tmp = bt_df.copy()
    tmp["Прогноз"] = pd.to_numeric(tmp.get("Прогноз"), errors="coerce").fillna(0.0)
    tmp["Факт"] = pd.to_numeric(tmp.get("Факт"), errors="coerce").fillna(0.0)
    tmp["Ошибка"] = pd.to_numeric(tmp.get("Ошибка"), errors="coerce").fillna(0.0)
    scale = tmp["Прогноз"].abs() + tmp["Факт"].abs()
    stable_mask = scale >= 20.0
    den = float(scale.loc[stable_mask].sum())
    num = float(tmp.loc[stable_mask, "Ошибка"].abs().sum())
    return (200.0 * num / den) if den > 1e-9 else None


def _get_nested_param(d: dict | None, path: list[Any], default: Any) -> Any:
    cur: Any = d if isinstance(d, dict) else {}
    for t in path:
        if not isinstance(cur, dict):
            return default
        if t in cur:
            cur = cur[t]
            continue
        if isinstance(t, int) and str(t) in cur:
            cur = cur[str(t)]
            continue
        if isinstance(t, str) and t.isdigit() and int(t) in cur:
            cur = cur[int(t)]
            continue
        return default
    return default if cur is None else cur


def _set_nested_param(d: dict, path: list[Any], value: Any) -> None:
    cur = d
    for t in path[:-1]:
        key = t
        if isinstance(cur, dict) and isinstance(t, int) and str(t) in cur and t not in cur:
            key = str(t)
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    leaf = path[-1]
    if isinstance(cur, dict) and isinstance(leaf, int) and str(leaf) in cur and leaf not in cur:
        leaf = str(leaf)
    cur[leaf] = value


def _render_param_inputs(key_prefix: str, source_params: dict, fallback_params: dict) -> dict[str, Any]:
    def _pick(path: list[Any], default: Any) -> Any:
        v = _get_nested_param(source_params, path, None)
        if v is not None:
            return v
        return _get_nested_param(fallback_params, path, default)

    def _pick_num(path: list[Any], default: float, lo: float | None = None, hi: float | None = None) -> float:
        raw = _pick(path, default)
        num = pd.to_numeric(raw, errors="coerce")
        val = float(default if pd.isna(num) else num)
        if lo is not None:
            val = max(float(lo), val)
        if hi is not None:
            val = min(float(hi), val)
        return float(val)

    def _pick_int(path: list[Any], default: int, lo: int | None = None, hi: int | None = None) -> int:
        val = int(round(_pick_num(path, float(default), None if lo is None else float(lo), None if hi is None else float(hi))))
        if lo is not None:
            val = max(int(lo), val)
        if hi is not None:
            val = min(int(hi), val)
        return int(val)

    def _num_input(
        label: str,
        *,
        min_value: float | int,
        max_value: float | int,
        value: float | int,
        step: float | int,
        key: str,
        **kwargs: Any,
    ) -> Any:
        is_int = isinstance(min_value, int) and isinstance(max_value, int) and isinstance(step, int)

        def _clip(v: Any) -> float:
            num = pd.to_numeric(v, errors="coerce")
            if pd.isna(num):
                num = value
            out = float(num)
            out = max(float(min_value), min(float(max_value), out))
            return out

        def _cast(v: float) -> float | int:
            return int(round(v)) if is_int else float(v)

        if key in st.session_state:
            st.session_state[key] = _cast(_clip(st.session_state.get(key)))

        safe_value = _cast(_clip(value))
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=safe_value,
            step=step,
            key=key,
            **kwargs,
        )

    st.markdown("**Сроки**")
    c1, c2 = st.columns(2)
    with c1:
        gest = _num_input(
            "Длительность стельности (дн.)",
            min_value=200,
            max_value=310,
            value=_pick_int(["GESTATION_DAYS"], 272, lo=200, hi=310),
            step=1,
            key=f"{key_prefix}_gest",
        )
    with c2:
        dry = _num_input(
            "Длительность сухостоя (дн.)",
            min_value=20,
            max_value=120,
            value=_pick_int(["DRY_DAYS"], 53, lo=20, hi=120),
            step=1,
            key=f"{key_prefix}_dry",
        )

    st.markdown("**Стельность**")
    avg_cow_dim_default = _pick_num(["CONCEPTION_PARAMS", "avg_cow_dim_global"], 104.0, lo=40.0, hi=250.0)
    c3 = st.columns(1)[0]
    with c3:
        avg_heifer_age_days = _num_input(
            "Тёлки: средний возраст наступления стельности (дн.)",
            min_value=250.0,
            max_value=700.0,
            value=_pick_num(["CONCEPTION_PARAMS", "avg_heifer_age_days"], 400.0, lo=250.0, hi=700.0),
            step=1.0,
            key=f"{key_prefix}_cp_heifer_age",
        )

    c5, c6 = st.columns(2)
    with c5:
        cp_l1 = _num_input(
            "Коровы: DIM наступления стельности — 1-я лактация",
            min_value=40.0,
            max_value=250.0,
            value=_pick_num(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 1], float(avg_cow_dim_default), lo=40.0, hi=250.0),
            step=1.0,
            key=f"{key_prefix}_cp_l1",
        )
        cp_l2 = _num_input(
            "Коровы: DIM наступления стельности — 2-я лактация",
            min_value=40.0,
            max_value=250.0,
            value=_pick_num(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 2], float(avg_cow_dim_default), lo=40.0, hi=250.0),
            step=1.0,
            key=f"{key_prefix}_cp_l2",
        )
    with c6:
        cp_l3 = _num_input(
            "Коровы: DIM наступления стельности — 3-я лактация",
            min_value=40.0,
            max_value=250.0,
            value=_pick_num(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 3], float(avg_cow_dim_default), lo=40.0, hi=250.0),
            step=1.0,
            key=f"{key_prefix}_cp_l3",
        )
        cp_l4 = _num_input(
            "Коровы: DIM наступления стельности — 4+ лактация",
            min_value=40.0,
            max_value=250.0,
            value=_pick_num(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 4], float(avg_cow_dim_default), lo=40.0, hi=250.0),
            step=1.0,
            key=f"{key_prefix}_cp_l4",
        )

    st.markdown("**Осеменения**")
    c7, c8 = st.columns(2)
    with c7:
        cow_spc = _num_input(
            "Коровы: осеменений до стельности (P), среднее",
            min_value=1.0,
            max_value=5.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_services_per_conception"], 2.0, lo=1.0, hi=5.0),
            step=0.01,
            key=f"{key_prefix}_ins_cow_spc",
        )
        cow_interval = _num_input(
            "Коровы: интервал между осеменениями (дн.)",
            min_value=14.0,
            max_value=90.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_ai_interval_days"], 45.0, lo=14.0, hi=90.0),
            step=0.5,
            key=f"{key_prefix}_ins_cow_interval",
        )
    with c8:
        heif_spc = _num_input(
            "Тёлки: осеменений до стельности (P), среднее",
            min_value=1.0,
            max_value=5.0,
            value=_pick_num(["INSEMINATION_PARAMS", "heifer_services_per_conception"], 2.0, lo=1.0, hi=5.0),
            step=0.01,
            key=f"{key_prefix}_ins_heif_spc",
        )
        heif_interval = _num_input(
            "Тёлки: интервал между осеменениями (дн.)",
            min_value=14.0,
            max_value=90.0,
            value=_pick_num(["INSEMINATION_PARAMS", "heifer_ai_interval_days"], 25.0, lo=14.0, hi=90.0),
            step=0.5,
            key=f"{key_prefix}_ins_heif_interval",
        )

    heif_first_ai = _num_input(
        "Тёлки: возраст первого осеменения (дн.)",
        min_value=250.0,
        max_value=700.0,
        value=_pick_num(["INSEMINATION_PARAMS", "heifer_first_ai_age_days"], 380.0, lo=250.0, hi=700.0),
        step=1.0,
        key=f"{key_prefix}_ins_heif_first_ai",
    )

    c9, c10 = st.columns(2)
    with c9:
        cow_first_ai_l1 = _num_input(
            "Коровы: DIM первого осеменения — 1-я лактация",
            min_value=30.0,
            max_value=220.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 1], 72.0, lo=30.0, hi=220.0),
            step=1.0,
            key=f"{key_prefix}_ins_first_l1",
        )
        cow_first_ai_l2 = _num_input(
            "Коровы: DIM первого осеменения — 2-я лактация",
            min_value=30.0,
            max_value=220.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 2], 72.0, lo=30.0, hi=220.0),
            step=1.0,
            key=f"{key_prefix}_ins_first_l2",
        )
    with c10:
        cow_first_ai_l3 = _num_input(
            "Коровы: DIM первого осеменения — 3-я лактация",
            min_value=30.0,
            max_value=220.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 3], 72.0, lo=30.0, hi=220.0),
            step=1.0,
            key=f"{key_prefix}_ins_first_l3",
        )
        cow_first_ai_l4 = _num_input(
            "Коровы: DIM первого осеменения — 4+ лактация",
            min_value=30.0,
            max_value=220.0,
            value=_pick_num(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 4], 72.0, lo=30.0, hi=220.0),
            step=1.0,
            key=f"{key_prefix}_ins_first_l4",
        )

    st.markdown("**Семя**")
    c_sem_1, c_sem_2 = st.columns(2)
    with c_sem_1:
        cow_sex_share = _num_input(
            "Коровы: доля сексированного семени",
            min_value=0.0,
            max_value=1.0,
            value=_pick_num(["SEMEN_USAGE_SHARES", "cow_sex"], 0.3, lo=0.0, hi=1.0),
            step=0.01,
            key=f"{key_prefix}_semen_cow_sex",
            format="%.4f",
        )
        trad_heifer_share = _num_input(
            "Обычное семя: доля тёлочек",
            min_value=0.0,
            max_value=1.0,
            value=_pick_num(["SEMEN_SEX_RATIOS", "trad", "heifer_share"], 0.7517, lo=0.0, hi=1.0),
            step=0.01,
            key=f"{key_prefix}_semen_trad_heifer",
            format="%.4f",
        )
    with c_sem_2:
        heifer_sex_share = _num_input(
            "Тёлки: доля сексированного семени",
            min_value=0.0,
            max_value=1.0,
            value=_pick_num(["SEMEN_USAGE_SHARES", "heifer_sex"], 0.7, lo=0.0, hi=1.0),
            step=0.01,
            key=f"{key_prefix}_semen_heifer_sex",
            format="%.4f",
        )
        sex_heifer_share = _num_input(
            "Сексированное семя: доля тёлочек",
            min_value=0.0,
            max_value=1.0,
            value=_pick_num(["SEMEN_SEX_RATIOS", "sex", "heifer_share"], 0.9417, lo=0.0, hi=1.0),
            step=0.01,
            key=f"{key_prefix}_semen_sex_heifer",
            format="%.4f",
        )
    st.caption("Доля обычного семени считается как 1 - доля сексированного. Доля бычков считается как 1 - доля тёлочек.")

    st.markdown("**Выбытие**")
    annual_disposal = _num_input(
        "Среднегодовой процент выбытия коров (доля)",
        min_value=0.0,
        max_value=0.5,
        value=_pick_num(["ANNUAL_DISPOSAL_RATE"], 0.0957, lo=0.0, hi=0.5),
        step=0.001,
        format="%.4f",
        key=f"{key_prefix}_annual_disp",
    )

    c11, c12 = st.columns(2)
    with c11:
        disp_median_l1 = _num_input(
            "Выбытие: DIM медиана — 1-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 1, "median_dim"], 111.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_median_l1",
        )
        disp_median_l2 = _num_input(
            "Выбытие: DIM медиана — 2-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 2, "median_dim"], 226.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_median_l2",
        )
        disp_median_l3 = _num_input(
            "Выбытие: DIM медиана — 3-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 3, "median_dim"], 194.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_median_l3",
        )
        disp_median_l4 = _num_input(
            "Выбытие: DIM медиана — 4+ лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 4, "median_dim"], 73.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_median_l4",
        )
    with c12:
        disp_mean_l1 = _num_input(
            "Выбытие: DIM среднее — 1-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 1, "mean_dim"], 160.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_mean_l1",
        )
        disp_mean_l2 = _num_input(
            "Выбытие: DIM среднее — 2-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 2, "mean_dim"], 235.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_mean_l2",
        )
        disp_mean_l3 = _num_input(
            "Выбытие: DIM среднее — 3-я лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 3, "mean_dim"], 192.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_mean_l3",
        )
        disp_mean_l4 = _num_input(
            "Выбытие: DIM среднее — 4+ лактация",
            min_value=10.0,
            max_value=500.0,
            value=_pick_num(["DISPOSAL_PARAMS", "by_lact", 4, "mean_dim"], 127.0, lo=10.0, hi=500.0),
            step=1.0,
            key=f"{key_prefix}_disp_mean_l4",
        )

    new_override: dict[str, Any] = {}
    _set_nested_param(new_override, ["GESTATION_DAYS"], int(gest))
    _set_nested_param(new_override, ["DRY_DAYS"], int(dry))

    _set_nested_param(new_override, ["CONCEPTION_PARAMS", "avg_heifer_age_days"], float(avg_heifer_age_days))
    _set_nested_param(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 1], float(cp_l1))
    _set_nested_param(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 2], float(cp_l2))
    _set_nested_param(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 3], float(cp_l3))
    _set_nested_param(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 4], float(cp_l4))

    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_services_per_conception"], float(cow_spc))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_ai_interval_days"], float(cow_interval))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "heifer_services_per_conception"], float(heif_spc))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "heifer_ai_interval_days"], float(heif_interval))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "heifer_first_ai_age_days"], float(heif_first_ai))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 1], float(cow_first_ai_l1))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 2], float(cow_first_ai_l2))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 3], float(cow_first_ai_l3))
    _set_nested_param(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 4], float(cow_first_ai_l4))
    _set_nested_param(new_override, ["SEMEN_USAGE_SHARES", "cow_sex"], float(cow_sex_share))
    _set_nested_param(new_override, ["SEMEN_USAGE_SHARES", "cow_trad"], float(max(0.0, 1.0 - float(cow_sex_share))))
    _set_nested_param(new_override, ["SEMEN_USAGE_SHARES", "heifer_sex"], float(heifer_sex_share))
    _set_nested_param(new_override, ["SEMEN_USAGE_SHARES", "heifer_trad"], float(max(0.0, 1.0 - float(heifer_sex_share))))
    _set_nested_param(new_override, ["SEMEN_SEX_RATIOS", "trad", "heifer_share"], float(trad_heifer_share))
    _set_nested_param(new_override, ["SEMEN_SEX_RATIOS", "trad", "bull_share"], float(max(0.0, 1.0 - float(trad_heifer_share))))
    _set_nested_param(new_override, ["SEMEN_SEX_RATIOS", "sex", "heifer_share"], float(sex_heifer_share))
    _set_nested_param(new_override, ["SEMEN_SEX_RATIOS", "sex", "bull_share"], float(max(0.0, 1.0 - float(sex_heifer_share))))

    _set_nested_param(new_override, ["ANNUAL_DISPOSAL_RATE"], float(annual_disposal))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 1, "median_dim"], float(disp_median_l1))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 2, "median_dim"], float(disp_median_l2))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 3, "median_dim"], float(disp_median_l3))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 4, "median_dim"], float(disp_median_l4))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 1, "mean_dim"], float(disp_mean_l1))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 2, "mean_dim"], float(disp_mean_l2))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 3, "mean_dim"], float(disp_mean_l3))
    _set_nested_param(new_override, ["DISPOSAL_PARAMS", "by_lact", 4, "mean_dim"], float(disp_mean_l4))
    return new_override


def _reset_tab3_farm_result_state() -> None:
    st.session_state.pop("tab3_monthly_all", None)
    st.session_state.pop("tab3_farm_infos", None)
    st.session_state.pop("tab3_target_month_end", None)
    st.session_state.pop("tab3_backtest_df", None)
    st.session_state.pop("tab3_backtest_sub_df", None)
    st.session_state.pop("tab3_backtest_cfg", None)


def _resolve_subdivision_params_runtime(
    subdivision: str,
    tables: dict[str, pd.DataFrame],
    farm_override: dict | None,
    subdivision_override: dict | None,
    capacity_override: dict | None = None,
    log_fn: Any | None = None,
) -> dict:
    try:
        sub_base = get_or_compute_subdivision_params(subdivision, tables)
    except Exception as e:
        if callable(log_fn):
            log_fn(f"{subdivision}: ошибка загрузки параметров подразделения ({e})")
        raise

    sub_base = apply_admin_overrides(sub_base)
    out = _build_subdivision_params(
        sub_base,
        farm_override=farm_override,
        subdivision_override=subdivision_override,
    )
    cap_ov = capacity_override if isinstance(capacity_override, dict) else {}
    if cap_ov:
        _deep_merge(out, cap_ov)
        disable_capacity = bool(cap_ov.get("DISABLE_CAPACITY", False)) or (cap_ov.get("APPLY_CAPACITY") is False)
        out["DISABLE_CAPACITY"] = bool(disable_capacity)
        out["APPLY_CAPACITY"] = False if disable_capacity else True
        if callable(log_fn) and not disable_capacity:
            cap_keys = sorted(str(k) for k in (cap_ov.get("HERD_CAPACITY", {}) or {}).keys())
            if cap_keys:
                log_fn(f"{subdivision}: применены скотоместа ({', '.join(cap_keys)})")
    return out


def _capacity_default_groups() -> list[str]:
    model_groups = [
        "Дойные коровы",
        "Сухостойные коровы",
        "Нетели",
        "Тёлки 0–3 мес",
        "Тёлки 3–8 мес",
        "Тёлки ≥9 мес",
    ]
    source_alias_groups = [
        "коровы дойные",
        "коровы сухостой",
        "сухостой",
        "молодняк 9-24",
        "молодняк 3-8",
        "молодняк 0-3",
        "телята 0-3",
        "телята 3-5",
        "телята",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for g in [*model_groups, *source_alias_groups]:
        k = str(g).strip().upper().replace("Ё", "Е")
        if k in seen:
            continue
        seen.add(k)
        out.append(g)
    return out


def _norm_capacity_header(x: Any) -> str:
    s = str(x or "").upper().replace("Ё", "Е")
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def _find_capacity_col(df: pd.DataFrame, variants: list[str]) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    norm_to_real = {_norm_capacity_header(c): c for c in df.columns}
    for v in variants:
        k = _norm_capacity_header(v)
        if k in norm_to_real:
            return str(norm_to_real[k])
    return None


def _read_capacity_upload(file_obj: Any) -> pd.DataFrame:
    if file_obj is None:
        return pd.DataFrame()
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass
    name = str(getattr(file_obj, "name", "")).lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        for sep in (";", ",", "\t"):
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=sep)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                continue
        return pd.DataFrame()
    try:
        return pd.read_excel(file_obj)
    except Exception:
        return pd.DataFrame()


def _norm_capacity_group_text(x: Any) -> str:
    s = str(x or "").upper().replace("Ё", "Е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _map_capacity_group_to_model_key(group_name: Any) -> str | None:
    s = _norm_capacity_group_text(group_name)
    s_compact = s.replace(" ", "")
    if "СУХОСТ" in s:
        return "Сухостойные коровы"
    if ("КОРОВ" in s and ("ДОЙ" in s or "СУХОСТ" not in s)) or ("КОРОВЫДОЙНЫЕ" in s_compact):
        return "Дойные коровы"
    if "НЕТЕЛ" in s:
        return "Нетели"
    if ("ТЕЛ" in s or "МОЛОДН" in s) and ("0-3" in s or "0 3" in s or "03" in s_compact):
        return "Тёлки 0–3 мес"
    if ("ТЕЛ" in s or "МОЛОДН" in s) and ("3-8" in s or "3 8" in s or "38" in s_compact):
        return "Тёлки 3–8 мес"
    if ("ТЕЛ" in s or "МОЛОДН" in s) and (
        ("9-24" in s)
        or ("9 24" in s)
        or ("924" in s_compact)
        or ("≥9" in s)
        or (">=9" in s_compact)
        or ("9МЕС" in s_compact)
        or ("СТАРШЕ 6" in s)
        or ("6+" in s_compact)
        or ("6МЕС" in s_compact)
    ):
        return "Тёлки 9–24 мес"
    return None


def _capacity_overrides_for_farm(farm_name: str) -> dict[str, dict]:
    cap_df = _load_capacity_rows_for_farm(farm_name)
    if not isinstance(cap_df, pd.DataFrame) or cap_df.empty:
        return {}

    model_groups = (
        "Дойные коровы",
        "Сухостойные коровы",
        "Нетели",
        "Тёлки 0–3 мес",
        "Тёлки 3–8 мес",
        "Тёлки 9–24 мес",
    )
    by_sub: dict[str, dict[str, float]] = {}
    for row in cap_df.to_dict(orient="records"):
        sub = _canon_subdivision_name(row.get("subdivision_name"))
        if not sub:
            continue
        key = _map_capacity_group_to_model_key(row.get("group_name"))
        if not key:
            continue
        by_sub.setdefault(sub, {g: 0.0 for g in model_groups})
        val = float(pd.to_numeric(row.get("places"), errors="coerce") or 0.0)
        val = max(0.0, val)
        by_sub[sub][key] = float(by_sub[sub].get(key, 0.0) + val)

    out: dict[str, dict] = {}
    for sub, caps in by_sub.items():
        if not caps:
            continue
        out[sub] = {
            "HERD_CAPACITY": caps,
            "APPLY_CAPACITY": True,
            "DISABLE_CAPACITY": False,
        }
    return out


def _tab3_demo_capacity_overrides_for_farm(farm_name: str) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    for sub in _subdivisions_for_farm(farm_name, ready_only=False):
        overrides[str(sub)] = {
            "DISABLE_CAPACITY": True,
            "APPLY_CAPACITY": False,
        }
    return overrides


def _parse_capacity_upload_df(df_raw: pd.DataFrame, default_farm: str | None = None) -> pd.DataFrame:
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame(columns=["farm_name", "subdivision_name", "group_name", "places"])

    farm_col = _find_capacity_col(df_raw, ["Хозяйство", "farm", "farm_name"])
    sub_col = _find_capacity_col(df_raw, ["Подразделение", "subdivision", "subdivision_name"])
    group_col = _find_capacity_col(df_raw, ["Наименование групп", "Группа", "group", "group_name"])
    places_col = _find_capacity_col(df_raw, ["Скотоместа", "мест", "places", "capacity"])

    if sub_col is None or group_col is None or places_col is None:
        raise ValueError(
            "В файле скотомест не найдены обязательные колонки: Подразделение, Наименование групп, Скотоместа."
        )

    out = pd.DataFrame()
    out["farm_name"] = (
        df_raw[farm_col].astype(str).map(_canon_farm_name)
        if farm_col is not None
        else _canon_farm_name(default_farm or "")
    )
    out["subdivision_name"] = df_raw[sub_col].astype(str).map(_canon_subdivision_name)
    out["group_name"] = df_raw[group_col].astype(str).str.strip()
    places_raw = df_raw[places_col].astype(str).str.replace(",", ".", regex=False)
    out["places"] = pd.to_numeric(places_raw, errors="coerce").fillna(0.0).clip(lower=0.0)
    out["farm_name"] = out["farm_name"].map(_canon_farm_name)
    out = out.loc[
        (out["farm_name"].astype(str) != "")
        & (out["subdivision_name"].astype(str) != "")
        & (out["group_name"].astype(str) != "")
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=["farm_name", "subdivision_name", "group_name", "places"])
    out = (
        out.groupby(["farm_name", "subdivision_name", "group_name"], as_index=False)["places"]
        .sum()
        .sort_values(["farm_name", "subdivision_name", "group_name"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def _prepare_capacity_editor_df_for_subdivision(farm_name: str, subdivision: str) -> pd.DataFrame:
    cap_df = _load_capacity_rows_for_farm(farm_name)
    sub_norm = _canon_subdivision_name(subdivision)
    if cap_df.empty:
        sub_df = pd.DataFrame(columns=["group_name", "places"])
    else:
        mask = cap_df["subdivision_name"].astype(str).map(_canon_subdivision_name) == sub_norm
        sub_df = cap_df.loc[mask, ["group_name", "places"]].copy()

    if sub_df.empty:
        rows = [{"Группа": g, "Скотоместа": 0.0} for g in _capacity_default_groups()]
        out = pd.DataFrame(rows, columns=["Группа", "Скотоместа"])
    else:
        out = sub_df.rename(columns={"group_name": "Группа", "places": "Скотоместа"}).copy()
        existing = {str(g).strip().upper().replace("Ё", "Е") for g in out["Группа"].astype(str).tolist()}
        add_rows = []
        for g in _capacity_default_groups():
            k = str(g).strip().upper().replace("Ё", "Е")
            if k not in existing:
                add_rows.append({"Группа": g, "Скотоместа": 0.0})
        if add_rows:
            out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True)

    out["Группа"] = out["Группа"].astype(str)
    out["Скотоместа"] = pd.to_numeric(out["Скотоместа"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out.sort_values(["Группа"], kind="mergesort").reset_index(drop=True)


def _render_subdivision_capacity_editor_block(
    farm_name: str,
    subdivision: str,
    *,
    key_scope: str = "default",
) -> None:
    st.markdown("**Скотоместа подразделения (по группам)**")
    sub_view = _prepare_capacity_editor_df_for_subdivision(farm_name, subdivision)
    key_base = f"tab3_sub_capacity_{_json_hash(f'{key_scope}|{farm_name}|{subdivision}')[:10]}"
    edited = st.data_editor(
        sub_view,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key=f"{key_base}_editor",
        column_config={
            "Группа": st.column_config.TextColumn("Наименование группы"),
            "Скотоместа": st.column_config.NumberColumn("Скотоместа", min_value=0.0, step=1.0),
        },
    )

    c1, c2 = st.columns(2)
    if c1.button("Сохранить скотоместа подразделения", use_container_width=True, key=f"{key_base}_save"):
        if not isinstance(edited, pd.DataFrame):
            edited = pd.DataFrame(columns=["Группа", "Скотоместа"])
        new_sub = edited.rename(columns={"Группа": "group_name", "Скотоместа": "places"})
        new_sub["subdivision_name"] = subdivision
        new_sub = new_sub[["subdivision_name", "group_name", "places"]].copy()

        all_cap = _load_capacity_rows_for_farm(farm_name)
        if not all_cap.empty:
            mask_other = all_cap["subdivision_name"].astype(str).map(_canon_subdivision_name) != _canon_subdivision_name(subdivision)
            all_cap = all_cap.loc[mask_other, ["subdivision_name", "group_name", "places"]].copy()
            merged = pd.concat([all_cap, new_sub], ignore_index=True)
        else:
            merged = new_sub
        _save_capacity_rows_for_farm(farm_name, merged)
        _reset_tab3_farm_result_state()
        st.success(f"Скотоместа подразделения «{subdivision}» сохранены.")
        st.rerun()

    if c2.button("Очистить скотоместа подразделения", use_container_width=True, key=f"{key_base}_clear"):
        all_cap = _load_capacity_rows_for_farm(farm_name)
        if not all_cap.empty:
            mask_other = all_cap["subdivision_name"].astype(str).map(_canon_subdivision_name) != _canon_subdivision_name(subdivision)
            rest = all_cap.loc[mask_other, ["subdivision_name", "group_name", "places"]].copy()
        else:
            rest = pd.DataFrame(columns=["subdivision_name", "group_name", "places"])
        _save_capacity_rows_for_farm(farm_name, rest)
        _reset_tab3_farm_result_state()
        st.success(f"Скотоместа подразделения «{subdivision}» очищены.")
        st.rerun()


def _render_capacity_editor_block(farm_name: str, subdivisions: list[str]) -> None:
    st.markdown("**Скотоместа (по группам) — хранятся в БД**")
    cap_df = _load_capacity_rows_for_farm(farm_name)

    default_groups = _capacity_default_groups()

    if cap_df.empty:
        seed_rows: list[dict[str, Any]] = []
        for sub in subdivisions:
            for grp in default_groups:
                seed_rows.append({"Подразделение": sub, "Группа": grp, "Скотоместа": 0.0})
        cap_view = pd.DataFrame(seed_rows, columns=["Подразделение", "Группа", "Скотоместа"])
    else:
        cap_view = cap_df.rename(
            columns={
                "subdivision_name": "Подразделение",
                "group_name": "Группа",
                "places": "Скотоместа",
            }
        )[["Подразделение", "Группа", "Скотоместа"]].copy()
        def _norm_key(x: Any) -> str:
            return str(x or "").strip().upper().replace("Ё", "Е")
        existing = {
            (_norm_key(r["Подразделение"]), _norm_key(r["Группа"]))
            for r in cap_view.to_dict(orient="records")
        }
        add_rows: list[dict[str, Any]] = []
        for sub in subdivisions:
            for grp in default_groups:
                k = (_norm_key(sub), _norm_key(grp))
                if k not in existing:
                    add_rows.append({"Подразделение": sub, "Группа": grp, "Скотоместа": 0.0})
        if add_rows:
            cap_view = pd.concat([cap_view, pd.DataFrame(add_rows)], ignore_index=True)
    if not cap_view.empty:
        cap_view["Подразделение"] = cap_view["Подразделение"].astype(str)
        cap_view["Группа"] = cap_view["Группа"].astype(str)
        cap_view["Скотоместа"] = pd.to_numeric(cap_view["Скотоместа"], errors="coerce").fillna(0.0)
        cap_view = cap_view.sort_values(["Подразделение", "Группа"], kind="mergesort").reset_index(drop=True)

    cap_key = f"tab3_capacity_editor_{_json_hash(farm_name)[:10]}"
    edited = st.data_editor(
        cap_view,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key=cap_key,
        column_config={
            "Подразделение": st.column_config.TextColumn("Подразделение"),
            "Группа": st.column_config.TextColumn("Наименование группы"),
            "Скотоместа": st.column_config.NumberColumn("Скотоместа", min_value=0.0, step=1.0),
        },
    )

    c1, c2 = st.columns(2)
    if c1.button("Сохранить скотоместа в БД", use_container_width=True, key=f"{cap_key}_save"):
        if not isinstance(edited, pd.DataFrame):
            edited = pd.DataFrame(columns=["Подразделение", "Группа", "Скотоместа"])
        to_save = edited.rename(
            columns={
                "Подразделение": "subdivision_name",
                "Группа": "group_name",
                "Скотоместа": "places",
            }
        )
        n_rows = _save_capacity_rows_for_farm(farm_name, to_save)
        _reset_tab3_farm_result_state()
        st.success(f"Сохранено строк скотомест: {n_rows}.")
        st.rerun()

    if c2.button("Очистить скотоместа хозяйства", use_container_width=True, key=f"{cap_key}_clear"):
        empty = pd.DataFrame(columns=["subdivision_name", "group_name", "places"])
        _save_capacity_rows_for_farm(farm_name, empty)
        _reset_tab3_farm_result_state()
        st.success(f"Скотоместа для «{farm_name}» очищены.")
        st.rerun()


def _farm_param_editor_block(farms: list[str], base_params: dict) -> None:
    if not _is_admin_mode():
        return
    with st.expander("Параметры прогноза по хозяйству", expanded=False):
        if not farms:
            return

        all_farm_overrides = _farm_param_overrides_state()
        all_sub_overrides = _subdivision_param_overrides_state()
        farm_name = st.selectbox("Хозяйство для настройки параметров", farms, index=0, key="tab3_param_farm_select")
        farm_override = deepcopy(all_farm_overrides.get(farm_name, {}))
        subdivisions = _subdivisions_for_farm(farm_name, ready_only=False)
        try:
            farm_base_params = inject_live_semen_params(base_params, tables=_load_farm_merged_tables_from_db(farm_name))
        except Exception:
            farm_base_params = base_params
        tab_labels = ["Хозяйство (общие)"] + [f"Подразделение: {sub}" for sub in subdivisions]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.caption(
                "Порядок применения параметров: базовые параметры подразделения -> параметры хозяйства -> "
                "параметры подразделения."
            )
            farm_key = f"tab3_param_farm_{_json_hash(farm_name)[:10]}"
            new_farm_override = _render_param_inputs(farm_key, farm_override, farm_base_params)
            c_save, c_reset, c_reset_subs = st.columns(3)
            if c_save.button("Сохранить параметры хозяйства", use_container_width=True, key=f"{farm_key}_save"):
                all_farm_overrides[farm_name] = new_farm_override
                st.session_state["tab3_farm_param_overrides"] = all_farm_overrides
                _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                _reset_tab3_farm_result_state()
                st.success(f"Параметры для «{farm_name}» сохранены.")
                st.rerun()
            if c_reset.button("Сбросить параметры хозяйства", use_container_width=True, key=f"{farm_key}_reset"):
                if farm_name in all_farm_overrides:
                    all_farm_overrides.pop(farm_name, None)
                    st.session_state["tab3_farm_param_overrides"] = all_farm_overrides
                    _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                    _reset_tab3_farm_result_state()
                    st.success(f"Параметры для «{farm_name}» сброшены.")
                    st.rerun()
            if c_reset_subs.button(
                "Сбросить параметры подразделений",
                use_container_width=True,
                key=f"{farm_key}_reset_subs",
            ):
                removed = 0
                for sub in subdivisions:
                    if sub in all_sub_overrides:
                        all_sub_overrides.pop(sub, None)
                        removed += 1
                st.session_state["tab3_subdivision_param_overrides"] = all_sub_overrides
                if removed > 0:
                    _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                    _reset_tab3_farm_result_state()
                    st.success(f"Сброшены параметры для {removed} подразделений хозяйства «{farm_name}».")
                    st.rerun()

        for idx, subdivision in enumerate(subdivisions, start=1):
            with tabs[idx]:
                sub_override = deepcopy(all_sub_overrides.get(subdivision, {}))
                try:
                    tables = _load_farm_tables_from_db(subdivision)
                    sub_base = inject_live_semen_params(get_or_compute_subdivision_params(subdivision, tables), tables=tables)
                    sub_base = apply_admin_overrides(sub_base)
                except Exception as e:
                    st.error(f"Не удалось загрузить параметры подразделения «{subdivision}»: {e}")
                    continue

                source_params = _build_subdivision_params(
                    sub_base,
                    farm_override=farm_override,
                    subdivision_override=sub_override,
                )
                fallback_params = _build_subdivision_params(
                    sub_base,
                    farm_override=farm_override,
                    subdivision_override=None,
                )
                st.caption(
                    "Порядок применения параметров: базовые параметры подразделения -> параметры хозяйства -> "
                    "параметры текущего подразделения."
                )
                sub_key = f"tab3_param_sub_{_json_hash(f'{farm_name}|{subdivision}')[:10]}"
                new_sub_override = _render_param_inputs(sub_key, source_params, fallback_params)
                s_save, s_reset = st.columns(2)
                if s_save.button(
                    f"Сохранить параметры подразделения «{subdivision}»",
                    use_container_width=True,
                    key=f"{sub_key}_save",
                ):
                    all_sub_overrides[subdivision] = new_sub_override
                    st.session_state["tab3_subdivision_param_overrides"] = all_sub_overrides
                    _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                    _reset_tab3_farm_result_state()
                    st.success(f"Параметры подразделения «{subdivision}» сохранены.")
                    st.rerun()
                if s_reset.button(
                    f"Сбросить параметры подразделения «{subdivision}»",
                    use_container_width=True,
                    key=f"{sub_key}_reset",
                ):
                    if subdivision in all_sub_overrides:
                        all_sub_overrides.pop(subdivision, None)
                        st.session_state["tab3_subdivision_param_overrides"] = all_sub_overrides
                        _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                        _reset_tab3_farm_result_state()
                        st.success(f"Параметры подразделения «{subdivision}» сброшены.")
                        st.rerun()

                st.markdown("---")
                st.caption("Скотоместа этого подразделения (по группам)")
                _render_subdivision_capacity_editor_block(
                    farm_name=farm_name,
                    subdivision=subdivision,
                    key_scope="param_tab",
                )



def _render_manual_capacity_panel(farms: list[str]) -> None:
    if not _is_admin_mode():
        return
    with st.expander("Скотоместа (ручная настройка)", expanded=False):
        if not farms:
            st.caption("Нет хозяйств для редактирования.")
            return
        farm_name = st.selectbox(
            "Хозяйство для редактирования скотомест",
            farms,
            index=0,
            key="tab3_manual_capacity_farm_select",
        )
        subdivisions = _subdivisions_for_farm(farm_name, ready_only=False)
        if not subdivisions:
            st.caption("У выбранного хозяйства пока нет подразделений.")
            return

        st.caption("Можно менять скотоместа сразу по всем подразделениям и группам.")
        _render_capacity_editor_block(farm_name=farm_name, subdivisions=subdivisions)

        with st.expander("Точечная настройка одного подразделения", expanded=False):
            sub = st.selectbox(
                "Подразделение",
                subdivisions,
                index=0,
                key=f"tab3_manual_capacity_sub_select_{_json_hash(farm_name)[:10]}",
            )
            _render_subdivision_capacity_editor_block(
                farm_name=farm_name,
                subdivision=sub,
                key_scope="manual_panel",
            )


def _render_capacity_limits_panel(farms: list[str]) -> None:
    with st.expander("Ограничения по скотоместам подразделений", expanded=False):
        if not farms:
            st.caption("Нет хозяйств для просмотра скотомест.")
            return
        farm_name = st.selectbox(
            "Хозяйство",
            farms,
            index=0,
            key="tab3_capacity_limits_farm_select",
        )
        subdivisions = _subdivisions_for_farm(farm_name, ready_only=False)
        if not subdivisions:
            st.caption("У выбранного хозяйства пока нет подразделений.")
            return
        sub = st.selectbox(
            "Подразделение",
            subdivisions,
            index=0,
            key=f"tab3_capacity_limits_sub_select_{_json_hash(farm_name)[:10]}",
        )
        if _is_admin_mode():
            st.caption("Редактирование скотомест по группам для выбранного подразделения.")
            _render_subdivision_capacity_editor_block(
                farm_name=farm_name,
                subdivision=sub,
                key_scope="limits_panel",
            )
        else:
            st.caption("Только просмотр. Для изменения включите админ-режим.")
            view_df = _prepare_capacity_editor_df_for_subdivision(farm_name, sub)
            st.dataframe(view_df, use_container_width=True, hide_index=True)

def _render_farm_backtesting_panel(default_farm: str | None = None) -> None:
    farm_status_df = _farm_status_df_from_db()
    ready_farms = (
        farm_status_df.loc[farm_status_df["Статус"] == "готово", "Хозяйство"].astype(str).tolist()
        if not farm_status_df.empty
        else []
    )
    if not ready_farms:
        return

    default_name = str(default_farm or "").strip()
    default_index = ready_farms.index(default_name) if default_name in ready_farms else 0

    with st.expander("Backtesting по хозяйству", expanded=False):
        st.session_state.setdefault("tab3_backtest_df", None)
        st.session_state.setdefault("tab3_backtest_sub_df", None)
        st.session_state.setdefault("tab3_backtest_cfg", None)

        bt_farm = st.selectbox(
            "Хозяйство для backtesting",
            ready_farms,
            index=default_index,
            key="tab3_bt_farm_select",
        )
        bt_target = st.selectbox(
            "Показатель для backtesting",
            FARM_BACKTEST_TARGETS,
            index=0,
            key="tab3_bt_target",
        )

        c_bt1, c_bt2 = st.columns(2)
        with c_bt1:
            bt_months = st.slider(
                "Глубина истории (последние месяцы)",
                min_value=3,
                max_value=24,
                value=6,
                step=1,
                key="tab3_bt_months",
            )
        with c_bt2:
            bt_horizon = st.slider(
                "Горизонт as-of (месяцев назад)",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key="tab3_bt_horizon",
            )
        bt_complete_only = st.checkbox(
            "Учитывать только полные месяцы факта",
            value=True,
            key="tab3_bt_complete_only",
        )

        if st.button("Запустить backtesting по хозяйству", key="tab3_btn_backtest", use_container_width=True):
            base_params_bt = apply_admin_overrides(get_param_source())
            bt_override = _farm_param_overrides_state().get(bt_farm) if _is_admin_mode() else None
            all_sub_overrides = _subdivision_param_overrides_state() if _is_admin_mode() else {}
            cap_overrides_by_sub = _tab3_demo_capacity_overrides_for_farm(bt_farm)
            farm_params_bt = _build_farm_params(base_params_bt, bt_override)
            bt_progress = st.progress(0.0)
            bt_status = st.empty()

            def _params_for_subdivision(subdivision: str, tables: dict[str, pd.DataFrame]) -> dict:
                sub_override = all_sub_overrides.get(subdivision)
                return _resolve_subdivision_params_runtime(
                    subdivision=subdivision,
                    tables=tables,
                    farm_override=bt_override,
                    subdivision_override=sub_override,
                    capacity_override=cap_overrides_by_sub.get(subdivision),
                    log_fn=None,
                )

            def _bt_progress_cb(step_idx: int, steps_total: int, msg: str) -> None:
                bt_progress.progress(step_idx / max(1, steps_total))
                bt_status.caption(f"Backtesting: {msg}")

            with st.spinner("Считаю backtesting по подразделениям и агрегирую метрики..."):
                bt_df, bt_sub_df, bt_summary = _run_farm_backtesting(
                    farm_name=bt_farm,
                    metric_name=bt_target,
                    bt_months=int(bt_months),
                    bt_horizon=int(bt_horizon),
                    complete_only=bool(bt_complete_only),
                    params=farm_params_bt,
                    params_for_subdivision=_params_for_subdivision,
                    progress_cb=_bt_progress_cb,
                )

            bt_progress.empty()
            bt_status.empty()

            st.session_state["tab3_backtest_df"] = bt_df
            st.session_state["tab3_backtest_sub_df"] = bt_sub_df
            st.session_state["tab3_backtest_cfg"] = {
                "farm": bt_farm,
                "metric": bt_target,
                "months": int(bt_months),
                "horizon": int(bt_horizon),
                "complete_only": bool(bt_complete_only),
                **(bt_summary or {}),
            }

        bt_df = st.session_state.get("tab3_backtest_df")
        bt_sub_df = st.session_state.get("tab3_backtest_sub_df")
        bt_cfg = st.session_state.get("tab3_backtest_cfg") or {}
        cfg_changed = (
            str(bt_cfg.get("farm", "")) != str(bt_farm)
            or str(bt_cfg.get("metric", "")) != str(bt_target)
            or int(bt_cfg.get("months", 0) or 0) != int(bt_months)
            or int(bt_cfg.get("horizon", 0) or 0) != int(bt_horizon)
            or bool(bt_cfg.get("complete_only", True)) != bool(bt_complete_only)
        )
        if cfg_changed:
            st.info("Выбери хозяйство и нажми запуск, чтобы увидеть метрики для него.")
            return

        if not isinstance(bt_df, pd.DataFrame) or bt_df.empty:
            st.info("Нажми «Запустить backtesting по хозяйству», чтобы увидеть метрики.")
            return

        metric_for_view = str(bt_cfg.get("metric") or bt_target)
        is_pct = metric_for_view in FARM_PERCENT_TARGETS
        mae_label = "Средняя погрешность, п.п." if is_pct else "Средняя погрешность, гол."
        bias_label = "Смещение (средняя ошибка), п.п." if is_pct else "Смещение (средняя ошибка), гол."

        mae_val = float(pd.to_numeric(bt_df["Ошибка"], errors="coerce").abs().mean())
        mape_val = _tab3_percent_metric_from_backtest_rows(bt_df, is_pct=is_pct)
        bias_val = float(pd.to_numeric(bt_df["Ошибка"], errors="coerce").mean())

        m1, m2, m3 = st.columns(3)
        perc_label = "Процентная погрешность, %" if is_pct else "Симметричная процентная погрешность, %"
        m1.metric(mae_label, "—" if mae_val is None else f"{float(mae_val):.1f}")
        m2.metric(perc_label, "—" if mape_val is None else f"{float(mape_val):.1f}")
        m3.metric(bias_label, "—" if bias_val is None else f"{float(bias_val):.1f}")

        skipped_months = int(bt_cfg.get("skipped_months", 0) or 0)
        skipped_sub_months = int(bt_cfg.get("skipped_sub_months", 0) or 0)
        st.caption(
            f"Итоговые метрики рассчитаны по сводному ряду хозяйства. "
            f"Пропущено месяцев: {skipped_months}, пропущено суб-месяцев: {skipped_sub_months}."
        )

        st.dataframe(bt_df, use_container_width=True, hide_index=True)
        chart_df = bt_df.set_index("Месяц факта")[["Прогноз", "Факт"]]
        st.line_chart(chart_df)

        if isinstance(bt_sub_df, pd.DataFrame) and not bt_sub_df.empty:
            st.markdown("**Вклад подразделений в итоговые метрики**")
            st.dataframe(
                bt_sub_df.style.format(fmt_cell),
                use_container_width=True,
                hide_index=True,
            )

def _render_results(monthly_all: pd.DataFrame, farm_infos: list[dict[str, Any]], target_month_end: date) -> None:
    _ = target_month_end
    farms = sorted(monthly_all["Хозяйство"].dropna().astype(str).unique().tolist())
    if not farms:
        return

    farm_detail = st.selectbox("Хозяйство для прогноза", farms, index=0, key="tab3_detail_farm")
    detail_df = monthly_all.loc[monthly_all["Хозяйство"] == farm_detail].copy().sort_values("Месяц")
    if detail_df.empty:
        return

    result_df = detail_df.set_index("Месяц")[INDICATORS].copy()
    overflow_df = detail_df.set_index("Месяц")[[c for c in OVERFLOW_COLS if c in detail_df.columns]].copy()

    lead_months = st.slider(
        "За сколько месяцев заранее продавать нетелей, если прогноз показывает переполнение по коровам",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        key="tab3_realization_lead",
    )
    realization_df = build_early_realization_plan(overflow_df, lead_months=int(lead_months))
    realization_view = realization_df.T

    forecast_view = result_df.T
    overflow_groups_only = overflow_df.reindex(columns=[c for c in OVERFLOW_GROUP_COLS if c in overflow_df.columns])
    overflow_view = overflow_groups_only.T

    info_by_farm: dict[str, dict[str, Any]] = {}
    for x in farm_infos:
        if isinstance(x, dict):
            fname = str(x.get("farm", "") or "")
            if fname:
                info_by_farm[fname] = x
    farm_info = info_by_farm.get(farm_detail, {})

    transfer_snapshot = pd.DataFrame(farm_info.get("transfer_snapshot", []))
    transfer_snapshot_monthly = pd.DataFrame(farm_info.get("transfer_snapshot_monthly", []))
    transfer_recs = pd.DataFrame(
        farm_info.get("transfer_recommendations_monthly", farm_info.get("transfer_recommendations", []))
    )
    transfer_flows = pd.DataFrame(farm_info.get("transfer_move_flows", []))
    remaining_cow_overflow_by_month: dict[str, float] = {}
    if not transfer_snapshot_monthly.empty and "Месяц" in transfer_snapshot_monthly.columns:
        over_col = (
            "Переполнение после перевода"
            if "Переполнение после перевода" in transfer_snapshot_monthly.columns
            else "Переполнение (оценка)"
        )
        if over_col in transfer_snapshot_monthly.columns:
            snap_work = transfer_snapshot_monthly.copy()
            snap_work["_farm_over"] = pd.to_numeric(snap_work[over_col], errors="coerce").fillna(0.0)
            remaining_cow_overflow_by_month = {
                str(k): float(v)
                for k, v in snap_work.groupby("Месяц", as_index=False)["_farm_over"].sum().set_index("Месяц")["_farm_over"].to_dict().items()
            }

    def _style_forecast(df_view: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df_view.index, columns=df_view.columns)
        for ind in df_view.index:
            if str(ind) in {"Дойные коровы", "Сухостойные коровы"}:
                for m in df_view.columns:
                    if float(remaining_cow_overflow_by_month.get(str(m), 0.0) or 0.0) > 0.0:
                        styles.loc[ind, m] = BAD
                continue
            ov_name = INDICATOR_TO_OVERFLOW.get(str(ind))
            if not ov_name or ov_name not in overflow_df.columns:
                continue
        return styles

    st.subheader("Прогноз ")
    st.dataframe(
        forecast_view.style.format(fmt_cell).apply(_style_forecast, axis=None),
        use_container_width=True,
    )

    st.subheader("Переполнение по группам ")
    st.dataframe(
        overflow_view.style.format(fmt_cell).apply(style_positive_red, axis=None),
        use_container_width=True,
    )

    st.subheader("План ранней реализации (рекомендация)")
    st.dataframe(
        realization_view.style.format(fmt_cell).apply(style_positive_red, axis=None),
        use_container_width=True,
    )

    if TAB3_SHOW_TRANSFER_SNAPSHOT:
        st.subheader("Распределение коров по подразделениям (оценка мест)")
        if not (transfer_snapshot.empty and transfer_snapshot_monthly.empty):
            snapshot_view = transfer_snapshot
            if not transfer_snapshot_monthly.empty and "Месяц" in transfer_snapshot_monthly.columns:
                month_opts = sorted(transfer_snapshot_monthly["Месяц"].astype(str).dropna().unique().tolist())
                if month_opts:
                    sel_month = st.selectbox(
                        "Месяц баланса подразделений",
                        options=month_opts,
                        index=len(month_opts) - 1,
                        key=f"tab3_transfer_snapshot_month_{farm_detail}",
                    )
                    snapshot_view = transfer_snapshot_monthly.loc[
                        transfer_snapshot_monthly["Месяц"].astype(str) == str(sel_month)
                    ].copy()

            cols_order = [
                "Месяц",
                "Подразделение",
                "Коровы всего",
                "Коровы всего (прогноз)",
                "Коровы до переводов",
                "Коровы после переводов",
                "Мест (коровы)",
                "Оценка мест (коровы)",
                "Источник мест",
                "Переполнение (оценка)",
                "Переполнение до перевода",
                "Переполнение после перевода",
                "Свободно мест (оценка)",
                "Свободно мест до перевода",
                "Свободно мест после перевода",
                "Переведено из подразделения",
                "Переведено в подразделение",
                "Корректировка переводами, накопленная",
                "Историческая доля",
                "Дойные коровы",
                "Сухостойные коровы",
            ]
            cols_order = [c for c in cols_order if c in snapshot_view.columns]
            st.dataframe(
                snapshot_view[cols_order].style.format(fmt_cell).apply(style_positive_red, axis=None),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Рекомендации по переводам между подразделениями по месяцам ")
    if not transfer_recs.empty:
        if "Детализация по группам" in transfer_recs.columns:
            all_groups = [
                "Дойные коровы",
                "Сухостойные коровы",
                "Тёлки 0–3 мес",
                "Тёлки 3–8 мес",
                "Тёлки ≥9 мес",
                "Нетели",
                "Бычки 0–2 мес",
            ]
            def _normalize_group_rows(raw: Any) -> list[dict[str, Any]]:
                if not isinstance(raw, list):
                    raw = []
                out_map = {g: 0 for g in all_groups}
                for it in raw:
                    if not isinstance(it, dict):
                        continue
                    gname = str(it.get("Группа", "") or "").strip()
                    gval = int(round(float(pd.to_numeric(it.get("Рекомендовано перевести, голов"), errors="coerce") or 0.0)))
                    if not gname:
                        continue
                    if gname not in out_map:
                        out_map[gname] = 0
                    out_map[gname] = int(gval)
                return [{"Группа": g, "Рекомендовано перевести, голов": int(out_map[g])} for g in list(all_groups) + [g for g in out_map if g not in all_groups]]

            transfer_recs = transfer_recs.copy()
            transfer_recs["Детализация по группам"] = transfer_recs["Детализация по группам"].map(_normalize_group_rows)
            if "По группам (гол.)" in transfer_recs.columns:
                transfer_recs["По группам (гол.)"] = transfer_recs["Детализация по группам"].map(
                    lambda rows: "; ".join(
                        f"{str(r.get('Группа') or '')}: {int(r.get('Рекомендовано перевести, голов') or 0)}"
                        for r in rows
                    )
                )

        cols_order = [
            "Месяц",
            "Источник (переполнен)",
            "Куда перевести",
            "Рекомендовано перевести, голов",
            "Свободно в приёмнике, мест",
            "Свободно в приёмнике, мест (оценка)",
        ]
        cols_order = [c for c in cols_order if c in transfer_recs.columns]
        st.dataframe(
            transfer_recs[cols_order].style.format(fmt_cell),
            use_container_width=True,
            hide_index=True,
        )
        if "Детализация по группам" in transfer_recs.columns:
            st.markdown("**Детализация по группам для каждой рекомендации**")
            view_recs = transfer_recs.reset_index(drop=True)
            for i, rr in view_recs.iterrows():
                month_v = str(rr.get("Месяц", "") or "")
                src_v = str(rr.get("Источник (переполнен)", "") or "")
                dst_v = str(rr.get("Куда перевести", "") or "")
                move_v = int(round(float(pd.to_numeric(rr.get("Рекомендовано перевести, голов"), errors="coerce") or 0.0)))
                title = f"{i + 1}. {month_v}: {src_v} -> {dst_v} ({move_v} гол.)"
                groups_raw = rr.get("Детализация по группам", [])
                groups_ok = groups_raw if isinstance(groups_raw, list) else []
                with st.expander(title, expanded=False):
                    if groups_ok:
                        groups_df = pd.DataFrame(groups_ok)
                        if "Рекомендовано перевести, голов" in groups_df.columns:
                            groups_df["Рекомендовано перевести, голов"] = pd.to_numeric(
                                groups_df["Рекомендовано перевести, голов"],
                                errors="coerce",
                            ).fillna(0.0).round().astype(int)
                        st.dataframe(
                            groups_df.style.format(fmt_cell),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption("Для этой рекомендации групповая детализация не рассчитана.")
    else:
        st.info("Рекомендаций нет: в выбранном горизонте не удалось подобрать переводы между подразделениями.")

    if TAB3_SHOW_TRANSFER_FLOWS and not transfer_flows.empty:
        with st.expander("Исторические маршруты переездов (CARX)", expanded=False):
            st.dataframe(transfer_flows.head(30), use_container_width=True, hide_index=True)

    st.subheader("Скачать результат (Excel)")
    excel_bytes = make_excel_bytes_highlight_months_columns(
        forecast_view=forecast_view,
        overflow_view=overflow_view,
        indicator_to_overflow=INDICATOR_TO_OVERFLOW,
        realization_view=realization_view,
    )
    months = detail_df["Месяц"].astype(str).tolist()
    if months:
        file_name = f"farm_forecast_{farm_detail}_{months[0]}_to_{months[-1]}.xlsx"
    else:
        file_name = f"farm_forecast_{farm_detail}.xlsx"

    st.download_button(
        label="Скачать Excel: прогноз + переполнение + план реализации",
        data=excel_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"tab3_dl_excel_{farm_detail}",
    )

def _data_files_from_workspace() -> list[Path]:
    out: list[Path] = []
    root = Path("data")
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xls", ".xlsx", ".txt"}:
            out.append(p)
    return out

def render_tab3_farm() -> None:
    st.subheader("Прогноз по хозяйствам")

    _ensure_farm_tables()
    _ensure_subdivision_map_table()
    _ensure_capacity_table()
    _ensure_forecast_cache_table()

    if st.session_state.get("tab3_ui_state_version") != TAB3_UI_STATE_VERSION:
        st.session_state["tab3_ui_state_version"] = TAB3_UI_STATE_VERSION
        st.session_state.pop("tab3_monthly_all", None)
        st.session_state.pop("tab3_farm_infos", None)
        st.session_state.pop("tab3_target_month_end", None)

    months_ru = [
        "01 — Январь", "02 — Февраль", "03 — Март", "04 — Апрель",
        "05 — Май", "06 — Июнь", "07 — Июль", "08 — Август",
        "09 — Сентябрь", "10 — Октябрь", "11 — Ноябрь", "12 — Декабрь",
    ]
    today = date.today()

    c1, c2 = st.columns(2)
    with c1:
        year_sel = st.number_input("Год прогноза", min_value=2000, max_value=2100, value=today.year, step=1, key="tab3_year")
    with c2:
        month_sel_label = st.selectbox("Месяц прогноза", months_ru, index=today.month - 1, key="tab3_month_label")
        month_sel = int(month_sel_label.split("—")[0].strip())

    target_month_end = month_end(int(year_sel), int(month_sel))

    sub_status_df = _subdivision_status_df_from_db()
    farm_status_df = _farm_status_df_from_db()
    if sub_status_df.empty:
        st.info("В БД пока нет подразделений. Сначала загрузи файлы ниже.")
    else:
        with st.expander("Состав хозяйств (подразделения)", expanded=False):
            rows = sub_status_df.copy()
            rows_real = rows.loc[rows["Хозяйство"].astype(str) != TAB3_UNASSIGNED_FARM].copy()

            if not rows_real.empty:
                farm_counts = rows_real.groupby("Хозяйство")["Подразделение"].count().to_dict()
                rows_real["Подразделение UI"] = rows_real.apply(
                    lambda r: _subdivision_display_name(
                        str(r["Подразделение"]),
                        str(r["Хозяйство"]),
                        int(farm_counts.get(str(r["Хозяйство"]), 0)),
                    ),
                    axis=1,
                )
                grp = rows_real.sort_values(["Хозяйство", "Подразделение UI"]).groupby("Хозяйство")["Подразделение UI"].apply(list)
                for farm_name, subs in grp.items():
                    st.markdown(f"**{farm_name}:** {', '.join(subs)}")

                rows_real["Последняя дата данных"] = pd.to_datetime(
                    rows_real.get("Последняя дата данных"),
                    errors="coerce",
                ).dt.strftime("%Y-%m-%d")
                sub_cols = [
                    "Хозяйство",
                    "Подразделение UI",
                    "Статус",
                    "Отёлы",
                    "Осеменения",
                    "Запуски",
                    "Выбытие",
                    "Быки",
                    "Последняя дата данных",
                ]
                sub_cols = [c for c in sub_cols if c in rows_real.columns]
                st.dataframe(rows_real[sub_cols], use_container_width=True, hide_index=True)

            if isinstance(farm_status_df, pd.DataFrame) and not farm_status_df.empty:
                farms_view = farm_status_df.copy()
                farms_view["Последняя дата данных"] = pd.to_datetime(
                    farms_view.get("Последняя дата данных"),
                    errors="coerce",
                ).dt.strftime("%Y-%m-%d")
                farm_cols = [
                    "Хозяйство",
                    "Статус",
                    "Подразделений",
                    "Готовых подразделений",
                    "Последняя дата данных",
                ]
                farm_cols = [c for c in farm_cols if c in farms_view.columns]
                st.markdown("**Сводка по хозяйствам**")
                st.dataframe(farms_view[farm_cols], use_container_width=True, hide_index=True)

    ready_farms = (
        farm_status_df.loc[farm_status_df["Статус"] == "готово", "Хозяйство"].astype(str).tolist()
        if not farm_status_df.empty
        else []
    )
    all_farms = (
        sorted(
            {
                str(x)
                for x in farm_status_df.get("Хозяйство", pd.Series(dtype=object)).astype(str).tolist()
                if str(x).strip()
            }
        )
        if isinstance(farm_status_df, pd.DataFrame)
        else []
    )
    selected_farms = st.multiselect(
        "Выбери хозяйства для расчёта",
        options=ready_farms,
        default=ready_farms[:1],
        key="tab3_selected_farms",
    )
    _render_capacity_limits_panel(all_farms)
    if _is_admin_mode():
        _farm_param_editor_block(sorted(set(ready_farms)), apply_admin_overrides(get_param_source()))
        _render_manual_capacity_panel(sorted(set(ready_farms)))
    else:
        st.caption("Изменение параметров доступно только в админ-режиме.")

    selected_last_min: date | None = None
    selected_last_max: date | None = None
    if isinstance(farm_status_df, pd.DataFrame) and not farm_status_df.empty and selected_farms:
        fs = farm_status_df.copy()
        fs["Последняя дата данных"] = pd.to_datetime(fs.get("Последняя дата данных"), errors="coerce")
        fs = fs[fs["Хозяйство"].astype(str).isin([str(x) for x in selected_farms])].copy()
        dts = fs["Последняя дата данных"].dropna()
        if not dts.empty:
            selected_last_min = pd.Timestamp(dts.min()).date()
            selected_last_max = pd.Timestamp(dts.max()).date()

    if selected_last_max is None:
        if selected_farms:
            last_data_caption = "Последняя дата данных по выбранным хозяйствам: нет данных"
        else:
            last_data_caption = "Последняя дата данных: выбери хозяйство"
    elif len(selected_farms) == 1 or selected_last_min == selected_last_max:
        last_data_caption = f"Последняя дата данных: {selected_last_max.strftime('%Y-%m-%d')}"
    else:
        last_data_caption = (
            "Последняя дата данных по выбранным хозяйствам: "
            f"{selected_last_min.strftime('%Y-%m-%d')} ... {selected_last_max.strftime('%Y-%m-%d')}"
        )
    st.caption(last_data_caption)
    c_run, c_cache = st.columns([3, 2])
    run_clicked = c_run.button("Посчитать прогноз по хозяйствам", key="tab3_run_db", use_container_width=True)
    clear_clicked = c_cache.button(
        "Очистить кэш выбранных хозяйств",
        key="tab3_clear_cache",
        use_container_width=True,
        disabled=not bool(selected_farms),
    )

    if clear_clicked:
        for farm in selected_farms:
            _clear_forecast_cache(entity_type="farm", entity_name=farm)
        _reset_tab3_farm_result_state()
        if len(selected_farms) == 1:
            st.success(f"Кэш прогнозов очищен для хозяйства «{selected_farms[0]}».")
        else:
            st.success(f"Кэш прогнозов очищен для выбранных хозяйств: {len(selected_farms)} шт.")

    if run_clicked:
        if not selected_farms:
            st.error("Выбери хотя бы одно хозяйство.")
        else:
            base_params = apply_admin_overrides(get_param_source())
            all_farm_overrides = _farm_param_overrides_state() if _is_admin_mode() else {}
            all_sub_overrides = _subdivision_param_overrides_state() if _is_admin_mode() else {}
            all_monthly: list[pd.DataFrame] = []
            farm_infos: list[dict[str, Any]] = []
            errors: list[str] = []
            cache_hits = 0
            cache_miss = 0
            live_logs: list[str] = []

            prog = st.progress(0.0)
            farm_prog = st.progress(0.0)
            live_status = st.empty()
            live_log_box = st.empty()

            def _push_log(msg: str) -> None:
                ts = pd.Timestamp.now().strftime("%H:%M:%S")
                live_logs.append(f"[{ts}] {msg}")
                live_log_box.code("\n".join(live_logs[-14:]), language="text")

            for i, farm in enumerate(selected_farms, start=1):
                try:
                    farm_override = all_farm_overrides.get(farm)
                    subs_all = _subdivisions_for_farm(farm, ready_only=True)
                    if not subs_all:
                        raise ValueError("Нет готовых подразделений для расчёта.")
                    cap_overrides_by_sub = _tab3_demo_capacity_overrides_for_farm(farm)
                    sub_overrides_for_farm = {
                        sub: ov
                        for sub in subs_all
                        for ov in [all_sub_overrides.get(sub)]
                        if isinstance(ov, dict) and ov
                    }
                    farm_params_fallback = _build_farm_params(base_params, farm_override)
                    runtime_ov = {}
                    if _is_admin_mode():
                        by_scope = st.session_state.get("runtime_overrides_by_scope")
                        if isinstance(by_scope, dict):
                            global_ov = by_scope.get("__global__")
                            if isinstance(global_ov, dict):
                                runtime_ov = global_ov
                    ph_farm = _params_hash(
                        {
                            "mode": "per_subdivision_params.v12",
                            "farm_override": farm_override or {},
                            "subdivision_overrides": sub_overrides_for_farm,
                            "runtime_overrides": runtime_ov or {},
                        }
                    )
                    sig = _farm_signature_from_db(farm)
                    _push_log(f"{farm}: подготовка данных и проверка кэша")
                    farm_prog.progress(0.02)
                    live_status.caption(f"{farm}: проверка кэша…")
                    cached = _load_forecast_cache(
                        entity_type="farm",
                        entity_name=farm,
                        target_month_end=target_month_end,
                        data_signature=sig,
                        params_hash=ph_farm,
                    )
                    if cached is not None:
                        monthly_df, info = cached
                        cache_hits += 1
                        farm_prog.progress(1.0)
                        live_status.caption(f"{farm}: готово из кэша")
                        _push_log(f"{farm}: кэш hit")
                    else:
                        _push_log(f"{farm}: подразделений в расчёте {len(subs_all)}")

                        def _params_for_subdivision(subdivision: str, tables: dict[str, pd.DataFrame]) -> dict:
                            sub_override = all_sub_overrides.get(subdivision)
                            return _resolve_subdivision_params_runtime(
                                subdivision=subdivision,
                                tables=tables,
                                farm_override=farm_override,
                                subdivision_override=sub_override,
                                capacity_override=cap_overrides_by_sub.get(subdivision),
                                log_fn=lambda msg: _push_log(f"{farm}/{msg}"),
                            )

                        def _sum_cb(sub: str, step_i: int, total_steps: int, d_end: date) -> None:
                            farm_prog.progress(step_i / max(1, total_steps))
                            live_status.caption(
                                f"{farm}: {sub}, месяц {step_i}/{total_steps} ({_month_label(d_end)})"
                            )
                            if step_i == 1 or step_i == total_steps or step_i % 2 == 0:
                                _push_log(f"{farm}: {sub} {step_i}/{total_steps} ({_month_label(d_end)})")

                        monthly_df, info = _compute_farm_forecast_sum_of_subdivisions(
                            farm_name=farm,
                            subdivisions=subs_all,
                            target_month_end=target_month_end,
                            params=farm_params_fallback,
                            params_for_subdivision=_params_for_subdivision,
                            progress_cb=_sum_cb,
                        )
                        info["subdivisions_n"] = len(subs_all)

                        _save_forecast_cache(
                            entity_type="farm",
                            entity_name=farm,
                            target_month_end=target_month_end,
                            data_signature=sig,
                            params_hash=ph_farm,
                            monthly_df=monthly_df,
                            info=info,
                        )
                        cache_miss += 1
                        farm_prog.progress(1.0)
                        live_status.caption(f"{farm}: расчёт завершён")
                        _push_log(f"{farm}: готово, результат записан в кэш")

                    if not isinstance(info, dict):
                        info = {}
                    rec_payload = info.get("transfer_recommendations_monthly", info.get("transfer_recommendations"))
                    snap_payload = info.get("transfer_snapshot_monthly", info.get("transfer_snapshot"))
                    need_rebuild_transfer = (
                        not isinstance(rec_payload, list)
                        or not isinstance(snap_payload, list)
                        or (len(rec_payload) == 0 and len(snap_payload) == 0)
                    )
                    if need_rebuild_transfer:
                        _push_log(f"{farm}: анализ переездов CARX и подбор переводов")

                        def _params_for_subdivision(subdivision: str, tables: dict[str, pd.DataFrame]) -> dict:
                            sub_override = all_sub_overrides.get(subdivision)
                            return _resolve_subdivision_params_runtime(
                                subdivision=subdivision,
                                tables=tables,
                                farm_override=farm_override,
                                subdivision_override=sub_override,
                                capacity_override=cap_overrides_by_sub.get(subdivision),
                                log_fn=lambda msg: _push_log(f"{farm}/{msg}"),
                            )

                        rec_df, flows_df, snap_df, snap_monthly_df, rec_meta = _build_transfer_recommendations(
                            farm_name=farm,
                            target_month_end=target_month_end,
                            params=farm_params_fallback,
                            params_for_subdivision=_params_for_subdivision,
                        )
                        info["transfer_recommendations"] = rec_df.to_dict(orient="records")
                        info["transfer_recommendations_monthly"] = rec_df.to_dict(orient="records")
                        info["transfer_move_flows"] = flows_df.to_dict(orient="records")
                        info["transfer_snapshot"] = snap_df.to_dict(orient="records")
                        info["transfer_snapshot_monthly"] = snap_monthly_df.to_dict(orient="records")
                        info["transfer_meta"] = rec_meta

                    all_monthly.append(monthly_df)
                    farm_infos.append(info)
                    if isinstance(info, dict) and info.get("sanity_violations"):
                        st.warning(
                            f"{farm}: обнаружены аномалии в сводном расчёте, "
                            "применён безопасный режим (сумма прогнозов подразделений)."
                        )
                except Exception as e:
                    errors.append(f"{farm}: {e}")
                prog.progress(i / max(1, len(selected_farms)))
            prog.empty()

            if errors:
                st.error("Ошибки по части хозяйств:\n- " + "\n- ".join(errors))

            if all_monthly:
                st.session_state["tab3_monthly_all"] = pd.concat(all_monthly, ignore_index=True)
                st.session_state["tab3_farm_infos"] = farm_infos
                st.session_state["tab3_target_month_end"] = target_month_end
            else:
                st.session_state.pop("tab3_monthly_all", None)
                st.session_state.pop("tab3_farm_infos", None)
                st.session_state.pop("tab3_target_month_end", None)

    default_bt_farm = selected_farms[0] if selected_farms else (ready_farms[0] if ready_farms else None)
    _render_farm_backtesting_panel(default_farm=default_bt_farm)

    monthly_all = st.session_state.get("tab3_monthly_all")
    farm_infos = st.session_state.get("tab3_farm_infos")
    target_cached = st.session_state.get("tab3_target_month_end", target_month_end)
    if isinstance(monthly_all, pd.DataFrame) and not monthly_all.empty and isinstance(farm_infos, list):
        _render_results(monthly_all, farm_infos, target_cached)

    with st.expander("Добавить/обновить подразделения хозяйства в БД", expanded=False):
        mode = st.radio(
            "Режим записи",
            options=["Заменить данные подразделения", "Добавить к данным подразделения"],
            index=0,
            horizontal=True,
            key="tab3_upload_mode",
        )

        files = st.file_uploader(
            "Файлы по подразделениям (можно сразу много): отёлы, осеменения, запуск, выбытие, быки",
            type=["xls", "xlsx", "txt"],
            accept_multiple_files=True,
            key="tab3_farm_files_upload",
        )
        st.caption(
            "Дедупликация включена: при режиме «Добавить» старые и новые строки объединяются, "
            "дубли по ключевым полям удаляются автоматически."
        )

        last_upload_logs = st.session_state.get("tab3_last_upload_logs")
        last_upload_status = st.session_state.get("tab3_last_upload_status")
        if isinstance(last_upload_status, dict) and last_upload_status.get("message"):
            kind = str(last_upload_status.get("kind") or "info")
            msg = str(last_upload_status.get("message") or "")
            if kind == "success":
                st.success(msg)
            elif kind == "error":
                st.error(msg)
            else:
                st.info(msg)
        if isinstance(last_upload_logs, list) and last_upload_logs:
            st.markdown("**Журнал последней загрузки**")
            st.code("\n".join(str(x) for x in last_upload_logs), language="text")

        if not files:
            return

        bundles, detect_df = _group_files(list(files))
        bundles, attached_bull_bundles = _merge_aux_bull_bundles(bundles)
        if attached_bull_bundles and isinstance(detect_df, pd.DataFrame) and not detect_df.empty:
            for source_name, target_name in attached_bull_bundles.items():
                mask = (
                    (detect_df["Подразделение"].astype(str) == str(source_name))
                    & (detect_df["Тип"].astype(str) == "bulls")
                )
                if not bool(mask.any()):
                    continue
                detect_df.loc[mask, "Подразделение"] = str(target_name)
                detect_df.loc[mask, "Статус"] = "быки добавлены к комплекту Excel"
        st.markdown("**Распознавание файлов**")
        st.dataframe(detect_df, use_container_width=True, hide_index=True)

        if attached_bull_bundles:
            attached_names = ", ".join(sorted(attached_bull_bundles.keys()))
            st.caption(
                "Файлы быков распознаны как отдельные подразделения по имени файла и автоматически "
                f"прикреплены к общему комплекту Excel: {attached_names}."
            )

        summary_rows: list[dict[str, str]] = []
        ready_upload_subs: list[str] = []
        for subdivision, b in bundles.items():
            miss = []
            if b.calv is None:
                miss.append("отёлы")
            if b.ins is None:
                miss.append("осеменения")
            if b.dry is None:
                miss.append("запуск")
            if b.disp is None:
                miss.append("выбытие")

            if miss:
                summary_rows.append({"Подразделение": subdivision, "Статус": "неполный комплект", "Отсутствует": ", ".join(miss)})
            else:
                summary_rows.append({"Подразделение": subdivision, "Статус": "готово к загрузке", "Отсутствует": ""})
                ready_upload_subs.append(subdivision)

        st.markdown("**Комплектность для загрузки**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if not ready_upload_subs:
            st.warning("Нет ни одного подразделения с полным набором файлов для загрузки.")
            return

        upload_log_box = st.empty()

        if st.button("Загрузить в БД", key="tab3_btn_upload_db", use_container_width=True):
            replace_subdivision = mode.startswith("Заменить")
            errors: list[str] = []
            updated_subdivisions: list[str] = []
            upload_logs: list[str] = []

            def _log(message: str) -> None:
                ts = datetime.now().strftime("%H:%M:%S")
                upload_logs.append(f"[{ts}] {message}")
                st.session_state["tab3_last_upload_logs"] = list(upload_logs)
                upload_log_box.code("\n".join(upload_logs), language="text")

            st.session_state["tab3_last_upload_logs"] = []
            st.session_state["tab3_last_upload_status"] = {"kind": "info", "message": "Загрузка в БД запущена."}
            _log(f"Старт загрузки. Режим: {'замена' if replace_subdivision else 'добавление'}.")
            _log(f"Готовых комплектов к загрузке: {len(ready_upload_subs)} -> {', '.join(map(str, ready_upload_subs))}")

            prog = st.progress(0.0)
            for i, subdivision in enumerate(ready_upload_subs, start=1):
                try:
                    _log(f"{subdivision}: подготовка таблиц из файлов.")
                    tables = _prepare_tables(bundles[subdivision])
                    table_sizes = ", ".join(
                        f"{key}={int(len(tables.get(key, pd.DataFrame())))}"
                        for key in ("calv", "ins", "dry", "disp", "bulls")
                    )
                    _log(f"{subdivision}: таблицы собраны ({table_sizes}).")
                    updated = _save_bundle_tables_to_db(
                        subdivision,
                        tables,
                        replace_subdivision=replace_subdivision,
                    )
                    updated_subdivisions.extend(updated)
                    if updated:
                        _log(f"{subdivision}: в БД обновлены подразделения -> {', '.join(map(str, updated))}.")
                    else:
                        _log(f"{subdivision}: загрузка завершилась без обновлённых подразделений.")
                except Exception as e:
                    errors.append(f"{subdivision}: {e}")
                    _log(f"{subdivision}: ошибка -> {e}")
                prog.progress(i / max(1, len(ready_upload_subs)))
            prog.empty()

            if errors:
                st.session_state["tab3_last_upload_status"] = {
                    "kind": "error",
                    "message": "Часть подразделений не загрузилась.",
                }
                _log(f"Загрузка завершена с ошибками. Ошибок: {len(errors)}.")
                st.error("Часть подразделений не загрузилась:\n- " + "\n- ".join(errors))
            else:
                n_upd = len(set(updated_subdivisions)) if updated_subdivisions else len(ready_upload_subs)
                st.session_state["tab3_last_upload_status"] = {
                    "kind": "success",
                    "message": f"Готово. В БД обновлено подразделений: {n_upd}",
                }
                _log(f"Загрузка завершена успешно. Обновлено подразделений: {n_upd}.")
                st.success(f"Готово. В БД обновлено подразделений: {n_upd}")
                st.rerun()


__all__ = [name for name in globals() if not name.startswith("__")]
