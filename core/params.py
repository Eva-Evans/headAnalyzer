from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text

from db import engine
import model_params as mp

def _get_db_signature() -> str:
    q = """
    SELECT
      COALESCE((SELECT MAX(event_date)::text FROM calvings_births_raw), '') AS calv_max,
      COALESCE((SELECT MAX(event_date)::text FROM inseminations_raw), '') AS ins_max,
      COALESCE((SELECT MAX(event_date)::text FROM dryoff_raw), '') AS dry_max,
      COALESCE((SELECT MAX(event_date)::text FROM disposals_raw), '') AS disp_max,

      (SELECT COUNT(*) FROM calvings_births_raw) AS calv_n,
      (SELECT COUNT(*) FROM inseminations_raw) AS ins_n,
      (SELECT COUNT(*) FROM dryoff_raw) AS dry_n,
      (SELECT COUNT(*) FROM disposals_raw) AS disp_n
    ;
    """
    try:
        df = pd.read_sql(q, con=engine)
        r = df.iloc[0].to_dict()
        return f"{r['calv_max']}|{r['ins_max']}|{r['dry_max']}|{r['disp_max']}|{r['calv_n']}|{r['ins_n']}|{r['dry_n']}|{r['disp_n']}"
    except Exception:
        return "no-db"

def _ensure_params_cache_table() -> None:
    q = """
    CREATE TABLE IF NOT EXISTS model_params_cache (
        signature TEXT PRIMARY KEY,
        params_json JSONB NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(q))

def _load_params_from_db_cache(sig: str) -> Optional[Dict[str, Any]]:
    _ensure_params_cache_table()
    q = "SELECT params_json FROM model_params_cache WHERE signature = :sig LIMIT 1;"
    try:
        df = pd.read_sql(text(q), con=engine, params={"sig": sig})
        if df.empty:
            return None
        return df.iloc[0]["params_json"]
    except Exception:
        return None

def _save_params_to_db_cache(sig: str, params: Dict[str, Any]) -> None:
    _ensure_params_cache_table()
    q = """
    INSERT INTO model_params_cache(signature, params_json, updated_at)
    VALUES (:sig, :params_json::jsonb, NOW())
    ON CONFLICT (signature)
    DO UPDATE SET params_json = EXCLUDED.params_json, updated_at = NOW();
    """
    with engine.begin() as conn:
        conn.execute(text(q), {"sig": sig, "params_json": json.dumps(params, ensure_ascii=False)})

def ensure_params_loaded_from_db_silent() -> None:
    if "computed_params" in st.session_state and isinstance(st.session_state.computed_params, dict):
        return
    sig = _get_db_signature()
    cached = _load_params_from_db_cache(sig)
    if isinstance(cached, dict) and cached:
        st.session_state.computed_params = cached
        return

def compute_params_from_db() -> Dict[str, Any]:
    raise NotImplementedError("Функция compute_params_from_db не реализована в core/params.py")

@st.cache_data(show_spinner=False)
def _compute_params_cached(db_signature: str) -> Dict[str, Any]:
    return compute_params_from_db()

def get_param_source() -> Dict[str, Any]:
    if "computed_params" in st.session_state and isinstance(st.session_state.computed_params, dict):
        return _inject_live_semen_usage(_normalize_param_aliases(st.session_state.computed_params))

    return _inject_live_semen_usage(_normalize_param_aliases({
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
    }))

from copy import deepcopy


def _deep_merge(dst: dict, src: dict) -> dict:
    """Рекурсивно накладывает src поверх dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _normalize_param_aliases(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Приводит алиасы ключей к каноническому формату UPPERCASE,
    который используется в UI и динамическом прогнозе.
    """
    out = deepcopy(params) if isinstance(params, dict) else {}
    aliases = (
        ("gestation_days", "GESTATION_DAYS"),
        ("dry_days", "DRY_DAYS"),
        ("DRY_DAYS_AVG", "DRY_DAYS"),
        ("conception", "CONCEPTION_PARAMS"),
        ("disposal_params", "DISPOSAL_PARAMS"),
        ("annual_disposal_rate", "ANNUAL_DISPOSAL_RATE"),
        ("insemination_params", "INSEMINATION_PARAMS"),
        ("semen_usage", "SEMEN_USAGE_SHARES"),
        ("semen_sex_ratios", "SEMEN_SEX_RATIOS"),
    )
    for low_key, up_key in aliases:
        if up_key not in out and low_key in out:
            out[up_key] = out[low_key]
        if low_key not in out and up_key in out:
            out[low_key] = out[up_key]
    return out


def _inject_live_semen_usage(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновляет блок долей использования семени по живым данным из БД.
    Это нужно, чтобы в табе параметров не показывались устаревшие fallback-значения.
    """
    out = deepcopy(params) if isinstance(params, dict) else {}
    try:
        from forecast_dynamic import load_tables, compute_semen_usage_from_db

        live = compute_semen_usage_from_db(load_tables())
        if isinstance(live, dict) and live:
            for k in ("cow_trad", "cow_sex", "heifer_trad", "heifer_sex"):
                if k in live:
                    live[k] = round(float(live[k]), 4)
            out["SEMEN_USAGE_SHARES"] = live
            out["semen_usage"] = live
    except Exception:
                                                                  
        pass
    return out


def apply_admin_overrides(
    base_params: dict,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Возвращает параметры для расчёта:
    base_params (из БД/кэша) + runtime_overrides (если админ включён).
    Ничего в БД НЕ пишет.
    """
    out = _normalize_param_aliases(base_params)

    if bool(st.session_state.get("is_admin", False)):
        ov = runtime_overrides if runtime_overrides is not None else st.session_state.get("runtime_overrides")
        ov = _normalize_param_aliases(ov)
        if isinstance(ov, dict) and ov:
            _deep_merge(out, ov)

    return out

def admin_key_true() -> str:
    return os.getenv("ADMIN_KEY", "admin")

def recompute_and_cache_params() -> Tuple[bool, str]:
    try:
        _compute_params_cached.clear()
        sig = _get_db_signature()
        params = _compute_params_cached(sig)
        st.session_state.computed_params = params
        _save_params_to_db_cache(sig, params)
        return True, "Параметры пересчитаны и сохранены в кэш БД."
    except Exception as e:
        return False, f"Не удалось пересчитать параметры из данных: {e}"

def save_params_cache_current_signature(params: Dict[str, Any]) -> None:
    _save_params_to_db_cache(_get_db_signature(), params)
