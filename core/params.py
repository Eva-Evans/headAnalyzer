from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text

from db import engine
import model_params as mp

PARAMS_CACHE_VERSION = "v3"


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
        return (
            f"{PARAMS_CACHE_VERSION}|"
            f"{r['calv_max']}|{r['ins_max']}|{r['dry_max']}|{r['disp_max']}|"
            f"{r['calv_n']}|{r['ins_n']}|{r['dry_n']}|{r['disp_n']}"
        )
    except Exception:
        return f"{PARAMS_CACHE_VERSION}|no-db"

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
    VALUES (:sig, CAST(:params_json AS jsonb), NOW())
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

def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _tables_signature(tables: Dict[str, pd.DataFrame]) -> str:
    parts: list[str] = [PARAMS_CACHE_VERSION]
    for key in ("calv", "ins", "dry", "disp", "bulls"):
        df = tables.get(key, pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            parts.append(f"{key}:0::")
            continue
        n = int(len(df))
        mn = ""
        mx = ""
        if "event_date" in df.columns:
            try:
                dt = pd.to_datetime(df["event_date"], errors="coerce")
                mn_ts = dt.min()
                mx_ts = dt.max()
                if pd.notna(mn_ts):
                    mn = str(pd.Timestamp(mn_ts).date())
                if pd.notna(mx_ts):
                    mx = str(pd.Timestamp(mx_ts).date())
            except Exception:
                mn = ""
                mx = ""
        extra_parts: list[str] = []
        if key == "calv":
            mother_n = 0
            born_n = 0
            otel_n = 0
            try:
                if "mother_reg" in df.columns:
                    mother = df["mother_reg"].astype("string").fillna("").str.strip()
                    mother_n = int((mother != "").sum())
                if "event_type" in df.columns:
                    ev = df["event_type"].astype("string").fillna("").str.upper().str.replace("Ё", "Е", regex=False)
                    born_n = int(ev.str.contains("РОЖ|BORN|BIRTH", regex=True, na=False).sum())
                    otel_n = int(ev.str.contains("ОТЕЛ|CALV", regex=True, na=False).sum())
            except Exception:
                mother_n = 0
                born_n = 0
                otel_n = 0
            extra_parts.extend([f"mother:{mother_n}", f"born:{born_n}", f"otel:{otel_n}"])
        elif key == "ins":
            p_n = 0
            reg_u = 0
            try:
                if "result" in df.columns:
                    res = df["result"].astype("string").fillna("").str.upper().str.strip()
                    p_n = int((res == "P").sum())
                if "reg" in df.columns:
                    reg = df["reg"].astype("string").fillna("").str.strip()
                    reg_u = int(reg[reg != ""].nunique())
            except Exception:
                p_n = 0
                reg_u = 0
            extra_parts.extend([f"p:{p_n}", f"regu:{reg_u}"])
        elif key in {"dry", "disp"}:
            reg_u = 0
            try:
                if "reg" in df.columns:
                    reg = df["reg"].astype("string").fillna("").str.strip()
                    reg_u = int(reg[reg != ""].nunique())
            except Exception:
                reg_u = 0
            extra_parts.append(f"regu:{reg_u}")
        elif key == "bulls":
            bull_u = 0
            try:
                if "bull_code" in df.columns:
                    bull = df["bull_code"].astype("string").fillna("").str.strip()
                    bull_u = int(bull[bull != ""].nunique())
            except Exception:
                bull_u = 0
            extra_parts.append(f"bullu:{bull_u}")
        extra = "|".join(extra_parts)
        parts.append(f"{key}:{n}:{mn}:{mx}:{extra}")
    return "|".join(parts)


def _runtime_params_to_dict(rp: Any) -> Dict[str, Any]:
    obj = _to_jsonable(rp)
    if not isinstance(obj, dict):
        return _normalize_param_aliases({})
    out = {
        "CONCEPTION_PARAMS": obj.get("conception_params"),
        "GESTATION_DAYS": obj.get("gestation_days"),
        "DRY_DAYS": obj.get("dry_days"),
        "DISPOSAL_PARAMS": obj.get("disposal_params"),
        "ANNUAL_DISPOSAL_RATE": obj.get("annual_disposal_rate"),
        "INSEMINATION_PARAMS": obj.get("insemination_params"),
        "meta": obj.get("meta"),
    }
    return _normalize_param_aliases(out)


def compute_params_from_db() -> Dict[str, Any]:
    from params_runtime import compute_params_from_db as _compute_runtime_params_from_db

    rp = _compute_runtime_params_from_db()
    return _runtime_params_to_dict(rp)


def compute_params_from_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    from params_runtime import (
        RuntimeParams,
        _compute_conception_params,
        _compute_disposal_params,
        _compute_dry_days,
        _compute_gestation_days,
        _compute_insemination_params,
    )

    calv = tables.get("calv", pd.DataFrame())
    ins = tables.get("ins", pd.DataFrame())
    dry = tables.get("dry", pd.DataFrame())
    disp = tables.get("disp", pd.DataFrame())

    conception_params = _compute_conception_params(ins.copy() if isinstance(ins, pd.DataFrame) else pd.DataFrame())
    gest, gest_meta = _compute_gestation_days(
        calv.copy() if isinstance(calv, pd.DataFrame) else pd.DataFrame(),
        ins.copy() if isinstance(ins, pd.DataFrame) else pd.DataFrame(),
    )
    dry_days, dry_meta = _compute_dry_days(
        calv.copy() if isinstance(calv, pd.DataFrame) else pd.DataFrame(),
        dry.copy() if isinstance(dry, pd.DataFrame) else pd.DataFrame(),
    )
    disposal_params, annual_rate, disp_meta = _compute_disposal_params(
        calv.copy() if isinstance(calv, pd.DataFrame) else pd.DataFrame(),
        disp.copy() if isinstance(disp, pd.DataFrame) else pd.DataFrame(),
    )
    insemination_params = _compute_insemination_params(
        ins.copy() if isinstance(ins, pd.DataFrame) else pd.DataFrame(),
        calv.copy() if isinstance(calv, pd.DataFrame) else pd.DataFrame(),
    )

    rp = RuntimeParams(
        conception_params=conception_params,
        gestation_days=float(gest),
        dry_days=int(dry_days),
        disposal_params=disposal_params,
        annual_disposal_rate=float(annual_rate),
        insemination_params=insemination_params,
        meta={"gestation": gest_meta, "dry": dry_meta, "disposal": disp_meta},
    )
    return _runtime_params_to_dict(rp)


def get_or_compute_subdivision_params(subdivision_name: str, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    sub = str(subdivision_name or "").strip()
    if not sub:
        raise ValueError("Пустое имя подразделения для расчёта параметров.")

    sig = f"subdivision::{sub}::{_tables_signature(tables)}"
    cached = _load_params_from_db_cache(sig)
    if isinstance(cached, dict) and cached:
        return _normalize_param_aliases(cached)

    params = compute_params_from_tables(tables)
    _save_params_to_db_cache(sig, params)
    return _normalize_param_aliases(params)

@st.cache_data(show_spinner=False)
def _compute_params_cached(db_signature: str) -> Dict[str, Any]:
    return compute_params_from_db()

def get_param_source() -> Dict[str, Any]:
    if "computed_params" in st.session_state and isinstance(st.session_state.computed_params, dict):
        return inject_live_semen_params(_normalize_param_aliases(st.session_state.computed_params))

    return inject_live_semen_params(_normalize_param_aliases({
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
            "cow_conception_month_factors": dict(mp.INSEMINATION_PARAMS.cow_conception_month_factors),
            "heifer_services_per_conception": float(mp.INSEMINATION_PARAMS.heifer_services_per_conception),
            "heifer_ai_interval_days": float(mp.INSEMINATION_PARAMS.heifer_ai_interval_days),
            "heifer_first_ai_age_days": float(mp.INSEMINATION_PARAMS.heifer_first_ai_age_days),
            "heifer_conception_month_factors": dict(mp.INSEMINATION_PARAMS.heifer_conception_month_factors),
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


def get_model_default_params() -> Dict[str, Any]:
    return _normalize_param_aliases({
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
            "cow_conception_month_factors": dict(mp.INSEMINATION_PARAMS.cow_conception_month_factors),
            "heifer_services_per_conception": float(mp.INSEMINATION_PARAMS.heifer_services_per_conception),
            "heifer_ai_interval_days": float(mp.INSEMINATION_PARAMS.heifer_ai_interval_days),
            "heifer_first_ai_age_days": float(mp.INSEMINATION_PARAMS.heifer_first_ai_age_days),
            "heifer_conception_month_factors": dict(mp.INSEMINATION_PARAMS.heifer_conception_month_factors),
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
        "HERD_CAPACITY": dict(mp.HERD_CAPACITY),
    })

from copy import deepcopy


def _normalize_nested_numeric_keys(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            nk: Any = k
            if isinstance(k, str) and k.isdigit():
                try:
                    nk = int(k)
                except Exception:
                    nk = k
            out[nk] = _normalize_nested_numeric_keys(v)
        return out
    if isinstance(value, list):
        return [_normalize_nested_numeric_keys(v) for v in value]
    return value


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
    out = _normalize_nested_numeric_keys(out)
    return out


def inject_live_semen_params(
    params: Dict[str, Any],
    tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Обновляет блоки по использованию семени и полу телят живыми данными.
    Если tables не переданы, используются общие таблицы из БД.
    """
    out = deepcopy(params) if isinstance(params, dict) else {}
    try:
        from forecast_dynamic import (
            compute_semen_sex_ratios_from_db,
            compute_semen_usage_from_db,
            load_tables,
        )

        live_tables = tables if isinstance(tables, dict) else load_tables()

        live_usage = compute_semen_usage_from_db(live_tables)
        if isinstance(live_usage, dict) and live_usage:
            usage_out = deepcopy(live_usage)
            for k in ("cow_trad", "cow_sex", "heifer_trad", "heifer_sex"):
                if k in usage_out:
                    usage_out[k] = round(float(usage_out[k]), 4)
            out["SEMEN_USAGE_SHARES"] = usage_out
            out["semen_usage"] = usage_out

        live_ratios = compute_semen_sex_ratios_from_db(live_tables)
        if isinstance(live_ratios, dict) and live_ratios:
            ratio_out: Dict[str, Any] = {}
            for semen_key in ("trad", "sex"):
                obj = live_ratios.get(semen_key)
                if obj is None:
                    continue
                bull_share = getattr(obj, "bull_share", None)
                heifer_share = getattr(obj, "heifer_share", None)
                if bull_share is None and isinstance(obj, dict):
                    bull_share = obj.get("bull_share")
                    heifer_share = obj.get("heifer_share")
                if bull_share is None and heifer_share is None:
                    continue
                if bull_share is None:
                    bull_share = 1.0 - float(heifer_share)
                if heifer_share is None:
                    heifer_share = 1.0 - float(bull_share)
                ratio_out[semen_key] = {
                    "bull_share": round(float(bull_share), 4),
                    "heifer_share": round(float(heifer_share), 4),
                }
            if ratio_out:
                out["SEMEN_SEX_RATIOS"] = ratio_out
                out["semen_sex_ratios"] = ratio_out
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
        if runtime_overrides is not None:
            ov = runtime_overrides
        else:
            by_scope = st.session_state.get("runtime_overrides_by_scope")
            if isinstance(by_scope, dict):
                ov = by_scope.get("__global__")
            else:
                ov = None
        ov = _normalize_param_aliases(ov)
        if isinstance(ov, dict) and ov:
            _deep_merge(out, ov)
            ov_sus = ov.get("SEMEN_USAGE_SHARES") or ov.get("semen_usage")
            out_sus = out.get("SEMEN_USAGE_SHARES") or out.get("semen_usage")
            if isinstance(ov_sus, dict) and isinstance(out_sus, dict):
                if "cow_sex" in ov_sus and "cow_trad" not in ov_sus:
                    out_sus["cow_trad"] = max(0.0, 1.0 - float(out_sus.get("cow_sex", 0.0)))
                if "cow_trad" in ov_sus and "cow_sex" not in ov_sus:
                    out_sus["cow_sex"] = max(0.0, 1.0 - float(out_sus.get("cow_trad", 0.0)))
                if "heifer_sex" in ov_sus and "heifer_trad" not in ov_sus:
                    out_sus["heifer_trad"] = max(0.0, 1.0 - float(out_sus.get("heifer_sex", 0.0)))
                if "heifer_trad" in ov_sus and "heifer_sex" not in ov_sus:
                    out_sus["heifer_sex"] = max(0.0, 1.0 - float(out_sus.get("heifer_trad", 0.0)))
                for left_key, right_key in (("cow_trad", "cow_sex"), ("heifer_trad", "heifer_sex")):
                    if left_key in out_sus or right_key in out_sus:
                        left = float(out_sus.get(left_key, 0.0))
                        right = float(out_sus.get(right_key, 0.0))
                        s = max(1e-9, left + right)
                        out_sus[left_key] = left / s
                        out_sus[right_key] = right / s
                out["SEMEN_USAGE_SHARES"] = out_sus
                out["semen_usage"] = out_sus

            ov_ssr = ov.get("SEMEN_SEX_RATIOS") or ov.get("semen_sex_ratios")
            out_ssr = out.get("SEMEN_SEX_RATIOS") or out.get("semen_sex_ratios")
            if isinstance(ov_ssr, dict) and isinstance(out_ssr, dict):
                for semen_key in ("trad", "sex"):
                    ov_part = ov_ssr.get(semen_key)
                    out_part = out_ssr.get(semen_key)
                    if not isinstance(ov_part, dict) or not isinstance(out_part, dict):
                        continue
                    if "heifer_share" in ov_part and "bull_share" not in ov_part:
                        out_part["bull_share"] = max(0.0, 1.0 - float(out_part.get("heifer_share", 0.0)))
                    if "bull_share" in ov_part and "heifer_share" not in ov_part:
                        out_part["heifer_share"] = max(0.0, 1.0 - float(out_part.get("bull_share", 0.0)))
                    bull = float(out_part.get("bull_share", 0.0))
                    heif = float(out_part.get("heifer_share", 0.0))
                    s = max(1e-9, bull + heif)
                    out_part["bull_share"] = bull / s
                    out_part["heifer_share"] = heif / s
                    out_ssr[semen_key] = out_part
                out["SEMEN_SEX_RATIOS"] = out_ssr
                out["semen_sex_ratios"] = out_ssr

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


def clear_model_params_cache_all() -> int:
    _ensure_params_cache_table()
    removed = 0
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM model_params_cache;"))
        removed = int(res.rowcount or 0)
    try:
        _compute_params_cached.clear()
    except Exception:
        pass
    st.session_state.pop("computed_params", None)
    return removed


def clear_model_params_cache_for_subdivision(subdivision_name: str) -> int:
    sub = str(subdivision_name or "").strip()
    if not sub:
        return 0
    _ensure_params_cache_table()
    q = """
    DELETE FROM model_params_cache
    WHERE split_part(signature, '::', 1) = 'subdivision'
      AND split_part(signature, '::', 2) = :sub;
    """
    with engine.begin() as conn:
        res = conn.execute(text(q), {"sub": sub})
        removed = int(res.rowcount or 0)
    try:
        _compute_params_cached.clear()
    except Exception:
        pass
    return removed
