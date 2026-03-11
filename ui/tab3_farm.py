from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.constants import INDICATORS, INDICATOR_TO_OVERFLOW, OVERFLOW_COLS, OVERFLOW_GROUP_COLS
from core.excel_export import make_excel_bytes_highlight_months_columns
from core.helpers import iter_month_ends, month_end, norm_label, vals_get
from core.params import apply_admin_overrides, get_param_source
from core.realization import build_early_realization_plan
from db import engine
from etl.bulls import read_bulls_txt
from etl.calvings_births import read_calvings_excel
from etl.disposals import read_disposals_excel
from etl.dryoff import read_dryoff_excel
from etl.inseminations import clean_inseminations, read_inseminations_excel
from forecast_dynamic import (
    compute_forecast_dynamic_from_tables,
    compute_semen_sex_ratios_from_db,
    compute_semen_usage_from_db,
    latest_data_date,
)
from ui.styles import BAD, fmt_cell, style_positive_red


TAB3_TABLES = {
    "calv": "tab3_calvings_farm_raw",
    "ins": "tab3_inseminations_farm_raw",
    "dry": "tab3_dryoff_farm_raw",
    "disp": "tab3_disposals_farm_raw",
    "bulls": "tab3_bulls_farm_raw",
}
TAB3_CACHE_TABLE = "tab3_forecast_cache"
TAB3_MAP_TABLE = "tab3_subdivision_farm_map"
TAB3_CACHE_SCHEMA_VERSION = "2026-03-03.v8"
TAB3_UI_STATE_VERSION = "2026-02-26.v3"
TAB3_SHOW_TRANSFER_SNAPSHOT = False
TAB3_SHOW_TRANSFER_FLOWS = False
TAB3_UNASSIGNED_FARM = "ВНЕ ХОЗЯЙСТВА"

FARM_BACKTEST_TARGETS: list[str] = [
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки (условно)",
    "Ожидаемые тёлочки (условно)",
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]

FARM_PERCENT_TARGETS = {
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
}


@dataclass
class FarmUploadBundle:
    farm_name: str
    calv: Any | None = None
    ins: Any | None = None
    dry: Any | None = None
    disp: Any | None = None
    bulls: list[Any] = field(default_factory=list)


_STOPWORDS = {
    "ОСЕМЕН", "ОСЕМЕНЕНИЯ", "INSEM", "INSEMINATION",
    "ОТЕЛ", "ОТЕЛЫ", "ОТЕЛА", "РОДИВ", "РОДИВШ", "CALV", "BIRTH", "BORN",
    "ЗАПУСК", "DRY", "DRYOFF",
    "ВЫБЫТИЕ", "DISPOSAL", "DISPOSALS",
    "БЫК", "БЫКИ", "BULL", "BULLS",
    "ПЛЮС", "DZ", "XLS", "XLSX", "TXT", "ДАННЫЕ", "ЖК", "МТФ", "РЖК",
}

_STOP_PREFIXES = (
    "ОСЕМЕН", "ОТЕЛ", "РОДИВ", "ЗАПУСК", "ВЫБЫТ", "БЫК", "DISPOS", "INSEM", "CALV", "BIRTH", "BORN", "DRY",
)


def _rewind(file_obj: Any) -> None:
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass


def _find_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {str(c).strip().upper(): c for c in df.columns}
    for x in cands:
        k = str(x).strip().upper()
        if k in cols:
            return cols[k]
    return None


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.normalize()


def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        return s.replace(".0", "")
    return s


def _norm_sex(x: Any) -> Optional[str]:
    if x is None:
        return None
    v = str(x).strip().upper().replace("Ё", "Е")
    if v in {"", "NAN", "NONE", "NULL", "0", "0.0"}:
        return None
    if v in {"F", "Ж"} or "ТЕЛ" in v or "ТЁЛ" in v or "HEIF" in v or "FEMALE" in v:
        return "F"
    if v in {"M", "М"} or "БЫЧ" in v or "BULL" in v or "MALE" in v:
        return "M"
    return None


def _norm_event_type(x: Any) -> str:
    if x is None:
        return ""
    v = str(x).strip().upper().replace("Ё", "Е")
    if "ОТЕЛ" in v or "CALV" in v:
        return "ОТЕЛ"
    if "РОЖ" in v or "BORN" in v or "BIRTH" in v:
        return "РОЖДЕН"
    return v


def _fallback_calvings(df_raw: pd.DataFrame) -> pd.DataFrame:
    mother_col = _find_col(df_raw, "DREG1", "DREG", "REG", "MOTHER_REG", "MOTHER")
    date_col = _find_col(df_raw, "DATE", "EVENT_DATE", "ARDAT", "CARX")
    ev_col = _find_col(df_raw, "EVENT", "EVENT_TYPE", "EVENTTYPE")
    sex_col = _find_col(df_raw, "GNDR", "GENDER", "SEX")
    lact_col = _find_col(df_raw, "LACT", "LACTATION")

    calf_cols = []
    for k in ("CALF1", "CALF2", "CALF3", "CALF4", "CALF5"):
        c = _find_col(df_raw, k)
        if c:
            calf_cols.append(c)

    if mother_col is None or date_col is None:
        raise ValueError("Не нашёл колонки матери/даты в файле отёлов (нужны DREG1/DATE или аналоги).")

    dts = _to_dt(df_raw[date_col])
    ev = df_raw[ev_col].map(_norm_event_type) if ev_col else "ОТЕЛ"
    mother = df_raw[mother_col].map(_norm_id)
    lact = pd.to_numeric(df_raw[lact_col], errors="coerce") if lact_col else pd.Series([pd.NA] * len(df_raw))

    out_rows: list[dict[str, Any]] = []
    for i in range(len(df_raw)):
        if pd.isna(dts.iloc[i]):
            continue
        mr = mother.iloc[i]
        if not mr:
            continue
        out_rows.append(
            {
                "reg": mr,
                "mother_reg": "",
                "birth_date": pd.NaT,
                "sex": None,
                "event_type": ev.iloc[i] if isinstance(ev, pd.Series) else "ОТЕЛ",
                "event_date": dts.iloc[i],
                "lact": lact.iloc[i],
            }
        )

    if calf_cols:
        sx = df_raw[sex_col].map(_norm_sex) if sex_col else None
        for i in range(len(df_raw)):
            dt = dts.iloc[i]
            if pd.isna(dt):
                continue
            mr = mother.iloc[i]
            if not mr:
                continue
            for cc in calf_cols:
                calf = _norm_id(df_raw[cc].iloc[i])
                if not calf or calf in {"0", "-"}:
                    continue
                out_rows.append(
                    {
                        "reg": calf,
                        "mother_reg": mr,
                        "birth_date": dt,
                        "sex": (sx.iloc[i] if sx is not None else None),
                        "event_type": "РОЖДЕН",
                        "event_date": dt,
                        "lact": pd.NA,
                    }
                )

    return pd.DataFrame(out_rows)


def _fallback_inseminations(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    lact_c = _find_col(df_raw, "LACT", "LACTATION")
    dim_c = _find_col(df_raw, "DIM", "DIM_AGE", "DAYS", "ВОЗРАСТ")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    bull_c = _find_col(df_raw, "REMARK", "BULL", "B", "BULL_CODE", "БЫК")
    res_c = _find_col(df_raw, "R", "RESULT", "RES", "RESULT ")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле осеменений.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "lact": pd.to_numeric(df_raw[lact_c], errors="coerce") if lact_c else 0,
            "dim_age": pd.to_numeric(df_raw[dim_c], errors="coerce") if dim_c else pd.NA,
            "event_date": _to_dt(df_raw[date_c]),
            "bull": df_raw[bull_c].map(_norm_id) if bull_c else "",
            "result": df_raw[res_c].astype(str).str.strip() if res_c else "",
        }
    )


def _fallback_disposals(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    reason_c = _find_col(df_raw, "REMARK", "DISPOSAL_REASON", "REM", "ПРИЧИНА ВЫБЫТИЯ")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле выбытия.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "event_date": _to_dt(df_raw[date_c]),
            "disposal_reason": df_raw[reason_c].astype(str).str.strip() if reason_c else "",
        }
    )


def _fallback_dryoff(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    dim_c = _find_col(df_raw, "DIM", "ВОЗРАСТ", "DIM_AGE", "DAYS")
    reason_c = _find_col(df_raw, "CARX", "ПРИЧИНА ВЫБЫТИЯ", "REASON", "REM", "REMARK")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле запусков.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "dim": pd.to_numeric(df_raw[dim_c], errors="coerce") if dim_c else pd.NA,
            "event_date": _to_dt(df_raw[date_c]),
            "move_reason": df_raw[reason_c].astype(str).str.strip() if reason_c else "",
        }
    )


def _detect_kind(filename: str) -> Optional[str]:
    n = filename.upper().replace("Ё", "Е")
    if any(x in n for x in ("ОСЕМЕН", "INSEM")):
        return "ins"
    if any(x in n for x in ("ОТЕЛ", "ОТЁЛ", "РОДИВ", "CALV", "BIRTH", "BORN")):
        return "calv"
    if any(x in n for x in ("ЗАПУСК", "DRY")):
        return "dry"
    if any(x in n for x in ("ВЫБЫТИ", "DISPOS")):
        return "disp"
    if any(x in n for x in ("БЫК", "BULL")):
        return "bulls"
    return None


def _extract_farm_name(filename: str, kind: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", filename, flags=re.IGNORECASE)
    tokens = re.findall(r"[0-9A-ZА-ЯЁ]+", stem.upper().replace("Ё", "Е"))

    out: list[str] = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if any(t.startswith(pref) for pref in _STOP_PREFIXES):
            continue
        if t.isdigit() and len(t) >= 4:
            continue
        if len(t) <= 1:
            continue
        out.append(t)

    name = " ".join(out).strip()
    return name or "ХОЗЯЙСТВО_1"


def _group_files(files: list[Any]) -> tuple[dict[str, FarmUploadBundle], pd.DataFrame]:
    bundles: dict[str, FarmUploadBundle] = {}
    rows: list[dict[str, str]] = []

    for f in files:
        kind = _detect_kind(f.name)
        if kind is None:
            rows.append({"Файл": f.name, "Тип": "не распознан", "Подразделение": "—", "Статус": "пропущен"})
            continue

        farm = _extract_farm_name(f.name, kind)
        b = bundles.setdefault(farm, FarmUploadBundle(farm_name=farm))

        status = "ok"
        if kind == "calv":
            if b.calv is not None:
                status = "заменён (последний файл)"
            b.calv = f
        elif kind == "ins":
            if b.ins is not None:
                status = "заменён (последний файл)"
            b.ins = f
        elif kind == "dry":
            if b.dry is not None:
                status = "заменён (последний файл)"
            b.dry = f
        elif kind == "disp":
            if b.disp is not None:
                status = "заменён (последний файл)"
            b.disp = f
        else:
            b.bulls.append(f)

        rows.append({"Файл": f.name, "Тип": kind, "Подразделение": farm, "Статус": status})

    return bundles, pd.DataFrame(rows, columns=["Файл", "Тип", "Подразделение", "Статус"])


def _prepare_tables(bundle: FarmUploadBundle) -> dict[str, pd.DataFrame]:
    if bundle.calv is None or bundle.ins is None or bundle.dry is None or bundle.disp is None:
        raise ValueError("Нужны 4 файла: отёлы, осеменения, запуски, выбытие.")

    _rewind(bundle.calv)
    try:
        calv_df = read_calvings_excel(bundle.calv, include_meta=True)
    except Exception:
        _rewind(bundle.calv)
        calv_df = _fallback_calvings(pd.read_excel(bundle.calv))

    calv_df = calv_df.copy()
    for c in ("reg", "mother_reg", "birth_date", "sex", "event_type", "event_date", "__farm", "__subdivision"):
        if c not in calv_df.columns:
            calv_df[c] = pd.NA
    calv_df["reg"] = calv_df["reg"].map(_norm_id)
    calv_df["mother_reg"] = calv_df["mother_reg"].map(_norm_id)
    calv_df["birth_date"] = pd.to_datetime(calv_df["birth_date"], errors="coerce")
    calv_df["event_date"] = pd.to_datetime(calv_df["event_date"], errors="coerce")
    calv_df["sex"] = calv_df["sex"].map(_norm_sex)
    calv_df["event_type"] = calv_df["event_type"].map(_norm_event_type)

    _rewind(bundle.ins)
    try:
        ins_df = clean_inseminations(read_inseminations_excel(bundle.ins, include_meta=True))
    except Exception:
        _rewind(bundle.ins)
        ins_df = _fallback_inseminations(pd.read_excel(bundle.ins))

    ins_df = ins_df.copy()
    for c in ("reg", "lact", "dim_age", "event_date", "bull", "result", "__farm", "__subdivision"):
        if c not in ins_df.columns:
            ins_df[c] = pd.NA
    ins_df["reg"] = ins_df["reg"].map(_norm_id)
    ins_df["lact"] = pd.to_numeric(ins_df["lact"], errors="coerce")
    ins_df["dim_age"] = pd.to_numeric(ins_df["dim_age"], errors="coerce")
    ins_df["event_date"] = pd.to_datetime(ins_df["event_date"], errors="coerce")
    ins_df["bull"] = ins_df["bull"].map(_norm_id)
    ins_df["result"] = ins_df["result"].astype(str).str.strip()

    _rewind(bundle.dry)
    try:
        dry_df = read_dryoff_excel(bundle.dry, include_meta=True)
    except Exception:
        _rewind(bundle.dry)
        dry_df = _fallback_dryoff(pd.read_excel(bundle.dry))

    dry_df = dry_df.copy()
    for c in ("reg", "dim", "event_date", "disposal_reason", "__farm", "__subdivision"):
        if c not in dry_df.columns:
            dry_df[c] = pd.NA
    dry_df["reg"] = dry_df["reg"].map(_norm_id)
    dry_df["dim"] = pd.to_numeric(dry_df["dim"], errors="coerce")
    dry_df["event_date"] = pd.to_datetime(dry_df["event_date"], errors="coerce")
    dry_df["move_reason"] = dry_df["disposal_reason"].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()

    _rewind(bundle.disp)
    try:
        disp_df = read_disposals_excel(bundle.disp, include_meta=True)
    except Exception:
        _rewind(bundle.disp)
        disp_df = _fallback_disposals(pd.read_excel(bundle.disp))

    disp_df = disp_df.copy()
    for c in ("reg", "event_date", "disposal_reason", "__farm", "__subdivision"):
        if c not in disp_df.columns:
            disp_df[c] = pd.NA
    disp_df["reg"] = disp_df["reg"].map(_norm_id)
    disp_df["event_date"] = pd.to_datetime(disp_df["event_date"], errors="coerce")

    bulls_frames: list[pd.DataFrame] = []
    for bf in bundle.bulls:
        try:
            _rewind(bf)
            bdf = read_bulls_txt(bf)
            if not bdf.empty:
                for c in ("bull_code", "bull_type"):
                    if c not in bdf.columns:
                        bdf[c] = pd.NA
                bdf = bdf[["bull_code", "bull_type"]].copy()
                bdf["bull_code"] = bdf["bull_code"].map(_norm_id)
                bdf["bull_type"] = bdf["bull_type"].astype(str).str.strip()
                bulls_frames.append(bdf)
        except Exception:
            continue

    bulls_df = (
        pd.concat(bulls_frames, ignore_index=True).drop_duplicates(subset=["bull_code"], keep="first")
        if bulls_frames
        else pd.DataFrame(columns=["bull_code", "bull_type"])
    )

    return {
        "calv": calv_df[["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date", "__farm", "__subdivision"]].copy(),
        "ins": ins_df[["reg", "lact", "dim_age", "event_date", "bull", "result", "__farm", "__subdivision"]].copy(),
        "dry": dry_df[["reg", "dim", "event_date", "move_reason", "__farm", "__subdivision"]].copy(),
        "disp": disp_df[["reg", "event_date", "disposal_reason", "__farm", "__subdivision"]].copy(),
        "bulls": bulls_df[["bull_code", "bull_type"]].copy(),
    }


def _ensure_farm_tables() -> None:
    ddl = [
        f"""
        CREATE TABLE IF NOT EXISTS {TAB3_TABLES['calv']} (
            farm_name TEXT NOT NULL,
            reg TEXT,
            mother_reg TEXT,
            birth_date DATE,
            sex TEXT,
            event_type TEXT,
            event_date DATE
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {TAB3_TABLES['ins']} (
            farm_name TEXT NOT NULL,
            reg TEXT,
            lact INTEGER,
            dim_age INTEGER,
            event_date DATE,
            bull TEXT,
            result TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {TAB3_TABLES['dry']} (
            farm_name TEXT NOT NULL,
            reg TEXT,
            dim INTEGER,
            event_date DATE,
            move_reason TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {TAB3_TABLES['disp']} (
            farm_name TEXT NOT NULL,
            reg TEXT,
            event_date DATE,
            disposal_reason TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {TAB3_TABLES['bulls']} (
            farm_name TEXT NOT NULL,
            bull_code TEXT,
            bull_type TEXT
        );
        """,
        f"CREATE INDEX IF NOT EXISTS idx_{TAB3_TABLES['calv']}_farm ON {TAB3_TABLES['calv']}(farm_name);",
        f"CREATE INDEX IF NOT EXISTS idx_{TAB3_TABLES['ins']}_farm ON {TAB3_TABLES['ins']}(farm_name);",
        f"CREATE INDEX IF NOT EXISTS idx_{TAB3_TABLES['dry']}_farm ON {TAB3_TABLES['dry']}(farm_name);",
        f"CREATE INDEX IF NOT EXISTS idx_{TAB3_TABLES['disp']}_farm ON {TAB3_TABLES['disp']}(farm_name);",
        f"CREATE INDEX IF NOT EXISTS idx_{TAB3_TABLES['bulls']}_farm ON {TAB3_TABLES['bulls']}(farm_name);",
    ]

    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))
        conn.execute(text(f"ALTER TABLE {TAB3_TABLES['dry']} ADD COLUMN IF NOT EXISTS move_reason TEXT"))


def _ensure_forecast_cache_table() -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TAB3_CACHE_TABLE} (
        entity_type TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        target_month DATE NOT NULL,
        data_signature TEXT NOT NULL,
        params_hash TEXT NOT NULL,
        monthly_json JSONB NOT NULL,
        info_json JSONB NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (entity_type, entity_name, target_month, data_signature, params_hash)
    );
    """
    idx = f"""
    CREATE INDEX IF NOT EXISTS idx_{TAB3_CACHE_TABLE}_entity_time
    ON {TAB3_CACHE_TABLE}(entity_type, entity_name, updated_at DESC);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(idx))


def _ensure_subdivision_map_table() -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TAB3_MAP_TABLE} (
        subdivision_name TEXT PRIMARY KEY,
        farm_name TEXT NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    backfill = f"""
    WITH subs AS (
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['calv']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['ins']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['dry']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['disp']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['bulls']}
    )
    INSERT INTO {TAB3_MAP_TABLE}(subdivision_name, farm_name, updated_at)
    SELECT s.subdivision_name, s.subdivision_name, NOW()
    FROM subs s
    WHERE COALESCE(s.subdivision_name, '') <> ''
    ON CONFLICT (subdivision_name) DO NOTHING;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(backfill))


def _upsert_subdivision_mapping(subdivision_name: str, farm_name: str | None = None, overwrite: bool = False) -> None:
    _ensure_subdivision_map_table()
    subdivision = (subdivision_name or "").strip()
    farm = (farm_name or subdivision).strip()
    if not subdivision or not farm:
        return
    if overwrite:
        sql = f"""
        INSERT INTO {TAB3_MAP_TABLE}(subdivision_name, farm_name, updated_at)
        VALUES (:s, :f, NOW())
        ON CONFLICT (subdivision_name)
        DO UPDATE SET farm_name = EXCLUDED.farm_name, updated_at = NOW();
        """
    else:
        sql = f"""
        INSERT INTO {TAB3_MAP_TABLE}(subdivision_name, farm_name, updated_at)
        VALUES (:s, :f, NOW())
        ON CONFLICT (subdivision_name) DO NOTHING;
        """
    with engine.begin() as conn:
        conn.execute(text(sql), {"s": subdivision, "f": farm})


def _json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _params_hash(params: dict) -> str:
    payload = {
        "__cache_schema_version__": TAB3_CACHE_SCHEMA_VERSION,
        "params": params or {},
    }
    return _json_hash(payload)


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _farm_param_overrides_state() -> dict[str, dict]:
    raw = st.session_state.get("tab3_farm_param_overrides")
    if not isinstance(raw, dict):
        raw = {}
    st.session_state["tab3_farm_param_overrides"] = raw
    return raw


def _is_admin_mode() -> bool:
    return bool(st.session_state.get("is_admin", False))


def _build_farm_params(base_params: dict, farm_override: dict | None) -> dict:
    params = deepcopy(base_params or {})
                                                                                     
    params.pop("SEMEN_USAGE_SHARES", None)
    params.pop("semen_usage", None)
    params.pop("SEMEN_SEX_RATIOS", None)
    params.pop("semen_sex_ratios", None)
    if isinstance(farm_override, dict) and farm_override:
        _deep_merge(params, farm_override)
    params.pop("HERD_CAPACITY", None)
    params.pop("herd_capacity", None)
    params["DISABLE_CAPACITY"] = True
    params["APPLY_CAPACITY"] = False
    return params


def _farm_param_editor_block(farms: list[str], base_params: dict) -> None:
    if not _is_admin_mode():
        return
    with st.expander("Параметры прогноза по хозяйству", expanded=False):
        if not farms:
            return

        all_overrides = _farm_param_overrides_state()
        farm_name = st.selectbox("Хозяйство для настройки параметров", farms, index=0, key="tab3_param_farm_select")
        farm_override = deepcopy(all_overrides.get(farm_name, {}))

        def _get_nested(d: dict | None, path: list[Any], default: Any) -> Any:
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

        def _pick(path: list[Any], default: Any) -> Any:
            ov_v = _get_nested(farm_override, path, None)
            if ov_v is not None:
                return ov_v
            return _get_nested(base_params, path, default)

        def _set_nested(d: dict, path: list[Any], value: Any) -> None:
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

        st.markdown("**Сроки**")
        c1, c2 = st.columns(2)
        with c1:
            gest = st.number_input(
                "Длительность стельности (дн.)",
                min_value=200,
                max_value=310,
                value=int(round(float(_pick(["GESTATION_DAYS"], 272) or 272))),
                step=1,
                key=f"tab3_param_gest_{farm_name}",
            )
        with c2:
            dry = st.number_input(
                "Длительность сухостоя (дн.)",
                min_value=20,
                max_value=120,
                value=int(round(float(_pick(["DRY_DAYS"], 53) or 53))),
                step=1,
                key=f"tab3_param_dry_{farm_name}",
            )

        st.markdown("**Стельность**")
        c3, c4 = st.columns(2)
        with c3:
            avg_cow_dim_global = st.number_input(
                "Коровы: средний DIM наступления стельности",
                min_value=40.0,
                max_value=250.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_cow_dim_global"], 104.0)),
                step=1.0,
                key=f"tab3_param_cp_cow_global_{farm_name}",
            )
        with c4:
            avg_heifer_age_days = st.number_input(
                "Тёлки: средний возраст наступления стельности (дн.)",
                min_value=250.0,
                max_value=700.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_heifer_age_days"], 400.0)),
                step=1.0,
                key=f"tab3_param_cp_heifer_age_{farm_name}",
            )

        c5, c6 = st.columns(2)
        with c5:
            cp_l1 = st.number_input(
                "Коровы: DIM наступления стельности — 1-я лактация",
                min_value=40.0,
                max_value=250.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 1], avg_cow_dim_global)),
                step=1.0,
                key=f"tab3_param_cp_l1_{farm_name}",
            )
            cp_l2 = st.number_input(
                "Коровы: DIM наступления стельности — 2-я лактация",
                min_value=40.0,
                max_value=250.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 2], avg_cow_dim_global)),
                step=1.0,
                key=f"tab3_param_cp_l2_{farm_name}",
            )
        with c6:
            cp_l3 = st.number_input(
                "Коровы: DIM наступления стельности — 3-я лактация",
                min_value=40.0,
                max_value=250.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 3], avg_cow_dim_global)),
                step=1.0,
                key=f"tab3_param_cp_l3_{farm_name}",
            )
            cp_l4 = st.number_input(
                "Коровы: DIM наступления стельности — 4+ лактация",
                min_value=40.0,
                max_value=250.0,
                value=float(_pick(["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 4], avg_cow_dim_global)),
                step=1.0,
                key=f"tab3_param_cp_l4_{farm_name}",
            )

        st.markdown("**Осеменения**")
        c7, c8 = st.columns(2)
        with c7:
            cow_spc = st.number_input(
                "Коровы: осеменений до стельности (P), среднее",
                min_value=1.0,
                max_value=5.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_services_per_conception"], 2.0)),
                step=0.01,
                key=f"tab3_param_ins_cow_spc_{farm_name}",
            )
            cow_interval = st.number_input(
                "Коровы: интервал между осеменениями (дн.)",
                min_value=14.0,
                max_value=90.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_ai_interval_days"], 45.0)),
                step=0.5,
                key=f"tab3_param_ins_cow_interval_{farm_name}",
            )
        with c8:
            heif_spc = st.number_input(
                "Тёлки: осеменений до стельности (P), среднее",
                min_value=1.0,
                max_value=5.0,
                value=float(_pick(["INSEMINATION_PARAMS", "heifer_services_per_conception"], 2.0)),
                step=0.01,
                key=f"tab3_param_ins_heif_spc_{farm_name}",
            )
            heif_interval = st.number_input(
                "Тёлки: интервал между осеменениями (дн.)",
                min_value=14.0,
                max_value=90.0,
                value=float(_pick(["INSEMINATION_PARAMS", "heifer_ai_interval_days"], 25.0)),
                step=0.5,
                key=f"tab3_param_ins_heif_interval_{farm_name}",
            )

        heif_first_ai = st.number_input(
            "Тёлки: возраст первого осеменения (дн.)",
            min_value=250.0,
            max_value=700.0,
            value=float(_pick(["INSEMINATION_PARAMS", "heifer_first_ai_age_days"], 380.0)),
            step=1.0,
            key=f"tab3_param_ins_heif_first_ai_{farm_name}",
        )

        c9, c10 = st.columns(2)
        with c9:
            cow_first_ai_l1 = st.number_input(
                "Коровы: DIM первого осеменения — 1-я лактация",
                min_value=30.0,
                max_value=220.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 1], 72.0)),
                step=1.0,
                key=f"tab3_param_ins_first_l1_{farm_name}",
            )
            cow_first_ai_l2 = st.number_input(
                "Коровы: DIM первого осеменения — 2-я лактация",
                min_value=30.0,
                max_value=220.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 2], 72.0)),
                step=1.0,
                key=f"tab3_param_ins_first_l2_{farm_name}",
            )
        with c10:
            cow_first_ai_l3 = st.number_input(
                "Коровы: DIM первого осеменения — 3-я лактация",
                min_value=30.0,
                max_value=220.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 3], 72.0)),
                step=1.0,
                key=f"tab3_param_ins_first_l3_{farm_name}",
            )
            cow_first_ai_l4 = st.number_input(
                "Коровы: DIM первого осеменения — 4+ лактация",
                min_value=30.0,
                max_value=220.0,
                value=float(_pick(["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 4], 72.0)),
                step=1.0,
                key=f"tab3_param_ins_first_l4_{farm_name}",
            )

        st.markdown("**Выбытие**")
        annual_disposal = st.number_input(
            "Среднегодовой процент выбытия коров (доля)",
            min_value=0.0,
            max_value=0.5,
            value=float(_pick(["ANNUAL_DISPOSAL_RATE"], 0.0957)),
            step=0.001,
            format="%.4f",
            key=f"tab3_param_annual_disp_{farm_name}",
        )

        c11, c12 = st.columns(2)
        with c11:
            disp_median_l1 = st.number_input(
                "Выбытие: DIM медиана — 1-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 1, "median_dim"], 111.0)),
                step=1.0,
                key=f"tab3_param_disp_median_l1_{farm_name}",
            )
            disp_median_l2 = st.number_input(
                "Выбытие: DIM медиана — 2-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 2, "median_dim"], 226.0)),
                step=1.0,
                key=f"tab3_param_disp_median_l2_{farm_name}",
            )
            disp_median_l3 = st.number_input(
                "Выбытие: DIM медиана — 3-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 3, "median_dim"], 194.0)),
                step=1.0,
                key=f"tab3_param_disp_median_l3_{farm_name}",
            )
            disp_median_l4 = st.number_input(
                "Выбытие: DIM медиана — 4+ лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 4, "median_dim"], 73.0)),
                step=1.0,
                key=f"tab3_param_disp_median_l4_{farm_name}",
            )
        with c12:
            disp_mean_l1 = st.number_input(
                "Выбытие: DIM среднее — 1-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 1, "mean_dim"], 160.0)),
                step=1.0,
                key=f"tab3_param_disp_mean_l1_{farm_name}",
            )
            disp_mean_l2 = st.number_input(
                "Выбытие: DIM среднее — 2-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 2, "mean_dim"], 235.0)),
                step=1.0,
                key=f"tab3_param_disp_mean_l2_{farm_name}",
            )
            disp_mean_l3 = st.number_input(
                "Выбытие: DIM среднее — 3-я лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 3, "mean_dim"], 192.0)),
                step=1.0,
                key=f"tab3_param_disp_mean_l3_{farm_name}",
            )
            disp_mean_l4 = st.number_input(
                "Выбытие: DIM среднее — 4+ лактация",
                min_value=10.0,
                max_value=500.0,
                value=float(_pick(["DISPOSAL_PARAMS", "by_lact", 4, "mean_dim"], 127.0)),
                step=1.0,
                key=f"tab3_param_disp_mean_l4_{farm_name}",
            )

        a1, a2 = st.columns(2)
        if a1.button("Сохранить параметры хозяйства", use_container_width=True, key=f"tab3_param_save_{farm_name}"):
            new_override: dict[str, Any] = {}
            _set_nested(new_override, ["GESTATION_DAYS"], int(gest))
            _set_nested(new_override, ["DRY_DAYS"], int(dry))

            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_global"], float(avg_cow_dim_global))
            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_heifer_age_days"], float(avg_heifer_age_days))
            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 1], float(cp_l1))
            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 2], float(cp_l2))
            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 3], float(cp_l3))
            _set_nested(new_override, ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", 4], float(cp_l4))

            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_services_per_conception"], float(cow_spc))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_ai_interval_days"], float(cow_interval))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "heifer_services_per_conception"], float(heif_spc))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "heifer_ai_interval_days"], float(heif_interval))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "heifer_first_ai_age_days"], float(heif_first_ai))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 1], float(cow_first_ai_l1))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 2], float(cow_first_ai_l2))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 3], float(cow_first_ai_l3))
            _set_nested(new_override, ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", 4], float(cow_first_ai_l4))

            _set_nested(new_override, ["ANNUAL_DISPOSAL_RATE"], float(annual_disposal))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 1, "median_dim"], float(disp_median_l1))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 2, "median_dim"], float(disp_median_l2))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 3, "median_dim"], float(disp_median_l3))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 4, "median_dim"], float(disp_median_l4))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 1, "mean_dim"], float(disp_mean_l1))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 2, "mean_dim"], float(disp_mean_l2))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 3, "mean_dim"], float(disp_mean_l3))
            _set_nested(new_override, ["DISPOSAL_PARAMS", "by_lact", 4, "mean_dim"], float(disp_mean_l4))

            all_overrides[farm_name] = new_override
            st.session_state["tab3_farm_param_overrides"] = all_overrides
            _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
            st.success(f"Параметры для «{farm_name}» сохранены.")
            st.rerun()

        if a2.button("Сбросить параметры хозяйства", use_container_width=True, key=f"tab3_param_reset_{farm_name}"):
            if farm_name in all_overrides:
                all_overrides.pop(farm_name, None)
                st.session_state["tab3_farm_param_overrides"] = all_overrides
                _clear_forecast_cache(entity_type="farm", entity_name=farm_name)
                st.success(f"Параметры для «{farm_name}» сброшены.")
                st.rerun()


def _subdivision_signature_from_db(subdivision_name: str) -> str:
    _ensure_farm_tables()
    sql = f"""
    SELECT
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['calv']} WHERE farm_name = :name), 0) AS n_calv,
      COALESCE((SELECT MAX(event_date)::text FROM {TAB3_TABLES['calv']} WHERE farm_name = :name), '') AS mx_calv,
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['ins']} WHERE farm_name = :name), 0) AS n_ins,
      COALESCE((SELECT MAX(event_date)::text FROM {TAB3_TABLES['ins']} WHERE farm_name = :name), '') AS mx_ins,
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['dry']} WHERE farm_name = :name), 0) AS n_dry,
      COALESCE((SELECT MAX(event_date)::text FROM {TAB3_TABLES['dry']} WHERE farm_name = :name), '') AS mx_dry,
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['dry']} WHERE farm_name = :name AND COALESCE(move_reason, '') <> ''), 0) AS n_dry_reason,
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['disp']} WHERE farm_name = :name), 0) AS n_disp,
      COALESCE((SELECT MAX(event_date)::text FROM {TAB3_TABLES['disp']} WHERE farm_name = :name), '') AS mx_disp,
      COALESCE((SELECT COUNT(*) FROM {TAB3_TABLES['bulls']} WHERE farm_name = :name), 0) AS n_bulls
    ;
    """
    df = pd.read_sql(text(sql), con=engine, params={"name": subdivision_name})
    if df.empty:
        return "empty"
    r = df.iloc[0].to_dict()
    return (
        f"{r.get('n_calv', 0)}|{r.get('mx_calv', '')}|"
        f"{r.get('n_ins', 0)}|{r.get('mx_ins', '')}|"
        f"{r.get('n_dry', 0)}|{r.get('mx_dry', '')}|"
        f"{r.get('n_dry_reason', 0)}|"
        f"{r.get('n_disp', 0)}|{r.get('mx_disp', '')}|"
        f"{r.get('n_bulls', 0)}"
    )


def _farm_signature_from_db(farm_name: str) -> str:
    parts: list[str] = []
    for sub in _subdivisions_for_farm(farm_name, ready_only=False):
        parts.append(f"{sub}:{_subdivision_signature_from_db(sub)}")
    if not parts:
        return "empty"
    return _json_hash(parts)


def _merge_tables(table_list: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    keys = ("calv", "ins", "dry", "disp", "bulls")
    out: dict[str, pd.DataFrame] = {}
    for k in keys:
        frames = [
            t.get(k) for t in table_list
            if isinstance(t, dict) and isinstance(t.get(k), pd.DataFrame) and not t.get(k).empty
        ]
        if not frames:
            out[k] = pd.DataFrame()
        else:
            merged = pd.concat(frames, ignore_index=True)
            if k == "bulls":
                merged = merged.drop_duplicates(subset=["bull_code"], keep="first")
            out[k] = merged
    return out


def _norm_reg_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _namespace_reg(subdivision: str, reg: Any) -> str:
    r = _norm_reg_value(reg)
    if not r:
        return ""
    return f"{subdivision}::{r}"


def _namespace_tables_for_subdivision(subdivision: str, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    calv = tables.get("calv")
    if isinstance(calv, pd.DataFrame) and not calv.empty:
        d = calv.copy()
        if "reg" in d.columns:
            d["reg"] = d["reg"].map(lambda x: _namespace_reg(subdivision, x))
        if "mother_reg" in d.columns:
            d["mother_reg"] = d["mother_reg"].map(lambda x: _namespace_reg(subdivision, x))
        out["calv"] = d
    else:
        out["calv"] = pd.DataFrame()

    ins = tables.get("ins")
    if isinstance(ins, pd.DataFrame) and not ins.empty:
        d = ins.copy()
        if "reg" in d.columns:
            d["reg"] = d["reg"].map(lambda x: _namespace_reg(subdivision, x))
        out["ins"] = d
    else:
        out["ins"] = pd.DataFrame()

    dry = tables.get("dry")
    if isinstance(dry, pd.DataFrame) and not dry.empty:
        d = dry.copy()
        if "reg" in d.columns:
            d["reg"] = d["reg"].map(lambda x: _namespace_reg(subdivision, x))
        out["dry"] = d
    else:
        out["dry"] = pd.DataFrame()

    disp = tables.get("disp")
    if isinstance(disp, pd.DataFrame) and not disp.empty:
        d = disp.copy()
        if "reg" in d.columns:
            d["reg"] = d["reg"].map(lambda x: _namespace_reg(subdivision, x))
        out["disp"] = d
    else:
        out["disp"] = pd.DataFrame()

    bulls = tables.get("bulls")
    out["bulls"] = bulls.copy() if isinstance(bulls, pd.DataFrame) else pd.DataFrame()
    return out


def _load_farm_merged_tables_from_db(farm_name: str) -> dict[str, pd.DataFrame]:
    subdivisions = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subdivisions:
        return {"calv": pd.DataFrame(), "ins": pd.DataFrame(), "dry": pd.DataFrame(), "disp": pd.DataFrame(), "bulls": pd.DataFrame()}
    all_tables = [_namespace_tables_for_subdivision(sub, _load_farm_tables_from_db(sub)) for sub in subdivisions]
    return _merge_tables(all_tables)


def _load_forecast_cache(
    entity_type: str,
    entity_name: str,
    target_month_end: date,
    data_signature: str,
    params_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    _ensure_forecast_cache_table()
    sql = f"""
    SELECT monthly_json, info_json
    FROM {TAB3_CACHE_TABLE}
    WHERE entity_type = :etype
      AND entity_name = :ename
      AND target_month = :tmonth
      AND data_signature = :dsig
      AND params_hash = :ph
    LIMIT 1;
    """
    df = pd.read_sql(
        text(sql),
        con=engine,
        params={
            "etype": entity_type,
            "ename": entity_name,
            "tmonth": target_month_end,
            "dsig": data_signature,
            "ph": params_hash,
        },
    )
    if df.empty:
        return None

    monthly_raw = df.loc[0, "monthly_json"]
    info_raw = df.loc[0, "info_json"]

    if isinstance(monthly_raw, str):
        monthly_raw = json.loads(monthly_raw)
    if isinstance(info_raw, str):
        info_raw = json.loads(info_raw)

    monthly_df = pd.DataFrame(monthly_raw or [])
    info = info_raw or {}
    return monthly_df, info


def _save_forecast_cache(
    entity_type: str,
    entity_name: str,
    target_month_end: date,
    data_signature: str,
    params_hash: str,
    monthly_df: pd.DataFrame,
    info: dict[str, Any],
) -> None:
    _ensure_forecast_cache_table()
    monthly_json = json.dumps(monthly_df.to_dict(orient="records"), ensure_ascii=False, default=str)
    info_json = json.dumps(info or {}, ensure_ascii=False, default=str)

    sql = f"""
    INSERT INTO {TAB3_CACHE_TABLE}
      (entity_type, entity_name, target_month, data_signature, params_hash, monthly_json, info_json, updated_at)
    VALUES
      (:etype, :ename, :tmonth, :dsig, :ph, CAST(:mjson AS jsonb), CAST(:ijson AS jsonb), NOW())
    ON CONFLICT (entity_type, entity_name, target_month, data_signature, params_hash)
    DO UPDATE SET
      monthly_json = EXCLUDED.monthly_json,
      info_json = EXCLUDED.info_json,
      updated_at = NOW();
    """
    with engine.begin() as conn:
        conn.execute(
            text(sql),
            {
                "etype": entity_type,
                "ename": entity_name,
                "tmonth": target_month_end,
                "dsig": data_signature,
                "ph": params_hash,
                "mjson": monthly_json,
                "ijson": info_json,
            },
        )


def _clear_forecast_cache(entity_type: str = "subdivision", entity_name: str | None = None) -> None:
    _ensure_forecast_cache_table()
    sql = f"DELETE FROM {TAB3_CACHE_TABLE} WHERE entity_type = :etype"
    params: dict[str, Any] = {"etype": entity_type}
    if entity_name:
        sql += " AND entity_name = :ename"
        params["ename"] = entity_name
    with engine.begin() as conn:
        conn.execute(text(sql), params)


def _save_farm_tables_to_db(farm_name: str, tables: dict[str, pd.DataFrame], replace_farm: bool = True) -> None:
    _ensure_farm_tables()
    _ensure_subdivision_map_table()
    farm = (farm_name or "").strip()
    if not farm:
        raise ValueError("Пустое имя подразделения.")

    if replace_farm:
        with engine.begin() as conn:
            for t in TAB3_TABLES.values():
                conn.execute(text(f"DELETE FROM {t} WHERE farm_name = :farm"), {"farm": farm})

    mapping = {
        "calv": ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"],
        "ins": ["reg", "lact", "dim_age", "event_date", "bull", "result"],
        "dry": ["reg", "dim", "event_date", "move_reason"],
        "disp": ["reg", "event_date", "disposal_reason"],
        "bulls": ["bull_code", "bull_type"],
    }

    for key, cols in mapping.items():
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        dfx = df.copy()
        for c in cols:
            if c not in dfx.columns:
                dfx[c] = pd.NA
        dfx = dfx[cols].copy()

        if "birth_date" in dfx.columns:
            dfx["birth_date"] = pd.to_datetime(dfx["birth_date"], errors="coerce").dt.date
        if "event_date" in dfx.columns:
            dfx["event_date"] = pd.to_datetime(dfx["event_date"], errors="coerce").dt.date

        dfx.insert(0, "farm_name", farm)
        dfx.to_sql(TAB3_TABLES[key], con=engine, if_exists="append", index=False, method="multi", chunksize=2000)

    _upsert_subdivision_mapping(farm, farm_name=farm, overwrite=False)


def _delete_subdivision_everywhere(subdivision_name: str) -> None:
    sub = (subdivision_name or "").strip()
    if not sub:
        return
    with engine.begin() as conn:
        for t in TAB3_TABLES.values():
            conn.execute(text(f"DELETE FROM {t} WHERE farm_name = :sub"), {"sub": sub})
        conn.execute(text(f"DELETE FROM {TAB3_MAP_TABLE} WHERE subdivision_name = :sub"), {"sub": sub})


def _mode_nonempty(values: list[str]) -> str | None:
    if not values:
        return None
    s = pd.Series(values, dtype="string").fillna("").str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    return str(s.value_counts().idxmax())


def _farm_values_in_df(df: pd.DataFrame) -> set[str]:
    if not isinstance(df, pd.DataFrame) or df.empty or "__farm" not in df.columns:
        return set()
    s = df["__farm"].map(_canon_name)
    s = s[s != ""]
    return set(s.astype(str).tolist())


def _farm_match_score(tables: dict[str, pd.DataFrame], farm_name: str) -> int:
    farm_u = _canon_name(farm_name)
    if not farm_u:
        return 0
    score = 0
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty or "__farm" not in df.columns:
            continue
        s = df["__farm"].map(_canon_name)
        score += int((s == farm_u).sum())
    return score


def _infer_bundle_farm_candidates(tables: dict[str, pd.DataFrame]) -> list[str]:
    per_table: list[set[str]] = []
    for key in ("calv", "ins", "dry", "disp"):
        farms = _farm_values_in_df(tables.get(key, pd.DataFrame()))
        if farms:
            per_table.append(farms)

    if not per_table:
        return []

    intersection = set.intersection(*per_table)
    if intersection:
        return sorted(intersection)

    union = set().union(*per_table)
    return sorted(union)


def _choose_bundle_farm(bundle_hint: str, tables: dict[str, pd.DataFrame]) -> str:
    hint = _canon_name(bundle_hint)
    candidates = _infer_bundle_farm_candidates(tables)
    if hint and hint in candidates:
        return hint
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda x: _farm_match_score(tables, x))
    return hint


def _canon_name(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_subdivision_name(x: Any) -> str:
    s = _canon_name(x)
    if not s:
        return ""
                                                                               
                                         
    return s


def _subdivision_display_name(subdivision_name: str, farm_name: str, farm_sub_count: int) -> str:
    _ = farm_name
    _ = farm_sub_count
    return (subdivision_name or "").strip()


def _normalize_bundle_meta(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            out[key] = pd.DataFrame()
            continue
        d = df.copy()
        if "__farm" in d.columns:
            d["__farm"] = d["__farm"].map(_canon_name)
        if "__subdivision" in d.columns:
            d["__subdivision"] = d["__subdivision"].map(_canon_subdivision_name)
        out[key] = d
    return out


def _filter_tables_by_farm(tables: dict[str, pd.DataFrame], farm_name: str) -> dict[str, pd.DataFrame]:
    hint = _canon_name(farm_name)
    if not hint:
        return tables
    out: dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            continue
        d = df.copy()
        if "__farm" in d.columns:
            s = d["__farm"].astype("string").fillna("").str.strip().str.upper().str.replace("Ё", "Е")
            nonempty = s != ""
            if bool(nonempty.any()):
                mask_exact = s == hint
                if bool(mask_exact.any()):
                    d = d.loc[mask_exact].copy()
                else:
                    uniq = sorted(set(s.loc[nonempty].astype(str).tolist()))
                                                                           
                    if len(uniq) == 1:
                        d = d.loc[s == uniq[0]].copy()
                    else:
                        d = d.iloc[0:0].copy()
        out[key] = d
    return out


def _subdivision_names_from_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    seen: dict[str, str] = {}
    presence: dict[str, set[str]] = {}
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty or "__subdivision" not in df.columns:
            continue
        vals = df["__subdivision"].astype("string").fillna("").str.strip()
        vals = vals[vals != ""]
        for v in vals.tolist():
            u = str(v).upper()
            if u not in seen:
                seen[u] = str(v)
            presence.setdefault(u, set()).add(key)

    if not seen:
        return []

                                                                             
                                            
    core = [seen[k] for k in sorted(seen.keys()) if len(presence.get(k, set())) >= 2]
    if core:
        return core
    return [seen[k] for k in sorted(seen.keys())]


def _farm_name_for_subdivision(tables: dict[str, pd.DataFrame], subdivision_name: str, default_farm: str) -> str:
    farm_vals: list[str] = []
    sub_u = (subdivision_name or "").strip().upper()
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        work = df
        if "__subdivision" in work.columns:
            mask = work["__subdivision"].astype("string").fillna("").str.strip().str.upper() == sub_u
            work = work.loc[mask]
        if "__farm" in work.columns and not work.empty:
            vals = work["__farm"].astype("string").fillna("").str.strip()
            vals = vals[vals != ""]
            if not vals.empty:
                farm_vals.extend(vals.tolist())
    return _mode_nonempty(farm_vals) or default_farm


def _tables_for_subdivision(tables: dict[str, pd.DataFrame], subdivision_name: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    sub_u = (subdivision_name or "").strip().upper()
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[key] = pd.DataFrame()
            continue
        work = df.copy()
        if "__subdivision" in work.columns:
            s = work["__subdivision"].astype("string").fillna("").str.strip().str.upper()
            nonempty = s != ""
                                                                         
                                                                               
            if bool(nonempty.any()):
                mask = s == sub_u
                work = work.loc[mask].copy()
        work = work.drop(columns=["__farm", "__subdivision"], errors="ignore")
        out[key] = work
    out["bulls"] = tables.get("bulls", pd.DataFrame()).copy()
    return out


def _save_bundle_tables_to_db(
    bundle_name: str,
    tables: dict[str, pd.DataFrame],
    *,
    replace_subdivision: bool = True,
) -> list[str]:
    tables = _normalize_bundle_meta(tables)
    inferred_farm = _choose_bundle_farm(bundle_name, tables)
    tables = _filter_tables_by_farm(tables, inferred_farm or bundle_name)
    subdivisions = _subdivision_names_from_tables(tables)
    if not subdivisions:
        subdivisions = [(bundle_name or "").strip() or "ПОДРАЗДЕЛЕНИЕ_1"]

    default_farm_vals: list[str] = []
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if isinstance(df, pd.DataFrame) and "__farm" in df.columns:
            vals = df["__farm"].astype("string").fillna("").str.strip()
            vals = vals[vals != ""]
            if not vals.empty:
                default_farm_vals.extend(vals.tolist())
    has_farm_meta = bool(default_farm_vals)
    default_farm = _canon_name(
        _mode_nonempty(default_farm_vals)
        or inferred_farm
        or (bundle_name or "").strip()
        or "ХОЗЯЙСТВО_1"
    )
    if not has_farm_meta:
        default_farm = TAB3_UNASSIGNED_FARM

    updated: list[str] = []
    for subdivision in subdivisions:
        stables = _tables_for_subdivision(tables, subdivision)
        n_rows = sum(int(len(stables.get(k, pd.DataFrame()))) for k in ("calv", "ins", "dry", "disp"))
        if n_rows <= 0:
            continue

        subdivision = _canon_subdivision_name(subdivision)
        if not subdivision:
            continue
        _save_farm_tables_to_db(subdivision, stables, replace_farm=replace_subdivision)
        farm_name = _farm_name_for_subdivision(tables, subdivision, default_farm=default_farm)
        farm_name = _canon_name(farm_name)
        _upsert_subdivision_mapping(subdivision, farm_name=farm_name, overwrite=True)
        _clear_forecast_cache(entity_type="subdivision", entity_name=subdivision)
        updated.append(subdivision)

    if updated:
                                                                                     
        if replace_subdivision and len(updated) >= 2 and default_farm:
            existing_df = pd.read_sql(
                text(
                    f"""
                    SELECT subdivision_name
                    FROM {TAB3_MAP_TABLE}
                    WHERE farm_name = :farm
                    """
                ),
                con=engine,
                params={"farm": default_farm},
            )
            existing = existing_df["subdivision_name"].astype(str).tolist() if not existing_df.empty else []
            updated_u = {_canon_subdivision_name(x) for x in updated}
            stale = [s for s in existing if _canon_subdivision_name(s) not in updated_u]
            for sub in stale:
                _delete_subdivision_everywhere(sub)
                _clear_forecast_cache(entity_type="subdivision", entity_name=sub)

        _clear_forecast_cache(entity_type="farm")
    return updated


def _load_farm_tables_from_db(farm_name: str) -> dict[str, pd.DataFrame]:
    _ensure_farm_tables()
    params = {"farm": farm_name}

    calv = pd.read_sql(
        text(
            f"""
            SELECT reg, mother_reg, birth_date, sex, event_type, event_date
            FROM {TAB3_TABLES['calv']}
            WHERE farm_name = :farm
            """
        ),
        con=engine,
        params=params,
    )
    ins = pd.read_sql(
        text(
            f"""
            SELECT reg, lact, dim_age, event_date, bull, result
            FROM {TAB3_TABLES['ins']}
            WHERE farm_name = :farm
            """
        ),
        con=engine,
        params=params,
    )
    dry = pd.read_sql(
        text(
            f"""
            SELECT reg, dim, event_date, move_reason
            FROM {TAB3_TABLES['dry']}
            WHERE farm_name = :farm
            """
        ),
        con=engine,
        params=params,
    )
    disp = pd.read_sql(
        text(
            f"""
            SELECT reg, event_date, disposal_reason
            FROM {TAB3_TABLES['disp']}
            WHERE farm_name = :farm
            """
        ),
        con=engine,
        params=params,
    )
    bulls = pd.read_sql(
        text(
            f"""
            SELECT bull_code, bull_type
            FROM {TAB3_TABLES['bulls']}
            WHERE farm_name = :farm
            """
        ),
        con=engine,
        params=params,
    )

    return {"calv": calv, "ins": ins, "dry": dry, "disp": disp, "bulls": bulls}


def _subdivision_status_df_from_db() -> pd.DataFrame:
    _ensure_farm_tables()
    _ensure_subdivision_map_table()

    sql = f"""
    WITH subs AS (
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['calv']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['ins']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['dry']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['disp']}
      UNION
      SELECT DISTINCT farm_name AS subdivision_name FROM {TAB3_TABLES['bulls']}
    )
    SELECT
      s.subdivision_name,
      COALESCE(m.farm_name, s.subdivision_name) AS farm_name,
      (SELECT COUNT(*) FROM {TAB3_TABLES['calv']} c WHERE c.farm_name = s.subdivision_name) AS n_calv,
      (SELECT COUNT(*) FROM {TAB3_TABLES['ins']} i WHERE i.farm_name = s.subdivision_name) AS n_ins,
      (SELECT COUNT(*) FROM {TAB3_TABLES['dry']} d WHERE d.farm_name = s.subdivision_name) AS n_dry,
      (SELECT COUNT(*) FROM {TAB3_TABLES['disp']} x WHERE x.farm_name = s.subdivision_name) AS n_disp,
      (SELECT COUNT(*) FROM {TAB3_TABLES['bulls']} b WHERE b.farm_name = s.subdivision_name) AS n_bulls
    FROM subs s
    LEFT JOIN {TAB3_MAP_TABLE} m
      ON m.subdivision_name = s.subdivision_name
    ORDER BY COALESCE(m.farm_name, s.subdivision_name), s.subdivision_name;
    """

    df = pd.read_sql(text(sql), con=engine)
    if df.empty:
        return pd.DataFrame(
            columns=["Хозяйство", "Подразделение", "Статус", "Отёлы", "Осеменения", "Запуски", "Выбытие", "Быки"]
        )

    for c in ("n_calv", "n_ins", "n_dry", "n_disp", "n_bulls"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["status"] = df.apply(
        lambda r: "готово" if (r["n_calv"] > 0 and r["n_ins"] > 0 and r["n_dry"] > 0 and r["n_disp"] > 0) else "неполный набор",
        axis=1,
    )

    return df.rename(
        columns={
            "farm_name": "Хозяйство",
            "subdivision_name": "Подразделение",
            "status": "Статус",
            "n_calv": "Отёлы",
            "n_ins": "Осеменения",
            "n_dry": "Запуски",
            "n_disp": "Выбытие",
            "n_bulls": "Быки",
        }
    )


def _farm_status_df_from_db() -> pd.DataFrame:
    sub = _subdivision_status_df_from_db()
    if sub.empty:
        return pd.DataFrame(
            columns=["Хозяйство", "Статус", "Подразделений", "Готовых подразделений", "Отёлы", "Осеменения", "Запуски", "Выбытие", "Быки"]
        )

    sub = sub.loc[sub["Хозяйство"].astype(str) != TAB3_UNASSIGNED_FARM].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["Хозяйство", "Статус", "Подразделений", "Готовых подразделений", "Отёлы", "Осеменения", "Запуски", "Выбытие", "Быки"]
        )

    agg = (
        sub.groupby("Хозяйство", dropna=False, as_index=False)
        .agg(
            n_sub=("Подразделение", "count"),
            n_ready=("Статус", lambda x: int((x == "готово").sum())),
            n_calv=("Отёлы", "sum"),
            n_ins=("Осеменения", "sum"),
            n_dry=("Запуски", "sum"),
            n_disp=("Выбытие", "sum"),
            n_bulls=("Быки", "sum"),
        )
    )
    agg["Статус"] = agg.apply(lambda r: "готово" if (r["n_ready"] > 0) else "неполный набор", axis=1)
    return agg.rename(
        columns={
            "n_sub": "Подразделений",
            "n_ready": "Готовых подразделений",
            "n_calv": "Отёлы",
            "n_ins": "Осеменения",
            "n_dry": "Запуски",
            "n_disp": "Выбытие",
            "n_bulls": "Быки",
        }
    )[
        ["Хозяйство", "Статус", "Подразделений", "Готовых подразделений", "Отёлы", "Осеменения", "Запуски", "Выбытие", "Быки"]
    ]


def _subdivisions_for_farm(farm_name: str, ready_only: bool = True) -> list[str]:
    sub = _subdivision_status_df_from_db()
    if sub.empty:
        return []
    mask = sub["Хозяйство"].astype(str) == str(farm_name)
    if ready_only:
        mask &= sub["Статус"].astype(str) == "готово"
    return sorted(sub.loc[mask, "Подразделение"].astype(str).tolist())


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
    empty = {k: 0.0 for k in FARM_BACKTEST_TARGETS}
    if not isinstance(calv_df, pd.DataFrame) or calv_df.empty:
        return empty

    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    c = calv_df.copy()
    c["event_date_n"] = pd.to_datetime(c.get("event_date"), errors="coerce").dt.normalize()
    c["event_type_n"] = c.get("event_type", pd.Series(dtype=object)).map(_norm_event_type)
    c["mother_reg_s"] = c.get("mother_reg", pd.Series(dtype=object)).map(_norm_id)
    c["reg_s"] = c.get("reg", pd.Series(dtype=object)).map(_norm_id)
    c["sex_norm"] = c.get("sex", pd.Series(dtype=object)).map(_norm_sex_marker_backtest)

    mask = (
        c["event_date_n"].notna()
        & (c["event_date_n"] >= pd.Timestamp(m_start))
        & (c["event_date_n"] < pd.Timestamp(m_next))
        & (c["event_type_n"] == "РОЖДЕН")
    )
    if as_of_date is not None:
        mask &= c["event_date_n"] <= pd.Timestamp(as_of_date)

    born = c.loc[mask, ["mother_reg_s", "reg_s", "event_date_n", "sex_norm"]].copy()
    if born.empty:
        return empty

    dam = born["mother_reg_s"].replace("", pd.NA).fillna(born["reg_s"].replace("", pd.NA))
    unknown_mask = dam.isna()
    if bool(unknown_mask.any()):
        unknown_ids = [f"__UNK__{i}" for i in range(int(unknown_mask.sum()))]
        dam.loc[unknown_mask] = unknown_ids
    born["dam_key"] = dam.astype(str)

    ev = born[["dam_key", "event_date_n"]].drop_duplicates().rename(columns={"event_date_n": "calv_dt"})
    total_calv = float(len(ev))
    cow_calv = total_calv
    heif_calv = 0.0

    if isinstance(ins_df, pd.DataFrame) and not ins_df.empty:
        ins = ins_df.copy()
        ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(_norm_id)
        ins["event_date_n"] = pd.to_datetime(ins.get("event_date"), errors="coerce").dt.normalize()
        ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
        if as_of_date is not None:
            ins = ins[ins["event_date_n"].notna() & (ins["event_date_n"] <= pd.Timestamp(as_of_date))]
        else:
            ins = ins[ins["event_date_n"].notna()]
        ins = ins[ins["reg_s"] != ""]
        if not ins.empty and not ev.empty:
            left = ev.rename(columns={"dam_key": "reg_s"}).sort_values(["reg_s", "calv_dt"], kind="mergesort")
            right = ins[["reg_s", "event_date_n", "lact_n"]].rename(columns={"event_date_n": "ins_dt"})
            right = right.sort_values(["reg_s", "ins_dt"], kind="mergesort")
            try:
                m = pd.merge_asof(
                    left,
                    right,
                    by="reg_s",
                    left_on="calv_dt",
                    right_on="ins_dt",
                    direction="backward",
                    allow_exact_matches=True,
                )
                lact = pd.to_numeric(m.get("lact_n"), errors="coerce")
                heif_calv = float((lact <= 0).sum())
                cow_calv = float(((lact > 0) | lact.isna()).sum())
            except Exception:
                cow_calv = total_calv
                heif_calv = 0.0

    bulls_known = float((born["sex_norm"] == "M").sum())
    heifers_known = float((born["sex_norm"] == "F").sum())
    total_birth_rows = float(len(born))
    known = bulls_known + heifers_known
    unknown = max(0.0, total_birth_rows - known)
    bull_share_known = (bulls_known / known) if known > 0 else 0.5
    bulls = bulls_known + unknown * bull_share_known
    heifers = max(0.0, total_birth_rows - bulls)
    total_by_sex = bulls + heifers
    bull_pct = (bulls / total_by_sex * 100.0) if total_by_sex > 0 else 0.0
    heif_pct = (heifers / total_by_sex * 100.0) if total_by_sex > 0 else 0.0

    return {
        "Ожидаемый отёл, всего": total_calv,
        "Ожидаемый отёл, из них коров": cow_calv,
        "Ожидаемый отёл, из них нетелей": heif_calv,
        "Ожидаемые бычки (условно)": bulls,
        "Ожидаемые тёлочки (условно)": heifers,
        "Доля бычков среди рождений, %": bull_pct,
        "Доля тёлочек среди рождений, %": heif_pct,
    }


def _is_fact_month_complete_for_subdivision(calv_df: pd.DataFrame, month_end_date: date) -> bool:
    if not isinstance(calv_df, pd.DataFrame) or calv_df.empty:
        return False

    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    c = calv_df.copy()
    c["event_date_n"] = pd.to_datetime(c.get("event_date"), errors="coerce").dt.normalize()
    c["event_type_n"] = c.get("event_type", pd.Series(dtype=object)).map(_norm_event_type)
    c = c[
        c["event_date_n"].notna()
        & (c["event_date_n"] >= pd.Timestamp(m_start))
        & (c["event_date_n"] < pd.Timestamp(m_next))
        & (c["event_type_n"] == "РОЖДЕН")
    ].copy()
    if c.empty:
        return False

    max_dt = c["event_date_n"].max()
    if pd.isna(max_dt):
        return False
    return bool(pd.Timestamp(max_dt).date() >= month_end_date)


def _pred_metric_value_for_backtest(pred_vals: dict, metric_name: str, nmap: dict[str, float]) -> float:
    if metric_name in FARM_PERCENT_TARGETS:
        pred_bull = float(vals_get(pred_vals, "Ожидаемые бычки (условно)", nmap) or 0.0)
        pred_heif = float(vals_get(pred_vals, "Ожидаемые тёлочки (условно)", nmap) or 0.0)
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
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subdivisions = _subdivisions_for_farm(farm_name, ready_only=True)
    if not subdivisions:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "no_ready_subdivisions"}

    tables_by_sub: dict[str, dict[str, pd.DataFrame]] = {}
    base_dates: list[date] = []
    for sub in subdivisions:
        tables = _load_farm_tables_from_db(sub)
        tables_by_sub[sub] = tables
        base_dates.append(latest_data_date(tables))

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
            is_complete = _is_fact_month_complete_for_subdivision(calv_df, target_me)
            if complete_only and not is_complete:
                skipped_sub_months += 1
                continue

            pred_vals = compute_forecast_dynamic_from_tables(
                tables,
                target_me,
                overrides=params,
                as_of_date=as_of_me,
            ) or {}
            nmap = {norm_label(k): v for k, v in pred_vals.items()}
            pred_val = float(_pred_metric_value_for_backtest(pred_vals, metric_name, nmap))
            fact_stats = _actual_birth_stats_month_from_tables(calv_df, ins_df, target_me, as_of_date=None)
            fact_val = float(fact_stats.get(metric_name, 0.0))

            pred_bulls = float(vals_get(pred_vals, "Ожидаемые бычки (условно)", nmap) or 0.0)
            pred_heifers = float(vals_get(pred_vals, "Ожидаемые тёлочки (условно)", nmap) or 0.0)
            fact_bulls = float(fact_stats.get("Ожидаемые бычки (условно)", 0.0))
            fact_heifers = float(fact_stats.get("Ожидаемые тёлочки (условно)", 0.0))

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
            ape_sub = (abs(err_sub) / fact_val * 100.0) if fact_val > 0 else None
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
        ape = (abs(err) / month_fact * 100.0) if month_fact > 0 else None
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
    metric_cols["Вес по факту"] = pd.to_numeric(metric_cols["Вес по факту"], errors="coerce").fillna(0.0)

    sub_summary = (
        metric_cols.groupby("Подразделение", as_index=False)
        .agg(
            n_months=("Месяц факта", "count"),
            mae=("Ошибка", lambda x: float(pd.to_numeric(x, errors="coerce").abs().mean())),
            bias=("Ошибка", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
            mape=("APE, %", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if not pd.to_numeric(x, errors="coerce").dropna().empty else float("nan")),
            weight_raw=("Вес по факту", "sum"),
        )
    )

    w_sum = float(pd.to_numeric(sub_summary["weight_raw"], errors="coerce").fillna(0.0).sum())
    if w_sum <= 1e-9:
        sub_summary["Вес, %"] = 100.0 / max(1, len(sub_summary))
    else:
        sub_summary["Вес, %"] = pd.to_numeric(sub_summary["weight_raw"], errors="coerce").fillna(0.0) / w_sum * 100.0

    sub_summary["MAE"] = pd.to_numeric(sub_summary["mae"], errors="coerce")
    sub_summary["Bias"] = pd.to_numeric(sub_summary["bias"], errors="coerce")
    sub_summary["MAPE, %"] = pd.to_numeric(sub_summary["mape"], errors="coerce")
    sub_summary = sub_summary[["Подразделение", "n_months", "Вес, %", "MAE", "MAPE, %", "Bias"]].sort_values(
        ["Вес, %", "Подразделение"],
        ascending=[False, True],
        kind="mergesort",
    )

    weights = pd.to_numeric(sub_summary["Вес, %"], errors="coerce").fillna(0.0) / 100.0
    weighted_mae = float((weights * pd.to_numeric(sub_summary["MAE"], errors="coerce").fillna(0.0)).sum())
    weighted_bias = float((weights * pd.to_numeric(sub_summary["Bias"], errors="coerce").fillna(0.0)).sum())

    mape_vals = pd.to_numeric(sub_summary["MAPE, %"], errors="coerce")
    mape_mask = mape_vals.notna()
    if bool(mape_mask.any()):
        w_mape = weights[mape_mask]
        w_norm = float(w_mape.sum())
        weighted_mape = float(((w_mape / w_norm) * mape_vals[mape_mask]).sum()) if w_norm > 1e-9 else None
    else:
        weighted_mape = None

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
    progress_cb: Optional[Callable[[int, int, date], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_date = latest_data_date(tables)
    base_month_end = month_end(base_date.year, base_date.month)

    if target_month_end < base_month_end:
        month_ends = [target_month_end]
    else:
        month_ends = iter_month_ends(base_month_end.year, base_month_end.month, target_month_end.year, target_month_end.month)

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
        "months_n": len(month_ends),
        "rows_calv": int(len(tables["calv"])),
        "rows_ins": int(len(tables["ins"])),
        "rows_dry": int(len(tables["dry"])),
        "rows_disp": int(len(tables["disp"])),
        "rows_bulls": int(len(tables["bulls"])),
    }
    return pd.DataFrame(rows), info


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
    progress_cb: Optional[Callable[[str, int, int, date], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sub_frames: list[pd.DataFrame] = []
    rows_meta: list[dict[str, Any]] = []

    for sub in subdivisions:
        tables = _load_farm_tables_from_db(sub)
        def _sub_cb(step_i: int, total_steps: int, d_end: date) -> None:
            if progress_cb is not None:
                progress_cb(sub, step_i, total_steps, d_end)

        monthly_sub, info_sub = _compute_farm_forecast(
            sub,
            tables,
            target_month_end,
            params,
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
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> pd.DataFrame:
    subs = _subdivisions_for_farm(farm_name, ready_only=False)
    if not subs:
        return pd.DataFrame(columns=["Месяц", "Подразделение", "Дойные коровы", "Сухостойные коровы", "Коровы всего"])

    rows: list[pd.DataFrame] = []
    total_subs = len(subs)
    for idx, sub in enumerate(subs, start=1):
        if progress_cb is not None:
            try:
                progress_cb(sub, idx, total_subs)
            except Exception:
                pass
        tables = _load_farm_tables_from_db(sub)
        monthly_sub, _ = _compute_farm_forecast(
            sub,
            tables,
            target_month_end,
            params,
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


def _build_transfer_recommendations(
    farm_name: str,
    target_month_end: date,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly_base = _subdivision_monthly_cows(farm_name, target_month_end, params)
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

        total_cows = float(sum(cows_before.values()))
        cap_by_sub = {sub: float(shares.get(sub, 0.0)) * total_cows for sub in subs}
        overflow_before = {sub: max(0.0, cows_before[sub] - cap_by_sub[sub]) for sub in subs}
        free_before = {sub: max(0.0, cap_by_sub[sub] - cows_before[sub]) for sub in subs}

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

                rec_rows.append(
                    {
                        "Месяц": str(month),
                        "Источник (переполнен)": src,
                        "Куда перевести": dst,
                        "Рекомендовано перевести, голов": float(move_n),
                        "Свободно в приёмнике, мест (оценка)": float(can_take),
                    }
                )

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
                    "Оценка мест (коровы)": float(cap_by_sub[sub]),
                    "Переполнение до перевода": float(overflow_before[sub]),
                    "Свободно мест до перевода": float(free_before[sub]),
                    "Переведено из подразделения": float(moved_out[sub]),
                    "Переведено в подразделение": float(moved_in[sub]),
                    "Коровы после переводов": float(cows_after[sub]),
                    "Переполнение после перевода": float(overflow_after[sub]),
                    "Свободно мест после перевода": float(free_after[sub]),
                    "Корректировка переводами, накопленная": float(cows_after[sub] - base_by_sub[sub]),
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
        "method": "monthly_historical_share_plus_carx_flows",
        "month_from": str(months[0]) if months else None,
        "month_to": str(months[-1]) if months else None,
        "months_n": int(len(months)),
        "sources_n": src_monthly_n,
        "destinations_n": dst_monthly_n,
        "recommendations_n": int(len(rec_df)),
        "total_moved": total_moved,
    }
    return rec_df, flows, snap_final, snap_monthly_df, meta


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
            farm_params_bt = _build_farm_params(base_params_bt, bt_override)
            bt_progress = st.progress(0.0)
            bt_status = st.empty()

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
        if str(bt_cfg.get("farm", "")) != str(bt_farm):
            st.info("Выбери хозяйство и нажми запуск, чтобы увидеть метрики для него.")
            return

        if not isinstance(bt_df, pd.DataFrame) or bt_df.empty:
            st.info("Нажми «Запустить backtesting по хозяйству», чтобы увидеть метрики.")
            return

        metric_for_view = str(bt_cfg.get("metric") or bt_target)
        is_pct = metric_for_view in FARM_PERCENT_TARGETS
        mae_label = "MAE, п.п." if is_pct else "MAE, гол."
        bias_label = "Bias, п.п." if is_pct else "Bias, гол."

        mae_val = bt_cfg.get("weighted_mae")
        mape_val = bt_cfg.get("weighted_mape")
        bias_val = bt_cfg.get("weighted_bias")
        if mae_val is None:
            mae_val = float(pd.to_numeric(bt_df["Ошибка"], errors="coerce").abs().mean())
        if bias_val is None:
            bias_val = float(pd.to_numeric(bt_df["Ошибка"], errors="coerce").mean())

        m1, m2, m3 = st.columns(3)
        m1.metric(mae_label, "—" if mae_val is None else f"{float(mae_val):.1f}")
        m2.metric("MAPE, %", "—" if mape_val is None else f"{float(mape_val):.1f}")
        m3.metric(bias_label, "—" if bias_val is None else f"{float(bias_val):.1f}")

        skipped_months = int(bt_cfg.get("skipped_months", 0) or 0)
        skipped_sub_months = int(bt_cfg.get("skipped_sub_months", 0) or 0)
        st.caption(
            f"Агрегация метрик выполнена по весам подразделений от факта. "
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

    def _style_forecast(df_view: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df_view.index, columns=df_view.columns)
        for ind in df_view.index:
            ov_name = INDICATOR_TO_OVERFLOW.get(str(ind))
            if not ov_name or ov_name not in overflow_df.columns:
                continue
            for m in df_view.columns:
                try:
                    ov = float(pd.to_numeric(overflow_df.loc[str(m), ov_name], errors="coerce") or 0.0)
                except Exception:
                    ov = 0.0
                if ov > 0.0:
                    styles.loc[ind, m] = BAD
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
                "Оценка мест (коровы)",
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
        cols_order = [
            "Месяц",
            "Источник (переполнен)",
            "Куда перевести",
            "Рекомендовано перевести, голов",
            "Свободно в приёмнике, мест (оценка)",
        ]
        cols_order = [c for c in cols_order if c in transfer_recs.columns]
        st.dataframe(
            transfer_recs[cols_order].style.format(fmt_cell),
            use_container_width=True,
            hide_index=True,
        )

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

    farm_status_df = _farm_status_df_from_db()
    ready_farms = farm_status_df.loc[farm_status_df["Статус"] == "готово", "Хозяйство"].astype(str).tolist() if not farm_status_df.empty else []
    selected_farms = st.multiselect(
        "Выбери хозяйства для расчёта",
        options=ready_farms,
        default=ready_farms[:1],
        key="tab3_selected_farms",
    )
    if _is_admin_mode():
        _farm_param_editor_block(sorted(set(ready_farms)), apply_admin_overrides(get_param_source()))
    else:
        st.caption("Изменение параметров доступно только в админ-режиме.")

    c_run, c_cache = st.columns([3, 2])
    run_clicked = c_run.button("Посчитать прогноз по хозяйствам", key="tab3_run_db", use_container_width=True)
    clear_clicked = c_cache.button("Очистить кэш прогнозов", key="tab3_clear_cache", use_container_width=True)

    if clear_clicked:
        _clear_forecast_cache(entity_type="farm")
        st.session_state.pop("tab3_monthly_all", None)
        st.session_state.pop("tab3_farm_infos", None)
        st.session_state.pop("tab3_target_month_end", None)
        st.success("Кэш прогнозов по хозяйствам очищен.")

    if run_clicked:
        if not selected_farms:
            st.error("Выбери хотя бы одно хозяйство.")
        else:
            base_params = apply_admin_overrides(get_param_source())
            all_farm_overrides = _farm_param_overrides_state() if _is_admin_mode() else {}
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
                    farm_params = _build_farm_params(base_params, all_farm_overrides.get(farm))
                    ph_farm = _params_hash(farm_params)
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
                        tables = _load_farm_merged_tables_from_db(farm)
                        n_calv = int(len(tables.get("calv", pd.DataFrame())))
                        n_ins = int(len(tables.get("ins", pd.DataFrame())))
                        n_dry = int(len(tables.get("dry", pd.DataFrame())))
                        n_disp = int(len(tables.get("disp", pd.DataFrame())))
                        _push_log(f"{farm}: входные строки calv={n_calv}, ins={n_ins}, dry={n_dry}, disp={n_disp}")

                        def _farm_month_cb(step_i: int, total_steps: int, d_end: date) -> None:
                            farm_prog.progress(step_i / max(1, total_steps))
                            live_status.caption(f"{farm}: расчёт месяца {step_i}/{total_steps} ({_month_label(d_end)})")
                            if step_i == 1 or step_i == total_steps or step_i % 2 == 0:
                                _push_log(f"{farm}: месяц {step_i}/{total_steps} ({_month_label(d_end)})")

                        monthly_df, info = _compute_farm_forecast(
                            farm,
                            tables,
                            target_month_end,
                            farm_params,
                            progress_cb=_farm_month_cb,
                        )
                        subs_all = _subdivisions_for_farm(farm, ready_only=False)
                        info["subdivisions_n"] = len(subs_all)

                        violations = _farm_sanity_violations_against_subdivisions(
                            monthly_df,
                            subdivisions=subs_all,
                            target_month_end=target_month_end,
                            params=farm_params,
                        )
                        if violations:
                            _push_log(f"{farm}: обнаружены аномалии, включаю safe-режим (сумма подразделений)")

                            def _sum_cb(sub: str, step_i: int, total_steps: int, d_end: date) -> None:
                                live_status.caption(
                                    f"{farm}: safe-режим, {sub}, месяц {step_i}/{total_steps} ({_month_label(d_end)})"
                                )
                                if step_i == 1 or step_i == total_steps or step_i % 2 == 0:
                                    _push_log(
                                        f"{farm}: safe-режим {sub} {step_i}/{total_steps} ({_month_label(d_end)})"
                                    )

                            monthly_df, info_fallback = _compute_farm_forecast_sum_of_subdivisions(
                                farm_name=farm,
                                subdivisions=subs_all,
                                target_month_end=target_month_end,
                                params=farm_params,
                                progress_cb=_sum_cb,
                            )
                            info = {**info, **info_fallback, "sanity_violations": violations}

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
                    if "transfer_recommendations_monthly" not in info:
                        _push_log(f"{farm}: анализ переездов CARX и подбор переводов")
                        rec_df, flows_df, snap_df, snap_monthly_df, rec_meta = _build_transfer_recommendations(
                            farm_name=farm,
                            target_month_end=target_month_end,
                            params=farm_params,
                        )
                        info["transfer_recommendations"] = rec_df.to_dict(orient="records")
                        info["transfer_recommendations_monthly"] = rec_df.to_dict(orient="records")
                        info["transfer_move_flows"] = flows_df.to_dict(orient="records")
                        info["transfer_snapshot"] = snap_df.to_dict(orient="records")
                        info["transfer_snapshot_monthly"] = snap_monthly_df.to_dict(orient="records")
                        info["transfer_meta"] = rec_meta

                    all_monthly.append(monthly_df)
                    farm_infos.append(info)
                    if isinstance(info, dict) and info.get("calc_mode") == "sum_subdivisions":
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

        if not files:
            return

        bundles, detect_df = _group_files(list(files))
        st.markdown("**Распознавание файлов**")
        st.dataframe(detect_df, use_container_width=True, hide_index=True)

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

        if st.button("Загрузить в БД", key="tab3_btn_upload_db", use_container_width=True):
            replace_subdivision = mode.startswith("Заменить")
            errors: list[str] = []
            updated_subdivisions: list[str] = []

            prog = st.progress(0.0)
            for i, subdivision in enumerate(ready_upload_subs, start=1):
                try:
                    tables = _prepare_tables(bundles[subdivision])
                    updated = _save_bundle_tables_to_db(
                        subdivision,
                        tables,
                        replace_subdivision=replace_subdivision,
                    )
                    updated_subdivisions.extend(updated)
                except Exception as e:
                    errors.append(f"{subdivision}: {e}")
                prog.progress(i / max(1, len(ready_upload_subs)))
            prog.empty()

            if errors:
                st.error("Часть подразделений не загрузилась:\n- " + "\n- ".join(errors))
            else:
                n_upd = len(set(updated_subdivisions)) if updated_subdivisions else len(ready_upload_subs)
                st.success(f"Готово. В БД обновлено подразделений: {n_upd}")
                st.rerun()
