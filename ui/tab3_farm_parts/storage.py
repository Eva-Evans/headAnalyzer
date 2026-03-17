from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from db import engine
from .common import *
from .common import _json_hash

_FARM_ALIAS_TARGET = "ЭНАЛБ"
_FARM_ALIAS_PREFIXES = ("БОДЕЕВ", "BODEEV")
_FARM_ALIAS_EXACT = {"ENALB"}
_BUILTIN_CAPACITY_FARM = _FARM_ALIAS_TARGET
_BUILTIN_CAPACITY_ROWS: tuple[tuple[str, str, float], ...] = (
    ("РЖК Добрино", "коровы дойные", 500.0),
    ("РЖК Добрино", "сухостой", 80.0),
    ("РЖК Добрино", "молодняк 9-24", 250.0),
    ("РЖК Добрино", "молодняк 3-8", 200.0),
    ("РЖК Добрино", "телята 0-3", 123.0),
    ("РЖК Добрино", "телята 3-5", 66.0),
    ("ЖК Высокое", "коровы дойные", 2400.0),
    ("ЖК Высокое", "сухостой", 400.0),
    ("ЖК Высокое", "молодняк 3-8", 800.0),
    ("ЖК Высокое", "молодняк 0-3", 600.0),
    ("ЖК Бодеевка", "коровы дойные", 2400.0),
    ("ЖК Бодеевка", "коровы сухостой", 400.0),
    ("ЖК Бодеевка", "молодняк 9-24", 3600.0),
    ("ЖК Бодеевка", "молодняк 3-8", 1000.0),
    ("ЖК Бодеевка", "молодняк 0-3", 600.0),
    ("ЖК Добрино", "коровы дойные", 2400.0),
    ("ЖК Добрино", "коровы сухостой", 400.0),
    ("ЖК Добрино", "молодняк 3-8", 820.0),
    ("ЖК Добрино", "молодняк 9-24", 2400.0),
    ("ЖК Добрино", "молодняк 0-3", 600.0),
    ("МТФ Высокое", "коровы дойные", 420.0),
    ("МТФ Высокое", "коровы сухостой", 80.0),
    ("МТФ Высокое", "молодняк 9-24", 1070.0),
    ("МТФ Высокое", "молодняк 0-3", 140.0),
    ("МТФ Высокое", "молодняк 3-8", 445.0),
    ("МТФ Садовое", "коровы дойные", 420.0),
    ("МТФ Садовое", "коровы сухостой", 50.0),
    ("МТФ Садовое", "молодняк 9-24", 270.0),
    ("МТФ Садовое", "молодняк 0-3", 87.0),
    ("МТФ Дракино", "коровы дойные", 400.0),
    ("МТФ Дракино", "коровы сухостой", 80.0),
    ("МТФ Дракино", "молодняк 9-24", 300.0),
    ("МТФ Дракино", "молодняк 3-8", 200.0),
    ("МТФ Дракино", "молодняк 0-3", 98.0),
    ("МТФ Старая Хвор", "коровы", 280.0),
    ("МТФ Старая Хвор", "коровы сухостой", 60.0),
    ("МТФ Старая Хвор", "молодняк", 200.0),
    ("МТФ Старая Хвор", "молодняк", 100.0),
    ("МТФ Старая Хвор", "телята", 5.0),
)


def _has_farm_alias_token(s: str) -> bool:
    if not s:
        return False
    return any(tok in s for tok in _FARM_ALIAS_PREFIXES)


def _capacity_seed_match_key(x: Any) -> str:
    s = str(x or "").upper().replace("Ё", "Е")
    s = s.replace("…", " ").replace("...", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"[^A-ZА-Я0-9]+", "", s)


def _capacity_seed_match_variants(x: Any) -> set[str]:
    base = _capacity_seed_match_key(x)
    if not base:
        return set()
    out = {base}
    for prefix in ("ЖК", "РЖК", "МТФ"):
        if base.startswith(prefix) and len(base) > len(prefix):
            out.add(base[len(prefix):])
    return {v for v in out if v}


def _resolve_builtin_capacity_subdivision_name(subdivision_hint: str, existing_subdivisions: list[str]) -> str:
    hint = _canon_subdivision_name(subdivision_hint)
    if not hint:
        return ""
    if not existing_subdivisions:
        return hint
    hint_vars = _capacity_seed_match_variants(hint)
    exact: list[str] = []
    fuzzy: list[str] = []
    for raw_sub in existing_subdivisions:
        sub = _canon_subdivision_name(raw_sub)
        if not sub:
            continue
        if sub == hint:
            return sub
        sub_vars = _capacity_seed_match_variants(sub)
        if hint_vars & sub_vars:
            exact.append(sub)
            continue
        if any((a in b) or (b in a) for a in hint_vars for b in sub_vars):
            fuzzy.append(sub)
    exact_unique = sorted(set(exact))
    if len(exact_unique) == 1:
        return exact_unique[0]
    fuzzy_unique = sorted(set(fuzzy))
    if len(fuzzy_unique) == 1:
        return fuzzy_unique[0]
    return hint


def _seed_builtin_capacity_rows() -> int:
    _ensure_subdivision_map_table()
    farm = _canon_farm_name(_BUILTIN_CAPACITY_FARM)
    if not farm:
        return 0
    try:
        existing_df = pd.read_sql(
            text(
                f"""
                SELECT subdivision_name
                FROM {TAB3_MAP_TABLE}
                WHERE farm_name = :farm
                ORDER BY subdivision_name
                """
            ),
            con=engine,
            params={"farm": farm},
        )
    except Exception:
        existing_df = pd.DataFrame(columns=["subdivision_name"])
    existing_subs = (
        existing_df["subdivision_name"].astype(str).map(_canon_subdivision_name).tolist()
        if not existing_df.empty and "subdivision_name" in existing_df.columns
        else []
    )
    rows: list[dict[str, Any]] = []
    for subdivision_hint, group_name, places in _BUILTIN_CAPACITY_ROWS:
        resolved_sub = _resolve_builtin_capacity_subdivision_name(subdivision_hint, existing_subs)
        if not resolved_sub:
            continue
        rows.append(
            {
                "farm_name": farm,
                "subdivision_name": resolved_sub,
                "group_name": str(group_name).strip(),
                "places": float(places),
            }
        )
    if not rows:
        return 0
    work = pd.DataFrame(rows)
    work["farm_name"] = work["farm_name"].astype(str).map(_canon_farm_name)
    work["subdivision_name"] = work["subdivision_name"].astype(str).map(_canon_subdivision_name)
    work["group_name"] = work["group_name"].astype(str).str.strip()
    work["places"] = pd.to_numeric(work["places"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.loc[
        (work["farm_name"] != "") & (work["subdivision_name"] != "") & (work["group_name"] != "")
    ].copy()
    if work.empty:
        return 0
    work = (
        work.groupby(["farm_name", "subdivision_name", "group_name"], as_index=False)["places"]
        .sum()
        .sort_values(["farm_name", "subdivision_name", "group_name"], kind="mergesort")
        .reset_index(drop=True)
    )
    sql = text(
        f"""
        INSERT INTO {TAB3_CAPACITY_TABLE}(farm_name, subdivision_name, group_name, places, updated_at)
        VALUES (:farm_name, :subdivision_name, :group_name, :places, NOW())
        ON CONFLICT (farm_name, subdivision_name, group_name)
        DO UPDATE SET
          places = EXCLUDED.places,
          updated_at = NOW()
        WHERE {TAB3_CAPACITY_TABLE}.places IS NULL OR {TAB3_CAPACITY_TABLE}.places <= 0
        """
    )
    changed = 0
    with engine.begin() as conn:
        for row in work.to_dict(orient="records"):
            res = conn.execute(sql, row)
            changed += int(res.rowcount or 0)
    if changed > 0:
        _clear_forecast_cache(entity_type="farm", entity_name=farm)
    return changed

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
    normalize_aliases = f"""
    UPDATE {TAB3_MAP_TABLE}
    SET farm_name = :target, updated_at = NOW()
    WHERE
      UPPER(REPLACE(COALESCE(farm_name, ''), 'Ё', 'Е')) LIKE '%БОДЕЕВ%'
      OR UPPER(COALESCE(farm_name, '')) LIKE '%BODEEV%'
      OR UPPER(COALESCE(farm_name, '')) = 'ENALB'
      OR UPPER(REPLACE(COALESCE(subdivision_name, ''), 'Ё', 'Е')) LIKE '%БОДЕЕВ%'
      OR UPPER(COALESCE(subdivision_name, '')) LIKE '%BODEEV%';
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(backfill))
        conn.execute(text(normalize_aliases), {"target": _FARM_ALIAS_TARGET})

def _ensure_capacity_table() -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TAB3_CAPACITY_TABLE} (
        farm_name TEXT NOT NULL,
        subdivision_name TEXT NOT NULL,
        group_name TEXT NOT NULL,
        places DOUBLE PRECISION NOT NULL DEFAULT 0,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (farm_name, subdivision_name, group_name)
    );
    """
    idx_farm = f"""
    CREATE INDEX IF NOT EXISTS idx_{TAB3_CAPACITY_TABLE}_farm
    ON {TAB3_CAPACITY_TABLE}(farm_name);
    """
    idx_sub = f"""
    CREATE INDEX IF NOT EXISTS idx_{TAB3_CAPACITY_TABLE}_sub
    ON {TAB3_CAPACITY_TABLE}(subdivision_name);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(idx_farm))
        conn.execute(text(idx_sub))
    _seed_builtin_capacity_rows()

def _upsert_subdivision_mapping(subdivision_name: str, farm_name: str | None = None, overwrite: bool = False) -> None:
    _ensure_subdivision_map_table()
    subdivision = (subdivision_name or "").strip()
    farm = _canon_farm_name((farm_name or subdivision).strip())
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

def _farm_capacity_signature_from_db(farm_name: str) -> str:
    _ensure_capacity_table()
    sql = f"""
    SELECT
      COALESCE(COUNT(*), 0) AS n_rows,
      COALESCE(MAX(updated_at)::text, '') AS mx_upd,
      COALESCE(SUM(COALESCE(places, 0)), 0) AS sum_places
    FROM {TAB3_CAPACITY_TABLE}
    WHERE farm_name = :farm
    ;
    """
    df = pd.read_sql(text(sql), con=engine, params={"farm": _canon_farm_name(farm_name)})
    if df.empty:
        return "0||0"
    r = df.iloc[0].to_dict()
    return f"{int(r.get('n_rows', 0) or 0)}|{r.get('mx_upd', '')}|{float(r.get('sum_places', 0) or 0):.6f}"

def _farm_signature_from_db(farm_name: str) -> str:
    parts: list[str] = []
    for sub in _subdivisions_for_farm(farm_name, ready_only=True):
        parts.append(f"{sub}:{_subdivision_signature_from_db(sub)}")
    parts.append(f"capacity:{_farm_capacity_signature_from_db(farm_name)}")
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
    subdivisions = _subdivisions_for_farm(farm_name, ready_only=True)
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

def _deduplicate_subdivision_rows(subdivision_name: str) -> None:
    _ensure_farm_tables()
    sub = (subdivision_name or "").strip()
    if not sub:
        return

    specs: list[tuple[str, list[str]]] = [
        (TAB3_TABLES["calv"], ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"]),
        (TAB3_TABLES["ins"], ["reg", "lact", "dim_age", "event_date", "bull", "result"]),
        (TAB3_TABLES["dry"], ["reg", "dim", "event_date", "move_reason"]),
        (TAB3_TABLES["disp"], ["reg", "event_date", "disposal_reason"]),
        (TAB3_TABLES["bulls"], ["bull_code", "bull_type"]),
    ]

    with engine.begin() as conn:
        for table_name, cols in specs:
            partition_expr = ", ".join([f"COALESCE({c}::text, '')" for c in cols])
            sql = f"""
            WITH ranked AS (
              SELECT ctid,
                     ROW_NUMBER() OVER (
                       PARTITION BY {partition_expr}
                       ORDER BY ctid
                     ) AS rn
              FROM {table_name}
              WHERE farm_name = :farm
            )
            DELETE FROM {table_name} t
            USING ranked r
            WHERE t.ctid = r.ctid
              AND r.rn > 1
            ;
            """
            conn.execute(text(sql), {"farm": sub})

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
        dfx = dfx.drop_duplicates(subset=cols, keep="last")

        if "birth_date" in dfx.columns:
            dfx["birth_date"] = pd.to_datetime(dfx["birth_date"], errors="coerce", dayfirst=True).dt.date
        if "event_date" in dfx.columns:
            dfx["event_date"] = pd.to_datetime(dfx["event_date"], errors="coerce", dayfirst=True).dt.date

        dfx.insert(0, "farm_name", farm)
        dfx.to_sql(TAB3_TABLES[key], con=engine, if_exists="append", index=False, method="multi", chunksize=2000)

    _deduplicate_subdivision_rows(farm)
    _upsert_subdivision_mapping(farm, farm_name=farm, overwrite=False)

def _delete_subdivision_everywhere(subdivision_name: str) -> None:
    sub = (subdivision_name or "").strip()
    if not sub:
        return
    _ensure_capacity_table()
    with engine.begin() as conn:
        for t in TAB3_TABLES.values():
            conn.execute(text(f"DELETE FROM {t} WHERE farm_name = :sub"), {"sub": sub})
        conn.execute(text(f"DELETE FROM {TAB3_MAP_TABLE} WHERE subdivision_name = :sub"), {"sub": sub})
        conn.execute(text(f"DELETE FROM {TAB3_CAPACITY_TABLE} WHERE subdivision_name = :sub"), {"sub": sub})

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
    s = df["__farm"].map(_canon_farm_name)
    s = s[s != ""]
    return set(s.astype(str).tolist())

def _farm_match_score(tables: dict[str, pd.DataFrame], farm_name: str) -> int:
    farm_u = _canon_farm_name(farm_name)
    if not farm_u:
        return 0
    score = 0
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty or "__farm" not in df.columns:
            continue
        s = df["__farm"].map(_canon_farm_name)
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
    hint = _canon_farm_name(bundle_hint)
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

def _canon_farm_name(x: Any) -> str:
    s = _canon_name(x)
    if not s:
        return ""
    if s in _FARM_ALIAS_EXACT:
        return _FARM_ALIAS_TARGET
    if _has_farm_alias_token(s):
        return _FARM_ALIAS_TARGET
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
            d["__farm"] = d["__farm"].map(_canon_farm_name)
        if "__subdivision" in d.columns:
            d["__subdivision"] = d["__subdivision"].map(_canon_subdivision_name)
        out[key] = d
    return out

def _filter_tables_by_farm(tables: dict[str, pd.DataFrame], farm_name: str) -> dict[str, pd.DataFrame]:
    hint = _canon_farm_name(farm_name)
    if not hint:
        return tables
    out: dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            continue
        d = df.copy()
        if "__farm" in d.columns:
            s = d["__farm"].map(_canon_farm_name)
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

def _farm_name_for_subdivision_from_tables(
    tables: dict[str, pd.DataFrame],
    subdivision_name: str,
    default_farm: str,
) -> str:
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
    out = _mode_nonempty(farm_vals) or default_farm
    farm = _canon_farm_name(out)
    if farm != TAB3_UNASSIGNED_FARM:
        return farm

    # Fallback: if subdivision name clearly points to alias farm, do not keep it unassigned.
    sub_farm = _canon_farm_name(subdivision_name)
    if sub_farm and sub_farm != TAB3_UNASSIGNED_FARM:
        return sub_farm
    return farm

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
    default_farm = _canon_farm_name(
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
        farm_name = _farm_name_for_subdivision_from_tables(tables, subdivision, default_farm=default_farm)
        farm_name = _canon_farm_name(farm_name)
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

def _load_capacity_rows_for_farm(farm_name: str) -> pd.DataFrame:
    _ensure_capacity_table()
    farm = _canon_farm_name(farm_name)
    if not farm:
        return pd.DataFrame(columns=["farm_name", "subdivision_name", "group_name", "places"])
    sql = f"""
    SELECT farm_name, subdivision_name, group_name, places
    FROM {TAB3_CAPACITY_TABLE}
    WHERE farm_name = :farm
    ORDER BY subdivision_name, group_name
    ;
    """
    df = pd.read_sql(text(sql), con=engine, params={"farm": farm})
    if df.empty:
        return pd.DataFrame(columns=["farm_name", "subdivision_name", "group_name", "places"])
    df["farm_name"] = df["farm_name"].astype(str).map(_canon_farm_name)
    df["subdivision_name"] = df["subdivision_name"].astype(str).map(_canon_subdivision_name)
    df["group_name"] = df["group_name"].astype(str).str.strip()
    df["places"] = pd.to_numeric(df["places"], errors="coerce").fillna(0.0)
    return df[["farm_name", "subdivision_name", "group_name", "places"]].copy()

def _save_capacity_rows_for_farm(farm_name: str, rows: pd.DataFrame) -> int:
    _ensure_capacity_table()
    farm = _canon_farm_name(farm_name)
    if not farm:
        return 0

    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(columns=["subdivision_name", "group_name", "places"])
    work = rows.copy()
    if "subdivision_name" not in work.columns:
        work["subdivision_name"] = ""
    if "group_name" not in work.columns:
        work["group_name"] = ""
    if "places" not in work.columns:
        work["places"] = 0

    work["subdivision_name"] = work["subdivision_name"].astype(str).map(_canon_subdivision_name)
    work["group_name"] = work["group_name"].astype(str).str.strip()
    work["places"] = pd.to_numeric(work["places"], errors="coerce").fillna(0.0)
    work["places"] = work["places"].clip(lower=0.0)
    work = work.loc[(work["subdivision_name"] != "") & (work["group_name"] != "")].copy()
    if not work.empty:
        work = (
            work.groupby(["subdivision_name", "group_name"], as_index=False)["places"].sum()
            .sort_values(["subdivision_name", "group_name"], kind="mergesort")
            .reset_index(drop=True)
        )

    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {TAB3_CAPACITY_TABLE} WHERE farm_name = :farm"), {"farm": farm})
        if not work.empty:
            payload = work.copy()
            payload.insert(0, "farm_name", farm)
            payload.to_sql(TAB3_CAPACITY_TABLE, con=conn, if_exists="append", index=False, method="multi", chunksize=500)

    _clear_forecast_cache(entity_type="farm", entity_name=farm)
    return int(len(work))

def _load_cow_capacity_by_subdivision(farm_name: str) -> dict[str, float]:
    df = _load_capacity_rows_for_farm(farm_name)
    if df.empty:
        return {}

    def _is_cow_group(x: Any) -> bool:
        s = str(x or "").upper().replace("Ё", "Е")
        return ("КОРОВ" in s) or ("COW" in s) or ("СУХОСТ" in s)

    work = df.loc[df["group_name"].map(_is_cow_group)].copy()
    if work.empty:
        return {}
    agg = work.groupby("subdivision_name", as_index=False)["places"].sum()
    return {
        str(r["subdivision_name"]): float(r["places"])
        for r in agg.to_dict(orient="records")
        if str(r.get("subdivision_name", "")).strip() != ""
    }

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
      (SELECT COUNT(*) FROM {TAB3_TABLES['bulls']} b WHERE b.farm_name = s.subdivision_name) AS n_bulls,
      GREATEST(
        COALESCE((SELECT MAX(event_date::date) FROM {TAB3_TABLES['calv']} c WHERE c.farm_name = s.subdivision_name), DATE '1900-01-01'),
        COALESCE((SELECT MAX(event_date::date) FROM {TAB3_TABLES['ins']} i WHERE i.farm_name = s.subdivision_name), DATE '1900-01-01'),
        COALESCE((SELECT MAX(event_date::date) FROM {TAB3_TABLES['dry']} d WHERE d.farm_name = s.subdivision_name), DATE '1900-01-01'),
        COALESCE((SELECT MAX(event_date::date) FROM {TAB3_TABLES['disp']} x WHERE x.farm_name = s.subdivision_name), DATE '1900-01-01')
      ) AS last_event_date
    FROM subs s
    LEFT JOIN {TAB3_MAP_TABLE} m
      ON m.subdivision_name = s.subdivision_name
    ORDER BY COALESCE(m.farm_name, s.subdivision_name), s.subdivision_name;
    """

    df = pd.read_sql(text(sql), con=engine)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Хозяйство",
                "Подразделение",
                "Статус",
                "Отёлы",
                "Осеменения",
                "Запуски",
                "Выбытие",
                "Быки",
                "Последняя дата данных",
            ]
        )

    for c in ("n_calv", "n_ins", "n_dry", "n_disp", "n_bulls"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["farm_name"] = df["farm_name"].map(_canon_farm_name)
    df["last_event_date"] = pd.to_datetime(df.get("last_event_date"), errors="coerce").dt.date
    sentinel = date(1900, 1, 1)
    df.loc[df["last_event_date"] == sentinel, "last_event_date"] = pd.NaT

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
            "last_event_date": "Последняя дата данных",
        }
    )


def _farm_name_for_subdivision(subdivision_name: str) -> str:
    _ensure_subdivision_map_table()
    sub = _canon_subdivision_name(subdivision_name)
    if not sub:
        return ""
    try:
        df = pd.read_sql(
            text(
                f"""
                SELECT farm_name
                FROM {TAB3_MAP_TABLE}
                WHERE subdivision_name = :sub
                LIMIT 1
                """
            ),
            con=engine,
            params={"sub": sub},
        )
        if not df.empty and "farm_name" in df.columns:
            farm = _canon_farm_name(df.loc[0, "farm_name"])
            if farm:
                return farm
    except Exception:
        pass
    sub_df = _subdivision_status_df_from_db()
    if not isinstance(sub_df, pd.DataFrame) or sub_df.empty:
        return ""
    if "Подразделение" not in sub_df.columns or "Хозяйство" not in sub_df.columns:
        return ""
    mask = sub_df["Подразделение"].astype(str).map(_canon_subdivision_name) == sub
    if not bool(mask.any()):
        return ""
    farms = sub_df.loc[mask, "Хозяйство"].astype(str).map(_canon_farm_name)
    farms = farms[farms != ""]
    if farms.empty:
        return ""
    return str(farms.iloc[0])

def _farm_status_df_from_db() -> pd.DataFrame:
    sub = _subdivision_status_df_from_db()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "Хозяйство",
                "Статус",
                "Подразделений",
                "Готовых подразделений",
                "Отёлы",
                "Осеменения",
                "Запуски",
                "Выбытие",
                "Быки",
                "Последняя дата данных",
            ]
        )

    sub = sub.loc[sub["Хозяйство"].astype(str) != TAB3_UNASSIGNED_FARM].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "Хозяйство",
                "Статус",
                "Подразделений",
                "Готовых подразделений",
                "Отёлы",
                "Осеменения",
                "Запуски",
                "Выбытие",
                "Быки",
                "Последняя дата данных",
            ]
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
            last_event_date=("Последняя дата данных", "max"),
        )
    )
    agg["Статус"] = agg.apply(
        lambda r: "готово" if (int(r["n_sub"]) > 0 and int(r["n_ready"]) == int(r["n_sub"])) else "неполный набор",
        axis=1,
    )
    return agg.rename(
        columns={
            "n_sub": "Подразделений",
            "n_ready": "Готовых подразделений",
            "n_calv": "Отёлы",
            "n_ins": "Осеменения",
            "n_dry": "Запуски",
            "n_disp": "Выбытие",
            "n_bulls": "Быки",
            "last_event_date": "Последняя дата данных",
        }
    )[
        [
            "Хозяйство",
            "Статус",
            "Подразделений",
            "Готовых подразделений",
            "Отёлы",
            "Осеменения",
            "Запуски",
            "Выбытие",
            "Быки",
            "Последняя дата данных",
        ]
    ]

def _subdivisions_for_farm(farm_name: str, ready_only: bool = True) -> list[str]:
    sub = _subdivision_status_df_from_db()
    if sub.empty:
        return []
    farm_u = _canon_farm_name(farm_name)
    mask = sub["Хозяйство"].astype(str).map(_canon_farm_name) == farm_u
    if ready_only:
        mask &= sub["Статус"].astype(str) == "готово"
    return sorted(sub.loc[mask, "Подразделение"].astype(str).tolist())


__all__ = [name for name in globals() if not name.startswith("__")]
