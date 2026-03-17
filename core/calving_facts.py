from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


BIRTH_STATS_KEYS = [
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]


def norm_fact_id(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ").strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def norm_fact_sex(x: Any) -> str | None:
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


def norm_fact_event_type(x: Any) -> str:
    if x is None:
        return ""
    v = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    if v == "" or v == "NAN":
        return ""
    if ("РОЖ" in v) or ("BORN" in v) or ("BIRTH" in v):
        return "РОЖДЕН"
    if ("ОТЕЛ" in v) or ("CALV" in v):
        return "ОТЕЛ"
    return v


def _month_bounds(month_end_date: date) -> tuple[date, date]:
    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)
    return m_start, m_next


def _empty_birth_stats() -> dict[str, float]:
    return {k: 0.0 for k in BIRTH_STATS_KEYS}


def _series(df: pd.DataFrame, col: str, default: Any = pd.NA) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _prepare_calving_rows(
    calv_df: pd.DataFrame,
    month_end_date: date,
    as_of_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(calv_df, pd.DataFrame) or calv_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    m_start, m_next = _month_bounds(month_end_date)
    as_of_ts = None if as_of_date is None else pd.Timestamp(as_of_date).normalize()

    c = calv_df.copy()
    c["event_type_n"] = _series(c, "event_type", "").map(norm_fact_event_type)
    c["event_date_n"] = pd.to_datetime(_series(c, "event_date", pd.NaT), errors="coerce").dt.normalize()
    c["birth_date_n"] = pd.to_datetime(_series(c, "birth_date", pd.NaT), errors="coerce").dt.normalize()
    c["reg_s"] = _series(c, "reg", "").map(norm_fact_id)
    c["mother_reg_s"] = _series(c, "mother_reg", "").map(norm_fact_id)
    c["sex_norm"] = _series(c, "sex", None).map(norm_fact_sex)
    c["lact_num"] = pd.to_numeric(_series(c, "lact", pd.NA), errors="coerce")

    c["fact_dt_n"] = c["event_date_n"]
    born_mask = (c["event_type_n"] == "РОЖДЕН") & c["birth_date_n"].notna()
    c.loc[born_mask, "fact_dt_n"] = c.loc[born_mask, "birth_date_n"]

    calving_like = c["event_type_n"].isin(["ОТЕЛ", "РОЖДЕН"])
    scoped = c[
        calving_like
        & c["fact_dt_n"].notna()
        & (c["fact_dt_n"] >= pd.Timestamp(m_start))
        & (c["fact_dt_n"] < pd.Timestamp(m_next))
    ].copy()
    if as_of_ts is not None:
        scoped = scoped.loc[scoped["fact_dt_n"] <= as_of_ts].copy()
    if scoped.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    born = scoped.loc[scoped["event_type_n"] == "РОЖДЕН"].copy()
    otel = scoped.loc[scoped["event_type_n"] == "ОТЕЛ"].copy()

    born_events = born[["mother_reg_s", "fact_dt_n", "lact_num"]].rename(
        columns={"mother_reg_s": "cow_reg_s", "fact_dt_n": "calv_dt"}
    )
    born_events["source_rank"] = 1

    otel_events = otel[["reg_s", "fact_dt_n", "lact_num"]].rename(
        columns={"reg_s": "cow_reg_s", "fact_dt_n": "calv_dt"}
    )
    otel_events["source_rank"] = 0

    events = pd.concat([otel_events, born_events], ignore_index=True)
    if events.empty:
        empty = pd.DataFrame()
        return scoped, born, empty

    missing = events["cow_reg_s"].astype(str).str.strip() == ""
    if bool(missing.any()):
        events.loc[missing, "cow_reg_s"] = [f"__UNK__{i}" for i in range(int(missing.sum()))]

    events["lact_missing_rank"] = events["lact_num"].isna().astype(int)
    events = events.sort_values(
        ["cow_reg_s", "calv_dt", "source_rank", "lact_missing_rank"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    events = events.drop_duplicates(subset=["cow_reg_s", "calv_dt"], keep="first").copy()
    return scoped, born, events


def actual_birth_stats_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame | None,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    scoped, born, events = _prepare_calving_rows(calv_df, month_end_date, as_of_date=as_of_date)
    if events.empty:
        return _empty_birth_stats()

    total_calv = float(len(events))
    events = events.copy()
    events["lact_eff"] = events["lact_num"]

    if isinstance(ins_df, pd.DataFrame) and not ins_df.empty:
        ins = ins_df.copy()
        ins["reg_s"] = _series(ins, "reg", "").map(norm_fact_id)
        ins["event_date_n"] = pd.to_datetime(_series(ins, "event_date", pd.NaT), errors="coerce").dt.normalize()
        ins["lact_n"] = pd.to_numeric(_series(ins, "lact", pd.NA), errors="coerce")
        ins = ins[ins["event_date_n"].notna() & (ins["reg_s"] != "")].copy()
        if as_of_date is not None:
            ins = ins.loc[ins["event_date_n"] <= pd.Timestamp(as_of_date).normalize()].copy()
        if not ins.empty:
            left = events[["cow_reg_s", "calv_dt"]].sort_values(["cow_reg_s", "calv_dt"], kind="mergesort")
            right = ins[["reg_s", "event_date_n", "lact_n"]].rename(
                columns={"reg_s": "cow_reg_s", "event_date_n": "ins_dt"}
            ).sort_values(["cow_reg_s", "ins_dt"], kind="mergesort")
            try:
                merged = pd.merge_asof(
                    left,
                    right,
                    by="cow_reg_s",
                    left_on="calv_dt",
                    right_on="ins_dt",
                    direction="backward",
                    allow_exact_matches=True,
                )
                merged = merged.rename(columns={"lact_n": "hist_lact"})
                events = events.merge(merged[["cow_reg_s", "calv_dt", "hist_lact"]], on=["cow_reg_s", "calv_dt"], how="left")
                events["lact_eff"] = events["lact_eff"].where(events["lact_eff"].notna(), events["hist_lact"])
            except Exception:
                pass

    lact_eff = pd.to_numeric(events["lact_eff"], errors="coerce")
    heif_calv = float((lact_eff <= 0).sum())
    cow_calv = float(((lact_eff > 0) | lact_eff.isna()).sum())

    bulls_known = float((born.get("sex_norm", pd.Series(dtype=object)) == "M").sum()) if not born.empty else 0.0
    heifers_known = float((born.get("sex_norm", pd.Series(dtype=object)) == "F").sum()) if not born.empty else 0.0
    total_birth_rows = float(len(born))

    if total_birth_rows <= 0 and total_calv > 0:
        bulls = total_calv * 0.5
        heifers = total_calv - bulls
    else:
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
        "Ожидаемые бычки": bulls,
        "Ожидаемые тёлочки": heifers,
        "Доля бычков среди рождений, %": bull_pct,
        "Доля тёлочек среди рождений, %": heif_pct,
    }


def is_calving_month_complete_from_tables(calv_df: pd.DataFrame, month_end_date: date) -> bool:
    scoped, _, _ = _prepare_calving_rows(calv_df, month_end_date, as_of_date=None)
    if scoped.empty:
        return False
    max_dt = pd.to_datetime(scoped["fact_dt_n"], errors="coerce").max()
    if pd.isna(max_dt):
        return False
    return bool(pd.Timestamp(max_dt).date() >= month_end_date)
