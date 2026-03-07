from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import streamlit as st

from etl.calvings_births import read_calvings_excel, load_calvings_to_db
from etl.inseminations import read_inseminations_excel, clean_inseminations, load_inseminations_to_db
from etl.disposals import read_disposals_excel, load_disposals_to_db


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

    out_rows: List[Dict[str, Any]] = []

                                              
    for i in range(len(df_raw)):
        if pd.isna(dts.iloc[i]):
            continue
        mr = mother.iloc[i]
        if not mr:
            continue
        out_rows.append({
            "reg": mr,
            "mother_reg": "",
            "birth_date": pd.NaT,
            "sex": "F",
            "event_type": "ОТЕЛ",
            "event_date": dts.iloc[i],
        })

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
                out_rows.append({
                    "reg": calf,
                    "mother_reg": mr,
                    "birth_date": dt,
                    "sex": (sx.iloc[i] if sx is not None else None),
                    "event_type": "РОЖДЕН",
                    "event_date": dt,
                })

    out = pd.DataFrame(out_rows, columns=["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"])
    return out


def _fallback_inseminations(df_raw: pd.DataFrame) -> pd.DataFrame:
                  
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    lact_c = _find_col(df_raw, "LACT", "LACTATION")
    dim_c = _find_col(df_raw, "DIM", "DIM_AGE", "DAYS")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE")
    bull_c = _find_col(df_raw, "REMARK", "BULL", "B", "BULL_CODE")
    res_c = _find_col(df_raw, "R", "RESULT", "RES")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле осеменений.")

    out = pd.DataFrame({
        "reg": df_raw[reg_c].map(_norm_id),
        "lact": pd.to_numeric(df_raw[lact_c], errors="coerce") if lact_c else 0,
        "dim_age": pd.to_numeric(df_raw[dim_c], errors="coerce") if dim_c else pd.NA,
        "event_date": _to_dt(df_raw[date_c]),
        "bull": df_raw[bull_c].map(_norm_id) if bull_c else "",
        "result": df_raw[res_c].astype(str).str.strip() if res_c else "",
    })
    return out


def _fallback_disposals(df_raw: pd.DataFrame) -> pd.DataFrame:
                  
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE")
    reason_c = _find_col(df_raw, "REMARK", "DISPOSAL_REASON", "REM")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле выбытия.")

    out = pd.DataFrame({
        "reg": df_raw[reg_c].map(_norm_id),
        "event_date": _to_dt(df_raw[date_c]),
        "disposal_reason": df_raw[reason_c].astype(str).str.strip() if reason_c else "",
    })
    return out


def _basic_stats(df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        out["min_date"] = None if d.dropna().empty else d.min().date()
        out["max_date"] = None if d.dropna().empty else d.max().date()
    return out


def render_tab3_data() -> None:
    st.subheader("Данные: загрузка и контроль качества (3 таблицы)")

    mode = st.radio(
        "Режим загрузки в БД",
        options=["replace (перезаписать таблицы)", "append (добавить строки)"],
        index=0,
        horizontal=True,
        key="tab3_mode",
    )
    if_exists = "replace" if mode.startswith("replace") else "append"

    c1, c2 = st.columns(2)
    with c1:
        f_calv = st.file_uploader("Отёлы + родившиеся (xlsx)", type=["xls", "xlsx"], key="tab3_calv")
        f_ins = st.file_uploader("Осеменения (xlsx)", type=["xls", "xlsx"], key="tab3_ins")
    with c2:
        f_disp = st.file_uploader("Выбытие (xlsx)", type=["xls", "xlsx"], key="tab3_disp")

    st.divider()

    def _preview_block(title: str, df: Optional[pd.DataFrame], stats: Dict[str, Any]) -> None:
        st.markdown(f"### {title}")
        st.write(stats)
        if df is not None and not df.empty:
            st.dataframe(df.head(200), use_container_width=True)

    calv_df = None
    ins_df = None
    disp_df = None

    if f_calv is not None:
        try:
            calv_df = read_calvings_excel(f_calv)
        except Exception:
            raw = pd.read_excel(f_calv)
            calv_df = _fallback_calvings(raw)

        calv_df = calv_df.copy()
        if "event_date" in calv_df.columns:
            calv_df["event_date"] = pd.to_datetime(calv_df["event_date"], errors="coerce").dt.normalize()
        if "birth_date" in calv_df.columns:
            calv_df["birth_date"] = pd.to_datetime(calv_df["birth_date"], errors="coerce").dt.normalize()
        calv_df["reg"] = calv_df.get("reg", "").map(_norm_id)
        calv_df["mother_reg"] = calv_df.get("mother_reg", "").map(_norm_id)
        calv_df["event_type"] = calv_df.get("event_type", "").map(_norm_event_type)
        calv_df["sex"] = calv_df.get("sex", None).map(_norm_sex) if "sex" in calv_df.columns else None

        st_ok = _basic_stats(calv_df, "event_date")
        _preview_block("Отёлы/Родившиеся → calvings_births_raw", calv_df, st_ok)

    if f_ins is not None:
        try:
            ins_df = read_inseminations_excel(f_ins)
            ins_df = clean_inseminations(ins_df)
        except Exception:
            raw = pd.read_excel(f_ins)
            ins_df = _fallback_inseminations(raw)

        ins_df = ins_df.copy()
        ins_df["reg"] = ins_df.get("reg", "").map(_norm_id)
        if "event_date" in ins_df.columns:
            ins_df["event_date"] = pd.to_datetime(ins_df["event_date"], errors="coerce").dt.normalize()
        if "bull" in ins_df.columns:
            ins_df["bull"] = ins_df["bull"].map(_norm_id)
        st_ok = _basic_stats(ins_df, "event_date")
        _preview_block("Осеменения → inseminations_raw", ins_df, st_ok)

    if f_disp is not None:
        try:
            disp_df = read_disposals_excel(f_disp)
        except Exception:
            raw = pd.read_excel(f_disp)
            disp_df = _fallback_disposals(raw)

        disp_df = disp_df.copy()
        disp_df["reg"] = disp_df.get("reg", "").map(_norm_id)
        if "event_date" in disp_df.columns:
            disp_df["event_date"] = pd.to_datetime(disp_df["event_date"], errors="coerce").dt.normalize()
        st_ok = _basic_stats(disp_df, "event_date")
        _preview_block("Выбытие → disposals_raw", disp_df, st_ok)

    st.divider()

    can_load = (calv_df is not None) and (ins_df is not None) and (disp_df is not None)
    if not can_load:
        st.info("Загрузи все 3 файла, чтобы появилась кнопка загрузки в БД.")
        return

    if st.button("Загрузить в БД", use_container_width=True, key="tab3_btn_load"):
        try:
            load_calvings_to_db(calv_df, if_exists=if_exists)
            load_inseminations_to_db(ins_df, if_exists=if_exists)
            load_disposals_to_db(disp_df, if_exists=if_exists)
        except Exception as e:
            st.error(f"Ошибка загрузки в БД: {e}")
            st.stop()

        st.success(f"Готово ✅ (mode={if_exists}). Таблицы в БД обновлены.")
