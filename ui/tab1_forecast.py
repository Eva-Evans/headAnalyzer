from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from forecast import compute_forecast_from_db
from forecast_dynamic import compute_forecast_dynamic_from_tables, latest_data_date
from core.calving_facts import actual_birth_stats_from_tables, is_calving_month_complete_from_tables
from core.constants import INDICATORS, OVERFLOW_COLS, OVERFLOW_GROUP_COLS, INDICATOR_TO_OVERFLOW
from core.helpers import month_end, iter_month_ends, ensure_month_col, get_max_event_date_from_db, norm_label, vals_get
from core.params import (
    apply_admin_overrides,
    clear_model_params_cache_for_subdivision,
    compute_params_from_db,
    get_or_compute_subdivision_params,
    get_param_source,
    save_params_cache_current_signature,
)
from core.realization import build_early_realization_plan
from core.excel_export import make_excel_bytes_highlight_months_columns
from ui.styles import fmt_cell, style_positive_red, BAD
from ui.tab3_farm import (
    _ensure_capacity_table,
    _farm_name_for_subdivision,
    _load_farm_tables_from_db,
    _prepare_capacity_editor_df_for_subdivision,
    _render_subdivision_capacity_editor_block,
    _subdivision_status_df_from_db,
)
from ui.tab3_farm_parts.compute import _actual_nonbirth_snapshot_from_tables

from etl.bulls import read_bulls_txt, load_bulls_to_db
from etl.calvings_births import read_calvings_excel, load_calvings_to_db
from etl.disposals import read_disposals_excel, load_disposals_to_db
from etl.dryoff import read_dryoff_excel, load_dryoff_to_db
from etl.inseminations import read_inseminations_excel, clean_inseminations, load_inseminations_to_db
from db import engine
import traceback


BACKTEST_BIRTH_TARGETS: list[str] = [
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]

BACKTEST_TARGETS_SUBDIVISION: list[str] = list(INDICATORS) + [
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]

PERCENT_TARGETS = {
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
}

BIRTH_BACKTEST_TARGETS = {
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
}


def _backtest_percent_error(pred_val: float, fact_val: float, *, is_pct: bool) -> float | None:
    err_abs = abs(float(pred_val) - float(fact_val))
    if is_pct:
        return (err_abs / abs(float(fact_val)) * 100.0) if abs(float(fact_val)) > 1e-9 else None
    scale = abs(float(pred_val)) + abs(float(fact_val))
    if scale < 20.0:
        return None
    return 200.0 * err_abs / scale


def _norm_sex_marker(x: Any) -> str | None:
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


def _actual_birth_stats_month(month_end_date: date) -> dict[str, float]:
    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    calv_sql = """
    SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
    """
    ins_sql = """
    SELECT reg, lact, event_date
    FROM inseminations_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    calv_df = pd.read_sql(text(calv_sql), con=engine, params={"m_start": m_start, "m_next": m_next})
    if calv_df.empty:
        return {k: 0.0 for k in BACKTEST_BIRTH_TARGETS}
    ins_df = pd.read_sql(text(ins_sql), con=engine, params={"m_end": month_end_date})
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=None)


def _actual_metric_month(month_end_date: date, metric_name: str) -> float:
    return float(_actual_birth_stats_month(month_end_date).get(metric_name, 0.0))


def _is_birth_event_type_series(s: pd.Series) -> pd.Series:
    ev = s.astype("string").fillna("").str.upper().str.replace("Ё", "Е", regex=False)
    return ev.str.contains("РОЖ|BORN|BIRTH", regex=True, na=False)


def _actual_birth_stats_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame | None,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=as_of_date)


def _actual_metric_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame | None,
    dry_df: pd.DataFrame | None,
    disp_df: pd.DataFrame | None,
    month_end_date: date,
    metric_name: str,
) -> float:
    if metric_name in BIRTH_BACKTEST_TARGETS or metric_name in PERCENT_TARGETS:
        return float(_actual_birth_stats_month_from_tables(calv_df, ins_df, month_end_date).get(metric_name, 0.0))
    snapshot = _actual_nonbirth_snapshot_from_tables(
        calv_df,
        ins_df if isinstance(ins_df, pd.DataFrame) else pd.DataFrame(),
        dry_df if isinstance(dry_df, pd.DataFrame) else pd.DataFrame(),
        disp_df if isinstance(disp_df, pd.DataFrame) else pd.DataFrame(),
        month_end_date,
    )
    return float(snapshot.get(metric_name, 0.0))


def _pred_metric_value(pred_vals: dict, metric_name: str, nmap: dict[str, float]) -> float:
    if metric_name in PERCENT_TARGETS:
        pred_bull = float(vals_get(pred_vals, "Ожидаемые бычки", nmap) or 0.0)
        pred_heif = float(vals_get(pred_vals, "Ожидаемые тёлочки", nmap) or 0.0)
        den = pred_bull + pred_heif
        if den <= 0:
            return 0.0
        if metric_name == "Доля бычков среди рождений, %":
            return pred_bull / den * 100.0
        return pred_heif / den * 100.0
    return float(vals_get(pred_vals, metric_name, nmap) or 0.0)


def _is_fact_month_complete(month_end_date: date) -> bool:
    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    sql = """
    SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
    """
    df = pd.read_sql(text(sql), con=engine, params={"m_start": m_start, "m_next": m_next})
    return is_calving_month_complete_from_tables(df, month_end_date)


def _is_fact_month_complete_from_tables(calv_df: pd.DataFrame, month_end_date: date) -> bool:
    return is_calving_month_complete_from_tables(calv_df, month_end_date)


def _is_target_fact_month_complete_from_tables(
    tables: dict[str, pd.DataFrame],
    metric_name: str,
    month_end_date: date,
) -> bool:
    if metric_name in BIRTH_BACKTEST_TARGETS or metric_name in PERCENT_TARGETS:
        return is_calving_month_complete_from_tables(tables.get("calv", pd.DataFrame()), month_end_date)
    try:
        return bool(latest_data_date(tables) >= month_end_date)
    except Exception:
        return False


def _month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return month_end(int(ts.year), int(ts.month))


def _rewind_fileobj(file_obj: Any) -> None:
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass


def _canon_name(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    s = " ".join(s.split())
    return s

def _canon_farm_name(x: Any) -> str:
    s = _canon_name(x)
    if not s:
        return ""
    if s == "ENALB":
        return "ЭНАЛБ"
    if s.startswith("БОДЕЕВ") or s.startswith("BODEEV"):
        return "ЭНАЛБ"
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


def _extract_scope_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    if not isinstance(df, pd.DataFrame):
        return set()
    if "__farm" not in df.columns and "__subdivision" not in df.columns:
        return set()

    farm_series = (
        df["__farm"].astype("string").fillna("").str.strip()
        if "__farm" in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype="string")
    )
    sub_series = (
        df["__subdivision"].astype("string").fillna("").str.strip()
        if "__subdivision" in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype="string")
    )

    pairs: set[tuple[str, str]] = set()
    for farm_val, sub_val in zip(farm_series.tolist(), sub_series.tolist()):
        farm = _canon_farm_name(farm_val)
        sub = _canon_subdivision_name(sub_val)
        if not farm and not sub:
            continue
        if not farm:
            farm = "ХОЗЯЙСТВО_НЕ_УКАЗАНО"
        if not sub:
            sub = "__ALL__"
        pairs.add((farm, sub))
    return pairs


def _scope_map_from_pairs(pairs: set[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for farm, sub in pairs:
        out.setdefault(farm, set()).add(sub)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _filter_by_scope(
    df: pd.DataFrame,
    farm_name: str,
    subdivision_name: str,
) -> tuple[pd.DataFrame, int, int, bool, str | None]:
    if not isinstance(df, pd.DataFrame):
        return df, 0, 0, False, "farm"

    n_before = int(len(df))
    out = df.copy()
    farm = _canon_farm_name(farm_name)
    subdivision = _canon_subdivision_name(subdivision_name)

    if farm:
        if "__farm" not in out.columns:
            return df, n_before, n_before, False, "farm"
        s_farm = out["__farm"].map(_canon_farm_name)
        out = out.loc[s_farm == farm].copy()

    if subdivision and subdivision != "__ALL__":
        if "__subdivision" not in out.columns:
            return out, n_before, int(len(out)), False, "subdivision"
        s_sub = out["__subdivision"].map(_canon_subdivision_name)
        out = out.loc[s_sub == subdivision].copy()

    return out, n_before, int(len(out)), True, None


def _drop_subdivision_meta(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    return df.drop(columns=["__farm", "__subdivision"], errors="ignore")


def _probe_subdivision_options(
    calvings_file: Any,
    disposals_file: Any,
    dryoff_file: Any,
    inseminations_file: Any,
) -> tuple[dict[str, list[str]], list[str]]:
    sets: list[set[tuple[str, str]]] = []
    errors: list[str] = []

    probes = [
        ("Отёлы + родившиеся", calvings_file, read_calvings_excel),
        ("Выбытие", disposals_file, read_disposals_excel),
        ("Запуски", dryoff_file, read_dryoff_excel),
        ("Осеменения", inseminations_file, read_inseminations_excel),
    ]

    for label, fobj, reader in probes:
        if fobj is None:
            continue
        try:
            _rewind_fileobj(fobj)
            df = reader(fobj, include_meta=True)
            vals = _extract_scope_pairs(df)
            if vals:
                sets.append(vals)
        except Exception as e:
            errors.append(f"{label}: {e}")
        finally:
            _rewind_fileobj(fobj)

    if not sets:
        return {}, errors

    common = set.intersection(*sets)
    if common:
        return _scope_map_from_pairs(common), errors
    return _scope_map_from_pairs(set.union(*sets)), errors


def _clear_tab1_outputs_state() -> None:
    for k in (
        "last_result_df",
        "last_overflow_df",
        "last_month_ends",
        "last_realization_view",
        "last_result_scope_label",
        "last_excel_bytes",
        "backtest_df",
        "backtest_cfg",
    ):
        st.session_state.pop(k, None)


def _tab1_runtime_overrides_for_selected_subdivision(subdivision_name: str | None) -> dict[str, Any]:
    sub = str(subdivision_name or "").strip()
    by_scope = st.session_state.get("runtime_overrides_by_scope")
    if isinstance(by_scope, dict):
        if sub:
            cand = by_scope.get(f"sub:{sub}")
            if isinstance(cand, dict):
                return cand
        global_ov = by_scope.get("__global__")
        if isinstance(global_ov, dict):
            return global_ov
    return {}


def _tab1_farm_name_for_subdivision(subdivision_name: str) -> str:
    return str(_farm_name_for_subdivision(subdivision_name) or "").strip()


def _render_tab1_subdivision_capacity_panel(subdivision_name: str) -> None:
    sub = str(subdivision_name or "").strip()
    if not sub:
        return
    _ensure_capacity_table()
    farm_name = _tab1_farm_name_for_subdivision(sub)
    st.markdown("**Скотоместа выбранного подразделения**")
    if not farm_name:
        st.caption("Не удалось определить хозяйство для выбранного подразделения.")
        return
    st.caption(f"Хозяйство: {farm_name}. Значения хранятся в БД по группам животных.")
    if bool(st.session_state.get("is_admin", False)):
        st.caption("В админ-режиме можно менять количество мест по каждой группе.")
        _render_subdivision_capacity_editor_block(
            farm_name=farm_name,
            subdivision=sub,
            key_scope="tab1_panel",
        )
        return
    view_df = _prepare_capacity_editor_df_for_subdivision(farm_name, sub)
    st.dataframe(view_df, use_container_width=True, hide_index=True)


def render_tab1_forecast() -> None:
    st.subheader("Источник данных")
    single_subdivision_mode = False
    selected_farm = ""
    selected_subdivision = ""
    db_subdivision_mode = False
    db_selected_subdivision = ""
    db_selected_last_data_date: date | None = None
    db_ready_subs_set: set[str] = set()

    data_mode = st.radio(
        "Откуда брать данные для расчёта?",
        options=[
            "Использовать данные из БД (по умолчанию)",
            "Обновить БД из файлов",
        ],
        index=0,
        horizontal=True,
        key="data_mode_radio",
    )
    need_files = data_mode.startswith("Обновить БД")

    if not need_files:
        try:
            sub_status = _subdivision_status_df_from_db()
        except Exception:
            sub_status = pd.DataFrame()
        if isinstance(sub_status, pd.DataFrame) and not sub_status.empty and "Статус" in sub_status.columns:
            work = sub_status.copy()
            farm_counts = work.groupby("Хозяйство")["Подразделение"].count().to_dict()
            if "Последняя дата данных" in work.columns:
                last_dates = pd.to_datetime(work["Последняя дата данных"], errors="coerce").dt.date
                sub_last_date_map = dict(zip(work["Подразделение"].astype(str).tolist(), last_dates.tolist()))
            else:
                sub_last_date_map = {}
            work["display"] = work.apply(
                lambda r: (
                    f"{str(r['Хозяйство'])} / "
                    f"{_subdivision_display_name(str(r['Подразделение']), str(r['Хозяйство']), int(farm_counts.get(str(r['Хозяйство']), 0)))}"
                ),
                axis=1,
            )
            work = work.sort_values(["Хозяйство", "Подразделение"]).reset_index(drop=True)
            all_subs = work["Подразделение"].astype(str).tolist()
            label_map = dict(zip(all_subs, work["display"].tolist()))
            ready_subs = work.loc[work["Статус"].astype(str) == "готово", "Подразделение"].astype(str).tolist()
            db_ready_subs_set = set(ready_subs)

            if ready_subs:
                db_subdivision_mode = True
                prev_sel = str(st.session_state.get("tab1_prev_db_subdivision", "") or "")
                default_sub = prev_sel if prev_sel in db_ready_subs_set else ready_subs[0]
                db_selected_subdivision = st.selectbox(
                    "Сначала выбери готовое подразделение из БД",
                    options=ready_subs,
                    index=ready_subs.index(default_sub),
                    format_func=lambda x: label_map.get(x, str(x)),
                    key="tab1_db_subdivision_select",
                )
                db_selected_last_data_date = sub_last_date_map.get(str(db_selected_subdivision))
                if prev_sel and prev_sel != str(db_selected_subdivision):
                    _clear_tab1_outputs_state()
                st.session_state["tab1_prev_db_subdivision"] = str(db_selected_subdivision)
            else:
                db_subdivision_mode = False
                db_selected_subdivision = ""
                st.warning("В БД нет готовых подразделений для расчёта. Сначала догрузи недостающие данные.")
        else:
            st.warning("В БД пока нет подразделений для расчёта. Сначала загрузи данные в разделе хозяйств.")
    else:

        st.subheader("Загрузка файлов (для обновления БД)")
        col1, col2 = st.columns(2)
        with col1:
            calvings_file = st.file_uploader("Отёлы + родившиеся", type=["xls", "xlsx"], key="u_calvings")
            disposals_file = st.file_uploader("Выбытие", type=["xls", "xlsx"], key="u_disposals")
        with col2:
            dryoff_file = st.file_uploader("Запуски", type=["xls", "xlsx"], key="u_dryoff")
            inseminations_file = st.file_uploader("Осеменения", type=["xls", "xlsx"], key="u_inseminations")
        bulls_file = st.file_uploader("Таблица быков (txt)", type=["txt"], key="u_bulls")

        single_subdivision_mode = st.checkbox(
            "Загрузить только одно подразделение из файла хозяйства",
            value=False,
            key="tab1_single_subdivision_mode",
            help="Если файл содержит несколько подразделений, можно выбрать одно и загрузить только его строки.",
        )

        if single_subdivision_mode:
            has_all_main = all(x is not None for x in (calvings_file, disposals_file, dryoff_file, inseminations_file))
            if has_all_main:
                scope_options, probe_errors = _probe_subdivision_options(
                    calvings_file=calvings_file,
                    disposals_file=disposals_file,
                    dryoff_file=dryoff_file,
                    inseminations_file=inseminations_file,
                )
                if scope_options:
                    farms = sorted(scope_options.keys())
                    selected_farm = st.selectbox(
                        "Хозяйство из файла",
                        options=farms,
                        index=0,
                        key="tab1_farm_select",
                    )
                    sub_options = scope_options.get(selected_farm, [])
                    real_subs = [x for x in sub_options if x != "__ALL__"]
                    if real_subs:
                        selected_subdivision = st.selectbox(
                        "Подразделение из файла хозяйства",
                        options=real_subs,
                        index=0,
                        key="tab1_subdivision_select",
                    )
                    else:
                        selected_subdivision = "__ALL__"
                else:
                    selected_farm = st.text_input(
                        "Хозяйство (вручную)",
                        key="tab1_farm_manual",
                        help="Если автоопределение не сработало, можно ввести название хозяйства вручную.",
                    ).strip()
                    selected_subdivision = st.text_input(
                        "Подразделение (вручную, опционально)",
                        key="tab1_subdivision_manual",
                        help="Если автоопределение не сработало, можно ввести название вручную.",
                    ).strip()
                if probe_errors:
                    with st.expander("Диагностика определения подразделений", expanded=False):
                        st.write("\n".join(f"- {x}" for x in probe_errors))

    if not need_files:
        calvings_file = None
        disposals_file = None
        dryoff_file = None
        inseminations_file = None
        bulls_file = None

    st.subheader("Месяц прогноза")
    months_ru = [
        "01 — Январь", "02 — Февраль", "03 — Март", "04 — Апрель",
        "05 — Май", "06 — Июнь", "07 — Июль", "08 — Август",
        "09 — Сентябрь", "10 — Октябрь", "11 — Ноябрь", "12 — Декабрь",
    ]

    today = date.today()
    sel_col1, sel_col2 = st.columns([1, 1])
    with sel_col1:
        year_sel = st.number_input("Год", min_value=2000, max_value=2100, value=today.year, step=1, key="year_sel")
    with sel_col2:
        month_sel_label = st.selectbox("Месяц", months_ru, index=today.month - 1, key="month_sel_label")
        month_sel = int(month_sel_label.split("—")[0].strip())

    target_month_end = month_end(int(year_sel), int(month_sel))

    st.subheader("Расчёт")
    clear_sub_disabled = not (
        (not need_files)
        and db_subdivision_mode
        and (db_selected_subdivision or "").strip()
    )
    clear_cache_sub = st.button(
        "Очистить кэш выбранного подразделения",
        key="tab1_clear_cache_sub",
        use_container_width=True,
        disabled=clear_sub_disabled,
        help="Доступно в режиме БД после выбора подразделения.",
    )

    if clear_cache_sub:
        sub = str(db_selected_subdivision or "").strip()
        removed = clear_model_params_cache_for_subdivision(sub)
        _clear_tab1_outputs_state()
        st.success(f"Кэш подразделения «{sub}» очищен: удалено записей в БД — {removed}.")

    last_data_caption = "Последняя дата данных: нет данных"
    if not need_files:
        if (db_selected_subdivision or "").strip():
            if db_selected_last_data_date is None:
                last_data_caption = "Последняя дата данных: нет данных"
            else:
                last_data_caption = f"Последняя дата данных: {pd.Timestamp(db_selected_last_data_date).strftime('%Y-%m-%d')}"
        else:
            last_data_caption = "Последняя дата данных: выбери подразделение"
    else:
        try:
            db_last_dt = get_max_event_date_from_db()
            last_data_caption = f"Последняя дата данных в БД: {pd.Timestamp(db_last_dt).strftime('%Y-%m-%d')}"
        except Exception:
            last_data_caption = "Последняя дата данных в БД: нет данных"
    st.caption(last_data_caption)
    if (not need_files) and db_subdivision_mode and (db_selected_subdivision or "").strip():
        _render_tab1_subdivision_capacity_panel(db_selected_subdivision)
    calculate = st.button("Рассчитать прогноз", key="btn_calc_forecast", use_container_width=True)

    st.session_state.setdefault("last_result_df", None)
    st.session_state.setdefault("last_overflow_df", None)
    st.session_state.setdefault("last_month_ends", None)
    st.session_state.setdefault("last_excel_bytes", None)
    st.session_state.setdefault("last_realization_view", None)
    st.session_state.setdefault("backtest_df", None)
    st.session_state.setdefault("backtest_cfg", None)

    if calculate:
        if (not need_files) and not (db_selected_subdivision or "").strip():
            st.error("Сначала выбери готовое подразделение из БД для расчёта прогноза.")
            st.stop()
        if (not need_files) and db_selected_subdivision and (db_selected_subdivision not in db_ready_subs_set):
            st.error(f"Подразделение «{db_selected_subdivision}» не готово: нужен полный набор данных.")
            st.stop()

        if need_files:
            missing = []
            if calvings_file is None: missing.append("Отёлы + родившиеся")
            if disposals_file is None: missing.append("Выбытие")
            if dryoff_file is None: missing.append("Запуски")
            if inseminations_file is None: missing.append("Осеменения")
            if missing:
                st.error("Не все файлы загружены: " + ", ".join(missing))
                st.stop()
            if single_subdivision_mode and not (selected_farm or "").strip():
                st.error("Выбери хозяйство для фильтрации данных из файла.")
                st.stop()

            try:
                calv_df = read_calvings_excel(calvings_file, include_meta=single_subdivision_mode)
                if single_subdivision_mode:
                    calv_df, n0, n1, applied, missing = _filter_by_scope(calv_df, selected_farm, selected_subdivision)
                    if not applied:
                        if missing == "farm":
                            st.error("В файле 'Отёлы + родившиеся' нет колонки хозяйства (Source.Name / Хозяйство).")
                        else:
                            st.error("В файле 'Отёлы + родившиеся' нет колонки подразделения (Столбец1 / Подразделение).")
                        st.stop()
                    if n1 == 0:
                        target_lbl = f"хозяйству «{selected_farm}»" + (
                            "" if selected_subdivision in {"", "__ALL__"} else f", подразделению «{selected_subdivision}»"
                        )
                        st.error(f"В файле 'Отёлы + родившиеся' нет строк по {target_lbl}.")
                        st.stop()
                calv_df = _drop_subdivision_meta(calv_df)
                load_calvings_to_db(calv_df, if_exists="replace")
            except Exception as e:
                st.error(f"Ошибка при загрузке 'Отёлы + родившиеся': {e}")
                st.stop()

            try:
                disp_df = read_disposals_excel(disposals_file, include_meta=single_subdivision_mode)
                if single_subdivision_mode:
                    disp_df, n0, n1, applied, missing = _filter_by_scope(disp_df, selected_farm, selected_subdivision)
                    if not applied:
                        if missing == "farm":
                            st.error("В файле 'Выбытие' нет колонки хозяйства (Source.Name / Хозяйство).")
                        else:
                            st.error("В файле 'Выбытие' нет колонки подразделения (Столбец1 / Подразделение).")
                        st.stop()
                    if n1 == 0:
                        target_lbl = f"хозяйству «{selected_farm}»" + (
                            "" if selected_subdivision in {"", "__ALL__"} else f", подразделению «{selected_subdivision}»"
                        )
                        st.error(f"В файле 'Выбытие' нет строк по {target_lbl}.")
                        st.stop()
                disp_df = _drop_subdivision_meta(disp_df)
                load_disposals_to_db(disp_df, if_exists="replace")
            except Exception as e:
                st.error(f"Ошибка при загрузке 'Выбытие': {e}")
                st.stop()

            try:
                dry_df = read_dryoff_excel(dryoff_file, include_meta=single_subdivision_mode)
                if single_subdivision_mode:
                    dry_df, n0, n1, applied, missing = _filter_by_scope(dry_df, selected_farm, selected_subdivision)
                    if not applied:
                        if missing == "farm":
                            st.error("В файле 'Запуски' нет колонки хозяйства (Source.Name / Хозяйство).")
                        else:
                            st.error("В файле 'Запуски' нет колонки подразделения (Столбец1 / Подразделение).")
                        st.stop()
                    if n1 == 0:
                        target_lbl = f"хозяйству «{selected_farm}»" + (
                            "" if selected_subdivision in {"", "__ALL__"} else f", подразделению «{selected_subdivision}»"
                        )
                        st.error(f"В файле 'Запуски' нет строк по {target_lbl}.")
                        st.stop()
                dry_df = _drop_subdivision_meta(dry_df)
                load_dryoff_to_db(dry_df, if_exists="replace")
            except Exception as e:
                st.error(f"Ошибка при загрузке 'Запуски': {e}")
                st.stop()

            try:
                ins_df = read_inseminations_excel(inseminations_file, include_meta=single_subdivision_mode)
                if single_subdivision_mode:
                    ins_df, n0, n1, applied, missing = _filter_by_scope(ins_df, selected_farm, selected_subdivision)
                    if not applied:
                        if missing == "farm":
                            st.error("В файле 'Осеменения' нет колонки хозяйства (Source.Name / Хозяйство).")
                        else:
                            st.error("В файле 'Осеменения' нет колонки подразделения (Столбец1 / Подразделение).")
                        st.stop()
                    if n1 == 0:
                        target_lbl = f"хозяйству «{selected_farm}»" + (
                            "" if selected_subdivision in {"", "__ALL__"} else f", подразделению «{selected_subdivision}»"
                        )
                        st.error(f"В файле 'Осеменения' нет строк по {target_lbl}.")
                        st.stop()
                ins_df = _drop_subdivision_meta(ins_df)
                ins_df = clean_inseminations(ins_df)
                load_inseminations_to_db(ins_df, if_exists="replace")
            except Exception as e:
                st.error(f"Ошибка при загрузке 'Осеменения': {e}")
                st.stop()

            if bulls_file is not None:
                try:
                    bulls_df = read_bulls_txt(bulls_file)
                    load_bulls_to_db(bulls_df, if_exists="replace")
                except Exception as e:
                    st.error(f"Ошибка при разборе 'Таблица быков': {e}")
                    st.stop()

            try:
                with st.spinner("Пересчитываю параметры из загруженных данных..."):
                    params = compute_params_from_db()
                    st.session_state.computed_params = params
                    save_params_cache_current_signature(params)
            except Exception as e:
                st.error(f"Не удалось пересчитать параметры из данных: {e}")
                st.stop()

        if (not need_files) and db_subdivision_mode and (db_selected_subdivision or "").strip():
            try:
                selected_tables = _load_farm_tables_from_db(db_selected_subdivision)
                base_date = latest_data_date(selected_tables)
            except Exception as e:
                st.error(f"Не удалось загрузить данные подразделения «{db_selected_subdivision}»: {e}")
                st.stop()
        else:
            base_date = get_max_event_date_from_db()
        base_month_end = month_end(base_date.year, base_date.month)

        if target_month_end < base_month_end:
            month_ends = [target_month_end]
        else:
            month_ends = iter_month_ends(base_date.year, base_date.month, target_month_end.year, target_month_end.month)

        st.markdown(f"**Период прогноза:** {month_ends[0].strftime('%m.%Y')} → {month_ends[-1].strftime('%m.%Y')}")

        if (not need_files) and db_subdivision_mode and (db_selected_subdivision or "").strip():
            try:
                base_params = get_or_compute_subdivision_params(db_selected_subdivision, selected_tables)
            except Exception:
                base_params = get_param_source()
        else:
            base_params = get_param_source()
        scoped_ov = _tab1_runtime_overrides_for_selected_subdivision(
            db_selected_subdivision if ((not need_files) and db_subdivision_mode) else None
        )
        final_params_for_forecast = apply_admin_overrides(base_params, runtime_overrides=scoped_ov)

        rows: list[dict] = []
        overflow_rows: list[dict] = []

        prog = st.progress(0.0)
        with st.spinner("Считаю прогноз..."):
            for i, d_end in enumerate(month_ends, start=1):
                try:
                    if (not need_files) and db_subdivision_mode and (db_selected_subdivision or "").strip():
                        vals = compute_forecast_dynamic_from_tables(
                            selected_tables,
                            d_end,
                            overrides=final_params_for_forecast,
                        ) or {}
                    else:
                        vals = compute_forecast_from_db(d_end, overrides=final_params_for_forecast) or {}
                except Exception as e:
                    st.error(f"Ошибка расчёта на {d_end.strftime('%Y-%m')}: {e}")
                    st.code(traceback.format_exc())                                         
                    st.stop()                             


                nmap = {norm_label(k2): v2 for k2, v2 in (vals or {}).items()}

                row = {"Месяц": d_end.strftime("%Y-%m")}
                for k in INDICATORS:
                    row[k] = vals_get(vals, k, nmap)
                rows.append(row)

                ov_row = {"Месяц": d_end.strftime("%Y-%m")}
                for k in OVERFLOW_COLS:
                    v = vals_get(vals, k, nmap)
                    ov_row[k] = 0.0 if v is None else v
                overflow_rows.append(ov_row)

                prog.progress(i / max(1, len(month_ends)))
        prog.empty()

        result_df = ensure_month_col(pd.DataFrame(rows), month_labels=[r.get("Месяц", "") for r in rows]).set_index("Месяц")
        overflow_df = ensure_month_col(pd.DataFrame(overflow_rows), month_labels=[r.get("Месяц", "") for r in overflow_rows]).set_index("Месяц")

        st.session_state["last_result_df"] = result_df
        st.session_state["last_overflow_df"] = overflow_df
        st.session_state["last_month_ends"] = month_ends
        if (not need_files) and db_subdivision_mode and (db_selected_subdivision or "").strip():
            st.session_state["last_result_scope_label"] = f"Подразделение: {db_selected_subdivision}"
        else:
            st.session_state["last_result_scope_label"] = "Источник: общая БД"

        st.subheader("План ранней реализации (рекомендация)")
        lead_months = st.slider(
            "За сколько месяцев заранее продавать нетелей, если прогноз показывает переполнение по коровам",
            min_value=0, max_value=6, value=2, step=1, key="realization_lead_months",
        )
        realization_df = build_early_realization_plan(overflow_df, lead_months=int(lead_months))
        realization_view = realization_df.T
        st.session_state["last_realization_view"] = realization_view

        st.dataframe(
            realization_view.style.format(fmt_cell).apply(style_positive_red, axis=None),
            use_container_width=True,
        )

    result = st.session_state.get("last_result_df")
    overflow_df = st.session_state.get("last_overflow_df")
    month_ends = st.session_state.get("last_month_ends")
    realization_view = st.session_state.get("last_realization_view")

    show_backtesting = st.checkbox(
        "Показать backtesting (историческая проверка)",
        value=False,
        key="tab1_show_backtesting",
    )
    if show_backtesting:
        st.subheader("Backtesting (историческая проверка)")
        bt_use_subdivision_mode = bool(
            (not need_files)
            and db_subdivision_mode
            and (db_selected_subdivision or "").strip()
        )
        if bt_use_subdivision_mode:
            st.caption(f"Режим: выбранное подразделение «{db_selected_subdivision}».")
        else:
            st.caption("Режим: общая БД.")
        bt_target_options = BACKTEST_TARGETS_SUBDIVISION if bt_use_subdivision_mode else BACKTEST_BIRTH_TARGETS
        prev_bt_target = str(st.session_state.get("bt_target_metric") or "")
        bt_target_index = bt_target_options.index(prev_bt_target) if prev_bt_target in bt_target_options else 0
        bt_target = st.selectbox(
            "Показатель для backtesting",
            bt_target_options,
            index=bt_target_index,
            key="bt_target_metric",
        )

        c_bt1, c_bt2 = st.columns(2)
        with c_bt1:
            bt_months = st.slider(
                "Глубина истории (сколько последних месяцев проверяем)",
                min_value=3,
                max_value=24,
                value=6,
                step=1,
                key="bt_months",
            )
        with c_bt2:
            bt_horizon = st.slider(
                "Горизонт (на сколько месяцев раньше ставим as-of)",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key="bt_horizon",
            )
        bt_complete_only = st.checkbox(
            "Учитывать только полные месяцы факта (рекомендуется)",
            value=True,
            key="bt_complete_only",
        )

        if st.button("Запустить backtesting", key="btn_run_backtest", use_container_width=True):
            bt_tables: dict[str, pd.DataFrame] | None = None
            if bt_use_subdivision_mode:
                try:
                    bt_tables = _load_farm_tables_from_db(db_selected_subdivision)
                    base_date_bt = latest_data_date(bt_tables)
                except Exception as e:
                    st.error(f"Не удалось загрузить данные подразделения «{db_selected_subdivision}» для backtesting: {e}")
                    st.stop()
            else:
                base_date_bt = get_max_event_date_from_db()
            last_me_bt = month_end(base_date_bt.year, base_date_bt.month)
            target_months = [_month_end_shift(last_me_bt, -i) for i in range(bt_months - 1, -1, -1)]
            unit = "pct" if bt_target in PERCENT_TARGETS else "heads"
            skipped_incomplete = 0

            if bt_use_subdivision_mode and bt_tables is not None:
                try:
                    bt_base_params = get_or_compute_subdivision_params(db_selected_subdivision, bt_tables)
                except Exception:
                    bt_base_params = get_param_source()
            else:
                bt_base_params = get_param_source()
            bt_scoped_ov = _tab1_runtime_overrides_for_selected_subdivision(
                db_selected_subdivision if bt_use_subdivision_mode else None
            )
            bt_params = apply_admin_overrides(bt_base_params, runtime_overrides=bt_scoped_ov)
            rows_bt: list[dict] = []
            prog_bt = st.progress(0.0)

            for i, target_me in enumerate(target_months, start=1):
                if bt_use_subdivision_mode and bt_tables is not None:
                    is_complete = _is_target_fact_month_complete_from_tables(bt_tables, bt_target, target_me)
                else:
                    is_complete = _is_fact_month_complete(target_me)
                if bt_complete_only and not is_complete:
                    skipped_incomplete += 1
                    prog_bt.progress(i / max(1, len(target_months)))
                    continue

                as_of_me = _month_end_shift(target_me, -int(bt_horizon))
                if bt_use_subdivision_mode and bt_tables is not None:
                    pred_vals = compute_forecast_dynamic_from_tables(
                        bt_tables,
                        target_me,
                        overrides=bt_params,
                        as_of_date=as_of_me,
                    ) or {}
                else:
                    pred_vals = compute_forecast_from_db(target_me, overrides=bt_params, as_of_date=as_of_me) or {}
                nmap = {norm_label(k): v for k, v in pred_vals.items()}
                pred_val = float(_pred_metric_value(pred_vals, bt_target, nmap))
                if bt_use_subdivision_mode and bt_tables is not None:
                    fact_val = float(
                        _actual_metric_month_from_tables(
                            bt_tables.get("calv", pd.DataFrame()),
                            bt_tables.get("ins", pd.DataFrame()),
                            bt_tables.get("dry", pd.DataFrame()),
                            bt_tables.get("disp", pd.DataFrame()),
                            target_me,
                            bt_target,
                        )
                    )
                else:
                    fact_val = float(_actual_metric_month(target_me, bt_target))
                err = pred_val - fact_val
                ape = _backtest_percent_error(pred_val, fact_val, is_pct=(bt_target in PERCENT_TARGETS))

                rows_bt.append(
                    {
                        "Месяц факта": target_me.strftime("%Y-%m"),
                        "as-of (на дату)": as_of_me.strftime("%Y-%m"),
                        "Показатель": bt_target,
                        "Прогноз": round(pred_val, 1),
                        "Факт": round(fact_val, 1),
                        "Ошибка": round(err, 1),
                        "APE, %": None if ape is None else round(float(ape), 1),
                        "Полный месяц факта": bool(is_complete),
                    }
                )
                prog_bt.progress(i / max(1, len(target_months)))
            prog_bt.empty()

            st.session_state["backtest_df"] = pd.DataFrame(rows_bt)
            st.session_state["backtest_cfg"] = {
                "months": int(bt_months),
                "horizon": int(bt_horizon),
                "metric": bt_target,
                "unit": unit,
                "scope": f"subdivision:{db_selected_subdivision}" if bt_use_subdivision_mode else "db:all",
                "complete_only": bool(bt_complete_only),
                "skipped_incomplete": int(skipped_incomplete),
            }

        bt_df = st.session_state.get("backtest_df")
        bt_cfg = st.session_state.get("backtest_cfg") or {}
        if isinstance(bt_df, pd.DataFrame) and not bt_df.empty:
            expected_scope = f"subdivision:{db_selected_subdivision}" if bt_use_subdivision_mode else "db:all"
            cfg_changed = (
                str(bt_cfg.get("metric") or "") != str(bt_target)
                or str(bt_cfg.get("scope") or "") != expected_scope
                or int(bt_cfg.get("months", 0) or 0) != int(bt_months)
                or int(bt_cfg.get("horizon", 0) or 0) != int(bt_horizon)
                or bool(bt_cfg.get("complete_only", True)) != bool(bt_complete_only)
            )
            if cfg_changed:
                st.info("Параметры backtesting изменены. Нажми «Запустить backtesting», чтобы пересчитать метрики.")
            else:
                if "Прогноз" not in bt_df.columns and "Прогноз, гол." in bt_df.columns:
                    bt_df = bt_df.rename(columns={"Прогноз, гол.": "Прогноз"})
                if "Прогноз" not in bt_df.columns and "Прогноз отёлов, гол." in bt_df.columns:
                    bt_df = bt_df.rename(columns={"Прогноз отёлов, гол.": "Прогноз"})
                if "Факт" not in bt_df.columns and "Факт, гол." in bt_df.columns:
                    bt_df = bt_df.rename(columns={"Факт, гол.": "Факт"})
                if "Факт" not in bt_df.columns and "Факт отёлов, гол." in bt_df.columns:
                    bt_df = bt_df.rename(columns={"Факт отёлов, гол.": "Факт"})
                if "Ошибка" not in bt_df.columns and "Ошибка, гол." in bt_df.columns:
                    bt_df = bt_df.rename(columns={"Ошибка, гол.": "Ошибка"})
                st.session_state["backtest_df"] = bt_df

                is_pct = bool(bt_cfg.get("unit") == "pct")
                mae_label = "Средняя погрешность, п.п." if is_pct else "Средняя погрешность, гол."
                bias_label = "Смещение (средняя ошибка), п.п." if is_pct else "Смещение (средняя ошибка), гол."

                bt_metrics = bt_df.copy()
                bt_metrics["Ошибка"] = pd.to_numeric(bt_metrics.get("Ошибка"), errors="coerce")
                bt_metrics["Факт"] = pd.to_numeric(bt_metrics.get("Факт"), errors="coerce")
                bt_metrics["Прогноз"] = pd.to_numeric(bt_metrics.get("Прогноз"), errors="coerce")

                mae = float(bt_metrics["Ошибка"].abs().mean())
                bias = float(bt_metrics["Ошибка"].mean())
                if is_pct:
                    perc_series = pd.to_numeric(bt_metrics["APE, %"], errors="coerce").dropna()
                    perc_err = float(perc_series.mean()) if not perc_series.empty else None
                    perc_label = "Средняя процентная погрешность, %"
                else:
                    scale = bt_metrics["Прогноз"].abs() + bt_metrics["Факт"].abs()
                    stable_mask = scale >= 20.0
                    den = float(scale.loc[stable_mask].sum())
                    num = float(bt_metrics.loc[stable_mask, "Ошибка"].abs().sum())
                    perc_err = (200.0 * num / den) if den > 1e-9 else None
                    perc_label = "Симметричная процентная погрешность, %"

                m1, m2, m3 = st.columns(3)
                m1.metric(mae_label, f"{mae:.1f}")
                m2.metric(perc_label, "—" if perc_err is None else f"{perc_err:.1f}")
                m3.metric(bias_label, f"{bias:.1f}")

                st.dataframe(bt_df, use_container_width=True, hide_index=True)
                chart_df = bt_df.set_index("Месяц факта")[["Прогноз", "Факт"]]
                st.line_chart(chart_df)
    if not isinstance(result, pd.DataFrame) or result.empty:
        st.info("Нажми «Рассчитать прогноз», чтобы увидеть таблицы.")
        return

    forecast_view = result.T
    overflow_groups_only = overflow_df.reindex(columns=[c for c in OVERFLOW_GROUP_COLS if c in overflow_df.columns])
    overflow_view = overflow_groups_only.T

    def style_forecast_months_as_columns(df_view: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df_view.index, columns=df_view.columns)
        for ind in df_view.index:
            ov_name = INDICATOR_TO_OVERFLOW.get(str(ind))
            if not ov_name:
                continue
            if ov_name not in overflow_df.columns:
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
        forecast_view.style.format(fmt_cell).apply(style_forecast_months_as_columns, axis=None),
        use_container_width=True,
    )

    st.subheader("Переполнение по группам ")
    st.dataframe(
        overflow_view.style.format(fmt_cell).apply(style_positive_red, axis=None),
        use_container_width=True,
    )

    st.subheader("Скачать результат (Excel)")
    excel_bytes = make_excel_bytes_highlight_months_columns(
        forecast_view=forecast_view,
        overflow_view=overflow_view,
        indicator_to_overflow=INDICATOR_TO_OVERFLOW,
        realization_view=realization_view,
    )
    st.session_state["last_excel_bytes"] = excel_bytes

    if isinstance(month_ends, list) and month_ends:
        file_name = f"herd_forecast_{month_ends[0].strftime('%Y-%m')}_to_{month_ends[-1].strftime('%Y-%m')}.xlsx"
    else:
        file_name = "herd_forecast.xlsx"

    st.download_button(
        label="Скачать Excel: прогноз + переполнение + план реализации",
        data=excel_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_excel",
    )
