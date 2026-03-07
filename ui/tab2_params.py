from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pandas as pd
import streamlit as st

from core.params import get_param_source, apply_admin_overrides


Token = Union[str, int]


def _dict_get_any_key(d: Any, key: Token) -> Any:
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    if isinstance(key, int):
        return d.get(str(key))
    if isinstance(key, str) and key.isdigit():
        return d.get(int(key))
    return None


@dataclass
class Spec:
    path: str
    tokens: List[Token]
    group_ru: str
    name_ru: str
    editable: bool = True


def _get_by_tokens(d: dict, tokens: List[Token]) -> Any:
    cur: Any = d
    for t in tokens:
        if isinstance(cur, dict):
            cur = _dict_get_any_key(cur, t)
            continue
        if isinstance(t, str) and hasattr(cur, t):
            cur = getattr(cur, t)
            continue
        return None
    return cur


def _set_by_tokens(d: dict, tokens: List[Token], value: Any) -> None:
    cur: Any = d
    for t in tokens[:-1]:
        key = t
        if isinstance(cur, dict) and isinstance(t, int) and str(t) in cur and t not in cur:
            key = str(t)
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]

    leaf = tokens[-1]
    if isinstance(cur, dict) and isinstance(leaf, int) and str(leaf) in cur and leaf not in cur:
        leaf = str(leaf)
    cur[leaf] = value


def _tokens_to_path(tokens: List[Token]) -> str:
    return ".".join(str(t) for t in tokens)


def _lact_label(k: int) -> str:
    if k in (1, 2, 3):
        return f"{k}-я лактация"
    return "4-я и старше"


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Да" if v else "Нет"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
                                                  
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def _parse(text: Any, ref: Any) -> Any:
    """
    Приведение значения из таблицы к нужному типу (int/float/bool/str).
    """
    if text is None:
        return None

    if isinstance(text, (int, float, bool)):
        return text

    s = str(text).strip()
    if s == "" or s == "—":
        return None

    if isinstance(ref, bool):
        s2 = s.lower()
        if s2 in ("да", "yes", "true", "1", "y"):
            return True
        if s2 in ("нет", "no", "false", "0", "n"):
            return False
        return bool(s)

    if isinstance(ref, int) and not isinstance(ref, bool):
        try:
            return int(float(s.replace(",", ".")))
        except Exception:
            return ref

    if isinstance(ref, float):
        try:
            return float(s.replace(",", "."))
        except Exception:
            return ref

    s2 = s.lower()
    if s2 in ("да", "нет", "true", "false", "yes", "no", "1", "0"):
        return s2 in ("да", "true", "yes", "1")
    try:
        if "." in s or "," in s:
            return float(s.replace(",", "."))
        return int(s)
    except Exception:
        return s


def _iter_groups(df_all: pd.DataFrame) -> List[tuple[str, pd.DataFrame]]:
    groups: List[tuple[str, pd.DataFrame]] = []
    for grp in pd.unique(df_all["Группа"]):
        groups.append((str(grp), df_all[df_all["Группа"] == grp].copy()))
    return groups


def _render_grouped_readonly(df_all: pd.DataFrame) -> None:
    for grp_name, grp_df in _iter_groups(df_all):
        st.markdown(f"### {grp_name}")
        st.dataframe(
            grp_df[["Параметр", "Значение"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


def _build_specs(final: dict) -> List[Spec]:
    specs: List[Spec] = []

    def add(group_ru: str, tokens: List[Token], name_ru: str, editable: bool = True):
        specs.append(Spec(path=_tokens_to_path(tokens), tokens=tokens, group_ru=group_ru, name_ru=name_ru, editable=editable))

                        
    cp = final.get("CONCEPTION_PARAMS")
    if isinstance(cp, dict):
        add("Стельность", ["CONCEPTION_PARAMS", "avg_cow_dim_global"], "Коровы: средний DIM наступления стельности (дн.)")
        add("Стельность", ["CONCEPTION_PARAMS", "avg_heifer_age_days"], "Тёлки: средний возраст наступления стельности (дн.)")

        by_lact = cp.get("avg_cow_dim_by_lact")
        if isinstance(by_lact, dict):
            for k in sorted(by_lact.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                try:
                    kk = int(k)
                except Exception:
                    continue
                add(
                    "Стельность",
                    ["CONCEPTION_PARAMS", "avg_cow_dim_by_lact", kk],
                    f"Коровы: DIM наступления стельности — {_lact_label(kk)} (дн.)",
                )

                   
    if "GESTATION_DAYS" in final:
        add("Сроки", ["GESTATION_DAYS"], "Средняя длительность стельности (дн.)")

    if "DRY_DAYS" in final:
        add("Сроки", ["DRY_DAYS"], "Средняя длительность сухостоя (дн.)")
    elif "DRY_DAYS_AVG" in final:
        add("Сроки", ["DRY_DAYS_AVG"], "Средняя длительность сухостоя (дн.)")

                        
    ip = final.get("INSEMINATION_PARAMS")
    if isinstance(ip, dict):
        add("Осеменения", ["INSEMINATION_PARAMS", "cow_services_per_conception"], "Коровы: осеменений до стельности (P), среднее (раз)")
        add("Осеменения", ["INSEMINATION_PARAMS", "cow_ai_interval_days"], "Коровы: интервал между осеменениями, средний (дн.)")

        add("Осеменения", ["INSEMINATION_PARAMS", "heifer_services_per_conception"], "Тёлки: осеменений до стельности (P), среднее (раз)")
        add("Осеменения", ["INSEMINATION_PARAMS", "heifer_ai_interval_days"], "Тёлки: интервал между осеменениями, средний (дн.)")
        add("Осеменения", ["INSEMINATION_PARAMS", "heifer_first_ai_age_days"], "Тёлки: возраст первого осеменения, средний (дн.)")

        first_dim = ip.get("cow_first_ai_dim_by_lact")
        if isinstance(first_dim, dict):
            for k in sorted(first_dim.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                try:
                    kk = int(k)
                except Exception:
                    continue
                add(
                    "Осеменения",
                    ["INSEMINATION_PARAMS", "cow_first_ai_dim_by_lact", kk],
                    f"Коровы: DIM первого осеменения после отёла — {_lact_label(kk)} (дн.)",
                )

                     
    if "ANNUAL_DISPOSAL_RATE" in final:
        add("Выбытие", ["ANNUAL_DISPOSAL_RATE"], "Среднегодовой процент выбытия коров (доля)")

    dp = final.get("DISPOSAL_PARAMS")
    if isinstance(dp, dict):
        by_lact = dp.get("by_lact")
        if isinstance(by_lact, dict):
            for k in sorted(by_lact.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                try:
                    kk = int(k)
                except Exception:
                    continue
                add("Выбытие", ["DISPOSAL_PARAMS", "by_lact", kk, "median_dim"], f"Выбытие: DIM (медиана) — {_lact_label(kk)} (дн.)")
                add("Выбытие", ["DISPOSAL_PARAMS", "by_lact", kk, "mean_dim"], f"Выбытие: DIM (среднее) — {_lact_label(kk)} (дн.)")

                                                                 
                by_lact_kk = _dict_get_any_key(by_lact, kk)
                if isinstance(by_lact_kk, dict) and "n" in by_lact_kk:
                    add("Выбытие", ["DISPOSAL_PARAMS", "by_lact", kk, "n"], f"Выбытие: кол-во наблюдений — {_lact_label(kk)} (гол.)", editable=False)
                if isinstance(by_lact_kk, dict) and "share" in by_lact_kk:
                    add("Выбытие", ["DISPOSAL_PARAMS", "by_lact", kk, "share"], f"Выбытие: доля наблюдений — {_lact_label(kk)} (доля)", editable=False)

        overall = dp.get("overall")
        if isinstance(overall, dict):
            if "median_dim" in overall:
                add("Выбытие", ["DISPOSAL_PARAMS", "overall", "median_dim"], "Выбытие: DIM (медиана) — все лактации (дн.)", editable=False)
            if "mean_dim" in overall:
                add("Выбытие", ["DISPOSAL_PARAMS", "overall", "mean_dim"], "Выбытие: DIM (среднее) — все лактации (дн.)", editable=False)
            if "n" in overall:
                add("Выбытие", ["DISPOSAL_PARAMS", "overall", "n"], "Выбытие: кол-во наблюдений — всего (гол.)", editable=False)

                                                  
    sus = final.get("SEMEN_USAGE_SHARES")
    if isinstance(sus, dict):
        if "cow_trad" in sus:
            add("Семя (использование) — Коровы", ["SEMEN_USAGE_SHARES", "cow_trad"], "Доля обычного (доля)")
        if "cow_sex" in sus:
            add("Семя (использование) — Коровы", ["SEMEN_USAGE_SHARES", "cow_sex"], "Доля сексированного (доля)")
        if "heifer_trad" in sus:
            add("Семя (использование) — Тёлки", ["SEMEN_USAGE_SHARES", "heifer_trad"], "Доля обычного (доля)")
        if "heifer_sex" in sus:
            add("Семя (использование) — Тёлки", ["SEMEN_USAGE_SHARES", "heifer_sex"], "Доля сексированного (доля)")

                                                        
    ssr = final.get("SEMEN_SEX_RATIOS")
    if isinstance(ssr, dict):
        trad = ssr.get("trad")
        sex = ssr.get("sex")

        trad_has = isinstance(trad, dict) or hasattr(trad, "bull_share") or hasattr(trad, "heifer_share")
        sex_has = isinstance(sex, dict) or hasattr(sex, "bull_share") or hasattr(sex, "heifer_share")

        if trad_has:
            add("Пол телят", ["SEMEN_SEX_RATIOS", "trad", "bull_share"], "Обычное семя: доля бычков (доля)")
            add("Пол телят", ["SEMEN_SEX_RATIOS", "trad", "heifer_share"], "Обычное семя: доля тёлочек (доля)")
        if sex_has:
            add("Пол телят", ["SEMEN_SEX_RATIOS", "sex", "bull_share"], "Сексированное семя: доля бычков (доля)")
            add("Пол телят", ["SEMEN_SEX_RATIOS", "sex", "heifer_share"], "Сексированное семя: доля тёлочек (доля)")

                                     
    if "HERD_CAPACITY" in final:
        add("Вместимость", ["HERD_CAPACITY"], "Вместимость стада (лимит поголовья, гол.)", editable=False)

    return specs


def render_tab2_params() -> None:
    st.subheader("Параметры модели")

    base = get_param_source()
    final = apply_admin_overrides(base)

    specs = _build_specs(final)

                                  
    rows = []
    for s in specs:
        v = _get_by_tokens(final, s.tokens)
        rows.append(
            {
                "_path": s.path,
                "Группа": s.group_ru,
                "_editable": s.editable,
                "Параметр": s.name_ru,
                "Значение": _fmt(v),
            }
        )

    df_all = pd.DataFrame(rows, columns=["_path", "Группа", "_editable", "Параметр", "Значение"])
    if df_all.empty:
        st.warning("Параметры не найдены: проверьте загрузку данных и формат вычисленных параметров.")
        return

                                          
    if not bool(st.session_state.get("is_admin", False)):
        _render_grouped_readonly(df_all)
        return

                                                                        
    st.divider()

                               
    has_overrides = isinstance(st.session_state.get("runtime_overrides"), dict) and bool(st.session_state.get("runtime_overrides"))
    if has_overrides:
        st.warning("Активны админ-правки: прогноз считается с учётом изменённых параметров.")

    c1, c2 = st.columns([1, 1])
    with c1:
        edit_mode = st.toggle("Редактировать", value=False, key="tab2_edit_mode")
    with c2:
        if st.button("Сбросить правки", use_container_width=True, key="tab2_reset_overrides"):
            st.session_state.pop("runtime_overrides", None)
            st.success("Админ-правки сброшены.")
            st.rerun()
    if not edit_mode:
        _render_grouped_readonly(df_all)
        return

                                                                    
    editable_paths = set(df_all[df_all["_editable"] == True]["_path"].tolist())
    edited_blocks: List[pd.DataFrame] = []
    for idx, (grp_name, grp_df) in enumerate(_iter_groups(df_all)):
        st.markdown(f"### {grp_name}")
        grp_idx = grp_df.set_index("_path", drop=True)[["Параметр", "Значение"]].copy()
        edited_grp = st.data_editor(
            grp_idx,
            use_container_width=True,
            hide_index=True,
            disabled=["Параметр"],
            column_config={"Значение": st.column_config.TextColumn("Значение")},
            key=f"tab2_editor_{idx}",
        )
        edited_blocks.append(edited_grp)

    if st.button("Сохранить и применить к прогнозу", use_container_width=True, key="tab2_save_apply"):
        overrides: Dict[str, Any] = {}
        spec_by_path: Dict[str, Spec] = {s.path: s for s in specs}

        changed = 0
        for edited in edited_blocks:
            for path, row in edited.iterrows():
                spec = spec_by_path.get(path)
                if not spec:
                    continue
                if path not in editable_paths:
                    continue

                new_raw = row["Значение"]

                base_val = _get_by_tokens(base, spec.tokens)
                final_val = _get_by_tokens(final, spec.tokens)
                ref = base_val if base_val is not None else final_val

                parsed = _parse(new_raw, ref)

                same = False
                if base_val is None and parsed is None:
                    same = True
                elif isinstance(base_val, (int, float)) and isinstance(parsed, (int, float)):
                    same = float(base_val) == float(parsed)
                else:
                    same = base_val == parsed

                if not same:
                    _set_by_tokens(overrides, spec.tokens, parsed)
                    changed += 1

        st.session_state["runtime_overrides"] = overrides if overrides else {}
        st.success(f"Готово. Изменено параметров: {changed}.")
        st.rerun()
