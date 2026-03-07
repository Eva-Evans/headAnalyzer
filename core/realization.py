from __future__ import annotations

import pandas as pd

def build_early_realization_plan(
    overflow_df: pd.DataFrame,
    *,
    lead_months: int = 2,
) -> pd.DataFrame:
    if not isinstance(overflow_df, pd.DataFrame) or overflow_df.empty:
        return pd.DataFrame()

    months = [str(x) for x in overflow_df.index.tolist()]

    def _get(m: str, col: str) -> float:
        if col not in overflow_df.columns:
            return 0.0
        try:
            v = pd.to_numeric(overflow_df.loc[m, col], errors="coerce")
            return float(0.0 if pd.isna(v) else v)
        except Exception:
            return 0.0

    plan_cols = [
        "Рекомендуем продать: нетели (заранее)",
        "Рекомендуем продать: нетели (в этот месяц)",
        "Рекомендуем продать: тёлки 9–24 мес",
        "Рекомендуем продать: тёлки 3–8 мес",
        "Рекомендуем продать: коровы (крайний случай)",
    ]
    plan = pd.DataFrame(0.0, index=months, columns=plan_cols)

    for i, m in enumerate(months):
        over_cows = _get(m, "Переполнение: Дойные коровы") + _get(m, "Переполнение: Сухостойные коровы")
        if over_cows > 0:
            j = i - int(lead_months)
            if j >= 0:
                plan.iloc[j, plan.columns.get_loc("Рекомендуем продать: нетели (заранее)")] += over_cows
            else:
                plan.loc[m, "Рекомендуем продать: коровы (крайний случай)"] += over_cows

        ov_heif_38 = _get(m, "Переполнение: Тёлки 3–8 мес")
        ov_heif_924 = _get(m, "Переполнение: Тёлки 9–24 мес")
        ov_preg = _get(m, "Переполнение: Нетели")

        if ov_preg > 0:
            plan.loc[m, "Рекомендуем продать: нетели (в этот месяц)"] += ov_preg
        if ov_heif_924 > 0:
            plan.loc[m, "Рекомендуем продать: тёлки 9–24 мес"] += ov_heif_924
        if ov_heif_38 > 0:
            plan.loc[m, "Рекомендуем продать: тёлки 3–8 мес"] += ov_heif_38

    plan = plan.where(plan.abs() >= 1e-6, 0.0)
    return plan
