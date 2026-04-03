from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

#from db import engine
from db_cloud import engine

from model_params import (
    GESTATION_DAYS,
    DRY_DAYS,
    CONCEPTION_PARAMS,
    DISPOSAL_PARAMS,
    ANNUAL_DISPOSAL_RATE,
    SEMEN_USAGE_PROBS,
    SEMEN_SEX_RATIOS,
    INSEMINATION_PARAMS,
    HERD_CAPACITY,
)

if TYPE_CHECKING:
    from model_params.defaults import SemenSexRatio  

import re
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class SemenSexRatio:
    bull_share: float
    heifer_share: float


def _to_semen_ratio(x) -> SemenSexRatio:
    """Приводит SEMEN_SEX_RATIOS из model_params к нашему dataclass (на всякий случай)."""
    if isinstance(x, SemenSexRatio):
        return x
    if hasattr(x, "bull_share") and hasattr(x, "heifer_share"):
        return SemenSexRatio(float(x.bull_share), float(x.heifer_share))
    if isinstance(x, dict):
        b = float(x.get("bull_share", 0.5))
        h = float(x.get("heifer_share", 1.0 - b))
        return SemenSexRatio(b, h)
    return SemenSexRatio(0.5, 0.5)

def _merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str | None = None,
    direction: str = "backward",
    allow_exact_matches: bool = True,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    if direction != "backward":
        raise ValueError("Эта версия fallback реализует только direction='backward'")

    l = left.copy()
    r = right.copy()

    l["_row_id"] = np.arange(len(l), dtype=np.int64)

    # on -> datetime64[ns]
    l[left_on] = pd.to_datetime(l[left_on], errors="coerce")
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce")

    if by is None:
        l = l[l[left_on].notna()].copy()
        r = r[r[right_on].notna()].copy()

        l = l.sort_values([left_on], kind="mergesort").reset_index(drop=True)
        r = r.sort_values([right_on], kind="mergesort").reset_index(drop=True)

        out = pd.merge_asof(
            l, r,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    # --- by нормализация ---
    l[by] = l[by].astype("string").fillna("").str.strip()
    r[by] = r[by].astype("string").fillna("").str.strip()

    l = l[(l[by] != "") & l[left_on].notna()].copy()
    r = r[(r[by] != "") & r[right_on].notna()].copy()

    # КЛЮЧЕВОЕ: сначала сортируем по времени (on), потом по by
    l = l.sort_values([left_on, by], kind="mergesort").reset_index(drop=True)
    r = r.sort_values([right_on, by], kind="mergesort").reset_index(drop=True)

    try:
        out = pd.merge_asof(
            l, r,
            by=by,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    except ValueError as e:
        # если всё равно "keys must be sorted" -> быстрый fallback
        if "keys must be sorted" not in str(e).lower():
            raise

        # ---- FAST fallback: numpy searchsorted по каждой группе ----
        # Подготовим "пустые" колонки справа (как после merge)
        right_cols = [c for c in r.columns if c not in l.columns or c == by or c == right_on]
        # но by/right_on уже есть в left, поэтому при назначении суффиксы не нужны (у нас right берётся только нужные колонки обычно)
        # В твоём use-case ты и так передаёшь right[[reg_s, ins_dt, semen]] и т.п.

        out = l.copy()
        for c in r.columns:
            if c not in out.columns:
                out[c] = pd.NA

        # Для ускорения: сгруппируем right по by ОДИН раз
        # r уже отсортирован по [right_on, by], но нам удобнее по by
        # (внутри группы right_on не обязательно монотонен после фильтра, поэтому ещё раз гарантируем)
        r_groups = {}
        for key, grp in r.groupby(by, sort=False):
            gg = grp.sort_values(right_on, kind="mergesort")
            r_groups[key] = gg

        # left тоже группируем
        for key, lg in out.groupby(by, sort=False):
            rg = r_groups.get(key)
            if rg is None or rg.empty:
                continue

            # int64 нанoseconds
            lt = lg[left_on].values.astype("datetime64[ns]").astype("int64")
            rt = rg[right_on].values.astype("datetime64[ns]").astype("int64")

            # позиции последнего rt <= lt
            pos = np.searchsorted(rt, lt, side="right") - 1
            ok = pos >= 0
            if not np.any(ok):
                continue

            # индексы строк в out, куда писать
            out_idx = lg.index.values[ok]
            take = rg.iloc[pos[ok]]

            # переносим все колонки right (кроме by и right_on, если они уже есть)
            for c in rg.columns:
                if c == by:
                    continue
                # right_on может совпадать с left_on по смыслу, но обычно у тебя разные имена ("ins_dt")
                # если колонка уже есть — перезапишем только если это не left_on
                if c in out.columns and c == left_on:
                    continue
                out.loc[out_idx, c] = take[c].values

        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

def _resolve_runtime_params(overrides: dict | None) -> dict:
    ov = overrides or {}

    # ---- defaults из model_params ----
    gest_default = float(GESTATION_DAYS)
    dry_default = int(DRY_DAYS)

    cp = ov.get("CONCEPTION_PARAMS") or {
        "avg_cow_dim_by_lact": dict(CONCEPTION_PARAMS.avg_cow_dim_by_lact),
        "avg_cow_dim_global": float(CONCEPTION_PARAMS.avg_cow_dim_global),
        "avg_heifer_age_days": float(CONCEPTION_PARAMS.avg_heifer_age_days),
    }

    disp = ov.get("DISPOSAL_PARAMS") or deepcopy(DISPOSAL_PARAMS)
    annual_disp = float(ov.get("ANNUAL_DISPOSAL_RATE", ANNUAL_DISPOSAL_RATE))

    ins = ov.get("INSEMINATION_PARAMS") or {
        "cow_services_per_conception": float(INSEMINATION_PARAMS.cow_services_per_conception),
        "cow_ai_interval_days": float(INSEMINATION_PARAMS.cow_ai_interval_days),
        "cow_first_ai_dim_by_lact": dict(INSEMINATION_PARAMS.cow_first_ai_dim_by_lact),
        "heifer_services_per_conception": float(INSEMINATION_PARAMS.heifer_services_per_conception),
        "heifer_ai_interval_days": float(INSEMINATION_PARAMS.heifer_ai_interval_days),
        "heifer_first_ai_age_days": float(INSEMINATION_PARAMS.heifer_first_ai_age_days),
    }

    # опционально: можно вручную задать доли семени; если нет — возьмём из БД как раньше
    semen_usage = ov.get("SEMEN_USAGE_SHARES")  # {"cow_trad":..,"cow_sex":..,"heifer_trad":..,"heifer_sex":..}

    # ---- normalize ----
    gest_days = int(round(float(ov.get("GESTATION_DAYS", gest_default))))
    gest_days = max(200, min(310, gest_days))

    dry_days = int(round(float(ov.get("DRY_DAYS", dry_default))))
    dry_days = max(20, min(120, dry_days))

    annual_disp = float(max(0.0, min(0.5, annual_disp)))

    return {
        "GESTATION_DAYS": gest_days,
        "DRY_DAYS": dry_days,
        "CONCEPTION_PARAMS": cp,
        "DISPOSAL_PARAMS": disp,
        "ANNUAL_DISPOSAL_RATE": annual_disp,
        "INSEMINATION_PARAMS": ins,
        "SEMEN_USAGE_SHARES": semen_usage,
    }

def norm_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    # "12345.0" -> "12345"
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s


# --- настройки модели ---
DRY_DAYS = int(DRY_DAYS)

# VWP берём из рассчитанного среднего DIM первого осеменения после отёла (по лактациям)
VWP_BY_LACT = {
    l: int(round(float(INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(l, 50))))
    for l in (1, 2, 3, 4)
}

# Для тёлок: минимальный возраст, раньше которого стельность не начинаем
HEIFER_VWP_AGE = int(round(float(INSEMINATION_PARAMS.heifer_first_ai_age_days)))

MAX_DIM = 500              # ограничиваем хвост DIM
MAX_AGE_DAYS = 900         # тёлки до ~30 мес
BULL_AGE_MAX = 90          # чтобы считать бычков 0–2 мес

def age_months(d: int) -> int:
    return int(d // 30)

def end_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)

def norm_sex(x: object) -> str | None:
    if x is None:
        return None
    v = str(x).strip().upper()
    if v in ("F", "Ж", "FEMALE"):
        return "F"
    if v in ("M", "М", "MALE"):
        return "M"
    return None

def norm_event_type(x: object) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")
    # чтобы "РОЖДЕНИЕ", "РОЖДЕН", "РОЖД." и т.п. всё стало "РОЖДЕН"
    if "РОЖ" in v:
        return "РОЖДЕН"
    return v

def norm_result(x: object) -> str:
    if x is None:
        return ""
    v = str(x).replace("\xa0", " ").strip().upper().replace("Ё", "Е")

    # канонизируем "плодотворное/стельная" -> P
    if v in {"P", "П"}:
        return "P"
    if "PREG" in v:
        return "P"
    if "СТЕЛ" in v or v in {"СТ", "СТ.", "СТ+", "СТЕЛЬНАЯ", "СТЕЛЬН"}:
        return "P"

    return v


def classify_semen_from_bull_type(bull_type: object) -> str:
    """
    Возвращаем ключи: 'sex' или 'trad' (как в SEMEN_SEX_RATIOS).
    """
    v = "" if bull_type is None else str(bull_type).strip().upper()
    if v == "S" or "SEX" in v:
        return "sex"
    return "trad"

def daily_base_disposal_prob() -> float:
    return 1.0 - (1.0 - float(ANNUAL_DISPOSAL_RATE)) ** (1.0 / 365.0)

def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4

def shift_right(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out

def shift_left(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out

def hazard_from_pdf(pdf: np.ndarray, *, vwp: int = 0) -> np.ndarray:
    """
    pdf -> распределение дня события -> hazard (условная вероятность события сегодня).
    """
    p = pdf.copy().astype(float)
    p[:vwp] = 0.0
    s = p.sum()
    if s <= 0:
        hz = np.zeros_like(p)
        return hz
    p /= s

    hz = np.zeros_like(p)
    surv = 1.0
    for i in range(len(p)):
        if i < vwp:
            hz[i] = 0.0
            continue
        pi = float(p[i])
        if surv <= 1e-12:
            hz[i] = 0.0
        else:
            hz[i] = pi / surv
            hz[i] = max(0.0, min(1.0, hz[i]))
            surv *= (1.0 - hz[i])
    return hz

def gaussian_hazard(length: int, mean: float, sd: float, *, vwp: int = 0) -> np.ndarray:
    x = np.arange(length, dtype=float)
    pdf = np.exp(-0.5 * ((x - mean) / max(sd, 1.0)) ** 2)
    return hazard_from_pdf(pdf, vwp=vwp)

def lognormal_hazard_by_dim(dim_max: int, mean: float, median: float) -> np.ndarray:
    """
    Строим hazard по DIM на основе логнормального распределения DIM выбытия.
    mean/median берём из DISPOSAL_PARAMS.
    """
    # защита от мусора
    mean = float(mean) if mean and mean > 1 else 1.0
    median = float(median) if median and median > 1 else max(1.0, mean * 0.8)

    # если mean < median (бывает от шума) — слегка поправим
    if mean < median:
        mean = median * 1.05

    # lognormal параметры
    sigma2 = 2.0 * np.log(max(1e-9, mean / median))
    sigma = float(np.sqrt(max(1e-9, sigma2)))
    mu = float(np.log(max(1e-9, median)))

    x = np.arange(dim_max + 1, dtype=float)
    pdf = np.zeros_like(x)
    # начинаем с 1 (log(0) нельзя)
    xx = x[1:]
    pdf[1:] = (1.0 / (xx * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((np.log(xx) - mu) ** 2) / (2.0 * sigma2))
    return hazard_from_pdf(pdf, vwp=0)

@dataclass
class HerdState:
    open_dim: Dict[int, np.ndarray]                       # lact_cat -> [0..MAX_DIM]
    preg_lact: Dict[Tuple[int, str], np.ndarray]          # (lact_cat, semen) -> [0..GESTATION_DAYS]
    preg_dry:  Dict[Tuple[int, str], np.ndarray]          # (lact_cat, semen) -> [0..GESTATION_DAYS]
    heifer_age: np.ndarray                                # [0..MAX_AGE_DAYS]
    heifer_preg: Dict[str, np.ndarray]                    # semen -> [0..GESTATION_DAYS]
    bull_age: np.ndarray                                  # [0..BULL_AGE_MAX]

def init_empty_state(gest_days: int) -> HerdState:
    open_dim = {l: np.zeros(MAX_DIM + 1, dtype=float) for l in (1, 2, 3, 4)}
    preg_lact = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    preg_dry  = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    heifer_age = np.zeros(MAX_AGE_DAYS + 1, dtype=float)
    heifer_preg = {s: np.zeros(gest_days + 1, dtype=float) for s in ("trad", "sex")}
    bull_age = np.zeros(BULL_AGE_MAX + 1, dtype=float)
    return HerdState(open_dim, preg_lact, preg_dry, heifer_age, heifer_preg, bull_age)

def load_tables() -> Dict[str, pd.DataFrame]:
    calv = pd.read_sql("SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw", con=engine)
    ins  = pd.read_sql("SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw", con=engine)
    dry  = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", con=engine)
    disp = pd.read_sql("SELECT reg, event_date, disposal_reason FROM disposals_raw", con=engine)
    bulls = pd.read_sql("SELECT bull_code, bull_type FROM bulls_raw", con=engine)
    return {"calv": calv, "ins": ins, "dry": dry, "disp": disp, "bulls": bulls}

def latest_data_date(tables: Dict[str, pd.DataFrame]) -> date:
    mx = None
    for key in ("calv", "ins", "dry", "disp"):
        df = tables[key]
        if "event_date" in df.columns and not df.empty:
            d = pd.to_datetime(df["event_date"], errors="coerce").max()
            if pd.notna(d):
                dx = d.date()
                mx = dx if mx is None else max(mx, dx)
    return mx or date.today()
def compute_semen_usage_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "cow_trad": float(SEMEN_USAGE_PROBS.cow_trad),
        "cow_sex": float(SEMEN_USAGE_PROBS.cow_sex),
        "heifer_trad": float(SEMEN_USAGE_PROBS.heifer_trad),
        "heifer_sex": float(SEMEN_USAGE_PROBS.heifer_sex),
    }

    if ins.empty or bulls.empty:
        return fallback

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    p = ins[(ins["event_date"].notna()) & (ins["result_norm"] == "P") & (ins["bull_s"] != "")].copy()
    if p.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["bull_type_s"] = bulls["bull_type"].astype(str).str.strip().str.upper()
    bulls["semen"] = np.where(bulls["bull_type_s"] == "S", "sex", "trad")
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p["semen_known"] = p["semen"].isin(["trad", "sex"])

    max_dt = p["event_date"].max()
    p_365 = p[p["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy() if pd.notna(max_dt) else p.copy()

    def _shares(df: pd.DataFrame) -> Tuple[float | None, float | None, int, float]:
        if df.empty:
            return None, None, 0, 0.0
        known = df[df["semen_known"]]
        if known.empty:
            return None, None, int(len(df)), 0.0
        total = int(len(known))
        sex = int((known["semen"] == "sex").sum())
        trad = total - sex
        known_rate = float(len(known)) / float(len(df))
        return trad / total, sex / total, total, known_rate

    def _mix(cond_cows: bool) -> Tuple[float, float]:
        all_df = p[p["lact"].gt(0) if cond_cows else p["lact"].le(0)].copy()
        y_df = p_365[p_365["lact"].gt(0) if cond_cows else p_365["lact"].le(0)].copy()

        trad_all, sex_all, n_all, known_rate_all = _shares(all_df)
        trad_y, sex_y, n_y, known_rate_y = _shares(y_df)

        # если мало матчинга быков — fallback
        if (trad_all is None) or (n_all < 200) or (known_rate_all < 0.3):
            return (fallback["cow_trad"], fallback["cow_sex"]) if cond_cows else (fallback["heifer_trad"], fallback["heifer_sex"])

        # вес "последнего года" только если он реально качественный
        w = 0.0
        if (trad_y is not None) and (n_y >= 200) and (known_rate_y >= 0.6):
            w = min(0.7, n_y / 2000.0) * min(1.0, known_rate_y / 0.9)

        sex_final = w * float(sex_y) + (1.0 - w) * float(sex_all)
        sex_final = float(max(0.0, min(1.0, sex_final)))
        trad_final = 1.0 - sex_final
        return trad_final, sex_final

    cow_trad, cow_sex = _mix(True)
    hef_trad, hef_sex = _mix(False)

    def _norm2(a: float, b: float) -> Tuple[float, float]:
        s = max(1e-9, a + b)
        return a / s, b / s

    cow_trad, cow_sex = _norm2(cow_trad, cow_sex)
    hef_trad, hef_sex = _norm2(hef_trad, hef_sex)

    return {
        "cow_trad": float(cow_trad),
        "cow_sex": float(cow_sex),
        "heifer_trad": float(hef_trad),
        "heifer_sex": float(hef_sex),
    }
import logging
from datetime import date
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def report_semen_and_calf_sex_params_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Логирует:
      1) доли использования semen trad/sex (коровы/тёлки) и покрытие матчинга быков
      2) доли пола телят bull/heifer для trad/sex и объём данных (сколько телят/отёлов)
    Возвращает словарь с теми же значениями, чтобы можно было показать в UI.
    """

    # -------------------------
    # A) SEMEN USAGE (trad/sex)
    # -------------------------
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback_usage = {
        "cow_trad": float(SEMEN_USAGE_PROBS.cow_trad),
        "cow_sex": float(SEMEN_USAGE_PROBS.cow_sex),
        "heifer_trad": float(SEMEN_USAGE_PROBS.heifer_trad),
        "heifer_sex": float(SEMEN_USAGE_PROBS.heifer_sex),
    }

    usage_diag = {
        "cow_all_P": 0, "cow_known": 0, "cow_known_rate": 0.0,
        "cow_365_P": 0, "cow_365_known": 0, "cow_365_known_rate": 0.0,
        "heifer_all_P": 0, "heifer_known": 0, "heifer_known_rate": 0.0,
        "heifer_365_P": 0, "heifer_365_known": 0, "heifer_365_known_rate": 0.0,
    }

    semen_shares = dict(fallback_usage)

    if not ins.empty and not bulls.empty:
        ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce")
        ins["result_norm"] = ins["result"].apply(norm_result)
        ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
        ins["bull_s"] = ins["bull"].apply(norm_id)

        p = ins[
            ins["event_date"].notna()
            & (ins["result_norm"] == "P")
            & (ins["bull_s"] != "")
        ].copy()

        if not p.empty:
            bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
            bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type)  # -> "trad"/"sex"
            semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

            p["semen"] = p["bull_s"].map(semen_by_bull)
            p["semen_known"] = p["semen"].isin(["trad", "sex"])

            max_dt = p["event_date"].max()
            p_365 = p[p["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy() if pd.notna(max_dt) else p.copy()

            def _shares(df: pd.DataFrame) -> Tuple[float | None, float | None, int, float]:
                # returns: (trad_share, sex_share, n_known, known_rate_over_all_df)
                if df.empty:
                    return None, None, 0, 0.0
                total_all = int(len(df))
                known = df[df["semen_known"]]
                n_known = int(len(known))
                known_rate = (n_known / total_all) if total_all > 0 else 0.0
                if n_known == 0:
                    return None, None, 0, known_rate
                sex = int((known["semen"] == "sex").sum())
                trad = n_known - sex
                return trad / n_known, sex / n_known, n_known, known_rate

            def _mix(is_cow: bool) -> Tuple[float, float]:
                df_all = p[p["lact"] > 0].copy() if is_cow else p[p["lact"] <= 0].copy()
                df_365 = p_365[p_365["lact"] > 0].copy() if is_cow else p_365[p_365["lact"] <= 0].copy()

                trad_all, sex_all, n_all, known_rate_all = _shares(df_all)
                trad_365, sex_365, n_365, known_rate_365 = _shares(df_365)

                # diagnostics
                if is_cow:
                    usage_diag["cow_all_P"] = int(len(df_all))
                    usage_diag["cow_known"] = int(n_all)
                    usage_diag["cow_known_rate"] = float(known_rate_all)
                    usage_diag["cow_365_P"] = int(len(df_365))
                    usage_diag["cow_365_known"] = int(n_365)
                    usage_diag["cow_365_known_rate"] = float(known_rate_365)
                else:
                    usage_diag["heifer_all_P"] = int(len(df_all))
                    usage_diag["heifer_known"] = int(n_all)
                    usage_diag["heifer_known_rate"] = float(known_rate_all)
                    usage_diag["heifer_365_P"] = int(len(df_365))
                    usage_diag["heifer_365_known"] = int(n_365)
                    usage_diag["heifer_365_known_rate"] = float(known_rate_365)

                # fallback if too little or too poor matching
                if n_all < 200 or known_rate_all < 0.3 or trad_all is None or sex_all is None:
                    if is_cow:
                        return fallback_usage["cow_trad"], fallback_usage["cow_sex"]
                    return fallback_usage["heifer_trad"], fallback_usage["heifer_sex"]

                # weight of last year
                w = 0.0
                if n_365 >= 200 and known_rate_365 >= 0.6 and trad_365 is not None and sex_365 is not None:
                    w = min(0.7, n_365 / 2000.0) * min(1.0, known_rate_365 / 0.9)

                sex_final = w * float(sex_365) + (1.0 - w) * float(sex_all)
                sex_final = float(max(0.0, min(1.0, sex_final)))
                trad_final = 1.0 - sex_final
                return float(trad_final), float(sex_final)

            cow_trad, cow_sex = _mix(True)
            heif_trad, heif_sex = _mix(False)

            # normalize safety
            def _norm2(a: float, b: float) -> Tuple[float, float]:
                s = max(1e-9, a + b)
                return a / s, b / s

            cow_trad, cow_sex = _norm2(cow_trad, cow_sex)
            heif_trad, heif_sex = _norm2(heif_trad, heif_sex)

            semen_shares = {
                "cow_trad": float(cow_trad),
                "cow_sex": float(cow_sex),
                "heifer_trad": float(heif_trad),
                "heifer_sex": float(heif_sex),
            }

    # -------------------------
    # B) CALF SEX RATIOS (by semen)
    # -------------------------
    # тут используем твою уже рабочую функцию (она делает merge_asof и фильтрацию gest_days)
    semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    # а чтобы отчитаться “сколько данных вошло” — пересоберём диагностику с минимумом логики
    sex_diag = {
        "matched_calvings_total": 0,
        "matched_calves_total": 0,
        "trad_matched_calvings": 0,
        "trad_calves_total": 0,
        "sex_matched_calvings": 0,
        "sex_calves_total": 0,
    }

    try:
        calv = tables["calv"].copy()
        ins2 = tables["ins"].copy()
        bulls2 = tables["bulls"].copy()

        if not calv.empty and not ins2.empty and not bulls2.empty:
            calv["event_type"] = calv["event_type"].apply(norm_event_type)
            calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce")
            calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
            calv["sex_norm"] = calv["sex"].apply(norm_sex)

            born = calv[
                (calv["event_type"] == "РОЖДЕН")
                & calv["event_date"].notna()
                & (calv["mother_reg_s"] != "")
                & calv["sex_norm"].isin(["M", "F"])
            ][["mother_reg_s", "event_date", "sex_norm"]].copy()

            if not born.empty:
                born["calving_dt"] = born["event_date"].dt.normalize()
                born["male"] = (born["sex_norm"] == "M").astype(int)
                born["female"] = (born["sex_norm"] == "F").astype(int)

                calv_ev = (
                    born.groupby(["mother_reg_s", "calving_dt"], sort=False)[["male", "female"]]
                    .sum()
                    .reset_index()
                    .rename(columns={"mother_reg_s": "reg_s"})
                )

                ins2["event_date"] = pd.to_datetime(ins2["event_date"], errors="coerce")
                ins2["result_norm"] = ins2["result"].apply(norm_result)
                ins2["reg_s"] = ins2["reg"].apply(norm_id)
                ins2["bull_s"] = ins2["bull"].apply(norm_id)

                p = ins2[
                    ins2["event_date"].notna()
                    & (ins2["result_norm"] == "P")
                    & (ins2["reg_s"] != "")
                    & (ins2["bull_s"] != "")
                ][["reg_s", "event_date", "bull_s"]].copy()

                if not p.empty:
                    bulls2["bull_code_s"] = bulls2["bull_code"].apply(norm_id)
                    bulls2["semen"] = bulls2["bull_type"].apply(classify_semen_from_bull_type)
                    semen_by_bull = dict(zip(bulls2["bull_code_s"], bulls2["semen"]))

                    p["semen"] = p["bull_s"].map(semen_by_bull)
                    p = p[p["semen"].isin(["trad", "sex"])].copy()
                    if not p.empty:
                        p["ins_dt"] = p["event_date"].dt.normalize()

                        left = calv_ev.sort_values(["reg_s", "calving_dt"], kind="mergesort")
                        right = p.sort_values(["reg_s", "ins_dt"], kind="mergesort")

                        m = _merge_asof_safe(
                            left,
                            right[["reg_s", "ins_dt", "semen"]],
                            by="reg_s",
                            left_on="calving_dt",
                            right_on="ins_dt",
                            direction="backward",
                            allow_exact_matches=True,
                        )

                        m = m[m["ins_dt"].notna()].copy()
                        if not m.empty:
                            m["gest_days"] = (m["calving_dt"] - m["ins_dt"]).dt.days
                            m = m[(m["gest_days"] >= 200) & (m["gest_days"] <= 310)].copy()

                            if not m.empty:
                                sex_diag["matched_calvings_total"] = int(len(m))
                                sex_diag["matched_calves_total"] = int((m["male"] + m["female"]).sum())

                                for semen in ("trad", "sex"):
                                    sub = m[m["semen"] == semen]
                                    sex_diag[f"{semen}_matched_calvings"] = int(len(sub))
                                    sex_diag[f"{semen}_calves_total"] = int((sub["male"] + sub["female"]).sum())
    except Exception:
        # отчёт — не должен ронять приложение
        logger.exception("Failed to compute sex ratio diagnostics; main ratios still computed.")

    # -------------------------
    # PRINT / LOG
    # -------------------------
    def pct(x: float) -> str:
        return f"{100.0 * float(x):.1f}%"

    logger.info(
        "SEMEM USAGE (final): cows trad=%s sex=%s | heifers trad=%s sex=%s",
        pct(semen_shares["cow_trad"]), pct(semen_shares["cow_sex"]),
        pct(semen_shares["heifer_trad"]), pct(semen_shares["heifer_sex"]),
    )
    logger.info(
        "SEMEM USAGE coverage: cows P=%s matched=%s (%.1f%%) | last365 P=%s matched=%s (%.1f%%)",
        usage_diag["cow_all_P"], usage_diag["cow_known"], 100.0 * usage_diag["cow_known_rate"],
        usage_diag["cow_365_P"], usage_diag["cow_365_known"], 100.0 * usage_diag["cow_365_known_rate"],
    )
    logger.info(
        "SEMEM USAGE coverage: heifers P=%s matched=%s (%.1f%%) | last365 P=%s matched=%s (%.1f%%)",
        usage_diag["heifer_all_P"], usage_diag["heifer_known"], 100.0 * usage_diag["heifer_known_rate"],
        usage_diag["heifer_365_P"], usage_diag["heifer_365_known"], 100.0 * usage_diag["heifer_365_known_rate"],
    )

    logger.info(
        "CALF SEX RATIOS: trad bull=%s heifer=%s | sex bull=%s heifer=%s",
        pct(semen_sex_ratios["trad"].bull_share), pct(semen_sex_ratios["trad"].heifer_share),
        pct(semen_sex_ratios["sex"].bull_share), pct(semen_sex_ratios["sex"].heifer_share),
    )
    logger.info(
        "CALF SEX RATIOS sample: matched calvings=%s, calves=%s | trad calvings=%s calves=%s | sex calvings=%s calves=%s",
        sex_diag["matched_calvings_total"], sex_diag["matched_calves_total"],
        sex_diag["trad_matched_calvings"], sex_diag["trad_calves_total"],
        sex_diag["sex_matched_calvings"], sex_diag["sex_calves_total"],
    )

    return {
        "semen_shares": semen_shares,
        "semen_usage_diag": usage_diag,
        "semen_sex_ratios": semen_sex_ratios,
        "semen_sex_diag": sex_diag,
    }

def compute_semen_sex_ratios_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, SemenSexRatio]:
    calv = tables["calv"].copy()
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "trad": _to_semen_ratio(SEMEN_SEX_RATIOS["trad"]),
        "sex": _to_semen_ratio(SEMEN_SEX_RATIOS["sex"]),
    }

    if calv.empty or ins.empty or bulls.empty:
        return fallback

    calv["event_type"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce").dt.normalize()
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    calv["sex_norm"] = calv["sex"].apply(norm_sex)

    born = calv[
        (calv["event_type"] == "РОЖДЕН")
        & calv["event_date"].notna()
        & (calv["mother_reg_s"] != "")
        & calv["sex_norm"].isin(["M", "F"])
    ][["mother_reg_s", "event_date", "sex_norm"]].copy()

    if born.empty:
        return fallback

    born["calving_dt"] = born["event_date"]
    born["male"] = (born["sex_norm"] == "M").astype(int)
    born["female"] = (born["sex_norm"] == "F").astype(int)

    calv_ev = (
        born.groupby(["mother_reg_s", "calving_dt"], sort=False)[["male", "female"]]
        .sum()
        .reset_index()
    )

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    p = ins[
        (ins["event_date"].notna())
        & (ins["result_norm"] == "P")
        & (ins["reg_s"] != "")
        & (ins["bull_s"] != "")
    ][["reg_s", "event_date", "bull_s"]].copy()

    if p.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["bull_type_s"] = bulls["bull_type"].astype(str).str.strip().str.upper()
    bulls["semen"] = np.where(bulls["bull_type_s"] == "S", "sex", "trad")
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p = p[p["semen"].isin(["trad", "sex"])].copy()
    if p.empty:
        return fallback

    p = p.rename(columns={"event_date": "ins_dt"})
    left = calv_ev.rename(columns={"mother_reg_s": "reg_s"}).copy()

    m = _merge_asof_safe(
        left,
        p[["reg_s", "ins_dt", "semen"]],
        by="reg_s",
        left_on="calving_dt",
        right_on="ins_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    m = m[m["ins_dt"].notna()].copy()
    if m.empty:
        return fallback

    m["gest_days"] = (m["calving_dt"] - m["ins_dt"]).dt.days
    m = m[(m["gest_days"] >= 200) & (m["gest_days"] <= 310)].copy()
    if m.empty:
        return fallback

    out = dict(fallback)
    for semen in ("trad", "sex"):
        sub = m[m["semen"] == semen]
        total = int(sub["male"].sum() + sub["female"].sum())
        if total < 300:
            continue
        bull_share = float(sub["male"].sum()) / float(total)
        bull_share = max(0.0, min(1.0, bull_share))
        out[semen] = SemenSexRatio(bull_share=bull_share, heifer_share=1.0 - bull_share)

    return out

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _effective_ai_interval_days(interval_raw: float, mean_target: float, first: float, spc: float) -> float:
    """
    Чтобы не ушли в нули и чтобы средняя стельность не “уплыла”:
    - raw interval из данных часто завышен (из-за длинных разрывов)
    - одновременно у нас есть mean_target (средний DIM/возраст стельности)
    Поэтому берём "целевой" интервал, который согласует first + (spc-1)*interval ≈ mean_target,
    и смешиваем его с raw.
    """
    interval_raw = _clamp(float(interval_raw), 14.0, 90.0)
    spc = float(spc)

    if spc <= 1.01:
        return interval_raw

    # интервал, который делает среднюю стельность близкой к наблюдаемой
    derived = (float(mean_target) - float(first)) / max(1e-9, (spc - 1.0))
    derived = _clamp(derived, 14.0, 60.0)

    # raw обычно шумный → даём ему меньший вес
    return 0.30 * interval_raw + 0.70 * derived
def build_initial_state(
    tables: Dict[str, pd.DataFrame],
    as_of: date,
    *,
    gest_days: int | None = None,
    dry_days: int | None = None,
) -> HerdState:
    gest_days = int(gest_days if gest_days is not None else GESTATION_DAYS)
    dry_days = int(dry_days if dry_days is not None else DRY_DAYS)
    # ВАЖНО: всё сравниваем в одном типе (Timestamp normalized)
    as_of_ts = pd.Timestamp(as_of).normalize()

    state = init_empty_state(gest_days)

    calv = tables["calv"].copy()
    ins  = tables["ins"].copy()
    dry  = tables["dry"].copy()
    disp = tables["disp"].copy()
    bulls = tables["bulls"].copy()

    # --- normalize calvings ---
    calv["event_type"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce").dt.normalize()
    calv["birth_date"] = pd.to_datetime(calv["birth_date"], errors="coerce").dt.normalize()

    calv["reg_s"] = calv["reg"].apply(norm_id)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    calv["sex_norm"] = calv["sex"].apply(norm_sex)

    # --- normalize inseminations ---
    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    # --- normalize dryoff ---
    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce").dt.normalize()
    dry["reg_s"] = dry["reg"].apply(norm_id)

    # --- normalize disposals ---
    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.normalize()
    disp["reg_s"] = disp["reg"].apply(norm_id)
    disp["disposal_reason"] = disp["disposal_reason"].astype(str).str.lower().str.replace("ё", "е")

    # --- normalize bulls ---
    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    # --- выбытия до as_of ---
    disposed_regs = set(
        disp.loc[
            disp["event_date"].notna() & (disp["event_date"] <= as_of_ts),
            "reg_s"
        ].astype(str)
    )
    disposed_regs.discard("")

    # --- last dryoff по коровам ---
    dry_ok = dry[(dry["event_date"].notna()) & (dry["event_date"] <= as_of_ts) & (dry["reg_s"] != "")]
    dry_last = dry_ok.groupby("reg_s", sort=False)["event_date"].max().to_dict()

    # --- calving dates через телят: event_type="РОЖДЕН" + mother_reg ---
    calves = calv[
        (calv["event_type"] == "РОЖДЕН")
        & (calv["mother_reg_s"].notna())
        & (calv["mother_reg_s"] != "")
        & (calv["event_date"].notna())
        & (calv["event_date"] <= as_of_ts)
    ][["mother_reg_s", "event_date"]].drop_duplicates()

    calv_stats = None
    if not calves.empty:
        calv_stats = (
            calves.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_date"})
            .groupby("reg_s", sort=False)
            .agg(
                n_calvings=("calving_date", "count"),
                last_calving=("calving_date", "max"),
            )
            .reset_index()
        )

    # --- восстановление last_calving из осеменений (DIM) для коров lact>0 ---
    ins_cow_dim = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] > 0)
        & (ins["dim_age"].notna())
        & (ins["dim_age"] >= 0)
        & (ins["dim_age"] <= MAX_DIM)
    ].copy()

    est_stats = None
    if not ins_cow_dim.empty:
        ins_cow_dim = ins_cow_dim.sort_values(["reg_s", "event_date"], kind="mergesort")
        last_dim_row = ins_cow_dim.groupby("reg_s", sort=False).tail(1).copy()
        last_dim_row["last_calving_est"] = last_dim_row["event_date"] - pd.to_timedelta(last_dim_row["dim_age"], unit="D")
        last_dim_row["lact_cat_est"] = last_dim_row["lact"].clip(lower=1, upper=4)
        est_stats = last_dim_row[["reg_s", "last_calving_est", "lact_cat_est", "dim_age"]].copy()

    # --- собрать список коров (union) ---
    if calv_stats is None and est_stats is None:
        cows = pd.DataFrame(columns=["reg_s", "last_calving", "n_calvings", "last_calving_est", "lact_cat_est", "dim_age"])
    elif calv_stats is None:
        cows = est_stats.copy()
        cows["n_calvings"] = pd.NA
        cows["last_calving"] = pd.NA
    elif est_stats is None:
        cows = calv_stats.copy()
        cows["last_calving_est"] = pd.NA
        cows["lact_cat_est"] = pd.NA
        cows["dim_age"] = pd.NA
    else:
        cows = calv_stats.merge(est_stats, on="reg_s", how="outer")

    cows = cows[(cows["reg_s"].notna()) & (cows["reg_s"] != "")].copy()
    cows = cows[~cows["reg_s"].isin(disposed_regs)].copy()

    # last_calving: факт > оценка
    cows["last_calving"] = cows["last_calving"].where(cows["last_calving"].notna(), cows["last_calving_est"])

    def _lcat(row) -> int:
        if pd.notna(row.get("n_calvings")):
            return lact_cat_from_count(int(row["n_calvings"]))
        if pd.notna(row.get("lact_cat_est")):
            return int(row["lact_cat_est"])
        return 1

    cows["lact_cat"] = cows.apply(_lcat, axis=1)

    # --- последние P-осеменения по reg (коровы) ---
    ins_p = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["result_norm"] == "P")
    ].copy()

    last_p = {}
    last_p_bull = {}
    if not ins_p.empty:
        ins_p = ins_p.sort_values(["reg_s", "event_date"], kind="mergesort")
        tail = ins_p.groupby("reg_s", sort=False).tail(1)
        last_p = dict(zip(tail["reg_s"], tail["event_date"]))
        last_p_bull = dict(zip(tail["reg_s"], tail["bull_s"]))

    # --- расклад коров в состояние ---
    OVERDUE_CLAMP_DAYS = 14  # если чуть "перелетели" срок из-за шумных дат — не теряем беременность

    for r in cows.itertuples(index=False):
        reg = str(r.reg_s)
        lact_cat = int(r.lact_cat)

        last_calv = getattr(r, "last_calving", pd.NaT)
        dim_guess = getattr(r, "dim_age", pd.NA)

        if pd.notna(last_calv):
            dim = int(max(0, min(MAX_DIM, (as_of_ts - pd.Timestamp(last_calv).normalize()).days)))
        else:
            dim = int(max(0, min(MAX_DIM, float(dim_guess))) if pd.notna(dim_guess) else 0)

        dry_last_dt = dry_last.get(reg, pd.NaT)
        is_dry_fact = pd.notna(dry_last_dt) and pd.notna(last_calv) and (pd.Timestamp(dry_last_dt) > pd.Timestamp(last_calv))

        p_date = last_p.get(reg, pd.NaT)
        bull = last_p_bull.get(reg, "") or ""
        semen = semen_by_bull.get(bull, "trad") if bull else "trad"

        if pd.notna(p_date):
            p_date = pd.Timestamp(p_date).normalize()

            # если есть факт отёла и P было ДО него — это не текущая беременность
            if pd.notna(last_calv) and p_date <= pd.Timestamp(last_calv).normalize():
                state.open_dim[lact_cat][dim] += 1.0
                continue

            days_to_calv = int(gest_days - (as_of_ts - p_date).days)

            # мягкий clamp для "чуть просроченных"
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0

            if 0 <= days_to_calv <= gest_days:
                is_dry = is_dry_fact or (days_to_calv <= dry_days)
                if is_dry:
                    state.preg_dry[(lact_cat, semen)][days_to_calv] += 1.0
                else:
                    state.preg_lact[(lact_cat, semen)][days_to_calv] += 1.0
                continue

        state.open_dim[lact_cat][dim] += 1.0

    # --- молодняк по birth_date/sex ---
    animals = calv[(calv["reg_s"].notna()) & (calv["reg_s"] != "")][["reg_s", "birth_date", "sex_norm"]].copy()
    animals = animals.drop_duplicates(subset=["reg_s"], keep="last")
    animals = animals[~animals["reg_s"].isin(disposed_regs)].copy()

    mothers = set(cows["reg_s"].astype(str))

    reg_to_birth = {}
    for rr in animals.itertuples(index=False):
        if pd.notna(rr.birth_date):
            reg_to_birth[str(rr.reg_s)] = pd.Timestamp(rr.birth_date).normalize()

    for rr in animals.itertuples(index=False):
        reg = str(rr.reg_s)
        bd = rr.birth_date
        sx = rr.sex_norm
        if pd.isna(bd) or sx not in ("F", "M"):
            continue
        if reg in mothers:
            continue

        age = int((as_of_ts - pd.Timestamp(bd).normalize()).days)
        if age < 0:
            continue

        if sx == "F":
            a = int(min(MAX_AGE_DAYS, max(0, age)))
            state.heifer_age[a] += 1.0
        else:
            if age <= BULL_AGE_MAX:
                state.bull_age[int(age)] += 1.0

    # --- беременные тёлки из inseminations (lact<=0, P) ---
    heifer_p = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] <= 0)
        & (ins["result_norm"] == "P")
    ].copy()

    heifer_p = heifer_p[~heifer_p["reg_s"].isin(disposed_regs)]
    heifer_p = heifer_p[~heifer_p["reg_s"].isin(mothers)]

    if not heifer_p.empty:
        heifer_p = heifer_p.sort_values(["reg_s", "event_date"], kind="mergesort")
        heifer_last = heifer_p.groupby("reg_s", sort=False).tail(1)

        for rr in heifer_last.itertuples(index=False):
            reg = str(rr.reg_s)
            p_date = rr.event_date
            bull = getattr(rr, "bull_s", "") or ""
            semen = semen_by_bull.get(bull, "trad")

            if pd.isna(p_date):
                continue

            p_date = pd.Timestamp(p_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - p_date).days)

            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0

            if not (0 <= days_to_calv <= gest_days):
                continue

            state.heifer_preg[semen][days_to_calv] += 1.0

            # убрать из heifer_age, если мы её там посчитали по birth_date
            bd = reg_to_birth.get(reg)
            if isinstance(bd, pd.Timestamp):
                age = int(min(MAX_AGE_DAYS, max(0, (as_of_ts - bd).days)))
                state.heifer_age[age] = max(0.0, state.heifer_age[age] - 1.0)

    return state

def build_disposal_shape(disposal_params: dict) -> Dict[int, np.ndarray]:
    shape = {}
    by_lact = disposal_params.get("by_lact", {})
    for lact_cat in (1, 2, 3, 4):
        s = by_lact.get(lact_cat, {})
        m = float(s.get("mean_dim", 150.0) or 150.0)
        md = float(s.get("median_dim", 120.0) or 120.0)
        hz = lognormal_hazard_by_dim(MAX_DIM, m, md)
        nz = hz[hz > 0]
        if nz.size == 0:
            sh = np.ones(MAX_DIM + 1, dtype=float)
        else:
            sh = hz / (float(nz.mean()) if float(nz.mean()) > 0 else 1.0)
        shape[lact_cat] = np.clip(sh, 0.1, 5.0)
    return shape

def _copy_state(s: HerdState) -> HerdState:
    return HerdState(
        open_dim={k: v.copy() for k, v in s.open_dim.items()},
        preg_lact={k: v.copy() for k, v in s.preg_lact.items()},
        preg_dry={k: v.copy() for k, v in s.preg_dry.items()},
        heifer_age=s.heifer_age.copy(),
        heifer_preg={k: v.copy() for k, v in s.heifer_preg.items()},
        bull_age=s.bull_age.copy(),
    )
def _cap(name: str) -> float | None:
    v = HERD_CAPACITY.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _take_from_array(arr: np.ndarray, idx_iter, need: float) -> float:
    taken = 0.0
    for i in idx_iter:
        if need <= 1e-9:
            break
        v = float(arr[i])
        if v <= 0:
            continue
        x = v if v < need else need
        arr[i] = v - x
        taken += x
        need -= x
    return taken


def _sell_cows_from_doy(state: HerdState, need: float, gest_days: int) -> float:
    sold = 0.0
    # 1) сначала open коровы: старые (большой DIM) и старшие лактации
    for l in (4, 3, 2, 1):
        sold += _take_from_array(state.open_dim[l], range(MAX_DIM, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    # 2) потом стельные в лактации: продаём тех, у кого до отёла дальше (большой days_to_calv)
    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_lact[(l, semen)], range(gest_days, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold

    return sold


def _sell_cows_from_dry(state: HerdState, need: float, gest_days: int, dry_days: int) -> float:
    sold = 0.0
    hi = min(dry_days, gest_days)
    # сухостой — это хвост близко к отёлу, но если прям не влезаем — режем от "дальше от отёла" к "ближе"
    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_dry[(l, semen)], range(hi, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold
    return sold


def _sell_heifers_by_age(state: HerdState, need: float, age_lo: int, age_hi: int) -> float:
    sold = 0.0
    lo = max(0, int(age_lo))
    hi = min(int(age_hi), len(state.heifer_age) - 1)
    sold += _take_from_array(state.heifer_age, range(hi, lo - 1, -1), need)
    return sold


def _sell_neteli_4_6_months(state: HerdState, need: float, gest_days: int) -> float:
    """
    Нетелей продаём при стельности 4–6 месяцев:
    это примерно 120–180 дней после осеменения.
    days_to_calv = gest - days_preg  ->  gest-180 .. gest-120  => ~100..160 (при gest≈280)
    """
    sold = 0.0
    pref_lo = max(0, min(gest_days, 100))
    pref_hi = max(0, min(gest_days, 160))

    # 1) сначала 4–6 месяцев (days_to_calv ~ 160..100)
    pref_range = list(range(pref_hi, pref_lo - 1, -1))
    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], pref_range, need - sold)
        if sold >= need - 1e-9:
            return sold

    # 2) если не хватило — режем остальных, но стараемся НЕ трогать "почти отёл" (малые days_to_calv)
    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(gest_days, pref_hi + 1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(pref_lo - 1, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    return sold


def _apply_capacity_month_end(state: HerdState, *, gest_days: int, dry_days: int) -> dict:
    """
    Возвращает сколько "срезали" в этом месяце по группам.
    """
    out = {
        "over_doy": 0.0,
        "over_dry": 0.0,
        "over_h0": 0.0,
        "over_h38": 0.0,
        "over_h9": 0.0,
        "over_neteli": 0.0,
        "sell_cows": 0.0,
        "sell_heifers": 0.0,
        "sell_neteli": 0.0,
    }

    # capacities
    cap_doy = _cap("Дойные коровы")
    cap_dry = _cap("Сухостойные коровы")
    cap_h0 = _cap("Тёлки 0–3 мес")          # в UI это мапится на "Тёлки 0–2 мес"
    cap_h38 = _cap("Тёлки 3–8 мес")
    cap_h9 = _cap("Тёлки 9–24 мес")
    cap_neteli = _cap("Нетели")            # если ключа нет — просто не режем

    # current counts
    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    # heifers by your UI buckets (d//30):
    h0 = float(state.heifer_age[:90].sum())          # 0–2 мес (0..89 дней)
    h38 = float(state.heifer_age[90:270].sum())      # 3–8 мес (90..269)
    h9 = float(state.heifer_age[270:].sum())         # >=9 мес (270+)
    neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

    # 1) DOY cows
    if cap_doy is not None and doy > cap_doy + 1e-9:
        need = doy - cap_doy
        sold = _sell_cows_from_doy(state, need, gest_days)
        out["over_doy"] = sold
        out["sell_cows"] += sold

    # recompute after possible sell
    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    doy = float(cows_open + cows_preg_lact)

    # 2) DRY cows
    if cap_dry is not None and dry > cap_dry + 1e-9:
        need = dry - cap_dry
        sold = _sell_cows_from_dry(state, need, gest_days, dry_days)
        out["over_dry"] = sold
        out["sell_cows"] += sold

    # 3) heifers groups
    if cap_h0 is not None and h0 > cap_h0 + 1e-9:
        need = h0 - cap_h0
        sold = _sell_heifers_by_age(state, need, 0, 89)
        out["over_h0"] = sold
        out["sell_heifers"] += sold

    if cap_h38 is not None and h38 > cap_h38 + 1e-9:
        need = h38 - cap_h38
        sold = _sell_heifers_by_age(state, need, 90, 269)
        out["over_h38"] = sold
        out["sell_heifers"] += sold

    if cap_h9 is not None and h9 > cap_h9 + 1e-9:
        need = h9 - cap_h9
        sold = _sell_heifers_by_age(state, need, 270, MAX_AGE_DAYS)
        out["over_h9"] = sold
        out["sell_heifers"] += sold

    # 4) neteli (если есть отдельная вместимость)
    if cap_neteli is not None and neteli > cap_neteli + 1e-9:
        need = neteli - cap_neteli
        sold = _sell_neteli_4_6_months(state, need, gest_days)
        out["over_neteli"] = sold
        out["sell_neteli"] += sold

    return out

def simulate_to_target(
    state: HerdState,
    *,
    start: date,
    target: date,
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    params: dict,
) -> Tuple[HerdState, Dict[str, float]]:

    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])
    cp = params["CONCEPTION_PARAMS"]
    disp_params = params["DISPOSAL_PARAMS"]
    annual_disp = float(params["ANNUAL_DISPOSAL_RATE"])
    ins = params["INSEMINATION_PARAMS"]

    # счётчики по месяцу target
    target_month = (target.year, target.month)
    calv_total = 0.0
    calv_cows = 0.0
    calv_heifers = 0.0
    exp_bulls = 0.0
    exp_heifers = 0.0

    meta: Dict[str, float] = {
        "cow_doses_total": 0.0, "cow_doses_sex": 0.0, "cow_doses_trad": 0.0,
        "heifer_doses_total": 0.0, "heifer_doses_sex": 0.0, "heifer_doses_trad": 0.0,
    }
    meta.update({
        "sell_cows": 0.0,
        "sell_heifers": 0.0,
        "sell_neteli": 0.0,
        "over_doy": 0.0,
        "over_dry": 0.0,
        "over_h0": 0.0,
        "over_h38": 0.0,
        "over_h9": 0.0,
        "over_neteli": 0.0,
    })

        # --- обработать "отёлы в день start" (если в initial state уже есть bucket 0) ---
    def _process_bucket0_for_day(curr_day: date) -> None:
        nonlocal calv_total, calv_cows, calv_heifers, exp_bulls, exp_heifers

        # коровы
        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                born = state.preg_dry[(l, semen)][0]
                if born > 0:
                    state.preg_dry[(l, semen)][0] = 0.0

                    if (curr_day.year, curr_day.month) == target_month:
                        calv_total += born
                        calv_cows += born
                        sr = semen_sex_ratios[semen]
                        exp_bulls += born * float(sr.bull_share)
                        exp_heifers += born * float(sr.heifer_share)

                    l2 = min(4, l + 1)
                    state.open_dim[l2][0] += born

                    sr = semen_sex_ratios[semen]
                    state.heifer_age[0] += born * float(sr.heifer_share)
                    state.bull_age[0] += born * float(sr.bull_share)

        # нетели
        for semen in ("trad", "sex"):
            born = state.heifer_preg[semen][0]
            if born > 0:
                state.heifer_preg[semen][0] = 0.0

                if (curr_day.year, curr_day.month) == target_month:
                    calv_total += born
                    calv_heifers += born
                    sr = semen_sex_ratios[semen]
                    exp_bulls += born * float(sr.bull_share)
                    exp_heifers += born * float(sr.heifer_share)

                state.open_dim[1][0] += born
                sr = semen_sex_ratios[semen]
                state.heifer_age[0] += born * float(sr.heifer_share)
                state.bull_age[0] += born * float(sr.bull_share)

    _process_bucket0_for_day(start)

    # disposal (масштаб из annual_disp)
    p_disp_day_base = 1.0 - (1.0 - annual_disp) ** (1.0 / 365.0)
    disp_shape = build_disposal_shape(disp_params)

    by_lact = disp_params.get("by_lact", {})
    total_n = float(disp_params.get("overall", {}).get("n", 1) or 1)
    shares = {l: (float(by_lact.get(l, {}).get("n", 0) or 0) / total_n) for l in (1,2,3,4)}
    avg_share = sum(shares.values()) / 4.0 if sum(shares.values()) > 0 else 1.0
    w = {l: (shares.get(l, avg_share) / avg_share) for l in (1,2,3,4)}

    # semen shares
    cow_trad_share = float(semen_shares["cow_trad"])
    cow_sex_share  = float(semen_shares["cow_sex"])
    heif_trad_share = float(semen_shares["heifer_trad"])
    heif_sex_share  = float(semen_shares["heifer_sex"])

    snapshot: HerdState | None = None
    if target <= start:
        snapshot = state

    end_sim = end_of_month(target)
    day = start

    idx_dry = min(dry_days, gest_days)

    while day < end_sim:
        day = day + timedelta(days=1)

        # 1) aging
        for l in (1,2,3,4):
            state.open_dim[l] = shift_right(state.open_dim[l])
        state.heifer_age = shift_right(state.heifer_age)
        state.bull_age = shift_right(state.bull_age)

        # 2) countdown pregnancy
        for l in (1,2,3,4):
            for semen in ("trad","sex"):
                state.preg_lact[(l, semen)] = shift_left(state.preg_lact[(l, semen)])
                state.preg_dry[(l, semen)]  = shift_left(state.preg_dry[(l, semen)])
        for semen in ("trad","sex"):
            state.heifer_preg[semen] = shift_left(state.heifer_preg[semen])

        # 3) auto dryoff
        for l in (1,2,3,4):
            for semen in ("trad","sex"):
                move = state.preg_lact[(l, semen)][idx_dry]
                if move > 0:
                    state.preg_lact[(l, semen)][idx_dry] = 0.0
                    state.preg_dry[(l, semen)][idx_dry] += move

        # 4) calvings (bucket 0)
        for l in (1,2,3,4):
            for semen in ("trad","sex"):
                born = state.preg_dry[(l, semen)][0]
                if born > 0:
                    state.preg_dry[(l, semen)][0] = 0.0

                    if (day.year, day.month) == target_month:
                        calv_total += born
                        calv_cows += born
                        sr = semen_sex_ratios[semen]
                        exp_bulls += born * float(sr.bull_share)
                        exp_heifers += born * float(sr.heifer_share)

                    l2 = min(4, l + 1)
                    state.open_dim[l2][0] += born

                    sr = semen_sex_ratios[semen]
                    state.heifer_age[0] += born * float(sr.heifer_share)
                    state.bull_age[0] += born * float(sr.bull_share)

        for semen in ("trad","sex"):
            born = state.heifer_preg[semen][0]
            if born > 0:
                state.heifer_preg[semen][0] = 0.0

                if (day.year, day.month) == target_month:
                    calv_total += born
                    calv_heifers += born
                    sr = semen_sex_ratios[semen]
                    exp_bulls += born * float(sr.bull_share)
                    exp_heifers += born * float(sr.heifer_share)

                state.open_dim[1][0] += born
                sr = semen_sex_ratios[semen]
                state.heifer_age[0] += born * float(sr.heifer_share)
                state.bull_age[0] += born * float(sr.bull_share)

        # 5) services -> conceptions (используем INSEMINATION_PARAMS)
        for l in (1,2,3,4):
            first_ai = float(ins["cow_first_ai_dim_by_lact"].get(l, 70.0))
            spc = float(ins["cow_services_per_conception"])
            interval_raw = float(ins["cow_ai_interval_days"])
            mean_target = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))

            interval = _effective_ai_interval_days(interval_raw, mean_target, first_ai, spc)
            p_service = 1.0 / max(1.0, interval)
            p_conc = _clamp(1.0 / max(1e-9, spc), 0.05, 0.95)

            open_arr = state.open_dim[l]
            first_i = int(_clamp(first_ai, 0.0, float(MAX_DIM)))
            if first_i >= len(open_arr):
                continue

            eligible = open_arr.copy()
            eligible[:first_i] = 0.0

            services_by_dim = eligible * p_service
            services_total = float(services_by_dim.sum())
            if services_total <= 0:
                continue

            conceived_by_dim = services_by_dim * p_conc
            conceived_total = float(conceived_by_dim.sum())
            if conceived_total <= 0:
                continue

            state.open_dim[l] = np.maximum(0.0, open_arr - conceived_by_dim)

            state.preg_lact[(l, "sex")][gest_days]  += services_total * cow_sex_share  * p_conc
            state.preg_lact[(l, "trad")][gest_days] += services_total * cow_trad_share * p_conc

            if (day.year, day.month) == target_month:
                meta["cow_doses_total"] += services_total
                meta["cow_doses_sex"] += services_total * cow_sex_share
                meta["cow_doses_trad"] += services_total * cow_trad_share

        first_ai_age = float(ins["heifer_first_ai_age_days"])
        spc_h = float(ins["heifer_services_per_conception"])
        interval_raw_h = float(ins["heifer_ai_interval_days"])
        mean_target_h = float(cp["avg_heifer_age_days"])

        interval_h = _effective_ai_interval_days(interval_raw_h, mean_target_h, first_ai_age, spc_h)
        p_service_h = 1.0 / max(1.0, interval_h)
        p_conc_h = _clamp(1.0 / max(1e-9, spc_h), 0.05, 0.95)

        first_h = int(_clamp(first_ai_age, 0.0, float(MAX_AGE_DAYS)))
        if first_h < len(state.heifer_age):
            eligible_h = state.heifer_age.copy()
            eligible_h[:first_h] = 0.0

            services_by_age = eligible_h * p_service_h
            services_total_h = float(services_by_age.sum())
            if services_total_h > 0:
                conceived_by_age = services_by_age * p_conc_h
                conceived_total_h = float(conceived_by_age.sum())
                if conceived_total_h > 0:
                    state.heifer_age = np.maximum(0.0, state.heifer_age - conceived_by_age)
                    state.heifer_preg["sex"][gest_days]  += services_total_h * heif_sex_share  * p_conc_h
                    state.heifer_preg["trad"][gest_days] += services_total_h * heif_trad_share * p_conc_h

                if (day.year, day.month) == target_month:
                    meta["heifer_doses_total"] += services_total_h
                    meta["heifer_doses_sex"] += services_total_h * heif_sex_share
                    meta["heifer_doses_trad"] += services_total_h * heif_trad_share

        # 6) disposal
        for l in (1,2,3,4):
            base = float(p_disp_day_base * w[l])
            base = max(0.0, min(0.02, base))

            haz_open = np.clip(base * disp_shape[l], 0.0, 0.05)
            state.open_dim[l] *= (1.0 - haz_open)

            # беременность -> оценка DIM
            mean_conc = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
            conc0 = int(round(mean_conc))

            idx = np.arange(gest_days + 1, dtype=int)
            gest_age = (gest_days - idx).astype(int)
            est_dim = np.clip(conc0 + gest_age, 0, MAX_DIM)

            haz_preg = np.clip(base * disp_shape[l][est_dim], 0.0, 0.05)
            for semen in ("trad","sex"):
                state.preg_lact[(l, semen)] *= (1.0 - haz_preg)
                state.preg_dry[(l, semen)]  *= (1.0 - haz_preg)
                # 7) month-end capacity -> "реализация"
        if day == end_of_month(day):
            sold = _apply_capacity_month_end(state, gest_days=gest_days, dry_days=dry_days)

            # учитываем в метриках только для target-месяца (как и отёлы)
            if (day.year, day.month) == target_month:
                meta["sell_cows"] += float(sold["sell_cows"])
                meta["sell_heifers"] += float(sold["sell_heifers"])
                meta["sell_neteli"] += float(sold["sell_neteli"])

                meta["over_doy"] += float(sold["over_doy"])
                meta["over_dry"] += float(sold["over_dry"])
                meta["over_h0"] += float(sold["over_h0"])
                meta["over_h38"] += float(sold["over_h38"])
                meta["over_h9"] += float(sold["over_h9"])
                meta["over_neteli"] += float(sold["over_neteli"])

        if day == target:
            snapshot = HerdState(
                open_dim={k: v.copy() for k, v in state.open_dim.items()},
                preg_lact={k: v.copy() for k, v in state.preg_lact.items()},
                preg_dry={k: v.copy() for k, v in state.preg_dry.items()},
                heifer_age=state.heifer_age.copy(),
                heifer_preg={k: v.copy() for k, v in state.heifer_preg.items()},
                bull_age=state.bull_age.copy(),
            )

    if snapshot is None:
        snapshot = state

    meta.update({
        "calv_total": float(calv_total),
        "calv_cows": float(calv_cows),
        "calv_heifers": float(calv_heifers),
        "exp_bulls": float(exp_bulls),
        "exp_heifers": float(exp_heifers),
    })

    return snapshot, meta
def compute_forecast_dynamic_from_db(target_date: date, overrides: dict | None = None) -> Dict[str, float]:
    tables = load_tables()
    base = latest_data_date(tables)
    start = min(base, target_date)

    ov = dict(overrides or {})

    # --- алиасы "русских" ключей из app.py / UI ---
    if "gestation_days" in ov and "GESTATION_DAYS" not in ov:
        ov["GESTATION_DAYS"] = ov["gestation_days"]
    if "dry_days" in ov and "DRY_DAYS" not in ov:
        ov["DRY_DAYS"] = ov["dry_days"]
    if "annual_disposal_rate" in ov and "ANNUAL_DISPOSAL_RATE" not in ov:
        ov["ANNUAL_DISPOSAL_RATE"] = ov["annual_disposal_rate"]
    if "conception" in ov and "CONCEPTION_PARAMS" not in ov:
        ov["CONCEPTION_PARAMS"] = ov["conception"]
    if "insemination_params" in ov and "INSEMINATION_PARAMS" not in ov:
        ov["INSEMINATION_PARAMS"] = ov["insemination_params"]
    if "semen_usage" in ov and "SEMEN_USAGE_SHARES" not in ov:
        ov["SEMEN_USAGE_SHARES"] = ov["semen_usage"]

    # 1) runtime params
    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

    # 2) доли семени: админ -> иначе БД
    semen_override = params.get("SEMEN_USAGE_SHARES")
    if isinstance(semen_override, dict) and semen_override:
        semen_shares = {
            "cow_trad": float(semen_override.get("cow_trad", 0.0)),
            "cow_sex": float(semen_override.get("cow_sex", 0.0)),
            "heifer_trad": float(semen_override.get("heifer_trad", 0.0)),
            "heifer_sex": float(semen_override.get("heifer_sex", 0.0)),
        }

        def _norm2(a: float, b: float) -> tuple[float, float]:
            s = max(1e-9, a + b)
            return a / s, b / s

        semen_shares["cow_trad"], semen_shares["cow_sex"] = _norm2(semen_shares["cow_trad"], semen_shares["cow_sex"])
        semen_shares["heifer_trad"], semen_shares["heifer_sex"] = _norm2(semen_shares["heifer_trad"], semen_shares["heifer_sex"])
    else:
        semen_shares = compute_semen_usage_from_db(tables)

    # 3) пол телят по типу семени: админ -> иначе БД
    # чтобы не словить NameError по SemenSexRatio — берём класс из дефолтов SEMEN_SEX_RATIOS
    ratio_cls = type(SEMEN_SEX_RATIOS["trad"])

    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj):
            bull = float(d.get("bull_share", getattr(fallback_obj, "bull_share", 0.5)))
            bull = max(0.0, min(1.0, bull))
            return ratio_cls(bull_share=bull, heifer_share=1.0 - bull)

        semen_sex_ratios = {
            "trad": _mk_ratio(trad, SEMEN_SEX_RATIOS["trad"]),
            "sex": _mk_ratio(sex, SEMEN_SEX_RATIOS["sex"]),
        }
    else:
        semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    # 4) initial state (важно: прокидываем gest/dry)
    state0 = build_initial_state(
        tables,
        as_of=start,
        gest_days=gest_days,
        dry_days=dry_days,
    )

    # 5) simulate
    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_date,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )

    # 6) метрики месяца target (сырые float)
    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    # --- агрегаты стада на target_date ---
    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    h0_2 = 0.0
    h3_8 = 0.0
    h9p = 0.0
    for age_d, cnt in enumerate(state_at_target.heifer_age):
        if cnt <= 0:
            continue
        m = age_months(age_d)
        if m <= 2:
            h0_2 += cnt
        elif 3 <= m <= 8:
            h3_8 += cnt
        else:
            h9p += cnt

    b0_2 = float(state_at_target.bull_age[:61].sum())

    # --- “показ” как ожидание (чтобы не пропадали месяцы) ---
    calv_total_show = round(calv_total_f, 1)
    calv_cows_show = round(calv_cows_f, 1)
    calv_heifers_show = round(calv_heifers_f, 1)
    exp_bulls_show = round(exp_bulls_f, 1)
    exp_heifers_show = round(exp_heifers_f, 1)

    # --- дополнительно целые (если очень нужно) ---
    calv_total_i = int(round(calv_total_f))
    calv_cows_i = int(round(calv_cows_f))
    calv_heifers_i = int(round(calv_heifers_f))

    denom = exp_bulls_f + exp_heifers_f
    bull_frac = (exp_bulls_f / denom) if denom > 1e-9 else 0.0
    exp_bulls_i = int(round(calv_total_i * bull_frac)) if calv_total_i > 0 else 0
    exp_heifers_i = max(0, calv_total_i - exp_bulls_i)

    return {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),
        "Тёлки 0–2 мес": round(h0_2),
        "Бычки 0–2 мес": round(b0_2),
        "Тёлки 3–8 мес": round(h3_8),
        "Тёлки ≥9 мес": round(h9p),
        "Нетели": round(neteli),

        # ожидание (float) — НЕ пропадает из-за округления
        "Ожидаемый отёл, всего": calv_total_show,
        "Ожидаемый отёл, из них коров": calv_cows_show,
        "Ожидаемый отёл, из них нетелей": calv_heifers_show,
        "Ожидаемые бычки (условно)": exp_bulls_show,
        "Ожидаемые тёлочки (условно)": exp_heifers_show,

        # если где-то UI/таблица ждёт именно int — используй эти поля
        "Ожидаемый отёл, всего (округл.)": calv_total_i,
        "Ожидаемые бычки (округл.)": exp_bulls_i,
        "Ожидаемые тёлочки (округл.)": exp_heifers_i,
                # --- реализация (в месяце target) ---
        "К реализации: коровы": round(float(meta.get("sell_cows", 0.0)), 1),
        "К реализации: тёлки": round(float(meta.get("sell_heifers", 0.0)), 1),
        "К реализации: нетели": round(float(meta.get("sell_neteli", 0.0)), 1),

        # где именно было переполнение (сколько пришлось "срезать")
        "Переполнение: Дойные коровы": round(float(meta.get("over_doy", 0.0)), 1),
        "Переполнение: Сухостойные коровы": round(float(meta.get("over_dry", 0.0)), 1),
        "Переполнение: Тёлки 0–3 мес": round(float(meta.get("over_h0", 0.0)), 1),
        "Переполнение: Тёлки 3–8 мес": round(float(meta.get("over_h38", 0.0)), 1),
        "Переполнение: Тёлки 9–24 мес": round(float(meta.get("over_h9", 0.0)), 1),
        "Переполнение: Нетели": round(float(meta.get("over_neteli", 0.0)), 1),
    }
