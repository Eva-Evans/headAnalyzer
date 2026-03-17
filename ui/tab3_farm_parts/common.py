from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import streamlit as st

from etl.bulls import read_bulls_txt
from etl.calvings_births import read_calvings_excel
from etl.disposals import read_disposals_excel
from etl.dryoff import read_dryoff_excel
from etl.inseminations import clean_inseminations, read_inseminations_excel

TAB3_TABLES = {
    "calv": "tab3_calvings_farm_raw",
    "ins": "tab3_inseminations_farm_raw",
    "dry": "tab3_dryoff_farm_raw",
    "disp": "tab3_disposals_farm_raw",
    "bulls": "tab3_bulls_farm_raw",
}

TAB3_CACHE_TABLE = "tab3_forecast_cache"

TAB3_MAP_TABLE = "tab3_subdivision_farm_map"

TAB3_CAPACITY_TABLE = "tab3_capacity_places"

TAB3_CACHE_SCHEMA_VERSION = "2026-03-03.v8"

TAB3_UI_STATE_VERSION = "2026-02-26.v3"

TAB3_SHOW_TRANSFER_SNAPSHOT = False

TAB3_SHOW_TRANSFER_FLOWS = False

TAB3_UNASSIGNED_FARM = "ВНЕ ХОЗЯЙСТВА"

FARM_BACKTEST_TARGETS: list[str] = [
    "Дойные коровы",
    "Сухостойные коровы",
    "Тёлки 0–3 мес",
    "Бычки 0–2 мес",
    "Тёлки 3–8 мес",
    "Тёлки ≥9 мес",
    "Нетели",
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]

FARM_BACKTEST_BIRTH_TARGETS = {
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
}

FARM_PERCENT_TARGETS = {
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
}

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

@dataclass
class FarmUploadBundle:
    farm_name: str
    calv: Any | None = None
    ins: Any | None = None
    dry: Any | None = None
    disp: Any | None = None
    bulls: list[Any] = field(default_factory=list)

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
    mother_col = _find_col(df_raw, "DREG", "DREG1", "REG", "MOTHER_REG", "MOTHER")
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


def _bundle_has_core_files(bundle: FarmUploadBundle) -> bool:
    return any(x is not None for x in (bundle.calv, bundle.ins, bundle.dry, bundle.disp))


def _bundle_has_only_bulls(bundle: FarmUploadBundle) -> bool:
    return bool(bundle.bulls) and not _bundle_has_core_files(bundle)


def _merge_aux_bull_bundles(
    bundles: dict[str, FarmUploadBundle],
) -> tuple[dict[str, FarmUploadBundle], dict[str, str]]:
    """
    Если пользователь загрузил один комплект из 4 Excel по всему хозяйству
    и отдельно несколько txt/xlsx по быкам для подразделений, не считаем
    эти bull-only файлы отдельными "подразделениями с неполным комплектом".

    Возвращает:
    - bundles после слияния
    - mapping source_bundle_name -> target_bundle_name для bull-only комплектов
    """
    if not isinstance(bundles, dict) or not bundles:
        return bundles, {}

    core_bundle_names = [name for name, bundle in bundles.items() if _bundle_has_core_files(bundle)]
    if len(core_bundle_names) != 1:
        return bundles, {}

    target_name = core_bundle_names[0]
    merged = dict(bundles)
    attached: dict[str, str] = {}

    for name, bundle in list(merged.items()):
        if name == target_name:
            continue
        if not _bundle_has_only_bulls(bundle):
            continue
        merged[target_name].bulls.extend(bundle.bulls)
        attached[name] = target_name
        del merged[name]

    return merged, attached

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
    calv_df["birth_date"] = pd.to_datetime(calv_df["birth_date"], errors="coerce", dayfirst=True)
    calv_df["event_date"] = pd.to_datetime(calv_df["event_date"], errors="coerce", dayfirst=True)
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
    ins_df["event_date"] = pd.to_datetime(ins_df["event_date"], errors="coerce", dayfirst=True)
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
    dry_df["event_date"] = pd.to_datetime(dry_df["event_date"], errors="coerce", dayfirst=True)
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
    disp_df["event_date"] = pd.to_datetime(disp_df["event_date"], errors="coerce", dayfirst=True)

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

def _subdivision_param_overrides_state() -> dict[str, dict]:
    raw = st.session_state.get("tab3_subdivision_param_overrides")
    if not isinstance(raw, dict):
        raw = {}
    st.session_state["tab3_subdivision_param_overrides"] = raw
    return raw

def _is_admin_mode() -> bool:
    return bool(st.session_state.get("is_admin", False))

def _build_farm_params(base_params: dict, farm_override: dict | None) -> dict:
    params = deepcopy(base_params or {})
    if isinstance(farm_override, dict) and farm_override:
        _deep_merge(params, farm_override)
    params.pop("HERD_CAPACITY", None)
    params.pop("herd_capacity", None)
    params["DISABLE_CAPACITY"] = True
    params["APPLY_CAPACITY"] = False
    return params

def _build_subdivision_params(
    base_params: dict,
    farm_override: dict | None = None,
    subdivision_override: dict | None = None,
) -> dict:
    params = _build_farm_params(base_params, farm_override)
    if isinstance(subdivision_override, dict) and subdivision_override:
        _deep_merge(params, subdivision_override)
    params.pop("HERD_CAPACITY", None)
    params.pop("herd_capacity", None)
    params["DISABLE_CAPACITY"] = True
    params["APPLY_CAPACITY"] = False
    return params


__all__ = [name for name in globals() if not name.startswith("__")]
