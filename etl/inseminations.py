from __future__ import annotations

import pandas as pd
import re
from io import BytesIO
from pathlib import Path
from typing import Dict

from db import engine

TABLE_NAME = "inseminations_raw"

                                                                       
COLUMN_MAP: Dict[str, str] = {
    "ID": "id",
    "REG": "reg",
    "LACT": "lact",
    "Событие": "event_type",
    "DIM/Возраст": "dim_age",
    "Дата": "event_date",
    "Бык": "bull",
    "Result": "result",
    "RESULT": "result",
    "R": "result",
    "T": "tech_id",
    "Тип осеменения": "insemination_type",
    "Техник": "technician",
}

_FARM_ALIASES = {
    "SOURCE.NAME",
    "SOURCE_NAME",
    "SOURCENAME",
    "ХОЗЯЙСТВО",
    "FARM",
    "COMPANY",
}

_SUBDIVISION_ALIASES = {
    "СТОЛБЕЦ1",
    "СТОЛБЕЦ 1",
    "ПОДРАЗДЕЛЕНИЕ",
    "ФЕРМА",
    "SUBDIVISION",
    "DEPARTMENT",
    "UNIT",
    "ЖК",
    "МТФ",
}


def _norm_col(c: object) -> str:
    """Нормализация названий колонок: NBSP -> space, trim, схлопывание пробелов."""
    s = str(c).replace("\u00a0", " ").strip()
                                      
    s = " ".join(s.split())
    return s


def _norm_key(c: object) -> str:
    s = _norm_col(c).upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я. ]+", "", s)


def _pick_col(columns, aliases: set[str]) -> str | None:
    for c in columns:
        if _norm_key(c) in aliases:
            return c
    return None


def _pick_first(columns, *candidates: str) -> str | None:
    norm_map = {_norm_key(c): c for c in columns}
    for cand in candidates:
        hit = norm_map.get(_norm_key(cand))
        if hit is not None:
            return hit
    return None


def _as_excel_source(path_or_buffer):
    if isinstance(path_or_buffer, (str, Path)):
        return path_or_buffer
    if isinstance(path_or_buffer, (bytes, bytearray)):
        return BytesIO(path_or_buffer)
    if hasattr(path_or_buffer, "getvalue"):
        return BytesIO(path_or_buffer.getvalue())
    if hasattr(path_or_buffer, "read"):
        data = path_or_buffer.read()
        try:
            path_or_buffer.seek(0)
        except Exception:
            pass
        return BytesIO(data)
    return path_or_buffer


def _rewind(src) -> None:
    if hasattr(src, "seek"):
        try:
            src.seek(0)
        except Exception:
            pass


def _read_excel_best_header(src, max_header: int = 20) -> pd.DataFrame:
    best_header: int | None = None
    best_score = -1
    last_err: Exception | None = None

    for h in range(0, max_header + 1):
        try:
            _rewind(src)
            tmp = pd.read_excel(src, header=h, dtype=object, nrows=300)
            tmp.columns = [_norm_col(c) for c in tmp.columns]
            has_reg = _pick_first(tmp.columns, "REG", "DREG", "IDREG") is not None
            has_dt = _pick_first(tmp.columns, "Дата", "DATE", "EVENT_DATE") is not None
            has_result = _pick_first(tmp.columns, "Result", "RESULT", "R", "Результат") is not None
            if not (has_reg and has_dt and has_result):
                continue
            score = 0
            for cands in (
                ("ID",),
                ("REG", "DREG", "IDREG"),
                ("LACT", "LACTATION"),
                ("Событие", "EVENT", "EVENT_TYPE"),
                ("DIM/Возраст", "DIM", "Возраст", "DIM_AGE"),
                ("Дата", "DATE", "EVENT_DATE"),
                ("Бык", "BULL", "REMARK"),
                ("Result", "RESULT", "R", "Результат"),
                ("T", "TECH_ID"),
                ("Тип осеменения", "INSEMINATION_TYPE"),
                ("Техник", "TECHNICIAN"),
            ):
                score += int(_pick_first(tmp.columns, *cands) is not None)
            if score > best_score:
                best_header = h
                best_score = score
        except Exception as e:
            last_err = e
            continue

    if best_header is None:
        raise ValueError(
            "Не удалось распознать шапку файла 'Осеменения'. "
            f"Последняя ошибка: {last_err}"
        )

    _rewind(src)
    df = pd.read_excel(src, header=best_header, dtype=object)
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def read_inseminations_excel(path_or_buffer, include_meta: bool = False) -> pd.DataFrame:
    """
    Читает Excel-файл 'Осеменения' в DataFrame и приводит имена колонок.
    path_or_buffer: путь к файлу или объект (например, из streamlit file_uploader).
    """
    src = _as_excel_source(path_or_buffer)
    df = _read_excel_best_header(src, max_header=20)
    farm_col = _pick_col(df.columns, _FARM_ALIASES) if include_meta else None
    subdivision_col = _pick_col(df.columns, _SUBDIVISION_ALIASES) if include_meta else None
    farm_series = None
    subdivision_series = None
    if farm_col is not None:
        farm_series = df[farm_col].astype("string").str.replace("\u00a0", " ", regex=False).str.strip()
    if subdivision_col is not None:
        subdivision_series = df[subdivision_col].astype("string").str.replace("\u00a0", " ", regex=False).str.strip()

    src_cols = {
        "id": _pick_first(df.columns, "ID"),
        "reg": _pick_first(df.columns, "REG", "DREG", "IDREG"),
        "lact": _pick_first(df.columns, "LACT", "LACTATION"),
        "event_type": _pick_first(df.columns, "Событие", "EVENT", "EVENT_TYPE"),
        "dim_age": _pick_first(df.columns, "DIM/Возраст", "DIM", "Возраст", "DIM_AGE", "AGE/DIM", "AGE_DIM"),
        "event_date": _pick_first(df.columns, "Дата", "DATE", "EVENT_DATE"),
        "bull": _pick_first(df.columns, "Бык", "BULL", "REMARK"),
        "result": _pick_first(df.columns, "Result", "RESULT", "R", "Результат"),
        "tech_id": _pick_first(df.columns, "T", "TECH_ID"),
        "insemination_type": _pick_first(df.columns, "Тип осеменения", "INSEMINATION_TYPE"),
        "technician": _pick_first(df.columns, "Техник", "TECHNICIAN"),
    }
    missing_critical = [x for x in ("reg", "event_date", "result") if src_cols.get(x) is None]
    if missing_critical:
        raise ValueError(
            "В файле 'Осеменения' не найдены обязательные колонки: "
            + ", ".join(missing_critical)
            + " (ожидаем REG/Дата/Result или аналоги)."
        )

    out = pd.DataFrame()
    for target in ("id", "reg", "lact", "event_type", "dim_age", "event_date", "bull", "result", "tech_id", "insemination_type", "technician"):
        col = src_cols.get(target)
        out[target] = df[col] if col is not None else pd.NA
    if include_meta:
        out["__farm"] = farm_series if farm_series is not None else pd.NA
        out["__subdivision"] = subdivision_series if subdivision_series is not None else pd.NA

    return out


def clean_inseminations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных 'Осеменения':
    - трим пробелов/nbps во всех строковых полях
    - result: strip+upper, пустые -> NULL
    - event_date: datetime
    - id/reg/lact/dim_age/tech_id: Int64
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .astype("string")
                .str.replace("\u00a0", " ", regex=False)
                .str.strip()
            )

    if "result" in df.columns:
        df["result"] = (
            df["result"]
            .astype("string")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
            .str.upper()
        )
        df.loc[df["result"] == "", "result"] = None

                  
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=True)

                   
    for col in ["id", "reg", "lact", "dim_age", "tech_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                                                               
    for col in ["bull", "event_type", "insemination_type", "technician"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "reg" in df.columns and "event_date" in df.columns:
        df = df[~(df["reg"].isna() & df["event_date"].isna())].copy()

    return df


def load_inseminations_to_db(
    df: pd.DataFrame,
    *,
    if_exists: str = "append",
    chunksize: int = 2000,
):
    """
    Загружает данные 'Осеменения' в таблицу inseminations_raw.
    Важно: df должен быть уже очищен clean_inseminations().
    """
    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
