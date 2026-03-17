import pandas as pd
import re
from io import BytesIO
from pathlib import Path

from db import engine

                                          
COLUMN_MAP = {
    "ID": "id",
    "REG": "reg",
    "Дата рождения": "birth_date",
    "LACT": "lact",
    "Пол": "sex",
    "Причина выбытия": "disposal_reason",
    "Событие": "event_type",
    "Возраст/DIM": "age_dim",
    "Дата": "event_date",
    "Примечание": "note",
}

TABLE_NAME = "disposals_raw"

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


def _norm_key(c: object) -> str:
    s = str(c).replace("\xa0", " ").strip().upper().replace("Ё", "Е")
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
            tmp.columns = [str(c).replace("\xa0", " ").strip() for c in tmp.columns]
            has_reg = _pick_first(tmp.columns, "REG", "DREG", "IDREG") is not None
            has_dt = _pick_first(tmp.columns, "Дата", "DATE", "EVENT_DATE") is not None
            if not (has_reg and has_dt):
                continue
            score = 0
            for cands in (
                ("ID",),
                ("REG", "DREG", "IDREG"),
                ("Дата рождения", "BDAT"),
                ("LACT", "LACTATION"),
                ("Пол", "SEX"),
                ("Причина выбытия", "CARX", "REM", "REMARK"),
                ("Событие", "EVENT", "EVENT_TYPE"),
                ("Возраст/DIM", "Возраст", "DIM", "DIM_AGE"),
                ("Дата", "DATE", "EVENT_DATE"),
                ("Примечание", "NOTE"),
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
            "Не удалось распознать шапку файла 'Выбытие'. "
            f"Последняя ошибка: {last_err}"
        )

    _rewind(src)
    df = pd.read_excel(src, header=best_header, dtype=object)
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    return df


def read_disposals_excel(path_or_buffer, include_meta: bool = False) -> pd.DataFrame:
    """
    Читает Excel-файл 'Выбытие' в DataFrame.
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
    src = {
        "id": _pick_first(df.columns, "ID"),
        "reg": _pick_first(df.columns, "REG", "DREG", "IDREG"),
        "birth_date": _pick_first(df.columns, "Дата рождения", "BDAT", "BIRTH_DATE"),
        "lact": _pick_first(df.columns, "LACT", "LACTATION"),
        "sex": _pick_first(df.columns, "Пол", "SEX"),
        "disposal_reason": _pick_first(df.columns, "Причина выбытия", "CARX", "REM", "REMARK"),
        "event_type": _pick_first(df.columns, "Событие", "EVENT", "EVENT_TYPE"),
        "age_dim": _pick_first(df.columns, "Возраст/DIM", "Возраст", "DIM", "AGE/DIM", "AGE_DIM", "DIM_AGE"),
        "event_date": _pick_first(df.columns, "Дата", "DATE", "EVENT_DATE"),
        "note": _pick_first(df.columns, "Примечание", "NOTE"),
    }
    missing_critical = [x for x in ("reg", "event_date") if src.get(x) is None]
    if missing_critical:
        raise ValueError(
            "В файле 'Выбытие' не найдены обязательные колонки: "
            + ", ".join(missing_critical)
            + " (ожидаем REG/Дата или аналоги)."
        )

    out = pd.DataFrame()
    for tgt in COLUMN_MAP.values():
        col = src.get(tgt)
        out[tgt] = df[col] if col is not None else pd.NA
    if include_meta:
        out["__farm"] = farm_series if farm_series is not None else pd.NA
        out["__subdivision"] = subdivision_series if subdivision_series is not None else pd.NA

    return out


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Безопасно парсит даты, ошибки → NaT."""
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.date


def clean_disposals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная очистка:
    - даты → DATE
    - lact, age_dim, id → числа
    - строки подчищаем
    """
    df = df.copy()

          
    df["birth_date"] = _parse_date_series(df["birth_date"])
    df["event_date"] = _parse_date_series(df["event_date"])
                   
    df["lact"] = pd.to_numeric(df["lact"], errors="coerce").astype("Int64")
    df["age_dim"] = pd.to_numeric(df["age_dim"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

                    
    for col in ["reg", "sex", "disposal_reason", "event_type", "note"]:
        df[col] = df[col].astype("string").str.strip()

    return df


def load_disposals_to_db(
    df: pd.DataFrame,
    if_exists: str = "append",
    chunksize: int = 1000,
):
    """
    Загружает данные 'Выбытие' в PostgreSQL в таблицу disposals_raw.
    if_exists: 'fail' | 'replace' | 'append'
    """
    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
