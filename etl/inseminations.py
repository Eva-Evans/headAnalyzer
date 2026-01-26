# etl/inseminations.py
from __future__ import annotations

import pandas as pd
from typing import Dict

from db import engine

TABLE_NAME = "inseminations_raw"

# Маппинг заголовков Excel → колонки в БД
# Если у тебя в файле названия отличаются — добавляй сюда альтернативы.
COLUMN_MAP: Dict[str, str] = {
    "ID": "id",
    "REG": "reg",
    "LACT": "lact",
    "Событие": "event_type",
    "DIM/Возраст": "dim_age",
    "Дата": "event_date",
    "Бык": "bull",
    "Result": "result",
    "T": "tech_id",
    "Тип осеменения": "insemination_type",
    "Техник": "technician",
}


def _norm_col(c: object) -> str:
    """Нормализация названий колонок: NBSP -> space, trim, схлопывание пробелов."""
    s = str(c).replace("\u00a0", " ").strip()
    # схлопываем множественные пробелы
    s = " ".join(s.split())
    return s


def read_inseminations_excel(path_or_buffer) -> pd.DataFrame:
    """
    Читает Excel-файл 'Осеменения' в DataFrame и приводит имена колонок.
    path_or_buffer: путь к файлу или объект (например, из streamlit file_uploader).
    """
    df = pd.read_excel(path_or_buffer)
    df.columns = [_norm_col(c) for c in df.columns]

    # проверяем, что есть обязательные колонки
    required = {"REG", "Дата", "Result"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"В файле 'Осеменения' не найдены обязательные колонки: {missing_req}")

    # выбираем только те колонки, которые реально есть в файле и есть в маппинге
    present_src_cols = [src for src in COLUMN_MAP.keys() if src in df.columns]
    df = df[present_src_cols].rename(columns={src: COLUMN_MAP[src] for src in present_src_cols})

    return df


def clean_inseminations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных 'Осеменения':
    - трим пробелов/nbps во всех строковых полях
    - result: strip+upper, пустые -> NULL
    - event_date: datetime
    - id/reg/lact/dim_age/tech_id: Int64
    """
    df = df.copy()

    # --- строки: NBSP -> пробел, trim ---
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .astype("string")
                .str.replace("\u00a0", " ", regex=False)
                .str.strip()
            )

    # --- result: критично! убираем хвостовые пробелы и аппер ---
    if "result" in df.columns:
        df["result"] = (
            df["result"]
            .astype("string")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
            .str.upper()
        )
        df.loc[df["result"] == "", "result"] = None

    # --- дата ---
    if "event_date" in df.columns:
        # делаем datetime (timestamp), чтобы в Postgres нормально улетало
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    # --- числа ---
    for col in ["id", "reg", "lact", "dim_age", "tech_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- bull / event_type / insemination_type / technician ---
    # (уже тримнули выше, но оставим как явный список — удобно)
    for col in ["bull", "event_type", "insemination_type", "technician"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # убираем полностью пустые строки (например, если REG пустой и даты нет)
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
