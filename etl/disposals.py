# etl/disposals.py
import pandas as pd

from db import engine

# соответствие  заголовков и названий в БД
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


def read_disposals_excel(path_or_buffer) -> pd.DataFrame:
    """
    Читает Excel-файл 'Выбытие' в DataFrame.
    path_or_buffer: путь к файлу или объект (например, из streamlit file_uploader).
    """
    df = pd.read_excel(path_or_buffer)

    df.columns = [
        str(c).replace("\xa0", " ").strip()
        for c in df.columns
    ]
    # Проверяем наличие нужных колонок
    missing = set(COLUMN_MAP.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"В файле 'Выбытие' не найдены колонки: {missing}")

    df = df.rename(columns=COLUMN_MAP)

    # Оставляем только нужные столбцы (если в файле есть лишние)
    df = df[list(COLUMN_MAP.values())]

    return df


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Безопасно парсит даты, ошибки → NaT."""
    return pd.to_datetime(s, errors="coerce").dt.date


def clean_disposals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная очистка:
    - даты → DATE
    - lact, age_dim, id → числа
    - строки подчищаем
    """
    df = df.copy()

    # даты
    df["birth_date"] = _parse_date_series(df["birth_date"])
    df["event_date"] = _parse_date_series(df["event_date"])
    #TODO: доделать парсинг столбца Дата/DIM
    # числовые поля
    df["lact"] = pd.to_numeric(df["lact"], errors="coerce").astype("Int64")
    df["age_dim"] = pd.to_numeric(df["age_dim"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    # строковые поля
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
