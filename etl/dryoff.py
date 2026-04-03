import pandas as pd

#from db import engine
from db_cloud import engine
# маппинг заголовков Excel → названия колонок в БД
COLUMN_MAP = {
    "ID": "id",
    "REG": "reg",
    "BDAT": "birth_date",
    "LACT": "lact",
    "ARDAT": "disposal_date",
    "CARX": "disposal_reason",
    "REM": "remark",
    "Событие": "event_type",
    "DIM": "dim",
    "Дата": "event_date",
    "Примечание": "note",
    "Протоколы;": "protocols",
    "Техник": "technician",
}

TABLE_NAME = "dryoff_raw"


def read_dryoff_excel(path_or_buffer) -> pd.DataFrame:
    """
    Читает Excel-файл 'Запуск' в DataFrame.
    path_or_buffer: путь к файлу или объект (например, из streamlit file_uploader).
    """
    df = pd.read_excel(path_or_buffer)

    df.columns = [
        str(c).replace("\xa0", " ").strip()
        for c in df.columns
    ]
    missing = set(COLUMN_MAP.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"В файле 'Запуск' не найдены колонки: {missing}")

    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())]

    return df


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Безопасно парсит даты, ошибки → NaT."""
    return pd.to_datetime(s, errors="coerce").dt.date


def clean_dryoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная очистка данных 'Запуск':
    - даты → DATE
    - id, lact, dim → числа
    - строки подчищаем от пробелов
    """
    df = df.copy()

    # даты
    df["birth_date"] = _parse_date_series(df["birth_date"])
    df["disposal_date"] = _parse_date_series(df["disposal_date"])
    df["event_date"] = _parse_date_series(df["event_date"])

    # числовые
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["lact"] = pd.to_numeric(df["lact"], errors="coerce").astype("Int64")
    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")

    # строковые
    for col in [
        "reg",
        "disposal_reason",
        "remark",
        "event_type",
        "note",
        "protocols",
        "technician",
    ]:
        df[col] = df[col].astype("string").str.strip()

    return df


def load_dryoff_to_db(
    df: pd.DataFrame,
    if_exists: str = "append",
    chunksize: int = 1000,
):
    """
    Загружает данные 'Запуск' в таблицу dryoff_raw в PostgreSQL.
    """
    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
