# etl/calvings_births.py
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from db import engine

# -----------------------------------------------------------------------------
# Helpers: column mapping
# -----------------------------------------------------------------------------

def _norm_colname(x: Any) -> str:
    """
    Нормализуем имя колонки так, чтобы:
    - 'Дата рождения', 'Дата_рождения', 'дата-рождения' матчились одинаково
    - убираем пробелы/знаки/скобки
    """
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    s = re.sub(r"[^0-9A-ZА-Я]+", "", s)
    return s


# алиасы -> канонические имена колонок, которые мы дальше используем в пайплайне
_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    # животное
    "reg": (
        "REG", "ANIMAL", "ANIMALID", "ID", "EAR", "EARTAG", "EARTAGID",
        "НОМЕР", "НОМЕРЖИВОТНОГО", "ЖИВОТНОЕ", "ИД", "ИДЖИВОТНОГО", "АБС", "ABS",
    ),
    # мать
    "mother_reg": (
        "MOTHERREG", "MOTHER", "DAM", "DAMID", "MOTHERID",
        "МАТЬ", "МАТКА", "НОМЕРМАТЕРИ", "МАТЕРЬ", "МАМА",
    ),
    # дата рождения (часто BDAT)
    "birth_date": (
        "BDAT", "BIRTHDATE", "BIRTH_DATE", "DOB",
        "ДАТАРОЖДЕНИЯ", "ДАТАРОЖД", "РОЖДЕНИЕ", "ДАТАРОЖДЖИВОТНОГО",
    ),
    # пол (часто GNDR)
    "sex": (
        "GNDR", "GENDER", "SEX", "SX",
        "ПОЛ", "ПОЛЖИВОТНОГО",
    ),
    # тип события
    "event_type": (
        "EVENTTYPE", "EVENT_TYPE", "TYPE", "EVTYPE", "EVENT",
        "СОБЫТИЕ", "ТИПСОБЫТИЯ", "ТИПСОБ", "ВИДСОБЫТИЯ",
    ),
    # дата события
    "event_date": (
        "EVENTDATE", "EVENT_DATE", "EDAT", "DATE", "DT", "EVENTDT",
        "ДАТА", "ДАТАСОБЫТИЯ", "ДАТАСОБ", "ДАТА_СОБЫТИЯ", "ДАТАОТЕЛА",
    ),
    # лактация (если есть)
    "lact": (
        "LACT", "LACTATION", "ЛАКТАЦИЯ", "НОМЕРЛАКТАЦИИ",
    ),
    # прочее (опционально — если вдруг есть в выгрузке)
    "note": ("NOTE", "КОММЕНТАРИЙ", "ПРИМЕЧАНИЕ", "ЗАМЕТКА"),
    "protocol": ("PROTOCOL", "ПРОТОКОЛ"),
    "technician": ("TECHNICIAN", "OPERATOR", "ТЕХНИК", "ОПЕРАТОР"),
    "age": ("AGE", "ВОЗРАСТ"),
    "disposal_date": ("DISPOSALDATE", "DISPOSAL_DATE", "ДАТАВЫБЫТИЯ", "ДАТАВЫВОДА"),
    "disposal_reason": ("DISPOSALREASON", "DISPOSAL_REASON", "ПРИЧИНАВЫБЫТИЯ", "ПРИЧИНАВЫВОДА"),
    "disposal_remark": ("DISPOSALREMARK", "DISPOSAL_REMARK", "КОММЕНТАРИЙВЫБЫТИЯ"),
}


def _pick_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    norm_to_real = {_norm_colname(c): c for c in df.columns}
    for a in aliases:
        key = _norm_colname(a)
        if key in norm_to_real:
            return norm_to_real[key]
    return None


def _build_rename_map(df: pd.DataFrame) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    for canon, aliases in _COL_ALIASES.items():
        real = _pick_column(df, aliases)
        if real is not None:
            rename[real] = canon
    return rename


# -----------------------------------------------------------------------------
# Helpers: value normalization
# -----------------------------------------------------------------------------

def _norm_id_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").fillna("").str.replace("\u00a0", " ", regex=False).str.strip()
    # "12345.0" -> "12345"
    out = out.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    out = out.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return out


def _norm_sex_value(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    v = str(x).strip().upper().replace("Ё", "Е")
    if v in ("F", "Ж", "ЖЕН", "ЖЕНСКИЙ", "FEMALE", "2"):
        return "F"
    if v in ("M", "М", "МУЖ", "МУЖСКОЙ", "MALE", "1"):
        return "M"
    return None


def _norm_event_type_value(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    v = str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")
    # лёгкая нормализация синонимов
    if "РОЖ" in v or "BIRTH" in v:
        return "РОЖДЕН"
    if "ОТЕЛ" in v or "CALV" in v:
        return "ОТЕЛ"
    return v


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _as_excel_source(file: Any):
    """
    Streamlit UploadedFile / file-like / bytes / path -> то, что понимает pd.read_excel.
    """
    if isinstance(file, (str, Path)):
        return file
    if isinstance(file, (bytes, bytearray)):
        return BytesIO(file)

    if hasattr(file, "getvalue"):  # streamlit UploadedFile
        return BytesIO(file.getvalue())

    if hasattr(file, "read"):  # file-like
        data = file.read()
        return BytesIO(data)

    return file  # на всякий случай


def _read_excel_best_header(src, max_header: int = 20) -> pd.DataFrame:
    """
    Часто шапка не на первой строке. Пробуем header=0..max_header,
    выбираем вариант, где нашлись обязательные колонки и максимум сматченных.
    """
    best_df: pd.DataFrame | None = None
    best_score = -1
    last_err: Exception | None = None

    # если src BytesIO — нужно уметь перематывать
    def _rewind():
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass

    for header in range(0, max_header + 1):
        try:
            _rewind()
            tmp = pd.read_excel(src, header=header, dtype=object)
            if tmp is None or tmp.empty:
                continue

            rename = _build_rename_map(tmp)
            tmp = tmp.rename(columns=rename)

            required_min = {"reg", "event_date", "event_type"}
            if not required_min.issubset(set(tmp.columns)):
                continue

            score = len(set(rename.values()))  # сколько канонических колонок нашли
            if score > best_score:
                best_df = tmp
                best_score = score
        except Exception as e:
            last_err = e
            continue

    if best_df is None:
        raise ValueError(
            "Не удалось распознать шапку/колонки в файле 'Отёлы+родившиеся'. "
            f"Последняя ошибка: {last_err}"
        )
    return best_df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def read_calvings_excel(file) -> pd.DataFrame:
    """
    Читает Excel (отёлы + родившиеся), приводит названия колонок к канону и нормализует значения.
    Возвращает df с минимумом колонок: reg, mother_reg, birth_date, sex, event_type, event_date (+ lact если есть).
    """
    src = _as_excel_source(file)
    df = _read_excel_best_header(src, max_header=20)

    # создаём необязательные, если их нет
    for col, default in (
        ("mother_reg", pd.NA),
        ("birth_date", pd.NaT),
        ("sex", pd.NA),
        ("lact", pd.NA),
        ("note", pd.NA),
        ("protocol", pd.NA),
        ("technician", pd.NA),
        ("age", pd.NA),
        ("disposal_date", pd.NaT),
        ("disposal_reason", pd.NA),
        ("disposal_remark", pd.NA),
    ):
        if col not in df.columns:
            df[col] = default

    # нормализация
    df["reg"] = _norm_id_series(df["reg"])
    df["mother_reg"] = _norm_id_series(df["mother_reg"])

    df["event_date"] = _to_datetime(df["event_date"])
    df["birth_date"] = _to_datetime(df["birth_date"])

    df["event_type"] = df["event_type"].apply(_norm_event_type_value)
    df["sex"] = df["sex"].apply(_norm_sex_value)

    df["lact"] = pd.to_numeric(df["lact"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # если birth_date пустая — для "РОЖДЕН" берём event_date
    mask_birth = (df["event_type"] == "РОЖДЕН") & df["birth_date"].isna()
    df.loc[mask_birth, "birth_date"] = df.loc[mask_birth, "event_date"]

    # минимальный набор + полезные поля (остальные clean_calvings добьёт)
    keep = [
        "reg",
        "mother_reg",
        "birth_date",
        "sex",
        "event_type",
        "event_date",
        "lact",
        "disposal_date",
        "disposal_reason",
        "disposal_remark",
        "age",
        "note",
        "protocol",
        "technician",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep].copy()


def clean_calvings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим типы, добавляем отсутствующие технические колонки и
    оставляем только те поля, которые хотим хранить в calvings_births_raw.
    """
    df = df.copy()

    target_cols = [
        "animal_id",
        "reg",
        "mother_reg",
        "mother_reg_intl",
        "calf1_reg",
        "calf2_reg",
        "calf3_reg",
        "birth_date",
        "lact",
        "disposal_date",
        "disposal_reason",
        "disposal_remark",
        "sex",
        "event_type",
        "age",
        "event_date",
        "note",
        "protocol",
        "technician",
    ]

    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[target_cols]

    # Типы
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["disposal_date"] = pd.to_datetime(df["disposal_date"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    df["lact"] = pd.to_numeric(df["lact"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # IDs как строки
    df["reg"] = _norm_id_series(df["reg"])
    df["mother_reg"] = _norm_id_series(df["mother_reg"])
    df["calf1_reg"] = _norm_id_series(df["calf1_reg"])
    df["calf2_reg"] = _norm_id_series(df["calf2_reg"])
    df["calf3_reg"] = _norm_id_series(df["calf3_reg"])

    # Нормализуем event_type/sex ещё раз на всякий
    df["event_type"] = df["event_type"].apply(_norm_event_type_value)
    df["sex"] = df["sex"].apply(_norm_sex_value)

    return df


def load_calvings_to_db(df: pd.DataFrame, if_exists: str = "replace") -> None:
    """
    Записываем отёлы+рождения в таблицу calvings_births_raw.
    if_exists: 'replace' (пересоздать), 'append' (добавить) и т.п.
    """
    cleaned = clean_calvings(df)
    cleaned.to_sql(
        "calvings_births_raw",
        con=engine,
        if_exists=if_exists,
        index=False,
    )
