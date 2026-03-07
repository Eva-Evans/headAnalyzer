from __future__ import annotations

from typing import IO, Optional, Dict, Tuple

import re
import pandas as pd

from db import engine


def _read_text(file_obj: IO[bytes] | IO[str]) -> str:
    """
    Универсально читаем содержимое файла (streamlit uploader или обычный файл).
    Удаляем NUL-байты (\x00), которые не любит Postgres.
    """
    raw = file_obj.read()
    if isinstance(raw, bytes):
        for enc in ("utf-8", "cp1251"):
            try:
                s = raw.decode(enc)
                break
            except UnicodeDecodeError:
                s = None
        if s is None:
                                                                     
            s = raw.decode("utf-8", errors="ignore")
    else:
        s = raw

    s = s.replace("\x00", "")
    return s


def _detect_header_and_spans(lines: list[str]) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Находим строку заголовка и вычисляем срезы (start, end) для колонок.
    """
    header_idx: Optional[int] = None
    header_line: Optional[str] = None

    for i, line in enumerate(lines):
        norm = line.replace("\t", " ")
        low = norm.lower()
                                                                       
        if "бык" in low and ("рег" in low or "клич" in low or "корот" in low or "порода" in low):
            header_idx = i
            header_line = norm.rstrip("\n")
            break

    if header_idx is None or header_line is None:
        raise ValueError("header-not-found")

                                                                         
    raw_cols = [seg for seg in re.split(r" {2,}", header_line.strip()) if seg]

                                                                       
    spans_raw: list[Tuple[str, int]] = []
    cursor = 0
    for seg in raw_cols:
        idx = header_line.find(seg, cursor)
        if idx == -1:
            idx = header_line.find(seg)
            if idx == -1:
                continue
        spans_raw.append((seg.strip(), idx))
        cursor = idx + len(seg)

    spans_raw.sort(key=lambda x: x[1])

    spans_named: list[Tuple[str, int, int]] = []
    for (name, start), next_item in zip(spans_raw, spans_raw[1:] + [(None, len(header_line))]):
        _, next_start = next_item
        end = next_start
        spans_named.append((name, start, end))

                                            
    col_spans: Dict[str, Tuple[int, int]] = {}

    for name, start, end in spans_named:
        lname = name.lower()
        if lname.startswith("бык"):
            col_spans["bull_code"] = (start, end)
        elif "короткая" in lname or "кличка" in lname:
            col_spans["short_name"] = (start, end)
        elif lname.startswith("рег"):
            col_spans["reg"] = (start, end)
        elif "вторичный" in lname:
            col_spans["secondary_id"] = (start, end)
        elif lname.startswith("плем"):
            col_spans["plem"] = (start, end)
        elif lname.startswith("порода"):
            col_spans["breed"] = (start, end)
        elif lname.startswith("тип"):
            col_spans["bull_type"] = (start, end)

    required = {"bull_code", "short_name", "reg"}
    missing = required - set(col_spans.keys())
    if missing:
        raise ValueError(f"header-spans-missing-{missing}")

    return header_idx, col_spans


def _parse_fixed_width(lines: list[str]) -> pd.DataFrame:
    """
    Основной вариант: парсим как fixed-width по заголовку.
    """
    header_idx, col_spans = _detect_header_and_spans(lines)
    data_lines = lines[header_idx + 1 :]

    def get_field(line: str, span: Tuple[int, int]) -> Optional[str]:
        start, end = span
        if start >= len(line):
            return None
        chunk = line[start:end]
        chunk = chunk.strip()
        return chunk or None

    rows = []
    for line in data_lines:
        if not line.strip():
            continue

        row = {
            "bull_code": get_field(line, col_spans["bull_code"]),
            "short_name": get_field(line, col_spans["short_name"]),
            "reg": get_field(line, col_spans["reg"]),
            "secondary_id": get_field(line, col_spans.get("secondary_id", (0, 0))),
            "plem": get_field(line, col_spans.get("plem", (0, 0))),
            "breed": get_field(line, col_spans.get("breed", (0, 0))),
            "bull_type": get_field(line, col_spans.get("bull_type", (0, 0))),
        }

        if not row["bull_code"]:
            continue

        rows.append(row)

    if not rows:
        raise ValueError("after-header-no-rows")

    df = pd.DataFrame(rows)
    if "plem" in df.columns:
        df["plem"] = pd.to_numeric(df["plem"], errors="coerce").astype("Int64")
    return df


def _parse_by_split(lines: list[str]) -> pd.DataFrame:
    """
    Запасной вариант: просто split по пробелам для строк,
    где хотя бы 3 "слова" и это не заголовок/шапка.
    Формат:
      0: bull_code
      1: short_name
      2: reg
      3: secondary_id
      4: plem
      5: breed
      6: bull_type
    """
    rows = []
    header_keywords = ("Бык", "бык", "Короткая", "короткая", "Рег", "рег", "Таблица", "таблица")

    for line in lines:
        line = line.strip()
        if not line:
            continue

                                             
        if any(kw in line for kw in header_keywords):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        bull_code = parts[0]
        short_name = parts[1] if len(parts) >= 2 else None
        reg = parts[2] if len(parts) >= 3 else None
        secondary_id = parts[3] if len(parts) >= 4 else None
        plem = parts[4] if len(parts) >= 5 else None
        breed = parts[5] if len(parts) >= 6 else None
        bull_type = parts[6] if len(parts) >= 7 else None

        rows.append(
            {
                "bull_code": bull_code,
                "short_name": short_name,
                "reg": reg,
                "secondary_id": secondary_id,
                "plem": plem,
                "breed": breed,
                "bull_type": bull_type,
            }
        )

    if not rows:
        raise ValueError("split-no-rows")

    df = pd.DataFrame(rows)
    if "plem" in df.columns:
        df["plem"] = pd.to_numeric(df["plem"], errors="coerce").astype("Int64")
    return df


def _clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляем NUL (\x00) из всех строковых колонок на всякий случай.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace("\x00", "", regex=False)
    return df


def read_bulls_txt(file_obj: IO[bytes] | IO[str]) -> pd.DataFrame:
    """
    Читает текстовую 'таблицу быков' из блокнота и возвращает DataFrame
    с колонками:
      bull_code, short_name, reg, secondary_id, plem, breed, bull_type

    Сначала пробуем как fixed-width по заголовку.
    Если не получилось – fallback на split() по пробелам.
    """
    text = _read_text(file_obj)
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
                                       
    try:
        df = _parse_fixed_width(lines)
    except Exception:
        df = _parse_by_split(lines)

    df = _clean_nulls(df)
    return df


def load_bulls_to_db(df: pd.DataFrame, if_exists: str = "replace") -> None:
    """
    Заливает таблицу быков в Postgres (таблица bulls_raw),
    предварительно вычищая мусорные строки.
    """
    df = _clean_nulls(df.copy())

                                                                       
    mask_code = df["bull_code"].astype(str).str.match(r"^[0-9A-Za-z]+$", na=False)

    mask_reg = df["reg"].notna() & (df["reg"].astype(str).str.len() > 1)

    df = df[mask_code & mask_reg]

    df.to_sql("bulls_raw", con=engine, if_exists=if_exists, index=False)
