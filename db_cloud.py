# db_cloud.py
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Определяем, где мы запущены
IN_CLOUD = os.environ.get("STREAMLIT_SHARING", False) or os.environ.get("STREAMLIT_CLOUD", False)

if IN_CLOUD:
    # SQLite для облака
    DB_PATH = "herd_data.db"
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL для локальной разработки
    from config import POSTGRES_DSN
    engine = create_engine(POSTGRES_DSN)