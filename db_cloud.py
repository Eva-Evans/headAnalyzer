# db_cloud.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

# Всегда используем SQLite на Streamlit Cloud
USE_SQLITE = True

if USE_SQLITE:
    # SQLite для облака
    DB_PATH = "herd_data.db"
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    # Функция для инициализации таблиц при первом запуске
    def init_db():
        """Создаёт таблицы, если их нет"""
        with engine.connect() as conn:
            # Таблица отёлов
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS calvings_births_raw (
                    reg TEXT,
                    mother_reg TEXT,
                    birth_date TIMESTAMP,
                    sex TEXT,
                    event_type TEXT,
                    event_date TIMESTAMP,
                    lact INTEGER,
                    disposal_date TIMESTAMP,
                    disposal_reason TEXT,
                    disposal_remark TEXT,
                    age INTEGER,
                    note TEXT,
                    protocol TEXT,
                    technician TEXT
                )
            """))
            
            # Таблица осеменений
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS inseminations_raw (
                    id INTEGER,
                    reg TEXT,
                    lact INTEGER,
                    event_type TEXT,
                    dim_age INTEGER,
                    event_date TIMESTAMP,
                    bull TEXT,
                    result TEXT,
                    tech_id TEXT,
                    insemination_type TEXT,
                    technician TEXT
                )
            """))
            
            # Таблица запусков
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dryoff_raw (
                    id INTEGER,
                    reg TEXT,
                    birth_date TIMESTAMP,
                    lact INTEGER,
                    disposal_date TIMESTAMP,
                    disposal_reason TEXT,
                    remark TEXT,
                    event_type TEXT,
                    dim INTEGER,
                    event_date TIMESTAMP,
                    note TEXT,
                    protocols TEXT,
                    technician TEXT
                )
            """))
            
            # Таблица выбытий
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS disposals_raw (
                    id INTEGER,
                    reg TEXT,
                    birth_date TIMESTAMP,
                    lact INTEGER,
                    sex TEXT,
                    disposal_reason TEXT,
                    event_type TEXT,
                    age_dim INTEGER,
                    event_date TIMESTAMP,
                    note TEXT
                )
            """))
            
            # Таблица быков
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bulls_raw (
                    bull_code TEXT,
                    short_name TEXT,
                    reg TEXT,
                    secondary_id TEXT,
                    plem INTEGER,
                    breed TEXT,
                    bull_type TEXT
                )
            """))
            
            # Таблица кэша параметров
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_params_cache (
                    signature TEXT PRIMARY KEY,
                    params_json TEXT,
                    updated_at TIMESTAMP
                )
            """))
            
            conn.commit()
            print("Database initialized successfully")
    
    # Инициализируем БД
    init_db()

else:
    # PostgreSQL для локальной разработки (Docker)
    try:
        from config import POSTGRES_DSN
        engine = create_engine(POSTGRES_DSN, echo=False, future=True)
    except ImportError:
        # fallback на SQLite если config нет
        engine = create_engine('sqlite:///herd_data.db', connect_args={"check_same_thread": False})