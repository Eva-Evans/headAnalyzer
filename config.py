import os

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql+psycopg2://herd_user:herd_password@db:5432/herd_forecast",
)
