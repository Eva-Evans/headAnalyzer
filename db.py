from sqlalchemy import create_engine
from config import POSTGRES_DSN

engine = create_engine(POSTGRES_DSN, echo=False, future=True)
