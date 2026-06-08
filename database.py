from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
import os

from dotenv import load_dotenv
load_dotenv()

raw_database_url = os.getenv("DATABASE_URL")
if not raw_database_url:
    raise RuntimeError("DATABASE_URL is not set")

DATABASE_URL = raw_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
DB_CONNECT_TIMEOUT = float(os.getenv("DB_CONNECT_TIMEOUT", "10"))

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true",
    pool_pre_ping=True,
    connect_args={"timeout": DB_CONNECT_TIMEOUT},
)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()
