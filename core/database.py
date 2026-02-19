import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get and Clean URL
raw_url = os.getenv("DATABASE_URL", "").strip()

# 2. Fix Render Prefix
if raw_url.startswith("postgres://"):
    DATABASE_URL = raw_url.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL = raw_url

# 3. Engine Setup with Fallback
# Agar URL galat ya khali hai toh auto-switch to SQLite
try:
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    else:
        engine = create_engine("sqlite:///./test.db")
except Exception:
    engine = create_engine("sqlite:///./test.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
