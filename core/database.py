import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get URL and handle None/Empty cases
db_url = os.getenv("DATABASE_URL", "").strip()

# 2. Fix Render/Postgres Prefix & Fallback
if not db_url:
    # Agar variable khali hai, toh crash hone ke bajaye SQLite use karo
    DATABASE_URL = "sqlite:///./test.db"
elif db_url.startswith("postgres://"):
    DATABASE_URL = db_url.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL = db_url

# 3. Engine Setup with extra safety
try:
    # connect_args sirf SQLite ke liye chahiye, Postgres ke liye nahi
    if "sqlite" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    print(f"⚠️ DB Engine Error: {e}. Falling back to SQLite.")
    engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
