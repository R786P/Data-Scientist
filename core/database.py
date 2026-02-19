import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get and Clean URL (Handle None/Empty cases)
raw_url = os.getenv("DATABASE_URL", "").strip() if os.getenv("DATABASE_URL") else ""

# 2. Fix Render Prefix & Standardize
if raw_url.startswith("postgres://"):
    DATABASE_URL = raw_url.replace("postgres://", "postgresql://", 1)
elif not raw_url:
    # Agar URL khali hai toh bina error ke SQLite par switch ho jao
    DATABASE_URL = "sqlite:///./test.db"
else:
    DATABASE_URL = raw_url

# 3. Engine Setup with Safety Fallback
try:
    # SQLite ko 'check_same_thread' chahiye hota hai, Postgres ko nahi
    if "sqlite" in DATABASE_URL:
        engine = create_engine(
            DATABASE_URL, 
            connect_args={"check_same_thread": False}
        )
    else:
        # Postgres/Other DBs
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
except Exception as e:
    # Last resort: Agar sab fail ho jaye toh local DB bana lo
    print(f"⚠️ Database Connection Failed: {e}. Falling back to SQLite.")
    engine = create_engine(
        "sqlite:///./test.db", 
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
