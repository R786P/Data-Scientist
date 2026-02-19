import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get URL and clean it (strip spaces)
raw_url = os.getenv("DATABASE_URL", "")
DATABASE_URL = raw_url.strip()

# 2. ✅ FIX: Handle Render's legacy 'postgres://' format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. Create Engine with a fallback
try:
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)
    else:
        # Fallback to local SQLite if no DB URL found
        engine = create_engine("sqlite:///./test.db")
except Exception as e:
    print(f"❌ Database connection error: {e}")
    # Last resort fallback
    engine = create_engine("sqlite:///./test.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
