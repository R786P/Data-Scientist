import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # ✅ Fix postgres:// to postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    try:
        engine = create_engine(DATABASE_URL)
        print("✅ PostgreSQL connected")
    except Exception as e:
        print(f"⚠️ PostgreSQL failed: {e}")
        print("📁 Falling back to SQLite")
        DATABASE_URL = "sqlite:///./local.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    print("⚠️ DATABASE_URL not set")
    print("📁 Using SQLite fallback")
    DATABASE_URL = "sqlite:///./local.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
