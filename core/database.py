import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get and Clean URL (Render Support)
raw_url = os.getenv("DATABASE_URL", "").strip()
if raw_url.startswith("postgres://"):
    DATABASE_URL = raw_url.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL = raw_url

# 2. Engine Setup with Fallback
try:
    if DATABASE_URL:
        # pool_pre_ping=True se database connection drop nahi hoga
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    else:
        engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})
except Exception:
    engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 🔐 MASTER FEATURE: USER MODEL FOR LOGIN ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False) # Hashed password storage
    email = Column(String(100), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- 📊 MASTER FEATURE: QUERY LOGS ---
class UserQuery(Base):
    __tablename__ = "user_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
