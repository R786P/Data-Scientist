import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # ✅ FIX: postgres:// ko postgresql:// mein convert karo
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # ✅ FIX: SSL parameters add karo Neon ke liye
    if "neon.tech" in DATABASE_URL:
        if "?" not in DATABASE_URL:
            DATABASE_URL += "?sslmode=require&sslcert=&sslkey="
        elif "sslmode" not in DATABASE_URL:
            DATABASE_URL += "&sslmode=require&sslcert=&sslkey="
    
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # ✅ Connection health check
            pool_recycle=300,    # ✅ Recycle connections every 5 min
            connect_args={"sslmode": "require"}  # ✅ SSL enforce
        )
        print("✅ PostgreSQL connected (Neon SSL configured)")
    except Exception as e:
        print(f"⚠️ PostgreSQL failed: {e}")
        print("📁 Falling back to SQLite")
        DATABASE_URL = "sqlite:///./local.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    print("⚠️ DATABASE_URL not set")
    DATABASE_URL = "sqlite:///./local.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserQuery(Base):
    __tablename__ = "user_queries"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    query_text = Column(String, nullable=False)
    response_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
