import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    try:
        engine = create_engine(DATABASE_URL)
        print("✅ PostgreSQL connected")
    except Exception as e:
        print(f"⚠️ PostgreSQL failed: {e}")
        DATABASE_URL = "sqlite:///./local.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    print("⚠️ DATABASE_URL not set")
    DATABASE_URL = "sqlite:///./local.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ UserQuery Model (Ye import ho raha hai agent.py mein)
class UserQuery(Base):
    __tablename__ = "user_queries"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    query_text = Column(String, nullable=False)
    response_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
