from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base

# 📊 User Query Model (Analysis log karne ke liye)
class UserQuery(Base):
    __tablename__ = "user_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# 🔐 User Auth Model (Login system ke liye)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    email = Column(String(100), unique=True)
