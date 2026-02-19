from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True)

class UserQuery(Base):
    __tablename__ = "user_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
