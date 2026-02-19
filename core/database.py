import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get URL from Environment
DATABASE_URL = os.getenv("DATABASE_URL")

# 2. ✅ FIX: Render's 'postgres://' to 'postgresql://' conversion
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. Create Engine
# Agar URL nahi milta toh temporary sqlite use karega
engine = create_engine(DATABASE_URL) if DATABASE_URL else create_engine("sqlite:///./test.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
