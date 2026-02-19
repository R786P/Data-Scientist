from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String
from flask_login import UserMixin
from core.database import Base, SessionLocal

class User(Base, UserMixin):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

def create_default_admin():
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                password_hash=generate_password_hash("admin123")
            )
            db.add(admin)
            db.commit()
            print("✅ Default admin created: admin / admin123")
        else:
            print("✅ Admin user already exists")
    except Exception as e:
        print(f"⚠️ Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()
