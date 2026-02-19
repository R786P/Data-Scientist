from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String
from core.database import Base, SessionLocal

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password_hash = Column(String)

def create_default_admin():
    db = SessionLocal()
    try:
        # Check if admin exists
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
