import os
import secrets
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, Boolean
from werkzeug.security import generate_password_hash
from core.database import Base, SessionLocal, ActiveSession, Subscription
from datetime import datetime

class User(Base, UserMixin):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String, unique=True, index=True)
    password_hash = Column(String)
    is_admin      = Column(Boolean, default=False)

def create_default_admin():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin = User(
                username="admin",
                password_hash=generate_password_hash(
                    os.getenv("ADMIN_PASSWORD", "admin123")
                ),
                is_admin=True
            )
            db.add(admin)
            db.flush()   # ← ID assign hoga bina commit ke
            # Give admin pro plan
            sub = Subscription(
                user_id=admin.id,
                plan="pro",
                is_active=True
            )
            db.add(sub)
            db.commit()
            print("✅ Admin user created with Pro plan")
        else:
            # Already exists — ensure pro plan
            sub = db.query(Subscription).filter_by(
                user_id=existing.id).first()
            if not sub:
                db.add(Subscription(
                    user_id=existing.id,
                    plan="pro",
                    is_active=True
                ))
                db.commit()
    except Exception as e:
        print(f"⚠️ Admin creation error: {e}")
        db.rollback()
    finally:
        db.close()

# ── Single Session ──────────────────────────────────────────────

def create_session_token(user_id: int) -> str:
    token = secrets.token_hex(32)
    db = SessionLocal()
    try:
        db.query(ActiveSession).filter(
            ActiveSession.user_id == user_id).delete()
        db.add(ActiveSession(
            user_id=user_id,
            session_token=token,
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Session create error: {e}")
    finally:
        db.close()
    return token

def validate_session_token(user_id: int, token: str) -> bool:
    db = SessionLocal()
    try:
        s = db.query(ActiveSession).filter_by(
            user_id=user_id,
            session_token=token
        ).first()
        if s:
            s.last_seen = datetime.utcnow()
            db.commit()
            return True
        return False
    except:
        return False
    finally:
        db.close()

def invalidate_session(user_id: int):
    db = SessionLocal()
    try:
        db.query(ActiveSession).filter(
            ActiveSession.user_id == user_id).delete()
        db.commit()
    except:
        db.rollback()
    finally:
        db.close()

# ── Subscription ────────────────────────────────────────────────

def get_user_plan(user_id: int) -> str:
    db = SessionLocal()
    try:
        sub = db.query(Subscription).filter_by(
            user_id=user_id, is_active=True).first()
        if sub:
            if sub.expires_at and sub.expires_at < datetime.utcnow():
                sub.is_active = False
                db.commit()
                return "free"
            return sub.plan
        return "free"
    except:
        return "free"
    finally:
        db.close()

def set_user_plan(user_id: int, plan: str,
                  razorpay_id: str = None, expires_at=None):
    db = SessionLocal()
    try:
        sub = db.query(Subscription).filter_by(user_id=user_id).first()
        if sub:
            sub.plan = plan
            sub.is_active = True
            if razorpay_id:
                sub.razorpay_id = razorpay_id
            if expires_at:
                sub.expires_at = expires_at
        else:
            db.add(Subscription(
                user_id=user_id, plan=plan,
                razorpay_id=razorpay_id,
                expires_at=expires_at,
                is_active=True
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Set plan error: {e}")
    finally:
        db.close()
