from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import relationship
from flask_login import UserMixin
from core.database import Base, SessionLocal
from datetime import datetime, timedelta
import os
import enum

# ✅ User Plan Enum (Free, Pro, Enterprise)
class UserPlan(enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class User(Base, UserMixin):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True)
    plan = Column(String, default=UserPlan.FREE.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # ✅ Relationship to queries
    queries = relationship("UserQuery", back_populates="user", lazy="dynamic")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_plan(self):
        return self.plan
    
    def set_plan(self, plan):
        if plan in [UserPlan.FREE.value, UserPlan.PRO.value, UserPlan.ENTERPRISE.value]:
            self.plan = plan
            return True
        return False

# ✅ Create Default Admin
def create_default_admin():
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            admin = User(
                username="admin",
                email="admin@example.com",
                plan=UserPlan.FREE.value
            )
            admin.set_password(admin_password)
            db.add(admin)
            db.commit()
            print(f"✅ Admin created: admin / {admin_password}")
        else:
            print("✅ Admin user already exists")
    except Exception as e:
        print(f"⚠️ Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()

# ✅ Session Token Functions (For API Authentication)
def create_session_token(user_id, expires_in_hours=24):
    """Create a session token for user"""
    import secrets
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(hours=expires_in_hours)
    
    db = SessionLocal()
    try:
        from core.database import SessionToken
        session_token = SessionToken(
            user_id=user_id,
            token=token,
            expires_at=expires
        )
        db.add(session_token)
        db.commit()
        return token
    except Exception as e:
        print(f"⚠️ Error creating session token: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def validate_session_token(token):
    """Validate session token"""
    db = SessionLocal()
    try:
        from core.database import SessionToken
        session_token = db.query(SessionToken).filter(
            SessionToken.token == token,
            SessionToken.expires_at > datetime.utcnow()
        ).first()
        
        if session_token:
            return session_token.user_id
        return None
    except Exception as e:
        print(f"⚠️ Error validating token: {e}")
        return None
    finally:
        db.close()

def invalidate_session(token):
    """Invalidate/delete session token"""
    db = SessionLocal()
    try:
        from core.database import SessionToken
        session_token = db.query(SessionToken).filter(SessionToken.token == token).first()
        if session_token:
            db.delete(session_token)
            db.commit()
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error invalidating token: {e}")
        db.rollback()
        return False
    finally:
        db.close()

# ✅ User Plan Functions (For Razorpay Integration)
def get_user_plan(user_id):
    """Get user's current plan"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return user.plan
        return UserPlan.FREE.value
    except Exception as e:
        print(f"⚠️ Error getting user plan: {e}")
        return UserPlan.FREE.value
    finally:
        db.close()

def set_user_plan(user_id, plan):
    """Set user's plan (after payment)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user and plan in [UserPlan.FREE.value, UserPlan.PRO.value, UserPlan.ENTERPRISE.value]:
            user.plan = plan
            user.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error setting user plan: {e}")
        db.rollback()
        return False
    finally:
        db.close()

# ✅ Check if User Has Permission
def has_permission(user_id, feature):
    """Check if user has permission for a feature based on plan"""
    plan = get_user_plan(user_id)
    
    # Free plan features
    free_features = ['basic_analysis', 'basic_charts', 'csv_export']
    
    # Pro plan features
    pro_features = ['advanced_analysis', 'dashboard', 'excel_export', 'email_reports']
    
    # Enterprise plan features
    enterprise_features = ['api_access', 'custom_models', 'priority_support', 'unlimited_exports']
    
    if plan == UserPlan.ENTERPRISE.value:
        return True
    elif plan == UserPlan.PRO.value:
        return feature in free_features + pro_features
    else:  # FREE
        return feature in free_features
