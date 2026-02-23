import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agent.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserQuery(Base):
    __tablename__ = "user_queries"
    id            = Column(Integer, primary_key=True, index=True)
    query_text    = Column(Text)
    response_text = Column(Text)
    user_id       = Column(Integer)
    timestamp     = Column(DateTime, default=datetime.utcnow)

class Subscription(Base):
    """Tracks user subscription plan."""
    __tablename__ = "subscriptions"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, unique=True, index=True)
    plan        = Column(String, default="free")   # "free" | "pro"
    razorpay_id = Column(String, nullable=True)    # Razorpay payment/subscription ID
    started_at  = Column(DateTime, default=datetime.utcnow)
    expires_at  = Column(DateTime, nullable=True)
    is_active   = Column(Boolean, default=True)

class ActiveSession(Base):
    """One session per user — for single-device enforcement."""
    __tablename__ = "active_sessions"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, unique=True, index=True)
    session_token = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen  = Column(DateTime, default=datetime.utcnow)

class AffiliateLink(Base):
    """Admin-managed affiliate links shown before downloads."""
    __tablename__ = "affiliate_links"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String)           # e.g. "Try Razorpay"
    url         = Column(String)           # affiliate URL
    description = Column(Text, nullable=True)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

class QueryCount(Base):
    """Daily query counter for free-plan limits."""
    __tablename__ = "query_counts"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, index=True)
    date       = Column(String)
    count      = Column(Integer, default=0)

class ScreenPost(Base):
    """Admin posts on /screen — text or image + affiliate link."""
    __tablename__ = "screen_posts"
    id            = Column(Integer, primary_key=True, index=True)
    title         = Column(String)
    content       = Column(Text, nullable=True)
    image_data    = Column(Text, nullable=True)   # base64
    image_mime    = Column(String, nullable=True)
    affiliate_url = Column(String, nullable=True)
    post_type     = Column(String, default="text")  # "text" | "image"
    is_active     = Column(Boolean, default=True)
    order_num     = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)

class MusicTrack(Base):
    """Admin uploaded music tracks — stored as base64."""
    __tablename__ = "music_tracks"
    id         = Column(Integer, primary_key=True, index=True)
    title      = Column(String)
    artist     = Column(String, nullable=True)
    audio_data = Column(Text)        # base64 MP3
    mime_type  = Column(String, default="audio/mpeg")
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class VideoTrack(Base):
    """Admin uploaded video clips — stored as base64."""
    __tablename__ = "video_tracks"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String)
    description = Column(Text, nullable=True)
    video_data  = Column(Text)       # base64 MP4/WebM
    mime_type   = Column(String, default="video/mp4")
    thumbnail   = Column(Text, nullable=True)   # base64 thumbnail image
    thumb_mime  = Column(String, nullable=True)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
