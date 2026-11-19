"""
VIGIL AI – Database Setup
SQLAlchemy async SQLite configuration with table definitions.
"""
import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config import settings

# Convert sqlite:/// URL to aiosqlite format
DB_URL = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(DB_URL, echo=settings.debug)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Stores every video inference result."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)
    prediction = Column(String(50), nullable=False)       # "Violent" | "Non-Violent"
    violence_prob = Column(Float, nullable=False)
    nonviolence_prob = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)       # seconds
    model_used = Column(String(100), nullable=False)
    is_demo = Column(Boolean, default=False)
    # Ground-truth label (optional – set when evaluating on labelled data)
    ground_truth = Column(String(50), nullable=True)
    # Activity category (optional – from annotations CSV or manual input)
    activity_category = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class CategoryAnnotation(Base):
    """Stores manual category annotations for predictions."""
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    activity_category = Column(String(100), nullable=False)
    ground_truth = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


async def init_db():
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency: provide a database session."""
    async with AsyncSessionLocal() as session:
        yield session
