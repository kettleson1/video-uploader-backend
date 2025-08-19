# models.py
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    text,           # included for convenience when doing raw SQL elsewhere
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)

    # Core upload fields
    s3_url = Column(String, nullable=False)
    foul_type = Column(String, nullable=False)
    notes = Column(String)

    # Original created-at (your table uses timestamp without time zone)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Review / prediction fields
    status = Column(String(20), nullable=False, default="queued")  # queued|processing|done|error
    prediction_label = Column(String(120))
    confidence = Column(Float)
    processed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)

    # NEW: model explanation / rationale text
    explanation = Column(Text)