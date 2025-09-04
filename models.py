# models.py
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float
)
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector  # pip install pgvector

Base = declarative_base()


# -----------------------------
# Video upload & prediction row
# -----------------------------
class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)

    # S3 location of the uploaded video
    s3_url = Column(String, nullable=False)

    # User-supplied metadata
    foul_type = Column(String, nullable=False)
    notes = Column(String)

    # Created-at (server time)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Processing / prediction fields
    status = Column(String(20), nullable=False, default="queued")  # queued|processing|done|error
    prediction_label = Column(String(120))  # model’s predicted foul (or None)
    confidence = Column(Float)              # 0..1

    processed_at = Column(DateTime(timezone=True))  # when background job finished
    error_message = Column(Text)                    # any error captured (stringified)

    # Optional RAG explanation from the model
    explanation = Column(Text)                      # human-readable reasoning text


# -----------------------------
# Rules / retrieval corpus
# -----------------------------
class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)

    # Natural keys for upsert (title + section)
    title = Column(Text)     # e.g., "holding.txt"
    section = Column(Text)   # e.g., "chunk-0"

    # Text content of this rule chunk
    body = Column(Text, nullable=False)

    # 3072-dimensional pgvector for retrieval
    embedding = Column(Vector(3072))  # requires: CREATE EXTENSION vector;