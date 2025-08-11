# models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float

Base = declarative_base()

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)

    # existing columns
    s3_url = Column(String, nullable=False)
    foul_type = Column(String, nullable=False)
    notes = Column(String)
    timestamp = Column(DateTime)  # your table is "timestamp without time zone"

    # new review/prediction fields (match your ALTER TABLE exactly)
    status = Column(String(20), nullable=False, default="queued")      # queued|processing|done|error
    prediction_label = Column(String(120))
    confidence = Column(Float)
    processed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)