from sqlalchemy import Column, BigInteger, Text, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(BigInteger, primary_key=True)
    s3_url = Column(Text, nullable=False)
    foul_type = Column(Text)
    notes = Column(Text)
    timestamp = Column(DateTime)
    status = Column(Text)
    prediction_label = Column(Text)
    confidence = Column(Float)
    processed_at = Column(DateTime)
    error_message = Column(Text)
    explanation = Column(Text)

    # New fields for expanded functionality
    retrieved_rules = Column(JSONB)        # store retrieved rules as JSON array
    human_label = Column(Text)             # corrected label from human reviewer
    human_notes = Column(Text)             # optional notes from reviewer
    reviewed_at = Column(DateTime)         # when human review was applied