from sqlalchemy import Column, BigInteger, Text, Float, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base   # ✅ Base correctly imported from db setup


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

    # New fields supporting human review
    retrieved_rules = Column(JSONB)
    human_label = Column(Text)
    human_notes = Column(Text)
    reviewed_at = Column(DateTime)


class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)

    def as_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content
        }