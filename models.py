from sqlalchemy import Column, BigInteger, Text, Float, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from database import Base   # ✅ Base correctly imported from db setup


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(BigInteger, primary_key=True)
    game_id = Column(BigInteger, index=True)
    play_number = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)
    segment_confidence = Column(Float)
    clip_s3_url = Column(Text)
    s3_url = Column(Text, nullable=False)
    foul_type = Column(Text)
    notes = Column(Text)
    ruleset = Column(Text, default="high_school")
    timestamp = Column(DateTime)
    status = Column(Text)
    result = Column(Text)
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
    ruleset = Column(Text, default="high_school", index=True)
    section = Column(Text)
    body = Column(Text)

    def as_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "section": self.section,
            "body": self.body,
        }


class Game(Base):
    __tablename__ = "games"

    id = Column(BigInteger, primary_key=True)
    title = Column(Text)
    source_type = Column(Text, default="file")
    source_url = Column(Text)
    s3_url = Column(Text, nullable=False)
    ruleset = Column(Text, default="high_school")
    status = Column(Text, default="queued")
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    segment_count = Column(Integer, default=0)
    analyzed_count = Column(Integer, default=0)
    created_at = Column(DateTime)
    segmented_at = Column(DateTime)
    analyzed_at = Column(DateTime)
