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
    source_duration = Column(Float)


class Play(Base):
    __tablename__ = "plays"

    id = Column(BigInteger, primary_key=True)
    game_id = Column(BigInteger, nullable=False, index=True)
    legacy_upload_id = Column(BigInteger, unique=True, index=True)
    play_number = Column(Integer, nullable=False)
    source_start_time = Column(Float, nullable=False)
    source_end_time = Column(Float, nullable=False)
    snap_time = Column(Float)
    live_ball_end_time = Column(Float)
    clip_start_time = Column(Float)
    clip_end_time = Column(Float)
    clip_s3_url = Column(Text)
    detection_method = Column(Text, default="coaches_activity")
    detection_confidence = Column(Float)
    boundary_flags = Column(JSONB)
    boundary_review_status = Column(Text, default="unreviewed")
    analysis_status = Column(Text, default="pending")
    analysis_version = Column(Text)
    broad_screen = Column(Text)
    retrieved_rules = Column(JSONB)
    analysis_payload = Column(JSONB)
    result = Column(Text)
    prediction_label = Column(Text)
    confidence = Column(Float)
    notes = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    analyzed_at = Column(DateTime)


class PlayFinding(Base):
    __tablename__ = "play_findings"

    id = Column(BigInteger, primary_key=True)
    play_id = Column(BigInteger, nullable=False, index=True)
    disposition = Column(Text, nullable=False)
    foul_label = Column(Text)
    committing_team = Column(Text)
    player_number = Column(Text)
    event_time = Column(Float)
    confidence = Column(Float)
    evidence_summary = Column(Text)
    reasoning = Column(Text)
    rule_citations = Column(JSONB)
    uncertainty_notes = Column(Text)
    review_status = Column(Text, default="unreviewed")
    human_label = Column(Text)
    human_disposition = Column(Text)
    human_committing_team = Column(Text)
    human_player_number = Column(Text)
    human_notes = Column(Text)
    reviewed_at = Column(DateTime)
    created_by = Column(Text, default="ai")
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(BigInteger, primary_key=True)
    game_id = Column(BigInteger, nullable=False, index=True)
    play_id = Column(BigInteger, index=True)
    job_key = Column(Text, nullable=False, unique=True)
    stage = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="queued")
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    error_message = Column(Text)
    available_at = Column(DateTime)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
