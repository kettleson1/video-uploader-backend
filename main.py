import os
import re
import io
import json
import base64
import hashlib
import hmac
import tempfile
import subprocess
import secrets
import time
from datetime import datetime, timezone
from typing import List, Optional
from models import Game, Play, PlayFinding, ProcessingJob, SegmentationComparison, Upload, Rule
from coaches_film_detector import (
    detect_cutup_play_boundaries,
    detect_play_boundaries,
    extract_activity_samples,
    extract_black_intervals,
    find_uncovered_intervals,
)
from job_queue import enqueue_job
from foul_findings import normalize_findings_payload
from segmentation_comparison import compare_segmentations

import boto3
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query, Body, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func, or_
from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text as sqltext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import asyncio
from openai import AsyncOpenAI

from database import async_session
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# Environment / Clients
# ------------------------------------------------------------------------------
DEFAULT_CORS_ALLOWED_ORIGINS = [
    "https://www.davesystemsinc.com",
    "https://davesystemsinc.com",
    "http://localhost:3000",
]


def _parse_cors_origins(value: Optional[str]) -> List[str]:
    if not value:
        return DEFAULT_CORS_ALLOWED_ORIGINS
    return [origin.strip() for origin in value.split(",") if origin.strip()]


AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072-D
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
PLAY_DETECTOR_MODE = os.getenv("PLAY_DETECTOR_MODE", "shadow_compare")
PLAY_FIRST_ACCEPTED = os.getenv("PLAY_FIRST_ACCEPTED", "false").strip().lower() in {"1", "true", "yes"}
LEGACY_GAME_SEGMENT_SECONDS = float(os.getenv("LEGACY_GAME_SEGMENT_SECONDS", "35"))
LEGACY_GAME_SEGMENT_GAP_SECONDS = float(os.getenv("LEGACY_GAME_SEGMENT_GAP_SECONDS", "5"))
DAVE_API_KEY = os.getenv("DAVE_API_KEY")
CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))

if not (BUCKET_NAME and DATABASE_URL and OPENAI_API_KEY):
    raise RuntimeError("Missing required env vars: AWS_S3_BUCKET, DATABASE_URL, OPENAI_API_KEY")

s3_client = boto3.client("s3", region_name=AWS_REGION)
SYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
engine = create_engine(SYNC_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

RULESETS = {
    "high_school": {
        "label": "NFHS high school football",
        "assistant": "an expert NFHS high school football officiating video analyst",
    },
    "college": {
        "label": "NCAA college football",
        "assistant": "an expert NCAA college football officiating video analyst",
    },
    "pro": {
        "label": "NFL pro football",
        "assistant": "an expert NFL pro football officiating video analyst",
    },
}

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Video Uploader API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def ensure_ruleset_columns():
    with engine.begin() as conn:
        conn.execute(sqltext("""
            CREATE TABLE IF NOT EXISTS games (
                id BIGSERIAL PRIMARY KEY,
                title TEXT,
                source_type TEXT DEFAULT 'file',
                source_url TEXT,
                s3_url TEXT NOT NULL,
                ruleset TEXT NOT NULL DEFAULT 'high_school',
                status TEXT DEFAULT 'queued',
                progress DOUBLE PRECISION DEFAULT 0,
                error_message TEXT,
                segment_count INTEGER DEFAULT 0,
                analyzed_count INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                segmented_at TIMESTAMP,
                analyzed_at TIMESTAMP
            )
        """))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS ruleset TEXT NOT NULL DEFAULT 'high_school'"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS game_id BIGINT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS play_number INTEGER"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS start_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS end_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS segment_confidence DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS clip_s3_url TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS uploads ADD COLUMN IF NOT EXISTS result TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS rules ADD COLUMN IF NOT EXISTS ruleset TEXT NOT NULL DEFAULT 'high_school'"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS games ADD COLUMN IF NOT EXISTS source_duration DOUBLE PRECISION"))
        conn.execute(sqltext("""
            CREATE TABLE IF NOT EXISTS plays (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL,
                legacy_upload_id BIGINT UNIQUE,
                play_number INTEGER NOT NULL,
                source_start_time DOUBLE PRECISION NOT NULL,
                source_end_time DOUBLE PRECISION NOT NULL,
                snap_time DOUBLE PRECISION,
                live_ball_end_time DOUBLE PRECISION,
                clip_start_time DOUBLE PRECISION,
                clip_end_time DOUBLE PRECISION,
                clip_s3_url TEXT,
                detection_method TEXT DEFAULT 'coaches_activity',
                detection_confidence DOUBLE PRECISION,
                boundary_flags JSONB,
                boundary_review_status TEXT DEFAULT 'unreviewed',
                analysis_status TEXT DEFAULT 'pending',
                analysis_version TEXT,
                analysis_revision INTEGER DEFAULT 0,
                broad_screen TEXT,
                retrieved_rules JSONB,
                analysis_payload JSONB,
                result TEXT,
                prediction_label TEXT,
                confidence DOUBLE PRECISION,
                notes TEXT,
                error_message TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                analyzed_at TIMESTAMP
            )
        """))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS snap_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS live_ball_end_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS boundary_flags JSONB"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS broad_screen TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS retrieved_rules JSONB"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS analysis_payload JSONB"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS analysis_revision INTEGER DEFAULT 0"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS original_start_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS original_end_time DOUBLE PRECISION"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS boundary_correction_count INTEGER DEFAULT 0"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS boundary_first_reviewed_at TIMESTAMP"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS plays ADD COLUMN IF NOT EXISTS boundary_last_reviewed_at TIMESTAMP"))
        conn.execute(sqltext("UPDATE plays SET original_start_time = source_start_time WHERE original_start_time IS NULL"))
        conn.execute(sqltext("UPDATE plays SET original_end_time = source_end_time WHERE original_end_time IS NULL"))
        conn.execute(sqltext("CREATE INDEX IF NOT EXISTS ix_plays_game_id ON plays (game_id)"))
        conn.execute(sqltext("CREATE UNIQUE INDEX IF NOT EXISTS ix_plays_legacy_upload_id ON plays (legacy_upload_id)"))
        conn.execute(sqltext("""
            INSERT INTO plays (
                game_id, legacy_upload_id, play_number,
                source_start_time, source_end_time, clip_start_time, clip_end_time,
                clip_s3_url, detection_method, detection_confidence,
                boundary_review_status, analysis_status, result,
                prediction_label, confidence, notes, error_message,
                created_at, updated_at, analyzed_at
            )
            SELECT
                game_id, id, COALESCE(play_number, 0),
                COALESCE(start_time, 0), COALESCE(end_time, 0), start_time, end_time,
                clip_s3_url, 'legacy_upload', segment_confidence,
                'unreviewed', COALESCE(status, 'pending'), result,
                prediction_label, confidence, notes, error_message,
                timestamp, COALESCE(processed_at, timestamp), processed_at
            FROM uploads
            WHERE game_id IS NOT NULL
            ON CONFLICT (legacy_upload_id) DO NOTHING
        """))
        conn.execute(sqltext("""
            CREATE TABLE IF NOT EXISTS play_findings (
                id BIGSERIAL PRIMARY KEY,
                play_id BIGINT NOT NULL,
                disposition TEXT NOT NULL,
                foul_label TEXT,
                committing_team TEXT,
                player_number TEXT,
                event_time DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                evidence_summary TEXT,
                reasoning TEXT,
                rule_citations JSONB,
                uncertainty_notes TEXT,
                review_status TEXT DEFAULT 'unreviewed',
                human_label TEXT,
                human_disposition TEXT,
                human_committing_team TEXT,
                human_player_number TEXT,
                human_notes TEXT,
                reviewed_at TIMESTAMP,
                created_by TEXT DEFAULT 'ai',
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        conn.execute(sqltext("ALTER TABLE IF EXISTS play_findings ADD COLUMN IF NOT EXISTS human_disposition TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS play_findings ADD COLUMN IF NOT EXISTS human_committing_team TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS play_findings ADD COLUMN IF NOT EXISTS human_player_number TEXT"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS play_findings ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS play_findings ADD COLUMN IF NOT EXISTS created_by TEXT DEFAULT 'ai'"))
        conn.execute(sqltext("CREATE INDEX IF NOT EXISTS ix_play_findings_play_id ON play_findings (play_id)"))
        conn.execute(sqltext("""
            INSERT INTO play_findings (
                play_id, disposition, foul_label, confidence, reasoning,
                rule_citations, review_status, human_label, human_notes,
                created_at, updated_at
            )
            SELECT
                p.id, COALESCE(u.result, 'UNCERTAIN'), u.prediction_label,
                u.confidence, u.explanation, '[]'::jsonb, 'unreviewed',
                u.human_label, u.human_notes,
                COALESCE(u.processed_at, u.timestamp), COALESCE(u.processed_at, u.timestamp)
            FROM uploads u
            JOIN plays p ON p.legacy_upload_id = u.id
            WHERE u.game_id IS NOT NULL
              AND u.result IN ('FOUL', 'UNCERTAIN')
              AND NOT EXISTS (
                  SELECT 1 FROM play_findings pf WHERE pf.play_id = p.id
              )
        """))
        conn.execute(sqltext("""
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL,
                play_id BIGINT,
                job_key TEXT,
                stage TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                error_message TEXT,
                available_at TIMESTAMP,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        conn.execute(sqltext("ALTER TABLE IF EXISTS processing_jobs ADD COLUMN IF NOT EXISTS job_key TEXT"))
        conn.execute(sqltext("UPDATE processing_jobs SET job_key = CONCAT('legacy:', id) WHERE job_key IS NULL"))
        conn.execute(sqltext("CREATE UNIQUE INDEX IF NOT EXISTS ix_processing_jobs_job_key ON processing_jobs (job_key)"))
        conn.execute(sqltext("ALTER TABLE IF EXISTS processing_jobs ALTER COLUMN job_key SET NOT NULL"))
        conn.execute(sqltext("CREATE INDEX IF NOT EXISTS ix_processing_jobs_game_id ON processing_jobs (game_id)"))
        conn.execute(sqltext("CREATE INDEX IF NOT EXISTS ix_processing_jobs_play_id ON processing_jobs (play_id)"))
        conn.execute(sqltext("""
            CREATE TABLE IF NOT EXISTS segmentation_comparisons (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL UNIQUE,
                detector_mode TEXT NOT NULL,
                selected_method TEXT NOT NULL,
                play_first_accepted BOOLEAN NOT NULL DEFAULT FALSE,
                legacy_count INTEGER NOT NULL DEFAULT 0,
                play_first_count INTEGER NOT NULL DEFAULT 0,
                matched_count INTEGER NOT NULL DEFAULT 0,
                unmatched_legacy_count INTEGER NOT NULL DEFAULT 0,
                unmatched_play_first_count INTEGER NOT NULL DEFAULT 0,
                possible_duplicate_play_first_count INTEGER NOT NULL DEFAULT 0,
                mean_matched_iou DOUBLE PRECISION DEFAULT 0,
                match_iou_threshold DOUBLE PRECISION DEFAULT 0.30,
                legacy_candidates JSONB,
                play_first_candidates JSONB,
                detector_error TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        conn.execute(sqltext("CREATE UNIQUE INDEX IF NOT EXISTS ix_segmentation_comparisons_game_id ON segmentation_comparisons (game_id)"))

# Dependency to get the database session
async def get_db():
    async with async_session() as session:
        yield session

def _decode_session_token(token: str) -> Optional[dict]:
    """Validate a short-lived token issued by the WordPress plugin."""
    try:
        encoded_payload, supplied_signature = token.split(".", 1)
        expected_signature = base64.urlsafe_b64encode(
            hmac.new(
                DAVE_API_KEY.encode("utf-8"),
                encoded_payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=").decode("ascii")
        if not secrets.compare_digest(supplied_signature, expected_signature):
            return None

        padding = "=" * (-len(encoded_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload + padding))
        now = int(time.time())
        expires_at = int(payload.get("exp", 0))
        if payload.get("v") != 1 or not payload.get("sub"):
            return None
        if expires_at <= now or expires_at > now + 3600:
            return None
        return payload
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def require_api_key(
    x_dave_api_key: Optional[str] = Header(None, alias="X-DAVE-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    if not DAVE_API_KEY:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if x_dave_api_key and secrets.compare_digest(x_dave_api_key, DAVE_API_KEY):
        return
    if authorization and authorization.startswith("Bearer "):
        if _decode_session_token(authorization[7:].strip()):
            return
    raise HTTPException(status_code=401, detail="Invalid or missing API authorization")

@app.get("/api/rules/list")
async def list_rules(
    ruleset: str = Query("high_school"),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    result = await db.execute(select(Rule).where(Rule.ruleset == _normalize_ruleset(ruleset)))
    rules = result.scalars().all()
    return [{"label": r.title, "value": r.title} for r in rules]

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class UploadResponse(BaseModel):
    id: int
    s3_url: str

class GameUploadResponse(BaseModel):
    id: int
    s3_url: str

class SegmentIn(BaseModel):
    id: Optional[int] = None
    play_number: Optional[int] = None
    start_time: float
    end_time: float
    notes: Optional[str] = None

class SegmentsPayload(BaseModel):
    segments: List[SegmentIn]

class ReviewPayload(BaseModel):
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    reviewed: bool = False

class OfficialFeedbackIn(BaseModel):
    official_agrees: bool
    human_label: Optional[str] = None
    human_notes: Optional[str] = None

class FindingReviewPayload(BaseModel):
    review_status: str
    human_disposition: Optional[str] = None
    human_label: Optional[str] = None
    human_committing_team: Optional[str] = None
    human_player_number: Optional[str] = None
    human_notes: Optional[str] = None

class ManualFindingPayload(BaseModel):
    disposition: str
    foul_label: str
    committing_team: Optional[str] = "unknown"
    player_number: Optional[str] = None
    event_time: Optional[float] = None
    confidence: float = 1.0
    evidence_summary: Optional[str] = None
    reasoning: Optional[str] = None
    rule_citations: List[str] = Field(default_factory=list)
    uncertainty_notes: Optional[str] = None

class PlayBoundaryPayload(BaseModel):
    start_time: float
    end_time: float
    notes: Optional[str] = None

class PlaySplitPayload(BaseModel):
    split_time: float

class PlayMergePayload(BaseModel):
    next_play_id: int

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _now_utc():
    return datetime.utcnow()

def _normalize_ruleset(ruleset: Optional[str]) -> str:
    normalized = (ruleset or "high_school").strip().lower().replace("-", "_")
    if normalized in {"nfhs", "highschool", "high_school"}:
        return "high_school"
    if normalized in {"ncaa", "college"}:
        return "college"
    if normalized in {"nfl", "pro", "professional"}:
        return "pro"
    raise HTTPException(status_code=400, detail=f"Unsupported ruleset: {ruleset}")

def _ruleset_context(ruleset: str) -> dict:
    return RULESETS.get(ruleset, RULESETS["high_school"])

def _safe_err_text(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return "Unknown error"

def _finding_to_dict(finding: PlayFinding) -> dict:
    return {
        "id": finding.id,
        "play_id": finding.play_id,
        "disposition": finding.disposition,
        "foul_label": finding.foul_label,
        "committing_team": finding.committing_team,
        "player_number": finding.player_number,
        "event_time": finding.event_time,
        "confidence": finding.confidence,
        "evidence_summary": finding.evidence_summary,
        "reasoning": finding.reasoning,
        "rule_citations": finding.rule_citations or [],
        "uncertainty_notes": finding.uncertainty_notes,
        "review_status": finding.review_status,
        "human_disposition": finding.human_disposition,
        "human_label": finding.human_label,
        "human_committing_team": finding.human_committing_team,
        "human_player_number": finding.human_player_number,
        "human_notes": finding.human_notes,
        "reviewed_at": finding.reviewed_at.isoformat() if finding.reviewed_at else None,
        "created_by": finding.created_by or "ai",
    }

async def _reset_play_for_reanalysis(db: AsyncSession, play: Play, clear_clip: bool = False):
    play.analysis_revision = int(play.analysis_revision or 0) + 1
    play.analysis_status = "queued_for_reanalysis"
    play.analysis_version = None
    play.broad_screen = None
    play.retrieved_rules = None
    play.analysis_payload = None
    play.result = None
    play.prediction_label = None
    play.confidence = None
    play.error_message = None
    play.analyzed_at = None
    play.updated_at = _now_utc()
    if clear_clip:
        play.clip_s3_url = None
    await db.execute(delete(PlayFinding).where(PlayFinding.play_id == play.id))
    upload = await db.get(Upload, play.legacy_upload_id)
    if upload:
        upload.status = "segmented"
        upload.result = None
        upload.prediction_label = None
        upload.confidence = None
        upload.error_message = None
        upload.explanation = None
        upload.retrieved_rules = None
        upload.processed_at = None
        if clear_clip:
            upload.clip_s3_url = None
            game = await db.get(Game, play.game_id)
            if game:
                upload.s3_url = game.s3_url

async def _enqueue_selective_analysis(db: AsyncSession, game: Game):
    signature_result = await db.execute(
        select(Play.id, Play.source_start_time, Play.source_end_time, Play.analysis_revision)
        .where(Play.game_id == game.id)
        .order_by(Play.play_number, Play.id)
    )
    signature = hashlib.sha256(json.dumps([
        [row.id, row.source_start_time, row.source_end_time, row.analysis_revision]
        for row in signature_result.all()
    ]).encode("utf-8")).hexdigest()[:16]
    await enqueue_job(
        db,
        job_key=f"game:{game.id}:extract_clips:selective:{signature}",
        game_id=game.id,
        stage="extract_clips",
    )
    game.status = "queued_for_analysis"
    game.error_message = None

def _assert_game_editable(game: Game):
    if game.status in {
        "queued_for_analysis", "extracting_clips", "analyzing",
        "screening", "retrieving_rules", "focused_analysis",
    }:
        raise HTTPException(status_code=409, detail="Play boundaries cannot change while analysis is running")

def _s3_key_from_url(url: str) -> str:
    # https://bucket.s3.region.amazonaws.com/<key>
    return url.split(".amazonaws.com/")[-1]

def _coerce_confidence(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    text = str(value).strip().lower()
    if not text:
        return default
    word_scores = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.60,
        "moderate": 0.60,
        "low": 0.35,
        "very low": 0.15,
    }
    if text in word_scores:
        return word_scores[text]
    if text.endswith("%"):
        try:
            return max(0.0, min(1.0, float(text[:-1]) / 100.0))
        except ValueError:
            return default
    try:
        num = float(text)
        if num > 1:
            num = num / 100.0
        return max(0.0, min(1.0, num))
    except ValueError:
        return default

def _presign(key: str, expires: int = 3600) -> Optional[str]:
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": BUCKET_NAME, "Key": key},
            ExpiresIn=expires,
        )
    except Exception:
        return None

def _extract_frames(video_bytes: bytes, fps: int = 1, max_frames: int = 6) -> List[str]:
    """
    Return a list of base64-encoded JPEG frames (strings).
    """
    frames_b64: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        out_tpl = os.path.join(td, "frame_%04d.jpg")
        # extract jpg frames
        cmd = [
            "ffmpeg", "-y",
            "-i", src,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            out_tpl
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        # load up to max_frames
        for name in sorted(os.listdir(td)):
            if not name.lower().endswith(".jpg"):
                continue
            if len(frames_b64) >= max_frames:
                break
            p = os.path.join(td, name)
            with open(p, "rb") as jf:
                b64 = base64.b64encode(jf.read()).decode("utf-8")  # defensive: return str
                frames_b64.append(b64)
    return frames_b64

def _probe_video_duration(video_bytes: bytes) -> float:
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "game.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        try:
            return max(0.0, float((proc.stdout or "0").strip()))
        except ValueError:
            return 0.0

def _detect_play_segments(video_bytes: bytes) -> List[dict]:
    """
    Detect editable play candidates from continuous coaches film.
    """
    return _detect_play_segments_with_comparison(video_bytes)[0]


def _detect_play_segments_with_comparison(video_bytes: bytes):
    """Detect candidates and return the optional legacy/play-first comparison."""
    duration = _probe_video_duration(video_bytes)
    if duration <= 0:
        return [], None
    legacy = _detect_play_segments_fixed_window(duration)
    if PLAY_DETECTOR_MODE == "fixed_window":
        return legacy, None
    detector_error = None
    try:
        samples = extract_activity_samples(video_bytes)
        black_intervals = extract_black_intervals(video_bytes)
        play_first = detect_cutup_play_boundaries(
            black_intervals, duration, samples=samples
        ) or detect_play_boundaries(samples, duration)
    except Exception as error:
        if PLAY_DETECTOR_MODE not in {"coaches_activity_with_fallback", "shadow_compare"}:
            raise
        play_first = []
        detector_error = _safe_err_text(error)

    if PLAY_DETECTOR_MODE == "shadow_compare":
        metrics = compare_segmentations(legacy, play_first)
        selected_method = (
            "coaches_activity" if PLAY_FIRST_ACCEPTED and play_first else "legacy_fixed_window"
        )
        selected = play_first if selected_method == "coaches_activity" else legacy
        for candidate in selected:
            candidate.setdefault("detection_method", selected_method)
        comparison = {
            **metrics,
            "detector_mode": PLAY_DETECTOR_MODE,
            "selected_method": selected_method,
            "play_first_accepted": PLAY_FIRST_ACCEPTED,
            "legacy_candidates": legacy,
            "play_first_candidates": play_first,
            "detector_error": detector_error,
        }
        return selected, comparison

    if play_first or PLAY_DETECTOR_MODE != "coaches_activity_with_fallback":
        for candidate in play_first:
            candidate.setdefault("detection_method", "coaches_activity")
        return play_first, None
    for candidate in legacy:
        candidate["detection_method"] = "legacy_fixed_window"
    return legacy, None


def _detect_play_segments_fixed_window(duration: float) -> List[dict]:
    """Temporary legacy fallback retained until play detection is validated."""
    segment_seconds = max(8.0, LEGACY_GAME_SEGMENT_SECONDS)
    gap_seconds = max(0.0, LEGACY_GAME_SEGMENT_GAP_SECONDS)
    step = segment_seconds + gap_seconds
    segments: List[dict] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + segment_seconds)
        if end - start >= 4.0:
            segments.append({
                "play_number": len(segments) + 1,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "segment_confidence": 0.35,
            })
        start += step
    return segments

def _extract_clip_bytes(video_bytes: bytes, start_time: float, end_time: float) -> bytes:
    start_time = max(0.0, float(start_time))
    duration = max(1.0, float(end_time) - start_time)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "game.mp4")
        out = os.path.join(td, "clip.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("Clip extraction failed")
        with open(out, "rb") as f:
            return f.read()

def _result_from_label(label: str, confidence: float) -> str:
    normalized = (label or "").strip().lower().replace("_", " ").replace("-", " ")
    if confidence < CONFIDENCE_THRESHOLD:
        return "UNCERTAIN"
    if normalized in {"", "none", "no foul", "no penalty"}:
        return "NO FOUL"
    if normalized == "uncertain":
        return "UNCERTAIN"
    return "FOUL"

async def _summarize_frames_async(
    frames_b64: List[str],
    foul_hint: str = "",
    notes: str = "",
    ruleset: str = "high_school",
) -> str:
    """
    Async: Summarize the play from a handful of actual video frames.
    """
    if not frames_b64:
        return "No visual context available."

    rules_context = _ruleset_context(ruleset)
    system_prompt = (
        f"You are {rules_context['assistant']}. "
        "Describe only what is visible in the frames. Be precise about timing, "
        "player roles, ball location, contact type, target area, and whether the "
        "action is clear or ambiguous. Do not invent unseen action."
    )

    user_text = (
        f"Analyze these {len(frames_b64)} chronological frames from one football play.\n"
        f"Ruleset to apply later: {rules_context['label']}\n"
        f"Selected foul type from the form: {foul_hint or 'not provided'}\n"
        f"Official/user notes: {notes or 'none'}\n\n"
        "Return a compact evidence summary for rule lookup with these fields:\n"
        "- play phase\n"
        "- key visible action\n"
        "- timing relative to snap/pass/kick/ball arrival if visible\n"
        "- players involved\n"
        "- possible foul indicators\n"
        "- uncertainty or missing camera evidence\n"
        "Keep it under 180 words."
    )

    content: list[dict] = [{"type": "text", "text": user_text}]
    for frame in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}",
                "detail": os.getenv("OPENAI_IMAGE_DETAIL", "high"),
            },
        })

    chat = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=320,
    )
    return (chat.choices[0].message.content or "").strip()

async def _embed_text(text_in: str) -> List[float]:
    """
    Async: Create a 3072-D embedding (text-embedding-3-large).
    """
    resp = await client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text_in)
    vec = resp.data[0].embedding
    return [float(x) for x in vec]



# ------------------------------------------------------------------------------
# Async rule retrieval using pgvector
# ------------------------------------------------------------------------------
async def _retrieve_rules_async(summary: str, top_k: int = 3, ruleset: str = "high_school") -> List[dict]:
    """
    Async: Retrieve closest rule chunks via pgvector (cosine).
    Column type: vector(3072)
    """
    emb = await _embed_text(summary)  # 3072-D
    qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    sql = sqltext("""
        SELECT
            id,
            title,
            section,
            body,
            1 - (embedding <=> (:qvec)::vector) AS score
        FROM rules
        WHERE ruleset = :ruleset
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT :k
    """)

    async with async_session() as session:
        rows = (await session.execute(sql, {
            "qvec": qvec_literal,
            "k": top_k,
            "ruleset": ruleset,
        })).mappings().all()

    # Make sure everything is JSON-serializable (no Decimals)
    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "title": r["title"],
            "section": r["section"],
            "body": r["body"],
            "score": float(r["score"]) if r["score"] is not None else None,
        })
    return out

async def _predict_with_rules(
    summary: str,
    retrieved: list[dict],
    foul_hint: str = "",
    notes: str = "",
    ruleset: str = "high_school",
) -> tuple[str, float, str]:
    """
    Use LLM to choose a label and produce confidence + explanation.
    Returned: (label, confidence[0..1], explanation)
    """
    # Build rule snippets text safely (avoid nested quotes in f-strings)
    parts: list[str] = []
    for r in (retrieved or []):
        title = str(r.get("title", ""))
        section = r.get("section")
        section_str = f" ({section})" if section else ""
        body = str(r.get("body", ""))
        parts.append(f"- {title}{section_str}\n{body}")
    rules_snips = "\n\n".join(parts).strip() or "No matching rule snippets."

    rules_context = _ruleset_context(ruleset)
    sys_prompt = (
        f"You are a {rules_context['label']} rules assistant. "
        "Choose the most likely foul label only when the video evidence and rule snippets support it. "
        "If the evidence is unclear or no provided rule fits, use 'None' or 'Uncertain'. "
        "Ground the explanation in visible evidence and the cited snippets."
    )
    user_prompt = (
        f"PLAY SUMMARY:\n{summary}\n\n"
        f"RULESET:\n{rules_context['label']}\n\n"
        f"SELECTED FOUL TYPE/HINT:\n{foul_hint or 'not provided'}\n\n"
        f"OFFICIAL/USER NOTES:\n{notes or 'none'}\n\n"
        f"CANDIDATE RULE SNIPPETS:\n{rules_snips}\n\n"
        "Respond as JSON with keys: label, confidence, explanation. "
        "The explanation must include the candidate rule title or section used."
    )

    chat = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    txt = (chat.choices[0].message.content or "").strip()

    label, conf, expl = "None", 0.0, ""
    try:
        if txt.startswith("```"):
            txt = "\n".join(txt.splitlines()[1:])
        if txt.endswith("```"):
            txt = "\n".join(txt.splitlines()[:-1])
        j = json.loads(txt)
        label = str(j.get("label", "None"))
        conf = _coerce_confidence(j.get("confidence"), 0.0)
        expl = str(j.get("explanation", "")).strip()
    except Exception:
        expl = f"Model returned non-JSON output: {txt[:300]}"

    if not expl:
        top_rule = retrieved[0] if retrieved else {}
        rule_title = top_rule.get("title") or "the retrieved rule snippets"
        expl = (
            f"Prediction is based on the visible play summary, the foul hint "
            f"'{foul_hint or 'not provided'}', and the top retrieved rule {rule_title}."
        )
    return label, conf, expl

async def _predict_batch_with_rules(
    summary: str,
    retrieved: list[dict],
    notes: str = "",
    ruleset: str = "high_school",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> dict:
    parts: list[str] = []
    for r in (retrieved or []):
        title = str(r.get("title", ""))
        section = r.get("section")
        section_str = f" ({section})" if section else ""
        body = str(r.get("body", ""))
        parts.append(f"- {title}{section_str}\n{body}")
    rules_snips = "\n\n".join(parts).strip() or "No matching rule snippets."

    rules_context = _ruleset_context(ruleset)
    time_text = ""
    if start_time is not None and end_time is not None:
        time_text = f"\nPLAY TIMESTAMP: {start_time:.2f}s to {end_time:.2f}s\n"

    sys_prompt = (
        f"You are a {rules_context['label']} officiating review assistant. "
        "You are not making an official ruling. Identify every independently supported foul "
        "or uncertain foul candidate visible on the play. Do not decide enforcement, acceptance, "
        "declination, offsetting administration, or disqualification. Only classify a foul when "
        "the visible evidence and rule snippets support its required elements."
    )
    user_prompt = (
        f"PLAY SUMMARY:\n{summary}\n"
        f"{time_text}\n"
        f"OFFICIAL/USER NOTES:\n{notes or 'none'}\n\n"
        f"CANDIDATE RULE SNIPPETS:\n{rules_snips}\n\n"
        "Respond as JSON with exactly these top-level keys:\n"
        "- confidence: confidence that the play review is complete, 0 to 1\n"
        "- play_summary: concise visible evidence for the entire play\n"
        "- no_foul_reason: why no foul is supported; use an empty string when findings exist\n"
        "- uncertainty_notes: important missing angles or timing evidence\n"
        "- findings: an array with zero or more independent findings\n\n"
        "Each finding must contain: disposition (FOUL or UNCERTAIN), foul_label, "
        "committing_team (offense, defense, kicking, receiving, or unknown), player_number "
        "(string or null), event_time (seconds from the start of this play clip or null), "
        "confidence, evidence_summary, reasoning, rule_citations (array), and uncertainty_notes.\n"
        "Never put NO FOUL in findings. Return an empty findings array when no foul candidate is supported."
    )

    chat = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    txt = (chat.choices[0].message.content or "").strip()

    fallback = {
        "result": "UNCERTAIN",
        "label": "Uncertain",
        "confidence": 0.0,
        "rule_citations": [],
        "evidence_summary": summary,
        "why_foul_or_no_foul": "The model did not return a usable structured result.",
        "uncertainty_notes": txt[:300],
        "findings": [{
            "disposition": "UNCERTAIN",
            "foul_label": "Uncertain",
            "confidence": 0.0,
            "evidence_summary": summary,
            "reasoning": "The model did not return a usable structured result.",
            "rule_citations": [],
            "uncertainty_notes": txt[:300],
        }],
    }
    try:
        if txt.startswith("```"):
            txt = "\n".join(txt.splitlines()[1:])
        if txt.endswith("```"):
            txt = "\n".join(txt.splitlines()[:-1])
        data = json.loads(txt)
    except Exception:
        return normalize_findings_payload(
            fallback, summary, CONFIDENCE_THRESHOLD,
            (end_time - start_time) if start_time is not None and end_time is not None else None,
        )

    return normalize_findings_payload(
        data, summary, CONFIDENCE_THRESHOLD,
        (end_time - start_time) if start_time is not None and end_time is not None else None,
    )
# ------------------------------------------------------------------------------
# Background worker (async version)
# ------------------------------------------------------------------------------
async def _process_upload_bg(
    upload_id: int,
    s3_url: str,
    foul_hint: str,
    notes: str = "",
    ruleset: str = "high_school",
):
    async with async_session() as db:
        try:
            print(f"📥 Starting processing for upload_id={upload_id}")
             
            # 1) Download bytes
            key = _s3_key_from_url(s3_url)
            obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            video_bytes: bytes = obj["Body"].read()

            print("✅ Video downloaded from S3")

            # 2) Frames -> summary
            frames = _extract_frames(video_bytes, fps=1, max_frames=6)
            print(f"📸 Extracted {len(frames)} frames")

            ruleset = _normalize_ruleset(ruleset)
            summary = await _summarize_frames_async(
                frames,
                foul_hint=foul_hint,
                notes=notes,
                ruleset=ruleset,
            )
            print(f"✍️ Summary: {summary}")

            # 3) Retrieve rules (async)
            retrieval_query = "\n".join(
                part for part in [summary, f"Foul hint: {foul_hint}" if foul_hint else "", notes] if part
            )
            retrieved = await _retrieve_rules_async(retrieval_query, top_k=6, ruleset=ruleset)
            print(f"📚 Retrieved {len(retrieved)} rules")
            
            # 4) Decide with LLM
            label, confidence, explanation_text = await _predict_with_rules(
                summary,
                retrieved,
                foul_hint=foul_hint,
                notes=notes,
                ruleset=ruleset,
            )
            print(f"📊 Predicted: {label} (confidence={confidence})")

            # 4b) Confidence thresholding
            final_label = label
            final_conf = float(confidence)
            if final_conf < CONFIDENCE_THRESHOLD:
                final_label = "Uncertain"

            # 5) Update DB (also SAVE retrieved_rules)
            result = await db.execute(select(Upload).where(Upload.id == upload_id))
            row = result.scalar_one_or_none()

            if row:
                row.status = "done"
                row.prediction_label = final_label
                row.result = _result_from_label(final_label, final_conf)
                row.confidence = final_conf
                row.explanation = explanation_text
                row.processed_at = _now_utc()
                row.error_message = None
                row.retrieved_rules = retrieved
                await db.commit()
        except Exception as e:
            print(f"❌ Error in _process_upload_bg: {str(e)}")
            result = await db.execute(select(Upload).where(Upload.id == upload_id))
            row = result.scalar_one_or_none()
            if row:
                row.status = "error"
                row.error_message = _safe_err_text(e)
                row.processed_at = _now_utc()
                await db.commit()

# ------------------------------------------------------------------------------
# Batch game processing
# ------------------------------------------------------------------------------
def _game_to_dict(game: Game) -> dict:
    return {
        "id": game.id,
        "title": game.title,
        "source_type": game.source_type,
        "source_url": game.source_url,
        "s3_url": game.s3_url,
        "ruleset": game.ruleset,
        "status": game.status,
        "progress": game.progress,
        "error_message": game.error_message,
        "segment_count": game.segment_count,
        "analyzed_count": game.analyzed_count,
        "created_at": game.created_at.isoformat() if game.created_at else None,
        "segmented_at": game.segmented_at.isoformat() if game.segmented_at else None,
        "analyzed_at": game.analyzed_at.isoformat() if game.analyzed_at else None,
    }

def _play_to_dict(
    play: Upload,
    timeline_play: Optional[Play] = None,
    findings: Optional[List[PlayFinding]] = None,
) -> dict:
    video_url = play.clip_s3_url or play.s3_url
    key = _s3_key_from_url(video_url) if video_url else ""
    presigned = _presign(key, 3600) if key else None
    official_agrees = None
    if getattr(play, "reviewed_at", None) and getattr(play, "human_label", None) and play.prediction_label:
        official_agrees = play.human_label == play.prediction_label
    return {
        "id": play.id,
        "game_id": play.game_id,
        "play_number": play.play_number,
        "start_time": play.start_time,
        "end_time": play.end_time,
        "segment_confidence": play.segment_confidence,
        "timeline_play_id": timeline_play.id if timeline_play else None,
        "detection_method": timeline_play.detection_method if timeline_play else None,
        "boundary_flags": (timeline_play.boundary_flags or []) if timeline_play else [],
        "snap_time": timeline_play.snap_time if timeline_play else None,
        "live_ball_end_time": timeline_play.live_ball_end_time if timeline_play else None,
        "foul_type": play.foul_type,
        "notes": play.notes,
        "ruleset": getattr(play, "ruleset", "high_school"),
        "timestamp": play.timestamp.isoformat() if play.timestamp else None,
        "status": play.status,
        "result": play.result,
        "prediction_label": play.prediction_label,
        "confidence": play.confidence,
        "processed_at": play.processed_at.isoformat() if play.processed_at else None,
        "error_message": play.error_message,
        "explanation": getattr(play, "explanation", None),
        "retrieved_rules": getattr(play, "retrieved_rules", None),
        "human_label": getattr(play, "human_label", None),
        "human_notes": getattr(play, "human_notes", None),
        "reviewed_at": play.reviewed_at.isoformat() if getattr(play, "reviewed_at", None) else None,
        "official_agrees": official_agrees,
        "s3_url": play.s3_url,
        "clip_s3_url": play.clip_s3_url,
        "presigned_url": presigned,
        "findings": [_finding_to_dict(finding) for finding in (findings or [])],
    }

async def _segment_game_bg(game_id: int):
    async with async_session() as db:
        try:
            result = await db.execute(select(Game).where(Game.id == game_id))
            game = result.scalar_one_or_none()
            if not game:
                return

            game.status = "segmenting"
            game.progress = 0.05
            game.error_message = None
            job_result = await db.execute(
                select(ProcessingJob)
                .where(ProcessingJob.game_id == game_id, ProcessingJob.stage == "detect_plays")
                .order_by(ProcessingJob.id.desc())
            )
            job = job_result.scalars().first()
            if job:
                job.status = "running"
                job.attempts = int(job.attempts or 0) + 1
                job.started_at = _now_utc()
                job.updated_at = _now_utc()
            await db.commit()

            key = _s3_key_from_url(game.s3_url)
            obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            video_bytes: bytes = obj["Body"].read()
            segments = _detect_play_segments(video_bytes)

            if not segments:
                raise RuntimeError("No playable segments could be detected in the uploaded video.")

            game.source_duration = _probe_video_duration(video_bytes)

            for segment in segments:
                upload = Upload(
                    game_id=game.id,
                    play_number=segment["play_number"],
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    segment_confidence=segment["segment_confidence"],
                    s3_url=game.s3_url,
                    foul_type="",
                    notes="",
                    ruleset=game.ruleset,
                    timestamp=_now_utc(),
                    status="segmented",
                    result=None,
                )
                db.add(upload)
                await db.flush()
                db.add(Play(
                    game_id=game.id,
                    legacy_upload_id=upload.id,
                    play_number=upload.play_number,
                    source_start_time=upload.start_time,
                    source_end_time=upload.end_time,
                    original_start_time=upload.start_time,
                    original_end_time=upload.end_time,
                    snap_time=segment.get("snap_time"),
                    live_ball_end_time=segment.get("live_ball_end_time"),
                    clip_start_time=upload.start_time,
                    clip_end_time=upload.end_time,
                    detection_method=segment.get("detection_method", "coaches_activity"),
                    detection_confidence=upload.segment_confidence,
                    boundary_flags=segment.get("boundary_flags") or [],
                    boundary_review_status="unreviewed",
                    analysis_status="segmented",
                    notes="",
                    created_at=_now_utc(),
                    updated_at=_now_utc(),
                ))

            game.status = "segments_ready"
            game.progress = 0.25
            game.segment_count = len(segments)
            game.segmented_at = _now_utc()
            if job:
                job.status = "completed"
                job.finished_at = _now_utc()
                job.updated_at = _now_utc()
            await db.commit()
        except Exception as e:
            result = await db.execute(select(Game).where(Game.id == game_id))
            game = result.scalar_one_or_none()
            if game:
                game.status = "error"
                game.error_message = _safe_err_text(e)
                job_result = await db.execute(
                    select(ProcessingJob)
                    .where(ProcessingJob.game_id == game_id, ProcessingJob.stage == "detect_plays")
                    .order_by(ProcessingJob.id.desc())
                )
                job = job_result.scalars().first()
                if job:
                    job.status = "failed"
                    job.error_message = _safe_err_text(e)
                    job.finished_at = _now_utc()
                    job.updated_at = _now_utc()
                await db.commit()

async def _analyze_game_bg(game_id: int):
    async with async_session() as db:
        try:
            result = await db.execute(select(Game).where(Game.id == game_id))
            game = result.scalar_one_or_none()
            if not game:
                return

            plays_result = await db.execute(
                select(Upload)
                .where(Upload.game_id == game_id)
                .order_by(Upload.play_number.asc(), Upload.start_time.asc())
            )
            plays = plays_result.scalars().all()
            if not plays:
                game.status = "error"
                game.error_message = "No play segments are available to analyze."
                await db.commit()
                return

            game.status = "analyzing"
            game.progress = 0.30
            game.error_message = None
            game.analyzed_count = 0
            job_result = await db.execute(
                select(ProcessingJob)
                .where(ProcessingJob.game_id == game_id, ProcessingJob.stage == "analyze_plays")
                .order_by(ProcessingJob.id.desc())
            )
            job = job_result.scalars().first()
            if job:
                job.status = "running"
                job.attempts = int(job.attempts or 0) + 1
                job.started_at = _now_utc()
                job.updated_at = _now_utc()
            await db.commit()

            key = _s3_key_from_url(game.s3_url)
            obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            video_bytes: bytes = obj["Body"].read()
            total = len(plays)

            failed_count = 0
            for index, play in enumerate(plays, start=1):
                play.status = "processing"
                play.error_message = None
                timeline_result = await db.execute(
                    select(Play).where(Play.legacy_upload_id == play.id)
                )
                timeline_play = timeline_result.scalar_one_or_none()
                if timeline_play:
                    timeline_play.analysis_status = "processing"
                    timeline_play.error_message = None
                    timeline_play.updated_at = _now_utc()
                await db.commit()

                try:
                    clip_bytes = _extract_clip_bytes(video_bytes, play.start_time or 0.0, play.end_time or 0.0)
                    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                    key_name = f"games/{game.id}/plays/{play.play_number or index}_{stamp}.mp4"
                    s3_client.put_object(Bucket=BUCKET_NAME, Key=key_name, Body=clip_bytes, ContentType="video/mp4")
                    clip_s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key_name}"

                    frames = _extract_frames(clip_bytes, fps=1, max_frames=8)
                    summary = await _summarize_frames_async(
                        frames,
                        foul_hint="",
                        notes=play.notes or "",
                        ruleset=game.ruleset,
                    )
                    retrieval_query = "\n".join(
                        part for part in [
                            summary,
                            f"Play timestamp: {play.start_time:.2f}s to {play.end_time:.2f}s",
                            play.notes or "",
                        ] if part
                    )
                    retrieved = await _retrieve_rules_async(retrieval_query, top_k=6, ruleset=game.ruleset)
                    prediction = await _predict_batch_with_rules(
                        summary,
                        retrieved,
                        notes=play.notes or "",
                        ruleset=game.ruleset,
                        start_time=play.start_time,
                        end_time=play.end_time,
                    )

                    label = prediction["label"]
                    explanation_payload = {
                        "rule_citations": prediction["rule_citations"],
                        "evidence_summary": prediction["evidence_summary"],
                        "why_foul_or_no_foul": prediction["why_foul_or_no_foul"],
                        "uncertainty_notes": prediction["uncertainty_notes"],
                    }

                    play.clip_s3_url = clip_s3_url
                    play.s3_url = clip_s3_url
                    play.status = "done"
                    play.result = prediction["result"]
                    play.prediction_label = label
                    play.confidence = prediction["confidence"]
                    play.explanation = json.dumps(explanation_payload)
                    play.retrieved_rules = retrieved
                    play.processed_at = _now_utc()
                    play.error_message = None

                    if timeline_play:
                        timeline_play.clip_s3_url = clip_s3_url
                        timeline_play.analysis_status = "done"
                        timeline_play.analysis_version = "legacy-single-finding-v1"
                        timeline_play.result = prediction["result"]
                        timeline_play.prediction_label = label
                        timeline_play.confidence = prediction["confidence"]
                        timeline_play.error_message = None
                        timeline_play.updated_at = _now_utc()
                        timeline_play.analyzed_at = _now_utc()
                        await db.execute(
                            delete(PlayFinding).where(PlayFinding.play_id == timeline_play.id)
                        )
                        if prediction["result"] != "NO FOUL":
                            db.add(PlayFinding(
                                play_id=timeline_play.id,
                                disposition=prediction["result"],
                                foul_label=label,
                                confidence=prediction["confidence"],
                                evidence_summary=prediction["evidence_summary"],
                                reasoning=prediction["why_foul_or_no_foul"],
                                rule_citations=prediction["rule_citations"],
                                uncertainty_notes=prediction["uncertainty_notes"],
                                review_status="unreviewed",
                                created_at=_now_utc(),
                                updated_at=_now_utc(),
                            ))
                except Exception as play_error:
                    play.status = "error"
                    play.error_message = _safe_err_text(play_error)
                    play.processed_at = _now_utc()
                    if timeline_play:
                        timeline_play.analysis_status = "error"
                        timeline_play.error_message = _safe_err_text(play_error)
                        timeline_play.updated_at = _now_utc()
                    failed_count += 1

                game.analyzed_count = index
                game.progress = 0.30 + (0.70 * index / total)
                await db.commit()

            game.status = "done" if failed_count == 0 else "error"
            game.progress = 1.0
            game.analyzed_at = _now_utc()
            game.error_message = (
                None
                if failed_count == 0
                else f"{failed_count} of {total} play analyses failed."
            )
            if job:
                job.status = "completed" if failed_count == 0 else "failed"
                job.error_message = game.error_message
                job.finished_at = _now_utc()
                job.updated_at = _now_utc()
            await db.commit()
        except Exception as e:
            result = await db.execute(select(Game).where(Game.id == game_id))
            game = result.scalar_one_or_none()
            if game:
                game.status = "error"
                game.error_message = _safe_err_text(e)
                job_result = await db.execute(
                    select(ProcessingJob)
                    .where(ProcessingJob.game_id == game_id, ProcessingJob.stage == "analyze_plays")
                    .order_by(ProcessingJob.id.desc())
                )
                job = job_result.scalars().first()
                if job:
                    job.status = "failed"
                    job.error_message = _safe_err_text(e)
                    job.finished_at = _now_utc()
                    job.updated_at = _now_utc()
                await db.commit()

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

    # ---- Human Review API ----
class ReviewIn(BaseModel):
    label: str
    notes: Optional[str] = None

@app.post("/api/games/upload", response_model=GameUploadResponse)
async def upload_game_video(
    file: UploadFile = File(...),
    title: str = Form(""),
    ruleset: str = Form("high_school"),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    try:
        ruleset = _normalize_ruleset(ruleset)
        data = await file.read()
        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "game.mp4")
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        key = f"games/{stamp}_{clean_name}"
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=file.content_type or "video/mp4")
        s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

        rec = Game(
            title=title.strip() or clean_name,
            source_type="file",
            source_url=None,
            s3_url=s3_url,
            ruleset=ruleset,
            status="queued",
            progress=0.0,
            created_at=_now_utc(),
        )
        db.add(rec)
        await db.flush()
        await enqueue_job(
            db,
            job_key=f"game:{rec.id}:validate_upload:v1",
            game_id=rec.id,
            stage="validate_upload",
        )
        await db.commit()
        return GameUploadResponse(id=rec.id, s3_url=s3_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_err_text(e))

@app.get("/api/games")
async def list_games(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ruleset: str = Query("high_school"),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    ruleset = _normalize_ruleset(ruleset)
    result = await db.execute(
        select(Game)
        .where(Game.ruleset == ruleset)
        .order_by(Game.id.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = result.scalars().all()
    total_result = await db.execute(select(func.count()).select_from(Game).where(Game.ruleset == ruleset))
    return {
        "items": [_game_to_dict(row) for row in rows],
        "total": total_result.scalar_one(),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/migration/readiness")
async def get_migration_readiness(
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game_count = await db.scalar(select(func.count()).select_from(Game)) or 0
    comparison_count = await db.scalar(
        select(func.count()).select_from(SegmentationComparison)
    ) or 0
    unmigrated_batch_uploads = await db.scalar(
        select(func.count()).select_from(Upload).where(
            Upload.game_id.isnot(None),
            ~Upload.id.in_(select(Play.legacy_upload_id).where(Play.legacy_upload_id.isnot(None))),
        )
    ) or 0
    games_without_plays = await db.scalar(
        select(func.count()).select_from(Game).where(
            ~Game.id.in_(select(Play.game_id))
        )
    ) or 0
    orphan_findings = await db.scalar(
        select(func.count()).select_from(PlayFinding).where(
            ~PlayFinding.play_id.in_(select(Play.id))
        )
    ) or 0
    orphan_jobs = await db.scalar(
        select(func.count()).select_from(ProcessingJob).where(
            ~ProcessingJob.game_id.in_(select(Game.id))
        )
    ) or 0
    integrity_ready = not any((
        unmigrated_batch_uploads, games_without_plays, orphan_findings, orphan_jobs,
    ))
    return {
        "detector_mode": PLAY_DETECTOR_MODE,
        "play_first_accepted": PLAY_FIRST_ACCEPTED,
        "legacy_fallback_available": True,
        "game_count": game_count,
        "comparison_game_count": comparison_count,
        "unmigrated_batch_upload_count": unmigrated_batch_uploads,
        "games_without_plays_count": games_without_plays,
        "orphan_finding_count": orphan_findings,
        "orphan_job_count": orphan_jobs,
        "existing_games_readable_through_new_model": integrity_ready,
        "ready_to_make_play_first_default": bool(
            PLAY_FIRST_ACCEPTED and comparison_count > 0 and integrity_ready
        ),
        "ready_to_remove_legacy": False,
        "legacy_removal_blockers": [
            "retain the administrative fallback for one accepted release",
            "verify rollback is no longer required",
        ],
    }


@app.get("/api/games/{game_id}/segmentation-comparison")
async def get_game_segmentation_comparison(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    comparison_result = await db.execute(
        select(SegmentationComparison).where(SegmentationComparison.game_id == game_id)
    )
    comparison = comparison_result.scalar_one_or_none()
    plays_result = await db.execute(select(Play).where(Play.game_id == game_id))
    plays = plays_result.scalars().all()
    corrected = [play for play in plays if int(play.boundary_correction_count or 0) > 0]
    boundary_delta_seconds = sum(
        abs(float(play.source_start_time) - float(play.original_start_time))
        + abs(float(play.source_end_time) - float(play.original_end_time))
        for play in corrected
        if play.original_start_time is not None and play.original_end_time is not None
    )
    first_review = min(
        [play.boundary_first_reviewed_at for play in corrected if play.boundary_first_reviewed_at]
        or [None]
    )
    correction_latency_seconds = (
        max(0.0, (first_review - game.segmented_at).total_seconds())
        if first_review and game.segmented_at else None
    )
    baseline = None if not comparison else {
        "detector_mode": comparison.detector_mode,
        "selected_method": comparison.selected_method,
        "play_first_accepted": bool(comparison.play_first_accepted),
        "legacy_count": comparison.legacy_count,
        "play_first_count": comparison.play_first_count,
        "count_delta": comparison.play_first_count - comparison.legacy_count,
        "matched_count": comparison.matched_count,
        "unmatched_legacy_count": comparison.unmatched_legacy_count,
        "unmatched_play_first_count": comparison.unmatched_play_first_count,
        "possible_duplicate_play_first_count": comparison.possible_duplicate_play_first_count,
        "mean_matched_iou": comparison.mean_matched_iou,
        "match_iou_threshold": comparison.match_iou_threshold,
        "detector_error": comparison.detector_error,
        "legacy_candidates": comparison.legacy_candidates or [],
        "play_first_candidates": comparison.play_first_candidates or [],
    }
    return {
        "game_id": game_id,
        "baseline_comparison": baseline,
        "human_corrected_timeline": {
            "current_play_count": len(plays),
            "corrected_play_count": len(corrected),
            "boundary_correction_count": sum(
                int(play.boundary_correction_count or 0) for play in plays
            ),
            "total_boundary_adjustment_seconds": round(boundary_delta_seconds, 2),
            "seconds_to_first_correction": correction_latency_seconds,
            "note": "Latency is elapsed time, not active official review time.",
        },
    }

@app.get("/api/games/{game_id}")
async def get_game(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    result = await db.execute(select(Game).where(Game.id == game_id))
    game = result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    data = _game_to_dict(game)
    key = _s3_key_from_url(game.s3_url)
    data["presigned_url"] = _presign(key, 3600)
    return data

@app.get("/api/games/{game_id}/plays")
async def get_game_plays(
    game_id: int,
    result: Optional[str] = Query(None),
    reviewed: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    filters = [Upload.game_id == game_id]
    if result:
        filters.append(Upload.result == result)
    if reviewed is not None:
        filters.append(Upload.reviewed_at.isnot(None) if reviewed else Upload.reviewed_at.is_(None))
    rows_result = await db.execute(
        select(Upload).where(*filters).order_by(Upload.play_number.asc(), Upload.start_time.asc())
    )
    rows = rows_result.scalars().all()
    upload_ids = [row.id for row in rows]
    timeline_by_upload = {}
    findings_by_play = {}
    if upload_ids:
        timeline_result = await db.execute(
            select(Play).where(Play.legacy_upload_id.in_(upload_ids))
        )
        timeline_by_upload = {
            play.legacy_upload_id: play for play in timeline_result.scalars().all()
        }
        timeline_ids = [play.id for play in timeline_by_upload.values()]
        if timeline_ids:
            findings_result = await db.execute(
                select(PlayFinding)
                .where(PlayFinding.play_id.in_(timeline_ids))
                .order_by(PlayFinding.id)
            )
            for finding in findings_result.scalars().all():
                findings_by_play.setdefault(finding.play_id, []).append(finding)
    return {
        "items": [
            _play_to_dict(
                row,
                timeline_by_upload.get(row.id),
                findings_by_play.get(timeline_by_upload[row.id].id, [])
                if row.id in timeline_by_upload else [],
            )
            for row in rows
        ],
        "total": len(rows),
    }


@app.get("/api/games/{game_id}/timeline")
async def get_game_timeline(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game_result = await db.execute(select(Game).where(Game.id == game_id))
    game = game_result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    plays_result = await db.execute(
        select(Play).where(Play.game_id == game_id).order_by(Play.play_number, Play.source_start_time)
    )
    plays = plays_result.scalars().all()
    play_ids = [play.id for play in plays]
    findings_by_play = {play_id: [] for play_id in play_ids}
    if play_ids:
        findings_result = await db.execute(
            select(PlayFinding).where(PlayFinding.play_id.in_(play_ids)).order_by(PlayFinding.id)
        )
        for finding in findings_result.scalars().all():
            findings_by_play[finding.play_id].append(_finding_to_dict(finding))
    upload_ids = [play.legacy_upload_id for play in plays if play.legacy_upload_id]
    reviewed_upload_ids = set()
    if upload_ids:
        reviewed_result = await db.execute(
            select(Upload.id).where(Upload.id.in_(upload_ids), Upload.reviewed_at.isnot(None))
        )
        reviewed_upload_ids = set(reviewed_result.scalars().all())
    jobs_result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.game_id == game_id)
        .order_by(ProcessingJob.id)
    )
    jobs = jobs_result.scalars().all()
    return {
        "items": [{
            "id": play.id,
            "legacy_upload_id": play.legacy_upload_id,
            "play_number": play.play_number,
            "source_start_time": play.source_start_time,
            "source_end_time": play.source_end_time,
            "snap_time": play.snap_time,
            "live_ball_end_time": play.live_ball_end_time,
            "clip_start_time": play.clip_start_time,
            "clip_end_time": play.clip_end_time,
            "clip_s3_url": play.clip_s3_url,
            "presigned_url": (
                _presign(_s3_key_from_url(play.clip_s3_url), 3600)
                if play.clip_s3_url else None
            ),
            "detection_method": play.detection_method,
            "detection_confidence": play.detection_confidence,
            "boundary_flags": play.boundary_flags or [],
            "boundary_review_status": play.boundary_review_status,
            "original_start_time": play.original_start_time,
            "original_end_time": play.original_end_time,
            "boundary_correction_count": play.boundary_correction_count or 0,
            "boundary_first_reviewed_at": play.boundary_first_reviewed_at,
            "boundary_last_reviewed_at": play.boundary_last_reviewed_at,
            "analysis_status": play.analysis_status,
            "analysis_version": play.analysis_version,
            "analysis_revision": play.analysis_revision or 0,
            "result": play.result,
            "prediction_label": play.prediction_label,
            "confidence": play.confidence,
            "notes": play.notes,
            "error_message": play.error_message,
            "findings": findings_by_play[play.id],
            "finding_count": len(findings_by_play[play.id]),
            "foul_families": sorted({
                finding["human_label"] or finding["foul_label"]
                for finding in findings_by_play[play.id]
                if finding["review_status"] != "dismissed"
            }),
            "human_review_state": (
                "reviewed" if (
                    findings_by_play[play.id]
                    and all(finding["review_status"] != "unreviewed" for finding in findings_by_play[play.id])
                ) or play.legacy_upload_id in reviewed_upload_ids
                else "partial" if any(
                    finding["review_status"] != "unreviewed" for finding in findings_by_play[play.id]
                )
                else "unreviewed"
            ),
        } for play in plays],
        "total": len(plays),
        "uncovered_intervals": find_uncovered_intervals(
            [{"start_time": play.source_start_time, "end_time": play.source_end_time} for play in plays],
            float(game.source_duration or max(
                [play.source_end_time for play in plays] or [0.0]
            )),
        ),
        "jobs": [{
            "id": job.id,
            "play_id": job.play_id,
            "stage": job.stage,
            "status": job.status,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "error_message": job.error_message,
            "available_at": job.available_at.isoformat() if job.available_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        } for job in jobs],
    }

@app.patch("/api/games/{game_id}/segments")
async def update_game_segments(
    game_id: int,
    payload: SegmentsPayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game_result = await db.execute(select(Game).where(Game.id == game_id))
    game = game_result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.status == "analyzing":
        raise HTTPException(status_code=409, detail="Cannot edit segments while analysis is running")

    clean_segments = sorted(payload.segments, key=lambda item: (item.start_time, item.end_time))
    existing_plays_result = await db.execute(select(Play.id).where(Play.game_id == game_id))
    existing_play_ids = list(existing_plays_result.scalars().all())
    if existing_play_ids:
        await db.execute(delete(PlayFinding).where(PlayFinding.play_id.in_(existing_play_ids)))
    await db.execute(delete(Play).where(Play.game_id == game_id))
    await db.execute(delete(Upload).where(Upload.game_id == game_id))
    for index, segment in enumerate(clean_segments, start=1):
        if segment.end_time <= segment.start_time:
            raise HTTPException(status_code=400, detail="Each segment end_time must be after start_time")
        upload = Upload(
            game_id=game.id,
            play_number=segment.play_number or index,
            start_time=segment.start_time,
            end_time=segment.end_time,
            segment_confidence=1.0,
            s3_url=game.s3_url,
            foul_type="",
            notes=segment.notes or "",
            ruleset=game.ruleset,
            timestamp=_now_utc(),
            status="segmented",
        )
        db.add(upload)
        await db.flush()
        db.add(Play(
            game_id=game.id,
            legacy_upload_id=upload.id,
            play_number=upload.play_number,
            source_start_time=upload.start_time,
            source_end_time=upload.end_time,
            original_start_time=upload.start_time,
            original_end_time=upload.end_time,
            snap_time=None,
            live_ball_end_time=None,
            clip_start_time=upload.start_time,
            clip_end_time=upload.end_time,
            detection_method="human_edited",
            detection_confidence=1.0,
            boundary_flags=[],
            boundary_review_status="approved",
            analysis_status="segmented",
            notes=upload.notes,
            created_at=_now_utc(),
            updated_at=_now_utc(),
        ))
    game.segment_count = len(clean_segments)
    game.analyzed_count = 0
    game.status = "segments_ready"
    game.progress = 0.25
    game.error_message = None
    await db.commit()
    return {"ok": True, "segment_count": len(clean_segments)}


@app.patch("/api/play-findings/{finding_id}/review")
async def review_play_finding(
    finding_id: int,
    payload: FindingReviewPayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    finding = await db.get(PlayFinding, finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")
    status = payload.review_status.strip().lower()
    if status not in {"agreed", "corrected", "dismissed", "unreviewed"}:
        raise HTTPException(status_code=400, detail="Unsupported finding review status")
    if status == "unreviewed":
        finding.review_status = status
        finding.human_disposition = None
        finding.human_label = None
        finding.human_committing_team = None
        finding.human_player_number = None
        finding.human_notes = None
        finding.reviewed_at = None
    elif status == "dismissed":
        finding.review_status = status
        finding.human_disposition = None
        finding.human_label = None
        finding.human_committing_team = None
        finding.human_player_number = None
        finding.human_notes = payload.human_notes
        finding.reviewed_at = _now_utc()
    else:
        disposition = (
            finding.disposition if status == "agreed" else
            str(payload.human_disposition or "").strip().upper()
        )
        label = finding.foul_label if status == "agreed" else str(payload.human_label or "").strip()
        if disposition not in {"FOUL", "UNCERTAIN"}:
            raise HTTPException(status_code=400, detail="Correction disposition must be FOUL or UNCERTAIN")
        if disposition == "FOUL" and not label:
            raise HTTPException(status_code=400, detail="Corrected foul label is required")
        team = (
            finding.committing_team if status == "agreed" else
            str(payload.human_committing_team or "unknown").strip().lower()
        )
        if team not in {"offense", "defense", "kicking", "receiving", "unknown"}:
            raise HTTPException(status_code=400, detail="Unsupported committing team")
        finding.review_status = status
        finding.human_disposition = disposition
        finding.human_label = label or "Uncertain"
        finding.human_committing_team = team
        finding.human_player_number = (
            finding.player_number if status == "agreed" else payload.human_player_number
        )
        finding.human_notes = payload.human_notes
        finding.reviewed_at = _now_utc()
    finding.updated_at = _now_utc()
    await db.commit()
    await db.refresh(finding)
    return {"ok": True, "finding": _finding_to_dict(finding)}


@app.patch("/api/games/{game_id}/plays/{play_id}/boundary")
async def update_play_boundary(
    game_id: int,
    play_id: int,
    payload: PlayBoundaryPayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    play = await db.get(Play, play_id)
    if not game or not play or play.game_id != game_id:
        raise HTTPException(status_code=404, detail="Game play not found")
    _assert_game_editable(game)
    if payload.start_time < 0 or payload.end_time <= payload.start_time:
        raise HTTPException(status_code=400, detail="End time must be after a non-negative start time")
    if game.source_duration and payload.end_time > game.source_duration:
        raise HTTPException(status_code=400, detail="Boundary exceeds the source video duration")
    reviewed_at = _now_utc()
    if play.original_start_time is None:
        play.original_start_time = play.source_start_time
    if play.original_end_time is None:
        play.original_end_time = play.source_end_time
    play.boundary_correction_count = int(play.boundary_correction_count or 0) + 1
    play.boundary_first_reviewed_at = play.boundary_first_reviewed_at or reviewed_at
    play.boundary_last_reviewed_at = reviewed_at
    play.source_start_time = payload.start_time
    play.source_end_time = payload.end_time
    play.clip_start_time = payload.start_time
    play.clip_end_time = payload.end_time
    play.notes = payload.notes or ""
    play.detection_method = "human_edited"
    play.detection_confidence = 1.0
    play.boundary_flags = []
    play.boundary_review_status = "approved"
    upload = await db.get(Upload, play.legacy_upload_id)
    if upload:
        upload.start_time = payload.start_time
        upload.end_time = payload.end_time
        upload.segment_confidence = 1.0
        upload.notes = play.notes
    await _reset_play_for_reanalysis(db, play, clear_clip=True)
    await _enqueue_selective_analysis(db, game)
    await db.commit()
    return {"ok": True, "play_id": play.id, "analysis_revision": play.analysis_revision}


@app.post("/api/games/{game_id}/plays/{play_id}/split")
async def split_game_play(
    game_id: int,
    play_id: int,
    payload: PlaySplitPayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    play = await db.get(Play, play_id)
    if not game or not play or play.game_id != game_id:
        raise HTTPException(status_code=404, detail="Game play not found")
    _assert_game_editable(game)
    if not play.source_start_time + 1.0 <= payload.split_time <= play.source_end_time - 1.0:
        raise HTTPException(status_code=400, detail="Split must leave at least one second on each side")
    later_result = await db.execute(
        select(Play).where(Play.game_id == game_id, Play.play_number > play.play_number)
        .order_by(Play.play_number.desc())
    )
    for later in later_result.scalars().all():
        later.play_number += 1
        later_upload = await db.get(Upload, later.legacy_upload_id)
        if later_upload:
            later_upload.play_number = later.play_number
    original_end = play.source_end_time
    reviewed_at = _now_utc()
    if play.original_start_time is None:
        play.original_start_time = play.source_start_time
    if play.original_end_time is None:
        play.original_end_time = play.source_end_time
    play.boundary_correction_count = int(play.boundary_correction_count or 0) + 1
    play.boundary_first_reviewed_at = play.boundary_first_reviewed_at or reviewed_at
    play.boundary_last_reviewed_at = reviewed_at
    original_upload = await db.get(Upload, play.legacy_upload_id)
    play.source_end_time = payload.split_time
    play.clip_end_time = payload.split_time
    if original_upload:
        original_upload.end_time = payload.split_time
    await _reset_play_for_reanalysis(db, play, clear_clip=True)
    new_upload = Upload(
        game_id=game.id,
        play_number=play.play_number + 1,
        start_time=payload.split_time,
        end_time=original_end,
        segment_confidence=1.0,
        s3_url=game.s3_url,
        foul_type="",
        notes=play.notes or "",
        ruleset=game.ruleset,
        timestamp=_now_utc(),
        status="segmented",
    )
    db.add(new_upload)
    await db.flush()
    new_play = Play(
        game_id=game.id,
        legacy_upload_id=new_upload.id,
        play_number=new_upload.play_number,
        source_start_time=payload.split_time,
        source_end_time=original_end,
        original_start_time=payload.split_time,
        original_end_time=original_end,
        boundary_correction_count=1,
        boundary_first_reviewed_at=reviewed_at,
        boundary_last_reviewed_at=reviewed_at,
        clip_start_time=payload.split_time,
        clip_end_time=original_end,
        detection_method="human_split",
        detection_confidence=1.0,
        boundary_flags=[],
        boundary_review_status="approved",
        analysis_status="queued_for_reanalysis",
        analysis_revision=1,
        notes=new_upload.notes,
        created_at=_now_utc(),
        updated_at=_now_utc(),
    )
    db.add(new_play)
    game.segment_count = int(game.segment_count or 0) + 1
    await db.flush()
    await _enqueue_selective_analysis(db, game)
    await db.commit()
    return {"ok": True, "play_ids": [play.id, new_play.id]}


@app.post("/api/games/{game_id}/plays/{play_id}/merge")
async def merge_game_plays(
    game_id: int,
    play_id: int,
    payload: PlayMergePayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    play = await db.get(Play, play_id)
    next_play = await db.get(Play, payload.next_play_id)
    if not game or not play or not next_play or play.game_id != game_id or next_play.game_id != game_id:
        raise HTTPException(status_code=404, detail="Game plays not found")
    _assert_game_editable(game)
    if next_play.play_number != play.play_number + 1:
        raise HTTPException(status_code=400, detail="Only consecutive plays can be merged")
    reviewed_at = _now_utc()
    if play.original_start_time is None:
        play.original_start_time = play.source_start_time
    if play.original_end_time is None:
        play.original_end_time = play.source_end_time
    play.boundary_correction_count = int(play.boundary_correction_count or 0) + 1
    play.boundary_first_reviewed_at = play.boundary_first_reviewed_at or reviewed_at
    play.boundary_last_reviewed_at = reviewed_at
    play.source_end_time = max(play.source_end_time, next_play.source_end_time)
    play.clip_end_time = play.source_end_time
    play.detection_method = "human_merge"
    play.detection_confidence = 1.0
    play.boundary_flags = []
    play.boundary_review_status = "approved"
    upload = await db.get(Upload, play.legacy_upload_id)
    removed_upload = await db.get(Upload, next_play.legacy_upload_id)
    if upload:
        upload.end_time = play.source_end_time
        upload.segment_confidence = 1.0
    await _reset_play_for_reanalysis(db, play, clear_clip=True)
    await db.execute(delete(PlayFinding).where(PlayFinding.play_id == next_play.id))
    await db.delete(next_play)
    if removed_upload:
        await db.delete(removed_upload)
    later_result = await db.execute(
        select(Play).where(Play.game_id == game_id, Play.play_number > play.play_number + 1)
        .order_by(Play.play_number)
    )
    for later in later_result.scalars().all():
        later.play_number -= 1
        later_upload = await db.get(Upload, later.legacy_upload_id)
        if later_upload:
            later_upload.play_number = later.play_number
    game.segment_count = max(0, int(game.segment_count or 0) - 1)
    await db.flush()
    await _enqueue_selective_analysis(db, game)
    await db.commit()
    return {"ok": True, "play_id": play.id}


@app.post("/api/games/{game_id}/plays/{play_id}/reanalyze")
async def reanalyze_game_play(
    game_id: int,
    play_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    play = await db.get(Play, play_id)
    if not game or not play or play.game_id != game_id:
        raise HTTPException(status_code=404, detail="Game play not found")
    _assert_game_editable(game)
    await _reset_play_for_reanalysis(db, play, clear_clip=False)
    await _enqueue_selective_analysis(db, game)
    await db.commit()
    return {"ok": True, "play_id": play.id, "analysis_revision": play.analysis_revision}


@app.delete("/api/games/{game_id}/plays/{play_id}")
async def remove_game_play(
    game_id: int,
    play_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game = await db.get(Game, game_id)
    play = await db.get(Play, play_id)
    if not game or not play or play.game_id != game_id:
        raise HTTPException(status_code=404, detail="Game play not found")
    _assert_game_editable(game)
    removed_number = play.play_number
    upload = await db.get(Upload, play.legacy_upload_id)
    await db.execute(delete(PlayFinding).where(PlayFinding.play_id == play.id))
    await db.delete(play)
    if upload:
        await db.delete(upload)
    later_result = await db.execute(
        select(Play).where(Play.game_id == game_id, Play.play_number > removed_number)
        .order_by(Play.play_number)
    )
    for later in later_result.scalars().all():
        later.play_number -= 1
        later_upload = await db.get(Upload, later.legacy_upload_id)
        if later_upload:
            later_upload.play_number = later.play_number
    game.segment_count = max(0, int(game.segment_count or 0) - 1)
    game.analyzed_count = min(int(game.analyzed_count or 0), game.segment_count)
    await db.commit()
    return {"ok": True, "removed_play_id": play_id}


@app.post("/api/plays/{play_id}/findings")
async def add_manual_play_finding(
    play_id: int,
    payload: ManualFindingPayload,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    play = await db.get(Play, play_id)
    if not play:
        raise HTTPException(status_code=404, detail="Play not found")
    disposition = payload.disposition.strip().upper()
    if disposition not in {"FOUL", "UNCERTAIN"}:
        raise HTTPException(status_code=400, detail="Finding disposition must be FOUL or UNCERTAIN")
    label = payload.foul_label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Foul label is required")
    team = str(payload.committing_team or "unknown").strip().lower()
    if team not in {"offense", "defense", "kicking", "receiving", "unknown"}:
        raise HTTPException(status_code=400, detail="Unsupported committing team")
    event_time = payload.event_time
    duration = max(0.0, play.source_end_time - play.source_start_time)
    if event_time is not None and not 0 <= event_time <= duration:
        raise HTTPException(status_code=400, detail="Event time must fall within the play clip")
    finding = PlayFinding(
        play_id=play.id,
        disposition=disposition,
        foul_label=label,
        committing_team=team,
        player_number=(payload.player_number or "").strip() or None,
        event_time=event_time,
        confidence=max(0.0, min(1.0, payload.confidence)),
        evidence_summary=payload.evidence_summary or "Added by an official reviewer.",
        reasoning=payload.reasoning or "Manual official finding.",
        rule_citations=payload.rule_citations,
        uncertainty_notes=payload.uncertainty_notes or "",
        review_status="agreed",
        human_disposition=disposition,
        human_label=label,
        human_committing_team=team,
        human_player_number=(payload.player_number or "").strip() or None,
        reviewed_at=_now_utc(),
        created_by="human",
        created_at=_now_utc(),
        updated_at=_now_utc(),
    )
    db.add(finding)
    if disposition == "FOUL" or play.result != "FOUL":
        play.result = disposition
        play.prediction_label = label
        play.confidence = finding.confidence
        play.updated_at = _now_utc()
        upload = await db.get(Upload, play.legacy_upload_id)
        if upload:
            upload.result = play.result
            upload.prediction_label = play.prediction_label
            upload.confidence = play.confidence
            upload.human_label = label
            upload.reviewed_at = _now_utc()
    await db.commit()
    await db.refresh(finding)
    return {"ok": True, "finding": _finding_to_dict(finding)}

@app.post("/api/games/{game_id}/analyze")
async def analyze_game(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    game_result = await db.execute(select(Game).where(Game.id == game_id))
    game = game_result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.status in {"queued_for_analysis", "analyzing"}:
        return {"ok": True, "id": game_id, "status": game.status}
    play_count_result = await db.execute(
        select(func.count()).select_from(Upload).where(Upload.game_id == game_id)
    )
    if play_count_result.scalar_one() == 0:
        raise HTTPException(status_code=409, detail="No play segments are available to analyze")
    signature_result = await db.execute(
        select(Play.id, Play.source_start_time, Play.source_end_time)
        .where(Play.game_id == game_id)
        .order_by(Play.play_number, Play.id)
    )
    segment_signature = hashlib.sha256(json.dumps([
        [row.id, row.source_start_time, row.source_end_time]
        for row in signature_result.all()
    ]).encode("utf-8")).hexdigest()[:16]
    game.status = "queued_for_analysis"
    game.progress = max(float(game.progress or 0), 0.25)
    game.error_message = None
    await enqueue_job(
        db,
        job_key=f"game:{game_id}:extract_clips:multi-v1:{segment_signature}",
        game_id=game_id,
        stage="extract_clips",
    )
    await db.commit()
    return {"ok": True, "id": game_id, "status": "queued_for_analysis"}

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str = Form(""),
    ruleset: str = Form("high_school"),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    print("📥 Upload request received")

    try:
        ruleset = _normalize_ruleset(ruleset)
        data = await file.read()
        print("✅ File read complete")

        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "clip.mp4")
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        key = f"videos/{stamp}_{clean_name}"
        print("🧹 Cleaned filename:", key)

        # S3 Upload
        try:
            s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType="video/mp4")
            print("✅ S3 upload complete")
        except Exception as e:
            print("❌ S3 upload failed:", e)
            raise HTTPException(status_code=500, detail="S3 upload failed")

        s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
        print("🌐 S3 URL:", s3_url)

        # DB Write (async)
        try:
            rec = Upload(
                s3_url=s3_url,
                foul_type=foul_type,
                notes=notes,
                ruleset=ruleset,
                timestamp=datetime.utcnow(),
                status="queued",
            )
            db.add(rec)
            await db.commit()
            await db.refresh(rec)
            print("✅ DB write complete with ID:", rec.id)

            asyncio.create_task(_process_upload_bg(rec.id, s3_url, foul_type, notes, ruleset))
            response = UploadResponse(id=rec.id, s3_url=s3_url)
            print("✅ UploadResponse ready:", response.dict())
            return response
        except Exception as db_error:
            print("❌ DB write failed:", db_error)
            raise HTTPException(status_code=500, detail="DB write failed")

    except Exception as e:
        print("❌ Unexpected error in /upload:", e)
        raise HTTPException(status_code=500, detail="Unexpected server error")

app.add_api_route("/api/upload", upload_video, methods=["POST"])

@app.get("/api/plays")
async def list_recent_plays(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    reviewed: Optional[bool] = Query(None),
    q: Optional[str] = Query(None, min_length=1),
    ruleset: str = Query("high_school"),
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    try:
        ruleset = _normalize_ruleset(ruleset)
        filters = []
        filters.append(Upload.ruleset == ruleset)
        if status:
            filters.append(Upload.status == status)
        if reviewed is not None:
            filters.append(Upload.reviewed_at.isnot(None) if reviewed else Upload.reviewed_at.is_(None))
        if q:
            pattern = f"%{q.strip()}%"
            filters.append(or_(
                Upload.foul_type.ilike(pattern),
                Upload.notes.ilike(pattern),
                Upload.prediction_label.ilike(pattern),
                Upload.human_label.ilike(pattern),
            ))

        result = await db.execute(
            select(Upload)
            .where(*filters)
            .order_by(Upload.id.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = result.scalars().all()
        total_result = await db.execute(
            select(func.count())
            .select_from(Upload)
            .where(*filters)
        )
        total = total_result.scalar_one()

        items = []
        for r in rows:
            key = _s3_key_from_url(r.s3_url)
            presigned = _presign(key, 3600)
            official_agrees = None
            if (
                getattr(r, "reviewed_at", None)
                and getattr(r, "human_label", None)
                and r.prediction_label
            ):
                official_agrees = r.human_label == r.prediction_label
            items.append({
                "id": r.id,
                "game_id": getattr(r, "game_id", None),
                "play_number": getattr(r, "play_number", None),
                "start_time": getattr(r, "start_time", None),
                "end_time": getattr(r, "end_time", None),
                "segment_confidence": getattr(r, "segment_confidence", None),
                "foul_type": r.foul_type,
                "notes": r.notes,
                "ruleset": getattr(r, "ruleset", "high_school"),
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "status": r.status,
                "result": getattr(r, "result", None),
                "prediction_label": r.prediction_label,
                "confidence": r.confidence,
                "processed_at": r.processed_at.isoformat() if r.processed_at else None,
                "error_message": r.error_message,
                "explanation": getattr(r, "explanation", None),
                "retrieved_rules": getattr(r, "retrieved_rules", None),
                "human_label": getattr(r, "human_label", None),
                "human_notes": getattr(r, "human_notes", None),
                "reviewed_at": r.reviewed_at.isoformat() if getattr(r, "reviewed_at", None) else None,
                "official_agrees": official_agrees,
                "s3_url": r.s3_url,
                "clip_s3_url": getattr(r, "clip_s3_url", None),
                "presigned_url": presigned,
            })
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/retry/{upload_id}")
async def retry_upload(
    upload_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    """Mark upload queued and re-run async background worker."""
    # Fetch the row using async session
    result = await db.execute(select(Upload).where(Upload.id == upload_id))
    row = result.scalar_one_or_none()

    if not row:
        return {"ok": False, "error": "Not found"}

    # Update status and clear error/processed flags
    row.status = "queued"
    row.error_message = None
    row.processed_at = None
    await db.commit()

    # Launch async background worker (non-blocking)
    asyncio.create_task(_process_upload_bg(
        row.id,
        row.s3_url,
        row.foul_type,
        row.notes or "",
        getattr(row, "ruleset", "high_school") or "high_school",
    ))

    return {"ok": True, "id": upload_id}

@app.patch("/api/plays/{upload_id}/review")
def set_human_review(
    upload_id: int,
    payload: ReviewPayload,
    _auth: None = Depends(require_api_key),
):
    db: Session = SessionLocal()
    try:
        row = db.query(Upload).get(upload_id)
        if not row:
            return {"ok": False, "error": "Not found"}

        if payload.reviewed:
            row.human_label = payload.human_label
            row.human_notes = payload.human_notes
            row.reviewed_at = datetime.now(timezone.utc)
        else:
            row.human_label = None
            row.human_notes = None
            row.reviewed_at = None

        db.commit()
        return {"ok": True, "id": upload_id}
    finally:
        db.close()


@app.post("/api/plays/{upload_id}/feedback")
def submit_official_feedback(
    upload_id: int,
    payload: OfficialFeedbackIn,
    _auth: None = Depends(require_api_key),
):
    """
    Save a low-friction official feedback signal without replacing the AI prediction.
    Agreement is stored as human_label == prediction_label; disagreement stores the
    official correction in human_label.
    """
    db: Session = SessionLocal()
    try:
        row = db.query(Upload).get(upload_id)
        if not row:
            return {"ok": False, "error": "Not found"}

        notes = payload.human_notes.strip() if payload.human_notes else None

        if payload.official_agrees:
            if not row.prediction_label:
                return {"ok": False, "error": "No prediction available to agree with"}
            row.human_label = row.prediction_label
        else:
            label = payload.human_label.strip() if payload.human_label else ""
            if not label:
                return {"ok": False, "error": "Correct label is required when disagreeing"}
            row.human_label = label

        row.human_notes = notes
        row.reviewed_at = _now_utc()
        db.commit()

        return {
            "ok": True,
            "id": upload_id,
            "official_agrees": payload.official_agrees,
            "human_label": row.human_label,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        }
    except Exception as e:
        return {"ok": False, "error": _safe_err_text(e)}
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Rules: quick search (already handy for sanity checks)
# ------------------------------------------------------------------------------
@app.get("/api/rules/search")
async def search_rules(
    q: str = Query(..., min_length=2),
    k: int = Query(3, ge=1, le=10),
    ruleset: str = Query("high_school"),
    _auth: None = Depends(require_api_key),
):
    """
    Simple retrieval endpoint to sanity-check embeddings & pgvector search.
    """
    try:
        ruleset = _normalize_ruleset(ruleset)
        emb = await _embed_text(q)  # 3072-D
        qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        sql = sqltext("""
            SELECT id, title, section, body,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM rules
            WHERE ruleset = :ruleset
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :k
        """)
        with engine.begin() as conn:
            rows = conn.execute(sql, {"qvec": qvec_literal, "k": k, "ruleset": ruleset}).mappings().all()
        return [
            {
                "id": int(r["id"]),
                "title": r["title"],
                "section": r["section"],
                "score": float(r["score"]) if r["score"] is not None else None,
                "body": r["body"],
            }
            for r in rows
        ]
    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------------------------------------
# 1.e Human review route
# ------------------------------------------------------------------------------
class ReviewIn(BaseModel):
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    # Optional: allow overriding prediction_label/confidence if you want
    override_prediction: Optional[bool] = False

@app.post("/api/review/{upload_id}")
def submit_review(
    upload_id: int,
    payload: ReviewIn = Body(...),
    _auth: None = Depends(require_api_key),
):
    """
    Save human review info. If override_prediction=True and human_label is provided,
    copy human_label into prediction_label.
    """
    db: Session = SessionLocal()
    try:
        row = db.query(Upload).get(upload_id)
        if not row:
            return {"ok": False, "error": "Not found"}

        row.human_label = payload.human_label
        row.human_notes = payload.human_notes
        row.reviewed_at = _now_utc()

        if payload.override_prediction and payload.human_label:
            row.prediction_label = payload.human_label

        db.commit()
        return {"ok": True, "id": upload_id}
    except Exception as e:
        return {"ok": False, "error": _safe_err_text(e)}
    finally:
        db.close()

# ------------------------------------------------------------------------------
# (Optional) list rules for populating UI dropdowns
# ------------------------------------------------------------------------------
# Removed duplicate /api/rules/list endpoint
