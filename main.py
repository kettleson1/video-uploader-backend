# main.py
import os
import logging
import random
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from models import Base, Upload

# -----------------------------------------------------------------------------
# Setup & config
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-uploader")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")

if not BUCKET_NAME or not DATABASE_URL:
    raise RuntimeError("Missing AWS_S3_BUCKET or DATABASE_URL in environment.")

# AWS S3 client
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Database
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if needed
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="Football Play Uploader")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _s3_key_from_url(url: str) -> str:
    """Extract the S3 object key from a bucket URL."""
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path

def _new_session() -> Session:
    return SessionLocal()

# -----------------------------------------------------------------------------
# Background processing (stubbed prediction)
# -----------------------------------------------------------------------------
def _fake_model_predict(local_path: str, foul_hint: str | None = None) -> tuple[str, float]:
    """
    Stub: pretend to run a model and return (label, confidence).
    Later you'll replace this with real inference (RAG / fine-tuned model).
    """
    labels = ["Targeting", "Holding", "Pass Interference", "False Start", "Offside", "No Foul"]
    if foul_hint and foul_hint in labels:
        weights = [0.32 if l == foul_hint else 0.68 / (len(labels) - 1) for l in labels]
    else:
        weights = [1 / len(labels)] * len(labels)
    label = random.choices(labels, weights=weights, k=1)[0]
    confidence = round(random.uniform(0.55, 0.95), 2)
    return label, confidence

def _download_s3_to_tmp(bucket: str, key: str) -> str:
    """
    Download the S3 object to a temp file and return local path.
    If you don't need the file (e.g., model uses the S3 URL), skip this.
    """
    import tempfile
    fd, path = tempfile.mkstemp(prefix="play_", suffix=os.path.splitext(key)[-1])
    os.close(fd)
    with open(path, "wb") as f:
        s3_client.download_fileobj(bucket, key, f)
    return path

def process_upload(upload_id: int):
    """
    Background job:
      - mark 'processing'
      - (optional) download video from S3
      - run stub model
      - write prediction + confidence + processed_at
    On failure: mark 'error' and record error_message.
    """
    db = _new_session()
    try:
        rec = db.query(Upload).filter(Upload.id == upload_id).one_or_none()
        if not rec:
            return

        # Move to processing
        rec.status = "processing"
        rec.error_message = None
        db.commit()

        key = _s3_key_from_url(rec.s3_url)

        # OPTIONAL: download for local model usage
        try:
            local_path = _download_s3_to_tmp(BUCKET_NAME, key)
        except ClientError as e:
            local_path = ""
            logger.warning("S3 download failed for %s: %s", key, e)

        # Run fake model
        label, conf = _fake_model_predict(local_path, foul_hint=rec.foul_type)

        # Persist results
        rec.prediction_label = label
        rec.confidence = conf
        rec.processed_at = datetime.now(timezone.utc)
        rec.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        try:
            rec = db.query(Upload).filter(Upload.id == upload_id).one_or_none()
            if rec:
                rec.status = "error"
                rec.error_message = str(e)[:2000]
                rec.processed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            db.rollback()
        logger.exception("Processing failed for upload_id=%s", upload_id)
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------------------------------------------------------
# Upload endpoint
# -----------------------------------------------------------------------------
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str | None = Form(None),
    background_tasks: BackgroundTasks | None = None,
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Unique S3 key
    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = os.path.basename(file.filename)
    s3_key = f"videos/{timestamp_str}_{safe_name}"

    # Upload to S3
    try:
        file.file.seek(0)
        extra_args = {"ContentType": file.content_type} if file.content_type else {}
        s3_client.upload_fileobj(
            Fileobj=file.file,
            Bucket=BUCKET_NAME,
            Key=s3_key,
            ExtraArgs=extra_args or None,
        )
    except Exception as e:
        logger.exception("S3 upload failed")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

    # Save initial record
    db = _new_session()
    try:
        rec = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes,
            status="queued",
            timestamp=datetime.utcnow(),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
    except Exception as e:
        db.rollback()
        logger.exception("DB save failed")
        raise HTTPException(status_code=500, detail=f"DB save failed: {e}")
    finally:
        db.close()

    # Queue background processing
    if background_tasks is not None:
        background_tasks.add_task(process_upload, rec.id)

    return {
        "id": rec.id,
        "status": rec.status,
        "s3_url": rec.s3_url,
        "foul_type": rec.foul_type,
        "notes": rec.notes,
        "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
    }

# -----------------------------------------------------------------------------
# List recent plays with presigned URLs
# -----------------------------------------------------------------------------
@app.get("/api/plays")
def list_recent_plays(limit: int = Query(20, ge=1, le=200)) -> List[dict]:
    db = _new_session()
    try:
        rows = (
            db.query(Upload)
              .order_by(Upload.id.desc())
              .limit(limit)
              .all()
        )
        out: List[dict] = []
        for r in rows:
            key = _s3_key_from_url(r.s3_url)
            try:
                presigned = s3_client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": BUCKET_NAME, "Key": key},
                    ExpiresIn=3600,  # 1 hour
                )
            except Exception as e:
                logger.warning("Failed to presign %s: %s", key, e)
                presigned = None

            out.append({
                "id": r.id,
                "foul_type": r.foul_type,
                "notes": r.notes,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "status": r.status,
                "prediction_label": r.prediction_label,
                "confidence": r.confidence,
                "processed_at": r.processed_at.isoformat() if r.processed_at else None,
                "error_message": r.error_message,
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Optional: manual retry
# -----------------------------------------------------------------------------
@app.post("/api/retry/{upload_id}")
def retry(upload_id: int):
    db = _new_session()
    try:
        rec = db.query(Upload).filter(Upload.id == upload_id).one_or_none()
        if not rec:
            raise HTTPException(status_code=404, detail="Not found")
        rec.status = "queued"
        rec.error_message = None
        db.commit()
    finally:
        db.close()
    # You can make this asynchronous by using BackgroundTasks too
    process_upload(upload_id)
    return {"ok": True}