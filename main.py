# main.py
import os
import logging
from datetime import datetime
from typing import List
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from models import Base, Upload  # <-- your SQLAlchemy model

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

# S3 client
s3_client = boto3.client("s3", region_name=AWS_REGION)

# DB engine / session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if not present
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="Football Play Uploader")

# CORS (dev: allow localhost 3000 & 3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _s3_key_from_url(url: str) -> str:
    """
    Converts https://<bucket>.s3.<region>.amazonaws.com/<key>
    to the S3 object key.
    """
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path

def _new_session() -> Session:
    return SessionLocal()

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
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Generate a unique S3 key
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = os.path.basename(file.filename)
    s3_key = f"videos/{timestamp}_{safe_name}"

    # Upload to S3
    try:
        # reset pointer just in case
        file.file.seek(0)
        extra_args = {}
        if file.content_type:
            extra_args["ContentType"] = file.content_type

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

    # Save metadata to DB
    db = _new_session()
    try:
        rec = Upload(s3_url=s3_url, foul_type=foul_type, notes=notes)
        db.add(rec)
        db.commit()
        db.refresh(rec)
    except Exception as e:
        db.rollback()
        logger.exception("DB save failed")
        # best-effort: do not delete S3 here; just surface error
        raise HTTPException(status_code=500, detail=f"DB save failed: {e}")
    finally:
        db.close()

    return {
        "status": "uploaded",
        "id": rec.id,
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
    """
    Return recent uploads with a short‑lived presigned URL for playback.
    """
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
                "s3_url": r.s3_url,         # original URL (may not be public)
                "presigned_url": presigned, # temporary playable link
            })
        return out
    finally:
        db.close()