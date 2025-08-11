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

from models import Base, Upload

# -------------------------------------------------------
# Env / logging
# -------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-uploader")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")

if not BUCKET_NAME or not DATABASE_URL:
    raise RuntimeError("Missing AWS_S3_BUCKET or DATABASE_URL in environment.")

# -------------------------------------------------------
# AWS + DB
# -------------------------------------------------------
s3_client = boto3.client("s3", region_name=AWS_REGION)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if missing (safe if they already exist)
Base.metadata.create_all(bind=engine)

# -------------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------------
app = FastAPI(title="Football Play Uploader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Utils
# -------------------------------------------------------
def _s3_key_from_url(url: str) -> str:
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path

def _new_session() -> Session:
    return SessionLocal()

# -------------------------------------------------------
# Health
# -------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -------------------------------------------------------
# Upload
# -------------------------------------------------------
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str | None = Form(None),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # S3 key
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

    # Save to DB (set status='queued', add a created timestamp)
    db = _new_session()
    try:
        rec = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes,
            status="queued",                     # NEW: initial status
            timestamp=datetime.utcnow(),         # since table has no server default
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

    return {
        "id": rec.id,
        "status": rec.status,
        "s3_url": rec.s3_url,
        "foul_type": rec.foul_type,
        "notes": rec.notes,
        "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
    }

# -------------------------------------------------------
# List plays with presigned URLs
# -------------------------------------------------------
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
                "status": r.status,                        # NEW
                "prediction_label": r.prediction_label,    # NEW
                "confidence": r.confidence,                # NEW
                "processed_at": r.processed_at.isoformat() if r.processed_at else None,  # NEW
                "error_message": r.error_message,          # NEW
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
    finally:
        db.close()