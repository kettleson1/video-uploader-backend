import os
import subprocess
from datetime import datetime
from typing import List

import boto3
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from models import Base, Upload

from openai import OpenAI

# --- Load env ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- DB setup ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(bind=engine)

# --- AWS S3 client ---
s3_client = boto3.client("s3", region_name=AWS_REGION)

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers
# -------------------------

def _embed_text(text: str) -> List[float]:
    """Embed text with OpenAI ada-002."""
    resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return resp.data[0].embedding

def _s3_key_from_url(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path

# ✅ Updated retrieval function
def _retrieve_rules(summary: str, top_k: int = 3):
    """
    Embed the summary and retrieve the closest rule chunks via pgvector.
    Uses cosine distance (<=>). We pass the embedding as a vector literal
    and cast with ::vector to avoid 'operator does not exist' errors.
    """
    emb = _embed_text(summary)  # length = 1536

    qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    sql = text("""
        SELECT
            id,
            title,
            section,
            body,
            1 - (embedding <=> :qvec::vector) AS score
        FROM rules
        ORDER BY embedding <=> :qvec::vector
        LIMIT :k
    """)

    with engine.begin() as conn:
        rows = conn.execute(sql, {"qvec": qvec_literal, "k": top_k}).mappings().all()

    return [
        {
            "id": r["id"],
            "title": r["title"],
            "section": r["section"],
            "body": r["body"],
            "score": float(r["score"]) if r["score"] is not None else None,
        }
        for r in rows
    ]

# -------------------------
# Routes
# -------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str = Form(None),
):
    key = f"videos/{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}_{file.filename}"
    s3_client.upload_fileobj(file.file, AWS_S3_BUCKET, key)
    s3_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    db: Session = SessionLocal()
    upload = Upload(s3_url=s3_url, foul_type=foul_type, notes=notes, status="queued")
    db.add(upload)
    db.commit()
    db.refresh(upload)
    db.close()

    background_tasks.add_task(_process_upload, upload.id)
    return {"id": upload.id, "s3_url": s3_url}

@app.get("/api/plays")
def list_recent_plays(limit: int = 20):
    db: Session = SessionLocal()
    rows = (
        db.query(Upload)
        .order_by(Upload.id.desc())
        .limit(limit)
        .all()
    )
    db.close()

    out = []
    for r in rows:
        key = _s3_key_from_url(r.s3_url)
        presigned = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": AWS_S3_BUCKET, "Key": key}, ExpiresIn=3600
        )
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

@app.post("/api/retry/{upload_id}")
def retry_upload(upload_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(_process_upload, upload_id)
    return {"ok": True, "id": upload_id}

# -------------------------
# Background processing stub (RAG pipeline hook goes here)
# -------------------------

def _process_upload(upload_id: int):
    db: Session = SessionLocal()
    upload = db.query(Upload).get(upload_id)
    if not upload:
        db.close()
        return

    try:
        # 1) Summarize video (stubbed)
        summary = "A play involving potential foul"
        # 2) Retrieve rules
        rules = _retrieve_rules(summary)
        # 3) Fake predictor
        upload.status = "done"
        upload.prediction_label = rules[0]["title"] if rules else "Unknown"
        upload.confidence = 0.70
        upload.explanation = "Stub predictor. Replace with RAG logic."
        upload.processed_at = datetime.utcnow()

    except Exception as e:
        upload.status = "error"
        upload.error_message = str(e)
    finally:
        db.add(upload)
        db.commit()
        db.close()