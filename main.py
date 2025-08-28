# main.py
from __future__ import annotations

import os
import io
import re
import base64
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

import boto3
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from openai import OpenAI

from models import Base, Upload  # Upload table with 'explanation' column etc.

# ---------- Env / Clients ----------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
if not BUCKET_NAME:
    raise RuntimeError("AWS_S3_BUCKET is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

s3_client = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI(title="Video Uploader RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
_slug_rx = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify_filename(name: str) -> str:
    # Remove path, collapse spaces, keep extension
    base = os.path.basename(name).strip()
    base = base.replace(" ", "_")
    base = _slug_rx.sub("", base)
    return base or "upload.bin"


def _s3_key_from_url(url: str) -> str:
    # https://<bucket>.s3.<region>.amazonaws.com/<key>  -> <key>
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path


def _b64(fp: str) -> str:
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_frames(video_path: str, max_frames: int = 6) -> List[str]:
    """
    Use ffmpeg to sample a handful of JPEG frames evenly across the clip.
    Returns a list of local file paths to the saved JPEG frames.
    """
    out_dir = tempfile.mkdtemp(prefix="frames_")
    # Get duration
    prob = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    duration = 0.0
    try:
        duration = float(prob.stdout.strip())
    except Exception:
        duration = 0.0

    timestamps = []
    if duration > 0:
        step = duration / (max_frames + 1)
        timestamps = [max(0.0, (i + 1) * step) for i in range(max_frames)]
    else:
        timestamps = [0.0] * max_frames

    frame_paths: List[str] = []
    for idx, ts in enumerate(timestamps):
        out_file = os.path.join(out_dir, f"frame_{idx:02d}.jpg")
        # One frame at timestamp
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", out_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if os.path.exists(out_file):
            frame_paths.append(out_file)

    return frame_paths


# ---------- 🔧 FIXED FUNCTION ----------
def _summarize_frames(frames: List[str], foul_hint: Optional[str]) -> str:
    """
    Summarize the play from sampled frames using OpenAI Vision.

    ✅ FIX: each image item must be:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..." }}
    """
    if not frames:
        return "No frames extracted."

    content = [{
        "type": "text",
        "text": (
            "You are an NCAA football video analyst. "
            "Given these still frames from a short clip, briefly describe what happens "
            "in the play in 3–5 sentences. Emphasize actions relevant to judging fouls. "
            f"Foul hint (may be wrong or missing): {foul_hint or '—'}"
        ),
    }]

    for fp in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_b64(fp)}"
            }
        })

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def _embed(text_chunk: str) -> List[float]:
    e = client.embeddings.create(model="text-embedding-3-small", input=text_chunk)
    return e.data[0].embedding


def _retrieve_rules(summary: str, limit: int = 3) -> List[dict]:
    """
    Retrieve relevant rule chunks from 'rules' table using pgvector.
    Assumes schema: rules(id, title, section, body, embedding vector)
    """
    emb = _embed(summary)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, title, section, body
                FROM rules
                ORDER BY embedding <-> :embedding
                LIMIT :k
            """),
            {"embedding": emb, "k": limit}
        ).mappings().all()
    return [dict(r) for r in rows]


def _decide(summary: str, rules: List[dict], foul_hint: Optional[str]) -> tuple[str, float, str]:
    """
    Ask the model to predict foul/no-foul and explain, grounded in the retrieved rules.
    Returns (label, confidence, explanation)
    """
    rule_blurbs = "\n\n".join(
        [f"Rule {r['section'] or ''} - {r['title'] or ''}:\n{r['body']}" for r in rules]
    ) or "No matching rules found."

    sys = (
        "You are an NCAA football officiating assistant. "
        "Use the provided rules to decide if a foul occurred. "
        "Return a JSON object with fields: prediction (string), confidence (0-1), explanation (short). "
        "Be concise and rely on the rules. If unsure, say 'Inconclusive' with low confidence."
    )

    user = (
        f"PLAY SUMMARY:\n{summary}\n\n"
        f"FOUL HINT (may be wrong): {foul_hint or '—'}\n\n"
        f"RULE EXCERPTS:\n{rule_blurbs}\n\n"
        "Respond ONLY with JSON like:\n"
        '{"prediction":"Holding","confidence":0.72,"explanation":"..."}'
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    import json
    try:
        data = json.loads(resp.choices[0].message.content)
        pred = str(data.get("prediction", "Inconclusive"))
        conf = float(data.get("confidence", 0.3))
        expl = str(data.get("explanation", ""))
        # Clamp confidence
        conf = max(0.0, min(1.0, conf))
        return pred, conf, expl
    except Exception:
        return "Inconclusive", 0.3, "Model output could not be parsed as JSON."


# ---------- Background worker ----------
def _process_upload_row(row_id: int):
    db: Session = SessionLocal()
    try:
        row: Upload | None = db.get(Upload, row_id)
        if not row:
            return

        # Mark processing
        row.status = "processing"
        row.processed_at = datetime.now(timezone.utc)
        row.error_message = None
        db.commit()

        # 1) Download from S3 to temp file
        key = _s3_key_from_url(row.s3_url)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            s3_client.download_fileobj(BUCKET_NAME, key, tmp)
            local_video = tmp.name

        # 2) Extract frames
        frames = _extract_frames(local_video, max_frames=6)

        # 3) Summarize (✅ fixed)
        summary = _summarize_frames(frames, row.foul_type)

        # 4) Retrieve rules
        rules = _retrieve_rules(summary, limit=3)

        # 5) Decide
        label, conf, explanation = _decide(summary, rules, row.foul_type)

        # 6) Update DB
        row.prediction_label = label
        row.confidence = conf
        row.explanation = explanation
        row.status = "done"
        row.processed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        row = db.get(Upload, row_id)
        if row:
            row.status = "error"
            row.error_message = f"{type(e).__name__}: {e}"
            row.processed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    foul_type: str = Query(...),
    notes: Optional[str] = Query(None),
):
    # Clean file name and build S3 key
    clean_name = _slugify_filename(file.filename or "upload.bin")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"videos/{ts}_{clean_name}"

    # Save to a temp file first
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Upload to S3
    with open(tmp_path, "rb") as f:
        s3_client.upload_fileobj(f, BUCKET_NAME, key, ExtraArgs={"ACL": "private"})

    s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

    # Create DB row
    db: Session = SessionLocal()
    try:
        new_row = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes or "",
            timestamp=datetime.utcnow(),
            status="queued",
        )
        db.add(new_row)
        db.commit()
        db.refresh(new_row)
        row_id = new_row.id
    finally:
        db.close()

    # Kick off background RAG pipeline
    background_tasks.add_task(_process_upload_row, row_id)

    return {"id": row_id, "s3_url": s3_url, "status": "queued"}


@app.get("/api/plays")
def list_recent_plays(limit: int = Query(25, ge=1, le=200)) -> List[dict]:
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(Upload)
              .order_by(Upload.id.desc())
              .limit(limit)
              .all()
        )
        out = []
        for r in rows:
            key = _s3_key_from_url(r.s3_url)
            presigned = s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
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
                "explanation": getattr(r, "explanation", None),
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
    finally:
        db.close()


@app.post("/api/retry/{upload_id}")
def retry_processing(upload_id: int):
    db: Session = SessionLocal()
    try:
        row = db.get(Upload, upload_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "Not found"})
        row.status = "queued"
        row.error_message = None
        db.commit()
    finally:
        db.close()

    # Resume background process
    _process_upload_row(upload_id)
    return {"ok": True, "id": upload_id}