import os
import re
import io
import base64
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from openai import OpenAI

from models import Upload  # SQLAlchemy model for table `uploads`

# =============================================================================
# Config
# =============================================================================
DATABASE_URL   = os.getenv("DATABASE_URL")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME    = os.getenv("AWS_S3_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert DATABASE_URL,   "DATABASE_URL not set"
assert BUCKET_NAME,    "AWS_S3_BUCKET not set"
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"

# --- Option B: keep 3072-D embeddings ---------------------------------------
EMBED_MODEL = "text-embedding-3-large"  # 3072-dim
EMBED_DIM   = 3072

# =============================================================================
# Clients
# =============================================================================
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

s3_client = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# App
# =============================================================================
app = FastAPI(title="Video Uploader Backend (RAG 3072D)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Helpers
# =============================================================================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe_err_text(err: Exception) -> str:
    """Always store a string in DB error_message."""
    try:
        return str(err)
    except Exception:
        return "Unhandled error"

def _s3_key_from_url(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.path[1:] if p.path.startswith("/") else p.path

def _presign(key: str, expires: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expires,
    )

def _extract_frames(video_bytes: bytes, fps: int = 1, max_frames: int = 6) -> List[bytes]:
    """
    Use ffmpeg to sample up to `max_frames` PNG frames at `fps`.
    Returns raw PNG bytes for each extracted frame.
    """
    frames: List[bytes] = []
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "clip.mp4")
        with open(in_path, "wb") as f:
            f.write(video_bytes)

        out_pattern = os.path.join(td, "f_%03d.png")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            out_pattern,
        ]
        subprocess.run(cmd, check=True)

        for name in sorted(p for p in os.listdir(td) if p.startswith("f_") and p.endswith(".png")):
            with open(os.path.join(td, name), "rb") as pf:
                frames.append(pf.read())
    return frames

def _summarize_frames(frames: List[bytes]) -> str:
    """
    Vision summary using GPT-4o-mini: send 2–3 frames as data URLs with a concise prompt.
    """
    if not frames:
        return "No visual content extracted."

    pick = frames[:3]
    content: List[dict] = [
        {"type": "text",
         "text": (
             "You are an analyst for NCAA football officiating. "
             "Summarize what happens in these frames (players, ball, snap/LOS, contact, likely fouls) "
             "in 3–5 concise bullet points."
         )}
    ]
    for b in pick:
        # Ensure base64 is a *string*
        b64 = base64.b64encode(b).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
    )
    return chat.choices[0].message.content or "No summary."

def _embed_text(text_: str) -> List[float]:
    """
    Create a 3072-D embedding, defensively padding/trimming to EMBED_DIM.
    """
    text_ = (text_ or "").strip()
    if not text_:
        return [0.0] * EMBED_DIM

    vec = client.embeddings.create(model=EMBED_MODEL, input=text_).data[0].embedding
    if len(vec) != EMBED_DIM:
        if len(vec) > EMBED_DIM:
            vec = vec[:EMBED_DIM]
        else:
            vec = vec + [0.0] * (EMBED_DIM - len(vec))
    return vec

def _retrieve_rules(summary: str, top_k: int = 3) -> List[dict]:
    """
    Embed the summary (3072-D) and retrieve closest rule chunks via pgvector cosine distance.
    We pass a vector literal (no spaces inside the brackets) and cast using ::vector.
    """
    emb = _embed_text(summary)  # 3072-D
    qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"  # IMPORTANT: no spaces

    sql = text("""
        SELECT
            id,
            title,
            section,
            body,
            1 - (embedding <=> (:qvec)::vector) AS score
        FROM rules
        ORDER BY embedding <=> (:qvec)::vector
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

def _decide_label(summary: str, retrieved: List[dict]) -> tuple[str, float, str]:
    """
    Ask the model for a foul label + 0–1 confidence + short explanation grounded in retrieved rules.
    """
    rules_text = "\n\n".join(
        f"[{i+1}] {r['title']} {r.get('section','')}\n{r['body']}"
        for i, r in enumerate(retrieved)
    ) or "(no rules retrieved)"

    prompt = (
        "You are an NCAA football rules analyst. Given the play summary and rule snippets, "
        "choose the most likely foul label from: "
        "['False Start','Offside','Holding','Pass Interference','Targeting','None'].\n"
        "Return JSON with keys: label (string), confidence (0..1), explanation (<=2 sentences).\n\n"
        f"Play summary:\n{summary}\n\nRules:\n{rules_text}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    import json
    try:
        js = json.loads(raw)
        label = str(js.get("label") or "None")
        conf = float(js.get("confidence") or 0.5)
        expl = str(js.get("explanation") or "")
    except Exception:
        label, conf, expl = "None", 0.5, "Model output was not valid JSON."

    return label, conf, expl

# =============================================================================
# Background worker
# =============================================================================
def _process_upload_bg(upload_id: int, s3_url: str, foul_hint: str):
    db: Session = SessionLocal()
    try:
        # 1) Download bytes
        key = _s3_key_from_url(s3_url)
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        video_bytes: bytes = obj["Body"].read()

        # 2) Frames -> summary
        frames = _extract_frames(video_bytes, fps=1, max_frames=6)
        summary = _summarize_frames(frames)

        # 3) Retrieve + 4) Decide
        retrieved = _retrieve_rules(summary, top_k=3)
        label, conf, expl = _decide_label(summary, retrieved)

        # 5) Update DB
        row = db.query(Upload).get(upload_id)
        if row:
            row.status = "done"
            row.prediction_label = label
            row.confidence = conf
            row.explanation = expl
            row.processed_at = _now_utc()
            row.error_message = None
            db.commit()
    except Exception as e:
        row = db.query(Upload).get(upload_id)
        if row:
            row.status = "error"
            row.error_message = _safe_err_text(e)  # stringify error
            row.processed_at = _now_utc()
            db.commit()
    finally:
        db.close()

# =============================================================================
# Schemas
# =============================================================================
class UploadResponse(BaseModel):
    id: int
    s3_url: str

# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str = Form(""),
):
    data = await file.read()

    # Clean filename (no spaces / weird chars)
    clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "clip.mp4")
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"videos/{stamp}_{clean_name}"

    # Upload to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType="video/mp4")
    s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

    # Create DB row
    db: Session = SessionLocal()
    try:
        rec = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes,
            timestamp=datetime.utcnow(),
            status="queued",
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        # Kick background
        background_tasks.add_task(_process_upload_bg, rec.id, s3_url, foul_type)

        return UploadResponse(id=rec.id, s3_url=s3_url)
    finally:
        db.close()

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
            presigned = _presign(key, 3600)
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
def retry_upload(upload_id: int):
    db: Session = SessionLocal()
    try:
        row = db.query(Upload).get(upload_id)
        if not row:
            return {"ok": False, "error": "Not found"}

        row.status = "queued"
        row.error_message = None
        row.processed_at = None
        db.commit()

        # Run synchronously here; for production use a background queue/worker
        _process_upload_bg(upload_id=row.id, s3_url=row.s3_url, foul_hint=row.foul_type)
        return {"ok": True, "id": upload_id}
    finally:
        db.close()