import os
import re
import io
import json
import base64
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import List, Optional
from models import Upload, Rule

import boto3
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query, Body, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text as sqltext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import asyncio
from openai import OpenAI

from database import async_session

from models import Upload  # must include retrieved_rules, human_label, human_notes, reviewed_at
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware


# ------------------------------------------------------------------------------
# Environment / Clients
# ------------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072-D
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

if not (BUCKET_NAME and DATABASE_URL and OPENAI_API_KEY):
    raise RuntimeError("Missing required env vars: AWS_S3_BUCKET, DATABASE_URL, OPENAI_API_KEY")

s3_client = boto3.client("s3", region_name=AWS_REGION)
engine = create_engine(
DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://3.135.12.183:3000",
        "http://3.135.12.183",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# or temporarily for quick testing:
# allow_origins=["*"]
# Dependency to get the database session
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/api/rules/list")
async def list_rules(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Rule))
    rules = result.scalars().all()
    return [{"label": r.title, "value": r.id} for r in rules]

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class UploadResponse(BaseModel):
    id: int
    s3_url: str

class ReviewPayload(BaseModel):
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    reviewed: bool = False

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _now_utc():
    return datetime.now(timezone.utc)

def _safe_err_text(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return "Unknown error"

def _s3_key_from_url(url: str) -> str:
    # https://bucket.s3.region.amazonaws.com/<key>
    return url.split(".amazonaws.com/")[-1]

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

async def _summarize_frames_async(frames_b64: List[str]) -> str:
    """
    Async: Summarize the play from a handful of frames.
    """
    if not frames_b64:
        return "No visual context available."

    prompt = (
        "You are an assistant that writes a concise description of a football play "
        "from a few snapshots. Mention formation/positions, motion, snap, contact, "
        "and any obvious infractions if clearly visible. Keep it under ~70 words."
    )

    msg = f"{prompt}\n\nFrames extracted: {len(frames_b64)} representative images (not attached)."

    chat = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": msg}],
        temperature=0.2,
        max_tokens=140,
    )
    return (chat.choices[0].message.content or "").strip()

def _embed_text(text_in: str) -> List[float]:
    """
    Create a 3072-D embedding (text-embedding-3-large).
    """
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text_in)
    vec = resp.data[0].embedding
    return [float(x) for x in vec]

def _retrieve_rules(summary: str, top_k: int = 3) -> List[dict]:
    """
    Retrieve closest rule chunks via pgvector (cosine).
    Column type: vector(3072)
    """
    emb = _embed_text(summary)  # 3072-D
    qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    sql = sqltext("""
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

def _predict_with_rules(summary: str, retrieved: list[dict]) -> tuple[str, float, str]:
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

    sys_prompt = (
        "You are a high school football rule assistant. "
        "Given a short play summary and a few rule snippets, choose the most likely foul label "
        "from the snippets (or 'None' if no foul). Provide a numeric confidence in [0,1] "
        "and a 1-2 sentence explanation grounded in the snippets."
    )
    user_prompt = (
        f"PLAY SUMMARY:\n{summary}\n\n"
        f"CANDIDATE RULE SNIPPETS:\n{rules_snips}\n\n"
        "Respond as JSON with keys: label, confidence, explanation."
    )

    chat = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=220,
    )
    txt = (chat.choices[0].message.content or "").strip()

    label, conf, expl = "None", 0.8, "No clear foul per snippets."
    try:
        j = json.loads(txt)
        label = str(j.get("label", "None"))
        conf = float(j.get("confidence", 0.8))
        expl = str(j.get("explanation", "")) or expl
    except Exception:
        pass
    return label, conf, expl
# ------------------------------------------------------------------------------
# Background worker (async version)
# ------------------------------------------------------------------------------
async def _process_upload_bg(upload_id: int, s3_url: str, foul_hint: str):
    async with async_session() as db:
        try:
            # 1) Download bytes
            key = _s3_key_from_url(s3_url)
            obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            video_bytes: bytes = obj["Body"].read()

            # 2) Frames -> summary
            frames = _extract_frames(video_bytes, fps=1, max_frames=6)
            summary = await _summarize_frames_async(frames)

            # 3) Retrieve rules
            retrieved = _retrieve_rules(summary, top_k=3)

            # 4) Decide with LLM
            label, confidence, explanation_text = _predict_with_rules(summary, retrieved)

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
                row.confidence = final_conf
                row.explanation = explanation_text
                row.processed_at = _now_utc()
                row.error_message = None
                row.retrieved_rules = retrieved
                await db.commit()
        except Exception as e:
            result = await db.execute(select(Upload).where(Upload.id == upload_id))
            row = result.scalar_one_or_none()
            if row:
                row.status = "error"
                row.error_message = _safe_err_text(e)
                row.processed_at = _now_utc()
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

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    print("📥 Upload request received")

    try:
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
                timestamp=datetime.utcnow(),
                status="queued",
            )
            db.add(rec)
            await db.commit()
            await db.refresh(rec)
            print("✅ DB write complete with ID:", rec.id)

            asyncio.create_task(_process_upload_bg(rec.id, s3_url, foul_type))
            response = UploadResponse(id=rec.id, s3_url=s3_url)
            print("✅ UploadResponse ready:", response.dict())
            return response
        except Exception as db_error:
            print("❌ DB write failed:", db_error)
            raise HTTPException(status_code=500, detail="DB write failed")

    except Exception as e:
        print("❌ Unexpected error in /upload:", e)
        raise HTTPException(status_code=500, detail="Unexpected server error")
    
@app.get("/api/plays")
def list_recent_plays(limit: int = Query(25, ge=1, le=200)) -> List[dict]:
    """
    Include retrieved_rules so the UI can open a drawer without another fetch.
    """
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
                "retrieved_rules": getattr(r, "retrieved_rules", None),  # <--- included
                "human_label": getattr(r, "human_label", None),
                "human_notes": getattr(r, "human_notes", None),
                "reviewed_at": r.reviewed_at.isoformat() if getattr(r, "reviewed_at", None) else None,
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/retry/{upload_id}")
async def retry_upload(upload_id: int, db: AsyncSession = Depends(get_db)):
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
    asyncio.create_task(_process_upload_bg(row.id, row.s3_url, row.foul_type))

    return {"ok": True, "id": upload_id}

@app.patch("/api/plays/{upload_id}/review")
def set_human_review(upload_id: int, payload: ReviewPayload):
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


# ------------------------------------------------------------------------------
# Rules: quick search (already handy for sanity checks)
# ------------------------------------------------------------------------------
@app.get("/api/rules/search")
def search_rules(q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=10)):
    """
    Simple retrieval endpoint to sanity-check embeddings & pgvector search.
    """
    try:
        emb = _embed_text(q)  # 3072-D
        qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        sql = sqltext("""
            SELECT id, title, section, body,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM rules
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :k
        """)
        with engine.begin() as conn:
            rows = conn.execute(sql, {"qvec": qvec_literal, "k": k}).mappings().all()
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
from typing import Optional  # Make sure this is imported at the top

class ReviewIn(BaseModel):
    human_label: Optional[str] = None
    human_notes: Optional[str] = None
    # Optional: allow overriding prediction_label/confidence if you want
    override_prediction: Optional[bool] = False

@app.post("/api/review/{upload_id}")
def submit_review(upload_id: int, payload: ReviewIn = Body(...)):
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
@app.get("/api/rules/list")
def get_rules():
    db = SessionLocal()
    rules = db.query(Rule).all()
    return [rule.as_dict() for rule in rules]