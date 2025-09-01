# main.py
import os
import io
import re
import time
import base64
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from models import Base, Upload  # Upload table lives in models.py

# ----- env & clients ---------------------------------------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (BUCKET_NAME and DATABASE_URL):
    raise RuntimeError("Missing required env vars: AWS_S3_BUCKET or DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base.metadata.create_all(bind=engine)

s3_client = boto3.client("s3", region_name=AWS_REGION)

# Optional OpenAI client (lazy import so running without key still works)
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

# ----- FastAPI ---------------------------------------------------------------
app = FastAPI(title="Video Uploader Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- helpers ---------------------------------------------------------------

def _safe_error(e: Exception | str) -> str:
    """Always return a plain string for DB storage/UI."""
    try:
        return str(e)
    except Exception:
        return "Unknown error"

def _safe_b64(data: bytes) -> str:
    """Base64-encode bytes and return a UTF-8 string (never bytes)."""
    return base64.b64encode(data).decode("utf-8")

def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    # replace spaces and bad chars
    base = re.sub(r"[^\w\-.]+", "_", base)
    # trim to something reasonable
    return base[:120] or "upload.bin"

def _s3_key_from_url(url: str) -> str:
    # Works for both virtual-hosted and path-style URLs if path contains the key.
    # We stored full https URL; everything after the bucket host is the key.
    # Example: https://bucket.s3.region.amazonaws.com/videos/file.mp4
    # Split on ".amazonaws.com/" and take the tail.
    marker = ".amazonaws.com/"
    if marker in url:
        return url.split(marker, 1)[1]
    # Fallback: try to strip the protocol/host and return the path minus leading slash
    return url.split("/", 3)[-1].lstrip("/")

# --- Embeddings / summarization stubs (kept simple; can be upgraded later) ---

def _embed_text(text_input: str) -> List[float]:
    """
    Returns a list[float] for pgvector. If OPENAI key isn't set, return a tiny fake vector.
    """
    if openai_client:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text_input.strip()[:8000],
        )
        return resp.data[0].embedding  # type: ignore[attr-defined]
    # fallback deterministic small vector (dimension 1536 expected; we can pad)
    import math
    vals = [math.sin(i) for i in range(64)]
    # pad to 1536 with zeros so cosine ops work
    return vals + [0.0] * (1536 - len(vals))

def _summarize_frames(s3_url: str, foul_hint: str, notes: str) -> str:
    """
    Very light 'summary' to keep pipeline flowing. If OpenAI is configured,
    produce a short description using text completion; otherwise return a heuristic string.
    """
    base_hint = foul_hint or "Unknown foul"
    base_notes = (notes or "").strip()
    base_desc = f"Video of a football play. Foul hint: {base_hint}."
    if base_notes:
        base_desc += f" Notes: {base_notes}."
    if not openai_client:
        return base_desc

    prompt = (
        "You are assisting a football officiating tool. "
        "Write a one-sentence neutral summary of the referenced play (do NOT judge legality). "
        f"Use this context as hints: foul hint={base_hint}; notes='{base_notes}'."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise, neutral descriptions of plays."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()  # type: ignore[attr-defined]
    except Exception as e:
        return f"{base_desc} (summary-fallback: {_safe_error(e)})"

# ------------------ FIXED RETRIEVAL FUNCTION (pgvector) ----------------------

from sqlalchemy import text as sql_text

def _retrieve_rules(summary: str, top_k: int = 3) -> list[dict]:
    """
    Embed the summary and retrieve the closest rule chunks via pgvector.
    Uses cosine distance (<=>). We pass the embedding as a vector literal
    and cast with ::vector to avoid 'operator does not exist' errors.
    """
    # 1) Embed the text (list of floats)
    emb = _embed_text(summary)  # length must match rules.embedding dimension

    # 2) Build a pgvector literal: "[0.123,0.456,...]"
    qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    # 3) Query with proper casting to vector
    sql = sql_text("""
        SELECT
            id,
            title,
            section,
            body,
            1 - (embedding <=> :qvec::vector) AS score  -- cosine similarity
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

# ------------- Decision model (simple; swap in your full RAG later) ----------

def _decide_label(summary: str, retrieved: list[dict]) -> tuple[str, float, str]:
    """
    Produce (label, confidence, explanation). Uses OpenAI if available; otherwise a stub.
    """
    if not openai_client:
        # Simple heuristic fallback
        label = "Holding" if "hold" in summary.lower() else "Offside"
        return label, 0.70, "Stub predictor. Replace with RAG logic."

    rules_text = "\n\n".join(
        f"[{r.get('title','Rule')}{' - ' + r.get('section','') if r.get('section') else ''}]\n{r.get('body','')}"
        for r in retrieved
    ) or "(no rules)"

    user_msg = (
        "Given this summary of a play and relevant NCAA rule snippets, "
        "predict the foul (single label), provide a 0-1 confidence, and a brief explanation.\n\n"
        f"Summary:\n{summary}\n\nRules:\n{rules_text}\n\n"
        "Return a strict JSON object with keys: label (string), confidence (number between 0 and 1), explanation (string)."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise assistant for football officiating decisions."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content  # type: ignore[attr-defined]
        import json
        data = json.loads(content)
        label = str(data.get("label") or "Unknown")
        conf = float(data.get("confidence") or 0.5)
        expl = str(data.get("explanation") or "")
        conf = max(0.0, min(1.0, conf))
        return label, conf, expl
    except Exception as e:
        return "Unknown", 0.5, f"Decision fallback: {_safe_error(e)}"

# --------------------------- background worker -------------------------------

def _process_upload(upload_id: int) -> None:
    db: Session = SessionLocal()
    try:
        row: Optional[Upload] = db.get(Upload, upload_id)
        if not row:
            return
        # mark running
        row.status = "processing"
        row.error_message = None
        db.commit()

        # Summarize
        summary = _summarize_frames(row.s3_url, row.foul_type or "", row.notes or "")

        # Retrieve rules
        retrieved = _retrieve_rules(summary, top_k=3)

        # Decide
        label, conf, expl = _decide_label(summary, retrieved)

        # update row
        row.prediction_label = label
        row.confidence = conf
        # ensure explanation column exists in your models.py (as Text or String)
        if hasattr(row, "explanation"):
            row.explanation = expl  # type: ignore[attr-defined]
        row.processed_at = datetime.now(timezone.utc)
        row.status = "done"
        db.commit()
    except Exception as e:
        # store stringified error
        try:
            row = db.get(Upload, upload_id)
            if row:
                row.status = "error"
                row.error_message = _safe_error(e)
                row.processed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()

# ------------------------------- routes --------------------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file"),
    foul_type: str = Form(...),
    notes: Optional[str] = Form(None),
):
    filename = _safe_filename(file.filename or "upload.bin")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"videos/{ts}_{filename}"

    # upload to S3
    try:
        body = await file.read()
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType=file.content_type or "application/octet-stream")
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {_safe_error(e)}")

    s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

    # write DB row
    db: Session = SessionLocal()
    try:
        row = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes or "",
            timestamp=datetime.utcnow(),
            status="queued",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        upload_id = row.id
    finally:
        db.close()

    # queue background work
    background_tasks.add_task(_process_upload, upload_id)

    return {"ok": True, "id": upload_id, "s3_url": s3_url}

@app.get("/api/plays")
def list_recent_plays(limit: int = Query(20, ge=1, le=200)) -> List[dict]:
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
            try:
                presigned = s3_client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": BUCKET_NAME, "Key": key},
                    ExpiresIn=3600,
                )
            except Exception:
                presigned = r.s3_url  # fallback

            out.append({
                "id": r.id,
                "foul_type": r.foul_type,
                "notes": r.notes,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "status": r.status,
                "prediction_label": r.prediction_label,
                "confidence": r.confidence,
                "processed_at": r.processed_at.isoformat() if getattr(r, "processed_at", None) else None,
                "error_message": r.error_message,
                "explanation": getattr(r, "explanation", None),
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
    finally:
        db.close()

@app.post("/api/retry/{upload_id}")
def retry_upload(upload_id: int, background_tasks: BackgroundTasks):
    db: Session = SessionLocal()
    try:
        row: Optional[Upload] = db.get(Upload, upload_id)
        if not row:
            raise HTTPException(status_code=404, detail="Upload not found")

        row.status = "queued"
        row.error_message = None
        row.processed_at = None
        db.commit()

        background_tasks.add_task(_process_upload, upload_id)
        return {"ok": True, "id": upload_id}
    finally:
        db.close()