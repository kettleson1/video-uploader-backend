import os
import re
import io
import json
import base64
import tempfile
import subprocess
import secrets
from datetime import datetime, timezone
from typing import List, Optional
from models import Upload, Rule

import boto3
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query, Body, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text as sqltext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import asyncio
from openai import AsyncOpenAI

from database import async_session

from models import Upload  # must include retrieved_rules, human_label, human_notes, reviewed_at
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
DAVE_API_KEY = os.getenv("DAVE_API_KEY")
CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))

if not (BUCKET_NAME and DATABASE_URL and OPENAI_API_KEY):
    raise RuntimeError("Missing required env vars: AWS_S3_BUCKET, DATABASE_URL, OPENAI_API_KEY")

s3_client = boto3.client("s3", region_name=AWS_REGION)
engine = create_engine(
DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
# Dependency to get the database session
async def get_db():
    async with async_session() as session:
        yield session

def require_api_key(x_dave_api_key: Optional[str] = Header(None, alias="X-DAVE-API-Key")):
    if not DAVE_API_KEY:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if not x_dave_api_key or not secrets.compare_digest(x_dave_api_key, DAVE_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/api/rules/list")
async def list_rules(
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
):
    result = await db.execute(select(Rule))
    rules = result.scalars().all()
    return [{"label": r.title, "value": r.title} for r in rules]

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

class OfficialFeedbackIn(BaseModel):
    official_agrees: bool
    human_label: Optional[str] = None
    human_notes: Optional[str] = None

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _now_utc():
    return datetime.utcnow()

def _safe_err_text(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return "Unknown error"

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

async def _summarize_frames_async(
    frames_b64: List[str],
    foul_hint: str = "",
    notes: str = "",
) -> str:
    """
    Async: Summarize the play from a handful of actual video frames.
    """
    if not frames_b64:
        return "No visual context available."

    system_prompt = (
        "You are an expert NFHS high school football officiating video analyst. "
        "Describe only what is visible in the frames. Be precise about timing, "
        "player roles, ball location, contact type, target area, and whether the "
        "action is clear or ambiguous. Do not invent unseen action."
    )

    user_text = (
        f"Analyze these {len(frames_b64)} chronological frames from one football play.\n"
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
async def _retrieve_rules_async(summary: str, top_k: int = 3) -> List[dict]:
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
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT :k
    """)

    async with async_session() as session:
        rows = (await session.execute(sql, {"qvec": qvec_literal, "k": top_k})).mappings().all()

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

    sys_prompt = (
        "You are an NFHS high school football rules assistant. "
        "Choose the most likely foul label only when the video evidence and rule snippets support it. "
        "If the evidence is unclear or no provided rule fits, use 'None' or 'Uncertain'. "
        "Ground the explanation in visible evidence and the cited snippets."
    )
    user_prompt = (
        f"PLAY SUMMARY:\n{summary}\n\n"
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
# ------------------------------------------------------------------------------
# Background worker (async version)
# ------------------------------------------------------------------------------
async def _process_upload_bg(upload_id: int, s3_url: str, foul_hint: str, notes: str = ""):
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

            summary = await _summarize_frames_async(frames, foul_hint=foul_hint, notes=notes)
            print(f"✍️ Summary: {summary}")

            # 3) Retrieve rules (async)
            retrieval_query = "\n".join(
                part for part in [summary, f"Foul hint: {foul_hint}" if foul_hint else "", notes] if part
            )
            retrieved = await _retrieve_rules_async(retrieval_query, top_k=6)
            print(f"📚 Retrieved {len(retrieved)} rules")
            
            # 4) Decide with LLM
            label, confidence, explanation_text = await _predict_with_rules(
                summary,
                retrieved,
                foul_hint=foul_hint,
                notes=notes,
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
    _auth: None = Depends(require_api_key),
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

            asyncio.create_task(_process_upload_bg(rec.id, s3_url, foul_type, notes))
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
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(require_api_key),
) -> List[dict]:
    try:
        result = await db.execute(
            select(Upload)
            .order_by(Upload.id.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

        out = []
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
                "retrieved_rules": getattr(r, "retrieved_rules", None),
                "human_label": getattr(r, "human_label", None),
                "human_notes": getattr(r, "human_notes", None),
                "reviewed_at": r.reviewed_at.isoformat() if getattr(r, "reviewed_at", None) else None,
                "official_agrees": official_agrees,
                "s3_url": r.s3_url,
                "presigned_url": presigned,
            })
        return out
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
    asyncio.create_task(_process_upload_bg(row.id, row.s3_url, row.foul_type, row.notes or ""))

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
    _auth: None = Depends(require_api_key),
):
    """
    Simple retrieval endpoint to sanity-check embeddings & pgvector search.
    """
    try:
        emb = await _embed_text(q)  # 3072-D
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
