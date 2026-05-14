"""
main_integration.py
===================
DAVE — Integration guide for video_processor.py into your existing FastAPI backend.

This file shows exactly WHERE and HOW to add the video processing step
into your current main.py. Do not replace your main.py with this file —
use it as a diff/guide to merge the changes in.

Changes needed in main.py:
  1. Import video_processor
  2. In your upload endpoint: call process_video() after S3 upload, before RAG query
  3. Pass the PlayAnalysis context into your RAG function
  4. Store the analysis in your database for the HITL feedback loop
  5. Return the analysis summary alongside the ruling in the API response
"""

# =============================================================================
# SECTION 1 — New imports to add at the top of your main.py
# =============================================================================
# Add these lines alongside your existing imports:
#
#   from video_processor import process_video, PlayAnalysis
#   import logging
#   logger = logging.getLogger(__name__)
# =============================================================================


# =============================================================================
# SECTION 2 — Updated Pydantic response model
# =============================================================================
# Add this model (or extend your existing PlayResponse) to return
# the vision analysis alongside the ruling.

from pydantic import BaseModel
from typing import Optional

class PlayResponse(BaseModel):
    """
    Full API response returned to the React frontend after play analysis.
    Extend your existing response model with the new vision fields.
    """
    # --- Existing fields (keep whatever you already have) ---
    id: int
    rule_type: str
    ruling: str                  # e.g. "FOUL — Pass Interference"
    explanation: str             # RAG-generated explanation with rule citations
    rule_citations: list[str]    # e.g. ["NFHS Rule 7-5-10", ...]

    # --- New fields from video_processor ---
    play_summary: str            # GPT-4o one-sentence play summary
    play_type: str               # run / pass / kick / punt / scrimmage
    action_description: str      # detailed description of what happened
    potential_violations: list[str]  # violations GPT-4o spotted
    frames_analyzed: int         # how many frames were reviewed

    # HITL fields — filled in by the official after reviewing the ruling
    official_agrees: Optional[bool] = None   # True = AI was right, False = wrong
    official_notes: Optional[str] = None


# =============================================================================
# SECTION 3 — Updated upload endpoint
# =============================================================================
# This shows the full flow. Merge this logic into your existing upload endpoint.
# Your current endpoint probably looks like:
#
#   @app.post("/upload")
#   async def upload_play(video: UploadFile, rule_id: int, notes: str = ""):
#       s3_key = upload_to_s3(video)
#       ruling = rag_query(rule_id, notes)
#       save_to_db(s3_key, rule_id, ruling)
#       return ruling
#
# Replace it with the version below:

from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
import boto3, os

app = FastAPI()

S3_BUCKET  = os.getenv("S3_BUCKET_NAME", "dave-video-uploads")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def upload_video_to_s3(video_file: UploadFile) -> str:
    """
    Upload the video to S3. Returns the S3 key.
    You likely already have this — keep your existing implementation.
    """
    import uuid
    s3_key = f"plays/{uuid.uuid4()}.mp4"
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_fileobj(video_file.file, S3_BUCKET, s3_key)
    return s3_key


def rag_query_with_video_context(
    rule_type: str,
    video_context: str,   # <-- NEW: PlayAnalysis.to_rag_context()
    notes: str = "",
) -> tuple[str, str, list[str]]:
    """
    Your existing RAG query function, updated to accept video context.

    The video_context string is injected into the prompt so the LLM
    reasons over WHAT HAPPENED in the video alongside the rule text.

    Returns: (ruling, explanation, citations)

    -----------------------------------------------------------------------
    HOW TO UPDATE YOUR EXISTING rag_query / _decide_label FUNCTION:

    In your current prompt, you likely have something like:

        prompt = f"Rule: {rule_text}\\nQuestion: Did a {rule_type} occur?"

    Update it to include the video context:

        prompt = f\"\"\"
        Rule: {rule_text}

        Play Description (from video analysis):
        {video_context}

        Official's Notes: {notes}

        Question: Based on the play description and the rule above,
        did a {rule_type} occur? Cite the specific rule clause.
        Respond with: RULING (FOUL or NO FOUL), EXPLANATION, CITATIONS.
        \"\"\"

    That's it — the RAG retrieval stays the same, only the prompt changes.
    -----------------------------------------------------------------------
    """
    # Placeholder — replace with your actual RAG call
    ruling      = "FOUL — Pass Interference (example)"
    explanation = "Based on the video analysis and NFHS Rule 7-5-10..."
    citations   = ["NFHS Rule 7-5-10", "NFHS Rule 2-32-1"]
    return ruling, explanation, citations


def save_play_to_db(
    s3_key: str,
    rule_type: str,
    ruling: str,
    explanation: str,
    citations: list[str],
    analysis,   # PlayAnalysis object
) -> int:
    """
    Save the play + analysis to your PostgreSQL database.
    Add the new columns listed in SECTION 4 below to capture vision data.
    Returns the new play's database ID.
    """
    # Replace with your actual SQLAlchemy / database.py call
    # Example:
    #   play = Play(
    #       s3_key=s3_key,
    #       rule_type=rule_type,
    #       ruling=ruling,
    #       explanation=explanation,
    #       citations=citations,
    #       play_summary=analysis.play_summary,
    #       play_type=analysis.play_type,
    #       action_description=analysis.action_description,
    #       frames_analyzed=analysis.frames_analyzed,
    #   )
    #   db.add(play)
    #   db.commit()
    #   return play.id
    return 1  # placeholder


@app.post("/upload", response_model=PlayResponse)
async def upload_play(
    video:     UploadFile = Form(...),
    rule_type: str        = Form(...),
    notes:     str        = Form(""),
):
    """
    Updated upload endpoint with video processing step.

    Step-by-step:
      1. Upload video to S3  (existing)
      2. Process video → frame extraction + GPT-4o analysis  (NEW)
      3. RAG query with video context injected into prompt    (updated)
      4. Save everything to DB                                (updated)
      5. Return full response to frontend                     (updated)
    """
    # --- Step 1: Upload to S3 (existing) ---
    try:
        s3_key = upload_video_to_s3(video)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {exc}")

    # --- Step 2: Process video with GPT-4o vision (NEW) ---
    try:
        from video_processor import process_video
        analysis = process_video(s3_key=s3_key, rule_type=rule_type, notes=notes or None)
    except Exception as exc:
        # Don't block the whole request if vision fails — log and continue
        # with an empty context. The RAG query will still work, just without
        # the video analysis context.
        import logging
        logging.getLogger(__name__).error("Video processing failed: %s", exc, exc_info=True)
        from video_processor import PlayAnalysis
        analysis = PlayAnalysis(
            play_summary="Video analysis unavailable",
            action_description=notes,
            analysis_notes=f"Vision processing error: {exc}",
        )

    # --- Step 3: RAG query with video context (updated) ---
    video_context = analysis.to_rag_context()
    ruling, explanation, citations = rag_query_with_video_context(
        rule_type=rule_type,
        video_context=video_context,
        notes=notes,
    )

    # --- Step 4: Save to database (updated) ---
    play_id = save_play_to_db(
        s3_key=s3_key,
        rule_type=rule_type,
        ruling=ruling,
        explanation=explanation,
        citations=citations,
        analysis=analysis,
    )

    # --- Step 5: Return full response ---
    return PlayResponse(
        id=play_id,
        rule_type=rule_type,
        ruling=ruling,
        explanation=explanation,
        rule_citations=citations,
        play_summary=analysis.play_summary,
        play_type=analysis.play_type,
        action_description=analysis.action_description,
        potential_violations=analysis.potential_violations,
        frames_analyzed=analysis.frames_analyzed,
    )


# =============================================================================
# SECTION 4 — Database model changes (models.py)
# =============================================================================
# Add these columns to your Play model in models.py:
#
#   from sqlalchemy import Column, String, Integer, Boolean, Text, ARRAY
#
#   class Play(Base):
#       __tablename__ = "plays"
#
#       # --- existing columns ---
#       id            = Column(Integer, primary_key=True, index=True)
#       s3_key        = Column(String)
#       rule_type     = Column(String)
#       ruling        = Column(String)
#       explanation   = Column(Text)
#       created_at    = Column(DateTime, default=datetime.utcnow)
#
#       # --- NEW columns for vision analysis ---
#       play_summary         = Column(Text, nullable=True)
#       play_type            = Column(String(20), nullable=True)
#       action_description   = Column(Text, nullable=True)
#       field_location       = Column(String, nullable=True)
#       ball_in_air          = Column(Boolean, nullable=True)
#       players_involved     = Column(ARRAY(String), nullable=True)
#       potential_violations = Column(ARRAY(String), nullable=True)
#       frames_analyzed      = Column(Integer, nullable=True)
#       raw_gpt_response     = Column(Text, nullable=True)   # for audit/debug
#
#       # --- HITL feedback columns (for your existing HITL work) ---
#       official_agrees      = Column(Boolean, nullable=True)
#       official_notes       = Column(Text, nullable=True)
#
# After updating models.py, generate an Alembic migration:
#   alembic revision --autogenerate -m "add vision analysis columns"
#   alembic upgrade head


# =============================================================================
# SECTION 5 — HITL feedback endpoint (bonus)
# =============================================================================
# Add this endpoint so officials can agree/disagree with the ruling.
# This data becomes your training signal for future improvements.

class HITLFeedback(BaseModel):
    play_id:        int
    official_agrees: bool
    official_notes: Optional[str] = None


@app.post("/plays/{play_id}/feedback")
async def submit_feedback(play_id: int, feedback: HITLFeedback):
    """
    Called by the React frontend when an official taps Agree/Disagree.
    Stores their verdict alongside the AI ruling for future analysis.
    """
    # Replace with your actual DB update
    # db.query(Play).filter(Play.id == play_id).update({
    #     "official_agrees": feedback.official_agrees,
    #     "official_notes":  feedback.official_notes,
    # })
    # db.commit()
    return {"status": "feedback recorded", "play_id": play_id}
