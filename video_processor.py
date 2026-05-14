"""
video_processor.py
==================
DAVE — Digital Artificial Video Evaluation
Video Processing Pipeline: S3 Download → Frame Extraction → GPT-4o Vision Analysis

This module sits between video upload and the RAG rules query.
It converts a raw football play video into a structured, text-based
play description that the RAG pipeline can reason over.

Flow:
    S3 video key
        → download to temp file
        → extract N evenly-spaced frames (+ motion-spike frames)
        → encode frames as base64 JPEG
        → send frame sequence to GPT-4o with football officiating prompt
        → return PlayAnalysis (structured description + detected violations)
        → feed into existing RAG rules query
"""

import os
import base64
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import boto3
import cv2
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

S3_BUCKET          = os.getenv("S3_BUCKET_NAME", "dave-video-uploads")
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
VISION_MODEL       = os.getenv("DAVE_VISION_MODEL", "gpt-4o")

# How many evenly-spaced frames to pull from the video.
# 8 frames covers most 3–10 second clips well; raise to 12 for longer clips.
BASE_FRAME_COUNT   = int(os.getenv("DAVE_BASE_FRAMES", "8"))

# Motion-spike sampling: pull extra frames when inter-frame difference
# exceeds this threshold (0–255 scale). Lower = more sensitive.
MOTION_THRESHOLD   = float(os.getenv("DAVE_MOTION_THRESHOLD", "30.0"))
MAX_MOTION_FRAMES  = int(os.getenv("DAVE_MAX_MOTION_FRAMES", "4"))

# GPT-4o image detail level: "low" (faster/cheaper) or "high" (more precise)
IMAGE_DETAIL       = os.getenv("DAVE_IMAGE_DETAIL", "high")

# JPEG quality for frames sent to GPT-4o (0–100).
# 85 is a good balance of quality vs. token cost.
JPEG_QUALITY       = int(os.getenv("DAVE_JPEG_QUALITY", "85"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayAnalysis:
    """
    Structured output from GPT-4o vision analysis of a football play.
    This is passed directly into the RAG query as additional context.
    """
    # One-sentence summary of what happened on the play
    play_summary: str = ""

    # Detailed description of relevant player actions (blocking, tackling, etc.)
    action_description: str = ""

    # Players / positions involved in the potential foul
    players_involved: list[str] = field(default_factory=list)

    # Field zone where the action occurred (e.g., "line of scrimmage", "open field")
    field_location: str = ""

    # Whether the ball was in the air at the time of the action
    ball_in_air: bool = False

    # Phase of play: "run", "pass", "kick", "punt", "scrimmage"
    play_type: str = ""

    # Any violations GPT-4o flagged, even if not matching the submitted rule
    potential_violations: list[str] = field(default_factory=list)

    # Free-form analysis notes from the model
    analysis_notes: str = ""

    # Number of frames actually analyzed
    frames_analyzed: int = 0

    # Raw text response from GPT-4o (kept for debugging / audit trail)
    raw_gpt_response: str = ""

    def to_rag_context(self) -> str:
        """
        Serialize to a concise string suitable for injection into the
        RAG prompt as additional context alongside the rule text.
        """
        lines = [
            f"PLAY SUMMARY: {self.play_summary}",
            f"PLAY TYPE: {self.play_type}",
            f"FIELD LOCATION: {self.field_location}",
            f"BALL IN AIR: {'Yes' if self.ball_in_air else 'No'}",
            f"PLAYERS INVOLVED: {', '.join(self.players_involved) or 'Not identified'}",
            f"ACTION DESCRIPTION: {self.action_description}",
        ]
        if self.potential_violations:
            lines.append(f"POTENTIAL VIOLATIONS OBSERVED: {', '.join(self.potential_violations)}")
        if self.analysis_notes:
            lines.append(f"ANALYST NOTES: {self.analysis_notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _download_from_s3(s3_key: str, local_path: str) -> None:
    """Download a video file from S3 to a local temp path."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    logger.info("Downloading s3://%s/%s → %s", S3_BUCKET, s3_key, local_path)
    s3.download_file(S3_BUCKET, s3_key, local_path)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Extract a representative set of frames from the video.

    Strategy:
      1. Pull BASE_FRAME_COUNT evenly-spaced frames across the full duration.
      2. Scan for motion spikes (large inter-frame differences) and add up to
         MAX_MOTION_FRAMES extra frames around those spikes.
      3. De-duplicate by timestamp and sort chronologically.

    Returns a list of BGR numpy arrays (OpenCV format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        raise ValueError("Video has no readable frames.")

    logger.info("Video: %d frames @ %.1f fps (%.1fs)", total_frames, fps, total_frames / fps)

    # --- Step 1: evenly-spaced indices ---
    step = max(1, total_frames // BASE_FRAME_COUNT)
    base_indices = set(range(0, total_frames, step))
    # Always include first and last frame
    base_indices.add(0)
    base_indices.add(total_frames - 1)

    # --- Step 2: motion-spike detection ---
    # Read every 5th frame for efficiency, compare with previous
    motion_indices: set[int] = set()
    prev_gray = None
    sample_step = max(1, total_frames // 50)  # sample ~50 points

    for idx in range(0, total_frames, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # small for speed
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff > MOTION_THRESHOLD:
                # Add this frame and its neighbours
                for offset in (-sample_step // 2, 0, sample_step // 2):
                    candidate = idx + offset
                    if 0 <= candidate < total_frames:
                        motion_indices.add(candidate)
        prev_gray = gray

    # Keep only the top MAX_MOTION_FRAMES motion frames (by index proximity to middle)
    mid = total_frames // 2
    motion_indices = set(
        sorted(motion_indices, key=lambda i: abs(i - mid))[:MAX_MOTION_FRAMES]
    )

    # --- Step 3: combine, sort, read full-res frames ---
    all_indices = sorted(base_indices | motion_indices)
    frames: list[np.ndarray] = []

    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    logger.info("Extracted %d frames (base=%d, motion=%d)", len(frames), len(base_indices), len(motion_indices))
    return frames


def _frames_to_base64(frames: list[np.ndarray]) -> list[str]:
    """Encode a list of BGR frames as base64 JPEG strings."""
    encoded = []
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    for frame in frames:
        success, buf = cv2.imencode(".jpg", frame, encode_params)
        if success:
            encoded.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    return encoded


# ---------------------------------------------------------------------------
# GPT-4o vision analysis
# ---------------------------------------------------------------------------

# System prompt — tuned for NFHS football officiating
_SYSTEM_PROMPT = """
You are an expert NFHS (National Federation of State High School Associations)
football officiating analyst. Your role is to analyze video frames from a
football play and provide a detailed, objective description that will be used
by an AI system to evaluate whether a penalty occurred.

Be precise about:
- Player positions (offense/defense, linemen/backs/receivers/DBs)
- Body contact details (hand placement, target area, force)
- Timing relative to the snap, release, and ball arrival
- Ball location and whether it was catchable
- Field zone (line of scrimmage, beyond LOS, near sideline, end zone, etc.)

Do NOT make the final ruling — only describe what you see factually.
Remain objective. If a frame is unclear or blurry, note that explicitly.
""".strip()

_ANALYSIS_PROMPT = """
Analyze these {n} frames from a high school football play. The official has
flagged this play for review under the rule: "{rule_type}".
{notes_section}

Respond in this exact JSON format (no markdown, no extra text):
{{
  "play_summary": "<one sentence summary of the play>",
  "play_type": "<run|pass|kick|punt|scrimmage>",
  "field_location": "<where the key action occurred>",
  "ball_in_air": <true|false>,
  "players_involved": ["<position/description>", ...],
  "action_description": "<detailed description of the relevant player actions>",
  "potential_violations": ["<violation if visible>", ...],
  "analysis_notes": "<anything unclear, ambiguous, or worth flagging>"
}}
""".strip()


def _analyze_with_gpt4o(
    frame_b64_list: list[str],
    rule_type: str,
    notes: Optional[str] = None,
) -> PlayAnalysis:
    """
    Send frames to GPT-4o and parse the structured response.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    notes_section = f'\nOfficial\'s notes: "{notes}"' if notes else ""
    user_text = _ANALYSIS_PROMPT.format(
        n=len(frame_b64_list),
        rule_type=rule_type,
        notes_section=notes_section,
    )

    # Build the content list: text prompt + all frames
    content: list[dict] = [{"type": "text", "text": user_text}]
    for b64 in frame_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": IMAGE_DETAIL,
            },
        })

    logger.info("Sending %d frames to %s for analysis", len(frame_b64_list), VISION_MODEL)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        max_tokens=1024,
        temperature=0.1,  # low temp for factual analysis
    )

    raw_text = response.choices[0].message.content or ""
    logger.debug("GPT-4o raw response: %s", raw_text)

    return _parse_gpt_response(raw_text, len(frame_b64_list))


def _parse_gpt_response(raw_text: str, frames_analyzed: int) -> PlayAnalysis:
    """Parse GPT-4o JSON response into a PlayAnalysis, with graceful fallback."""
    import json

    analysis = PlayAnalysis(raw_gpt_response=raw_text, frames_analyzed=frames_analyzed)

    # Strip any accidental markdown fences
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])

    try:
        data = json.loads(clean)
        analysis.play_summary         = data.get("play_summary", "")
        analysis.play_type            = data.get("play_type", "")
        analysis.field_location       = data.get("field_location", "")
        analysis.ball_in_air          = bool(data.get("ball_in_air", False))
        analysis.players_involved     = data.get("players_involved", [])
        analysis.action_description   = data.get("action_description", "")
        analysis.potential_violations = data.get("potential_violations", [])
        analysis.analysis_notes       = data.get("analysis_notes", "")
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not parse GPT-4o JSON response: %s. Using raw text.", exc)
        analysis.play_summary       = "Vision analysis completed (unstructured)"
        analysis.action_description = raw_text

    return analysis


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_video(
    s3_key: str,
    rule_type: str,
    notes: Optional[str] = None,
) -> PlayAnalysis:
    """
    Main entry point — called from your FastAPI endpoint BEFORE the RAG query.

    Args:
        s3_key:     S3 object key of the uploaded video (e.g. "plays/2024/clip.mp4")
        rule_type:  The foul type the official selected (e.g. "Holding", "Pass Interference")
        notes:      Optional free-text notes from the official

    Returns:
        PlayAnalysis with .to_rag_context() for injection into the RAG prompt

    Example:
        analysis = process_video("plays/abc123.mp4", "Pass Interference", "DB hit WR before ball")
        rag_context = analysis.to_rag_context()
        ruling = your_rag_query(rule_type, rag_context)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # 1. Download from S3
        _download_from_s3(s3_key, tmp_path)

        # 2. Extract frames
        frames = _extract_frames(tmp_path)
        if not frames:
            raise ValueError("No frames could be extracted from the video.")

        # 3. Encode frames
        frame_b64_list = _frames_to_base64(frames)

        # 4. Analyze with GPT-4o
        analysis = _analyze_with_gpt4o(frame_b64_list, rule_type, notes)

        logger.info(
            "Video processing complete: %d frames analyzed, play_type=%s",
            analysis.frames_analyzed,
            analysis.play_type,
        )
        return analysis

    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Cleaned up temp file: %s", tmp_path)
