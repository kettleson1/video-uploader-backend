"""Frame extraction helpers for chronological, full-clip visual coverage."""

import base64
import math
import os
import subprocess
import tempfile
from typing import List, Sequence


def probe_video_path_duration(src: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        src,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    try:
        return max(0.0, float((proc.stdout or "0").strip()))
    except ValueError:
        return 0.0


def frame_sample_timestamps(duration: float, fps: int = 1, max_frames: int = 6) -> List[float]:
    """Return chronological sample points distributed across the complete clip."""
    if duration <= 0 or fps <= 0 or max_frames <= 0:
        return []
    sample_count = min(max_frames, max(1, int(math.ceil(duration * fps))))
    return [duration * (index + 0.5) / sample_count for index in range(sample_count)]


def focused_sample_timestamps(
    duration: float,
    event_times: Sequence[float],
    window_seconds: float = 2.0,
    fps: float = 4.0,
    max_frames: int = 20,
) -> List[float]:
    """Return dense, bounded samples around one or more suspected contact events."""
    if duration <= 0 or window_seconds <= 0 or fps <= 0 or max_frames <= 0:
        return []

    step = 1.0 / fps
    timestamps: set[float] = set()
    for raw_event_time in event_times:
        try:
            event_time = max(0.0, min(duration, float(raw_event_time)))
        except (TypeError, ValueError):
            continue
        start = max(0.0, event_time - window_seconds)
        end = min(duration, event_time + window_seconds)
        count = max(1, int(math.floor((end - start) / step)) + 1)
        for index in range(count):
            timestamps.add(round(min(end, start + index * step), 6))
        timestamps.add(round(event_time, 6))
        timestamps.add(round(end, 6))

    ordered = sorted(timestamps)
    if len(ordered) <= max_frames:
        return ordered
    if max_frames == 1:
        return [ordered[len(ordered) // 2]]
    selected_indexes = {
        round(index * (len(ordered) - 1) / (max_frames - 1))
        for index in range(max_frames)
    }
    return [ordered[index] for index in sorted(selected_indexes)]


def _extract_at_timestamps(video_bytes: bytes, timestamps: Sequence[float]) -> List[dict]:
    """Extract timestamped base64 JPEG frames from the requested clip positions."""
    frames: List[dict] = []
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4")
        with open(src, "wb") as handle:
            handle.write(video_bytes)

        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = os.path.join(td, f"frame_{index:04d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.6f}",
                "-i", src,
                "-map", "0:v:0",
                "-frames:v", "1",
                "-q:v", "2",
                frame_path,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0 or not os.path.exists(frame_path):
                continue
            with open(frame_path, "rb") as frame:
                frames.append({
                    "timestamp": float(timestamp),
                    "image_b64": base64.b64encode(frame.read()).decode("utf-8"),
                })
    return frames


def extract_timestamped_frames(video_bytes: bytes, fps: int = 1, max_frames: int = 6) -> List[dict]:
    """Return timestamped frames distributed chronologically across the full clip."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "probe.mp4")
        with open(src, "wb") as handle:
            handle.write(video_bytes)
        duration = probe_video_path_duration(src)
    timestamps = frame_sample_timestamps(duration, fps=fps, max_frames=max_frames)
    return _extract_at_timestamps(video_bytes, timestamps)


def extract_focused_frames(
    video_bytes: bytes,
    event_times: Sequence[float],
    window_seconds: float = 2.0,
    fps: float = 4.0,
    max_frames: int = 20,
) -> List[dict]:
    """Return a dense timestamped sequence around suspected contact events."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "probe.mp4")
        with open(src, "wb") as handle:
            handle.write(video_bytes)
        duration = probe_video_path_duration(src)
    timestamps = focused_sample_timestamps(
        duration,
        event_times,
        window_seconds=window_seconds,
        fps=fps,
        max_frames=max_frames,
    )
    return _extract_at_timestamps(video_bytes, timestamps)


def extract_frames(video_bytes: bytes, fps: int = 1, max_frames: int = 6) -> List[str]:
    """Backward-compatible base64-only wrapper for full-clip samples."""
    return [
        frame["image_b64"]
        for frame in extract_timestamped_frames(video_bytes, fps=fps, max_frames=max_frames)
    ]
