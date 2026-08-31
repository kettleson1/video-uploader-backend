"""Coaches-film play candidate detection.

The detector intentionally has no third-party Python dependencies. ffmpeg samples
small grayscale frames; the pure boundary function is deterministic and testable.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ActivitySample:
    time: float
    score: float


def extract_activity_samples(
    video_bytes: bytes,
    sample_fps: float = 2.0,
    width: int = 160,
    height: int = 90,
) -> List[ActivitySample]:
    """Return frame-difference activity scores sampled across a video."""
    frame_size = width * height
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "coaches-film-input")
        with open(source_path, "wb") as source_file:
            source_file.write(video_bytes)

        command = [
            "ffmpeg", "-v", "error", "-i", source_path,
            "-vf", f"fps={sample_fps},scale={width}:{height},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Could not open coaches-film activity stream")

        samples: List[ActivitySample] = []
        previous = process.stdout.read(frame_size)
        frame_index = 1
        # Every fourth pixel is sufficient at this resolution and bounds CPU cost.
        pixel_step = 4
        compared_pixels = (frame_size + pixel_step - 1) // pixel_step
        while len(previous) == frame_size:
            current = process.stdout.read(frame_size)
            if len(current) != frame_size:
                break
            difference = sum(
                abs(current[index] - previous[index])
                for index in range(0, frame_size, pixel_step)
            ) / compared_pixels
            samples.append(ActivitySample(frame_index / sample_fps, difference))
            previous = current
            frame_index += 1

        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not sample coaches-film activity: {detail[-300:]}")
        return samples


def detect_play_boundaries(
    samples: Sequence[ActivitySample],
    video_duration: float,
    pre_roll: float = 8.0,
    post_roll: float = 10.0,
    merge_inactive_seconds: float = 5.0,
    minimum_active_seconds: float = 2.0,
) -> List[dict]:
    """Convert adaptive activity runs into editable play candidates."""
    if video_duration <= 0 or len(samples) < 2:
        return []

    scores = [max(0.0, sample.score) for sample in samples]
    baseline = statistics.median(scores)
    deviations = [abs(score - baseline) for score in scores]
    mad = statistics.median(deviations)
    threshold = max(2.0, baseline + max(1.5, 3.0 * mad))

    active_times = [sample.time for sample in samples if sample.score >= threshold]
    if not active_times:
        return []

    sample_interval = statistics.median(
        max(0.01, right.time - left.time)
        for left, right in zip(samples, samples[1:])
    )
    runs: List[tuple[float, float]] = []
    run_start = active_times[0]
    run_end = active_times[0]
    for active_time in active_times[1:]:
        if active_time - run_end <= merge_inactive_seconds + sample_interval:
            run_end = active_time
        else:
            runs.append((run_start, run_end + sample_interval))
            run_start = run_end = active_time
    runs.append((run_start, run_end + sample_interval))

    candidates = []
    for active_start, active_end in runs:
        if active_end - active_start < minimum_active_seconds:
            continue
        candidates.append({
            "start_time": round(max(0.0, active_start - pre_roll), 2),
            "end_time": round(min(video_duration, active_end + post_roll), 2),
            "segment_confidence": 0.65,
        })

    # Padding may overlap for no-huddle sequences. Split the overlap at its
    # midpoint so every candidate retains context without duplicate footage.
    for index in range(len(candidates) - 1):
        current = candidates[index]
        following = candidates[index + 1]
        if current["end_time"] > following["start_time"]:
            midpoint = round((current["end_time"] + following["start_time"]) / 2.0, 2)
            current["end_time"] = midpoint
            following["start_time"] = midpoint

    return [
        {"play_number": index, **candidate}
        for index, candidate in enumerate(candidates, start=1)
        if candidate["end_time"] - candidate["start_time"] >= 4.0
    ]
