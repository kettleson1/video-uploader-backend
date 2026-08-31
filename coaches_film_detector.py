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
    active_region_ratio: float = 0.0
    spatial_spread: float = 0.0


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
            "-vf", f"fps={sample_fps},scale={width}:{height}:flags=area,setsar=1,format=gray",
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
        grid_columns = 10
        grid_rows = 6
        region_count = grid_columns * grid_rows
        while len(previous) == frame_size:
            current = process.stdout.read(frame_size)
            if len(current) != frame_size:
                break
            region_sums = [0.0] * region_count
            region_pixels = [0] * region_count
            difference_sum = 0.0
            for index in range(0, frame_size, pixel_step):
                pixel_difference = abs(current[index] - previous[index])
                difference_sum += pixel_difference
                y, x = divmod(index, width)
                region_x = min(grid_columns - 1, x * grid_columns // width)
                region_y = min(grid_rows - 1, y * grid_rows // height)
                region_index = region_y * grid_columns + region_x
                region_sums[region_index] += pixel_difference
                region_pixels[region_index] += 1
            region_scores = [
                total / count if count else 0.0
                for total, count in zip(region_sums, region_pixels)
            ]
            active_regions = [score >= 6.0 for score in region_scores]
            active_region_ratio = sum(active_regions) / region_count
            occupied_quadrants = set()
            for region_index, active in enumerate(active_regions):
                if not active:
                    continue
                region_y, region_x = divmod(region_index, grid_columns)
                occupied_quadrants.add((region_x >= grid_columns / 2, region_y >= grid_rows / 2))
            samples.append(ActivitySample(
                time=frame_index / sample_fps,
                score=difference_sum / compared_pixels,
                active_region_ratio=active_region_ratio,
                spatial_spread=len(occupied_quadrants) / 4.0,
            ))
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
    maximum_plays: int = 200,
) -> List[dict]:
    """Convert set-to-motion activity runs into editable play candidates."""
    if video_duration <= 0 or len(samples) < 2:
        return []

    scores = [max(0.0, sample.score) for sample in samples]
    baseline = statistics.median(scores)
    deviations = [abs(score - baseline) for score in scores]
    mad = statistics.median(deviations)
    threshold = max(2.0, baseline + max(1.5, 3.0 * mad))

    active_samples = [sample for sample in samples if sample.score >= threshold]
    if not active_samples:
        return []

    sample_interval = statistics.median(
        max(0.01, right.time - left.time)
        for left, right in zip(samples, samples[1:])
    )
    runs: List[list[ActivitySample]] = []
    run = [active_samples[0]]
    for active_sample in active_samples[1:]:
        if active_sample.time - run[-1].time <= merge_inactive_seconds + sample_interval:
            run.append(active_sample)
        else:
            runs.append(run)
            run = [active_sample]
    runs.append(run)

    candidates = []
    for run in runs:
        active_start = run[0].time
        active_end = run[-1].time + sample_interval
        flags = []
        preceding = [
            sample for sample in samples
            if active_start - 3.0 <= sample.time < active_start
        ]
        set_detected = len(preceding) >= 2 and sum(
            sample.score < threshold for sample in preceding
        ) / len(preceding) >= 0.75
        if not set_detected:
            flags.append("no_clear_set_period")
        if active_end - active_start < minimum_active_seconds:
            flags.append("short_activity_burst")
        mean_region_ratio = statistics.mean(sample.active_region_ratio for sample in run)
        if mean_region_ratio >= 0.85:
            flags.append("possible_camera_motion")
        mean_spread = statistics.mean(sample.spatial_spread for sample in run)
        if mean_spread < 0.25 and any(sample.spatial_spread for sample in run):
            flags.append("localized_motion_only")
        confidence = 0.78 if set_detected else 0.58
        confidence -= 0.15 * len(flags)
        if "short_activity_burst" in flags:
            confidence = min(confidence, 0.45)
        if "possible_camera_motion" in flags:
            confidence = min(confidence, 0.40)
        candidates.append({
            "start_time": round(max(0.0, active_start - pre_roll), 2),
            "end_time": round(min(video_duration, active_end + post_roll), 2),
            "snap_time": round(active_start, 2),
            "live_ball_end_time": round(active_end, 2),
            "segment_confidence": round(max(0.2, confidence), 2),
            "boundary_flags": flags,
        })

    while len(candidates) > maximum_plays:
        merge_index = min(
            range(len(candidates) - 1),
            key=lambda index: candidates[index + 1]["start_time"] - candidates[index]["end_time"],
        )
        current = candidates[merge_index]
        following = candidates.pop(merge_index + 1)
        current["end_time"] = following["end_time"]
        current["live_ball_end_time"] = following["live_ball_end_time"]
        current["segment_confidence"] = min(current["segment_confidence"], 0.3)
        current["boundary_flags"] = sorted(set(
            current["boundary_flags"] + following["boundary_flags"] + ["candidate_limit_merge"]
        ))

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


def find_uncovered_intervals(
    candidates: Sequence[dict],
    video_duration: float,
    minimum_gap_seconds: float = 1.0,
) -> List[dict]:
    """Make every source interval omitted from play clips explicit for review."""
    gaps = []
    cursor = 0.0
    for candidate in sorted(candidates, key=lambda item: item["start_time"]):
        start = max(0.0, float(candidate["start_time"]))
        if start - cursor >= minimum_gap_seconds:
            gaps.append({"start_time": round(cursor, 2), "end_time": round(start, 2)})
        cursor = max(cursor, min(video_duration, float(candidate["end_time"])))
    if video_duration - cursor >= minimum_gap_seconds:
        gaps.append({"start_time": round(cursor, 2), "end_time": round(video_duration, 2)})
    return gaps
