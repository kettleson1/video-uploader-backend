"""Pure utilities for comparing legacy and play-first segmentation candidates."""

from typing import List, Sequence


def interval_iou(left: dict, right: dict) -> float:
    start = max(float(left["start_time"]), float(right["start_time"]))
    end = min(float(left["end_time"]), float(right["end_time"]))
    intersection = max(0.0, end - start)
    union = max(
        float(left["end_time"]), float(right["end_time"])
    ) - min(float(left["start_time"]), float(right["start_time"]))
    return intersection / union if union > 0 else 0.0


def compare_segmentations(
    legacy: Sequence[dict],
    play_first: Sequence[dict],
    match_iou: float = 0.30,
) -> dict:
    """Return one-to-one overlap metrics without treating either input as truth."""
    possible = []
    legacy_overlap_counts = [0 for _ in legacy]
    for new_index, new_candidate in enumerate(play_first):
        for legacy_index, legacy_candidate in enumerate(legacy):
            score = interval_iou(new_candidate, legacy_candidate)
            if score >= match_iou:
                legacy_overlap_counts[legacy_index] += 1
                possible.append((score, new_index, legacy_index))

    matched_new = set()
    matched_legacy = set()
    matched_scores: List[float] = []
    for score, new_index, legacy_index in sorted(possible, reverse=True):
        if new_index in matched_new or legacy_index in matched_legacy:
            continue
        matched_new.add(new_index)
        matched_legacy.add(legacy_index)
        matched_scores.append(score)

    return {
        "legacy_count": len(legacy),
        "play_first_count": len(play_first),
        "count_delta": len(play_first) - len(legacy),
        "matched_count": len(matched_scores),
        "unmatched_legacy_count": len(legacy) - len(matched_legacy),
        "unmatched_play_first_count": len(play_first) - len(matched_new),
        "possible_duplicate_play_first_count": sum(
            max(0, count - 1) for count in legacy_overlap_counts
        ),
        "mean_matched_iou": (
            round(sum(matched_scores) / len(matched_scores), 4)
            if matched_scores else 0.0
        ),
        "match_iou_threshold": match_iou,
    }
