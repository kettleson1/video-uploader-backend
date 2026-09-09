import unittest

from coaches_film_detector import (
    ActivitySample,
    detect_play_boundaries,
    find_uncovered_intervals,
)
from segmentation_comparison import compare_segmentations, interval_iou


def samples(scores, interval=0.5):
    return [
        ActivitySample(time=(index + 1) * interval, score=score)
        for index, score in enumerate(scores)
    ]


class DetectPlayBoundariesTests(unittest.TestCase):
    def test_detects_and_numbers_separate_activity_runs(self):
        scores = [0.5] * 20 + [12.0] * 12 + [0.5] * 30 + [10.0] * 10 + [0.5] * 10
        result = detect_play_boundaries(samples(scores), video_duration=41.0)

        self.assertEqual([play["play_number"] for play in result], [1, 2])
        self.assertEqual(result[0]["start_time"], 2.5)
        self.assertEqual(result[-1]["end_time"], 41.0)

    def test_keeps_isolated_camera_change_as_uncertain_candidate(self):
        scores = [0.5] * 10 + [20.0] + [0.5] * 20
        result = detect_play_boundaries(samples(scores), video_duration=15.5)

        self.assertEqual(len(result), 1)
        self.assertIn("short_activity_burst", result[0]["boundary_flags"])
        self.assertLess(result[0]["segment_confidence"], 0.6)

    def test_returns_no_candidates_for_static_video(self):
        result = detect_play_boundaries(samples([0.5] * 40), video_duration=20.0)

        self.assertEqual(result, [])

    def test_clamps_padding_to_video_bounds(self):
        scores = [10.0] * 8 + [0.5] * 20
        result = detect_play_boundaries(samples(scores), video_duration=14.0)

        self.assertEqual(result[0]["start_time"], 0.0)
        self.assertEqual(result[0]["end_time"], 14.0)

    def test_caps_candidates_by_merging_nearest_neighbors(self):
        scores = []
        for _ in range(5):
            scores.extend([0.5] * 16)
            scores.extend([10.0] * 6)
        result = detect_play_boundaries(
            samples(scores),
            video_duration=55.0,
            merge_inactive_seconds=2.0,
            maximum_plays=3,
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(any(
            "candidate_limit_merge" in play["boundary_flags"] for play in result
        ))

    def test_reports_every_uncovered_timeline_interval(self):
        gaps = find_uncovered_intervals([
            {"start_time": 5.0, "end_time": 12.0},
            {"start_time": 15.0, "end_time": 18.0},
        ], video_duration=20.0)

        self.assertEqual(gaps, [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 12.0, "end_time": 15.0},
            {"start_time": 18.0, "end_time": 20.0},
        ])

    def test_compares_shadow_candidates_without_calling_either_ground_truth(self):
        legacy = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 15.0, "end_time": 25.0},
        ]
        play_first = [
            {"start_time": 1.0, "end_time": 9.0},
            {"start_time": 16.0, "end_time": 20.0},
            {"start_time": 21.0, "end_time": 24.0},
            {"start_time": 30.0, "end_time": 35.0},
        ]

        result = compare_segmentations(legacy, play_first, match_iou=0.25)

        self.assertAlmostEqual(interval_iou(legacy[0], play_first[0]), 0.8)
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["unmatched_play_first_count"], 2)
        self.assertEqual(result["possible_duplicate_play_first_count"], 1)
        self.assertNotIn("missed_play_count", result)


if __name__ == "__main__":
    unittest.main()
