import unittest

from coaches_film_detector import ActivitySample, detect_play_boundaries


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

    def test_ignores_isolated_camera_change(self):
        scores = [0.5] * 10 + [20.0] + [0.5] * 20
        result = detect_play_boundaries(samples(scores), video_duration=15.5)

        self.assertEqual(result, [])

    def test_returns_no_candidates_for_static_video(self):
        result = detect_play_boundaries(samples([0.5] * 40), video_duration=20.0)

        self.assertEqual(result, [])

    def test_clamps_padding_to_video_bounds(self):
        scores = [10.0] * 8 + [0.5] * 20
        result = detect_play_boundaries(samples(scores), video_duration=14.0)

        self.assertEqual(result[0]["start_time"], 0.0)
        self.assertEqual(result[0]["end_time"], 14.0)


if __name__ == "__main__":
    unittest.main()
