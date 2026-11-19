import unittest

from video_frames import focused_sample_timestamps, frame_sample_timestamps


class FrameSamplingTests(unittest.TestCase):
    def test_samples_span_the_complete_clip(self):
        timestamps = frame_sample_timestamps(18.0, fps=1, max_frames=6)

        self.assertEqual(len(timestamps), 6)
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertLess(timestamps[0], 2.0)
        self.assertGreater(timestamps[-1], 16.0)

    def test_short_clip_does_not_duplicate_more_than_requested_rate(self):
        timestamps = frame_sample_timestamps(2.2, fps=1, max_frames=8)

        self.assertEqual(len(timestamps), 3)
        self.assertGreater(timestamps[-1], 1.8)

    def test_default_analysis_budget_covers_a_long_play_through_its_end(self):
        timestamps = frame_sample_timestamps(19.0, fps=1, max_frames=12)

        self.assertEqual(len(timestamps), 12)
        self.assertLess(timestamps[0], 1.0)
        self.assertGreater(timestamps[-1], 18.0)

    def test_invalid_limits_return_no_samples(self):
        self.assertEqual(frame_sample_timestamps(10.0, fps=0, max_frames=8), [])
        self.assertEqual(frame_sample_timestamps(10.0, fps=1, max_frames=0), [])

    def test_focused_samples_surround_the_contact_time(self):
        timestamps = focused_sample_timestamps(
            18.0, [10.0], window_seconds=2.0, fps=4.0, max_frames=20
        )

        self.assertEqual(timestamps[0], 8.0)
        self.assertEqual(timestamps[-1], 12.0)
        self.assertIn(10.0, timestamps)
        self.assertGreaterEqual(len(timestamps), 17)

    def test_focused_samples_are_clamped_and_capped(self):
        timestamps = focused_sample_timestamps(
            10.0, [0.5, 9.5], window_seconds=2.0, fps=8.0, max_frames=12
        )

        self.assertEqual(len(timestamps), 12)
        self.assertGreaterEqual(timestamps[0], 0.0)
        self.assertLessEqual(timestamps[-1], 10.0)


if __name__ == "__main__":
    unittest.main()
