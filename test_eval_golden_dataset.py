import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from eval_golden_dataset import GoldenClip, result_from_label, score_clip, upload_clip


class BlindGoldenEvalTests(unittest.TestCase):
    def test_uncertain_is_not_scored_as_a_foul(self):
        self.assertEqual(result_from_label("Uncertain"), "UNCERTAIN")

    def test_score_prefers_stored_uncertain_result(self):
        clip = GoldenClip(
            clip_id="001",
            filename="positive.mov",
            expected_label="holding",
            expected_result="FOUL",
            rule_reference="NFHS 9-2-1",
            notes="",
            video_path=Path("positive.mov"),
        )
        scored = score_clip(clip, {
            "id": 1,
            "status": "done",
            "result": "UNCERTAIN",
            "prediction_label": "holding",
            "confidence": 0.5,
        })
        self.assertEqual(scored["predicted_result"], "UNCERTAIN")
        self.assertFalse(scored["result_correct"])

    @patch("eval_golden_dataset.http_json", return_value={"id": 42})
    def test_blind_upload_withholds_expected_answer(self, mocked_http):
        with NamedTemporaryFile(suffix=".mov") as video:
            video.write(b"fake video bytes")
            video.flush()
            clip = GoldenClip(
                clip_id="001",
                filename="positive.mov",
                expected_label="pass_interference_defense",
                expected_result="FOUL",
                rule_reference="NFHS 7-5-10",
                notes="Defender restricts the receiver.",
                video_path=Path(video.name),
            )

            upload_clip("http://localhost", "test-key", clip, "run-1", blind=True)

        body = mocked_http.call_args.kwargs["body"]
        self.assertIn(b"Blind evaluation", body)
        self.assertNotIn(b"pass_interference_defense", body)
        self.assertNotIn(b"Defender restricts the receiver", body)
        self.assertNotIn(b"Expected: FOUL", body)


if __name__ == "__main__":
    unittest.main()
