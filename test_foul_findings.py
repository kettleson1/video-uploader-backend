import unittest
from pathlib import Path

from foul_findings import normalize_findings_payload


class NormalizeFindingsTests(unittest.TestCase):
    def test_preserves_multiple_independent_fouls(self):
        result = normalize_findings_payload({
            "confidence": 0.9,
            "findings": [
                {"disposition": "FOUL", "foul_label": "holding", "confidence": 0.91,
                 "committing_team": "offense", "event_time": 4.2},
                {"disposition": "FOUL", "foul_label": "face_mask", "confidence": 0.82,
                 "committing_team": "defense", "player_number": "24", "event_time": 9.1},
            ],
        }, "two contacts", 0.6, 12.0)

        self.assertEqual(result["result"], "FOUL")
        self.assertEqual(result["label"], "holding")
        self.assertEqual(len(result["findings"]), 2)

    def test_no_foul_is_empty_findings_plus_play_conclusion(self):
        result = normalize_findings_payload({
            "confidence": 0.88,
            "no_foul_reason": "Contact was incidental.",
            "findings": [{"disposition": "NO FOUL", "foul_label": "None"}],
        }, "legal contact", 0.6, 8.0)

        self.assertEqual(result["result"], "NO FOUL")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["no_foul_reason"], "Contact was incidental.")

    def test_low_confidence_foul_becomes_uncertain_finding(self):
        result = normalize_findings_payload({
            "findings": [{
                "disposition": "FOUL", "foul_label": "targeting",
                "confidence": 0.4, "event_time": 30,
            }],
        }, "angle blocked", 0.6, 10.0)

        self.assertEqual(result["result"], "UNCERTAIN")
        self.assertEqual(result["findings"][0]["disposition"], "UNCERTAIN")
        self.assertEqual(result["findings"][0]["event_time"], 10.0)

    def test_deduplicates_same_finding(self):
        finding = {
            "disposition": "FOUL", "foul_label": "holding", "confidence": 0.9,
            "committing_team": "offense", "player_number": "72", "event_time": 3,
        }
        result = normalize_findings_payload(
            {"findings": [finding, finding]}, "restriction", 0.6, 8.0
        )

        self.assertEqual(len(result["findings"]), 1)

    def test_review_and_manual_finding_routes_are_exposed(self):
        source = Path(__file__).with_name("main.py").read_text()
        self.assertIn('@app.patch("/api/play-findings/{finding_id}/review")', source)
        self.assertIn('@app.post("/api/plays/{play_id}/findings")', source)


if __name__ == "__main__":
    unittest.main()
