import ast
import unittest
from pathlib import Path


MODEL_TREE = ast.parse(Path(__file__).with_name("models.py").read_text())


def model_columns(class_name):
    for node in MODEL_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.targets[0].id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Call)
                and getattr(statement.value.func, "id", None) == "Column"
            }
    raise AssertionError(f"Model {class_name} was not found")


class FullGameModelContractTests(unittest.TestCase):
    def test_single_play_upload_model_remains_available(self):
        columns = model_columns("Upload")
        self.assertTrue({"foul_type", "prediction_label", "human_label"} <= columns)

    def test_game_and_play_have_separate_responsibilities(self):
        self.assertIn("s3_url", model_columns("Game"))
        self.assertTrue({
            "legacy_upload_id", "source_start_time",
            "detection_confidence", "analysis_status", "snap_time",
            "live_ball_end_time", "boundary_flags",
        } <= model_columns("Play"))

    def test_findings_support_multiple_rulings_per_play(self):
        self.assertTrue({
            "play_id", "foul_label", "rule_citations", "review_status",
        } <= model_columns("PlayFinding"))

    def test_processing_jobs_track_retry_state(self):
        self.assertTrue({
            "stage", "attempts", "max_attempts", "available_at",
        } <= model_columns("ProcessingJob"))


if __name__ == "__main__":
    unittest.main()
