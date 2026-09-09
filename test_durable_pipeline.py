import ast
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).parent


def function_source(path, function_name):
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"Function {function_name} not found")


class DurablePipelineContractTests(unittest.TestCase):
    def test_full_game_routes_enqueue_without_create_task(self):
        main_path = BACKEND_DIR / "main.py"
        for function_name in ("upload_game_video", "analyze_game"):
            source = function_source(main_path, function_name)
            self.assertIn("enqueue_job", source)
            self.assertNotIn("create_task", source)

    def test_worker_registers_every_pipeline_stage(self):
        source = (BACKEND_DIR / "full_game_worker.py").read_text()
        for stage in (
            "validate_upload",
            "detect_plays",
            "extract_clips",
            "screen_play",
            "retrieve_rules",
            "analyze_foul",
            "store_findings",
            "finalize_game",
        ):
            self.assertIn(f'"{stage}"', source)

    def test_queue_claim_uses_skip_locked(self):
        source = (BACKEND_DIR / "job_queue.py").read_text()
        self.assertIn("FOR UPDATE SKIP LOCKED", source)
        self.assertIn("on_conflict_do_nothing", source)
        self.assertIn("heartbeat_job", source)

    def test_timeline_edit_routes_are_selective(self):
        source = (BACKEND_DIR / "main.py").read_text()
        for route in (
            '@app.patch("/api/games/{game_id}/plays/{play_id}/boundary")',
            '@app.post("/api/games/{game_id}/plays/{play_id}/split")',
            '@app.post("/api/games/{game_id}/plays/{play_id}/merge")',
            '@app.post("/api/games/{game_id}/plays/{play_id}/reanalyze")',
            '@app.delete("/api/games/{game_id}/plays/{play_id}")',
        ):
            self.assertIn(route, source)
        self.assertIn("analysis_revision", source)

    def test_shadow_migration_controls_remain_reversible(self):
        source = (BACKEND_DIR / "main.py").read_text()
        worker = (BACKEND_DIR / "full_game_worker.py").read_text()
        self.assertIn('os.getenv("PLAY_DETECTOR_MODE", "shadow_compare")', source)
        self.assertIn("PLAY_FIRST_ACCEPTED", source)
        self.assertIn("_detect_play_segments_fixed_window", source)
        self.assertIn("SegmentationComparison", worker)
        self.assertIn('@app.get("/api/migration/readiness")', source)


if __name__ == "__main__":
    unittest.main()
