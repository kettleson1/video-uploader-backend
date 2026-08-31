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


if __name__ == "__main__":
    unittest.main()
