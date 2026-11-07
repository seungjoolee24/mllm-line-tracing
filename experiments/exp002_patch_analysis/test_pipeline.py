"""Offline integrity/resume checks. All calculated stage data are synthetic."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline


REPO = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.run = pipeline.AnalysisRun(REPO, self.artifacts, self.root / "out", self.root / "cache")

    def tearDown(self):
        self.temporary.cleanup()

    def stage(self, key="synthetic"):
        folder = self.run.output / key
        def compute(staging):
            (staging / "result.txt").write_text("synthetic-stage\n")
            return {"synthetic": True}
        return self.run._stage(key, folder, compute)

    def test_resume_requires_verified_inventory_and_checksums(self):
        self.stage()
        def forbidden():
            raise AssertionError("A completed stage must be reused")
        self.assertEqual(self.run._stage("synthetic", self.run.output / "synthetic", forbidden), {"synthetic": True})
        (self.run.output / "synthetic/result.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run._stage("synthetic", self.run.output / "synthetic", forbidden)

    def test_extra_stage_file_is_not_silently_accepted(self):
        self.stage()
        (self.run.output / "synthetic/extra.txt").write_text("untracked")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.run._stage("synthetic", self.run.output / "synthetic", lambda: {})

    def test_interrupted_copy_is_resumed_without_overwriting_different_data(self):
        folder = self.run.output / "interrupted"
        folder.mkdir()
        (folder / "one.txt").write_text("same bytes")
        def compute(staging):
            (staging / "one.txt").write_text("same bytes")
            (staging / "two.txt").write_text("new bytes")
            return {"path": str(staging / "two.txt")}
        result = self.run._stage("interrupted", folder, compute)
        self.assertEqual(result["path"], str(folder / "two.txt"))
        self.assertEqual((folder / "one.txt").read_text(), "same bytes")
        other = self.run.output / "conflicting"
        other.mkdir()
        (other / "one.txt").write_text("different bytes")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run._stage("conflicting", other, compute)

    def test_interrupted_local_calculation_is_restarted_cleanly(self):
        def fail(staging):
            (staging / "half.txt").write_text("unfinished")
            raise RuntimeError("simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run._stage("synthetic", self.run.output / "synthetic", fail)
        self.assertFalse((self.run.output / "synthetic").exists())
        self.stage()
        self.assertFalse((self.run.output / "synthetic/half.txt").exists())

    def test_configuration_and_environment_must_match_for_resume(self):
        pipeline.AnalysisRun(REPO, self.artifacts, self.root / "out", self.root / "cache")
        with self.assertRaisesRegex(ValueError, "different"):
            pipeline.AnalysisRun(REPO, self.artifacts, self.root / "out", self.root / "cache", {"seed": 1})
        with patch.object(pipeline, "_environment", return_value={"python": "different"}):
            with self.assertRaisesRegex(ValueError, "different"):
                pipeline.AnalysisRun(REPO, self.artifacts, self.root / "out", self.root / "cache")

    def test_cache_and_output_cannot_overlap_inputs(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            pipeline.AnalysisRun(REPO, self.artifacts, self.artifacts / "out", self.root / "cache2")
        with self.assertRaisesRegex(ValueError, "separate"):
            pipeline.AnalysisRun(REPO, self.artifacts, self.root / "out2", self.artifacts / "cache")

    def prepare_synthetic_completion(self):
        self.run.cases = {case_id: SimpleNamespace(meta={"generation": {"token_details": {
            "color_spans": [{"color": "red"}]}}}) for case_id in pipeline.CASES}
        inputs = self.run.output / "inputs"
        inputs.mkdir()
        source = inputs / "synthetic.txt"
        source.write_text("synthetic fixture, not a capture\n")
        pipeline.atomic_json(self.run.output / "input_manifest.json", {
            "analysis_fingerprint": self.run.fingerprint,
            "files": {"inputs/synthetic.txt": {"bytes": source.stat().st_size,
                                                   "sha256": pipeline.sha256(source)}}})
        for case_id in pipeline.CASES:
            for stage in ("pca", "attention", "continuity"):
                self.stage(case_id + "_" + stage)
            for mode in ("raw", "conditional"):
                self.stage(f"{case_id}_word_00_{mode}")

    def test_finalization_rejects_missing_stage_and_input_tampering(self):
        self.prepare_synthetic_completion()
        receipt = self.run.output / "stage_receipts" / (pipeline.CASES[0] + "_pca.json")
        original = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaisesRegex(RuntimeError, "Incomplete stage"):
            self.run.finalize()
        receipt.write_bytes(original)
        (self.run.output / "inputs/synthetic.txt").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "input verification"):
            self.run.finalize()

    def test_completion_manifest_reconstructs_all_saved_files(self):
        self.prepare_synthetic_completion()
        result = self.run.finalize()
        self.assertEqual(result["status"], "complete")
        manifest = json.loads((self.run.output / "output_manifest.json").read_text())
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        self.assertEqual(manifest["total_bytes"], sum(x["bytes"] for x in manifest["files"].values()))
        for relative, record in manifest["files"].items():
            path = self.run.output / relative
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(pipeline.sha256(path), record["sha256"])
        self.assertEqual(json.loads((self.run.output / "environment.json").read_text()), self.run.identity["environment"])

    def test_incomplete_file_cannot_produce_complete_summary(self):
        self.prepare_synthetic_completion()
        (self.run.output / "orphan.partial").write_text("interrupted")
        with self.assertRaisesRegex(ValueError, "Incomplete files"):
            self.run.finalize()
        self.assertFalse((self.run.output / "summary.json").exists())


class NotebookTests(unittest.TestCase):
    def test_cells_are_unexecuted_valid_python_and_bootstrap_hashes_match(self):
        import nbformat
        notebook_path = Path(__file__).with_name("patch_analysis.ipynb")
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        source_manifest = None
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            tree = ast.parse(cell.source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets):
                    source_manifest = ast.literal_eval(node.value)
        self.assertIsNotNone(source_manifest)
        for relative, record in source_manifest.items():
            data = (REPO / relative).read_bytes()
            import hashlib
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
            self.assertEqual(hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), record["git_blob_sha"])


if __name__ == "__main__":
    unittest.main()
