"""Offline input preflight tests; fixtures contain no model data or inference.

Only the authoritative-manifest download is mocked. No network or Drive APIs
are used. Missing unselected cases deliberately remain absent from the fixture.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline


SELECTED_CASES = (
    "case_00_curve", "case_01_straight", "case_02_elbow", "case_03_swirl", "case_04_curve",
)
ALL_CASES = SELECTED_CASES + (
    "case_05_straight", "case_06_elbow", "case_07_swirl", "case_08_curve", "case_09_elbow",
)
REPO = Path(__file__).resolve().parents[2]


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _required_names(result):
    """The API may return filenames or the original manifest entry records."""
    return {row["file"] if isinstance(row, dict) else str(row)
            for row in result["required_archives"]}


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.manifest_path = self.artifacts / "artifact_manifest.json"
        self.expected_names = {"run002_metadata.zip"} | {f"run002_{case}.zip" for case in SELECTED_CASES}
        entries = []
        for case in (None, *ALL_CASES):
            name = "run002_metadata.zip" if case is None else f"run002_{case}.zip"
            # Preflight inspects existence and byte size; ZIP SHA/extraction is
            # deliberately deferred to the existing DriveBundle reader.
            data = ("synthetic archive fixture: " + name + "\n").encode()
            entry = {"file": name, "bytes": len(data), "sha256": _digest(data),
                     "kind": "metadata" if case is None else "case", "sample_id": case,
                     "tensor_file_count": 0 if case is None else 206,
                     "source_file_count": 1 if case is None else 209,
                     "uncompressed_bytes": len(data),
                     "archive_path_prefix": "run002/" if case is None else f"run002/samples/{case}/"}
            entries.append(entry)
            if name in self.expected_names:
                (self.artifacts / name).write_bytes(data)
        manifest = {"schema_version": "1.0", "package_protocol": "qwen3vl_trace_case_zip_v1",
                    "collection_schema_version": "1.1", "run_name": "run002", "status": "complete",
                    "case_count": 10, "expected_case_count": 10, "archive_count": 11,
                    "tensor_file_count": 2060, "expected_tensor_file_count": 2060,
                    "source_tensor_bytes": 4253107416,
                    "total_archive_bytes": sum(entry["bytes"] for entry in entries),
                    "archives": entries}
        self.canonical_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        self.expected_digest = _digest(self.canonical_bytes)
        self.patches = [patch.object(pipeline, "CASES", SELECTED_CASES),
                        patch.object(pipeline, "MANIFEST_SHA256", self.expected_digest)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def install_manifest(self):
        self.manifest_path.write_bytes(self.canonical_bytes)

    def mock_download(self, url, target, expected_sha256):
        self.assertEqual(url, pipeline.CANONICAL_MANIFEST_URL)
        self.assertEqual(expected_sha256, self.expected_digest)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.canonical_bytes)
        return target

    def test_missing_manifest_is_distinct_and_does_not_download_without_opt_in(self):
        with patch.object(pipeline, "_download") as download:
            with self.assertRaises(FileNotFoundError) as error:
                pipeline.input_preflight(self.artifacts, recover_manifest=False)
            download.assert_not_called()
        self.assertIn("artifact_manifest.json", str(error.exception))
        self.assertFalse(self.manifest_path.exists())

    def test_existing_wrong_manifest_hash_is_reported_without_replacement(self):
        original = b'{"run_name": "some_other_run"}\n'
        self.manifest_path.write_bytes(original)
        for recover in (False, True):
            with self.subTest(recover_manifest=recover):
                with patch.object(pipeline, "_download") as download:
                    with self.assertRaises(ValueError):
                        pipeline.input_preflight(self.artifacts, recover_manifest=recover)
                    download.assert_not_called()
                self.assertEqual(self.manifest_path.read_bytes(), original)

    def test_only_available_selected_five_and_metadata_are_required(self):
        self.install_manifest()
        with patch.object(pipeline, "_download") as download:
            result = pipeline.input_preflight(self.artifacts)
            download.assert_not_called()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["artifact_dir"], str(self.artifacts.resolve()))
        self.assertEqual(result["manifest_sha256"], self.expected_digest)
        self.assertFalse(result["manifest_recovered"])
        self.assertEqual(result["selected_cases"], list(SELECTED_CASES))
        self.assertEqual(_required_names(result), self.expected_names)
        for case in ALL_CASES[5:]:
            self.assertFalse((self.artifacts / f"run002_{case}.zip").exists())

    def test_recovery_copies_exact_authoritative_bytes_then_validates_selection(self):
        with patch.object(pipeline, "_download", side_effect=self.mock_download) as download:
            result = pipeline.input_preflight(self.artifacts, recover_manifest=True)
            download.assert_called_once()
        self.assertEqual(self.manifest_path.read_bytes(), self.canonical_bytes)
        self.assertEqual(pipeline.sha256(self.manifest_path), self.expected_digest)
        self.assertTrue(result["manifest_recovered"])
        self.assertEqual(_required_names(result), self.expected_names)
        # A verified second call must not fetch or rewrite the manifest.
        with patch.object(pipeline, "_download") as download:
            pipeline.input_preflight(self.artifacts, recover_manifest=True)
            download.assert_not_called()

    def test_missing_selected_archive_is_not_confused_with_missing_manifest(self):
        self.install_manifest()
        missing = self.artifacts / "run002_case_04_curve.zip"
        missing.unlink()
        with self.assertRaises(FileNotFoundError) as error:
            pipeline.input_preflight(self.artifacts)
        self.assertIn(missing.name, str(error.exception))
        self.assertEqual(self.manifest_path.read_bytes(), self.canonical_bytes)

    def test_wrong_archive_size_fails_before_analysis_output_creation(self):
        self.install_manifest()
        archive = self.artifacts / "run002_case_03_swirl.zip"
        archive.write_bytes(archive.read_bytes() + b"truncated_or_extra_byte")
        with self.assertRaises(ValueError) as error:
            pipeline.input_preflight(self.artifacts)
        self.assertIn(archive.name, str(error.exception))
        output = self.root / "analysis_output"
        # Mirror the notebook's explicit preflight-before-construction order.
        with self.assertRaises(ValueError), patch.object(pipeline, "AnalysisRun") as constructor:
            pipeline.input_preflight(self.artifacts)
            pipeline.AnalysisRun(REPO, self.artifacts, output, self.root / "cache",
                                 config={"cases": list(SELECTED_CASES)})
        constructor.assert_not_called()
        self.assertFalse(output.exists(), "input failures must precede analysis identity/source creation")

    def test_recovery_does_not_publish_manifest_if_selected_zip_is_missing(self):
        (self.artifacts / "run002_case_04_curve.zip").unlink()
        with patch.object(pipeline, "_download", side_effect=self.mock_download):
            with self.assertRaises(FileNotFoundError):
                pipeline.input_preflight(self.artifacts, recover_manifest=True)
        self.assertFalse(self.manifest_path.exists())

    def test_declared_selection_and_duplicate_archive_structure_are_checked(self):
        for invalid in ("missing_selected_case", "duplicate_filename"):
            with self.subTest(invalid=invalid):
                manifest = json.loads(self.canonical_bytes)
                if invalid == "missing_selected_case":
                    manifest["archives"] = [row for row in manifest["archives"]
                                            if row["sample_id"] != "case_03_swirl"]
                else:
                    manifest["archives"][2]["file"] = manifest["archives"][1]["file"]
                payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
                self.manifest_path.write_bytes(payload)
                with patch.object(pipeline, "MANIFEST_SHA256", _digest(payload)):
                    with self.assertRaises(ValueError):
                        pipeline.input_preflight(self.artifacts)

    def test_archive_sha_is_deferred_to_verified_bundle_reader(self):
        self.install_manifest()
        archive = self.artifacts / "run002_case_00_curve.zip"
        content = archive.read_bytes()
        archive.write_bytes(b"x" * len(content))
        result = pipeline.input_preflight(self.artifacts)
        self.assertEqual(_required_names(result), self.expected_names)


if __name__ == "__main__":
    unittest.main()
