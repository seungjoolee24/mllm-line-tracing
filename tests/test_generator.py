"""Geometry, raster, and dataset contract checks for the easy tracing baseline."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageColor

from line_tracing.audit import audit_dataset
from line_tracing.dataset import generate_dataset
from line_tracing.geometry import FAMILIES, GeneratorConfig, make_sample, validate_sample
from line_tracing.render import render_masks, render_sample


REPOSITORY = Path(__file__).resolve().parents[1]


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    projection = 0.0 if denominator == 0 else max(
        0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator)
    )
    return math.hypot(point[0] - start[0] - projection * dx,
                      point[1] - start[1] - projection * dy)


def component_count(image):
    """Count foreground components with 8-connectivity, independently of geometry."""
    width, height = image.size
    foreground = {index for index, value in enumerate(image.getdata()) if value}
    components = 0
    while foreground:
        components += 1
        stack = [foreground.pop()]
        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    neighbor = yy * width + xx
                    if 0 <= xx < width and 0 <= yy < height and neighbor in foreground:
                        foreground.remove(neighbor)
                        stack.append(neighbor)
    return components


class GeometryTests(unittest.TestCase):
    def test_every_family_preserves_ground_truth_and_clearance(self):
        for family in FAMILIES:
            for seed in (0, 7, 42):
                with self.subTest(family=family, seed=seed):
                    config = GeneratorConfig(family=family)
                    sample = make_sample(config, seed=seed)
                    validate_sample(sample)
                    points = sample["path"]["points_px"]
                    objects = sorted(sample["objects"], key=lambda item: item["sequence_index"])
                    self.assertEqual(len(objects), config.num_objects)
                    self.assertEqual(len({obj["color"] for obj in objects}), config.num_objects)
                    self.assertEqual(sample["answer_colors"], [obj["color"] for obj in objects])
                    self.assertEqual(sample["answer"], " ".join(sample["answer_colors"] + ["end"]))
                    self.assertTrue(all(a["arc_length_px"] < b["arc_length_px"]
                                        for a, b in zip(objects, objects[1:])))
                    measured_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
                    self.assertAlmostEqual(measured_length, sample["path"]["length_px"], places=4)
                    self.assertAlmostEqual(measured_length, config.path_length or 0.75 * config.image_size, delta=0.05)
                    self.assertLessEqual(max(math.dist(a, b) for a, b in zip(points, points[1:])), 2.01)
                    for index, obj in enumerate(objects):
                        center = obj["center_px"]
                        on_path = min(point_segment_distance(center, a, b)
                                      for a, b in zip(points, points[1:]))
                        self.assertLess(on_path, 1e-4)
                        self.assertAlmostEqual(math.hypot(*obj["tangent"]), 1.0, places=5)
                        bbox = obj["bbox_px"]
                        self.assertGreaterEqual(min(bbox), 0)
                        self.assertLessEqual(max(bbox), config.image_size)
                        self.assertEqual(obj["predecessor_id"], objects[index - 1]["id"] if index else None)
                        self.assertEqual(obj["successor_id"], objects[index + 1]["id"] if index + 1 < len(objects) else None)
                    for index, obj in enumerate(objects):
                        for other in objects[index + 1:]:
                            self.assertGreater(math.dist(obj["center_px"], other["center_px"]),
                                               obj["radius_px"] + other["radius_px"])

    def test_seed_reproduces_geometry_and_rendering(self):
        config = GeneratorConfig(family="swirl")
        first = make_sample(config, seed=91)
        second = make_sample(config, seed=91)
        self.assertEqual(first, second)
        self.assertEqual(render_sample(first).tobytes(), render_sample(second).tobytes())
        self.assertNotEqual(first["answer_colors"], make_sample(config, seed=92)["answer_colors"])

    def test_explicit_direction_reverses_the_same_centerline(self):
        forward = make_sample(GeneratorConfig(family="curve", reverse=False), seed=17)
        backward = make_sample(GeneratorConfig(family="curve", reverse=True), seed=17)
        for first, second in zip(forward["path"]["points_px"], reversed(backward["path"]["points_px"])):
            self.assertLess(math.dist(first, second), 1e-6)
        self.assertEqual(forward["answer_colors"], list(reversed(backward["answer_colors"])))

    def test_rejects_invalid_or_impossible_configuration(self):
        bad_options = (
            {"family": "unknown"}, {"num_objects": 0}, {"num_objects": 9},
            {"image_size": 0}, {"line_width": 0}, {"object_radius": 0},
            {"path_length": float("nan")}, {"rotation_degrees": float("inf")},
            {"swirl_turns": -1}, {"object_radius": 300}, {"path_length": 40},
        )
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_sample(GeneratorConfig(**options), seed=0)

    def test_validator_rejects_corrupted_annotation(self):
        original = make_sample(GeneratorConfig(), seed=5)
        moved = copy.deepcopy(original)
        moved["objects"][0]["center_px"][0] += 50
        overlapping = copy.deepcopy(original)
        overlapping["objects"][1]["center_px"] = overlapping["objects"][0]["center_px"][:]
        wrong_answer = copy.deepcopy(original)
        wrong_answer["answer_colors"] = list(reversed(wrong_answer["answer_colors"]))
        for name, sample in (("off_path", moved), ("overlap", overlapping), ("answer", wrong_answer)):
            with self.subTest(corruption=name), self.assertRaises(ValueError):
                validate_sample(sample)

    def test_explicit_object_spacing_is_independent_of_path_length(self):
        samples = []
        for length in (480, 520):
            sample = make_sample(GeneratorConfig(
                family="curve", num_objects=8, path_length=length,
                object_spacing=56, rotation_degrees=0, reverse=False,
            ), seed=17)
            arcs = [obj["arc_length_px"] for obj in sorted(
                sample["objects"], key=lambda item: item["sequence_index"]
            )]
            self.assertAlmostEqual(sample["path"]["length_px"], length, places=5)
            self.assertAlmostEqual(arcs[-1] - arcs[0], 392, places=5)
            self.assertAlmostEqual(arcs[0], (length - 392) / 2, places=5)
            self.assertAlmostEqual(length - arcs[-1], arcs[0], places=5)
            self.assertLessEqual(arcs[0], 1.25 * 56)
            for left, right in zip(arcs, arcs[1:]):
                self.assertAlmostEqual(right - left, 56, places=5)
            samples.append(arcs)
        self.assertAlmostEqual(samples[1][0] - samples[0][0], 20, places=5)

    def test_position_jitter_is_seeded_and_bounded_in_pixels(self):
        config = GeneratorConfig(
            family="curve", num_objects=8, path_length=480,
            object_spacing=56, position_jitter=2, rotation_degrees=0, reverse=False,
        )
        sample = make_sample(config, seed=31)
        self.assertEqual(sample, make_sample(config, seed=31))
        offsets = [obj["arc_length_px"] - (44 + 56 * index)
                   for index, obj in enumerate(sorted(
                       sample["objects"], key=lambda item: item["sequence_index"]
                   ))]
        self.assertLessEqual(max(abs(offset) for offset in offsets), 2 + 1e-7)
        self.assertTrue(any(abs(offset) > 1e-7 for offset in offsets))
        boundary_sample = make_sample(GeneratorConfig(
            family="curve", num_objects=8, path_length=520,
            object_spacing=56, position_jitter=6, rotation_degrees=0, reverse=False,
        ), seed=31)
        boundary_arcs = sorted(obj["arc_length_px"] for obj in boundary_sample["objects"])
        self.assertLessEqual(max(boundary_arcs[0], 520 - boundary_arcs[-1]), 70 + 1e-7)

    def test_impossible_spacing_and_negative_jitter_are_explicit_errors(self):
        for options in ({"object_spacing": 200}, {"object_spacing": 20},
                        {"path_length": 480, "object_spacing": 40},
                        {"path_length": 520, "object_spacing": 56, "position_jitter": 7},
                        {"object_spacing": -1}, {"position_jitter": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_sample(GeneratorConfig(**options), seed=0)


class RasterTests(unittest.TestCase):
    def test_clean_image_colors_and_single_connected_trace(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                sample = make_sample(GeneratorConfig(family=family), seed=24)
                image = render_sample(sample)
                path_mask, objects_mask = render_masks(sample)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (640, 640))
                self.assertEqual(path_mask.mode, "L")
                self.assertEqual(objects_mask.mode, "L")
                self.assertEqual(set(path_mask.getdata()), {0, 255})
                self.assertEqual(component_count(path_mask), 1)
                self.assertEqual(set(objects_mask.getdata()), set(range(9)))
                for obj in sample["objects"]:
                    center = tuple(round(value) for value in obj["center_px"])
                    self.assertEqual(image.getpixel(center), ImageColor.getrgb(obj["hex"]))
                    self.assertEqual(objects_mask.getpixel(center), obj["sequence_index"] + 1)


class DatasetTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "line_tracing", *map(str, arguments)],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
        )

    def test_cli_dataset_has_all_artifacts_and_is_extendably_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, large = root / "small", root / "large"
            for output, count in ((small, 1), (large, 4)):
                result = self.run_cli("--output", output, "--count", count,
                                      "--family", "mixed", "--seed", 123, "--preview",
                                      "--object-spacing", 56, "--path-length", 480, "--position-jitter", 0)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((output / "manifest.json").is_file())
                self.assertTrue((output / "preview.png").is_file())
                samples = [json.loads(line) for line in (output / "annotations.jsonl").read_text().splitlines()]
                self.assertEqual(len(samples), count)
                for sample in samples:
                    arcs = sorted(obj["arc_length_px"] for obj in sample["objects"])
                    for left, right in zip(arcs, arcs[1:]):
                        self.assertAlmostEqual(right - left, 56, places=5)
                for index in range(count):
                    name = f"sample_{index:06d}"
                    for artifact in (f"images/{name}.png", f"masks/{name}_path.png", f"masks/{name}_objects.png"):
                        with Image.open(output / artifact) as image:
                            self.assertEqual(image.size, (640, 640))
            first_small = (small / "annotations.jsonl").read_text().splitlines()[0]
            first_large = (large / "annotations.jsonl").read_text().splitlines()[0]
            self.assertEqual(json.loads(first_small), json.loads(first_large))
            self.assertEqual((small / "images/sample_000000.png").read_bytes(),
                             (large / "images/sample_000000.png").read_bytes())

    def test_existing_output_and_failed_generation_leave_data_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep me")
            result = self.run_cli("--output", output, "--count", 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "keep me")
            self.assertEqual(list(output.iterdir()), [sentinel])
            impossible = Path(directory) / "impossible"
            result = self.run_cli("--output", impossible, "--count", 1, "--num-objects", 9)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(impossible.exists())


class AuditTests(unittest.TestCase):
    def test_saved_dataset_passes_then_png_tampering_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            generate_dataset(output, count=2, seed=12, family="mixed")
            report = audit_dataset(output)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["count"], 2)
            self.assertEqual(report["checks"]["file_sha256"], 6)
            (output / "images/sample_000000.png").write_bytes(b"corrupted PNG bytes")
            report = audit_dataset(output)
            self.assertFalse(report["valid"])
            self.assertTrue(any("SHA-256 mismatch" in error for error in report["errors"]))

    def test_audit_rejects_wrong_answer_and_duplicate_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            generate_dataset(output, count=2, seed=12, family="mixed")
            annotation_path = output / "annotations.jsonl"
            original = [json.loads(line) for line in annotation_path.read_text().splitlines()]
            bad_answer = copy.deepcopy(original)
            bad_answer[0]["answer_colors"] = list(reversed(bad_answer[0]["answer_colors"]))
            duplicate_id = copy.deepcopy(original)
            duplicate_id[1]["id"] = duplicate_id[0]["id"]
            for description, records, expected_error in (
                ("wrong answer", bad_answer, "answer"),
                ("duplicate identity", duplicate_id, "duplicate sample ID"),
            ):
                with self.subTest(corruption=description):
                    annotation_path.write_text("".join(json.dumps(record) + "\n" for record in records))
                    report = audit_dataset(output)
                    self.assertFalse(report["valid"])
                    self.assertTrue(any(expected_error in error for error in report["errors"]), report["errors"])

    def test_audit_cli_missing_dataset_returns_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            result = subprocess.run(
                [sys.executable, "-m", "line_tracing.audit", str(missing)],
                cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["valid"])
            self.assertTrue(report["errors"])
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
