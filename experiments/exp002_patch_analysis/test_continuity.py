"""CPU checks of geometric controls and full-feature continuity measurements.

Run with ``python -m unittest discover -s experiments/exp002_patch_analysis
-p test_continuity.py -v``. Synthetic fixtures do not need model weights or Drive.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "continuity-test-mpl"))

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuity


def cumulative_length(points):
    points = np.asarray(points, dtype=np.float64)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


class GeometryTests(unittest.TestCase):
    def test_clipping_detects_crossing_without_an_interior_vertex(self):
        actual = continuity.polyline_cell_intervals([[-100, 5], [100, 5]], [0, 0, 10, 10])
        np.testing.assert_allclose(actual, [[100, 110]], atol=1e-10)

    def test_touching_segment_intervals_merge_at_bend(self):
        actual = continuity.polyline_cell_intervals([[-5, 5], [5, 5], [5, 15]], [0, 0, 10, 10])
        np.testing.assert_allclose(actual, [[5, 15]], atol=1e-10)

    def test_reentry_is_two_separated_arc_intervals(self):
        points = [[-5, 5], [15, 5], [15, 15], [-5, 15], [-5, 5], [15, 5]]
        actual = continuity.polyline_cell_intervals(points, [0, 0, 10, 10])
        np.testing.assert_allclose(actual, [[5, 15], [65, 75]], atol=1e-10)

    def test_very_small_outside_gap_is_not_merged(self):
        # Arc intervals touch in screen space but are separated by travel outside the cell.
        epsilon = 1e-10
        points = [[-1, 5], [10 + epsilon, 5], [10 + epsilon, 5 + epsilon], [-1, 5 + epsilon]]
        actual = continuity.polyline_cell_intervals(points, [0, 0, 10, 10])
        self.assertEqual(len(actual), 2)
        self.assertGreater(actual[1][0], actual[0][1])

    def test_interpolation_uses_arclength_across_unequal_segments(self):
        points = np.array([[0, 0], [3, 0], [3, 4]], dtype=float)
        cumulative = cumulative_length(points)
        np.testing.assert_allclose(continuity.interpolate_path(points, 5), [3, 2])
        np.testing.assert_allclose(continuity.interpolate_path(points, 5, cumulative), [3, 2])
        np.testing.assert_allclose(continuity.interpolate_path(points, 0), [0, 0])
        np.testing.assert_allclose(continuity.interpolate_path(points, 7), [3, 4])

    def test_straight_arclength_distance_equals_screen_distance(self):
        points = np.array([[10, 20], [70, 100]], dtype=float)
        positions = np.array([0.0, 11.0, 32.0, 64.0, 100.0])
        xy = np.asarray([continuity.interpolate_path(points, s) for s in positions])
        screen = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        np.testing.assert_allclose(screen, np.abs(positions[:, None] - positions[None, :]), atol=1e-10)


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.arc = np.array([0.0, 32.0, 160.0, 192.0])
        self.xy = np.array([[0, 0], [20, 0], [20, 1], [0, 1]], dtype=float)
        self.coverage = np.full(4, 0.1)

    def test_folded_geometry_yields_screen_matched_near_and_far(self):
        matches = continuity.select_geometry_matches(self.arc, self.xy, self.coverage)
        self.assertEqual(len(matches), 4)
        for match in matches:
            anchor, near, far = (match[k] for k in ("anchor", "near", "far"))
            self.assertEqual(len({anchor, near, far}), 3)
            self.assertLessEqual(abs(self.arc[near] - self.arc[anchor]), 64)
            self.assertGreaterEqual(abs(self.arc[far] - self.arc[anchor]), 128)
            self.assertLessEqual(match["screen_distance_error_px"], 8)
            self.assertLessEqual(match["coverage_error"], 0.03)
            self.assertAlmostEqual(match["near_screen_px"], np.linalg.norm(self.xy[near] - self.xy[anchor]))
            self.assertAlmostEqual(match["far_screen_px"], np.linalg.norm(self.xy[far] - self.xy[anchor]))

    def test_screen_distance_confound_cannot_pass(self):
        xy = np.c_[self.arc, np.zeros(4)]
        self.assertEqual(continuity.select_geometry_matches(self.arc, xy, self.coverage), [])

    def test_coverage_confound_cannot_pass(self):
        coverage = np.array([0.1, 0.1, 0.15, 0.15])
        self.assertEqual(continuity.select_geometry_matches(self.arc, self.xy, coverage), [])

    def test_matching_is_feature_blind_and_deterministic(self):
        self.assertNotIn("H", inspect.signature(continuity.select_geometry_matches).parameters)
        first = continuity.select_geometry_matches(self.arc, self.xy, self.coverage)
        for _ in range(3):
            self.assertEqual(first, continuity.select_geometry_matches(self.arc.copy(), self.xy.copy(), self.coverage.copy()))

    def test_best_screen_match_takes_priority_over_shortest_near_arc(self):
        matches = continuity.select_geometry_matches(
            [0, 32, 48, 160, 192], [[0, 0], [20, 0], [30, 0], [20, 1], [30, 0]], [0.1] * 5
        )
        selected = next(m for m in matches if m["anchor"] == 0)
        self.assertEqual((selected["near"], selected["far"]), (2, 4))

    def test_arc_cutoffs_are_inclusive(self):
        matches = continuity.select_geometry_matches([0, 64, 128], [[0, 0], [20, 0], [20, 0]], [0.1] * 3)
        anchor_zero = [m for m in matches if m["anchor"] == 0]
        self.assertEqual(len(anchor_zero), 1)
        self.assertEqual((anchor_zero[0]["near"], anchor_zero[0]["far"]), (1, 2))

    def test_zero_or_one_candidate_has_no_match(self):
        self.assertEqual(continuity.select_geometry_matches(np.empty(0), np.empty((0, 2)), np.empty(0)), [])
        self.assertEqual(continuity.select_geometry_matches([0], [[0, 0]], [0.1]), [])


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="continuity-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, points, centers, *, family="curve", objects=(), endpoint_radius=1, width=192, height=80):
        points = np.asarray(points, dtype=float)
        length = cumulative_length(points)
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).line([tuple(p) for p in points], fill=255, width=1)
        image_path = self.root / "input.png"
        Image.new("RGB", (width, height), "white").save(image_path)
        cells = [{"bbox_px": [x - 4, y - 4, x + 4, y + 4], "merged_index": i} for i, (x, y) in enumerate(centers)]
        sample = {
            "id": "synthetic", "family": family, "image": {"width": width, "height": height},
            "config": {"line_width": 1},
            "objects": list(objects),
            "endpoints": {name: {"center_px": point.tolist(), "radius_px": endpoint_radius}
                          for name, point in (("start", points[0]), ("end", points[-1]))},
            "path": {"points_px": points.tolist(), "cumulative_length_px": length.tolist(),
                     "length_px": float(length[-1]), "start_px": points[0].tolist(), "end_px": points[-1].tolist()},
        }
        return cells, np.asarray(mask), sample, image_path

    def run_analysis(self, H, fixture, folder="analysis", config=None):
        cells, mask, sample, image_path = fixture
        output = self.root / folder
        summary = continuity.analyze_continuity(np.asarray(H, dtype=float), cells, mask, sample, image_path, output, config or {})
        # Reports must be JSON-safe, including cases without an identifiable contrast.
        json.dumps(summary, allow_nan=False)
        self.assertTrue((output / "summary.json").is_file())
        return summary, output

    @staticmethod
    def read_csv(path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_straight_geometry_is_non_identifiable_but_exports_pairs(self):
        fixture = self.fixture([[0, 16], [160, 16]], [(32, 16), (64, 16), (96, 16), (128, 16)], family="straight")
        summary, output = self.run_analysis([[1, 0], [0, 1], [1, 1], [-1, 1]], fixture)
        self.assertEqual(summary["status"], "structurally_non_identifiable")
        self.assertEqual(summary["n_clean_cells"], 4)
        self.assertIsNone(summary["conditional"]["raw_median"])
        self.assertIsNone(summary["conditional"]["centered_median"])
        self.assertEqual(len(self.read_csv(output / "all_pairs.csv")), 6)
        for name in ("all_pairs_scatter.png", "similarity_matrices.png", "selected_patches.png", "clean_patches.png", "similarities.npz"):
            self.assertTrue((output / name).is_file(), name)

    def test_collinear_centerline_is_not_identifiable_despite_family_label(self):
        fixture = self.fixture([[0, 16], [80, 16], [160, 16]], [(32, 16), (64, 16)], family="curve")
        summary, _ = self.run_analysis([[1, 0], [0, 1]], fixture)
        self.assertEqual(summary["status"], "structurally_non_identifiable")

    def test_no_line_and_single_clean_cell_are_graceful(self):
        for centers, expected in (([(32, 48)], "no_line_cells"), ([(32, 16)], "insufficient_clean_cells")):
            with self.subTest(expected=expected):
                fixture = self.fixture([[0, 16], [160, 16], [160, 48]], centers)
                summary, output = self.run_analysis([[1, 0]], fixture, folder=expected)
                self.assertEqual(summary["status"], expected)
                self.assertEqual(summary["n_matched_anchors"], 0)
                self.assertIsNone(summary["conditional"]["raw_median"])
                self.assertIsNone(summary["conditional"]["centered_median"])
                self.assertEqual(self.read_csv(output / "all_pairs.csv"), [])

    def test_reentered_cell_is_removed_from_clean_set(self):
        fixture = self.fixture([[0, 16], [160, 16], [160, 48], [0, 48], [0, 16], [160, 16]], [(64, 16), (64, 48)])
        summary, _ = self.run_analysis([[1, 0], [0, 1]], fixture)
        self.assertEqual(summary["n_line_cells"], 2)
        self.assertEqual(summary["n_clean_cells"], 1)
        self.assertEqual(summary["status"], "insufficient_clean_cells")

    def test_object_disk_margin_tangent_excludes_without_center_in_cell(self):
        # Cell spans [28,36]x[12,20]; center y=26 and radius4+margin2 touches its edge.
        objects = [{"id": "tangent", "center_px": [32, 26], "radius_px": 4}]
        fixture = self.fixture([[0, 16], [160, 16], [160, 48]], [(32, 16), (64, 16)], objects=objects)
        summary, _ = self.run_analysis([[1, 0], [0, 1]], fixture)
        self.assertEqual(summary["n_line_cells"], 2)
        self.assertEqual(summary["n_clean_cells"], 1)

    def test_endpoint_disk_excludes_cells_without_endpoint_center_in_cell(self):
        fixture = self.fixture([[20, 16], [160, 16], [160, 48]], [(32, 16), (64, 16)], endpoint_radius=6)
        summary, _ = self.run_analysis([[1, 0], [0, 1]], fixture)
        self.assertEqual(summary["n_line_cells"], 2)
        self.assertEqual(summary["n_clean_cells"], 1)

    def test_all_object_cells_have_no_clean_line_status(self):
        fixture = self.fixture(
            [[0, 16], [160, 16], [160, 48]], [(32, 16)],
            objects=[{"id": "cover", "center_px": [32, 16], "radius_px": 4}],
        )
        summary, _ = self.run_analysis([[1, 0]], fixture)
        self.assertEqual(summary["status"], "no_clean_line_cells")
        self.assertEqual(summary["n_clean_cells"], 0)
        self.assertIsNone(summary["conditional"]["raw_median"])

    def test_geometry_matches_are_unchanged_when_features_change(self):
        centers = [(x, y) for y in (16, 48) for x in (32, 64, 96, 128)]
        fixture = self.fixture([[0, 16], [160, 16], [160, 48], [0, 48]], centers)
        rng = np.random.default_rng(19)
        first, first_dir = self.run_analysis(rng.normal(size=(8, 7)), fixture, folder="features_a")
        second, second_dir = self.run_analysis(rng.normal(size=(8, 13)) * 100 + 30, fixture, folder="features_b")
        self.assertGreater(first["n_matched_anchors"], 0)
        self.assertEqual(first["n_matched_anchors"], second["n_matched_anchors"])
        # Feature-derived outcome columns may differ; only geometric selection may be used.
        a, b = self.read_csv(first_dir / "matches.csv"), self.read_csv(second_dir / "matches.csv")
        self.assertEqual(len(a), len(b))
        for row_a, row_b in zip(a, b):
            for key in ("anchor", "near", "far", "near_arc_px", "far_arc_px", "near_screen_px", "far_screen_px", "screen_distance_error_px", "coverage_error"):
                self.assertEqual(row_a[key], row_b[key], key)

    def test_full_feature_cosine_and_all_cell_centering_with_arc_sort(self):
        # Last row is a background token; its large features must affect the center.
        fixture = self.fixture(
            [[0, 16], [160, 16], [160, 48]],
            [(96, 16), (32, 16), (64, 16), (128, 16), (32, 64)],
        )
        H = np.array([[1, 2, 3, 4], [2, -1, 0, 0.1], [3, 0, -2, 0.2], [-4, 1, 2, -0.3], [30, 20, 10, 5]])
        summary, output = self.run_analysis(H, fixture)
        unit = H / np.linalg.norm(H, axis=1, keepdims=True)
        centered = H - H.mean(axis=0)
        centered_unit = centered / np.linalg.norm(centered, axis=1, keepdims=True)
        raw_expected = unit @ unit.T
        centered_expected = centered_unit @ centered_unit.T
        with np.load(output / "similarities.npz") as arrays:
            np.testing.assert_allclose(arrays["all_cells_mean"], H.mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(arrays["raw_norms"], np.linalg.norm(H, axis=1), atol=1e-12)
            np.testing.assert_allclose(arrays["centered_norms"], np.linalg.norm(centered, axis=1), atol=1e-12)
            np.testing.assert_allclose(arrays["raw_cosine_all"], raw_expected, atol=1e-12)
            np.testing.assert_allclose(arrays["centered_cosine_all"], centered_expected, atol=1e-12)
            np.testing.assert_array_equal(arrays["clean_token_indices"], [1, 2, 0, 3])
            order = [1, 2, 0, 3]
            np.testing.assert_allclose(arrays["clean_raw_cosine"], raw_expected[np.ix_(order, order)], atol=1e-12)
            np.testing.assert_allclose(arrays["clean_centered_cosine"], centered_expected[np.ix_(order, order)], atol=1e-12)
        self.assertEqual(summary["representation"]["cosine_dimensions"], 4)
        self.assertEqual(summary["representation"]["centering_cell_count"], 5)
        self.assertFalse(summary["representation"]["pca_used"])
        for pair in self.read_csv(output / "all_pairs.csv"):
            i, j = int(pair["token_i"]), int(pair["token_j"])
            self.assertAlmostEqual(float(pair["raw_cosine"]), raw_expected[i, j], places=12)
            self.assertAlmostEqual(float(pair["centered_cosine"]), centered_expected[i, j], places=12)

    def test_conditional_summary_and_reuse_are_exactly_reconstructible(self):
        fixture = self.fixture(
            [[0, 16], [160, 16], [160, 48], [0, 48]],
            [(x, y) for y in (16, 48) for x in (32, 64, 96, 128)],
        )
        H = np.random.default_rng(881).normal(size=(8, 9))
        summary, output = self.run_analysis(H, fixture)
        self.assertEqual(summary["status"], "ok")
        matches = self.read_csv(output / "matches.csv")
        raw = H / np.linalg.norm(H, axis=1, keepdims=True)
        centered = H - H.mean(axis=0)
        centered /= np.linalg.norm(centered, axis=1, keepdims=True)
        for key, features in (("raw", raw), ("centered", centered)):
            deltas = []
            for match in matches:
                a, near, far = (int(match[k]) for k in ("anchor_token", "near_token", "far_token"))
                delta = features[a] @ features[near] - features[a] @ features[far]
                self.assertAlmostEqual(float(match[f"{key}_delta"]), delta, places=12)
                deltas.append(delta)
            self.assertAlmostEqual(summary["conditional"][f"{key}_median"], float(np.median(deltas)), places=12)
        usage = self.read_csv(output / "token_usage.csv")
        self.assertEqual(sum(int(row["uses_total"]) for row in usage), 3 * len(matches))
        self.assertEqual(summary["token_reuse"]["total_token_uses"], 3 * len(matches))

    def test_zero_norm_activations_do_not_create_numeric_evidence(self):
        fixture = self.fixture(
            [[0, 16], [160, 16], [160, 48], [0, 48]],
            [(x, y) for y in (16, 48) for x in (32, 64, 96, 128)],
        )
        summary, output = self.run_analysis(np.zeros((8, 3)), fixture)
        self.assertEqual(summary["status"], "degenerate_activations")
        self.assertGreaterEqual(summary["n_matched_anchors"], 5)
        self.assertEqual(summary["zero_norm_raw_cells"], 8)
        self.assertEqual(summary["zero_norm_centered_cells"], 8)
        self.assertIsNone(summary["conditional"]["raw_median"])
        self.assertIsNone(summary["conditional"]["centered_median"])
        with np.load(output / "similarities.npz") as arrays:
            self.assertTrue(np.isnan(arrays["raw_cosine_all"]).all())
            self.assertTrue(np.isnan(arrays["centered_cosine_all"]).all())

    def pre_merge_fixture(self):
        _, mask, sample, image_path = self.fixture(
            [[0, 8], [64, 8]], [], family="straight", width=64, height=32
        )
        # Actual Qwen vision ordering: each 2x2 merge block precedes the next block.
        coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        cells = [
            {"vision_index": index, "row": row, "column": column, "merged_index": column // 2,
             "bbox_px": [16 * column, 16 * row, 16 * (column + 1), 16 * (row + 1)]}
            for index, (row, column) in enumerate(coordinates)
        ]
        # Remove endpoint exclusions to isolate the ordering and numerical correspondence.
        sample["endpoints"] = {}
        return cells, mask, sample, image_path

    def test_pre_merge_block_major_cells_preserve_feature_row_correspondence(self):
        fixture = self.pre_merge_fixture()
        H = np.arange(1, 8 * 5 + 1, dtype=float).reshape(8, 5)
        summary, output = self.run_analysis(H, fixture)
        self.assertEqual(summary["representation"]["geometry_kind"], "pre_merge_vision")
        self.assertEqual(summary["representation"]["geometry_index_field"], "vision_index")
        self.assertEqual(summary["representation"]["spatial_grid_rows"], 2)
        self.assertEqual(summary["representation"]["spatial_grid_columns"], 4)
        self.assertEqual(summary["representation"]["patch_width_px"], 16)
        self.assertEqual(summary["representation"]["patch_height_px"], 16)
        self.assertIn("block-major", summary["representation"]["sequence_order"])
        self.assertEqual(summary["n_total_cells"], 8)
        self.assertEqual(summary["n_line_cells"], 4)
        self.assertEqual(summary["n_clean_cells"], 4)
        with np.load(output / "similarities.npz") as arrays:
            np.testing.assert_array_equal(arrays["line_token_indices"], [0, 1, 4, 5])
            np.testing.assert_array_equal(arrays["clean_token_indices"], [0, 1, 4, 5])
            np.testing.assert_allclose(arrays["clean_arc_positions_px"], [8, 24, 40, 56])
            unit = H / np.linalg.norm(H, axis=1, keepdims=True)
            expected = (unit @ unit.T)[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])]
            np.testing.assert_allclose(arrays["clean_raw_cosine"], expected, atol=1e-12)
        patches = self.read_csv(output / "patch_table.csv")
        self.assertEqual([int(row["geometry_index"]) for row in patches], list(range(8)))
        self.assertEqual([row["geometry_kind"] for row in patches], ["pre_merge_vision"] * 8)
        self.assertEqual([int(row["parent_merged_index"]) for row in patches], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertNotIn("merged_index", patches[0])

    def test_pre_merge_reordered_vision_ids_are_rejected(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        cells[0], cells[1] = cells[1], cells[0]
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "bad_ids")

    def test_pre_merge_row_major_coordinates_are_rejected(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        for index, (row, column) in enumerate((r, c) for r in range(2) for c in range(4)):
            cells[index].update(row=row, column=column, merged_index=column // 2,
                                bbox_px=[16 * column, 16 * row, 16 * (column + 1), 16 * (row + 1)])
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "row_major")

    def test_merged_cell_index_must_also_match_feature_row(self):
        cells, mask, sample, image_path = self.fixture([[0, 16], [160, 16]], [(32, 16), (64, 16)])
        cells[0]["merged_index"], cells[1]["merged_index"] = 1, 0
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((2, 3)), cells, mask, sample, image_path, self.root / "bad_merged_ids")

    def test_geometry_dictionary_selects_unique_sequence_by_feature_row_count(self):
        vision, mask, sample, image_path = self.pre_merge_fixture()
        merged = [
            {"merged_index": index, "bbox_px": [32 * index, 0, 32 * (index + 1), 32]}
            for index in range(2)
        ]
        geometry = {"vision_tokens": vision, "merged_tokens": merged}
        for count, kind, lines in ((8, "pre_merge_vision", 4), (2, "merged_visual", 2)):
            with self.subTest(kind=kind):
                H = np.arange(1, count * 3 + 1, dtype=float).reshape(count, 3)
                summary, _ = self.run_analysis(H, (geometry, mask, sample, image_path), folder=kind)
                self.assertEqual(summary["representation"]["geometry_kind"], kind)
                self.assertEqual(summary["n_total_cells"], count)
                self.assertEqual(summary["n_line_cells"], lines)

    def test_ambiguous_geometry_dictionary_is_rejected(self):
        vision, mask, sample, image_path = self.pre_merge_fixture()
        geometry = {"vision_tokens": vision, "merged_tokens": vision}
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), geometry, mask, sample, image_path, self.root / "ambiguous")

    def test_pre_merge_row_without_column_cannot_bypass_order_validation(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        for cell in cells:
            del cell["column"]
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "row_only")

    def test_pre_merge_missing_merge_block_is_rejected_despite_sorted_coordinates(self):
        _, mask, sample, image_path = self.fixture([[0, 8], [96, 8]], [], width=96, height=32)
        coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 4), (0, 5), (1, 4), (1, 5)]
        cells = [
            {"vision_index": index, "row": row, "column": column, "merged_index": column // 2,
             "bbox_px": [16 * column, 16 * row, 16 * (column + 1), 16 * (row + 1)]}
            for index, (row, column) in enumerate(coordinates)
        ]
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "missing_block")

    def test_pre_merge_bbox_must_agree_with_grid_coordinates(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        cells[4]["bbox_px"] = cells[5]["bbox_px"].copy()
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "bad_bbox")

    def test_pre_merge_parent_index_must_agree_with_merge_block(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        cells[4]["merged_index"] = 0
        with self.assertRaises(ValueError):
            continuity.analyze_continuity(np.ones((8, 3)), cells, mask, sample, image_path, self.root / "bad_parent")

    def test_pre_merge_bbox_only_coordinates_preserve_block_major_order(self):
        cells, mask, sample, image_path = self.pre_merge_fixture()
        for cell in cells:
            del cell["row"]
            del cell["column"]
        H = np.arange(1, 25, dtype=float).reshape(8, 3)
        summary, output = self.run_analysis(H, (cells, mask, sample, image_path))
        self.assertEqual(summary["representation"]["geometry_kind"], "pre_merge_vision")
        self.assertEqual(summary["representation"]["spatial_grid_rows"], 2)
        self.assertEqual(summary["representation"]["spatial_grid_columns"], 4)
        with np.load(output / "similarities.npz") as arrays:
            np.testing.assert_array_equal(arrays["clean_token_indices"], [0, 1, 4, 5])
            unit = H / np.linalg.norm(H, axis=1, keepdims=True)
            expected = (unit @ unit.T)[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])]
            np.testing.assert_allclose(arrays["clean_raw_cosine"], expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
