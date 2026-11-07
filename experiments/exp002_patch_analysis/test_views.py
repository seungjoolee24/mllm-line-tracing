"""CPU-only checks of PCA selection and saved predictor attention alignment."""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from views import extract_attention, pca_case, pca_rgb, render_attention


def cells(rows=2, columns=2, size=4):
    return [{"merged_index": row * columns + column,
             "bbox_px": [column*size, row*size, (column+1)*size, (row+1)*size]}
            for row in range(rows) for column in range(columns)]


class FakeCase:
    def __init__(self, image_path):
        self.image_path = image_path
        self.sample = {"id": "synthetic", "answer_colors": ["red", "green"]}
        self.geometry = {"merged_tokens": cells()}
        self.meta = {
            "capture_spec": {"decoder_layers": 2},
            "layout": {"attention_query_positions": [5, 20, 21, 22, 23],
                       "predictor_positions": [20, 21, 22, 23],
                       "predictor_attention_rows": [1, 2, 3, 4]},
            "generation": {"token_details": {"color_spans": [
                {"color": "red", "generation_steps": [0, 1], "predictor_positions": [20, 21]},
                {"color": "blue", "generation_steps": [2], "predictor_positions": [22]}]}}}
        weights = np.zeros((2, 5, 4), dtype=np.float32)
        weights[0, 1, 0] = 0.4
        weights[1, 1, 1] = 0.2
        weights[:, 2, 3] = 0.9     # second BPE must not replace first-BPE predictor
        weights[:, 3, 2] = 0.2
        weights[:, 4, 3] = 0.8     # generated-token row would be an off-by-one error
        self.weights = [weights, weights.copy()]
        self.weights[1][:, 3] = 0  # conditional attention must be missing, not zero
        self.loads = []

    def load(self, name):
        self.loads.append(name)
        if not name.endswith("/attention_to_image"):
            raise AssertionError("attention views must not load hidden states")
        return self.weights[int(name.split("/")[1].split("_")[1])]


class PcaTests(unittest.TestCase):
    def test_sample_space_pca_matches_svd_and_has_fixed_signs(self):
        rng = np.random.default_rng(13)
        features = rng.normal(size=(18, 9)) * np.arange(1, 10)
        selection = np.arange(13)
        result = pca_rgb(features, selection)
        centered = features[selection] - features[selection].mean(axis=0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        np.testing.assert_allclose(result["ratios"], singular[:3]**2 / (singular**2).sum(), atol=1e-12)
        np.testing.assert_allclose(result["components"].T @ result["components"], vt[:3].T @ vt[:3], atol=1e-11)
        np.testing.assert_allclose(result["mean"], features[selection].mean(axis=0))
        for component in result["components"]:
            self.assertGreaterEqual(component[np.argmax(abs(component))], 0)
        self.assertEqual(result["rgb"].shape, (18, 3))
        self.assertTrue(np.all((result["rgb"] >= 0) & (result["rgb"] <= 1)))
        np.testing.assert_allclose(result["scaling_low"], result["scores"][selection].min(axis=0))
        np.testing.assert_allclose(result["scaling_high"], result["scores"][selection].max(axis=0))
        # Unselected tokens must not affect fit axes, min-max scales, or selected colors.
        changed = features.copy()
        changed[13:] *= 1e6
        other = pca_rgb(changed, selection)
        for key in ["components", "mean", "ratios", "scaling_low", "scaling_high"]:
            np.testing.assert_array_equal(result[key], other[key])
        np.testing.assert_array_equal(result["rgb"][selection], other["rgb"][selection])

    def test_degenerate_axes_are_neutral_and_invalid_selection_is_rejected(self):
        features = np.arange(6)[:, None] * np.array([[1., 2., 3., 4.]])
        result = pca_rgb(features, np.arange(6))
        np.testing.assert_allclose(result["rgb"][:, 1:], 0.5)
        np.testing.assert_allclose(result["ratios"], [1, 0, 0], atol=1e-12)
        with self.assertRaises(ValueError):
            pca_rgb(features, np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            pca_rgb(features, np.array([0, 1, 2, 2]))
        with self.assertRaises(ValueError):
            pca_rgb(np.ones((6, 4)), np.arange(6))

    def test_line_selection_uses_any_positive_path_pixel_and_geometry_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "source.png"
            Image.new("RGB", (12, 8), "white").save(image)
            geometry = list(reversed(cells(2, 3)))  # storage order is intentionally scrambled
            mask = np.zeros((8, 12), dtype=np.uint8)
            for y, x in [(0, 0), (1, 4), (5, 4), (7, 11)]:
                mask[y, x] = 1
            features = np.random.default_rng(9).normal(size=(6, 8))
            result = pca_case(features, geometry, mask, image, root / "pca")
            self.assertEqual(result["conditions"]["all"]["fit_token_count"], 6)
            self.assertEqual(result["conditions"]["line"]["fit_token_count"], 4)
            with np.load(result["conditions"]["line"]["npz"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["selected_indices"], [0, 1, 4, 5])
            summary = json.loads(Path(result["conditions"]["line"]["json"]).read_text())
            self.assertEqual(summary["fit_token_count"], 4)
            self.assertTrue(Path(result["conditions"]["line"]["png"]).is_file())
            before = Path(result["conditions"]["line"]["npz"]).read_bytes()
            pca_case(features, geometry, mask, image, root / "pca")
            self.assertEqual(before, Path(result["conditions"]["line"]["npz"]).read_bytes())
            with self.assertRaises(FileExistsError):
                pca_case(features + 1, geometry, mask, image, root / "pca")


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (8, 8), "white").save(self.image)
        self.case = FakeCase(self.image)

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def test_first_bpe_predictor_and_head_mean_before_conditioning(self):
        destination = self.root / "attention"
        destination.mkdir()
        interrupted = destination / "attention_head_mean.npz.partial"
        interrupted.write_bytes(b"interrupted previous write")
        result = extract_attention(self.case, self.root / "attention")
        self.assertFalse(interrupted.exists())
        self.assertEqual(len(self.case.loads), 2)
        self.assertEqual(result["words"][0]["bpe_token_count"], 2)
        self.assertEqual(result["words"][0]["predictor_position"], 20)
        self.assertEqual(result["words"][1]["expected_color"], "green")
        with np.load(result["npz"], allow_pickle=False) as saved:
            self.assertEqual(saved["raw"].shape, (2, 2, 4))
            np.testing.assert_allclose(saved["raw"][0, 0], [0.2, 0.1, 0, 0])
            np.testing.assert_allclose(saved["conditional"][0, 0], [2/3, 1/3, 0, 0], atol=1e-7)
            self.assertTrue(np.isnan(saved["conditional"][1, 1]).all())
            np.testing.assert_allclose(saved["mass"][0], [0.3, 0.2])
            np.testing.assert_array_equal(saved["attention_rows"], [1, 3])
        before = Path(result["npz"]).read_bytes()
        again = extract_attention(self.case, self.root / "attention")
        self.assertEqual(before, Path(again["npz"]).read_bytes())

    def test_validation_includes_unselected_rows_and_zero_mass_json_is_finite(self):
        self.case.weights[0][:] = 0
        self.case.weights[1][:] = 0
        result = extract_attention(self.case, self.root / "zero")
        self.assertEqual(result["conditional_vmax"], 0)
        def reject_nonfinite(value):
            raise AssertionError(f"JSON contains a non-finite number: {value}")
        json.loads(Path(result["json"]).read_text(), parse_constant=reject_nonfinite)
        self.case.weights[1][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_attention(self.case, self.root / "invalid")

    def test_render_limits_shared_over_layers_and_explicit_head_is_lazy(self):
        result = extract_attention(self.case, self.root / "attention")
        self.case.loads.clear()
        figure = render_attention(self.case, result["npz"], self.root / "grids", 0.8, 0.9, 0, save=False)
        self.assertEqual(self.case.loads, [])
        image_axes = [axis for axis in figure.axes if len(axis.images) == 2]
        self.assertEqual(len(image_axes), 2)
        self.assertTrue(all(axis.images[-1].get_clim() == (0, 0.8) for axis in image_axes))
        self.assertTrue(image_axes[0].get_title().startswith("Layer 01"))
        self.assertIn("predictor position 20", figure._suptitle.get_text())
        plt.close(figure)
        figure = render_attention(self.case, result["npz"], self.root / "grids", 0.8, 0.9, 0,
                                  head=0, save=False)
        self.assertEqual(len(self.case.loads), 2)
        image_axes = [axis for axis in figure.axes if len(axis.images) == 2]
        for axis in image_axes:
            self.assertAlmostEqual(axis.images[-1].get_clim()[1], 0.4)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
