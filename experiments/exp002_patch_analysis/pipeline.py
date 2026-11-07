"""User-run CPU analysis of the available run002 case00-04 captures.

No model is instantiated. All authoritative outputs are written to the user's
mounted Drive. Local storage is used only to cache verified input archives.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image

import views
from continuity import analyze_continuity


CASES = ("case_00_curve", "case_01_straight", "case_02_elbow", "case_03_swirl", "case_04_curve")
DATASET_COMMIT = "957385e35c3cc43af96f650771285a70ea72f3e2"
DATASET_ANNOTATIONS_SHA256 = "24686bc95539b559f73f12d45f58681e7af87d567d3ea8738341cd65f2555e8f"
CAPTURE_IO_SHA256 = "4aef2c4057e32433522ff8f71121fe4b42434a5ceee15639c5e3acf63f30ffbe"
MANIFEST_SHA256 = "7d86f02d1d52ab50a40a1abe0028334e1773f5a4b854bb645efe7f7ddca35e5e"
CANONICAL_MANIFEST_URL = (
    f"https://raw.githubusercontent.com/seungjoolee24/mllm-line-tracing/{DATASET_COMMIT}/"
    "experiments/exp001_qwen3vl4b_trace10/results/run002_release/artifact_manifest.json"
)
DEFAULT_CONFIG = {
    "cases": list(CASES), "seed": 0, "feature_tap": "merger/final/output",
    "feature_shape": [400, 2560], "decoder_layers": 36,
    "continuity_feature_tap": "merger/final/input",
    "continuity_feature_shape": [1600, 1024],
    "feature_match_atol": 0.001, "feature_match_rtol": 0.001,
    "pca_conditions": ["all_patches", "line_patches"],
    "word_alignment": "first_bpe_predictor", "attention_heads": "mean",
    "attention_modes": ["raw", "conditional"],
    "continuity": {
        "min_line_coverage": 0.01, "exclusion_margin_px": 2,
        "near_arc_max_px": 64, "far_arc_min_px": 128,
        "max_screen_distance_error_px": 8, "max_coverage_error": 0.03,
        "min_matched_anchors": 5,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=_json_default)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def _download(url, target, expected_sha256):
    target = Path(target)
    if target.is_file() and sha256(target) == expected_sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "mllm-exp002"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, 1024 * 1024)
    if sha256(temporary) != expected_sha256:
        raise ValueError(f"Source checksum differs: {target}")
    temporary.replace(target)
    return target


def _copy_exact(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Existing output differs; choose a new OUTPUT_DIR: {target}")
        return
    temporary = target.with_name(target.name + ".partial")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256(temporary) != expected:
        raise ValueError(f"Copy checksum differs: {target}")
    temporary.replace(target)


def _environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "torch", "matplotlib", "Pillow", "ipywidgets"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def input_preflight(artifact_dir, recover_manifest=False):
    """Check selected archives; optionally recover only the pinned tiny manifest.

    No raw archive is downloaded or regenerated here. An existing different
    manifest is never replaced. A recovered manifest is published only after
    all selected archive names and sizes pass the preflight.
    """
    root = Path(artifact_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input artifact directory is missing: {root}. Mount Drive and check ARTIFACT_DIR.")
    manifest = root / "artifact_manifest.json"
    if manifest.exists() and not manifest.is_file():
        raise ValueError(f"Manifest is not a regular file: {manifest}")
    if manifest.is_file():
        actual = sha256(manifest)
        if actual != MANIFEST_SHA256:
            raise ValueError(f"Manifest SHA-256 mismatch: {manifest}\nexpected={MANIFEST_SHA256}\nactual={actual}. Existing file was not changed.")
    elif not recover_manifest:
        raise FileNotFoundError(f"Manifest is missing: {manifest}. Run input_preflight(ARTIFACT_DIR, recover_manifest=True) before creating AnalysisRun.")
    recovered = not manifest.is_file()
    with tempfile.TemporaryDirectory(prefix="exp002-manifest-") as temporary:
        source = Path(temporary) / "artifact_manifest.json" if recovered else manifest
        if recovered:
            _download(CANONICAL_MANIFEST_URL, source, MANIFEST_SHA256)
            if sha256(source) != MANIFEST_SHA256:
                raise ValueError("Recovered manifest SHA-256 mismatch")
        data = json.loads(source.read_text(encoding="utf-8"))
        if (data.get("package_protocol") != "qwen3vl_trace_case_zip_v1"
                or data.get("collection_schema_version") != "1.1"
                or data.get("run_name") != "run002" or data.get("status") != "complete"
                or data.get("case_count") != 10 or data.get("tensor_file_count") != 2060):
            raise ValueError("Manifest protocol/run/status/count differs from the collected run002")
        archives = data.get("archives", [])
        names = [entry.get("file") for entry in archives]
        if len(archives) != 11 or len(set(names)) != len(names):
            raise ValueError("Manifest archive filenames are missing or duplicate")
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError("Manifest contains an invalid archive filename")
        metadata = [entry for entry in archives if entry.get("kind") == "metadata"]
        if len(metadata) != 1:
            raise ValueError("Manifest must declare exactly one metadata archive")
        required = list(metadata)
        for case_id in CASES:
            matches = [entry for entry in archives if entry.get("kind") == "case" and entry.get("sample_id") == case_id]
            if len(matches) != 1:
                raise ValueError(f"Manifest selected case is missing or ambiguous: {case_id}")
            required.append(matches[0])
        missing, wrong_size = [], []
        for entry in required:
            path = root / entry["file"]
            if not path.is_file():
                missing.append(entry["file"])
            elif path.stat().st_size != entry["bytes"]:
                wrong_size.append(f"{entry['file']}: expected={entry['bytes']}, actual={path.stat().st_size}")
        if missing:
            raise FileNotFoundError(f"Selected input archives are missing in {root}: {', '.join(missing)}. No raw data were downloaded.")
        if wrong_size:
            raise ValueError("Input archive size mismatch: " + "; ".join(wrong_size))
        if recovered:
            _copy_exact(source, manifest)
    print(f"INPUT PREFLIGHT READY: {len(CASES)} selected cases + metadata; manifest recovered={recovered}", flush=True)
    return {"status": "ready", "artifact_dir": str(root), "manifest_sha256": MANIFEST_SHA256,
            "manifest_recovered": recovered, "selected_cases": list(CASES),
            "required_archives": required}


def verify_output_bundle(output_dir):
    """Re-read all declared files, including after Drive sync/unmount/remount."""
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "output_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("file_count") != len(manifest["files"]):
        raise ValueError("Output manifest is incomplete")
    for relative, record in manifest["files"].items():
        path = (root / relative).resolve()
        path.relative_to(root)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Output verification failed: {relative}")
    return {"status": "verified", "output_dir": str(root), "file_count": manifest["file_count"],
            "total_bytes": sum(row["bytes"] for row in manifest["files"].values())}


class AnalysisRun:
    def __init__(self, repo_root, artifact_dir, output_dir, cache_dir, config=None):
        self.repo = Path(repo_root).resolve()
        self.artifact_dir = Path(artifact_dir).resolve()
        self.output = Path(output_dir).resolve()
        self.cache = Path(cache_dir).resolve()
        # Keep inputs and outputs in separate directory trees.
        if (self.output == self.artifact_dir or self.output in self.artifact_dir.parents
                or self.artifact_dir in self.output.parents):
            raise ValueError("Input artifacts and output must be in separate directory trees")
        for other in (self.output, self.artifact_dir):
            if self.cache == other or self.cache in other.parents or other in self.cache.parents:
                raise ValueError("Cache, output, and input artifacts must use separate directory trees")
        self.config = json.loads(json.dumps(DEFAULT_CONFIG))
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
            for key, value in config.items():
                if key == "continuity":
                    if set(value) - set(self.config[key]):
                        raise ValueError("Unknown continuity configuration key")
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        if self.config["cases"] != list(CASES):
            raise ValueError("This notebook is fixed to the approved case00-04 selection")
        for key in ("feature_tap", "feature_shape", "continuity_feature_tap", "continuity_feature_shape", "decoder_layers", "pca_conditions",
                    "word_alignment", "attention_heads", "attention_modes"):
            if self.config[key] != DEFAULT_CONFIG[key]:
                raise ValueError(f"This analysis requires the approved {key}")
        self.code_files = {
            "pipeline.py": Path(__file__).resolve(),
            "views.py": Path(views.__file__).resolve(),
            "continuity.py": Path(sys.modules[analyze_continuity.__module__].__file__).resolve(),
            "capture_io.py": self.repo / "experiments/exp001_qwen3vl4b_trace10/analysis.py",
        }
        if sha256(self.code_files["capture_io.py"]) != CAPTURE_IO_SHA256:
            raise ValueError("The pinned capture reader differs")
        self.identity = {
            "protocol": "exp002_patch_analysis_v1", "config": self.config,
            "output_dir": str(self.output),
            "artifact_manifest_sha256": MANIFEST_SHA256,
            "dataset_commit": DATASET_COMMIT,
            "annotations_sha256": DATASET_ANNOTATIONS_SHA256,
            "code_sha256": {name: sha256(path) for name, path in self.code_files.items()},
            "environment": _environment(),
        }
        self.fingerprint = hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.output.mkdir(parents=True, exist_ok=True)
        identity_path = self.output / "analysis_config.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != self.identity:
                raise ValueError("Output belongs to different code/config/data/environment. Use a new OUTPUT_DIR.")
        elif any(self.output.iterdir()):
            raise ValueError("Nonempty output has no analysis_config.json; choose a new OUTPUT_DIR")
        else:
            atomic_json(identity_path, self.identity)
        self.cache.mkdir(parents=True, exist_ok=True)
        for name, source in self.code_files.items():
            _copy_exact(source, self.output / "source" / name)
        spec = importlib.util.spec_from_file_location("exp002_capture_io", self.code_files["capture_io.py"])
        self.io = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.io)
        self.cases = {}
        self.features = {}
        self.continuity_features = {}
        self.path_masks = {}
        self.attention_summary = None

    def _check_prepared(self):
        if tuple(self.cases) != CASES:
            raise RuntimeError("Run the input preparation cell first")

    def _stage(self, key, folder, compute):
        """Reuse a stage only after checking every saved output file."""
        folder = Path(folder)
        receipt_path = self.output / "stage_receipts" / (key + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["analysis_fingerprint"] != self.fingerprint:
                raise ValueError(f"Stage identity differs: {key}")
            current_paths = {str(p.relative_to(self.output)) for p in folder.rglob("*") if p.is_file()}
            if current_paths != set(receipt["files"]):
                raise ValueError(f"Stage file inventory differs: {key}")
            for relative, record in receipt["files"].items():
                path = self.output / relative
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError(f"Stage output differs: {relative}")
            print(f"VERIFIED / SKIP: {key}", flush=True)
            return receipt["result"]
        # Calculate into disposable local staging first. If interrupted before
        # the receipt, rebuild the local stage and reuse only byte-identical
        # files already copied to Drive. Completed Drive results never change.
        staging = self.cache / "staging" / self.fingerprint / key
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        result = json.loads(canonical(compute(staging)))
        def rebase(value):
            if isinstance(value, dict):
                return {k: rebase(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rebase(v) for v in value]
            if isinstance(value, str) and value.startswith(str(staging) + "/"):
                return str(folder / Path(value).relative_to(staging))
            return value
        result = rebase(result)
        staged = [p for p in sorted(staging.rglob("*")) if p.is_file()]
        if not staged or any(p.name.endswith((".partial", ".part")) for p in staged):
            raise ValueError(f"Stage did not finish local calculation: {key}")
        for source in staged:
            _copy_exact(source, folder / source.relative_to(staging))
        expected_files = {str(p.relative_to(staging)) for p in staged}
        actual_files = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
        if actual_files != expected_files:
            raise ValueError(f"Unexpected files in incomplete stage: {key}")
        files = {str(p.relative_to(self.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(folder.rglob("*")) if p.is_file()}
        if not files:
            raise ValueError(f"Stage produced no output: {key}")
        atomic_json(receipt_path, {"analysis_fingerprint": self.fingerprint,
                                  "files": files, "result": result})
        print(f"SAVED / VERIFIED: {key} ({len(files)} files)", flush=True)
        return result

    def prepare(self):
        input_preflight(self.artifact_dir, recover_manifest=False)
        manifest = self.artifact_dir / "artifact_manifest.json"
        bundle = self.io.DriveBundle(self.artifact_dir, self.cache / "captures")
        print(f"Restoring metadata and only the {len(CASES)} selected case ZIPs...", flush=True)
        run = bundle.metadata()
        dataset = self.cache / "dataset"
        prefix = f"https://raw.githubusercontent.com/seungjoolee24/mllm-line-tracing/{DATASET_COMMIT}/experiments/exp001_qwen3vl4b_trace10/dataset/"
        annotations = _download(prefix + "annotations.jsonl", dataset / "annotations.jsonl",
                                DATASET_ANNOTATIONS_SHA256)
        samples = self.io.dataset_samples(dataset)
        rows = []
        for case_id in CASES:
            bundle.case(case_id)
            sample = samples[case_id]
            for name in ("image", "path_mask", "object_mask"):
                relative = sample["files"][name]
                _download(prefix + relative, dataset / relative, sample["sha256"][name])
                _copy_exact(dataset / relative, self.output / "inputs/dataset" / relative)
            case = self.io.CaseAnalysis(run, case_id, dataset)
            if case.meta["score"]["exact_match"] != (case_id != "case_03_swirl"):
                raise ValueError("Case outcome differs from the approved selection")
            if len(case.meta["tensors"]) != 206:
                raise ValueError("Case tensor inventory differs")
            if case.meta["capture_spec"]["decoder_layers"] != 36:
                raise ValueError("Decoder layer count differs")
            H = case.load("merger/final/output").float().numpy().copy()
            H0 = case.load("decoder/layer_000/input")[0, case.image_positions, :].float().numpy()
            if list(H.shape) != [400, 2560] or H0.shape != H.shape:
                raise ValueError("Pre-LLM feature shape differs")
            difference = float(np.max(np.abs(H - H0)))
            if not np.allclose(H, H0, atol=self.config["feature_match_atol"],
                               rtol=self.config["feature_match_rtol"]):
                raise ValueError("Final merger output and first decoder image input differ")
            V = case.load("merger/final/input").float().numpy().copy()
            Vref = case.load("vision/block_023/output").float().numpy()
            if list(V.shape) != [1600, 1024] or Vref.shape != V.shape:
                raise ValueError("Pre-merge vision feature shape differs")
            vision_difference = float(np.max(np.abs(V - Vref)))
            if not np.allclose(V, Vref, atol=self.config["feature_match_atol"],
                               rtol=self.config["feature_match_rtol"]):
                raise ValueError("Final ViT output and merger input differ")
            if [cell["vision_index"] for cell in case.geometry["vision_tokens"]] != list(range(1600)):
                raise ValueError("Vision geometry is not aligned with the saved feature rows")
            with Image.open(dataset / sample["files"]["path_mask"]) as image:
                path_mask = np.asarray(image.convert("L")).copy()
            self.cases[case_id] = case
            self.features[case_id] = H
            self.continuity_features[case_id] = V
            self.path_masks[case_id] = path_mask
            rows.append({"case_id": case_id, "family": sample["family"],
                         "exact_match": case.meta["score"]["exact_match"],
                         "objects": len(sample["objects"]), "feature_shape": list(H.shape),
                         "max_abs_merger_decoder0_difference": difference,
                         "continuity_feature_shape": list(V.shape),
                         "max_abs_vit_merger_input_difference": vision_difference,
                         "generated_answer": case.meta["generation"]["raw_answer"]})
            _copy_exact(case.directory / "meta.json", self.output / "inputs" / case_id / "meta.json")
            _copy_exact(case.directory / "token_geometry.json", self.output / "inputs" / case_id / "token_geometry.json")
            print(f"INPUT VERIFIED: {case_id}, PCA H={H.shape}, continuity V={V.shape}, tap differences={difference:g}/{vision_difference:g}", flush=True)
        _copy_exact(annotations, self.output / "inputs/dataset/annotations.jsonl")
        _copy_exact(manifest, self.output / "inputs/artifact_manifest.json")
        input_files = {str(p.relative_to(self.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                       for p in sorted((self.output / "inputs").rglob("*")) if p.is_file()}
        atomic_json(self.output / "input_manifest.json", {
            "analysis_fingerprint": self.fingerprint,
            "artifact_manifest_sha256": MANIFEST_SHA256,
            "cases": rows, "files": input_files,
            "restored_case_archives": list(CASES),
        })
        return rows

    def pca(self):
        self._check_prepared()
        result = {}
        for case_id, case in self.cases.items():
            folder = self.output / case_id / "pca"
            result[case_id] = self._stage(case_id + "_pca", folder, lambda f, c=case, k=case_id:
                views.pca_case(self.features[k], c.geometry["merged_tokens"], self.path_masks[k], c.image_path, f))
        return result

    def attention(self):
        self._check_prepared()
        result = {}
        for case_id, case in self.cases.items():
            folder = self.output / case_id / "attention/data"
            result[case_id] = self._stage(case_id + "_attention", folder,
                                         lambda f, c=case: views.extract_attention(c, f))
        self.attention_summary = {
            "cases": result,
            "raw_vmax": max(item["raw_vmax"] for item in result.values()),
            "conditional_vmax": max(item["conditional_vmax"] for item in result.values()),
        }
        atomic_json(self.output / "attention_scales.json", self.attention_summary)
        return self.attention_summary

    def word_options(self, case_id):
        self._check_prepared()
        spans = self.cases[case_id].meta["generation"]["token_details"]["color_spans"]
        return [(f"{i+1}: {span['color']}", i) for i, span in enumerate(spans)]

    def _attention_arguments(self, case_id):
        if self.attention_summary is None:
            path = self.output / "attention_scales.json"
            if not path.is_file():
                raise RuntimeError("Run attention extraction first")
            self.attention_summary = json.loads(path.read_text())
        item = self.attention_summary["cases"][case_id]
        return item["npz"], self.attention_summary["raw_vmax"], self.attention_summary["conditional_vmax"]

    def show_attention(self, case_id, word_index=0, mode="raw", head=None):
        self._check_prepared()
        npz, raw_vmax, conditional_vmax = self._attention_arguments(case_id)
        return views.render_attention(self.cases[case_id], npz, self.cache / "viewer",
                                       raw_vmax, conditional_vmax, word_index,
                                       mode=mode, head=head, save=False)

    def render_all_attention(self):
        self._check_prepared()
        result = {}
        for case_id, case in self.cases.items():
            npz, raw_vmax, conditional_vmax = self._attention_arguments(case_id)
            result[case_id] = []
            for _, index in self.word_options(case_id):
                for mode in ("raw", "conditional"):
                    folder = self.output / case_id / f"attention/grids/word_{index:02d}/{mode}"
                    key = f"{case_id}_word_{index:02d}_{mode}"
                    def compute(f, c=case, n=npz, i=index, m=mode):
                        path = views.render_attention(c, n, f, raw_vmax, conditional_vmax,
                                                       i, mode=m, head=None, save=True)
                        return {"path": str(path), "word_index": i, "mode": m}
                    result[case_id].append(self._stage(key, folder, compute))
        return result

    def continuity(self):
        self._check_prepared()
        result = {}
        for case_id, case in self.cases.items():
            folder = self.output / case_id / "continuity"
            result[case_id] = self._stage(case_id + "_continuity", folder,
                lambda f, c=case, k=case_id: analyze_continuity(
                    self.continuity_features[k], c.geometry["vision_tokens"], self.path_masks[k],
                    c.sample, c.image_path, f, self.config["continuity"]))
        return result

    def finalize(self):
        self._check_prepared()
        if json.loads((self.output / "analysis_config.json").read_text()) != self.identity:
            raise ValueError("Analysis identity was modified after preparation")
        inputs = json.loads((self.output / "input_manifest.json").read_text())
        for relative, record in inputs["files"].items():
            path = self.output / relative
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Saved input verification failed: {relative}")
        for name, expected_sha in self.identity["code_sha256"].items():
            if sha256(self.output / "source" / name) != expected_sha:
                raise ValueError(f"Saved source verification failed: {name}")
        incomplete = [str(p.relative_to(self.output)) for p in self.output.rglob("*")
                      if p.is_file() and p.name.endswith((".partial", ".part"))]
        if incomplete:
            raise ValueError(f"Incomplete files remain: {incomplete}")
        expected = []
        for case_id in CASES:
            expected += [case_id + "_" + stage for stage in ("pca", "attention", "continuity")]
            for _, index in self.word_options(case_id):
                expected += [f"{case_id}_word_{index:02d}_{mode}" for mode in ("raw", "conditional")]
        receipts = {}
        for key in expected:
            path = self.output / "stage_receipts" / (key + ".json")
            if not path.is_file():
                raise RuntimeError(f"Incomplete stage: {key}. Run its notebook cell first.")
            receipt = json.loads(path.read_text())
            if receipt["analysis_fingerprint"] != self.fingerprint:
                raise ValueError(f"Stage identity differs: {key}")
            for relative, record in receipt["files"].items():
                output = self.output / relative
                if not output.is_file() or output.stat().st_size != record["bytes"] or sha256(output) != record["sha256"]:
                    raise ValueError(f"Output verification failed: {relative}")
            receipts[key] = receipt["result"]
        atomic_json(self.output / "environment.json", self.identity["environment"])
        summary = {
            "status": "complete", "analysis_fingerprint": self.fingerprint,
            "cases": list(CASES), "pca_condition_count": len(CASES) * 2,
            "attention_grid_count": sum(len(self.word_options(k)) for k in CASES) * 2,
            "stages": receipts,
            "interpretation": ["Descriptive case analysis; no causal or population-level conclusion.",
                               "PCA axes and RGB scales are independent per image and condition.",
                               "Straight-line arc and screen distances are not independently identifiable.",
                               "Insufficient matched anchors are reported without relaxing thresholds."],
        }
        atomic_json(self.output / "summary.json", summary)
        files = {str(p.relative_to(self.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(self.output.rglob("*"))
                 if p.is_file() and p.name != "output_manifest.json"}
        incomplete = [name for name in files if name.endswith((".partial", ".part"))]
        if incomplete:
            raise ValueError(f"Incomplete files remain: {incomplete}")
        atomic_json(self.output / "output_manifest.json", {
            "status": "complete", "analysis_fingerprint": self.fingerprint,
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values()),
            "files": files,
        })
        print(f"ANALYSIS FILES VERIFIED (mounted Drive; run the sync cell next): {self.output}", flush=True)
        return {"status": "local_verified", "output_dir": str(self.output),
                "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values())}
