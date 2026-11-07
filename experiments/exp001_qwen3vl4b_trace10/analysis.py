"""CPU analysis of exp001 captures; no model weights or GPU are loaded.

Release metadata is downloaded first. Raw tensors are opt-in, one case archive
at a time. Every archive and loaded tensor is verified against its saved SHA256.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile


REPOSITORY = "seungjoolee24/mllm-line-tracing"
DEFAULT_RELEASE = "exp001-greedy-run002"
PROTOCOL = "qwen3vl_trace_case_zip_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _request(url):
    return urllib.request.Request(url, headers={"User-Agent": "mllm-line-tracing-analysis"})


def _download(url, destination, digest, expected_bytes=None):
    """Stream an artifact to disk, then verify before publishing the cache file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == digest:
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            return destination
    temporary = destination.with_name(destination.name + ".part")
    try:
        with urllib.request.urlopen(_request(url), timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
            raise ValueError(f"download size mismatch: {destination.name}")
        if sha256_file(temporary) != digest:
            raise ValueError(f"download SHA256 mismatch: {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _extract_verified_archive(archive, destination, prefix):
    """Extract to disk without traversal/symlinks or replacing different data."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    expected_prefix = Path(prefix).parts
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = Path(member.filename).parts
            if Path(member.filename).is_absolute() or ".." in parts:
                raise ValueError("archive contains an unsafe path")
            if parts[:len(expected_prefix)] != expected_prefix:
                raise ValueError(f"archive path is outside declared prefix: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("archive contains a symbolic link")
            target = _inside(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with source.open(member) as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if target.stat().st_size != member.file_size or sha256_file(target) != digest.hexdigest():
                    raise ValueError(f"existing extracted file differs: {target}")
                continue
            temporary = target.with_name(target.name + ".extracting")
            try:
                with source.open(member) as stream, temporary.open("wb") as out:
                    shutil.copyfileobj(stream, out, length=1024 * 1024)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


class ReleaseBundle:
    """Public GitHub release reader with metadata-first, per-case downloads."""

    def __init__(self, cache, tag=DEFAULT_RELEASE, repository=REPOSITORY,
                 expected_manifest_sha256=None):
        self.cache = Path(cache).resolve() / tag
        self.cache.mkdir(parents=True, exist_ok=True)
        url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            release = json.load(response)
        self.assets = {asset["name"]: asset for asset in release["assets"]}
        manifest_asset = self.assets.get("artifact_manifest.json")
        if manifest_asset is None:
            raise ValueError("release has no artifact_manifest.json")
        github_digest = manifest_asset.get("digest") or ""
        github_digest = github_digest.removeprefix("sha256:") if github_digest.startswith("sha256:") else None
        if expected_manifest_sha256 and github_digest and expected_manifest_sha256 != github_digest:
            raise ValueError("pinned manifest digest differs from GitHub release digest")
        digest = expected_manifest_sha256 or github_digest
        if not digest or len(digest) != 64:
            raise ValueError("manifest SHA256 unavailable; set EXPECTED_MANIFEST_SHA256 from the published release")
        self.manifest_path = _download(manifest_asset["browser_download_url"],
                                       self.cache / "artifact_manifest.json", digest,
                                       manifest_asset.get("size"))
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("package_protocol") != PROTOCOL:
            raise ValueError("unsupported release package protocol")
        if self.manifest.get("collection_schema_version") != "1.1":
            raise ValueError("release must contain protocol 1.1 greedy captures")
        if self.manifest.get("status") != "complete":
            raise ValueError("release run is not complete")
        self.run_name = self.manifest["run_name"]
        if self.run_name != "run002":
            raise ValueError("this notebook expects run002")
        self.run_dir = self.cache / self.run_name
        self.manifest_sha256 = digest

    def _restore(self, entry):
        asset = self.assets.get(entry["file"])
        if asset is None:
            raise ValueError(f"missing release asset: {entry['file']}")
        archive = _download(asset["browser_download_url"], self.cache / entry["file"],
                            entry["sha256"], entry["bytes"])
        _extract_verified_archive(archive, self.cache, entry["archive_path_prefix"])
        return self.run_dir

    def metadata(self):
        entries = [entry for entry in self.manifest["archives"] if entry["kind"] == "metadata"]
        if len(entries) != 1:
            raise ValueError("release must contain exactly one metadata archive")
        return self._restore(entries[0])

    def case(self, sample_id="case_03_swirl"):
        """Explicitly download one case, never every case in the release."""
        entries = [entry for entry in self.manifest["archives"]
                   if entry["kind"] == "case" and entry["sample_id"] == sample_id]
        if len(entries) != 1:
            raise ValueError(f"case archive is missing or ambiguous: {sample_id}")
        self._restore(entries[0])
        return self.run_dir / "samples" / sample_id

    def inventory(self):
        return [{"file": row["file"], "kind": row["kind"], "sample_id": row.get("sample_id"),
                 "MiB": row["bytes"] / 1024**2, "sha256": row["sha256"]}
                for row in self.manifest["archives"]]


def _validate_metadata(meta, allow_legacy=False):
    if meta.get("schema_version") != "1.1":
        if not allow_legacy or meta.get("schema_version") != "1.0" or meta.get("status") != "complete":
            raise ValueError("protocol 1.0 is excluded: checkpoint sampling defaults were inherited")
        return
    if meta.get("status") != "complete":
        raise ValueError("case capture is incomplete")
    generation = meta["generation"]
    effective = generation["effective_generation_config"]
    if generation.get("use_model_defaults") is not False or effective.get("do_sample") is not False:
        raise ValueError("case is not verified explicit greedy generation")
    if effective.get("num_beams", 1) != 1 or effective.get("repetition_penalty", 1) != 1:
        raise ValueError("case generation policy differs from the neutral greedy protocol")
    alignment = meta["alignment"]
    if not alignment.get("replay_matches_generation_argmax") or not alignment.get("generation_raw_argmax_matches_emitted"):
        raise ValueError("case does not have verified generation/replay alignment")


def dataset_samples(dataset):
    return {sample["id"]: sample for sample in
            (json.loads(line) for line in (Path(dataset) / "annotations.jsonl").read_text().splitlines() if line.strip())}


def overview(run, dataset, allow_legacy=False):
    """Ten-case behavioral summary, using only small JSON metadata files."""
    run = Path(run)
    samples = dataset_samples(dataset)
    config = read_json(run / "run_config.json")
    if config.get("annotations_sha256") != sha256_file(Path(dataset) / "annotations.jsonl"):
        raise ValueError("dataset annotations differ from the collection configuration")
    rows = []
    for path in sorted((run / "samples").glob("*/meta.json")):
        meta = read_json(path)
        _validate_metadata(meta, allow_legacy=allow_legacy)
        sample = samples[meta["sample_id"]]
        score = meta["score"]
        rows.append({"sample_id": meta["sample_id"], "family": meta["family"],
                     "expected": sample["answer"], "answer": meta["generation"]["raw_answer"],
                     "exact_match": score["exact_match"], "correct_prefix_colors": score["correct_prefix_colors"],
                     "num_objects": len(sample["objects"]), "format_valid": score["format_valid"],
                     "raw_argmax_alignment": meta["alignment"].get("generation_raw_argmax_matches_emitted"),
                     "replay_argmax_alignment": meta["alignment"].get("replay_matches_generation_argmax"),
                     "schema_version": meta["schema_version"],
                     "generated_tokens": meta["layout"]["generated_length"],
                     "tensor_MiB": meta["storage"]["file_bytes"] / 1024**2})
    return rows


class CaseAnalysis:
    """Lazy, one-tensor-at-a-time reader. Outputs are NumPy arrays on CPU."""

    def __init__(self, run, sample_id, dataset, allow_legacy=False):
        self.run, self.dataset = Path(run), Path(dataset)
        self.directory = self.run / "samples" / sample_id
        self.meta = read_json(self.directory / "meta.json")
        _validate_metadata(self.meta, allow_legacy=allow_legacy)
        self.geometry = read_json(self.directory / "token_geometry.json")
        self.sample = dataset_samples(dataset)[sample_id]
        self.records = {row["name"]: row for row in self.meta["tensors"]}
        self.verified = set()
        self.image_positions = self.meta["layout"]["image_positions"]
        self.objects = sorted(self.sample["objects"], key=lambda item: item["sequence_index"])
        self.image_path = _inside(self.dataset, self.sample["files"]["image"])
        if sha256_file(self.image_path) != self.meta["input"]["image_sha256"]:
            raise ValueError("source image differs from the collected image")
        grid = self.geometry["image_grid_thw"]
        if len(grid) != 1 or grid[0][0] != 1:
            raise ValueError("analysis expects exactly one still image")
        self.grid_height, self.grid_width = grid[0][1:]
        self.merge_size = self.geometry["merge_size"]
        if len(self.geometry["vision_tokens"]) != self.grid_height * self.grid_width:
            raise ValueError("pre-merger token geometry has the wrong length")
        if len(self.geometry["merged_tokens"]) != len(self.image_positions):
            raise ValueError("merged token geometry has the wrong length")

    def load(self, name):
        import torch
        row = self.records[name]
        path = _inside(self.directory, row["file"])
        if name not in self.verified:
            if path.stat().st_size != row["file_bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError(f"tensor integrity check failed: {name}")
            self.verified.add(name)
        tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if list(tensor.shape) != row["shape"] or str(tensor.dtype) != row["dtype"]:
            raise ValueError(f"tensor shape or dtype differs from metadata: {name}")
        return tensor

    def mapping_summary(self):
        return {"image_grid_thw": self.geometry["image_grid_thw"],
                "vision_patches": len(self.geometry["vision_tokens"]),
                "merged_visual_tokens": len(self.geometry["merged_tokens"]),
                "decoder_image_positions": len(self.image_positions),
                "vision_sequence_order": self.geometry["vision_sequence_order"],
                "capture_semantics": self.meta["capture_semantics"]}

    def activation_statistics(self, stage="decoder", phase="output"):
        """Per-position L2 norm and cosine to previous layer; do not pool heads/tokens."""
        import numpy as np
        import torch
        import torch.nn.functional as functional
        if stage not in {"vision", "decoder"} or phase not in {"input", "output"}:
            raise ValueError("stage must be vision/decoder and phase input/output")
        count = self.meta["capture_spec"]["vision_layers" if stage == "vision" else "decoder_layers"]
        norms, cosines, previous = [], [], None
        for layer in range(count):
            tail = "output_before_deepstack" if stage == "decoder" and phase == "output" else phase
            prefix = "block" if stage == "vision" else "layer"
            tensor = self.load(f"{stage}/{prefix}_{layer:03d}/{tail}")
            values = tensor.squeeze(0).float()
            norms.append(values.norm(dim=-1).numpy().copy())
            cosines.append(np.full(values.shape[0], np.nan) if previous is None else
                           functional.cosine_similarity(previous, values, dim=-1).numpy().copy())
            previous = values
            del tensor
        del previous
        gc.collect()
        return {"stage": stage, "phase": phase, "norm": np.stack(norms),
                "cosine_previous_layer": np.stack(cosines),
                "note": "decoder outputs are before same-layer DeepStack injection; layer transitions include that injection"}

    def position_groups(self):
        layout = self.meta["layout"]
        return {"image": layout["image_positions"], "question": layout["question_positions"],
                "answer_predictor": layout["predictor_positions"], "generated_token": layout["generated_positions"]}

    def activation_group_curves(self, stats):
        """Descriptive norms, not semantic activation strength or causal importance."""
        if stats["stage"] != "decoder":
            raise ValueError("question and answer position groups exist only in the decoder")
        return {key: stats["norm"][:, positions].mean(axis=1)
                for key, positions in self.position_groups().items() if positions}

    def attention_rows(self, layer, kind="predictor", head=None):
        """Return query×image values, preserving the original full-sequence denominator."""
        names = {"predictor": "predictor_attention_rows", "generated": "generated_attention_rows",
                 "question": "question_attention_rows"}
        if kind not in names:
            raise ValueError("kind must be predictor, generated, or question")
        weights = self.load(f"decoder/layer_{layer:03d}/attention_to_image")
        if head is not None and not 0 <= head < weights.shape[0]:
            raise ValueError("head index is outside the saved head range")
        rows = self.meta["layout"][names[kind]]
        selected = weights[:, rows, :].float()
        values = selected.mean(dim=0) if head is None else selected[head]
        result = values.numpy().copy()
        del weights, selected, values
        return result

    def object_attention(self, layer, kind="predictor", head=None):
        """Generated color occurrence×ground-truth object coverage-weighted mass.

        Coarse token attention is treated as uniform within each merged cell.
        This is a spatial projection, not pixel-resolution attention or causality.
        """
        import numpy as np
        if kind not in {"predictor", "generated"}:
            raise ValueError("color alignment is defined for predictor or generated rows")
        values = self.attention_rows(layer, kind=kind, head=head)
        spans = self.meta["generation"]["token_details"]["color_spans"]
        coverage = np.asarray([self.geometry["object_fraction_of_merged_cell_area"][obj["id"]]
                               for obj in self.objects], dtype=np.float64)
        if not spans:
            return np.empty((0, len(self.objects))), []
        aligned = np.stack([values[span["generation_steps"]].mean(axis=0) for span in spans])
        return aligned @ coverage.T, [span["color"] for span in spans]

    def spatial_grid(self, values, stage="merged"):
        import numpy as np
        values = np.asarray(values)
        if stage == "vision":
            mapping = self.geometry["vision_tokens"]
            grid = np.full((self.grid_height, self.grid_width), np.nan)
            if len(values) != len(mapping):
                raise ValueError("vision values must match the pre-merger patch sequence")
            for item in mapping:
                grid[item["row"], item["column"]] = values[item["vision_index"]]
            return grid
        height, width = self.grid_height // self.merge_size, self.grid_width // self.merge_size
        if len(values) != height * width:
            raise ValueError("merged values must match the decoder image-token count")
        return values.reshape(height, width)

    def plot_overlay(self, values, title, stage="merged", cmap="magma", ax=None, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        image = Image.open(self.image_path)
        ax.imshow(image)
        grid = self.spatial_grid(values, stage)
        artist = ax.imshow(grid, extent=(0, image.width, image.height, 0), interpolation="nearest",
                           alpha=0.62, cmap=cmap,
                           vmin=float(np.nanmin(grid)) if vmin is None else vmin,
                           vmax=float(np.nanmax(grid)) if vmax is None else vmax)
        ax.set(title=title, xlim=(0, image.width), ylim=(image.height, 0))
        ax.set_axis_off()
        plt.colorbar(artist, ax=ax, shrink=0.72)
        return ax

    def plot_attention(self, layer, kind="predictor", step=0, head=None):
        rows = self.attention_rows(layer, kind, head)
        values = rows.mean(axis=0) if step is None else rows[step]
        label = "mean of all heads" if head is None else f"head {head}"
        return self.plot_overlay(values, f"L{layer} {kind} row {step} | {label}")

    def plot_object_attention(self, layer, kind="predictor", head=None):
        import matplotlib.pyplot as plt
        import numpy as np
        matrix, predicted = self.object_attention(layer, kind, head)
        if not len(predicted):
            raise ValueError("answer has no recognized color spans")
        _, ax = plt.subplots(figsize=(max(7, len(self.objects)), max(4.5, 0.5 * len(predicted))))
        image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0)
        ax.set_xticks(range(len(self.objects)), [f"{i}: {obj['color']}" for i, obj in enumerate(self.objects)], rotation=45)
        ax.set_yticks(range(len(predicted)), [f"{i}: {color}" for i, color in enumerate(predicted)])
        ax.set(xlabel="Ground-truth object order", ylabel=f"Color occurrence / {kind}",
               title=f"Layer {layer}: coverage-weighted image attention mass")
        for index in range(min(matrix.shape)):
            ax.plot(index, index, marker="s", markersize=18, fillstyle="none", color="cyan")
        plt.colorbar(image, ax=ax)
        plt.tight_layout()
        return ax

    def plot_layer_statistics(self, stats, log_norm=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        _, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, key, label in zip(axes, ("norm", "cosine_previous_layer"),
                                  ("L2 norm", "Cosine to previous layer")):
            options = {"norm": LogNorm()} if key == "norm" and log_norm else {}
            if key == "cosine_previous_layer":
                options = {"vmin": -1, "vmax": 1}
            artist = ax.imshow(stats[key], aspect="auto", interpolation="nearest", cmap="viridis", **options)
            if key == "norm" and log_norm:
                label += " (log color scale)"
            ax.set(xlabel=f"{stats['stage']} sequence position", ylabel="Layer index", title=label)
            plt.colorbar(artist, ax=ax)
        plt.tight_layout()
        return axes

    def plot_group_curves(self, stats):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(9, 4))
        for name, values in self.activation_group_curves(stats).items():
            ax.plot(values, label=name)
        ax.set(xlabel="Decoder layer index", ylabel="Mean token L2 norm", title="Position groups (descriptive only)")
        ax.legend()
        return ax

    def attention_layer_profile(self, kind="predictor", head=None):
        """Layer×query total attention probability on image tokens, before pooling image keys."""
        import numpy as np
        return np.stack([self.attention_rows(layer, kind, head).sum(axis=-1)
                         for layer in range(self.meta["capture_spec"]["decoder_layers"])])

    def plot_attention_layer_profile(self, kind="predictor", head=None):
        import matplotlib.pyplot as plt
        values = self.attention_layer_profile(kind, head)
        _, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set(xlabel=f"{kind} query index", ylabel="Decoder layer", title="Attention mass on all image tokens")
        plt.colorbar(image, ax=ax)
        return ax
