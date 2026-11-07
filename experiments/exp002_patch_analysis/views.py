"""PCA and attention views of saved exp001 tensors; no model inference.

PCA fits are independent for each image and each selection. Their RGB axes
must not be compared across images or selections. Attention is decoder causal
self-attention cropped to image keys, not an explanation of causal influence.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np


def _array(value):
    """Accept NumPy arrays or the CPU tensors returned by CaseAnalysis.load."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _write(path, content):
    """Permit identical reruns; never silently replace a different result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"different result already exists: {path}; choose a new output folder")
    else:
        temporary = path.with_name(path.name + ".partial")
        try:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _json(path, content):
    return _write(path, (json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def _npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return _write(path, buffer.getvalue())


def _figure(path, figure, description=None):
    buffer = io.BytesIO()
    metadata = {"Software": "mllm-line-tracing exp002"}
    if description is not None:
        metadata["Description"] = json.dumps(description, ensure_ascii=False)
    figure.savefig(buffer, format="png", dpi=130, metadata=metadata)
    return _write(path, buffer.getvalue())


def pca_rgb(features, selected_indices):
    """Fit three PCs on selected tokens, transform every token, map PCs to RGB.

    Uses a sample-space eigendecomposition (at most 400 x 400 here), not a
    2560 x 2560 covariance decomposition. RGB uses selected-score minimum and
    maximum per component. Constant components are neutral gray. PC signs are
    fixed by making the largest absolute loading positive; tied eigenspaces
    still have no intrinsic orientation. No feature standardization is used.
    """
    values = np.asarray(_array(features), dtype=np.float64)
    selected = np.asarray(selected_indices)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("features must be a finite token x feature matrix with at least 3 features")
    if selected.ndim != 1 or selected.dtype.kind not in "iu":
        raise ValueError("selected_indices must be a one-dimensional integer array")
    selected = selected.astype(np.int64, copy=False)
    if len(selected) < 4 or len(np.unique(selected)) != len(selected):
        raise ValueError("PCA needs at least four distinct selected tokens")
    if np.any(selected < 0) or np.any(selected >= len(values)):
        raise ValueError("selected token index is outside the feature matrix")
    fit = values[selected]
    mean = fit.mean(axis=0)
    centered = fit - mean
    gram = centered @ centered.T / (len(selected) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    total = eigenvalues.sum()
    if not np.isfinite(total) or total <= np.finfo(np.float64).tiny:
        raise ValueError("selected tokens have no PCA variance")
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * eigenvalues[0]
    components = np.zeros((3, values.shape[1]), dtype=np.float64)
    ratios = np.zeros(3, dtype=np.float64)
    for component in range(3):
        variance = eigenvalues[component]
        if variance <= tolerance:
            continue
        vector = eigenvectors[:, order[component]] @ centered / np.sqrt(variance * (len(selected) - 1))
        vector *= 1 if vector[np.argmax(np.abs(vector))] >= 0 else -1
        components[component] = vector
        ratios[component] = variance / total
    scores = (values - mean) @ components.T
    low, high = scores[selected].min(axis=0), scores[selected].max(axis=0)
    rgb = np.full(scores.shape, 0.5, dtype=np.float64)
    active = (high - low) > np.finfo(np.float64).eps * np.maximum(1, np.maximum(abs(low), abs(high)))
    rgb[:, active] = np.clip((scores[:, active] - low[active]) / (high[active] - low[active]), 0, 1)
    return {"rgb": rgb.astype(np.float32), "scores": scores, "components": components,
            "mean": mean, "ratios": ratios, "scaling_low": low, "scaling_high": high,
            "selected_indices": selected}


def _boxes(cells, count, width, height):
    if len(cells) != count or sorted(cell["merged_index"] for cell in cells) != list(range(count)):
        raise ValueError("merged token geometry must contain each feature index exactly once")
    boxes = [None] * count
    for cell in cells:
        raw = np.asarray(cell["bbox_px"], dtype=float)
        if raw.shape != (4,) or not np.isfinite(raw).all() or not np.allclose(raw, np.round(raw)):
            raise ValueError("merged token bounding boxes must use integer pixel boundaries")
        x0, y0, x1, y1 = np.round(raw).astype(int).tolist()
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("merged token bounding box is outside the source image")
        boxes[cell["merged_index"]] = (x0, y0, x1, y1)
    return boxes


def pca_case(H, cells, path_mask, image_path, out_dir):
    """Save two independent PCA fits: all tokens and any-path-pixel tokens.

    A line-containing cell can also contain a colored circle or S/E. Excluding
    those belongs to the separate continuity analysis, not this PCA selection.
    Only selected cells are colored in each PCA panel; all-token transforms are
    retained in NPZ. A three-panel PNG includes the explicit fit selection.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    features = _array(H)
    if features.ndim != 2:
        raise ValueError("H must have shape [merged tokens, hidden dimension]")
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    mask = np.asarray(path_mask)
    if mask.shape != (height, width):
        raise ValueError("path_mask must match the source image height and width")
    boxes = _boxes(cells, len(features), width, height)
    line_indices = np.array([i for i, (x0, y0, x1, y1) in enumerate(boxes)
                             if np.any(mask[y0:y1, x0:x1] > 0)], dtype=np.int64)
    conditions = {"all": np.arange(len(features), dtype=np.int64), "line": line_indices}
    out_dir = Path(out_dir)
    result = {"conditions": {}, "feature_shape": list(features.shape)}
    for condition, indices in conditions.items():
        fit = pca_rgb(features, indices)
        stem = out_dir / f"pca_{condition}"
        npz_path = _npz(stem.with_suffix(".npz"), **fit)
        summary = {"condition": condition, "fit_token_count": len(indices),
                   "total_token_count": len(features), "feature_dimension": features.shape[1],
                   "explained_variance_ratio": fit["ratios"].tolist(),
                   "fit_selection": "all tokens" if condition == "all" else "any positive path-mask pixel in cell",
                   "rgb_scaling": "per-component min-max over fitted tokens", "rgb_scaling_low": fit["scaling_low"].tolist(),
                   "rgb_scaling_high": fit["scaling_high"].tolist(),
                   "note": "Independent per-image, per-condition fit; RGB axes are not comparable across fits."}
        json_path = _json(stem.with_suffix(".json"), summary)
        canvas = np.ones_like(image, dtype=np.float32)
        for index in indices:
            x0, y0, x1, y1 = boxes[index]
            canvas[y0:y1, x0:x1] = fit["rgb"][index]
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        axes[0].imshow(image)
        axes[0].set_title("Source image")
        axes[1].imshow(canvas, interpolation="nearest")
        axes[1].set_title("PC1 / PC2 / PC3 → RGB")
        axes[2].imshow(image)
        for index in indices:
            x0, y0, x1, y1 = boxes[index]
            axes[2].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                        edgecolor="deepskyblue", linewidth=0.45))
        axes[2].set_title(f"Fit selection: {len(indices)} / {len(features)} cells")
        for axis in axes:
            axis.set_axis_off()
        figure.suptitle(f"{condition}: independent PCA | variance ratios " +
                       ", ".join(f"{ratio:.3f}" for ratio in fit["ratios"]))
        figure.tight_layout()
        try:
            png_path = _figure(stem.with_suffix(".png"), figure, summary)
        finally:
            plt.close(figure)
        result["conditions"][condition] = {"png": str(png_path), "npz": str(npz_path),
                                             "json": str(json_path), **summary}
    return result


def _word_records(case):
    """Align each actual color occurrence to the predictor of its first BPE."""
    layout = case.meta["layout"]
    query_positions = layout["attention_query_positions"]
    rows = {position: row for row, position in enumerate(query_positions)}
    if len(rows) != len(query_positions):
        raise ValueError("attention query positions are not unique")
    expected = case.sample["answer_colors"]
    records = []
    for index, span in enumerate(case.meta["generation"]["token_details"]["color_spans"]):
        steps = span["generation_steps"]
        if not steps or len(steps) != len(span["predictor_positions"]):
            raise ValueError("color span has no aligned predictor steps")
        first = int(np.argmin(steps))
        step, position = int(steps[first]), int(span["predictor_positions"][first])
        if layout["predictor_positions"][step] != position or layout["predictor_attention_rows"][step] != rows[position]:
            raise ValueError("color span and saved predictor layout disagree")
        records.append({"word_index": index, "actual_color": span["color"],
                        "expected_color": expected[index] if index < len(expected) else "<extra>",
                        "generation_step": step, "predictor_position": position,
                        "attention_row": rows[position], "bpe_token_count": len(steps)})
    if not records:
        raise ValueError("answer has no recognized color words")
    return records


def _conditional(raw, mass):
    return np.divide(raw, mass[..., None], out=np.full_like(raw, np.nan), where=mass[..., None] > 0)


def _maximum(values):
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else 0.0


def _saved_attention(case, layer):
    weights = _array(case.load(f"decoder/layer_{layer:03d}/attention_to_image"))
    queries = len(case.meta["layout"]["attention_query_positions"])
    count = len(case.geometry["merged_tokens"])
    if weights.ndim != 3 or weights.shape[0] < 1 or weights.shape[1:] != (queries, count):
        raise ValueError(f"unexpected saved attention shape in layer {layer}: {weights.shape}")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"saved attention contains invalid probabilities in layer {layer}")
    return weights


def extract_attention(case, out_dir):
    """Save head-mean attention for every layer and actual color word.

    raw[L,W,P] keeps the original full-sequence denominator. conditional uses
    the same head-mean raw matrix divided by its image mass: mean first, then
    normalize. Zero image mass becomes NaN. This is not the mean of separately
    normalized heads. Only first-BPE predictors are included; end/EOS are not.
    """
    records = _word_records(case)
    rows = [record["attention_row"] for record in records]
    layers = int(case.meta["capture_spec"]["decoder_layers"])
    raw, heads = [], None
    for layer in range(layers):
        weights = _saved_attention(case, layer)
        if heads is not None and weights.shape[0] != heads:
            raise ValueError("head count differs between decoder layers")
        heads = weights.shape[0]
        selected = weights[:, rows, :].astype(np.float32).mean(axis=0)
        if not np.isfinite(selected).all() or np.any(selected < 0):
            raise ValueError("saved attention contains invalid probabilities")
        raw.append(selected)
        del weights
    raw = np.stack(raw)
    mass = raw.sum(axis=-1, dtype=np.float32)
    conditional = _conditional(raw, mass)
    sample_id = case.sample["id"]
    npz_path = _npz(Path(out_dir) / "attention_head_mean.npz", raw=raw, conditional=conditional, mass=mass,
                    actual_colors=np.array([r["actual_color"] for r in records]),
                    expected_colors=np.array([r["expected_color"] for r in records]),
                    generation_steps=np.array([r["generation_step"] for r in records]),
                    predictor_positions=np.array([r["predictor_position"] for r in records]),
                    attention_rows=np.array(rows), sample_id=np.array(sample_id), head_count=np.array(heads))
    summary = {"sample_id": sample_id, "layers": layers, "word_count": len(records), "head_count": heads,
               "raw_shape": list(raw.shape), "conditional_shape": list(conditional.shape),
               "mass_shape": list(mass.shape), "raw_vmax": _maximum(raw),
               "conditional_vmax": _maximum(conditional), "words": records,
               "alignment": "predictor of first BPE of each actual generated color occurrence",
               "conditional_definition": "head-mean raw attention divided by its image mass; zero mass is NaN",
               "note": "Image-key crop of causal self-attention; attention alone does not establish causality."}
    json_path = _json(Path(out_dir) / "attention_head_mean.json", summary)
    return {"npz": str(npz_path), "json": str(json_path), **summary}


def _spatial_grid(values, cells):
    """Map by recorded bounding boxes rather than assume token sequence order."""
    xs = sorted({cell["bbox_px"][0] for cell in cells})
    ys = sorted({cell["bbox_px"][1] for cell in cells})
    widths = {cell["bbox_px"][2] - cell["bbox_px"][0] for cell in cells}
    heights = {cell["bbox_px"][3] - cell["bbox_px"][1] for cell in cells}
    if len(widths) != 1 or len(heights) != 1 or len(xs) * len(ys) != len(cells):
        raise ValueError("attention overlay requires a complete uniform merged-token grid")
    grid = np.full((len(ys), len(xs)), np.nan)
    for cell in cells:
        x0, y0, _, _ = cell["bbox_px"]
        grid[ys.index(y0), xs.index(x0)] = values[cell["merged_index"]]
    return grid, (xs[0], xs[-1] + next(iter(widths)), ys[-1] + next(iter(heights)), ys[0])


def render_attention(case, output_npz, output_dir, raw_vmax, conditional_vmax,
                     word_index, mode="raw", head=None, save=True):
    """One 6x6 all-layer image overlay for one color predictor.

    Mean-head grids use caller-supplied shared cohort limits. An explicitly
    selected head is read lazily from the original saved tensors and uses one
    common limit across its 36 layers. save=False returns an open Figure.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if mode not in {"raw", "conditional"}:
        raise ValueError("mode must be raw or conditional")
    records = _word_records(case)
    if not 0 <= word_index < len(records):
        raise IndexError("word_index is outside the generated color list")
    record = records[word_index]
    with np.load(output_npz, allow_pickle=False) as data:
        if str(data["sample_id"].item()) != case.sample["id"]:
            raise ValueError("attention NPZ belongs to another sample")
        if data["predictor_positions"][word_index] != record["predictor_position"]:
            raise ValueError("attention NPZ predictor alignment differs from the case")
        layers = data["raw"].shape[0]
        if head is None:
            values = data[mode][:, word_index, :].copy()
            mass = data["mass"][:, word_index].copy()
    if not 1 <= layers <= 36:
        raise ValueError("the all-layer grid supports 1 to 36 saved decoder layers")
    if head is not None:
        if not isinstance(head, (int, np.integer)) or head < 0:
            raise ValueError("head must be a nonnegative integer")
        head = int(head)
        raw_rows = []
        for layer in range(layers):
            weights = _saved_attention(case, layer)
            if head >= weights.shape[0]:
                raise ValueError("head is outside the saved head range")
            raw_rows.append(weights[head, record["attention_row"]].astype(np.float32).copy())
            del weights
        raw_rows = np.stack(raw_rows)
        mass = raw_rows.sum(axis=-1)
        values = raw_rows if mode == "raw" else _conditional(raw_rows, mass)
        vmax = _maximum(values)
        scale_label = "shared across this head's layers"
    else:
        vmax = float(raw_vmax if mode == "raw" else conditional_vmax)
        scale_label = "shared across selected cases / words / layers"
    if not np.isfinite(vmax) or vmax < 0:
        raise ValueError("attention color limit must be finite and nonnegative")
    display_vmax = vmax if vmax > 0 else 1.0
    with Image.open(case.image_path) as source:
        image = np.asarray(source.convert("RGB"))
    figure, axes = plt.subplots(6, 6, figsize=(16, 17), constrained_layout=True)
    artist = None
    for layer, axis in enumerate(axes.flat):
        if layer >= layers:
            axis.set_visible(False)
            continue
        grid, extent = _spatial_grid(values[layer], case.geometry["merged_tokens"])
        axis.imshow(image)
        artist = axis.imshow(grid, extent=extent, origin="upper", interpolation="nearest",
                             cmap="magma", vmin=0, vmax=display_vmax, alpha=0.45)
        axis.set_title(f"Layer {layer + 1:02d} | image mass {mass[layer]:.4f}", fontsize=9)
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_axis_off()
    head_label = "head mean" if head is None else f"head {head}"
    figure.suptitle(f"{case.sample['id']} | word {word_index}: actual {record['actual_color']} / "
                   f"expected {record['expected_color']}\nfirst-BPE predictor position {record['predictor_position']} | "
                   f"{mode} | {head_label} | {scale_label}", fontsize=12)
    figure.colorbar(artist, ax=[a for a in axes.flat if a.get_visible()], shrink=0.65,
                    label="Original attention probability" if mode == "raw" else "Probability conditional on image keys")
    if not save:
        return figure
    suffix = "mean" if head is None else f"head_{head:03d}"
    path = Path(output_dir) / f"word_{word_index:02d}_{record['actual_color']}_{mode}_{suffix}.png"
    try:
        return _figure(path, figure, {**record, "mode": mode, "head": head, "vmax": vmax,
                                      "scale": scale_label})
    finally:
        plt.close(figure)
