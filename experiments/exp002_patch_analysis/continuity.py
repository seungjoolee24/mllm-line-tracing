"""Exact centerline geometry and full-feature cosine continuity diagnostics.

No model execution, dimensionality reduction, threshold relaxation, or hypothesis
tests are performed here. Matching depends exclusively on stored image geometry.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = {
    "min_line_coverage": 0.01,
    "exclusion_margin_px": 2.0,
    "near_arc_max_px": 64.0,
    "far_arc_min_px": 128.0,
    "max_screen_distance_error_px": 8.0,
    "max_coverage_error": 0.03,
    "min_matched_anchors": 5,
}
_EPS = 1e-8  # Floating-point boundary tolerance, not a matching threshold.


def _configuration(config):
    result = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown continuity settings: {sorted(unknown)}")
        result.update(config)
    for key, value in result.items():
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
        result[key] = int(value) if key == "min_matched_anchors" else float(value)
    if result["min_matched_anchors"] < 1 or result["min_matched_anchors"] != result.get("min_matched_anchors"):
        raise ValueError("min_matched_anchors must be a positive integer")
    if config is not None and "min_matched_anchors" in config and int(config["min_matched_anchors"]) != config["min_matched_anchors"]:
        raise ValueError("min_matched_anchors must be a positive integer")
    if result["min_line_coverage"] > 1 or result["max_coverage_error"] > 1:
        raise ValueError("Coverage settings must lie between zero and one")
    if not 0 < result["near_arc_max_px"] < result["far_arc_min_px"]:
        raise ValueError("Require 0 < near_arc_max_px < far_arc_min_px")
    return result


def _path_arrays(points, cumulative=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Centerline points must be a finite array with shape [N>=2,2]")
    computed = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if computed[-1] <= 0:
        raise ValueError("The centerline must have positive length")
    if cumulative is not None:
        supplied = np.asarray(cumulative, dtype=np.float64)
        if supplied.shape != computed.shape or not np.isfinite(supplied).all() or not np.allclose(supplied, computed, rtol=1e-8, atol=1e-6):
            raise ValueError("Stored cumulative lengths do not match the exact centerline")
    return points, computed


def _rectangle(bbox):
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Cell bbox_px must be finite [x0,y0,x1,y1] with positive area")
    return box


def _clip_segment(a, b, box):
    """Liang--Barsky clipping, retaining exact tangent/endpoint contacts."""
    low, high = 0.0, 1.0
    delta = b - a
    for axis in (0, 1):
        if delta[axis] == 0:
            if a[axis] < box[axis] or a[axis] > box[axis + 2]:
                return None
            continue
        enter = (box[axis] - a[axis]) / delta[axis]
        leave = (box[axis + 2] - a[axis]) / delta[axis]
        low = max(low, min(enter, leave))
        high = min(high, max(enter, leave))
        if high < low:
            return None
    return low, high


def polyline_cell_intervals(points, bbox, cumulative=None):
    """Return connected arc-length intervals inside a closed cell rectangle.

    Adjacent/touching arc intervals merge. A later re-entry with positive arc
    length outside the cell remains a separate interval, even at the same point.
    """
    points, arc = _path_arrays(points, cumulative)
    box = _rectangle(bbox)
    intervals = []
    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        clipped = _clip_segment(a, b, box)
        if clipped is None:
            continue
        length = arc[index + 1] - arc[index]
        # Canonical segment endpoints give adjacent clips the identical arc
        # coordinate; a positive outside gap is never merged by a tolerance.
        start, end = (float(arc[index] if t == 0 else arc[index + 1] if t == 1 else arc[index] + t * length)
                      for t in clipped)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def interpolate_path(points, s, cumulative=None):
    """Interpolate the exact polyline at scalar arc length s (in pixels)."""
    points, arc = _path_arrays(points, cumulative)
    s = float(s)
    if not np.isfinite(s) or s < -_EPS or s > arc[-1] + _EPS:
        raise ValueError("Arc coordinate lies outside the centerline")
    s = np.clip(s, 0.0, arc[-1])
    return np.array([np.interp(s, arc, points[:, axis]) for axis in (0, 1)])


def select_geometry_matches(arc_positions, path_positions, coverages, config=None):
    """Choose one deterministic (anchor,near,far) triple per eligible anchor.

    No activations are accepted by this function. Among valid candidates use
    lexicographic (screen error, coverage error, near arc distance, far arc
    distance, near index, far index), making ties reproducible.
    """
    settings = _configuration(config)
    s, positions, coverage = (np.asarray(value, dtype=np.float64)
                              for value in (arc_positions, path_positions, coverages))
    if s.ndim != 1 or positions.shape != (len(s), 2) or coverage.shape != s.shape:
        raise ValueError("Expected arc[N], positions[N,2], coverage[N]")
    if not all(np.isfinite(value).all() for value in (s, positions, coverage)):
        raise ValueError("Matching geometry must be finite")
    matches = []
    for anchor in range(len(s)):
        arc_distance = np.abs(s - s[anchor])
        screen = np.linalg.norm(positions - positions[anchor], axis=1)
        near = np.flatnonzero((arc_distance > 0) & (arc_distance <= settings["near_arc_max_px"]))
        far = np.flatnonzero(arc_distance >= settings["far_arc_min_px"])
        candidates = []
        for j in near:
            for k in far:
                screen_error = abs(screen[j] - screen[k])
                coverage_error = abs(coverage[j] - coverage[k])
                if screen_error <= settings["max_screen_distance_error_px"] and coverage_error <= settings["max_coverage_error"]:
                    candidates.append((float(screen_error), float(coverage_error), float(arc_distance[j]),
                                       float(arc_distance[k]), int(j), int(k)))
        if not candidates:
            continue
        screen_error, coverage_error, near_arc, far_arc, j, k = min(candidates)
        matches.append({"anchor": anchor, "near": j, "far": k,
                        "near_arc_px": near_arc, "far_arc_px": far_arc,
                        "near_screen_px": float(screen[j]), "far_screen_px": float(screen[k]),
                        "screen_distance_error_px": screen_error, "coverage_error": coverage_error})
    return matches


def _circle_intersects(box, center, radius):
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius < 0:
        raise ValueError("Object/endpoint circles must have finite centers and nonnegative radii")
    nearest = np.clip(center, box[:2], box[2:])
    return bool(np.dot(center - nearest, center - nearest) <= radius * radius)


def _cosine_matrix(features):
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(features, norms[:, None], out=np.zeros_like(features), where=norms[:, None] > 0)
    matrix = np.clip(normalized @ normalized.T, -1.0, 1.0)
    invalid = norms == 0
    matrix[invalid, :] = np.nan
    matrix[:, invalid] = np.nan
    return matrix, norms


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value):
    return float(value) if np.isfinite(value) else None


def _straight_path(points, arc):
    chord = np.linalg.norm(points[-1] - points[0])
    return bool(np.isclose(arc[-1], chord, rtol=1e-9, atol=1e-7))


def _geometry_rows(cells, count):
    """Validate sequence indices without ever reshaping tokens into a raster.

    Qwen3-VL pre-merger vision rows follow 2x2 spatial merge blocks, then inner
    row-major order. Its vision_index takes precedence over parent merged_index.
    """
    if isinstance(cells, dict):
        candidates = [cells[key] for key in ("vision_tokens", "merged_tokens")
                      if key in cells and len(cells[key]) == count]
        if len(candidates) != 1:
            raise ValueError("Geometry must identify one token sequence matching H rows")
        cells = candidates[0]
    cells = list(cells)
    if len(cells) != count:
        raise ValueError("H rows and geometry cells have different lengths")
    has_vision = ["vision_index" in cell for cell in cells]
    if any(has_vision) and not all(has_vision):
        raise ValueError("Every pre-merge cell must have a vision_index")
    if all(has_vision):
        kind, index_field = "pre_merge_vision", "vision_index"
    elif all("merged_index" in cell for cell in cells):
        kind, index_field = "merged_visual", "merged_index"
    elif any("merged_index" in cell for cell in cells):
        raise ValueError("Geometry indices must be present consistently")
    else:
        kind, index_field = "cell_rows", None
    if index_field is not None:
        indices = [cell[index_field] for cell in cells]
        if indices != list(range(count)) or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in indices):
            raise ValueError(f"{index_field} must match H row order exactly; geometry is reordered or incomplete")
    boxes = [_rectangle(cell["bbox_px"]) for cell in cells]
    grid_rows, grid_columns = None, None
    has_coordinates = [("row" in cell or "column" in cell) for cell in cells]
    if kind == "pre_merge_vision" and count:
        if any(has_coordinates):
            if not all("row" in cell and "column" in cell for cell in cells):
                raise ValueError("Pre-merge row/column coordinates must be present consistently")
            coordinates = [(cell["row"], cell["column"]) for cell in cells]
        else:
            # bbox-only callers still get full block-order validation; spatial
            # coordinates are inferred from bbox origins, never H.reshape(...).
            columns = {value: index for index, value in enumerate(sorted({box[0] for box in boxes}))}
            rows = {value: index for index, value in enumerate(sorted({box[1] for box in boxes}))}
            coordinates = [(rows[box[1]], columns[box[0]]) for box in boxes]
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0
               for pair in coordinates for value in pair) or len(set(coordinates)) != count:
            raise ValueError("Pre-merge row/column coordinates must be unique nonnegative integers")
        expected = sorted(coordinates, key=lambda rc: (rc[0] // 2, rc[1] // 2, rc[0] % 2, rc[1] % 2))
        if coordinates != expected:
            raise ValueError("Pre-merge geometry must preserve spatial merge block-major sequence order")
        grid_rows = max(row for row, _ in coordinates) + 1
        grid_columns = max(column for _, column in coordinates) + 1
    widths = np.array([box[2] - box[0] for box in boxes])
    heights = np.array([box[3] - box[1] for box in boxes])
    if kind == "pre_merge_vision" and grid_rows is not None:
        if grid_rows % 2 or grid_columns % 2 or grid_rows * grid_columns != count:
            raise ValueError("Pre-merge geometry must describe the complete even-sized spatial grid")
        if not np.all(widths == widths[0]) or not np.all(heights == heights[0]):
            raise ValueError("Pre-merge cell sizes must be constant across the spatial grid")
        origin_x = min(box[0] for box in boxes)
        origin_y = min(box[1] for box in boxes)
        for cell, box, (row, column) in zip(cells, boxes, coordinates):
            expected_box = np.array([origin_x + column * widths[0], origin_y + row * heights[0],
                                     origin_x + (column + 1) * widths[0], origin_y + (row + 1) * heights[0]])
            if not np.array_equal(box, expected_box):
                raise ValueError("Pre-merge bbox coordinates disagree with stored row/column geometry")
            parent = (row // 2) * (grid_columns // 2) + column // 2
            if "merged_index" in cell and cell["merged_index"] != parent:
                raise ValueError("Pre-merge parent merged_index disagrees with spatial merge blocks")
    representation = {
        "geometry_kind": kind, "geometry_index_field": index_field or "input row position",
        "spatial_grid_rows": grid_rows, "spatial_grid_columns": grid_columns,
        "patch_width_px": float(widths[0]) if len(widths) and np.all(widths == widths[0]) else None,
        "patch_height_px": float(heights[0]) if len(heights) and np.all(heights == heights[0]) else None,
        "sequence_order": "spatial merge block-major (merge size 2), then inner patch row-major" if kind == "pre_merge_vision" else "explicit input geometry row order",
        "row_alignment": "H row i corresponds to geometry_index i; no raster reshape is used",
    }
    return cells, boxes, representation


def _plots(out, image_path, mask_shape, boxes, selected, clean, raw, centered, pairs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    height, width = mask_shape
    image = plt.imread(image_path) if image_path is not None else np.ones((height, width, 3))
    if image.shape[:2] != (height, width):
        raise ValueError("The image and path mask must have identical pixel dimensions")
    clean_set = set(clean)
    for filename, show_clean in (("selected_patches.png", False), ("clean_patches.png", True)):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image, origin="upper", extent=(0, width, height, 0))
        for index in selected:
            x0, y0, x1, y1 = boxes[index]
            color = "#008b45" if index in clean_set else "#d62728"
            if not show_clean:
                color = "#0072b2"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=color, linewidth=1.2,
                                   linestyle="-" if index in clean_set or not show_clean else "--"))
            ax.text(x0 + 1, y0 + 6, str(index), fontsize=5, color=color)
        ax.set(xlim=(0, width), ylim=(height, 0), title=("Clean patches (green); excluded line patches (red)" if show_clean else "All patches with path-mask pixels"))
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, matrix, title in zip(axes, (raw, centered), ("Raw full-feature cosine", "Centered full-feature cosine")):
        if len(matrix):
            plot = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", origin="upper")
            fig.colorbar(plot, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "No clean patches", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Clean patch in arc order", ylabel="Clean patch in arc order")
    fig.tight_layout()
    fig.savefig(out / "similarity_matrices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ("raw_cosine", "centered_cosine"), ("Raw full-feature cosine", "Centered full-feature cosine")):
        if pairs:
            values = np.array([np.nan if row[key] is None else row[key] for row in pairs])
            scatter = ax.scatter([row["arc_distance_px"] for row in pairs], values,
                                 c=[row["screen_distance_px"] for row in pairs], s=12, alpha=0.65, cmap="viridis")
            fig.colorbar(scatter, ax=ax, label="Screen distance (px)")
        else:
            ax.text(0.5, 0.5, "Fewer than two clean patches", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Arc distance (px)", ylabel="Cosine similarity", ylim=(-1.05, 1.05))
    fig.tight_layout()
    fig.savefig(out / "all_pairs_scatter.png", dpi=150)
    plt.close(fig)


def analyze_continuity(H, cells, path_mask, sample, image_path, out_dir, config=None):
    """Write geometry, cosine plots, pair/triple tables and a compact JSON summary.

    H rows and cells must have identical order. Production continuity input is
    pre-merger [1600,1024] with its explicit block-major vision token geometry;
    other arrays are accepted for explicit geometry/numerical unit tests. The
    centered representation subtracts the mean of *all* H rows before per-token
    L2 normalization. PCA is never involved in these quantitative calculations.
    """
    settings = _configuration(config)
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional activation array")
    cells, boxes, geometry_representation = _geometry_rows(cells, H.shape[0])
    mask = np.asarray(path_mask)
    if H.ndim != 2 or H.shape[0] != len(cells) or H.shape[1] < 1 or len(cells) < 1 or not np.isfinite(H).all():
        raise ValueError("H must be finite [number_of_cells,feature_dimension] with at least one cell/feature")
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("path_mask must be a nonempty two-dimensional array")
    binary = mask > 0
    points, cumulative = _path_arrays(sample["path"]["points_px"], sample["path"].get("cumulative_length_px"))
    out = Path(out_dir)
    filenames = {key: filename for key, filename in (
        ("patches", "patch_table.csv"), ("pairs", "all_pairs.csv"), ("matches", "matches.csv"),
        ("usage", "token_usage.csv"), ("arrays", "similarities.npz"),
        ("selected_overlay", "selected_patches.png"), ("clean_overlay", "clean_patches.png"),
        ("matrices", "similarity_matrices.png"), ("scatter", "all_pairs_scatter.png"), ("summary", "summary.json"))}
    if any((out / filename).exists() for filename in filenames.values()):
        raise FileExistsError(f"Continuity artifacts already exist in {out}; use a new output directory")
    height, width = binary.shape
    margin = settings["exclusion_margin_px"]
    endpoints = sample.get("endpoints", {})
    endpoints = list(endpoints.values()) if isinstance(endpoints, dict) else list(endpoints)
    disks = [("object_disk", obj) for obj in sample.get("objects", [])] + [("endpoint_disk", obj) for obj in endpoints]
    selected, clean, patch_rows, exclusions = [], [], [], Counter()
    arc_by_token, point_by_token, coverage_by_token = {}, {}, {}
    for index, box in enumerate(boxes):
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError("Every cell bbox must be inside the original path mask")
        x0, y0, x1, y1 = np.ceil(box).astype(int)
        region = binary[y0:y1, x0:x1]
        coverage = float(np.count_nonzero(region) / region.size) if region.size else 0.0
        is_line = bool(np.any(region))
        intervals = polyline_cell_intervals(points, box, cumulative) if is_line else []
        reasons = []
        if is_line:
            selected.append(index)
            if coverage < settings["min_line_coverage"]:
                reasons.append("low_line_coverage")
            if len(intervals) != 1:
                reasons.append("centerline_interval_count")
            for reason, disk in disks:
                if _circle_intersects(box, disk["center_px"], float(disk["radius_px"]) + margin) and reason not in reasons:
                    reasons.append(reason)
            exclusions.update(reasons)
        arc = float(sum(intervals[0]) / 2) if len(intervals) == 1 else None
        point = interpolate_path(points, arc, cumulative) if arc is not None else None
        is_clean = is_line and not reasons
        if is_clean:
            clean.append(index)
            arc_by_token[index], point_by_token[index], coverage_by_token[index] = arc, point, coverage
        patch_rows.append({"token_index": index, "geometry_index": index,
                           "geometry_kind": geometry_representation["geometry_kind"],
                           "parent_merged_index": cells[index].get("merged_index") if geometry_representation["geometry_kind"] == "pre_merge_vision" else None,
                           "x0": float(box[0]), "y0": float(box[1]), "x1": float(box[2]), "y1": float(box[3]),
                           "line_cell": is_line, "clean": is_clean, "path_coverage": coverage,
                           "interval_count": len(intervals), "intervals_px": json.dumps(intervals),
                           "arc_midpoint_px": arc, "path_x_px": None if point is None else float(point[0]),
                           "path_y_px": None if point is None else float(point[1]),
                           "exclusion_reasons": ";".join(reasons)})
    clean.sort(key=lambda index: (arc_by_token[index], index))
    s = np.array([arc_by_token[index] for index in clean], dtype=np.float64)
    positions = np.array([point_by_token[index] for index in clean], dtype=np.float64).reshape(-1, 2)
    coverages = np.array([coverage_by_token[index] for index in clean], dtype=np.float64)
    raw_all, raw_norms = _cosine_matrix(H)
    all_mean = H.mean(axis=0)
    centered_all, centered_norms = _cosine_matrix(H - all_mean)
    raw = raw_all[np.ix_(clean, clean)]
    centered = centered_all[np.ix_(clean, clean)]
    pairs = []
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            pairs.append({"token_i": clean[i], "token_j": clean[j], "arc_i_px": float(s[i]), "arc_j_px": float(s[j]),
                          "arc_distance_px": float(abs(s[i] - s[j])), "screen_distance_px": float(np.linalg.norm(positions[i] - positions[j])),
                          "coverage_i": float(coverages[i]), "coverage_j": float(coverages[j]),
                          "coverage_error": float(abs(coverages[i] - coverages[j])),
                          "raw_cosine": _finite_float(raw[i, j]), "centered_cosine": _finite_float(centered[i, j])})
    matches = select_geometry_matches(s, positions, coverages, settings)
    match_rows, usage = [], Counter()
    for match in matches:
        a, j, k = (match[key] for key in ("anchor", "near", "far"))
        usage.update((clean[a], clean[j], clean[k]))
        row = {**match, "anchor_token": clean[a], "near_token": clean[j], "far_token": clean[k],
               "raw_near_cosine": _finite_float(raw[a, j]), "raw_far_cosine": _finite_float(raw[a, k]),
               "raw_delta": _finite_float(raw[a, j] - raw[a, k]),
               "centered_near_cosine": _finite_float(centered[a, j]), "centered_far_cosine": _finite_float(centered[a, k]),
               "centered_delta": _finite_float(centered[a, j] - centered[a, k])}
        match_rows.append(row)
    straight = sample.get("family") == "straight" or _straight_path(points, cumulative)
    minimum = settings["min_matched_anchors"]
    valid_raw = [row["raw_delta"] for row in match_rows if row["raw_delta"] is not None]
    valid_centered = [row["centered_delta"] for row in match_rows if row["centered_delta"] is not None]
    if straight:
        status = "structurally_non_identifiable"
    elif not selected:
        status = "no_line_cells"
    elif not clean:
        status = "no_clean_line_cells"
    elif len(clean) < 3:
        status = "insufficient_clean_cells"
    elif len(matches) < minimum:
        status = "insufficient_matched_anchors"
    elif min(len(valid_raw), len(valid_centered)) < minimum:
        status = "degenerate_activations"
    else:
        status = "ok"
    conditional = {"raw_median": None, "centered_median": None,
                   "definition": "median(cosine(anchor,near)-cosine(anchor,far)) across geometry-matched anchors",
                   "raw_valid_anchors": len(valid_raw), "centered_valid_anchors": len(valid_centered)}
    if not straight and len(matches) >= minimum:
        if len(valid_raw) >= minimum:
            conditional["raw_median"] = float(np.median(valid_raw))
        if len(valid_centered) >= minimum:
            conditional["centered_median"] = float(np.median(valid_centered))
    usage_rows = [{"token_index": index, "uses_total": usage[index],
                   "uses_anchor": sum(row["anchor_token"] == index for row in match_rows),
                   "uses_near": sum(row["near_token"] == index for row in match_rows),
                   "uses_far": sum(row["far_token"] == index for row in match_rows)} for index in clean]
    summary = {
        "schema_version": "1.0", "sample_id": sample.get("id"), "status": status, "config": settings,
        "n_total_cells": len(cells), "feature_dimension": H.shape[1], "n_line_cells": len(selected),
        "n_clean_cells": len(clean), "n_excluded_line_cells": len(selected) - len(clean),
        "exclusion_counts_nonexclusive": dict(sorted(exclusions.items())),
        "n_all_pairs": len(pairs), "n_matched_anchors": len(matches), "conditional": conditional,
        "straight_geometry": straight, "zero_norm_raw_cells": int(np.count_nonzero(raw_norms == 0)),
        "zero_norm_centered_cells": int(np.count_nonzero(centered_norms == 0)),
        "token_reuse": {"distinct_tokens_used": len(usage), "tokens_used_more_than_once": sum(count > 1 for count in usage.values()),
                        "maximum_uses": max(usage.values(), default=0), "total_token_uses": sum(usage.values()),
                        "note": "Triples share tokens and are descriptive dependent observations; no p-values are computed."},
        "representation": {**geometry_representation, "cosine_dimensions": H.shape[1], "centering_population": "all input image cells",
                           "centering_cell_count": len(cells), "normalization": "per-token L2 after optional centering", "pca_used": False},
        "geometry": {"coordinate_units": "original-image pixels", "arc_assignment": "midpoint of the single exact clipped centerline interval",
                     "circle_exclusion": "any object or S/E disk, radius plus exclusion_margin_px, intersects the closed cell rectangle",
                     "line_selection": "any path_mask pixel > 0", "matrix_population": "clean cells ordered by centerline arc midpoint",
                     "matching_rank": ["screen_distance_error", "coverage_error", "near_arc", "far_arc", "near_index", "far_index"]},
        "files": filenames,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / filenames["patches"], list(patch_rows[0]), patch_rows)
    pair_fields = ["token_i", "token_j", "arc_i_px", "arc_j_px", "arc_distance_px", "screen_distance_px", "coverage_i", "coverage_j", "coverage_error", "raw_cosine", "centered_cosine"]
    _write_csv(out / filenames["pairs"], pair_fields, pairs)
    match_fields = ["anchor", "near", "far", "near_arc_px", "far_arc_px", "near_screen_px", "far_screen_px", "screen_distance_error_px", "coverage_error", "anchor_token", "near_token", "far_token", "raw_near_cosine", "raw_far_cosine", "raw_delta", "centered_near_cosine", "centered_far_cosine", "centered_delta"]
    _write_csv(out / filenames["matches"], match_fields, match_rows)
    _write_csv(out / filenames["usage"], ["token_index", "uses_total", "uses_anchor", "uses_near", "uses_far"], usage_rows)
    np.savez_compressed(out / filenames["arrays"], raw_cosine_all=raw_all, centered_cosine_all=centered_all,
                        all_cells_mean=all_mean, raw_norms=raw_norms, centered_norms=centered_norms,
                        all_geometry_indices=np.arange(len(cells), dtype=np.int64), all_bboxes_px=np.asarray(boxes),
                        geometry_kind=np.asarray(geometry_representation["geometry_kind"]),
                        line_token_indices=np.asarray(selected, dtype=np.int64), clean_token_indices=np.asarray(clean, dtype=np.int64),
                        clean_arc_positions_px=s, clean_path_positions_px=positions, clean_coverages=coverages,
                        clean_raw_cosine=raw, clean_centered_cosine=centered,
                        clean_bboxes_px=np.asarray([boxes[index] for index in clean]).reshape(-1, 4))
    _plots(out, image_path, binary.shape, boxes, selected, clean, raw, centered, pairs)
    (out / filenames["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary
