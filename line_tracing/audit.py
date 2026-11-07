"""Verify a saved dataset, including geometry, PNG contents, and file hashes.

Run ``python -m line_tracing.audit DATASET [--report report.json]``. This reads
the published files rather than trusting successful generation in memory.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageColor

from .geometry import validate_sample


FILE_KEYS = ("image", "path_mask", "object_mask")


def _confined_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("file path must be a nonempty relative string")
    if Path(relative).is_absolute():
        raise ValueError(f"absolute file path is not allowed: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"file path escapes dataset directory: {relative}") from None
    if not path.is_file():
        raise ValueError(f"missing file: {relative}")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_component(mask):
    """Check one nonempty 8-connected foreground using only standard Python."""
    width, height = mask.size
    remaining = {i for i, value in enumerate(mask.tobytes()) if value}
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        index = stack.pop()
        x, y = index % width, index // width
        for row in range(max(0, y - 1), min(height, y + 2)):
            for column in range(max(0, x - 1), min(width, x + 2)):
                neighbor = row * width + column
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return not remaining


def _read_png(path, mode, size):
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path.name}: expected PNG, got {image.format}")
        if image.mode != mode:
            raise ValueError(f"{path.name}: expected mode {mode}, got {image.mode}")
        if image.size != size:
            raise ValueError(f"{path.name}: expected size {size}, got {image.size}")
        image.load()
        return image.copy()


def _check_sample(root, sample, checks):
    validate_sample(sample)
    checks["sample_geometry"] += 1
    files = sample["files"]
    hashes = sample["sha256"]
    if not isinstance(files, dict) or set(files) != set(FILE_KEYS):
        raise ValueError(f"files must contain exactly {', '.join(FILE_KEYS)}")
    if not isinstance(hashes, dict) or set(hashes) != set(FILE_KEYS):
        raise ValueError("sha256 keys must match the three file keys")
    paths = {}
    for key in FILE_KEYS:
        paths[key] = _confined_file(root, files[key])
        recorded = hashes[key]
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise ValueError(f"{key}: expected a SHA-256 hex digest")
        if _sha256(paths[key]) != recorded:
            raise ValueError(f"{key}: SHA-256 mismatch ({files[key]})")
        checks["file_sha256"] += 1
    size = (sample["image"]["width"], sample["image"]["height"])
    image = _read_png(paths["image"], "RGB", size)
    path_mask = _read_png(paths["path_mask"], "L", size)
    instances = _read_png(paths["object_mask"], "L", size)
    checks["png_mode_size"] += 3
    path_values = set(path_mask.tobytes())
    if not path_values <= {0, 255} or 255 not in path_values:
        raise ValueError("path_mask must contain a nonempty binary 0/255 path")
    checks["binary_path_mask"] += 1
    if not _one_component(path_mask):
        raise ValueError("path_mask must have exactly one 8-connected foreground component")
    checks["connected_path_mask"] += 1
    expected_values = {0, *(obj["mask_value"] for obj in sample["objects"])}
    if set(instances.tobytes()) != expected_values:
        raise ValueError("object_mask values do not match background 0 plus all object instance IDs")
    checks["object_mask_values"] += 1
    for obj in sample["objects"]:
        center = tuple(round(value) for value in obj["center_px"])
        expected_rgb = ImageColor.getrgb(obj["hex"])
        if image.getpixel(center) != expected_rgb:
            raise ValueError(f"object {obj['sequence_index']}: center pixel does not match its color")
        if instances.getpixel(center) != obj["mask_value"]:
            raise ValueError(f"object {obj['sequence_index']}: center instance ID mismatch")
        if path_mask.getpixel(center) != 255:
            raise ValueError(f"object {obj['sequence_index']}: center is outside raster path mask")
        checks["object_centers"] += 1


def audit_dataset(directory):
    """Return a JSON-compatible audit report. Invalid datasets never pass silently."""
    root = Path(directory).resolve()
    report = {
        "dataset": str(root), "valid": False, "count": 0,
        "families": {}, "checks": {}, "geometry": {}, "errors": [],
    }
    checks, families, ids = Counter(), Counter(), set()
    lengths, minimum_spacings = [], []
    try:
        if not root.is_dir():
            raise ValueError(f"dataset directory does not exist: {root}")
        manifest_path = _confined_file(root, "manifest.json")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        count = manifest.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("manifest count must be a positive integer")
        expected_families = manifest.get("family_counts")
        if not isinstance(expected_families, dict) or not all(
            isinstance(name, str) and isinstance(value, int)
            and not isinstance(value, bool) and value >= 0
            for name, value in expected_families.items()
        ):
            raise ValueError("manifest family_counts must map family names to nonnegative counts")
        annotations = _confined_file(root, manifest.get("annotations"))
        with annotations.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                sample_id = f"line {line_number}"
                try:
                    sample = json.loads(line)
                    if not isinstance(sample, dict):
                        raise ValueError("annotation must be a JSON object")
                    report["count"] += 1
                    identity = sample.get("id")
                    if not isinstance(identity, str) or not identity:
                        raise ValueError("sample ID must be a nonempty string")
                    sample_id = identity
                    if identity in ids:
                        raise ValueError("duplicate sample ID")
                    ids.add(identity)
                    checks["unique_sample_ids"] += 1
                    family = sample.get("family")
                    if not isinstance(family, str) or not family:
                        raise ValueError("sample family must be a nonempty string")
                    families[family] += 1
                    _check_sample(root, sample, checks)
                    length = float(sample["path"]["length_px"])
                    if not math.isfinite(length) or length <= 0:
                        raise ValueError("path length_px must be finite and positive")
                    lengths.append(length)
                    centers = [obj["center_px"] for obj in sample["objects"]]
                    if len(centers) > 1:
                        minimum_spacings.append(min(
                            math.dist(a, b) for i, a in enumerate(centers) for b in centers[i + 1:]
                        ))
                except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    report["errors"].append(f"{sample_id}: {exc}")
        if report["count"] != count:
            report["errors"].append(f"manifest count {count} != annotation count {report['count']}")
        else:
            checks["manifest_count"] += 1
        if dict(families) != expected_families:
            report["errors"].append(
                f"manifest family_counts {expected_families} != observed {dict(families)}"
            )
        else:
            checks["manifest_family_counts"] += 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report["errors"].append(str(exc))
    report["families"] = dict(sorted(families.items()))
    report["checks"] = dict(sorted(checks.items()))
    if lengths:
        report["geometry"]["path_length_px"] = {"min": min(lengths), "max": max(lengths)}
    if minimum_spacings:
        report["geometry"]["minimum_object_center_spacing_px"] = {
            "min": min(minimum_spacings), "max": max(minimum_spacings),
        }
    report["valid"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit a saved single-line tracing dataset.")
    parser.add_argument("dataset", help="Dataset directory containing manifest.json")
    parser.add_argument("--report", help="Write the JSON report to this file as well as stdout")
    args = parser.parse_args(argv)
    report = audit_dataset(args.dataset)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        try:
            Path(args.report).write_text(encoded, encoding="utf-8")
        except OSError as exc:
            parser.exit(2, f"error writing report: {exc}\n")
    print(encoded, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
