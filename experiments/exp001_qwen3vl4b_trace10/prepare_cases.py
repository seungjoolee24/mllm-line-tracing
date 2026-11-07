"""Freeze ten varied, readable single-line cases for the activation pilot."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from line_tracing.geometry import GeneratorConfig, make_sample
from line_tracing.render import preview_sheet, render_masks, render_sample
from line_tracing.audit import audit_dataset


RECIPES = [
    {"family": "curve", "num_objects": 6, "path_length": 480, "object_spacing": 70, "rotation_degrees": 20, "reverse": False},
    {"family": "straight", "num_objects": 4, "path_length": 400, "object_spacing": 100, "rotation_degrees": 105, "reverse": True},
    {"family": "elbow", "num_objects": 8, "path_length": 480, "object_spacing": 56, "rotation_degrees": 25, "reverse": False},
    {"family": "swirl", "num_objects": 8, "path_length": 480, "object_spacing": 56, "position_jitter": 4, "rotation_degrees": 80, "reverse": True},
    {"family": "curve", "num_objects": 8, "path_length": 480, "object_spacing": 56, "position_jitter": 8, "rotation_degrees": 210, "reverse": False},
    {"family": "straight", "num_objects": 8, "path_length": 480, "object_spacing": 56, "position_jitter": 6, "rotation_degrees": 150, "reverse": True},
    {"family": "elbow", "num_objects": 6, "path_length": 440, "object_spacing": 64, "position_jitter": 4, "rotation_degrees": 270, "reverse": False},
    {"family": "swirl", "num_objects": 6, "path_length": 480, "object_spacing": 70, "swirl_turns": 1.0, "rotation_degrees": 160, "reverse": False},
    {"family": "curve", "num_objects": 4, "path_length": 400, "object_spacing": 96, "position_jitter": 6, "rotation_degrees": 300, "reverse": True},
    {"family": "elbow", "num_objects": 8, "path_length": 520, "object_spacing": 56, "position_jitter": 5, "rotation_degrees": 120, "reverse": True},
]


def prepare(output=None):
    output = Path(output) if output else Path(__file__).resolve().parent / "dataset"
    if output.exists():
        report = audit_dataset(output)
        if not report["valid"] or report["count"] != 10:
            raise ValueError(f"Existing dataset is invalid; refusing to overwrite: {report}")
        print(json.dumps({"dataset": str(output), "count": 10, "reused": True}))
        return output
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    samples, counts = [], {}
    for i, options in enumerate(RECIPES):
        sample = make_sample(GeneratorConfig(**options), seed=2026110700 + i,
                             sample_id=f"case_{i:02d}_{options['family']}")
        sample["files"] = {
            "image": f"images/{sample['id']}.png",
            "path_mask": f"masks/{sample['id']}_path.png",
            "object_mask": f"masks/{sample['id']}_objects.png",
        }
        images = [render_sample(sample), *render_masks(sample)]
        for key, image in zip(("image", "path_mask", "object_mask"), images):
            image.save(output / sample["files"][key])
        sample["sha256"] = {key: hashlib.sha256((output / path).read_bytes()).hexdigest()
                            for key, path in sample["files"].items()}
        samples.append(sample)
        counts[sample["family"]] = counts.get(sample["family"], 0) + 1
    (output / "annotations.jsonl").write_text(
        "".join(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n" for sample in samples), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "count": 10, "family_counts": counts,
        "annotations": "annotations.jsonl", "recipes": RECIPES,
        "purpose": "Frozen pilot pool; execute first case only before reviewing tensor size.",
    }, indent=2) + "\n", encoding="utf-8")
    preview_sheet(samples, columns=5, thumbnail_size=256).save(output / "preview.png")
    report = audit_dataset(output)
    if not report["valid"]:
        raise ValueError(report)
    print(json.dumps({"dataset": str(output), "count": len(samples), "valid": True}))
    return output


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else None)
