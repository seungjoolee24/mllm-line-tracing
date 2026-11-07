"""Inspect saved captures, add explicit spatial token mapping, and package a run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def token_geometry(sample, meta, dataset, patch_size=16, merge_size=2):
    grid = meta["input"]["image_grid_thw"]
    if len(grid) != 1 or grid[0][0] != 1:
        raise ValueError("pilot token mapping requires one still image")
    _, gh, gw = grid[0]
    if gh % merge_size or gw % merge_size:
        raise ValueError("grid is not divisible by merge size")
    width, height = sample["image"]["width"], sample["image"]["height"]
    if [width, height] != [gw * patch_size, gh * patch_size] or meta["input"]["resize"]:
        raise ValueError("pixel mapping requires unchanged source image dimensions")
    merged, vision = [], []
    positions = meta["layout"]["image_positions"]
    with Image.open(Path(dataset) / sample["files"]["object_mask"]) as loaded:
        instances = loaded.convert("L")
    objects = sorted(sample["objects"], key=lambda obj: obj["sequence_index"])
    coverage = {obj["id"]: [] for obj in objects}
    for row in range(gh // merge_size):
        for column in range(gw // merge_size):
            index = row * (gw // merge_size) + column
            stride = patch_size * merge_size
            bbox = [column * stride, row * stride, (column + 1) * stride, (row + 1) * stride]
            merged.append({"merged_index": index, "decoder_position": positions[index], "bbox_px": bbox})
            pixels = list(instances.crop(tuple(bbox)).getdata())
            for obj in objects:
                coverage[obj["id"]].append(pixels.count(obj["mask_value"]) / len(pixels))
            # Actual pre-merger patch sequence uses 2x2 block-major ordering.
            for inner_row in range(merge_size):
                for inner_column in range(merge_size):
                    r, c = row * merge_size + inner_row, column * merge_size + inner_column
                    vision.append({"vision_index": len(vision), "row": r, "column": c,
                                   "merged_index": index,
                                   "bbox_px": [c * patch_size, r * patch_size,
                                               (c + 1) * patch_size, (r + 1) * patch_size]})
    return {"coordinate_system": "source pixels; x right, y down", "patch_size": patch_size,
            "merge_size": merge_size, "image_grid_thw": grid,
            "vision_sequence_order": "spatial merge block-major, then inner patch row-major",
            "vision_tokens": vision, "merged_tokens": merged,
            "object_ids_in_sequence": [obj["id"] for obj in objects],
            "object_fraction_of_merged_cell_area": coverage}


def summarize(run, dataset, archive=False):
    run, dataset = Path(run), Path(dataset)
    summary = json.loads((run / "run_summary.json").read_text())
    samples = {s["id"]: s for s in (json.loads(x) for x in (dataset / "annotations.jsonl").read_text().splitlines())}
    rows = []
    for directory in sorted((run / "samples").glob("*")):
        if directory.name.startswith(".") or not (directory / "meta.json").exists():
            continue
        meta = json.loads((directory / "meta.json").read_text())
        if meta["status"] != "complete":
            continue
        geometry = token_geometry(samples[meta["sample_id"]], meta, dataset)
        write_json(directory / "token_geometry.json", geometry)
        row = {"sample_id": meta["sample_id"], "answer": meta["generation"]["raw_answer"],
               "expected": samples[meta["sample_id"]]["answer"], "exact_match": meta["score"]["exact_match"],
               "storage": meta["storage"], "performance": meta["performance"],
               "generation_tokens": meta["layout"]["generated_length"],
               "question_tokens": len(meta["layout"]["question_positions"]),
               "visual_tokens": len(meta["layout"]["image_positions"]),
               "alignment": meta["alignment"]}
        rows.append(row)
    all_files = [p for p in run.rglob("*") if p.is_file()]
    report = {"completed_cases": len(rows), "intended_cases": summary["intended_cases"],
              "status": summary["status"], "cases": rows,
              "total_result_bytes_before_report": sum(p.stat().st_size for p in all_files),
              "tensor_file_bytes": summary["total_tensor_file_bytes"],
              "tensor_file_mib": summary["total_tensor_file_bytes"] / 1024**2,
              "full_ten_case_linear_estimate_gib": (summary["total_tensor_file_bytes"] / max(1, len(rows)) * 10 / 1024**3),
              "estimate_note": "Linear estimate only; actual prompt/output lengths change attention size."}
    write_json(run / "size_report.json", report)
    if archive:
        archive_path = run.parent / (run.name + ".zip")
        # Streaming ZIP avoids assembling all activation tensors in RAM.
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as output:
            for path in sorted(run.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(run.parent))
        print(json.dumps({"archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
                          "sha256": hashlib.file_digest(archive_path.open("rb"), "sha256").hexdigest()}))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    summarize(args.run, args.dataset, args.archive)
