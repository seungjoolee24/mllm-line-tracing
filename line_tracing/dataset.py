"""Write reproducible, self-contained datasets with atomic directory publication."""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import PIL

from .geometry import FAMILIES, GeneratorConfig, make_sample
from .render import SUPERSAMPLE, preview_sheet, render_masks, render_sample


def sample_seed(master_seed, index):
    """Stable across Python processes, dataset sizes, and output directories."""
    payload = f"single-line-v0:{master_seed}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_dataset(output, count=16, seed=42, family="curve", config=None, preview=False):
    """Write a new directory; existing paths are never overwritten.

    For family='mixed', family assignment cycles in FAMILIES order. config holds
    the remaining settings. Each sample is generated/validated before rendering.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if family not in (*FAMILIES, "mixed"):
        raise ValueError(f"unknown family: {family}")
    base_config = config if config is not None else GeneratorConfig()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    examples, family_counts = [], Counter()
    try:
        (temporary / "images").mkdir()
        (temporary / "masks").mkdir()
        with (temporary / "annotations.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(count):
                selected = FAMILIES[index % len(FAMILIES)] if family == "mixed" else family
                sample_id = f"sample_{index:06d}"
                sample = make_sample(replace(base_config, family=selected), sample_seed(seed, index), sample_id)
                files = {
                    "image": f"images/{sample_id}.png",
                    "path_mask": f"masks/{sample_id}_path.png",
                    "object_mask": f"masks/{sample_id}_objects.png",
                }
                image = render_sample(sample)
                path_mask, object_mask = render_masks(sample)
                image.save(temporary / files["image"], format="PNG")
                path_mask.save(temporary / files["path_mask"], format="PNG")
                object_mask.save(temporary / files["object_mask"], format="PNG")
                sample["files"] = files
                sample["sha256"] = {
                    key: hashlib.sha256((temporary / path).read_bytes()).hexdigest()
                    for key, path in files.items()
                }
                stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                family_counts[selected] += 1
                if preview and len(examples) < 16:
                    examples.append(sample)
        manifest = {
            "schema_version": "1.0", "generator_version": "0.1.0",
            "task": "single_line_color_sequence", "count": count, "master_seed": seed,
            "family": family, "family_counts": dict(family_counts), "base_config": asdict(base_config),
            "annotations": "annotations.jsonl",
            "runtime": {"python": platform.python_version(), "pillow": PIL.__version__},
            "render": {"supersample": SUPERSAMPLE, "font": "Pillow bundled default"},
            "source": "https://arxiv.org/html/2605.15672v2",
            "scope": "Adapted low-complexity baseline; not an exact paper reproduction.",
            "preview": "preview.png" if preview else None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if preview:
            preview_sheet(examples).save(temporary / "preview.png", format="PNG")
        if output.exists():
            raise FileExistsError(f"output appeared during generation: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
