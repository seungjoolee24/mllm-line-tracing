"""Command-line dataset generation, suitable for local CPU or Google Colab."""

import argparse
import json
import sys

from .dataset import generate_dataset
from .geometry import FAMILIES, GeneratorConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate one-line ordered-color tracing problems.")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=(*FAMILIES, "mixed"), default="curve")
    parser.add_argument("--num-objects", type=int, default=8)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--path-length", type=float, help="Pixels; default 0.75 times image size")
    parser.add_argument("--line-width", type=float, default=4)
    parser.add_argument("--object-radius", type=float, default=12)
    parser.add_argument("--object-spacing", type=float, help="Center-to-center arc distance in pixels; default evenly spread")
    parser.add_argument("--position-jitter", type=float, default=0, help="Maximum per-object arc-position jitter in pixels")
    parser.add_argument("--rotation", type=float, help="Degrees in image coordinates; default seeded random rotation")
    parser.add_argument("--direction", choices=("random", "forward", "reverse"), default="random")
    parser.add_argument("--swirl-turns", type=float, default=0.75)
    parser.add_argument("--preview", action="store_true", help="Separate QA sheet with answers, up to 16 samples")
    args = parser.parse_args(argv)
    try:
        config = GeneratorConfig(
            family=FAMILIES[0] if args.family == "mixed" else args.family,
            image_size=args.size, num_objects=args.num_objects, path_length=args.path_length,
            line_width=args.line_width, object_radius=args.object_radius,
            object_spacing=args.object_spacing, position_jitter=args.position_jitter,
            rotation_degrees=args.rotation,
            reverse={"random": None, "forward": False, "reverse": True}[args.direction],
            swirl_turns=args.swirl_turns,
        )
        manifest = generate_dataset(args.output, args.count, args.seed, args.family, config, args.preview)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output": args.output, "count": manifest["count"], "families": manifest["family_counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
