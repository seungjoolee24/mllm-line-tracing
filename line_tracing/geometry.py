"""Deterministic geometry and annotations for single-path color tracing.

Coordinates use pixel centers with x increasing rightward and y downward.  This
module needs only the Python standard library; rasterization belongs in render.
The families are deliberately simple adaptations, not the original Swirl data.
"""

from __future__ import annotations

import bisect
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
FAMILIES = ("straight", "curve", "elbow", "swirl")
PALETTE = {
    "red": "#e53935",
    "orange": "#f57c00",
    "yellow": "#fbc02d",
    "green": "#43a047",
    "blue": "#1e88e5",
    "purple": "#7e57c2",
    "pink": "#ec7eaa",
    "brown": "#795548",
}
ENDPOINT_RADIUS = 14.0
MAX_ENDPOINT_GAP_RATIO = 1.25
EPS = 1e-7


@dataclass(frozen=True)
class GeneratorConfig:
    """Image-space controls. ``None`` length means 0.75 * image_size pixels."""

    family: str = "curve"
    image_size: int = 640
    num_objects: int = 8
    path_length: Optional[float] = None
    line_width: float = 4
    object_radius: float = 12
    margin: float = 56
    rotation_degrees: Optional[float] = None
    reverse: Optional[bool] = None
    swirl_turns: float = 0.75
    object_spacing: Optional[float] = None
    position_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("family must be one of " + ", ".join(FAMILIES))
        if type(self.image_size) is not int or not 128 <= self.image_size <= 4096:
            raise ValueError("image_size must be an integer between 128 and 4096")
        if type(self.num_objects) is not int or not 2 <= self.num_objects <= len(PALETTE):
            raise ValueError("num_objects must be an integer between 2 and 8")
        for name in ("line_width", "object_radius", "margin", "swirl_turns"):
            _positive_number(name, getattr(self, name))
        if self.path_length is not None:
            _positive_number("path_length", self.path_length)
        if self.object_spacing is not None:
            _positive_number("object_spacing", self.object_spacing)
        _finite_number("position_jitter", self.position_jitter)
        if self.position_jitter < 0:
            raise ValueError("position_jitter must be nonnegative")
        if self.rotation_degrees is not None:
            _finite_number("rotation_degrees", self.rotation_degrees)
        if self.reverse is not None and type(self.reverse) is not bool:
            raise ValueError("reverse must be True, False, or None")
        if self.line_width < 1:
            raise ValueError("line_width must be at least 1 pixel for raster rendering")
        if self.object_radius < 4:
            raise ValueError("object_radius must be at least 4 pixels for raster rendering")
        if self.object_radius < self.line_width:
            raise ValueError("object_radius must be at least line_width")
        if self.margin < max(self.object_radius, ENDPOINT_RADIUS) + 4:
            raise ValueError("margin must reserve endpoint and object radii plus 4 pixels")
        if 2 * self.margin >= self.image_size:
            raise ValueError("margin leaves no usable image area")
        if not 0.25 <= self.swirl_turns <= 1.25:
            raise ValueError("swirl_turns must be between 0.25 and 1.25 for this easy baseline")
        length = self.effective_path_length
        if length > 4 * self.image_size:
            raise ValueError("path_length exceeds the supported easy-family range")
        reserve = self.object_radius + ENDPOINT_RADIUS + 9
        minimum = 2 * reserve + (self.num_objects - 1) * (2 * self.object_radius + 8)
        if length < minimum:
            raise ValueError(
                "path_length is too short for the requested objects and endpoint clearance "
                "(at least %.1f pixels required)" % minimum
            )
        _placement_layout(self, length)

    @property
    def effective_path_length(self) -> float:
        return float(self.path_length if self.path_length is not None else 0.75 * self.image_size)


def _finite_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")


def _positive_number(name: str, value: Any) -> None:
    _finite_number(name, value)
    if value <= 0:
        raise ValueError(name + " must be positive")


def _placement_layout(config: GeneratorConfig, length: float) -> Tuple[float, float]:
    """Return centered nominal arc-length positions without altering controls."""
    reserve = config.object_radius + ENDPOINT_RADIUS + 9
    if config.object_spacing is None:
        padding = max(0.10 * length, reserve)
        step = (length - 2 * padding) / (config.num_objects - 1)
    else:
        step = float(config.object_spacing)
        padding = (length - (config.num_objects - 1) * step) / 2
    if padding - config.position_jitter < reserve - EPS:
        raise ValueError("object spacing and position_jitter leave insufficient endpoint clearance")
    if step - 2 * config.position_jitter < 2 * config.object_radius + 8 - EPS:
        raise ValueError("object spacing and position_jitter leave insufficient object separation")
    if padding + config.position_jitter > MAX_ENDPOINT_GAP_RATIO * step + EPS:
        minimum_spacing = max(
            2 * config.object_radius + 8 + 2 * config.position_jitter,
            (length + 2 * config.position_jitter) / (config.num_objects - 1 + 2 * MAX_ENDPOINT_GAP_RATIO),
        )
        maximum_spacing = (length - 2 * (reserve + config.position_jitter)) / (config.num_objects - 1)
        suggestion = ""
        if minimum_spacing <= maximum_spacing:
            suggestion = " (feasible object_spacing range: %.2f to %.2f arc-length pixels)" % (minimum_spacing, maximum_spacing)
        raise ValueError(
            "object_spacing leaves excessive empty endpoint segments; increase spacing or shorten path_length" + suggestion
        )
    return padding, step


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cumulative(points: Sequence[Sequence[float]]) -> List[float]:
    result = [0.0]
    for a, b in zip(points, points[1:]):
        result.append(result[-1] + _distance(a, b))
    return result


def _at_arc(points: Sequence[Sequence[float]], cumulative: Sequence[float], arc: float) -> Tuple[Point, Point]:
    index = min(max(0, bisect.bisect_right(cumulative, arc) - 1), len(points) - 2)
    length = cumulative[index + 1] - cumulative[index]
    fraction = min(1.0, max(0.0, (arc - cumulative[index]) / length))
    a, b = points[index], points[index + 1]
    tangent = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1])), tangent


def _raw_path(config: GeneratorConfig, rng: random.Random) -> List[Point]:
    # Dense analytic samples are resampled to a common image-space arc length.
    count = 2048
    if config.family == "straight":
        return [(i / count - 0.5, 0.0) for i in range(count + 1)]
    if config.family == "curve":
        bend = rng.uniform(0.14, 0.28) * rng.choice((-1, 1))
        return [(i / count - 0.5, bend * math.sin(math.pi * i / count)) for i in range(count + 1)]
    if config.family == "elbow":
        a = rng.uniform(0.43, 0.57)
        b = 1.0 - a
        radius = rng.uniform(0.18, 0.28) * min(a, b)
        points = [(0.0, 0.0), (a - radius, 0.0)]
        for i in range(1, 513):
            angle = -math.pi / 2 + (math.pi / 2) * i / 512
            points.append((a - radius + radius * math.cos(angle), radius + radius * math.sin(angle)))
        points.append((a, b))
        return points
    turns = config.swirl_turns
    inner_radius = rng.uniform(0.18, 0.24)
    return [
        (
            (inner_radius + (1 - inner_radius) * i / count) * math.cos(2 * math.pi * turns * i / count),
            (inner_radius + (1 - inner_radius) * i / count) * math.sin(2 * math.pi * turns * i / count),
        )
        for i in range(count + 1)
    ]


def _make_path(config: GeneratorConfig, rng: random.Random) -> Tuple[List[Point], float]:
    raw = _raw_path(config, rng)
    raw_arc = _cumulative(raw)
    target = config.effective_path_length
    # 1.5 pixels leaves headroom below the 2-pixel annotation spacing contract.
    steps = max(2, math.ceil(target / 1.5))
    samples = [_at_arc(raw, raw_arc, raw_arc[-1] * i / steps)[0] for i in range(steps + 1)]
    scale = target / _cumulative(samples)[-1]
    angle = float(config.rotation_degrees if config.rotation_degrees is not None else rng.uniform(0, 360)) % 360
    theta = math.radians(angle)
    co, si = math.cos(theta), math.sin(theta)
    rotated = [(scale * (x * co - y * si), scale * (x * si + y * co)) for x, y in samples]
    xs, ys = zip(*rotated)
    available = config.image_size - 2 * config.margin
    if max(xs) - min(xs) > available + EPS or max(ys) - min(ys) > available + EPS:
        raise ValueError("path_length and rotation do not fit inside the image margin; shorten path_length or enlarge image_size")
    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    middle = config.image_size / 2
    return [(x - center_x + middle, y - center_y + middle) for x, y in rotated], angle


def _point_segment_distance(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return _distance(point, a)
    t = min(1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom))
    return _distance(point, (a[0] + t * dx, a[1] + t * dy))


def _cross(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    if (max(a[0], b[0]) < min(c[0], d[0]) - EPS or max(c[0], d[0]) < min(a[0], b[0]) - EPS
            or max(a[1], b[1]) < min(c[1], d[1]) - EPS or max(c[1], d[1]) < min(a[1], b[1]) - EPS):
        return False
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ab_c * ab_d <= EPS and cd_a * cd_b <= EPS


def _segment_distance(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _candidate_segment_pairs(points: Sequence[Sequence[float]], padding: float) -> Iterable[Tuple[int, int]]:
    """Spatial broad phase, avoiding a quadratic scan for long dense paths."""
    cell = max(16.0, 2 * padding)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(zip(points, points[1:])):
        x0, x1 = math.floor((min(a[0], b[0]) - padding) / cell), math.floor((max(a[0], b[0]) + padding) / cell)
        y0, y1 = math.floor((min(a[1], b[1]) - padding) / cell), math.floor((max(a[1], b[1]) + padding) / cell)
        keys = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        candidates = set()
        for key in keys:
            candidates.update(buckets.get(key, ()))
        for j in candidates:
            if j < i - 1:
                yield j, i
        for key in keys:
            buckets.setdefault(key, []).append(i)


def _check_path(points: Sequence[Sequence[float]], cumulative: Sequence[float], line_width: float) -> None:
    clearance = line_width + 8.0
    for i, j in _candidate_segment_pairs(points, clearance):
        a, b, c, d = points[i], points[i + 1], points[j], points[j + 1]
        if _segments_intersect(a, b, c, d):
            raise ValueError("path has a self-intersection or self-touch")
        # Ignore the unavoidable proximity of neighboring points on one stroke.
        if cumulative[j] - cumulative[i + 1] >= 2 * clearance:
            if _segment_distance(a, b, c, d) < clearance - EPS:
                raise ValueError("nonlocal path sections have insufficient clearance")


def _object_clearance(objects: Sequence[Dict[str, Any]], points: Sequence[Sequence[float]],
                      cumulative: Sequence[float], radius: float, line_width: float) -> None:
    for i, obj in enumerate(objects):
        center = obj["center_px"]
        if min(_distance(center, points[0]), _distance(center, points[-1])) < radius + ENDPOINT_RADIUS + 8 - EPS:
            raise ValueError("object overlaps an endpoint badge or its clearance")
        for other in objects[:i]:
            if _distance(center, other["center_px"]) < 2 * radius + 8 - EPS:
                raise ValueError("objects have insufficient separation")
        for k, (a, b) in enumerate(zip(points, points[1:])):
            midpoint_arc = (cumulative[k] + cumulative[k + 1]) / 2
            if abs(midpoint_arc - obj["arc_length_px"]) > 2 * radius + line_width + 8:
                if _point_segment_distance(center, a, b) < radius + line_width / 2 + 8 - EPS:
                    raise ValueError("object touches a nonlocal section of the path")


def _check_endpoint_clearance(points: Sequence[Sequence[float]], cumulative: Sequence[float],
                              line_width: float) -> None:
    """Keep S/E badges distinct and away from other sections of the stroke."""
    if _distance(points[0], points[-1]) < 2 * ENDPOINT_RADIUS + 8 - EPS:
        raise ValueError("endpoint badges have insufficient separation")
    adjacent_arc = ENDPOINT_RADIUS + line_width + 8
    required_distance = ENDPOINT_RADIUS + line_width / 2 + 8
    length = cumulative[-1]
    for k, (a, b) in enumerate(zip(points, points[1:])):
        midpoint_arc = (cumulative[k] + cumulative[k + 1]) / 2
        if midpoint_arc > adjacent_arc:
            if _point_segment_distance(points[0], a, b) < required_distance - EPS:
                raise ValueError("start badge touches a nonlocal section of the path")
        if length - midpoint_arc > adjacent_arc:
            if _point_segment_distance(points[-1], a, b) < required_distance - EPS:
                raise ValueError("end badge touches a nonlocal section of the path")


def make_sample(config: GeneratorConfig, seed: int, sample_id: Optional[str] = None) -> Dict[str, Any]:
    """Make one JSON-serializable sample, rejecting geometrically invalid requests.

    Explicit ``reverse=False`` / ``True`` pairs preserve physical object colors
    and centers, and reverse only the traversal and the S/E endpoint badges.
    """
    if not isinstance(config, GeneratorConfig):
        raise ValueError("config must be a GeneratorConfig")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    sample_id = sample_id if sample_id is not None else "%s_%s" % (config.family, seed)
    if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError("sample_id must contain only letters, digits, underscores, and hyphens")
    points, angle = _make_path(config, random.Random("%s:geometry" % seed))
    cumulative = _cumulative(points)
    length = cumulative[-1]
    _check_path(points, cumulative, config.line_width)
    _check_endpoint_clearance(points, cumulative, config.line_width)
    object_rng = random.Random("%s:objects" % seed)
    colors = object_rng.sample(list(PALETTE), config.num_objects)
    identities = object_rng.sample(range(1000, 10000), config.num_objects)
    padding, step = _placement_layout(config, length)
    jitter = config.position_jitter
    physical_objects = []
    last_error = None
    for _ in range(64 if jitter else 1):
        physical_objects = []
        for i, color in enumerate(colors):
            arc = padding + i * step + object_rng.uniform(-jitter, jitter)
            center, tangent = _at_arc(points, cumulative, arc)
            physical_objects.append({
                "id": "object_%d" % identities[i],
                "object_id": "object_%d" % identities[i],
                "color": color, "hex": PALETTE[color], "center_px": list(center),
                "arc_length_px": arc, "tangent": list(tangent),
            })
        try:
            _object_clearance(physical_objects, points, cumulative, config.object_radius, config.line_width)
            break
        except ValueError as error:
            last_error = error
    else:
        raise ValueError("cannot place objects with the requested geometry: " + str(last_error))
    reversed_path = config.reverse if config.reverse is not None else random.Random("%s:direction" % seed).choice((False, True))
    if reversed_path:
        points.reverse()
        cumulative = _cumulative(points)
        length = cumulative[-1]
        for obj in physical_objects:
            obj["arc_length_px"] = length - obj["arc_length_px"]
            obj["tangent"] = [-v for v in obj["tangent"]]
    ordered = sorted(physical_objects, key=lambda obj: obj["arc_length_px"])
    radius = float(config.object_radius)
    for i, obj in enumerate(ordered):
        x, y = obj["center_px"]
        obj.update({
            "sequence_index": i, "mask_value": i + 1,
            "center_normalized": [x / config.image_size, y / config.image_size],
            "bbox_px": [x - radius, y - radius, x + radius, y + radius],
            "radius_px": radius, "arc_fraction": obj["arc_length_px"] / length,
            "predecessor_id": ordered[i - 1]["id"] if i else None,
            "successor_id": ordered[i + 1]["id"] if i + 1 < len(ordered) else None,
        })
    answer_colors = [obj["color"] for obj in ordered]
    adjacent_arc_spacings = [b["arc_length_px"] - a["arc_length_px"] for a, b in zip(ordered, ordered[1:])]
    storage = list(ordered)
    random.Random("%s:storage" % seed).shuffle(storage)
    endpoint_data = {}
    for key, label, center in (("start", "S", points[0]), ("end", "E", points[-1])):
        x, y = center
        endpoint_data[key] = {"label": label, "center_px": list(center), "radius_px": ENDPOINT_RADIUS,
                              "bbox_px": [x - ENDPOINT_RADIUS, y - ENDPOINT_RADIUS, x + ENDPOINT_RADIUS, y + ENDPOINT_RADIUS]}
    chord = _distance(points[0], points[-1])
    turning = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        turning += abs(math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))
    sample = {
        "schema_version": "1.0", "id": sample_id, "seed": seed, "family": config.family,
        "config": asdict(config),
        "image": {"width": config.image_size, "height": config.image_size,
                  "coordinate_system": "pixel centers; origin top-left; x right; y down", "background": "#ffffff"},
        "style": {"line_color": "#202020", "line_width_px": config.line_width,
                  "object_radius_px": radius, "endpoint_radius_px": ENDPOINT_RADIUS},
        "prompt": "Follow the single continuous line from S to E. List the colors of the circles in the order encountered. "
                  "Output only lowercase color names separated by single spaces, followed by end. "
                  "Allowed color names: red orange yellow green blue purple pink brown.",
        "answer_colors": answer_colors, "answer": " ".join(answer_colors + ["end"]),
        "objects": storage, "endpoints": endpoint_data,
        "path": {"points_px": [list(point) for point in points], "cumulative_length_px": cumulative,
                 "length_px": length, "start_px": list(points[0]), "end_px": list(points[-1]),
                 "rotation_degrees": angle, "reversed": reversed_path, "direction": "S_to_E"},
        "metrics": {"path_length_px": length, "endpoint_distance_px": chord,
                    "tortuosity": length / chord, "total_absolute_turn_radians": turning,
                    "required_nonlocal_line_clearance_px": config.line_width + 8.0,
                    "min_adjacent_arc_spacing_px": min(adjacent_arc_spacings),
                    "max_adjacent_arc_spacing_px": max(adjacent_arc_spacings),
                    "endpoint_arc_gap_start_px": ordered[0]["arc_length_px"],
                    "endpoint_arc_gap_end_px": length - ordered[-1]["arc_length_px"],
                    "max_allowed_endpoint_arc_gap_px": MAX_ENDPOINT_GAP_RATIO * step,
                    "min_object_center_distance_px": min(_distance(a["center_px"], b["center_px"])
                        for i, a in enumerate(ordered) for b in ordered[:i]),
                    "min_endpoint_object_distance_px": min(_distance(obj["center_px"], endpoint)
                        for obj in ordered for endpoint in (points[0], points[-1]))},
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Independently check geometry, traversal labels, and annotation consistency.

    A valid report certifies these checks, not model performance or human ease.
    Raises ``ValueError`` for invalid or inconsistent records.
    """
    try:
        config = GeneratorConfig(**sample["config"])
        size = config.image_size
        points = sample["path"]["points_px"]
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("path needs at least two points")
        for point in points:
            if len(point) != 2:
                raise ValueError("path points need x and y coordinates")
            for value in point:
                _finite_number("path coordinate", value)
            if any(value < config.margin - EPS or value > size - config.margin + EPS for value in point):
                raise ValueError("path is outside the configured image margin")
        cumulative = _cumulative(points)
        if any(b <= a for a, b in zip(cumulative, cumulative[1:])):
            raise ValueError("path contains repeated consecutive points")
        maximum_step = max(b - a for a, b in zip(cumulative, cumulative[1:]))
        if maximum_step > 2.001:
            raise ValueError("path annotations are more than 2 pixels apart")
        declared = sample["path"]["cumulative_length_px"]
        for value in declared:
            _finite_number("cumulative path length", value)
        if len(declared) != len(cumulative) or any(abs(a - b) > 1e-5 for a, b in zip(declared, cumulative)):
            raise ValueError("cumulative path lengths are inconsistent")
        length = cumulative[-1]
        _finite_number("declared path length", sample["path"]["length_px"])
        if abs(length - config.effective_path_length) > 1e-4 or abs(sample["path"]["length_px"] - length) > 1e-5:
            raise ValueError("path length is inconsistent with config or annotation")
        if sample["path"]["start_px"] != points[0] or sample["path"]["end_px"] != points[-1]:
            raise ValueError("path endpoints are inconsistent")
        if sample["image"]["width"] != size or sample["image"]["height"] != size:
            raise ValueError("image dimensions are inconsistent")
        if sample["family"] != config.family:
            raise ValueError("family is inconsistent with config")
        _finite_number("path rotation", sample["path"]["rotation_degrees"])
        for key in ("path_length_px", "endpoint_distance_px", "tortuosity", "total_absolute_turn_radians",
                    "required_nonlocal_line_clearance_px", "min_object_center_distance_px",
                    "min_endpoint_object_distance_px", "min_adjacent_arc_spacing_px", "max_adjacent_arc_spacing_px",
                    "endpoint_arc_gap_start_px", "endpoint_arc_gap_end_px", "max_allowed_endpoint_arc_gap_px"):
            _finite_number("metric " + key, sample["metrics"][key])
        for key, expected in (("line_width_px", config.line_width), ("object_radius_px", config.object_radius),
                              ("endpoint_radius_px", ENDPOINT_RADIUS)):
            if sample["style"][key] != expected:
                raise ValueError("style is inconsistent with config")
        _check_path(points, cumulative, config.line_width)
        _check_endpoint_clearance(points, cumulative, config.line_width)
        objects = sample["objects"]
        if len(objects) != config.num_objects:
            raise ValueError("object count is inconsistent")
        ordered = sorted(objects, key=lambda obj: obj["sequence_index"])
        if [obj["sequence_index"] for obj in ordered] != list(range(config.num_objects)):
            raise ValueError("sequence indices must be consecutive and zero based")
        if len({obj["id"] for obj in ordered}) != len(ordered):
            raise ValueError("object identities must be unique")
        colors = [obj["color"] for obj in ordered]
        if len(set(colors)) != len(colors) or any(color not in PALETTE for color in colors):
            raise ValueError("object colors must be unique supported palette colors")
        arcs = [obj["arc_length_px"] for obj in ordered]
        if not all(0 < arc < length for arc in arcs) or not all(a < b for a, b in zip(arcs, arcs[1:])):
            raise ValueError("object order disagrees with path arc length")
        padding, step = _placement_layout(config, length)
        for i, arc in enumerate(arcs):
            if abs(arc - (padding + i * step)) > config.position_jitter + 1e-5:
                raise ValueError("object position disagrees with configured spacing or jitter")
        spacings = [b - a for a, b in zip(arcs, arcs[1:])]
        for key, expected in (("min_adjacent_arc_spacing_px", min(spacings)),
                              ("max_adjacent_arc_spacing_px", max(spacings)),
                              ("endpoint_arc_gap_start_px", arcs[0]),
                              ("endpoint_arc_gap_end_px", length - arcs[-1]),
                              ("max_allowed_endpoint_arc_gap_px", MAX_ENDPOINT_GAP_RATIO * step)):
            if abs(sample["metrics"][key] - expected) > 1e-5:
                raise ValueError("arc spacing or endpoint gap metric is inconsistent")
        if max(arcs[0], length - arcs[-1]) > MAX_ENDPOINT_GAP_RATIO * step + 1e-5:
            raise ValueError("actual endpoint arc gap exceeds the configured spacing cap")
        for i, obj in enumerate(ordered):
            _finite_number("object arc length", obj["arc_length_px"])
            _finite_number("object arc fraction", obj["arc_fraction"])
            for key in ("center_px", "center_normalized", "bbox_px", "tangent"):
                expected_size = 4 if key == "bbox_px" else 2
                if len(obj[key]) != expected_size:
                    raise ValueError("object " + key + " has an invalid coordinate count")
                for value in obj[key]:
                    _finite_number("object " + key, value)
            center, tangent = _at_arc(points, cumulative, obj["arc_length_px"])
            if _distance(center, obj["center_px"]) > 1e-5:
                raise ValueError("object center is not at its declared path position")
            if abs(obj["arc_fraction"] - obj["arc_length_px"] / length) > EPS:
                raise ValueError("object arc fraction is inconsistent")
            if _distance(tangent, obj["tangent"]) > 1e-5:
                raise ValueError("object tangent is inconsistent")
            if obj["radius_px"] != config.object_radius or obj["hex"] != PALETTE[obj["color"]]:
                raise ValueError("object appearance is inconsistent")
            x, y = center
            r = config.object_radius
            expected_bbox = (x - r, y - r, x + r, y + r)
            if len(obj["bbox_px"]) != 4 or any(abs(a - b) > EPS for a, b in zip(obj["bbox_px"], expected_bbox)):
                raise ValueError("object bounding box is inconsistent")
            if any(a < 0 or a > size for a in obj["bbox_px"]):
                raise ValueError("object is outside the image")
            if _distance(obj["center_normalized"], (x / size, y / size)) > EPS:
                raise ValueError("normalized object center is inconsistent")
            if obj["mask_value"] != i + 1 or obj["object_id"] != obj["id"]:
                raise ValueError("object identity or mask value is inconsistent")
            expected_previous = ordered[i - 1]["id"] if i else None
            expected_next = ordered[i + 1]["id"] if i + 1 < len(ordered) else None
            if obj["predecessor_id"] != expected_previous or obj["successor_id"] != expected_next:
                raise ValueError("predecessor or successor identity is inconsistent")
        _object_clearance(objects, points, cumulative, config.object_radius, config.line_width)
        if sample["answer_colors"] != colors or sample["answer"] != " ".join(colors + ["end"]):
            raise ValueError("answer is inconsistent with traversal order")
        for name, label, point in (("start", "S", points[0]), ("end", "E", points[-1])):
            endpoint = sample["endpoints"][name]
            if len(endpoint["center_px"]) != 2 or len(endpoint["bbox_px"]) != 4:
                raise ValueError("endpoint has an invalid coordinate count")
            for value in endpoint["center_px"] + endpoint["bbox_px"]:
                _finite_number("endpoint coordinate", value)
            if endpoint["label"] != label or _distance(endpoint["center_px"], point) > EPS:
                raise ValueError("endpoint badge is inconsistent")
            if endpoint["radius_px"] != ENDPOINT_RADIUS:
                raise ValueError("endpoint badge radius is inconsistent")
            x, y = point
            expected_bbox = (x - ENDPOINT_RADIUS, y - ENDPOINT_RADIUS, x + ENDPOINT_RADIUS, y + ENDPOINT_RADIUS)
            if len(endpoint["bbox_px"]) != 4 or any(abs(a - b) > EPS for a, b in zip(endpoint["bbox_px"], expected_bbox)):
                raise ValueError("endpoint bounding box is inconsistent")
        return {"valid": True, "num_paths": 1, "self_intersections": 0, "num_objects": len(objects),
                "path_length_px": length, "maximum_annotation_step_px": maximum_step,
                "checked": ["frame", "path_spacing", "self_intersection", "nonlocal_line_clearance",
                            "object_separation", "object_on_path", "endpoint_clearance", "endpoint_path_clearance",
                            "finite_annotations", "configured_spacing", "endpoint_gap_cap", "traversal_order", "answer"]}
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("malformed sample: " + str(error)) from error
