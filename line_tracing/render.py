"""Render model inputs and separate geometry masks without external fonts."""

import math

from PIL import Image, ImageDraw, ImageFont


SUPERSAMPLE = 4


def _box(center, radius, scale=1):
    x, y = center
    return tuple(v * scale for v in (x - radius, y - radius, x + radius, y + radius))


def _draw_path(draw, sample, scale=1, fill=255):
    points = [tuple(v * scale for v in p) for p in sample["path"]["points_px"]]
    width = max(1, round(sample["style"]["line_width_px"] * scale))
    draw.line(points, fill=fill, width=width, joint="curve")
    for point in (points[0], points[-1]):
        draw.ellipse(_box(point, width / 2), fill=fill)


def render_sample(sample):
    """Return a clean RGB input. No answer, sequence IDs, or arrows are drawn."""
    size = (sample["image"]["width"], sample["image"]["height"])
    scale = SUPERSAMPLE
    image = Image.new("RGB", tuple(v * scale for v in size), "white")
    draw = ImageDraw.Draw(image)
    _draw_path(draw, sample, scale, sample["style"]["line_color"])
    for obj in sample["objects"]:
        draw.ellipse(_box(obj["center_px"], obj["radius_px"], scale), fill=obj["hex"])
    font = ImageFont.load_default(size=18 * scale)
    for endpoint in sample["endpoints"].values():
        draw.ellipse(
            _box(endpoint["center_px"], endpoint["radius_px"], scale),
            fill="white", outline="black", width=2 * scale,
        )
        center = tuple(v * scale for v in endpoint["center_px"])
        draw.text(center, endpoint["label"], font=font, fill="black", anchor="mm")
    return image.resize(size, Image.Resampling.LANCZOS)


def render_masks(sample):
    """Return path-only binary mask and disk-only instance mask (background=0).

    The path mask includes the centerline underneath disks and endpoint badges.
    Object mask values map to objects[].mask_value; S/E are not object instances.
    """
    size = (sample["image"]["width"], sample["image"]["height"])
    scale = SUPERSAMPLE
    large_size = tuple(v * scale for v in size)
    path = Image.new("L", large_size, 0)
    _draw_path(ImageDraw.Draw(path), sample, scale, 255)
    path = path.resize(size, Image.Resampling.LANCZOS).point(lambda p: 255 if p >= 128 else 0)
    instances = Image.new("L", size, 0)
    draw = ImageDraw.Draw(instances)
    for obj in sample["objects"]:
        draw.ellipse(_box(obj["center_px"], obj["radius_px"]), fill=obj["mask_value"])
    return path, instances


def preview_sheet(samples, columns=4, thumbnail_size=256):
    """QA sheet, separate from model inputs. Captions intentionally contain answers."""
    if not samples or columns < 1:
        raise ValueError("preview requires samples and at least one column")
    columns = min(columns, len(samples))
    rows = math.ceil(len(samples) / columns)
    pad, caption = 12, 64
    cell_w, cell_h = thumbnail_size + 2 * pad, thumbnail_size + caption + 2 * pad
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "#ededed")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=12)
    for index, sample in enumerate(samples):
        x, y = (index % columns) * cell_w + pad, (index // columns) * cell_h + pad
        image = render_sample(sample).resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x, y + thumbnail_size + 5), f"{sample['id']} | {sample['family']}", fill="black", font=font)
        colors = sample["answer_colors"]
        midpoint = math.ceil(len(colors) / 2)
        draw.text((x, y + thumbnail_size + 22), " ".join(colors[:midpoint]), fill="black", font=font)
        draw.text((x, y + thumbnail_size + 38), " ".join(colors[midpoint:]) + " end", fill="black", font=font)
    return sheet
