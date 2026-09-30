"""Render absolute xyxy candidates and legacy xywh scene objects."""
from pathlib import Path
from PIL import Image, ImageDraw


def open_image(source):
    if isinstance(source, Image.Image):
        return source.convert("RGB").copy()
    with Image.open(source) as image:
        return image.convert("RGB")


def xyxy(candidate):
    box = candidate.get("bbox")
    if isinstance(box, dict):
        return box["x"], box["y"], box["x"]+box["w"], box["y"]+box["h"]
    return tuple(box) if box and len(box) == 4 else None


def draw_candidates(source, output_path, candidates):
    image = open_image(source)
    draw = ImageDraw.Draw(image)
    for candidate in candidates:
        box = xyxy(candidate)
        if not box:
            continue
        x, y, r, b = box
        if r <= x or b <= y:
            continue
        color = "orange" if candidate.get("is_truncated") else "lime"
        draw.rectangle((x, y, r-1, b-1), outline=color, width=max(1, image.width//800))
        score = candidate.get('confidence')
        label = f"{candidate.get('category', '?')} {score:.2f}" if score is not None else candidate.get('category', '?')
        label += f" {candidate.get('tile_id') or candidate.get('source', '')}"
        draw.text((x+2, max(0, y-12)), label, fill=color, stroke_width=1, stroke_fill="black")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)
