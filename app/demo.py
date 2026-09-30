"""Create a small original synthetic isometric scene for the mock demo."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from services.grounding_service import GroundingService


def create_demo(path: Path, width: int = 640, height: int = 480) -> Path:
    """Draw four simple scene objects aligned with the mock detection regions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), "#76955d")
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, height // 2), (width // 2, 0), (width, height // 2), (width // 2, height)], fill="#a8be7a")
    for y in range(0, height, 24):
        draw.line([(0, y), (width, min(height, y + width // 3))], fill="#8fa66e")
    image.save(path)
    detections = GroundingService(mock=True).detect(str(path), ["tree", "rock", "building", "mountain"])
    for item in detections:
        box = item["bbox"]
        x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
        if item["category"] == "tree":
            draw.rectangle((x + w * .42, y + h * .48, x + w * .58, y + h - 1), fill="#775437")
            draw.ellipse((x, y, x + w - 1, y + h * .76), fill="#365f40", outline="#244934", width=3)
            draw.ellipse((x + w * .12, y + h * .10, x + w * .74, y + h * .57), fill="#59824a")
        elif item["category"] == "rock":
            draw.polygon([(x, y + h * .7), (x + w * .25, y), (x + w * .75, y + h * .12), (x + w - 1, y + h * .65), (x + w * .6, y + h - 1)], fill="#7c8581", outline="#56635e")
        elif item["category"] == "building":
            draw.polygon([(x, y + h * .35), (x + w * .55, y + h * .55), (x + w * .55, y + h - 1), (x, y + h * .75)], fill="#d7c39b")
            draw.polygon([(x + w * .55, y + h * .55), (x + w - 1, y + h * .32), (x + w - 1, y + h * .75), (x + w * .55, y + h - 1)], fill="#a98e67")
            draw.polygon([(x, y + h * .35), (x + w * .4, y), (x + w - 1, y + h * .32), (x + w * .55, y + h * .55)], fill="#8e5144", outline="#653c38")
        else:
            draw.polygon([(x, y + h - 1), (x + w * .35, y), (x + w * .66, y + h - 1)], fill="#889987")
            draw.polygon([(x + w * .34, y + h - 1), (x + w * .65, y + h * .18), (x + w - 1, y + h - 1)], fill="#586f67")
            draw.polygon([(x + w * .22, y + h * .38), (x + w * .35, y), (x + w * .46, y + h * .34)], fill="#dfe1ca")
    image.save(path, "PNG")
    return path


def main() -> None:
    """Create the reproducible demo without any model downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("input/test.png"))
    args = parser.parse_args()
    print(create_demo(args.output).resolve())


if __name__ == "__main__":
    main()
