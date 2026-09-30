"""Coordinate placement, transparent padding and alpha/Y-order tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from cv.bbox import get_tight_bbox
from cv.crop import crop_rgba_by_mask
from cv.reconstruct import reconstruct_scene


class ReconstructionTests(unittest.TestCase):
    def test_crop_roundtrip_preserves_source_coordinates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (40, 30), (231, 22, 67))
            mask = np.zeros((30, 40), dtype=np.uint8)
            mask[12:16, 21:26] = 255
            box = get_tight_bbox(mask, padding=2)
            crop_rgba_by_mask(source, mask, box, root / "asset.png")
            path = reconstruct_scene(40, 30, [{"asset_path": str(root / "asset.png"), "crop_bbox": box, "z_order": 16}], root / "preview.png")
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((21, 12)), (231, 22, 67, 255))
                self.assertEqual(image.getpixel((20, 12))[3], 0)
                self.assertEqual(image.getpixel((25, 15))[3], 255)
                self.assertEqual(image.getpixel((26, 15))[3], 0)

    def test_z_order_blends_translucent_foreground(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(root / "back.png")
            Image.new("RGBA", (2, 2), (0, 0, 255, 128)).save(root / "front.png")
            box = {"x": 2, "y": 3, "w": 2, "h": 2}
            objects = [
                {"asset_path": str(root / "front.png"), "crop_bbox": box, "z_order": 9},
                {"asset_path": str(root / "back.png"), "crop_bbox": box, "z_order": 1},
            ]
            reconstruct_scene(8, 8, objects, root / "preview.png")
            with Image.open(root / "preview.png") as image:
                self.assertEqual(image.getpixel((2, 3)), (127, 0, 128, 255))

    def test_empty_scene_reconstructs_transparent_canvas(self) -> None:
        with TemporaryDirectory() as directory:
            output = reconstruct_scene(5, 7, [], Path(directory) / "empty.png")
            with Image.open(output) as image:
                self.assertEqual(image.size, (5, 7))
                self.assertIsNone(image.getbbox())
