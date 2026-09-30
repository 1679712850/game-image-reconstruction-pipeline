"""True RGB/alpha preservation in compact crops."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from cv.crop import crop_rgba_by_mask


class CropTests(unittest.TestCase):
    def test_crop_is_compact_and_preserves_rgb_and_alpha(self) -> None:
        source = Image.new("RGB", (100, 80), (10, 20, 30))
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:30, 40:50] = 180
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tree.png"
            crop = crop_rgba_by_mask(source, mask, output_path=path, padding=2)
            self.assertEqual(crop.size, (14, 14))
            self.assertEqual(crop.getpixel((2, 2)), (10, 20, 30, 180))
            self.assertEqual(crop.getpixel((0, 0)), (10, 20, 30, 0))
            with Image.open(path) as saved:
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.size, crop.size)

    def test_empty_mask_does_not_write_a_fake_asset(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "empty.png"
            result = crop_rgba_by_mask(Image.new("RGB", (10, 10)), np.zeros((10, 10), dtype=np.uint8), output_path=target)
            self.assertIsNone(result)
            self.assertFalse(target.exists())

    def test_mask_dimensions_must_match(self) -> None:
        with self.assertRaises(ValueError):
            crop_rgba_by_mask(Image.new("RGB", (10, 10)), np.ones((5, 5), dtype=np.uint8))

    def test_explicit_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            crop_rgba_by_mask(Image.new("RGB", (10, 10)), np.full((10, 10), 255, dtype=np.uint8), {"x": 9, "y": 0, "w": 2, "h": 2})
