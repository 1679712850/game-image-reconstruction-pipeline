"""Tight bbox geometry and empty-mask regression tests."""
import unittest

import numpy as np

from cv.bbox import get_tight_bbox


class BBoxTests(unittest.TestCase):
    def test_exact_bbox_uses_exclusive_right_bottom(self) -> None:
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:10, 8:12] = 255
        self.assertEqual(get_tight_bbox(mask, padding=0), {"x": 8, "y": 5, "w": 4, "h": 5})

    def test_padding_is_clipped_to_image(self) -> None:
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[0:2, 10:12] = 255
        self.assertEqual(get_tight_bbox(mask, padding=16), {"x": 0, "y": 0, "w": 12, "h": 10})

    def test_threshold_is_strict(self) -> None:
        mask = np.array([[8, 9]], dtype=np.uint8)
        self.assertEqual(get_tight_bbox(mask, padding=0), {"x": 1, "y": 0, "w": 1, "h": 1})

    def test_empty_mask_returns_none(self) -> None:
        self.assertIsNone(get_tight_bbox(np.zeros((5, 5), dtype=np.uint8)))

    def test_bool_mask_is_supported(self) -> None:
        self.assertEqual(get_tight_bbox(np.array([[False, True]]), padding=0), {"x": 1, "y": 0, "w": 1, "h": 1})

    def test_invalid_padding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_tight_bbox(np.ones((2, 2), dtype=np.uint8), padding=-1)
