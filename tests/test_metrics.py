"""CV metric and texture scale boundary tests."""
import unittest

import numpy as np

from cv.metrics import bbox_area, occupancy, touches_edge
from cv.pivot import ground_pivot
from services.upscale_service import choose_scale


class MetricTests(unittest.TestCase):
    def test_mask_metrics(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        self.assertEqual(occupancy(mask), .25)
        self.assertEqual(bbox_area({"x": 0, "y": 0, "w": 4, "h": 4}), 16)
        self.assertFalse(touches_edge(mask))
        mask[0, 1] = 255
        self.assertTrue(touches_edge(mask))

    def test_ground_pivot_is_in_logical_crop_coordinates(self) -> None:
        mask = np.zeros((10, 8), dtype=np.uint8)
        mask[4:8, 2:6] = 255
        self.assertEqual(ground_pivot(mask), {"x": 4.0, "y": 7.5})

    def test_scale_boundaries(self) -> None:
        for edge, scale in ((127, 4), (128, 3), (256, 3), (257, 2), (512, 2), (513, 1)):
            with self.subTest(edge=edge):
                self.assertEqual(choose_scale(edge, 10), scale)
