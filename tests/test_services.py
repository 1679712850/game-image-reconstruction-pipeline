"""All service boundaries are usable in mock mode without model downloads."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from services.runtime import ServiceBundle


class ServiceTests(unittest.TestCase):
    def test_mock_services_support_even_tiny_images(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.png"
            Image.new("RGB", (1, 1), (20, 60, 40)).save(path)
            services = ServiceBundle.create(mock=True)
            analysis = services.vlm.analyze_scene(str(path))
            self.assertEqual(analysis.projection, "isometric")
            detections = services.grounding.detect(str(path), analysis.categories)
            self.assertEqual(len(detections), 4)
            self.assertTrue(all(obj["bbox"] == {"x": 0, "y": 0, "w": 1, "h": 1} for obj in detections))
            masks = services.sam.segment(str(path), detections)
            self.assertTrue(all(obj["mask"].shape == (1, 1) for obj in masks))
            hd = services.upscale.upscale(str(path), 4)
            with Image.open(hd) as image:
                self.assertEqual(image.size, (4, 4))
            self.assertTrue(services.layered.decompose_layers(str(path))[0]["mock"])
            self.assertEqual(services.image_edit.analyze_occlusion([]), [])
            self.assertEqual(services.image_edit.complete_object(str(path), str(path)), str(path.resolve()))

    def test_real_services_fail_explicitly_until_configured(self) -> None:
        services = ServiceBundle.create(mock=False)
        with self.assertRaises(NotImplementedError):
            services.vlm.analyze_scene("missing.png")
        with self.assertRaises(NotImplementedError):
            services.grounding.detect("missing.png", [])
        with self.assertRaises(NotImplementedError):
            services.sam.segment("missing.png", [])
        with self.assertRaises(NotImplementedError):
            services.upscale.upscale("missing.png", 2)
