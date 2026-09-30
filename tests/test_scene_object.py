"""Schemas reject invalid geometry and retain required metadata."""
import unittest

from pydantic import ValidationError

from schemas.object import SceneObject
from schemas.scene import SceneAnalysis


class SchemaTests(unittest.TestCase):
    def test_required_object_fields(self) -> None:
        obj = SceneObject(id="tree_001", category="tree", confidence=.9, bbox={"x": 1, "y": 2, "w": 3, "h": 4})
        self.assertEqual(obj.bbox.w, 3)
        self.assertIn("texture_scale", obj.model_dump())

    def test_invalid_box_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SceneObject(id="x", category="rock", confidence=.5, bbox={"x": 0, "y": 0, "w": 0, "h": 1})

    def test_invalid_projection_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SceneAnalysis(projection="flat", categories=[], description="invalid")
