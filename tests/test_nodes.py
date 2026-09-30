"""Exercise state-only node contracts without an orchestration dependency."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from agent.routers import route_after_qa
from app.config import PipelineConfig
from nodes.analyze_scene import make_analyze_scene
from nodes.crop_objects import make_crop_objects
from nodes.detect_instances import make_detect_instances
from nodes.load_image import load_image
from nodes.plan_layers import make_plan_layers
from nodes.qa_objects import make_qa_objects
from nodes.refine_masks import make_refine_masks
from nodes.retry_objects import make_retry_objects
from nodes.segment_instances import make_segment_instances
from services.runtime import ServiceBundle
from services.sam_service import SAMService


class EmptyMaskService(SAMService):
    def segment(self, image_path: str, detections: list[dict]) -> list[dict]:
        return [{**obj, "mask": np.zeros_like(obj["mask"])} for obj in super().segment(image_path, detections)]


class NodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (180, 140), (20, 100, 60)).save(self.source)
        self.config = PipelineConfig(exercise_retry=True)
        self.services = ServiceBundle.create()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def state_through_crop(self, sam: SAMService | None = None) -> dict:
        state = {"source_path": str(self.source), "output_dir": str(self.root / "out"), "retry_count": 0, "max_retry": 1}
        taxonomy = Path(__file__).resolve().parents[1] / "config" / "categories.yaml"
        nodes = [
            load_image, make_analyze_scene(self.services.vlm), make_plan_layers(taxonomy),
            make_detect_instances(self.services.grounding, exercise_retry=True),
            make_segment_instances(sam or self.services.sam),
            make_refine_masks(self.config.crop.alpha_threshold), make_crop_objects(self.config.crop),
        ]
        for node in nodes:
            before = deepcopy(state)
            update = node(state)
            self.assertEqual(state, before)
            state.update(update)
            json.dumps(state)
        return state

    def test_retry_only_changes_failed_records_and_keeps_confidence(self) -> None:
        state = self.state_through_crop()
        qa = make_qa_objects(self.config)
        state.update(qa(state))
        self.assertEqual(state["failed_objects"], ["tree_001"])
        untouched = deepcopy(state["objects"][1:])
        state.update(make_retry_objects(self.config, self.services.sam)(state))
        self.assertEqual(state["objects"][1:], untouched)
        self.assertEqual(state["objects"][0]["confidence"], .1)
        state.update(qa(state))
        self.assertEqual(state["failed_objects"], [])
        self.assertEqual(route_after_qa(state), "continue")

    def test_persistent_empty_masks_become_manual_review(self) -> None:
        sam = EmptyMaskService()
        state = self.state_through_crop(sam)
        qa = make_qa_objects(self.config)
        state.update(qa(state))
        self.assertEqual(route_after_qa(state), "retry")
        state.update(make_retry_objects(self.config, sam)(state))
        state.update(qa(state))
        self.assertEqual(route_after_qa(state), "continue")
        self.assertEqual(len(state["failed_objects"]), 4)
        self.assertTrue(all(obj["status"] == "manual_review" for obj in state["objects"]))
        self.assertTrue(all(obj["asset_path"] is None for obj in state["objects"]))

    def test_retry_guard_does_not_exceed_budget(self) -> None:
        retry = make_retry_objects(self.config, self.services.sam)
        self.assertEqual(retry({"retry_count": 1, "max_retry": 1}), {})
