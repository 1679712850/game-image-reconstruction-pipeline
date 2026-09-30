"""Integration tests against the actual LangGraph runtime."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from app.config import CropConfig, PipelineConfig, ReconstructionConfig, UpscaleConfig
from schemas.scene import SceneManifest
from services.grounding_service import GroundingService
from services.runtime import ServiceBundle
from services.sam_service import SAMService


class EmptySAM(SAMService):
    """Persistent failure injection verifies that retry exhaustion terminates."""

    def segment(self, image_path: str, detections: list[dict]) -> list[dict]:
        outputs = super().segment(image_path, detections)
        return [{**obj, "mask": np.zeros_like(obj["mask"])} for obj in outputs]


class NoDetections(GroundingService):
    """A legitimate empty detection result must still reach export."""

    def detect(self, image_path: str, categories: list[str]) -> list[dict]:
        return []


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (180, 140), (40, 130, 60)).save(self.source)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def initial(self, name: str = "result") -> dict:
        return {"source_path": str(self.source), "output_dir": str(self.root / name)}

    def test_mock_graph_reaches_export_with_portable_metadata(self) -> None:
        visits = []
        initial = self.initial()
        state = build_graph(progress=visits.append).invoke(initial)
        self.assertEqual(visits[-1], "export")
        self.assertEqual(visits.count("qa_scene"), 3)
        self.assertEqual(visits.count("update_remaining"), 3)
        self.assertEqual(state["scene_stop_reason"], "max_rounds")
        self.assertEqual(initial, self.initial())  # Input state remains untouched.
        self.assertEqual(len(state["objects"]), 4)
        self.assertEqual(state["failed_objects"], [])
        self.assertEqual(state["retry_count"], 0)
        json.dumps(state)  # No NumPy arrays/model objects in checkpoint state.
        manifest = SceneManifest.model_validate_json(Path(state["scene_json"]).read_text())
        self.assertTrue(manifest.mock)
        for obj in manifest.objects:
            self.assertFalse(Path(obj.asset).is_absolute())
            with Image.open(Path(state["output_dir"]) / obj.asset) as asset:
                self.assertLess(asset.width, state["width"])
                self.assertLess(asset.height, state["height"])
                self.assertEqual(list(asset.size), list(obj.logical_size))
                self.assertEqual(asset.mode, "RGBA")
            self.assertEqual(obj.z_order, obj.crop_bbox.y + obj.pivot.y)
            self.assertEqual(obj.texture_size, tuple(round(v * obj.texture_scale) for v in obj.logical_size))
        self.assertEqual(manifest.reconstruction, "reconstruction.png")
        self.assertTrue(all(Path(path).exists() for path in state["exported_assets"]))

    def test_forced_mock_retry_reenters_qa_once(self) -> None:
        visits = []
        state = build_graph(PipelineConfig(exercise_retry=True), progress=visits.append).invoke(self.initial())
        self.assertEqual(visits.count("retry_objects"), 1)
        self.assertEqual(visits.count("qa_objects"), state["detection_round"] + 1)
        self.assertEqual(state["retry_count"], 1)
        self.assertEqual(state["failed_objects"], [])
        first = state["objects"][0]
        self.assertEqual(first["confidence"], 0.1)
        self.assertTrue(first["mock_retry_resolved"])
        self.assertIn("simulation", first["qa"]["reason"])

    def test_zero_retry_budget_exports_manual_review(self) -> None:
        config = PipelineConfig(exercise_retry=True, max_retry=0)
        visits = []
        state = build_graph(config, progress=visits.append).invoke(self.initial())
        self.assertNotIn("retry_objects", visits)
        self.assertEqual(state["objects"][0]["status"], "manual_review")
        self.assertEqual(len(state["failed_objects"]), 1)

    def test_persistent_empty_masks_terminate_without_fake_assets(self) -> None:
        services = replace(ServiceBundle.create(), sam=EmptySAM())
        visits = []
        graph = build_graph(PipelineConfig(max_retry=2), services, progress=visits.append)
        state = graph.invoke(self.initial(), {"recursion_limit": 30})
        self.assertEqual(visits.count("retry_objects"), 4)
        self.assertEqual(state["detection_round"], 2)
        self.assertEqual(state["scene_stop_reason"], "no_progress")
        self.assertEqual(len(state["failed_objects"]), 4)
        self.assertTrue(all(obj["asset_path"] is None for obj in state["objects"]))
        self.assertTrue(all(obj["status"] == "manual_review" for obj in state["objects"]))
        self.assertTrue(Path(state["scene_json"]).exists())

    def test_no_detections_still_exports_empty_scene(self) -> None:
        services = replace(ServiceBundle.create(), grounding=NoDetections())
        state = build_graph(services=services).invoke(self.initial())
        self.assertEqual(state["objects"], [])
        self.assertEqual(state["failed_objects"], [])
        self.assertTrue(Path(state["reconstruction_path"]).exists())

    def test_configuration_controls_crop_and_optional_stages(self) -> None:
        config = PipelineConfig(
            crop=CropConfig(padding=0), upscale=UpscaleConfig(enabled=False),
            reconstruction=ReconstructionConfig(enabled=False),
        )
        state = build_graph(config).invoke(self.initial())
        for obj in state["objects"]:
            self.assertEqual(obj["crop_bbox"], obj["bbox"])
            self.assertIsNone(obj["hd_asset_path"])
            self.assertEqual(obj["texture_scale"], 1)
        self.assertEqual(state["reconstruction_path"], "")
        manifest = SceneManifest.model_validate_json(Path(state["scene_json"]).read_text())
        self.assertIsNone(manifest.reconstruction)
        self.assertIsNone(manifest.reconstruction_score)

    def test_checkpoint_interrupt_and_resume_reach_end(self) -> None:
        saver = InMemorySaver()
        graph = build_graph(checkpointer=saver, interrupt_before=["upscale_objects"])
        run_config = {"configurable": {"thread_id": "review-test"}}
        paused = graph.invoke(self.initial(), run_config)
        self.assertNotIn("scene_json", paused)
        self.assertEqual(graph.get_state(run_config).next, ("upscale_objects",))
        result = graph.invoke(None, run_config)
        self.assertTrue(Path(result["scene_json"]).exists())
        self.assertEqual(graph.get_state(run_config).next, ())
