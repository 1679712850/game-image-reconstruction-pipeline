"""Optional Qwen stages, request validation and portable export integration."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from app.config import OptionalStageConfig, PipelineConfig
from app.edits import load_edit_requests, validate_edit_requests
from schemas.scene import SceneManifest
from services.model_support import ModelUnavailableError
from services.runtime import ServiceBundle


class QwenPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (180, 140), (40, 130, 60)).save(self.source)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def initial(self, name: str) -> dict:
        return {"source_path": str(self.source), "output_dir": str(self.root / name)}

    def test_enabled_mock_stages_export_portable_candidates_and_reach_end(self) -> None:
        config = PipelineConfig(
            layer_decomposition=OptionalStageConfig(enabled=True),
            object_completion=OptionalStageConfig(enabled=True), exercise_retry=True,
        )
        saver = InMemorySaver()
        visits = []
        graph = build_graph(config, checkpointer=saver, interrupt_before=["complete_objects"], progress=visits.append)
        run = {"configurable": {"thread_id": "qwen-test"}}
        paused = graph.invoke(self.initial("result"), run)
        self.assertEqual(graph.get_state(run).next, ("complete_objects",))
        obj = paused["objects"][0]
        mask = self.root / "edit.png"
        Image.new("L", (obj["crop_bbox"]["w"], obj["crop_bbox"]["h"]), 255).save(mask)
        graph.update_state(run, {"edit_requests": [{"object_id": obj["id"], "mask_path": str(mask), "prompt": "Repair the tree"}]})
        result = graph.invoke(None, run)
        self.assertEqual(graph.get_state(run).next, ())
        self.assertEqual(visits.count("decompose_layers"), 1)
        self.assertEqual(visits.count("complete_objects"), 1)
        self.assertEqual(visits.count("retry_objects"), 1)
        self.assertLess(visits.index("plan_layers"), visits.index("decompose_layers"))
        self.assertLess(visits.index("complete_objects"), visits.index("upscale_objects"))
        json.dumps(result)
        manifest = SceneManifest.model_validate_json(Path(result["scene_json"]).read_text())
        self.assertEqual(manifest.schema_version, "1.2")
        self.assertEqual(manifest.backends["object_completion"], "mock_noop")
        self.assertEqual(manifest.decomposed_layers[0].status, "mock_passthrough")
        edit = manifest.object_edits[0]
        self.assertEqual(edit.status, "mock_noop")
        self.assertGreater(edit.crop_bbox.w, obj["crop_bbox"]["w"])
        self.assertGreater(edit.crop_bbox.h, obj["crop_bbox"]["h"])
        self.assertEqual(edit.alpha_policy, "visible_hint")
        root = Path(result["output_dir"])
        for path in (edit.asset_path, edit.source_asset_path, edit.mask_path, manifest.decomposed_layers[0].asset_path):
            self.assertFalse(Path(path).is_absolute())
            self.assertTrue((root / path).is_file())
            self.assertIn(str((root / path).resolve()), result["exported_assets"])
        with Image.open(root / edit.asset_path) as edited, Image.open(root / edit.source_asset_path) as original:
            np.testing.assert_array_equal(edited, original)
        self.assertEqual(result["objects"][0]["asset_path"], obj["asset_path"])

    def test_empty_models_allowed_when_disabled_and_rejected_before_real_run(self) -> None:
        # Isolate optional image generation from the mandatory VLM preflight.
        mock = ServiceBundle.create()
        real = replace(ServiceBundle.create(mock=False), vlm=mock.vlm, reviewer=mock.reviewer)
        with patch.object(real.grounding, "load") as load:
            build_graph(PipelineConfig(mock=False, scene_loop={'reviewer': 'rules'}), real)
            for stage in ("layer_decomposition", "object_completion"):
                with self.subTest(stage=stage), self.assertRaisesRegex(ModelUnavailableError, "model_path is empty"):
                    build_graph(PipelineConfig.model_validate({"mock": False, 'scene_loop': {'reviewer': 'rules'}, stage: {"enabled": True}}), real)
            load.assert_not_called()

    def test_empty_edit_requests_skip_inference(self) -> None:
        services = ServiceBundle.create()
        config = PipelineConfig(object_completion=OptionalStageConfig(enabled=True))
        with patch.object(services.image_edit, "complete_object") as edit:
            result = build_graph(config, services).invoke(self.initial("empty"))
            edit.assert_not_called()
        self.assertEqual(result["object_edits"], [])

    def test_unknown_ids_and_disabled_requests_are_not_silently_ignored(self) -> None:
        initial = {**self.initial("bad"), "edit_requests": [{"object_id": "missing", "mask_path": str(self.source), "prompt": "repair"}]}
        with self.assertRaisesRegex(ValueError, "requires object_completion"):
            build_graph().invoke(initial)
        config = PipelineConfig(object_completion=OptionalStageConfig(enabled=True))
        with self.assertRaisesRegex(ValueError, "unknown/uncropped"):
            build_graph(config).invoke(initial)

    def test_request_file_resolves_paths_and_rejects_duplicates(self) -> None:
        path = self.root / "requests.json"
        request = {"object_id": "tree_001", "mask_path": "source.png", "prompt": "repair"}
        path.write_text(json.dumps([request]), encoding="utf-8")
        self.assertEqual(load_edit_requests(path)[0]["mask_path"], str(self.source.resolve()))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_edit_requests([request, request])
        with self.assertRaises(ValueError):
            validate_edit_requests([{**request, "object_id": "../escape"}])
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON list"):
            load_edit_requests(path)


if __name__ == "__main__":
    unittest.main()
