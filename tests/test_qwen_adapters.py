"""Qwen adapter contracts use injected pipelines; no model packages or downloads."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.models import ModelConfig, QwenConfig, load_models
from services.image_edit_service import ImageEditService
from services.model_support import ModelUnavailableError
from services.qwen_layered_service import QwenLayeredService
from services.qwen_support import load_local_pipeline


def fake_torch() -> SimpleNamespace:
    """Provide only the generator/context contract used by the adapters."""
    return SimpleNamespace(Generator=MagicMock(), inference_mode=nullcontext)


class QwenLoadingTests(unittest.TestCase):
    def test_unconfigured_and_missing_paths_fail_before_import(self) -> None:
        for value in (None, "", "  ", "/missing/qwen/model"):
            with self.subTest(value=value):
                with patch("services.qwen_support.import_module") as imports:
                    with self.assertRaises(ModelUnavailableError):
                        load_local_pipeline("QwenImageEditPipeline", QwenConfig(model_path=value), "cpu")
                    imports.assert_not_called()

    def test_loader_requires_model_index_and_enforces_local_files_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QwenConfig(model_path=root)
            with self.assertRaisesRegex(ModelUnavailableError, "model_index"):
                load_local_pipeline("QwenImageEditPipeline", settings, "cpu")
            (root / "model_index.json").write_text("{}", encoding="utf-8")
            pipe_type, pipe = MagicMock(), MagicMock()
            pipe_type.from_pretrained.return_value = pipe
            pipe.to.return_value = pipe
            module = SimpleNamespace(QwenImageEditPipeline=pipe_type)
            torch = SimpleNamespace(float32="float32")
            with patch("services.qwen_support.import_module", return_value=module):
                with patch("services.qwen_support.torch_runtime", return_value=(torch, "cpu")):
                    result, _ = load_local_pipeline("QwenImageEditPipeline", settings, "cpu")
            self.assertIs(result, pipe)
            pipe_type.from_pretrained.assert_called_once_with(str(root.resolve()), torch_dtype="float32", local_files_only=True)
            pipe.to.assert_called_once_with("cpu")

    def test_qwen_paths_are_relative_to_yaml_and_blank_stays_empty(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text("qwen_layered:\n  model_path: ./layered\nqwen_image_edit:\n  model_path: '  '\n", encoding="utf-8")
            config = load_models(path)
            self.assertEqual(config.qwen_layered.model_path, path.resolve().parent / "layered")
            self.assertIsNone(config.qwen_image_edit.model_path)
            self.assertIsNone(ModelConfig().qwen_layered.model_path)

    def test_real_services_load_once_and_mock_never_imports(self) -> None:
        for service_type, namespace in (
            (QwenLayeredService, "services.qwen_layered_service"),
            (ImageEditService, "services.image_edit_service"),
        ):
            with self.subTest(service=service_type.__name__):
                with patch(f"{namespace}.load_local_pipeline", return_value=(MagicMock(), fake_torch())) as load:
                    service = service_type(False)
                    load.assert_not_called()
                    service.load()
                    service.load()
                    load.assert_called_once()
                    service_type(True).load()
                    load.assert_called_once()


class QwenImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "source.png"
        pixels = np.full((12, 20, 4), [20, 60, 40, 255], dtype=np.uint8)
        pixels[:, :3, 3] = 0
        Image.fromarray(pixels).save(self.source)
        self.mask = self.root / "mask.png"
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask[3:8, 6:11] = 255
        Image.fromarray(mask).save(self.mask)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def layered(self, images: list) -> QwenLayeredService:
        service = QwenLayeredService(False)
        service._pipeline = MagicMock(return_value=SimpleNamespace(images=images))
        service._torch = fake_torch()
        return service

    def test_layered_nested_rgba_output_retains_alpha_and_source_coordinates(self) -> None:
        service = self.layered([[Image.new("RGBA", (10, 6), (255, 20, 30, 128)), Image.new("RGBA", (10, 6), (0, 30, 50, 0))]])
        records = service.decompose_layers(str(self.source), output_dir=self.root / "layers")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["model_size"], [10, 6])
        self.assertEqual(records[0]["canvas_size"], [20, 12])
        self.assertFalse(records[0]["mock"])
        self.assertEqual(records[0]["status"], "manual_review")
        with Image.open(records[0]["asset_path"]) as layer:
            self.assertEqual(layer.size, (20, 12))
            self.assertEqual(layer.getchannel("A").getextrema(), (128, 128))
        args = service._pipeline.call_args.kwargs
        self.assertEqual(args["layers"], 4)
        self.assertEqual(args["resolution"], 640)
        self.assertEqual(args["image"].mode, "RGBA")

    def test_layered_rejects_missing_alpha_empty_and_inconsistent_canvases(self) -> None:
        for images in ([], [Image.new("RGB", (10, 6))], [[Image.new("RGBA", (10, 6)), Image.new("RGBA", (9, 6))]]):
            with self.subTest(images=images), self.assertRaises(ValueError):
                self.layered(images).decompose_layers(str(self.source), output_dir=self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_edit_returns_rgb_for_new_segmentation_without_source_alpha(self) -> None:
        original_bytes = self.source.read_bytes()
        service = ImageEditService(False)
        service._pipeline = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (40, 24), (200, 10, 90))]))
        service._torch = fake_torch()
        result = service.complete_object(str(self.source), str(self.mask), prompt="Repair stone texture", output_path=self.root / "edited.png")
        with Image.open(result) as image, Image.open(self.source) as original, Image.open(self.mask) as mask:
            out, src, edit_mask = np.array(image), np.array(original), np.array(mask)
        self.assertEqual(out.shape[2], 3)
        np.testing.assert_array_equal(out[edit_mask == 0], src[edit_mask == 0, :3])
        np.testing.assert_array_equal(out[edit_mask > 0, :3], np.tile([200, 10, 90], (25, 1)))
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertNotIn("mask_image", service._pipeline.call_args.kwargs)
        self.assertEqual(service._pipeline.call_args.kwargs["prompt"], "Repair stone texture")

    def test_bad_edit_inputs_fail_before_loading(self) -> None:
        service = ImageEditService(False)
        empty = self.root / "empty.png"
        Image.new("L", (20, 12)).save(empty)
        mismatch = self.root / "mismatch.png"
        Image.new("L", (1, 1), 255).save(mismatch)
        for mask, prompt, output in ((empty, "repair", self.root / "out.png"), (mismatch, "repair", self.root / "out.png"), (self.mask, " ", self.root / "out.png"), (self.mask, "repair", self.source)):
            with self.subTest(mask=mask, prompt=prompt), patch.object(service, "load") as load:
                with self.assertRaises(ValueError):
                    service.complete_object(str(self.source), str(mask), prompt=prompt, output_path=output)
                load.assert_not_called()

    def test_mock_outputs_do_not_claim_semantic_decomposition_or_editing(self) -> None:
        with patch("services.qwen_support.import_module") as imports:
            layers = QwenLayeredService().decompose_layers(str(self.source), output_dir=self.root / "layers")
            edited = ImageEditService().complete_object(str(self.source), str(self.mask), prompt="Repair", output_path=self.root / "edited.png")
            imports.assert_not_called()
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["status"], "mock_passthrough")
        with Image.open(edited) as image, Image.open(self.source) as source:
            np.testing.assert_array_equal(image, source)

    def test_edit_rejects_invalid_model_output(self) -> None:
        service = ImageEditService(False)
        service._torch = fake_torch()
        service._pipeline = MagicMock(return_value=SimpleNamespace(images=[]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            service.complete_object(str(self.source), str(self.mask), prompt="repair", output_path=self.root / "edited.png")


if __name__ == "__main__":
    unittest.main()
