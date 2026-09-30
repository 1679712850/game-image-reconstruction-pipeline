"""Local VLM contracts without model weights, torch, GPU or network access."""
import base64
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.config import PipelineConfig, load_config
from app.models import ModelConfig, QwenVLConfig, load_models
from nodes.scene_loop import make_qa_scene
from schemas.scene import SceneAnalysis
from schemas.scene_qa import SceneReviewDecision
from services.execution import ExecutionRuntime
from services.model_support import ModelUnavailableError
from services.qwen_vl_backend import QwenVLBackend
from services.runtime import ServiceBundle


class Inputs(dict):
    """Minimal tensor batch recording device placement."""

    def to(self, device: str) -> 'Inputs':
        self.device = device
        return self


class QwenVLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        Image.new('RGB', (400, 200), 'green').save(self.source)
        self.weights = self.root / 'model'
        self.weights.mkdir()
        (self.weights / 'config.json').write_text('{"model_type":"qwen2_5_vl"}')
        (self.weights / 'model.safetensors').write_bytes(b'contract-test-placeholder')
        self.config = ModelConfig(device='cpu', qwen_vl=QwenVLConfig(model_path=self.weights, image_long_edge=256))
        self.backend = QwenVLBackend(self.config)

    def messages(self, count: int = 1) -> list:
        buffer = BytesIO()
        with Image.open(self.source) as image:
            image.save(buffer, 'PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        return [HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': url}}] * count)]

    def fake_generation(self, text: str) -> tuple:
        inputs = Inputs(input_ids=np.array([[10, 11, 12]]))
        processor = MagicMock()
        processor.apply_chat_template.return_value = inputs
        processor.batch_decode.return_value = [text]
        model = MagicMock()
        model.generate.return_value = np.array([[10, 11, 12, 90, 91]])
        self.backend._processor, self.backend._model = processor, model
        self.backend._torch = SimpleNamespace(inference_mode=nullcontext)
        self.backend._device = 'cpu'
        return inputs, processor, model

    def test_default_configuration_and_blank_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(load_config(root / 'config/pipeline.yaml').scene_loop.reviewer, 'llm')
        self.assertIsNone(load_models(root / 'config/models.yaml').qwen_vl.model_path)
        for blank in (None, '', '  '):
            backend = QwenVLBackend(ModelConfig(qwen_vl={'model_path': blank}))
            with patch('services.qwen_vl_backend.torch_runtime') as runtime:
                with self.assertRaisesRegex(ModelUnavailableError, 'qwen_vl.model_path is empty'):
                    backend.load()
                runtime.assert_not_called()
        path = self.root / 'models.yaml'
        path.write_text('qwen_vl:\n  model_path: ./model\n')
        self.assertEqual(load_models(path).qwen_vl.model_path.resolve(), self.weights.resolve())

    def test_local_directory_preflight_rejects_invalid_weights(self) -> None:
        self.assertEqual(self.backend.validate_ready(), self.weights.resolve())
        for metadata in ('{}', '[]', 'broken', '{"model_type":"qwen3_vl_moe"}'):
            (self.weights / 'config.json').write_text(metadata)
            with self.assertRaises(ModelUnavailableError):
                self.backend.validate_ready()
        (self.weights / 'config.json').write_text('{"model_type":"qwen3_vl"}')
        self.assertEqual(self.backend.validate_ready(), self.weights.resolve())
        (self.weights / 'model.safetensors').unlink()
        with self.assertRaisesRegex(ModelUnavailableError, 'weights are missing'):
            self.backend.validate_ready()

    def test_loader_is_lazy_local_only_and_reused(self) -> None:
        processor_factory, model_factory = MagicMock(), MagicMock()
        fake_transformers = SimpleNamespace(AutoProcessor=processor_factory, AutoModelForImageTextToText=model_factory)
        fake_torch = SimpleNamespace(float32='fp32')
        self.assertIsNone(self.backend._model)
        with patch.dict('sys.modules', {'transformers': fake_transformers}), patch(
            'services.qwen_vl_backend.torch_runtime', return_value=(fake_torch, 'cpu'),
        ):
            self.backend.load()
            self.backend.load()
        processor_factory.from_pretrained.assert_called_once_with(
            str(self.weights.resolve()), local_files_only=True, trust_remote_code=False)
        model_factory.from_pretrained.assert_called_once_with(
            str(self.weights.resolve()), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype='fp32', attn_implementation='sdpa')
        model_factory.from_pretrained.return_value.to.assert_called_once_with('cpu')
        model_factory.from_pretrained.return_value.to.return_value.eval.assert_called_once()

    def test_generation_handles_images_schema_and_only_new_tokens(self) -> None:
        expected = SceneAnalysis(projection='isometric', categories=['tree'], description='树木')
        inputs, processor, model = self.fake_generation(expected.model_dump_json())
        result = self.backend.with_structured_output(SceneAnalysis).invoke(self.messages(2))
        self.assertEqual(result, expected)
        conversation = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(conversation[0]['role'], 'system')
        self.assertIn('JSON schema:', conversation[0]['content'][0]['text'])
        self.assertEqual([b['image'].size for b in conversation[1]['content']], [(256, 128)] * 2)
        self.assertTrue(all(b['image'].mode == 'RGB' for b in conversation[1]['content']))
        self.assertEqual(processor.apply_chat_template.call_args.kwargs,
                         dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'))
        np.testing.assert_array_equal(processor.batch_decode.call_args.args[0][0], [90, 91])
        self.assertEqual(inputs.device, 'cpu')
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])
        self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], 2048)

    def test_bounds_reject_images_urls_and_excess_context(self) -> None:
        with self.assertRaisesRegex(ValueError, 'max_images'):
            self.backend.generate(self.messages(5), SceneAnalysis)
        self.assertIsNone(self.backend._model)
        remote = [HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image.png'}}])]
        with self.assertRaisesRegex(ValueError, 'not remote URLs'):
            self.backend.generate(remote, SceneAnalysis)
        inputs, _, model = self.fake_generation('{}')
        inputs['input_ids'] = np.zeros((1, 16385), dtype=int)
        with self.assertRaisesRegex(ValueError, 'max_input_tokens'):
            self.backend.generate(self.messages(), SceneAnalysis)
        model.generate.assert_not_called()
        self.assertFalse(hasattr(inputs, 'device'))

    def test_json_fences_validate_and_malformed_output_never_passes(self) -> None:
        structured = self.backend.with_structured_output(SceneReviewDecision)
        good = '{"continue_detection":false,"reason":"No further objects","status":"sufficient"}'
        with patch.object(self.backend, 'generate', return_value='```json\n' + good + '\n```'):
            self.assertEqual(structured.invoke([]).status, 'sufficient')
        for value in ('{"status":', '{}', good.replace('sufficient', 'invalid'), 'Thinking...\n' + good):
            with self.subTest(value=value), patch.object(self.backend, 'generate', return_value=value):
                with self.assertRaises(ValidationError):
                    structured.invoke([])

    def test_analysis_and_scene_review_share_multimodal_backend(self) -> None:
        bundle = ServiceBundle.create(False, self.config)
        backend = bundle.vlm._local_backend
        self.assertIs(backend, bundle.reviewer._local_backend)
        self.assertIsNone(backend._model)
        responses = [SceneAnalysis(projection='isometric', categories=['tree'], description='Tree').model_dump_json(),
                     SceneReviewDecision(continue_detection=False, reason='Done', status='sufficient').model_dump_json()]
        with patch.object(backend, 'generate', side_effect=responses) as generate:
            bundle.vlm.analyze_scene(str(self.source))
            result = bundle.reviewer.review(str(self.source), str(self.source), {'accepted_coverage': .2})
        self.assertEqual(result.status, 'sufficient')
        messages = generate.call_args.args[0]
        self.assertEqual(sum(b['type'] == 'image_url' for b in messages[1].content), 2)

    def test_invalid_local_review_becomes_manual_review_without_fallback(self) -> None:
        reviewer = ServiceBundle.create(False, self.config).reviewer
        state = {'source_path': str(self.source), 'working_path': str(self.source), 'output_dir': str(self.root / 'out'),
                 'scene_qa': {'accepted_coverage': .1, 'target_coverage': .9}, 'detection_round': 1,
                 'objects': [], 'failed_objects': [], 'retry_count': 0}
        with patch.object(reviewer._local_backend, 'generate', return_value='malformed'):
            result = make_qa_scene(PipelineConfig(), reviewer)(state)
        self.assertEqual(result['scene_qa']['status'], 'manual_review')
        self.assertEqual(result['scene_qa']['backend'], 'llm')
        self.assertEqual(result['scene_qa']['reviewer_error'], 'ValidationError')
        self.assertFalse(result['scene_continue'])

    def test_auxiliary_visual_reviews_use_the_shared_local_model(self) -> None:
        reviewer = ServiceBundle.create(False, self.config).reviewer
        from schemas.candidate import CandidateQA
        from schemas.reconstruction import OcclusionAnalysis
        from schemas.scene_qa import CategoryReview
        responses = [
            CategoryReview(category='tree', confidence=.9, reason='Visible trunk', uncertain=False),
            CandidateQA(status='ACCEPT', overall=.9, evaluator='vision_llm'),
            OcclusionAnalysis(object_id='tree_001', object_type='tree', occlusion_ratio=0,
                              reconstruction_confidence=.9, needs_completion=False),
        ]
        with patch.object(reviewer._local_backend, 'generate', side_effect=[r.model_dump_json() for r in responses]) as generate:
            self.assertEqual(reviewer.classify_crop(str(self.source), 'rock'), responses[0])
            self.assertEqual(reviewer.review_candidate(str(self.source), str(self.source), {}), responses[1])
            self.assertEqual(reviewer.analyze_occlusion(str(self.source), str(self.source),
                             {'id': 'tree_001', 'category': 'tree'}), responses[2])
        self.assertEqual([call.args[1] for call in generate.call_args_list],
                         [CategoryReview, CandidateQA, OcclusionAnalysis])

    def test_mock_remains_offline_and_uses_rules(self) -> None:
        with patch.object(QwenVLBackend, 'load') as load:
            bundle = ServiceBundle.create(True)
            self.assertEqual(bundle.vlm.analyze_scene('unused').projection, 'isometric')
            self.assertEqual(bundle.reviewer.backend, 'rules')
            self.assertIsNone(bundle.vlm._local_backend)
            load.assert_not_called()

    def test_shared_lifecycle_and_reviewer_cache_invalidation(self) -> None:
        real = ServiceBundle.create(False, self.config)
        bundle = replace(ServiceBundle.create(), vlm=real.vlm, reviewer=real.reviewer)
        backend = real.vlm._local_backend
        loads = []

        def fake_load() -> None:
            loads.append(True)
            backend._model, backend._processor = MagicMock(), MagicMock()

        backend.load = fake_load
        runtime = ExecutionRuntime(PipelineConfig(), bundle)
        self.assertEqual(set(runtime.manager.entries), {'qwen_vl'})
        self.assertIs(runtime.manager.entries['qwen_vl'].service, backend)
        response = SceneReviewDecision(continue_detection=False, reason='Done', status='sufficient').model_dump_json()
        metrics = {'accepted_coverage': .2}
        node = lambda state: {'decision': bundle.reviewer.review(str(self.source), str(self.source), metrics)}
        memory = {'rss': 1024, 'system_ram_available': 128 * 1024 ** 3}
        with patch.object(backend, 'generate', return_value=response) as generate, patch(
            'services.model_manager.snapshot', return_value=memory,
        ), patch.object(runtime.manager, 'release_cache'):
            runtime.run_node('qa_scene', node, {'output_dir': str(self.root / 'out')})
            self.assertIsNone(backend._model)
            self.assertIsNone(backend._processor)
            runtime.run_node('qa_scene', node, {'output_dir': str(self.root / 'out')})
            self.assertEqual(len(loads), 1)  # Cached QA does not load weights.
            self.assertEqual(generate.call_count, 1)
            # New task rescans checkpoint metadata, including QA's shared weights.
            (self.weights / 'model.safetensors').write_bytes(b'updated-placeholder-weights')
            runtime.run_node('load_image', node, {'output_dir': str(self.root / 'out')})
            self.assertEqual(len(loads), 2)
            backend.config = backend.config.model_copy(update={'qwen_vl': backend.config.qwen_vl.model_copy(update={'max_new_tokens': 512})})
            runtime.run_node('qa_scene', node, {'output_dir': str(self.root / 'out')})
            self.assertEqual(len(loads), 3)
        self.assertTrue(all(e.get('model') == 'qwen_vl' for e in runtime.profiler.events if e['kind'] == 'inference'))
