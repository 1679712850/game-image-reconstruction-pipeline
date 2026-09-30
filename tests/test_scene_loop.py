"""Scene loop budgets, actual pixel removal, reviewer contracts and tiled coordinates."""
from contextlib import nullcontext
from dataclasses import replace
import json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from app.config import PipelineConfig, SceneLoopConfig
from app.models import GroundingConfig, ModelConfig, SAMConfig, SceneReviewerConfig
from cv.coverage import write_remaining
from cv.mask import read_mask, save_mask
from cv.metrics import mask_outside_bbox
from cv.tiles import tile_windows
from schemas.scene_qa import SceneReviewDecision
from services.detection_postprocess import canonical_prompt_label
from services.grounding_service import GroundingService
from services.runtime import ServiceBundle
from services.sam_service import SAMService
from services.scene_review_service import SceneReviewService
from services.model_support import ModelUnavailableError
from nodes.scene_loop import make_update_remaining


class SequencedDetector(GroundingService):
    """One known object per pass, recording the detector's actual input pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs = []

    def detect_round(self, image_path: str, categories: list[str], round_index: int) -> tuple[list[dict], dict]:
        with Image.open(image_path) as image:
            self.inputs.append((np.array(image.convert("RGB")), list(categories)))
        return [{"category": "tree", "confidence": .9,
                 "bbox": {"x": round_index * 20, "y": 15, "w": 10, "h": 12}}], {}


class SceneLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "scene.png"
        Image.new("RGB", (120, 90), (15, 35, 65)).save(self.source)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def initial(self) -> dict:
        return {"source_path": str(self.source), "output_dir": str(self.root / "out")}

    def test_llm_continues_then_stops_on_remaining_image(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = [
            SceneReviewDecision(continue_detection=True, reason="Stone pillars remain", suggested_categories=["stone pillar"], status="needs_detection"),
            SceneReviewDecision(continue_detection=False, reason="No further separable objects", status="sufficient"),
        ]
        reviewer = SceneReviewService("llm", llm=llm)
        detector = SequencedDetector()
        bundle = replace(ServiceBundle.create(), grounding=detector, reviewer=reviewer)
        state = build_graph(services=bundle).invoke(self.initial())
        self.assertEqual(state["detection_round"], 2)
        self.assertEqual([o["id"] for o in state["objects"]], ["tree_001", "tree_002"])
        np.testing.assert_array_equal(detector.inputs[1][0][15:27, 20:30], 255)
        np.testing.assert_array_equal(detector.inputs[1][0][0, 0], [15, 35, 65])
        self.assertIn("stone pillar", detector.inputs[1][1])
        self.assertIn("stone pillar", [c for layer in state["layer_plan"] for c in layer["categories"]])
        with Image.open(self.source) as source:
            self.assertEqual(source.getpixel((20, 15)), (15, 35, 65))
        for obj in state["objects"]:
            with Image.open(obj["asset_path"]) as asset:
                pixels = np.array(asset)
                np.testing.assert_array_equal(pixels[pixels[:, :, 3] > 0][:, :3], np.tile([15,35,65], (120, 1)))
        call = llm.with_structured_output.return_value.invoke.call_args.args[0]
        self.assertEqual(len(call[1].content), 3)  # Metrics + original + remaining.
        self.assertEqual(state["scene_qa"]["status"], "sufficient")
        manifest = json.loads(Path(state["scene_json"]).read_text())
        self.assertTrue((self.root / "out" / manifest["coverage"]["remaining_image"]).exists())
        json.dumps(state)

    def test_llm_continue_cannot_exceed_round_budget(self) -> None:
        reviewer = SceneReviewService("llm", llm=MagicMock())
        reviewer._structured.invoke.return_value = SceneReviewDecision(continue_detection=True, reason="More objects", status="needs_detection")
        bundle = replace(ServiceBundle.create(), grounding=SequencedDetector(), reviewer=reviewer)
        options = PipelineConfig(scene_loop=SceneLoopConfig(max_rounds=2))
        state = build_graph(options, bundle).invoke(self.initial())
        self.assertEqual(state["detection_round"], 2)
        self.assertFalse(state["scene_continue"])
        self.assertEqual(state["scene_stop_reason"], "max_rounds")
        self.assertEqual(state["scene_qa"]["status"], "manual_review")

    def test_llm_exception_exports_manual_review_without_rules_fallback(self) -> None:
        reviewer = SceneReviewService("llm", llm=MagicMock())
        reviewer._structured.invoke.side_effect = RuntimeError("private provider details")
        bundle = replace(ServiceBundle.create(), reviewer=reviewer)
        state = build_graph(services=bundle).invoke(self.initial())
        self.assertEqual(state["detection_round"], 1)
        self.assertEqual(state["scene_qa"]["backend"], "llm")
        self.assertEqual(state["scene_qa"]["reviewer_error"], "RuntimeError")
        self.assertNotIn("private provider details", json.dumps(state))
        self.assertTrue(Path(state["scene_json"]).is_file())

    def test_object_budget_and_checkpoint_resume_keep_unique_ids(self) -> None:
        detector = SequencedDetector()
        bundle = replace(ServiceBundle.create(), grounding=detector)
        options = PipelineConfig(scene_loop=SceneLoopConfig(max_objects=2))
        saver = InMemorySaver()
        graph = build_graph(options, bundle, checkpointer=saver, interrupt_before=["qa_scene"])
        run = {"configurable": {"thread_id": "round-resume"}}
        graph.invoke(self.initial(), run)
        graph.invoke(None, run)
        state = graph.invoke(None, run)
        self.assertEqual(graph.get_state(run).next, ())
        self.assertEqual(len(state["objects"]), 2)
        self.assertEqual(state["scene_stop_reason"], "max_objects")
        self.assertEqual(len({o["id"] for o in state["objects"]}), 2)

    def test_mask_union_only_removes_passed_pixels_and_ignores_transparent_source(self) -> None:
        source = np.zeros((10, 10, 4), dtype=np.uint8)
        source[:5] = [20, 30, 40, 255]
        Image.fromarray(source).save(self.source)
        a = np.zeros((10,10), dtype=bool)
        a[:2, :5] = True
        b = np.zeros_like(a)
        b[1:4, :5] = True
        objects = [{"id":"a", "status":"pass", "asset_path":"unused", "mask_path":save_mask(a, self.root/'a.png')},
                   {"id":"b", "status":"manual_review", "asset_path":"unused", "mask_path":save_mask(b, self.root/'b.png')}]
        result = write_remaining(str(self.source), objects, self.root/'residual', 8)
        self.assertEqual(result["accepted_coverage"], .2)
        self.assertEqual(result["candidate_coverage"], .4)
        self.assertEqual(result["eligible_pixels"], 50)
        self.assertEqual(int(np.count_nonzero(read_mask(result["coverage_mask_path"]))), 10)
        with Image.open(result["working_path"]) as remaining:
            self.assertEqual(remaining.getpixel((1, 3)), (20,30,40))
            self.assertEqual(remaining.getpixel((1, 1)), (255,255,255))

    def test_disabled_scene_loop_preserves_single_pass(self) -> None:
        options = PipelineConfig(scene_loop=SceneLoopConfig(enabled=False))
        state = build_graph(options).invoke(self.initial())
        self.assertEqual(state["detection_round"], 1)
        self.assertEqual(state["scene_history"], [])
        self.assertTrue(Path(state["scene_json"]).is_file())

    def test_failed_later_segmentation_keeps_earlier_candidate_asset(self) -> None:
        mask = np.zeros((90,120), dtype=bool)
        mask[10:20,10:20] = True
        record = {'id':'tree_001','category':'tree','confidence':.3,'status':'manual_review',
                  'asset_path':str(self.source),'mask_path':save_mask(mask, self.root/'mask.png')}
        state = {**self.initial(), 'detection_round':2, 'archived_objects':[record],
                 'objects':[{**record,'confidence':.9,'error':'empty mask','asset_path':None}]}
        update = make_update_remaining(PipelineConfig())(state)
        self.assertEqual(update['objects'],[record])
        self.assertGreater(update['scene_qa']['candidate_coverage'],0)
        self.assertEqual(update['scene_coverage'],0)

    def test_sparse_mask_inside_box_is_not_mask_leakage(self) -> None:
        mask = np.zeros((30,30), dtype=np.uint8)
        mask[10,10] = 255
        box = {'x':5,'y':5,'w':20,'h':20}
        self.assertEqual(mask_outside_bbox(mask,box),0)
        mask[1,1] = 255
        self.assertEqual(mask_outside_bbox(mask,box),.5)

    def test_unconfigured_llm_fails_before_detector_loading(self) -> None:
        config = PipelineConfig(mock=False, scene_loop=SceneLoopConfig(reviewer="llm"))
        services = ServiceBundle.create(False, reviewer_backend="llm")
        with patch.dict('os.environ', {}, clear=True), patch.object(services.grounding, 'load') as load:
            with self.assertRaisesRegex(ModelUnavailableError, 'qwen_vl.model_path'):
                build_graph(config, services)
            load.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('langchain_openai'), 'Optional requirements-llm.txt not installed')
    def test_real_langchain_structured_request_without_network(self) -> None:
        import httpx
        from langchain_openai import ChatOpenAI
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                'id': 'test-completion', 'object': 'chat.completion', 'created': 0, 'model': 'vision-test',
                'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': 'test-call', 'type': 'function', 'function': {
                            'name': 'SceneReviewDecision', 'arguments': json.dumps({
                                'continue_detection': True, 'reason': 'Visible stone pillars remain',
                                'suggested_categories': ['stone pillar'], 'status': 'needs_detection',
                            }),
                        },
                    }],
                }}],
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch.dict('os.environ', {'TEST_SCENE_KEY': 'test-only-no-network'}), patch(
                'langchain_openai.ChatOpenAI', side_effect=lambda **kw: ChatOpenAI(http_client=client, **kw),
            ):
                reviewer = SceneReviewService('llm', SceneReviewerConfig(
                    provider='api', model='vision-test', base_url='https://test.invalid/v1', api_key_env='TEST_SCENE_KEY',
                ))
                reviewer.validate_ready()
                result = reviewer.review(str(self.source),str(self.source), {'accepted_coverage': .1})
        self.assertTrue(result.continue_detection)
        self.assertEqual(result.suggested_categories,['stone pillar'])
        content = requests[0]['messages'][1]['content']
        self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
        self.assertEqual(requests[0]['tools'][0]['function']['name'], 'SceneReviewDecision')


class TiledDetectionTests(unittest.TestCase):
    def test_tiles_cover_borders_without_gaps(self) -> None:
        covered = np.zeros((103, 155), dtype=bool)
        windows = tile_windows(155,103,64,.25)
        for x0,y0,x1,y1 in windows:
            covered[y0:y1,x0:x1] = True
            self.assertLessEqual(x1-x0,64)
            self.assertLessEqual(y1-y0,64)
        self.assertTrue(covered.all())
        self.assertEqual(tile_windows(12,9,64,.25),[(0,0,12,9)])

    def test_global_offsets_nms_and_only_late_threshold_relaxation(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)/'scene.png'
            Image.new('RGB',(100,60)).save(path)
            cfg = GroundingConfig(tile_size=64, include_full_image=False, prompt_group_size=1)
            service = GroundingService(False, ModelConfig(grounding=cfg))
            service._model = object()
            service._infer_image = MagicMock(side_effect=[
                [{'category':'tree','confidence':.9,'bbox':{'x':40,'y':10,'w':10,'h':12}}],
                [{'category':'tree','confidence':.8,'bbox':{'x':4,'y':10,'w':10,'h':12}}],
            ])
            found, stats = service.detect_round(str(path),['tree'],1)
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]['bbox'],{'x':40,'y':10,'w':10,'h':12})
            self.assertEqual(stats['before_merge'],2)
            self.assertEqual(stats['box_threshold'],.30)
            service._infer_image.side_effect = None
            service._infer_image.return_value = []
            _, stats = service.detect_round(str(path),['tree'],3)
            self.assertEqual(stats['box_threshold'],.25)
            self.assertEqual(stats['text_threshold'],.20)

    def test_specific_prompt_aliases_and_ambiguous_labels(self) -> None:
        cats, phrases = ['pillar','gate','rock_debris'], ['stone pillar','ornate gate','broken rock debris']
        self.assertEqual(canonical_prompt_label('pillar', cats, phrases), 'pillar')
        self.assertEqual(canonical_prompt_label('rock debris', cats, phrases), 'rock_debris')
        self.assertEqual(canonical_prompt_label('pillar gate', cats, phrases), '')

    def test_local_sam_restores_original_coordinates(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)/'scene.png'
            Image.new('RGB',(100,80)).save(path)
            service = SAMService(False, ModelConfig(sam=SAMConfig(local_padding=5)))
            service._torch = SimpleNamespace(inference_mode=nullcontext)
            service._predictor = MagicMock()
            service.load = MagicMock()
            masks = np.zeros((2,22,20))
            masks[1,5:17,5:15] = 1
            service._predictor.predict.return_value = (masks,np.array([.2,.9]),None)
            record = {'id':'tree_001','category':'tree','confidence':.3,'bbox':{'x':40,'y':20,'w':10,'h':12}}
            result = service.segment_local(str(path),[record])[0]
            self.assertEqual(result['mask'].shape,(80,100))
            self.assertEqual(int(np.count_nonzero(result['mask'])),120)
            self.assertTrue((result['mask'][20:32,40:50] == 255).all())
            np.testing.assert_array_equal(service._predictor.predict.call_args.kwargs['box'],[5,5,15,17])
            self.assertEqual(result['confidence'],.3)


if __name__ == '__main__':
    unittest.main()
