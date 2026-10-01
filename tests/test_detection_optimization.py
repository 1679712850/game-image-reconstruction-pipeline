"""Differential tests for optimizations: compare decisions, ordering and pixels."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import unittest

import cv2
import numpy as np
from PIL import Image

from app.config import PipelineConfig
from app.detection_config import DedupConfig, DetectionBudget
from app.models import ModelConfig
from detection.budget import BudgetTracker
from detection.grouped_detector import scan_window
from fusion.candidate_fusion import fuse_candidates, merge_candidates
from fusion.cross_tile_dedup import same_object
from retry.detection_retry import recover_regions
from segmentation.local_refiner import candidate_metrics, nearby_edges, LocalRefiner
from services.execution import _active, ExecutionRuntime
from services.runtime import ServiceBundle
from services.grounding_service import GroundingService
from tests.fixtures.detection_retry_reference import recover_regions as reference_recover
from tests.test_p0_detection import candidate, settings, item
from tests.test_p1_pipeline import BoxSAM


def reference_fuse(candidates, config):
    kept, rejected = [], []
    for value in sorted(candidates, key=lambda c: (not c.is_truncated, c.confidence), reverse=True):
        match = next((i for i, old in enumerate(kept) if same_object(old, value, config)), None)
        if match is None:
            kept.append(value.model_copy(deep=True))
        else:
            old = kept[match]
            merged = merge_candidates(old, value)
            loser = value if merged.id != value.id else old
            duplicate = loser.model_copy(deep=True)
            duplicate.reject_reason = 'cross_tile_duplicate' if old.window != value.window else 'duplicate'
            duplicate.parent_id = merged.id
            rejected.append(duplicate)
            kept[match] = merged
    return kept, rejected


class GeometryEquivalenceTests(unittest.TestCase):
    def test_spatial_index_matches_greedy_order_with_growth_and_giant_boxes(self):
        rng = np.random.default_rng(19)
        values = []
        for index in range(900):
            x, y = rng.integers(0, 12000, 2)
            w, h = rng.integers(1, 1500, 2)
            value = candidate(str(index), tuple(int(v) for v in (x, y, x+w, y+h)),
                category=['tree', 'rock', 'building'][index % 3], score=float(rng.choice([.5, .7, .9])),
                edges=['left'] if index % 5 == 0 else (), window=(0, 0, 14000, 14000))
            values.append(value)
            if index % 7 == 0:
                values.append(value.model_copy(update={'id': f'{index}_copy', 'confidence': .55}, deep=True))
        # A union crosses cell boundaries; a giant query/index uses bounded fallback.
        values.extend([candidate('left', (240, 20, 320, 90), score=.9, edges=['left']),
                       candidate('right', (280, 20, 360, 90), score=.8, edges=['right']),
                       candidate('last', (300, 20, 380, 90), score=.7, edges=['right']),
                       candidate('giant', (0, 0, 1000000, 1000000), category='rock')])
        before = [c.model_dump() for c in values]
        actual = fuse_candidates(values, DedupConfig())
        expected = reference_fuse(values, DedupConfig())
        self.assertEqual([[c.model_dump() for c in part] for part in actual],
                         [[c.model_dump() for c in part] for part in expected])
        self.assertEqual(before, [c.model_dump() for c in values])

    def test_no_recovery_reuses_fusion_but_unresolved_new_observations_do_not(self):
        from detection.p0_pipeline import P0DetectionPipeline
        with patch('detection.p0_pipeline.fuse_candidates', wraps=fuse_candidates) as fusion:
            P0DetectionPipeline(settings()).run(Image.new('RGB', (100, 60)), ['tree'], lambda *a: [])
            self.assertEqual(fusion.call_count, 1)
        def infer(crop, group, ctx):
            if ctx['source'] == 'tile' and ctx['window'][0] == 0:
                return [item('tree', (55, 20, 64, 35))]
            if ctx['source'] == 'redetection':
                x, y, right, bottom = ctx['window']
                return [item('tree', (55-x, 20-y, right-x, 35-y))]
            return []
        with patch('detection.p0_pipeline.fuse_candidates', wraps=fuse_candidates) as fusion:
            P0DetectionPipeline(settings()).run(Image.new('RGB', (100, 60)), ['tree'], infer)
            self.assertEqual(fusion.call_count, 2)

    def test_shared_edges_leave_every_score_unchanged_and_canny_runs_once(self):
        rng = np.random.default_rng(17)
        image = Image.fromarray(rng.integers(0, 256, (87, 119, 3), dtype=np.uint8))
        masks = [rng.random((87, 119)) > .6, np.zeros((87, 119)), np.ones((87, 119))]
        expected = [candidate_metrics(mask, [5, 8, 70, 50], image, .9, [[-1, 4], [130, 60]]) for mask in masks]
        with patch('segmentation.local_refiner.cv2.Canny', wraps=cv2.Canny) as canny:
            shared = nearby_edges(image)
            actual = [candidate_metrics(mask, [5, 8, 70, 50], image, .9, [[-1, 4], [130, 60]], nearby=shared) for mask in masks]
            self.assertEqual(canny.call_count, 1)
        self.assertEqual(expected, actual)


class OrderedRecoveryTests(unittest.TestCase):
    def test_mock_bounds_prevent_prefetch_beyond_object_cap(self):
        from retry.detection_schedule import RegionDetectionSchedule
        config = PipelineConfig(scene_loop={'max_objects': 1})
        detector = GroundingService(mock=True)
        self.assertEqual(detector.detect_bounds((100, 100), ['tree', 'tree', 'unknown'], 1), (0, 1))
        regions = [{'category': 'tree', 'approx_bbox': {'x': x, 'y': 20, 'w': 10, 'h': 10}}
                   for x in (20, 60)]
        schedule = RegionDetectionSchedule({}, regions, detector, config,
                                           Image.new('RGB', (100, 100)), ['tree'], 1)
        self.assertEqual(schedule._admitted(0, 0), [])

    def run_case(self, root, reference=False, limit=None, budget_limit=100, fail_region=False):
        config = PipelineConfig(mock=False, crop={'padding': 0}, scene_loop={'enabled': False, 'max_objects': limit},
                                resources={'max_oom_retry': 0}, p1={'max_problem_regions': 6})
        budget = BudgetTracker(DetectionBudget(max_total_inference_calls=budget_limit, max_retry_calls=budget_limit))
        events = []
        class Detector:
            def __init__(self):
                self.config = ModelConfig()
            def detect_bounds(self, size, categories, category_limit):
                return 1, 2
            def detect(self, path, categories):
                index, attempt = map(int, Path(path).stem.split('_')[-2:])
                events.append(f'd{index}:{attempt}')
                if not budget.consume('redetection'):
                    raise RuntimeError('Detection retry inference budget exhausted')
                if index == 0 and attempt == 0:
                    return []
                if index == 1 and attempt == 0 and fail_region:
                    raise ValueError('bad detector output')
                pad = 32*(attempt+1)*2
                value = {'category': 'tree', 'confidence': .95, 'bbox': {'x': pad, 'y': pad, 'w': 40, 'h': 40}}
                # A duplicate within a region exercises commit-time dedup.
                return [value, deepcopy(value)]
        class SAM(BoxSAM):
            def predict_candidates(self, image, prompts):
                events.append('sam')
                return super().predict_candidates(image, prompts)
        source = root/'scene.png'
        Image.new('RGB', (500, 240), (40, 130, 60)).save(source)
        state = {'source_path': str(source), 'output_dir': str(root), 'objects': [], 'all_detections': [],
                 'layer_plan': [{'categories': ['tree']}], 'retry_history': []}
        regions = [{'category': 'tree', 'approx_bbox': {'x': x, 'y': 80, 'w': 20, 'h': 20}}
                   for x in (80, 210, 340)]
        regions.insert(2, deepcopy(regions[1]))
        runtime = SimpleNamespace(detection_budget=budget, config=config, instance_id=None, retry_count=0)
        token = _active.set(runtime)
        try:
            objects, history = (reference_recover if reference else recover_regions)(state, regions, Detector(), SAM(), config)
        finally:
            _active.reset(token)
        def portable(value):
            if isinstance(value, str):
                return value.replace(str(root), '<root>')
            if isinstance(value, list):
                return [portable(v) for v in value]
            if isinstance(value, dict):
                return {k: portable(v) for k, v in value.items()}
            return value
        pixels = []
        for obj in objects:
            with Image.open(obj['asset_path']) as image:
                pixels.append(np.array(image))
        return portable(({'objects': objects, 'history': history})), budget.report(), events, pixels

    def test_phases_preserve_retries_ids_neighbours_histories_and_pixels(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for fail in (False, True):
                a, b = root/f'a{fail}', root/f'b{fail}'
                a.mkdir(); b.mkdir()
                old = self.run_case(a, reference=True, fail_region=fail)
                new = self.run_case(b, fail_region=fail)
                self.assertEqual(old[0], new[0])
                self.assertEqual(old[1], new[1])
                self.assertNotEqual(old[2], new[2])
                for expected, actual in zip(old[3], new[3]):
                    np.testing.assert_array_equal(expected, actual)
                self.assertEqual(new[2][:3], ['d0:0', 'd1:0', 'd3:0'] if not fail else ['d0:0', 'd1:0', 'd0:1'])
                if not fail:
                    from itertools import groupby
                    phases = lambda events: len(list(groupby('sam' if e == 'sam' else 'dino' for e in events)))
                    self.assertEqual((phases(old[2]), phases(new[2])), (6, 2))

    def test_tight_object_or_call_budgets_preserve_serial_order(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index, kwargs in enumerate([{'limit': 1}, {'budget_limit': 2}]):
                a, b = root/f'a{index}', root/f'b{index}'
                a.mkdir(); b.mkdir()
                old, new = self.run_case(a, reference=True, **kwargs), self.run_case(b, **kwargs)
                self.assertEqual(old[:3], new[:3])

    def test_last_prefetched_region_can_reach_object_cap_without_changing_results(self):
        with TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            a.mkdir(); b.mkdir()
            old, new = self.run_case(a, reference=True, limit=3), self.run_case(b, limit=3)
            self.assertEqual(old[:2], new[:2])
            self.assertNotEqual(old[2], new[2])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional Torch is not installed')
class InputAndBatchTests(unittest.TestCase):
    def service(self):
        import torch
        class Inputs(dict):
            def to(self, device):
                return Inputs({k: v.to(device) for k, v in self.items()})
        class Processor:
            def __init__(self):
                self.images = 0
            def __call__(self, *, text, images=None, return_tensors=None, **kwargs):
                result = Inputs(input_ids=torch.tensor([[ord(c) for c in text]]))
                if images is not None:
                    self.images += 1
                    result['pixel_values'] = torch.tensor(np.array(images).transpose(2, 0, 1)).unsqueeze(0).float()/255
                    result['pixel_mask'] = torch.ones((1, images.height, images.width), dtype=torch.long)
                return result
            def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
                return [{'boxes': torch.tensor([[1., 1., 7., 8.]]),
                         'scores': outputs[i].reshape(1),
                         'text_labels': [''.join(chr(c) for c in ids.tolist()).strip('.')]} for i, ids in enumerate(input_ids)]
        class Model:
            def __init__(self):
                self.calls = []
                self.oom = False
            def to(self, *args, **kwargs):
                return self
            def __call__(self, input_ids, pixel_values, pixel_mask):
                self.calls.append(len(input_ids))
                if self.oom and len(input_ids) > 1:
                    raise RuntimeError('CUDA out of memory')
                return pixel_values.mean((1, 2, 3))*.1 + .8
        service = GroundingService(False, ModelConfig(device='cpu'))
        service._torch, service._device = torch, 'cpu'
        service._processor, service._model = Processor(), Model()
        return service

    def test_inputs_reuse_exact_pixels_text_and_invalidate_at_window_or_scale_change(self):
        import torch
        service = self.service()
        image = Image.new('RGB', (16, 12), (40, 130, 60))
        with service.prepared_window(image):
            first = service._prepare_inputs(image, 'tree.')
            second = service._prepare_inputs(image, 'rock.')
            self.assertIs(first['pixel_values'], second['pixel_values'])
            expected = service._processor(images=image, text='rock.', return_tensors='pt')
            for key in expected:
                self.assertTrue(torch.equal(expected[key], second[key]))
            self.assertEqual(service._processor.images, 2)  # one cached + explicit oracle
            service._inference_scale = .5
            service._prepare_inputs(image, 'tree.')
            self.assertEqual(service._processor.images, 3)
        self.assertIsNone(service._prepared_image)
        with self.assertRaises(ValueError):
            with service.prepared_window(image):
                service._prepare_inputs(image, 'tree.')
                raise ValueError('stop')
        self.assertIsNone(service._prepared_image)
        self.assertIsNone(service._window_image)

    def test_same_shape_batches_and_oom_split_preserve_results_and_order(self):
        image = Image.new('RGB', (16, 12), (40, 130, 60))
        service = self.service()
        groups = [['tree'], ['building'], ['rock'], ['flag']]
        expected = [service._infer_image(image, group, .1, .1, preserve_candidates=True) for group in groups]
        for oom in (False, True):
            service._model.calls.clear()
            service._model.oom = oom
            with service.prepared_window(image):
                actual = service._infer_batch(image, groups, .1, .1)
            self.assertEqual(expected, actual)
            self.assertIn(3, service._model.calls)
            if oom:
                self.assertEqual(service._model.calls.count(1), 4)
            self.assertEqual(getattr(service, '_inference_scale', 1), 1)
            self.assertEqual(service.config.device, 'cpu')

    def test_batch_scanner_preserves_logical_budget_failure_isolation_and_crop_identity(self):
        seen, crops, collected, scans = [], [], [], []
        def infer(crop, group, context):
            crops.append(crop)
            if group == ['building']:
                raise ValueError('bad input')
            return [{'category': group[0]}]
        def batch(crop, groups):
            seen.extend(groups)
            raise RuntimeError('batch failure')
        infer.batch_size = 3
        infer.infer_many = batch
        budget = BudgetTracker(DetectionBudget(max_total_inference_calls=2))
        scan_window(Image.new('RGB', (20, 20)), (0, 0, 20, 20), ['tree', 'rock', 'building'], infer,
                    lambda value, context: collected.append(value), scans, source='tile', group_size=1, budget=budget)
        self.assertEqual(budget.calls, 2)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(crop is crops[0] for crop in crops))
        self.assertEqual(sum(scan['status'] == 'budget_exhausted' for scan in scans), 1)
        self.assertEqual(sum(scan['status'] == 'failed' for scan in scans), 1)
        self.assertEqual(len(collected), 1)

    def test_batch_runtime_cache_bypasses_reload_and_stage_cleanup_drops_inputs(self):
        from tests.test_p2_resources import memory
        service = self.service()
        model = service._model
        image = Image.new('RGB', (16, 12), (40, 130, 60))
        bundle = replace(ServiceBundle.create(), grounding=service)
        runtime = ExecutionRuntime(PipelineConfig(), bundle)
        def node(state):
            with service.prepared_window(image):
                return {'detections': service._infer_batch(image, [['tree'], ['rock']], .1, .1)}
        with TemporaryDirectory() as folder:
            with patch.object(runtime.manager, 'release_cache'), patch('services.model_manager.snapshot', side_effect=lambda: memory()):
                first = runtime.run_node('detect_instances', node, {'output_dir': folder})
                calls = len(model.calls)
                second = runtime.run_node('detect_instances', node, {'output_dir': folder})
        self.assertEqual(first['detections'], second['detections'])
        self.assertEqual(len(model.calls), calls)
        self.assertIsNone(service._prepared_image)
        self.assertIsNone(service._model)
        self.assertEqual(runtime.cache.stats['detection']['hit'], 1)
