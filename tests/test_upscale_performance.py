"""Progress, model scheduling and sparse-tile correctness regressions."""
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib.util
import threading
import unittest

import numpy as np
from PIL import Image

from app.config import PipelineConfig
from app.models import ModelConfig
from nodes.upscale_objects import make_upscale_objects
from services.execution import ExecutionRuntime
from services.progress import StageProgress
from services.runtime import ServiceBundle
from services.upscale_service import UpscaleService
from tests.test_p2_candidates import scores
from tests.test_p2_resources import Loader, memory


class ProgressTests(unittest.TestCase):
    def test_percent_duration_average_eta_and_midnight(self):
        progress = StageProgress('Upscale', 300)
        progress.completed = 87
        output = progress.format(elapsed=1902, now=datetime(2026, 10, 1, 23, 50))
        self.assertIn('[Upscale] 87/300 (29.0%)', output)
        self.assertIn('elapsed=00:31:42', output)
        self.assertIn('avg=21.9s/object', output)
        self.assertIn('eta=01:17:37', output)
        self.assertIn('estimated_finish=2026-10-02 01:07', output)

    def test_unknown_initial_eta_empty_work_and_completion(self):
        progress = StageProgress('HD QA', 2)
        self.assertIn('eta=--:--:--', progress.format(elapsed=20))
        progress.completed = 2
        self.assertIn('eta=00:00:00', progress.format(elapsed=40))
        self.assertIn('(100.0%)', StageProgress('Upscale', 0).format(elapsed=0))

    def test_long_item_heartbeat_and_cleanup_on_error(self):
        beat = threading.Event()
        messages = []
        def report(message):
            messages.append(message)
            if 'current=tree' in message:
                beat.set()
        progress = StageProgress('Upscale', 1, report, interval=.01)
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with progress:
                progress.start_item('tree')
                self.assertTrue(beat.wait(2))
                raise RuntimeError('stop')
        self.assertFalse(progress._thread.is_alive())
        self.assertIn('interrupted=1', messages[-1])
        self.assertIn('0/1', messages[-1])


class SchedulingTests(unittest.TestCase):
    def test_300_assets_load_each_gpu_model_once_and_keep_required_status(self):
        events = []
        class Upscaler(Loader):
            backend = 'real_esrgan'
            def upscale(self, image_path, scale):
                events.append('upscale')
                path = Path(image_path).with_name('hd.png')
                with Image.open(image_path) as source:
                    source.resize((source.width*scale, source.height*scale)).save(path)
                return str(path)
        class Reviewer(Loader):
            def review_candidate(self, *args):
                events.append('review')
                return {'status': 'ACCEPT', 'scores': scores(), 'evaluator': 'test'}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'source.png'
            Image.new('RGBA', (8, 8), 'red').save(path)
            objects = [{'id': str(i), 'asset_path': str(path), 'mask_path': str(path),
                        'category': 'tree', 'bbox': {'x': 0, 'y': 0, 'w': 8, 'h': 8},
                        'status': 'pass', 'qa': {'status': 'pass'}} for i in range(300)]
            upscale, reviewer = Upscaler(), Reviewer()
            bundle = replace(ServiceBundle.create(), upscale=upscale, reviewer=reviewer)
            config = PipelineConfig(cache={'enabled': False})
            runtime = ExecutionRuntime(config, bundle)
            messages = []
            node = make_upscale_objects(upscale, True, reviewer, required=True, progress=messages.append)
            with patch.object(runtime.manager, 'release_cache'), patch('services.model_manager.snapshot', side_effect=lambda: memory()):
                result = runtime.run_node('upscale_objects', node, {'output_dir': folder, 'objects': objects})
            self.assertEqual(events, ['upscale']*300 + ['review']*300)
            self.assertEqual((upscale.loads, reviewer.loads), (1, 1))
            self.assertEqual(result['failed_objects'], [])
            self.assertTrue(all(o['enhancement_status'] == 'ready' and not o['review']['required'] for o in result['objects']))
            self.assertTrue(all(o['base_asset_status'] == 'ready' for o in result['objects']))
            self.assertTrue(any('[HD QA] 300/300 (100.0%)' in m for m in messages))
            self.assertEqual(objects[0].get('enhancement'), None)
            ids = {e.get('instance_id') for e in runtime.profiler.events if e['kind'] == 'inference'}
            self.assertEqual(ids, {str(i) for i in range(300)})

    def test_skips_and_failed_review_are_finalized_without_accepting_candidate(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'source.png'
            Image.new('RGBA', (8, 8), 'red').save(path)
            candidate = Path(folder)/'hd.png'
            Image.new('RGBA', (32, 32), 'red').save(candidate)
            base = {'id': 'a', 'asset_path': str(path), 'mask_path': str(path), 'status': 'pass',
                    'qa': {'status': 'pass'}, 'category': 'tree', 'bbox': {'x': 0, 'y': 0, 'w': 8, 'h': 8}}
            service = UpscaleService(False)
            messages = []
            with patch.object(service, 'upscale', return_value=str(candidate)) as restore:
                result = make_upscale_objects(service, True, required=True, progress=messages.append)({'objects': [
                    base, {**base, 'id': 'skip', 'completion_required': True, 'needs_manual_review': True}]})
            restore.assert_called_once()
            obj = result['objects'][0]
            self.assertEqual(obj['enhancement_status'], 'failed')
            self.assertEqual(obj['base_asset_status'], 'ready')
            self.assertEqual(obj['review'], {'required': True, 'reason': 'required_enhancement'})
            self.assertEqual(obj['status'], 'manual_review')
            self.assertIsNone(obj['hd_asset_path'])
            self.assertEqual(obj['texture_scale'], 1)
            self.assertTrue(any('[Upscale] 1/1' in message for message in messages))
            self.assertEqual(result['objects'][1]['enhancement_status'], 'skipped')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional Torch is not installed')
class SparseTileTests(unittest.TestCase):
    def test_skip_transparent_tiles_preserves_visible_rgba_at_both_scales(self):
        import torch
        class LocalModel(torch.nn.Module):
            def forward(self, tensor):
                tensor = torch.nn.functional.avg_pool2d(tensor, 3, stride=1, padding=1)
                return tensor.repeat_interleave(4, 2).repeat_interleave(4, 3)
        with TemporaryDirectory() as folder:
            pixels = np.zeros((97, 131, 4), np.uint8)
            pixels[27:38, 29:39] = [200, 60, 20, 255]
            pixels[40, 49] = [15, 180, 240, 1]  # Preserve even faint/disconnected alpha.
            pixels[0:3, 130] = [30, 40, 190, 128]
            source = Path(folder)/'asset.png'
            Image.fromarray(pixels).save(source)
            for scale in (2, 4):
                outputs, stats = [], []
                for skip in (False, True):
                    service = UpscaleService(False, config=ModelConfig(device='cpu', upscale={
                        'tile': 16, 'tile_pad': 4, 'skip_transparent_tiles': skip}))
                    service._torch, service._device, service._model = torch, 'cpu', LocalModel()
                    result = service.upscale(source, scale)
                    with Image.open(result) as image:
                        outputs.append(np.array(image))
                    stats.append(service.last_restore_stats)
                np.testing.assert_array_equal(outputs[0], outputs[1])
                self.assertGreater(stats[1]['tiles_skipped'], 0)
                self.assertLess(stats[1]['tiles_run'], stats[0]['tiles_run'] / 2)
