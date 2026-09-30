"""Resource lifecycle, lazy cache reuse, portable blobs and recoverable failures."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import numpy as np
from PIL import Image

from agent.graph import build_graph
from app.config import PipelineConfig
from app.models import ModelConfig
from app.resource_config import CacheConfig, ResourcesConfig
from services.cache_manager import CacheManager
from services.execution import ExecutionRuntime
from services.model_manager import ModelManager, ResourceBudgetError
from services.profiler import Profiler, GIB
from services.runtime import ServiceBundle


def memory(**overrides):
    return {'rss': GIB, 'vram_allocated': 0, 'vram_free': 24*GIB, 'vram_total': 24*GIB, **overrides}


class Handle:
    def __init__(self):
        self.devices=[]
    def to(self, device):
        self.devices.append(device)
        return self


class Loader:
    mock=False
    def __init__(self, device='cuda'):
        self.config=ModelConfig(device=device)
        self._model=None
        self.loads=0
    def load(self):
        self.loads+=1
        self._model=Handle()
        self._model.to(self.config.device)


class CacheTests(unittest.TestCase):
    def test_content_prompt_model_and_versions_invalidate(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); path=root/'source.png'
            Image.new('RGB',(2,2),'red').save(path)
            cache=CacheManager(CacheConfig(),root)
            params={'image':str(path),'prompt':'first','steps':20}
            key=cache.key('generation',{'model':'a','revision':'v1'},params)
            self.assertNotEqual(key,cache.key('generation',{'model':'a','revision':'v2'},params))
            self.assertNotEqual(key,cache.key('generation',{'model':'a','revision':'v1'},{**params,'prompt':'second'}))
            version=CacheManager(CacheConfig(pipeline_version='v2'),root)
            self.assertNotEqual(key,version.key('generation',{'model':'a','revision':'v1'},params))
            Image.new('RGB',(2,2),'blue').save(path)
            self.assertNotEqual(key,cache.key('generation',{'model':'a','revision':'v1'},params))

    def test_arrays_and_generated_files_restore_in_new_output_root(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); a=root/'a'; b=root/'b'; a.mkdir()
            settings=CacheConfig(directory=root/'shared')
            source=a/'candidate.png'; Image.new('RGBA',(4,5),'red').save(source)
            array=np.ones((5,4),np.uint8)*255
            first=CacheManager(settings,a); first.put('segmentation','key',({'mask':array},str(source)))
            source.unlink()
            second=CacheManager(settings,b); hit,result=second.get('segmentation','key')
            self.assertTrue(hit); np.testing.assert_array_equal(result[0]['mask'],array)
            self.assertEqual(result[1],str((b/'candidate.png').resolve())); self.assertTrue((b/'candidate.png').exists())
            manifest=root/'shared/segmentation/key/result.json'; data=json.loads(manifest.read_text())
            data['data'][1]['relative']='../../escaped.png'; manifest.write_text(json.dumps(data))
            self.assertFalse(second.get('segmentation','key')[0])
            self.assertFalse((root.parent/'escaped.png').exists())

    def test_corrupt_blob_is_a_miss_not_an_exception(self):
        with TemporaryDirectory() as folder:
            cache=CacheManager(CacheConfig(),folder)
            cache.put('segmentation','key',np.ones((2,2)))
            path=cache.root/'segmentation/key'; blob=next(p for p in path.iterdir() if p.name != 'result.json')
            blob.write_bytes(b'corrupt')
            self.assertFalse(cache.get('segmentation','key')[0])
            self.assertEqual(cache.stats['segmentation']['corrupt'],1)


class ModelTests(unittest.TestCase):
    def test_lazy_load_lru_and_offload_cpu_reuse_then_final_release(self):
        a,b=Loader(),Loader()
        manager=ModelManager(ResourcesConfig(keep_alive={'a':True},estimated_vram={'a':3,'b':4}),Profiler())
        manager.register('a',a); manager.register('b',b)
        self.assertEqual(a.loads,0)
        with patch('services.model_manager.snapshot',side_effect=lambda: memory()), patch.object(manager,'release_cache'):
            manager.load('a'); manager.load('a')
            self.assertEqual(a.loads,1)
            manager.load('b')
            self.assertEqual(manager.entries['a'].status,'OFFLOADED')
            self.assertEqual(a._model.devices,['cuda','cpu'])
            manager.load('a')
            self.assertEqual(a.loads,1)
            self.assertEqual(manager.entries['b'].status,'NOT_LOADED')
            manager.end_stage(final=True)
        self.assertIsNone(a._model); self.assertIsNone(b._model)

    def test_hard_vram_limit_uses_cpu_and_ram_budget_blocks_loading(self):
        service=Loader(); manager=ModelManager(ResourcesConfig(max_vram=4,soft_vram=3,estimated_vram={'a':8},estimated_ram={'a':1}),Profiler())
        manager.register('a',service)
        with patch('services.model_manager.snapshot',return_value=memory()), patch.object(manager,'release_cache'):
            manager.load('a')
            self.assertEqual(service.config.device,'cpu'); self.assertEqual(manager.entries['a'].status,'CPU')
        service=Loader('cpu'); manager=ModelManager(ResourcesConfig(max_ram=1,estimated_ram={'a':3}),Profiler());manager.register('a',service)
        with patch('services.model_manager.snapshot',return_value=memory()), patch.object(manager,'release_cache'):
            with self.assertRaises(ResourceBudgetError):
                manager.load('a')
        self.assertEqual(service.loads,0)

    def test_active_gpu_model_cannot_be_evicted_or_exceed_resident_limit(self):
        manager=ModelManager(ResourcesConfig(estimated_vram={'a':1,'b':1}),Profiler())
        a,b=Loader(),Loader();manager.register('a',a);manager.register('b',b)
        with patch('services.model_manager.snapshot',return_value=memory()), patch.object(manager,'release_cache'):
            manager.load('a'); manager.entries['a'].active=1;manager.load('b')
            self.assertEqual(manager.entries['a'].status,'GPU')
            self.assertEqual(manager.entries['b'].status,'CPU')


class RuntimeTests(unittest.TestCase):
    def test_cached_results_bypass_model_load_and_oom_recovery_is_bounded(self):
        class Upscale(Loader):
            def __init__(self):
                super().__init__('cpu'); self.calls=0
            def upscale(self,image_path,scale):
                self.calls+=1
                if self.calls==1:
                    raise RuntimeError('CUDA out of memory')
                return 123
        with TemporaryDirectory() as folder:
            service=Upscale(); bundle=replace(ServiceBundle.create(),upscale=service)
            runtime=ExecutionRuntime(PipelineConfig(),bundle)
            state={'output_dir':folder}
            with patch.object(runtime.manager,'release_cache'), patch('services.model_manager.snapshot',return_value=memory()):
                result=runtime.run_node('upscale_objects',lambda s:{'value':service.upscale('image',2)},state)
                self.assertEqual(result['value'],123); self.assertEqual(service.calls,2)
                self.assertTrue(any(e.get('oom_retry')==1 for e in runtime.profiler.events))
                # Degraded results are deliberately not cached under a full-quality key.
                service._inference_scale=1
                runtime.run_node('upscale_objects',lambda s:{'value':service.upscale('image',2)},state)
                loads=service.loads
                runtime.run_node('upscale_objects',lambda s:{'value':service.upscale('image',2)},state)
                self.assertEqual(service.loads,loads);self.assertEqual(service.calls,3)

    def test_repeated_graph_hits_cache_and_writes_complete_reports(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); source=root/'source.png'; Image.new('RGB',(80,60),'green').save(source)
            config=PipelineConfig(scene_loop={'enabled':False},p1={'enabled':False},detection={'diagnostics':{'enabled':False}})
            graph=build_graph(config)
            state={'source_path':str(source),'output_dir':str(root/'out')}
            first=graph.invoke(state);second=graph.invoke(state)
            report=json.loads(Path(second['performance_report_path']).read_text())
            self.assertGreater(report['cache']['hit'],0)
            self.assertEqual(report['nodes'][-1]['name'],'export')
            self.assertFalse(any(e['kind']=='load' for e in report['events']))
            self.assertTrue(Path(second['timeline_path']).exists())
            self.assertEqual(len(first['objects']),len(second['objects']))
            self.assertTrue(all(Path(p).exists() for p in second['exported_assets']))

    def test_exhausted_detector_oom_exports_reviewable_task(self):
        from services.grounding_service import GroundingService
        class Detector(GroundingService):
            def detect_p0(self,*args,**kwargs):
                raise RuntimeError('CUDA out of memory')
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.png';Image.new('RGB',(20,20),'green').save(source)
            config=PipelineConfig(scene_loop={'enabled':False},p1={'enabled':False},resources={'max_oom_retry':1},detection={'diagnostics':{'enabled':False}})
            bundle=replace(ServiceBundle.create(),grounding=Detector())
            graph=build_graph(config,bundle)
            result=graph.invoke({'source_path':str(source),'output_dir':str(root/'out')})
            self.assertEqual(result['resource_failures'][0]['node'],'detect_instances')
            self.assertTrue(Path(result['scene_json']).exists())
            manifest=json.loads(Path(result['scene_json']).read_text())
            self.assertTrue(manifest['resource_failures'])
