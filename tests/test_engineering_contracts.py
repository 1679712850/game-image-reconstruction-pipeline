"""Regression contracts for engineering changes; synthetic fixtures verify mechanics only."""
import json
import os
import struct
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PIL import Image
from app.config import PipelineConfig, UpscaleConfig
from app.detection_config import DetectionConfig
from benchmarks.metrics import evaluate, mask_metrics
from benchmarks.schema import Annotation, EvaluationConfig
from benchmarks.report import regression
from detection.budget import BudgetTracker, restore_budget
from detection.p0_pipeline import P0DetectionPipeline
from exporters.psd_exporter import export_layers
from nodes.upscale_objects import make_upscale_objects
from schemas.scene import SceneManifest
from services.model_fingerprint import fingerprint_path
from services.upscale_service import UpscaleService


class MetricsTests(unittest.TestCase):
    def test_exact_duplicates_fragments_false_positives_and_misses(self):
        ann=Annotation(scene_id='s',image='s.png',roi=(0,0,100,100),provenance='test',objects=[
            {'id':'a','category':'tree','bbox':[0,0,10,10]},
            {'id':'b','category':'tree','bbox':[20,0,40,10]},
            {'id':'c','category':'tree','bbox':[60,0,70,10]}])
        predictions=[{'id':str(i),'category':'tree','bbox':b,'confidence':1-i/10} for i,b in enumerate([
            [0,0,10,10],[0,0,10,10],[20,0,29,10],[31,0,40,10],[80,0,90,10]])]
        r=evaluate(ann,predictions,(100,100),Path('.'),Path('.'),EvaluationConfig())
        self.assertAlmostEqual(r['detection']['recall'],1/3)
        self.assertAlmostEqual(r['detection']['precision'],1/5)
        self.assertAlmostEqual(r['detection']['duplicate_rate'],1/5)
        self.assertAlmostEqual(r['detection']['fragmentation_rate'],1/3)
        self.assertEqual(r['missed'],['b','c'])

    def test_mask_metrics_and_semantic_region_coexist(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); mask=np.zeros((10,10),np.uint8);mask[2:8,2:8]=255
            Image.fromarray(mask).save(root/'gt.png')
            ann=Annotation(scene_id='s',image='s.png',roi=(0,0,10,10),provenance='test',objects=[
                {'id':'a','category':'tree','bbox':[2,2,8,8],'mask':'gt.png'},
                {'id':'region','category':'tree','bbox':[2,2,8,8],'mask':'gt.png','kind':'semantic_region'}])
            r=evaluate(ann,[{'id':'p','category':'tree','bbox':[2,2,8,8],'visible_mask_path':'gt.png'}],(10,10),root,root,EvaluationConfig())
            self.assertEqual(r['mask_quality']['iou'],1)
            self.assertEqual(r['mask_quality']['boundary_fscore'],1)
            self.assertEqual(r['semantic_regions'][0]['completeness'],1)
            partial=mask.copy();partial[2:5]=0
            m=mask_metrics(mask>0,partial>0)
            self.assertEqual(m['completeness'],.5);self.assertEqual(m['contamination'],0)

    def test_ignore_regions_and_outside_roi_do_not_create_false_positives(self):
        ann=Annotation(scene_id='s',image='s.png',roi=(0,0,50,50),provenance='test',categories=['tree'],
                       ignore_regions=[[10,10,20,20]],objects=[])
        r=evaluate(ann,[{'id':'ignore','category':'tree','bbox':[10,10,20,20]},
                       {'id':'outside','category':'tree','bbox':[70,70,90,90]}],(100,100),Path('.'),Path('.'),EvaluationConfig())
        self.assertEqual(r['ignored_predictions'],['ignore']);self.assertEqual(r['detection']['prediction_count'],0)

    def test_pixel_perfect_residual_still_reports_215_percent_semantics(self):
        from cv.completion_metrics import completion_metrics
        from cv.metrics import reconstruction_similarity
        from cv.reconstruct import reconstruct_scene
        with TemporaryDirectory() as folder:
            root=Path(folder)
            original=Image.new('RGBA',(20,10),'red');original.save(root/'source.png')
            mask=np.zeros((10,20),np.uint8);mask.flat[:43]=255
            Image.fromarray(mask).save(root/'mask.png')
            asset=original.copy();asset.putalpha(Image.fromarray(mask));asset.save(root/'asset.png')
            residual=original.copy();residual.putalpha(Image.fromarray(255-mask));residual.save(root/'residual.png')
            box={'x':0,'y':0,'w':20,'h':10}
            obj={'id':'x','asset_path':str(root/'asset.png'),'visible_mask_path':str(root/'mask.png'),
                 'base_asset_status':'ready','crop_bbox':box}
            preview=reconstruct_scene(20,10,[{**obj,'z_order':1},{'asset_path':str(root/'residual.png'),'crop_bbox':box,'z_order':0}],root/'preview.png')
            with Image.open(preview) as image:
                score=reconstruction_similarity(original,image)
            m=completion_metrics({'source_path':str(root/'source.png'),'width':20,'height':10,'objects':[obj],
                'reconstruction_score':score,'ownership':{'eligible_pixels':200,'coverage_ratio':.215,
                'residual_background':{'asset_path':str(root/'residual.png'),'pixel_count':157}}})
            self.assertEqual(score,1);self.assertEqual(m['scene']['semantic_coverage'],.215)
            self.assertEqual(m['scene']['unassigned_ratio'],.785);self.assertEqual(m['status'],'partial')

    def test_gate_does_not_hide_tradeoffs_or_unknown_truncation(self):
        base={'scene_id':'s','annotation_digest':'a','image_digest':'b','inventory_kind':'x','annotation_status':'reviewed',
              'truncated':False,'detection':{'recall':.8,'precision':.9,'duplicate_rate':.1},'performance':{'runtime':100},'mask_quality':{'boundary_fscore':.8}}
        current={**base,'detection':{**base['detection'],'recall':.7},'performance':{'runtime':70}}
        gate=regression(base,current,EvaluationConfig().thresholds)
        self.assertEqual(gate['status'],'FAIL');self.assertEqual(gate['checks']['runtime']['status'],'PASS')
        self.assertEqual(regression(base,{**base,'truncated':None},EvaluationConfig().thresholds)['status'],'WARN')


class EnhancementTests(unittest.TestCase):
    def test_optional_disabled_missing_failed_and_required(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'a.png';Image.new('RGBA',(10,10),'red').save(path)
            obj={'id':'a','asset_path':str(path),'mask_path':str(path),'status':'pass',
                 'qa':{'status':'pass','reason':'passed'},'asset_library_eligible':True}
            service=UpscaleService(False)
            for enabled,expected in [(False,'disabled'),('auto','skipped'),(True,'unavailable')]:
                out=make_upscale_objects(service,enabled)({'objects':[obj]})['objects'][0]
                self.assertEqual(out['base_asset_status'],'ready');self.assertEqual(out['enhancement']['status'],expected)
                self.assertFalse(out['review']['required']);self.assertTrue(out['asset_library_eligible'])
            with patch.object(service,'upscale',side_effect=RuntimeError('backend unavailable')):
                out=make_upscale_objects(service,True)({'objects':[obj]})['objects'][0]
                self.assertEqual(out['enhancement']['status'],'failed');self.assertEqual(out['status'],'pass')
            out=make_upscale_objects(service,True,required=True)({'objects':[obj]})['objects'][0]
            self.assertEqual(out['base_asset_status'],'ready');self.assertTrue(out['review']['required'])
            self.assertEqual(out['status'],'manual_review')


class ConfigBudgetTests(unittest.TestCase):
    def test_defaults_aliases_precedence_and_warnings(self):
        self.assertFalse(UpscaleConfig().enabled)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            cfg=PipelineConfig.model_validate({'max_retry':9,'pipeline':{'max_retry':1,'detection':{
                'tiled':{'tile_size':800},'tiling':{'tile_size':900}}}})
        self.assertEqual(cfg.max_retry,1);self.assertEqual(cfg.detection.tiling.tile_size,900)
        self.assertEqual(len(messages),2)
        self.assertIsNone(PipelineConfig(scene_loop={'max_objects':None}).scene_loop.max_objects)

    def test_budget_restored_across_rounds_and_retry(self):
        cfg=DetectionConfig(budget={'max_total_inference_calls':3,'max_retry_calls':1}).budget
        budget=BudgetTracker(cfg)
        self.assertTrue(budget.begin_pass('global'))
        self.assertTrue(budget.consume('global'))
        restored=restore_budget(cfg,budget.report())
        self.assertFalse(restored.begin_pass('global'))
        self.assertTrue(restored.consume('redetection'));self.assertFalse(restored.consume('redetection'))
        self.assertTrue(restored.consume('tile'));self.assertFalse(restored.consume('tile'))

    def test_expanded_taxonomy_planning_reduces_calls_without_an_extra_model(self):
        from tests.test_p0_detection import settings
        def count(enabled):
            cfg=settings(expand_categories=True,category_planner={'enabled':enabled,'max_groups_per_tile':2})
            _,report=P0DetectionPipeline(cfg).run(Image.new('RGB',(100,60)),['tree'],lambda *args:[])
            return report['scan_cost']['inference_calls']
        self.assertLess(count(True),count(False))

    def test_object_limit_is_explicit_in_production_export(self):
        from agent.graph import build_graph
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'s.png';Image.new('RGB',(100,100),'green').save(source)
            cfg=PipelineConfig(scene_loop={'max_rounds':1,'max_objects':1},p1={'enabled':False},
                               candidates={'export_psd':False},detection={'diagnostics':{'enabled':False}})
            state=build_graph(cfg).invoke({'source_path':str(source),'output_dir':str(root/'out')})
            limit=json.loads(Path(state['scene_json']).read_text())['detection']['object_limit']
            self.assertTrue(limit['truncated']);self.assertGreater(limit['objects_before_limit'],limit['objects_after_limit'])

    def test_explicit_prompt_members_survive_planning_to_avoid_precision_regression(self):
        from tests.test_p0_detection import settings
        cfg=settings(category_planner={'max_groups_per_tile':1})
        categories=['tree','building','stone_lantern','tombstone']
        calls=[]
        P0DetectionPipeline(cfg).run(Image.new('RGB',(100,60)),categories,lambda im,group,ctx:calls.append(ctx) or [])
        first_tile=[c for c in calls if c['tile_id']=='s64_tile_00_00']
        self.assertEqual({c for call in first_tile for c in call['categories']},set(categories))

    def test_scene_groups_bound_calls_and_preserve_global_discovery(self):
        from tests.test_p0_detection import settings
        image=Image.new('RGB',(100,60))
        calls=[]
        cfg=settings(category_planner={'max_groups_per_tile':1},budget={'max_total_inference_calls':4})
        _,report=P0DetectionPipeline(cfg).run(image,['tree','building','stone_lantern'],lambda image,group,ctx:calls.append(ctx) or [])
        self.assertLessEqual(len(calls),4)
        self.assertTrue(any(c['source']=='global' for c in calls))
        self.assertTrue(all(len(t['category_groups'])==1 for t in report['tiles']))


class PerformanceSchemaTests(unittest.TestCase):
    def test_fingerprint_uses_stats_not_weight_contents_and_invalidates(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pth';path.write_bytes(b'first')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('weight read')):
                first=fingerprint_path(folder)
                path.write_bytes(b'second-longer')
                self.assertNotEqual(first,fingerprint_path(folder))

    def test_model_snapshot_scans_once_per_task_and_new_task_invalidates(self):
        from services.model_fingerprint import service_versions
        from app.models import ModelConfig
        with TemporaryDirectory() as folder:
            root=Path(folder); (root/'weights.safetensors').write_bytes(b'weights')
            settings=ModelConfig(qwen_image_edit={'model_path':root})
            cache={}
            with patch('services.model_fingerprint.fingerprint_path',wraps=fingerprint_path) as scan:
                first=service_versions(settings,'image_edit',cache)
                self.assertEqual(first,service_versions(settings,'image_edit',cache))
                self.assertEqual(scan.call_count,1)
                (root/'weights.safetensors').write_bytes(b'changed-weights')
                self.assertNotEqual(first,service_versions(settings,'image_edit',{}))
                self.assertEqual(scan.call_count,2)

    def test_schema_loads_old_and_exports_new_without_mutating(self):
        payload={'schema_version':'1.1','mock':False,'scene':{'width':10,'height':10,'projection':'unknown'},
                 'description':'old','retry_count':0,'objects':[{'id':'x','category':'tree','confidence':1,
                 'bbox':{'x':1,'y':1,'w':2,'h':2},'crop_bbox':{'x':-2,'y':-3,'w':8,'h':9}}]}
        model=SceneManifest.model_validate(payload)
        self.assertEqual(model.schema_version,'1.2');self.assertEqual(payload['schema_version'],'1.1')
        self.assertEqual(model.objects[0].geometry['full_asset_bbox']['x'],-2)

    def test_geometry_exports_accepted_placement_not_failed_amodal_proposal(self):
        from exporters.json_exporter import export_geometry
        record={'bbox':{'x':2,'y':3,'w':4,'h':5},'crop_bbox':{'x':1,'y':2,'w':6,'h':7},
                'bbox_full':{'x':-30,'y':-20,'w':100,'h':100},'logical_size':[6,7]}
        geometry=export_geometry(record,Path('.'))
        self.assertEqual(geometry['full_asset_bbox'],record['crop_bbox'])
        self.assertEqual(geometry['full_asset_canvas'],[6,7])

    def test_psd_crops_layers_spools_and_records_budget(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);path=root/'small.png'
            image=Image.new('RGBA',(100,100));image.paste('red',(40,50,45,57));image.save(path)
            obj={'id':'树','asset_path':str(path),'crop_bbox':{'x':-1,'y':-2,'w':100,'h':100}}
            output=export_layers([obj],100,100,root/'out.psd',memory_budget_mb=.001)
            with Image.open(output) as result:
                self.assertEqual(result.convert('RGB').getpixel((39,48)),(255,0,0))
            info=json.loads((root/'out.export.json').read_text())
            self.assertTrue(info['budget_exceeded']);self.assertEqual(info['encoded_pixels'],35)
            with self.assertRaises(MemoryError):
                export_layers([obj],100,100,root/'blocked.psd',memory_budget_mb=.001,low_memory=False)
