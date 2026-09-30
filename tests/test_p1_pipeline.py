"""Integration contracts for P1 export, bounded retries and graph variants."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from agent.graph import build_graph
from app.config import PipelineConfig
from app.objects import crop_records
from cv.mask import rectangle_mask, save_mask
from diagnostics.p1_report import portable
from nodes.assign_ownership import make_assign_ownership
from nodes.build_metadata import build_metadata
from nodes.export import make_export
from nodes.p1_scene import make_p1_scene
from nodes.qa_objects import make_qa_objects
from nodes.reconstruct_scene import make_reconstruct_scene
from nodes.retry_objects import make_retry_objects
from retry.detection_retry import recover_regions
from retry.segmentation_retry import repair_record
from scene.element_classifier import SceneElementClassifier
from schemas.object import SceneObject
from schemas.scene_qa import CategoryReview, SceneReviewDecision
from services.sam_service import SAMService
from services.scene_review_service import SceneReviewService


class BoxSAM(SAMService):
    """Exercise the real local-refinement path without loading model weights."""
    def __init__(self, leak=False):
        super().__init__(mock=False)
        self.leak = leak
        self.calls = []

    def predict_candidates(self, image, prompts):
        self.calls.append((image.size, deepcopy(prompts)))
        x, y, right, bottom = np.rint(prompts['box']).astype(int)
        mask = np.zeros((1, image.height, image.width), np.uint8)
        if self.leak:
            mask[:] = 1
        else:
            mask[0, y:bottom, x:right] = 1
        return mask, np.array([.9])


class RegionDetector:
    def __init__(self, empty=False):
        self.calls = []
        self.empty = empty

    def detect(self, path, categories):
        self.calls.append((path, list(categories)))
        if self.empty:
            return []
        # At attempt 1 a 32-pixel pad and 2x crop restore this to the requested box.
        return [{'category': 'tree', 'confidence': .95, 'bbox': {'x':64, 'y':64, 'w':40, 'h':40}}]


class P1Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        Image.new('RGB', (400, 200), (40, 130, 60)).save(self.source)
        self.config = PipelineConfig(mock=False, crop={'padding': 0}, scene_loop={'enabled': False})

    def tearDown(self):
        self.temp.cleanup()

    def state(self, name='out'):
        return {'source_path': str(self.source), 'output_dir': str(self.root/name),
                'width':400, 'height':200, 'objects':[], 'all_detections':[],
                'scene_analysis': {'description':'test source', 'projection':'top_down'},
                'layer_plan':[{'name':'vegetation', 'categories':['tree'], 'order':0}]}

    def record(self, state, ident, category, box, **extra):
        mask = save_mask(rectangle_mask(400,200,box), Path(state['output_dir'])/'masks'/f'{ident}.png')
        record = SceneObject(id=ident, category=category, bbox=box, confidence=.95,
                             mask_path=mask, candidate_mask_path=mask, **extra).model_dump(mode='json')
        record = SceneElementClassifier().classify(record, (400,200))
        record = crop_records(str(self.source), [record], Path(state['output_dir']), self.config.crop)[0]
        record['status'] = 'pass'
        return record

    def region(self, x=70):
        return {'category':'tree', 'approx_bbox':{'x':x,'y':50,'w':20,'h':20},
                'reason':'visible object not covered', 'failure_type':'MISSED_DETECTION'}

    def test_export_contains_summary_retry_and_all_portable_references(self):
        state = self.state()
        state['objects'] = [self.record(state,'ground','grass',{'x':0,'y':0,'w':400,'h':200}),
                            self.record(state,'tree','tree',{'x':70,'y':50,'w':20,'h':20})]
        state['all_detections'] = deepcopy(state['objects'])
        evidence = Path(state['output_dir'])/'debug'/'retry_crop.png'
        evidence.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB',(20,20)).save(evidence)
        state['retry_history'] = [{'retry_reason':'MISSED_DETECTION', 'crop_path':str(evidence),
                                   'previous_score':0.0, 'new_score':.8, 'improved':True}]
        state['objects'][1]['retry_history'] = deepcopy(state['retry_history'])
        state.update(make_p1_scene(PipelineConfig(), Mock(), Mock(), SceneReviewService())(state))
        state.update(make_assign_ownership(self.config)(state))
        state.update(build_metadata(state))
        state.update(make_reconstruct_scene(True, self.config.p1)(state))
        before = deepcopy(state)
        result = make_export(True)(state)
        self.assertEqual(state, before)
        root = Path(state['output_dir'])
        manifest = json.loads(Path(result['scene_json']).read_text())
        summary = json.loads((root/'metadata'/'summary.json').read_text())
        self.assertEqual(manifest['p1_summary'], portable(result['p1_summary'], root))
        self.assertEqual(manifest['p1_summary'], summary)
        self.assertTrue(summary['mock'])
        self.assertEqual(summary['detections'], 2)
        self.assertEqual(summary['instances'], 1)
        self.assertEqual(summary['terrain_layers'], 1)
        self.assertEqual(summary['reconstruction']['path'], 'diagnostics/reconstruction_diff.png')
        self.assertEqual(manifest['retry_history'], json.loads((root/'metadata'/'retries.json').read_text()))
        self.assertNotIn(str(root), json.dumps(manifest))
        self.assertTrue(all(Path(path).is_file() for path in result['exported_assets']))

        def references(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if (key.endswith('_path') or key == 'path') and isinstance(child, str) and child:
                        yield child
                    else:
                        yield from references(child)
            elif isinstance(value, list):
                for child in value:
                    yield from references(child)

        for path in references(manifest):
            self.assertFalse(Path(path).is_absolute(), path)
            self.assertTrue((root/path).is_file(), path)
            self.assertIn(str((root/path).resolve()), result['exported_assets'])
        # Resuming or retrying the export node is safe and reproducible.
        again = make_export(True)(state)
        self.assertEqual(result['p1_summary'], again['p1_summary'])

    def test_graph_feature_matrix_exports_with_zero_retry_budgets(self):
        for p1 in (False, True):
            for loop in (False, True):
                for completion in (False, True):
                    with self.subTest(p1=p1, loop=loop, completion=completion):
                        config = PipelineConfig(
                            max_retry=0, exercise_retry=True, upscale={'enabled':False},
                            p1={'enabled':p1,'max_scene_retry':0,'max_detection_retry':0,'max_segmentation_retry':0},
                            scene_loop={'enabled':loop,'max_rounds':1},
                            detection={'diagnostics':{'enabled':False}},
                            object_completion={'enabled':completion})
                        visits = []
                        state = build_graph(config, progress=visits.append).invoke(
                            self.state(f'matrix_{p1}_{loop}_{completion}'))
                        self.assertEqual(visits[-1], 'export')
                        self.assertNotIn('retry_objects', visits)
                        self.assertEqual('p1_scene' in visits, p1)
                        self.assertEqual('qa_scene' in visits, loop)
                        self.assertEqual('complete_objects' in visits, completion)
                        self.assertEqual(state['retry_count'], 0)
                        if p1:
                            self.assertLess(visits.index('p1_scene'), visits.index('assign_ownership'))
                            target = 'complete_objects' if completion else 'upscale_objects'
                            self.assertLess(visits.index('assign_ownership'), visits.index(target))
                            self.assertTrue(state['p1_summary'])
                        else:
                            self.assertFalse(state.get('ownership'))
                            self.assertFalse((Path(state['output_dir'])/'diagnostics').exists())

    def test_local_recovery_preserves_coordinates_and_unique_ids_across_rounds(self):
        state, detector, sam = self.state(), RegionDetector(), BoxSAM()
        first, history = recover_regions(state,[self.region()],detector,sam,self.config)
        self.assertEqual(len(first),1)
        self.assertEqual(first[0]['bbox'],self.region()['approx_bbox'])
        self.assertEqual(first[0]['group'],'vegetation')
        original_mask = Path(first[0]['mask_path']).read_bytes()
        state.update(objects=first, all_detections=first, retry_history=history)
        second, _ = recover_regions(state,[self.region(230)],detector,sam,self.config,scene_attempt=2)
        self.assertEqual(len(second),1)
        self.assertNotEqual(first[0]['id'], second[0]['id'])
        self.assertNotEqual(first[0]['mask_path'], second[0]['mask_path'])
        self.assertEqual(second[0]['bbox'],self.region(230)['approx_bbox'])
        self.assertEqual(original_mask,Path(first[0]['mask_path']).read_bytes())

    def test_detection_budget_is_shared_across_rounds_and_respects_object_limit(self):
        state, detector = self.state(), RegionDetector(empty=True)
        first, history = recover_regions(state,[self.region()],detector,BoxSAM(),self.config)
        self.assertEqual(first,[])
        self.assertEqual(len(detector.calls),2)
        state['retry_history'] = history
        second, logs = recover_regions(state,[self.region()],detector,BoxSAM(),self.config,scene_attempt=2)
        self.assertEqual(second,[])
        self.assertEqual(logs,[])
        self.assertEqual(len(detector.calls),2)
        limited = PipelineConfig(mock=False, scene_loop={'max_objects':1},crop={'padding':0})
        objects, _ = recover_regions(self.state('limited'),[self.region(),self.region(230)],
                                     RegionDetector(),BoxSAM(),limited)
        self.assertEqual(len(objects),1)

    def test_vision_missed_proposals_survive_zero_budget_and_smooth_background(self):
        config = PipelineConfig(mock=False,p1={'max_detection_retry':0,'max_scene_retry':0})
        reviewer = Mock(backend='llm')
        reviewer.review.return_value = SceneReviewDecision(
            continue_detection=True,reason='Missing tree',status='needs_detection',
            missed_objects=[{key:value for key,value in self.region().items() if key != 'failure_type'}])
        detector = Mock()
        result = make_p1_scene(config,detector,BoxSAM(),reviewer)(self.state())
        self.assertIn('tree',[r['category'] for r in result['missed_object_candidates']])
        detector.detect.assert_not_called()
        self.assertFalse(result['retry_history'])

    def test_p1_scene_exports_recovered_detections_and_provenance(self):
        state = self.state()
        reviewer = Mock(backend='llm')
        reviewer.review.return_value = SceneReviewDecision(
            continue_detection=True,reason='Missing tree',status='needs_detection',
            missed_objects=[{key:value for key,value in self.region().items() if key != 'failure_type'}])
        config = PipelineConfig(mock=False,crop={'padding':0},
                                p1={'max_scene_retry':1,'max_detection_retry':1,'max_problem_regions':1})
        result = make_p1_scene(config,RegionDetector(),BoxSAM(),reviewer)(state)
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['all_detections'][0]['id'],result['objects'][0]['id'])
        self.assertTrue(result['retry_history'][0]['improved'])
        self.assertTrue(Path(result['retry_history'][0]['crop_path']).exists())
        self.assertFalse(any(r['category']=='tree' for r in result['missed_object_candidates']))

    def test_worse_segmentation_cannot_replace_accepted_mask(self):
        state = self.state()
        record = self.record(state,'tree','tree',{'x':70,'y':50,'w':20,'h':20})
        record['qa'] = {'status':'retry','reason':'test conflict','retry_strategy':'rerun_segmentation',
                        'failure_types':['BACKGROUND_LEAK']}
        state['objects'] = [record]
        before = deepcopy(record)
        result, log = repair_record(record,state,BoxSAM(leak=True),self.config,1)
        self.assertFalse(log['improved'])
        self.assertLess(log['new_score'],log['previous_score'])
        self.assertEqual(result,before)
        self.assertEqual(record,before)

    def test_wrong_category_uses_vision_classification_without_sam(self):
        state = self.state()
        record = self.record(state,'tree','tree',{'x':70,'y':50,'w':20,'h':20})
        record.update(confidence=.1,status='retry',qa={
            'status':'retry','reason':'wrong category','retry_strategy':'change_prompt','failure_types':['WRONG_CATEGORY']})
        state.update(objects=[record], failed_objects=['tree'],retry_count=0,max_retry=1)
        reviewer = Mock(backend='llm')
        reviewer.classify_crop.return_value = CategoryReview(category='rock',confidence=.99,
                                                             uncertain=False,reason='Clearly a rock')
        sam = Mock()
        update = make_retry_objects(self.config,sam,reviewer)(state)
        self.assertEqual(update['objects'][0]['category'],'rock')
        self.assertEqual(update['objects'][0]['group'],'rock')
        self.assertEqual(update['objects'][0]['mask_path'],record['mask_path'])
        reviewer.classify_crop.assert_called_once()
        sam.predict_candidates.assert_not_called()
        qa = make_qa_objects(self.config)({**state,**update})
        self.assertEqual(qa['failed_objects'],[])

    def test_fragment_retry_requires_complete_detection_and_improved_qa(self):
        from retry.fragment_retry import repair_fragment
        state = self.state()
        record = self.record(state,'fragment','tree',{'x':70,'y':50,'w':20,'h':20})
        record.update(is_truncated=True,truncated_edges=['right'],status='retry',qa={
            'status':'retry','reason':'tile edge','retry_strategy':'merge_neighbor_tiles',
            'failure_types':['CROSS_TILE_FRAGMENT']})
        state.update(objects=[record],failed_objects=['fragment'],retry_count=0,max_retry=1)
        detector = RegionDetector()
        result = make_retry_objects(self.config,BoxSAM(),detector=detector)(state)
        repaired = result['objects'][0]
        self.assertEqual(repaired['id'],record['id'])
        self.assertFalse(repaired['is_truncated'])
        self.assertTrue(repaired['redetected'])
        self.assertEqual(len(detector.calls),1)
        self.assertGreater(result['retry_history'][0]['new_score'],result['retry_history'][0]['previous_score'])
        self.assertEqual(make_qa_objects(self.config)({**state,**result})['failed_objects'],[])
        empty = RegionDetector(empty=True)
        unchanged, log = repair_fragment(record,state,empty,BoxSAM(),self.config,1)
        self.assertEqual(unchanged,record)
        self.assertFalse(log['improved'])
        self.assertTrue(unchanged['is_truncated'])

    def test_fully_occluded_object_has_no_visible_asset_but_keeps_candidate(self):
        state = self.state()
        box = {'x':70,'y':50,'w':20,'h':20}
        tree = self.record(state,'tree','tree',box)
        pillar = self.record(state,'pillar','pillar',box)
        tree['occludes'] = ['pillar']
        state['objects'] = [tree,pillar]
        result = make_assign_ownership(self.config)(state)
        hidden = next(o for o in result['objects'] if o['id'] == 'pillar')
        self.assertEqual(hidden['status'],'pass')
        self.assertEqual(hidden['visible_pixel_count'],0)
        self.assertEqual(hidden['occluded_pixel_count'],400)
        self.assertIsNone(hidden['asset_path'])
        self.assertIsNone(hidden['error'])
        self.assertTrue(Path(hidden['candidate_mask_path']).is_file())
        self.assertEqual(result['failed_objects'],[])

    def test_terrain_completion_preserves_visible_alpha_and_reports_terrain_failures(self):
        state = self.state()
        rgba = np.full((200,400,4),255,np.uint8)
        rgba[:,:,:3] = (40,130,60)
        rgba[10,10,3] = 128
        Image.fromarray(rgba).save(self.source)
        grass = self.record(state,'grass','grass',{'x':0,'y':0,'w':400,'h':200})
        grass['status'] = 'manual_review'
        tree = self.record(state,'tree','tree',{'x':70,'y':50,'w':20,'h':20})
        state['objects'] = [grass,tree]
        state['all_detections'] = deepcopy(state['objects'])
        state.update(make_assign_ownership(self.config)(state))
        terrain = state['terrain_layers'][0]
        self.assertEqual(terrain['occluded_pixel_count'],400)
        self.assertEqual(terrain['qa_status'],'manual_review')
        with Image.open(terrain['complete_asset_path']) as completed:
            self.assertEqual(completed.getpixel((10,10))[3],128)
        result = make_export(False)(state)
        self.assertEqual(result['p1_summary']['terrain_qa_failed_layers'],1)
        self.assertEqual(result['p1_summary']['status'],'manual_review')

    def test_unresolved_p0_fragments_are_targeted_even_without_texture_proposals(self):
        state = self.state()
        state['detection_runs'] = [{'review_candidate_pool':[{
            'id':'p0_fragment','category':'tree','bbox':[70,50,90,70],'is_truncated':True}]}]
        config = PipelineConfig(mock=False,crop={'padding':0},
                                p1={'max_problem_regions':1,'max_scene_retry':1,'max_detection_retry':1})
        result = make_p1_scene(config,RegionDetector(),BoxSAM(),SceneReviewService())(state)
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['retry_history'][0]['retry_reason'],'CROSS_TILE_FRAGMENT')
        self.assertTrue(result['retry_history'][0]['improved'])


if __name__ == '__main__':
    unittest.main()
