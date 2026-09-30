"""P1 contracts: bounded windows, prompt transforms, mask ranking and ownership."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from app.config import PipelineConfig
from app.p1_config import P1Config
from cv.mask import read_mask, rectangle_mask, save_mask
from diagnostics.reconstruction_diff import write_reconstruction_diff
from nodes.assign_ownership import make_assign_ownership
from nodes.reconstruct_scene import make_reconstruct_scene
from ownership.pixel_owner import build_pixel_ownership
from scene.element_classifier import SceneElementClassifier
from schemas.object import SceneObject
from segmentation.local_refiner import LocalRefiner, candidate_metrics, local_windows
from segmentation.mask_postprocess import mask_postprocess_by_category
from segmentation.prompt_generator import generate_prompts


class FakeLocalSAM:
    mock = False

    def __init__(self):
        self.calls = []

    def predict_candidates(self, image, prompts):
        self.calls.append((image.size, deepcopy(prompts)))
        x, y, r, b = np.rint(prompts['box']).astype(int)
        masks = np.zeros((3, image.height, image.width), np.uint8)
        masks[0] = 1  # Highest SAM score, but leaks through the entire crop.
        masks[1, y:b, x:r] = 1
        masks[2, y:y+1, x:x+1] = 1
        return masks, np.array([.99, .8, .7])


class LocalGeometryTests(unittest.TestCase):
    def test_adaptive_padding_and_bounds(self):
        tiny = {'x':200, 'y':200, 'w':20, 'h':30}
        self.assertEqual(local_windows(tiny, (1000, 1000), 512), [(170, 170, 250, 260)])
        for box in [tiny, {'x':0, 'y':0, 'w':1000, 'h':700},
                    {'x':975, 'y':965, 'w':25, 'h':35}]:
            cover = np.zeros((1000,1000), bool)
            for x,y,r,b in local_windows(box, (1000,1000), 256):
                self.assertTrue(0 <= x < r <= 1000 and 0 <= y < b <= 1000)
                self.assertLessEqual(max(r-x,b-y), 256)
                cover[y:b,x:r] = True
            self.assertTrue(cover[box['y']:box['y']+box['h'],box['x']:box['x']+box['w']].all())

    def test_prompts_are_original_global_and_transformed_once(self):
        obj = {'id':'flag', 'bbox':{'x':100,'y':80,'w':12,'h':24},
               'positive_points':[[101,88]],'negative_points':[[97,90]]}
        other = {'id':'pillar','bbox':{'x':120,'y':80,'w':6,'h':20}}
        prompts, trace = generate_prompts(obj, [obj,other], (90,60,140,120), 4)
        np.testing.assert_array_equal(prompts['box'], [40,80,88,176])
        self.assertIn([101,88], trace['positive_points'])
        self.assertIn([123,90], trace['negative_points'])
        self.assertTrue(set(prompts['point_labels']) == {0,1})

    def test_local_multimask_rejects_high_confidence_background(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'source.png'
            Image.new('RGB', (900,800), 'green').save(path)
            box = {'x':701,'y':613,'w':20,'h':30}
            record = {'id':'flag_001','category':'flag','confidence':.9,'bbox':box}
            service = FakeLocalSAM()
            result = LocalRefiner(service, P1Config()).segment(str(path), [record])[0]
            np.testing.assert_array_equal(result['mask'], rectangle_mask(900,800,box))
            self.assertEqual(result['segmentation']['windows'][0]['selected'], 1)
            self.assertEqual(service.calls[0][0], (320,360))
            self.assertEqual(result['bbox'],box)

    def test_thin_rods_and_disconnected_foliage_survive(self):
        mask = np.zeros((25,25),np.uint8)
        mask[2:22,12] = 255
        mask[3,3] = 255
        for category in ('flag','wood_pillar','tree','bush'):
            np.testing.assert_array_equal(mask_postprocess_by_category(mask,category),mask)

    def test_empty_mask_and_wrong_crop_limits_fail_safely(self):
        image = Image.new('RGB',(10,10))
        self.assertEqual(candidate_metrics(np.zeros((10,10)),[2,2,5,5],image,.99)['score'],-1)
        with self.assertRaises(ValueError):
            P1Config(max_local_crop_size=2048,max_sam_input_size=1024)
        with self.assertRaises(ValueError):
            local_windows({'x':9,'y':9,'w':2,'h':2},(10,10),64)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        self.rgba = np.full((40,50,4),255,np.uint8)
        self.rgba[:,:,:3] = (40,110,70)
        self.rgba[:3,:,3] = 0
        self.rgba[12:18,10:14,3] = 128
        Image.fromarray(self.rgba).save(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, ident, category, box, **extra):
        path = save_mask(rectangle_mask(50,40,box), self.root/f'{ident}.png')
        r = SceneObject(id=ident,category=category,confidence=.9,bbox=box,mask_path=path,
                        candidate_mask_path=path,status='pass',**extra).model_dump(mode='json')
        return SceneElementClassifier().classify(r,(50,40))

    def test_exclusive_pixels_terrain_merge_and_semantic_residual(self):
        records = [self.record('g1','grass',{'x':0,'y':0,'w':30,'h':40}),
                   self.record('g2','grass_ground',{'x':20,'y':0,'w':30,'h':40}),
                   self.record('tree','tree',{'x':10,'y':12,'w':12,'h':20})]
        payload,table,terrain = build_pixel_ownership(str(self.source),records,self.root/'out')
        owner = np.load(payload['owner_map_path'])
        self.assertEqual(len(terrain),1)
        self.assertEqual(terrain[0]['category'],'grass')
        tree_owner = next(r['owner_id'] for r in table if 'tree' in r.get('instance_ids',[]))
        self.assertTrue((owner[12:32,10:22] == tree_owner).all())
        self.assertFalse(owner[:3].any())
        self.assertEqual(payload['unassigned_ratio'],0)
        masks = [read_mask(r['visible_mask_path'])>0 for r in records if r['element_type']!='terrain']
        masks += [read_mask(t['visible_mask_path'])>0 for t in terrain]
        self.assertEqual(int(np.max(np.sum(masks,axis=0))),1)
        self.assertGreater(payload['overlap_ratio'],0)
        self.assertGreater(records[0]['occluded_pixel_count'],0)
        self.assertIsNone(records[-1]['full_mask_path'])
        self.assertTrue(payload['conflicts'])

    def test_explicit_occlusion_overrides_class_without_changing_ids(self):
        box={'x':5,'y':5,'w':12,'h':12}
        records=[self.record('tree','tree',box,occludes=['pillar']),self.record('pillar','pillar',box)]
        payload,table,_=build_pixel_ownership(str(self.source),records,self.root/'out')
        owner=np.load(payload['owner_map_path'])
        tree_id=next(r['owner_id'] for r in table if 'tree' in r.get('instance_ids',[]))
        self.assertTrue((owner[5:17,5:17]==tree_id).all())
        self.assertEqual(records[1]['visible_pixel_count'],0)

    def test_visible_reconstruction_preserves_alpha_and_disallows_double_paste(self):
        records=[self.record('grass','grass',{'x':0,'y':0,'w':50,'h':40}),
                 self.record('tree','tree',{'x':10,'y':10,'w':12,'h':20})]
        from app.objects import crop_records
        config=PipelineConfig()
        records=crop_records(str(self.source),records,self.root/'out',config.crop)
        for r in records: r['status']='pass'
        state={'source_path':str(self.source),'output_dir':str(self.root/'out'),'width':50,'height':40,'objects':records}
        before=deepcopy(state)
        state.update(make_assign_ownership(config)(state))
        self.assertEqual(before['objects'],records)
        self.assertEqual(len(state['objects']),1)
        state.update(make_reconstruct_scene(True,config.p1)(state))
        with Image.open(state['reconstruction_path']) as result:
            np.testing.assert_array_equal(np.array(result)[self.rgba[:,:,3]>8],self.rgba[self.rgba[:,:,3]>8])
        self.assertAlmostEqual(state['ownership']['reconstruction_diff']['ssim'],1,places=6)
        self.assertEqual(state['ownership']['reconstruction_diff']['alpha_gap'],0)

    def test_residual_is_not_semantic_coverage_and_can_be_disabled(self):
        records=[self.record('tree','tree',{'x':10,'y':10,'w':5,'h':8})]
        a,_,_=build_pixel_ownership(str(self.source),deepcopy(records),self.root/'a')
        b,_,_=build_pixel_ownership(str(self.source),deepcopy(records),self.root/'b',preserve_residual=False)
        self.assertEqual(a['unassigned_ratio'],b['unassigned_ratio'])
        self.assertGreater(a['unassigned_ratio'],.9)
        self.assertEqual(a['visible_coverage_ratio'],1)
        self.assertLess(b['visible_coverage_ratio'],.1)
        self.assertIsNone(b['residual_background']['asset_path'])

    def test_diff_detects_alpha_gaps_not_hidden_rgb(self):
        altered=self.rgba.copy(); altered[10:14,15:20,3]=0
        path=self.root/'gap.png'; Image.fromarray(altered).save(path)
        diff=write_reconstruction_diff(self.source,path,self.root/'diff.png')
        self.assertEqual(diff['missing_pixel_count'],20)
        self.assertLess(diff['ssim'],1)
        with Image.open(self.root/'diff.png') as image:
            self.assertEqual(image.getpixel((16,11)),(255,0,0))

    def test_hidden_rgb_does_not_create_reconstruction_error(self):
        altered = self.rgba.copy()
        altered[:3,:,:3] = (240,10,200)
        path = self.root/'hidden_rgb.png'
        Image.fromarray(altered).save(path)
        diff = write_reconstruction_diff(self.source,path,self.root/'hidden_diff.png')
        self.assertAlmostEqual(diff['ssim'],1,places=6)
        self.assertEqual(diff['mean_pixel_difference'],0)
        self.assertEqual(diff['high_difference_pixel_count'],0)


if __name__ == '__main__':
    unittest.main()
