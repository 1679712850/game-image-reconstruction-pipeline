"""Occlusion completion acceptance cases with deterministic model doubles.

These validate data/geometry/QA contracts, not pretrained model image quality.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import importlib.util
import numpy as np
from PIL import Image, ImageDraw
from app.config import PipelineConfig, AmodalConfig
from app.models import ModelConfig
from candidates.pipeline import run_candidates
from candidates.reconstruction import prepare, completion_metrics
from candidates.alpha import refine_alpha, matte_edges
from schemas.reconstruction import OcclusionAnalysis
from schemas.object import SceneObject
from services.upscale_service import UpscaleService
from nodes.upscale_objects import make_upscale_objects
from services.scene_review_service import SceneReviewService
from tests.test_p2_candidates import Review, Generator, Segmenter, scores
from schemas.candidate import CandidateQA


class AmodalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.source=self.root/'visible.png'
        image=Image.new('RGBA',(32,40)); ImageDraw.Draw(image).rectangle((8,4,23,29),fill=(40,130,50,255));image.save(self.source)
        self.obj={'id':'asset','category':'tree','confidence':.95,'status':'pass',
            'bbox':{'x':10,'y':10,'w':32,'h':40},'crop_bbox':{'x':10,'y':10,'w':32,'h':40},
            'asset_path':str(self.source),'requires_inpainting':True,'visible_pixel_count':416,'occluded_pixel_count':128}
        self.state={'source_path':str(self.source),'output_dir':str(self.root),'objects':[self.obj],'width':100,'height':100}
        self.config=PipelineConfig(object_completion={'enabled':True},resources={'max_generation_retry':1})

    def tearDown(self):
        self.tmp.cleanup()

    def run_asset(self, generator=None, reviewer=None, sam=None):
        return run_candidates(self.state,self.config,generator or Generator(),sam=sam or Segmenter(),reviewer=reviewer or Review())

    def test_tree_mountain_building_completion_expands_alpha_and_preserves_coordinates(self):
        for category in ('tree','mountain','building','pillar'):
            with self.subTest(category=category):
                self.obj['category']=category
                result=self.run_asset(); obj=result['objects'][0]
                self.assertEqual(obj['status'],'pass')
                self.assertTrue(obj['asset_library_eligible'])
                self.assertGreater(obj['completion_qa']['expansion_ratio'], .1)
                self.assertGreater(obj['completion_qa']['outside_visible_area'],0)
                self.assertGreaterEqual(obj['completion_qa']['visible_retention'],.99)
                self.assertEqual(obj['placement']['normalization_scale'],1)
                self.assertNotEqual(obj['bbox_visible'],obj['crop_bbox'])
                with Image.open(obj['accepted_asset']) as image:
                    self.assertEqual(image.mode,'RGBA')
                    self.assertGreater(np.count_nonzero(np.asarray(image)[:,:,3]>8),416)
                SceneObject.model_validate(obj)
                self.assertEqual(self.obj['crop_bbox'],{'x':10,'y':10,'w':32,'h':40})

    def test_canvas_is_transparent_and_does_not_end_at_semantic_mask(self):
        from candidates.reconstruction import analyze
        analysis=analyze(self.obj,self.state,Review())
        planned=prepare(self.obj,analysis,self.root/'plan',AmodalConfig())
        with Image.open(planned['expanded_crop_path']) as canvas, Image.open(planned['amodal_mask_path']) as mask:
            self.assertEqual(canvas.getpixel((0,0)),(0,0,0,0))
            box=mask.getbbox()
            self.assertGreater(box[0],0);self.assertLess(box[2],canvas.width)
            self.assertGreater(box[1],0);self.assertLess(box[3],canvas.height)
        self.assertLess(planned['canvas_bbox']['x'],0)

    def test_visible_only_first_result_triggers_bounded_retry(self):
        class FirstVisible(Generator):
            def _output(self,source,output_path):
                if len(self.prompts)==1:
                    with Image.open(source) as image:
                        array=np.array(image); pixels=np.zeros_like(array[:,:,:3]);pixels[array[:,:,3]>8]=[240,15,25]
                    Image.fromarray(pixels).save(output_path);return str(output_path)
                return super()._output(source,output_path)
        generator=FirstVisible(); result=self.run_asset(generator)
        self.assertEqual(len(generator.prompts),2)
        self.assertIn('previous result still preserves',generator.prompts[1])
        self.assertEqual(result['objects'][0]['accepted_candidate_id'],'inpaint_v2')
        first=result['candidate_registry']['asset']['candidates'][1]
        self.assertEqual(first['qa']['status'],'RETRY')
        self.assertIn('alpha expansion missing',str(first['qa']['reasons']))

    def test_no_segmenter_never_accepts_model_alpha(self):
        result=run_candidates(self.state,self.config,Generator(),reviewer=Review())
        obj=result['objects'][0]
        self.assertEqual(obj['accepted_candidate_id'],'original')
        self.assertTrue(obj['needs_manual_review']);self.assertFalse(obj['asset_library_eligible'])
        self.assertIn('independent foreground segmentation',str(result['candidate_registry']))

    def test_low_confidence_explicit_edit_cannot_bypass_review(self):
        class Uncertain(Review):
            def analyze_occlusion(self,*args):
                return {**super().analyze_occlusion(*args),'reconstruction_confidence':.1}
        mask=self.root/'request.png';Image.new('L',(32,40),255).save(mask)
        self.state['edit_requests']=[{'object_id':'asset','mask_path':str(mask),'prompt':'Restore the trunk'}]
        result=self.run_asset(reviewer=Uncertain())
        self.assertFalse(result['objects'][0]['asset_library_eligible'])
        self.assertTrue(result['objects'][0]['needs_manual_review'])
        self.assertFalse(any(c['qa']['status']=='ACCEPT' for c in result['candidate_registry']['asset']['candidates'][1:]))

    def test_severe_occlusion_has_three_candidates_and_tiny_visible_region_requires_review(self):
        self.obj.update(visible_pixel_count=10,occluded_pixel_count=90)
        generator=Generator();result=self.run_asset(generator)
        self.assertEqual(len(generator.prompts),3)
        self.assertEqual(len(set(generator.prompts)),3)
        self.assertFalse(result['objects'][0]['asset_library_eligible'])
        self.assertTrue(result['objects'][0]['needs_manual_review'])

    def test_outside_scene_completion_retains_full_asset_png(self):
        self.obj['bbox']['y']=80;self.obj['crop_bbox']['y']=80
        result=self.run_asset(); obj=result['objects'][0]
        self.assertGreater(obj['crop_bbox']['y']+obj['crop_bbox']['h'],100)
        with Image.open(obj['asset_mask_path']) as mask,Image.open(obj['accepted_asset']) as image:
            self.assertEqual(mask.size,(100,100))
            self.assertGreater(np.count_nonzero(np.asarray(image)[:,:,3]),np.count_nonzero(np.asarray(mask)))
        from cv.reconstruct import reconstruct_scene
        from exporters.psd_exporter import export_layers
        reconstruct_scene(100,100,[obj],self.root/'preview.png')
        export_layers([obj],100,100,self.root/'preview.psd')
        SceneObject.model_validate(obj)

    def test_noise_holes_and_semitransparent_edges_keep_thin_structure(self):
        mask=np.zeros((30,30),np.uint8);mask[5:25,10:20]=255;mask[2:5,15]=255
        mask[10,15]=0;mask[0,0]=255;mask[15,10]=100
        refined=refine_alpha(mask)
        self.assertEqual(refined[0,0],0);self.assertEqual(refined[10,15],255)
        self.assertTrue(np.all(refined[2:5,15]==255));self.assertEqual(refined[15,10],100)
        result=matte_edges(Image.new('RGB',(30,30),'red'),refined)
        self.assertEqual(result[15,10],100)
        self.assertTrue(np.all(result[2:5,15]>8))

    def test_excessive_expansion_and_border_contact_fail_qa(self):
        analysis=OcclusionAnalysis(object_id='x',object_type='tree',occlusion_ratio=.3,
            reconstruction_confidence=0,needs_completion=True,evidence='unavailable',max_expansion_ratio=1.5)
        visible=np.zeros((30,30),np.uint8);visible[10:20,10:20]=255
        metrics,reasons=completion_metrics(np.full((30,30),255,np.uint8),visible,analysis)
        self.assertTrue(metrics['touches_border']);self.assertIn('exceeds semantic limit',str(reasons))

    def test_border_contact_retries_on_a_larger_canvas(self):
        class AtBorder(Generator):
            def _output(self,source,output_path):
                if len(self.prompts)==1:
                    with Image.open(source) as image:
                        array=np.array(image);pixels=np.zeros_like(array[:,:,:3]);pixels[array[:,:,3]>8]=[240,15,25]
                    pixels[-1,:]=[240,15,25];Image.fromarray(pixels).save(output_path);return str(output_path)
                return super()._output(source,output_path)
        result=self.run_asset(AtBorder())
        candidates=result['candidate_registry']['asset']['candidates'][1:]
        self.assertEqual(len(candidates),2)
        self.assertTrue(candidates[0]['completion']['touches_border'])
        self.assertGreater(candidates[1]['completion']['canvas_bbox']['w'],candidates[0]['completion']['canvas_bbox']['w'])
        self.assertEqual(candidates[1]['qa']['status'],'ACCEPT')

    def test_invalid_semantic_mask_budget_is_isolated(self):
        self.config=PipelineConfig(object_completion={'enabled':True},amodal={'max_canvas_edge':64})
        result=self.run_asset()
        self.assertIn('planning_failed',str(result['candidate_registry']))
        self.assertTrue(result['objects'][0]['needs_manual_review'])


class HDTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional Torch is not installed')
    def test_rrdb_architecture_checkpoint_keys_and_4x_shape(self):
        import torch
        from services.rrdbnet import rrdbnet
        network=rrdbnet(num_feat=4,num_block=2,num_grow_ch=2)
        self.assertIn('body.1.rdb3.conv5.weight',network.state_dict())
        self.assertIn('conv_up2.weight',network.state_dict())
        with torch.inference_mode():
            output=network(torch.zeros(1,3,3,5))
        self.assertEqual(tuple(output.shape),(1,3,12,20))

    def test_neural_output_uses_completed_alpha_and_requires_visual_qa(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'completed.png'
            image=Image.new('RGBA',(16,24));ImageDraw.Draw(image).rectangle((3,2,12,20),fill=(150,30,20,255));image.save(source)
            service=UpscaleService(False)
            with patch.object(service,'_restore',return_value=Image.new('RGB',(64,96),(200,30,20))) as restore:
                path=service.upscale(str(source),4)
                restore.assert_called_once()
            with Image.open(path) as result:
                self.assertEqual(result.mode,'RGBA')
                np.testing.assert_array_equal(result.getchannel('A'),image.getchannel('A').resize((64,96),Image.Resampling.LANCZOS))
                self.assertEqual(result.getpixel((20,20))[0],200)
            obj={'id':'x','asset_path':str(source),'status':'pass','category':'tree','bbox':{'x':0,'y':0,'w':16,'h':24}}
            with patch.object(service,'upscale',return_value=path):
                output=make_upscale_objects(service,True,Review())({'objects':[obj]})['objects'][0]
            self.assertEqual(output['hd_asset_path'],path);self.assertTrue(output['hd_qa']['silhouette_preserved'])
            with patch.object(service,'upscale',return_value=path):
                output=make_upscale_objects(service,True)({'objects':[obj]})['objects'][0]
            self.assertIsNone(output['hd_asset_path']);self.assertEqual(output['status'],'pass')
            self.assertEqual(output['enhancement']['status'],'failed')

    def test_missing_weights_preserves_native_asset_and_reports_unavailable(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'asset.png';Image.new('RGBA',(16,16),'red').save(path)
            obj={'id':'x','asset_path':str(path),'status':'pass'}
            result=make_upscale_objects(UpscaleService(False),True)({'objects':[obj]})['objects'][0]
            self.assertEqual(result['asset_path'],str(path));self.assertIsNone(result['hd_asset_path'])
            self.assertEqual(result['texture_scale'],1);self.assertEqual(result['hd_qa']['status'],'unavailable')

    def test_hd_reviewer_failure_is_isolated_per_asset(self):
        class Broken(Review):
            def review_candidate(self,*args):
                raise Exception('vision endpoint offline')
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.png';hd=root/'hd.png'
            Image.new('RGBA',(16,16),'red').save(source);Image.new('RGBA',(64,64),'red').save(hd)
            service=UpscaleService(False)
            obj={'id':'x','asset_path':str(source),'status':'pass','category':'tree','bbox':{'x':0,'y':0,'w':16,'h':16}}
            with patch.object(service,'upscale',return_value=str(hd)):
                result=make_upscale_objects(service,True,Broken())({'objects':[obj]})['objects'][0]
            self.assertIsNone(result['hd_asset_path'])
            self.assertIn('vision endpoint offline',result['hd_qa']['reason'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional Torch is not installed')
    def test_rrdb_tiling_stitches_exact_coordinates(self):
        import torch
        service=UpscaleService(False,config=ModelConfig(upscale={'tile':7,'tile_pad':2}))
        class Repeat(torch.nn.Module):
            def forward(self,tensor):
                return tensor.repeat_interleave(4,2).repeat_interleave(4,3)
        service._model=Repeat();service._torch=torch;service._device='cpu'
        rng=np.random.default_rng(0);pixels=rng.integers(0,256,(19,23,3),dtype=np.uint8)
        restored=service._restore(Image.fromarray(pixels))
        np.testing.assert_array_equal(np.array(restored),pixels.repeat(4,0).repeat(4,1))
