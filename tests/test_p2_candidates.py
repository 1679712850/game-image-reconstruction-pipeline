"""P2 integration: selected pixels, regenerated alpha, retries, fallback and PSD."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import struct
import unittest
import numpy as np
from PIL import Image, ImageDraw

from app.config import PipelineConfig
from candidates.pipeline import run_candidates
from candidates.qa import select, retry_prompt
from cv.reconstruct import reconstruct_scene
from exporters.psd_exporter import export_layers
from schemas.candidate import Candidate, CandidateQA, CandidateRegistry


def scores(value=.95, **overrides):
    return {**dict.fromkeys(['shape','style','perspective','scale','color','lighting','edge',
                            'occlusion_reconstruction_quality','semantic'], value), 'background_leak': 0., **overrides}


class Review:
    def __init__(self, decisions=None):
        self.decisions = decisions or [CandidateQA(status='ACCEPT', scores=scores(), evaluator='test_vision')]
        self.calls = 0

    def review_candidate(self, *args):
        value = self.decisions[min(self.calls, len(self.decisions)-1)]
        self.calls += 1
        return value


class Generator:
    mock = False
    def __init__(self, fail=False):
        self.prompts = []; self.fail = fail

    def generate_object(self, source, *, prompt, output_path):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError('CUDA out of memory')
        image = Image.new('RGBA', (100,100))
        ImageDraw.Draw(image).rectangle((40,25,59,74), fill=(240,15,25,255))
        path = Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); image.save(path)
        return str(path)

    def complete_object(self, source, mask, *, prompt, output_path):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError('generation unavailable')
        with Image.open(source) as original:
            image=original.copy()
        image.paste((240,15,25), (0,0,image.width,image.height))
        image.putalpha(Image.open(source).getchannel('A'))
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); image.save(path)
        return str(path)


class CandidateIntegration(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name)
        self.original=self.root/'original.png'; self.mask=self.root/'mask.png'
        image=Image.new('RGBA',(20,30)); ImageDraw.Draw(image).rectangle((3,2,16,29),fill=(20,150,40,255)); image.save(self.original)
        Image.new('L',(20,30),255).save(self.mask)
        self.obj={'id':'tree_001','category':'tree','confidence':.95,'status':'pass',
                  'bbox':{'x':10,'y':20,'w':20,'h':30},'crop_bbox':{'x':10,'y':20,'w':20,'h':30},
                  'asset_path':str(self.original),'mask_path':str(self.mask),'requires_inpainting':False}
        self.state={'output_dir':str(self.root),'source_path':str(self.original),'objects':[self.obj], 'width':80,'height':80}
        self.config=PipelineConfig(object_completion={'enabled':True},resources={'max_generation_retry':2})

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_generated_pixels_flow_to_scene_png_and_psd(self):
        self.obj.update(occluded_pixel_count=80,visible_pixel_count=20,status='manual_review')
        result=run_candidates(self.state,self.config,Generator(),reviewer=Review())
        obj=result['objects'][0]; registry=result['candidate_registry']['tree_001']
        self.assertEqual(obj['accepted_candidate_id'],'generation_v1')
        self.assertEqual(obj['asset_path'],obj['accepted_asset'])
        self.assertNotEqual(obj['mask_path'],self.obj['mask_path'])
        self.assertEqual(obj['placement']['anchor'],registry['source']['placement']['anchor'])
        with Image.open(obj['accepted_asset']) as image:
            self.assertLess(image.width,100); self.assertLess(image.height,100)
            self.assertEqual(image.getchannel('A').getbbox(),(0,0,image.width,image.height))
            expected=np.asarray(image).copy()
        path=reconstruct_scene(80,80,[obj],self.root/'scene.png')
        box=obj['crop_bbox']
        with Image.open(path) as image:
            actual=np.asarray(image)[box['y']:box['y']+box['h'],box['x']:box['x']+box['w']]
        np.testing.assert_array_equal(actual,expected)
        psd=export_layers([obj],80,80,self.root/'scene.psd')
        with Image.open(psd) as image:
            np.testing.assert_array_equal(np.asarray(image.convert('RGB'))[box['y']:box['y']+box['h'],box['x']:box['x']+box['w']],expected[:,:,:3])
        raw=Path(psd).read_bytes(); self.assertEqual(raw[:4],b'8BPS')
        self.assertEqual(struct.unpack('>h',raw[42:44])[0],-1)
        self.assertTrue((self.root/'debug/candidates/tree_001_contact_sheet.png').exists())
        self.assertEqual(json.loads((self.root/'metadata/candidates.json').read_text())['tree_001']['accepted_candidate_id'],'generation_v1')

    def test_retry_changes_prompt_and_later_reject_does_not_win(self):
        self.obj['status']='manual_review'; self.obj['requires_inpainting']=True
        generator=Generator()
        review=Review([CandidateQA(status='RETRY',scores=scores(perspective=.6),reasons=['perspective mismatch']),
                       CandidateQA(status='ACCEPT',scores=scores())])
        result=run_candidates(self.state,self.config,generator,reviewer=review)
        self.assertEqual(len(generator.prompts),2)
        self.assertNotEqual(*generator.prompts)
        self.assertIn('Do not change viewpoint',generator.prompts[1])
        self.assertEqual(result['objects'][0]['accepted_candidate_id'],'inpaint_v2')
        self.assertEqual(result['objects'][0]['provenance']['retry_count'],1)

    def test_all_generation_failures_preserve_source_and_continue_other_objects(self):
        self.obj['requires_inpainting']=True
        other={**self.obj,'id':'rock_001','requires_inpainting':False}
        self.state['objects'].append(other)
        result=run_candidates(self.state,self.config,Generator(fail=True),reviewer=Review())
        first,second=result['objects']
        self.assertEqual(first['accepted_asset'],str(self.original))
        self.assertTrue(first['needs_manual_review']); self.assertFalse(second['needs_manual_review'])
        self.assertEqual(result['failed_objects'],['tree_001'])

    def test_clean_source_skips_generation_and_unknown_qa_does_not_accept(self):
        generator=Generator()
        result=run_candidates(self.state,self.config,generator,reviewer=None)
        self.assertFalse(generator.prompts)
        self.assertEqual(result['objects'][0]['accepted_candidate_id'],'original')
        self.obj['requires_inpainting']=True
        result=run_candidates(self.state,self.config,generator,reviewer=None)
        self.assertEqual(result['objects'][0]['accepted_candidate_id'],'original')
        self.assertTrue(result['objects'][0]['needs_manual_review'])

    def test_ranking_and_fallback_never_select_last_or_rejected(self):
        registry=CandidateRegistry(instance_id='tree',source={})
        registry.candidates=[Candidate(candidate_id=ident,type=kind,image_path=ident,qa=CandidateQA(status=status,overall=score))
            for ident,kind,status,score in [('original','segmentation','ACCEPT',.8),('best','generation','ACCEPT',.96),('last','generation','ACCEPT',.85),('wrong','generation','REJECT',1.)]]
        self.assertEqual(select(registry)[0].candidate_id,'best')
        registry.candidates=[c.model_copy(update={'qa':c.qa.model_copy(update={'status':'REJECT'})}) for c in registry.candidates[1:]]+[registry.candidates[0]]
        self.assertEqual(select(registry)[0].candidate_id,'original')

    def test_occlusion_graph_overrides_y_order_in_png_and_psd(self):
        a=self.root/'a.png'; b=self.root/'b.png'
        Image.new('RGBA',(5,5),'red').save(a); Image.new('RGBA',(5,5),'blue').save(b)
        box={'x':0,'y':0,'w':5,'h':5}
        records=[{'id':'front','accepted_asset':str(a),'crop_bbox':box,'z_order':0,'occludes':['back']},
                 {'id':'back','accepted_asset':str(b),'crop_bbox':box,'z_order':100}]
        path=reconstruct_scene(5,5,records,self.root/'occlusion.png')
        self.assertEqual(Image.open(path).getpixel((1,1)),(255,0,0,255))
        path=export_layers(records,5,5,self.root/'occlusion.psd')
        with Image.open(path) as image:
            self.assertEqual(image.convert('RGB').getpixel((1,1)),(255,0,0))

    def test_layered_candidates_join_registry_and_are_selected_after_qa(self):
        layer=self.root/'layer.png'; image=Image.new('RGBA',(80,80))
        ImageDraw.Draw(image).rectangle((12,22,26,48),fill=(220,20,30,255));image.save(layer)
        self.state['decomposed_layers']=[{'asset_path':str(layer),'mock':False}]
        config=PipelineConfig(object_completion={'enabled':False})
        result=run_candidates(self.state,config,Generator(),reviewer=Review())
        self.assertEqual(result['objects'][0]['accepted_candidate_id'],'layered_0_v1')
        self.assertEqual(len(result['candidate_registry']['tree_001']['candidates']),2)

    def test_graph_accepted_edit_reaches_manifest_upscale_and_scene(self):
        from dataclasses import replace
        from langgraph.checkpoint.memory import InMemorySaver
        from agent.graph import build_graph
        from services.runtime import ServiceBundle
        from services.image_edit_service import ImageEditService
        from services.scene_review_service import SceneReviewService
        class Edit(ImageEditService):
            def complete_object(self,*args,**kwargs):
                return Generator().complete_object(*args,**kwargs)
        class Vision(SceneReviewService):
            def review_candidate(self,*args,**kwargs):
                return CandidateQA(status='ACCEPT',scores=scores(),evaluator='test_vision')
        source=self.root/'scene_source.png';Image.new('RGB',(80,80),(20,150,40)).save(source)
        config=PipelineConfig(object_completion={'enabled':True},candidates={'automatic':False},
                              p1={'enabled':False},scene_loop={'enabled':False},detection={'diagnostics':{'enabled':False}})
        services=replace(ServiceBundle.create(),image_edit=Edit(),reviewer=Vision())
        graph=build_graph(config,services,checkpointer=InMemorySaver(),interrupt_before=['complete_objects'])
        run={'configurable':{'thread_id':'p2-accepted'}}
        paused=graph.invoke({'source_path':str(source),'output_dir':str(self.root/'graph')},run)
        obj=paused['objects'][0];mask=self.root/'request.png'
        Image.new('L',(obj['crop_bbox']['w'],obj['crop_bbox']['h']),255).save(mask)
        graph.update_state(run,{'edit_requests':[{'object_id':obj['id'],'mask_path':str(mask),'prompt':'Repair tree'}]})
        result=graph.invoke(None,run)
        accepted=result['objects'][0]
        self.assertEqual(accepted['accepted_candidate_id'],'inpaint_v1')
        manifest=json.loads(Path(result['scene_json']).read_text())
        exported=next(o for o in manifest['objects'] if o['id']==obj['id'])
        self.assertEqual(exported['asset'],exported['accepted_asset'])
        with Image.open(accepted['hd_asset_path']) as image:
            active=np.asarray(image)[:,:,3]>0
            self.assertGreater(np.asarray(image)[:,:,0][active].mean(),200)
        box=accepted['crop_bbox']
        with Image.open(result['reconstruction_path']) as image:
            self.assertGreater(image.getpixel((box['x']+box['w']//2,box['y']+box['h']//2))[0],200)

    def test_existing_scene_transform_is_inherited(self):
        from candidates.geometry import source_placement
        obj={**self.obj, 'z_order':18, 'placement':{'anchor':[20,49.5], 'center':[20,35],
             'scale':1.25,'rotation':15,'depth':.7}}
        placement=source_placement(obj)
        for field,value in obj['placement'].items():
            self.assertEqual(placement[field],value)
        self.assertEqual(placement['z_order'],18)
