"""Scene coverage review and bounded targeted detection before ownership rebuild."""
from pathlib import Path
from PIL import Image, ImageDraw
from qa.scene_qa import coverage_review
from retry.detection_retry import recover_regions
from services.detection_postprocess import iou


def make_p1_scene(config, detector, sam, reviewer):
    def p1_scene(state):
        records=list(state.get('objects',[])); history=list(state.get('retry_history',[]))
        root=Path(state['output_dir']); directory=root/'diagnostics'; directory.mkdir(parents=True,exist_ok=True)
        regions,coverage=coverage_review(state['source_path'],records,config.p1)
        reviews=[]
        fragments=[]
        for run in state.get('detection_runs', []):
            for candidate in run.get('review_candidate_pool', []):
                if not candidate.get('is_truncated'):
                    continue
                x,y,right,bottom = candidate['bbox']
                fragments.append({'category':candidate['category'],
                                  'approx_bbox':{'x':x,'y':y,'w':right-x,'h':bottom-y},
                                  'reason':'P0 fragment has no complete re-observation',
                                  'failure_type':'CROSS_TILE_FRAGMENT'})
        # The vision reviewer sees original pixels and current detections; its proposals remain untrusted until detection+mask QA.
        if reviewer.backend=='llm':
            try:
                metrics={'accepted_coverage':state.get('scene_coverage',0), 'target_coverage':config.scene_loop.target_coverage,
                         'image_size':[state['width'],state['height']], 'detections':[{'category':o['category'],'bbox':o['bbox']} for o in records],
                         'request':'Find missed objects. approx_bbox uses original image global integer x,y,w,h.'}
                decision=reviewer.review(state['source_path'],state.get('working_path',state['source_path']),metrics)
                reviews=[m.model_dump() for m in decision.missed_objects]
            except Exception as error:
                history.append({'retry_reason':'MISSED_DETECTION','action':'vision_review_failed','error':type(error).__name__,'improved':False})
        regions=[*reviews,*fragments,*regions][:config.p1.max_problem_regions]
        all_regions=list(regions)
        def remaining_vision_proposals():
            return [r for r in [*reviews,*fragments] if not any(
                o.get('status') == 'pass' and o['category'] == r['category'].lower().replace(' ', '_')
                and iou(o['bbox'], r['approx_bbox']) > .4 for o in records)]
        # Mock detections are geometric fixtures, so re-detecting arbitrary crops would fabricate objects.
        for scene_attempt in range(config.p1.max_scene_retry if not config.mock else 0):
            if not regions: break
            recovered,logs=recover_regions({**state,'objects':records,'retry_history':history},regions,detector,sam,config,
                                            scene_attempt=scene_attempt+1)
            history.extend({**log,'scene_attempt':scene_attempt+1} for log in logs)
            records.extend(recovered)
            if not recovered: break
            regions,coverage=coverage_review(state['source_path'],records,config.p1)
            regions=[*remaining_vision_proposals(),*regions][:config.p1.max_problem_regions]
            all_regions.extend(regions)
        regions,coverage=coverage_review(state['source_path'],records,config.p1)
        regions=[*remaining_vision_proposals(),*regions][:config.p1.max_problem_regions]
        coverage.save(directory/'coverage_map.png')
        coverage.save(directory/'p1_coverage_map.png')
        for name,items,color in [('qa_failed_regions',[o['bbox'] for o in records if o['status']!='pass'],(255,40,40)),
                                 ('retry_regions',[r['approx_bbox'] for r in all_regions],(255,180,30)),
                                 ('detection_boxes',[o['bbox'] for o in records],(30,220,60))]:
            with Image.open(state['source_path']) as image: canvas=image.convert('RGB')
            draw=ImageDraw.Draw(canvas)
            for b in items: draw.rectangle((b['x'],b['y'],b['x']+b['w']-1,b['y']+b['h']-1),outline=color,width=2)
            canvas.save(directory/f'{name}.png')
        all_detections = {item['id']: item for item in state.get('all_detections', [])}
        all_detections.update({item['id']: item for item in records})
        return {'objects':records, 'all_detections':list(all_detections.values()),
                'retry_history':history,'missed_object_candidates':regions,
                'detection_coverage_review':{'backend':reviewer.backend,'vision_proposals':reviews,
                                           'remaining_candidates':len(regions),'mock_local_detection_skipped':config.mock}}
    return p1_scene
