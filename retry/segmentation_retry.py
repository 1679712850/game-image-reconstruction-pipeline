"""Prepare failure-specific prompts and accept only objectively better masks."""
from pathlib import Path
import numpy as np
from cv.mask import read_mask
from app.objects import segment_records, refine_records, crop_records
from qa.instance_qa import inspect_instance


def repair_record(record, state, service, config, attempt):
    old_metrics, failures = inspect_instance(record, state['source_path'], config)
    failure = ((record.get('qa') or {}).get('failure_types') or failures or ['BAD_MASK'])[0]
    log = {'instance_id':record['id'], 'retry_reason':failure, 'retry_count':attempt,
           'previous_score':old_metrics['quality_score'], 'new_score':old_metrics['quality_score'], 'improved':False}
    if failure in {'WRONG_CATEGORY','DUPLICATE_INSTANCE'}:
        log['action']='manual_review_requires_classification' if failure=='WRONG_CATEGORY' else 'deduplicate_in_ownership'
        return record,log
    candidate=dict(record)
    if failure in {'BACKGROUND_LEAK','MASK_TOO_LARGE','PIXEL_CONFLICT'}:
        negative=list(candidate.get('negative_points',[]))
        for other in state.get('objects',[]):
            if other['id']==record['id'] or not other.get('mask_path'): continue
            if other.get('element_type')=='terrain' or other.get('ownership_priority',0)>record.get('ownership_priority',0):
                ys,xs=np.where(read_mask(other['mask_path'])>8)
                step=max(1,len(xs)//16)
                negative.extend([[float(x),float(y)] for x,y in zip(xs[::step][:16],ys[::step][:16])])
        candidate['negative_points']=negative
    root=Path(state['output_dir'])
    version=f"_r{state.get('detection_round',1)}_repair{attempt}"
    new=segment_records(state['source_path'],[candidate],service,root,p1=config.p1,attempt=attempt,version=version,
                        neighbors=state.get('objects', []))
    new=refine_records(new,config.crop.alpha_threshold)
    new=crop_records(state['source_path'],new,root,config.crop,version=version)[0]
    metrics,new_failures=inspect_instance(new,state['source_path'],config)
    log.update(new_score=metrics['quality_score'],action='local_crop_upscale_alternative_prompts',
               improved=bool(new.get('asset_path') and not new.get('error') and
                             metrics['quality_score']>old_metrics['quality_score']+1e-6 and len(new_failures)<=len(failures)))
    return (new if log['improved'] else record),log
