"""Failure-specific repairs with strict improvement gates and finite budgets."""
from pathlib import Path
from app.objects import crop_records, refine_records, segment_records
from retry.segmentation_retry import repair_record


def make_retry_objects(config, service, reviewer=None, detector=None):
    def retry_objects(state):
        count=state.get('retry_count',0)
        if count>=state.get('max_retry',config.max_retry): return {}
        failed=set(state.get('failed_objects',[])); history=list(state.get('retry_history',[])); output=[]
        for original in state['objects']:
            if original['id'] not in failed:
                output.append(original); continue
            if config.p1.enabled and not config.mock:
                attempts=sum(r.get('instance_id')==original['id'] for r in history)
                if attempts>=config.p1.max_segmentation_retry or (original.get('qa') or {}).get('status')=='manual_review':
                    output.append(original); continue
                if 'WRONG_CATEGORY' in ((original.get('qa') or {}).get('failure_types') or []):
                    from retry.classification_retry import reclassify_record
                    candidate,log=reclassify_record(original,state,reviewer,attempts+1)
                elif 'CROSS_TILE_FRAGMENT' in ((original.get('qa') or {}).get('failure_types') or []):
                    from retry.fragment_retry import repair_fragment
                    candidate,log=repair_fragment(original,state,detector,service,config,attempts+1)
                else:
                    candidate,log=repair_record(original,state,service,config,attempts+1)
            else:
                root=Path(state['output_dir']); version=f"_r{state.get('detection_round',1):02d}_retry{count+1}"
                repaired=segment_records(state['source_path'],[original],service,root,local=True,
                                         coverage_path=state.get('coverage_mask_path'),version=version,p1=config.p1,attempt=count+1)
                repaired=refine_records(repaired,config.crop.alpha_threshold)
                candidate=crop_records(state['source_path'],repaired,root,config.crop,version=version)[0]
                candidate['mock_retry_resolved']=config.mock and not candidate.get('error')
                log={'instance_id':original['id'],'retry_reason':((original.get('qa') or {}).get('failure_types') or ['BAD_MASK'])[0],
                     'retry_count':count+1,'previous_score':None,'new_score':None,'improved':candidate['mock_retry_resolved'],
                     'action':'mock_simulation' if config.mock else 'legacy_local_retry'}
            history.append(log)
            candidate={**candidate,'retry_history':[*original.get('retry_history',[]),log]}
            output.append(candidate)
        return {'objects':output,'retry_count':count+1,'retry_history':history}
    return retry_objects
