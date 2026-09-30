"""Clear artificial tile truncation only after a complete local re-observation."""
from pathlib import Path

from PIL import Image

from app.objects import crop_records, refine_records, segment_records
from detection.candidate_adapter import observation_candidate
from fusion.cross_tile_dedup import appearance_similarity, geometry
from qa.instance_qa import inspect_instance
from segmentation.local_refiner import local_windows


def repair_fragment(record, state, detector, sam, config, attempt):
    old_metrics, failures = inspect_instance(record, state['source_path'], config)
    log = {'instance_id':record['id'], 'retry_reason':'CROSS_TILE_FRAGMENT',
           'retry_count':attempt, 'previous_score':old_metrics['quality_score'],
           'new_score':old_metrics['quality_score'], 'improved':False,
           'action':'expanded_local_detection'}
    if detector is None or attempt > config.p1.max_detection_retry:
        log['action'] = 'fragment_detection_unavailable' if detector is None else 'fragment_detection_budget_exhausted'
        return record, log
    with Image.open(state['source_path']) as image:
        source = image.convert('RGB')
    root = Path(state['output_dir'])
    category = record['category']
    context = {'window':(0,0,source.width,source.height), 'source':'global', 'group':[category]}
    parent, error = observation_candidate(record,context,source,record['id'],config.detection,'qa_fragment',[category])
    if error:
        log['error'] = error['reason']
        return record, log
    best = record
    windows = local_windows(record['bbox'],source.size,config.p1.max_local_crop_size,attempt)
    for index, window in enumerate(windows):
        crop = source.crop(window)
        scale = 2 if max(crop.size) < 512 else 1
        crop = crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)
        path = root/'debug'/f"{record['id']}_fragment_{attempt}_{index}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        crop.save(path)
        log['crop_path'] = str(path.resolve())
        try:
            for serial, item in enumerate(detector.detect(str(path),[category])):
                box = item['bbox']
                # Detector coordinates refer to the upscaled crop exactly once.
                local = {k:box[k]/scale for k in ('x','y','w','h')}
                context = {'window':window,'source':'redetection','group':[category],
                           'tile_id':f'qa_fragment_{attempt}_{index}','parent_id':record['id']}
                ident = f"{record['id']}_fragment_{attempt}_{index}_{serial}"
                child, invalid = observation_candidate({**item,'bbox':local},context,source,ident,
                                                       config.detection,'qa_fragment',[category])
                if invalid or child.is_truncated or child.category != category:
                    continue
                _, overlap, distance, ratio = geometry(parent,child)
                similarity = appearance_similarity(parent,child)
                if (overlap < config.detection.dedup.overlap_ratio or distance > 1.25 or ratio < .1
                        or similarity is None or similarity < config.detection.dedup.appearance_similarity
                        or child.confidence < config.qa.min_confidence):
                    continue
                candidate = {**record,'bbox':child.as_bbox(),'is_truncated':False,'truncated_edges':[],
                             'redetected':True,'confidence':child.confidence,
                             'source_candidates':[*record.get('source_candidates',[]),ident],
                             'observations':[*record.get('observations',[]),*child.observations]}
                version = f'_fragment_{attempt}_{index}_{serial}'
                candidates = segment_records(state['source_path'],[candidate],sam,root,p1=config.p1,
                                             attempt=attempt,version=version,neighbors=state.get('objects',[]))
                candidates = refine_records(candidates,config.crop.alpha_threshold)
                candidate = crop_records(state['source_path'],candidates,root,config.crop,version=version)[0]
                metrics, new_failures = inspect_instance(candidate,state['source_path'],config)
                if (candidate.get('asset_path') and not candidate.get('error')
                        and metrics['quality_score'] > log['new_score'] + 1e-6
                        and len(new_failures) < len(failures)):
                    best = candidate
                    log.update(improved=True,new_score=metrics['quality_score'],
                               global_bbox=child.as_bbox(),overlap=overlap,appearance_similarity=similarity)
        except (ValueError,RuntimeError,OSError) as error:
            log['error'] = type(error).__name__
    return best, log
