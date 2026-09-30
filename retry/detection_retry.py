"""Bounded problem-region detection with original-coordinate, unique instances."""
from pathlib import Path

from PIL import Image

from app.objects import segment_records, refine_records, crop_records
from qa.instance_qa import inspect_instance
from postprocess.truncation_detector import truncated_edges
from scene.element_classifier import SceneElementClassifier
from schemas.object import BBox, SceneObject
from services.detection_postprocess import iou


def region_key(region):
    """Stable budget key, shared across P1 scene review rounds."""
    box = region['approx_bbox']
    category = (region.get('category') or '').lower().replace(' ', '_')
    return [category, *(box[k] for k in ('x', 'y', 'w', 'h'))]


def recover_regions(state, regions, detector, sam, config, *, scene_attempt=1):
    root = Path(state['output_dir'])
    recovered, history = [], []
    with Image.open(state['source_path']) as image:
        source = image.convert('RGB')
    categories = list(dict.fromkeys(c for layer in state.get('layer_plan', []) for c in layer['categories']))
    known = state.get('objects', [])
    used_ids = {r['id'] for r in [*known, *state.get('all_detections', [])]}
    sequence = 0
    seen = set()
    for index, region in enumerate(regions[:config.p1.max_problem_regions]):
        key = region_key(region)
        if tuple(key) in seen:
            continue
        seen.add(tuple(key))
        previous = sum(log.get('region_key') == key for log in state.get('retry_history', [])
                       if log.get('action') == 'local_detection')
        for attempt in range(previous, config.p1.max_detection_retry):
            if len(known) + len(recovered) >= config.scene_loop.max_objects:
                return recovered, history
            log = {'retry_reason': region.get('failure_type', 'MISSED_DETECTION'),
                   'retry_count': attempt + 1, 'scene_attempt': scene_attempt,
                   'region_key': key, 'bbox': region['approx_bbox'], 'action': 'local_detection',
                   'previous_score': 0.0, 'new_score': 0.0, 'accepted_ids': [], 'rejected': []}
            try:
                box = BBox.model_validate(region['approx_bbox'])
                pad = 32 * (attempt + 1)
                x, y = max(0, box.x-pad), max(0, box.y-pad)
                right, bottom = min(source.width, box.x+box.w+pad), min(source.height, box.y+box.h+pad)
                if right <= x or bottom <= y:
                    raise ValueError('Problem region lies outside the original image')
                crop = source.crop((x, y, right, bottom))
                scale = 2 if max(crop.size) < 512 else 1
                crop = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.LANCZOS)
                path = root/'debug'/f'qa_detect_s{scene_attempt}_{index}_{attempt}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path)
                log['crop_path'] = str(path.resolve())
                prompts = [region['category']] if region.get('category') else categories
                for item in detector.detect(str(path), prompts):
                    if len(known) + len(recovered) >= config.scene_loop.max_objects:
                        break
                    local = BBox.model_validate(item['bbox'])
                    left, top = max(x, x+round(local.x/scale)), max(y, y+round(local.y/scale))
                    end_x = min(right, x+round((local.x+local.w)/scale))
                    end_y = min(bottom, y+round((local.y+local.h)/scale))
                    if end_x <= left or end_y <= top or item['confidence'] < config.qa.min_confidence:
                        continue
                    global_box = {'x': left, 'y': top, 'w': end_x-left, 'h': end_y-top}
                    edges = truncated_edges((left,top,end_x,end_y),(x,y,right,bottom),source.size,
                                            config.detection.truncation.edge_threshold)
                    if edges:
                        log['rejected'].append({'bbox':global_box,'failure_types':['CROSS_TILE_FRAGMENT'],
                                                'truncated_edges':edges})
                        continue
                    category = item['category'].lower().replace(' ', '_')
                    if any(o['category'] == category and iou(o['bbox'], global_box) > .4 for o in [*known, *recovered]):
                        continue
                    while True:
                        sequence += 1
                        ident = f'qa_s{scene_attempt:02d}_{sequence:05d}'
                        if ident not in used_ids:
                            used_ids.add(ident)
                            break
                    record = SceneObject(id=ident, category=category, bbox=global_box, confidence=item['confidence'],
                                         source='qa_local', detection_method='targeted_local', redetected=True).model_dump(mode='json')
                    record = SceneElementClassifier().classify(record, source.size)
                    candidates = segment_records(state['source_path'], [record], sam, root, p1=config.p1,
                                                 neighbors=[*known, *recovered])
                    candidates = refine_records(candidates, config.crop.alpha_threshold)
                    record = crop_records(state['source_path'], candidates, root, config.crop)[0]
                    metrics, failures = inspect_instance(record, state['source_path'], config)
                    if record.get('asset_path') and not record.get('error') and not failures:
                        record.update(status='pass', metrics=metrics, qa={
                            'status': 'pass', 'reason': 'Targeted detection and local mask QA passed',
                            'retry_strategy': 'none', 'failure_types': []})
                        recovered.append(record)
                        log['accepted_ids'].append(ident)
                        log['new_score'] = max(log['new_score'], metrics['quality_score'])
                    else:
                        log['rejected'].append({'id': ident, 'bbox': global_box, 'failure_types': failures,
                                                'error': record.get('error'), 'score': metrics['quality_score']})
            except (ValueError, RuntimeError, OSError) as error:
                log['error'] = type(error).__name__
            log['improved'] = bool(log['accepted_ids'])
            history.append(log)
            if log['improved']:
                break
    return recovered, history
