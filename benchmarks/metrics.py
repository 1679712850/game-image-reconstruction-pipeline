"""Category-aware confidence-ordered one-to-one matching on an annotated ROI."""
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from benchmarks.schema import Annotation, EvaluationConfig
from diagnostics.bbox_visualizer import xyxy


def intersection(a: tuple, b: tuple) -> float:
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(box: tuple) -> float:
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def box_iou(a: tuple, b: tuple) -> float:
    overlap = intersection(a, b)
    return overlap/max(1, area(a)+area(b)-overlap)


def roi_mask(path: Path, size: tuple[int, int], roi: tuple) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L')) > 8
    x,y,r,b = roi
    if mask.shape == (size[1], size[0]):
        return mask[y:b,x:r]
    if mask.shape != (b-y,r-x):
        raise ValueError(f'Mask {path} must be scene-sized or ROI-sized')
    return mask


def mask_metrics(gt: np.ndarray, pred: np.ndarray, tolerance: int = 2) -> dict[str, float]:
    common = int((gt & pred).sum()); n, p = int(gt.sum()), int(pred.sum())
    kernel = np.ones((3,3), np.uint8)
    def boundary(mask: np.ndarray) -> np.ndarray:
        return mask & ~cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    a,b = boundary(gt),boundary(pred)
    dilate = np.ones((2*tolerance+1,2*tolerance+1),np.uint8)
    recall = float((a & cv2.dilate(b.astype(np.uint8),dilate).astype(bool)).sum())/max(1,int(a.sum()))
    precision = float((b & cv2.dilate(a.astype(np.uint8),dilate).astype(bool)).sum())/max(1,int(b.sum()))
    return {'iou': common/max(1,n+p-common), 'dice': 2*common/max(1,n+p),
            'boundary_fscore': 2*recall*precision/max(1e-12,recall+precision),
            'completeness': common/max(1,n), 'contamination': (p-common)/max(1,p)}


def evaluate(annotation: Annotation, predictions: list[dict], image_size: tuple[int,int],
             annotation_root: Path, prediction_root: Path, config: EvaluationConfig) -> dict[str, Any]:
    roi = annotation.roi
    ignored = [*annotation.ignore_regions, *(o.bbox for o in annotation.objects if o.ignore)]
    gt = [o for o in annotation.objects if not o.ignore and o.kind == 'instance']
    categories = set(annotation.categories or [o.category for o in annotation.objects])
    selected = []
    ignored_ids = []
    for index, original in enumerate(predictions):
        box = xyxy(original)
        if not box or original.get('category') not in categories or original.get('kind') == 'terrain':
            continue
        # Center-in-ROI prevents partial outside observations being counted as FPs.
        cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
        if not (roi[0] <= cx < roi[2] and roi[1] <= cy < roi[3]):
            continue
        item = {**original, 'id': original.get('id', f'prediction_{index}'),
                'bbox': [max(roi[0],box[0]),max(roi[1],box[1]),min(roi[2],box[2]),min(roi[3],box[3])]}
        if any(intersection(item['bbox'], region)/max(1,area(item['bbox'])) >= .5 for region in ignored):
            ignored_ids.append(item['id']); continue
        selected.append(item)
    selected.sort(key=lambda p: -p.get('confidence',0))
    matches, duplicates, false_positives = [], [], []
    used = set()
    for pred in selected:
        eligible = sorted(((box_iou(pred['bbox'],g.bbox),g) for g in gt if g.category == pred['category']), key=lambda pair:-pair[0])
        match = next(((score,g) for score,g in eligible if score >= config.iou_threshold and g.id not in used),None)
        if match:
            score,g = match; used.add(g.id)
            matches.append({'gt_id':g.id,'prediction_id':pred['id'],'bbox_iou':score})
        elif eligible and eligible[0][0] >= config.iou_threshold:
            duplicates.append(pred['id'])
        else:
            false_positives.append(pred['id'])
    fragments = []
    for g in gt:
        pieces = [p for p in selected if p['category'] == g.category
                  and config.fragment_min_area <= area(p['bbox'])/area(g.bbox) < .8
                  and intersection(p['bbox'],g.bbox)/max(1,area(p['bbox'])) >= config.fragment_containment]
        # Distinct fragments, not repeated boxes or a full object plus a small duplicate.
        distinct = []
        for p in pieces:
            if all(box_iou(p['bbox'],q['bbox']) < .2 for q in distinct):
                distinct.append(p)
        if len(distinct) >= 2:
            fragments.append({'gt_id':g.id,'prediction_ids':[p['id'] for p in distinct]})
    missed = [g.id for g in gt if g.id not in used]
    masks = []
    evaluated_pixels = np.ones((roi[3]-roi[1],roi[2]-roi[0]),dtype=bool)
    for a,b,c,d in ignored:
        evaluated_pixels[max(0,b-roi[1]):max(0,d-roi[1]), max(0,a-roi[0]):max(0,c-roi[0])] = False
    pred_by_id = {p['id']:p for p in selected}
    gt_by_id = {g.id:g for g in gt}
    for match in matches:
        g,p = gt_by_id[match['gt_id']],pred_by_id[match['prediction_id']]
        path = p.get('visible_mask_path') or p.get('source_mask_path') or p.get('mask_path')
        if g.mask:
            target = roi_mask(annotation_root/g.mask,image_size,roi)
            actual = roi_mask(prediction_root/path,image_size,roi) if path else np.zeros_like(target)
            masks.append({**match, **mask_metrics(target & evaluated_pixels,actual & evaluated_pixels,config.boundary_tolerance), 'prediction_mask_missing':not bool(path)})
    semantic = []
    for g in annotation.objects:
        if g.ignore or g.kind != 'semantic_region' or not g.mask:
            continue
        target = roi_mask(annotation_root/g.mask,image_size,roi)
        union = np.zeros_like(target)
        for p in predictions:
            path = p.get('visible_mask_path') or p.get('mask_path')
            if path and p.get('category') == g.category:
                union |= roi_mask(prediction_root/path,image_size,roi)
        semantic.append({'gt_id':g.id, **mask_metrics(target & evaluated_pixels,union & evaluated_pixels,config.boundary_tolerance)})
    size_stats = {}
    for name in ('small','medium','large'):
        def size_class(g):
            ratio = area(g.bbox)/(image_size[0]*image_size[1])
            return 'small' if ratio < config.small_area_ratio else 'large' if ratio >= config.large_area_ratio else 'medium'
        group = [g for g in gt if size_class(g) == name]
        size_stats[name] = {'gt':len(group),'detected':sum(g.id in used for g in group),
                            'recall':sum(g.id in used for g in group)/len(group) if group else None}
    return {'detection': {'recall':len(matches)/len(gt) if gt else None,
                          'precision':len(matches)/len(selected) if selected else (0.0 if gt else None),
                          'duplicate_rate':len(duplicates)/max(1,len(selected)),
                          'fragmentation_rate':len(fragments)/max(1,len(gt)),
                          'miss_rate':len(missed)/max(1,len(gt)), 'gt_count':len(gt),
                          'prediction_count':len(selected),'true_positive_count':len(matches),
                          'false_positive_count':len(selected)-len(matches),'duplicate_count':len(duplicates),
                          'fragmented_gt_count':len(fragments),'by_size':size_stats},
            'mask_quality': {key:float(np.mean([m[key] for m in masks])) if masks else None
                             for key in ('iou','dice','boundary_fscore','completeness','contamination')},
            'matches':matches,'missed':missed,'false_positives':false_positives,'duplicates':duplicates,
            'fragments':fragments,'masks':masks,'semantic_regions':semantic,'ignored_predictions':ignored_ids,
            'predictions':selected,'poor_masks':[m['prediction_id'] for m in masks if m['iou'] < config.poor_mask_iou]}
