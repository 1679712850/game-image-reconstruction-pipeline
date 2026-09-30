"""Source-visible semantic coverage; residual and inferred pixels never increase it."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from app.asset_status import base_status
from cv.mask import read_mask


def completion_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    owner = state.get('ownership') or {}
    objects = list(state.get('objects') or [])
    n = max(1,int(owner.get('eligible_pixels',state.get('width',0)*state.get('height',0))))
    statuses = [o.get('base_asset_status') or base_status(o) for o in objects]
    counts = {s:statuses.count(s) for s in ('ready','needs_review','rejected')}
    tables = owner.get('owner_table',[])
    instance = sum(t.get('visible_pixel_count',0) for t in tables if t.get('kind') in {'instance','hybrid','effect'})/n
    terrain = sum(t.get('visible_pixel_count',0) for t in tables if t.get('kind') == 'terrain')/n
    predicted = float(owner.get('coverage_ratio',1-float(owner['unassigned_ratio']) if 'unassigned_ratio' in owner else state.get('scene_coverage',0)))
    semantic = predicted
    # QA-passed coverage is measured from SOURCE masks, not generated amodal alpha.
    if state.get('source_path') and Path(state['source_path']).exists():
        with Image.open(state['source_path']) as image:
            eligible = np.asarray(image.convert('RGBA'))[:,:,3] > 8
        union = np.zeros_like(eligible)
        instance_union = np.zeros_like(eligible)
        for o in objects:
            path = o.get('visible_mask_path') or o.get('source_mask_path') or o.get('mask_path')
            if (o.get('base_asset_status') or base_status(o)) == 'ready' and path:
                mask = read_mask(path) > 8
                if mask.shape == eligible.shape:
                    instance_union |= mask & eligible
        union |= instance_union
        for layer in state.get('terrain_layers',[]):
            path = layer.get('visible_mask_path') or layer.get('mask_path')
            if layer.get('qa_status') == 'pass' and not layer.get('uncertain') and path:
                mask = read_mask(path) > 8
                if mask.shape == union.shape:
                    union |= mask & eligible
        n = max(1,int(eligible.sum()))
        semantic = float(union.sum()/n)
        instance = float(instance_union.sum()/n)
        terrain = float((union & ~instance_union).sum()/n)
    residual_info = owner.get('residual_background') or {}
    residual = residual_info.get('pixel_count',0)/n if residual_info.get('asset_path') else 0.0
    runs = state.get('detection_runs',[])
    observed = sum(r.get('combined_candidates',0) for r in runs)
    merged = sum(sum('duplicate' in f.get('reason','') for f in r.get('filtered',[])) for r in runs)
    quality = {'duplicate_rate':None,'fragmentation_rate':None,'mean_mask_iou':None,
               'candidate_dedup_ratio':merged/max(1,observed)}
    required_review = sum(bool(o.get('review',{}).get('required')) for o in objects)
    status = 'partial' if semantic < 1-state.get('p1_thresholds',{}).get('unassigned',.01) else 'completed_with_review' if required_review or counts['needs_review'] or counts['rejected'] else 'completed'
    if state.get('resource_failures') and not objects:
        status = 'failed'
    return {'status':status,
            'scene':{'semantic_coverage':semantic,'predicted_semantic_coverage':predicted,
                     'unassigned_ratio':1-semantic,'unassigned_pixel_ratio':1-semantic,
                     'residual_pixel_ratio':residual,'semantic_pixel_ratio':semantic,
                     'background_pixel_ratio':terrain,'instance_pixel_ratio':instance},
            'objects':{'detected':len(state.get('all_detections',objects)), 'evaluated_assets':len(objects),
                       'accepted':counts['ready'],'asset_ready':counts['ready'],
                       'needs_review':counts['needs_review'],'rejected':counts['rejected'],
                       'asset_ready_ratio':counts['ready']/max(1,len(objects)),
                       'review_ratio':counts['needs_review']/max(1,len(objects)),
                       'failed_ratio':counts['rejected']/max(1,len(objects)), 'delivery_review':required_review},
            'quality':quality, 'reconstruction':{'similarity':state.get('reconstruction_score')},
            'definitions':{'semantic_coverage':'Union of QA-passed source-visible instance/terrain masks over nontransparent source pixels; predicted quality, not GT recall.',
                           'predicted_semantic_coverage':'All assigned source masks including review candidates; excludes residual.',
                           'unassigned_ratio':'Pixels without QA-passed semantics, including residual. Not additive with residual.',
                           'background_pixel_ratio':'QA-passed visible terrain pixels; excludes residual and generated completion.',
                           'asset_ready':'Persisted base cutout and segmentation, QA pass, no unresolved content/boundary review; optional HD excluded.',
                           'quality':'GT recall, precision, final duplicate/fragmentation/mask quality are unavailable until Benchmark evaluation.',
                           'reconstruction':'Pixel fidelity only; residual may produce 1.0 with low semantic coverage.'}}
