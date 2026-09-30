"""ROI-level source-visible coverage and production inventory normalization."""
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from benchmarks.metrics import roi_mask
from benchmarks.schema import Annotation


def prediction_inventory(manifest: dict[str, Any]) -> tuple[list[dict], str]:
    final = {o['id']: o for o in [*manifest.get('objects',[]), *manifest.get('environment_effects',[])]}
    inventory = manifest.get('detection',{}).get('instances')
    if inventory is None:
        return [*final.values(), *manifest.get('terrain_layers',[])], 'legacy_exported_assets_only'
    # Keep rejected/missing/occluded detections in the denominator. Add final masks/QA.
    predictions = []
    for original in inventory:
        asset = final.get(original['id'],{})
        predictions.append({**original, **asset, 'bbox':original['bbox']})
    predictions += [o for key,o in final.items() if key not in {p['id'] for p in predictions}]
    predictions += manifest.get('terrain_layers',[])
    return predictions, 'production_detection_inventory'


def roi_completion(source: Path, annotation: Annotation, manifest: dict, root: Path) -> dict[str, Any]:
    roi = annotation.roi
    with Image.open(source) as image:
        size = image.size
        eligible = np.asarray(image.convert('RGBA').crop(roi))[:,:,3] > 8
    for a,b,c,d in [*annotation.ignore_regions, *(o.bbox for o in annotation.objects if o.ignore)]:
        eligible[b-roi[1]:d-roi[1],a-roi[0]:c-roi[0]] = False
    instance = np.zeros_like(eligible)
    terrain = np.zeros_like(eligible)
    assets = [*manifest.get('objects',[]), *manifest.get('environment_effects',[])]
    for obj in assets:
        path = obj.get('visible_mask_path') or obj.get('source_mask_path')
        if obj.get('base_asset_status') == 'ready' and path:
            instance |= roi_mask(root/path,size,roi) & eligible
    for layer in manifest.get('terrain_layers',[]):
        path = layer.get('visible_mask_path') or layer.get('mask_path')
        if layer.get('qa_status') == 'pass' and not layer.get('uncertain') and path:
            terrain |= roi_mask(root/path,size,roi) & eligible
    info = manifest.get('ownership',{}).get('residual_background',{})
    residual = roi_mask(root/info['mask_path'],size,roi) & eligible if info.get('mask_path') and info.get('asset_path') else np.zeros_like(eligible)
    n = max(1,int(eligible.sum()))
    value = float((instance|terrain).sum()/n)
    return {'semantic_coverage':value,'unassigned_ratio':1-value,
            'residual_pixel_ratio':float(residual.sum()/n),
            'instance_pixel_ratio':float(instance.sum()/n),
            'background_pixel_ratio':float((terrain & ~instance).sum()/n),
            'definition':'QA-passed source-visible pixels inside ROI, excluding ignore regions. Residual is a subset of unassigned, not additive. Not GT pixel correctness.'}


def truncation(manifest: dict, root: Path) -> dict[str, Any]:
    known = manifest.get('detection',{}).get('object_limit')
    if known:
        return known
    # Earlier production manifests omit limits, but their run evidence may preserve drops.
    candidates = [root/'debug'/'run.json', root/'diagnostics'/'summary.json']
    for path in candidates:
        if path.name == 'run.json' and path.exists():
            import json
            runs = json.loads(path.read_text()).get('state',{}).get('detection_runs',[])
            if runs:
                drops = sum(f.get('reason') == 'max_objects' for r in runs for f in r.get('filtered',[]))
                kept = sum(r.get('selected',r.get('after_filter',0)) for r in runs)
                return {'truncated':bool(drops),'objects_before_limit':kept+drops,'objects_after_limit':kept,
                        'provenance':'legacy_debug_run'}
    return {'truncated':None,'objects_before_limit':None,'objects_after_limit':None,'provenance':'unknown'}
