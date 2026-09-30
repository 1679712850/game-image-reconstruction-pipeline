"""Exclusive source-pixel ownership, with auditable conflict resolution."""
from pathlib import Path
import numpy as np
from PIL import Image
from cv.mask import read_mask, save_mask
from scene.element_classifier import TERRAIN
import json


def build_pixel_ownership(source_path, records, output_dir, alpha_threshold=8, preserve_residual=True):
    root = Path(output_dir)
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Pixel ownership requires unique instance IDs')
    for name in ('masks', 'terrain', 'metadata', 'diagnostics'):
        (root/name).mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgba = np.asarray(image.convert('RGBA'))
    h, w = rgba.shape[:2]
    eligible = rgba[:, :, 3] > alpha_threshold
    owner = np.zeros((h, w), np.int32)
    depth = np.zeros((h, w), np.uint16)
    table = [{'owner_id': 0, 'kind': 'unassigned', 'category': 'unassigned'}]
    masks, ids, groups = {}, {}, {}
    for record in records:
        path = record.get('candidate_mask_path') or record.get('mask_path')
        if not path:
            continue
        mask = read_mask(path) > alpha_threshold
        if mask.shape != owner.shape:
            raise ValueError('Ownership mask is not in original-image coordinates')
        mask &= eligible
        record['candidate_mask_path'] = path
        masks[record['id']] = mask
        key = ('terrain', TERRAIN.get(record['category'], record['category'])) if record.get('element_type') == 'terrain' else ('instance', record['id'])
        if key not in groups:
            groups[key] = len(table)
            table.append({'owner_id': len(table), 'kind': record.get('element_type', 'instance'),
                          'category': key[1] if key[0] == 'terrain' else record['category'], 'instance_ids': []})
        ids[record['id']] = groups[key]
        table[groups[key]]['instance_ids'].append(record['id'])
    # Explicit occlusion relationships supersede class order. Cycles are not guessed.
    pending = sorted([r for r in records if r['id'] in masks], key=lambda r: (
        r.get('ownership_priority', 0), r['bbox']['y']+r['bbox']['h'],
        r.get('confidence', 0), r.get('segmentation', {}).get('score', 0), r['id']))
    ordered, resolved, cycles = [], set(), []
    while pending:
        ready = next((r for r in pending if not (set(r.get('occludes', [])) & {p['id'] for p in pending})), None)
        if ready is None:
            cycles = [r['id'] for r in pending]
            ordered.extend(pending)
            break
        ordered.append(ready); resolved.add(ready['id']); pending.remove(ready)
    # Same-category terrain observations are one owner, not duplicate layers.
    owner_union = {}
    for record in ordered:
        oid = ids[record['id']]
        owner_union.setdefault(oid, np.zeros_like(eligible))
        owner_union[oid] |= masks[record['id']]
    for mask in owner_union.values():
        depth += mask
    conflicts = []
    for record in ordered:
        mask, oid = masks[record['id']], ids[record['id']]
        for old_id, count in zip(*np.unique(owner[mask & (owner > 0) & (owner != oid)], return_counts=True)):
            conflicts.append({'winner': oid, 'loser': int(old_id), 'pixels': int(count),
                              'reason': 'explicit_occlusion_then_class_priority_ground_y_confidence_mask_score_id'})
        owner[mask] = oid
    unassigned = eligible & (owner == 0)
    semantic_unassigned = int(unassigned.sum())
    residual_mask = save_mask(unassigned, root/'masks'/'unclassified_residual.png')
    residual_asset = None
    if preserve_residual and np.any(unassigned):
        oid = len(table)
        table.append({'owner_id': oid, 'kind': 'background', 'category': 'unclassified_residual', 'uncertain': True})
        owner[unassigned] = oid
        layer = rgba.copy(); layer[:, :, 3] = np.where(unassigned, rgba[:, :, 3], 0)
        residual_asset = str((root/'terrain'/'unclassified_residual.png').resolve())
        Image.fromarray(layer).save(residual_asset)
    terrain = []
    for item in table[1:]:
        visible = owner == item['owner_id']
        item['visible_pixel_count'] = int(visible.sum())
        item['mask_path'] = save_mask(visible, root/'masks'/f"owner_{item['owner_id']}.png")
        if item['kind'] == 'terrain':
            observations = [r for r in records if r['id'] in item['instance_ids']]
            item['qa_status'] = 'pass' if all(r.get('status') == 'pass' for r in observations) else 'manual_review'
            item['uncertain'] = any(r.get('uncertain',False) for r in observations)
            item['occluded_pixel_count'] = int((owner_union[item['owner_id']] & ~visible).sum())
            path = root/'terrain'/f"{item['category']}.png"
            layer = rgba.copy(); layer[:, :, 3] = np.where(visible, rgba[:, :, 3], 0)
            Image.fromarray(layer).save(path)
            terrain.append({**item, 'id': f"terrain_{item['category']}", 'asset_path': str(path.resolve()),
                            'visible_mask_path': item['mask_path'], 'complete_mask_path': None,
                            'completion_status': 'not_inferred', 'crop_bbox': {'x': 0, 'y': 0, 'w': w, 'h': h}, 'z_order': -1})
    for record in records:
        if record['id'] not in masks:
            continue
        oid, candidate = ids[record['id']], masks[record['id']]
        visible = (owner == oid) & candidate
        record['visible_mask_path'] = save_mask(visible, root/'masks'/f"{record['id']}_visible.png")
        # A SAM candidate is not an amodal completed object.
        record['full_mask_path'] = None
        record['visible_pixel_count'] = int(visible.sum())
        record['ownership_pixel_count'] = int(visible.sum())
        record['occluded_pixel_count'] = int((candidate & ~visible).sum())
    np.save(root/'metadata'/'pixel_owner.npy', owner, allow_pickle=False)
    palette = np.array([[(i*73)%256, (i*137)%256, (i*199)%256] for i in range(len(table))], np.uint8)
    Image.fromarray(palette[owner]).save(root/'diagnostics'/'ownership_map.png')
    heat = np.zeros((h, w, 3), np.uint8); heat[depth > 1] = (255, 30, 30)
    Image.fromarray(heat).save(root/'diagnostics'/'overlap_heatmap.png')
    save_mask(unassigned, root/'diagnostics'/'unassigned_regions.png')
    n = max(1, int(eligible.sum()))
    payload = {'coordinate_space': 'original_image_global', 'width': w, 'height': h,
               'owner_map_path': str((root/'metadata'/'pixel_owner.npy').resolve()),
               'owner_table': table, 'conflicts': conflicts, 'occlusion_cycles': cycles,
               'residual_background': {'kind': 'visible_unclassified_residual', 'mask_path': residual_mask,
                                       'asset_path': residual_asset, 'pixel_count': semantic_unassigned},
               'unassigned_ratio': semantic_unassigned/n, 'unassigned_pixel_count': semantic_unassigned,
               'coverage_ratio': 1-semantic_unassigned/n, 'visible_coverage_ratio': float((eligible & (owner > 0)).sum()/n),
               'overlap_pixel_count': int((depth > 1).sum()), 'overlap_ratio': float((depth > 1).sum()/n),
               'resolved_overlap_ratio': 0.0, 'eligible_pixels': int(eligible.sum()),
               'definition': 'unassigned/coverage measured before uncertain residual fallback; overlap measured before conflict resolution'}
    (root/'metadata'/'ownership.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
    return payload, table, terrain
