"""Finalize disjoint assets while preserving previous QA and candidate evidence."""
from pathlib import Path
import numpy as np
from app.objects import crop_records
from cv.mask import read_mask
from ownership.pixel_owner import build_pixel_ownership


def make_assign_ownership(config):
    def assign_ownership(state):
        records = [dict(obj) for obj in state.get('objects', [])]
        for record in records:
            record.update(source_asset_path=record.get('asset_path'),
                          source_mask_path=record.get('mask_path'), source_crop_bbox=record.get('crop_bbox'))
        payload, _, terrain = build_pixel_ownership(state['source_path'], records, state['output_dir'],
                                                   config.crop.alpha_threshold, config.p1.preserve_residual_background)
        if config.p1.terrain_completion:
            from scene.terrain_manager import complete_terrain
            terrain = complete_terrain(state['source_path'], terrain, state['output_dir'])
        objects = []
        for record in records:
            if record.get('element_type') == 'terrain':
                continue
            path = record.get('visible_mask_path')
            if path and record.get('visible_pixel_count') == 0 and record.get('occluded_pixel_count',0) > 0:
                # A correctly hidden object is not a failed/empty SAM mask.
                record.update(mask_path=path, asset_path=None, hd_asset_path=None, crop_bbox=None,
                              logical_size=None, texture_size=None, pivot=None, error=None)
                record['metrics'] = {**record.get('metrics', {}), 'fully_occluded': True}
            elif path and not np.array_equal(read_mask(path), read_mask(record['mask_path'])):
                status = record['status']
                record['mask_path'] = path
                record = crop_records(state['source_path'], [record], Path(state['output_dir']), config.crop, version='_visible')[0]
                record['status'] = status if not record.get('error') else 'manual_review'
            elif path:
                record['mask_path'] = path
            objects.append(record)
        return {'objects': objects, 'ownership': payload, 'terrain_layers': terrain,
                'p1_thresholds': {'unassigned': config.p1.unassigned_threshold, 'overlap': config.p1.overlap_threshold},
                'failed_objects': [o['id'] for o in objects if o['status'] != 'pass']}
    return assign_ownership
