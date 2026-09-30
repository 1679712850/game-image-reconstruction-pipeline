"""Recover usable source evidence without manufacturing a segmentation mask."""
from pathlib import Path
import numpy as np
from PIL import Image
from cv.bbox import get_tight_bbox
from cv.crop import crop_rgba_by_mask
from cv.mask import read_mask


def recover_source(obj, state, enabled):
    obj = dict(obj)
    if obj.get('asset_path') and Path(obj['asset_path']).is_file():
        return obj
    obj['asset_path'] = None
    # Restore the pre-ownership segmented object only when generation is enabled.
    if enabled and obj.get('source_asset_path') and Path(obj['source_asset_path']).is_file():
        obj.update(asset_path=obj['source_asset_path'], crop_bbox=obj['source_crop_bbox'],
                   mask_path=obj.get('source_mask_path'), pivot=None)
        return obj
    mask_path = obj.get('mask_path')
    if mask_path and Path(mask_path).is_file():
        mask = read_mask(mask_path)
        if mask.shape == (state['height'], state['width']):
            box = get_tight_bbox(mask, 8, 0)
            if box:
                path = Path(state['output_dir'])/'candidates'/obj['id']/'source_recovered.png'
                with Image.open(state['source_path']) as image:
                    crop_rgba_by_mask(image.convert('RGBA'), mask, box, path, threshold=8)
                obj.update(asset_path=str(path.resolve()), crop_bbox=box, pivot=None,
                           status='manual_review', needs_manual_review=True)
    return obj
