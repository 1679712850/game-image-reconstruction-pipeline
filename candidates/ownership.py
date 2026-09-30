"""Final layer attribution from accepted alpha, separate from source-pixel diagnostics."""
from pathlib import Path
import numpy as np
from PIL import Image
from cv.layer_order import ordered_layers


def accepted_ownership(objects, root, size):
    owner = np.zeros((size[1], size[0]), dtype=np.int32)
    indices = {o['id']: i+1 for i,o in enumerate(objects)}
    for obj in ordered_layers(objects):
        if not obj.get('asset_mask_path'):
            continue
        with Image.open(obj['asset_mask_path']) as image:
            active = np.asarray(image) > 8
        owner[active] = indices[obj['id']]
    path = Path(root)/'metadata'/'accepted_owner.npy'
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, owner, allow_pickle=False)
    for obj in objects:
        if not obj.get('asset_mask_path'):
            continue
        visible = (owner == indices[obj['id']]).astype(np.uint8)*255
        path_mask = Path(root)/'accepted_masks'/f"{obj['id']}_visible.png"
        Image.fromarray(visible).save(path_mask)
        obj['visible_mask_path'] = str(path_mask.resolve())
        with Image.open(obj['asset_mask_path']) as image:
            full_pixels = int((np.asarray(image) > 8).sum())
        obj['visible_pixel_count'] = int((visible > 0).sum())
        obj['ownership_pixel_count'] = obj['visible_pixel_count']
        obj['occluded_pixel_count'] = max(0, full_pixels-obj['visible_pixel_count'])
    return {'owner_map_path': str(path.resolve()), 'owner_ids': indices,
            'definition': 'Accepted asset alpha with explicit occlusion edges then source depth; separate from source segmentation coverage.'}
