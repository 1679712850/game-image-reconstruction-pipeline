"""Normalize generated alpha and restore the source ground contact in scene space."""
from pathlib import Path
from PIL import Image
from cv.pivot import ground_pivot
import numpy as np


def source_placement(obj):
    box = obj.get('crop_bbox') or obj['bbox']
    pivot = obj.get('pivot')
    if not pivot and obj.get('asset_path'):
        with Image.open(obj['asset_path']) as image:
            pivot = ground_pivot(np.array(image.convert('RGBA').getchannel('A')))
    pivot = pivot or {'x': box['w'] / 2, 'y': box['h']}
    anchor = [box['x'] + pivot['x'], box['y'] + pivot['y']]
    existing = obj.get('placement') or {}
    return {'bbox_original': dict(obj['bbox']), 'crop_bbox_original': dict(box),
            'center': existing.get('center', [box['x'] + box['w']/2, box['y'] + box['h']/2]),
            'anchor': existing.get('anchor', anchor), 'scale': existing.get('scale', 1.0), 'rotation': existing.get('rotation', 0),
            'z_order': existing.get('z_order', obj['z_order'] if obj.get('pivot') or obj.get('z_order', 0) != 0 else anchor[1]),
            'depth': obj.get('placement', {}).get('depth', anchor[1]),
            'mask_position': [box['x'], box['y']], 'crop_bbox': dict(box), 'pivot': pivot}


def normalize(path, original, placement, target, scene_size, *, aligned=False):
    with Image.open(path) as image:
        image = image.convert('RGBA')
    tight = image.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    if tight is None:
        raise ValueError('empty generated alpha')
    image = image.crop(tight)
    with Image.open(original) as reference:
        ref_box = reference.convert('RGBA').getchannel('A').getbbox()
        rw, rh = (ref_box[2]-ref_box[0], ref_box[3]-ref_box[1]) if ref_box else reference.size
    if aligned:
        # Local inpainting is already in source-crop space. Cropping its alpha must not
        # resize or translate the surviving source pixels.
        factor = 1.0
        x, y = placement['crop_bbox']['x']+tight[0], placement['crop_bbox']['y']+tight[1]
    else:
        target_box = placement['bbox_original']
        rw, rh = max(rw, target_box['w']), max(rh, target_box['h'])
        factor = min(rw/image.width, rh/image.height)
        image = image.resize((max(1, round(image.width*factor)), max(1, round(image.height*factor))), Image.Resampling.LANCZOS)
        pivot = ground_pivot(np.array(image.getchannel('A')))
        x, y = (round(placement['anchor'][0]-pivot['x']), round(placement['anchor'][1]-pivot['y']))
    # Clip only at the original scene boundary; never move the anchor to fit.
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(image.width, scene_size[0]-x), min(image.height, scene_size[1]-y)
    if right <= left or bottom <= top:
        raise ValueError('generated placement outside scene')
    image = image.crop((left, top, right, bottom)); x += left; y += top
    box = {'x': x, 'y': y, 'w': image.width, 'h': image.height}
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); image.save(target)
    mask = target.with_name(target.stem+'_mask.png'); image.getchannel('A').save(mask)
    placed = {**placement, 'crop_bbox': box, 'pivot': {'x': placement['anchor'][0]-x, 'y': placement['anchor'][1]-y},
              'mask_position': [x, y], 'normalization_scale': factor}
    return str(target.resolve()), str(mask.resolve()), placed


def rebuild_mask(image_path, box, scene_size, target):
    with Image.open(image_path) as image:
        alpha = image.convert('RGBA').getchannel('A')
    mask = Image.new('L', scene_size); mask.paste(alpha, (box['x'], box['y']))
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); mask.save(target)
    return str(target.resolve())
