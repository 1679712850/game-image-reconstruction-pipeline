"""Semantic amodal planning, transparent canvas preparation and geometric QA.

The prediction is a hint, not an output alpha. Missing vision is recorded explicitly;
we never describe dilation or a bounding rectangle as a predicted object silhouette.
"""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from schemas.reconstruction import OcclusionAnalysis
from candidates.geometry import source_placement

COMPLETION_PROMPT = '''Reconstruct the complete object as if all foreground occluders were removed.
Infer and generate all hidden geometry. Extend the object beyond its currently visible alpha boundary wherever necessary.
Do not preserve the original alpha silhouette. Generate a new complete silhouette corresponding to the full unoccluded object.
The output must represent one isolated complete game asset. Remove every overlapping foreground object,
including unrelated trees, grass, buildings, rocks, characters, shadows and neighboring objects.
Do not leave remnants, ghost pixels, shadows or fragments from occluding objects.
Preserve the exact artistic style, camera angle, material language, isometric projection, line weight,
shading direction and color palette. Preserve hand-painted 2D game art, existing line work and stylized shading.
Do not convert to photorealism, 3D render or smooth AI painting. Do not redesign the object.
Only infer missing occluded regions with plausible continuation of edges, structures, textures and geometry.
Keep visible landmarks at their original canvas coordinates and scale. Leave empty context around the completed object.'''

RETRY_PROMPT = '''The previous result still preserves the visible-only silhouette or has incomplete geometry.
Do not preserve the original alpha shape. Expand the silhouette and reconstruct its hidden portions outside
the previous object boundary only in the predicted occlusion directions. Remove all occluder remnants.
The final object should look complete and naturally shaped, within the predicted expansion limit.'''


def analyze(obj, state, reviewer):
    total = obj.get('visible_pixel_count', 0) + obj.get('occluded_pixel_count', 0)
    ratio = obj.get('occluded_pixel_count', 0) / max(1, total)
    needed = bool(obj.get('requires_inpainting') or obj.get('is_truncated') or ratio >= .05)
    context = {k: obj.get(k) for k in ('id', 'category', 'bbox', 'crop_bbox', 'occludes',
                'truncated_edges', 'visible_pixel_count', 'occluded_pixel_count', 'requires_inpainting', 'is_truncated')}
    context['neighbors'] = [{k: o.get(k) for k in ('id', 'category', 'bbox', 'occludes')}
                            for o in state.get('objects', []) if o['id'] != obj['id']]
    context['projection'] = state.get('scene_analysis', {}).get('projection', 'unknown')
    error = 'Semantic amodal analysis unavailable; configure a vision reviewer'
    if reviewer is not None and hasattr(reviewer, 'analyze_occlusion'):
        try:
            result = OcclusionAnalysis.model_validate(reviewer.analyze_occlusion(
                state['source_path'], obj['asset_path'], context))
            if result.object_id != obj['id'] or result.object_type != obj['category']:
                raise ValueError('Occlusion analysis changed object identity')
            # Observed ownership loss cannot be overruled by a "complete" guess.
            if needed and not result.needs_completion:
                raise ValueError('Occlusion analysis contradicts observed missing geometry')
            return result
        except Exception as exc:
            error = f'Semantic amodal analysis unavailable: {exc}'
    return OcclusionAnalysis(object_id=obj['id'], object_type=obj['category'], occlusion_ratio=ratio,
        reconstruction_confidence=0, needs_completion=needed, evidence='unavailable',
        boundary_reasoning=error, max_expansion_ratio=min(8, max(1.3, 1 / max(.125, 1-ratio) * 1.2)))


def prepare(obj, analysis, folder, config, request=None, *, growth=1):
    """Rasterize semantic polygons, union visible evidence, then pad an independent canvas."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with Image.open(obj['asset_path']) as source:
        source = source.convert('RGBA')
    w, h = source.size
    visible_box = source.getchannel('A').getbbox()
    if visible_box is None:
        raise ValueError('Cannot infer an amodal shape without visible evidence')
    points = [(x*w, y*h) for polygon in analysis.full_shape_polygons for x, y in polygon]
    left = math.floor(min([0] + [p[0] for p in points]))
    top = math.floor(min([0] + [p[1] for p in points]))
    right = math.ceil(max([w] + [p[0] for p in points]))
    bottom = math.ceil(max([h] + [p[1] for p in points]))
    factor = config.severe_padding if analysis.occlusion_ratio >= .30 else config.context_padding
    pad = math.ceil(max(2, config.minimum_padding, max(right-left, bottom-top)*factor)*growth)
    left -= pad; top -= pad; right += pad; bottom += pad
    size = (right-left, bottom-top)
    if max(size) > config.max_canvas_edge or size[0]*size[1] > config.max_canvas_pixels:
        raise ValueError('Amodal canvas exceeds configured memory/geometry budget')
    offset = (-left, -top)
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    source_array = np.array(source); source_array[source_array[:, :, 3] == 0] = 0
    canvas.paste(Image.fromarray(source_array), offset)
    visible = Image.new('L', size); visible.paste(source.getchannel('A'), offset)
    prediction = Image.new('L', size)
    draw = ImageDraw.Draw(prediction)
    for polygon in analysis.full_shape_polygons:
        draw.polygon([(round(x*w-left), round(y*h-top)) for x, y in polygon], fill=255)
    amodal = np.maximum(np.asarray(prediction), np.asarray(visible))
    visible_area = max(1, np.count_nonzero(np.asarray(visible) > 8))
    if analysis.full_shape_polygons and np.count_nonzero(amodal > 8)/visible_area > analysis.max_expansion_ratio * 1.1:
        raise ValueError('Semantic mask exceeds its predicted expansion bound')
    # Edit permission spans context as well; it is NEVER copied to final alpha.
    edit = Image.new('L', size, 255)
    if request:
        with Image.open(request.mask_path) as supplied:
            if supplied.size != source.size or not supplied.convert('L').getbbox():
                raise ValueError('Edit mask must be nonempty and match the original cropped asset')
            supplied.convert('L').save(folder/'requested_edit_mask.png')
    paths = {}
    for key, image in [('expanded_crop_path', canvas), ('observed_mask_path', visible),
                       ('amodal_mask_path', Image.fromarray(amodal)), ('edit_mask_path', edit)]:
        path = folder/f'{key.removesuffix("_path")}.png'; image.save(path); paths[key] = str(path.resolve())
    box = obj['crop_bbox']
    canvas_box = {'x': box['x']+left, 'y': box['y']+top, 'w': size[0], 'h': size[1]}
    placement = source_placement(obj)
    placement['crop_bbox'] = canvas_box
    amodal_box = Image.fromarray(amodal).getbbox()
    full_box = {'x': canvas_box['x']+amodal_box[0], 'y': canvas_box['y']+amodal_box[1],
                'w': amodal_box[2]-amodal_box[0], 'h': amodal_box[3]-amodal_box[1]}
    bbox_visible = {'x': box['x']+visible_box[0], 'y': box['y']+visible_box[1],
                    'w': visible_box[2]-visible_box[0], 'h': visible_box[3]-visible_box[1]}
    return {**paths, 'canvas_bbox': canvas_box, 'bbox_full': full_box, 'bbox_visible': bbox_visible, 'placement': placement,
            'source_offset': list(offset), 'amodal_evidence': analysis.evidence,
            'analysis': {**analysis.model_dump(mode='json'), 'visible_ratio': analysis.visible_ratio}}


def completion_metrics(alpha, visible, analysis):
    full = np.asarray(alpha) > 8; observed = np.asarray(visible) > 8
    if full.shape != observed.shape:
        raise ValueError('Completion QA requires a shared canvas')
    count = max(1, int(observed.sum())); area = int(full.sum())
    ratio = area/count
    reasons = []
    outside = int((full & ~observed).sum())
    retained = float((full & observed).sum()/count)
    if analysis.needs_completion and (ratio <= 1.01 or outside/count <= .01):
        reasons.append('alpha expansion missing: visible-only silhouette')
    if ratio > analysis.max_expansion_ratio:
        reasons.append('alpha expansion exceeds semantic limit')
    if retained < .90:
        reasons.append('visible geometry lost or displaced')
    ys, xs = np.where(full)
    border = bool(len(xs) and min(xs.min(), ys.min(), full.shape[1]-1-xs.max(), full.shape[0]-1-ys.max()) < 2)
    if border:
        reasons.append('object reaches canvas border: expand canvas')
    return {'alpha_area': area, 'visible_area': count, 'expansion_ratio': ratio-1,
            'outside_visible_area': outside, 'visible_retention': retained, 'touches_border': border}, reasons
