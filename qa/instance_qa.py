"""Shape quality independent of SAM confidence, suitable for retry comparison."""
import cv2
import numpy as np
from PIL import Image
from cv.mask import read_mask
from segmentation.local_refiner import candidate_metrics


def inspect_instance(record, source_path, config):
    if not record.get('mask_path'):
        return {'quality_score': -1.0}, ['BAD_MASK']
    mask = read_mask(record['mask_path'])
    with Image.open(source_path) as image:
        if mask.shape != (image.height, image.width):
            return {'quality_score': -1.0}, ['BAD_MASK']
        box = record['bbox']; x, y, w, h = (box[k] for k in ('x', 'y', 'w', 'h'))
        m = candidate_metrics(mask, [x,y,x+w,y+h], image, record['confidence'])
    active = (mask > config.crop.alpha_threshold).astype(np.uint8)
    components = cv2.connectedComponents(active, connectivity=8)[0]-1
    ratio = int(active.sum())/max(1, w*h)
    failures = []
    if not active.any(): failures.append('BAD_MASK')
    elif ratio < config.p1.min_mask_bbox_ratio: failures.append('MASK_TOO_SMALL')
    if ratio > config.p1.max_mask_bbox_ratio: failures.append('MASK_TOO_LARGE')
    if m['background_leak'] > config.qa.max_mask_outside_bbox: failures.append('BACKGROUND_LEAK')
    if record.get('is_truncated'): failures.append('CROSS_TILE_FRAGMENT')
    if record.get('uncertain') and record.get('element_type') != 'hybrid': failures.append('WRONG_CATEGORY')
    # A complete re-observation is objective improvement even with equal mask scores.
    return {**m, 'quality_score': m['score'] - (.2 if record.get('is_truncated') else 0), 'mask_bbox_ratio': ratio,
            'fragment_count': components, 'alpha_integrity': bool(np.all((mask == 0) | (mask == 255)))}, failures
