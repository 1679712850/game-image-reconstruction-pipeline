"""Texture proposals are uncertain QA regions, never automatic detections."""
import cv2
import numpy as np
from PIL import Image, ImageDraw
from cv.mask import read_mask


def coverage_review(source_path, records, config):
    with Image.open(source_path) as image:
        image = image.convert('RGBA')
        rgb = np.array(image)[:, :, :3]
        eligible = np.array(image)[:, :, 3] > 8
    h, w = eligible.shape
    eligible_count = max(1, int(eligible.sum()))
    terrain = np.zeros((h,w), bool); covered = np.zeros_like(terrain)
    overlay = np.full((h,w,3), 45, np.uint8)
    for r in records:
        x,y,bw,bh = (r['bbox'][k] for k in ('x','y','w','h'))
        if r.get('element_type') == 'terrain' and r.get('mask_path'):
            terrain |= read_mask(r['mask_path']) > 8
        else:
            covered[y:y+bh,x:x+bw] = True
            overlay[y:y+bh,x:x+bw] = (40,210,80) if r['confidence'] >= .4 else (255,220,40)
    overlay[terrain & ~covered] = (130,130,130)
    edges = cv2.Canny(rgb, 60, 150) > 0
    regions = []
    cell = config.coverage_cell_size
    for y in range(0,h,cell):
        for x in range(0,w,cell):
            bh,bw = min(cell,h-y), min(cell,w-x)
            free = eligible[y:y+bh,x:x+bw] & ~covered[y:y+bh,x:x+bw] & ~terrain[y:y+bh,x:x+bw]
            score = float((edges[y:y+bh,x:x+bw] & free).sum()/max(1,free.sum()))
            fraction = float(free.sum()/eligible_count)
            if free.sum() >= 16 and (score >= config.edge_density_threshold or fraction > config.unassigned_threshold):
                textured = score >= config.edge_density_threshold
                regions.append({'category': None, 'approx_bbox': {'x':x,'y':y,'w':bw,'h':bh},
                                'failure_type': 'MISSED_DETECTION' if textured else 'UNASSIGNED_REGION',
                                'reason': 'uncovered high edge density; unverified object proposal' if textured else
                                          'unassigned source region; verify terrain or missed objects',
                                'score':score})
    regions = sorted(regions,key=lambda r:r['score'],reverse=True)[:config.max_problem_regions]
    for r in regions:
        b=r['approx_bbox']; overlay[b['y']:b['y']+b['h'],b['x']:b['x']+b['w']] = (230,45,45)
    return regions, Image.blend(image.convert('RGB'), Image.fromarray(overlay), .5)
