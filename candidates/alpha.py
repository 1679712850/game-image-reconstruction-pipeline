"""Independent generated-object segmentation and conservative alpha edge recovery."""
import cv2
import numpy as np
from PIL import Image
from cv.mask import as_mask


def extend_colors(image):
    rgba = np.array(image.convert('RGBA'))
    known = rgba[:, :, 3] > 8
    if not known.any():
        raise ValueError('Cannot restore an empty asset')
    _, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    colors = np.zeros((labels.max()+1, 3), np.uint8)
    colors[labels[known]] = rgba[:, :, :3][known]
    rgba[:, :, :3][~known] = colors[labels[~known]]
    return Image.fromarray(rgba)


def refine_alpha(mask, *, protected=None, max_region=4):
    alpha = as_mask(mask).copy()
    active = (alpha > 8).astype(np.uint8)
    if not active.any():
        raise ValueError('Generated segmentation is empty')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for label in range(1, count):
        region = labels == label
        if label != largest and stats[label, cv2.CC_STAT_AREA] <= max_region and not (
            protected is not None and np.any(region & (np.asarray(protected) > 8))):
            alpha[region] = 0
    holes = (alpha <= 8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    border = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    for label in range(1, count):
        if label not in border and stats[label, cv2.CC_STAT_AREA] <= max_region:
            alpha[labels == label] = 255
    return alpha


def matte_edges(image, mask):
    """Color-line trimap matting, with conservative antialiasing in ambiguous pixels.

    Interior geometry is untouched. This is a local color estimator, not a neural
    matting claim; semantic transparency still needs model/visual QA.
    """
    alpha = mask.astype(np.float32)/255
    binary = (mask > 8).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.erode(binary, kernel) > 0
    background = cv2.dilate(binary, kernel) == 0
    if not foreground.any() or not background.any():
        return mask
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    def nearest(known):
        _, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        palette = np.zeros((labels.max()+1, 3), np.float32)
        palette[labels[known]] = rgb[known]
        return palette[labels]
    fg, bg = nearest(foreground), nearest(background)
    vector = fg-bg
    denom = np.sum(vector*vector, axis=2)
    estimate = np.clip(np.sum((rgb-bg)*vector, axis=2)/np.maximum(denom, 1), 0, 1)
    unknown = ~foreground & ~background
    # Do not erase thin segmented geometry when colors are ambiguous.
    smooth = cv2.GaussianBlur(alpha, (3, 3), .5)
    estimate = np.where(denom > 64, estimate, smooth)
    estimate = np.where(binary > 0, np.maximum(estimate, .55), np.minimum(estimate, .45))
    alpha[unknown] = estimate[unknown]
    # A supplied soft alpha is evidence from the generated output, never source alpha.
    soft = (mask > 8) & (mask < 247)
    alpha[soft] = mask[soft]/255
    return np.rint(alpha*255).astype(np.uint8)


def recover_alpha(image, path, prepared, obj, sam, *, mock_noop=False):
    if mock_noop:
        # Mock passthrough never claims fresh segmentation or completion.
        return image.convert('RGBA')
    if sam is None:
        raise ValueError('Object completion requires independent foreground segmentation')
    with Image.open(prepared['amodal_mask_path']) as prediction:
        prediction = np.asarray(prediction.convert('L'))
    with Image.open(prepared['observed_mask_path']) as visible:
        visible = np.asarray(visible.convert('L'))
    if prediction.shape != (image.height, image.width):
        raise ValueError('Amodal hint and generated canvas differ')
    ys, xs = np.where(visible > 8)
    if not len(xs):
        raise ValueError('No visible landmarks for target mask selection')
    if hasattr(sam, 'predict_candidates') and not getattr(sam, 'mock', False):
        py, px = np.where(prediction > 8)
        pad = max(4, round(max(px.max()-px.min()+1, py.max()-py.min()+1)*.1))
        box = [max(0, px.min()-pad), max(0, py.min()-pad),
               min(image.width, px.max()+pad+1), min(image.height, py.max()+pad+1)]
        sample = np.linspace(0, len(xs)-1, min(8, len(xs)), dtype=int)
        positives = np.column_stack([xs[sample], ys[sample]])
        negatives = np.array([[0, 0], [image.width-1, 0], [0, image.height-1], [image.width-1, image.height-1]])
        masks, scores = sam.predict_candidates(image, {'box': np.array(box, np.float32),
            'point_coords': np.concatenate([positives, negatives]).astype(np.float32),
            'point_labels': np.array([1]*len(positives)+[0]*4)})
        ranked = []
        for mask, score in zip(masks, scores):
            active = np.asarray(mask) > 0
            if active.shape != visible.shape or not np.isfinite(mask).all() or not np.isfinite(score):
                continue
            retention = np.count_nonzero(active & (visible > 8))/max(1, len(xs))
            leak = np.count_nonzero(active & (prediction <= 8))/max(1, active.sum())
            border = np.mean(np.concatenate([active[0], active[-1], active[:, 0], active[:, -1]]))
            ranked.append((retention + .25*float(score)-.3*leak-border, active.astype(np.uint8)*255))
        if not ranked:
            raise ValueError('No valid generated segmentation candidates')
        mask = max(ranked, key=lambda item: item[0])[1]
    else:
        outputs = sam.segment(path, [{'id': obj['id'], 'category': obj['category'], 'confidence': obj['confidence'],
            'bbox': {'x': 0, 'y': 0, 'w': image.width, 'h': image.height}}])
        if len(outputs) != 1 or outputs[0]['id'] != obj['id']:
            raise ValueError('Generated segmentation changed target identity')
        mask = as_mask(outputs[0]['mask'])
        if mask.shape != visible.shape:
            raise ValueError('Generated segmentation dimensions differ')
    mask = matte_edges(image, refine_alpha(mask, protected=visible))
    rgba = np.array(image.convert('RGBA'))
    rgba[:, :, 3] = mask
    rgba[mask == 0] = 0
    return Image.fromarray(rgba)
