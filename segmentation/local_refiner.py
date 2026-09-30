"""Bounded local crops, contextual prompts and evidence-weighted SAM candidates."""
import cv2
import numpy as np
from PIL import Image

from app.paths import read_rgba
from cv.tiles import tile_windows
from schemas.object import BBox
from segmentation.prompt_generator import generate_prompts
from segmentation.mask_postprocess import mask_postprocess_by_category


def local_windows(bbox, size, maximum, attempt=0):
    box = BBox.model_validate(bbox)
    width, height = size
    if box.x+box.w > width or box.y+box.h > height:
        raise ValueError("Local box lies outside original image")
    side = max(box.w, box.h)
    ratio = 1.0 if side < 64 else .6 if side < 128 else .3 if side < 256 else .15
    padding = round(side * ratio * (1 + .5*attempt))
    # Oversized objects use overlapping bounded windows, never a truncated center crop.
    if side > maximum:
        x0, y0 = max(0,box.x-padding), max(0,box.y-padding)
        right, bottom = min(width,box.x+box.w+padding), min(height,box.y+box.h+padding)
        return [(x0+x, y0+y, x0+r, y0+b)
                for x, y, r, b in tile_windows(right-x0,bottom-y0,maximum,.25)
                if x0+x < box.x+box.w and y0+y < box.y+box.h and x0+r > box.x and y0+b > box.y]
    def span(start, extent, limit):
        length = min(limit, maximum, extent+2*padding)
        begin = min(max(0, start-(length-extent)//2), limit-length)
        return begin, begin+length
    x, r = span(box.x, box.w, width)
    y, b = span(box.y, box.h, height)
    return [(x, y, r, b)]


def candidate_metrics(mask, box, image, confidence, negative=None):
    active = np.asarray(mask) > 0
    h, w = active.shape
    x, y, r, b = np.rint(box).astype(int)
    x, y, r, b = max(0, x), max(0, y), min(w, r), min(h, b)
    area = int(active.sum())
    inside = int(active[y:b, x:r].sum())
    ratio = inside / max(1, (r-x)*(b-y))
    leak = (area-inside)/max(1, area)
    boundary = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edges = cv2.Canny(np.asarray(image.convert("RGB")), 60, 150)
    nearby = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
    edge = float((nearby & boundary).sum()/max(1, boundary.sum()))
    overlap = 0.0
    if negative:
        hits = [active[min(h-1, max(0, round(py))), min(w-1, max(0, round(px)))] for px, py in negative]
        overlap = float(np.mean(hits))
    score = .3*float(confidence)+.25*min(ratio/.35, 1)+.2*(inside/max(1, area))+.15*edge-.35*leak-.2*overlap
    if not area:
        score = -1.0
    return {"score": float(score), "sam_confidence": float(confidence), "bbox_coverage": ratio,
            "detection_alignment": inside/max(1, area), "edge_consistency": edge,
            "background_leak": leak, "overlap_penalty": overlap, "mask_area": area}


class LocalRefiner:
    def __init__(self, service, config):
        self.service, self.config = service, config

    def segment(self, image_path, records, attempt=0, neighbors=None):
        if not records:
            return []
        source = read_rgba(image_path)
        results = []
        for record in records:
            windows = local_windows(record["bbox"], source.size, self.config.max_local_crop_size, attempt)
            # Mock adapters can inject failures while exercising original-coordinate contracts.
            if self.service.mock:
                output = self.service.segment(image_path, [record])[0]
                output["segmentation"] = {"backend": "mock", "windows": [list(w) for w in windows], "attempt": attempt}
                results.append(output)
                continue
            canvas = np.zeros((source.height, source.width), np.uint8)
            traces = []
            for window in windows:
                x, y, r, b = window
                crop = source.crop(window).convert("RGB")
                desired = self.config.min_local_size*(1+.5*attempt)
                scale = 4 if min(crop.size)*2 < desired else 2 if min(crop.size) < desired else 1
                scale = max(1, min(scale, self.config.max_sam_input_size//max(crop.size)))
                image = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.LANCZOS)
                prompts, trace = generate_prompts(record, records if neighbors is None else neighbors, window, scale, attempt)
                masks, scores = self.service.predict_candidates(image, prompts)
                negatives = prompts["point_coords"][prompts["point_labels"] == 0].tolist()
                metrics = [candidate_metrics(mask, prompts["box"], image, score, negatives) for mask, score in zip(masks, scores)]
                best = max(range(len(metrics)), key=lambda i: metrics[i]["score"])
                chosen = Image.fromarray((masks[best] > 0).astype(np.uint8)*255).resize(crop.size, Image.Resampling.NEAREST)
                canvas[y:b, x:r] = np.maximum(canvas[y:b, x:r], np.asarray(chosen))
                traces.append({**trace, "candidates": metrics, "selected": best})
            canvas = mask_postprocess_by_category(canvas, record["category"])
            canvas[np.asarray(source)[:, :, 3] == 0] = 0
            results.append({**record, "mask": canvas, "segmentation": {"backend": "local_sam2", "attempt": attempt,
                            "windows": traces, "score": float(np.mean([t["candidates"][t["selected"]]["score"] for t in traces]))}})
        return results
