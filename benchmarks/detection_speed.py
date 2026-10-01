"""Reproducible CPU microbenchmarks for detection postprocessing (no weights).

Run: python -m benchmarks.detection_speed --output /tmp/detection-speed.json
The frozen reference and assertions live with the differential regression tests.
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image

from app.detection_config import DedupConfig
from fusion.candidate_fusion import fuse_candidates
from segmentation.local_refiner import candidate_metrics, nearby_edges
from tests.test_detection_optimization import reference_fuse
from tests.test_p0_detection import candidate


def timed(function, repeats=5):
    function()  # warm-up
    values = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        values.append(perf_counter()-started)
    return median(values)


def run():
    rng = np.random.default_rng(23)
    candidates = []
    for index in range(1500):
        x, y = (index % 50)*180, (index // 50)*180
        value = candidate(str(index), (x, y, x+80, y+90), category=['tree', 'rock', 'building'][index % 3])
        candidates.append(value)
        if index % 5 == 0:
            candidates.append(value.model_copy(update={'id': f'copy_{index}', 'confidence': .6}, deep=True))
    config = DedupConfig()
    old = lambda: reference_fuse(candidates, config)
    new = lambda: fuse_candidates(candidates, config)
    serialize = lambda parts: [[c.model_dump() for c in part] for part in parts]
    assert serialize(old()) == serialize(new())
    fusion_old, fusion_new = timed(old), timed(new)

    image = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
    masks = [rng.random((512, 512)) > .6 for _ in range(3)]
    def uncached():
        return [candidate_metrics(mask, [50, 50, 400, 450], image, .9) for mask in masks]
    def cached():
        edges = nearby_edges(image)
        return [candidate_metrics(mask, [50, 50, 400, 450], image, .9, nearby=edges) for mask in masks]
    assert uncached() == cached()
    edges_old, edges_new = timed(uncached, 11), timed(cached, 11)
    return {
        'seed': 23, 'device': 'cpu', 'measurement': 'median after warm-up',
        'fusion': {'input_candidates': len(candidates), 'output_equal': True,
                   'baseline_seconds': fusion_old, 'optimized_seconds': fusion_new,
                   'speedup': fusion_old/fusion_new},
        'sam_candidate_scoring': {'image_size': [512, 512], 'candidates': 3, 'scores_equal': True,
                                  'canny_calls_before': 3, 'canny_calls_after': 1,
                                  'baseline_seconds': edges_old, 'optimized_seconds': edges_new,
                                  'speedup': edges_old/edges_new},
        'limitations': 'Synthetic CPU microbenchmarks, not whole-pipeline or CUDA speedup measurements.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
