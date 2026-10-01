"""Compare real local DINO inputs/outputs. No downloads or CUDA are required.

Run: python -m benchmarks.check_detection_inputs --output /tmp/dino-check.json
Timing is a single-run diagnostic; baseline runs first and includes warm-up.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from app.models import ModelConfig
from services.grounding_service import GroundingService


def check_inputs(service, image, groups):
    """Compare every tensor against a fresh full processor invocation."""
    with service.prepared_window(image):
        for group in groups:
            phrases = [service.config.grounding.prompts.get(c, c.replace('_', ' ')) for c in group]
            prompt = '. '.join(phrases) + '.'
            expected = service._processor(images=image, text=prompt, return_tensors='pt').to(service._device)
            actual = service._prepare_inputs(image, prompt)
            assert expected.keys() == actual.keys(), 'Processor tensor keys changed'
            for key in expected:
                assert service._torch.equal(expected[key], actual[key]), f'Input tensor changed: {key}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = ModelConfig(device='cpu', local_files_only=True, cache_dir=Path('.cache/models'))
    service = GroundingService(False, config)
    service.load()
    with Image.open('benchmarks/scenes/sect_ruins.png') as source:
        image = source.convert('RGB').crop((0, 0, 768, 768))
    groups = [['tree', 'rock'], ['building', 'flag'], ['stone_lantern', 'tombstone']]
    check_inputs(service, image, groups)
    results, timings = {}, {}
    for name, reuse in [('baseline', False), ('reuse', True)]:
        service.config = config.model_copy(update={
            'grounding': config.grounding.model_copy(update={'reuse_image_inputs': reuse})})
        start = time.perf_counter()
        with service.prepared_window(image):
            results[name] = [service._infer_image(image, group, .1, .1, preserve_candidates=True)
                             for group in groups]
        timings[name] = time.perf_counter() - start
        print(name, timings[name], [len(v) for v in results[name]], flush=True)
    assert results['baseline'] == results['reuse'], 'Serial preprocessing changed outputs'
    start = time.perf_counter()
    with service.prepared_window(image):
        results['batch'] = service._infer_batch(image, groups, .1, .1)
    timings['batch'] = time.perf_counter() - start
    max_score_error = max_box_error = 0
    same_labels = True
    for old, new in zip(results['baseline'], results['batch']):
        assert len(old) == len(new), 'Batched candidate count changed'
        for a, b in zip(old, new):
            same_labels &= a['category'] == b['category'] and a['raw_label'] == b['raw_label']
            max_score_error = max(max_score_error, abs(a['confidence'] - b['confidence']))
            max_box_error = max(max_box_error, max(abs(x-y) for x, y in zip(a['bbox'], b['bbox'])))
    report = {
        'device': 'cpu', 'model': config.grounding.model_id, 'crop_size': list(image.size),
        'groups': groups, 'counts': [len(v) for v in results['baseline']],
        'serial_inputs_exact': True, 'serial_outputs_exact': results['baseline'] == results['reuse'],
        'batch_labels_equal': same_labels, 'batch_max_confidence_error': max_score_error,
        'batch_max_bbox_error_pixels': max_box_error, 'timings_seconds': timings,
        'note': 'Single CPU run; baseline first includes warm-up. Timings do not establish speedup. Batch remains opt-in.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
