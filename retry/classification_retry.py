"""A category failure invokes crop classification, never SAM."""
from pathlib import Path
from PIL import Image
from scene.element_classifier import SceneElementClassifier


def reclassify_record(record, state, reviewer, attempt):
    log = {'instance_id': record['id'], 'retry_reason': 'WRONG_CATEGORY',
           'retry_count': attempt, 'previous_score': record.get('classification_confidence') or record['confidence'],
           'new_score': None, 'improved': False, 'action': 'classification_unavailable'}
    if reviewer is None or reviewer.backend != 'llm':
        return record, log
    box = record['bbox']
    with Image.open(state['source_path']) as image:
        size = image.size
        crop = image.crop((max(0, box['x']-16), max(0, box['y']-16),
                           min(image.width, box['x']+box['w']+16), min(image.height, box['y']+box['h']+16)))
    path = Path(state['output_dir'])/'debug'/f"{record['id']}_classify_{attempt}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)
    try:
        decision = reviewer.classify_crop(str(path), record['category'])
        log.update(action='vision_crop_classification', new_score=decision.confidence,
                   reason=decision.reason, proposed_category=decision.category)
        log['improved'] = not decision.uncertain and decision.confidence > log['previous_score']
        if log['improved']:
            record = SceneElementClassifier().classify({**record,
                'category': decision.category.lower().replace(' ', '_'),
                'classification_confidence': decision.confidence}, size)
    except Exception as error:
        log.update(action='classification_unavailable', error=type(error).__name__)
    return record, log
