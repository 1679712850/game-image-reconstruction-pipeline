"""Quality gates and deterministic ranking, independent from generation order."""
import math
import numpy as np
from PIL import Image
from schemas.candidate import CandidateQA

WEIGHTS = {'shape': .20, 'style': .20, 'perspective': .15, 'scale': .15,
           'color': .10, 'edge': .10, 'semantic': .10}
RETRY_HINTS = {
    'perspective': "Maintain exact original isometric camera angle. Do not change viewpoint. Match the source object's orientation.",
    'color': 'Match the original palette and saturation exactly.',
    'lighting': 'Match the source light direction, highlights and shadows.',
    'shape': 'Restore the missing silhouette while retaining the original structure and proportions.',
    'semantic': 'Keep only the original object category. Remove extra objects and decorations.',
    'style': 'Match the original rendering style and level of detail. Remove extra decorations.',
    'edge': 'Produce a clean complete silhouette without background contamination.',
}


def evaluate(candidate, original, obj, reviewer, config, *, force_review=False):
    with Image.open(candidate) as image:
        rgba = np.array(image.convert('RGBA'))
    if not np.any(rgba[:, :, 3] > 8):
        return CandidateQA(status='REJECT', reasons=['empty alpha'])
    with Image.open(original) as image:
        source = image.convert('RGBA')
        # No-op mock edits cannot gain ranking credit for generation.
        if not force_review and source.size == (rgba.shape[1], rgba.shape[0]) and np.array_equal(np.array(source), rgba):
            return original_qa(obj)
    if reviewer is None or not hasattr(reviewer, 'review_candidate'):
        return CandidateQA(status='RETRY', reasons=['semantic/style/perspective QA unavailable'])
    decision = CandidateQA.model_validate(reviewer.review_candidate(original, candidate, {
        'category': obj['category'], 'bbox': obj['bbox'], 'projection': obj.get('projection', 'unknown')}))
    scores = decision.scores
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        return CandidateQA(status='REJECT', reasons=['invalid QA scores'])
    required = {*WEIGHTS, 'lighting', 'background_leak', 'occlusion_reconstruction_quality'}
    if not required.issubset(scores):
        return CandidateQA(status='RETRY', scores=scores, reasons=['incomplete semantic/style/perspective QA'], evaluator=decision.evaluator)
    overall = sum(weight * scores[key] for key, weight in WEIGHTS.items())
    reasons = list(decision.reasons)
    weak = [key for key in required - {'background_leak'} if scores[key] < config.accept_threshold]
    reasons.extend(f'{key} mismatch' for key in sorted(weak))
    fatal = any(scores[key] < .35 for key in ('semantic', 'shape', 'style'))
    if fatal or decision.status == 'REJECT' or overall < config.retry_threshold:
        status = 'REJECT'
    elif decision.status == 'ACCEPT' and overall >= config.accept_threshold and not weak and scores['background_leak'] <= .08:
        status = 'ACCEPT'
    else:
        status = 'RETRY'
    if scores['background_leak'] > .08:
        reasons.append('edge background_leak')
    return CandidateQA(status=status, scores=scores, overall=overall, reasons=reasons, evaluator=decision.evaluator)


def original_qa(obj):
    passed = obj.get('status') == 'pass'
    damaged = obj.get('requires_inpainting') or obj.get('is_truncated') or obj.get('occluded_pixel_count', 0)
    return CandidateQA(status='ACCEPT' if passed else 'RETRY', overall=(.65 if damaged else .80) if passed else .40,
                       reasons=['source segmentation QA passed' if passed else 'source needs repair or manual review'],
                       evaluator='source_segmentation_rules')


def retry_prompt(prompt, reasons, attempt):
    hints = [hint for key, hint in RETRY_HINTS.items() if any(key in reason.lower() for reason in reasons)]
    return prompt + '\nCorrection pass ' + str(attempt) + ': ' + ' '.join(hints or [
        'Preserve the source identity, geometry and clean alpha. Correct these QA failures: ' + '; '.join(reasons)])


def select(registry):
    viable = [c for c in registry.candidates if c.image_path and c.qa.status == 'ACCEPT']
    if viable:
        return max(viable, key=lambda c: (c.qa.overall, c.type == 'segmentation', -c.attempt)), False
    # Rejected assets are never safe fallback. RETRY assets need human review.
    for kind in ('generation', 'inpaint', 'segmentation', 'crop_mask'):
        pool = [c for c in registry.candidates if c.image_path and c.type == kind and c.qa.status != 'REJECT'
                and (kind in {'segmentation', 'crop_mask'} or bool(c.qa.scores) and c.qa.overall > 0)]
        if pool:
            return max(pool, key=lambda c: (c.qa.overall, -c.attempt)), True
    return None, True
