"""Restore all assets, then review all results to avoid GPU model thrashing."""
from collections.abc import Callable
import logging
import numpy as np
from PIL import Image
from agent.state import SceneState
from app.asset_status import normalize_status
from app.paths import read_rgba
from services.execution import instance_scope
from services.model_support import ModelUnavailableError
from services.progress import StageProgress
from services.upscale_service import UpscaleService, resolve_upscale_factor


def _review_enhancement(candidate, obj, service, reviewer, candidate_config):
    from candidates.qa import evaluate
    from app.resource_config import CandidateConfig
    decision = evaluate(candidate, obj['asset_path'], obj, reviewer,
                        candidate_config or CandidateConfig(), force_review=True)
    details = {**decision.model_dump(mode='json'), 'candidate_path': candidate,
               'backend': service.backend, 'silhouette_preserved': True}
    return (candidate if decision.status == 'ACCEPT' else None), details


def enhance(obj: dict, service: UpscaleService, reviewer: object, candidate_config: object,
            *, review: bool = True) -> tuple[str | None, dict]:
    """Lock alpha before enqueueing review; only paths survive between passes."""
    original = read_rgba(obj['asset_path'])
    scale = resolve_upscale_factor(*original.size)
    candidate = service.upscale(obj['asset_path'], scale)
    restored = read_rgba(candidate)
    expected_alpha = original.getchannel('A').resize(restored.size, Image.Resampling.LANCZOS)
    if restored.size != (original.width*scale, original.height*scale) or not np.array_equal(
            np.asarray(restored.getchannel('A')), np.asarray(expected_alpha)):
        raise ValueError('HD restoration changed completed silhouette or dimensions')
    if service.backend != 'real_esrgan' or service.mock:
        return candidate, {'status': 'preview_only', 'backend': service.backend,
                           'reason': 'Interpolation preview; neural detail restoration not performed'}
    if not review:
        return candidate, {'status': 'pending_review', 'candidate_path': candidate,
                           'backend': service.backend, 'silhouette_preserved': True}
    return _review_enhancement(candidate, obj, service, reviewer, candidate_config)


def _set_enhancement(obj, hd, details, status, reason, enabled, required):
    obj.update(hd_asset_path=hd, hd_qa=details, enhancement_status=status,
               enhancement={'requested': bool(enabled), 'required': required,
                            'status': status, 'reason': reason})


def _failed(obj, error, service, enabled, required):
    status = 'unavailable' if isinstance(error, ModelUnavailableError) else 'failed'
    reason = str(error)
    _set_enhancement(obj, None, {'status': status, 'reason': reason, 'backend': service.backend},
                     status, reason, enabled, required)
    logging.getLogger(__name__).warning('[Upscale] object=%s status=%s reason=%s', obj['id'], status, reason)


def make_upscale_objects(service: UpscaleService, enabled: bool | str, reviewer: object = None,
                         candidate_config: object = None, required: bool = False,
                         progress: Callable[[str], None] | None = None) -> Callable[[SceneState], dict]:
    def upscale_objects(state: SceneState) -> dict:
        objects, queued, pending_reviews = [], [], []
        for record in state.get('objects', []):
            obj = dict(record)
            if obj.get('accepted_asset'):
                obj['asset_path'] = obj['accepted_asset']
            reason = 'disabled_by_config' if not enabled else 'missing_base_asset'
            status = 'disabled' if not enabled else 'skipped'
            if enabled:
                if obj.get('completion_required') and (obj.get('needs_manual_review') or obj.get('accepted_candidate_id') == 'original'):
                    reason = 'completion_not_accepted'
                elif obj.get('asset_path'):
                    if enabled == 'auto' and not service.available():
                        reason = 'backend_or_weights_unavailable'
                    else:
                        queued.append(obj)
            _set_enhancement(obj, None, {'status': status, 'reason': reason, 'backend': service.backend},
                             status, reason, enabled, required)
            objects.append(obj)

        if enabled:
            logging.getLogger(__name__).info('[Upscale] queued=%d skipped=%d; HD QA follows restoration',
                                              len(queued), len(objects)-len(queued))
            with StageProgress('Upscale', len(queued), progress) as tracker:
                for obj in queued:
                    tracker.start_item(obj['id'])
                    try:
                        with instance_scope(obj['id']):
                            hd, details = enhance(obj, service, reviewer, candidate_config, review=False)
                        status = details['status']
                        _set_enhancement(obj, hd, details, status, details.get('reason', ''), enabled, required)
                        if status == 'pending_review':
                            pending_reviews.append(obj)
                    except Exception as error:
                        _failed(obj, error, service, enabled, required)
                    tracker.advance('restored' if obj['enhancement_status'] == 'pending_review' else obj['enhancement_status'])

        if pending_reviews:
            with StageProgress('HD QA', len(pending_reviews), progress) as tracker:
                for obj in pending_reviews:
                    tracker.start_item(obj['id'])
                    try:
                        with instance_scope(obj['id']):
                            hd, details = _review_enhancement(obj['hd_asset_path'], obj, service, reviewer, candidate_config)
                        status = 'ready' if hd else 'failed'
                        reason = details.get('reason', 'hd_qa_failed' if hd is None else '')
                        _set_enhancement(obj, hd, details, status, reason, enabled, required)
                    except Exception as error:
                        _failed(obj, error, service, enabled, required)
                    tracker.advance(obj['enhancement_status'])

        for index, obj in enumerate(objects):
            if obj.get('asset_path'):
                # Header reads suffice here: pixels were already validated above.
                with Image.open(obj['asset_path']) as image:
                    size = image.size
                texture = size
                if obj.get('hd_asset_path'):
                    with Image.open(obj['hd_asset_path']) as image:
                        texture = image.size
                obj.update(logical_size=list(size), texture_size=list(texture), texture_scale=texture[0]/size[0])
            obj = normalize_status(obj)
            if required and obj['enhancement_status'] not in {'ready', 'preview_only'}:
                obj.update(status='manual_review', needs_manual_review=True)
            objects[index] = obj
        return {'objects': objects, 'failed_objects': [o['id'] for o in objects if o.get('status') != 'pass']}
    return upscale_objects
