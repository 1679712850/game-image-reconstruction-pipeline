"""Optional HD restoration never changes the quality of the base asset."""
from collections.abc import Callable
import logging
import numpy as np
from PIL import Image
from agent.state import SceneState
from app.asset_status import normalize_status
from app.paths import read_rgba
from services.model_support import ModelUnavailableError
from services.upscale_service import UpscaleService, resolve_upscale_factor


def enhance(obj: dict, service: UpscaleService, reviewer: object, candidate_config: object) -> tuple[str | None, dict]:
    """Restore and independently QA texture detail while locking alpha."""
    original = read_rgba(obj['asset_path'])
    scale = resolve_upscale_factor(*original.size)
    candidate = service.upscale(obj['asset_path'], scale)
    restored = read_rgba(candidate)
    expected_alpha = original.getchannel('A').resize(restored.size, Image.Resampling.LANCZOS)
    if restored.size != (original.width*scale, original.height*scale) or not np.array_equal(
            np.asarray(restored.getchannel('A')), np.asarray(expected_alpha)):
        raise ValueError('HD restoration changed completed silhouette or dimensions')
    if service.backend != 'real_esrgan' or service.mock:
        return candidate, {'status':'preview_only','backend':service.backend,
                           'reason':'Interpolation preview; neural detail restoration not performed'}
    from candidates.qa import evaluate
    from app.resource_config import CandidateConfig
    decision = evaluate(candidate,obj['asset_path'],obj,reviewer,candidate_config or CandidateConfig(),force_review=True)
    details = {**decision.model_dump(mode='json'),'candidate_path':candidate,'backend':service.backend,'silhouette_preserved':True}
    return (candidate if decision.status == 'ACCEPT' else None), details


def make_upscale_objects(service: UpscaleService, enabled: bool | str, reviewer: object = None,
                         candidate_config: object = None, required: bool = False) -> Callable[[SceneState], dict]:
    def upscale_objects(state: SceneState) -> dict:
        objects = []
        for record in state.get('objects', []):
            obj = dict(record)
            if obj.get('accepted_asset'):
                obj['asset_path'] = obj['accepted_asset']
            hd = None
            status, reason = 'disabled', 'disabled_by_config'
            details = {'status': status, 'reason': reason}
            if enabled:
                status, reason = 'skipped', 'missing_base_asset'
                if obj.get('completion_required') and (obj.get('needs_manual_review') or obj.get('accepted_candidate_id') == 'original'):
                    reason = 'completion_not_accepted'
                elif obj.get('asset_path'):
                    try:
                        if enabled == 'auto' and not service.available():
                            status, reason = 'skipped', 'backend_or_weights_unavailable'
                        else:
                            hd, details = enhance(obj, service, reviewer, candidate_config)
                            status = 'preview_only' if details['status'] == 'preview_only' else 'ready' if hd else 'failed'
                            reason = details.get('reason','hd_qa_failed' if hd is None else '')
                    except Exception as error:
                        status = 'unavailable' if isinstance(error, ModelUnavailableError) else 'failed'
                        reason = str(error)
                        logging.getLogger(__name__).warning('[Upscale] object=%s status=%s reason=%s',obj['id'],status,reason)
                if details['status'] == 'disabled':
                    details = {'status':status,'reason':reason,'backend':service.backend}
            obj['hd_qa'] = details
            obj['enhancement_status'] = status
            obj['enhancement'] = {'requested':bool(enabled),'required':required,'status':status,'reason':reason}
            if obj.get('asset_path'):
                size = read_rgba(obj['asset_path']).size
                texture = read_rgba(hd).size if hd else size
                obj.update(hd_asset_path=hd,logical_size=list(size),texture_size=list(texture),texture_scale=texture[0]/size[0])
            obj = normalize_status(obj)
            if required and status not in {'ready','preview_only'}:
                # Compatibility status signals strict delivery failure; base status stays factual.
                obj.update(status='manual_review',needs_manual_review=True)
            objects.append(obj)
        # `failed_objects` remains the graph's QA retry contract; optional
        # enhancement/review is exposed through completion_metrics instead.
        return {'objects':objects,'failed_objects':[o['id'] for o in objects if o.get('status') != 'pass']}
    return upscale_objects
