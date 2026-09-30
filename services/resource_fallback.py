"""Visible per-node recovery after supported resource strategies are exhausted."""
from services.model_manager import ResourceBudgetError, is_oom


def recover_node(name, state, error, services):
    if not (is_oom(error) or isinstance(error, ResourceBudgetError)):
        raise error
    failures = [*state.get('resource_failures', []), {'node': name, 'reason': str(error), 'needs_manual_review': True}]
    updates = {'resource_failures': failures}
    if name == 'analyze_scene':
        model_config = getattr(services.grounding, 'config', None)
        updates['scene_analysis'] = {
            'projection': getattr(model_config, 'projection', 'unknown'),
            'categories': list(getattr(model_config, 'categories', [])),
            'description': 'Resource recovery: configured categories only; visual understanding unavailable.'}
    elif name == 'detect_instances':
        updates.update(detections=[], detection_round=state.get('detection_round', 0)+1,
                       detection_diagnostics={'resource_failure': str(error)}, retry_count=0)
    elif name in {'segment_instances', 'retry_objects'}:
        records = state.get('detections' if name == 'segment_instances' else 'objects', [])
        updates['objects'] = [{**o, 'error': str(error), 'status': 'manual_review',
                               'needs_manual_review': True} for o in records]
        updates['failed_objects'] = [o['id'] for o in records]
        updates['retryable_objects'] = []
        updates['retry_count'] = state.get('max_retry', 0)
    elif name == 'decompose_layers':
        updates['decomposed_layers'] = []
    elif name == 'upscale_objects':
        from nodes.upscale_objects import make_upscale_objects
        updates.update(make_upscale_objects(services.upscale, False)(state))
    elif name == 'p1_scene':
        updates['missed_object_candidates'] = [*state.get('missed_object_candidates', []),
            {'failure_type': 'RESOURCE_EXHAUSTED', 'reason': str(error)}]
    else:
        # Generation and VLM QA have per-instance exception isolation. CPU export/IO errors
        # cannot be repaired by guessing image data and must still be reported to the caller.
        raise error
    return updates
