"""Base quality is independent of optional enhancement availability."""
from typing import Any


def base_status(record: dict[str, Any]) -> str:
    """QA acceptance plus persisted segmentation and exportable pixels."""
    if record.get('error') or not (record.get('accepted_asset') or record.get('asset_path')):
        return 'rejected'
    qa = record.get('qa') or {}
    # Re-evaluation after a required HD failure must not rewrite base quality.
    if record.get('enhancement', {}).get('required') and record.get('review', {}).get('reason') == 'required_enhancement':
        return record.get('base_asset_status', 'needs_review')
    if (record.get('status') != 'pass' or qa.get('status') != 'pass'
            or record.get('needs_manual_review') or record.get('review_required')
            or record.get('uncertain') or not (record.get('visible_mask_path') or record.get('mask_path'))):
        return 'needs_review'
    return 'ready'


def normalize_status(record: dict[str, Any]) -> dict[str, Any]:
    obj = dict(record)
    obj['base_asset_status'] = base_status(obj)
    enhancement = obj.get('enhancement', {})
    required_failure = enhancement.get('required') and enhancement.get('status') not in {'ready','preview_only'}
    obj['review'] = {'required': obj['base_asset_status'] != 'ready' or bool(required_failure),
                     'reason': 'required_enhancement' if required_failure else (obj.get('qa') or {}).get('reason','')}
    return obj
