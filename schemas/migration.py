"""Normalize portable manifests from schema 1.1 to the current 1.2 shape."""
from copy import deepcopy
from typing import Any
from app.asset_status import base_status

CURRENT_SCHEMA = '1.2'


def migrate_schema_v1_1_to_v1_2(payload: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(payload)
    data['schema_version'] = CURRENT_SCHEMA
    data.setdefault('completion_metrics', {})
    data.setdefault('pipeline_status', 'completed')
    for obj in [*data.get('objects',[]), *data.get('environment_effects',[])]:
        geometry = obj.setdefault('geometry', {})
        geometry.setdefault('visible_bbox', obj.get('bbox_visible') or obj.get('bbox'))
        geometry.setdefault('visible_mask', obj.get('visible_mask_path') or obj.get('mask_path'))
        geometry.setdefault('full_asset_bbox', obj.get('placement',{}).get('crop_bbox') or obj.get('crop_bbox') or obj.get('bbox_full'))
        geometry.setdefault('full_asset_canvas', obj.get('logical_size'))
        geometry.setdefault('full_asset_mask', obj.get('reconstructed_mask_path'))
        obj.setdefault('base_asset_status', base_status(obj))
        obj.setdefault('enhancement', {'requested':bool(obj.get('hd_asset_path')), 'status':'ready' if obj.get('hd_asset_path') else 'disabled', 'required':False})
    return data


def normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    version = str(payload.get('schema_version','1.1'))
    if version == '1.1':
        return migrate_schema_v1_1_to_v1_2(payload)
    if version != CURRENT_SCHEMA:
        raise ValueError(f'Unsupported scene schema {version}; expected 1.1 or {CURRENT_SCHEMA}')
    return migrate_schema_v1_1_to_v1_2(payload)
