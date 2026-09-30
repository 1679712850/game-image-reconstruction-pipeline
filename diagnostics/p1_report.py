"""Portable P1 artifacts and summary; residual coverage is never semantic recall."""
from collections import Counter
from pathlib import Path
import json
import shutil
from app.paths import relative_asset


def portable(value, root):
    if isinstance(value,dict): return {k:portable(v,root) for k,v in value.items()}
    if isinstance(value,list): return [portable(v,root) for v in value]
    if isinstance(value,str) and value.startswith('/'):
        try: return relative_asset(value,root)
        except ValueError: return value
    return value


def export_p1(state, *, mock=None):
    root=Path(state['output_dir']); metadata=root/'metadata'; metadata.mkdir(parents=True,exist_ok=True)
    objects=[]
    for record in state.get('objects',[]):
        obj=dict(record)
        if obj.get('asset_path') and obj.get('element_type')!='effect':
            path=root/'objects'/f"{obj['id']}.png"; path.parent.mkdir(parents=True,exist_ok=True)
            if Path(obj['asset_path']).resolve() != path.resolve():
                shutil.copyfile(obj['asset_path'],path)
            obj['asset_path']=str(path.resolve())
        obj.update(instance_id=obj['id'],sub_category=obj.get('subtype'),layer=obj.get('layer_group'))
        objects.append(obj)
    owner=state['ownership']
    reconstruction=owner.get('reconstruction_diff',{})
    retry=Counter(r.get('retry_reason','UNKNOWN').lower() for r in state.get('retry_history',[]))
    instances = [o for o in objects if o.get('element_type') != 'effect']
    summary={'detections':len(state.get('all_detections',state.get('detections',[]))), 'instances':len(instances),
             'effects':len(objects)-len(instances),
             'terrain_layers':len(state.get('terrain_layers',[])),
             'coverage_ratio':owner['coverage_ratio'],'unassigned_ratio':owner['unassigned_ratio'],
             'visible_coverage_ratio':owner['visible_coverage_ratio'],
             'overlap_ratio':owner['overlap_ratio'],'resolved_overlap_ratio':owner['resolved_overlap_ratio'],
             'qa_failed_instances':sum(o['status']!='pass' for o in instances),
             'remaining_missed_candidates':sum(r.get('failure_type','MISSED_DETECTION') == 'MISSED_DETECTION'
                                                for r in state.get('missed_object_candidates',[])),
             'remaining_problem_regions':len(state.get('missed_object_candidates',[])),
             'detection_density':len(objects)/(state['width']*state['height']/1_000_000),
             'uncertain_categories':sum(o.get('uncertain',False) for o in objects),
             'terrain_visible_pixels':sum(t['visible_pixel_count'] for t in state.get('terrain_layers',[])),
             'retry':dict(retry),'reconstruction':reconstruction,
             'definition':owner['definition'], 'mock':mock if mock is not None else state.get('detection_coverage_review',{}).get('mock_local_detection_skipped',False)}
    summary['completion_metrics'] = state.get('completion_metrics', {})
    summary['pipeline_status'] = state.get('pipeline_status', 'completed')
    failures=[]
    if owner['unassigned_ratio']>state.get('p1_thresholds',{}).get('unassigned',.01): failures.append('UNASSIGNED_REGION')
    if owner['overlap_ratio']>state.get('p1_thresholds',{}).get('overlap',.01): failures.append('PIXEL_CONFLICT')
    failures.extend(r.get('failure_type','MISSED_DETECTION') for r in state.get('missed_object_candidates',[]))
    summary['terrain_qa_failed_layers'] = sum(t.get('qa_status') == 'manual_review' for t in state.get('terrain_layers', []))
    summary['vision_review_failed'] = any(r.get('action') == 'vision_review_failed' for r in state.get('retry_history', []))
    if reconstruction.get('alpha_gap', 0) > 0 or reconstruction.get('extra_alpha', 0) > 0:
        failures.append('RECONSTRUCTION_ALPHA_GAP')
    summary['scene_failure_types']=list(dict.fromkeys(failures))
    summary['status']='manual_review' if (failures or summary['qa_failed_instances'] or
                                          summary['terrain_qa_failed_layers'] or summary['vision_review_failed'] or
                                          summary['uncertain_categories']) else 'pass'
    payloads={'detections':state.get('all_detections',state.get('detections',[])), 'instances':objects,
              'terrain':state.get('terrain_layers',[]),'ownership':owner,'retries':state.get('retry_history',[]),
              'summary':summary,'missed_object_candidates':state.get('missed_object_candidates',[])}
    for name,data in payloads.items():
        (metadata/f'{name}.json').write_text(json.dumps(portable(data,root),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report=root/'diagnostics'/'report.html'
    import html
    figures=''.join(f'<figure><figcaption>{name}</figcaption><img src="{name}.png"></figure>' for name in
                    ('detection_boxes','coverage_map','ownership_map','overlap_heatmap','unassigned_regions','qa_failed_regions','retry_regions','reconstruction_diff')
                    if (root/'diagnostics'/f'{name}.png').exists())
    marker = '<!-- P1 REPORT -->'
    base = report.read_text(encoding='utf-8').split(marker)[0] if report.exists() else (
        '<!doctype html><meta charset="utf-8"><title>P1 scene diagnostics</title>'
        '<style>body{font:15px system-ui;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(base+marker+'<h1>P1 scene decomposition</h1><p>Green: detection; yellow: low confidence; red: missed-object or unassigned region; gray: terrain. Residual fallback is excluded from semantic coverage.</p><pre>'+html.escape(json.dumps(portable(summary,root),ensure_ascii=False,indent=2))+f'</pre>{figures}',encoding='utf-8')
    return summary
