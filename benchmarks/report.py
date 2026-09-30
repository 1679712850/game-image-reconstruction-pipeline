"""Readable, separate quality dimensions and a non-scalar regression gate."""
import json
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw
from diagnostics.bbox_visualizer import open_image, xyxy
from benchmarks.schema import Annotation


def regression(baseline: dict, current: dict, thresholds: dict[str,float]) -> dict:
    checks = {}
    fields = [('recall','detection','recall_drop',False), ('precision','detection','precision_drop',False),
              ('duplicate_rate','detection','duplicate_rate_increase',True), ('runtime','performance','runtime_increase',True),
              ('boundary_fscore','mask_quality','boundary_fscore_drop',False)]
    for field,section,limit,higher_bad in fields:
        old,new = baseline.get(section,{}).get(field),current.get(section,{}).get(field)
        if old is None or new is None:
            checks[field] = {'status':'WARN','reason':'measurement unavailable'}; continue
        delta = (new-old)/max(old,1e-9) if field == 'runtime' else new-old
        regression_amount = delta if higher_bad else -delta
        checks[field] = {'baseline':old,'current':new,'delta':delta,'limit':thresholds.get(limit,0),
                         'status':'FAIL' if regression_amount > thresholds.get(limit,0) else 'PASS'}
    if any(baseline.get(k) != current.get(k) for k in ('annotation_digest','image_digest','scene_id','inventory_kind','config')):
        checks['comparability'] = {'status':'FAIL','reason':'annotation or scene changed'}
    if any(r.get('truncated') is not False or r.get('mock') or r.get('annotation_status') != 'reviewed' for r in (baseline,current)):
        checks['validity'] = {'status':'WARN','reason':'truncated/mock/draft GT; not an acceptance gate'}
    if any(r.get('budget_exhausted') for r in (baseline,current)):
        checks['scan_budget'] = {'status':'WARN','reason':'Scan budget exhausted; inspect omitted work'}
    if any(r.get('resource_failures') for r in (baseline,current)):
        checks['execution'] = {'status':'FAIL','reason':'Resource recovery prevented full inference'}
    status = 'FAIL' if any(c['status']=='FAIL' for c in checks.values()) else 'WARN' if any(c['status']=='WARN' for c in checks.values()) else 'PASS'
    return {'status':status,'checks':checks}


def visualizations(source: Path, annotation: Annotation, report: dict, directory: Path, annotation_root: Path, prediction_root: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    predictions = report['predictions']
    tp = {m['prediction_id'] for m in report['matches']}
    fragments = {i for f in report['fragments'] for i in f['prediction_ids']}
    gt = [o.model_dump() for o in annotation.objects]
    selections = {
        'gt': (gt,'yellow'), 'predicted':(predictions,'lime'),
        'needs_review':([p for p in predictions if p.get('base_asset_status') == 'needs_review'],'orange'),
        'missed':([g for g in gt if g['id'] in report['missed']],'yellow'),
        'false_positives':([p for p in predictions if p['id'] in report['false_positives']],'red'),
        'duplicates':([p for p in predictions if p['id'] in report['duplicates']],'purple'),
        'fragmented':([p for p in predictions if p['id'] in fragments],'orange'),
        'poor_masks':([p for p in predictions if p['id'] in report['poor_masks']],'orange'),
        'accepted_assets':([p for p in predictions if p.get('base_asset_status')=='ready' and p['id'] in tp],'lime')}
    for name,(records,color) in selections.items():
        image = open_image(source).crop(annotation.roi).convert('RGBA')
        from benchmarks.metrics import roi_mask
        from PIL import ImageColor
        for record in records:
            mask_path = record.get('mask') if name == 'gt' else record.get('visible_mask_path') or record.get('source_mask_path')
            if mask_path:
                base = annotation_root if name == 'gt' else prediction_root
                with Image.open(source) as original:
                    mask = roi_mask(base/mask_path,original.size,annotation.roi)
                alpha = Image.fromarray((mask*70).astype('uint8'))
                overlay = Image.new('RGBA',image.size,ImageColor.getrgb(color)+(0,)); overlay.putalpha(alpha)
                image = Image.alpha_composite(image,overlay)
        draw = ImageDraw.Draw(image)
        for record in records:
            box = xyxy(record)
            if box:
                box = (box[0]-annotation.roi[0],box[1]-annotation.roi[1],box[2]-annotation.roi[0],box[3]-annotation.roi[1])
                draw.rectangle(box,outline=color,width=2)
                draw.text((box[0]+2,box[1]+2),record['id'],fill=color,stroke_width=1,stroke_fill='black')
        image.save(directory/f'{name}.png')


def write_report(report: dict[str,Any], directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'benchmark_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines = [f"# Benchmark: {report['scene_id']}", '',
             f"GT: {report['annotation_status']}; mock: {report.get('mock')}; truncated: {report.get('truncated')}",
             '', 'Mask metrics are measured only where masks are annotated; bbox GT does not imply pixel accuracy.', '']
    for section in ('roi_completion','completion','detection','mask_quality','semantic_regions','scan_cost','performance','gate','metric_definitions'):
        lines += [f'## {section}', '', '```json', json.dumps(report.get(section,{}),ensure_ascii=False,indent=2), '```', '']
    lines += ['## Missed objects','', ', '.join(report['missed']) or 'None', '', '## Visualizations','']
    lines += [f'![{p.stem}](visualizations/{p.name})' for p in sorted((directory/'visualizations').glob('*.png'))]
    (directory/'benchmark_report.md').write_text('\n'.join(lines)+'\n')
