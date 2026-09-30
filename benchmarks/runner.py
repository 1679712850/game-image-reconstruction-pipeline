"""Run the production graph, or evaluate its existing portable manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from PIL import Image
import yaml
from benchmarks.schema import Annotation, EvaluationConfig
from benchmarks.metrics import evaluate
from benchmarks.report import regression, visualizations, write_report
from benchmarks.pipeline_metrics import prediction_inventory, roi_completion, truncation


def run(annotation_path: Path, output: Path, pipeline_path: Path, models_path: Path,
        evaluation_path: Path | None = None, manifest_path: Path | None = None,
        baseline_path: Path | None = None, mock: bool = False) -> dict[str,Any]:
    annotation_bytes = annotation_path.read_bytes()
    annotation = Annotation.model_validate_json(annotation_bytes)
    source = (annotation_path.parent/annotation.image).resolve()
    config = EvaluationConfig.model_validate(yaml.safe_load(evaluation_path.read_text()) or {}) if evaluation_path else EvaluationConfig()
    output.mkdir(parents=True,exist_ok=True)
    if manifest_path is None:
        from agent.graph import build_graph
        from app.config import PipelineConfig, load_config
        from app.models import load_models
        from services.runtime import ServiceBundle
        values = load_config(pipeline_path).model_dump()
        values['mock'] = mock
        values['scene_loop']['max_objects'] = None
        pipeline = PipelineConfig.model_validate(values)
        models = None if mock else load_models(models_path)
        services = ServiceBundle.create(mock, models,pipeline.scene_loop.reviewer)
        (output/'resolved_config.json').write_text(json.dumps({'pipeline':pipeline.model_dump(mode='json'),
            'models':models.model_dump(mode='json') if models else None},indent=2)+'\n')
        state = build_graph(pipeline,services).invoke({'source_path':str(source),'output_dir':str((output/'pipeline').resolve())})
        manifest_path = Path(state['scene_json'])
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    historical_inventory = root/'metadata'/'detections.json'
    if 'instances' not in manifest.get('detection',{}) and historical_inventory.exists():
        manifest.setdefault('detection',{})['instances'] = json.loads(historical_inventory.read_text())
    predictions, inventory_kind = prediction_inventory(manifest)
    with Image.open(source) as image:
        size = image.size
    if annotation.roi[2] > size[0] or annotation.roi[3] > size[1]:
        raise ValueError('Annotation ROI exceeds source image')
    result = evaluate(annotation,predictions,size,annotation_path.parent,root,config)
    performance_path = root/'performance_report.json'
    performance = json.loads(performance_path.read_text()) if performance_path.exists() else {}
    limits = truncation(manifest,root)
    result.update(scene_id=annotation.scene_id, annotation_status=annotation.annotation_status,
                  annotation_digest=hashlib.sha256(annotation_bytes).hexdigest(), roi=annotation.roi,
                  image_digest=hashlib.sha256(source.read_bytes()).hexdigest(), evaluation_scope=annotation.scope,
                  inventory_kind=inventory_kind, resource_failures=manifest.get('resource_failures',[]),
                  roi_completion=roi_completion(source,annotation,manifest,root),
                  scan_cost=manifest.get('detection',{}).get('budget',{}),
                  scan_groups=[r.get('scan_cost',{}) for r in manifest.get('detection',{}).get('rounds',[])],
                  budget_exhausted=any(r.get('budget',{}).get('exhausted',False) for r in manifest.get('detection',{}).get('rounds',[])),
                  mock=manifest.get('mock'), completion=manifest.get('completion_metrics',{}),
                  truncated=limits.get('truncated'), objects_before_limit=limits.get('objects_before_limit'),
                  objects_after_limit=limits.get('objects_after_limit'),
                  performance={'runtime':performance.get('wall_time'), 'peak_ram_mb':performance.get('peak_ram_mb'),
                               'peak_vram_mb':performance.get('peak_vram_mb')},
                  config=config.model_dump(), production_manifest=str(manifest_path.resolve()),
                  metric_definitions={'matching':'Confidence-ordered category-aware one-to-one IoU matching within ROI; duplicate predictions also lower precision.',
                  'duplicates':'Unmatched predictions overlapping an already matched GT at IoU threshold / predictions.',
                  'fragmentation':'GT instances containing at least two distinct nonoverlapping partial boxes / instance GT; heuristic, not amodal GT.',
                  'size':'BBox area / original scene area; no resizing-dependent thresholds.',
                  'mask_quality':'Visible-mask metrics conditional on a matched, mask-annotated GT; missing predicted masks score zero. Unmatched GT reported as misses.',
                  'semantic_coverage':'QA-passed visible pixel union; GT semantic-region scores are separate and unavailable without mask GT.'})
    if baseline_path:
        result['gate'] = regression(json.loads(baseline_path.read_text()),result,config.thresholds)
    # A first baseline run can also recover from a model failure and export an
    # incomplete manifest. Never report that run as usable just because no
    # comparison baseline was supplied.
    if result['resource_failures']:
        gate = result.setdefault('gate', {'checks': {}})
        gate['status'] = 'FAIL'
        gate['checks']['execution'] = {'status': 'FAIL', 'reason': 'Resource recovery prevented full inference'}
    visualizations(source,annotation,result,output/'visualizations',annotation_path.parent,root)
    write_report(result,output)
    return result


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pipeline',type=Path,default=Path('config/pipeline.yaml'))
    parser.add_argument('--models',type=Path,default=Path('config/models.yaml'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--manifest',type=Path,help='Evaluate existing production output without inference')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--mock',action='store_true',help='Mechanics only; never real-image quality evidence')
    args = parser.parse_args()
    try:
        result = run(args.annotation,args.output,args.pipeline,args.models,args.config,args.manifest,args.baseline,args.mock)
    except (OSError,ValueError,RuntimeError) as error:
        args.output.mkdir(parents=True,exist_ok=True)
        failure = {'status':'failed','error':str(error)}
        (args.output/'benchmark_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        print('[Benchmark] '+str(error))
        return 1
    print(json.dumps({k:result[k] for k in ('scene_id','detection','mask_quality','performance')},indent=2))
    return 2 if result.get('gate',{}).get('status') == 'FAIL' else 0

if __name__ == '__main__':
    raise SystemExit(main())
