"""Rebuild only exclusive visible layers; inferred terrain remains separate."""
from pathlib import Path
from app.paths import read_rgba
from cv.metrics import reconstruction_similarity
from cv.reconstruct import reconstruct_scene as rebuild
from cv.completion_metrics import completion_metrics


def make_reconstruct_scene(enabled, p1=None):
    def reconstruct_scene(state):
        if not enabled:
            return {'reconstruction_path':'','reconstruction_score':None}
        records=[*state.get('terrain_layers',[]),*state.get('objects',[])]
        ownership=dict(state.get('ownership',{}))
        residual=ownership.get('residual_background',{}).get('asset_path')
        if residual:
            records.append({'asset_path':residual,'crop_bbox':{'x':0,'y':0,'w':state['width'],'h':state['height']},'z_order':-2})
        path=rebuild(state['width'],state['height'],records,Path(state['output_dir'])/'reconstruction.png')
        score=reconstruction_similarity(read_rgba(state['source_path']),read_rgba(path))
        if ownership:
            from diagnostics.reconstruction_diff import write_reconstruction_diff
            ownership['reconstruction_diff']=write_reconstruction_diff(state['source_path'],path,
                Path(state['output_dir'])/'diagnostics'/'reconstruction_diff.png', lpips_enabled=bool(p1 and p1.lpips))
        updates = {'reconstruction_path':path,'reconstruction_score':score,'ownership':ownership}
        metrics = completion_metrics({**state, **updates})
        status = metrics['status']
        return {**updates, 'completion_metrics': metrics, 'pipeline_status': status}
    return reconstruct_scene
