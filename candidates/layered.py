"""Offer scene-aligned model layers as object candidates only with local overlap evidence."""
from pathlib import Path
import numpy as np
from PIL import Image


def layer_jobs(state, objects, root):
    jobs = []
    for index, layer in enumerate(state.get('decomposed_layers', [])):
        if layer.get('mock'):
            continue
        with Image.open(layer['asset_path']) as source:
            image = source.convert('RGBA')
        alpha = np.asarray(image.getchannel('A')) > 8
        if image.size != (state['width'], state['height']) or not alpha.any():
            continue
        for ident, obj in objects.items():
            if not obj.get('asset_path'):
                continue
            box = obj['bbox']; x,y,w,h = (box[k] for k in ('x','y','w','h'))
            if not alpha[y:y+h,x:x+w].any():
                continue
            # Whole-scene opaque backgrounds are not independent object candidates.
            left,top = max(0,x-w//2),max(0,y-h//2)
            right,bottom = min(image.width,x+w+w//2),min(image.height,y+h+h//2)
            if alpha[top:bottom,left:right].sum() < .5*alpha.sum():
                continue
            path = Path(root)/'candidates'/ident/f'layered_{index}_input.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            image.crop((left,top,right,bottom)).save(path)
            jobs.append({'id':ident, 'type':'generation', 'prefix':f'layered_{index}',
                         'existing_path':str(path.resolve()), 'mask':None, 'explicit':False,
                         'prompt':f'Repair this {obj["category"]}; retain the original scene appearance and camera.'})
    return jobs
