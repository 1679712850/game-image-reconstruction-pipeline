"""Minimal PSD v1 writer with editable RGBA layers, offsets and composite preview."""
from pathlib import Path
import struct
from PIL import Image
from cv.layer_order import ordered_layers
import logging


def estimate_export_memory(records: list[dict], width: int, height: int) -> int:
    """Conservative live RAM estimate: merged canvas + largest decoded/encoded layer."""
    largest = max(((r.get('placement', {}).get('crop_bbox') or r.get('crop_bbox') or {}).get('w',0) *
                   (r.get('placement', {}).get('crop_bbox') or r.get('crop_bbox') or {}).get('h',0)
                   for r in records), default=0)
    return width*height*8 + largest*16


def _u32(value):
    return struct.pack('>I', value)


def _layer_record(width: int, height: int, name: str, box: dict) -> bytes:
    rectangle = struct.pack('>iiii',box['y'],box['x'],box['y']+height,box['x']+width)
    descriptors = b''.join(struct.pack('>hI',ident,width*height+2) for ident in (0,1,2,-1))
    encoded = name.encode('ascii','replace')[:255]
    pascal = bytes([len(encoded)])+encoded
    pascal += b'\0'*((-len(pascal))%4)
    unicode_name = name.encode('utf-16-be')
    unicode_data = _u32(len(unicode_name)//2)+unicode_name
    extra = _u32(0)+_u32(0)+pascal+b'8BIMluni'+_u32(len(unicode_data))+unicode_data
    extra += b'\0'*(len(unicode_data)%2)
    return rectangle+struct.pack('>H',4)+descriptors+b'8BIMnorm'+bytes([255,0,0,0])+_u32(len(extra))+extra


def export_layers(records: list[dict], width: int, height: int, output_path: str | Path,
                  *, memory_budget_mb: float = 2048, low_memory: bool = True) -> str:
    """Spool channels to disk, retain small headers only, crop transparent borders."""
    import tempfile
    import os
    import json
    if max(width,height) > 30000:
        raise ValueError('PSD v1 supports at most 30000 pixels per dimension')
    estimate = estimate_export_memory(records,width,height)
    if estimate > memory_budget_mb*1024**2:
        logging.getLogger(__name__).warning('[Export] PSD estimate %.1f MiB exceeds budget %s MiB; disk spooling enabled',estimate/1024**2,memory_budget_mb)
        if not low_memory:
            raise MemoryError('PSD estimated memory exceeds export.memory_budget_mb')
    path = Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    layers: list[tuple[bytes, int, int]] = []
    canvas = Image.new('RGBA',(width,height))
    cropped_pixels = 0
    with tempfile.TemporaryFile(dir=path.parent) as pixels:
        for obj in ordered_layers(records):
            asset = obj.get('accepted_asset') or obj.get('asset_path')
            box = obj.get('placement',{}).get('crop_bbox') or obj.get('crop_bbox')
            if not asset or not box:
                continue
            with Image.open(asset) as source:
                rgba = source.convert('RGBA')
            if rgba.size != (box['w'],box['h']):
                raise ValueError('PSD layer dimensions differ from placement')
            bounds = rgba.getchannel('A').getbbox()
            if bounds is None:
                rgba.close(); continue
            image = rgba.crop(bounds); rgba.close()
            placement = {'x':box['x']+bounds[0], 'y':box['y']+bounds[1]}
            # Header construction keeps no pixel bytes; channels are written individually.
            record = _layer_record(image.width,image.height,obj.get('id','background'),placement)
            offset = pixels.tell()
            for channel in ('R','G','B','A'):
                pixels.write(b'\0\0')
                pixels.write(image.getchannel(channel).tobytes())
            layers.append((record,offset,pixels.tell()-offset))
            cropped_pixels += image.width*image.height
            canvas.alpha_composite(image,(placement['x'],placement['y']))
            image.close()
        if len(layers)>32767:
            raise ValueError('Too many PSD layers')
        length = 2+sum(len(r)+n for r,_,n in layers)
        padded = length+(length%2)
        if padded+8 > 0xffffffff:
            raise ValueError("PSD v1 layer section exceeds 4 GiB; PSB is not supported")
        header = b'8BPS'+struct.pack('>H6sHIIHH',1,b'\0'*6,4,height,width,8,3)
        fd,temporary = tempfile.mkstemp(dir=path.parent,suffix='.psd.tmp')
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(header+_u32(0)+_u32(0)+_u32(4+padded+4)+_u32(padded))
                stream.write(struct.pack('>h',-len(layers)))
                for record,_,_ in reversed(layers):
                    stream.write(record)
                for _,offset,count in reversed(layers):
                    pixels.seek(offset)
                    while count:
                        chunk = pixels.read(min(count,1024*1024)); stream.write(chunk); count -= len(chunk)
                stream.write(b'\0'*(length%2)+_u32(0)+b'\0\0')
                for channel in ('R','G','B','A'):
                    stream.write(canvas.getchannel(channel).tobytes())
            os.replace(temporary,path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
            canvas.close()
    path.with_suffix('.export.json').write_text(json.dumps({
        'estimated_memory_mb':estimate/1024**2,'memory_budget_mb':memory_budget_mb,
        'budget_exceeded':estimate>memory_budget_mb*1024**2,
        'mode':'disk_spool_cropped_channels','layer_count':len(layers),'encoded_pixels':cropped_pixels},indent=2)+'\n')
    return str(path.resolve())


def export_psd(scene, output_path, *, asset_root=None, memory_budget_mb=2048, low_memory=True):
    """Export a portable SceneManifest; asset_root defaults to the manifest directory."""
    root = Path(asset_root or Path(output_path).parent)
    records = []
    for obj in [*scene.objects, *scene.environment_effects]:
        data = obj.model_dump()
        for key in ('accepted_asset', 'asset_path'):
            if data.get(key):
                data[key] = str(root/data[key])
        records.append(data)
    for layer in scene.terrain_layers:
        records.append({**layer, 'asset_path': str(root/layer['asset_path'])})
    residual = scene.ownership.get('residual_background', {}).get('asset_path')
    if residual:
        records.append({'id': 'residual_background', 'asset_path': str(root/residual), 'z_order': -2,
                        'crop_bbox': {'x':0,'y':0,'w':scene.scene.width,'h':scene.scene.height}})
    return Path(export_layers(records,scene.scene.width,scene.scene.height,output_path,
                             memory_budget_mb=memory_budget_mb,low_memory=low_memory))
