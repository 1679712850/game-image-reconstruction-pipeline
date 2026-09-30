"""Minimal PSD v1 writer with editable RGBA layers, offsets and composite preview."""
from pathlib import Path
import struct
from PIL import Image
from cv.layer_order import ordered_layers


def _u32(value):
    return struct.pack('>I', value)


def _layer(image, name, box):
    channels = [image.getchannel(c).tobytes() for c in ('R', 'G', 'B', 'A')]
    rectangle = struct.pack('>iiii', box['y'], box['x'], box['y']+image.height, box['x']+image.width)
    descriptors = b''.join(struct.pack('>hI', ident, len(data)+2) for ident,data in zip((0,1,2,-1), channels))
    encoded = name.encode('ascii', 'replace')[:255]
    pascal = bytes([len(encoded)])+encoded
    pascal += b'\0'*((-len(pascal)) % 4)
    unicode_name = name.encode('utf-16-be')
    unicode_data = _u32(len(unicode_name)//2)+unicode_name
    extra = _u32(0)+_u32(0)+pascal+b'8BIMluni'+_u32(len(unicode_data))+unicode_data
    extra += b'\0'*(len(unicode_data)%2)
    record = rectangle+struct.pack('>H',4)+descriptors+b'8BIMnorm'+bytes([255,0,0,0])+_u32(len(extra))+extra
    pixels = b''.join(b'\0\0'+data for data in channels)
    return record,pixels


def export_layers(records, width, height, output_path):
    if max(width,height) > 30000:
        raise ValueError('PSD v1 supports at most 30000 pixels per dimension')
    layers = []
    canvas = Image.new('RGBA', (width,height))
    for obj in ordered_layers(records):
        path = obj.get('accepted_asset') or obj.get('asset_path')
        box = obj.get('placement', {}).get('crop_bbox') or obj.get('crop_bbox')
        if not path or not box:
            continue
        with Image.open(path) as source:
            image = source.convert('RGBA')
        if image.size != (box['w'],box['h']):
            raise ValueError('PSD layer dimensions differ from placement')
        layers.append(_layer(image, obj.get('id', 'background'), box))
        canvas.alpha_composite(image,(box['x'],box['y']))
    if len(layers)>32767:
        raise ValueError('Too many PSD layers')
    # Negative count indicates that the fourth merged channel is transparency.
    ordered = list(reversed(layers))
    layer_info = struct.pack('>h', -len(layers))+b''.join(r for r,_ in ordered)+b''.join(p for _,p in ordered)
    layer_info += b'\0'*(len(layer_info)%2)
    section = _u32(len(layer_info))+layer_info+_u32(0)
    header = b'8BPS'+struct.pack('>H6sHIIHH',1,b'\0'*6,4,height,width,8,3)
    merged = b'\0\0'+b''.join(canvas.getchannel(c).tobytes() for c in ('R','G','B','A'))
    path = Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(header+_u32(0)+_u32(0)+_u32(len(section))+section+merged)
    return str(path.resolve())


def export_psd(scene, output_path, *, asset_root=None):
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
    return Path(export_layers(records,scene.scene.width,scene.scene.height,output_path))
