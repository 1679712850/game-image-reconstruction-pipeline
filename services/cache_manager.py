"""Content addressed, versioned JSON/NPY caches. No pickle or executable payloads."""
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from PIL import Image
from pydantic import BaseModel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    if isinstance(value, BaseModel):
        return fingerprint(value.model_dump(mode='json'))
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.ndarray):
        return {'array': digest(value.tobytes()), 'dtype': str(value.dtype), 'shape': value.shape}
    if isinstance(value, Image.Image):
        return {'image': digest(value.tobytes()), 'mode': value.mode, 'size': value.size}
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and len(value) < 4096:
        try:
            path = Path(value)
            if path.is_file():
                if path.suffix.lower() in {'.pth','.pt','.bin','.safetensors'}:
                    stat = path.stat()
                    return {'model_file':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
                return {'file': digest(path.read_bytes())}
        except OSError:
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value


class CacheManager:
    def __init__(self, config, output_root):
        self.config = config
        self.output_root = Path(output_root).resolve()
        self.root = (config.directory or self.output_root/'cache').resolve()
        self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0, 'corrupt': 0})

    def key(self, namespace, model, parameters):
        payload = {'pipeline_version': self.config.pipeline_version, 'prompt_version': self.config.prompt_version,
                   'namespace': namespace, 'model': model, 'parameters': fingerprint(parameters)}
        return digest(json.dumps(payload, sort_keys=True, allow_nan=False).encode())

    def _encode(self, value, blobs):
        if isinstance(value, BaseModel):
            return {'__type__': 'model', 'class': value.__class__.__name__, 'data': self._encode(value.model_dump(mode='json'), blobs)}
        if isinstance(value, np.ndarray):
            buffer = BytesIO(); np.save(buffer, value, allow_pickle=False)
            raw = buffer.getvalue(); key = digest(raw); blobs[key] = raw
            return {'__type__': 'array', 'blob': key}
        if isinstance(value, tuple):
            return {'__type__': 'tuple', 'data': [self._encode(v, blobs) for v in value]}
        if isinstance(value, list):
            return [self._encode(v, blobs) for v in value]
        if isinstance(value, dict):
            return {k: self._encode(v, blobs) for k, v in value.items()}
        if isinstance(value, str):
            try:
                path = Path(value).resolve()
                if path.is_file() and path.is_relative_to(self.output_root):
                    raw = path.read_bytes(); key = digest(raw); blobs[key] = raw
                    return {'__type__': 'file', 'blob': key, 'relative': str(path.relative_to(self.output_root))}
            except OSError:
                pass
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _decode(self, value, folder):
        if isinstance(value, list):
            return [self._decode(v, folder) for v in value]
        if not isinstance(value, dict):
            return value
        kind = value.get('__type__')
        if kind in {'file', 'array'}:
            key = value['blob']
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('Invalid cache blob key')
            raw = (folder/key).read_bytes()
            if digest(raw) != key:
                raise ValueError('Cache checksum mismatch')
            if kind == 'array':
                return np.load(BytesIO(raw), allow_pickle=False)
            path = (self.output_root/value['relative']).resolve()
            if not path.is_relative_to(self.output_root):
                raise ValueError('Cache output escapes task directory')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            return str(path)
        if kind == 'tuple':
            return tuple(self._decode(v, folder) for v in value['data'])
        if kind == 'model':
            from schemas.scene import SceneAnalysis
            from schemas.scene_qa import SceneReviewDecision, CategoryReview
            from schemas.candidate import CandidateQA
            types = {cls.__name__: cls for cls in (SceneAnalysis, SceneReviewDecision, CategoryReview, CandidateQA)}
            return types[value['class']].model_validate(self._decode(value['data'], folder))
        return {k: self._decode(v, folder) for k, v in value.items()}

    def get(self, namespace, key):
        if self.config.enabled:
            folder = self.root/namespace/key
            try:
                value = self._decode(json.loads((folder/'result.json').read_text()), folder)
                self.stats[namespace]['hit'] += 1
                return True, value
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, OSError, TypeError):
                self.stats[namespace]['corrupt'] += 1
        self.stats[namespace]['miss'] += 1
        return False, None

    def put(self, namespace, key, value):
        if not self.config.enabled:
            return
        blobs = {}; payload = self._encode(value, blobs)
        folder = self.root/namespace/key; folder.mkdir(parents=True, exist_ok=True)
        for name, raw in blobs.items():
            self._atomic(folder/name, raw)
        self._atomic(folder/'result.json', json.dumps(payload, allow_nan=False).encode())

    @staticmethod
    def _atomic(path, raw):
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.cache-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
            os.replace(temporary, path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
