"""Task snapshots of cheap model metadata; weight contents are never read."""
from pathlib import Path
import hashlib
import json
from typing import Any


def fingerprint_path(path: str | Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {'path':str(target),'missing':True}
    files: list[tuple[str,int,int]] = []
    candidates = [target] if target.is_file() else sorted(target.rglob('*'))
    for item in candidates:
        if item.is_file() and (target.is_file() or item.suffix.lower() in {'.json','.yaml','.yml','.safetensors','.bin','.pt','.pth'}):
            stat = item.stat()
            files.append((item.name if target.is_file() else str(item.relative_to(target)),stat.st_size,stat.st_mtime_ns))
    manifest = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:20]
    config_hash = hashlib.sha256(json.dumps(config or {},sort_keys=True,default=str).encode()).hexdigest()[:20]
    return {'path':str(target.resolve()),'manifest':manifest,'config_hash':config_hash,
            'files':len(files),'size':sum(item[1] for item in files),
            'mtime_ns':max((item[2] for item in files),default=0)}


def service_versions(settings: Any, name: str, cache: dict) -> list[dict[str,Any]]:
    """Resolve each local model path once per task/config; each new task rescans.

    Hub refs resolve to the cached snapshot when present. In-place weight updates
    during a run are unsupported; restart the run to invalidate its frozen identity.
    """
    section = {'grounding':'grounding','sam':'sam','image_edit':'qwen_image_edit',
               'layered':'qwen_layered','upscale':'upscale','qwen_vl':'qwen_vl'}.get(name)
    options = getattr(settings,section,None) if section else None
    if options is None:
        return []
    config = options.model_dump(mode='json')
    task_key = (section,json.dumps(config,sort_keys=True,default=str))
    if task_key not in cache:
        paths = []
        for field in ('model_id','model_path','quantized_model_path','checkpoint'):
            value = getattr(options,field,None)
            if value and Path(value).exists():
                paths.append(Path(value))
        repo = getattr(options,'repo_id',None) or getattr(options,'model_id',None)
        if repo and not Path(repo).exists():
            hub = settings.cache_dir / ('models--'+repo.replace('/','--'))
            revision = getattr(options,'revision','main')
            ref = hub/'refs'/revision
            if ref.is_file():
                revision = ref.read_text().strip()
            snapshot = hub/'snapshots'/revision
            if snapshot.is_dir():
                paths.append(snapshot)
        cache[task_key] = [fingerprint_path(p,config=config) for p in dict.fromkeys(paths)]
    return cache[task_key]
