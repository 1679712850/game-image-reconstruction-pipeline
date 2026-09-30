"""Instrument existing service boundaries without serializing clients/models in state."""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import inspect
import time
import threading
import sys

from services.cache_manager import CacheManager
from services.model_manager import ModelManager, is_oom
from services.profiler import Profiler

_active = ContextVar('pipeline_execution', default=None)
_depth = ContextVar('pipeline_service_depth', default=0)
METHODS = {
    'vlm': ('vlm', ['analyze_scene']),
    'grounding': ('detection', ['detect', 'detect_round', 'detect_p0', '_infer_image']),
    'sam': ('segmentation', ['segment', 'segment_local', 'predict_candidates']),
    'image_edit': ('generation', ['complete_object', 'generate_object']),
    'layered': ('generation', ['decompose_layers']),
    'reviewer': ('qa', ['review', 'classify_crop', 'review_candidate']),
    'upscale': ('upscale', ['upscale']),
}


class ExecutionRuntime:
    """Graph-local serialized device access; per-invocation reports survive checkpoints."""
    def __init__(self, config, services):
        self.config = config
        self.services = services
        self.profiler = Profiler()
        self.manager = ModelManager(config.resources, self.profiler)
        self.cache = None
        self.instance_id = None
        self.retry_count = 0
        self.root = None
        self.lock = threading.RLock()
        self._sessions = {}
        for name, (namespace, methods) in METHODS.items():
            service = getattr(services, name, None)
            if service is None:
                continue
            self.manager.register(name, service)
            for method in methods:
                if hasattr(service, method):
                    self._wrap(service, name, namespace, method)

    def _wrap(self, service, name, namespace, method):
        function = getattr(service, method)
        if hasattr(function, '_pipeline_original'):
            function = function._pipeline_original
        @wraps(function)
        def invoke(*args, **kwargs):
            current = _active.get()
            if current is None or (_depth.get() and method != '_infer_image'):
                return function(*args, **kwargs)
            token = _depth.set(_depth.get()+1)
            try:
                return current.infer(service, name, namespace, method, function, args, kwargs)
            finally:
                _depth.reset(token)
        invoke._pipeline_original = function
        setattr(service, method, invoke)

    def infer(self, service, name, namespace, method, function, args, kwargs):
        parameters = dict(inspect.signature(function).bind(*args, **kwargs).arguments)
        instance_id = self.instance_id
        items = parameters.get('detections', [])
        if instance_id is None and len(items) == 1:
            instance_id = items[0].get('id')
        for key in ('output_path', 'output_dir'):
            if parameters.get(key) is not None:
                target = Path(parameters[key]).resolve()
                parameters[key] = str(target.relative_to(self.root)) if target.is_relative_to(self.root) else str(target)
        identity = {'adapter': service.__class__.__module__+'.'+service.__class__.__qualname__,
                    'method': method, 'mock': getattr(service, 'mock', None), 'backend': getattr(service, 'backend', None),
                    'settings': getattr(service, 'config', None).model_dump(mode='json') if hasattr(getattr(service, 'config', None), 'model_dump') else None, 'prompt': getattr(service, '_prompt', None)}
        # In-place local model updates invalidate the key even when the path is unchanged.
        settings = getattr(service, 'config', None)
        if settings is not None and hasattr(settings, 'model_dump'):
            model_paths = []
            for section in ('grounding', 'sam', 'qwen_image_edit', 'qwen_layered'):
                options = getattr(settings, section, None)
                if options is None:
                    continue
                for field in ('model_id', 'model_path', 'quantized_model_path', 'checkpoint'):
                    value = getattr(options, field, None)
                    if value:
                        path = Path(value)
                        if path.is_dir():
                            model_paths.extend((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin', '.pt'})
                        elif path.is_file():
                            model_paths.append((str(path), path.stat().st_size, path.stat().st_mtime_ns))
            identity['local_versions'] = model_paths
        key = self.cache.key(namespace, identity, parameters)
        hit, result = self.cache.get(namespace, key)
        if hit:
            with self.profiler.span(name+'.'+method, 'cache', model=name, instance_id=self.instance_id, cache_hit=True):
                pass
            return result
        for attempt in range(self.config.resources.max_oom_retry+1):
            started = len(self.profiler.events)
            outer_detection = name == 'grounding' and method != '_infer_image'
            empty_segmentation = method in {'segment', 'segment_local'} and parameters.get('detections') == []
            managed = name in self.manager.entries and not outer_detection and not empty_segmentation
            try:
                with self.profiler.span(name+'.'+method, 'service' if outer_detection else 'inference', model=name, instance_id=instance_id,
                    retry_count=max(self.retry_count, attempt), oom_retry=attempt, cache_hit=False) as event:
                    # Keep load timing separate from actual inference.
                    if managed:
                        self.manager.load(name)
                        self.manager.entries[name].active += 1
                    try:
                        torch = sys.modules.get('torch')
                        gpu_start = gpu_end = None
                        if torch is not None and torch.cuda.is_available() and managed and self.manager.entries[name].status == 'GPU':
                            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            gpu_start.record()
                        result = function(*args, **kwargs)
                        if gpu_start is not None:
                            gpu_end.record(); gpu_end.synchronize()
                            event['gpu_time'] = gpu_start.elapsed_time(gpu_end)/1000
                        else:
                            event['gpu_time'] = None
                    finally:
                        if managed:
                            self.manager.entries[name].active -= 1
                    event['exclusive_duration'] = max(0, time.time()-event['start_time']-sum(
                        e['duration'] for e in self.profiler.events[started:] if e['kind'] in {'load', 'postprocess'}))
                def failed(value):
                    if isinstance(value, dict):
                        return bool(value.get('error') or value.get('failed_tiles')) or any(failed(v) for v in value.values())
                    if isinstance(value, (tuple, list)):
                        return any(failed(v) for v in value)
                    return False
                # Degraded inference has different effective parameters. Do not cache it
                # under the original full-quality key or persist partial failures.
                if attempt == 0 and not failed(result) and getattr(service, '_inference_scale', 1) == 1:
                    self.cache.put(namespace, key, result)
                return result
            except Exception as error:
                if not is_oom(error):
                    raise
                if attempt >= self.config.resources.max_oom_retry:
                    from services.model_support import ModelUnavailableError
                    raise ModelUnavailableError(f'{name} OOM recovery exhausted after {attempt} retries') from error
                self.manager.recover(name, attempt+1)
        raise AssertionError('unreachable')

    def run_node(self, name, function, state):
        with self.lock:
            root = Path(state['output_dir']).resolve()
            if name == 'load_image' or root not in self._sessions:
                self._sessions[root] = (Profiler(), CacheManager(self.config.cache, root))
            self.profiler, self.cache = self._sessions[root]
            self.manager.profiler = self.profiler
            self.root = root
            self.instance_id = None
            self.retry_count = state.get('retry_count', 0)+1 if name == 'retry_objects' else 0
            self.profiler.start_sampling()
            token = _active.set(self)
            try:
                with self.profiler.span(name):
                    try:
                        updates = function(state)
                    except Exception as error:
                        from services.resource_fallback import recover_node
                        updates = recover_node(name, state, error, self.services)
                    self.manager.end_stage(final=name == 'export')
                paths = self.profiler.write(root, self.cache.stats, self.manager.states()) if name == 'export' else None
                if paths:
                    updates.update(performance_report_path=paths[0], timeline_path=paths[1])
                    updates['exported_assets'] = sorted(set(updates.get('exported_assets', [])) | set(paths))
                return updates
            except Exception:
                self.manager.end_stage(final=True)
                try:
                    self.profiler.write(root, self.cache.stats, self.manager.states())
                except OSError:
                    pass  # Preserve the original failure if the output disk is full.
                raise
            finally:
                self.profiler.stop_sampling()
                _active.reset(token)
                if name == 'export':
                    self._sessions.pop(root, None)


def operation_span(name, *, model=None, kind='postprocess', instance_id=None):
    """Instrument CPU geometry/postprocessing inside a service or candidate stage."""
    from contextlib import nullcontext
    runtime = _active.get()
    if runtime is None:
        return nullcontext()
    return runtime.profiler.span(name, kind, model=model,
                                 instance_id=instance_id or runtime.instance_id,
                                 retry_count=runtime.retry_count)


def instance_scope(instance_id, retry_count=0):
    """Attach local SAM calls to the instance, keeping execution metadata out of prompts."""
    from contextlib import contextmanager
    @contextmanager
    def scope():
        runtime = _active.get()
        if runtime is None:
            yield
            return
        previous = runtime.instance_id, runtime.retry_count
        runtime.instance_id, runtime.retry_count = instance_id, retry_count
        try:
            yield
        finally:
            runtime.instance_id, runtime.retry_count = previous
    return scope()
