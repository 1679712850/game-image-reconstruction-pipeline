"""Instrument existing service boundaries without serializing clients/models in state."""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import inspect
import logging
import time
import threading
import sys

from services.cache_manager import CacheManager
from services.model_manager import ModelManager, is_oom
from services.profiler import Profiler
from services.progress import duration

_active = ContextVar('pipeline_execution', default=None)
_depth = ContextVar('pipeline_service_depth', default=0)
METHODS = {
    'vlm': ('vlm', ['analyze_scene']),
    'grounding': ('detection', ['detect', 'detect_round', 'detect_p0', '_infer_image', '_infer_batch']),
    'sam': ('segmentation', ['segment', 'segment_local', 'predict_candidates']),
    'image_edit': ('generation', ['complete_object', 'generate_object']),
    'layered': ('generation', ['decompose_layers']),
    'reviewer': ('qa', ['review', 'classify_crop', 'review_candidate', 'analyze_occlusion']),
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
        self.model_fingerprints = {}
        from detection.budget import BudgetTracker
        self.detection_budget = BudgetTracker(config.detection.budget)
        for name, (namespace, methods) in METHODS.items():
            service = getattr(services, name, None)
            if service is None:
                continue
            local_backend = getattr(service, '_local_backend', None)
            model_name = 'qwen_vl' if local_backend is not None else name
            if model_name not in self.manager.entries:
                self.manager.register(model_name, local_backend if local_backend is not None else service)
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
            if current is None or (_depth.get() and method not in {'_infer_image', '_infer_batch'}):
                return function(*args, **kwargs)
            token = _depth.set(_depth.get()+1)
            try:
                return current.infer(service, name, namespace, method, function, args, kwargs)
            finally:
                _depth.reset(token)
        invoke._pipeline_original = function
        setattr(service, method, invoke)

    def infer(self, service, name, namespace, method, function, args, kwargs):
        local_backend = getattr(service, '_local_backend', None)
        model_name = 'qwen_vl' if local_backend is not None else name
        model_service = local_backend if local_backend is not None else service
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
        from services.model_fingerprint import service_versions
        if local_backend is not None:
            identity['local_model'] = local_backend.config.qwen_vl.model_dump(mode='json')
            identity['structured_output_version'] = local_backend.structured_output_version
        identity['model_fingerprints'] = service_versions(getattr(model_service,'config',None),model_name,self.model_fingerprints)
        # P0 scanning owns call reservations. Legacy local retry calls also obey the run budget.
        if name == 'grounding' and method == '_infer_image' and self.node_name in {'retry_objects','p1_scene'}:
            if not self.detection_budget.consume('redetection'):
                raise RuntimeError('Detection retry inference budget exhausted')
        if name == 'grounding' and method in {'detect','detect_round','detect_p0'}:
            return function(*args, **kwargs)
        key = self.cache.key(namespace, identity, parameters)
        hit, result = self.cache.get(namespace, key)
        if hit:
            with self.profiler.span(name+'.'+method, 'cache', model=model_name, instance_id=self.instance_id, cache_hit=True):
                pass
            return result
        for attempt in range(self.config.resources.max_oom_retry+1):
            if attempt and name == 'grounding' and method == '_infer_image':
                if not self.detection_budget.consume('redetection'):
                    raise RuntimeError('Detection OOM retry budget exhausted')
            started = len(self.profiler.events)
            outer_detection = name == 'grounding' and method not in {'_infer_image', '_infer_batch'}
            empty_segmentation = method in {'segment', 'segment_local'} and parameters.get('detections') == []
            managed = model_name in self.manager.entries and not outer_detection and not empty_segmentation
            try:
                with self.profiler.span(name+'.'+method, 'service' if outer_detection else 'inference', model=model_name, instance_id=instance_id,
                    retry_count=max(self.retry_count, attempt), oom_retry=attempt, cache_hit=False) as event:
                    # Keep load timing separate from actual inference.
                    if managed:
                        self.manager.load(model_name)
                        self.manager.entries[model_name].active += 1
                    try:
                        torch = sys.modules.get('torch')
                        gpu_start = gpu_end = None
                        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available() and managed and self.manager.entries[model_name].status == 'GPU':
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
                            self.manager.entries[model_name].active -= 1
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
                if attempt == 0 and not failed(result) and getattr(model_service, '_inference_scale', 1) == 1:
                    self.cache.put(namespace, key, result)
                return result
            except Exception as error:
                # The detector splits batches without changing precision/device.
                # Failed singletons are replayed through the normal single-call path.
                if method == '_infer_batch':
                    raise
                if not is_oom(error):
                    raise
                if attempt >= self.config.resources.max_oom_retry:
                    from services.model_support import ModelUnavailableError
                    raise ModelUnavailableError(f'{name} OOM recovery exhausted after {attempt} retries') from error
                self.manager.recover(model_name, attempt+1)
        raise AssertionError('unreachable')

    def run_node(self, name, function, state):
        with self.lock:
            root = Path(state['output_dir']).resolve()
            if name == 'load_image' or root not in self._sessions:
                self._sessions[root] = (Profiler(), CacheManager(self.config.cache, root), {})
            self.profiler, self.cache, self.model_fingerprints = self._sessions[root]
            from detection.budget import restore_budget
            self.detection_budget = restore_budget(self.config.detection.budget,
                                                  {} if name == 'load_image' else state.get('detection_budget',{}))
            self.node_name = name
            self.manager.profiler = self.profiler
            self.root = root
            self.instance_id = None
            self.retry_count = state.get('retry_count', 0)+1 if name == 'retry_objects' else 0
            self.profiler.start_sampling()
            token = _active.set(self)
            try:
                label = {'detect_instances':'Detection','segment_instances':'Segmentation','qa_objects':'QA',
                         'upscale_objects':'Upscale','export':'Export','reconstruct_scene':'Reconstruction'}.get(name,'Pipeline')
                logging.getLogger(__name__).info('[%s] node=%s event=start',label,name)
                node_started = time.monotonic()
                with self.profiler.span(name):
                    try:
                        updates = function(state)
                    except Exception as error:
                        from services.resource_fallback import recover_node
                        updates = recover_node(name, state, error, self.services)
                    self.manager.end_stage(final=name == 'export')
                logging.getLogger(__name__).info('[%s] node=%s event=end status=%s elapsed=%s',
                    label,name,updates.get('pipeline_status','completed_stage'),duration(time.monotonic()-node_started))
                updates['detection_budget'] = self.detection_budget.report()
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
