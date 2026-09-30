"""One lifecycle authority for registered lazy model factories and LRU eviction."""
from dataclasses import dataclass
import gc
import sys
import time
from services.profiler import snapshot, GIB
from services.model_support import ModelUnavailableError


class ResourceBudgetError(ModelUnavailableError):
    pass


def is_oom(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if 'out of memory' in str(error).lower() or 'outofmemory' in type(error).__name__.lower():
            return True
        error = error.__cause__ or error.__context__
    return False


@dataclass
class ModelEntry:
    service: object
    loader: object
    status: str = 'NOT_LOADED'
    last_used: float = 0
    vram: int = 0
    active: int = 0


class ModelManager:
    def __init__(self, config, profiler):
        self.config = config
        self.profiler = profiler
        self.entries = {}

    def register(self, name, service):
        if hasattr(service, 'load') and not getattr(service, 'mock', False):
            # Reuse the original factory if a graph is rebuilt with the same adapters.
            loader = getattr(service, '_resource_loader', service.load)
            service._resource_loader = loader
            self.entries[name] = ModelEntry(service, loader)
            def managed_load():
                from services.execution import _active
                runtime = _active.get()
                return (runtime.manager if runtime is not None else self).load(name)
            service.load = managed_load

    @staticmethod
    def handle(service):
        if getattr(service, '_predictor', None) is not None:
            return service._predictor.model
        for key in ('_pipeline', '_model'):
            value = getattr(service, key, None)
            if value is not None:
                return value
        return None

    def release_cache(self):
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
        elif torch is not None and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def unload(self, name, *, offload=False):
        entry = self.entries[name]
        if entry.active:
            return
        service = entry.service; handle = self.handle(service)
        if handle is None:
            entry.status = 'NOT_LOADED'; return
        if offload and entry.status == 'GPU':
            with self.profiler.span(name+'.offload', 'offload', model=name):
                handle.to('cpu')
                if hasattr(service, '_device'):
                    service._device = 'cpu'
            entry.status = 'OFFLOADED'; entry.vram = 0
        elif offload and entry.status in {'CPU', 'OFFLOADED'}:
            return
        else:
            # Predictors and pipelines must release every owner reference, not a local alias.
            for key in ('_predictor', '_pipeline', '_model', '_processor'):
                if hasattr(service, key):
                    setattr(service, key, None)
            entry.status = 'NOT_LOADED'; entry.vram = 0
        del handle
        self.release_cache()

    def _make_room(self, name, required):
        memory = snapshot()
        while True:
            residents = [(key, e) for key,e in self.entries.items() if key != name and e.status == 'GPU']
            used = max(memory.get('vram_allocated') or 0, sum(e.vram for _, e in residents))
            soft = min(self.config.soft_vram*GIB, memory.get('vram_total') or float('inf'))
            free = memory.get('vram_free')
            if len(residents) < self.config.gpu_models_max_resident and used+required <= soft and (free is None or required <= free):
                break
            evictable = [(key,e) for key,e in residents if not e.active]
            if not evictable:
                break
            key, _ = min(evictable, key=lambda pair: pair[1].last_used)
            rss = memory.get('rss') or 0
            self.unload(key, offload=self.config.keep_alive.get(key, False) and rss+required < self.config.max_ram*GIB)
            memory = snapshot()
        free = memory.get('vram_free')
        hard = min(self.config.max_vram*GIB, memory.get('vram_total') or float('inf'))
        residents = [e for key,e in self.entries.items() if key != name and e.status == 'GPU']
        return (len(residents) < self.config.gpu_models_max_resident
                and (memory.get('vram_allocated') or 0)+required <= hard
                and (free is None or required <= free))

    def load(self, name):
        entry = self.entries[name]; service = entry.service
        entry.last_used = time.time()
        handle = self.handle(service)
        if handle is not None and entry.status in {'GPU', 'CPU'}:
            return handle
        options = service.config
        requested = options.device
        torch = sys.modules.get('torch')
        gpu = requested in {'cuda', 'mps'} or requested == 'auto' and torch is not None and torch.cuda.is_available()
        selected = requested if requested != 'auto' else ('cuda' if gpu else 'cpu')
        # Device discovery is lazy, only on a cache miss that needs the model.
        if requested == 'auto' and torch is None:
            from services.model_support import torch_runtime
            _, selected = torch_runtime(requested); gpu = selected in {'cuda', 'mps'}
        required = self.config.estimated_vram.get(name, 2)*GIB
        room = self._make_room(name, required) if gpu else True
        if gpu and not room:
            # CPU placement is supported by every current local adapter. No pretend quantization.
            service.config = options.model_copy(update={'device': 'cpu'})
            gpu = False; selected = 'cpu'
        memory = snapshot()
        additional_ram = self.config.estimated_ram.get(name, self.config.estimated_vram.get(name, 2)*2)*GIB if handle is None else 0
        def ram_exceeded(current):
            available = current.get('system_ram_available')
            return ((current.get('rss') or 0)+additional_ram > self.config.max_ram*GIB
                    or available is not None and additional_ram > available)
        if ram_exceeded(memory):
            for key,e in sorted(self.entries.items(), key=lambda pair: pair[1].last_used):
                if key != name and not e.active:
                    self.unload(key)
            if ram_exceeded(snapshot()):
                raise ResourceBudgetError('CPU RAM budget exhausted before model load')
        before = memory.get('vram_allocated') or 0
        with self.profiler.span(name+'.load', 'load', model=name):
            try:
                if handle is not None:
                    device = selected
                    handle.to(device)
                    if hasattr(service, '_device'):
                        service._device = device
                else:
                    entry.loader()
                entry.status = 'GPU' if gpu else 'CPU'
                entry.vram = max(0, (snapshot().get('vram_allocated') or 0)-before)
            except Exception:
                for key in ('_predictor', '_pipeline', '_model', '_processor'):
                    if hasattr(service, key):
                        setattr(service, key, None)
                entry.status = 'NOT_LOADED'; self.release_cache()
                raise
        memory = snapshot()
        if (memory.get('vram_allocated') or 0) > self.config.max_vram*GIB:
            self.unload(name, offload=True)
            service.config = service.config.model_copy(update={'device': 'cpu'})
            entry.status = 'CPU'
        if (snapshot().get('rss') or 0) > self.config.max_ram*GIB:
            self.unload(name)
            raise ResourceBudgetError('CPU RAM budget exceeded by loaded model')
        return self.handle(service)

    def recover(self, name, attempt):
        # Invoked between failed attempts, after the inference context has unwound.
        for key, entry in self.entries.items():
            if not entry.active:
                self.unload(key)
        service = self.entries[name].service if name in self.entries else None
        if service:
            # Current adapters already infer one item at a time. Offload plus lower input resolution
            # is the supported degradation; unsupported quantizers are not advertised as applied.
            service.config = service.config.model_copy(update={'device': 'cpu'})
            service._inference_scale = .5**attempt
            # A prebuilt local quantized pipeline is an explicit alternative, never
            # an implicit download or an unsupported in-place quantizer.
            section = {'image_edit': 'qwen_image_edit', 'layered': 'qwen_layered'}.get(name)
            if section and attempt > 1:
                settings = getattr(service.config, section)
                if settings.quantized_model_path:
                    service.config = service.config.model_copy(update={section: settings.model_copy(
                        update={'model_path': settings.quantized_model_path})})
        self.release_cache()

    def end_stage(self, final=False):
        for key, entry in self.entries.items():
            if entry.status != 'NOT_LOADED' and not entry.active:
                keep = not final and self.config.keep_alive.get(key, False)
                try:
                    self.unload(key, offload=keep)
                except (RuntimeError, AttributeError):
                    self.unload(key)

    def states(self):
        return {key: {'status': entry.status, 'vram': entry.vram, 'last_used': entry.last_used}
                for key,entry in self.entries.items()}
