"""Portable node/model spans and sampled memory; absent GPU metrics remain null."""
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import html
import json
import resource
import sys
import threading
import time

GIB = 1024**3


def snapshot():
    value = {'rss': None, 'system_ram_total': None, 'system_ram_available': None,
             'vram_total': None, 'vram_free': None, 'vram_allocated': None, 'vram_reserved': None,
             'gpu_util': None}
    try:
        import psutil
        value['rss'] = psutil.Process().memory_info().rss
        ram = psutil.virtual_memory()
        value.update(system_ram_total=ram.total, system_ram_available=ram.available)
    except ImportError:
        # ru_maxrss is a process lifetime peak, not current RSS.
        value['rss_lifetime_peak'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            value.update(vram_total=total, vram_free=free, vram_allocated=torch.cuda.memory_allocated(),
                         vram_reserved=torch.cuda.memory_reserved())
            try:
                value['gpu_util'] = torch.cuda.utilization()
            except (RuntimeError, AttributeError, ImportError, ModuleNotFoundError):
                pass
        except (RuntimeError, AttributeError):
            pass
    elif torch is not None and hasattr(torch, 'mps') and torch.backends.mps.is_available():
        value['vram_allocated'] = torch.mps.current_allocated_memory()
        value['vram_reserved'] = torch.mps.driver_allocated_memory()
    return value


class Profiler:
    def __init__(self):
        self.started = time.time()
        self.events = []
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        value = {'time': time.time()-self.started, **snapshot()}
        self.samples.append(value)
        return value

    def start_sampling(self):
        if self._thread is not None:
            return
        self._stop.clear()
        def run():
            while not self._stop.wait(.2):
                self.sample()
        self._thread = threading.Thread(target=run, daemon=True, name='pipeline-memory-profiler')
        self._thread.start()

    def stop_sampling(self):
        if self._thread is not None:
            self._stop.set(); self._thread.join(timeout=1); self._thread = None

    @contextmanager
    def span(self, name, kind='node', **metadata):
        start = time.time(); before = self.sample(); index = len(self.samples)-1
        record = {'name': name, 'kind': kind, 'start_time': start, **metadata}
        torch = sys.modules.get('torch')
        gpu_peak = kind in {'load', 'inference'} and torch is not None and torch.cuda.is_available()
        if gpu_peak:
            torch.cuda.reset_peak_memory_stats()
        try:
            yield record
        except Exception as error:
            record['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            after = self.sample(); end = time.time()
            samples = self.samples[index:]
            peak = lambda key: max((s[key] for s in samples if s.get(key) is not None), default=None)
            measured_peak = peak('vram_allocated')
            if gpu_peak:
                measured_peak = max(measured_peak or 0, torch.cuda.max_memory_allocated())
            record.update(end_time=end, duration=end-start, start=start-self.started,
                          memory_before=before, memory_after=after,
                          vram_peak=measured_peak, ram_peak=peak('rss'))
            self.events.append(record)

    def report(self, cache, models):
        aggregates = defaultdict(lambda: {'load_time': 0., 'inference_time': 0., 'postprocess_time': 0., 'peak_vram': None})
        instances = defaultdict(lambda: defaultdict(float))
        retry_duration = 0.; retry_gpu = 0.; retry_calls = 0; gpu_measured = False
        for event in self.events:
            if event.get('model'):
                stats = aggregates[event['model']]
                field = {'load': 'load_time', 'inference': 'inference_time', 'postprocess': 'postprocess_time'}.get(event['kind'])
                if field:
                    stats[field] += event.get('exclusive_duration', event['duration'])
                if event['vram_peak'] is not None:
                    stats['peak_vram'] = max(stats['peak_vram'] or 0, event['vram_peak'])
            if event.get('instance_id') and event['kind'] == 'inference':
                instances[event['instance_id']][event['model']] += event['duration']
            if event.get('retry_count', 0) and event['kind'] == 'inference':
                retry_duration += event['duration']; retry_calls += 1
                retry_gpu += event.get('gpu_time') or 0
                gpu_measured |= event.get('gpu_time') is not None
        for record in instances.values():
            record['total'] = sum(record.values())
        hits = sum(v['hit'] for v in cache.values()); misses = sum(v['miss'] for v in cache.values())
        execution_time = sum(e['duration'] for e in self.events if e['kind'] == 'node')
        return {'total_time': execution_time, 'wall_time': time.time()-self.started,
                'units': {'memory': 'bytes', 'time': 'seconds'},
                'models': dict(aggregates), 'model_states': models, 'instances': dict(instances),
                'retry': {'calls': retry_calls, 'duration': retry_duration, 'gpu_time': retry_gpu if gpu_measured else None,
                          'fraction_of_total': retry_duration/max(.001, execution_time)},
                'cache': {'hit': hits, 'miss': misses, 'hit_rate': hits/max(1, hits+misses),
                          'namespaces': {k: {**v, 'hit_rate': v['hit']/max(1, v['hit']+v['miss'])} for k, v in cache.items()}},
                'nodes': [e for e in self.events if e['kind'] == 'node'], 'events': self.events, 'samples': self.samples,
                'measurement': 'RAM and device metrics are sampled every 200ms and at span boundaries. CUDA inference/load spans also read the allocator peak; nested load peaks are included in model aggregates. GPU time is null when CUDA events are unavailable.'}

    def write(self, root, cache, models):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        report = self.report(cache, models)
        path = root/'performance_report.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        duration = max(report['wall_time'], .001)
        rows = ''.join('<div class="row"><label>'+html.escape(e['name'])+'</label><div class="track"><div class="bar" style="margin-left:'+str(100*e['start']/duration)+'%;width:'+str(max(.2,100*e['duration']/duration))+'%" title="'+html.escape(f'{e["duration"]:.3f}s')+'"></div></div></div>' for e in report['nodes'])
        charts = ''
        for key, label, denominator in [('rss', 'CPU RSS (GiB)', GIB), ('vram_allocated', 'GPU allocated (GiB)', GIB), ('gpu_util', 'GPU utilization (%)', 1)]:
            points = [(s['time'], s[key]/denominator) for s in self.samples if s.get(key) is not None]
            if not points:
                charts += f'<p>{label}: unavailable on this runtime</p>'; continue
            maximum = max(.001, max(v for _, v in points))
            coordinates = ' '.join(f'{t/duration*1000:.2f},{140-v/maximum*130:.2f}' for t,v in points)
            charts += f'<h3>{label} · peak {maximum:.2f}</h3><svg viewBox="0 0 1000 150"><polyline fill="none" stroke="#387bd1" stroke-width="2" points="{coordinates}"/></svg>'
        timeline = root/'timeline.html'
        timeline.write_text('<!doctype html><meta charset="utf-8"><title>Pipeline profiling</title><style>body{font:14px system-ui;margin:32px}.row{display:flex;margin:7px 0}label{width:180px}.track{flex:1;background:#eee}.bar{height:18px;background:#387bd1}svg{width:100%;background:#f5f5f5}</style><h1>Pipeline profiling</h1>'+rows+charts+'<p>'+html.escape(report['measurement'])+'</p>', encoding='utf-8')
        return str(path.resolve()), str(timeline.resolve())
