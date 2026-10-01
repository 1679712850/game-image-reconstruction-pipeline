"""Stage progress with monotonic timing and a heartbeat during long model calls."""
from collections import Counter
from datetime import datetime, timedelta
import logging
import threading
import time


def duration(seconds: float) -> str:
    hours, rest = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class StageProgress:
    """Count finished work (including failures), not work merely submitted."""
    def __init__(self, label, total, callback=None, *, interval=10, unit='object'):
        self.label, self.total, self.unit = label, total, unit
        self.callback = callback or logging.getLogger(__name__).info
        self.interval = interval
        self.started = time.monotonic()
        self.completed = 0
        self.current = ''
        self.counts = Counter()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    def format(self, *, elapsed=None, now=None):
        elapsed = max(0, time.monotonic() - self.started if elapsed is None else elapsed)
        now = now or datetime.now().astimezone()
        percent = 100 * self.completed / self.total if self.total else 100
        average = elapsed / self.completed if self.completed else None
        eta = average * max(0, self.total - self.completed) if average is not None else None
        if not self.total:
            eta = 0
        finish = now + timedelta(seconds=eta) if eta is not None else None
        finish_format = '%H:%M' if finish and finish.date() == now.date() else '%Y-%m-%d %H:%M'
        return '\n'.join([
            f'[{self.label}] {self.completed}/{self.total} ({percent:.1f}%)'
            + (f' current={self.current}' if self.current else ''),
            f'elapsed={duration(elapsed)}',
            f'avg={average:.1f}s/{self.unit}' if average is not None else f'avg=--s/{self.unit}',
            f'eta={duration(eta)}' if eta is not None else 'eta=--:--:--',
            'estimated_finish=' + (finish.strftime(finish_format) if finish else '--:--'),
            ' '.join(f'{key}={count}' for key, count in sorted(self.counts.items())),
        ]).rstrip()

    def emit(self):
        with self._lock:
            self.callback(self.format())

    def start_item(self, item):
        with self._lock:
            self.current = str(item)

    def advance(self, status='completed'):
        with self._lock:
            self.completed += 1
            self.counts[status] += 1
            self.current = ''
            self.emit()

    def __enter__(self):
        self.started = time.monotonic()
        self.emit()
        if self.total:
            def heartbeat():
                while not self._stop.wait(self.interval):
                    self.emit()
            self._thread = threading.Thread(target=heartbeat, daemon=True, name='pipeline-progress')
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if exc_type is not None:
            with self._lock:
                self.counts['interrupted'] += 1
                self.emit()
