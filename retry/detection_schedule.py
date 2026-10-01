"""Bounded DINO lookahead; SAM/QA commits retain the original region/attempt order.

Do not prefetch when an earlier retry could consume a later region's budget or
object slots. Adapters must explicitly provide a conservative detect_bounds.
"""
from pathlib import Path
from PIL import Image
from schemas.object import BBox


def region_key(region):
    """Stable budget key, shared across P1 scene review rounds."""
    box = region['approx_bbox']
    category = (region.get('category') or '').lower().replace(' ', '_')
    return [category, *(box[k] for k in ('x', 'y', 'w', 'h'))]


class ScheduledDetectionError(RuntimeError):
    def __init__(self, name, message):
        super().__init__(message)
        self.original_type = name


def region_geometry(source, region, attempt):
    box = BBox.model_validate(region['approx_bbox'])
    pad = 32 * (attempt + 1)
    x, y = max(0, box.x-pad), max(0, box.y-pad)
    right, bottom = min(source.width, box.x+box.w+pad), min(source.height, box.y+box.h+pad)
    if right <= x or bottom <= y:
        raise ValueError('Problem region lies outside the original image')
    scale = 2 if max(right-x, bottom-y) < 512 else 1
    return x, y, right, bottom, scale


def write_crop(source, region, attempt, path):
    x, y, right, bottom, scale = region_geometry(source, region, attempt)
    crop = source.crop((x, y, right, bottom))
    crop = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)
    return x, y, right, bottom, scale


class RegionDetectionSchedule:
    def __init__(self, state, regions, detector, config, source, categories, scene_attempt):
        self.state, self.detector, self.config = state, detector, config
        self.source, self.categories, self.scene_attempt = source, categories, scene_attempt
        self.entries, seen = {}, set()
        for index, region in enumerate(regions[:config.p1.max_problem_regions]):
            try:
                key = region_key(region)
            except (KeyError, TypeError, AttributeError):
                # Do not evaluate a malformed future region earlier than the
                # original loop (it may never be reached due to the object cap).
                break
            if tuple(key) in seen:
                continue
            seen.add(tuple(key))
            previous = sum(log.get('region_key') == key and log.get('action') == 'local_detection'
                           for log in state.get('retry_history', []))
            if previous < config.p1.max_detection_retry:
                self.entries[index] = (region, previous)
        self.pending = {}

    def _admitted(self, index, count):
        # Third-party/stateful detectors opt in through an explicit bound method.
        bounds = getattr(type(self.detector), 'detect_bounds', None)
        if bounds is None or self.config.p1.detection_prefetch_regions == 1:
            return []
        from services.execution import _active
        runtime = _active.get()
        budget = runtime.detection_budget if runtime else None
        calls = objects = 0
        selected = []
        for ident, (region, previous) in self.entries.items():
            if ident < index:
                continue
            extra_calls = extra_objects = 0
            try:
                for attempt in range(previous, self.config.p1.max_detection_retry):
                    x, y, right, bottom, scale = region_geometry(self.source, region, attempt)
                    prompts = [region['category']] if region.get('category') else self.categories
                    limit = runtime.config.detection.budget.max_categories_per_pass if runtime else self.detector.config.grounding.prompt_group_size
                    n, m = self.detector.detect_bounds(((right-x)*scale, (bottom-y)*scale), prompts, limit)
                    extra_calls += n * (1 + self.config.resources.max_oom_retry)
                    # A region stops retrying as soon as any object is accepted.
                    extra_objects = max(extra_objects, m)
            except (ValueError, TypeError, AttributeError):
                break
            if budget and (budget.calls+calls+extra_calls > budget.config.max_total_inference_calls
                           or budget.retry_calls+calls+extra_calls > budget.config.max_retry_calls):
                break
            maximum = self.config.scene_loop.max_objects
            # Only prior regions must fit completely: the last region is still
            # visited with one free slot, then its ordinary commit loop enforces
            # the cap. This avoids disabling staging for small remaining quotas.
            if maximum is not None and count+objects >= maximum:
                break
            calls += extra_calls
            objects += extra_objects
            selected.append(ident)
            if len(selected) == self.config.p1.detection_prefetch_regions:
                break
        return selected if len(selected) > 1 else []

    def detect(self, index, attempt, path, prompts, count):
        if index not in self.pending and index in self.entries and attempt == self.entries[index][1]:
            selected = self._admitted(index, count)
            for ident in selected:
                region, first_attempt = self.entries[ident]
                target = Path(self.state['output_dir'])/'debug'/f'qa_detect_s{self.scene_attempt}_{ident}_{first_attempt}.png'
                categories = [region['category']] if region.get('category') else self.categories
                try:
                    if ident != index:
                        write_crop(self.source, region, first_attempt, target)
                    found = self.detector.detect(str(target), categories)
                    self.pending[ident] = (found, None)
                except (ValueError, RuntimeError, OSError) as error:
                    # Store only the exception type/message, never GPU tracebacks.
                    self.pending[ident] = (None, (type(error).__name__, str(error)))
                    break  # Avoid further speculation after a resource failure.
        if index in self.pending and attempt == self.entries[index][1]:
            found, error = self.pending.pop(index)
            if error is not None:
                name, message = error
                raise ScheduledDetectionError(name, message)
            return found
        return self.detector.detect(str(path), prompts)
