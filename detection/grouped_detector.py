"""Adapter-neutral prompt scheduling, with each failed scan isolated."""
from contextlib import nullcontext
import logging
import time
from taxonomy.categories import category_group
from taxonomy.prompt_groups import groups_for

TILE_INSTRUCTION = (
    "This image is a local tile of a larger 2D / 2.5D isometric game map. "
    "Find every independently reusable environmental object in the requested group, "
    "including tiny, occluded and edge-truncated objects and building attachments. "
    "Include uncertain candidates for later review. Inspect under eaves, roofs, "
    "stairs, bridges and trees for lanterns, flags, pots, signs, pillars, stones, "
    "plants, tombstones, incense burners, talismans and cultivation props."
)


def grouped_categories(categories, size):
    groups = groups_for(categories)
    result = []
    for name, members in groups:
        if name == "other":
            by_group = {}
            for member in members:
                by_group.setdefault(category_group(member), []).append(member)
        else:
            by_group = {name: members}
        for group, values in by_group.items():
            result.extend((group, values[i:i+size]) for i in range(0, len(values), size))
    return result


def scan_window(image, window, categories, infer, collect, scans, *, source, tile_id=None,
                group_size=6, parent_id=None, budget=None, pass_id=None, scale=None,
                allowed_groups=None):
    groups = [(name, group) for name, group in grouped_categories(categories, group_size)
              if allowed_groups is None or name in allowed_groups]
    if not groups:
        return
    # One immutable crop per window, shared by every prompt and its cache key.
    crop = image.crop(window)
    prepared_window = getattr(infer, 'prepared_window', None)
    batch_size = getattr(infer, 'batch_size', 1)
    with prepared_window(crop) if prepared_window else nullcontext():
        for start in range(0, len(groups), batch_size):
            jobs = []
            for group_name, group in groups[start:start+batch_size]:
                context = {"source": source, "tile_id": tile_id, "window": list(window),
                           "group": group_name, "categories": group, "parent_id": parent_id,
                           "pass_id": pass_id or source, "scale": scale,
                           "instruction": TILE_INSTRUCTION if source != "global" else "Find whole scene objects and terrain with global context."}
                if budget is not None and not budget.consume(source):
                    jobs.append((context, False))
                else:
                    jobs.append((context, True))
            active = [context for context, admitted in jobs if admitted]
            batch_results = None
            batch_elapsed = 0
            if len(active) > 1 and batch_size > 1:
                started = time.perf_counter()
                try:
                    batch_results = infer.infer_many(crop, [c['categories'] for c in active])
                    if len(batch_results) != len(active):
                        raise ValueError('Detector returned incorrect batch length')
                except Exception:
                    # Preserve per-prompt failure isolation. Singletons retain the
                    # existing runtime's bounded retry/resource recovery contract.
                    batch_results = None
                batch_elapsed = time.perf_counter() - started
            result_index = 0
            for context, admitted in jobs:
                if not admitted:
                    scans.append({"source": source, "tile_id": tile_id, "window": list(window),
                                  "group": context['group'], "categories": context['categories'],
                                  "status": "budget_exhausted", "candidates": 0, "pass_id": pass_id, "scale": scale})
                    continue
                started = time.perf_counter()
                try:
                    found = (batch_results[result_index] if batch_results is not None
                             else infer(crop, context['categories'], context))
                    context['runtime'] = (batch_elapsed/len(active) if batch_results is not None
                                          else time.perf_counter()-started)
                    scan = {**context, 'status': 'ok', 'candidates': len(found)}
                    for item in found:
                        collect(item, context)
                except Exception as error:
                    scan = {**context, 'status': 'failed', 'candidates': 0,
                            'error': f'{type(error).__name__}: {error}'}
                    logging.getLogger(__name__).warning('[%s] tile=%s group=%s failed: %s',
                        'GlobalDetection' if source == 'global' else 'TileDetection', tile_id, context['group'], error)
                scan['runtime'] = time.perf_counter()-started + batch_elapsed/len(active)
                if batch_results is not None:
                    scan['batch_size'] = len(active)
                    scan['runtime_allocation'] = 'equal_share_of_batch_plus_collect'
                scans.append(scan)
                result_index += 1
                logging.getLogger(__name__).info('[%s] tile=%s group=%s objects=%s',
                    'GlobalDetection' if source == 'global' else 'TileDetection', tile_id, context['group'], scan['candidates'])
