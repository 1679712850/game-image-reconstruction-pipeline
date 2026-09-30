"""Batch generation -> generated segmentation -> QA -> bounded retry -> selection."""
from pathlib import Path
import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw

from app.edits import validate_edit_requests
from services.execution import operation_span
from candidates.geometry import source_placement, normalize, rebuild_mask
from candidates.qa import evaluate, original_qa, retry_prompt, select
from schemas.candidate import Candidate, CandidateQA, CandidateRegistry
from diagnostics.p1_report import portable

CONSTRAINTS = (' Preserve object type, structure, orientation, proportions, palette, lighting, rendering style '
               'and exact original camera angle/isometric perspective. No additional decoration or objects. '
               'Isolate one complete object on a plain background.')


def strategy(obj, config):
    failures = set((obj.get('qa') or {}).get('failure_types', []))
    total = obj.get('visible_pixel_count', 0) + obj.get('occluded_pixel_count', 0)
    occlusion = obj.get('occluded_pixel_count', 0) / max(1, total)
    if failures & {'BACKGROUND_LEAK', 'BAD_MASK'} or occlusion >= config.severe_occlusion:
        return 'generation'
    if obj.get('requires_inpainting') or obj.get('is_truncated') or occlusion > 0 or failures & {'MASK_TOO_SMALL', 'OCCLUSION'}:
        return 'inpaint'
    return None


def contact_sheet(registry, root):
    columns = [*registry.candidates]
    accepted = next((c for c in columns if c.candidate_id == registry.accepted_candidate_id), None)
    if accepted:
        columns.append(accepted)
    sheet = Image.new('RGB', (240*max(1, len(columns)), 380), '#252830')
    draw = ImageDraw.Draw(sheet)
    for i, candidate in enumerate(columns):
        x = 240*i
        if candidate.image_path:
            with Image.open(candidate.image_path) as image:
                thumb = image.convert('RGBA'); thumb.thumbnail((220, 220))
                sheet.paste(thumb, (x+10, 28), thumb)
        title = 'ACCEPTED' if accepted and i == len(columns)-1 else candidate.candidate_id
        draw.text((x+8, 4), title, fill='white')
        draw.text((x+8, 246), candidate.qa.status + f' {candidate.qa.overall:.3f}', fill='white')
        reason = '; '.join(candidate.qa.reasons).encode('ascii', 'replace').decode()
        for j, line in enumerate(textwrap.wrap(reason, 38)[:4]):
            draw.text((x+8, 269+j*16), line, fill='white')
        box = candidate.placement.get('crop_bbox', {})
        draw.text((x+8, 346), ' '.join(f'{key}={box[key]}' for key in ('x', 'y', 'w', 'h') if key in box), fill='white')
    path = root/'debug'/'candidates'/f'{registry.instance_id}_contact_sheet.png'
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)


def run_candidates(state, config, service, sam=None, reviewer=None, runtime=None):
    root = Path(state['output_dir']).resolve()
    from candidates.source import recover_source
    objects = {o['id']: recover_source(o, state, config.object_completion.enabled) for o in state.get('objects', [])}
    if len(objects) != len(state.get('objects', [])):
        raise ValueError('Candidate registry requires unique instance IDs')
    requests = {r.object_id: r for r in validate_edit_requests(state.get('edit_requests', []))}
    for ident in requests:
        if ident not in objects or not objects[ident].get('asset_path') or not objects[ident].get('crop_bbox'):
            raise ValueError(f'Edit request references an unknown/uncropped object: {ident}')
    registries, jobs, edits = {}, [], []
    for ident, obj in objects.items():
        placement = source_placement(obj)
        registry = CandidateRegistry(instance_id=ident, source={
            'image_path': obj.get('asset_path'), 'mask_path': obj.get('mask_path'), 'bbox': obj['bbox'],
            'segmented_asset_path': obj.get('source_asset_path'),
            'segmented_mask_path': obj.get('source_mask_path'),
            'placement': placement})
        registries[ident] = registry
        if not obj.get('asset_path'):
            # Preserve a real empty-mask record; never invent an opaque rectangular object.
            registry.candidates.append(Candidate(candidate_id='original', type='segmentation',
                qa=CandidateQA(status='REJECT', reasons=[obj.get('error') or 'no visible asset']), placement=placement))
            continue
        registry.candidates.append(Candidate(candidate_id='original', type='segmentation', image_path=obj['asset_path'],
            mask_path=obj.get('mask_path'), placement=placement, qa=original_qa(obj)))
        if not config.object_completion.enabled:
            continue
        request = requests.get(ident)
        kind = 'inpaint' if request else strategy(obj, config.candidates) if config.candidates.automatic else None
        if not kind:
            continue
        folder = root/'candidates'/ident; folder.mkdir(parents=True, exist_ok=True)
        mask = folder/'edit_mask.png'
        if request:
            with Image.open(request.mask_path) as image, Image.open(obj['asset_path']) as original:
                if image.size != original.size or image.convert('L').getbbox() is None:
                    raise ValueError('Edit mask must be nonempty and match the original cropped asset')
                image.convert('L').save(mask)
        else:
            # Border and holes are local repair targets; the generation is composited through this mask.
            import cv2
            with Image.open(obj['asset_path']) as image:
                alpha = np.array(image.convert('RGBA').getchannel('A'))
            binary = (alpha > 8).astype(np.uint8)*255
            kernel = np.ones((5, 5), np.uint8)
            edge = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
            Image.fromarray(edge if edge.any() else binary).save(mask)
        prompt = (request.prompt if request else f'Repair the missing parts of this {obj["category"]}.') + CONSTRAINTS
        jobs.append({'id': ident, 'type': kind, 'prompt': prompt, 'mask': str(mask), 'explicit': bool(request)})

    from candidates.layered import layer_jobs
    jobs.extend(layer_jobs(state, objects, root))
    for attempt in range(config.resources.max_generation_retry+1):
        if not jobs:
            break
        pending = []
        for job in jobs:
            ident = job['id']; obj = objects[ident]; registry = registries[ident]
            candidate_id = f'{job.get("prefix", job["type"])}_v{attempt+1}'
            path = root/'candidates'/ident/f'{candidate_id}_raw.png'
            generation_source = obj.get('source_asset_path') if job['type'] == 'generation' else None
            generation_source = generation_source or obj['asset_path']
            job['source'] = generation_source
            original_geometry = {**obj, 'asset_path': generation_source, 'crop_bbox': obj.get('source_crop_bbox') or obj['crop_bbox'], 'pivot': None}
            job['placement'] = source_placement(original_geometry) if job['type'] == 'generation' else registry.source['placement']
            candidate = Candidate(candidate_id=candidate_id, type=job['type'], prompt=job['prompt'], attempt=attempt,
                                  qa=CandidateQA(status='REJECT', reasons=['generation failed']))
            registry.candidates.append(candidate)
            try:
                if runtime:
                    runtime.instance_id, runtime.retry_count = ident, attempt
                if job.get('existing_path'):
                    raw = job['existing_path']
                elif job['type'] == 'inpaint' and job['explicit']:
                    raw = service.complete_object(obj['asset_path'], job['mask'], prompt=job['prompt'], output_path=path)
                else:
                    raw = service.generate_object(generation_source, prompt=job['prompt'], output_path=path)
                candidate.raw_path = raw
                pending.append((job, candidate))
                if job['explicit'] and attempt == 0:
                    edits.append({'object_id': ident, 'source_asset_path': obj['asset_path'], 'mask_path': job['mask'],
                        'asset_path': raw, 'prompt': job['prompt'], 'crop_bbox': obj['crop_bbox'],
                        'logical_size': [obj['crop_bbox']['w'], obj['crop_bbox']['h']], 'mock': service.mock,
                        'status': 'mock_noop' if service.mock else 'manual_review', 'alpha_policy': 'preserve_source'})
            except Exception as error:
                candidate.error = str(error); candidate.qa.reasons = [f'generation failed: {error}']
        if runtime:
            runtime.manager.end_stage()
        # Batch all generated alpha recovery before switching to the QA model.
        for job, candidate in pending:
            obj = objects[job['id']]; registry = registries[job['id']]
            try:
                if runtime:
                    runtime.instance_id, runtime.retry_count = job['id'], attempt
                with Image.open(candidate.raw_path) as image:
                    image = image.copy()
                if image.mode != 'RGBA' or image.getchannel('A').getextrema()[0] == 255:
                    if not sam:
                        raise ValueError('Generated RGB candidate requires object segmentation')
                    w, h = image.size
                    masks = sam.segment(candidate.raw_path, [{'id': obj['id'], 'category': obj['category'],
                        'confidence': obj['confidence'], 'bbox': {'x': 0, 'y': 0, 'w': w, 'h': h}}])
                    mask = np.asarray(masks[0]['mask'], dtype=np.uint8)
                    if mask.shape != (h, w) or not mask.any():
                        raise ValueError('Generated segmentation is empty or invalid')
                    image = image.convert('RGBA'); image.putalpha(Image.fromarray(mask))
                # Automatic local repairs preserve source pixels outside the repair region.
                if job['type'] == 'inpaint' and not job['explicit']:
                    with Image.open(obj['asset_path']) as original, Image.open(job['mask']) as mask:
                        image = Image.composite(image.resize(original.size, Image.Resampling.LANCZOS), original.convert('RGBA'), mask.convert('L'))
                alpha_path = Path(candidate.raw_path).with_name(candidate.candidate_id+'_alpha.png')
                image.save(alpha_path)
                path = alpha_path.with_name(candidate.candidate_id+'.png')
                with operation_span('candidate.normalize', model='image_edit', instance_id=obj['id']):
                    candidate.image_path, candidate.mask_path, candidate.placement = normalize(
                        alpha_path, job['source'], job['placement'], path, (state['width'], state['height']), aligned=job['type'] == 'inpaint')
            except (RuntimeError, ValueError, OSError, IndexError, KeyError) as error:
                candidate.error = str(error); candidate.qa = CandidateQA(status='REJECT', reasons=[f'alpha recovery failed: {error}'])
        if runtime:
            runtime.manager.end_stage()
        jobs = []
        for job, candidate in pending:
            if candidate.error:
                continue
            obj = objects[job['id']]
            if runtime:
                runtime.instance_id, runtime.retry_count = job['id'], attempt
            try:
                candidate.qa = evaluate(candidate.image_path, job['source'], obj, reviewer, config.candidates)
            except Exception as error:
                candidate.qa = CandidateQA(status='RETRY', reasons=[f'QA unavailable: {error}'])
            unavailable = any('unavailable' in reason.lower() or 'incomplete' in reason.lower() for reason in candidate.qa.reasons)
            if candidate.qa.status == 'RETRY' and not unavailable and attempt < config.resources.max_generation_retry and config.object_completion.enabled:
                jobs.append({**job, 'existing_path': None, 'prompt': retry_prompt(job['prompt'], candidate.qa.reasons, attempt+1) + CONSTRAINTS})
        if runtime:
            runtime.manager.end_stage()
    if runtime:
        runtime.instance_id, runtime.retry_count = None, 0

    for ident, registry in registries.items():
        obj = objects[ident]
        chosen, manual = select(registry)
        # If a requested repair failed, keep the source but make that failure visible.
        attempted = registry.candidates[1:]
        manual |= bool(attempted and not any(c.qa.status == 'ACCEPT' for c in attempted))
        registry.needs_manual_review = manual
        obj['needs_manual_review'] = manual
        if chosen:
            registry.accepted_asset = chosen.image_path; registry.accepted_candidate_id = chosen.candidate_id
            placement = chosen.placement
            box = placement['crop_bbox']
            full_mask = rebuild_mask(chosen.image_path, box, (state['width'], state['height']),
                                     root/'accepted_masks'/f'{ident}.png')
            registry.provenance = {'final_source': chosen.type, 'original_asset': registry.source['image_path'],
                'accepted_asset': chosen.image_path, 'qa_score': chosen.qa.overall,
                'qa_status': chosen.qa.status, 'retry_count': max((c.attempt for c in registry.candidates), default=0),
                'needs_manual_review': manual}
            obj.update(accepted_asset=chosen.image_path, accepted_candidate_id=chosen.candidate_id,
                asset_path=chosen.image_path, asset_mask_path=full_mask, mask_path=full_mask, full_mask_path=full_mask,
                placement=placement, provenance=registry.provenance, crop_bbox=box,
                pivot=placement['pivot'], z_order=placement['z_order'], hd_asset_path=None)
            if manual:
                obj['status'] = 'manual_review'
            elif chosen.type != 'segmentation':
                obj['status'] = 'pass'; obj['error'] = None
        elif not obj.get('metrics', {}).get('fully_occluded'):
            obj['status'] = 'manual_review'
        contact_sheet(registry, root)
    from candidates.ownership import accepted_ownership
    final_ownership = accepted_ownership(list(objects.values()), root, (state['width'], state['height']))
    payload = {key: r.model_dump(mode='json') for key, r in registries.items()}
    path = root/'metadata'/'candidates.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(payload, root), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {'accepted_ownership': final_ownership, 'objects': list(objects.values()), 'candidate_registry': payload, 'object_edits': edits,
            'failed_objects': [o['id'] for o in objects.values() if o['status'] != 'pass']}
