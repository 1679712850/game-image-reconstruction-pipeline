"""Batch generation -> generated segmentation -> QA -> bounded retry -> selection."""
from pathlib import Path
import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw

from app.edits import validate_edit_requests
from services.execution import operation_span
from candidates.geometry import source_placement, normalize, rebuild_mask
from candidates.reconstruction import analyze, prepare, COMPLETION_PROMPT, RETRY_PROMPT, completion_metrics
from candidates.qa import evaluate, original_qa, retry_prompt, select
from schemas.candidate import Candidate, CandidateQA, CandidateRegistry
from diagnostics.p1_report import portable

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
            'segmented_mask_path': obj.get('source_mask_path'), 'placement': placement})
        registries[ident] = registry
        if not obj.get('asset_path'):
            registry.candidates.append(Candidate(candidate_id='original', type='segmentation',
                qa=CandidateQA(status='REJECT', reasons=[obj.get('error') or 'no visible asset']), placement=placement))
            continue
        registry.candidates.append(Candidate(candidate_id='original', type='segmentation', image_path=obj['asset_path'],
            mask_path=obj.get('mask_path'), placement=placement, qa=original_qa(obj)))
        request = requests.get(ident)
        if not config.object_completion.enabled or not (config.candidates.automatic or request):
            continue
        analysis = analyze(obj, state, reviewer)
        obj.update(occlusion_ratio=analysis.occlusion_ratio, occlusion_directions=analysis.occluded_directions,
            completion_required=analysis.needs_completion, reconstruction_confidence=analysis.reconstruction_confidence,
            bbox_visible=obj['crop_bbox'], original_crop_path=obj['asset_path'])
        kind = 'inpaint' if request else strategy(obj, config.candidates)
        if analysis.needs_completion and not kind:
            kind = 'inpaint'
        if not kind:
            continue
        folder = root/'candidates'/ident
        prompt = (request.prompt if request else f'Reconstruct this {obj["category"]}.') + '\n' + COMPLETION_PROMPT
        try:
            prepared = prepare(obj, analysis, folder/'plan', config.amodal, request)
            obj.update(amodal_mask_path=prepared['amodal_mask_path'], expanded_crop_path=prepared['expanded_crop_path'],
                edit_mask_path=prepared['edit_mask_path'], bbox_full=prepared['bbox_full'],
                bbox_visible=prepared['bbox_visible'], reconstruction=prepared)
            prompt += '\nOcclusion reasoning and full-shape constraints: ' + json.dumps(prepared['analysis'], ensure_ascii=False)
            prompt += '\nSource crop offset in edit canvas: ' + json.dumps(prepared['source_offset'])
            prompt += '\nEdit canvas size: ' + json.dumps([prepared['canvas_bbox']['w'], prepared['canvas_bbox']['h']])
        except (ValueError, OSError) as error:
            registry.candidates.append(Candidate(candidate_id='planning_failed', type=kind,
                qa=CandidateQA(status='REJECT', reasons=[f'amodal planning failed: {error}'])))
            continue
        count = config.amodal.severe_candidates if analysis.occlusion_ratio >= config.candidates.severe_occlusion else 1
        for variant in range(count):
            jobs.append({'id': ident, 'type': kind, 'prompt': prompt + f'\nCandidate hypothesis {variant+1}.',
                'mask': prepared['edit_mask_path'], 'explicit': bool(request), 'analysis': analysis,
                'prepared': prepared, 'prefix': kind if count == 1 else f'{kind}_c{variant+1}'})

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
            prepared = job.get('prepared')
            job['source'] = prepared['expanded_crop_path'] if prepared else obj['asset_path']
            job['placement'] = prepared['placement'] if prepared else registry.source['placement']
            candidate = Candidate(candidate_id=candidate_id, type=job['type'], prompt=job['prompt'], attempt=attempt,
                qa=CandidateQA(status='REJECT', reasons=['generation failed']))
            registry.candidates.append(candidate)
            try:
                if runtime:
                    runtime.instance_id, runtime.retry_count = ident, attempt
                if job.get('existing_path'):
                    raw = job['existing_path']
                elif job['type'] == 'inpaint':
                    raw = service.complete_object(job['source'], job['mask'], prompt=job['prompt'], output_path=path)
                else:
                    raw = service.generate_object(job['source'], prompt=job['prompt'], output_path=path)
                candidate.raw_path = raw
                pending.append((job, candidate))
                if job['explicit'] and attempt == 0:
                    box = job['placement']['crop_bbox']
                    edits.append({'object_id': ident, 'source_asset_path': job['source'], 'mask_path': job['mask'],
                        'asset_path': raw, 'prompt': job['prompt'], 'crop_bbox': box,
                        'logical_size': [box['w'], box['h']], 'mock': service.mock,
                        'status': 'mock_noop' if service.mock else 'manual_review', 'alpha_policy': 'visible_hint' if service.mock else 'pending_segmentation'})
            except Exception as error:
                candidate.error = str(error); candidate.qa.reasons = [f'generation failed: {error}']
        if runtime:
            runtime.manager.end_stage()
        for job, candidate in pending:
            obj = objects[job['id']]
            try:
                if runtime:
                    runtime.instance_id, runtime.retry_count = job['id'], attempt
                with Image.open(candidate.raw_path) as image:
                    image = image.copy()
                prepared = job.get('prepared')
                if prepared:
                    with Image.open(prepared['observed_mask_path']) as visible:
                        visible = visible.copy()
                    # Restore canvas coordinates after model resolution rounding.
                    if image.size != visible.size:
                        image = image.resize(visible.size, Image.Resampling.LANCZOS)
                    aligned_path = Path(candidate.raw_path).with_name(candidate.candidate_id+'_canvas.png')
                    image.save(aligned_path)
                    from candidates.alpha import recover_alpha
                    with Image.open(job['source']) as source_image:
                        mock_noop = bool(service.mock and np.array_equal(np.asarray(image.convert('RGBA')),
                            np.asarray(source_image.convert('RGBA'))))
                    image = recover_alpha(image, str(aligned_path), prepared, obj, sam, mock_noop=mock_noop)
                    metrics, reasons = completion_metrics(image.getchannel('A'), visible, job['analysis'])
                    candidate.completion = {**metrics, 'reasons': reasons, 'analysis': prepared['analysis'],
                        'alpha_policy': 'visible_hint' if mock_noop else 'resegmented',
                        'canvas_bbox': prepared['canvas_bbox'], 'amodal_mask_path': prepared['amodal_mask_path'],
                        'expanded_crop_path': prepared['expanded_crop_path'], 'edit_mask_path': prepared['edit_mask_path']}
                elif image.mode != 'RGBA' or image.getchannel('A').getextrema()[0] == 255:
                    if not sam:
                        raise ValueError('Generated RGB candidate requires object segmentation')
                    w, h = image.size
                    masks = sam.segment(candidate.raw_path, [{'id': obj['id'], 'category': obj['category'],
                        'confidence': obj['confidence'], 'bbox': {'x': 0, 'y': 0, 'w': w, 'h': h}}])
                    from candidates.alpha import refine_alpha
                    mask = refine_alpha(masks[0]['mask'])
                    if mask.shape != (h, w) or not mask.any():
                        raise ValueError('Generated segmentation is empty or invalid')
                    image = image.convert('RGBA'); image.putalpha(Image.fromarray(mask))
                alpha_path = Path(candidate.raw_path).with_name(candidate.candidate_id+'_alpha.png')
                image.save(alpha_path)
                if job['explicit'] and attempt == 0:
                    for edit in edits:
                        if edit['asset_path'] == candidate.raw_path and not service.mock:
                            edit.update(asset_path=str(alpha_path), alpha_policy='resegmented')
                path = alpha_path.with_name(candidate.candidate_id+'.png')
                with operation_span('candidate.normalize', model='image_edit', instance_id=obj['id']):
                    candidate.image_path, candidate.mask_path, candidate.placement = normalize(
                        alpha_path, job['source'], job['placement'], path, (state['width'], state['height']),
                        aligned=bool(prepared) or job['type'] == 'inpaint', allow_outside_scene=bool(prepared))
            except Exception as error:
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
                candidate.qa = evaluate(candidate.image_path, obj['asset_path'], obj, reviewer, config.candidates)
                reasons = candidate.completion.get('reasons', [])
                if reasons:
                    candidate.qa.reasons.extend(reasons)
                    if candidate.qa.status != 'REJECT':
                        candidate.qa.status = 'RETRY'
                analysis = job.get('analysis')
                if analysis and (analysis.evidence != 'vision' or analysis.reconstruction_confidence < config.amodal.confidence_threshold
                                 or analysis.visible_ratio < .20):
                    candidate.qa.status = 'RETRY' if candidate.qa.status != 'REJECT' else 'REJECT'
                    candidate.qa.reasons.append('amodal confidence unavailable or too low; manual review required')
            except Exception as error:
                candidate.qa = CandidateQA(status='RETRY', reasons=[f'QA unavailable: {error}'])
            unavailable = any('unavailable' in reason.lower() or 'incomplete semantic' in reason.lower() or 'manual review required' in reason.lower()
                              for reason in candidate.qa.reasons)
            if candidate.qa.status == 'RETRY' and not unavailable and attempt < config.resources.max_generation_retry and config.object_completion.enabled:
                next_job = {**job, 'existing_path': None,
                    'prompt': retry_prompt(job['prompt'], candidate.qa.reasons, attempt+1) + '\n' + RETRY_PROMPT}
                if candidate.completion.get('touches_border') and job.get('prepared'):
                    try:
                        next_job['prepared'] = prepare(obj, job['analysis'], root/'candidates'/obj['id']/f'canvas_r{attempt+1}',
                                                       config.amodal, growth=1.5**(attempt+1))
                        next_job['mask'] = next_job['prepared']['edit_mask_path']
                        next_job['prompt'] += '\nUpdated canvas coordinates supersede the previous offset: ' + json.dumps({
                            'source_offset': next_job['prepared']['source_offset'],
                            'canvas_bbox': next_job['prepared']['canvas_bbox']})
                    except (ValueError, OSError) as error:
                        candidate.qa.reasons.append(f'canvas expansion budget: {error}')
                        continue
                jobs.append(next_job)
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
        manual |= bool(obj.get('completion_required') and (not chosen or chosen.candidate_id == 'original'))
        registry.needs_manual_review = manual
        obj['needs_manual_review'] = manual
        obj['asset_library_eligible'] = bool(chosen and chosen.qa.status == 'ACCEPT' and not manual)
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
                pivot=placement['pivot'], z_order=placement['z_order'], hd_asset_path=None,
                reconstructed_mask_path=chosen.mask_path if chosen.type != 'segmentation' else None,
                completion_qa=chosen.completion)
            if chosen.completion:
                for key in ('amodal_mask_path', 'expanded_crop_path', 'edit_mask_path'):
                    obj[key] = chosen.completion[key]
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
