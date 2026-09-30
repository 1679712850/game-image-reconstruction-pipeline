# P1 scene decomposition

The default graph runs source-coordinate classification and local refinement during segmentation, followed by explicit `p1_scene` and `assign_ownership` nodes after scene QA. Targeted review and retries finish before visible ownership is rebuilt, terrain is completed, and objects are passed to optional editing and upscaling. Both nodes remain active when `scene_loop.enabled` is false. They have their own progress events and checkpoint boundaries.

`config/pipeline.yaml:p1` controls local crop limits, retry budgets, QA thresholds, terrain completion, and optional LPIPS. Detection records are still global `xywh`; local windows and SAM point prompts are transformed only at inference and every saved mask is restored to the original image dimensions.

The classifier labels each record as `terrain`, `instance`, `hybrid`, or `effect`. Terrain observations with the same semantic category share one Owner and are written under `terrain/`. Objects retain `candidate_mask_path`, `visible_mask_path`, and `full_mask_path`; the current implementation never fabricates an amodal `full_mask_path`.

`metadata/ownership.json` records the integer source-coordinate ownership map, conflict decisions, predicted semantic `unassigned_ratio`, visible coverage, overlap diagnostics, and the explicit `unclassified_residual` fallback. Residual pixels can make a reconstruction visually complete while remaining semantic failures, so they are excluded from the coverage metric. `metadata/summary.json` contains the run-level detection, terrain, QA, retry, and reconstruction report.

The retry manager classifies failures as `MISSED_DETECTION`, `BAD_MASK`, `BACKGROUND_LEAK`, `MASK_TOO_SMALL`, `MASK_TOO_LARGE`, `DUPLICATE_INSTANCE`, `CROSS_TILE_FRAGMENT`, `WRONG_CATEGORY`, `PIXEL_CONFLICT`, or `UNASSIGNED_REGION`. Detection failures use bounded enlarged problem crops; mask failures use local high-resolution SAM with candidate ranking and negative points; category failures use crop classification. A replacement is accepted only when its QA score improves or the explicit mock policy is active.

Diagnostics include `coverage_map.png`, `detection_boxes.png`, `ownership_map.png`, `overlap_heatmap.png`, `unassigned_regions.png`, `qa_failed_regions.png`, `retry_regions.png`, and `reconstruction_diff.png`. The reconstruction report uses premultiplied-RGB SSIM, alpha gaps, edge difference, and optional cached LPIPS weights; it never downloads model weights automatically.

The implementation is covered by `tests/test_p1_geometry.py` for adaptive windows, prompt coordinates, candidate ranking, thin structures, exclusive ownership, terrain merging, occlusion precedence, residual semantics, and alpha-aware reconstruction differences.

## Retry contracts

- `max_retry` bounds graph-level object QA/retry passes; `p1.max_segmentation_retry` additionally caps each object's repair attempts. The smaller remaining budget applies.
- `p1.max_detection_retry` caps local detection attempts per problem-region/category key across scene passes. `p1.max_scene_retry` caps scene passes; `max_problem_regions` and `scene_loop.max_objects` bound work and output size. A zero budget executes no associated inference.
- `MISSED_DETECTION` and `UNASSIGNED_REGION` invoke the detector on expanded crops, then local SAM and instance QA. Recovered boxes are mapped once to original coordinates. IDs and evidence filenames stay distinct across retries, and accepted detections enter the exported detection inventory.
- Unresolved P0 tile fragments enter the targeted queue. `CROSS_TILE_FRAGMENT` repairs require a complete local re-observation with global overlap and appearance evidence, followed by improved QA. A mask alone cannot clear an artificial tile-boundary flag.
- `WRONG_CATEGORY` uses vision crop classification. Without a configured vision reviewer it remains manual review; rerunning SAM cannot resolve it.
- Mask repairs vary context, resolution and point prompts. Neighbor context survives single-object retries. Failed/worse candidates do not replace prior assets. Every attempted repair records its reason and acceptance result.
- Vision missed-object proposals remain unresolved until a corresponding accepted detection appears, even if texture heuristics produce no candidate. Smooth unassigned regions can also enter the queue.
- Mock mode skips local discovery because crop-relative fixture boxes would fabricate objects. `--exercise-retry` is explicitly labeled simulation.

## Export and diagnostics

`scene.json.p1_summary` and `metadata/summary.json` are generated from the same final state. `scene.json.retry_history` mirrors `metadata/retries.json`. Asset references in these manifests, ownership tables, terrain records, and retry evidence are relative to the output root, including JSON files under `metadata/`. Node updates return the summary without mutating their input state.

P0 scan coverage is preserved as `diagnostics/detection_scan_coverage.png`; P1 semantic/detection review uses `coverage_map.png`. The HTML report distinguishes them and can be generated with P0 diagnostics disabled. Fully occluded instances keep their candidate evidence but have no visible PNG; they are not relabeled as failed SAM masks. Terrain QA failures remain explicit after terrain records are removed from the instance list.

The reconstruction comparison ignores RGB hidden behind zero alpha. Visible and inferred terrain assets are separate, and completion preserves observed alpha. Source ownership is measured only from visible masks. Scene composition consumes the accepted asset (possibly amodal) and residual; generated alpha is clipped only on the scene canvas.

## Configuration and validation

Set `p1.enabled: false` to retain the P0 segmentation/refinement path. P1 diagnostics are independent of `detection.diagnostics.enabled`; disable P1 as well when running a P0-only diagnostics-off test. Set `p1.preserve_residual_background: false` to make unassigned source pixels remain transparent in reconstruction.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m app.demo --output input/p1_demo.png
.venv/bin/python main.py --input input/p1_demo.png --output output/p1_demo --mock
```

`tests/test_p1_pipeline.py` covers export consistency and portability, graph feature combinations, zero/exhausted budgets, unique retry IDs, improved-only replacements, classification and fragment repair, terrain completion, occlusion, and missed-object retention. Existing P0, Qwen, checkpoint, and failure-exhaustion tests remain part of the full suite.

## Evidence limits

Tests and the synthetic mock demo validate mechanics, not real-image segmentation accuracy. A residual-backed SSIM of 1.0 does not establish semantic completeness; inspect `unassigned_ratio`, problem regions, and the summary status. `terrain_completion` currently uses conservative enclosed-hole inference with OpenCV, not generative scene understanding. Hybrid component names remain uncertain proposals until separate geometry is supplied. Source SAM masks are not amodal masks; accepted amodal candidates now have explicit reconstructed alpha and expanded geometry (see AMODAL_RECONSTRUCTION.md). LPIPS remains disabled by default and reports unavailable when optional dependencies or cached weights are absent.

## Completion metrics and schema 1.2

`completion_metrics` distinguishes semantic decomposition completeness from reconstruction fidelity. `semantic_coverage` is the union of QA-passed source-visible instance/terrain masks and excludes residual/background; `reconstruction.similarity` is full-canvas pixel fidelity and may be 1.0 while semantic coverage is low. `asset_ready` requires a passing base asset and exportable pixels; optional upscale is excluded.

The portable manifest is now schema `1.2`. `schemas.migration.normalize_manifest` reads 1.1 and adds `geometry.visible_*` and `geometry.full_asset_*`, `base_asset_status`, and `enhancement` without rejecting old checkpoints.
