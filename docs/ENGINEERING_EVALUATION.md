# Engineering audit, contracts, and acceptance

## Architecture review

The existing LangGraph backbone remains authoritative: `GroundingService.detect_p0` → `P0DetectionPipeline` → grouped global/tile inference → coordinate restoration → fusion/filtering → SAM/local refinement → QA/retry → ownership/terrain → candidate selection/amodal repair → optional HD → metadata/reconstruction/export. Benchmark invokes `build_graph`, or evaluates its portable manifest and detection inventory. There is no benchmark detector.

Audit confirmed P0 had exhaustive taxonomy×window scanning, no labeled evaluator, HD exceptions overwrote base QA, and PSD concatenated all encoded layers. P1 already excluded residual from its predicted ownership coverage. P2 already supplied profiling, model lifecycle/cache, and crop offsets; these are reused. Existing local edits to amodal reconstruction were preserved.

## Active / deprecated / unused configuration

| Location | Active consumer / migration |
|---|---|
| `pipeline.yaml:detection` | `PipelineConfig.detection`, production global/tile planner, confidence, dedup, truncation, diagnostics |
| `pipeline.yaml:p1` | local SAM, terrain/ownership and failure-specific retries |
| `pipeline.yaml:scene_loop` | scene rounds, QA stop criteria, object limit; `max_objects:null` removes the limit |
| `pipeline.yaml:max_retry` | graph object QA retry cap; P1 per-object cap also applies |
| `pipeline.yaml:upscale` | enabled false/true/auto, required false; both Pydantic and YAML default false |
| `models.yaml:upscale` | backend/checkpoint/tile/model parameters; empty checkpoint never downloads a model |
| `models.yaml:grounding` prompts/model/revision/text thresholds | still active in production inference |
| `models.yaml:grounding` tile/NMS/max_detections/box threshold | legacy `detect/detect_round` API and P1 legacy local discovery retain these; **not** production P0 discovery settings. Graph warns for explicitly supplied legacy discovery fields. NMS also controls cross-round archived duplicate matching. |
| `detection.tiled` → `detection.tiling` | accepted alias + DeprecationWarning; canonical wins on conflict |
| `detection.multiscale` → `detection.multi_scale` | accepted alias + DeprecationWarning; canonical wins on conflict |
| root options plus `pipeline: {...}` | accepted; nested options override conflicting root fields with DeprecationWarning |

No second top-level legacy `detection` model exists in the current `ModelConfig`; unknown fields remain validation errors, not silently unused options. `config/categories.yaml:layers` controls layer assignment, `taxonomy/categories.py` canonical hierarchy, `taxonomy/prompt_groups.py` short inference prompts. These have distinct purposes. `candidates.export_psd` is still the PSD switch; `export` controls memory policy. Benchmark evaluator configuration is the separate `--config` YAML; `--pipeline` selects production configuration, avoiding another unused `benchmark.enabled` toggle.

Default changes are explicit: HD is now opt-in; category planner is enabled with at most 4 expansion groups/tile plus explicitly requested scene categories; one global pass and two tile rounds per run are permitted. To compare exhaustive allocation set `category_planner.enabled:false`; finite inference budgets remain active. Existing commands and service injection remain supported.

## Detection budget and allocation

The run context holds a budget persisted in checkpoint state. Global discovery runs once; regional tiling runs first, then a bounded remaining-image gap pass. Whitened resolved tiles are skipped in gap passes. Global/nearby candidate groups rank first, scene analysis categories supply fallback, and architecture→prop / vegetation→rock provide small bounded extensions. Explicit scene categories are preserved by default (`preserve_scene_categories:true`) because deleting members of a shared DINO prompt can change precision. The group cap applies to automatic taxonomy expansion. Unknown semantic groups are not declared absent; omitted groups are a recall trade-off. No additional VLM call is introduced. Multi-scale windows and original dedup remain intact.

`max_categories_per_pass` caps each inference prompt. `max_total_inference_calls` bounds scheduled model calls (cache hits also consume a reservation for reproducible scheduling); `max_retry_calls` additionally bounds truncated-region, local QA and OOM recovery inference. Global/tile pass counts are separate from individual window calls. Legacy custom detector implementations that do not use `_infer_image` remain governed by their existing adapter contract; their opaque internal inference cannot be counted reliably.

Candidates keep source/pass/tile/scale/category group/category/model/runtime in observations. Scan costs count inference/runtime/candidates/retained by semantic group; shared multi-group prompts can appear in multiple groups, and retained is not GT validity. Budget omissions and runtime failures remain visible.

## Metrics

| Metric | Definition |
|---|---|
| Recall / miss | one-to-one class-matched bbox IoU≥0.5 GT matches / instance GT; miss lists IDs |
| Precision | matched predictions / predictions in annotated category scope and ROI; duplicates count against precision |
| Duplicate | unmatched prediction meeting IoU threshold with already matched GT / predictions |
| Fragmentation | GT containing ≥2 distinct partial boxes / GT; containment and min area configurable, heuristic |
| Size recall | bbox area/original scene area, <0.001 small, ≥0.05 large, otherwise medium |
| IoU / Dice | visible GT/predicted foreground overlap; never bbox as a substitute for mask |
| Boundary F | precision/recall of morphological mask boundaries within configurable pixel tolerance |
| Completeness / contamination | intersection/GT pixels; extra predicted pixels/predicted pixels |
| Semantic coverage | union of QA-passed source-visible asset and terrain masks / nontransparent source pixels |
| Predicted semantic coverage | P1 assigned source masks including review candidates, residual excluded |
| Unassigned | 1 − QA-passed semantic coverage; residual is a **subset**, not an additive partition |
| Residual | unclassified source pixels actually preserved by fallback / eligible pixels |
| Background / instance pixels | QA-passed visible terrain / instance union; generated hidden parts and residual excluded |
| Asset-ready | base pixels and segmentation exist, detection/QA pass, no unresolved category/content/boundary review |
| Review / failed ratio | review / rejected base assets divided by evaluated asset count; detected inventory reported separately |
| Reconstruction fidelity | full-canvas RGBA similarity; may be 1.0 with very low semantic coverage |
| RAM / VRAM | sampled RSS/device allocation; CUDA allocator peaks included, unavailable values null |

Coverage is **predicted** completeness, not GT semantic correctness. Rule QA can accept a wrong object. GT mask metrics are conditional on matched mask-annotated instances; missing predicted mask scores zero, unmatched GT is reported as missed. Semantic-region GT receives separate mask metrics. Ignore regions are excluded from mask scoring. Predictions outside ROI or unannotated categories are unscored. ROI coverage and whole-scene completion are separately named.

Pipeline status: `completed` means the configured coverage threshold is met without review; `completed_with_review` means coverage met but review remains; `partial` means coverage is below threshold; resource recovery producing no assets is `failed`. Exceptions still exit nonzero. No status establishes human-certified asset-library fitness. `asset_library_eligible` is the existing stricter candidate/completion field and remains separate from base asset readiness.

## Upscale and geometry

`base_asset_status`, `enhancement`, and `review` are independent. Optional missing weights/backend/OOM/QA failure never changes the base asset or eligibility. `required:true` requests delivery review and sets the legacy delivery status to manual_review while preserving the factual base assessment. Mock/Lanczos is `preview_only`. Scale policy lives in `resolve_upscale_factor`: long edge <128 →4x, otherwise 2x.

Visible masks remain in source-image coordinates. Full/amodal assets may extend outside the scene; reconstructed local alpha and signed full bbox describe their own canvas. Scene composition clips only for preview. Geometry contains visible_bbox/visible_mask, full_asset_bbox/full_asset_canvas (width,height), full_asset_mask (local reconstructed alpha when available). Export derives visible bounds from the source mask when available and full bounds from the final accepted placement, never an unaccepted amodal proposal. Legacy mask fields remain for consumers.

## Schema migration

Exports are **1.2**, an additive upgrade: old field names, signed crop placement and path conventions remain. The signed full bbox behavior existed in the audited amodal implementation; no coordinate contract is newly reinterpreted. Consumers must opt into new geometry fields rather than interpreting visible SAM masks as completed alpha.

```python
from schemas.scene import SceneManifest
manifest = SceneManifest.model_validate_json(open('old_scene.json').read())
assert manifest.schema_version == '1.2'
# Or use schemas.migration.normalize_manifest(dict) before custom loading.
```

1.1 data is copied/normalized, new fields added conservatively; migration does not invent missing GT metrics or recover base readiness from historically contaminated HD status. Unknown versions fail clearly. Existing v1.1 readers using strict extra-field validation must update before reading 1.2.

## Performance

Model fingerprints use resolved path/file size/mtime/config/revision and cached Hub snapshot metadata, once per task/model configuration. No weight bytes are hashed. New run, different path or changed configuration regenerates the snapshot; in-place model changes during a run require restart. Input images and inference result blobs retain content hashing.

PSD v1 layers are cropped to alpha bounds with adjusted signed offsets, decoded one at a time, channels spooled into temporary files, then copied in blocks. The merged canvas still consumes scene-sized RAM and the largest input PNG is decoded once. Estimated peak RAM and budget policy are written to `scene.export.json`; over-budget mode warns and uses disk spooling, or `low_memory:false` refuses. This is not a hard OS memory ceiling and disk capacity can still fail export. Atomic final replacement prevents corrupt partial PSD files.

## Benchmark commands

```bash
# Full production graph, offline local models, unlimited objects:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m benchmarks.runner \
  --annotation benchmarks/annotations/sect_ruins_courtyard.json \
  --pipeline benchmarks/configs/sect_ruins_baseline.yaml \
  --models benchmarks/configs/sect_ruins_models.yaml \
  --config benchmarks/configs/default.yaml --output benchmarks/reports/my_baseline

# Same graph with conservative allocation, compare dimensions separately:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m benchmarks.runner \
  --annotation benchmarks/annotations/sect_ruins_courtyard.json \
  --pipeline benchmarks/configs/sect_ruins_safe.yaml \
  --models benchmarks/configs/sect_ruins_models.yaml \
  --config benchmarks/configs/default.yaml --output benchmarks/reports/my_current \
  --baseline benchmarks/reports/my_baseline/benchmark_report.json

# Re-evaluate without model inference:
.venv/bin/python -m benchmarks.runner \
  --annotation benchmarks/annotations/sect_ruins_courtyard.json \
  --manifest benchmarks/reports/my_current/pipeline/scene.json \
  --config benchmarks/configs/default.yaml --output benchmarks/reports/review
```

Outputs: JSON/Markdown report, GT/prediction/miss/FP/duplicate/fragment/mask/review/accepted visualizations, resolved run configuration and production artifacts. `max_objects` is forced to null for fresh benchmarks; re-evaluation reports actual recorded drops or unknown, never assumes no truncation. Baseline gate compares recall/precision/duplicates/runtime/boundary metrics separately, returns PASS/WARN/FAIL (exit 2 for FAIL), and checks annotation/image/inventory/evaluation config identity. Draft GT, mock, unknown truncation or missing mask data cannot produce a clean PASS. Resource failures force FAIL.

The included seed is three manually boxed stone lanterns plus one coarse hand-traced visible mask in a courtyard ROI, pending independent annotation review. It is a real image baseline but not an exhaustive multi-region/full-map acceptance set. More building/vegetation/water/occlusion regions and expert mask review are required before claiming semantic completeness improvements.

## Delivered validation and remaining risks

Full unittest suite: **177 tests passed** (6.452 seconds), including accepted-placement geometry and rejection of an incomplete initial baseline after resource recovery. `git diff --check` passes. Normal Mock CLI reaches export and now says `partial` despite 100% reconstruction: predicted ownership 21.5%, stricter QA-passed semantic coverage 11.5%, unassigned 88.5%, residual 78.5%. A separate regression fixture builds a real composite with residual and verifies exactly 21.5% semantic / 78.5% unassigned at similarity 1.0. Mock is used only for contracts, never the real-map metrics. See [the delivery report](IMPLEMENTATION_REPORT.md) for the file map, reproducible commands, and real-image results.

Real runs and rejected strategies are documented in [benchmarks/reports/README.md](../benchmarks/reports/README.md). Defaults preserve explicit scene prompt members after two measured regressions; 19 calls for the small six-category baseline and final configuration. This does not prove real full-taxonomy speedup, reduced memory, or map completeness. Expand GT to buildings/vegetation/water/occlusion and review the draft polygons before promoting WARN to acceptance PASS. Rule QA remains weaker than human asset QA. Opaque third-party detector internals cannot be fully budgeted; PSD still needs a merged canvas and disk space; CUDA/Real-ESRGAN/amodal neural quality were not run in this CPU benchmark. No claims are made about 8K peak memory from this 1536×1024 scene.
