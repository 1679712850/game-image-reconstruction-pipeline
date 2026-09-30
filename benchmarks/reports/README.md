# Real-image engineering runs

All results use the repository's 1536×1024 real scene, local Grounding DINO tiny + SAM 2.1 on CPU, no generation, no HD, no cache, unlimited objects, one production scene round and at most three truncation recovery calls. Both models actually ran. Scope is the annotated courtyard ROI, three standing stone lanterns, one manually traced **draft** visible mask. No full-map accuracy claim is made.

| Run | Planner | Calls | Recall | Precision | Duplicates | Mask IoU | Boundary F | Runtime s | Peak RAM MiB | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [baseline](baseline/benchmark_report.md) | disabled | 19 | 1.0 | 1.0 | 0 | .586 | .808 | 216.91 | 2079.4 | reference |
| [aggressive](current/benchmark_report.md) | 2 groups, explicit classes may be omitted | 13 | 0 | 0 | 0 | unavailable | unavailable | 95.11 | 2258.3 | FAIL |
| [4 groups without preservation](conservative/benchmark_report.md) | drops tombstone from shared prompt | 19 | 1.0 | .375 | .25 | .594 | .808 | 249.31 | 2309.9 | FAIL |
| [safe default strategy](safe/benchmark_report.md) | preserves explicit scene classes | 19 | 1.0 | 1.0 | 0 | .586 | .808 | 187.20 | 2264.5 | WARN (draft GT) |

Removing a class from a DINO prompt changed its predictions, not just scan cost. Therefore the default preserves explicit scene categories and caps automatic taxonomy expansion groups. The final 6-class benchmark uses the same prompts/call count as baseline. Its runtime difference is a single-run observation, **not demonstrated optimization speedup**. Peak RAM did not improve in this comparison. Full-taxonomy call reduction is tested at the scheduler level; real quality needs a broader annotated set.

Final whole-scene prediction summary: 122 detections; 31 base-ready, 87 review, 4 rejected; QA-passed semantic coverage 42.29%; unassigned 57.71%; residual 48.83%; reconstruction 1.0. Residual is contained in unassigned. The QA heuristic can accept incorrect semantics; the high reconstruction score does not certify decomposition.

Initial coarse annotation mistakenly included a roofless pillar and omitted a small lantern. It was corrected from magnified source pixels, and all saved predictions were re-evaluated against the same corrected annotation (identical annotation/image/config digest). Earlier intermediate results are superseded and ignored in Git. The GT remains draft, not human-reviewed. Baseline initially hit a RAM estimate guard; that incomplete attempt was excluded, and the recorded complete baseline used explicit tiny-model RAM estimates (1.0 / 0.6 GiB). No weights were downloaded. A profiler/Torch import race discovered during the run was fixed; initial baseline peak RSS uses span samples, subsequent runs also use the periodic sampler. These differences further limit performance conclusions.

The large `pipeline/` folders remain locally for inspection but are ignored in Git. JSON/Markdown reports and visualizations are retained. Reproduce with the configs in `../configs/`; see [the engineering guide](../../docs/ENGINEERING_EVALUATION.md) for complete commands and limitations.
