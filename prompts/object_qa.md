Future QA agent prompt; the V1 graph uses deterministic CV rules.
Assess an object's visible cutout together with its source crop and CV metrics.
Return ObjectQA with status (pass, retry, manual_review), reason, and one
retry_strategy (none, expand_crop, change_prompt, rerun_segmentation,
merge_neighbor_tiles). Explain uncertainty; do not claim occluded pixels exist.
Retry limits are enforced by the graph, not the model.
