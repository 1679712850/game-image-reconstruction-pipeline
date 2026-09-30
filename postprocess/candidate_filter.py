"""Every rejected observation gets an explicit reason; weak props stay reviewable."""
from postprocess.small_object_filter import protect_small, threshold_for


def filter_candidates(candidates, image_area, config):
    kept, rejected, review = [], [], []
    for candidate in candidates:
        if candidate.area <= 0:
            candidate.reject_reason = "invalid_bbox"
        elif candidate.is_truncated:
            candidate.review_required = True
            candidate.review_reason = "unresolved_truncation"
            review.append(candidate)
            continue  # Never emit a half-object asset.
        elif candidate.confidence < threshold_for(candidate, image_area, config):
            if protect_small(candidate):
                candidate.review_required = True
                candidate.review_reason = "low_confidence_small_object"
                review.append(candidate)
            else:
                candidate.reject_reason = "low_confidence"
        if candidate.reject_reason:
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected, review
