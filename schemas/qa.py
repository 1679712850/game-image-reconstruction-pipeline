"""QA outcomes and the bounded retry decision vocabulary."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ObjectQA(BaseModel):
    """A decision and an explanation; V1 uses deterministic CV rules."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["pass", "retry", "manual_review"]
    reason: str
    failure_types: list[Literal['MISSED_DETECTION', 'BAD_MASK', 'BACKGROUND_LEAK', 'MASK_TOO_SMALL',
        'MASK_TOO_LARGE', 'DUPLICATE_INSTANCE', 'CROSS_TILE_FRAGMENT', 'WRONG_CATEGORY',
        'PIXEL_CONFLICT', 'UNASSIGNED_REGION']] = Field(default_factory=list)
    retry_strategy: Literal[
        "none", "expand_crop", "change_prompt",
        "rerun_segmentation", "merge_neighbor_tiles",
    ] = "none"
