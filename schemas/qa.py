"""QA outcomes and the bounded retry decision vocabulary."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ObjectQA(BaseModel):
    """A decision and an explanation; V1 uses deterministic CV rules."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["pass", "retry", "manual_review"]
    reason: str
    retry_strategy: Literal[
        "none", "expand_crop", "change_prompt",
        "rerun_segmentation", "merge_neighbor_tiles",
    ] = "none"
