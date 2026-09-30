"""Validated pixel-space geometry and object records."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.qa import ObjectQA


class BBox(BaseModel):
    """Integer xywh; right/bottom boundaries are exclusive."""

    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class Pivot(BaseModel):
    """Ground contact point in the original crop's pixel space."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)


class SceneObject(BaseModel):
    """Paths are absolute in state and relative in the exported manifest."""

    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    crop_bbox: BBox | None = None
    mask_path: str | None = None
    asset_path: str | None = None
    hd_asset_path: str | None = None
    pivot: Pivot | None = None
    z_order: float = 0
    status: Literal["detected", "segmented", "cropped", "pass", "retry", "manual_review"] = "detected"
    logical_size: tuple[int, int] | None = None
    texture_size: tuple[int, int] | None = None
    texture_scale: float = Field(default=1, gt=0)
    qa: ObjectQA | None = None
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    error: str | None = None
    mock_retry_resolved: bool = False
