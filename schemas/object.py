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
    group: str = "other"
    subtype: str | None = None
    aliases: list[str] = Field(default_factory=list)
    source: str = "global"
    tile_id: str | None = None
    is_truncated: bool = False
    truncated_edges: list[str] = Field(default_factory=list)
    detection_method: str = "legacy"
    parent_id: str | None = None
    source_candidates: list[str] = Field(default_factory=list)
    merged_from: list[str] = Field(default_factory=list)
    merged: bool = False
    observations: list[dict] = Field(default_factory=list)
    redetected: bool = False
    review_required: bool = False
    review_reason: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    crop_bbox: BBox | None = None
    mask_path: str | None = None
    asset_path: str | None = None
    source_asset_path: str | None = None
    source_mask_path: str | None = None
    source_crop_bbox: BBox | None = None
    accepted_asset: str | None = None
    accepted_candidate_id: str | None = None
    asset_mask_path: str | None = None
    placement: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    needs_manual_review: bool = False
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
    element_type: Literal["terrain", "instance", "hybrid", "effect"] = "instance"
    layer_group: str = "other"
    requires_individual_export: bool = True
    requires_inpainting: bool = False
    classification_reason: str = ""
    classification_confidence: float | None = Field(default=None, ge=0, le=1)
    uncertain: bool = False
    hybrid_components: list[str] = Field(default_factory=list)
    candidate_mask_path: str | None = None
    visible_mask_path: str | None = None
    full_mask_path: str | None = None
    segmentation: dict = Field(default_factory=dict)
    ownership_priority: int = 0
    occludes: list[str] = Field(default_factory=list)
    negative_points: list[list[float]] = Field(default_factory=list)
    positive_points: list[list[float]] = Field(default_factory=list)
    retry_history: list[dict] = Field(default_factory=list)
    occluded_pixel_count: int = Field(default=0, ge=0)
    visible_pixel_count: int = Field(default=0, ge=0)
    ownership_pixel_count: int = Field(default=0, ge=0)
