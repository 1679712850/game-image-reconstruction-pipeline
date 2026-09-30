"""Serializable generated candidates, separate from verified source-image assets."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.object import AssetBBox


class LayerAsset(BaseModel):
    """Scene-aligned RGBA layer; order records model output, not inferred Y-sort."""

    model_config = ConfigDict(extra="forbid")
    id: str
    asset_path: str
    order: int = Field(ge=0)
    canvas_size: tuple[int, int]
    model_size: tuple[int, int]
    mock: bool
    status: Literal["mock_passthrough", "manual_review"]


class ObjectEditRequest(BaseModel):
    """White mask pixels permit RGB edits in the object's original crop space."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    object_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    mask_path: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class ObjectEditResult(BaseModel):
    """A completion record whose alpha is independently reconstructed."""

    model_config = ConfigDict(extra="forbid")
    object_id: str
    source_asset_path: str
    mask_path: str
    asset_path: str
    prompt: str
    crop_bbox: AssetBBox
    logical_size: tuple[int, int]
    mock: bool
    status: Literal["mock_noop", "manual_review"]
    alpha_policy: Literal["resegmented", "visible_hint", "pending_segmentation"] = "pending_segmentation"
