"""Serializable high-recall detection candidates.

Candidates are intentionally richer than :class:`SceneObject`: they retain
where and how a model saw an object, including rejected and merged records.
The bbox uses absolute pixel ``(x1, y1, x2, y2)`` coordinates. SceneObject
conversion is explicit at the existing xywh boundary.
"""
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionCandidate(BaseModel):
    """A model observation before segmentation and conservative filtering."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    id: str
    category: str
    group: str = "other"
    subtype: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    source: str = "global"
    tile_id: str | None = None
    is_truncated: bool = False
    truncated_edges: list[str] = Field(default_factory=list)
    detection_method: str = "grounding"
    parent_id: str | None = None
    merged_from: list[str] = Field(default_factory=list)
    reject_reason: str | None = None
    observations: list[dict] = Field(default_factory=list)
    review_required: bool = False
    review_reason: str | None = None
    redetected: bool = False
    appearance: list[float] = Field(default_factory=list)
    window: tuple[int, int, int, int] | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, box):
        if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Candidate bbox must be positive-area absolute pixel xyxy")
        return box

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])

    def as_bbox(self) -> dict[str, int]:
        x, y, x2, y2 = self.bbox
        return {"x": x, "y": y, "w": x2-x, "h": y2-y}

    def serializable(self) -> dict:
        return self.model_dump(mode="json", exclude={"mask", "appearance"})
