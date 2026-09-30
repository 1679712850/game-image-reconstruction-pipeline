"""Original-image xyxy annotations; masks are scene-sized or ROI-sized PNGs."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class GTObject(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    category: str
    bbox: tuple[int, int, int, int]
    mask: str | None = None
    group: str = 'other'
    ignore: bool = False
    kind: Literal['instance', 'semantic_region'] = 'instance'

    @model_validator(mode='after')
    def valid_box(self):
        x, y, r, b = self.bbox
        if x < 0 or y < 0 or r <= x or b <= y:
            raise ValueError('GT bbox must be nonempty original-image xyxy')
        return self

class Annotation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene_id: str
    image: str
    roi: tuple[int, int, int, int]
    objects: list[GTObject]
    ignore_regions: list[tuple[int, int, int, int]] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    annotation_status: Literal['draft', 'reviewed'] = 'draft'
    provenance: str
    scope: str = 'Exhaustive for listed categories within ROI; other categories are unscored'

    @model_validator(mode='after')
    def validate_roi(self):
        x, y, r, b = self.roi
        if x < 0 or y < 0 or r <= x or b <= y:
            raise ValueError('ROI must be nonempty original-image xyxy')
        if len({o.id for o in self.objects}) != len(self.objects):
            raise ValueError('GT instance IDs must be unique')
        for obj in self.objects:
            a, c, d, e = obj.bbox
            if not (x <= a < d <= r and y <= c < e <= b):
                raise ValueError('GT objects must fit ROI; ignore ambiguous cut objects')
        if self.categories and any(o.category not in self.categories for o in self.objects if not o.ignore):
            raise ValueError('Every nonignored GT category must be in categories')
        for region in self.ignore_regions:
            a,c,d,e = region
            if not (x <= a < d <= r and y <= c < e <= b):
                raise ValueError('Ignore regions must be nonempty xyxy inside ROI')
        return self

class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    iou_threshold: float = Field(default=.5, gt=0, le=1)
    fragment_containment: float = Field(default=.8, gt=0, le=1)
    fragment_min_area: float = Field(default=.05, gt=0, le=1)
    small_area_ratio: float = Field(default=.001, gt=0, le=1)
    large_area_ratio: float = Field(default=.05, gt=0, le=1)
    boundary_tolerance: int = Field(default=2, ge=0)
    poor_mask_iou: float = Field(default=.5, ge=0, le=1)
    thresholds: dict[str, float] = Field(default_factory=lambda: {
        'recall_drop': .02, 'precision_drop': .03,
        'runtime_increase': .20, 'duplicate_rate_increase': .02})
