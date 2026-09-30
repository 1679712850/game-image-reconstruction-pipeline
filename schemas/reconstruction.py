"""Amodal predictions use normalized source-crop coordinates, including outside it."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OcclusionAnalysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    object_id: str
    object_type: str
    occlusion_ratio: float = Field(ge=0, le=1)
    reconstruction_confidence: float = Field(ge=0, le=1)
    needs_completion: bool
    occluded_directions: list[Literal['top', 'bottom', 'left', 'right', 'top_left',
        'top_right', 'bottom_left', 'bottom_right', 'interior']] = Field(default_factory=list)
    likely_full_shape: str = ''
    boundary_reasoning: str = ''
    symmetry: str = ''
    repeated_structure: str = ''
    # Polygon coordinates are relative to ORIGINAL CROP width/height; negative and >1 allowed.
    full_shape_polygons: list[list[tuple[float, float]]] = Field(default_factory=list)
    max_expansion_ratio: float = Field(default=2.5, ge=1, le=8)
    evidence: Literal['vision', 'unavailable'] = 'vision'

    @model_validator(mode='after')
    def geometry(self):
        if self.occlusion_ratio >= .05 and not self.needs_completion:
            raise ValueError('An occluded object requires completion')
        for polygon in self.full_shape_polygons:
            if len(polygon) < 3 or len(polygon) > 256:
                raise ValueError('Full-shape polygons require 3..256 vertices')
            if any(not math.isfinite(v) or not -2 <= v <= 3 for point in polygon for v in point):
                raise ValueError('Amodal polygon exceeds bounded crop coordinates [-2, 3]')
        if len(self.full_shape_polygons) > 16:
            raise ValueError('Too many amodal polygons')
        if self.evidence == 'vision' and self.needs_completion and not self.full_shape_polygons:
            raise ValueError('Semantic completion requires full-shape polygons')
        return self

    @property
    def visible_ratio(self):
        return 1 - self.occlusion_ratio
