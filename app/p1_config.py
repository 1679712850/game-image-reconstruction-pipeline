"""Budgets and thresholds for source-coordinate scene decomposition."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class P1Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    max_local_crop_size: int = Field(default=1024, ge=64)
    min_local_size: int = Field(default=512, ge=64)
    max_sam_input_size: int = Field(default=2048, ge=64)
    min_mask_bbox_ratio: float = Field(default=.03, gt=0, le=1)
    max_mask_bbox_ratio: float = Field(default=1.5, ge=1)
    max_detection_retry: int = Field(default=2, ge=0, le=10)
    max_segmentation_retry: int = Field(default=3, ge=0, le=10)
    max_scene_retry: int = Field(default=2, ge=0, le=10)
    max_problem_regions: int = Field(default=12, ge=1, le=100)
    detection_prefetch_regions: int = Field(default=4, ge=1, le=16)
    coverage_cell_size: int = Field(default=128, ge=16)
    edge_density_threshold: float = Field(default=.08, gt=0, le=1)
    unassigned_threshold: float = Field(default=.01, ge=0, le=1)
    overlap_threshold: float = Field(default=.01, ge=0, le=1)
    preserve_residual_background: bool = True
    terrain_completion: bool = True
    lpips: bool = False

    @model_validator(mode='after')
    def validate_sizes(self):
        if self.max_sam_input_size < self.max_local_crop_size:
            raise ValueError('max_sam_input_size must cover max_local_crop_size')
        return self
