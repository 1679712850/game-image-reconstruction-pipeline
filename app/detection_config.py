"""Independent P0 detection settings; model options remain in models.yaml."""
import warnings
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class GlobalDetectionConfig(Settings):
    enabled: bool = True


class TilingConfig(Settings):
    enabled: bool = True
    tile_size: int = Field(default=1024, ge=32)
    overlap: float = Field(default=.25, gt=0, lt=1)
    min_tile_size: int = Field(default=64, ge=1)


class MultiScaleConfig(Settings):
    enabled: bool = True
    scales: list[int] = Field(default_factory=lambda: [1024, 1536])


class ConfidenceConfig(Settings):
    default: float = Field(default=.4, ge=0, le=1)
    large_object: float = Field(default=.5, ge=0, le=1)
    small_object: float = Field(default=.25, ge=0, le=1)
    candidate_floor: float = Field(default=.10, ge=0, le=1)
    small_area_ratio: float = Field(default=.001, gt=0, lt=1)
    large_area_ratio: float = Field(default=.05, gt=0, le=1)


class DedupConfig(Settings):
    bbox_iou: float = Field(default=.45, gt=0, le=1)
    mask_iou: float = Field(default=.5, gt=0, le=1)
    overlap_ratio: float = Field(default=.7, gt=0, le=1)
    center_distance: float = Field(default=.65, gt=0)
    appearance_similarity: float = Field(default=.8, ge=0, le=1)


class TruncationConfig(Settings):
    edge_threshold: int = Field(default=12, ge=0)
    redetect_padding: int = Field(default=192, gt=0)
    max_redetections: int = Field(default=100, ge=0)


class DiagnosticsConfig(Settings):
    enabled: bool = True


class DetectionBudget(Settings):
    """Finite per-run inference budget shared by global, tile and recovery passes."""
    max_global_passes: int = Field(default=1, ge=0)
    max_tile_passes: int = Field(default=2, ge=0)
    max_categories_per_pass: int = Field(default=6, ge=1)
    max_total_inference_calls: int = Field(default=2000, ge=1)
    max_retry_calls: int = Field(default=100, ge=0)


class CategoryPlannerConfig(Settings):
    enabled: bool = True
    max_groups_per_tile: int = Field(default=4, ge=1)
    preserve_scene_categories: bool = True


class DetectionConfig(Settings):
    global_detection: GlobalDetectionConfig = Field(default_factory=GlobalDetectionConfig, alias="global")
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    multi_scale: MultiScaleConfig = Field(default_factory=MultiScaleConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    budget: DetectionBudget = Field(default_factory=DetectionBudget)
    category_planner: CategoryPlannerConfig = Field(default_factory=CategoryPlannerConfig)
    expand_categories: bool = True
    prompt_group_size: int = Field(default=6, ge=1, le=32)

    @model_validator(mode="before")
    @classmethod
    def aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for alias, active in [('tiled','tiling'),('multiscale','multi_scale')]:
            if alias in value:
                warnings.warn(f'detection.{alias} is deprecated; use detection.{active}', DeprecationWarning, stacklevel=2)
                old = value.pop(alias)
                value[active] = {**old, **value.get(active,{})}
        return value

    @model_validator(mode="after")
    def validate_scales(self):
        sizes = [self.tiling.tile_size, *self.multi_scale.scales]
        if any(size < self.tiling.min_tile_size for size in sizes):
            raise ValueError("Detection tile sizes must be >= min_tile_size")
        if self.multi_scale.enabled and not self.multi_scale.scales:
            raise ValueError("multi_scale.scales must not be empty")
        if not self.global_detection.enabled and not self.tiling.enabled:
            raise ValueError("At least one detection path must be enabled")
        if self.confidence.candidate_floor > self.confidence.small_object:
            raise ValueError("candidate_floor must not exceed small_object threshold")
        if self.confidence.small_area_ratio >= self.confidence.large_area_ratio:
            raise ValueError("small_area_ratio must be below large_area_ratio")
        return self
