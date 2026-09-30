"""Validated configuration, separate from workflow state."""
from pathlib import Path
from typing import Self

import yaml
from app.detection_config import DetectionConfig
from app.p1_config import P1Config
from app.resource_config import ResourcesConfig, CacheConfig, CandidateConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Options(BaseModel):
    """Reject configuration typos instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CropConfig(Options):
    alpha_threshold: int = Field(default=8, ge=0, le=254)
    padding: int = Field(default=16, ge=0)


class QAConfig(Options):
    min_confidence: float = Field(default=0.35, ge=0, le=1)
    min_occupancy: float = Field(default=0.08, ge=0, le=1)
    max_mask_outside_bbox: float = Field(default=0.35, ge=0, le=1)


class SceneLoopConfig(Options):
    """Hard budgets override agent decisions; coverage is diagnostic, not accuracy."""

    enabled: bool = True
    max_rounds: int = Field(default=3, ge=1, le=20)
    max_objects: int = Field(default=300, ge=1, le=2000)
    target_coverage: float = Field(default=0.85, gt=0, le=1)
    min_coverage_gain: float = Field(default=0.002, ge=0, le=1)
    no_progress_patience: int = Field(default=2, ge=1, le=10)
    covered_box_threshold: float = Field(default=0.85, gt=0, le=1)
    reviewer: str = "rules"

    @model_validator(mode="after")
    def validate_reviewer(self) -> Self:
        if self.reviewer not in {"rules", "llm"}:
            raise ValueError("scene_loop.reviewer must be rules or llm")
        return self


class UpscaleConfig(Options):
    enabled: bool = True


class ReconstructionConfig(Options):
    enabled: bool = True


class OptionalStageConfig(Options):
    """Generative stages are opt-in and leave the default backbone unchanged."""

    enabled: bool = False


class PipelineConfig(Options):
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    p1: P1Config = Field(default_factory=P1Config)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    mock: bool = True
    max_retry: int = Field(default=1, ge=0, le=100)
    exercise_retry: bool = False
    crop: CropConfig = Field(default_factory=CropConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    upscale: UpscaleConfig = Field(default_factory=UpscaleConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    layer_decomposition: OptionalStageConfig = Field(default_factory=OptionalStageConfig)
    object_completion: OptionalStageConfig = Field(default_factory=OptionalStageConfig)
    scene_loop: SceneLoopConfig = Field(default_factory=SceneLoopConfig)

    @model_validator(mode="after")
    def validate_retry_demo(self) -> Self:
        """Synthetic failures are only permitted in mock mode."""
        if self.exercise_retry and not self.mock:
            raise ValueError("exercise_retry requires mock mode")
        return self


def load_config(path: Path) -> PipelineConfig:
    """Read UTF-8 YAML and validate all supplied options."""
    with path.open(encoding="utf-8") as stream:
        return PipelineConfig.model_validate(yaml.safe_load(stream) or {})
