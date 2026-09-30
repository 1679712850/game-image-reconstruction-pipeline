"""Validated configuration, separate from workflow state."""
from pathlib import Path
from typing import Self

import yaml
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


class UpscaleConfig(Options):
    enabled: bool = True


class ReconstructionConfig(Options):
    enabled: bool = True


class PipelineConfig(Options):
    mock: bool = True
    max_retry: int = Field(default=1, ge=0, le=100)
    exercise_retry: bool = False
    crop: CropConfig = Field(default_factory=CropConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    upscale: UpscaleConfig = Field(default_factory=UpscaleConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)

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
