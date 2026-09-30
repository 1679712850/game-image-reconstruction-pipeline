"""Task budgets and versioned inference caching (GiB, seconds)."""
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResourcesConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    max_vram: float = Field(default=22, gt=0)
    soft_vram: float = Field(default=20, gt=0)
    max_ram: float = Field(default=64, gt=0)
    gpu_models_max_resident: int = Field(default=1, ge=1)
    max_generation_retry: int = Field(default=2, ge=0, le=10)
    max_oom_retry: int = Field(default=2, ge=0, le=5)
    keep_alive: dict[str, bool] = Field(default_factory=lambda: {'sam': True})
    estimated_vram: dict[str, float] = Field(default_factory=lambda: {
        'grounding': 2, 'sam': 3, 'image_edit': 18, 'layered': 18, 'qwen_vl': 18})
    estimated_ram: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode='after')
    def limits(self):
        if self.soft_vram > self.max_vram or any(v < 0 for v in [*self.estimated_vram.values(), *self.estimated_ram.values()]):
            raise ValueError('soft_vram must not exceed max_vram; estimates must be nonnegative')
        return self


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    enabled: bool = True
    directory: Path | None = None  # Default: output_dir/cache; shared directories are opt-in.
    pipeline_version: str = 'engineering-v1.2'
    prompt_version: str = 'amodal-candidate-v1'


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    automatic: bool = True
    accept_threshold: float = Field(default=.80, ge=0, le=1)
    retry_threshold: float = Field(default=.55, ge=0, le=1)
    severe_occlusion: float = Field(default=.35, ge=0, le=1)
    export_psd: bool = True

    @model_validator(mode='after')
    def thresholds(self):
        if self.retry_threshold > self.accept_threshold:
            raise ValueError('retry_threshold must not exceed accept_threshold')
        return self
