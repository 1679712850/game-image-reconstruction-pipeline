"""Validated real-model settings; paths are relative to the YAML file."""
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
import yaml

from app.config import Options


class GroundingConfig(Options):
    """Transformers Grounding DINO and postprocessing options."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    revision: str = "main"
    box_threshold: float = Field(default=0.30, ge=0, le=1)
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    nms_iou: float = Field(default=0.5, ge=0, le=1)
    max_detections: int = Field(default=100, gt=0)


class SAMConfig(Options):
    """Official SAM 2 config plus either a local checkpoint or Hub source."""

    model_config_name: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint: Path | None = None
    repo_id: str = "facebook/sam2.1-hiera-tiny"
    filename: str = "sam2.1_hiera_tiny.pt"
    revision: str = "main"
    multimask_output: bool = False
    mask_threshold: float = 0.0


class QwenConfig(Options):
    """Local Diffusers settings; a blank path never resolves to a Hub ID."""

    model_path: Path | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    num_inference_steps: int = Field(default=50, gt=0)
    true_cfg_scale: float = Field(default=4.0, ge=1)
    negative_prompt: str = " "
    seed: int = Field(default=0, ge=0)

    @field_validator("model_path", mode="before")
    @classmethod
    def blank_path_is_unconfigured(cls, value: object) -> object:
        """Null/empty strings mean unconfigured, never the current directory."""
        return None if isinstance(value, str) and not value.strip() else value


class QwenLayeredConfig(QwenConfig):
    """Layer generation controls, independent of semantic detection categories."""

    layers: int = Field(default=4, gt=0, le=32)
    resolution: Literal[640, 1024] = 640
    cfg_normalize: bool = True
    use_en_prompt: bool = True


class ModelConfig(Options):
    """Explicit classical scene analysis and real detection/segmentation."""

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    cache_dir: Path = Path(".cache/models")
    local_files_only: bool = False
    categories: list[str] = Field(default_factory=lambda: ["tree", "rock", "mountain", "building", "bridge", "bush", "water", "road"])
    projection: Literal["isometric", "top_down", "perspective", "unknown"] = "unknown"
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    qwen_layered: QwenLayeredConfig = Field(default_factory=QwenLayeredConfig)
    qwen_image_edit: QwenConfig = Field(default_factory=QwenConfig)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        """Prevent ambiguous empty or multi-sentence detection prompts."""
        result = list(dict.fromkeys(value.strip().lower() for value in values))
        if not result or any(not value or any(c in value for c in ".\n\r") for value in result):
            raise ValueError("categories must be nonempty phrases without periods/newlines")
        return result

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """A configured local checkpoint must exist; do not fall back to Hub."""
        if self.sam.checkpoint is not None and not self.sam.checkpoint.is_file():
            raise ValueError(f"SAM 2 checkpoint not found: {self.sam.checkpoint}")
        return self


def load_models(path: Path) -> ModelConfig:
    """Load model options and resolve filesystem paths relative to their file."""
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("Model configuration must be a YAML mapping")
    data.setdefault("cache_dir", ".cache/models")
    for parent, key in (
        (data, "cache_dir"), (data.get("sam", {}), "checkpoint"),
        (data.get("qwen_layered", {}), "model_path"),
        (data.get("qwen_image_edit", {}), "model_path"),
    ):
        if parent.get(key) and str(parent[key]).strip():
            value = Path(parent[key]).expanduser()
            parent[key] = value if value.is_absolute() else path.parent / value
    model_id = data.get("grounding", {}).get("model_id", "")
    if model_id.startswith((".", "/", "~")):
        value = Path(model_id).expanduser()
        data["grounding"]["model_id"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return ModelConfig.model_validate(data)
