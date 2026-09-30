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
    tiled: bool = True
    tile_size: int = Field(default=768, ge=64)
    tile_overlap: float = Field(default=0.25, ge=0, lt=0.75)
    include_full_image: bool = True
    prompt_group_size: int = Field(default=6, ge=1, le=32)
    prompts: dict[str, str] = Field(default_factory=dict)
    relax_from_round: int = Field(default=3, ge=2)
    relaxed_box_threshold: float = Field(default=0.25, ge=0, le=1)
    relaxed_text_threshold: float = Field(default=0.20, ge=0, le=1)

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, values: dict[str, str]) -> dict[str, str]:
        """Each canonical category maps to one unambiguous English phrase."""
        phrases = [v.strip().lower() for v in values.values()]
        if len(set(phrases)) != len(phrases) or any(not v or any(c in v for c in ".\n\r") for v in phrases):
            raise ValueError("Detection prompts must be unique nonempty phrases without periods/newlines")
        return dict(zip(values, phrases))


class SAMConfig(Options):
    """Official SAM 2 config plus either a local checkpoint or Hub source."""

    model_config_name: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint: Path | None = None
    repo_id: str = "facebook/sam2.1-hiera-tiny"
    filename: str = "sam2.1_hiera_tiny.pt"
    revision: str = "main"
    multimask_output: bool = False
    mask_threshold: float = 0.0
    local_retry: bool = True
    local_padding: int = Field(default=64, ge=0)


class SceneReviewerConfig(Options):
    """OpenAI-compatible vision endpoint; credentials stay in environment only."""

    model: str = ""
    base_url: str | None = None
    api_key_env: str = "VLM_API_KEY"
    timeout: float = Field(default=60, gt=0, le=300)
    max_retries: int = Field(default=1, ge=0, le=3)
    image_long_edge: int = Field(default=1536, ge=256, le=4096)


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
    scene_reviewer: SceneReviewerConfig = Field(default_factory=SceneReviewerConfig)

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
