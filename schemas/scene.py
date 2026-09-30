"""Scene understanding and portable export schema."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.object import SceneObject


class SceneAnalysis(BaseModel):
    """Structured-output target for a future LangChain VLM adapter."""

    model_config = ConfigDict(extra="forbid")
    projection: Literal["isometric", "top_down", "perspective", "unknown"]
    categories: list[str] = Field(default_factory=list)
    description: str


class SceneInfo(BaseModel):
    """Logical scene coordinates are independent of texture scale."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    projection: Literal["isometric", "top_down", "perspective", "unknown"]


class ExportObject(SceneObject):
    """Portable asset aliases for downstream scene consumers."""

    asset: str | None = None
    hd_asset: str | None = None


class SceneManifest(BaseModel):
    """Versioned manifest with explicit mock and review information."""

    schema_version: str = "1.0"
    mock: bool
    scene: SceneInfo
    description: str
    layers: list[dict] = Field(default_factory=list)
    objects: list[ExportObject]
    retry_count: int
    unresolved_objects: list[str] = Field(default_factory=list)
    reconstruction: str | None = None
    reconstruction_score: float | None = None
    reconstruction_score_definition: str = "1 - mean absolute RGBA error / 255 over the full canvas"
