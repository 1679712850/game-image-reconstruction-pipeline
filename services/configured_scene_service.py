"""Deterministic scene description for real detection without requiring a VLM."""
from app.models import ModelConfig
from schemas.scene import SceneAnalysis
from services.vlm_service import VLMService


class ConfiguredSceneService(VLMService):
    """Categories and projection are user configuration, not VLM predictions."""

    def __init__(self, config: ModelConfig):
        self.mock = False
        self.config = config

    def analyze_scene(self, image_path: str) -> SceneAnalysis:
        """Return explicit detection prompts without inferring scene semantics."""
        return SceneAnalysis(
            projection=self.config.projection,
            categories=self.config.categories,
            description="Configured categories/projection; Grounding DINO detects instances and SAM 2 segments them. No VLM scene inference.",
        )
