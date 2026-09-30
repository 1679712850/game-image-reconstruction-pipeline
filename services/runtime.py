"""Create service instances once and inject them into node factories."""
from dataclasses import dataclass
from pathlib import Path

from app.models import ModelConfig, load_models
from services.configured_scene_service import ConfiguredSceneService

from services.grounding_service import GroundingService
from services.image_edit_service import ImageEditService
from services.qwen_layered_service import QwenLayeredService
from services.sam_service import SAMService
from services.upscale_service import UpscaleService
from services.vlm_service import VLMService
from services.scene_review_service import SceneReviewService


@dataclass(frozen=True)
class ServiceBundle:
    """Replace individual services with real adapters or test doubles."""

    vlm: VLMService
    grounding: GroundingService
    sam: SAMService
    upscale: UpscaleService
    layered: QwenLayeredService
    image_edit: ImageEditService
    reviewer: SceneReviewService | None = None

    @classmethod
    def create(cls, mock: bool = True, models: ModelConfig | None = None, reviewer_backend: str = "rules") -> "ServiceBundle":
        """The only default composition point for service initialization."""
        if not mock:
            options = models or load_models(Path(__file__).resolve().parents[1] / "config" / "models.yaml")
            return cls(
                vlm=ConfiguredSceneService(options),
                grounding=GroundingService(False, options), sam=SAMService(False, options),
                upscale=UpscaleService(False, backend="lanczos"),
                layered=QwenLayeredService(False, options), image_edit=ImageEditService(False, options),
                reviewer=SceneReviewService(reviewer_backend, options.scene_reviewer),
            )
        return cls(
            vlm=VLMService(mock), grounding=GroundingService(mock),
            sam=SAMService(mock), upscale=UpscaleService(mock),
            layered=QwenLayeredService(mock, models), image_edit=ImageEditService(mock, models),
            reviewer=SceneReviewService("rules"),
        )

    def provenance(self, *, layered_enabled: bool = False, image_edit_enabled: bool = False) -> dict[str, str]:
        """Describe each active stage without claiming real VLM or super-resolution."""
        return {
            "analysis": "configured_categories" if isinstance(self.vlm, ConfiguredSceneService) else ("mock" if self.vlm.mock else "langchain_vlm"),
            "grounding": "mock" if self.grounding.mock else "grounding_dino_transformers",
            "segmentation": "mock" if self.sam.mock else "sam2_official",
            "scene_review": self.reviewer.backend if self.reviewer else "rules",
            "upscale": self.upscale.backend,
            "layer_decomposition": ("mock_passthrough" if self.layered.mock else "qwen_image_layered_local") if layered_enabled else "disabled",
            "object_completion": ("mock_noop" if self.image_edit.mock else "qwen_image_edit_local") if image_edit_enabled else "disabled",
        }
