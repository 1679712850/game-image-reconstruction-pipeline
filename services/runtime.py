"""Create service instances once and inject them into node factories."""
from dataclasses import dataclass

from services.grounding_service import GroundingService
from services.image_edit_service import ImageEditService
from services.qwen_layered_service import QwenLayeredService
from services.sam_service import SAMService
from services.upscale_service import UpscaleService
from services.vlm_service import VLMService


@dataclass(frozen=True)
class ServiceBundle:
    """Replace individual services with real adapters or test doubles."""

    vlm: VLMService
    grounding: GroundingService
    sam: SAMService
    upscale: UpscaleService
    layered: QwenLayeredService
    image_edit: ImageEditService

    @classmethod
    def create(cls, mock: bool = True) -> "ServiceBundle":
        """The only default composition point for service initialization."""
        return cls(
            vlm=VLMService(mock), grounding=GroundingService(mock),
            sam=SAMService(mock), upscale=UpscaleService(mock),
            layered=QwenLayeredService(mock), image_edit=ImageEditService(mock),
        )
