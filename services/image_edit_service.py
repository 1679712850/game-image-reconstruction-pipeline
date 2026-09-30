"""Extension contracts for occlusion analysis and object completion."""
from pathlib import Path


class ImageEditService:
    """No editing model is required for the first workflow."""

    def __init__(self, mock: bool = True):
        self.mock = mock

    def analyze_occlusion(self, objects: list[dict]) -> list[dict]:
        """Mock has no inferred occlusions."""
        if self.mock:
            return []
        raise NotImplementedError("TODO: Qwen-VL occlusion diagnosis")

    def complete_object(self, image_path: str, mask_path: str) -> str:
        """Mock is an explicit no-op; real inpainting is a future capability."""
        if self.mock:
            return str(Path(image_path).resolve())
        raise NotImplementedError("TODO: Qwen-Image-Edit / inpainting")
