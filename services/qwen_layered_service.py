"""Extension contract for semantic RGBA decomposition, unused by V1."""
from pathlib import Path


class QwenLayeredService:
    """Keep the adapter callable in mock mode without loading weights."""

    def __init__(self, mock: bool = True):
        self.mock = mock

    def decompose_layers(self, image_path: str) -> list[dict]:
        """Mock returns one explicit passthrough reference, not inferred layers."""
        if self.mock:
            return [{"name": "mock_source", "image_path": str(Path(image_path).resolve()), "mock": True}]
        raise NotImplementedError("TODO: connect Qwen-Image-Layered")
