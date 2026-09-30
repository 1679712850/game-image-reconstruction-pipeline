"""Grounding DINO boundary with dimension-relative mock detections."""
from app.paths import read_rgba


class GroundingService:
    """Initialize a future detector here or inject a subclass at graph creation."""

    def __init__(self, mock: bool = True):
        self.mock = mock

    def detect(self, image_path: str, categories: list[str]) -> list[dict]:
        """Return original-image xywh boxes and bounded confidence scores."""
        if not self.mock:
            raise NotImplementedError("TODO: connect Grounding DINO in GroundingService.detect")
        width, height = read_rgba(image_path).size
        specs = [
            ("tree", 0.91, 0.08, 0.38, 0.14, 0.35),
            ("rock", 0.84, 0.52, 0.65, 0.16, 0.13),
            ("building", 0.88, 0.69, 0.38, 0.19, 0.24),
            ("mountain", 0.76, 0.23, 0.08, 0.40, 0.25),
        ]
        found = []
        for category, confidence, xf, yf, wf, hf in specs:
            if category not in categories:
                continue
            x, y = min(width - 1, int(width * xf)), min(height - 1, int(height * yf))
            w, h = min(width - x, max(1, int(width * wf))), min(height - y, max(1, int(height * hf)))
            found.append({"category": category, "confidence": confidence, "bbox": {"x": x, "y": y, "w": w, "h": h}})
        return found
