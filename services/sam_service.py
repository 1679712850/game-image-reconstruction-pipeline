"""SAM 2 boundary; transient masks are persisted by the calling node."""
from app.paths import read_rgba
from cv.mask import rectangle_mask


class SAMService:
    """Own segmentation model lifecycle outside the nodes."""

    def __init__(self, mock: bool = True):
        self.mock = mock

    def segment(self, image_path: str, detections: list[dict]) -> list[dict]:
        """Return detection records with full-resolution uint8 masks."""
        if not self.mock:
            raise NotImplementedError("TODO: connect SAM 2 in SAMService.segment")
        width, height = read_rgba(image_path).size
        return [{**item, "mask": rectangle_mask(width, height, item["bbox"])} for item in detections]
