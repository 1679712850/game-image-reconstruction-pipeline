"""Official SAM 2 box-prompt segmentation with a lightweight mock mode."""
import numpy as np

from app.models import ModelConfig
from app.paths import read_rgba
from cv.mask import rectangle_mask
from schemas.object import BBox
from services.model_support import ModelUnavailableError, torch_runtime


class SAMService:
    """One cached model per service, one image embedding per segment call."""

    def __init__(self, mock: bool = True, config: ModelConfig | None = None):
        self.mock = mock
        self.config = config or ModelConfig()
        self._predictor = None
        self._torch = None
        self._device = None

    def load(self) -> None:
        """Load official SAM 2 from a local checkpoint or a cached Hub file."""
        if self.mock or self._predictor is not None:
            return
        torch, device = torch_runtime(self.config.device)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download
            settings = self.config.sam
            checkpoint = settings.checkpoint
            if checkpoint is None:
                checkpoint = hf_hub_download(
                    repo_id=settings.repo_id, filename=settings.filename,
                    revision=settings.revision, cache_dir=str(self.config.cache_dir),
                    local_files_only=self.config.local_files_only,
                )
            model = build_sam2(settings.model_config_name, str(checkpoint), device=device, apply_postprocessing=False)
            model.eval()
            predictor = SAM2ImagePredictor(model, mask_threshold=settings.mask_threshold)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ModelUnavailableError(f"SAM 2 load failed: {error}. Install requirements-sam2.txt and check checkpoint/config/device settings.") from error
        self._torch, self._device, self._predictor = torch, device, predictor

    def segment(self, image_path: str, detections: list[dict]) -> list[dict]:
        """Return full-image uint8 masks, keeping IDs, categories and confidence."""
        if not detections:
            return []
        image = read_rgba(image_path).convert("RGB")
        width, height = image.size
        for detection in detections:
            box = BBox.model_validate(detection["bbox"])
            if box.x + box.w > width or box.y + box.h > height:
                raise ValueError("SAM 2 box prompt lies outside source image")
        if self.mock:
            return [{**item, "mask": rectangle_mask(width, height, item["bbox"])} for item in detections]
        self.load()
        objects = []
        with self._torch.inference_mode():
            self._predictor.set_image(np.array(image, copy=True))
            try:
                for item in detections:
                    box = item["bbox"]
                    prompt = np.array([box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']], dtype=np.float32)
                    masks, scores, _ = self._predictor.predict(
                        box=prompt, multimask_output=self.config.sam.multimask_output,
                        return_logits=False, normalize_coords=True,
                    )
                    masks, scores = np.asarray(masks), np.asarray(scores).reshape(-1)
                    if masks.ndim == 2:
                        masks = masks[None]
                    if masks.ndim != 3 or masks.shape[1:] != (height, width) or masks.shape[0] != scores.size:
                        raise ValueError("SAM 2 returned invalid mask dimensions")
                    if scores.size == 0 or not np.isfinite(scores).all() or not np.isfinite(masks).all():
                        raise ValueError("SAM 2 returned empty/non-finite predictions")
                    selected = masks[int(np.argmax(scores))]
                    objects.append({**item, "mask": (selected > 0).astype(np.uint8) * 255})
            finally:
                self._predictor.reset_predictor()
        return objects
