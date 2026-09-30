"""Transformers Grounding DINO inference and dimension-relative mock mode."""
from app.paths import read_rgba
from app.models import ModelConfig
from services.model_support import ModelUnavailableError, torch_runtime
from services.detection_postprocess import postprocess_detections


class GroundingService:
    """Lazily load one detector per service, outside serialized graph state."""

    def __init__(self, mock: bool = True, config: ModelConfig | None = None):
        self.mock = mock
        self.config = config or ModelConfig()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def detect(self, image_path: str, categories: list[str]) -> list[dict]:
        """Return original-image xywh boxes and bounded confidence scores."""
        if not self.mock:
            return self._detect_real(image_path, categories)
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

    def load(self) -> None:
        """Load processor and weights once; never silently fall back to mock."""
        if self.mock or self._model is not None:
            return
        torch, device = torch_runtime(self.config.device)
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            options = dict(
                cache_dir=str(self.config.cache_dir),
                revision=self.config.grounding.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
            processor = AutoProcessor.from_pretrained(self.config.grounding.model_id, **options)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.grounding.model_id, **options)
            model.to(device).eval()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ModelUnavailableError(f"Grounding DINO load failed: {error}. Install requirements-vision.txt and check model/cache/device settings.") from error
        self._torch, self._device = torch, device
        self._processor, self._model = processor, model

    def _detect_real(self, image_path: str, categories: list[str]) -> list[dict]:
        """Infer grounded boxes in source pixels using period-separated prompts."""
        if not categories:
            return []
        image = read_rgba(image_path).convert("RGB")
        self.load()
        prompt = ". ".join(categories) + "."
        with self._torch.inference_mode():
            inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=self.config.grounding.box_threshold,
                text_threshold=self.config.grounding.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
        return postprocess_detections(
            result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist(),
            list(result["text_labels"]), categories, image.width, image.height,
            self.config.grounding.box_threshold, self.config.grounding.nms_iou,
            self.config.grounding.max_detections,
        )
