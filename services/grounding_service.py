"""Transformers Grounding DINO inference and dimension-relative mock mode."""
from app.paths import read_rgba
from app.models import ModelConfig
from services.model_support import ModelUnavailableError, torch_runtime
from services.detection_postprocess import postprocess_detections, canonical_prompt_label
from cv.tiles import tile_windows
from PIL import Image


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
        """Run grouped full-image and overlapping tile detection."""
        return self.detect_round(image_path, categories, 1)[0]

    def detect_round(self, image_path: str, categories: list[str], round_index: int) -> tuple[list[dict], dict]:
        """Return global boxes plus per-window diagnostics; only late rounds relax thresholds."""
        if self.mock:
            found = self.detect(image_path, categories)
            return found, {"backend": "mock", "kept": len(found)}
        if not categories:
            return [], {"kept": 0}
        image = read_rgba(image_path).convert("RGB")
        self.load()
        cfg = self.config.grounding
        relaxed = round_index >= cfg.relax_from_round
        box_threshold = min(cfg.box_threshold, cfg.relaxed_box_threshold) if relaxed else cfg.box_threshold
        text_threshold = min(cfg.text_threshold, cfg.relaxed_text_threshold) if relaxed else cfg.text_threshold
        full = (0, 0, image.width, image.height)
        windows = tile_windows(image.width, image.height, cfg.tile_size, cfg.tile_overlap) if cfg.tiled else [full]
        if cfg.include_full_image:
            windows = list(dict.fromkeys([full, *windows]))
        candidates, scans = [], []
        groups = [categories[i:i + cfg.prompt_group_size] for i in range(0, len(categories), cfg.prompt_group_size)]
        for window in windows:
            for group in groups:
                found = self._infer_image(image.crop(window), group, box_threshold, text_threshold)
                scans.append({"window": list(window), "categories": group, "candidates": len(found)})
                for item in found:
                    box = item["bbox"]
                    candidates.append({**item, "bbox": {**box, "x": box["x"] + window[0], "y": box["y"] + window[1]}})
        result = postprocess_detections(
            [[o['bbox']['x'], o['bbox']['y'], o['bbox']['x'] + o['bbox']['w'], o['bbox']['y'] + o['bbox']['h']] for o in candidates],
            [o['confidence'] for o in candidates], [o['category'] for o in candidates], categories,
            image.width, image.height, box_threshold, cfg.nms_iou, cfg.max_detections,
        )
        return result, {"backend": "grounding_dino_transformers", "box_threshold": box_threshold,
                        "text_threshold": text_threshold, "scans": scans,
                        "before_merge": len(candidates), "kept": len(result)}

    def _infer_image(self, image: Image.Image, categories: list[str], box_threshold: float, text_threshold: float) -> list[dict]:
        """Infer a single window, mapping explicit prompt phrases back to canonical categories."""
        phrases = [self.config.grounding.prompts.get(c, c.replace('_', ' ')) for c in categories]
        prompt = ". ".join(phrases) + "."
        with self._torch.inference_mode():
            inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        # Transformers 5.17 tokenizer batch_decode([]) returns ['']; no boxes is still empty.
        if not boxes and not scores:
            return []
        found = postprocess_detections(
            boxes, scores,
            [canonical_prompt_label(label, categories, phrases) for label in result["text_labels"]],
            categories, image.width, image.height,
            box_threshold, self.config.grounding.nms_iou,
            self.config.grounding.max_detections,
        )
        return found
