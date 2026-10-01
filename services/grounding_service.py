"""Transformers Grounding DINO inference and dimension-relative mock mode."""
from collections.abc import Mapping
from contextlib import contextmanager
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
        self._window_image = None
        self._prepared_image = None

    def clear_prepared_inputs(self):
        """Drop device tensors before model offload/unload or a new crop."""
        self._prepared_image = None

    @contextmanager
    def prepared_window(self, image):
        self.clear_prepared_inputs()
        self._window_image = image
        try:
            yield
        finally:
            self._window_image = None
            self.clear_prepared_inputs()

    def _prepare_inputs(self, image, prompt):
        scale = getattr(self, '_inference_scale', 1.0)
        stamp = (id(self._processor), self._device, scale)
        reusable = self.config.grounding.reuse_image_inputs and self._window_image is image
        if reusable and self._prepared_image is not None and self._prepared_image[0] == stamp:
            # Use the processor's text-only API, preserving its tokenizer defaults
            # and label formatting (rather than calling the tokenizer directly).
            text = self._processor(text=prompt, return_tensors='pt').to(self._device)
            return {**text, **self._prepared_image[1]}
        size_options = {'size': {'shortest_edge': max(128, round(800*scale)),
                                 'longest_edge': max(128, round(1333*scale))}} if scale < 1 else {}
        inputs = self._processor(images=image, text=prompt, return_tensors='pt', **size_options).to(self._device)
        if reusable and isinstance(inputs, Mapping) and 'pixel_values' in inputs:
            self._prepared_image = (stamp, {key: inputs[key] for key in ('pixel_values', 'pixel_mask') if key in inputs})
        return inputs

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

    def detect_bounds(self, size, categories, category_limit):
        """Worst-case calls and returned objects for the legacy local detector.

        Only this explicit contract permits ordered lookahead in region recovery.
        """
        if not categories:
            return 0, 0
        if self.mock:
            return 0, len({'tree', 'rock', 'building', 'mountain'}.intersection(categories))
        cfg = self.config.grounding
        width, height = size
        full = (0, 0, width, height)
        windows = tile_windows(width, height, cfg.tile_size, cfg.tile_overlap) if cfg.tiled else [full]
        if cfg.include_full_image:
            windows = list(dict.fromkeys([full, *windows]))
        group_size = min(cfg.prompt_group_size, category_limit)
        return len(windows)*((len(categories)+group_size-1)//group_size), cfg.max_detections

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
        from services.execution import _active
        runtime = _active.get()
        group_size = min(cfg.prompt_group_size,runtime.config.detection.budget.max_categories_per_pass) if runtime else cfg.prompt_group_size
        groups = [categories[i:i+group_size] for i in range(0,len(categories),group_size)]
        for window in windows:
            crop = image.crop(window)
            with self.prepared_window(crop):
                for group in groups:
                    found = self._infer_image(crop, group, box_threshold, text_threshold)
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

    def detect_p0(self, image_path, categories, detection_config, round_index=1):
        """Run the production high-recall path with explicit pipeline configuration."""
        from detection.p0_pipeline import P0DetectionPipeline
        from taxonomy.aliases import normalize_category
        image = read_rgba(image_path).convert("RGB")
        cfg = self.config.grounding
        relaxed = round_index >= cfg.relax_from_round
        text_threshold = min(cfg.text_threshold, cfg.relaxed_text_threshold) if relaxed else cfg.text_threshold
        # Otherwise low-score boxes survive the box gate but get empty text labels.
        text_threshold = min(text_threshold, detection_config.confidence.candidate_floor)
        if self.mock:
            # Stable source-space fixtures: overlapping crops observe the SAME objects.
            fixtures = self.detect(image_path, categories)
            def infer(crop, group, context):
                wx, wy, wr, wb = context["window"]
                found = []
                for obj in fixtures:
                    category = normalize_category(obj["category"], default=obj["category"])
                    if category not in group:
                        continue
                    b = obj["bbox"]
                    x, y = max(wx, b["x"]), max(wy, b["y"])
                    r, bottom = min(wr, b["x"]+b["w"]), min(wb, b["y"]+b["h"])
                    if r > x and bottom > y:
                        found.append({"category": category, "confidence": obj["confidence"],
                                      "bbox": {"x": x-wx, "y": y-wy, "w": r-x, "h": bottom-y}})
                return found
        else:
            self.load()  # A broken model setup must fail explicitly before per-tile isolation.
            def infer(crop, group, context):
                # Grounding DINO consumes noun phrases, not VLM instructions. The shared
                # context retains the full instruction for instruction-following adapters.
                return self._infer_image(crop, group, detection_config.confidence.candidate_floor,
                                         text_threshold, preserve_candidates=True)
            infer.prepared_window = self.prepared_window
            infer.batch_size = cfg.batch_size if self._device == 'cuda' else 1
            infer.infer_many = lambda crop, groups: self._infer_batch(
                crop, groups, detection_config.confidence.candidate_floor, text_threshold)
        records, stats = P0DetectionPipeline(detection_config).run(
            image, categories, infer, method="mock" if self.mock else cfg.model_id, round_index=round_index)
        return records, stats

    def _infer_image(self, image: Image.Image, categories: list[str], box_threshold: float, text_threshold: float, *, preserve_candidates: bool = False) -> list[dict]:
        """Infer a single window, mapping explicit prompt phrases back to canonical categories."""
        phrases = [self.config.grounding.prompts.get(c, c.replace('_', ' ')) for c in categories]
        prompt = ". ".join(phrases) + "."
        with self._torch.inference_mode():
            inputs = self._prepare_inputs(image, prompt)
            outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
        return self._decode_result(result, image.size, categories, phrases, box_threshold, preserve_candidates)

    def _decode_result(self, result, size, categories, phrases, box_threshold, preserve_candidates):
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        # Transformers 5.17 tokenizer batch_decode([]) returns ['']; no boxes is still empty.
        if not boxes and not scores:
            return []
        if preserve_candidates:
            from taxonomy.aliases import normalize_category
            labels = result["text_labels"]
            if not len(boxes) == len(scores) == len(labels):
                raise ValueError("Detector returned mismatched boxes/scores/labels")
            return [{"category": normalize_category(label) or canonical_prompt_label(label, categories, phrases) or label,
                     "raw_label": label, "confidence": score, "bbox": box, "bbox_format": "xyxy", "coordinate_space": "pixel"}
                    for box, score, label in zip(boxes, scores, labels)]
        found = postprocess_detections(
            boxes, scores,
            [canonical_prompt_label(label, categories, phrases) for label in result["text_labels"]],
            categories, *size,
            box_threshold, self.config.grounding.nms_iou,
            self.config.grounding.max_detections,
        )
        return found

    def _infer_batch(self, image, groups, box_threshold, text_threshold):
        """Batch only identical image AND token shapes; no extra padding or AMP.

        OOM splits happen here before the runtime can degrade resolution/device.
        The scanner owns logical-call reservations, even if a batch is split.
        """
        from collections import defaultdict
        from services.model_manager import is_oom
        torch = self._torch
        prepared, phrases_by_group, buckets = [], [], defaultdict(list)
        with torch.inference_mode():
            for index, group in enumerate(groups):
                phrases = [self.config.grounding.prompts.get(c, c.replace('_', ' ')) for c in group]
                inputs = self._prepare_inputs(image, '. '.join(phrases) + '.')
                signature = tuple((k, tuple(v.shape), v.dtype, v.device) for k, v in sorted(inputs.items()))
                prepared.append(inputs)
                phrases_by_group.append(phrases)
                buckets[signature].append(index)
            results = [None] * len(groups)

            def forward(indices):
                # This frame must unwind before OOM recovery to release activations.
                inputs = {key: torch.cat([prepared[i][key] for i in indices], dim=0)
                          for key in prepared[indices[0]]}
                outputs = self._model(**inputs)
                return self._processor.post_process_grounded_object_detection(
                    outputs, input_ids=inputs['input_ids'], threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[(image.height, image.width)]*len(indices))

            def run(indices):
                retry = False
                try:
                    outputs = forward(indices)
                except Exception as error:
                    if not is_oom(error) or len(indices) == 1:
                        raise
                    retry = True
                if retry:
                    if self._device == 'cuda':
                        torch.cuda.empty_cache()
                    midpoint = len(indices)//2
                    run(indices[:midpoint])
                    run(indices[midpoint:])
                    return
                if len(outputs) != len(indices):
                    raise ValueError('Detector returned incorrect batch length')
                for index, result in zip(indices, outputs):
                    results[index] = self._decode_result(result, image.size, groups[index],
                        phrases_by_group[index], box_threshold, True)

            for indices in buckets.values():
                run(indices)
        return results
