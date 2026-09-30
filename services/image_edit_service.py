"""Prompt-based image editing with a separate visible hint and reconstructed alpha."""
from pathlib import Path

from PIL import Image

from app.models import ModelConfig
from app.paths import read_rgba
from services.execution import operation_span
from services.qwen_support import load_local_pipeline, local_pipeline_path, inference_image
import hashlib


class ImageEditService:
    """Generate review candidates; QwenImageEditPipeline is not a native inpaint API."""

    def __init__(self, mock: bool = True, config: ModelConfig | None = None):
        self.mock = mock
        self.config = config or ModelConfig()
        self._pipeline = None
        self._torch = None

    def validate_ready(self) -> None:
        """Validate the explicitly configured local model, without model imports."""
        if not self.mock:
            local_pipeline_path(self.config.qwen_image_edit, "QwenImageEditPipeline")

    def load(self) -> None:
        """Load and cache the optional local pipeline only when inference is requested."""
        if not self.mock and self._pipeline is None:
            self._pipeline, self._torch = load_local_pipeline(
                "QwenImageEditPipeline", self.config.qwen_image_edit, self.config.device,
            )

    def analyze_occlusion(self, objects: list[dict]) -> list[dict]:
        """Return conservative, serializable occlusion evidence for each visible object."""
        from candidates.reconstruction import analyze
        state = {'objects': objects, 'source_path': objects[0].get('source_path', '') if objects else ''}
        results = []
        for obj in objects:
            if not obj.get('asset_path'):
                continue
            result = analyze(obj, state, None)
            results.append(result.model_dump(mode='json'))
        return results

    def complete_object(
        self, image_path: str, mask_path: str, *, prompt: str, output_path: str | Path,
    ) -> str:
        """Return model RGB on the supplied canvas for independent segmentation.

        QwenImageEditPipeline has no native mask argument. The permission mask is
        used only for RGB compositing. No source alpha is applied to a real output.
        """
        if not prompt.strip():
            raise ValueError("An explicit nonempty edit prompt is required")
        source = read_rgba(image_path)
        with Image.open(Path(mask_path)) as image:
            mask = image.convert("L")
        if mask.size != source.size:
            raise ValueError("Edit mask size must match the edit canvas, not the scene or HD texture")
        if mask.getbbox() is None:
            raise ValueError("Edit mask is empty")
        target = Path(output_path).expanduser().resolve()
        if target in (Path(image_path).resolve(), Path(mask_path).resolve()):
            raise ValueError("Edit output must not overwrite its source image or mask")
        if self.mock:
            edited = source.copy()
        else:
            self.load()
            settings = self.config.qwen_image_edit
            seed = (settings.seed + int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)) % (2**32)
            generator = self._torch.Generator(device="cpu").manual_seed(seed)
            with self._torch.inference_mode():
                result = self._pipeline(
                    image=inference_image(self, source), prompt=prompt.strip(),
                    negative_prompt=settings.negative_prompt, true_cfg_scale=settings.true_cfg_scale,
                    num_inference_steps=settings.num_inference_steps,
                    generator=generator, num_images_per_prompt=1, output_type="pil",
                )
            if not isinstance(result.images, (list, tuple)) or len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
                raise ValueError("Qwen-Image-Edit must return exactly one PIL image")
            with operation_span("image_edit.composite", model="image_edit"):
                candidate = result.images[0].convert("RGB")
                if candidate.size != source.size:
                    candidate = candidate.resize(source.size, Image.Resampling.LANCZOS)
                edited = Image.composite(candidate, source.convert("RGB"), mask)
        with operation_span("image_edit.save", model="image_edit"):
            target.parent.mkdir(parents=True, exist_ok=True)
            edited.save(target, format="PNG")
        return str(target)

    def generate_object(self, image_path: str, *, prompt: str, output_path: str | Path) -> str:
        """Generate a complete silhouette; RGB results must be segmented before QA."""
        if not prompt.strip():
            raise ValueError('Generation requires a nonempty constrained prompt')
        source = read_rgba(image_path)
        target = Path(output_path).resolve()
        if target == Path(image_path).resolve():
            raise ValueError('Generation must not overwrite source')
        if self.mock:
            generated = source
        else:
            self.load()
            settings = self.config.qwen_image_edit
            with self._torch.inference_mode():
                result = self._pipeline(image=inference_image(self, source), prompt=prompt,
                    negative_prompt=settings.negative_prompt, true_cfg_scale=settings.true_cfg_scale,
                    num_inference_steps=settings.num_inference_steps,
                    generator=self._torch.Generator(device='cpu').manual_seed(
                        (settings.seed + int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)) % (2**32)),
                    num_images_per_prompt=1, output_type='pil')
            if not isinstance(result.images, (list, tuple)) or len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
                raise ValueError('Qwen must return exactly one PIL image')
            generated = result.images[0]
        with operation_span('image_edit.save', model='image_edit'):
            target.parent.mkdir(parents=True, exist_ok=True)
            generated.save(target, 'PNG')
        return str(target)
