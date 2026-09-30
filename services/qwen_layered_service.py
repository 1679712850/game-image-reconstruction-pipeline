"""Optional Qwen-Image-Layered adapter with local-only model loading."""
from pathlib import Path

from PIL import Image

from app.models import ModelConfig
from app.paths import read_rgba
from schemas.generation import LayerAsset
from services.qwen_support import load_local_pipeline, local_pipeline_path, inference_image


class QwenLayeredService:
    """Export scene-aligned RGBA layers without changing instance detection input."""

    def __init__(self, mock: bool = True, config: ModelConfig | None = None):
        self.mock = mock
        self.config = config or ModelConfig()
        self._pipeline = None
        self._torch = None

    def validate_ready(self) -> None:
        """Fail early on absent local weights, without importing model packages."""
        if not self.mock:
            local_pipeline_path(self.config.qwen_layered, "QwenImageLayeredPipeline")

    def load(self) -> None:
        """Lazily cache one local Diffusers pipeline per service instance."""
        if not self.mock and self._pipeline is None:
            self._pipeline, self._torch = load_local_pipeline(
                "QwenImageLayeredPipeline", self.config.qwen_layered, self.config.device,
            )

    def decompose_layers(self, image_path: str, *, output_dir: str | Path | None = None) -> list[dict]:
        """Persist model RGBA output at source size; mock writes one unchanged layer."""
        source = read_rgba(image_path)
        root = Path(output_dir) if output_dir is not None else Path(image_path).parent / f"{Path(image_path).stem}_layers"
        if self.mock:
            layers = [source]
        else:
            self.load()
            settings = self.config.qwen_layered
            generator = self._torch.Generator(device="cpu").manual_seed(settings.seed)
            with self._torch.inference_mode():
                result = self._pipeline(
                    image=inference_image(self, source), layers=settings.layers,
                    resolution=640 if getattr(self, "_inference_scale", 1) < 1 else settings.resolution,
                    num_inference_steps=settings.num_inference_steps,
                    true_cfg_scale=settings.true_cfg_scale, negative_prompt=settings.negative_prompt,
                    cfg_normalize=settings.cfg_normalize, use_en_prompt=settings.use_en_prompt,
                    generator=generator, num_images_per_prompt=1, output_type="pil",
                )
            # Layered output groups layers by batch; tolerate a flat single batch.
            layers = result.images
            if isinstance(layers, (list, tuple)) and len(layers) == 1 and isinstance(layers[0], (list, tuple)):
                layers = layers[0]
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError("Qwen-Image-Layered returned no layers")
        if any(not isinstance(layer, Image.Image) or layer.mode != "RGBA" for layer in layers):
            raise ValueError("Qwen-Image-Layered must return RGBA PIL layers with model alpha")
        if len({layer.size for layer in layers}) != 1:
            raise ValueError("Qwen-Image-Layered layers must share one canvas size")
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for index, layer in enumerate(layers):
            target = root / f"layer_{index:03d}.png"
            aligned = layer if layer.size == source.size else layer.resize(source.size, Image.Resampling.LANCZOS)
            aligned.save(target, format="PNG")
            records.append(LayerAsset(
                id=f"layer_{index:03d}", asset_path=str(target.resolve()), order=index,
                canvas_size=source.size, model_size=layer.size, mock=self.mock,
                status="mock_passthrough" if self.mock else "manual_review",
            ).model_dump(mode="json"))
        return records
