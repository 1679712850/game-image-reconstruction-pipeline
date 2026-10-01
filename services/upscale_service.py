"""Independent neural detail restoration; completed alpha is the shape authority."""
from pathlib import Path
import numpy as np
from PIL import Image
from app.models import ModelConfig
from app.paths import read_rgba
from services.model_support import ModelUnavailableError, torch_runtime

HD_PROMPT = '''Restore this isolated 2D game asset at high resolution. Preserve its exact shape,
silhouette, proportions, perspective, isometric camera angle, colors, materials, lighting and art style.
Recover fine line work, texture detail, clean edges and material definition. Remove compression artifacts,
blurry edges, pixel noise, generation artifacts and inconsistent textures. Do not redesign or reinterpret
the object. Do not introduce components or change its silhouette. Preserve hand-painted 2D game art,
existing line work and stylized shading. Do not convert to photorealism or a 3D-rendered style.'''


def resolve_upscale_factor(width: int, height: int) -> int:
    return 4 if max(width, height) < 128 else 2


# Public compatibility alias; one implementation defines all factor boundaries.
choose_scale = resolve_upscale_factor


class UpscaleService:
    """RealESRGAN_x4plus local weights; interpolation is explicitly a preview backend."""
    def __init__(self, mock=True, backend=None, config=None):
        self.mock = mock
        self.config = config or ModelConfig()
        self.backend = backend or ('lanczos' if mock else self.config.upscale.backend)
        self._model = None
        self._torch = None
        self._device = None

    @staticmethod
    def _target(source: Path, scale: int) -> Path:
        root = source.parent.parent if source.parent.name in {'assets', 'effects'} else source.parent
        return root/('effects_hd' if source.parent.name == 'effects' else 'assets_hd')/f'{source.stem}@{scale:g}x.png'

    def available(self) -> bool:
        return self.backend == 'lanczos' or bool(self.config.upscale.checkpoint and self.config.upscale.checkpoint.is_file())

    def load(self):
        if self.backend == 'lanczos' or self._model is not None:
            return
        path = self.config.upscale.checkpoint
        if path is None or not path.is_file():
            raise ModelUnavailableError('Real-ESRGAN requires local upscale.checkpoint (RealESRGAN_x4plus.pth)')
        try:
            from services.rrdbnet import rrdbnet
            torch, device = torch_runtime(self.config.device)
            model = rrdbnet()
            state = torch.load(str(path), map_location='cpu', weights_only=True)
            model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
            self._model = model.eval().to(device)
            if device == 'cuda':
                # NHWC enables optimized CUDA convolution kernels without changing
                # model precision or weights.
                self._model = self._model.to(memory_format=torch.channels_last)
            self._torch, self._device = torch, device
        except (ImportError, RuntimeError, OSError, ValueError) as error:
            raise ModelUnavailableError(f'Real-ESRGAN load failed: {error}; install requirements-upscale.txt') from error

    def _restore(self, image):
        self.load()
        torch = self._torch
        rgb = np.array(image.convert('RGB'), dtype=np.float32)/255
        height, width = rgb.shape[:2]
        tile = self.config.upscale.tile or max(width, height)
        pad = self.config.upscale.tile_pad
        output = np.zeros((height*4, width*4, 3), np.uint8)
        alpha = np.asarray(image.getchannel('A')) if image.mode == 'RGBA' else None
        tiles_run = tiles_skipped = 0
        with torch.inference_mode():
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    right, bottom = min(width, x+tile), min(height, y+tile)
                    # Only omit output that cannot affect visible pixels. Keep an
                    # 8-input-pixel guard for alpha upsampling and the final RGB
                    # Lanczos 4x -> 2x resample. Visible tiles retain full context.
                    if (self.config.upscale.skip_transparent_tiles and alpha is not None
                            and not alpha[max(0,y-8):min(height,bottom+8),
                                          max(0,x-8):min(width,right+8)].any()):
                        tiles_skipped += 1
                        continue
                    tiles_run += 1
                    left, top = max(0, x-pad), max(0, y-pad)
                    r, b = min(width, right+pad), min(height, bottom+pad)
                    tensor = torch.from_numpy(rgb[top:b, left:r].transpose(2, 0, 1).copy()).unsqueeze(0)
                    if self._device == 'cuda':
                        tensor = tensor.contiguous(memory_format=torch.channels_last)
                    tensor = tensor.to(self._device, non_blocking=self._device == 'cuda')
                    restored = self._model(tensor).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                    if restored.shape != ((b-top)*4, (r-left)*4, 3):
                        raise ValueError('Real-ESRGAN returned incorrect dimensions')
                    output[y*4:bottom*4, x*4:right*4] = np.rint(restored[(y-top)*4:(bottom-top)*4,
                        (x-left)*4:(right-left)*4]*255).astype(np.uint8)
        self.last_restore_stats = {'tiles_run': tiles_run, 'tiles_skipped': tiles_skipped}
        return Image.fromarray(output)

    def upscale(self, image_path, scale):
        if scale not in (2, 4):
            raise ValueError('High resolution restoration supports 2x or 4x')
        source = Path(image_path)
        image = read_rgba(source)
        if image.width*image.height*16 > self.config.upscale.max_output_pixels:
            raise ValueError('Neural 4x intermediate exceeds upscale.max_output_pixels')
        size = (image.width*scale, image.height*scale)
        if self.backend == 'lanczos':
            result = image.resize(size, Image.Resampling.LANCZOS)
        elif self.backend == 'real_esrgan':
            from candidates.alpha import extend_colors
            rgb = self._restore(extend_colors(image))
            result = rgb.resize(size, Image.Resampling.LANCZOS).convert('RGBA')
            # Freeze Stage A's NEW silhouette, never the original visible mask.
            result.putalpha(image.getchannel('A').resize(size, Image.Resampling.LANCZOS))
        else:
            raise ValueError(f'Unknown upscale backend: {self.backend}')
        pixels = np.array(result); pixels[pixels[:, :, 3] == 0] = 0
        target = self._target(source, scale)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target, 'PNG')
        return str(target.resolve())
