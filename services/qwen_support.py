"""Lazy, local-only loading for optional Qwen Diffusers pipelines."""
from importlib import import_module
from pathlib import Path
from typing import Any

from app.models import QwenConfig
from services.model_support import ModelUnavailableError, torch_runtime


def local_pipeline_path(settings: QwenConfig, name: str) -> Path:
    """Reject missing/incomplete local directories before importing model packages."""
    if settings.model_path is None:
        raise ModelUnavailableError(f"{name}: model_path is empty. Configure a local Diffusers directory; automatic downloads are disabled.")
    path = settings.model_path.expanduser().resolve()
    if not path.is_dir() or not (path / "model_index.json").is_file():
        raise ModelUnavailableError(f"{name}: expected a local Diffusers directory containing model_index.json: {path}")
    return path


def load_local_pipeline(name: str, settings: QwenConfig, device: str) -> tuple[Any, Any]:
    """Load no remote repository; missing components fail instead of downloading."""
    path = local_pipeline_path(settings, name)
    try:
        pipeline_type = getattr(import_module("diffusers"), name)
        torch, selected = torch_runtime(device)
        dtype = settings.dtype
        if dtype == "auto":
            dtype = "bfloat16" if selected == "cuda" else "float32"
        pipeline = pipeline_type.from_pretrained(
            str(path), torch_dtype=getattr(torch, dtype), local_files_only=True,
        )
        # Diffusers exposes these only when the model supports them. VAE tiling
        # reduces decode peaks without changing scene-space placement.
        for method in ('enable_vae_tiling', 'enable_vae_slicing'):
            if hasattr(pipeline, method):
                getattr(pipeline, method)()
        pipeline = pipeline.to(selected)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as error:
        raise ModelUnavailableError(
            f"{name} local load failed: {error}. Install requirements-qwen.txt and provide the complete local pipeline; no weights are downloaded."
        ) from error
    return pipeline, torch


def inference_image(service, image):
    """Bound OOM retry resolution; caller restores source-space size after inference."""
    from PIL import Image
    scale = getattr(service, '_inference_scale', 1.0)
    if scale >= 1:
        return image
    width, height = (max(16, round(v*scale/16)*16) for v in image.size)
    return image.resize((width, height), Image.Resampling.LANCZOS)
