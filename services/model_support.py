"""Optional dependency and device handling without imports in mock mode."""
from typing import Any


class ModelUnavailableError(RuntimeError):
    """Actionable model dependency, loading or inference error."""


def torch_runtime(device: str) -> tuple[Any, str]:
    """Resolve device explicitly; auto prefers CUDA then the portable CPU path."""
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailableError("Install optional dependencies: pip install -r requirements-vision.txt") from error
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailableError("CUDA requested but unavailable; select device: cpu")
    if selected == "mps" and not torch.backends.mps.is_available():
        raise ModelUnavailableError("MPS requested but unavailable; select device: cpu")
    return torch, selected
