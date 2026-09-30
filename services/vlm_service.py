"""VLM boundary with mock responses and optional LangChain structured output."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from app.paths import read_rgba
from schemas.scene import SceneAnalysis
from services.model_support import ModelUnavailableError
from services.qwen_vl_backend import QwenVLBackend


class VLMService:
    """A provider client is injected once; the graph never initializes models."""

    def __init__(self, mock: bool = True, llm: BaseChatModel | None = None, *, local_backend: QwenVLBackend | None = None):
        self.mock = mock
        self._local_backend = local_backend if not mock and llm is None else None
        self.config = self._local_backend.config if self._local_backend is not None else None
        client = llm if llm is not None else self._local_backend
        self._structured = client.with_structured_output(SceneAnalysis) if client is not None else None
        self._prompt = (Path(__file__).resolve().parents[1] / "prompts" / "scene_analysis.md").read_text(encoding="utf-8")

    def validate_ready(self) -> None:
        """Preflight configuration without loading weights or changing mock behavior."""
        if self.mock:
            return
        if self._local_backend is not None:
            self._local_backend.validate_ready()
        elif self._structured is None:
            raise ModelUnavailableError("Configure local qwen_vl.model_path or inject a vision chat model")

    def analyze_scene(self, image_path: str) -> SceneAnalysis:
        """Analyze scene semantics only, without calling workflow tools."""
        if self.mock:
            return SceneAnalysis(
                projection="isometric",
                categories=["tree", "rock", "building", "mountain"],
                description="Mock scene analysis: categories are fixed, not inferred from pixels.",
            )
        self.validate_ready()
        from langchain_core.messages import HumanMessage, SystemMessage

        buffer = BytesIO()
        image = read_rgba(image_path)
        if self._local_backend is not None:
            edge = self._local_backend.config.qwen_vl.image_long_edge
            image.thumbnail((edge, edge))
        image.save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages = [
            SystemMessage(content=self._prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this game terrain scene."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]),
        ]
        return SceneAnalysis.model_validate(self._structured.invoke(messages))
