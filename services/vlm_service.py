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


class VLMService:
    """A provider client is injected once; the graph never initializes models."""

    def __init__(self, mock: bool = True, llm: BaseChatModel | None = None):
        self.mock = mock
        self._structured = llm.with_structured_output(SceneAnalysis) if llm is not None else None
        self._prompt = (Path(__file__).resolve().parents[1] / "prompts" / "scene_analysis.md").read_text(encoding="utf-8")

    def analyze_scene(self, image_path: str) -> SceneAnalysis:
        """Analyze scene semantics only, without calling workflow tools."""
        if self.mock:
            return SceneAnalysis(
                projection="isometric",
                categories=["tree", "rock", "building", "mountain"],
                description="Mock scene analysis: categories are fixed, not inferred from pixels.",
            )
        if self._structured is None:
            raise NotImplementedError("Inject a Qwen-VL-compatible LangChain chat model into VLMService.")
        from langchain_core.messages import HumanMessage, SystemMessage

        buffer = BytesIO()
        read_rgba(image_path).save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages = [
            SystemMessage(content=self._prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this game terrain scene."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]),
        ]
        return SceneAnalysis.model_validate(self._structured.invoke(messages))
