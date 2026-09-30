"""LangChain structured vision review and an explicit deterministic fallback."""
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from app.models import SceneReviewerConfig
from app.paths import read_rgba
from schemas.scene_qa import SceneReviewDecision
from services.model_support import ModelUnavailableError


class SceneReviewService:
    """Initialize once at composition; LLM failure never masquerades as success."""

    def __init__(self, backend: str = "rules", config: SceneReviewerConfig | None = None, llm: Any = None):
        self.backend = backend
        self.config = config or SceneReviewerConfig()
        self._structured = llm.with_structured_output(SceneReviewDecision) if llm is not None else None
        self._prompt = (Path(__file__).resolve().parents[1] / "prompts" / "scene_qa.md").read_text(encoding="utf-8")

    def validate_ready(self) -> None:
        """Fail before expensive vision inference if explicit LLM settings are absent."""
        if self.backend != "llm" or self._structured is not None:
            return
        model = self.config.model or os.getenv("VLM_MODEL", "")
        key = os.getenv(self.config.api_key_env, "")
        if not model or not key:
            raise ModelUnavailableError("LLM scene QA requires VLM_MODEL (or scene_reviewer.model) and the configured API key environment variable")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise ModelUnavailableError("Install requirements-llm.txt for LLM scene QA") from error
        client = ChatOpenAI(
            model=model, api_key=key, base_url=self.config.base_url or os.getenv("VLM_BASE_URL") or None,
            temperature=0, timeout=self.config.timeout, max_retries=self.config.max_retries,
        )
        self._structured = client.with_structured_output(SceneReviewDecision, method="function_calling")

    def _image(self, path: str) -> dict:
        image = read_rgba(path).convert("RGB")
        image.thumbnail((self.config.image_long_edge, self.config.image_long_edge))
        buffer = BytesIO()
        image.save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

    def review(self, source: str, remaining: str, metrics: dict) -> SceneReviewDecision:
        """Only propose continue/stop and categories; graph enforces all budgets."""
        if self.backend == "rules":
            more = metrics["accepted_coverage"] < metrics["target_coverage"]
            return SceneReviewDecision(
                continue_detection=more,
                reason="Rules only: accepted coverage below target" if more else "Rules only: pixel coverage target reached; semantic completeness unverified",
                status="needs_detection" if more else "manual_review",
            )
        self.validate_ready()
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=self._prompt), HumanMessage(content=[
            {"type": "text", "text": json.dumps(metrics, ensure_ascii=False)},
            self._image(source), self._image(remaining),
        ])]
        return SceneReviewDecision.model_validate(self._structured.invoke(messages))
