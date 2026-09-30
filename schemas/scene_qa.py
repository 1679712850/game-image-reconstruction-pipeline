"""Constrained scene reviewer output; the agent cannot invoke arbitrary tools."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SceneReviewDecision(BaseModel):
    """Suggest further detection and bounded English noun phrases."""

    model_config = ConfigDict(extra="forbid")
    continue_detection: bool
    reason: str = Field(min_length=1, max_length=2000)
    suggested_categories: list[str] = Field(default_factory=list, max_length=16)
    status: Literal["needs_detection", "sufficient", "manual_review"]

    @field_validator("suggested_categories")
    @classmethod
    def safe_categories(cls, values: list[str]) -> list[str]:
        """Reject long instructions, punctuation and non-English detector prompts."""
        import re
        if any(not re.fullmatch(r"[a-zA-Z][a-zA-Z _-]{0,63}", v) for v in values):
            raise ValueError("Suggested categories must be short English noun phrases")
        return list(dict.fromkeys(v.strip().lower() for v in values))
