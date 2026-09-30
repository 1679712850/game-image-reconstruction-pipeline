"""Serializable candidate decisions; unknown semantic scores are never invented."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CandidateQA(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['ACCEPT', 'RETRY', 'REJECT']
    scores: dict[str, float] = Field(default_factory=dict)
    overall: float = Field(default=0, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    evaluator: str = 'rules'


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidate_id: str
    type: Literal['segmentation', 'inpaint', 'generation', 'crop_mask']
    image_path: str | None = None
    raw_path: str | None = None
    mask_path: str | None = None
    prompt: str = ''
    attempt: int = 0
    qa: CandidateQA
    placement: dict = Field(default_factory=dict)
    error: str | None = None


class CandidateRegistry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    instance_id: str
    source: dict
    candidates: list[Candidate] = Field(default_factory=list)
    accepted_candidate_id: str | None = None
    accepted_asset: str | None = None
    needs_manual_review: bool = False
    provenance: dict = Field(default_factory=dict)
