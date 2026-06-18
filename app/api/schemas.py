"""Pydantic-схемы запросов API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .. import config


class StartRequest(BaseModel):
    mode: str = Field(default="bilingual")  # bilingual | ru | en
    k: int = Field(default=config.DEFAULT_K, ge=1, le=50)


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str
