"""接口的请求 / 响应模型。

校验放在这里而不是靠前端：接口是公开的，任何人都能直接 POST。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION = 500
MAX_HISTORY_TURNS = 10
MAX_TURN_CHARS = 4000


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=MAX_TURN_CHARS)


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        # min_length 拦不住 "   " 这种全是空白的输入
        if not stripped:
            raise ValueError("问题不能为空")
        return stripped
