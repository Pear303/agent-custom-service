"""工单相关请求模型"""
from __future__ import annotations

from pydantic import BaseModel


class RequirementRequest(BaseModel):
    user_id: str
    project_name: str
    project_type: str = ""
    description: str
    deadline: str = ""
    budget: str = ""
