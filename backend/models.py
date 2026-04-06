from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TaskStatus = Literal["queued", "running", "succeeded", "failed"]
StepStatus = Literal["queued", "running", "succeeded", "failed"]


class TaskCreateRequest(BaseModel):
    goal: str = Field(..., min_length=5, description="High-level task goal")
    max_steps: int = Field(6, ge=1, le=25)
    enable_code_run: bool = Field(False, description="Allow the code agent to run code (risky)")


class TaskResponse(BaseModel):
    task_id: str
    goal: str
    status: TaskStatus
    created_at: str
    updated_at: str
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[str] = None


class PlanStep(BaseModel):
    step_type: Literal["web_research", "rag_index", "synthesize_report", "code_demo", "finalize"]
    agent: Literal["supervisor", "researcher", "rager", "coder", "analyst", "reflector"] = "analyst"
    details: Dict[str, Any] = Field(default_factory=dict)


class StepUpdate(BaseModel):
    step_id: str
    agent: str
    step_type: str
    status: StepStatus
    output: Optional[str] = None
    error: Optional[str] = None

