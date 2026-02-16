"""Pydantic 데이터 모델 정의: Action, Observation, AgentState 및 LangGraph용 GraphState."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────── Enums ────────────────────────────


class ActionType(str, Enum):
    """Policy가 반환할 수 있는 Action 유형."""

    CALL_TOOL = "CALL_TOOL"
    WRITE_NOTE = "WRITE_NOTE"
    SYNTHESIZE = "SYNTHESIZE"
    STOP = "STOP"


class ToolName(str, Enum):
    """사용 가능한 Tool 이름."""

    VECTOR_SEARCH = "vector_search"
    WEB_SEARCH = "web_search"


class ObservationKind(str, Enum):
    """Observation 유형."""

    TOOL_RESULT = "TOOL_RESULT"
    NOTE = "NOTE"


class AgentStatus(str, Enum):
    """에이전트 실행 상태."""

    RUNNING = "RUNNING"
    DONE = "DONE"


# ──────────────────────────── Pydantic Models ────────────────────────────


class Action(BaseModel):
    """Policy가 반환하는 다음 Action."""

    type: ActionType
    tool_name: Optional[ToolName] = None
    tool_args: Optional[dict[str, Any]] = None
    note: Optional[str] = None


class Observation(BaseModel):
    """Action 실행 결과를 기록하는 append-only 로그 엔트리."""

    seq: int
    step_id: str
    kind: ObservationKind
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    payload: Optional[dict[str, Any]] = None
    summary: str = ""


class Budget(BaseModel):
    """루프 예산."""

    max_steps: int = 5


class AgentState(BaseModel):
    """에이전트 실행의 전체 상태."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    cursor: int = 0
    budget: Budget = Field(default_factory=Budget)
    status: AgentStatus = AgentStatus.RUNNING
    observations: list[Observation] = Field(default_factory=list)
    final_answer: Optional[str] = None


# ──────────────────────────── LangGraph TypedDict State ────────────────────────────


class GraphState(TypedDict, total=False):
    """LangGraph StateGraph에서 사용하는 상태 딕셔너리.

    LangGraph는 TypedDict 기반 상태를 사용하므로,
    Pydantic 모델을 직렬화하여 담는다.
    """

    run_id: str
    goal: str
    cursor: int
    max_steps: int
    status: str  # "RUNNING" | "DONE"
    observations: list[dict[str, Any]]  # Observation.model_dump() 리스트
    final_answer: str
    current_action: dict[str, Any]  # Action.model_dump()


# ──────────────────────────── Tool 입출력 스키마 ────────────────────────────


class VectorSearchInput(BaseModel):
    """vector_search Tool 입력."""

    collection: str = "docs"
    query_text: str
    top_k: int = 5
    filter: Optional[dict[str, Any]] = None


class VectorSearchHit(BaseModel):
    """vector_search 결과 단건."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchOutput(BaseModel):
    """vector_search Tool 출력."""

    hits: list[VectorSearchHit] = Field(default_factory=list)


class WebSearchInput(BaseModel):
    """web_search Tool 입력."""

    query: str
    top_k: int = 5
    region: Optional[str] = None


class WebSearchResult(BaseModel):
    """web_search 결과 단건."""

    title: str
    snippet: str
    url: str


class WebSearchOutput(BaseModel):
    """web_search Tool 출력."""

    results: list[WebSearchResult] = Field(default_factory=list)

