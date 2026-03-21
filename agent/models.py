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

    SQL_QUERY = "sql_query"
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
    """LangGraph StateGraph에서 사용하는 상태 딕셔너리."""

    run_id: str
    goal: str
    cursor: int
    max_steps: int
    status: str
    observations: list[dict[str, Any]]
    final_answer: str
    current_action: dict[str, Any]


# ──────────────────────────── Tool 입출력 스키마 ────────────────────────────


class SqlQueryInput(BaseModel):
    """sql_query Tool 입력."""

    sql: str
    params: list[Any] = Field(default_factory=list)


class SqlQueryOutput(BaseModel):
    """sql_query Tool 출력."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


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
