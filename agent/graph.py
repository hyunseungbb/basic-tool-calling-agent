"""LangGraph StateGraph 정의 및 컴파일.

정의서 §3 실행 플로우 및 AI_AGENT_ARCHITECTURE.md 플로우차트에 따라
Tool Agent Loop를 LangGraph 그래프로 구성한다.

플로우 (스트리밍 모드):
  normalize_goal → policy_node → (route_action) →
    CALL_TOOL → tool_executor → (check_budget) → policy_node 또는 force_synthesizer (placeholder)
    WRITE_NOTE → write_note → (check_budget) → policy_node 또는 force_synthesizer (placeholder)
    SYNTHESIZE → synthesizer (placeholder) → END
    STOP → stop_node → END
  
  합성 단계는 그래프 외부에서 stream_synthesis()로 스트리밍 처리
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from agent.config import settings
from agent.models import ActionType, AgentStatus, GraphState
from agent.nodes.executor import stop_node, tool_executor_node, write_note_node
from agent.nodes.policy import policy_node

logger = logging.getLogger(__name__)


# ──────────────────────────── 노드 함수들 ────────────────────────────


def normalize_goal(state: GraphState) -> dict[str, Any]:
    """목표를 정규화하고 초기 상태를 설정한다.

    정의서 §8.1 의사코드 1번:
    - 입력 메시지로 goal 정규화 (MVP: user_input 그대로 사용)
    - state 생성 및 저장
    """
    run_id = state.get("run_id") or uuid.uuid4().hex[:12]
    goal = state.get("goal", "")
    max_steps = state.get("max_steps") or settings.max_steps

    logger.info("에이전트 시작: run_id=%s, goal='%s'", run_id, goal[:50])

    return {
        "run_id": run_id,
        "goal": goal,
        "cursor": 0,
        "max_steps": max_steps,
        "status": AgentStatus.RUNNING.value,
        "observations": [],
        "final_answer": "",
        "current_action": {},
    }


# ──────────────────────────── 라우팅 함수들 ────────────────────────────


def route_action(state: GraphState) -> str:
    """Policy의 Action 유형에 따라 다음 노드를 결정한다.

    정의서 §8.1 의사코드 Action 실행 분기:
    - CALL_TOOL → tool_executor
    - WRITE_NOTE → write_note
    - SYNTHESIZE → synthesizer
    - STOP → stop
    """
    action = state.get("current_action", {})
    action_type = action.get("type", "STOP")

    logger.debug("Action 라우팅: %s", action_type)

    if action_type == ActionType.CALL_TOOL.value:
        return "tool_executor"
    elif action_type == ActionType.WRITE_NOTE.value:
        return "write_note"
    elif action_type == ActionType.SYNTHESIZE.value:
        return "synthesizer"
    else:  # STOP 또는 알 수 없는 경우
        return "stop"


def check_budget(state: GraphState) -> str:
    """예산(budget) 체크로 루프 계속 여부를 결정한다.

    정의서 §3.1 Continue loop 판단:
    - cursor >= max_steps 이면 종료 후 Synthesizer 1회 호출로 마무리
    - status == DONE 이면 종료
    """
    cursor = state.get("cursor", 0)
    max_steps = state.get("max_steps", 5)
    status = state.get("status", "RUNNING")

    if status == AgentStatus.DONE.value:
        logger.info("상태 DONE - 루프 종료")
        return "force_synthesizer"

    if cursor >= max_steps:
        logger.info("예산 초과 (%d/%d) - 강제 종료", cursor, max_steps)
        return "force_synthesizer"

    logger.debug("예산 확인 OK (%d/%d) - 계속", cursor, max_steps)
    return "policy"


# ──────────────────────── 스트리밍 전용 그래프 ────────────────────────


def _synthesis_placeholder(state: GraphState) -> dict[str, Any]:
    """스트리밍 전용 placeholder 노드.

    LLM 호출 없이 status=DONE만 설정한다.
    실제 합성은 그래프 외부에서 stream_synthesis()를 통해 스트리밍된다.
    """
    logger.info("Synthesis placeholder - 스트리밍 모드 (LLM 호출 생략)")
    return {"status": AgentStatus.DONE.value}


def build_streaming_graph():
    """스트리밍 전용 LangGraph StateGraph를 구성하고 컴파일한다.

    synthesizer와 force_synthesizer를 placeholder로 교체하여,
    그래프는 observations 수집까지만 수행하고
    합성 단계는 외부에서 스트리밍으로 처리할 수 있게 한다.

    Returns:
        컴파일된 LangGraph 그래프
    """
    graph = StateGraph(GraphState)

    # 노드 등록 (synthesizer를 placeholder로 교체)
    graph.add_node("normalize_goal", normalize_goal)
    graph.add_node("policy", policy_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("write_note", write_note_node)
    graph.add_node("synthesizer", _synthesis_placeholder)
    graph.add_node("stop", stop_node)
    graph.add_node("force_synthesizer", _synthesis_placeholder)

    # 엣지 연결 (기존 그래프와 동일한 구조)
    graph.set_entry_point("normalize_goal")
    graph.add_edge("normalize_goal", "policy")

    graph.add_conditional_edges(
        "policy",
        route_action,
        {
            "tool_executor": "tool_executor",
            "write_note": "write_note",
            "synthesizer": "synthesizer",
            "stop": "stop",
        },
    )

    graph.add_conditional_edges(
        "tool_executor",
        check_budget,
        {
            "policy": "policy",
            "force_synthesizer": "force_synthesizer",
        },
    )

    graph.add_conditional_edges(
        "write_note",
        check_budget,
        {
            "policy": "policy",
            "force_synthesizer": "force_synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)
    graph.add_edge("stop", END)
    graph.add_edge("force_synthesizer", END)

    compiled = graph.compile()
    logger.info("스트리밍 전용 LangGraph 그래프 컴파일 완료")
    return compiled


# ──────────────────────────── 싱글턴 인스턴스 ────────────────────────────

_compiled_streaming_graph = None


def get_streaming_graph():
    """스트리밍 전용 그래프 싱글턴을 반환한다."""
    global _compiled_streaming_graph
    if _compiled_streaming_graph is None:
        _compiled_streaming_graph = build_streaming_graph()
    return _compiled_streaming_graph

