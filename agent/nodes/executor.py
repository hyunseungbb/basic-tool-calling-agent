"""Executor 노드: Action을 실행하고 Observation을 누적한다.

CALL_TOOL, WRITE_NOTE, STOP 세 가지 Action 유형을 처리한다.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.models import GraphState, ObservationKind
from agent.tools.tool_runtime import execute_tool

logger = logging.getLogger(__name__)


def _make_observation(
    state: GraphState,
    kind: ObservationKind,
    summary: str,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Observation 딕셔너리를 생성한다."""
    cursor = state.get("cursor", 0)
    observations = state.get("observations", [])
    seq = len(observations)

    return {
        "seq": seq,
        "step_id": f"turn_{cursor}",
        "kind": kind.value,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "payload": payload,
        "summary": summary,
    }


def tool_executor_node(state: GraphState) -> dict[str, Any]:
    """CALL_TOOL Action을 처리하는 노드."""
    action = state.get("current_action", {})
    tool_name = action.get("tool_name", "")
    tool_args = action.get("tool_args", {})

    logger.info("Executor: Tool 실행 - %s", tool_name)

    try:
        result = execute_tool(tool_name, tool_args)
        summary = _summarize_tool_result(tool_name, result)
        observation = _make_observation(
            state,
            kind=ObservationKind.TOOL_RESULT,
            summary=summary,
            tool_name=tool_name,
            tool_args=tool_args,
            payload=result,
        )
    except Exception as e:
        logger.error("Tool 실행 오류: %s", str(e))
        observation = _make_observation(
            state,
            kind=ObservationKind.TOOL_RESULT,
            summary=f"Tool '{tool_name}' 실행 오류: {str(e)}",
            tool_name=tool_name,
            tool_args=tool_args,
        )

    observations = list(state.get("observations", []))
    observations.append(observation)
    cursor = state.get("cursor", 0) + 1

    return {"observations": observations, "cursor": cursor}


def write_note_node(state: GraphState) -> dict[str, Any]:
    """WRITE_NOTE Action을 처리하는 노드."""
    action = state.get("current_action", {})
    note = action.get("note", "")

    logger.info("Executor: 노트 작성 - %s", note[:50])

    observation = _make_observation(
        state,
        kind=ObservationKind.NOTE,
        summary=note,
    )

    observations = list(state.get("observations", []))
    observations.append(observation)
    cursor = state.get("cursor", 0) + 1

    return {"observations": observations, "cursor": cursor}


def stop_node(state: GraphState) -> dict[str, Any]:
    """STOP Action을 처리하는 노드."""
    action = state.get("current_action", {})
    note = action.get("note", "에이전트가 중단되었습니다.")

    logger.info("Executor: STOP - %s", note[:50])

    return {
        "final_answer": note,
        "status": "DONE",
    }


def _summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Tool 결과를 요약 문자열로 변환한다."""
    if tool_name == "sql_query":
        error = result.get("error")
        if error:
            return f"sql_query 오류: {error}"

        columns = result.get("columns", [])
        rows = result.get("rows", [])
        row_count = result.get("row_count", 0)

        if not rows:
            return "sql_query: 결과 없음 (0행)"

        # 컬럼명과 함께 결과 요약
        lines = [f"sql_query: {row_count}행 반환"]
        lines.append("컬럼: " + ", ".join(columns))
        for i, row in enumerate(rows[:10]):
            row_str = " | ".join(str(v) for v in row)
            lines.append(f"  [{i+1}] {row_str}")
        if row_count > 10:
            lines.append(f"  ... (총 {row_count}행)")
        return "\n".join(lines)

    elif tool_name == "web_search":
        results = result.get("results", [])
        if not results:
            return "web_search: 결과 없음"
        lines = [f"web_search: {len(results)}건 검색됨"]
        for i, item in enumerate(results[:5]):
            title = item.get("title", "")[:80]
            snippet = item.get("snippet", "")[:200]
            lines.append(f"  [{i+1}] {title}: {snippet}")
        return "\n".join(lines)

    return f"{tool_name}: 실행 완료"
