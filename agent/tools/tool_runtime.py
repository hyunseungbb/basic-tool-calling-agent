"""Tool 디스패치 모듈.

tool_name을 기반으로 적절한 Tool 함수를 실행한다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent.tools.vector_search import vector_search
from agent.tools.web_search import web_search

logger = logging.getLogger(__name__)

# Tool 이름 → 실행 함수 매핑 레지스트리
TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "vector_search": vector_search,
    "web_search": web_search,
}

# Tool 설명 (Policy 프롬프트에 제공)
TOOL_DESCRIPTIONS: dict[str, str] = {
    "vector_search": (
        "Milvus 벡터DB에서 내부 문서를 유사도 검색합니다. "
        "파라미터: collection(str, 기본 'docs'), query_text(str, 필수), "
        "top_k(int, 기본 5), filter(dict, optional)"
    ),
    "web_search": (
        "DuckDuckGo를 사용하여 웹에서 정보를 검색합니다. "
        "파라미터: query(str, 필수), top_k(int, 기본 5), region(str, optional)"
    ),
}


def get_available_tools() -> list[str]:
    """사용 가능한 Tool 이름 목록을 반환한다."""
    return list(TOOL_REGISTRY.keys())


def get_tool_descriptions() -> str:
    """Policy 프롬프트에 포함할 Tool 설명 문자열을 생성한다."""
    lines: list[str] = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _normalize_tool_name(tool_name: str) -> str:
    """Tool 이름을 정규화한다.

    Enum 값(예: ToolName.WEB_SEARCH)이 문자열로 전달될 경우를 처리한다.
    """
    # Enum의 .value가 직렬화된 경우 처리
    normalized = str(tool_name)
    # "ToolName.WEB_SEARCH" → "web_search" 등 처리
    if "." in normalized:
        normalized = normalized.split(".")[-1].lower()
    return normalized


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """지정된 Tool을 실행하고 결과를 반환한다.

    Args:
        tool_name: 실행할 Tool 이름
        tool_args: Tool에 전달할 인자

    Returns:
        Tool 실행 결과 딕셔너리

    Raises:
        ValueError: 등록되지 않은 Tool 이름인 경우
    """
    tool_name = _normalize_tool_name(tool_name)

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"등록되지 않은 Tool: '{tool_name}'. "
            f"사용 가능: {get_available_tools()}"
        )

    logger.info("Tool 실행: %s, args=%s", tool_name, tool_args)
    result = TOOL_REGISTRY[tool_name](tool_args)
    logger.info("Tool 실행 완료: %s", tool_name)
    return result

