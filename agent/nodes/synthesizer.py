"""Synthesizer 노드: 누적된 Observation을 근거로 최종 답변을 생성한다.

정의서 §2.1(3) Synthesizer:
- 입력: goal, observations
- 출력: final_answer 문자열

스트리밍 지원:
- stream_synthesis(state)를 사용하여 토큰 단위로 최종 답변을 생성할 수 있다.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from agent.llm_client import get_llm_client
from agent.models import AgentStatus, GraphState

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """\
당신은 AI 에이전트의 Synthesizer 모듈입니다.
사용자의 목표(goal)와 지금까지 수집된 관측 기록(observations)을 바탕으로
**최종 답변**을 작성해야 합니다.

## 규칙
1. 관측 기록에 있는 정보만을 근거로 답변하세요.
2. 근거가 부족하면 솔직하게 "충분한 정보를 찾지 못했습니다"라고 답하세요.
3. 출처(URL, 문서 ID 등)가 있다면 답변 끝에 참고자료로 정리하세요.
4. 답변은 명확하고 구조화된 형태로 작성하세요.
5. 한국어로 답변하세요.
"""


def build_synthesis_prompt(state: GraphState) -> str:
    """Synthesizer에 전달할 user prompt를 구성한다.

    synthesizer_node와 stream_synthesis 모두에서 사용한다.
    """
    observations_text = "없음"
    if state.get("observations"):
        obs_lines = []
        for obs in state["observations"]:
            kind = obs.get("kind", "?")
            summary = obs.get("summary", "")
            payload = obs.get("payload")

            line = f"[{obs.get('step_id', '?')}] ({kind}) {summary}"
            if payload:
                # payload에서 주요 정보 추출
                if "hits" in payload:
                    for hit in payload["hits"][:3]:
                        line += f"\n  - [{hit.get('score', 0):.3f}] {hit.get('text', '')[:200]}"
                elif "results" in payload:
                    for result in payload["results"][:3]:
                        line += f"\n  - {result.get('title', '')}: {result.get('snippet', '')[:200]}"
                        line += f"\n    URL: {result.get('url', '')}"
            obs_lines.append(line)
        observations_text = "\n".join(obs_lines)

    return f"""\
## 사용자 목표
{state.get("goal", "")}

## 수집된 관측 기록
{observations_text}

위 정보를 바탕으로 최종 답변을 작성하세요.
"""


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """Synthesizer 노드: 최종 답변을 생성한다.

    Args:
        state: 현재 그래프 상태

    Returns:
        final_answer와 status가 포함된 상태 업데이트 딕셔너리
    """
    logger.info("Synthesizer 노드 실행")

    user_prompt = build_synthesis_prompt(state)
    llm = get_llm_client()

    try:
        final_answer = llm.generate(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)
        logger.info("Synthesizer 완료: %d chars", len(final_answer))
    except Exception as e:
        logger.error("Synthesizer 에러: %s", str(e))
        final_answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    return {
        "final_answer": final_answer,
        "status": AgentStatus.DONE.value,
    }


def stream_synthesis(state: GraphState) -> Generator[str, None, None]:
    """Synthesizer를 스트리밍 방식으로 실행한다.

    Anthropic의 messages.stream() API를 사용하여 토큰 단위로
    최종 답변 텍스트 청크를 yield한다.

    Args:
        state: 현재 그래프 상태 (observations가 수집된 상태)

    Yields:
        텍스트 청크 문자열
    """
    logger.info("Synthesizer 스트리밍 시작")

    user_prompt = build_synthesis_prompt(state)
    llm = get_llm_client()

    yield from llm.generate_stream(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)

