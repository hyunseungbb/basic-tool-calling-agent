"""Synthesizer 노드: SQL 결과를 바탕으로 자연어 답변을 생성한다.

SQL 쿼리 결과(테이블 데이터)를 분석하여 사용자가 이해하기 쉬운
자연어 답변으로 변환한다. 단위 변환(백만→조/억)도 처리한다.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from agent.llm_client import get_llm_client
from agent.models import AgentStatus, GraphState

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """\
당신은 글로벌 주식시장 데이터 분석 AI의 Synthesizer 모듈입니다.
SQL 쿼리 결과 데이터를 바탕으로 사용자의 질문에 대한 **최종 답변**을 작성합니다.

## 답변 규칙
1. **SQL 결과 데이터에만 근거**하여 답변하세요. 추측하지 마세요.
2. 데이터가 없거나 부족하면 솔직하게 "해당 데이터가 없습니다"라고 답하세요.
3. **단위 변환**을 반드시 수행하세요:
   - value_usd는 **백만 USD** 단위입니다.
   - 큰 금액은 읽기 쉽게 변환: 31,401,660 백만$ → **약 31.4조 달러** (trillion)
   - 중간 금액: 372,339 백만$ → **약 3,723억 달러** (billion)
   - 작은 금액: 5,234 백만$ → **약 52.3억 달러**
   - 원화 참고: 1 USD ≈ 1,400원 기준으로 원화 환산도 병기하면 좋습니다.
4. **순위 정보**가 있으면 명확하게 표시하세요.
5. **전월대비/전년대비 변화율**은 퍼센트로 표시하세요 (0.05 → 5%).
6. 답변은 **구조화된 형태**로 작성하세요 (표, 목록 등 활용).
7. 한국어로 답변하세요.
8. 일평균 값이 있으면 "일평균 거래대금: 약 X억 달러"와 같이 표시하세요.
9. **여러 거래소가 매칭**된 경우 (예: Nasdaq - US, Nasdaq Nordic and Baltics), 각각의 데이터를 구분하여 모두 보여주세요. 사용자가 특정 거래소를 지정하지 않았다면 어떤 거래소들이 포함되었는지 안내하세요.
"""


def build_synthesis_prompt(state: GraphState) -> str:
    """Synthesizer에 전달할 user prompt를 구성한다."""
    observations_text = "없음"
    if state.get("observations"):
        obs_lines = []
        for obs in state["observations"]:
            kind = obs.get("kind", "?")
            summary = obs.get("summary", "")
            payload = obs.get("payload")
            tool_name = obs.get("tool_name", "")
            tool_args = obs.get("tool_args", {})

            line = f"### [{obs.get('step_id', '?')}] {kind}"
            if tool_name == "sql_query" and tool_args:
                line += f"\n실행 SQL: `{tool_args.get('sql', '')}`"

            if payload:
                # SQL 결과를 테이블 형태로 전달
                if "columns" in payload and "rows" in payload:
                    columns = payload["columns"]
                    rows = payload["rows"]
                    if columns and rows:
                        line += f"\n결과: {len(rows)}행"
                        # 헤더
                        line += "\n| " + " | ".join(str(c) for c in columns) + " |"
                        line += "\n| " + " | ".join("---" for _ in columns) + " |"
                        # 데이터 (최대 20행)
                        for row in rows[:20]:
                            line += "\n| " + " | ".join(str(v) for v in row) + " |"
                        if len(rows) > 20:
                            line += f"\n... (총 {len(rows)}행 중 20행만 표시)"
                    elif payload.get("error"):
                        line += f"\nSQL 오류: {payload['error']}"
                    else:
                        line += "\n결과: 0행 (데이터 없음)"
                # 웹 검색 결과
                elif "results" in payload:
                    for result in payload["results"][:5]:
                        line += f"\n- {result.get('title', '')}: {result.get('snippet', '')[:200]}"
            else:
                line += f"\n{summary}"

            obs_lines.append(line)
        observations_text = "\n".join(obs_lines)

    return f"""\
## 사용자 질문
{state.get("goal", "")}

## 수집된 데이터
{observations_text}

위 데이터를 바탕으로 최종 답변을 작성하세요. 반드시 단위 변환(백만 USD → 조/억 달러)을 수행하세요.
"""


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """Synthesizer 노드: 최종 답변을 생성한다."""
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
    """Synthesizer를 스트리밍 방식으로 실행한다."""
    logger.info("Synthesizer 스트리밍 시작")

    user_prompt = build_synthesis_prompt(state)
    llm = get_llm_client()

    yield from llm.generate_stream(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)
