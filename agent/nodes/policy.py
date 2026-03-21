"""Policy 노드: LLM을 호출하여 다음 Action을 결정한다.

사용자 질문을 분석하고, DB 스키마를 참고하여 SQL 쿼리를 직접 생성한다.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.llm_client import get_llm_client
from agent.models import Action, ActionType, GraphState
from agent.tools.tool_runtime import get_db_schema, get_tool_descriptions

logger = logging.getLogger(__name__)

POLICY_SYSTEM_PROMPT = """\
당신은 글로벌 주식시장 데이터 분석 AI 에이전트의 Policy 모듈입니다.
사용자의 질문을 분석하고, SQLite DB에서 데이터를 조회하기 위한 **SQL 쿼리를 직접 생성**해야 합니다.

## 사용 가능한 Action 유형
1. **CALL_TOOL** — Tool을 호출합니다.
   - `sql_query`: SQL SELECT 쿼리를 실행합니다. 아래 DB 스키마를 참고하여 정확한 SQL을 작성하세요.
   - `web_search`: DB에 없는 최신 정보나 일반 지식이 필요할 때 사용합니다.
2. **WRITE_NOTE** — 중간 분석 메모를 작성합니다.
3. **SYNTHESIZE** — 충분한 데이터가 수집되었으면, 최종 답변 생성을 요청합니다.
4. **STOP** — 더 이상 진행할 수 없을 때 중단합니다. note에 이유를 포함하세요.

## 사용 가능한 Tool
{tool_descriptions}

## DB 스키마
{db_schema}

## SQL 작성 규칙
1. **시가총액 질문** → indicator = 'Total Equity Market - Market Capitalisation'
2. **거래대금 질문** → indicator = 'Total Equity Market - Value traded (EOB Total)'
3. **일평균 거래대금** → value_usd / trading_days (trading_days 테이블 JOIN 필요)
4. **순위 질문** → RANK() OVER (ORDER BY ... DESC) 사용
5. **TOP N** → ORDER BY ... DESC LIMIT N
6. **비교 질문** → 여러 거래소/지역을 한 번에 조회
7. **전월대비** → pct_mtm 컬럼 사용 (0.05 = 5%)
8. **전년대비** → pct_yty 컬럼 사용
9. **value_usd 단위**: Monetary 데이터는 **백만 USD** 단위입니다.
   - 시가총액 31,401,660 → 약 31.4조 달러(trillion)
   - 거래대금 372,339 → 약 3,723억 달러(billion)
10. 월은 영문 3글자: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
11. **거래소명은 반드시 LIKE 검색 사용**: 동일 브랜드에 여러 거래소가 있을 수 있습니다.
   - 예: "Nasdaq"으로 검색하면 "Nasdaq - US", "Nasdaq Nordic and Baltics" 모두 매칭
   - `WHERE exchange_name LIKE '%Nasdaq%'` 형태로 작성하세요
   - 한국어 거래소명 키워드: 한국거래소=Korea, 뉴욕=NYSE, 상하이=Shanghai, 도쿄=Japan Exchange, 나스닥=Nasdaq, 홍콩=Hong Kong, 선전=Shenzhen, 런던=London Stock Exchange
12. **결과가 0건이면**: LIKE 범위를 넓히거나, 먼저 `SELECT DISTINCT exchange_name FROM market_data WHERE exchange_name LIKE '%키워드%'`로 거래소명을 확인한 뒤 재시도하세요.
13. **여러 거래소가 매칭될 때**: 모두 포함하여 결과를 보여주세요. Synthesizer가 사용자에게 어떤 거래소인지 안내합니다.

## 판단 기준
- 주식시장 데이터 질문 → **반드시 sql_query 사용** (SQL을 직접 작성)
- DB에 없는 정보 (뉴스, 일반 지식) → web_search 사용
- 이전 SQL 결과가 충분하면 → SYNTHESIZE
- 이전 SQL에 오류가 있었으면 → SQL을 수정하여 다시 CALL_TOOL
- 이전 SQL 결과가 0건이면 → LIKE 범위를 넓히거나 거래소명 확인 SQL을 먼저 실행
- 남은 단계가 1이면 → SYNTHESIZE 우선

## 응답 형식 (반드시 이 JSON만 반환)
```json
{{
  "type": "CALL_TOOL",
  "tool_name": "sql_query",
  "tool_args": {{
    "sql": "SELECT exchange_name, value_usd, RANK() OVER (ORDER BY value_usd DESC) AS rank FROM market_data WHERE indicator = 'Total Equity Market - Market Capitalisation' AND year = 2025 AND month = 'Dec'"
  }}
}}
```
- CALL_TOOL일 때만 tool_name, tool_args 포함
- WRITE_NOTE, STOP일 때만 note 포함
- SYNTHESIZE일 때는 type만 포함
"""


def _build_user_prompt(state: GraphState) -> str:
    """Policy에 전달할 user prompt를 구성한다."""
    observations_text = "없음"
    if state.get("observations"):
        obs_lines = []
        for obs in state["observations"]:
            kind = obs.get("kind", "?")
            summary = obs.get("summary", "")
            tool_name = obs.get("tool_name", "")
            tool_args = obs.get("tool_args", {})

            line = f"[{obs.get('step_id', '?')}] ({kind})"
            if tool_name:
                line += f" Tool={tool_name}"
            if tool_name == "sql_query" and tool_args:
                line += f"\n  실행한 SQL: {tool_args.get('sql', '')}"
            line += f"\n  결과: {summary}"

            obs_lines.append(line)
        observations_text = "\n".join(obs_lines)

    remaining = state.get("max_steps", 5) - state.get("cursor", 0)

    return f"""\
## 현재 상태
- 사용자 질문: {state.get("goal", "")}
- 진행 단계: {state.get("cursor", 0)} / {state.get("max_steps", 5)} (남은 단계: {remaining})

## 이전 실행 기록
{observations_text}

위 정보를 바탕으로 다음 Action을 JSON으로 반환하세요.
"""


def policy_node(state: GraphState) -> dict[str, Any]:
    """Policy 노드: LLM을 호출하여 다음 Action을 결정한다."""
    logger.info("Policy 노드 실행 (cursor=%d)", state.get("cursor", 0))

    system_prompt = POLICY_SYSTEM_PROMPT.format(
        tool_descriptions=get_tool_descriptions(),
        db_schema=get_db_schema(),
    )
    user_prompt = _build_user_prompt(state)

    llm = get_llm_client()

    try:
        action_dict = llm.generate_json(system_prompt, user_prompt)
        action = Action(**action_dict)
        logger.info("Policy 결정: %s", action.type.value)
        return {"current_action": action.model_dump(mode="json")}
    except Exception as e:
        logger.error("Policy 에러, STOP 반환: %s", str(e))
        fallback = Action(
            type=ActionType.STOP,
            note=f"Policy 결정 중 오류 발생: {str(e)}",
        )
        return {"current_action": fallback.model_dump(mode="json")}
