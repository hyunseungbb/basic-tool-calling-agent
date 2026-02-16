"""FastAPI 엔드포인트.

정의서 §11:
- POST /agent/run: 에이전트 실행
- GET /agent/state/{run_id}: 상태 조회
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent MVP",
    description="Tool Agent Loop 기반 AI 에이전트 API",
    version="0.1.0",
)

# 인메모리 상태 저장소 (MVP)
_state_store: dict[str, dict[str, Any]] = {}


# ──────────────────────────── Request/Response 모델 ────────────────────────────


class RunRequest(BaseModel):
    """에이전트 실행 요청."""

    message: str


class RunResponse(BaseModel):
    """에이전트 실행 응답."""

    answer: str
    run_id: str


class StateResponse(BaseModel):
    """상태 조회 응답."""

    run_id: str
    goal: str
    cursor: int
    max_steps: int
    status: str
    observations: list[dict[str, Any]]
    final_answer: str


# ──────────────────────────── 엔드포인트 ────────────────────────────


@app.post("/agent/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    """에이전트를 실행하여 사용자 메시지에 대한 답변을 생성한다.

    Args:
        request: 사용자 메시지가 포함된 요청

    Returns:
        답변과 run_id가 포함된 응답
    """
    run_id = uuid.uuid4().hex[:12]
    logger.info("에이전트 실행 시작: run_id=%s, message='%s'", run_id, request.message[:50])

    # 초기 상태 구성
    initial_state = {
        "run_id": run_id,
        "goal": request.message,
    }

    try:
        # LangGraph 그래프 실행
        graph = get_graph()
        final_state = graph.invoke(initial_state)

        # 상태 저장
        _state_store[run_id] = final_state

        answer = final_state.get("final_answer", "답변을 생성하지 못했습니다.")
        logger.info("에이전트 실행 완료: run_id=%s", run_id)

        return RunResponse(answer=answer, run_id=run_id)

    except Exception as e:
        logger.error("에이전트 실행 에러: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"에이전트 실행 중 오류 발생: {str(e)}")


@app.get("/agent/state/{run_id}", response_model=StateResponse)
async def get_agent_state(run_id: str) -> StateResponse:
    """에이전트 실행 상태를 조회한다.

    Args:
        run_id: 조회할 실행 ID

    Returns:
        에이전트 상태 정보
    """
    if run_id not in _state_store:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}'를 찾을 수 없습니다.")

    state = _state_store[run_id]
    return StateResponse(
        run_id=state.get("run_id", run_id),
        goal=state.get("goal", ""),
        cursor=state.get("cursor", 0),
        max_steps=state.get("max_steps", 5),
        status=state.get("status", "UNKNOWN"),
        observations=state.get("observations", []),
        final_answer=state.get("final_answer", ""),
    )


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    return {"status": "ok"}

