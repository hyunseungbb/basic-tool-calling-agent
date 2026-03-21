"""FastAPI 엔드포인트.

정의서 §11:
- POST /agent/run: 에이전트 실행 (SSE 스트리밍)
- GET /agent/state/{run_id}: 상태 조회
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import get_streaming_graph
from agent.nodes.synthesizer import stream_synthesis

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent MVP",
    description="Tool Agent Loop 기반 AI 에이전트 API (SSE 스트리밍)",
    version="0.2.0",
)

# CORS 설정 (React 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인메모리 상태 저장소 (MVP)
_state_store: dict[str, dict[str, Any]] = {}


# ──────────────────────────── Request/Response 모델 ────────────────────────────


class RunRequest(BaseModel):
    """에이전트 실행 요청."""

    message: str


class StateResponse(BaseModel):
    """상태 조회 응답."""

    run_id: str
    goal: str
    cursor: int
    max_steps: int
    status: str
    observations: list[dict[str, Any]]
    final_answer: str


# ──────────────────────────── SSE 유틸리티 ────────────────────────────


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """SSE 형식의 이벤트 문자열을 생성한다.

    Args:
        event: 이벤트 이름 (step, token, done, error)
        data: JSON 직렬화할 데이터

    Returns:
        SSE 형식 문자열 (event: ...\\ndata: ...\\n\\n)
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ──────────────────────────── 엔드포인트 ────────────────────────────


@app.post("/agent/run")
async def run_agent(request: RunRequest) -> StreamingResponse:
    """에이전트를 실행하여 SSE 스트리밍으로 답변을 생성한다.

    SSE 이벤트 형식:
    - event: step  — 그래프 루프 진행 상황 (policy 결정, tool 실행 등)
    - event: token — Synthesizer의 최종 답변 텍스트 청크
    - event: done  — 완료 (run_id, status 포함)
    - event: error — 오류 발생

    Args:
        request: 사용자 메시지가 포함된 요청

    Returns:
        SSE StreamingResponse
    """
    run_id = uuid.uuid4().hex[:12]
    logger.info("에이전트 스트리밍 시작: run_id=%s, message='%s'", run_id, request.message[:50])

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 스트리밍 그래프 실행 (합성 직전까지)
            graph = get_streaming_graph()
            initial_state = {"run_id": run_id, "goal": request.message}

            # LangGraph의 stream()을 사용하여 노드별 진행 상황 전달
            final_state: dict[str, Any] = {}
            for chunk in graph.stream(initial_state):
                # chunk: {node_name: state_update}
                for node_name, state_update in chunk.items():
                    final_state.update(state_update)

                    # step 이벤트: 클라이언트에 진행 상황 알림
                    step_data = _build_step_event(node_name, final_state)
                    if step_data:
                        yield _sse_event("step", step_data)

            # STOP 액션인 경우 스트리밍 합성 없이 바로 종료
            if final_state.get("final_answer"):
                yield _sse_event("token", {"text": final_state["final_answer"]})
                _state_store[run_id] = final_state
                yield _sse_event(
                    "done",
                    {"run_id": run_id, "status": final_state.get("status", "DONE")},
                )
                return

            # 2. Synthesizer 스트리밍: 토큰 단위로 SSE 전송
            accumulated_answer = ""
            for text_chunk in stream_synthesis(final_state):
                accumulated_answer += text_chunk
                yield _sse_event("token", {"text": text_chunk})

            # 3. 상태 저장
            final_state["final_answer"] = accumulated_answer
            final_state["status"] = "DONE"
            _state_store[run_id] = final_state

            logger.info(
                "에이전트 스트리밍 완료: run_id=%s, answer_len=%d",
                run_id,
                len(accumulated_answer),
            )

            # 4. done 이벤트
            yield _sse_event("done", {"run_id": run_id, "status": "DONE"})

        except Exception as e:
            logger.error("에이전트 스트리밍 에러: %s", str(e), exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _build_step_event(
    node_name: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    """노드 실행 정보를 step 이벤트 데이터로 변환한다.

    내부 상세 노드는 건너뛰고, 클라이언트에 유의미한 정보만 전달한다.
    """
    if node_name == "policy":
        action = state.get("current_action", {})
        action_type = action.get("type", "")
        data: dict[str, Any] = {
            "type": "policy",
            "cursor": state.get("cursor", 0),
            "action": action_type,
        }
        if action_type == "CALL_TOOL":
            data["tool"] = action.get("tool_name", "")
        return data

    if node_name in ("tool_executor", "write_note"):
        observations = state.get("observations", [])
        if observations:
            latest = observations[-1]
            return {
                "type": node_name,
                "cursor": state.get("cursor", 0),
                "summary": latest.get("summary", "")[:200],
            }

    return None


@app.get("/agent/state/{run_id}", response_model=StateResponse)
async def get_agent_state(run_id: str) -> StateResponse:
    """에이전트 실행 상태를 조회한다.

    Args:
        run_id: 조회할 실행 ID

    Returns:
        에이전트 상태 정보
    """
    if run_id not in _state_store:
        raise HTTPException(
            status_code=404, detail=f"run_id '{run_id}'를 찾을 수 없습니다."
        )

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
