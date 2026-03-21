"""Streamlit 챗봇 UI - basic-tool-calling-agent 백엔드 연동."""

import json
import requests
import streamlit as st

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="글로벌 주식시장 AI 챗봇",
    page_icon="📈",
    layout="wide",
)

BACKEND_URL = "http://localhost:8000"

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")
    backend_url = st.text_input("백엔드 URL", value=BACKEND_URL)

    # 백엔드 상태 확인
    try:
        res = requests.get(f"{backend_url}/health", timeout=2)
        if res.status_code == 200:
            st.success("✅ 백엔드 연결됨")
        else:
            st.error("❌ 백엔드 응답 오류")
    except Exception:
        st.error("❌ 백엔드 연결 실패\n`uvicorn agent.app:app` 실행 확인")

    st.divider()
    st.markdown("**💡 질문 예시**")
    examples = [
        "상하이 거래소 시가총액은?",
        "Americas 지역 시가총액 TOP 5",
        "2026년 2월 아시아 거래량 비교",
        "IPO 자본조달 현황 알려줘",
        "전월 대비 가장 많이 오른 거래소는?",
        "한국 거래소 현황",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_input"] = ex

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 메인 영역 ─────────────────────────────────────────────────
st.title("📈 글로벌 주식시장 AI 챗봇")
st.caption("WFE 데이터 기반 | Tool Agent Loop (Policy → Executor → Synthesizer)")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("🔍 Agent 실행 과정"):
                for step in msg["steps"]:
                    st.markdown(f"- {step}")

# 사이드바 예시 버튼으로 입력된 경우 처리
pending = st.session_state.pop("pending_input", None)

# 사용자 입력
prompt = st.chat_input("주식시장에 대해 질문하세요...") or pending

if prompt:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 백엔드 호출 (SSE 스트리밍)
    with st.chat_message("assistant"):
        steps = []
        answer_placeholder = st.empty()
        step_placeholder = st.empty()
        full_answer = ""

        try:
            with requests.post(
                f"{backend_url}/agent/run",
                json={"message": prompt},
                stream=True,
                timeout=60,
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            data = json.loads(data_str)
                        except Exception:
                            continue

                        if event_type == "step":
                            action = data.get("action", "")
                            tool = data.get("tool", "")
                            cursor = data.get("cursor", "")
                            summary = data.get("summary", "")

                            if action == "CALL_TOOL":
                                step_msg = f"🔧 **Step {cursor}**: `{tool}` 실행 중..."
                            elif action == "SYNTHESIZE":
                                step_msg = f"✍️ **Step {cursor}**: 답변 생성 중..."
                            elif summary:
                                step_msg = f"📝 **Step {cursor}**: {summary[:100]}"
                            else:
                                step_msg = f"⚙️ **Step {cursor}**: {data.get('type', '')}"

                            steps.append(step_msg)
                            with step_placeholder.expander("🔍 Agent 실행 과정", expanded=True):
                                for s in steps:
                                    st.markdown(s)

                        elif event_type == "token":
                            full_answer += data.get("text", "")
                            answer_placeholder.markdown(full_answer + "▌")

                        elif event_type == "done":
                            answer_placeholder.markdown(full_answer)
                            step_placeholder.empty()
                            if steps:
                                with st.expander("🔍 Agent 실행 과정"):
                                    for s in steps:
                                        st.markdown(s)

                        elif event_type == "error":
                            st.error(f"오류: {data.get('message', '')}")

        except requests.exceptions.ConnectionError:
            full_answer = "❌ 백엔드 서버에 연결할 수 없습니다. `uvicorn agent.app:app --port 8000`을 실행하세요."
            answer_placeholder.error(full_answer)
        except Exception as e:
            full_answer = f"❌ 오류 발생: {str(e)}"
            answer_placeholder.error(full_answer)

    # 히스토리 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "steps": steps,
    })
