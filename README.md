# AI Agent MVP

LangGraph 기반 Tool Agent Loop 아키텍처의 AI 에이전트입니다.
WFE 글로벌 주식시장 데이터를 SQLite에 저장하고, 자연어 질문에 대해 검색 + 웹 검색을 결합하여 답변합니다.

## 아키텍처

```
사용자 질문 → Policy(LLM) → Executor(Tool 실행) → Synthesizer(답변 생성)
                ↑                    │
                └────────────────────┘  (루프: 최대 5회)
```

- **Policy**: LLM이 다음 액션 결정 (CALL_TOOL / WRITE_NOTE / SYNTHESIZE / STOP)
- **Executor**: vector_search(SQLite FTS5), web_search(DuckDuckGo) 실행
- **Synthesizer**: 수집된 정보를 종합하여 최종 답변 스트리밍

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 워크플로우 | LangGraph (StateGraph) |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| 벡터 검색 | SQLite FTS5 (Docker 불필요) |
| 웹 검색 | duckduckgo-search |
| 백엔드 | FastAPI + SSE 스트리밍 |
| 프론트엔드 | Streamlit |

## 프로젝트 구조

```
agent/
  app.py                # FastAPI 엔드포인트 (SSE 스트리밍)
  graph.py              # LangGraph StateGraph 정의
  config.py             # 환경변수 설정 (pydantic-settings)
  models.py             # 데이터 모델
  llm_client.py         # Anthropic LLM 클라이언트
  nodes/
    policy.py           # Policy 노드
    synthesizer.py      # Synthesizer 노드
    executor.py         # Executor 노드
  tools/
    tool_runtime.py     # Tool 디스패치
    vector_search.py    # SQLite FTS5 검색
    web_search.py       # DuckDuckGo 웹 검색
    ingest.py           # 엑셀 → SQLite 적재
streamlit_app.py        # Streamlit 챗봇 UI
requirements.txt
.env.example
```

## macOS 로컬 실행 가이드

### 사전 요구사항

- Python 3.10 이상 (macOS 기본 python3 또는 Homebrew로 설치)
- Anthropic API 키

### 1. 가상환경 생성 및 활성화

```bash
cd /path/to/agent_modified
python3 -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
pip install streamlit
```

### 3. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키를 입력합니다:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxx   # 실제 키로 교체
LLM_MODEL=claude-sonnet-4-20250514
SQLITE_PATH=data/knowledge.db
MAX_STEPS=5
VECTOR_SEARCH_TOP_K=5
WEB_SEARCH_TOP_K=5
```

### 4. 데이터 적재 (최초 1회)

엑셀 데이터를 SQLite DB에 적재합니다:

```bash
python -m agent.tools.ingest --excel <엑셀파일경로>
```

예시:
```bash
python -m agent.tools.ingest --excel report__9_.xlsx
```

`data/knowledge.db` 파일이 생성됩니다.

### 5. 백엔드 서버 실행

```bash
uvicorn agent.app:app --host 0.0.0.0 --port 8000
```

헬스 체크 확인:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 6. 프론트엔드 실행 (별도 터미널)

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속하여 챗봇을 사용합니다.

## API 사용법

### POST `/agent/run` — 에이전트 실행 (SSE 스트리밍)

```bash
curl -N -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "상하이 거래소 시가총액은?"}'
```

SSE 이벤트:
- `step` — 실행 과정 (policy 결정, tool 실행)
- `token` — 답변 텍스트 청크
- `done` — 완료
- `error` — 오류

### GET `/agent/state/{run_id}` — 상태 조회

```bash
curl http://localhost:8000/agent/state/{run_id}
```

### GET `/health` — 헬스 체크

```bash
curl http://localhost:8000/health
```

## 질문 예시

- 상하이 거래소 시가총액은?
- Americas 지역 시가총액 TOP 5
- 2026년 2월 아시아 거래량 비교해줘
- 전월 대비 가장 많이 오른 거래소는?
- IPO 자본조달 현황 알려줘
- 한국 거래소 현황

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `ModuleNotFoundError` | `source venv/bin/activate` 확인 |
| 백엔드 연결 실패 | `uvicorn agent.app:app --port 8000` 실행 확인 |
| API 키 오류 | `.env` 파일에 `ANTHROPIC_API_KEY` 설정 확인 |
| 데이터 없음 | `python -m agent.tools.ingest --excel <파일>` 실행 |
