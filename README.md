# AI Agent MVP

LLM 기반 Tool Agent Loop 아키텍처를 사용하는 AI 에이전트입니다. LangGraph를 활용하여 Policy → Executor → Synthesizer 플로우로 사용자 질문에 대한 답변을 생성합니다.

## 📋 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 설정](#설치-및-설정)
- [사용 방법](#사용-방법)
- [API 엔드포인트](#api-엔드포인트)
- [주요 기능](#주요-기능)

## 개요

이 프로젝트는 **Tool Agent Loop** 패턴을 구현한 AI 에이전트입니다. 사용자의 질문을 받아 다음 단계를 결정하고, 필요한 Tool을 실행하며, 수집된 정보를 종합하여 최종 답변을 생성합니다.

### 핵심 컴포넌트

1. **Policy**: LLM을 사용하여 다음 Action을 결정 (CALL_TOOL, WRITE_NOTE, SYNTHESIZE, STOP)
2. **Executor**: Policy가 결정한 Action을 실행하고 Observation을 기록
3. **Synthesizer**: 누적된 Observation을 바탕으로 최종 답변 생성
4. **Tools**: 
   - `vector_search`: Milvus 벡터DB에서 내부 문서 검색
   - `web_search`: DuckDuckGo를 사용한 웹 검색

## 아키텍처

### 실행 플로우

```
normalize_goal → policy → [route_action]
  ├─ CALL_TOOL → tool_executor → [check_budget] → policy (계속) 또는 force_synthesizer (예산 초과)
  ├─ WRITE_NOTE → write_note → [check_budget] → policy (계속) 또는 force_synthesizer (예산 초과)
  ├─ SYNTHESIZE → synthesizer → END
  └─ STOP → stop → END
```

### 상태 관리

- **AgentState**: 에이전트 실행 상태와 히스토리를 관리
- **Observation**: 각 Action 실행 결과를 append-only 로그로 기록
- **Budget**: 최대 실행 단계 수로 루프 폭주 방지 (기본값: 5단계)

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| **워크플로우 엔진** | LangGraph (StateGraph) |
| **LLM Provider** | Anthropic Claude (claude-sonnet-4-20250514) |
| **임베딩 모델** | sentence-transformers (all-MiniLM-L6-v2) |
| **벡터 DB** | Milvus standalone (Docker Compose) |
| **웹 검색** | duckduckgo-search |
| **API 서버** | FastAPI |
| **데이터 모델** | Pydantic v2 |
| **설정 관리** | pydantic-settings |

## 프로젝트 구조

```
agent/
  app.py                  # FastAPI 엔드포인트
  graph.py                # LangGraph StateGraph 정의 및 컴파일
  config.py               # 환경변수 및 설정 관리
  models.py               # Pydantic 데이터 모델 (Action, Observation, GraphState)
  llm_client.py           # LLMClient 추상 인터페이스 + AnthropicClient 구현
  embeddings.py           # sentence-transformers 기반 EmbeddingProvider
  nodes/
    policy.py             # Policy 노드 (LLM으로 다음 Action 결정)
    synthesizer.py        # Synthesizer 노드 (LLM으로 최종 답변 생성)
    executor.py           # Executor 노드 (Tool 실행, WRITE_NOTE, STOP 처리)
  tools/
    tool_runtime.py       # Tool 디스패치 레지스트리
    vector_search.py      # Milvus 벡터 검색
    web_search.py         # DuckDuckGo 웹 검색
docker/
  docker-compose.yml      # Milvus standalone (etcd + minio + milvus)
requirements.txt          # Python 패키지 의존성
.env                      # API 키 및 설정 (gitignore)
```

## 설치 및 설정

### 1. 저장소 클론 및 디렉토리 이동

```bash
cd /path/to/RAG
```

### 2. 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

`.env` 파일을 생성하고 다음 내용을 입력하세요:

```env
# LLM
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_MODEL=claude-sonnet-4-20250514

# Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Agent
MAX_STEPS=5
VECTOR_SEARCH_TOP_K=5
WEB_SEARCH_TOP_K=5
```

### 5. (선택) Milvus 벡터DB 실행

`vector_search` Tool을 사용하려면 Milvus를 실행해야 합니다:

```bash
cd docker
docker compose up -d
cd ..
```

Milvus는 `http://127.0.0.1:19530`에서 서비스됩니다.

## 사용 방법

### FastAPI 서버 실행

```bash
source venv/bin/activate
uvicorn agent.app:app --host 0.0.0.0 --port 8000
```

서버가 실행되면 `http://localhost:8000`에서 접근할 수 있습니다.

### Python 코드로 직접 실행

```python
from agent.graph import get_graph

graph = get_graph()
result = graph.invoke({"goal": "오늘 서울 날씨는 어때?"})
print(result["final_answer"])
```

## API 엔드포인트

### POST `/agent/run`

에이전트를 실행하여 사용자 메시지에 대한 답변을 생성합니다.

**Request:**
```json
{
  "message": "파이썬 리스트 컴프리헨션이 뭐야?"
}
```

**Response:**
```json
{
  "answer": "파이썬 리스트 컴프리헨션은...",
  "run_id": "10c35df32bff"
}
```

**예시:**
```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "오늘 서울 날씨는 어때?"}'
```

### GET `/agent/state/{run_id}`

에이전트 실행 상태를 조회합니다.

**Response:**
```json
{
  "run_id": "10c35df32bff",
  "goal": "오늘 서울 날씨는 어때?",
  "cursor": 2,
  "max_steps": 5,
  "status": "DONE",
  "observations": [...],
  "final_answer": "..."
}
```

**예시:**
```bash
curl http://localhost:8000/agent/state/10c35df32bff
```

### GET `/health`

헬스 체크 엔드포인트입니다.

**Response:**
```json
{
  "status": "ok"
}
```

## 주요 기능

### 1. Tool Agent Loop

- **Policy**: LLM이 현재 상태를 분석하여 다음 Action 결정
- **Executor**: Tool 실행 및 Observation 기록
- **Synthesizer**: 수집된 정보를 종합하여 최종 답변 생성

### 2. 예산 기반 루프 제어

- 최대 실행 단계 수(`max_steps`)로 무한 루프 방지
- 예산 초과 시 자동으로 Synthesizer 호출하여 종료

### 3. Tool 지원

#### vector_search
- Milvus 벡터DB에서 유사도 검색
- sentence-transformers로 쿼리 임베딩 생성
- 컬렉션, 쿼리 텍스트, top_k, 필터 옵션 지원

#### web_search
- DuckDuckGo 기반 무료 웹 검색
- 제목, 스니펫, URL 반환
- 지역 설정 옵션 지원

### 4. 상태 관리

- 각 실행마다 고유한 `run_id` 생성
- Observation을 append-only 로그로 기록
- 실행 상태(RUNNING/DONE) 추적

## 참고 문서

- [개발 정의서](agent_mvp_spec.md): 상세한 아키텍처 및 구현 사양
- [아키텍처 플로우차트](AI_AGENT_ARCHITECTURE.md): 실행 플로우 시각화

## 라이선스

이 프로젝트는 MVP(Minimum Viable Product) 단계입니다.

