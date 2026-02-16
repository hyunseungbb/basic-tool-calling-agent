# 에이전트 개발 정의서 (MVP) — Tool Agent Loop 기반

## 1. 문서 개요

### 1.1 목적
- 사용자의 요청을 받아 **LLM 기반 Tool Agent Loop** 아키텍처로 처리하는 에이전트를 Python으로 개발한다.
- Tool은 2개만 제공한다.
  1) `vector_search`: Milvus 벡터DB 검색 (로컬 docker-compose로 실행 후 연동)  
     - Milvus standalone docker-compose 실행 가이드: https://milvus.io/docs/ko/install_standalone-docker-compose.md
  2) `web_search`: 무료 웹검색(뼈대). 우선 DuckDuckGo 기반으로 구현(키/유료 API 연동은 추후).  
     - DuckDuckGo Search 라이브러리: https://pypi.org/project/duckduckgo-search/

### 1.2 범위(MVP)
- Router는 별도 모듈로 분리하지 않고, **Policy가 Action을 결정**하는 구조로 단순화한다.
- 아래 flowchart 개념을 따른다:
  - **Action 결정(Policy)** → **Executor 실행** → **Observation/State 누적** → **종료 시 Synthesizer 응답 생성**
- 포함: 상태 관리(AgentState), 관측(Observation) 기록, 예산(budget) 기반 루프 종료
- 제외: 재시도, fallback, 검증(critic), tool-error 처리 정책, 멀티툴 병렬, rerank, 세션 메모리 장기화


## 2. 아키텍처 개념 정리(용어 정의)

### 2.1 핵심 컴포넌트

#### (1) Policy (LLM 호출 1)
- 역할: “다음에 무엇을 할지” 결정해서 **Action 1개**를 반환한다.
- 입력: `goal`, `observations`, `budget`, `available_tools`
- 출력(Action):
  - `CALL_TOOL` (vector_search 또는 web_search 호출)
  - `WRITE_NOTE` (중간 요약/정리 메모를 Observation에 남김)
  - `SYNTHESIZE` (최종 답변 생성 요청)
  - `STOP` (중단 메시지로 종료)

> **MVP에서는** Policy가 `SYNTHESIZE`를 내리면 “충분하다”고 판단한 것으로 간주한다.

#### (2) Executor (코드 로직)
- 역할: Policy가 준 Action을 실행하고, 결과를 Observation으로 쌓고, 상태를 저장한다.
- 특징: Executor는 “결정”을 거의 하지 않는다.  
  단, **하드 게이트(Continue loop)** 는 Executor가 코드로 체크한다.
  - `cursor < max_steps` (루프 예산)
  - `status != DONE`

#### (3) Synthesizer (LLM 호출 2)
- 역할: 누적된 Observation들을 근거로 최종 답변을 생성한다.
- 입력: `goal`, `observations`
- 출력: `final_answer` 문자열

#### (4) AgentState
- 역할: 에이전트 실행의 “현재 상태 + 히스토리”를 담는 데이터 구조
- 저장: MVP는 로컬 파일(JSON) 또는 메모리(개발 편의).  
  (추후 Redis/DB로 교체 가능하도록 인터페이스 분리)

#### (5) Observation
- 역할: Action 결과를 기록하는 append-only 로그
- 예: tool 호출 파라미터, 결과 요약, 결과 payload 일부/포인터


## 3. 실행 플로우(MVP)

### 3.1 루프 종료(Continue loop) 판단은 누가/언제?
- **Executor가 매 턴 시작 시** 다음만 확인한다(하드 게이트):
  - `state.status == DONE` 이면 종료
  - `state.cursor >= state.budget.max_steps` 이면 종료 후 Synthesizer 1회 호출로 마무리

> “정보가 충분한지”는 Policy가 `SYNTHESIZE` Action을 반환함으로써 결정한다.

### 3.2 LLM 호출 지점
- LLM 호출은 딱 2군데:
  1) Policy: 다음 Action 결정
  2) Synthesizer: 최종 응답 생성


## 4. 기술 스택(최소/적합)

### 4.1 Python 런타임
- Python 3.11+

### 4.2 필수 라이브러리
- API 서버(선택): **FastAPI** (엔드포인트로 실행/테스트 편리)
- 데이터 모델: **Pydantic** (Action/State/Observation 스키마 강제)
- Milvus 연동: **pymilvus** (host/port 19530 연결)
  - connect() 참고: https://milvus.io/api-reference/pymilvus/v2.2.x/Connections/connect%28%29.md
- 웹검색 뼈대: **duckduckgo-search** (무료 DuckDuckGo 검색 라이브러리)
  - https://pypi.org/project/duckduckgo-search/
- LLM 클라이언트: 특정 벤더 고정 대신 `LLMClient` 인터페이스로 추상화(추후 교체 용이)


## 5. Tool 정의(MVP)

### 5.1 Tool #1: vector_search (Milvus)

#### 목적
- 사용자의 질의로부터 생성된 임베딩(또는 이미 계산된 vector)을 사용해 Milvus에서 Top-K 유사 문서를 반환

#### Milvus 로컬 실행
- docker-compose 기반 standalone 실행
- Milvus standalone은 기본적으로 **19530 포트로 서비스**
  - https://milvus.io/docs/ko/install_standalone-docker-compose.md

#### 연결
- `connections.connect(host="127.0.0.1", port=19530)` 형태
  - https://milvus.io/api-reference/pymilvus/v2.2.x/Connections/connect%28%29.md

#### 입력 스키마(예시)
- `collection`: string (예: `"docs"`)
- `query_text`: string (MVP에선 text를 받아서 내부에서 임베딩 생성)
- `top_k`: int (기본 5~10)
- `filter`: dict (MVP에선 optional, 사용 안 해도 됨)

#### 출력 스키마(예시)
- `hits`: 리스트
  - `id`, `score`, `text`, `metadata`

> MVP에서는 “임베딩 생성”은 별도 컴포넌트(EmbeddingProvider)로 두고, 우선 더미/간단 모델로 시작 가능.

---

### 5.2 Tool #2: web_search (무료 웹검색)

#### 목적
- 외부 웹 검색 결과(제목/요약/URL)를 Top-K로 반환

#### 구현 선택(뼈대)
- DuckDuckGo 기반 `duckduckgo-search` 사용
  - https://pypi.org/project/duckduckgo-search/
- 키/유료 API 연동은 추후로 미룬다.

#### 입력 스키마(예시)
- `query`: string
- `top_k`: int (기본 5)
- `region`: string (optional)

#### 출력 스키마(예시)
- `results`: 리스트
  - `title`, `snippet`, `url`


## 6. 데이터 모델 정의(Pydantic 기준)

### 6.1 Action
- `type`: `"CALL_TOOL" | "WRITE_NOTE" | "SYNTHESIZE" | "STOP"`
- `tool_name?`: `"vector_search" | "web_search"`
- `tool_args?`: dict
- `note?`: string

### 6.2 Observation
- `seq`: int
- `step_id`: string (예: `"turn_1"`)
- `kind`: `"TOOL_RESULT" | "NOTE"`
- `tool_name?`
- `tool_args?`
- `payload?`: dict (MVP는 일부만)
- `summary`: string (짧게)

### 6.3 AgentState
- `run_id`: string
- `goal`: string
- `cursor`: int
- `budget`: `{ max_steps: int }`
- `status`: `"RUNNING" | "DONE"`
- `observations`: `Observation[]`
- `final_answer?`: string


## 7. 모듈 설계(파일 구조 예시)

```
agent/
  app.py                  # FastAPI 엔드포인트(선택)
  executor.py             # Executor 루프
  policy.py               # LLM Policy (decide next Action)
  synthesizer.py          # LLM Synthesizer (final answer)
  state_store.py          # State 저장소(메모리/파일)
  models.py               # Pydantic: Action/Observation/AgentState
  tools/
    tool_runtime.py       # Tool dispatch
    vector_search.py      # Milvus client + search
    web_search.py         # DuckDuckGo 검색 wrapper
  config.py               # 환경변수/설정
  embeddings.py           # EmbeddingProvider (MVP는 간단/더미 가능)
docker/
  docker-compose.yml      # Milvus standalone
```


## 8. Executor 동작 정의(MVP 알고리즘)

### 8.1 의사코드
1. 입력 메시지로 goal 정규화(간단히: user_input 그대로 or 짧게 요약)
2. state 생성 및 저장
3. 반복:
   - 하드 게이트 체크: `cursor >= max_steps` 또는 `status == DONE`
   - Policy(LLM) 호출 → Action 1개 획득
   - Action 실행:
     - CALL_TOOL: tool_runtime 호출 → Observation(TOOL_RESULT) append
     - WRITE_NOTE: Observation(NOTE) append
     - SYNTHESIZE: Synthesizer(LLM) 호출 → final_answer 저장, DONE
     - STOP: stop 메시지 저장, DONE
   - cursor += 1
   - state 저장
4. 루프가 예산 종료면 Synthesizer 1회 호출로 마무리


## 9. Milvus 로컬 구성 가이드(최소)

### 9.1 docker-compose 가져오기/실행
- Milvus 공식 문서는 standalone docker-compose 실행을 안내하며, standalone은 19530 포트를 로컬에 서비스한다.
  - https://milvus.io/docs/ko/install_standalone-docker-compose.md
- compose 파일은 릴리즈에서 다운로드 형태로 제공된다.
  - https://milvus.io/docs/ko/configure-docker.md

예시:
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

### 9.2 Python 연결
```python
from pymilvus import connections
connections.connect(alias="default", host="127.0.0.1", port="19530")
```


## 10. web_search 뼈대 가이드(무료)

- `duckduckgo-search`는 DuckDuckGo 기반 검색을 제공하는 PyPI 패키지로, pip 설치 후 사용 가능하다.
  - https://pypi.org/project/duckduckgo-search/

```bash
pip install duckduckgo-search
```


## 11. API(선택) – FastAPI 최소 엔드포인트
- `POST /agent/run`
  - 입력: `{ "message": "..." }`
  - 출력: `{ "answer": "...", "run_id": "..." }`
- 개발 편의상 run_id로 state 조회 엔드포인트를 추가해도 됨(옵션):
  - `GET /agent/state/{run_id}`


## 12. MVP에서 결정해야 하는 기본값
- `budget.max_steps`: 3~5 권장(루프 폭주 방지)
- `vector_search.top_k`: 5 또는 10 고정(Policy가 바꾸게 하지 않음)
- `web_search.top_k`: 5 고정
- Policy가 tool 선택 기준:
  - 일반 지식 질문: web_search 또는 SYNTHESIZE(검색 없이)
  - 내부 문서 지식: vector_search 우선
