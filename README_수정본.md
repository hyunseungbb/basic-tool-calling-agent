# AI Agent MVP - Docker 없이 SQLite로 실행하기

## 변경 사항 요약
| 항목 | 기존 | 변경 |
|------|------|------|
| 벡터DB | Milvus (Docker 필요) | SQLite FTS5 (로컬 파일) |
| 임베딩 | sentence-transformers | 불필요 (FTS로 대체) |
| CORS | 없음 | React 프론트 연동 추가 |

## 전체 구조
```
[React 프론트엔드 :3000] ←→ [FastAPI 백엔드 :8000] ←→ [SQLite DB]
  tool-agent-frontend              basic-tool-calling-agent    data/knowledge.db
```

---

## 백엔드 설치 및 실행

### 1. 패키지 설치
```powershell
cd basic-tool-calling-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. .env 파일 설정
```powershell
copy .env.example .env
# .env 열어서 ANTHROPIC_API_KEY 입력
```

### 3. 데이터 적재 (최초 1회)
```powershell
python -m agent.tools.ingest --excel report__9_.xlsx
```

### 4. 백엔드 서버 실행
```powershell
uvicorn agent.app:app --host 0.0.0.0 --port 8000
```

---

## 프론트엔드 설치 및 실행

### 1. Node.js 설치 (없는 경우)
https://nodejs.org 에서 LTS 버전 설치

### 2. 패키지 설치 및 실행
```powershell
cd tool-agent-frontend
npm install
npm run dev
```

브라우저에서 http://localhost:3000 접속

---

## 질문 예시
- 상하이 거래소 시가총액은?
- Americas 지역 시가총액 TOP 5
- 2026년 2월 아시아 거래량 비교해줘
- 전월 대비 가장 많이 오른 거래소는?
- IPO 자본조달 현황 알려줘
