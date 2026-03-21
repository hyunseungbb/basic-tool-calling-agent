"""SQLite FTS5 기반 문서 검색 Tool (Milvus 대체).

Docker 없이 로컬 SQLite DB에서 전문 검색(Full-Text Search)을 수행한다.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from agent.config import settings
from agent.models import VectorSearchHit, VectorSearchInput, VectorSearchOutput

logger = logging.getLogger(__name__)


# 한국어 → 영어 동의어 매핑 (FTS 검색 확장용)
_SYNONYMS: dict[str, list[str]] = {
    "시가총액": ["Market", "Capitalisation"],
    "거래량": ["Trading", "Volume"],
    "상장": ["listed", "companies"],
    "거래소": ["Exchange"],
    "주식": ["Equity"],
    "자본": ["Capital"],
    "조달": ["Raising"],
    "IPO": ["IPO"],
    "채권": ["Bond"],
    "파생": ["Derivatives"],
}

# 한국어 월 → 영어 월 매핑
_MONTH_MAP: dict[str, str] = {
    "1월": "Jan", "2월": "Feb", "3월": "Mar", "4월": "Apr",
    "5월": "May", "6월": "Jun", "7월": "Jul", "8월": "Aug",
    "9월": "Sep", "10월": "Oct", "11월": "Nov", "12월": "Dec",
}


def _build_fts_query(raw_query: str) -> str:
    """사용자 쿼리를 FTS5 OR 검색 쿼리로 변환한다.

    - 토큰을 OR로 연결하여 부분 매칭 허용
    - 한국어 키워드를 영어 동의어로 확장
    - 한국어 월 표현을 영어로 변환
    """
    import re

    tokens: list[str] = []

    # 한국어 월 변환 (예: "3월" → "Mar")
    for kr_month, en_month in _MONTH_MAP.items():
        if kr_month in raw_query:
            tokens.append(en_month)

    # 한국어 동의어 확장
    for kr_word, en_words in _SYNONYMS.items():
        if kr_word in raw_query:
            tokens.extend(en_words)

    # 원본 토큰 추가 (숫자, 영문, 한글 단어)
    for tok in re.findall(r"[A-Za-z]+|\d+", raw_query):
        if len(tok) >= 2:
            tokens.append(tok)

    # 중복 제거 후 OR 연결
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            unique.append(t)

    if not unique:
        return raw_query.replace('"', '""')

    return " OR ".join(unique)


def _get_conn() -> sqlite3.Connection:
    """SQLite 연결을 반환한다."""
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def vector_search(args: dict[str, Any]) -> dict[str, Any]:
    """SQLite FTS5로 문서 검색을 수행한다.

    Args:
        args: VectorSearchInput에 맞는 딕셔너리
            - collection: 테이블 이름 (기본 'docs')
            - query_text: 검색 쿼리 텍스트
            - top_k: 반환할 결과 수
            - filter: 필터 조건 (optional, 예: {"region": "Americas"})

    Returns:
        VectorSearchOutput을 딕셔너리로 변환한 결과
    """
    params = VectorSearchInput(**args)

    try:
        conn = _get_conn()
        cursor = conn.cursor()

        # FTS5 전문 검색 쿼리
        table_fts = f"{params.collection}_fts"

        # 필터 조건 구성
        filter_clause = ""
        filter_vals: list[Any] = []
        if params.filter:
            conditions = []
            for key, value in params.filter.items():
                conditions.append(f"d.{key} = ?")
                filter_vals.append(value)
            filter_clause = "AND " + " AND ".join(conditions)

        sql = f"""
            SELECT d.id, d.text, d.region, d.exchange, d.indicator,
                   d.year, d.month, d.value, d.currency,
                   bm25({table_fts}) AS score
            FROM {table_fts}
            JOIN {params.collection} d ON d.id = {table_fts}.rowid
            WHERE {table_fts} MATCH ?
            {filter_clause}
            ORDER BY score
            LIMIT ?
        """

        # FTS5 쿼리: 토큰을 OR로 연결하여 부분 매칭 허용
        safe_query = _build_fts_query(params.query_text)
        query_vals = [safe_query] + filter_vals + [params.top_k]

        rows = cursor.execute(sql, query_vals).fetchall()

        hits: list[VectorSearchHit] = []
        for row in rows:
            text = row["text"]
            meta = {
                "region": row["region"],
                "exchange": row["exchange"],
                "indicator": row["indicator"],
                "year": row["year"],
                "month": row["month"],
                "value": row["value"],
                "currency": row["currency"],
            }
            hit = VectorSearchHit(
                id=str(row["id"]),
                score=float(row["score"]) * -1,  # bm25는 음수가 좋은 값
                text=text,
                metadata=meta,
            )
            hits.append(hit)

        conn.close()
        output = VectorSearchOutput(hits=hits)
        logger.info("vector_search(SQLite) 완료: query='%s', hits=%d", params.query_text, len(hits))
        return output.model_dump()

    except Exception as e:
        logger.error("vector_search(SQLite) 오류: %s", str(e))
        # FTS 테이블 없으면 LIKE 검색으로 폴백
        return _fallback_like_search(params)


def _fallback_like_search(params: VectorSearchInput) -> dict[str, Any]:
    """FTS 실패 시 LIKE 검색으로 폴백."""
    try:
        conn = _get_conn()
        keywords = params.query_text.split()
        conditions = " OR ".join([f"text LIKE ?" for _ in keywords])
        vals = [f"%{kw}%" for kw in keywords] + [params.top_k]

        rows = conn.execute(
            f"SELECT id, text, region, exchange, indicator, year, month, value, currency "
            f"FROM {params.collection} WHERE {conditions} LIMIT ?",
            vals,
        ).fetchall()

        hits = [
            VectorSearchHit(
                id=str(r["rowid"]),
                score=1.0,
                text=r["text"],
                metadata={"region": r["region"], "exchange": r["exchange"],
                          "indicator": r["indicator"], "year": r["year"],
                          "month": r["month"], "value": r["value"]},
            )
            for r in rows
        ]
        conn.close()
        return VectorSearchOutput(hits=hits).model_dump()
    except Exception as e2:
        logger.error("fallback 검색도 실패: %s", str(e2))
        return VectorSearchOutput(hits=[]).model_dump()
