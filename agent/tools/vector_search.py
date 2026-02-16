"""Milvus 벡터 검색 Tool.

pymilvus를 사용하여 Milvus standalone에 연결하고 유사도 검색을 수행한다.
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import Collection, connections, utility

from agent.config import settings
from agent.embeddings import get_embedding_provider
from agent.models import VectorSearchHit, VectorSearchInput, VectorSearchOutput

logger = logging.getLogger(__name__)

_connected = False


def _ensure_connection() -> None:
    """Milvus 연결을 보장한다."""
    global _connected
    if not _connected:
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=str(settings.milvus_port),
        )
        _connected = True
        logger.info(
            "Milvus 연결 완료: %s:%s", settings.milvus_host, settings.milvus_port
        )


def vector_search(args: dict[str, Any]) -> dict[str, Any]:
    """Milvus에서 벡터 유사도 검색을 수행한다.

    Args:
        args: VectorSearchInput에 맞는 딕셔너리
            - collection: 컬렉션 이름
            - query_text: 검색 쿼리 텍스트
            - top_k: 반환할 결과 수
            - filter: 필터 조건 (optional)

    Returns:
        VectorSearchOutput을 딕셔너리로 변환한 결과
    """
    params = VectorSearchInput(**args)
    _ensure_connection()

    # 컬렉션 존재 여부 확인
    if not utility.has_collection(params.collection):
        logger.warning("컬렉션 '%s'이 존재하지 않습니다.", params.collection)
        return VectorSearchOutput(hits=[]).model_dump()

    # 임베딩 생성
    provider = get_embedding_provider()
    query_vector = provider.embed(params.query_text)

    # 컬렉션 로드 및 검색
    collection = Collection(params.collection)
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    # 필터 표현식
    expr = ""
    if params.filter:
        conditions = []
        for key, value in params.filter.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            else:
                conditions.append(f"{key} == {value}")
        expr = " and ".join(conditions)

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=params.top_k,
        expr=expr if expr else None,
        output_fields=["text", "metadata"],
    )

    # 결과 변환
    hits: list[VectorSearchHit] = []
    for result in results[0]:
        hit = VectorSearchHit(
            id=str(result.id),
            score=float(result.score),
            text=result.entity.get("text", ""),
            metadata=result.entity.get("metadata", {}),
        )
        hits.append(hit)

    output = VectorSearchOutput(hits=hits)
    logger.info(
        "vector_search 완료: collection=%s, hits=%d",
        params.collection,
        len(hits),
    )
    return output.model_dump()

