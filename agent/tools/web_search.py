"""DuckDuckGo 기반 웹 검색 Tool.

duckduckgo-search 라이브러리를 사용하여 무료 웹 검색을 수행한다.
"""

from __future__ import annotations

import logging
from typing import Any

from duckduckgo_search import DDGS

from agent.models import WebSearchInput, WebSearchOutput, WebSearchResult

logger = logging.getLogger(__name__)


def web_search(args: dict[str, Any]) -> dict[str, Any]:
    """DuckDuckGo를 사용하여 웹 검색을 수행한다.

    Args:
        args: WebSearchInput에 맞는 딕셔너리
            - query: 검색 쿼리
            - top_k: 반환할 결과 수
            - region: 검색 지역 (optional)

    Returns:
        WebSearchOutput을 딕셔너리로 변환한 결과
    """
    params = WebSearchInput(**args)

    try:
        ddgs = DDGS()
        raw_results = ddgs.text(
            keywords=params.query,
            region=params.region or "wt-wt",
            max_results=params.top_k,
        )

        results: list[WebSearchResult] = []
        for item in raw_results:
            result = WebSearchResult(
                title=item.get("title", ""),
                snippet=item.get("body", ""),
                url=item.get("href", ""),
            )
            results.append(result)

        output = WebSearchOutput(results=results)
        logger.info("web_search 완료: query='%s', results=%d", params.query, len(results))
        return output.model_dump()

    except Exception as e:
        logger.error("web_search 오류: %s", str(e))
        return WebSearchOutput(results=[]).model_dump()

