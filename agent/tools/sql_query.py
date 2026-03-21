"""SQL 쿼리 실행 Tool.

Policy(LLM)가 생성한 SQL을 SQLite에서 실행하고 결과를 반환한다.
읽기 전용(SELECT)만 허용하며, RANK(), 일평균 계산 등을 지원한다.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from agent.config import settings

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    """SQLite 연결을 반환한다."""
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_sql(sql: str) -> None:
    """SQL이 읽기 전용(SELECT)인지 검증한다."""
    stripped = sql.strip().upper()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH"]
    first_word = stripped.split()[0] if stripped.split() else ""
    if first_word in forbidden:
        raise ValueError(f"읽기 전용 쿼리만 허용됩니다. '{first_word}'는 사용할 수 없습니다.")


def sql_query(args: dict[str, Any]) -> dict[str, Any]:
    """LLM이 생성한 SQL 쿼리를 실행한다.

    Args:
        args: 딕셔너리
            - sql: 실행할 SELECT 쿼리
            - params: 바인딩 파라미터 리스트 (optional)

    Returns:
        {"columns": [...], "rows": [[...], ...], "row_count": int}
    """
    sql = args.get("sql", "")
    params = args.get("params", [])

    if not sql.strip():
        return {"columns": [], "rows": [], "row_count": 0, "error": "빈 SQL 쿼리"}

    try:
        _validate_sql(sql)
        conn = _get_conn()
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchall()]
        conn.close()

        logger.info("sql_query 완료: %d행 반환", len(rows))
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }

    except Exception as e:
        logger.error("sql_query 오류: %s", str(e))
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": str(e),
        }


def get_schema_description() -> str:
    """Policy 프롬프트에 제공할 DB 스키마 설명을 반환한다."""
    return """\
## SQLite DB 스키마

### market_data 테이블 (핵심 데이터)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 자동 증가 |
| region | TEXT | 지역 (Americas, Asia - Pacific, Europe - Africa - Middle East) |
| exchange_name | TEXT | 거래소명 (NYSE, Korea Exchange, 등) |
| indicator | TEXT | 지표명 (아래 목록 참조) |
| year | INTEGER | 연도 (2025, 2026) |
| month | TEXT | 월 (Jan, Feb, ..., Dec) |
| value | REAL | 원시 값 |
| value_usd | REAL | USD 환산 값 (Monetary: 백만 USD 단위) |
| currency | TEXT | 통화 |
| nominal | REAL | 단위 배수 (1, 1000, 1000000) |
| data_type | TEXT | Monetary / Full Number / Decimal / Percentage |
| pct_mtm | REAL | 전월대비 변화율 (0.05 = 5%) |
| pct_yty | REAL | 전년대비 변화율 |
| agg_type | TEXT | Stock(잔액) / Flow(유량) |

### trading_days 테이블 (거래일수)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| exchange_name | TEXT | 거래소명 |
| year | INTEGER | 연도 |
| month | TEXT | 월 |
| days | INTEGER | 해당 월 거래일수 |

### 주요 indicator 목록
- Market Capitalisation: 시가총액
- Value traded (EOB Total): 거래대금(총합)
- Number of listed companies (Total): 상장회사 수
- Capital raised through IPO (Total): IPO 자본조달
- Capital raised (Total): 총 자본조달
- Number of trades (EOB): 거래건수
- Share turnover velocity: 회전율
- Number of new listings (Total): 신규상장 수

모든 indicator는 "Total Equity Market - " 접두사가 붙습니다.
예: "Total Equity Market - Market Capitalisation"

### 일평균 계산 예시
```sql
SELECT m.exchange_name,
       m.value_usd AS total_value,
       t.days AS trading_days,
       m.value_usd / t.days AS daily_avg
FROM market_data m
JOIN trading_days t
  ON m.exchange_name = t.exchange_name
  AND m.year = t.year AND m.month = t.month
WHERE m.indicator = 'Total Equity Market - Value traded (EOB Total)'
  AND m.year = 2025 AND m.month = 'Dec'
ORDER BY daily_avg DESC;
```

### 순위 계산 예시
```sql
SELECT exchange_name, value_usd,
       RANK() OVER (ORDER BY value_usd DESC) AS rank
FROM market_data
WHERE indicator = 'Total Equity Market - Market Capitalisation'
  AND year = 2025 AND month = 'Dec';
```

### 일평균 + 순위 복합 예시
```sql
SELECT m.exchange_name,
       m.value_usd / t.days AS daily_avg,
       RANK() OVER (ORDER BY m.value_usd / t.days DESC) AS rank
FROM market_data m
JOIN trading_days t
  ON m.exchange_name = t.exchange_name
  AND m.year = t.year AND m.month = t.month
WHERE m.indicator = 'Total Equity Market - Value traded (EOB Total)'
  AND m.year = 2025 AND m.month = 'Dec'
ORDER BY daily_avg DESC;
```

### 월 표기: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
### 현재 데이터 범위: 2020년 Jan ~ 2026년 Feb (연도별 전체 월 데이터 보유)
"""
