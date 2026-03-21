"""엑셀 데이터를 SQLite DB에 구조화하여 적재하는 스크립트.

여러 파일을 한 번에 적재할 수 있습니다.

사용법:
    # 단일 파일
    python -m agent.tools.ingest --excel report__9_.xlsx

    # 여러 파일
    python -m agent.tools.ingest --excel "report (1).xlsx" "report (2).xlsx" report__9_.xlsx
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3

import pandas as pd

from agent.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _create_tables(conn: sqlite3.Connection) -> None:
    """테이블을 DROP 후 재생성한다."""
    conn.execute("DROP TABLE IF EXISTS market_data")
    conn.execute("DROP TABLE IF EXISTS trading_days")

    conn.execute("""
        CREATE TABLE market_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            region          TEXT NOT NULL,
            exchange_name   TEXT NOT NULL,
            indicator       TEXT NOT NULL,
            year            INTEGER NOT NULL,
            month           TEXT NOT NULL,
            value           REAL,
            value_usd       REAL,
            currency        TEXT,
            nominal         REAL,
            data_type       TEXT,
            pct_mtm         REAL,
            pct_yty         REAL,
            agg_type        TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE trading_days (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange_name   TEXT NOT NULL,
            year            INTEGER NOT NULL,
            month           TEXT NOT NULL,
            days            INTEGER NOT NULL,
            UNIQUE(exchange_name, year, month)
        )
    """)
    conn.commit()


def _create_indexes(conn: sqlite3.Connection) -> None:
    """인덱스를 생성한다."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_md_exchange ON market_data(exchange_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_md_indicator ON market_data(indicator)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_md_year_month ON market_data(year, month)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_md_region ON market_data(region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_td_exchange ON trading_days(exchange_name, year, month)")
    conn.commit()


def _load_one_file(conn: sqlite3.Connection, excel_path: str) -> tuple[int, int]:
    """엑셀 파일 1개를 읽어 DB에 삽입한다. (market_rows, trading_rows) 건수 반환."""
    logger.info("엑셀 로드 중: %s", excel_path)
    df = pd.read_excel(excel_path, sheet_name="Data")
    df = df.dropna(subset=["Value"])
    logger.info("  로드 완료: %d행", len(df))

    trading_days_indicator = "Total Equity Market - Number of trading days"

    market_rows = []
    trading_rows = []

    for _, row in df.iterrows():
        exchange = str(row.get("ExchangeName", ""))
        indicator = str(row.get("Indicator Name", ""))
        year = int(row.get("Year", 0))
        month = str(row.get("Month", ""))
        value = float(row["Value"]) if pd.notna(row.get("Value")) else None
        nominal = float(row["Nominal"]) if pd.notna(row.get("Nominal")) else 1.0
        region = str(row.get("Region", ""))
        currency = str(row.get("CurrencyName", "USD"))
        data_type = str(row.get("DataType", ""))
        pct_mtm = float(row["% Change (MTM)"]) if pd.notna(row.get("% Change (MTM)")) else None
        pct_yty = float(row["% Change (YTY)"]) if pd.notna(row.get("% Change (YTY)")) else None
        agg_type = str(row.get("AggregationType", ""))

        if indicator == trading_days_indicator:
            if value is not None:
                trading_rows.append((exchange, year, month, int(value)))
            continue

        value_usd = None
        if value is not None and data_type == "Monetary":
            value_usd = value
        elif value is not None:
            value_usd = value

        market_rows.append((
            region, exchange, indicator, year, month,
            value, value_usd, currency, nominal, data_type,
            pct_mtm, pct_yty, agg_type,
        ))

    conn.executemany("""
        INSERT INTO market_data
        (region, exchange_name, indicator, year, month,
         value, value_usd, currency, nominal, data_type,
         pct_mtm, pct_yty, agg_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, market_rows)

    conn.executemany("""
        INSERT OR REPLACE INTO trading_days
        (exchange_name, year, month, days)
        VALUES (?, ?, ?, ?)
    """, trading_rows)

    conn.commit()
    logger.info("  삽입 완료: market_data=%d행, trading_days=%d행", len(market_rows), len(trading_rows))
    return len(market_rows), len(trading_rows)


def ingest_excel(excel_paths: list[str], db_path: str | None = None) -> None:
    """여러 엑셀 파일을 SQLite DB에 구조화하여 적재한다."""
    db_path = db_path or settings.sqlite_path
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    _create_tables(conn)

    total_market = 0
    total_trading = 0

    for path in excel_paths:
        m, t = _load_one_file(conn, path)
        total_market += m
        total_trading += t

    _create_indexes(conn)
    conn.close()

    logger.info(
        "전체 적재 완료: %d개 파일 → market_data=%d행, trading_days=%d행 → %s",
        len(excel_paths), total_market, total_trading, db_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="엑셀 → SQLite 적재")
    parser.add_argument("--excel", nargs="+", required=True, help="엑셀 파일 경로 (여러 개 가능)")
    parser.add_argument("--db", default=None, help="SQLite DB 경로 (기본: .env의 SQLITE_PATH)")
    args = parser.parse_args()
    ingest_excel(args.excel, args.db)
