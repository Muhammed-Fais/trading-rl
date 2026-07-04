from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from trading_rl.data.features import add_basic_features
from trading_rl.data.schemas import KLINE_COLUMNS
from trading_rl.data.storage import write_parquet

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    limit: int = 1000,
    request_sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    start_ms = _to_millis(start)
    end_ms = _to_millis(end)

    with httpx.Client(base_url=BINANCE_SPOT_BASE_URL, timeout=30.0) as client:
        while start_ms < end_ms:
            response = client.get(
                KLINES_ENDPOINT,
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            next_start = int(batch[-1][6]) + 1
            if next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(request_sleep_seconds)

    return normalize_klines(rows)


def normalize_klines(rows: list[list[object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.drop(columns=["ignore"]).drop_duplicates("open_time").sort_values("open_time")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _to_millis(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance spot klines.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-output")
    args = parser.parse_args()

    raw = fetch_klines(
        symbol=args.symbol,
        interval=args.interval,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    output = write_parquet(raw, args.output)
    print(f"Wrote {len(raw)} raw rows to {output}")

    if args.features_output:
        features = add_basic_features(raw)
        features_output = write_parquet(features, Path(args.features_output))
        print(f"Wrote {len(features)} feature rows to {features_output}")


if __name__ == "__main__":
    main()
