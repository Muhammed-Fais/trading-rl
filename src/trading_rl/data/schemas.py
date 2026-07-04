from __future__ import annotations

from dataclasses import dataclass

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: slice
    validation: slice
    test: slice
