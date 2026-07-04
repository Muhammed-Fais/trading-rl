from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "return_3",
    "return_12",
    "return_24",
    "hl_range",
    "atr_14",
    "volume_change",
    "volume_zscore_48",
    "volatility_24",
    "volatility_168",
    "ma_ratio_12",
    "ma_ratio_48",
    "ma_ratio_168",
    "rsi_14",
    "taker_buy_ratio",
    "trade_count_zscore_48",
]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    out["log_return"] = np.log(close / close.shift(1))
    out["return_3"] = close.pct_change(3)
    out["return_12"] = close.pct_change(12)
    out["return_24"] = close.pct_change(24)
    out["hl_range"] = (high - low) / close.replace(0, np.nan)
    out["atr_14"] = _atr(out, 14) / close.replace(0, np.nan)
    out["volume_change"] = np.log(volume.replace(0, np.nan) / volume.shift(1).replace(0, np.nan))
    out["volume_zscore_48"] = _zscore(volume, 48)
    out["volatility_24"] = out["log_return"].rolling(24, min_periods=2).std()
    out["volatility_168"] = out["log_return"].rolling(168, min_periods=2).std()
    out["ma_ratio_12"] = close / close.rolling(12, min_periods=2).mean() - 1.0
    out["ma_ratio_48"] = close / close.rolling(48, min_periods=2).mean() - 1.0
    out["ma_ratio_168"] = close / close.rolling(168, min_periods=2).mean() - 1.0
    out["rsi_14"] = _rsi(close, 14)

    if "taker_buy_base_volume" in out.columns:
        out["taker_buy_ratio"] = (
            out["taker_buy_base_volume"].astype(float) / volume.replace(0, np.nan)
        )
    else:
        out["taker_buy_ratio"] = 0.5
    if "number_of_trades" in out.columns:
        out["trade_count_zscore_48"] = _zscore(out["number_of_trades"].astype(float), 48)
    else:
        out["trade_count_zscore_48"] = 0.0

    feature_cols = feature_columns(out)
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns and df[col].dtype.kind in "fiu"]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period, min_periods=2).mean()
    loss = (-diff.clip(upper=0)).rolling(period, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=2).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=2).mean()
    rolling_std = series.rolling(window, min_periods=2).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)
