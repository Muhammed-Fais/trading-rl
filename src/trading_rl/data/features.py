from __future__ import annotations

import numpy as np
import pandas as pd


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
    out["hl_range"] = (high - low) / close.replace(0, np.nan)
    out["volume_change"] = np.log(volume.replace(0, np.nan) / volume.shift(1).replace(0, np.nan))
    out["volatility_24"] = out["log_return"].rolling(24, min_periods=2).std()
    out["ma_ratio_12"] = close / close.rolling(12, min_periods=2).mean() - 1.0
    out["ma_ratio_48"] = close / close.rolling(48, min_periods=2).mean() - 1.0
    out["rsi_14"] = _rsi(close, 14)

    feature_cols = [
        "log_return",
        "hl_range",
        "volume_change",
        "volatility_24",
        "ma_ratio_12",
        "ma_ratio_48",
        "rsi_14",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ignore",
    }
    return [col for col in df.columns if col not in excluded and df[col].dtype.kind in "fiu"]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period, min_periods=2).mean()
    loss = (-diff.clip(upper=0)).rolling(period, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0
