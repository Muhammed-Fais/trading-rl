from __future__ import annotations

from trading_rl.data.schemas import ChronologicalSplit


def chronological_split_indices(
    n_rows: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    if n_rows < 3:
        raise ValueError("Need at least three rows to create train/validation/test splits")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave room for test data")

    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    return ChronologicalSplit(
        train=slice(0, train_end),
        validation=slice(train_end, validation_end),
        test=slice(validation_end, n_rows),
    )
