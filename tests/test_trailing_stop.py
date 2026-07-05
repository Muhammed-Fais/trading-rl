import pytest

from trading_rl.agents.evaluate import _active_trailing_stop


def test_percent_trailing_stop_uses_configured_value() -> None:
    stop = _active_trailing_stop(
        mode="percent",
        trailing_stop=0.15,
        true_ranges=[0.01, 0.02],
        atr_window=2,
        atr_multiplier=3.0,
        min_trailing_stop=0.06,
        max_trailing_stop=0.20,
    )

    assert stop == pytest.approx(0.15)


def test_atr_trailing_stop_uses_recent_true_range_with_clamps() -> None:
    stop = _active_trailing_stop(
        mode="atr",
        trailing_stop=0.15,
        true_ranges=[0.01, 0.02, 0.03],
        atr_window=2,
        atr_multiplier=3.0,
        min_trailing_stop=0.06,
        max_trailing_stop=0.20,
    )

    assert stop == pytest.approx(0.075)


def test_atr_trailing_stop_respects_minimum() -> None:
    stop = _active_trailing_stop(
        mode="atr",
        trailing_stop=0.15,
        true_ranges=[0.001, 0.002],
        atr_window=2,
        atr_multiplier=3.0,
        min_trailing_stop=0.06,
        max_trailing_stop=0.20,
    )

    assert stop == pytest.approx(0.06)
