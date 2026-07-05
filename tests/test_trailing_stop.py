import pytest

from trading_rl.agents.evaluate import _active_trailing_stop, _momentum_participation_exposure


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


def test_momentum_participation_requires_positive_momentum() -> None:
    exposure = _momentum_participation_exposure(
        prices=[100.0, 103.0, 108.0],
        short_ma=104.0,
        participation_floor=0.25,
        momentum_window=2,
        momentum_threshold=0.05,
        max_exposure=1.0,
    )

    assert exposure == pytest.approx(0.25)


def test_momentum_participation_stays_off_without_floor() -> None:
    exposure = _momentum_participation_exposure(
        prices=[100.0, 103.0, 108.0],
        short_ma=104.0,
        participation_floor=0.0,
        momentum_window=2,
        momentum_threshold=0.05,
        max_exposure=1.0,
    )

    assert exposure == pytest.approx(0.0)
