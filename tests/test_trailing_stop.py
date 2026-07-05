import pytest

from trading_rl.agents.evaluate import (
    _active_trailing_stop,
    _confirmed_recovery_reset_allowed,
    _momentum_participation_exposure,
    _recovery_reentry_exposure,
    trend_risk_policy,
)


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
        participation_mode="momentum",
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
        participation_mode="momentum",
        momentum_window=2,
        momentum_threshold=0.05,
        max_exposure=1.0,
    )

    assert exposure == pytest.approx(0.0)


def test_always_participation_uses_floor_without_momentum_condition() -> None:
    exposure = _momentum_participation_exposure(
        prices=[100.0],
        short_ma=120.0,
        participation_floor=0.25,
        participation_mode="always",
        momentum_window=72,
        momentum_threshold=0.05,
        max_exposure=0.20,
    )

    assert exposure == pytest.approx(0.20)


def test_drawdown_cooldown_reset_allows_reentry_after_risk_stop() -> None:
    policy = trend_risk_policy(
        short_window=2,
        long_window=3,
        realized_window=3,
        max_portfolio_drawdown=0.10,
        participation_floor=0.2,
        participation_mode="always",
        cooldown_steps=2,
        reset_peak_after_drawdown=True,
    )
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    values = [10000.0, 11500.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
    actions = []

    for price, value in zip(prices, values, strict=True):
        actions.append(
            policy(
                _obs=None,
                info={
                    "price": price,
                    "high": price,
                    "low": price,
                    "portfolio_value": value,
                    "action_mode": "continuous",
                },
            )
        )

    assert float(actions[-1][0]) > 0.0


def test_recovery_reentry_requires_momentum_and_trend_confirmation() -> None:
    exposure = _recovery_reentry_exposure(
        prices=[100.0, 103.0, 108.0],
        short_ma=105.0,
        long_ma=102.0,
        mode="momentum",
        exposure=0.2,
        momentum_window=2,
        momentum_threshold=0.05,
        max_exposure=1.0,
    )

    assert exposure == pytest.approx(0.2)


def test_confirmed_recovery_reset_requires_fresh_high_confirmation() -> None:
    allowed = _confirmed_recovery_reset_allowed(
        prices=[100.0, 103.0, 106.0, 108.0],
        short_ma=106.0,
        long_ma=103.0,
        mode="confirmed_reset",
        confirm_window=4,
        confirm_threshold=0.03,
    )

    assert allowed


def test_confirmed_recovery_reset_rejects_weak_recovery() -> None:
    allowed = _confirmed_recovery_reset_allowed(
        prices=[100.0, 110.0, 104.0, 105.0],
        short_ma=104.0,
        long_ma=103.0,
        mode="confirmed_reset",
        confirm_window=4,
        confirm_threshold=0.03,
    )

    assert not allowed


def test_recovery_reentry_allows_capped_reentry_after_drawdown_stop() -> None:
    policy = trend_risk_policy(
        short_window=2,
        long_window=3,
        realized_window=3,
        max_portfolio_drawdown=0.10,
        participation_floor=0.2,
        participation_mode="always",
        cooldown_steps=2,
        recovery_reentry_mode="momentum",
        recovery_exposure=0.15,
        recovery_momentum_window=2,
        recovery_momentum_threshold=0.01,
        recovery_drawdown_buffer=0.05,
    )
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 109.0]
    values = [10000.0, 11500.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
    actions = []

    for price, value in zip(prices, values, strict=True):
        actions.append(
            policy(
                _obs=None,
                info={
                    "price": price,
                    "high": price,
                    "low": price,
                    "portfolio_value": value,
                    "action_mode": "continuous",
                },
            )
        )

    assert float(actions[-1][0]) == pytest.approx(0.15)


def test_confirmed_recovery_reset_allows_normal_reentry_after_drawdown_stop() -> None:
    policy = trend_risk_policy(
        short_window=2,
        long_window=3,
        realized_window=3,
        target_hourly_vol=1.0,
        max_portfolio_drawdown=0.10,
        participation_floor=0.0,
        cooldown_steps=2,
        recovery_reentry_mode="confirmed_reset",
        recovery_confirm_window=3,
        recovery_confirm_threshold=0.03,
    )
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 109.0]
    values = [10000.0, 11500.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
    actions = []

    for price, value in zip(prices, values, strict=True):
        actions.append(
            policy(
                _obs=None,
                info={
                    "price": price,
                    "high": price,
                    "low": price,
                    "portfolio_value": value,
                    "action_mode": "continuous",
                },
            )
        )

    assert float(actions[-1][0]) > 0.15
