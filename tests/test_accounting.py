import pytest

from trading_rl.envs.accounting import rebalance_to_fraction


def test_buy_rebalance_pays_costs_and_updates_position() -> None:
    state = rebalance_to_fraction(
        cash=1000.0,
        asset_quantity=0.0,
        price=100.0,
        target_fraction=0.5,
        fee_rate=0.001,
        slippage_rate=0.001,
    )

    assert state.cash == 499.0
    assert state.asset_quantity == 5.0
    assert state.portfolio_value == 999.0
    assert state.fee_paid == 0.5
    assert state.slippage_paid == 0.5


def test_sell_rebalance_does_not_short_spot() -> None:
    state = rebalance_to_fraction(
        cash=0.0,
        asset_quantity=1.0,
        price=100.0,
        target_fraction=0.0,
        fee_rate=0.001,
        slippage_rate=0.001,
    )

    assert state.asset_quantity == 0.0
    assert state.cash == pytest.approx(99.8)
    assert state.position_fraction == 0.0
