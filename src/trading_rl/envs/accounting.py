from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    asset_quantity: float
    portfolio_value: float
    position_fraction: float
    turnover: float
    fee_paid: float
    slippage_paid: float


def mark_to_market(cash: float, asset_quantity: float, price: float) -> float:
    return cash + asset_quantity * price


def rebalance_to_fraction(
    cash: float,
    asset_quantity: float,
    price: float,
    target_fraction: float,
    fee_rate: float,
    slippage_rate: float,
    min_trade_notional: float = 0.0,
) -> PortfolioState:
    if price <= 0:
        raise ValueError("price must be positive")
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("spot target_fraction must be in [0, 1]")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")

    before_value = mark_to_market(cash, asset_quantity, price)
    target_asset_value = before_value * target_fraction
    current_asset_value = asset_quantity * price
    trade_notional = target_asset_value - current_asset_value

    if abs(trade_notional) < min_trade_notional:
        after_value = mark_to_market(cash, asset_quantity, price)
        position_fraction = 0.0 if after_value <= 0 else asset_quantity * price / after_value
        return PortfolioState(cash, asset_quantity, after_value, position_fraction, 0.0, 0.0, 0.0)

    cost_rate = fee_rate + slippage_rate
    fee_paid = abs(trade_notional) * fee_rate
    slippage_paid = abs(trade_notional) * slippage_rate

    if trade_notional > 0:
        total_cash_needed = trade_notional * (1.0 + cost_rate)
        if total_cash_needed > cash:
            trade_notional = cash / (1.0 + cost_rate)
            fee_paid = trade_notional * fee_rate
            slippage_paid = trade_notional * slippage_rate
            total_cash_needed = trade_notional + fee_paid + slippage_paid
        cash -= total_cash_needed
        asset_quantity += trade_notional / price
    else:
        sell_notional = min(abs(trade_notional), asset_quantity * price)
        fee_paid = sell_notional * fee_rate
        slippage_paid = sell_notional * slippage_rate
        cash += sell_notional - fee_paid - slippage_paid
        asset_quantity -= sell_notional / price
        trade_notional = -sell_notional

    after_value = mark_to_market(cash, asset_quantity, price)
    position_fraction = 0.0 if after_value <= 0 else asset_quantity * price / after_value
    turnover = 0.0 if before_value <= 0 else abs(trade_notional) / before_value
    return PortfolioState(
        cash=cash,
        asset_quantity=asset_quantity,
        portfolio_value=after_value,
        position_fraction=position_fraction,
        turnover=turnover,
        fee_paid=fee_paid,
        slippage_paid=slippage_paid,
    )
