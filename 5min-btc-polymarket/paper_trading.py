"""Paper-only position lifecycle and two independent exit simulations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class PaperPosition:
    market_slug: str
    side: str
    entry_time: datetime
    entry_price: float
    stake_usd: float = 1.0
    model_probability: float = 0.0
    entry_edge: float = 0.0
    entry_confidence: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def shares(self) -> float:
        return self.stake_usd / self.entry_price

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl(self) -> Optional[float]:
        return None if self.exit_price is None else self.shares * self.exit_price - self.stake_usd

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("entry_time", "exit_time"):
            if value[key] is not None:
                value[key] = value[key].isoformat().replace("+00:00", "Z")
        value["shares"] = self.shares
        value["pnl"] = self.pnl
        return value


@dataclass(frozen=True)
class RepricingConfig:
    take_profit_price_delta: float = 0.10
    stop_loss_price_delta: float = -0.08
    max_hold_seconds: float = 180.0
    confidence_drop: float = 0.20
    opposite_probability: float = 0.65


def settle(position: PaperPosition, official_winner: str, settled_at: datetime) -> PaperPosition:
    price = 1.0 if position.side.upper() == official_winner.upper() else 0.0
    return replace(position, exit_time=settled_at, exit_price=price, exit_reason="official_settlement")


def evaluate_repricing_exit(
    position: PaperPosition, now: datetime, current_bid: float,
    selected_side_probability: float, confidence: float,
    opposite_probability: float, current_edge: float,
    config: RepricingConfig = RepricingConfig(),
) -> PaperPosition:
    if not position.is_open:
        return position
    delta = current_bid - position.entry_price
    elapsed = (now - position.entry_time).total_seconds()
    reason = None
    if delta >= config.take_profit_price_delta:
        reason = "take_profit"
    elif delta <= config.stop_loss_price_delta:
        reason = "stop_loss"
    elif current_edge <= 0:
        reason = "edge_disappeared"
    elif confidence <= max(0.0, position.entry_confidence - config.confidence_drop):
        reason = "confidence_drop"
    elif opposite_probability >= config.opposite_probability and opposite_probability > selected_side_probability:
        reason = "strong_opposite_signal"
    elif elapsed >= config.max_hold_seconds:
        reason = "max_hold"
    return position if reason is None else replace(position, exit_time=now, exit_price=current_bid, exit_reason=reason)
