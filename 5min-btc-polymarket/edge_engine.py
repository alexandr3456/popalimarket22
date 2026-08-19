"""Symmetric UP/DOWN effective-cost, edge and safety-gate engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

from probability_engine import ProbabilityEstimate


@dataclass(frozen=True)
class StrategyConfig:
    min_probability: float = 0.60
    min_edge: float = 0.07
    min_confidence: float = 0.65
    min_seconds_left: float = 45
    max_seconds_left: float = 180
    max_spread: float = 0.04
    min_liquidity: float = 10.0
    slippage_buffer: float = 0.01
    estimated_taker_fee: float = 0.01
    daily_trade_limit: int = 10
    max_entry_price: float = 0.85

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: values[k] for k in cls.__dataclass_fields__ if k in values})


@dataclass(frozen=True)
class SideQuote:
    bid: Optional[float]
    ask: Optional[float]
    liquidity: Optional[float]

    @property
    def spread(self) -> Optional[float]:
        return None if self.bid is None or self.ask is None else max(0.0, self.ask - self.bid)


@dataclass(frozen=True)
class DecisionContext:
    seconds_left: float
    has_open_position: bool = False
    market_already_traded: bool = False
    daily_trade_count: int = 0


@dataclass(frozen=True)
class TradeDecision:
    trade: bool
    side: Optional[str]
    model_probability: Optional[float]
    effective_cost: Optional[float]
    edge: Optional[float]
    up_edge: Optional[float]
    down_edge: Optional[float]
    confidence: float
    rejection_reasons: tuple[str, ...]
    gates: dict[str, bool]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["rejection_reasons"] = list(self.rejection_reasons)
        return value


def _side_result(side: str, probability: float, quote: SideQuote, cfg: StrategyConfig) -> tuple[Optional[float], Optional[float]]:
    if quote.ask is None:
        return None, None
    cost = quote.ask + cfg.estimated_taker_fee + cfg.slippage_buffer
    return cost, probability - cost


def decide_trade(estimate: ProbabilityEstimate, up: SideQuote, down: SideQuote, context: DecisionContext, config: StrategyConfig = StrategyConfig()) -> TradeDecision:
    up_cost, up_edge = _side_result("UP", estimate.p_up, up, config)
    down_cost, down_edge = _side_result("DOWN", estimate.p_down, down, config)
    candidates = []
    if up_edge is not None:
        candidates.append((up_edge, "UP", estimate.p_up, up_cost, up))
    if down_edge is not None:
        candidates.append((down_edge, "DOWN", estimate.p_down, down_cost, down))
    if not candidates:
        return TradeDecision(False, None, None, None, None, up_edge, down_edge, estimate.confidence, ("missing asks",), {"quotes_available": False})
    edge, side, probability, cost, quote = max(candidates, key=lambda row: row[0])
    spread = quote.spread
    gates = {
        "quotes_available": quote.ask is not None and spread is not None and quote.liquidity is not None,
        "probability": probability >= config.min_probability,
        "edge": edge >= config.min_edge,
        "entry_price": quote.ask is not None and quote.ask <= config.max_entry_price,
        "spread": spread is not None and spread <= config.max_spread,
        "liquidity": quote.liquidity is not None and quote.liquidity >= config.min_liquidity,
        "time_window": config.min_seconds_left <= context.seconds_left <= config.max_seconds_left,
        "confidence": estimate.confidence >= config.min_confidence,
        "no_open_position": not context.has_open_position,
        "market_not_traded": not context.market_already_traded,
        "daily_limit": context.daily_trade_count < config.daily_trade_limit,
    }
    reasons = tuple(name for name, passed in gates.items() if not passed)
    return TradeDecision(not reasons, side if not reasons else None, probability, cost, edge, up_edge, down_edge, estimate.confidence, reasons, gates)
