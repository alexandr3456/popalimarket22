"""Independent BTC probability model. Polymarket prices are intentionally absent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Mapping, Optional

from feature_engine import MarketFeatures


@dataclass(frozen=True)
class ProbabilityEstimate:
    p_up: float
    p_down: float
    confidence: float
    disagreement_penalty: float
    raw_score: float
    top_signals: tuple[str, ...]
    mode: str = "rule_based"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["top_signals"] = list(self.top_signals)
        return value


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _sign(value: Optional[float], epsilon: float = 1e-7) -> int:
    if value is None or abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else -1


def rule_based_probability(features: MarketFeatures) -> ProbabilityEstimate:
    vol = max(features.realized_volatility_60s or 0.00015, 0.00005)
    distance_z = _clip(features.distance_from_market_open_pct / vol, -4.0, 4.0)
    time_weight = 0.7 + 1.3 * (1.0 - features.seconds_left / 300.0)
    components: list[tuple[str, float]] = [("distance from window open", 0.65 * distance_z * time_weight)]
    momentum_values = [features.return_10s, features.return_20s, features.return_30s, features.return_60s]
    available = [x for x in momentum_values if x is not None]
    momentum = fsum(available) / len(available) if available else 0.0
    components.append(("multi-horizon momentum", 0.45 * _clip(momentum / vol, -3, 3)))
    components.append(("momentum acceleration", 0.18 * _clip((features.momentum_acceleration or 0) / vol, -2, 2)))
    components.append(("trade volume imbalance", 0.22 * (features.volume_imbalance or 0)))
    components.append(("BTC book imbalance", 0.22 * (features.order_book_imbalance or 0)))
    signs = [_sign(x) for x in available]
    nonzero = [s for s in signs if s]
    agreement = abs(sum(nonzero)) / len(nonzero) if nonzero else 0.0
    disagreement = 1.0 - agreement
    raw = sum(value for _, value in components)
    p_up = _clip(1.0 / (1.0 + exp(-raw)), 0.01, 0.99)
    volatility_penalty = _clip((vol - 0.0008) / 0.0025, 0.0, 0.35)
    missing_optional = sum(x is None for x in (features.volume_imbalance, features.order_book_imbalance))
    data_penalty = 0.04 * missing_optional
    confidence = _clip(0.48 + 0.16 * min(abs(raw), 2.0) + 0.28 * agreement - 0.35 * disagreement - volatility_penalty - data_penalty, 0.0, 1.0)
    top = tuple(name for name, value in sorted(components, key=lambda item: abs(item[1]), reverse=True) if abs(value) > 0)[:3]
    return ProbabilityEstimate(p_up, 1.0 - p_up, confidence, disagreement, raw, top)


def logistic_probability(features: MarketFeatures, coefficients: Mapping[str, float]) -> ProbabilityEstimate:
    """Simple statistical mode using externally trained coefficients only."""
    values = features.to_dict()
    raw = float(coefficients.get("intercept", 0.0))
    used = []
    for name, coefficient in coefficients.items():
        if name == "intercept":
            continue
        value = values.get(name)
        if isinstance(value, (int, float)):
            contribution = float(coefficient) * float(value)
            raw += contribution
            used.append((name, contribution))
    p_up = _clip(1.0 / (1.0 + exp(-raw)), 0.01, 0.99)
    confidence = _clip(0.5 + abs(p_up - 0.5), 0.0, 1.0)
    top = tuple(name for name, _ in sorted(used, key=lambda item: abs(item[1]), reverse=True)[:3])
    return ProbabilityEstimate(p_up, 1.0 - p_up, confidence, 0.0, raw, top, "logistic")


def fsum(values):
    return sum(values, 0.0)
