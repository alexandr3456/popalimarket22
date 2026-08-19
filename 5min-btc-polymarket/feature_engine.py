"""BTC-only feature engineering for a Polymarket five-minute window.

The engine is deliberately data-source agnostic.  Callers must supply timestamped
public BTC observations and the exact market start.  Missing optional public data
stays ``None``; it is never imputed from Polymarket prices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from statistics import fmean
from typing import Optional, Sequence

UTC = timezone.utc


@dataclass(frozen=True)
class PricePoint:
    timestamp: datetime
    price: float
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None


@dataclass(frozen=True)
class MarketFeatures:
    timestamp_utc: str
    source_timestamp_utc: str
    market_start_utc: str
    market_end_utc: str
    seconds_left: float
    seconds_elapsed: float
    normalized_time: float
    btc_reference_open: float
    btc_current: float
    distance_from_market_open_usd: float
    distance_from_market_open_pct: float
    return_5s: Optional[float]
    return_10s: Optional[float]
    return_20s: Optional[float]
    return_30s: Optional[float]
    return_60s: Optional[float]
    return_120s: Optional[float]
    momentum_acceleration: Optional[float]
    momentum_slope: Optional[float]
    realized_volatility_30s: Optional[float]
    realized_volatility_60s: Optional[float]
    range_high_low_60s: Optional[float]
    distance_over_volatility: Optional[float]
    buy_volume: Optional[float]
    sell_volume: Optional[float]
    volume_imbalance: Optional[float]
    volume_acceleration: Optional[float]
    bid_depth: Optional[float]
    ask_depth: Optional[float]
    order_book_imbalance: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _at_or_before(points: Sequence[PricePoint], when: datetime) -> Optional[PricePoint]:
    candidates = [p for p in points if _utc(p.timestamp) <= when]
    return max(candidates, key=lambda p: p.timestamp) if candidates else None


def _return(points: Sequence[PricePoint], now: datetime, seconds: int) -> Optional[float]:
    old = _at_or_before(points, now - timedelta(seconds=seconds))
    if old is None or old.price <= 0:
        return None
    return points[-1].price / old.price - 1.0


def _window(points: Sequence[PricePoint], now: datetime, seconds: int) -> list[PricePoint]:
    cutoff = now - timedelta(seconds=seconds)
    return [p for p in points if cutoff <= _utc(p.timestamp) <= now]


def _realized_vol(points: Sequence[PricePoint]) -> Optional[float]:
    returns = [b.price / a.price - 1.0 for a, b in zip(points, points[1:]) if a.price > 0]
    if not returns:
        return None
    return sqrt(sum(x * x for x in returns))


def _slope(points: Sequence[PricePoint]) -> Optional[float]:
    if len(points) < 2:
        return None
    x0 = _utc(points[0].timestamp).timestamp()
    xs = [_utc(p.timestamp).timestamp() - x0 for p in points]
    ys = [p.price for p in points]
    xm, ym = fmean(xs), fmean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    return None if den == 0 else sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den


def _sum_optional(points: Sequence[PricePoint], name: str) -> Optional[float]:
    vals = [getattr(p, name) for p in points if getattr(p, name) is not None]
    return sum(vals) if vals else None


def build_features(
    points: Sequence[PricePoint], market_start: datetime, market_end: datetime,
    now: Optional[datetime] = None, reference_tolerance_seconds: float = 2.0,
) -> MarketFeatures:
    """Build features, requiring an observation at the actual market boundary.

    The first observation at/after start (within tolerance), or last observation
    just before it (within tolerance), is the reference. A rolling 5m value is not
    accepted as a substitute.
    """
    if not points:
        raise ValueError("at least one BTC price point is required")
    start, end = _utc(market_start), _utc(market_end)
    ordered = sorted(points, key=lambda p: _utc(p.timestamp))
    current_time = _utc(now or ordered[-1].timestamp)
    usable = [p for p in ordered if _utc(p.timestamp) <= current_time]
    if not usable:
        raise ValueError("no BTC observation at or before now")
    reference = min(ordered, key=lambda p: abs((_utc(p.timestamp) - start).total_seconds()))
    if abs((_utc(reference.timestamp) - start).total_seconds()) > reference_tolerance_seconds:
        raise ValueError("missing BTC reference price at market start")
    current = usable[-1]
    if reference.price <= 0 or current.price <= 0:
        raise ValueError("BTC prices must be positive")
    elapsed = min(300.0, max(0.0, (current_time - start).total_seconds()))
    left = max(0.0, (end - current_time).total_seconds())
    rets = {s: _return(usable, current_time, s) for s in (5, 10, 20, 30, 60, 120)}
    w30, w60 = _window(usable, current_time, 30), _window(usable, current_time, 60)
    vol30, vol60 = _realized_vol(w30), _realized_vol(w60)
    distance = current.price - reference.price
    range60 = max((p.price for p in w60), default=current.price) - min((p.price for p in w60), default=current.price)
    buy, sell = _sum_optional(w60, "buy_volume"), _sum_optional(w60, "sell_volume")
    total_vol = (buy or 0.0) + (sell or 0.0)
    volume_imb = None if buy is None or sell is None or total_vol <= 0 else (buy - sell) / total_vol
    last30 = w30
    older30 = [p for p in w60 if p not in last30]
    old_vol = (_sum_optional(older30, "buy_volume") or 0) + (_sum_optional(older30, "sell_volume") or 0)
    new_vol = (_sum_optional(last30, "buy_volume") or 0) + (_sum_optional(last30, "sell_volume") or 0)
    volume_acc = None if buy is None or sell is None or old_vol <= 0 else new_vol / old_vol - 1.0
    bid, ask = current.bid_depth, current.ask_depth
    depth_total = (bid or 0) + (ask or 0)
    ob_imb = None if bid is None or ask is None or depth_total <= 0 else (bid - ask) / depth_total
    accel = None if rets[10] is None or rets[30] is None else rets[10] - rets[30] / 3.0
    return MarketFeatures(
        _iso(current_time), _iso(current.timestamp), _iso(start), _iso(end), left, elapsed,
        elapsed / 300.0, reference.price, current.price, distance, distance / reference.price,
        rets[5], rets[10], rets[20], rets[30], rets[60], rets[120], accel, _slope(w60),
        vol30, vol60, range60, None if not vol60 else (distance / current.price) / vol60,
        buy, sell, volume_imb, volume_acc, bid, ask, ob_imb,
    )
