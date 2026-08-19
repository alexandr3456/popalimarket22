#!/usr/bin/env python3
"""Backtest BTC 5m baselines and the probability+edge paper strategy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def get(row: dict, *paths: str, default=None):
    for path in paths:
        value: Any = row
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and value != "":
            return value
    return default


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: str | Path) -> list[dict]:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return [unflatten_csv_row(row) for row in csv.DictReader(handle)]


def unflatten_csv_row(row: dict[str, str]) -> dict:
    result: dict = {}
    for path, raw in row.items():
        if raw in (None, ""):
            continue
        value: Any = raw
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        cursor = result
        keys = path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return result


def resolved_snapshots(rows: Iterable[dict]) -> list[dict]:
    winners: dict[str, str] = {}
    snapshots: dict[str, list[dict]] = {}
    for row in rows:
        slug = str(get(row, "market_slug", default=""))
        if not slug:
            continue
        winner = get(row, "official_winner", "final_outcome.winner")
        if winner:
            winners[slug] = str(winner).upper()
        if get(row, "record_type", default="snapshot") == "snapshot":
            snapshots.setdefault(slug, []).append(row)
    out = []
    for slug, market_rows in snapshots.items():
        market_rows.sort(key=lambda r: str(get(r, "timestamp_utc", "features.timestamp_utc", default="")))
        winner = winners.get(slug) or str(get(market_rows[-1], "official_winner", default="")).upper()
        if winner in {"UP", "DOWN"}:
            for row in market_rows:
                row["_winner"] = winner
            out.extend(market_rows)
    return out


@dataclass(frozen=True)
class Trade:
    market_slug: str
    timestamp: str
    side: str
    winner: str
    entry: float
    probability: Optional[float]
    edge: Optional[float]
    seconds_left: Optional[float]
    fee: float
    slippage: float
    exit_value: Optional[float] = None

    @property
    def won(self): return self.side == self.winner
    @property
    def stake(self): return self.entry + self.fee + self.slippage
    @property
    def gross_pnl(self): return (self.exit_value if self.exit_value is not None else (1.0 if self.won else 0.0)) - self.entry
    @property
    def net_pnl(self): return (self.exit_value if self.exit_value is not None else (1.0 if self.won else 0.0)) - self.stake


def quote(row: dict, side: str) -> Optional[float]:
    side = side.lower()
    return as_float(get(row, f"polymarket.{side}_ask", f"{side}_ask"))


def make_trade(row: dict, side: str, probability=None, edge=None, fee=0.01, slippage=0.01, exit_value=None) -> Optional[Trade]:
    entry = quote(row, side)
    if entry is None or not 0 < entry < 1:
        return None
    return Trade(str(get(row, "market_slug")), str(get(row, "timestamp_utc", default="")), side.upper(), row["_winner"], entry,
                 as_float(probability), as_float(edge), as_float(get(row, "seconds_left", "features.seconds_left")), fee, slippage, as_float(exit_value))


def select_one_per_market(rows: Iterable[dict], selector: Callable[[dict], Optional[Trade]]) -> list[Trade]:
    selected: dict[str, Trade] = {}
    for row in rows:
        slug = str(get(row, "market_slug"))
        if slug not in selected:
            trade = selector(row)
            if trade:
                selected[slug] = trade
    return sorted(selected.values(), key=lambda t: t.timestamp)


def strategies(rows: list[dict], repricing: bool = False) -> dict[str, list[Trade]]:
    def always_up(r): return make_trade(r, "UP")
    def favorite(r):
        up, down = quote(r, "UP"), quote(r, "DOWN")
        return None if up is None or down is None else make_trade(r, "UP" if up >= down else "DOWN")
    def old_threshold(r):
        up, down = quote(r, "UP"), quote(r, "DOWN")
        candidates = [(x, side) for x, side in ((up, "UP"), (down, "DOWN")) if x is not None and x >= 0.70]
        return None if not candidates else make_trade(r, max(candidates)[1])
    def momentum(r):
        move = as_float(get(r, "features.distance_from_market_open_usd", "distance_from_market_open_usd"))
        if move is None or not 70 <= abs(move) <= 100:
            return None
        return make_trade(r, "UP" if move > 0 else "DOWN")
    def probability_edge(r):
        if not bool(get(r, "trade", "decision.trade", default=False)):
            return None
        side = get(r, "decision.side")
        if side not in {"UP", "DOWN"}:
            return None
        probability = get(r, "decision.model_probability")
        edge = get(r, "decision.edge")
        exit_value = get(r, "paper_repricing.exit_price", "repricing_exit_price") if repricing else None
        if repricing and exit_value is None:
            return None
        return make_trade(r, side, probability, edge, exit_value=exit_value)
    selectors = {"always_up": always_up, "polymarket_favorite": favorite,
                 "old_threshold_0.70": old_threshold, "momentum_70_100_usd": momentum,
                 "probability_edge": probability_edge}
    return {name: select_one_per_market(rows, selector) for name, selector in selectors.items()}


def metrics(trades: list[Trade]) -> dict:
    wins = sum(t.won for t in trades); losses = len(trades) - wins
    gross = sum(t.gross_pnl for t in trades); fees = sum(t.fee for t in trades)
    slippage = sum(t.slippage for t in trades); net = sum(t.net_pnl for t in trades)
    curve, peak, max_dd, running = [], 0.0, 0.0, 0.0
    longest, streak = 0, 0
    for t in trades:
        running += t.net_pnl; curve.append(running); peak = max(peak, running); max_dd = max(max_dd, peak - running)
        streak = 0 if t.won else streak + 1; longest = max(longest, streak)
    gains = sum(max(0, t.net_pnl) for t in trades); losses_cash = -sum(min(0, t.net_pnl) for t in trades)
    modeled = [t.probability for t in trades if t.probability is not None]
    edges = [t.edge for t in trades if t.edge is not None]
    invested = sum(t.stake for t in trades)
    return {"trades": len(trades), "wins": wins, "losses": losses,
            "winrate": wins / len(trades) if trades else None,
            "average_entry_price": sum(t.entry for t in trades) / len(trades) if trades else None,
            "average_modeled_probability": sum(modeled) / len(modeled) if modeled else None,
            "average_edge": sum(edges) / len(edges) if edges else None,
            "gross_pnl": gross, "estimated_fees": fees, "estimated_slippage": slippage,
            "net_pnl": net, "roi": net / invested if invested else None, "max_drawdown": max_dd,
            "longest_losing_streak": longest,
            "profit_factor": gains / losses_cash if losses_cash else ("Infinity" if gains else None)}


BUCKETS = {
    "model_probability": [(0.55,.60),(0.60,.65),(.65,.70),(.70,.75),(.75,.80),(.80,.85),(.85,1.01)],
    "entry_price": [(.45,.55),(.55,.65),(.65,.75),(.75,.85)],
    "seconds_left": [(150,180.0001),(120,150),(90,120),(60,90),(45,60)],
    "edge": [(.03,.05),(.05,.07),(.07,.10),(.10,.15),(.15,1.01)],
}


def bucket_analysis(trades: list[Trade]) -> dict:
    attrs = {"model_probability": lambda t:t.probability, "entry_price":lambda t:t.entry,
             "seconds_left":lambda t:t.seconds_left, "edge":lambda t:t.edge}
    report = {}
    for name, bounds in BUCKETS.items():
        rows = []
        for low, high in bounds:
            sample = [t for t in trades if attrs[name](t) is not None and low <= attrs[name](t) < high]
            m = metrics(sample)
            rows.append({"bucket": f"{low:g}-{high:g}", "sample_size":len(sample), "winrate":m["winrate"],
                         "avg_entry":m["average_entry_price"], "net_pnl":m["net_pnl"], "roi":m["roi"]})
        report[name] = rows
    return report


CALIBRATION_BOUNDS = [(.50,.55),(.55,.60),(.60,.65),(.65,.70),(.70,.75),(.75,.80),(.80,.85),(.85,.90),(.90,1.01)]


def calibration(trades: list[Trade]) -> dict:
    usable = [t for t in trades if t.probability is not None]
    buckets=[]
    for low, high in CALIBRATION_BOUNDS:
        sample=[t for t in usable if low <= t.probability < high]
        predicted=sum(t.probability for t in sample)/len(sample) if sample else None
        actual=sum(t.won for t in sample)/len(sample) if sample else None
        buckets.append({"bucket":f"{low:g}-{high:g}","predicted_probability":predicted,"actual_winrate":actual,
                        "count":len(sample),"calibration_error":None if not sample else actual-predicted})
    brier=sum((t.probability-float(t.won))**2 for t in usable)/len(usable) if usable else None
    log_loss=None
    if usable:
        log_loss=-sum(float(t.won)*math.log(max(t.probability,1e-15))+(1-float(t.won))*math.log(max(1-t.probability,1e-15)) for t in usable)/len(usable)
    return {"buckets":buckets,"brier_score":brier,"log_loss":log_loss}


def chronological_split(rows: list[dict], ratio: float = .70) -> tuple[list[dict], list[dict]]:
    slugs = sorted({str(get(r,"market_slug")) for r in rows}, key=lambda slug:min(str(get(r,"market_start","features.market_start_utc","timestamp_utc",default="")) for r in rows if get(r,"market_slug")==slug))
    cut = int(len(slugs)*ratio); train=set(slugs[:cut])
    return ([r for r in rows if get(r,"market_slug") in train], [r for r in rows if get(r,"market_slug") not in train])


def generate_report(rows: list[dict]) -> dict:
    resolved=resolved_snapshots(rows); train,test=chronological_split(resolved)
    report={}
    for label, sample in (("TRAIN",train),("TEST",test)):
        runs=strategies(sample); new=runs["probability_edge"]
        repricing_runs = strategies(sample, repricing=True)
        report[label]={"strategies":{k:metrics(v) for k,v in runs.items()},
                       "exit_strategies": {"settlement_mode": metrics(new),
                                           "repricing_mode": metrics(repricing_runs["probability_edge"])},
                       "probability_edge_buckets":bucket_analysis(new),"calibration":calibration(new)}
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("dataset"); parser.add_argument("--output")
    args=parser.parse_args(); report=generate_report(load_rows(args.dataset)); text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output: Path(args.output).write_text(text+"\n",encoding="utf-8")
    print(text); return 0


if __name__ == "__main__": raise SystemExit(main())
