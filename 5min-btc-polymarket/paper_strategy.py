#!/usr/bin/env python3
"""Offline/paper probability+edge strategy runner.

Input is one JSON snapshot from a public-data adapter. This command has no live
execution flag, wallet code, credential loading, or order-placement dependency.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from debug_audit import append_debug, debug_record
from edge_engine import DecisionContext, SideQuote, StrategyConfig, decide_trade
from feature_engine import PricePoint, build_features
from probability_engine import logistic_probability, rule_based_probability
from research_data_collector import ResearchDataCollector, build_snapshot
from telegram_paper import paper_entry_message, status_message


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluate_payload(payload: dict, config: StrategyConfig, coefficients: dict | None = None):
    points = [PricePoint(parse_time(p["timestamp"]), float(p["price"]), p.get("buy_volume"),
                         p.get("sell_volume"), p.get("bid_depth"), p.get("ask_depth"))
              for p in payload["btc_points"]]
    features = build_features(points, parse_time(payload["market_start"]), parse_time(payload["market_end"]),
                              parse_time(payload.get("timestamp") or payload["btc_points"][-1]["timestamp"]))
    estimate = logistic_probability(features, coefficients) if coefficients else rule_based_probability(features)
    poly = payload["polymarket"]
    up = SideQuote(poly.get("up_bid"), poly.get("up_ask"), poly.get("up_liquidity"))
    down = SideQuote(poly.get("down_bid"), poly.get("down_ask"), poly.get("down_liquidity"))
    state = payload.get("paper_state", {})
    context = DecisionContext(features.seconds_left, bool(state.get("has_open_position")),
                              bool(state.get("market_already_traded")), int(state.get("daily_trade_count", 0)))
    return features, estimate, decide_trade(estimate, up, down, context, config)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON snapshot; stdin when omitted")
    parser.add_argument("--config", default="config/research_strategy.json")
    parser.add_argument("--coefficients", help="optional TRAIN-only logistic coefficients JSON")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    raw = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    payload = json.loads(raw)
    cfg = StrategyConfig.from_json(args.config)
    coefficients = json.loads(Path(args.coefficients).read_text(encoding="utf-8")) if args.coefficients else None
    features, estimate, decision = evaluate_payload(payload, cfg, coefficients)
    snapshot = build_snapshot({"slug": payload["market_slug"]}, features, estimate, decision, payload["polymarket"])
    data = Path(args.data_dir)
    ResearchDataCollector(data / "research_data.jsonl", data / "research_data.csv").append(snapshot)
    audit = debug_record({"btc": features.source_timestamp_utc, "market": payload.get("market_source_timestamp"),
                          "polymarket": payload.get("polymarket_source_timestamp")}, features, estimate, decision, cfg)
    if args.debug_data:
        append_debug(audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    if decision.trade:
        print(paper_entry_message(payload["market_slug"], decision.side, features, estimate,
                                  payload["polymarket"], decision, 1.0))
    elif args.status:
        print(status_message(payload["market_slug"], features, estimate, decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
