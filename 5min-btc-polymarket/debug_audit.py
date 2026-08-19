"""Structured debug-data output for reproducible strategy decisions."""
from __future__ import annotations

import json
from pathlib import Path


def debug_record(raw_timestamps: dict, features, estimate, decision, config) -> dict:
    return {
        "raw_source_timestamps": raw_timestamps,
        "features": features.to_dict(),
        "probability_calculation": estimate.to_dict(),
        "cost_calculation": {
            "estimated_taker_fee": config.estimated_taker_fee,
            "slippage_buffer": config.slippage_buffer,
            "up_edge": decision.up_edge, "down_edge": decision.down_edge,
            "effective_cost": decision.effective_cost,
        },
        "gates": decision.gates,
        "final_decision": decision.to_dict(),
    }


def append_debug(record: dict, path: str | Path = "logs/debug_data.jsonl") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
