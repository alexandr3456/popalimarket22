"""Append-only JSONL/CSV research snapshots, including NO TRADE observations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


class ResearchDataCollector:
    def __init__(self, jsonl_path: str | Path, csv_path: str | Path):
        self.jsonl_path, self.csv_path = Path(jsonl_path), Path(csv_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def flatten(record: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, (list, tuple)):
                flat[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                flat[prefix] = value
        visit("", record)
        return flat

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        flat = self.flatten(record)
        existing: list[dict[str, Any]] = []
        fields = list(flat)
        if self.csv_path.exists() and self.csv_path.stat().st_size:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                existing = list(reader)
                fields = list(reader.fieldnames or [])
            for key in flat:
                if key not in fields:
                    fields.append(key)
        existing.append(flat)
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)

    def append_resolution(self, market_slug: str, official_winner: str, final_outcome: Any) -> None:
        """Append a resolution event; immutable JSONL history avoids unsafe rewrites."""
        self.append({"record_type": "resolution", "market_slug": market_slug,
                     "official_winner": official_winner, "final_outcome": final_outcome})


def build_snapshot(market: dict, features, estimate, decision, quotes: dict) -> dict:
    return {
        "record_type": "snapshot", "timestamp_utc": features.timestamp_utc,
        "market_slug": market["slug"], "market_start": features.market_start_utc,
        "market_end": features.market_end_utc, "seconds_left": features.seconds_left,
        "features": features.to_dict(), "polymarket": quotes,
        "model": estimate.to_dict(), "decision": decision.to_dict(),
        "trade": decision.trade, "rejection_reasons": list(decision.rejection_reasons),
    }
