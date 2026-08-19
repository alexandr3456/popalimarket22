import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest_confidence_strategy import generate_report, load_rows, metrics, Trade
from paper_trading import PaperPosition, RepricingConfig, evaluate_repricing_exit, settle
from research_data_collector import ResearchDataCollector

UTC = timezone.utc


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.position = PaperPosition("m", "DOWN", self.now, .56, 1, .71, .13, .8)

    def test_settlement_pnl(self):
        closed = settle(self.position, "DOWN", self.now + timedelta(seconds=200))
        self.assertAlmostEqual(closed.pnl, 1/.56 - 1)

    def test_repricing_take_profit(self):
        closed = evaluate_repricing_exit(self.position, self.now + timedelta(seconds=10), .67, .75, .8, .25, .10)
        self.assertEqual(closed.exit_reason, "take_profit")
        self.assertGreater(closed.pnl, 0)

    def test_repricing_edge_disappears(self):
        closed = evaluate_repricing_exit(self.position, self.now + timedelta(seconds=10), .55, .55, .7, .45, 0)
        self.assertEqual(closed.exit_reason, "edge_disappeared")


class CollectorBacktestTests(unittest.TestCase):
    def test_collector_writes_trade_and_no_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = ResearchDataCollector(Path(tmp)/"d.jsonl", Path(tmp)/"d.csv")
            collector.append({"market_slug":"a", "trade":False, "rejection_reasons":["edge"]})
            collector.append({"market_slug":"b", "trade":True, "extra":{"p":.7}})
            lines = (Path(tmp)/"d.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertFalse(json.loads(lines[0])["trade"])
            self.assertIn("extra.p", (Path(tmp)/"d.csv").read_text(encoding="utf-8-sig").splitlines()[0])
            loaded = load_rows(Path(tmp)/"d.csv")
            self.assertEqual(loaded[1]["extra"]["p"], .7)

    def rows(self):
        rows=[]
        for i in range(10):
            slug=f"m{i}"
            rows.append({"record_type":"snapshot","market_slug":slug,"timestamp_utc":f"2026-01-01T00:{i:02d}:00Z",
                         "market_start":f"2026-01-01T00:{i:02d}:00Z","seconds_left":120,
                         "polymarket":{"up_ask":.55,"down_ask":.45},
                         "features":{"distance_from_market_open_usd":80},
                         "model":{"p_up":.70,"p_down":.30},
                         "decision":{"trade":True,"side":"UP","model_probability":.70,"edge":.13},"trade":True})
            rows.append({"record_type":"resolution","market_slug":slug,"official_winner":"UP" if i%2==0 else "DOWN"})
        return rows

    def test_chronological_train_test_and_report(self):
        report=generate_report(self.rows())
        self.assertEqual(report["TRAIN"]["strategies"]["always_up"]["trades"],7)
        self.assertEqual(report["TEST"]["strategies"]["probability_edge"]["trades"],3)
        self.assertIn("brier_score", report["TEST"]["calibration"])
        self.assertIn("repricing_mode", report["TEST"]["exit_strategies"])

    def test_metrics_drawdown_and_streak(self):
        ts="2026-01-01T00:00:00Z"
        trades=[Trade(str(i),ts,"UP","DOWN" if i<2 else "UP",.5,.7,.18,120,.01,.01) for i in range(3)]
        result=metrics(trades)
        self.assertEqual(result["longest_losing_streak"],2)
        self.assertGreater(result["max_drawdown"],0)


if __name__ == "__main__": unittest.main()
