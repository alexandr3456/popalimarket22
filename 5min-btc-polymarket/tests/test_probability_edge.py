import unittest
from datetime import datetime, timedelta, timezone

from edge_engine import DecisionContext, SideQuote, StrategyConfig, decide_trade
from feature_engine import PricePoint, build_features
from probability_engine import ProbabilityEstimate, rule_based_probability

UTC = timezone.utc


class FeatureProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 1, tzinfo=UTC)
        self.end = self.start + timedelta(seconds=300)

    def points(self, direction=1):
        return [PricePoint(self.start + timedelta(seconds=s), 100000 + direction * s * 2,
                           buy_volume=8 if direction > 0 else 2, sell_volume=2 if direction > 0 else 8,
                           bid_depth=70 if direction > 0 else 30, ask_depth=30 if direction > 0 else 70)
                for s in range(0, 181, 5)]

    def test_reference_is_exact_market_open(self):
        features = build_features(self.points(), self.start, self.end, self.start + timedelta(seconds=150))
        self.assertEqual(features.btc_reference_open, 100000)
        self.assertEqual(features.distance_from_market_open_usd, 300)
        self.assertEqual(features.seconds_left, 150)

    def test_missing_market_open_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            build_features(self.points()[1:], self.start, self.end, self.start + timedelta(seconds=150))

    def test_probability_is_directionally_symmetric(self):
        up = rule_based_probability(build_features(self.points(1), self.start, self.end, self.start + timedelta(seconds=150)))
        down = rule_based_probability(build_features(self.points(-1), self.start, self.end, self.start + timedelta(seconds=150)))
        self.assertGreater(up.p_up, 0.5)
        self.assertLess(down.p_up, 0.5)
        self.assertAlmostEqual(up.p_up, 1 - down.p_up, places=5)

    def test_disagreement_reduces_confidence(self):
        consistent = build_features(self.points(), self.start, self.end, self.start + timedelta(seconds=150))
        mixed_points = self.points()
        mixed_points[-7:] = [PricePoint(p.timestamp, p.price - (i + 1) * 100,
                                       buy_volume=p.buy_volume, sell_volume=p.sell_volume,
                                       bid_depth=p.bid_depth, ask_depth=p.ask_depth)
                             for i, p in enumerate(mixed_points[-7:])]
        mixed = build_features(mixed_points, self.start, self.end, self.start + timedelta(seconds=150))
        self.assertGreater(rule_based_probability(consistent).confidence, rule_based_probability(mixed).confidence)


class EdgeTests(unittest.TestCase):
    def estimate(self, up=.72, confidence=.8):
        return ProbabilityEstimate(up, 1-up, confidence, 0, 1, ("distance",))

    def test_high_probability_but_small_edge_is_no_trade(self):
        result = decide_trade(self.estimate(.82), SideQuote(.79, .80, 100), SideQuote(.18, .20, 100), DecisionContext(120))
        self.assertFalse(result.trade)
        self.assertIn("edge", result.rejection_reasons)

    def test_down_trade_is_symmetric(self):
        result = decide_trade(self.estimate(.29), SideQuote(.70, .72, 100), SideQuote(.54, .56, 100), DecisionContext(120))
        self.assertTrue(result.trade)
        self.assertEqual(result.side, "DOWN")
        self.assertAlmostEqual(result.edge, .13)

    def test_every_gate_is_enforced(self):
        cfg = StrategyConfig()
        result = decide_trade(self.estimate(.75, .5), SideQuote(.40, .50, 1), SideQuote(.48, .50, 1),
                              DecisionContext(10, True, True, cfg.daily_trade_limit), cfg)
        for gate in ("confidence", "liquidity", "time_window", "no_open_position", "market_not_traded", "daily_limit"):
            self.assertFalse(result.gates[gate])


if __name__ == "__main__": unittest.main()
