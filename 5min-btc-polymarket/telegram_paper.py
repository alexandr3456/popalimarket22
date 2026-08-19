"""Pure Telegram text formatting. No network or trading capability exists here."""
from __future__ import annotations


def paper_entry_message(market: str, side: str, features, estimate, quotes: dict, decision, stake: float = 1.0) -> str:
    signals = "\n".join(f"- {signal}" for signal in estimate.top_signals) or "- insufficient public BTC data"
    return (
        "🟡 PAPER ENTRY\n\n"
        f"Market: {market}\nSide: {side}\nBTC: ${features.btc_current:,.2f}\n"
        f"Reference BTC: ${features.btc_reference_open:,.2f}\n"
        f"Distance from reference: ${features.distance_from_market_open_usd:+,.2f} "
        f"({features.distance_from_market_open_pct:+.4%})\nSeconds left: {features.seconds_left:.0f}\n\n"
        f"P(UP): {estimate.p_up:.1%}\nP(DOWN): {estimate.p_down:.1%}\n\n"
        f"UP ask: {quotes.get('up_ask')}\nDOWN ask: {quotes.get('down_ask')}\n\n"
        f"Selected side: {side}\nModel probability: {decision.model_probability:.1%}\n"
        f"Effective cost: {decision.effective_cost:.3f}\nEdge: {decision.edge:.1%}\n"
        f"Confidence: {decision.confidence:.1%}\n\nTop signals:\n{signals}\n\n"
        f"Paper stake: ${stake:g}\n\nNO REAL ORDER WAS PLACED"
    )


def status_message(market: str, features, estimate, decision) -> str:
    edge = max(x for x in (decision.up_edge, decision.down_edge) if x is not None)
    reason = ", ".join(decision.rejection_reasons) or "PAPER ENTRY"
    return (f"BTC ${features.btc_current:,.2f}\nMarket: {market}\n"
            f"P(UP)/P(DOWN): {estimate.p_up:.1%}/{estimate.p_down:.1%}\n"
            f"Best edge: {edge:.1%}\nStatus: {reason}")
