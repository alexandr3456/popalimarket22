# Data availability and provenance

## Present in the original repository

- Gamma event/market metadata: current slug, active/closed state, end time,
  outcome labels, indicative outcome prices and CLOB token IDs.
- Public Polymarket CLOB order books: best bid/ask. The original helper did not
  retain per-side spread, top liquidity or timestamps; a data adapter must provide
  those fields to the research engine.
- Legacy threshold decision: `scripts/test_btc_5m_session_exit_sl.py` selects the
  larger UP/DOWN CLOB ask when it is at least 0.70.
- Legacy P/L: external runner cashflow is aggregated by `scripts/btc5m_report.py`.
- Legacy exit: stop loss and time-before-end close in the canonical old runner.

## Not present and therefore not fabricated

- A BTC spot/index price feed or historical tick store.
- A verified BTC reference/open tied to the market's official resolution source.
- Public BTC trade aggressor volume, volume acceleration or exchange order book.
- Historical Polymarket top-of-book liquidity and quote timestamps.
- Official resolution ingestion/history.
- A Telegram bot implementation or credentials.
- A training dataset for fitted logistic-regression coefficients.

The paper CLI expects these public observations from a separate read-only adapter.
Optional volume and BTC book fields may be omitted and remain `null`; confidence is
penalized. The window-open BTC point is mandatory and must be within two seconds of
`market_start`. The adapter should record source names and raw timestamps so data
from different BTC venues is never silently mixed.

## Polymarket isolation rule

Polymarket bid/ask, spread and liquidity are passed only to `edge_engine.py` as a
benchmark and execution-cost estimate. They are not passed to
`probability_engine.py`, so the independent BTC probability cannot accidentally
learn the market price in rule-based mode.
