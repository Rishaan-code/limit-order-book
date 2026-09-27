# limit-order-book

I built this to actually understand how exchanges work at the data structure level. Started reading about market microstructure and kept wondering what the matching engine looks like under the hood, so I just built one.

It's a full limit order book in Python w/ price-time priority matching, all four order types, cancel/amend, a backtesting framework, synthetic market data via GBM, and 32 tests, 4 of them property-based with Hypothesis. Hits 200K+ orders/sec on a single core.


## what it does

- price-time priority matching engine, O(log n) insert/cancel, O(1) best price
- LIMIT, MARKET, IOC, and FOK order types
- cancel and amend (qty-down preserves time priority, price change loses it)
- multi-symbol exchange routing to per-symbol books
- backtesting framework with 3 built-in strategies and a performance report (Sharpe, drawdown, slippage)
- synthetic market data from a latent GBM fair value with evenly spaced events
- 32 tests (28 unit, 4 Hypothesis property-based)

---

## numbers

```
Benchmark                    Result
─────────────────────────────────────────────
Limit order throughput       204,618 orders/sec   (4.89 µs avg)
Matching throughput          163,412 orders/sec   (6.12 µs avg)
Cancel throughput          1,324,964 cancels/sec  (0.75 µs avg)
Latency p50                    8.4 µs
Latency p95                   12.6 µs
Latency p99                   32.8 µs
Latency p99.9                 60.5 µs
Market data generation        36,848 steps/sec
─────────────────────────────────────────────
Python 3.12, single core
```

---

## interesting design decisions

### no floats anywhere
prices are stored as integers with 4 decimal places (so $100.05 becomes 1000500). floating point is non-associative so you get rounding errors that can silently break price-time priority. a fill at the wrong price is a real bug so I just avoided floats entirely on the matching path.

### deque + dict hybrid for each price level
each price level needs two things: FIFO ordering (time priority) and O(1) cancel by order ID. a pure deque can't do O(1) cancel. a pure dict loses insertion order. so I use both — deque for ordering, dict for lookup. cancelled orders get lazily removed at match time instead of immediately, which avoids scanning the deque on every cancel.

### SortedDict for the price ladder
`sortedcontainers.SortedDict` gives O(log n) insert/delete and O(1) best price. bids use negated keys so highest bid is always at index 0. in production you'd use a custom C++ red-black tree but SortedDict has the same complexity in Python.

### FOK checks liquidity before touching anything
fill-or-kill orders first probe how much is available at or better than the limit price, without modifying any state. only if the full qty is available does it actually execute. the naive approach partially fills then realizes it can't complete. that corrupts the book. the probe is O(k levels checked) with zero side effects.

### no thread safety by design
each OrderBook is single-threaded intentionally. in a real system each symbol runs in its own thread/process with no shared state between books. making it thread-safe with a lock would hide the concurrency model instead of making it explicit.

---

## quickstart

```bash
pip install -r requirements.txt
```

```python
from orderbook import Exchange, Side, OrderType

ex = Exchange()
ex.add_symbol("AAPL")

ex.submit("AAPL", Side.ASK, OrderType.LIMIT, qty=100, price=150.05)
ex.submit("AAPL", Side.BID, OrderType.LIMIT, qty=100, price=149.95)

# this crosses the spread and fills
result = ex.submit("AAPL", Side.BID, OrderType.LIMIT, qty=50, price=150.05)
print(result.fills)
# [Fill(#1 BID 50 @ 150.0500, agg=3 pas=1)]

print(ex.display("AAPL"))
```

---

## backtesting

```python
from orderbook import (
    Exchange, SyntheticMarket, MarketDataConfig,
    MidPriceMeanReversion, BacktestEngine,
)

ex     = Exchange()
config = MarketDataConfig(symbol="SIM", volatility=0.02, seed=42)
market = SyntheticMarket(config, ex)

snapshots = market.generate(5_000)

strat  = MidPriceMeanReversion("SIM", ex, window=20, threshold=0.0002)
engine = BacktestEngine(strat)
report = engine.run(snapshots)
print(report)
```

```
══════════════════════════════════════════════════
  Performance Report: MidPriceMeanReversion
══════════════════════════════════════════════════
  Total P&L:       $     -0.1140
  Realized P&L:    $     -0.1140
  Unrealized P&L:  $      0.0000
  Max Drawdown:    $      0.1140
  Total Fills:                 4
  Fill Rate:              36.4%
  Trade Count:                22
  Final Position:              0
══════════════════════════════════════════════════
```

**The strategy loses money, and that is the correct result.** The price process
is geometric Brownian motion, which is a random walk with no mean reversion in
it. There is nothing to revert to, so a mean-reversion strategy should pay the
spread and bleed. If it printed a positive Sharpe on this data, that would be a
bug in the simulator, not alpha.

The same holds for `SpreadCapture`: market making against informed flow on a
random walk loses to adverse selection (5,306 orders, 113 fills, 2.1% fill rate,
-$2.08 over 5,000 events).

Note that `threshold` has to be scaled to the market's volatility. At the default
`0.0002` the strategy never trades at all, because the deepest excursion this
configuration produces is -0.000168, shallower than the trigger. That is a
property of the parameters, not a fault in the engine, and `0.0001` is used above
to produce a run that actually trades.

Sharpe is reported but is not meaningful on these runs: the P&L series is flat
for most events with a handful of jumps, so the ratio blows up on a near-zero
denominator. Treat it as unimplemented rather than as a result.

---

## strategies included

| Strategy | what it does |
|---|---|
| `MidPriceMeanReversion` | IOC orders when price moves away from rolling mid |
| `SpreadCapture` | post both sides of the spread, cancel/re-quote each update |
| `VWAPExecution` | slice a large order over time to minimize market impact |

---

## tests and benchmarks

```bash
pytest tests/ -v        # 32 tests
python -m benchmarks.bench
```

---

## structure

```
orderbook/
├── orderbook/
│   ├── types.py        # Order, Fill, enums — fixed-point prices, no floats
│   ├── price_level.py  # per-price FIFO queue: deque + dict hybrid
│   ├── engine.py       # matching engine: SortedDict price ladder
│   ├── exchange.py     # multi-symbol router + fill log
│   ├── backtest.py     # strategy base class + P&L + performance report
│   └── market_data.py  # GBM synthetic data + Binance CSV loader
├── tests/
│   └── test_orderbook.py   # 32 tests: 28 unit, 4 property-based
├── benchmarks/
│   └── bench.py
└── README.md
```

---

## invariants the tests prove

1. fills always happen at the passive order's price, never the aggressor's
2. at the same price, earlier orders fill first
3. total fill qty always equals min(aggressor qty, available qty) — no shares appear or disappear
4. a FOK that can't fully fill leaves the book completely unchanged
5. bid levels always descending, ask levels always ascending

---

## references that actually helped

- Harris, *Trading and Exchanges* — best book on market microstructure, explains why all these rules exist
- Gould et al. (2013), "Limit order books" in *Quantitative Finance*
- NASDAQ ITCH 5.0 spec — what production message formats look like
