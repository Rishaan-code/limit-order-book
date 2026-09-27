# limit-order-book

A high-performance limit order book engine with a backtesting framework, written from scratch in Python.

Built to understand the core data structures and algorithms behind market microstructure. Every design decision is documented with explicit reasoning about correctness and performance tradeoffs.

---

## What it does

- **Matching engine** with price-time priority: O(log n) insertion/cancellation, O(1) best price access
- **All four order types**: LIMIT, MARKET, IOC (immediate-or-cancel), FOK (fill-or-kill)
- **Cancel and amend** with correct time-priority semantics (qty-down preserves priority; price change loses it)
- **Multi-symbol exchange** routing orders to per-symbol books
- **Backtesting framework** with three built-in strategies and a full performance report (Sharpe, drawdown, slippage)
- **Synthetic market data**: a latent GBM fair-value process with evenly spaced order events, exponential order sizes, noise traders quoting around fair value, informed market orders, and random cancellation
- **32 tests**: 28 targeted unit tests plus 4 Hypothesis property-based tests, covering 5 core invariants
- **Benchmarks** showing 200K+ orders/sec throughput, ~5µs average latency, 1.3M+ cancels/sec, p99 under 35µs

---

## Performance

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
Hardware: Python 3.12, single core
```

---

## Design decisions

### Fixed-point integer arithmetic
All prices are stored as `int` with 4 decimal places of precision (multiply by 10,000). Floating-point is avoided entirely on the matching path. This is deliberate: floating-point arithmetic is non-associative and produces rounding errors that can silently violate price-time priority invariants. A fill at the wrong price is a serious bug in a real system.

### Hybrid deque + dict for price levels
Each price level uses a `collections.deque` for FIFO ordering plus a `dict[order_id, Order]` for O(1) cancellation. A pure deque gives O(n) cancellation (can't find by ID). A pure dict loses insertion order. The hybrid gives O(1) for all hot paths at the cost of one extra dict entry per resting order. Cancelled orders are removed lazily (tombstone approach) — the deque front is purged at match time rather than on every cancel, which avoids O(n) deque scans.

### SortedDict for the price ladder
`sortedcontainers.SortedDict` gives O(log n) insert/delete and O(1) best price access (peekitem). This is equivalent to a red-black tree or skip list. In production this would be a custom C++ data structure but SortedDict gives the same asymptotic complexity in Python. Bid side uses negated keys (`-price`) so the highest bid is always `peekitem(0)`.

### FOK correctness: probe before execute
Fill-or-kill orders check available liquidity *before* touching any book state, then execute only if the full quantity is available. A naive implementation might partially fill before discovering the FOK condition cannot be satisfied — this would leave the book in an incorrect state (shares consumed that should have been preserved). The probe is O(k) where k is the number of levels checked, with no state mutation.

### Thread safety
Each OrderBook is intentionally not thread-safe. In a multi-symbol system, each symbol's book runs in its own thread/process with no shared mutable state between books. Intra-book concurrency would require a lock or lock-free data structure. This design makes the concurrency model explicit rather than hiding it behind incorrect mutex usage.

---

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from orderbook import Exchange, Side, OrderType

ex = Exchange()
ex.add_symbol("AAPL")

# post resting orders
ex.submit("AAPL", Side.ASK, OrderType.LIMIT, qty=100, price=150.05)
ex.submit("AAPL", Side.BID, OrderType.LIMIT, qty=100, price=149.95)

# aggressive order crosses the spread
result = ex.submit("AAPL", Side.BID, OrderType.LIMIT, qty=50, price=150.05)
print(result.fills)
# [Fill(#1 BID 50 @ 150.0500, agg=3 pas=1)]

print(ex.display("AAPL"))
```

---

## Backtesting

```python
from orderbook import (
    Exchange, SyntheticMarket, MarketDataConfig,
    MidPriceMeanReversion, BacktestEngine,
)

ex     = Exchange()
config = MarketDataConfig(symbol="SIM", volatility=0.02, seed=42)
market = SyntheticMarket(config, ex)

# generate 5,000 market events
snapshots = market.generate(5_000)

# run a mean-reversion strategy on the snapshots
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

## Built-in strategies

| Strategy | Description |
|---|---|
| `MidPriceMeanReversion` | IOC orders when price deviates from rolling mid average |
| `SpreadCapture` | Market-making: post both sides, cancel/re-quote on each update |
| `VWAPExecution` | Break a large order into slices, execute over time |

---

## Running tests

```bash
# all 32 tests (28 unit, 4 Hypothesis property-based)
pytest tests/ -v

# benchmarks
python -m benchmarks.bench
```

---

## Project structure

```
orderbook/
├── orderbook/
│   ├── types.py        # Order, Fill, enums — all fixed-point, no floats
│   ├── price_level.py  # FIFO queue per price: deque + dict hybrid
│   ├── engine.py       # Matching engine: SortedDict price ladder
│   ├── exchange.py     # Multi-symbol router + fill log
│   ├── backtest.py     # Strategy interface + P&L + performance report
│   └── market_data.py  # GBM synthetic data + Binance CSV loader
├── tests/
│   └── test_orderbook.py   # 32 tests: 28 unit, 4 property-based
├── benchmarks/
│   └── bench.py        # Throughput + latency distribution
└── README.md
```

---

## Key invariants (enforced by property-based tests)

1. **Price priority**: fills always occur at the passive (resting) order's price
2. **Time priority**: at equal prices, earlier-arriving orders fill first
3. **Conservation**: sum of fill quantities equals min(aggressor qty, available qty) — no shares created or destroyed
4. **FOK atomicity**: a FOK order that cannot be fully filled leaves the book completely unchanged
5. **Book ordering**: bid levels always descending, ask levels always ascending

---

## References

- Harris, L. (2003). *Trading and Exchanges: Market Microstructure for Practitioners*. Oxford University Press.
- Gould, M. et al. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709–1742.
- NASDAQ ITCH 5.0 Protocol Specification (for production message format reference)

### The index scan that made submits O(n)

`_order_index` maps `order_id -> (side, price_key)` so cancel and amend are O(1)
lookups. Originally, when a price level emptied during matching, the engine
dropped that level's stale index entries by walking the entire index:

```python
if level.is_empty():
    del opposite[price_key]
    for oid, (s, k) in list(self._order_index.items()):
        if k == price_key:
            del self._order_index[oid]
```

That made an otherwise O(log n) submit O(n) in the number of resting orders, and
only on the submits that happened to clear a level. The mean stayed fine and the
tail did not, which is why it survived as long as it did. A fully-filled passive
order is now dropped from the index at fill time, while its id is still in hand,
and pruning a level is just the dict delete.

Measured with `benchmarks/bench.py` (`bench_latency_distribution`, n=10,000),
three runs on one machine:

| | p95 | p99 |
|---|---|---|
| with index scan | 1074 to 1141 us | 3837 to 5183 us |
| current | 14.1 to 15.8 us | 37.4 to 42.7 us |

Roughly 75x at p95. The absolute numbers move with the machine, the ratio does
not. To reproduce the regression, restore the loop above and re-run the
benchmark.
