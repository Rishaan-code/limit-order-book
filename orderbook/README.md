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
- **Synthetic market data** via Geometric Brownian Motion with Poisson-distributed order arrivals
- **32 tests** including property-based tests with Hypothesis proving core invariants
- **Benchmarks** showing 130K+ orders/sec throughput, ~7µs average latency, 1M+ cancels/sec

---

## Performance

```
Benchmark                    Result
─────────────────────────────────────────────
Limit order throughput       136,780 orders/sec   (7.31 µs avg)
Matching throughput          132,060 orders/sec   (7.57 µs avg)
Cancel throughput          1,015,426 cancels/sec  (0.98 µs avg)
Latency p50                   14.3 µs
Latency p95                 1978.5 µs
Latency p99                 2206.9 µs
Market data generation        36,877 steps/sec
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
  Total P&L:           $    42.3812
  Realized P&L:        $    38.1200
  Unrealized P&L:      $     4.2612
  Sharpe Ratio:             0.847
  Max Drawdown:        $    12.4400
  Total Fills:                  187
  Fill Rate:                  73.2%
  Avg Slippage:            0.42 bps
  Trade Count:                   94
  Final Position:                10
══════════════════════════════════════════════════
```

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
# all 32 tests including property-based
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
│   └── test_orderbook.py   # 32 tests, property-based with Hypothesis
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
