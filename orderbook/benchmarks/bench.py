"""
benchmarks/bench.py

Performance benchmarks for the order book engine.
Measures throughput (orders/second) and latency distribution.

Run with: python -m benchmarks.bench
"""

import time
import statistics
import random
from typing import Callable

from orderbook import Exchange, Side, OrderType, TimeInForce


def timed(fn: Callable, n: int = 1) -> tuple[float, float]:
    """Returns (total_seconds, ops_per_second)."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed, n / elapsed


def bench_limit_order_throughput(n: int = 100_000) -> dict:
    """
    Measures raw limit order submission throughput.
    Posts only bids far below market so nothing matches — pure insertion.
    """
    ex = Exchange()
    ex.add_symbol("BENCH")
    rng = random.Random(42)

    # All bids at prices well below 90 so they never cross
    prices = [round(50.0 + rng.random() * 10, 2) for _ in range(n)]
    qtys   = [rng.randint(1, 100) for _ in range(n)]

    start = time.perf_counter()
    for i in range(n):
        ex.submit("BENCH", Side.BID, OrderType.LIMIT, qtys[i], prices[i])
    elapsed = time.perf_counter() - start

    return {
        "benchmark":      "limit_order_throughput",
        "n_orders":       n,
        "total_seconds":  elapsed,
        "orders_per_sec": n / elapsed,
        "avg_latency_us": (elapsed / n) * 1e6,
    }


def bench_matching_throughput(n: int = 50_000) -> dict:
    """
    Measures matching throughput: alternating bid/ask at same price,
    so every order immediately fills. Tests the hot matching path.
    """
    ex = Exchange()
    ex.add_symbol("BENCH")
    rng = random.Random(42)

    start = time.perf_counter()
    for i in range(n):
        if i % 2 == 0:
            ex.submit("BENCH", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        else:
            ex.submit("BENCH", Side.BID, OrderType.LIMIT, qty=10, price=100.0)
    elapsed = time.perf_counter() - start

    return {
        "benchmark":      "matching_throughput",
        "n_orders":       n,
        "total_seconds":  elapsed,
        "orders_per_sec": n / elapsed,
        "fills":          ex.get_book("BENCH").total_fills,
        "avg_latency_us": (elapsed / n) * 1e6,
    }


def bench_cancel_throughput(n: int = 50_000) -> dict:
    """
    Measures cancel throughput. Posts n orders then cancels them all.
    Tests the O(log n) cancel path.
    """
    ex = Exchange()
    ex.add_symbol("BENCH")
    rng = random.Random(42)

    order_ids = []
    for i in range(n):
        price = round(90.0 + rng.random() * 10, 2)
        side  = Side.BID if i % 2 == 0 else Side.ASK
        r     = ex.submit("BENCH", side, OrderType.LIMIT, qty=10, price=price)
        order_ids.append(r.order.order_id)

    start = time.perf_counter()
    for oid in order_ids:
        ex.cancel("BENCH", oid)
    elapsed = time.perf_counter() - start

    return {
        "benchmark":      "cancel_throughput",
        "n_cancels":      n,
        "total_seconds":  elapsed,
        "cancels_per_sec": n / elapsed,
        "avg_latency_us": (elapsed / n) * 1e6,
    }


def bench_latency_distribution(n: int = 10_000) -> dict:
    """
    Measures per-order latency distribution for LIMIT orders with matching.
    Reports p50, p95, p99, p999 latencies in microseconds.
    """
    ex = Exchange()
    ex.add_symbol("BENCH")
    rng = random.Random(42)

    latencies = []

    for i in range(n):
        # maintain a live book by alternating sides
        if i % 3 != 0:
            price = round(99.0 + rng.random() * 0.5, 2)
            ex.submit("BENCH", Side.ASK, OrderType.LIMIT, qty=10, price=price)
            price = round(99.0 - rng.random() * 0.5, 2)
            ex.submit("BENCH", Side.BID, OrderType.LIMIT, qty=10, price=price)
        else:
            t0    = time.perf_counter_ns()
            price = round(99.5 + rng.random() * 0.1, 2)
            ex.submit("BENCH", Side.BID, OrderType.LIMIT, qty=5, price=price)
            t1    = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000)   # ns -> us

    latencies.sort()
    n_lat = len(latencies)

    return {
        "benchmark":    "latency_distribution_us",
        "n_samples":    n_lat,
        "p50":          latencies[int(n_lat * 0.50)],
        "p95":          latencies[int(n_lat * 0.95)],
        "p99":          latencies[int(n_lat * 0.99)],
        "p999":         latencies[int(n_lat * 0.999)] if n_lat >= 1000 else None,
        "mean":         statistics.mean(latencies),
        "stdev":        statistics.stdev(latencies),
    }


def bench_market_data_generation(n: int = 10_000) -> dict:
    """
    Measures synthetic market data generation throughput.
    """
    from orderbook import SyntheticMarket, MarketDataConfig

    ex     = Exchange()
    config = MarketDataConfig(symbol="SIM", seed=42)
    market = SyntheticMarket(config, ex)

    start = time.perf_counter()
    for _ in range(n):
        market.step()
    elapsed = time.perf_counter() - start

    return {
        "benchmark":     "market_data_generation",
        "n_steps":       n,
        "total_seconds": elapsed,
        "steps_per_sec": n / elapsed,
    }


def format_result(result: dict) -> str:
    lines = [f"\n  {'─'*44}",
             f"  {result['benchmark']}"]
    for k, v in result.items():
        if k == "benchmark":
            continue
        if isinstance(v, float):
            if "per_sec" in k:
                lines.append(f"    {k:<28} {v:>12,.0f}")
            elif "us" in k or "latency" in k:
                lines.append(f"    {k:<28} {v:>12.2f} µs")
            elif "seconds" in k:
                lines.append(f"    {k:<28} {v:>12.4f} s")
            else:
                lines.append(f"    {k:<28} {v:>12.4f}")
        elif isinstance(v, int):
            lines.append(f"    {k:<28} {v:>12,}")
        elif v is None:
            lines.append(f"    {k:<28} {'N/A':>12}")
        else:
            lines.append(f"    {k:<28} {v!r:>12}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n" + "═"*48)
    print("  Order Book Engine — Performance Benchmarks")
    print("═"*48)

    benchmarks = [
        bench_limit_order_throughput,
        bench_matching_throughput,
        bench_cancel_throughput,
        bench_latency_distribution,
        bench_market_data_generation,
    ]

    for bench in benchmarks:
        try:
            result = bench()
            print(format_result(result))
        except Exception as e:
            print(f"\n  ERROR in {bench.__name__}: {e}")

    print("\n" + "═"*48 + "\n")
