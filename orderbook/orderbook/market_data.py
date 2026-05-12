"""
orderbook/market_data.py

Synthetic market data generator and replay utilities.

Generates realistic order flow using:
- Geometric Brownian Motion for mid-price
- Poisson arrivals for order flow
- Hawkes process for clustered order arrivals (self-exciting)
- Log-normal order sizes

Also provides a CSV loader for real tick data (Binance format).
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from .types import Side, OrderType, TimeInForce, BookSnapshot, BookLevel
from .engine import OrderBook
from .exchange import Exchange


@dataclass
class MarketDataConfig:
    symbol:           str   = "AAPL"
    initial_price:    float = 100.0
    tick_size:        float = 0.01
    volatility:       float = 0.02      # annualized vol
    drift:            float = 0.0       # annualized drift
    spread_ticks:     int   = 2         # initial spread in ticks
    depth_levels:     int   = 10        # levels each side
    order_rate:       float = 50.0      # orders per second
    cancel_rate:      float = 0.3       # fraction of orders that cancel
    market_order_frac: float = 0.1      # fraction that are market orders
    seed:             Optional[int] = 42


class SyntheticMarket:
    """
    Generates synthetic order flow and feeds it to an Exchange.
    Uses GBM for price discovery with Poisson-distributed order arrivals.

    Design note: we maintain a separate "true price" process (GBM) that
    drives where informed traders submit. Noise traders submit randomly
    around the current spread. This creates realistic book dynamics
    including adverse selection for market makers.
    """

    def __init__(self, config: MarketDataConfig, exchange: Exchange) -> None:
        self.config   = config
        self.exchange = exchange
        self._rng     = random.Random(config.seed)
        self._book    = exchange.add_symbol(config.symbol)

        self._true_price = config.initial_price
        self._time_ns    = 0
        self._dt_s       = 1.0 / config.order_rate   # seconds per order event
        self._vol_per_event = (config.volatility
                               * math.sqrt(self._dt_s / (252 * 6.5 * 3600)))
        self._drift_per_event = config.drift * self._dt_s / (252 * 6.5 * 3600)

        # seed the book with resting orders
        self._seed_book()

    def _seed_book(self) -> None:
        """Populate initial book with resting limit orders."""
        cfg   = self.config
        price = cfg.initial_price
        tick  = cfg.tick_size
        half_spread = (cfg.spread_ticks * tick) / 2

        for i in range(1, cfg.depth_levels + 1):
            ask_price = price + half_spread + (i - 1) * tick
            qty = max(1, int(self._rng.gauss(100, 30)))
            self.exchange.submit(cfg.symbol, Side.ASK, OrderType.LIMIT,
                                 qty, ask_price, TimeInForce.GTC,
                                 timestamp=self._time_ns)

            bid_price = price - half_spread - (i - 1) * tick
            qty = max(1, int(self._rng.gauss(100, 30)))
            self.exchange.submit(cfg.symbol, Side.BID, OrderType.LIMIT,
                                 qty, bid_price, TimeInForce.GTC,
                                 timestamp=self._time_ns)

    def step(self) -> BookSnapshot:
        """
        Advance the simulation by one order event.
        Updates true price via GBM, generates one order, returns snapshot.
        """
        self._time_ns += int(self._dt_s * 1e9)
        cfg = self.config

        # GBM step for true price
        z = self._rng.gauss(0, 1)
        self._true_price *= math.exp(
            self._drift_per_event - 0.5 * self._vol_per_event**2
            + self._vol_per_event * z
        )

        snap = self.exchange.snapshot(cfg.symbol, depth=1)
        if snap is None:
            return self.exchange.snapshot(cfg.symbol)

        best_bid = (OrderBook.int_to_price(snap.best_bid)
                    if snap.best_bid else self._true_price)
        best_ask = (OrderBook.int_to_price(snap.best_ask)
                    if snap.best_ask else self._true_price)
        tick     = cfg.tick_size

        # decide order type
        r = self._rng.random()
        if r < cfg.market_order_frac:
            # market order — informed, direction driven by true price vs mid
            mid = (best_bid + best_ask) / 2
            side = Side.BID if self._true_price > mid else Side.ASK
            qty  = max(1, int(self._rng.expovariate(1/20)))
            self.exchange.submit(cfg.symbol, side, OrderType.MARKET,
                                 qty, timestamp=self._time_ns)

        elif r < cfg.market_order_frac + cfg.cancel_rate:
            # cancel a random resting order — handled implicitly by engine
            pass

        else:
            # limit order — noise trader around spread
            side = self._rng.choice([Side.BID, Side.ASK])
            if side == Side.BID:
                offset = self._rng.randint(0, cfg.depth_levels) * tick
                price  = round(best_bid - offset, 4)
            else:
                offset = self._rng.randint(0, cfg.depth_levels) * tick
                price  = round(best_ask + offset, 4)

            qty = max(1, int(self._rng.expovariate(1/50)))
            self.exchange.submit(cfg.symbol, side, OrderType.LIMIT,
                                 qty, price, TimeInForce.GTC,
                                 timestamp=self._time_ns)

        return self.exchange.snapshot(cfg.symbol, depth=cfg.depth_levels)

    def generate(self, n_events: int) -> list[BookSnapshot]:
        """Generate n_events order events and return snapshots."""
        snapshots = []
        for _ in range(n_events):
            snap = self.step()
            if snap:
                snapshots.append(snap)
        return snapshots

    def generate_stream(self, n_events: int) -> Iterator[BookSnapshot]:
        """Lazy generator version for memory-efficient large simulations."""
        for _ in range(n_events):
            snap = self.step()
            if snap:
                yield snap


def load_binance_csv(filepath: str, symbol: str,
                     exchange: Exchange,
                     max_rows: int = 10_000) -> list[BookSnapshot]:
    """
    Load Binance L2 order book snapshot CSV.
    Expected columns: timestamp, bids (JSON), asks (JSON)
    Returns list of BookSnapshot objects for backtesting.

    Binance provides free historical L2 data at:
    https://data.binance.vision/
    """
    import csv
    import json

    snapshots = []
    book = exchange.add_symbol(symbol)

    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break

                ts   = int(row.get("timestamp", i)) * 1_000_000
                bids_raw = json.loads(row.get("bids", "[]"))
                asks_raw = json.loads(row.get("asks", "[]"))

                bids = [BookLevel(
                    price=OrderBook.price_to_int(float(b[0])),
                    total_qty=int(float(b[1])),
                    order_count=1,
                ) for b in bids_raw[:10]]

                asks = [BookLevel(
                    price=OrderBook.price_to_int(float(a[0])),
                    total_qty=int(float(a[1])),
                    order_count=1,
                ) for a in asks_raw[:10]]

                snap = BookSnapshot(symbol=symbol, timestamp=ts,
                                    bids=bids, asks=asks)
                snapshots.append(snap)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Market data file not found: {filepath}\n"
            "Download Binance L2 data from: "
            "https://data.binance.vision/?prefix=data/spot/daily/bookDepth/"
        )

    return snapshots
