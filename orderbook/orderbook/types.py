"""
orderbook/types.py

Core types for the limit order book engine.
All IDs are integers. All prices are integers in fixed-point (price * 10000).
All quantities are integers. This avoids floating-point entirely — a deliberate
correctness decision: floating-point arithmetic is non-associative and produces
rounding errors that violate price-time priority invariants.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class Side(Enum):
    BID = auto()   # buy side
    ASK = auto()   # sell side

    def opposite(self) -> "Side":
        return Side.ASK if self is Side.BID else Side.BID


class OrderType(Enum):
    LIMIT  = auto()   # rest on book if not filled
    MARKET = auto()   # fill at any price, no resting
    IOC    = auto()   # immediate-or-cancel: fill what you can, cancel rest
    FOK    = auto()   # fill-or-kill: fill entire qty or cancel entirely


class OrderStatus(Enum):
    NEW           = auto()   # accepted, not yet processed
    OPEN          = auto()   # resting on book, partially filled or untouched
    PARTIALLY_FILLED = auto()
    FILLED        = auto()   # fully filled
    CANCELLED     = auto()   # cancelled by client or IOC/FOK logic
    REJECTED      = auto()   # rejected at entry (e.g. bad price)


class TimeInForce(Enum):
    GTC = auto()   # good-till-cancelled
    DAY = auto()   # good for session
    IOC = auto()   # immediate-or-cancel
    FOK = auto()   # fill-or-kill


@dataclass
class Order:
    """
    Represents a single order. Once created, order_id, side, symbol,
    order_type are immutable. price and qty are updated as fills arrive.
    """
    order_id:    int
    symbol:      str
    side:        Side
    order_type:  OrderType
    qty:         int              # original quantity
    price:       int              # fixed-point price (0 for MARKET orders)
    timestamp:   int              # nanosecond timestamp for time priority
    tif:         TimeInForce = TimeInForce.GTC

    # mutable state
    filled_qty:  int = field(default=0, init=False)
    status:      OrderStatus = field(default=OrderStatus.NEW, init=False)
    fills:       list = field(default_factory=list, init=False)

    @property
    def leaves_qty(self) -> int:
        """Remaining unfilled quantity."""
        return self.qty - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.OPEN,
                               OrderStatus.PARTIALLY_FILLED)

    @property
    def avg_fill_price(self) -> Optional[float]:
        if not self.fills:
            return None
        total_qty  = sum(f.qty for f in self.fills)
        total_notl = sum(f.qty * f.price for f in self.fills)
        return total_notl / total_qty if total_qty else None

    def __repr__(self) -> str:
        return (f"Order(id={self.order_id}, {self.side.name} {self.leaves_qty}"
                f"/{self.qty} @ {self.price/10000:.4f}, {self.status.name})")


@dataclass
class Fill:
    """
    Represents a matched trade between an aggressor and a resting order.
    """
    fill_id:       int
    symbol:        str
    aggressor_id:  int
    passive_id:    int
    side:          Side           # aggressor side
    price:         int            # fixed-point, always passive order's price
    qty:           int
    timestamp:     int

    def __repr__(self) -> str:
        return (f"Fill(#{self.fill_id} {self.side.name} {self.qty}"
                f" @ {self.price/10000:.4f},"
                f" agg={self.aggressor_id} pas={self.passive_id})")


@dataclass
class CancelRequest:
    order_id: int
    symbol:   str
    timestamp: int = field(default_factory=lambda: time.time_ns())


@dataclass
class AmendRequest:
    """
    Price amendment. Qty-down preserves time priority; qty-up or price change
    loses time priority (new timestamp assigned).
    """
    order_id:  int
    symbol:    str
    new_price: Optional[int] = None
    new_qty:   Optional[int] = None
    timestamp: int = field(default_factory=lambda: time.time_ns())


@dataclass
class BookLevel:
    """Aggregated view of a single price level."""
    price:      int
    total_qty:  int
    order_count: int

    def __repr__(self) -> str:
        return f"{self.price/10000:.4f} x {self.total_qty} ({self.order_count} orders)"


@dataclass
class BookSnapshot:
    """Full L2 snapshot of the order book at a point in time."""
    symbol:    str
    timestamp: int
    bids:      list[BookLevel]   # sorted descending by price
    asks:      list[BookLevel]   # sorted ascending by price

    @property
    def best_bid(self) -> Optional[int]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[int]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def __repr__(self) -> str:
        bid_str = f"{self.best_bid/10000:.4f}" if self.best_bid else "---"
        ask_str = f"{self.best_ask/10000:.4f}" if self.best_ask else "---"
        spread  = f"{self.spread/10000:.4f}" if self.spread else "---"
        return f"Book({self.symbol} bid={bid_str} ask={ask_str} spread={spread})"
