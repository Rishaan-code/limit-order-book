"""
orderbook/price_level.py

A PriceLevel holds all orders resting at a single price point.
Orders within a level are matched in FIFO (time-priority) order.

Data structure choice: collections.deque for O(1) append and popleft,
plus a dict for O(1) cancel by order_id. This is the standard production
approach — a pure deque gives O(n) cancellation, a pure dict loses FIFO
ordering. The hybrid gives O(1) for all hot paths.

Memory layout is intentionally compact: we store Order references, not copies.
"""

from __future__ import annotations
from collections import deque
from typing import Iterator, Optional
from .types import Order, Fill, Side
import itertools


class PriceLevel:
    """
    Ordered queue of resting orders at a single price.

    Invariants maintained:
    - Orders are dequeued in arrival (time) order
    - Cancelled orders are lazily removed on dequeue (tombstone approach)
      to avoid O(n) removal from the middle of the deque
    - total_qty is always consistent with sum of active order leaves_qty
    """

    __slots__ = ("price", "side", "_queue", "_active", "total_qty", "order_count")

    def __init__(self, price: int, side: Side) -> None:
        self.price       = price
        self.side        = side
        self._queue: deque[Order] = deque()
        self._active: dict[int, Order] = {}   # order_id -> Order
        self.total_qty   = 0
        self.order_count = 0

    # ── public interface ───────────────────────────────────────────────────

    def add(self, order: Order) -> None:
        """Add a resting order. O(1)."""
        assert order.price == self.price
        assert order.side  == self.side
        self._queue.append(order)
        self._active[order.order_id] = order
        self.total_qty   += order.leaves_qty
        self.order_count += 1

    def cancel(self, order_id: int) -> Optional[Order]:
        """
        Mark order as cancelled. O(1) via tombstone — the deque is NOT
        modified; the cancelled order will be skipped at match time.
        Returns the cancelled order or None if not found.
        """
        order = self._active.pop(order_id, None)
        if order is None:
            return None
        self.total_qty   -= order.leaves_qty
        self.order_count -= 1
        return order

    def match(self, aggressor_qty: int, fill_id_counter: Iterator[int],
              symbol: str, timestamp: int, aggressor_id: int,
              aggressor_side: Side) -> list[Fill]:
        """
        Match aggressor_qty against resting orders in FIFO order.
        Returns list of Fill objects. Updates resting order fill state.
        Lazy-purges tombstoned (cancelled/filled) orders from the front.
        """
        fills: list[Fill] = []
        remaining = aggressor_qty

        while remaining > 0 and self._queue:
            passive = self._queue[0]

            # lazy tombstone removal
            if passive.order_id not in self._active:
                self._queue.popleft()
                continue
            if not passive.is_active:
                self._queue.popleft()
                self._active.pop(passive.order_id, None)
                continue

            fill_qty = min(remaining, passive.leaves_qty)
            fill = Fill(
                fill_id      = next(fill_id_counter),
                symbol       = symbol,
                aggressor_id = aggressor_id,
                passive_id   = passive.order_id,
                side         = aggressor_side,
                price        = self.price,
                qty          = fill_qty,
                timestamp    = timestamp,
            )
            fills.append(fill)

            # update passive order
            passive.filled_qty += fill_qty
            passive.fills.append(fill)
            self.total_qty -= fill_qty

            if passive.leaves_qty == 0:
                from .types import OrderStatus
                passive.status = OrderStatus.FILLED
                self._queue.popleft()
                self._active.pop(passive.order_id)
                self.order_count -= 1
            else:
                from .types import OrderStatus
                passive.status = OrderStatus.PARTIALLY_FILLED

            remaining -= fill_qty

        return fills

    def peek(self) -> Optional[Order]:
        """Return the first active order without removing it. O(1) amortized."""
        while self._queue:
            front = self._queue[0]
            if front.order_id in self._active and front.is_active:
                return front
            self._queue.popleft()
            self._active.pop(front.order_id, None)
        return None

    def is_empty(self) -> bool:
        return self.order_count == 0

    def __len__(self) -> int:
        return self.order_count

    def __repr__(self) -> str:
        return (f"PriceLevel(price={self.price/10000:.4f}, "
                f"qty={self.total_qty}, orders={self.order_count})")
