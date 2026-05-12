"""
orderbook/engine.py

The limit order book matching engine.

Design decisions documented:
- sortedcontainers.SortedDict for the price ladder: O(log n) insert/delete,
  O(1) best price access. Better than heapq (no O(1) cancel) or plain dict
  (no ordering). In production this would be a custom red-black tree or
  skip list but SortedDict gives equivalent asymptotic complexity in Python.
- Fixed-point integer arithmetic throughout: no floats on the critical path.
- Fill ID and Order ID counters are local itertools.count() — thread-unsafe
  by design. For multi-symbol concurrent books, each engine gets its own
  counters. Shared state requires a lock or atomic counter.
- FOK correctness: we probe available liquidity before touching any state,
  then execute. This is critical — a naive implementation might partially
  fill before discovering the FOK condition cannot be satisfied.
"""

from __future__ import annotations
import itertools
import time
from typing import Optional
from sortedcontainers import SortedDict

from .types import (
    Order, Fill, CancelRequest, AmendRequest,
    Side, OrderType, OrderStatus, TimeInForce,
    BookLevel, BookSnapshot,
)
from .price_level import PriceLevel


class MatchResult:
    """Result of processing a single order through the engine."""
    __slots__ = ("order", "fills", "cancelled", "reject_reason")

    def __init__(self, order: Order, fills: list[Fill],
                 cancelled: bool = False,
                 reject_reason: Optional[str] = None) -> None:
        self.order         = order
        self.fills         = fills
        self.cancelled     = cancelled
        self.reject_reason = reject_reason

    @property
    def total_filled(self) -> int:
        return sum(f.qty for f in self.fills)

    def __repr__(self) -> str:
        return (f"MatchResult(order={self.order.order_id}, "
                f"fills={len(self.fills)}, filled={self.total_filled}, "
                f"cancelled={self.cancelled})")


class OrderBook:
    """
    Single-symbol limit order book with price-time priority matching.

    Supported order types: LIMIT, MARKET, IOC, FOK
    Supported operations: submit, cancel, amend

    Price representation: integer fixed-point with 4 decimal places.
    Use price_to_int(p) and int_to_price(p) for conversion.

    Complexity:
        submit (no match):   O(log n)  — insert into sorted price ladder
        submit (with match): O(k log n) — k fills, each O(log n) for level cleanup
        cancel:              O(log n)  — price level lookup + O(1) tombstone
        amend (price):       O(log n)  — cancel + reinsert
        best bid/ask:        O(1)      — SortedDict peekitem
        snapshot (depth d):  O(d)
    """

    PRICE_SCALE = 10_000   # 4 decimal places

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        # price -> PriceLevel, bids sorted descending, asks ascending
        # SortedDict uses ascending key order by default.
        # For bids we negate the key so highest price = first item.
        self._bids: SortedDict[int, PriceLevel] = SortedDict()   # neg_price -> level
        self._asks: SortedDict[int, PriceLevel] = SortedDict()   # price -> level

        # order_id -> (side, price_key) for O(log n) cancel/amend
        self._order_index: dict[int, tuple[Side, int]] = {}

        # counters — thread-unsafe, intentionally so (see module docstring)
        self._order_id_counter = itertools.count(1)
        self._fill_id_counter  = itertools.count(1)

        # stats
        self.total_orders    = 0
        self.total_fills     = 0
        self.total_cancelled = 0
        self.total_volume    = 0

    # ── public API ─────────────────────────────────────────────────────────

    @classmethod
    def price_to_int(cls, price: float) -> int:
        """Convert decimal price to fixed-point integer."""
        return round(price * cls.PRICE_SCALE)

    @classmethod
    def int_to_price(cls, price_int: int) -> float:
        """Convert fixed-point integer back to decimal price."""
        return price_int / cls.PRICE_SCALE

    def new_order_id(self) -> int:
        return next(self._order_id_counter)

    def submit(self, side: Side, order_type: OrderType, qty: int,
               price: float = 0.0,
               tif: TimeInForce = TimeInForce.GTC,
               order_id: Optional[int] = None,
               timestamp: Optional[int] = None) -> MatchResult:
        """
        Submit a new order. Returns a MatchResult with all generated fills.
        price is ignored for MARKET orders.
        """
        if qty <= 0:
            dummy = self._make_order(order_id, side, order_type,
                                     qty, 0, tif, timestamp or time.time_ns())
            dummy.status = OrderStatus.REJECTED
            return MatchResult(dummy, [], reject_reason="qty must be positive")

        ts        = timestamp or time.time_ns()
        price_int = self.price_to_int(price) if order_type != OrderType.MARKET else 0
        order     = self._make_order(order_id, side, order_type,
                                     qty, price_int, tif, ts)
        self.total_orders += 1

        if order_type == OrderType.MARKET:
            return self._process_market(order)
        elif order_type == OrderType.LIMIT:
            return self._process_limit(order)
        elif order_type == OrderType.IOC:
            return self._process_ioc(order)
        elif order_type == OrderType.FOK:
            return self._process_fok(order)
        else:
            order.status = OrderStatus.REJECTED
            return MatchResult(order, [], reject_reason="unknown order type")

    def cancel(self, request: CancelRequest) -> Optional[Order]:
        """
        Cancel a resting order. Returns the cancelled order or None.
        O(log n) via price-level index lookup.
        """
        entry = self._order_index.pop(request.order_id, None)
        if entry is None:
            return None

        side, price_key = entry
        ladder = self._bids if side == Side.BID else self._asks
        level  = ladder.get(price_key)
        if level is None:
            return None

        order = level.cancel(request.order_id)
        if order is None:
            return None

        order.status = OrderStatus.CANCELLED
        self.total_cancelled += 1

        # prune empty level
        if level.is_empty():
            del ladder[price_key]

        return order

    def amend(self, request: AmendRequest) -> Optional[Order]:
        """
        Amend an order's price or quantity.

        Qty-down: preserves time priority (in-place update).
        Qty-up or price change: loses time priority (cancel + reinsert).

        Returns the updated order or None if not found.
        """
        entry = self._order_index.get(request.order_id)
        if entry is None:
            return None

        side, price_key = entry
        ladder = self._bids if side == Side.BID else self._asks
        level  = ladder.get(price_key)
        if level is None:
            return None

        order = level._active.get(request.order_id)
        if order is None:
            return None

        new_price = request.new_price
        new_qty   = request.new_qty

        # determine if we lose time priority
        price_change = new_price is not None and new_price != order.price
        qty_up       = new_qty is not None and new_qty > order.qty

        if price_change or qty_up:
            # cancel and reinsert with new timestamp
            cancel_req = CancelRequest(order.order_id, order.symbol,
                                       request.timestamp)
            self.cancel(cancel_req)
            order.status = OrderStatus.NEW

            submit_price = self.int_to_price(new_price) if new_price else \
                           self.int_to_price(order.price)
            submit_qty   = new_qty if new_qty else order.leaves_qty

            result = self.submit(
                side       = order.side,
                order_type = order.order_type,
                qty        = submit_qty,
                price      = submit_price,
                tif        = order.tif,
                order_id   = order.order_id,
                timestamp  = request.timestamp,
            )
            return result.order
        else:
            # qty-down: in-place, preserve priority
            if new_qty is not None and new_qty < order.qty:
                delta = order.leaves_qty - (new_qty - order.filled_qty)
                if delta > 0:
                    level.total_qty -= delta
                    order.qty = new_qty
            return order

    def snapshot(self, depth: int = 10) -> BookSnapshot:
        """Return L2 book snapshot up to `depth` levels each side."""
        bids = []
        for _, level in self._bids.items():
            if len(bids) >= depth:
                break
            if not level.is_empty():
                bids.append(BookLevel(level.price, level.total_qty,
                                      level.order_count))

        asks = []
        for _, level in self._asks.items():
            if len(asks) >= depth:
                break
            if not level.is_empty():
                asks.append(BookLevel(level.price, level.total_qty,
                                      level.order_count))

        return BookSnapshot(
            symbol    = self.symbol,
            timestamp = time.time_ns(),
            bids      = bids,
            asks      = asks,
        )

    @property
    def best_bid(self) -> Optional[int]:
        if not self._bids:
            return None
        neg_price, level = self._bids.peekitem(0)
        return level.price

    @property
    def best_ask(self) -> Optional[int]:
        if not self._asks:
            return None
        _, level = self._asks.peekitem(0)
        return level.price

    @property
    def spread(self) -> Optional[int]:
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if bb and ba else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return ((bb + ba) / 2) if bb and ba else None

    # ── matching logic ─────────────────────────────────────────────────────

    def _process_market(self, order: Order) -> MatchResult:
        fills = self._match_against_book(order, limit_price=None)
        if order.leaves_qty > 0:
            # market orders that can't be fully filled are cancelled
            order.status = OrderStatus.CANCELLED
            self.total_cancelled += 1
        self._finalize_order_status(order, fills)
        return MatchResult(order, fills)

    def _process_limit(self, order: Order) -> MatchResult:
        fills = self._match_against_book(order, limit_price=order.price)
        if order.leaves_qty == 0:
            order.status = OrderStatus.FILLED
        elif fills:
            # partially filled — rest remainder on book
            order.status = OrderStatus.PARTIALLY_FILLED
            self._rest_order(order)
        else:
            # no fills — rest on book as OPEN
            self._rest_order(order)
        return MatchResult(order, fills)

    def _process_ioc(self, order: Order) -> MatchResult:
        fills = self._match_against_book(order, limit_price=order.price)
        cancelled = order.leaves_qty > 0
        # IOC always cancels the remainder — override status
        if order.leaves_qty > 0:
            order.status = OrderStatus.CANCELLED
            self.total_cancelled += 1
        else:
            order.status = OrderStatus.FILLED
        return MatchResult(order, fills, cancelled=cancelled)

    def _process_fok(self, order: Order) -> MatchResult:
        """
        FOK: check available liquidity first without touching any state.
        Only execute if the full qty can be filled.
        """
        available = self._available_liquidity(order.side, order.price,
                                               order.qty)
        if available < order.qty:
            order.status = OrderStatus.CANCELLED
            self.total_cancelled += 1
            return MatchResult(order, [], cancelled=True,
                               reject_reason="FOK: insufficient liquidity")

        fills = self._match_against_book(order, limit_price=order.price)
        order.status = OrderStatus.FILLED
        self._finalize_order_status(order, fills)
        return MatchResult(order, fills)

    def _match_against_book(self, aggressor: Order,
                             limit_price: Optional[int]) -> list[Fill]:
        """
        Walk the opposite side of the book and generate fills.
        Stops when qty exhausted or no more price-compatible levels.
        """
        fills: list[Fill] = []
        opposite = self._asks if aggressor.side == Side.BID else self._bids

        while aggressor.leaves_qty > 0 and opposite:
            # peek at best level on opposite side
            if aggressor.side == Side.BID:
                price_key, level = opposite.peekitem(0)
                # bids match asks: ask price <= limit_price (or no limit)
                if limit_price is not None and level.price > limit_price:
                    break
            else:
                price_key, level = opposite.peekitem(0)
                # asks match bids: bid price >= limit_price (or no limit)
                # bids are stored with neg key, so peekitem(0) = highest bid
                if limit_price is not None and level.price < limit_price:
                    break

            level_fills = level.match(
                aggressor_qty  = aggressor.leaves_qty,
                fill_id_counter= self._fill_id_counter,
                symbol         = self.symbol,
                timestamp      = aggressor.timestamp,
                aggressor_id   = aggressor.order_id,
                aggressor_side = aggressor.side,
            )

            for f in level_fills:
                aggressor.filled_qty += f.qty
                aggressor.fills.append(f)
                self.total_volume += f.qty
                self.total_fills  += 1
                # also update order index for passive order (already done in level)

            fills.extend(level_fills)

            # prune empty level
            if level.is_empty():
                del opposite[price_key]
                # remove all (should be zero) orders from index on this level
                for oid in list(self._order_index.keys()):
                    if self._order_index[oid] == (aggressor.side.opposite(),
                                                   price_key):
                        del self._order_index[oid]

        return fills

    def _available_liquidity(self, aggressor_side: Side,
                              limit_price: int, needed: int) -> int:
        """
        Probe how much qty is available at or better than limit_price.
        Used by FOK to check feasibility without modifying book state.
        """
        available = 0
        opposite  = self._asks if aggressor_side == Side.BID else self._bids

        for price_key, level in opposite.items():
            if aggressor_side == Side.BID:
                if level.price > limit_price:
                    break
            else:
                if level.price < limit_price:
                    break
            available += level.total_qty
            if available >= needed:
                return available

        return available

    def _rest_order(self, order: Order) -> None:
        """Place a resting limit order onto the book."""
        # preserve PARTIALLY_FILLED status; only set OPEN for fresh orders
        if order.status == OrderStatus.NEW:
            order.status = OrderStatus.OPEN
        if order.side == Side.BID:
            key    = -order.price   # negate for descending sort
            ladder = self._bids
        else:
            key    = order.price
            ladder = self._asks

        if key not in ladder:
            ladder[key] = PriceLevel(order.price, order.side)

        ladder[key].add(order)
        self._order_index[order.order_id] = (order.side, key)

    def _finalize_order_status(self, order: Order,
                                fills: list[Fill]) -> None:
        if order.filled_qty == 0:
            pass  # status unchanged
        elif order.leaves_qty == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

    def _make_order(self, order_id: Optional[int], side: Side,
                    order_type: OrderType, qty: int, price: int,
                    tif: TimeInForce, timestamp: int) -> Order:
        oid = order_id if order_id is not None else next(self._order_id_counter)
        return Order(
            order_id   = oid,
            symbol     = self.symbol,
            side       = side,
            order_type = order_type,
            qty        = qty,
            price      = price,
            timestamp  = timestamp,
            tif        = tif,
        )

    # ── display ────────────────────────────────────────────────────────────

    def display(self, depth: int = 5) -> str:
        snap   = self.snapshot(depth)
        lines  = [f"\n{'─'*42}",
                  f"  {self.symbol} Order Book",
                  f"{'─'*42}",
                  f"  {'ASKS':>20}",
                  f"{'─'*42}"]

        for level in reversed(snap.asks[:depth]):
            price_str = f"{self.int_to_price(level.price):.4f}"
            lines.append(f"  {price_str:>10}  {level.total_qty:>8}  "
                         f"({level.order_count})")

        spread_str = (f"{self.int_to_price(snap.spread):.4f}"
                      if snap.spread else "---")
        lines.append(f"{'─'*42}")
        lines.append(f"  spread: {spread_str}    mid: "
                     f"{self.int_to_price(snap.mid_price):.4f}"
                     if snap.mid_price else f"  spread: {spread_str}")
        lines.append(f"{'─'*42}")

        for level in snap.bids[:depth]:
            price_str = f"{self.int_to_price(level.price):.4f}"
            lines.append(f"  {price_str:>10}  {level.total_qty:>8}  "
                         f"({level.order_count})")

        lines.append(f"{'─'*42}")
        lines.append(f"  {'BIDS':>20}")
        lines.append(f"{'─'*42}\n")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"OrderBook({self.symbol}, "
                f"bid={self.int_to_price(self.best_bid):.4f} "
                f"ask={self.int_to_price(self.best_ask):.4f})"
                if self.best_bid and self.best_ask
                else f"OrderBook({self.symbol}, empty)")
