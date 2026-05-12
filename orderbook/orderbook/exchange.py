"""
orderbook/exchange.py

A multi-symbol exchange that routes orders to per-symbol OrderBook instances.
Maintains a global fill log and provides exchange-level statistics.

In production, each symbol's book would run in its own thread or process.
Here they share a single thread for simplicity -- the design makes parallelism
straightforward since books share no mutable state.
"""

from __future__ import annotations
import time
from typing import Optional, Callable

from .engine import OrderBook, MatchResult
from .types import (
    Side, OrderType, TimeInForce, Fill,
    CancelRequest, AmendRequest, BookSnapshot,
)


class Exchange:
    """
    Routes orders to the correct OrderBook by symbol.
    Maintains a global chronological fill log for backtesting.
    Supports pluggable fill callbacks (e.g. for strategy notification).
    """

    def __init__(self) -> None:
        self._books:         dict[str, OrderBook] = {}
        self._fill_log:      list[Fill] = []
        self._fill_callbacks: list[Callable[[Fill], None]] = []

    # ── symbol management ──────────────────────────────────────────────────

    def add_symbol(self, symbol: str) -> OrderBook:
        if symbol not in self._books:
            self._books[symbol] = OrderBook(symbol)
        return self._books[symbol]

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        return self._books.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._books.keys())

    # ── order entry ────────────────────────────────────────────────────────

    def submit(self, symbol: str, side: Side, order_type: OrderType,
               qty: int, price: float = 0.0,
               tif: TimeInForce = TimeInForce.GTC,
               order_id: Optional[int] = None,
               timestamp: Optional[int] = None) -> MatchResult:
        book = self._books.get(symbol)
        if book is None:
            raise KeyError(f"Unknown symbol: {symbol}. "
                           f"Call add_symbol('{symbol}') first.")

        result = book.submit(side, order_type, qty, price, tif,
                             order_id, timestamp)
        self._record_fills(result.fills)
        return result

    def cancel(self, symbol: str, order_id: int,
               timestamp: Optional[int] = None) -> Optional[object]:
        book = self._books.get(symbol)
        if book is None:
            return None
        req = CancelRequest(order_id, symbol,
                            timestamp or time.time_ns())
        return book.cancel(req)

    def amend(self, symbol: str, order_id: int,
              new_price: Optional[float] = None,
              new_qty:   Optional[int]   = None,
              timestamp: Optional[int]   = None) -> Optional[object]:
        book = self._books.get(symbol)
        if book is None:
            return None
        price_int = (OrderBook.price_to_int(new_price)
                     if new_price is not None else None)
        req = AmendRequest(order_id, symbol, price_int, new_qty,
                           timestamp or time.time_ns())
        return book.amend(req)

    # ── market data ────────────────────────────────────────────────────────

    def snapshot(self, symbol: str, depth: int = 10) -> Optional[BookSnapshot]:
        book = self._books.get(symbol)
        return book.snapshot(depth) if book else None

    def best_bid(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        if book is None or book.best_bid is None:
            return None
        return OrderBook.int_to_price(book.best_bid)

    def best_ask(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        if book is None or book.best_ask is None:
            return None
        return OrderBook.int_to_price(book.best_ask)

    def mid_price(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        return book.mid_price / 2 if book and book.mid_price else None

    # ── fill log ───────────────────────────────────────────────────────────

    @property
    def fill_log(self) -> list[Fill]:
        return self._fill_log

    def register_fill_callback(self, cb: Callable[[Fill], None]) -> None:
        """Register a function to be called on every fill."""
        self._fill_callbacks.append(cb)

    def _record_fills(self, fills: list[Fill]) -> None:
        for fill in fills:
            self._fill_log.append(fill)
            for cb in self._fill_callbacks:
                cb(fill)

    # ── stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            sym: {
                "total_orders":    book.total_orders,
                "total_fills":     book.total_fills,
                "total_cancelled": book.total_cancelled,
                "total_volume":    book.total_volume,
                "best_bid":        (OrderBook.int_to_price(book.best_bid)
                                    if book.best_bid else None),
                "best_ask":        (OrderBook.int_to_price(book.best_ask)
                                    if book.best_ask else None),
            }
            for sym, book in self._books.items()
        }

    def display(self, symbol: str, depth: int = 5) -> str:
        book = self._books.get(symbol)
        return book.display(depth) if book else f"Unknown symbol: {symbol}"
