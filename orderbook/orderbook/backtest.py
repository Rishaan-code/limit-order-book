"""
orderbook/backtest.py

A backtesting framework for evaluating trading strategies against
historical or synthetic order book data.

Architecture:
- Strategy: abstract base with on_book_update() callback
- BacktestEngine: replays events, calls strategy, tracks P&L
- PerformanceReport: Sharpe, fill rate, slippage, drawdown

P&L accounting:
- Realized P&L: from closed positions (matched buy/sell pairs)
- Unrealized P&L: mark-to-market at current mid price
- Total P&L = realized + unrealized
- Slippage: difference between arrival mid-price and avg fill price
"""

from __future__ import annotations
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .exchange import Exchange
from .engine import OrderBook, MatchResult
from .types import (
    Side, OrderType, TimeInForce, Fill, BookSnapshot,
)


# ── Strategy interface ─────────────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract base for all strategies.
    Subclasses implement on_book_update() and optionally on_fill().
    Strategies interact with the market via self.exchange.
    """

    def __init__(self, symbol: str, exchange: Exchange) -> None:
        self.symbol   = symbol
        self.exchange = exchange
        self.position = 0          # net position (positive = long)
        self.cash     = 0.0        # running cash P&L from fills
        self._pending_orders: dict[int, tuple[Side, float]] = {}
                                   # order_id -> (side, arrival_mid)

    @abstractmethod
    def on_book_update(self, snapshot: BookSnapshot) -> None:
        """Called on every book update. Strategy submits orders here."""
        ...

    def on_fill(self, fill: Fill, is_aggressor: bool) -> None:
        """Called when one of our orders is filled. Override optionally."""
        pass

    def buy(self, qty: int, price: float,
            order_type: OrderType = OrderType.LIMIT,
            tif: TimeInForce = TimeInForce.GTC) -> MatchResult:
        snap = self.exchange.snapshot(self.symbol, depth=1)
        arrival_mid = snap.mid_price if snap else price
        result = self.exchange.submit(self.symbol, Side.BID, order_type,
                                      qty, price, tif)
        if arrival_mid:
            self._pending_orders[result.order.order_id] = (Side.BID,
                                                            arrival_mid)
        self._process_result(result)
        return result

    def sell(self, qty: int, price: float,
             order_type: OrderType = OrderType.LIMIT,
             tif: TimeInForce = TimeInForce.GTC) -> MatchResult:
        snap = self.exchange.snapshot(self.symbol, depth=1)
        arrival_mid = snap.mid_price if snap else price
        result = self.exchange.submit(self.symbol, Side.ASK, order_type,
                                      qty, price, tif)
        if arrival_mid:
            self._pending_orders[result.order.order_id] = (Side.ASK,
                                                            arrival_mid)
        self._process_result(result)
        return result

    def cancel(self, order_id: int) -> None:
        self.exchange.cancel(self.symbol, order_id)
        self._pending_orders.pop(order_id, None)

    def _process_result(self, result: MatchResult) -> None:
        for fill in result.fills:
            is_agg = fill.aggressor_id == result.order.order_id
            price  = OrderBook.int_to_price(fill.price)
            if fill.side == Side.BID:
                self.position += fill.qty
                self.cash     -= price * fill.qty
            else:
                self.position -= fill.qty
                self.cash     += price * fill.qty
            self.on_fill(fill, is_agg)

    @property
    def unrealized_pnl(self) -> float:
        snap = self.exchange.snapshot(self.symbol, depth=1)
        if snap is None or snap.mid_price is None:
            return 0.0
        return self.position * OrderBook.int_to_price(int(snap.mid_price))

    @property
    def total_pnl(self) -> float:
        return self.cash + self.unrealized_pnl


# ── Built-in strategies ────────────────────────────────────────────────────

class MidPriceMeanReversion(Strategy):
    """
    Simple mean-reversion strategy.
    Buys when ask is below rolling mid average, sells when bid is above.
    Uses IOC orders to avoid inventory buildup.
    """

    def __init__(self, symbol: str, exchange: Exchange,
                 window: int = 20, threshold: float = 0.0002,
                 order_qty: int = 10) -> None:
        super().__init__(symbol, exchange)
        self.window     = window
        self.threshold  = threshold
        self.order_qty  = order_qty
        self._mid_history: list[float] = []
        self.trade_count = 0

    def on_book_update(self, snapshot: BookSnapshot) -> None:
        if snapshot.mid_price is None:
            return

        mid = OrderBook.int_to_price(int(snapshot.mid_price))
        self._mid_history.append(mid)

        if len(self._mid_history) < self.window:
            return

        avg_mid = sum(self._mid_history[-self.window:]) / self.window

        if snapshot.best_ask is None or snapshot.best_bid is None:
            return

        ask = OrderBook.int_to_price(snapshot.best_ask)
        bid = OrderBook.int_to_price(snapshot.best_bid)

        # buy if ask is sufficiently below average mid
        if ask < avg_mid * (1 - self.threshold) and self.position <= 0:
            self.buy(self.order_qty, ask, OrderType.IOC)
            self.trade_count += 1

        # sell if bid is sufficiently above average mid
        elif bid > avg_mid * (1 + self.threshold) and self.position >= 0:
            self.sell(self.order_qty, bid, OrderType.IOC)
            self.trade_count += 1


class SpreadCapture(Strategy):
    """
    Market-making strategy: post limit orders on both sides of the spread
    to capture the bid-ask spread. Cancels and re-posts on each update.
    Classic passive strategy — profitable in liquid markets, dangerous when
    trending.
    """

    def __init__(self, symbol: str, exchange: Exchange,
                 order_qty: int = 5,
                 max_position: int = 50,
                 edge_ticks: int = 1) -> None:
        super().__init__(symbol, exchange)
        self.order_qty   = order_qty
        self.max_position = max_position
        self.edge_ticks  = edge_ticks   # ticks inside best bid/ask
        self._bid_order_id: Optional[int] = None
        self._ask_order_id: Optional[int] = None
        self.trade_count = 0

    def on_book_update(self, snapshot: BookSnapshot) -> None:
        if snapshot.best_bid is None or snapshot.best_ask is None:
            return

        spread = snapshot.spread
        if spread is None or spread <= 0:
            return

        # cancel existing quotes
        if self._bid_order_id is not None:
            self.cancel(self._bid_order_id)
        if self._ask_order_id is not None:
            self.cancel(self._ask_order_id)

        best_bid = OrderBook.int_to_price(snapshot.best_bid)
        best_ask = OrderBook.int_to_price(snapshot.best_ask)
        tick     = 1 / OrderBook.PRICE_SCALE * self.edge_ticks

        # only post if within position limits
        if self.position > -self.max_position:
            r = self.sell(self.order_qty,
                          best_ask - tick,
                          OrderType.LIMIT, TimeInForce.GTC)
            self._ask_order_id = r.order.order_id

        if self.position < self.max_position:
            r = self.buy(self.order_qty,
                         best_bid + tick,
                         OrderType.LIMIT, TimeInForce.GTC)
            self._bid_order_id = r.order.order_id


class VWAPExecution(Strategy):
    """
    VWAP execution strategy: break a large order into slices and execute
    over time proportional to historical volume distribution.
    Minimizes market impact.
    """

    def __init__(self, symbol: str, exchange: Exchange,
                 total_qty: int, side: Side,
                 num_slices: int = 10) -> None:
        super().__init__(symbol, exchange)
        self.total_qty   = total_qty
        self.exec_side   = side
        self.num_slices  = num_slices
        self.slice_qty   = total_qty // num_slices
        self.slices_done = 0
        self._update_count = 0

    def on_book_update(self, snapshot: BookSnapshot) -> None:
        self._update_count += 1
        # execute one slice every N updates
        trigger_interval = max(1, 100 // self.num_slices)
        if (self._update_count % trigger_interval != 0 or
                self.slices_done >= self.num_slices):
            return

        if snapshot.best_ask is None and self.exec_side == Side.BID:
            return
        if snapshot.best_bid is None and self.exec_side == Side.ASK:
            return

        qty = self.slice_qty
        if self.slices_done == self.num_slices - 1:
            qty = self.total_qty - (self.slice_qty * (self.num_slices - 1))

        if self.exec_side == Side.BID:
            price = OrderBook.int_to_price(snapshot.best_ask)
            self.buy(qty, price, OrderType.IOC)
        else:
            price = OrderBook.int_to_price(snapshot.best_bid)
            self.sell(qty, price, OrderType.IOC)

        self.slices_done += 1


# ── Performance reporting ──────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    strategy_name:  str
    total_pnl:      float
    realized_pnl:   float
    unrealized_pnl: float
    sharpe_ratio:   Optional[float]
    max_drawdown:   float
    total_fills:    int
    fill_rate:      float        # filled qty / submitted qty
    avg_slippage:   float        # bps
    trade_count:    int
    final_position: int

    def __str__(self) -> str:
        sharpe = f"{self.sharpe_ratio:.3f}" if self.sharpe_ratio else "N/A"
        return (
            f"\n{'═'*50}\n"
            f"  Performance Report: {self.strategy_name}\n"
            f"{'═'*50}\n"
            f"  Total P&L:       ${self.total_pnl:>12.4f}\n"
            f"  Realized P&L:    ${self.realized_pnl:>12.4f}\n"
            f"  Unrealized P&L:  ${self.unrealized_pnl:>12.4f}\n"
            f"  Sharpe Ratio:    {sharpe:>13}\n"
            f"  Max Drawdown:    ${self.max_drawdown:>12.4f}\n"
            f"  Total Fills:     {self.total_fills:>13}\n"
            f"  Fill Rate:       {self.fill_rate:>12.1%}\n"
            f"  Avg Slippage:    {self.avg_slippage:>10.2f} bps\n"
            f"  Trade Count:     {self.trade_count:>13}\n"
            f"  Final Position:  {self.final_position:>13}\n"
            f"{'═'*50}\n"
        )


class BacktestEngine:
    """
    Replays a sequence of BookSnapshot events against a strategy,
    then computes a PerformanceReport.
    """

    def __init__(self, strategy: Strategy) -> None:
        self.strategy    = strategy
        self._pnl_series: list[float] = []

    def run(self, snapshots: list[BookSnapshot]) -> PerformanceReport:
        """Feed snapshots to strategy and collect results."""
        submitted_qty = 0
        filled_qty    = 0
        slippage_bps_list: list[float] = []

        # register fill callback to track slippage
        def on_fill(fill: Fill) -> None:
            nonlocal filled_qty
            filled_qty += fill.qty
            # compute slippage vs arrival mid
            entry = self.strategy._pending_orders.get(fill.passive_id)
            if entry:
                side, arrival_mid = entry
                fill_price = OrderBook.int_to_price(fill.price)
                if arrival_mid > 0:
                    if side == Side.BID:
                        slip = (fill_price - arrival_mid) / arrival_mid * 10000
                    else:
                        slip = (arrival_mid - fill_price) / arrival_mid * 10000
                    slippage_bps_list.append(slip)

        self.strategy.exchange.register_fill_callback(on_fill)

        for snap in snapshots:
            self.strategy.on_book_update(snap)
            self._pnl_series.append(self.strategy.total_pnl)

        # compute stats
        sharpe       = self._compute_sharpe(self._pnl_series)
        max_dd       = self._compute_max_drawdown(self._pnl_series)
        fills        = len(self.strategy.exchange.fill_log)
        avg_slip     = (sum(slippage_bps_list) / len(slippage_bps_list)
                        if slippage_bps_list else 0.0)
        fill_rate    = filled_qty / submitted_qty if submitted_qty > 0 else 0.0

        return PerformanceReport(
            strategy_name  = type(self.strategy).__name__,
            total_pnl      = self.strategy.total_pnl,
            realized_pnl   = self.strategy.cash,
            unrealized_pnl = self.strategy.unrealized_pnl,
            sharpe_ratio   = sharpe,
            max_drawdown   = max_dd,
            total_fills    = fills,
            fill_rate       = fill_rate,
            avg_slippage   = avg_slip,
            trade_count    = getattr(self.strategy, "trade_count", 0),
            final_position = self.strategy.position,
        )

    @staticmethod
    def _compute_sharpe(pnl_series: list[float],
                         risk_free: float = 0.0) -> Optional[float]:
        if len(pnl_series) < 2:
            return None
        returns = [pnl_series[i] - pnl_series[i-1]
                   for i in range(1, len(pnl_series))]
        n    = len(returns)
        mean = sum(returns) / n
        var  = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std  = math.sqrt(var)
        if std == 0:
            return None
        return (mean - risk_free) / std * math.sqrt(252 * 6.5 * 3600)

    @staticmethod
    def _compute_max_drawdown(pnl_series: list[float]) -> float:
        if not pnl_series:
            return 0.0
        peak    = pnl_series[0]
        max_dd  = 0.0
        for v in pnl_series:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        return max_dd
