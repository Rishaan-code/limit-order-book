"""
orderbook — A high-performance limit order book engine with backtesting.

Core components:
    OrderBook     — single-symbol matching engine
    Exchange      — multi-symbol router
    Strategy      — abstract base for trading strategies
    BacktestEngine — strategy evaluation framework
    SyntheticMarket — GBM-driven synthetic order flow

Quick start:
    from orderbook import Exchange, Side, OrderType

    ex = Exchange()
    ex.add_symbol("AAPL")
    result = ex.submit("AAPL", Side.BID, OrderType.LIMIT, qty=100, price=150.00)
    print(result)
    print(ex.display("AAPL"))
"""

from .types import (
    Side, OrderType, OrderStatus, TimeInForce,
    Order, Fill, CancelRequest, AmendRequest,
    BookLevel, BookSnapshot,
)
from .price_level import PriceLevel
from .engine import OrderBook, MatchResult
from .exchange import Exchange
from .backtest import (
    Strategy, BacktestEngine, PerformanceReport,
    MidPriceMeanReversion, SpreadCapture, VWAPExecution,
)
from .market_data import SyntheticMarket, MarketDataConfig, load_binance_csv

__all__ = [
    "Side", "OrderType", "OrderStatus", "TimeInForce",
    "Order", "Fill", "CancelRequest", "AmendRequest",
    "BookLevel", "BookSnapshot",
    "PriceLevel", "OrderBook", "MatchResult",
    "Exchange",
    "Strategy", "BacktestEngine", "PerformanceReport",
    "MidPriceMeanReversion", "SpreadCapture", "VWAPExecution",
    "SyntheticMarket", "MarketDataConfig", "load_binance_csv",
]
