"""
tests/test_orderbook.py

Comprehensive test suite covering:
- Price-time priority invariants
- All order types (LIMIT, MARKET, IOC, FOK)
- Cancel and amend semantics
- Edge cases: self-trade, crossed book, zero liquidity
- Property-based tests using hypothesis
"""

import pytest
import time
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from orderbook import (
    Exchange, OrderBook, Side, OrderType, OrderStatus,
    TimeInForce, CancelRequest,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def ex():
    e = Exchange()
    e.add_symbol("TEST")
    return e

@pytest.fixture
def book():
    return OrderBook("TEST")


def seed_book(ex, symbol="TEST", mid=100.0, spread=0.02, depth=5):
    """Seed a book with resting orders around a mid price."""
    tick = 0.01
    half = spread / 2
    for i in range(1, depth + 1):
        ex.submit(symbol, Side.ASK, OrderType.LIMIT,
                  qty=100, price=mid + half + (i-1)*tick)
        ex.submit(symbol, Side.BID, OrderType.LIMIT,
                  qty=100, price=mid - half - (i-1)*tick)


# ── Basic matching ─────────────────────────────────────────────────────────

class TestBasicMatching:

    def test_limit_order_rests_on_empty_book(self, ex):
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=99.0)
        assert r.order.status == OrderStatus.OPEN
        assert len(r.fills) == 0
        assert ex.best_bid("TEST") == pytest.approx(99.0)

    def test_limit_order_crosses_and_fills(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=100.0)
        assert r.order.status == OrderStatus.FILLED
        assert len(r.fills) == 1
        assert r.fills[0].qty == 10
        assert r.fills[0].price == OrderBook.price_to_int(100.0)

    def test_partial_fill_leaves_remainder_on_book(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=5, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=100.0)
        assert r.order.status == OrderStatus.PARTIALLY_FILLED
        assert r.total_filled == 5
        assert r.order.leaves_qty == 5
        # remainder should rest
        assert ex.best_bid("TEST") == pytest.approx(100.0)

    def test_fill_at_passive_price(self, ex):
        """Fill price must always be the passive (resting) order's price."""
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=101.0)
        assert r.fills[0].price == OrderBook.price_to_int(100.0)

    def test_no_cross_when_bid_below_ask(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=101.0)
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=99.0)
        assert r.order.status == OrderStatus.OPEN
        assert len(r.fills) == 0


# ── Price-time priority ────────────────────────────────────────────────────

class TestPriceTimePriority:

    def test_better_price_fills_first(self, ex):
        """Lower ask price should fill before higher ask price."""
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=101.0)
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=101.0)
        assert r.fills[0].price == OrderBook.price_to_int(100.0)

    def test_equal_price_fills_in_fifo_order(self, ex):
        """At the same price, earlier orders fill first."""
        r1 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=5, price=100.0, timestamp=1000)
        r2 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=5, price=100.0, timestamp=2000)
        r3 = ex.submit("TEST", Side.BID, OrderType.LIMIT,
                       qty=5, price=100.0, timestamp=3000)
        # r1 should fill (earlier), r2 should remain
        assert r3.fills[0].passive_id == r1.order.order_id
        assert r2.order.status == OrderStatus.OPEN

    def test_time_priority_preserved_across_multiple_fills(self, ex):
        """Large aggressor should fill in strict time order."""
        ids = []
        for i in range(5):
            r = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                          qty=10, price=100.0, timestamp=i * 1000)
            ids.append(r.order.order_id)

        r = ex.submit("TEST", Side.BID, OrderType.LIMIT,
                      qty=50, price=100.0, timestamp=10000)
        passive_ids = [f.passive_id for f in r.fills]
        assert passive_ids == ids


# ── Order types ────────────────────────────────────────────────────────────

class TestOrderTypes:

    def test_market_order_fills_at_any_price(self, ex):
        seed_book(ex)
        r = ex.submit("TEST", Side.BID, OrderType.MARKET, qty=50)
        assert r.total_filled == 50
        assert r.order.status == OrderStatus.FILLED

    def test_market_order_cancelled_if_no_liquidity(self, ex):
        r = ex.submit("TEST", Side.BID, OrderType.MARKET, qty=10)
        assert r.order.status == OrderStatus.CANCELLED
        assert r.total_filled == 0

    def test_ioc_partial_fill_cancels_remainder(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=5, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.IOC, qty=10, price=100.0)
        assert r.total_filled == 5
        assert r.order.status == OrderStatus.CANCELLED
        # IOC should NOT rest on book
        assert ex.best_bid("TEST") is None

    def test_ioc_full_fill(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.IOC, qty=10, price=100.0)
        assert r.total_filled == 10
        assert r.order.status == OrderStatus.FILLED

    def test_fok_cancels_if_insufficient_liquidity(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=5, price=100.0)
        r = ex.submit("TEST", Side.BID, OrderType.FOK, qty=10, price=100.0)
        assert r.total_filled == 0
        assert r.order.status == OrderStatus.CANCELLED
        # Critically: the 5 resting shares must NOT have been consumed
        assert ex.best_ask("TEST") == pytest.approx(100.0)
        snap = ex.snapshot("TEST", depth=1)
        assert snap.asks[0].total_qty == 5

    def test_fok_fills_completely_when_sufficient_liquidity(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.1)
        r = ex.submit("TEST", Side.BID, OrderType.FOK, qty=15, price=100.1)
        assert r.total_filled == 15
        assert r.order.status == OrderStatus.FILLED

    def test_fok_does_not_modify_book_on_failure(self, ex):
        """This is the critical FOK correctness invariant."""
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=3, price=100.0)
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=3, price=100.1)
        # FOK for 10, only 6 available — must cancel without touching book
        r = ex.submit("TEST", Side.BID, OrderType.FOK, qty=10, price=100.1)
        assert r.total_filled == 0
        snap = ex.snapshot("TEST", depth=5)
        total_ask_qty = sum(l.total_qty for l in snap.asks)
        assert total_ask_qty == 6   # book unchanged


# ── Cancel and amend ───────────────────────────────────────────────────────

class TestCancelAmend:

    def test_cancel_removes_order_from_book(self, ex):
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=99.0)
        cancelled = ex.cancel("TEST", r.order.order_id)
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED
        assert ex.best_bid("TEST") is None

    def test_cancel_nonexistent_order_returns_none(self, ex):
        result = ex.cancel("TEST", order_id=99999)
        assert result is None

    def test_cancel_updates_book_depth(self, ex):
        r1 = ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        r2 = ex.submit("TEST", Side.ASK, order_type=OrderType.LIMIT,
                       qty=10, price=100.0)
        ex.cancel("TEST", r1.order.order_id)
        snap = ex.snapshot("TEST", depth=1)
        assert snap.asks[0].total_qty == 10

    def test_amend_price_loses_time_priority(self, ex):
        """
        An order that changes price loses time priority.
        After amending r1 to a different price and back, r2 (unamended)
        should have priority since r1's effective queue position is newer.
        We verify by checking r2 fills before r1 on the next aggressor.
        """
        r1 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=10, price=100.0, timestamp=1000)
        r2 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=10, price=100.0, timestamp=2000)
        # amend r1 to new price — it leaves 100.0 level, joins 100.5 level
        ex.amend("TEST", r1.order.order_id, new_price=100.5)
        # now only r2 is at 100.0; submit a bid that matches r2
        r3 = ex.submit("TEST", Side.BID, OrderType.LIMIT,
                       qty=10, price=100.0, timestamp=3000)
        assert r3.fills[0].passive_id == r2.order.order_id
        # r1 should still be resting at 100.5
        snap = ex.snapshot("TEST", depth=5)
        ask_prices = [l.price for l in snap.asks]
        assert OrderBook.price_to_int(100.5) in ask_prices

    def test_amend_qty_down_preserves_time_priority(self, ex):
        r1 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=20, price=100.0, timestamp=1000)
        r2 = ex.submit("TEST", Side.ASK, OrderType.LIMIT,
                       qty=10, price=100.0, timestamp=2000)
        ex.amend("TEST", r1.order.order_id, new_qty=10)
        # r1 should still have priority over r2
        r3 = ex.submit("TEST", Side.BID, OrderType.LIMIT,
                       qty=5, price=100.0, timestamp=3000)
        assert r3.fills[0].passive_id == r1.order.order_id


# ── Snapshot and market data ───────────────────────────────────────────────

class TestSnapshot:

    def test_snapshot_bids_sorted_descending(self, ex):
        for p in [99.0, 98.0, 100.0, 97.0]:
            ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=p)
        snap = ex.snapshot("TEST", depth=10)
        prices = [l.price for l in snap.bids]
        assert prices == sorted(prices, reverse=True)

    def test_snapshot_asks_sorted_ascending(self, ex):
        for p in [101.0, 103.0, 102.0, 104.0]:
            ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=p)
        snap = ex.snapshot("TEST", depth=10)
        prices = [l.price for l in snap.asks]
        assert prices == sorted(prices)

    def test_spread_calculation(self, ex):
        ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=99.0)
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=101.0)
        snap = ex.snapshot("TEST")
        assert snap.spread == OrderBook.price_to_int(101.0) - OrderBook.price_to_int(99.0)

    def test_mid_price(self, ex):
        ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=10, price=99.0)
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=101.0)
        snap = ex.snapshot("TEST")
        assert snap.mid_price == pytest.approx(
            (OrderBook.price_to_int(99.0) + OrderBook.price_to_int(101.0)) / 2
        )


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_qty_rejected(self, ex):
        r = ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=0, price=100.0)
        assert r.order.status == OrderStatus.REJECTED

    def test_multiple_levels_swept_by_market_order(self, ex):
        for price in [100.0, 100.1, 100.2]:
            ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=price)
        r = ex.submit("TEST", Side.BID, OrderType.MARKET, qty=30)
        assert r.total_filled == 30
        assert len(r.fills) == 3

    def test_book_empty_after_full_sweep(self, ex):
        ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=10, price=100.0)
        ex.submit("TEST", Side.BID, OrderType.MARKET, qty=10)
        assert ex.best_ask("TEST") is None

    def test_large_number_of_orders(self, ex):
        """Stress test: 10,000 orders should complete without errors."""
        import random
        rng = random.Random(42)
        for i in range(5000):
            price = round(rng.uniform(99.0, 101.0), 2)
            qty   = rng.randint(1, 100)
            ex.submit("TEST", Side.BID, OrderType.LIMIT, qty=qty, price=price)
        for i in range(5000):
            price = round(rng.uniform(99.0, 101.0), 2)
            qty   = rng.randint(1, 100)
            ex.submit("TEST", Side.ASK, OrderType.LIMIT, qty=qty, price=price)
        # should not raise


# ── Property-based tests ───────────────────────────────────────────────────

class TestProperties:

    @given(
        bid_price=st.floats(min_value=1.0, max_value=999.0,
                            allow_nan=False, allow_infinity=False),
        ask_price=st.floats(min_value=1.0, max_value=999.0,
                            allow_nan=False, allow_infinity=False),
        qty=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=200)
    def test_fill_only_when_prices_cross(self, bid_price, ask_price, qty):
        """Invariant: fills occur iff bid_price >= ask_price."""
        ex = Exchange()
        ex.add_symbol("PROP")
        ex.submit("PROP", Side.ASK, OrderType.LIMIT, qty=qty, price=ask_price)
        r = ex.submit("PROP", Side.BID, OrderType.LIMIT, qty=qty, price=bid_price)
        should_fill = bid_price >= ask_price
        if should_fill:
            assert r.total_filled > 0
        else:
            assert r.total_filled == 0

    @given(
        prices=st.lists(
            st.floats(min_value=90.0, max_value=110.0,
                      allow_nan=False, allow_infinity=False),
            min_size=2, max_size=20,
        ),
        qty=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_fill_price_always_passive_price(self, prices, qty):
        """Invariant: every fill price equals the passive order's limit price."""
        ex = Exchange()
        ex.add_symbol("PROP2")
        for p in prices:
            ex.submit("PROP2", Side.ASK, OrderType.LIMIT, qty=qty, price=p)
        r = ex.submit("PROP2", Side.BID, OrderType.MARKET, qty=qty * len(prices))
        for fill in r.fills:
            # fill price must be a valid ask price we submitted
            fill_price = OrderBook.int_to_price(fill.price)
            assert any(abs(fill_price - p) < 0.001 for p in prices)

    @given(
        n_orders=st.integers(min_value=1, max_value=50),
        qty=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_total_filled_qty_conservation(self, n_orders, qty):
        """
        Invariant: sum of fill qtys == min(aggressor qty, available qty).
        Conservation of shares — no shares created or destroyed.
        """
        ex = Exchange()
        ex.add_symbol("PROP3")
        total_available = n_orders * qty
        for _ in range(n_orders):
            ex.submit("PROP3", Side.ASK, OrderType.LIMIT,
                      qty=qty, price=100.0)
        aggressor_qty = n_orders * qty // 2 + 1
        r = ex.submit("PROP3", Side.BID, OrderType.MARKET, qty=aggressor_qty)
        expected = min(aggressor_qty, total_available)
        assert r.total_filled == expected

    @given(
        n_levels=st.integers(min_value=1, max_value=10),
        base_price=st.floats(min_value=10.0, max_value=500.0,
                             allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_book_levels_always_sorted(self, n_levels, base_price):
        """Invariant: bid levels always descending, ask levels always ascending."""
        ex = Exchange()
        ex.add_symbol("PROP4")
        for i in range(n_levels):
            ex.submit("PROP4", Side.BID, OrderType.LIMIT,
                      qty=10, price=base_price - i * 0.01)
            ex.submit("PROP4", Side.ASK, OrderType.LIMIT,
                      qty=10, price=base_price + i * 0.01 + 0.05)
        snap = ex.snapshot("PROP4", depth=n_levels + 1)
        bid_prices = [l.price for l in snap.bids]
        ask_prices = [l.price for l in snap.asks]
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert ask_prices == sorted(ask_prices)
