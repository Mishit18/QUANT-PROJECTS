# Matching Engine

## Overview

The matching engine implements a continuous limit order book (CLOB) with strict price-time priority, the standard for most modern electronic exchanges.

## Matching Rules

### Price-Time Priority

1. **Price priority**: Orders at better prices match first
   - Bids: Higher price has priority
   - Asks: Lower price has priority

2. **Time priority**: At the same price, earlier orders match first
   - FIFO queue at each price level
   - Order timestamp determines position

### Example

```
Order Book:
BID                     ASK
100 @ 150.02 (t=1)     100 @ 150.05 (t=3)
200 @ 150.01 (t=2)     150 @ 150.06 (t=4)
150 @ 150.00 (t=5)     200 @ 150.07 (t=6)

Incoming: SELL 250 @ 150.01

Matches:
1. 100 @ 150.02 (best bid, full fill)
2. 150 @ 150.01 (next best, partial fill of 200)

Result:
- Incoming order: 250 filled (100 + 150)
- Bid @ 150.02: Fully filled, removed
- Bid @ 150.01: Partially filled, 50 remaining
```

## Order Lifecycle

### States

```
PENDING → ACTIVE → PARTIALLY_FILLED → FILLED
                ↓
            CANCELLED
```

- **PENDING**: Order received, not yet in book
- **ACTIVE**: Resting in order book
- **PARTIALLY_FILLED**: Some quantity filled, remainder active
- **FILLED**: Fully executed, removed from book
- **CANCELLED**: User-requested cancellation

### Transitions

- Submit → ACTIVE (if no immediate match)
- Submit → PARTIALLY_FILLED (if partial match)
- Submit → FILLED (if full match)
- ACTIVE → PARTIALLY_FILLED (on partial fill)
- PARTIALLY_FILLED → FILLED (on final fill)
- ACTIVE/PARTIALLY_FILLED → CANCELLED (on cancel request)

## Operations

### Submit Order

```cpp
std::vector<Fill> submitOrder(const Order& order);
```

1. Check for immediate matches against opposite side
2. Execute all possible matches (price-time priority)
3. If quantity remains, add to book at limit price
4. Return vector of fills (both sides)

**Complexity**: O(log m + k) where m = price levels, k = matches

### Cancel Order

```cpp
bool cancelOrder(uint64_t order_id);
```

1. Locate order by ID (O(1) hash lookup)
2. Remove from price level (O(p) where p = orders at level)
3. Remove empty price levels
4. Mark order as CANCELLED

**Complexity**: O(log m + p)

### Amend Order

```cpp
bool amendOrder(uint64_t order_id, double new_price, uint64_t new_quantity);
```

1. Remove order from book (loses time priority)
2. Create new order with amended parameters
3. Resubmit (goes to back of queue at new price)

**Note**: In production, amends may preserve time priority for quantity reductions only.

**Complexity**: O(log m + p + k) where k = potential matches

## Matching Algorithm

### Buy Order Matching

```cpp
void matchBuyOrder(Order& order, OrderBook& book) {
    while (order.remaining_quantity > 0) {
        auto best_ask = book.getBestAsk();
        if (!best_ask || order.price < best_ask->price) {
            break;  // No match possible
        }
        
        auto ask_order = best_ask->getFrontOrder();
        uint64_t fill_qty = min(order.remaining_quantity,
                                ask_order->remaining_quantity);
        
        // Execute at passive order price (ask price)
        executeTrade(order, ask_order, best_ask->price, fill_qty);
        
        if (ask_order->remaining_quantity == 0) {
            best_ask->popFront();
        }
    }
}
```

### Sell Order Matching

Symmetric to buy order matching, but checks against bids.

## Trade Execution

### Price Determination

- Trades execute at the **passive order's price**
- Aggressive order "crosses the spread"
- Example:
  - Resting ask @ 150.05
  - Incoming bid @ 150.10
  - Trade executes @ 150.05 (ask price)

### Fill Generation

Both sides of the trade receive fill notifications:

```cpp
Fill {
    order_id: 123,
    price: 150.05,
    quantity: 100,
    timestamp: 1000000
}
```

### Partial Fills

Orders can be filled across multiple matches:

```
Incoming: BUY 500 @ 150.05
Book: SELL 200 @ 150.04, SELL 300 @ 150.05

Fills:
1. 200 @ 150.04 (full fill of first ask)
2. 300 @ 150.05 (full fill of second ask)

Result: Incoming order fully filled in 2 trades
```

## Order Book Structure

### Price Levels

```cpp
std::map<double, PriceLevel> bids_;  // Descending order
std::map<double, PriceLevel> asks_;  // Ascending order
```

- Bids: Highest price first (reverse iterator for best bid)
- Asks: Lowest price first (forward iterator for best ask)

### FIFO Queue at Each Level

```cpp
class PriceLevel {
    double price_;
    std::deque<Order> orders_;  // FIFO
    uint64_t total_quantity_;
};
```

- Orders at same price maintained in time order
- Front of deque = earliest order (best time priority)
- Back of deque = latest order

### Fast Order Lookup

```cpp
std::unordered_map<uint64_t, Order*> order_map_;
```

- O(1) average case lookup by order ID
- Required for cancels and amends
- Pointer to order in price level deque

## Edge Cases

### Self-Trade Prevention

Not implemented in basic version. Production systems typically:
- Reject orders that would match own orders
- Cancel resting order before matching
- Decrement both sides (no trade)

### Minimum Quantity

Not implemented. Some exchanges support:
- Minimum execution size
- All-or-none orders
- Fill-or-kill orders

### Price Bands

Not implemented. Production systems have:
- Circuit breakers (halt trading)
- Price collars (reject out-of-band orders)
- Reference price checks

## Performance Optimization

### Current Implementation

- Price level lookup: O(log m) using `std::map`
- Order insertion: O(1) at price level (deque push_back)
- Order cancellation: O(p) linear scan at level
- Matching: O(k) where k = number of matches

### Potential Improvements

1. **Intrusive Linked Lists**
   - O(1) order cancellation
   - Requires custom memory management

2. **Skip List for Price Levels**
   - Better cache locality than tree
   - O(log m) average case, simpler than tree

3. **Lock-Free Order Book**
   - Parallel matching across symbols
   - Complex, requires careful design

4. **Memory Pools**
   - Reduce allocation overhead
   - Pre-allocate order objects

## Testing

### Unit Tests

```cpp
TEST(MatchingEngine, PriceTimePriority) {
    // Submit orders at same price
    // Verify FIFO matching
}

TEST(MatchingEngine, PartialFill) {
    // Submit large order
    // Match against multiple small orders
    // Verify quantities
}

TEST(MatchingEngine, CancelOrder) {
    // Submit and cancel
    // Verify removal from book
}
```

### Property-Based Tests

- Invariant: Total quantity in book = sum of order quantities
- Invariant: Best bid < best ask (no crossed book)
- Invariant: Orders at level are time-ordered

## Comparison to Real Exchanges

### NASDAQ

- Price-time priority: ✓
- Partial fills: ✓
- Order types: Limit only (missing market, stop, etc.)
- Hidden orders: ✗
- Routing: ✗

### CME

- Price-time priority: ✓ (some products)
- Pro-rata matching: ✗ (some products use this)
- Implied orders: ✗
- Spread trading: ✗

### Crypto Exchanges

- Price-time priority: ✓
- Maker-taker fees: ✗
- Post-only orders: ✗
- Self-trade prevention: ✗

## Conclusion

This matching engine implements the core logic of a modern exchange:
- Correct price-time priority
- Partial fill support
- Deterministic matching
- Microsecond-level performance

It is suitable for research and education, but lacks many production features (order types, risk checks, self-trade prevention, etc.).
