#pragma once

#include <string>
#include <cstdint>
#include <chrono>

namespace execution {

// Type aliases for clarity and precision
using Price = double;
using Quantity = int64_t;
using OrderId = uint64_t;
using VenueId = std::string;
using Timestamp = std::chrono::microseconds;

// Order side
enum class Side {
    BUY,
    SELL
};

// Order type
enum class OrderType {
    LIMIT,
    MARKET
};

// Order status
enum class OrderStatus {
    PENDING,
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED
};

// Execution strategy type
enum class StrategyType {
    TWAP,
    VWAP,
    POV,
    AGGRESSIVE_LIQUIDITY_SEEKER
};

// Basic order structure
struct Order {
    OrderId id;
    Side side;
    OrderType type;
    Price price;  // 0 for market orders
    Quantity quantity;
    Quantity filled_quantity;
    OrderStatus status;
    VenueId venue_id;
    Timestamp submit_time;
    Timestamp last_update_time;
    
    Order() : id(0), side(Side::BUY), type(OrderType::LIMIT), 
              price(0.0), quantity(0), filled_quantity(0),
              status(OrderStatus::PENDING), submit_time(0), last_update_time(0) {}
    
    Quantity remaining_quantity() const {
        return quantity - filled_quantity;
    }
    
    bool is_complete() const {
        return status == OrderStatus::FILLED || 
               status == OrderStatus::CANCELLED ||
               status == OrderStatus::REJECTED;
    }
};

// Trade execution record
struct Fill {
    OrderId order_id;
    VenueId venue_id;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
    Side side;
    
    Fill(OrderId oid, const VenueId& vid, Price p, Quantity q, Timestamp ts, Side s)
        : order_id(oid), venue_id(vid), price(p), quantity(q), timestamp(ts), side(s) {}
};

// Market data quote
struct Quote {
    VenueId venue_id;
    Price bid_price;
    Price ask_price;
    Quantity bid_size;
    Quantity ask_size;
    Timestamp timestamp;
    
    Quote() : bid_price(0.0), ask_price(0.0), bid_size(0), ask_size(0), timestamp(0) {}
    
    Price mid_price() const {
        return (bid_price + ask_price) / 2.0;
    }
    
    Price spread() const {
        return ask_price - bid_price;
    }
};

// Parent order specification
struct ParentOrder {
    OrderId id;
    Side side;
    Quantity total_quantity;
    Quantity filled_quantity;
    StrategyType strategy;
    Timestamp start_time;
    Timestamp end_time;  // Execution horizon
    double urgency;  // 0.0 = passive, 1.0 = aggressive
    
    ParentOrder() : id(0), side(Side::BUY), total_quantity(0), 
                    filled_quantity(0), strategy(StrategyType::TWAP),
                    start_time(0), end_time(0), urgency(0.5) {}
    
    Quantity remaining_quantity() const {
        return total_quantity - filled_quantity;
    }
    
    bool is_complete() const {
        return filled_quantity >= total_quantity;
    }
};

} // namespace execution
