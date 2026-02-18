#pragma once

#include "types.hpp"
#include <map>
#include <queue>
#include <vector>
#include <memory>

namespace execution {

// Price level in order book
struct PriceLevel {
    Price price;
    std::queue<std::shared_ptr<Order>> orders;  // FIFO queue for price-time priority
    Quantity total_quantity;
    
    explicit PriceLevel(Price p) : price(p), total_quantity(0) {}
    
    void add_order(std::shared_ptr<Order> order) {
        orders.push(order);
        total_quantity += order->remaining_quantity();
    }
    
    Quantity remove_quantity(Quantity qty) {
        Quantity removed = 0;
        while (!orders.empty() && removed < qty) {
            auto& order = orders.front();
            Quantity available = order->remaining_quantity();
            Quantity to_fill = std::min(available, qty - removed);
            
            order->filled_quantity += to_fill;
            removed += to_fill;
            total_quantity -= to_fill;
            
            if (order->remaining_quantity() == 0) {
                order->status = OrderStatus::FILLED;
                orders.pop();
            } else {
                order->status = OrderStatus::PARTIALLY_FILLED;
            }
        }
        return removed;
    }
};

// Central Limit Order Book
class OrderBook {
public:
    explicit OrderBook(const VenueId& venue_id);
    
    // Add order to book
    void add_order(std::shared_ptr<Order> order);
    
    // Match incoming order against book
    std::vector<Fill> match_order(std::shared_ptr<Order> order, Timestamp current_time);
    
    // Get best bid/ask
    Quote get_quote(Timestamp current_time) const;
    
    // Get depth at price level
    Quantity get_depth_at_price(Side side, Price price) const;
    
    // Get total depth for N levels
    Quantity get_total_depth(Side side, int levels) const;
    
    // Estimate queue position for new order
    int estimate_queue_position(Side side, Price price) const;
    
    // Get venue ID
    const VenueId& get_venue_id() const { return venue_id_; }
    
    // Clear book (for testing)
    void clear();

private:
    VenueId venue_id_;
    
    // Bids: highest price first (descending)
    std::map<Price, PriceLevel, std::greater<Price>> bids_;
    
    // Asks: lowest price first (ascending)
    std::map<Price, PriceLevel, std::less<Price>> asks_;
    
    // Helper to get best price
    Price get_best_bid() const;
    Price get_best_ask() const;
    
    // Helper to get price levels map
    std::map<Price, PriceLevel, std::greater<Price>>& get_side_book(Side side);
    const std::map<Price, PriceLevel, std::greater<Price>>& get_bids() const { return bids_; }
    const std::map<Price, PriceLevel, std::less<Price>>& get_asks() const { return asks_; }
};

} // namespace execution
