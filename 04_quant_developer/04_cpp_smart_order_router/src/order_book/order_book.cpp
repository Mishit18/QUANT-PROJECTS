#include "order_book.hpp"
#include <algorithm>
#include <limits>

namespace execution {

OrderBook::OrderBook(const VenueId& venue_id) : venue_id_(venue_id) {}

void OrderBook::add_order(std::shared_ptr<Order> order) {
    if (order->type != OrderType::LIMIT) {
        return;  // Only limit orders rest in book
    }
    
    order->status = OrderStatus::SUBMITTED;
    
    if (order->side == Side::BUY) {
        auto it = bids_.find(order->price);
        if (it == bids_.end()) {
            bids_.emplace(order->price, PriceLevel(order->price));
            it = bids_.find(order->price);
        }
        it->second.add_order(order);
    } else {
        auto it = asks_.find(order->price);
        if (it == asks_.end()) {
            asks_.emplace(order->price, PriceLevel(order->price));
            it = asks_.find(order->price);
        }
        it->second.add_order(order);
    }
}

std::vector<Fill> OrderBook::match_order(std::shared_ptr<Order> order, Timestamp current_time) {
    std::vector<Fill> fills;
    
    if (order->side == Side::BUY) {
        // Match against asks
        while (!asks_.empty() && order->remaining_quantity() > 0) {
            auto& level = asks_.begin()->second;
            
            // Check if price crosses
            if (order->type == OrderType::LIMIT && order->price < level.price) {
                break;  // No more matches possible
            }
            
            Quantity to_fill = std::min(order->remaining_quantity(), level.total_quantity);
            Quantity filled = level.remove_quantity(to_fill);
            
            order->filled_quantity += filled;
            fills.emplace_back(order->id, venue_id_, level.price, filled, current_time, order->side);
            
            // Remove empty level
            if (level.total_quantity == 0) {
                asks_.erase(asks_.begin());
            }
        }
    } else {
        // Match against bids
        while (!bids_.empty() && order->remaining_quantity() > 0) {
            auto& level = bids_.begin()->second;
            
            // Check if price crosses
            if (order->type == OrderType::LIMIT && order->price > level.price) {
                break;  // No more matches possible
            }
            
            Quantity to_fill = std::min(order->remaining_quantity(), level.total_quantity);
            Quantity filled = level.remove_quantity(to_fill);
            
            order->filled_quantity += filled;
            fills.emplace_back(order->id, venue_id_, level.price, filled, current_time, order->side);
            
            // Remove empty level
            if (level.total_quantity == 0) {
                bids_.erase(bids_.begin());
            }
        }
    }
    
    // Update order status
    if (order->remaining_quantity() == 0) {
        order->status = OrderStatus::FILLED;
    } else if (order->filled_quantity > 0) {
        order->status = OrderStatus::PARTIALLY_FILLED;
    }
    
    return fills;
}

Quote OrderBook::get_quote(Timestamp current_time) const {
    Quote quote;
    quote.venue_id = venue_id_;
    quote.timestamp = current_time;
    
    if (!bids_.empty()) {
        const auto& best_bid = bids_.begin()->second;
        quote.bid_price = best_bid.price;
        quote.bid_size = best_bid.total_quantity;
    }
    
    if (!asks_.empty()) {
        const auto& best_ask = asks_.begin()->second;
        quote.ask_price = best_ask.price;
        quote.ask_size = best_ask.total_quantity;
    }
    
    return quote;
}

Quantity OrderBook::get_depth_at_price(Side side, Price price) const {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        return (it != bids_.end()) ? it->second.total_quantity : 0;
    } else {
        auto it = asks_.find(price);
        return (it != asks_.end()) ? it->second.total_quantity : 0;
    }
}

Quantity OrderBook::get_total_depth(Side side, int levels) const {
    Quantity total = 0;
    int count = 0;
    
    if (side == Side::BUY) {
        for (const auto& [price, level] : bids_) {
            if (count >= levels) break;
            total += level.total_quantity;
            count++;
        }
    } else {
        for (const auto& [price, level] : asks_) {
            if (count >= levels) break;
            total += level.total_quantity;
            count++;
        }
    }
    
    return total;
}

int OrderBook::estimate_queue_position(Side side, Price price) const {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            return static_cast<int>(it->second.orders.size());
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            return static_cast<int>(it->second.orders.size());
        }
    }
    return 0;
}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
}

Price OrderBook::get_best_bid() const {
    return bids_.empty() ? 0.0 : bids_.begin()->first;
}

Price OrderBook::get_best_ask() const {
    return asks_.empty() ? std::numeric_limits<Price>::max() : asks_.begin()->first;
}

} // namespace execution
