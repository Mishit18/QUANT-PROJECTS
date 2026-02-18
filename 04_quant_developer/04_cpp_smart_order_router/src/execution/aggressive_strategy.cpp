#include "strategies/aggressive_strategy.hpp"
#include <algorithm>
#include <cmath>

namespace execution {

AggressiveStrategy::AggressiveStrategy(const Config& config)
    : config_(config), last_order_time_(0) {}

void AggressiveStrategy::initialize(const ParentOrder& parent_order) {
    parent_order_ = parent_order;
    last_order_time_ = parent_order.start_time;
}

std::vector<std::shared_ptr<Order>> AggressiveStrategy::generate_orders(
    Timestamp current_time,
    const std::vector<Quote>& venue_quotes,
    Quantity remaining_quantity
) {
    std::vector<std::shared_ptr<Order>> orders;
    
    if (remaining_quantity == 0) {
        return orders;
    }
    
    // Check minimum time between orders
    Timestamp time_since_last = current_time - last_order_time_;
    if (time_since_last < Timestamp(static_cast<int64_t>(config_.order_interval_us))) {
        return orders;
    }
    
    // Calculate current urgency
    double urgency = calculate_urgency(current_time, remaining_quantity);
    
    // Find best venue
    if (venue_quotes.empty()) {
        return orders;
    }
    
    const Quote* best_quote = &venue_quotes[0];
    for (const auto& quote : venue_quotes) {
        if (parent_order_.side == Side::BUY) {
            if (quote.ask_price < best_quote->ask_price) {
                best_quote = &quote;
            }
        } else {
            if (quote.bid_price > best_quote->bid_price) {
                best_quote = &quote;
            }
        }
    }
    
    // Determine order characteristics based on urgency
    bool cross_spread = should_cross_spread(urgency);
    Quantity order_size = calculate_sweep_size(*best_quote, remaining_quantity);
    
    if (order_size == 0) {
        return orders;
    }
    
    // Create order
    if (cross_spread && config_.allow_market_orders) {
        // Market order to guarantee execution
        auto order = create_child_order(
            parent_order_.side,
            order_size,
            0.0,
            OrderType::MARKET
        );
        orders.push_back(order);
    } else {
        // Aggressive limit order (cross spread but with price protection)
        Price order_price;
        if (parent_order_.side == Side::BUY) {
            // Buy at ask or slightly above
            order_price = best_quote->ask_price * (1.0 + urgency * 0.001);
        } else {
            // Sell at bid or slightly below
            order_price = best_quote->bid_price * (1.0 - urgency * 0.001);
        }
        
        auto order = create_child_order(
            parent_order_.side,
            order_size,
            order_price,
            OrderType::LIMIT
        );
        orders.push_back(order);
    }
    
    last_order_time_ = current_time;
    
    return orders;
}

void AggressiveStrategy::on_fill(const Fill& fill, Timestamp current_time) {
    // Fills increase urgency implicitly by reducing remaining quantity
}

bool AggressiveStrategy::is_complete(Timestamp current_time) const {
    return current_time >= parent_order_.end_time;
}

double AggressiveStrategy::calculate_urgency(Timestamp current_time, Quantity remaining) const {
    // Base urgency from parent order
    double base_urgency = parent_order_.urgency;
    
    // Time pressure: increase urgency as deadline approaches
    int64_t total_time = (parent_order_.end_time - parent_order_.start_time).count();
    int64_t elapsed_time = (current_time - parent_order_.start_time).count();
    
    double time_pressure = 0.0;
    if (total_time > 0) {
        time_pressure = static_cast<double>(elapsed_time) / static_cast<double>(total_time);
    }
    
    // Quantity pressure: increase urgency if we're behind schedule
    double expected_fill_rate = static_cast<double>(elapsed_time) / static_cast<double>(total_time);
    double actual_fill_rate = 1.0 - (static_cast<double>(remaining) / static_cast<double>(parent_order_.total_quantity));
    double quantity_pressure = std::max(0.0, expected_fill_rate - actual_fill_rate);
    
    // Combined urgency
    double urgency = base_urgency + 0.3 * time_pressure + 0.4 * quantity_pressure;
    
    return std::min(1.0, urgency);
}

bool AggressiveStrategy::should_cross_spread(double urgency) const {
    return urgency >= config_.urgency_threshold;
}

Quantity AggressiveStrategy::calculate_sweep_size(const Quote& quote, Quantity remaining) const {
    // Determine available liquidity
    Quantity available;
    if (parent_order_.side == Side::BUY) {
        available = quote.ask_size;
    } else {
        available = quote.bid_size;
    }
    
    // Take a percentage of available liquidity
    Quantity sweep_size = static_cast<Quantity>(available * config_.sweep_percentage);
    
    // Limit to remaining quantity
    sweep_size = std::min(sweep_size, remaining);
    
    // Ensure minimum size
    const Quantity min_size = 100;
    if (sweep_size < min_size && remaining >= min_size) {
        sweep_size = min_size;
    }
    
    return sweep_size;
}

} // namespace execution
