#include "strategies/pov_strategy.hpp"
#include <algorithm>
#include <numeric>

namespace execution {

POVStrategy::POVStrategy(const Config& config)
    : config_(config), last_order_time_(0), our_filled_in_window_(0) {}

void POVStrategy::initialize(const ParentOrder& parent_order) {
    parent_order_ = parent_order;
    last_order_time_ = parent_order.start_time;
    volume_history_.clear();
    our_filled_in_window_ = 0;
}

std::vector<std::shared_ptr<Order>> POVStrategy::generate_orders(
    Timestamp current_time,
    const std::vector<Quote>& venue_quotes,
    Quantity remaining_quantity
) {
    std::vector<std::shared_ptr<Order>> orders;
    
    if (remaining_quantity == 0) {
        return orders;
    }
    
    // Check if enough time has passed since last order
    Timestamp time_since_last = current_time - last_order_time_;
    if (time_since_last < Timestamp(static_cast<int64_t>(config_.order_interval_us))) {
        return orders;
    }
    
    // Calculate recent market volume
    Quantity market_volume = calculate_market_volume(current_time);
    
    if (market_volume == 0) {
        return orders;  // No market activity to participate in
    }
    
    // Check current participation rate
    double current_participation = calculate_participation_rate(current_time);
    
    // If we're above max participation, wait
    if (current_participation >= config_.max_participation) {
        return orders;
    }
    
    // Calculate order size to achieve target participation
    Quantity order_size = calculate_order_size(market_volume, remaining_quantity);
    
    if (order_size == 0) {
        return orders;
    }
    
    // Find best quote
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
    
    // Determine order type and price
    OrderType order_type = (config_.aggression > 0.7) ? OrderType::MARKET : OrderType::LIMIT;
    Price order_price = determine_order_price(*best_quote, parent_order_.side);
    
    // Create child order
    auto order = create_child_order(
        parent_order_.side,
        order_size,
        order_price,
        order_type
    );
    
    orders.push_back(order);
    last_order_time_ = current_time;
    
    return orders;
}

void POVStrategy::on_fill(const Fill& fill, Timestamp current_time) {
    our_filled_in_window_ += fill.quantity;
    clean_old_observations(current_time);
}

bool POVStrategy::is_complete(Timestamp current_time) const {
    return current_time >= parent_order_.end_time;
}

void POVStrategy::update_market_volume(Quantity volume, Timestamp timestamp) {
    volume_history_.push_back({volume, timestamp});
    clean_old_observations(timestamp);
}

Quantity POVStrategy::calculate_market_volume(Timestamp current_time) const {
    Quantity total = 0;
    Timestamp cutoff = current_time - Timestamp(static_cast<int64_t>(config_.measurement_window_us));
    
    for (const auto& obs : volume_history_) {
        if (obs.timestamp >= cutoff) {
            total += obs.volume;
        }
    }
    
    return total;
}

double POVStrategy::calculate_participation_rate(Timestamp current_time) const {
    Quantity market_vol = calculate_market_volume(current_time);
    
    if (market_vol == 0) {
        return 0.0;
    }
    
    return static_cast<double>(our_filled_in_window_) / static_cast<double>(market_vol);
}

Quantity POVStrategy::calculate_order_size(Quantity market_volume, Quantity remaining) const {
    // Target quantity based on participation rate
    Quantity target = static_cast<Quantity>(market_volume * config_.target_participation);
    
    // Ensure we don't exceed remaining quantity
    Quantity order_size = std::min(target, remaining);
    
    // Ensure minimum order size
    const Quantity min_order = 100;
    if (order_size < min_order && remaining >= min_order) {
        order_size = min_order;
    }
    
    return order_size;
}

Price POVStrategy::determine_order_price(const Quote& quote, Side side) const {
    if (config_.aggression > 0.7) {
        return 0.0;  // Market order
    }
    
    // Price based on aggression level
    if (side == Side::BUY) {
        double offset = config_.aggression * quote.spread();
        return quote.bid_price + offset;
    } else {
        double offset = config_.aggression * quote.spread();
        return quote.ask_price - offset;
    }
}

void POVStrategy::clean_old_observations(Timestamp current_time) {
    Timestamp cutoff = current_time - Timestamp(static_cast<int64_t>(config_.measurement_window_us));
    
    auto it = std::remove_if(volume_history_.begin(), volume_history_.end(),
        [cutoff](const VolumeObservation& obs) {
            return obs.timestamp < cutoff;
        });
    
    volume_history_.erase(it, volume_history_.end());
    
    // Recalculate our filled quantity in window
    // This is simplified - in production would track our fills with timestamps
}

} // namespace execution
