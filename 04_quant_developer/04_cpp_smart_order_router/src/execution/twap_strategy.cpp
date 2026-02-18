#include "strategies/twap_strategy.hpp"
#include <algorithm>
#include <cmath>

namespace execution {

TWAPStrategy::TWAPStrategy(const Config& config)
    : config_(config), current_slice_(0), last_slice_time_(0),
      slice_size_(0), unfilled_from_previous_slices_(0) {}

void TWAPStrategy::initialize(const ParentOrder& parent_order) {
    parent_order_ = parent_order;
    current_slice_ = 0;
    last_slice_time_ = parent_order.start_time;
    unfilled_from_previous_slices_ = 0;
    calculate_slice_size();
}

void TWAPStrategy::calculate_slice_size() {
    slice_size_ = parent_order_.total_quantity / config_.num_slices;
    // Handle remainder in last slice
    if (parent_order_.total_quantity % config_.num_slices != 0) {
        slice_size_ += 1;
    }
}

std::vector<std::shared_ptr<Order>> TWAPStrategy::generate_orders(
    Timestamp current_time,
    const std::vector<Quote>& venue_quotes,
    Quantity remaining_quantity
) {
    std::vector<std::shared_ptr<Order>> orders;
    
    if (remaining_quantity == 0 || current_slice_ >= config_.num_slices) {
        return orders;
    }
    
    // Check if it's time for next slice
    Timestamp time_since_last_slice = current_time - last_slice_time_;
    if (time_since_last_slice < Timestamp(static_cast<int64_t>(config_.slice_interval_us))) {
        return orders;
    }
    
    // Calculate quantity for this slice (including unfilled from previous)
    Quantity slice_quantity = std::min(
        slice_size_ + unfilled_from_previous_slices_,
        remaining_quantity
    );
    
    if (slice_quantity == 0) {
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
    
    // Determine order type and price based on aggression
    OrderType order_type = (config_.aggression > 0.7) ? OrderType::MARKET : OrderType::LIMIT;
    Price order_price = determine_order_price(*best_quote, parent_order_.side);
    
    // Create child order
    auto order = create_child_order(
        parent_order_.side,
        slice_quantity,
        order_price,
        order_type
    );
    
    orders.push_back(order);
    
    // Update state
    current_slice_++;
    last_slice_time_ = current_time;
    unfilled_from_previous_slices_ = 0;  // Will be updated in on_fill
    
    return orders;
}

void TWAPStrategy::on_fill(const Fill& fill, Timestamp current_time) {
    // Track unfilled quantity for next slice
    // This is handled by the execution engine tracking remaining quantity
}

bool TWAPStrategy::is_complete(Timestamp current_time) const {
    return current_slice_ >= config_.num_slices ||
           current_time >= parent_order_.end_time;
}

Price TWAPStrategy::determine_order_price(const Quote& quote, Side side) const {
    if (config_.aggression > 0.7) {
        return 0.0;  // Market order
    }
    
    // Passive to moderately aggressive limit orders
    if (side == Side::BUY) {
        // Place bid between best bid and mid
        double offset = (1.0 - config_.aggression) * quote.spread() / 2.0;
        return quote.bid_price + offset;
    } else {
        // Place ask between mid and best ask
        double offset = (1.0 - config_.aggression) * quote.spread() / 2.0;
        return quote.ask_price - offset;
    }
}

} // namespace execution
