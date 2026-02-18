#include "strategies/vwap_strategy.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace execution {

VWAPStrategy::VWAPStrategy(const Config& config)
    : config_(config), current_interval_(0), last_interval_time_(0),
      cumulative_target_(0), cumulative_filled_(0),
      initial_spread_(0.0), expected_volume_(0), observed_volume_(0) {}

void VWAPStrategy::initialize(const ParentOrder& parent_order) {
    parent_order_ = parent_order;
    current_interval_ = 0;
    last_interval_time_ = parent_order.start_time;
    cumulative_target_ = 0;
    cumulative_filled_ = 0;
    initial_spread_ = 0.0;
    observed_volume_ = 0;
    calculate_targets();
    
    // Calculate expected volume based on profile
    expected_volume_ = parent_order.total_quantity;
}

void VWAPStrategy::calculate_targets() {
    target_quantities_.clear();
    target_quantities_.reserve(config_.num_intervals);
    
    // Normalize volume profile if needed
    double total = std::accumulate(config_.volume_profile.begin(), 
                                   config_.volume_profile.end(), 0.0);
    
    Quantity allocated = 0;
    for (size_t i = 0; i < config_.volume_profile.size() && i < static_cast<size_t>(config_.num_intervals); ++i) {
        double fraction = config_.volume_profile[i] / total;
        Quantity target = static_cast<Quantity>(parent_order_.total_quantity * fraction);
        target_quantities_.push_back(target);
        allocated += target;
    }
    
    // Allocate remainder to last interval
    if (!target_quantities_.empty()) {
        target_quantities_.back() += (parent_order_.total_quantity - allocated);
    }
}

std::vector<std::shared_ptr<Order>> VWAPStrategy::generate_orders(
    Timestamp current_time,
    const std::vector<Quote>& venue_quotes,
    Quantity remaining_quantity
) {
    std::vector<std::shared_ptr<Order>> orders;
    
    if (remaining_quantity == 0) {
        return orders;
    }
    
    int interval = get_current_interval(current_time);
    if (interval >= config_.num_intervals || interval < 0) {
        return orders;
    }
    
    // Check if we've moved to a new interval
    if (interval > current_interval_) {
        current_interval_ = interval;
        last_interval_time_ = current_time;
    }
    
    // Store initial spread if not set
    if (initial_spread_ == 0.0 && !venue_quotes.empty()) {
        initial_spread_ = venue_quotes[0].spread();
    }
    
    // Check if should deviate from VWAP schedule
    bool deviate = should_deviate(venue_quotes, current_time);
    
    // Calculate how much we should have filled by now
    Quantity target_by_now = 0;
    for (int i = 0; i <= current_interval_ && i < static_cast<int>(target_quantities_.size()); ++i) {
        target_by_now += target_quantities_[i];
    }
    
    // Calculate catch-up quantity (if we're behind schedule)
    Quantity shortfall = target_by_now - parent_order_.filled_quantity;
    
    // If deviating, adjust order size
    if (deviate) {
        // Reduce order size when conditions are unfavorable
        shortfall = static_cast<Quantity>(shortfall * 0.5);
    }
    
    if (shortfall <= 0) {
        return orders;  // We're ahead of schedule, wait
    }
    
    // Limit order size to remaining quantity
    Quantity order_quantity = std::min(shortfall, remaining_quantity);
    
    if (order_quantity == 0) {
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
        order_quantity,
        order_price,
        order_type
    );
    
    orders.push_back(order);
    
    return orders;
}

void VWAPStrategy::on_fill(const Fill& fill, Timestamp current_time) {
    cumulative_filled_ += fill.quantity;
}

bool VWAPStrategy::is_complete(Timestamp current_time) const {
    return current_interval_ >= config_.num_intervals ||
           current_time >= parent_order_.end_time;
}

Price VWAPStrategy::determine_order_price(const Quote& quote, Side side) const {
    if (config_.aggression > 0.7) {
        return 0.0;  // Market order
    }
    
    // Adjust price based on aggression
    if (side == Side::BUY) {
        double offset = config_.aggression * quote.spread();
        return quote.bid_price + offset;
    } else {
        double offset = config_.aggression * quote.spread();
        return quote.ask_price - offset;
    }
}

int VWAPStrategy::get_current_interval(Timestamp current_time) const {
    if (current_time < parent_order_.start_time) {
        return -1;
    }
    
    int64_t elapsed = (current_time - parent_order_.start_time).count();
    int interval = static_cast<int>(elapsed / static_cast<int64_t>(config_.interval_us));
    
    return interval;
}

bool VWAPStrategy::should_deviate(const std::vector<Quote>& quotes, Timestamp current_time) {
    // Reset deviation flags
    last_deviation_.spread_widened = false;
    last_deviation_.impact_increased = false;
    last_deviation_.volume_deviated = false;
    
    if (quotes.empty()) {
        return false;
    }
    
    // Check spread condition
    if (is_spread_widened(quotes)) {
        last_deviation_.spread_widened = true;
        return true;
    }
    
    // Check impact condition
    Quantity next_order_size = parent_order_.total_quantity / config_.num_intervals;
    if (is_impact_increased(next_order_size, quotes)) {
        last_deviation_.impact_increased = true;
        return true;
    }
    
    // Check volume condition
    if (is_volume_deviated(current_time)) {
        last_deviation_.volume_deviated = true;
        return true;
    }
    
    return false;
}

bool VWAPStrategy::is_spread_widened(const std::vector<Quote>& quotes) const {
    if (initial_spread_ == 0.0) {
        return false;
    }
    
    // Find average current spread
    double total_spread = 0.0;
    int count = 0;
    for (const auto& quote : quotes) {
        if (quote.bid_price > 0 && quote.ask_price > 0) {
            total_spread += quote.spread();
            count++;
        }
    }
    
    if (count == 0) {
        return false;
    }
    
    double avg_spread = total_spread / count;
    double spread_increase_bps = ((avg_spread - initial_spread_) / initial_spread_) * 10000.0;
    
    // Deviate if spread widened by more than threshold
    return spread_increase_bps > last_deviation_.spread_threshold_bps;
}

bool VWAPStrategy::is_impact_increased(Quantity order_size, const std::vector<Quote>& quotes) const {
    // Estimate if our order would have significant impact
    // Impact increases when order size is large relative to available depth
    
    if (quotes.empty()) {
        return false;
    }
    
    // Find minimum depth across venues
    Quantity min_depth = std::numeric_limits<Quantity>::max();
    for (const auto& quote : quotes) {
        Quantity depth = (parent_order_.side == Side::BUY) ? quote.ask_size : quote.bid_size;
        if (depth > 0) {
            min_depth = std::min(min_depth, depth);
        }
    }
    
    if (min_depth == std::numeric_limits<Quantity>::max()) {
        return true;  // No liquidity
    }
    
    // Calculate participation rate
    double participation = static_cast<double>(order_size) / static_cast<double>(min_depth);
    
    // Deviate if participation exceeds threshold
    return participation > last_deviation_.impact_threshold;
}

bool VWAPStrategy::is_volume_deviated(Timestamp current_time) const {
    // Check if observed volume deviates significantly from expected
    
    if (expected_volume_ == 0) {
        return false;
    }
    
    // Calculate expected volume by this time
    int interval = get_current_interval(current_time);
    if (interval < 0 || interval >= config_.num_intervals) {
        return false;
    }
    
    double expected_fraction = 0.0;
    for (int i = 0; i <= interval && i < static_cast<int>(config_.volume_profile.size()); ++i) {
        expected_fraction += config_.volume_profile[i];
    }
    
    Quantity expected_by_now = static_cast<Quantity>(expected_volume_ * expected_fraction);
    
    if (expected_by_now == 0) {
        return false;
    }
    
    // Calculate deviation
    double deviation = std::abs(static_cast<double>(observed_volume_) - 
                                static_cast<double>(expected_by_now)) / 
                       static_cast<double>(expected_by_now);
    
    // Deviate if volume differs by more than threshold
    return deviation > last_deviation_.volume_deviation_pct;
}

} // namespace execution
