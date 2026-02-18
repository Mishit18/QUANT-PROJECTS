#include "venue.hpp"
#include <algorithm>
#include <cmath>

namespace execution {

Venue::Venue(const VenueConfig& config, std::mt19937& rng)
    : config_(config), order_book_(config.id), rng_(rng), next_order_id_(1000000) {
    initialize_liquidity();
}

void Venue::submit_order(std::shared_ptr<Order> order, Timestamp current_time) {
    order->venue_id = config_.id;
    order->submit_time = current_time;
    
    // Simulate network latency
    Timestamp arrival_time = current_time + get_latency();
    pending_orders_.push_back({order, arrival_time});
}

std::vector<Fill> Venue::process_orders(Timestamp current_time) {
    std::vector<Fill> all_fills;
    
    // Process pending orders that have arrived
    auto it = pending_orders_.begin();
    while (it != pending_orders_.end()) {
        if (it->arrival_time <= current_time) {
            auto order = it->order;
            
            // Try to match order
            auto fills = order_book_.match_order(order, current_time);
            all_fills.insert(all_fills.end(), fills.begin(), fills.end());
            
            // If order not fully filled and is limit order, add to book
            if (order->remaining_quantity() > 0 && order->type == OrderType::LIMIT) {
                order_book_.add_order(order);
            }
            
            it = pending_orders_.erase(it);
        } else {
            ++it;
        }
    }
    
    return all_fills;
}

Quote Venue::get_quote(Timestamp current_time) const {
    return order_book_.get_quote(current_time);
}

void Venue::simulate_market_activity(Timestamp current_time, double activity_rate) {
    std::poisson_distribution<int> poisson(activity_rate);
    int num_orders = poisson(rng_);
    
    Quote quote = get_quote(current_time);
    if (quote.bid_price == 0.0 || quote.ask_price == 0.0) {
        return;  // No valid market
    }
    
    std::uniform_real_distribution<double> side_dist(0.0, 1.0);
    std::uniform_real_distribution<double> price_offset(-0.02, 0.02);
    std::uniform_int_distribution<Quantity> size_dist(100, 5000);
    
    for (int i = 0; i < num_orders; ++i) {
        auto order = std::make_shared<Order>();
        order->id = next_order_id_++;
        order->side = (side_dist(rng_) < 0.5) ? Side::BUY : Side::SELL;
        order->type = OrderType::LIMIT;
        order->quantity = size_dist(rng_);
        order->filled_quantity = 0;
        order->status = OrderStatus::PENDING;
        
        // Price near current market
        if (order->side == Side::BUY) {
            order->price = quote.bid_price * (1.0 + price_offset(rng_));
        } else {
            order->price = quote.ask_price * (1.0 + price_offset(rng_));
        }
        
        submit_order(order, current_time);
    }
}

Timestamp Venue::get_latency() const {
    std::normal_distribution<double> latency_dist(config_.base_latency_us, config_.latency_std_dev_us);
    double latency = std::max(1.0, latency_dist(rng_));
    return Timestamp(static_cast<int64_t>(latency));
}

void Venue::initialize_liquidity() {
    Price mid = config_.initial_mid_price;
    Price spread = mid * config_.initial_spread_bps / 10000.0;
    Price tick = spread / 10.0;  // Reasonable tick size
    
    // Create bid levels
    for (int i = 0; i < config_.num_levels; ++i) {
        Price price = mid - spread / 2.0 - i * tick;
        auto order = std::make_shared<Order>();
        order->id = next_order_id_++;
        order->side = Side::BUY;
        order->type = OrderType::LIMIT;
        order->price = price;
        order->quantity = config_.initial_depth;
        order->filled_quantity = 0;
        order->status = OrderStatus::PENDING;
        order->venue_id = config_.id;
        
        order_book_.add_order(order);
    }
    
    // Create ask levels
    for (int i = 0; i < config_.num_levels; ++i) {
        Price price = mid + spread / 2.0 + i * tick;
        auto order = std::make_shared<Order>();
        order->id = next_order_id_++;
        order->side = Side::SELL;
        order->type = OrderType::LIMIT;
        order->price = price;
        order->quantity = config_.initial_depth;
        order->filled_quantity = 0;
        order->status = OrderStatus::PENDING;
        order->venue_id = config_.id;
        
        order_book_.add_order(order);
    }
}

} // namespace execution
