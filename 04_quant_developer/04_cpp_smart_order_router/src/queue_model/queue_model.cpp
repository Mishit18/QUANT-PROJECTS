#include "queue_model.hpp"
#include <cmath>
#include <algorithm>

namespace execution {

QueueModel::QueueModel(const Config& config) : config_(config) {}

double QueueModel::estimate_fill_probability(
    int queue_position,
    Quantity order_size,
    Quantity available_depth,
    double latency_us,
    double time_horizon_us
) const {
    if (available_depth == 0) {
        return 0.0;
    }
    
    // Adjust queue position for latency-induced changes
    int arrival_position = estimate_arrival_queue_position(
        queue_position,
        latency_us,
        config_.market_order_arrival_rate
    );
    
    // If we're beyond available depth, no fill
    if (arrival_position > static_cast<int>(available_depth)) {
        return 0.0;
    }
    
    // Calculate queue depletion over time horizon
    double depletion = calculate_queue_depletion(
        time_horizon_us,
        config_.market_order_arrival_rate
    );
    
    // Probability that queue depletes to our position
    double queue_reach_prob = std::min(1.0, depletion / arrival_position);
    
    // Probability we're not cancelled before fill
    double no_cancel_prob = calculate_fill_before_cancel_prob(
        arrival_position,
        time_horizon_us
    );
    
    // Combined probability
    double fill_prob = queue_reach_prob * no_cancel_prob;
    
    // Adjust for order size relative to depth
    double size_factor = std::min(1.0, 
        static_cast<double>(available_depth) / static_cast<double>(order_size));
    
    return fill_prob * size_factor;
}

int QueueModel::estimate_arrival_queue_position(
    int current_position,
    double latency_us,
    double market_activity_rate
) const {
    // During latency period, queue can change due to:
    // 1. Market orders arriving (deplete queue)
    // 2. Cancellations (improve position)
    // 3. New limit orders (worsen position)
    
    double time_sec = latency_us / 1e6;
    
    // Expected market orders during latency
    double expected_market_orders = market_activity_rate * time_sec;
    
    // Expected cancellations
    double expected_cancellations = current_position * config_.cancellation_rate * time_sec;
    
    // Net change in position
    // Market orders improve position, cancellations improve position
    double position_improvement = expected_market_orders + expected_cancellations;
    
    // New position (can't go below 0)
    int new_position = std::max(0, 
        current_position - static_cast<int>(position_improvement));
    
    return new_position;
}

double QueueModel::estimate_time_to_fill(
    int queue_position,
    Quantity order_size,
    double market_order_rate
) const {
    if (market_order_rate == 0.0) {
        return 1e9;  // Infinite time
    }
    
    // Expected time for queue to deplete to our position
    // Assuming exponential inter-arrival times
    double expected_time_sec = queue_position / market_order_rate;
    
    return expected_time_sec * 1e6;  // Convert to microseconds
}

double QueueModel::calculate_queue_depletion(
    double time_us,
    double market_order_rate
) const {
    double time_sec = time_us / 1e6;
    
    // Expected number of market orders in time period
    // Poisson process: E[N(t)] = lambda * t
    double expected_orders = market_order_rate * time_sec;
    
    // Apply decay rate (queue doesn't deplete linearly)
    double depletion = expected_orders * (1.0 - std::exp(-config_.queue_decay_rate * time_sec));
    
    return depletion;
}

double QueueModel::calculate_fill_before_cancel_prob(
    int queue_position,
    double time_horizon_us
) const {
    double time_sec = time_horizon_us / 1e6;
    
    // Probability of not being cancelled
    // Exponential survival: P(T > t) = exp(-lambda * t)
    double cancel_rate_per_sec = config_.cancellation_rate;
    double no_cancel_prob = std::exp(-cancel_rate_per_sec * time_sec);
    
    return no_cancel_prob;
}

} // namespace execution
